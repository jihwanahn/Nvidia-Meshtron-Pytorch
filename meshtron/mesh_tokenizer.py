import os
import torch
import trimesh
import numpy as np

class MeshTokenizer:

    def __init__(self, bins: int):
        "Quantize and add special tokens"
        self.box_dim = 1.0
        self.bins = bins

        #Special tokens
        self.SOS = torch.tensor([bins], dtype=torch.int64)
        self.EOS = torch.tensor([bins+1], dtype=torch.int64)
        self.PAD = torch.tensor([bins+2], dtype=torch.int64) 

        self.vocab_size = bins + 3 # add 3 for special tokens

    
    def __extract_faces_bot_top(self, mesh: trimesh.Trimesh, vertices_yzx: torch.Tensor):
        faces = mesh.faces
        
        face_data = []
        for face in faces:
            face_verts = vertices_yzx[face].numpy()
            sorted_verts = face_verts[np.lexsort((face_verts[:, 2], face_verts[:, 1], face_verts[:, 0]))]
            sort_key = tuple(sorted_verts[0].tolist())
            face_data.append((sort_key, face))
        
        face_data.sort(key=lambda x: x[0])
        faces = np.array([face for _, face in face_data])
        return torch.from_numpy(faces)
    
    def __normalize_verts_to_box(self, file_path: str):
        """
        Normalize vertices of mesh so that it fits inside a cube bounding box of size 1.0 and zero centers it.

        Parameters:
            mesh (trimesh.Trimesh): Input mesh
        """

        vertices = self.__get_vertices(file_path)
        min_coord = np.min(vertices, axis=0)
        max_coord = np.max(vertices, axis=0)

        # Center of bounding box
        center = (max_coord + min_coord) / 2.0
        vertices -= center  # shift to zero-center
    
        dimension = max_coord - min_coord
        
        scale = 1.0 / np.max(dimension)

        vertices *= scale

        return torch.from_numpy(vertices)
    
    def __lex_sort_verts(self, face: torch.Tensor, all_vertices: torch.Tensor):
        face_vertices = np.array([all_vertices[vert] for vert in face])
        sorted_idx = np.lexsort((face_vertices[:, 2], face_vertices[:, 1], face_vertices[:, 0]))
        return face_vertices[sorted_idx]
    
    def __get_vertices(self, obj_file: str):
        if not os.path.exists(obj_file):
            raise FileNotFoundError(f"File not found {obj_file}")
        vertices = []
        with open(obj_file, 'r') as obj:
            for line in obj:
                line = line.strip()

                if not line or line.startswith('#'):
                    continue

                parts  = line.split()

                if parts[0] == 'v':
                    vertices.append(parts[1:])
        return np.array(vertices, dtype=float)

    def quantize(self, sequence: torch.Tensor):
        "converts float values to discrete int bins"
        return (torch.clamp(torch.floor((sequence + (self.box_dim / 2)) * (self.bins / self.box_dim)), 0, self.bins - 1)).to(dtype=torch.int64)

    def dequantize(self, tokens: torch.Tensor):
        "converts integer bins to float values (returns bin center)"
        return (tokens.float() + 0.5) / self.bins * self.box_dim - (self.box_dim / 2)
    
    def encode(self, mesh_path: str):

        mesh = trimesh.load(mesh_path)

        vertices = self.__normalize_verts_to_box(mesh_path)
       
        mesh.vertices = vertices.numpy()

        vertices_yzx = vertices[:, [1,2,0]]

        face_list = self.__extract_faces_bot_top(mesh, vertices_yzx)

        sorted_faces_verts = torch.from_numpy(np.array([self.__lex_sort_verts(face, vertices_yzx) for face in face_list]))

        # Flatten the (N, 3, 3) list to (N*9)
        sequence = torch.flatten(sorted_faces_verts)

        sequence = self.quantize(sequence)

        return sequence
    
    def decode(self, x:  torch.Tensor):
        """Converts integer tokens to corresponding float coordinates in YZX order"""

        coordinates = self.dequantize(x)

        points = coordinates.view([-1, 3])

        return points