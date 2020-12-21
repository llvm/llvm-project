// RUN: mlir-opt %s -split-input-file -verify-diagnostics

#trait_memref = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a
    affine_map<(i) -> (i)>   // x (out)
  ],
  sparse = [
    [ "S" ],  // a
    [ "D" ]   // x
  ],
  iterator_types = ["parallel"]
}

func @invalid_memref(%arga: memref<32xf32>, %argb: f32, %shape: tensor<32xf32>)
  -> tensor<32xf32>
{
  // expected-error@+1 {{'linalg.generic' op expected sparse annotations on tensors only}}
  %0 = linalg.generic #trait_memref
     ins(%arga: memref<32xf32>)
    outs(%shape: tensor<32xf32>) {
      ^bb(%a: f32, %s: f32):
        %0 = addf %a, %argb  : f32
        linalg.yield %0 : f32
  } -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// -----

#trait_too_many = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a
    affine_map<(i) -> (i)>   // x (out)
  ],
  sparse = [
    [ "S" ],  // a
    [ "S" ],  // b
    [ "D" ]   // x
  ],
  iterator_types = ["parallel"]
}

func @invalid_too_many(%arga: tensor<32xf32>, %argb: f32) -> tensor<32xf32> {
  // expected-error@+1 {{'linalg.generic' op expected one sparse annotation for each tensor}}
  %0 = linalg.generic #trait_too_many
     ins(%arga: tensor<32xf32>)
    outs(%arga: tensor<32xf32>) {
      ^bb(%a: f32, %s: f32):
        %0 = addf %a, %argb  : f32
        linalg.yield %0 : f32
  } -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// -----

#trait_no_array = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a
    affine_map<(i) -> (i)>   // x (out)
  ],
  sparse = [ 1, 2 ],
  iterator_types = ["parallel"]
}

func @invalid_no_array(%arga: tensor<32xf32>, %argb: f32) -> tensor<32xf32> {
  // expected-error@+1 {{'linalg.generic' op expected sparse annotation array for tensor 0}}
  %0 = linalg.generic #trait_no_array
     ins(%arga: tensor<32xf32>)
    outs(%arga: tensor<32xf32>) {
      ^bb(%a: f32, %s: f32):
        %0 = addf %a, %argb  : f32
        linalg.yield %0 : f32
  } -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// -----

#trait_wrong_rank = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a
    affine_map<(i) -> (i)>   // x (out)
  ],
  sparse = [
    [ "S" ],
    [ "D", "D" ]
  ],
  iterator_types = ["parallel"]
}

func @invalid_wrong_rank(%arga: tensor<32xf32>, %argb: f32) -> tensor<32xf32> {
  // expected-error@+1 {{'linalg.generic' op expected sparse annotation with rank 1 for tensor 1}}
  %0 = linalg.generic #trait_wrong_rank
     ins(%arga: tensor<32xf32>)
    outs(%arga: tensor<32xf32>) {
      ^bb(%a: f32, %s: f32):
        %0 = addf %a, %argb  : f32
        linalg.yield %0 : f32
  } -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// -----

#trait_no_string = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // a
    affine_map<(i,j) -> (i,j)>   // x (out)
  ],
  sparse = [
    [ "S", 1 ],
    [ "D", "D" ]
  ],
  iterator_types = ["parallel","parallel"]
}

func @invalid_no_string(%arga: tensor<32x16xf32>, %argb: f32) -> tensor<32x16xf32> {
  // expected-error@+1 {{'linalg.generic' op expected sparse annotation at position 1 for tensor 0}}
  %0 = linalg.generic #trait_no_string
     ins(%arga: tensor<32x16xf32>)
    outs(%arga: tensor<32x16xf32>) {
      ^bb(%a: f32, %s: f32):
        %0 = addf %a, %argb  : f32
        linalg.yield %0 : f32
  } -> tensor<32x16xf32>
  return %0 : tensor<32x16xf32>
}

// -----

#trait_wrong_symbol = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // a
    affine_map<(i,j) -> (i,j)>   // x (out)
  ],
  sparse = [
    [ "S", "S" ],
    [ "D", "X" ]
  ],
  iterator_types = ["parallel","parallel"]
}

func @invalid_wrong_symbol(%arga: tensor<32x16xf32>, %argb: f32) -> tensor<32x16xf32> {
  // expected-error@+1 {{'linalg.generic' op expected sparse annotation at position 1 for tensor 1}}
  %0 = linalg.generic #trait_wrong_symbol
     ins(%arga: tensor<32x16xf32>)
    outs(%arga: tensor<32x16xf32>) {
      ^bb(%a: f32, %s: f32):
        %0 = addf %a, %argb  : f32
        linalg.yield %0 : f32
  } -> tensor<32x16xf32>
  return %0 : tensor<32x16xf32>
}

// -----

#trait_no_sparse_output = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // a
    affine_map<(i,j) -> (i,j)>   // x (out)
  ],
  sparse = [
    [ "S", "S" ],
    [ "D", "S" ]
  ],
  iterator_types = ["parallel","parallel"]
}

func @invalid_no_sparse_output(%arga: tensor<32x16xf32>, %argb: f32) -> tensor<32x16xf32> {
  // expected-error@+1 {{'linalg.generic' op sparse output tensors not supported (yet)}}
  %0 = linalg.generic #trait_no_sparse_output
     ins(%arga: tensor<32x16xf32>)
    outs(%arga: tensor<32x16xf32>) {
      ^bb(%a: f32, %s: f32):
        %0 = addf %a, %argb  : f32
        linalg.yield %0 : f32
  } -> tensor<32x16xf32>
  return %0 : tensor<32x16xf32>
}
