// RUN: mlir-opt %s --sparsification-and-bufferization        | FileCheck %s --check-prefix=CHECK-NOPARA
// RUN: mlir-opt %s --sparsification-and-bufferization="parallelization-strategy=any-storage-any-loop" | FileCheck %s --check-prefix=CHECK-PARA

// Test to ensure we can pass parallelization flags into
// the mini sparsification and bufferization pipeline.

#SparseMatrix = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed, d1 : compressed)
}>

#trait_ss = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(i,j) * SCALE"
}

//
// CHECK-NOPARA-LABEL: func.func @scale_ss
// CHECK-NOPARA:       scf.for
//
// CHECK-PARA-LABEL: func.func @scale_ss
// CHECK-PARA:       scf.parallel
//
func.func @scale_ss(%scale: f32,
               %arga: tensor<?x?xf32, #SparseMatrix>,
	       %argx: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic #trait_ss
     ins(%arga: tensor<?x?xf32, #SparseMatrix>)
    outs(%argx: tensor<?x?xf32>) {
      ^bb(%a: f32, %x: f32):
        %0 = arith.mulf %a, %scale : f32
        linalg.yield %0 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
