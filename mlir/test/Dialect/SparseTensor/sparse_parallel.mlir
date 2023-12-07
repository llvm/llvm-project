// RUN: mlir-opt %s -sparsification="parallelization-strategy=none" | \
// RUN:   FileCheck %s --check-prefix=CHECK-PAR0
// RUN: mlir-opt %s -sparsification="parallelization-strategy=dense-outer-loop" | \
// RUN:   FileCheck %s --check-prefix=CHECK-PAR1
// RUN: mlir-opt %s -sparsification="parallelization-strategy=any-storage-outer-loop" | \
// RUN:   FileCheck %s --check-prefix=CHECK-PAR2
// RUN: mlir-opt %s -sparsification="parallelization-strategy=dense-any-loop" | \
// RUN:   FileCheck %s --check-prefix=CHECK-PAR3
// RUN: mlir-opt %s -sparsification="parallelization-strategy=any-storage-any-loop" | \
// RUN:   FileCheck %s --check-prefix=CHECK-PAR4

#DenseMatrix = #sparse_tensor.encoding<{
  lvlTypes = [ "dense", "dense" ]
}>

#SparseMatrix = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed", "compressed" ]
}>

#CSR = #sparse_tensor.encoding<{
  lvlTypes = [ "dense", "compressed" ]
}>

#trait_dd = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(i,j) * SCALE"
}

//
// CHECK-PAR0-LABEL: func @scale_dd
// CHECK-PAR0:         scf.for
// CHECK-PAR0:           scf.for
// CHECK-PAR0:         return
//
// CHECK-PAR1-LABEL: func @scale_dd
// CHECK-PAR1:         scf.parallel
// CHECK-PAR1:           scf.for
// CHECK-PAR1:         return
//
// CHECK-PAR2-LABEL: func @scale_dd
// CHECK-PAR2:         scf.parallel
// CHECK-PAR2:           scf.for
// CHECK-PAR2:         return
//
// CHECK-PAR3-LABEL: func @scale_dd
// CHECK-PAR3:         scf.parallel
// CHECK-PAR3:           scf.parallel
// CHECK-PAR3:         return
//
// CHECK-PAR4-LABEL: func @scale_dd
// CHECK-PAR4:         scf.parallel
// CHECK-PAR4:           scf.parallel
// CHECK-PAR4:         return
//
func.func @scale_dd(%scale: f32,
               %arga: tensor<?x?xf32, #DenseMatrix>,
	       %argx: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic #trait_dd
     ins(%arga: tensor<?x?xf32, #DenseMatrix>)
    outs(%argx: tensor<?x?xf32>) {
      ^bb(%a: f32, %x: f32):
        %0 = arith.mulf %a, %scale : f32
        linalg.yield %0 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

#trait_ss = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A
    affine_map<(i,j) -> (i,j)>   // X (out)
  ],
  iterator_types = ["parallel", "parallel"],
  doc = "X(i,j) = A(i,j) * SCALE"
}

//
// CHECK-PAR0-LABEL: func @scale_ss
// CHECK-PAR0:         scf.for
// CHECK-PAR0:           scf.for
// CHECK-PAR0:         return
//
// CHECK-PAR1-LABEL: func @scale_ss
// CHECK-PAR1:         scf.for
// CHECK-PAR1:           scf.for
// CHECK-PAR1:         return
//
// CHECK-PAR2-LABEL: func @scale_ss
// CHECK-PAR2:         scf.parallel
// CHECK-PAR2:           scf.for
// CHECK-PAR2:         return
//
// CHECK-PAR3-LABEL: func @scale_ss
// CHECK-PAR3:         scf.for
// CHECK-PAR3:           scf.for
// CHECK-PAR3:         return
//
// CHECK-PAR4-LABEL: func @scale_ss
// CHECK-PAR4:         scf.parallel
// CHECK-PAR4:           scf.parallel
// CHECK-PAR4:         return
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

#trait_matvec = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>,  // A
    affine_map<(i,j) -> (j)>,    // b
    affine_map<(i,j) -> (i)>     // x (out)
  ],
  iterator_types = ["parallel", "reduction"],
  doc = "x(i) += A(i,j) * b(j)"
}

//
// CHECK-PAR0-LABEL: func @matvec
// CHECK-PAR0:         scf.for
// CHECK-PAR0:           scf.for
// CHECK-PAR0:         return
//
// CHECK-PAR1-LABEL: func @matvec
// CHECK-PAR1:         scf.parallel
// CHECK-PAR1:           scf.for
// CHECK-PAR1:         return
//
// CHECK-PAR2-LABEL: func @matvec
// CHECK-PAR2:         scf.parallel
// CHECK-PAR2:           scf.for
// CHECK-PAR2:         return
//
// CHECK-PAR3-LABEL: func @matvec
// CHECK-PAR3:         scf.parallel
// CHECK-PAR3:           scf.for
// CHECK-PAR3:         return
//
// CHECK-PAR4-LABEL: func @matvec
// CHECK-PAR4:         scf.parallel
// CHECK-PAR4:           scf.parallel
// CHECK-PAR4:             scf.reduce
// CHECK-PAR4:         return
//
func.func @matvec(%arga: tensor<16x32xf32, #CSR>,
             %argb: tensor<32xf32>,
	     %argx: tensor<16xf32>) -> tensor<16xf32> {
  %0 = linalg.generic #trait_matvec
      ins(%arga, %argb : tensor<16x32xf32, #CSR>, tensor<32xf32>)
     outs(%argx: tensor<16xf32>) {
    ^bb(%A: f32, %b: f32, %x: f32):
      %0 = arith.mulf %A, %b : f32
      %1 = arith.addf %0, %x : f32
      linalg.yield %1 : f32
  } -> tensor<16xf32>
  return %0 : tensor<16xf32>
}
