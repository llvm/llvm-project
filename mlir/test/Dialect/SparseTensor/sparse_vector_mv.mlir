// RUN: mlir-opt %s -sparse-compiler="vl=8" |  FileCheck %s

#Dense = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "dense" ]
}>

#matvec = {
  indexing_maps = [
    affine_map<(i,j) -> (i,j)>, // A
    affine_map<(i,j) -> (j)>,   // b
    affine_map<(i,j) -> (i)>    // x (out)
  ],
  iterator_types = ["parallel", "reduction"],
  doc = "X(i) += A(i,j) * B(j)"
}

// CHECK-LABEL: llvm.func @kernel_matvec
// CHECK:       llvm.intr.vector.reduce.fadd
func.func @kernel_matvec(%arga: tensor<?x?xf32, #Dense>,
                         %argb: tensor<?xf32>,
			 %argx: tensor<?xf32>) -> tensor<?xf32> {
  %x = linalg.generic #matvec
    ins(%arga, %argb: tensor<?x?xf32, #Dense>, tensor<?xf32>)
    outs(%argx: tensor<?xf32>) {
    ^bb(%a: f32, %b: f32, %x: f32):
      %0 = arith.mulf %a, %b : f32
      %1 = arith.addf %x, %0 : f32
      linalg.yield %1 : f32
  } -> tensor<?xf32>
  return %x : tensor<?xf32>
}
