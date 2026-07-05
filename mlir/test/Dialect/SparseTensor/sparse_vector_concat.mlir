// RUN: mlir-opt %s --sparsifier="enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true"

#MAT_D_C = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

#MAT_C_C_P = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : compressed, d0 : compressed)
}>

#MAT_C_D_P = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d1 : compressed, d0 : dense)
}>

//
// Ensures only last loop is vectorized
// (vectorizing the others would crash).
//
// CHECK-LABEL: llvm.func @foo
// CHECK:       llvm.intr.masked.load
// CHECK:       llvm.intr.masked.scatter
//
func.func @foo(%arg0: tensor<2x4xf64, #MAT_C_C_P>,
               %arg1: tensor<3x4xf64, #MAT_C_D_P>,
           %arg2: tensor<4x4xf64, #MAT_D_C>) -> tensor<9x4xf64> {
  %0 = sparse_tensor.concatenate %arg0, %arg1, %arg2 {dimension = 0 : index}
       : tensor<2x4xf64, #MAT_C_C_P>, tensor<3x4xf64, #MAT_C_D_P>, tensor<4x4xf64, #MAT_D_C> to tensor<9x4xf64>
  return %0 : tensor<9x4xf64>
}
