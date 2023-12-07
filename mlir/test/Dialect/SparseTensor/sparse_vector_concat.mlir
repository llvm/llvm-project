// RUN: mlir-opt %s --sparse-compiler="enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true"

#MAT_D_C = #sparse_tensor.encoding<{
  lvlTypes = ["dense", "compressed"]
}>

#MAT_C_C_P = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed", "compressed" ],
  dimToLvl = affine_map<(i,j) -> (j,i)>
}>

#MAT_C_D_P = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed", "dense" ],
  dimToLvl = affine_map<(i,j) -> (j,i)>
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
