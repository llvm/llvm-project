module {
  func @multiply_transpose(%arg0: tensor<*xf64> loc("/home/shkumar/LLVM/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":4:1), %arg1: tensor<*xf64> loc("/home/shkumar/LLVM/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":4:1)) -> tensor<*xf64> {
    %0 = toy.matmul(%arg0 : tensor<*xf64>, %arg1 : tensor<*xf64>) to tensor<*xf64> loc("/home/shkumar/LLVM/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":5:10)
    toy.return %0 : tensor<*xf64> loc("/home/shkumar/LLVM/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":5:3)
  } loc("/home/shkumar/LLVM/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":4:1)
  func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64> loc("/home/shkumar/LLVM/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":9:17)
    %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64> loc("/home/shkumar/LLVM/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":9:3)
    %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64> loc("/home/shkumar/LLVM/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":10:17)
    %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64> loc("/home/shkumar/LLVM/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":10:3)
    %4 = toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("/home/shkumar/LLVM/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":11:11)
    %5 = toy.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("/home/shkumar/LLVM/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":12:11)
    toy.print %5 : tensor<*xf64> loc("/home/shkumar/LLVM/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":13:3)
    toy.return loc("/home/shkumar/LLVM/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":8:1)
  } loc("/home/shkumar/LLVM/llvm-project/mlir/test/Examples/Toy/Ch2/codegen.toy":8:1)
} loc(unknown)
