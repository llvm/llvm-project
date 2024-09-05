func.func @test_add_0d(%arg0: tensor<2x32xf32>, %arg1: tensor<2x32xf32>) -> (tensor<2x32xf32>, tensor<2x32xf32>) {
  %0 = tosa.add %arg0, %arg1 : (tensor<2x32xf32>, tensor<2x32xf32>) -> tensor<2x32xf32>
  %1 = tosa.add %arg0, %arg1 : (tensor<2x32xf32>, tensor<2x32xf32>) -> tensor<2x32xf32>
  return %0, %1 : tensor<2x32xf32>, tensor<2x32xf32>
}