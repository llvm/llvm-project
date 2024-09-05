func.func @test_add_0d(%arg0: tensor<2x4x32xf32>, %arg1: tensor<2x4x32xf32>) -> (tensor<1x4x32xf32>, tensor<2x1x32xf32>) {
  %0 = tosa.reduce_max %arg0 {axis = 0 : i32} : (tensor<2x4x32xf32>) -> tensor<1x4x32xf32>
  %1 = tosa.reduce_min %arg1 {axis = 1 : i32} : (tensor<2x4x32xf32>) -> tensor<2x1x32xf32>
  return %0, %1 : tensor<1x4x32xf32>, tensor<2x1x32xf32>
}