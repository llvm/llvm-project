// RUN: env MLIR_NUM_THREADS=8 mlir-opt %s -test-tosa-op-availability > /dev/null

func.func @f0(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tosa.identity %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

func.func @f1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = tosa.identity %arg0 : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}