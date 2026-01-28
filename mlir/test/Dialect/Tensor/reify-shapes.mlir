// RUN: mlir-opt -reify-result-shapes  %s | FileCheck %s

// The test below checks concat op reification. In the first case, no cast is inserted while on the second a cast gets inserted.
// CHECK-LABEL:  func.func @concat_reification
func.func @concat_reification(%arg0: tensor<4x7x3xf32>, %arg1 : tensor<4x4x3xf32>, %arg2: tensor<?x?x?xf32>)
  -> (tensor<4x11x3xf32>, tensor<?x?x?xf32>) {
  // CHECK: %[[RES0:.*]] = tensor.concat dim(1) %{{.*}} : (tensor<4x7x3xf32>, tensor<4x4x3xf32>) -> tensor<4x11x3xf32>
  %1 = tensor.concat dim(1) %arg0, %arg1 : (tensor<4x7x3xf32>, tensor<4x4x3xf32>) -> tensor<4x11x3xf32>
  // CHECK: %[[V0:.*]] = tensor.concat dim(2) %{{.*}} : (tensor<4x7x3xf32>, tensor<?x?x?xf32>) -> tensor<4x7x?xf32>
  // CHECK: %[[RES1:.*]] = tensor.cast %[[V0]] : tensor<4x7x?xf32> to tensor<?x?x?xf32>
  %2 = tensor.concat dim(2) %arg0, %arg2 : (tensor<4x7x3xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  // CHECK: return %[[RES0]], %[[RES1]] : tensor<4x11x3xf32>, tensor<?x?x?xf32>
  return %1, %2 : tensor<4x11x3xf32>, tensor<?x?x?xf32>
}

// CHECK-LABEL:  func.func @pad_reification
func.func @pad_reification(%cst : f32, %idx : index, %t: tensor<64x?x64xf32>) -> tensor<1x?x64xf32> {
  %pad_amt = affine.apply affine_map<(d0) -> (-d0 + 256)>(%idx)
  %es = tensor.extract_slice %t[0, 0, 0] [1, %idx, 64] [1, 1, 1] 
    : tensor<64x?x64xf32> to tensor<1x?x64xf32>

  // CHECK: tensor.pad
  // CHECK: : tensor<1x?x64xf32> to tensor<1x256x64xf32>
  // CHECK: tensor.cast %{{.*}} : tensor<1x256x64xf32> to tensor<1x?x64xf32>
  %padded = tensor.pad %es low[0, 0, 0] high[0, %pad_amt, 0] {
    ^bb0(%a: index, %b: index, %c: index):
    tensor.yield %cst : f32
  } : tensor<1x?x64xf32> to tensor<1x?x64xf32>

  return %padded : tensor<1x?x64xf32>
}
