// RUN: mlir-opt %s -test-linalg-transform-patterns="test-softmax-matmul-fusion-rewrite softmax-matmul-fusion-tile-size=32" -split-input-file | FileCheck %s

// Test basic softmax -> matmul pattern match and rewrite.

// CHECK-LABEL: func.func @softmax_matmul_basic
// CHECK-SAME: (%[[Q:.*]]: tensor<4x16xf32>, %[[KT:.*]]: tensor<16x128xf32>, %[[V:.*]]: tensor<128x64xf32>)
// CHECK: %[[S:.*]] = linalg.matmul ins(%[[Q]], %[[KT]] : tensor<4x16xf32>, tensor<16x128xf32>) outs({{.*}}) -> tensor<4x128xf32>
// CHECK: %[[LOCAL_SOFTMAX:.*]]:3 = linalg.local_softmax dimension(1) tile_size(32)
// CHECK-SAME: ins(%[[S]] : tensor<4x128xf32>)
// CHECK-SAME: -> tensor<4x4x32xf32>, tensor<4x4xf32>, tensor<4x4xf32>
// CHECK: tensor.expand_shape %[[V]] {{\[\[}}0, 1], [2]] output_shape [4, 32, 64] : tensor<128x64xf32> into tensor<4x32x64xf32>
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction", "reduction", "parallel"]
// CHECK: ^bb0({{.*}}: f32, {{.*}}: f32, {{.*}}: f32, {{.*}}: f32, {{.*}}: f32, {{.*}}: f32, {{.*}}: f32):
// CHECK:   arith.maximumf
// CHECK:   arith.subf
// CHECK:   math.exp
// CHECK:   arith.mulf
// CHECK:   arith.subf
// CHECK:   math.exp
// CHECK:   arith.mulf
// CHECK:   arith.mulf
// CHECK:   arith.addf
// CHECK:   arith.divf
// CHECK:   arith.mulf
// CHECK:   arith.mulf
// CHECK:   arith.divf
// CHECK:   arith.addf
// CHECK:   linalg.yield
// CHECK-NOT: linalg.matmul ins(%{{.*}}, %[[V]]
func.func @softmax_matmul_basic(%Q: tensor<4x16xf32>, %KT: tensor<16x128xf32>, %V: tensor<128x64xf32>) -> tensor<4x64xf32> {
  %S_init = tensor.empty() : tensor<4x128xf32>
  %zero = arith.constant 0.0 : f32
  %S_fill = linalg.fill ins(%zero : f32) outs(%S_init : tensor<4x128xf32>) -> tensor<4x128xf32>
  %S = linalg.matmul ins(%Q, %KT : tensor<4x16xf32>, tensor<16x128xf32>) outs(%S_fill : tensor<4x128xf32>) -> tensor<4x128xf32>

  %softmax_init = tensor.empty() : tensor<4x128xf32>
  %softmax = linalg.softmax dimension(1) ins(%S : tensor<4x128xf32>) outs(%softmax_init : tensor<4x128xf32>) -> tensor<4x128xf32>

  %O_init = tensor.empty() : tensor<4x64xf32>
  %O_fill = linalg.fill ins(%zero : f32) outs(%O_init : tensor<4x64xf32>) -> tensor<4x64xf32>
  %O = linalg.matmul ins(%softmax, %V : tensor<4x128xf32>, tensor<128x64xf32>) outs(%O_fill : tensor<4x64xf32>) -> tensor<4x64xf32>

  return %O : tensor<4x64xf32>
}

// -----

// Negative test: softmax with no matmul user should not be rewritten.

// CHECK-LABEL: func.func @softmax_no_matmul_user
// CHECK: linalg.softmax
// CHECK-NOT: linalg.local_softmax
func.func @softmax_no_matmul_user(%input: tensor<4x128xf32>) -> tensor<4x128xf32> {
  %output_init = tensor.empty() : tensor<4x128xf32>
  %result = linalg.softmax dimension(1) ins(%input : tensor<4x128xf32>) outs(%output_init : tensor<4x128xf32>) -> tensor<4x128xf32>
  return %result : tensor<4x128xf32>
}

// -----

// Negative test: softmax dim does not match matmul contraction dim.
// Here softmax is along dim 0 but matmul contracts dim 1 (the last dim of LHS).

// CHECK-LABEL: func.func @softmax_wrong_dim
// CHECK: linalg.softmax
// CHECK: linalg.matmul
// CHECK-NOT: linalg.local_softmax
func.func @softmax_wrong_dim(%input: tensor<128x4xf32>, %V: tensor<4x64xf32>) -> tensor<128x64xf32> {
  %softmax_init = tensor.empty() : tensor<128x4xf32>
  %softmax = linalg.softmax dimension(0) ins(%input : tensor<128x4xf32>) outs(%softmax_init : tensor<128x4xf32>) -> tensor<128x4xf32>

  %O_init = tensor.empty() : tensor<128x64xf32>
  %zero = arith.constant 0.0 : f32
  %O_fill = linalg.fill ins(%zero : f32) outs(%O_init : tensor<128x64xf32>) -> tensor<128x64xf32>
  %O = linalg.matmul ins(%softmax, %V : tensor<128x4xf32>, tensor<4x64xf32>) outs(%O_fill : tensor<128x64xf32>) -> tensor<128x64xf32>

  return %O : tensor<128x64xf32>
}

// -----

// Negative test: N not divisible by tile_size (128 is divisible by 32, but 96 is not for tile_size=32... wait, 96/32=3, so it IS divisible).
// Let's use N=100 which is not divisible by 32.

// CHECK-LABEL: func.func @softmax_not_divisible
// CHECK: linalg.softmax
// CHECK: linalg.matmul
// CHECK-NOT: linalg.local_softmax
func.func @softmax_not_divisible(%input: tensor<4x100xf32>, %V: tensor<100x64xf32>) -> tensor<4x64xf32> {
  %softmax_init = tensor.empty() : tensor<4x100xf32>
  %softmax = linalg.softmax dimension(1) ins(%input : tensor<4x100xf32>) outs(%softmax_init : tensor<4x100xf32>) -> tensor<4x100xf32>

  %O_init = tensor.empty() : tensor<4x64xf32>
  %zero = arith.constant 0.0 : f32
  %O_fill = linalg.fill ins(%zero : f32) outs(%O_init : tensor<4x64xf32>) -> tensor<4x64xf32>
  %O = linalg.matmul ins(%softmax, %V : tensor<4x100xf32>, tensor<100x64xf32>) outs(%O_fill : tensor<4x64xf32>) -> tensor<4x64xf32>

  return %O : tensor<4x64xf32>
}

// -----

// Test: softmax with multiple users emits rescaling_softmax to recover global softmax.

// CHECK-LABEL: func.func @softmax_multiple_users
// CHECK: linalg.local_softmax dimension(1) tile_size(32)
// Three linalg.generic ops: rescaling_matmul, identity matrix, rescaling_softmax
// CHECK: linalg.generic {
// CHECK: linalg.generic {
// CHECK: linalg.generic {
func.func @softmax_multiple_users(%input: tensor<4x128xf32>, %V: tensor<128x64xf32>) -> (tensor<4x64xf32>, tensor<4x128xf32>) {
  %softmax_init = tensor.empty() : tensor<4x128xf32>
  %softmax = linalg.softmax dimension(1) ins(%input : tensor<4x128xf32>) outs(%softmax_init : tensor<4x128xf32>) -> tensor<4x128xf32>

  %O_init = tensor.empty() : tensor<4x64xf32>
  %zero = arith.constant 0.0 : f32
  %O_fill = linalg.fill ins(%zero : f32) outs(%O_init : tensor<4x64xf32>) -> tensor<4x64xf32>
  %O = linalg.matmul ins(%softmax, %V : tensor<4x128xf32>, tensor<128x64xf32>) outs(%O_fill : tensor<4x64xf32>) -> tensor<4x64xf32>

  // The softmax result is also used directly (e.g., for backward pass)
  return %O, %softmax : tensor<4x64xf32>, tensor<4x128xf32>
}
