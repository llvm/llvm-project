// RUN: mlir-opt %s \
// RUN:   --test-linalg-transform-patterns="test-softmax-matmul-fusion-rewrite softmax-matmul-fusion-tile-size=32" \
// RUN:   --canonicalize --cse | FileCheck %s

// Test: softmax with multiple users triggers rescaling_softmax emission.
// The softmax result is used by both a matmul AND returned directly.
// The rewrite should produce:
//   - expand_shape + 4 generics (local softmax for P, m, l)
//   - rescaling_matmul generic (replaces the matmul)
//   - 2 generics + collapse_shape (recovers global softmax for the other user)
// No identity matrix should be materialized.

// CHECK-LABEL: func.func @softmax_multi_user
//
// Local softmax generics (max, exp, sum — no per-tile divide; matmul consumes
// the unnormalized num. l (sum) is kept for the recovery path below):
// CHECK: tensor.expand_shape
// CHECK: linalg.generic
// CHECK: linalg.generic
// CHECK: linalg.generic
//
// Second GEMM, emitted split so the matmul is a standalone contraction:
//   op1 (pv): contraction over ts (tn parallel); op2: recurrence over tn.
// CHECK: tensor.expand_shape
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction", "parallel"]
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction", "parallel"]
// CHECK:   arith.maximumf
//
// Final divide O = O / L (elementwise):
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel"]
// CHECK:   arith.divf
//
// Rescaling softmax — Generic 1: reduce m, l over tn to get M_global, L_global
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK:   arith.maxnumf
//
// Rescaling softmax — Generic 2: elementwise correction of P
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]
// CHECK:   math.exp
// CHECK:   arith.divf
//
// collapse_shape merges [tn, ts] back to [N]
// CHECK: tensor.collapse_shape
// CHECK-SAME: tensor<4x4x32xf32> into tensor<4x128xf32>
//
// No identity matrix (no tensor of shape [N, N] or [tn, ts, N])
// CHECK-NOT: tensor<128x128xf32>
// CHECK-NOT: tensor<4x32x128xf32>
//
// CHECK: return

func.func @softmax_multi_user(%input : tensor<4x128xf32>, %V : tensor<128x64xf32>) -> (tensor<4x64xf32>, tensor<4x128xf32>) {
  %softmax_init = tensor.empty() : tensor<4x128xf32>
  %softmax = linalg.softmax dimension(1) ins(%input : tensor<4x128xf32>) outs(%softmax_init : tensor<4x128xf32>) -> tensor<4x128xf32>
  %O_init = tensor.empty() : tensor<4x64xf32>
  %O = linalg.matmul ins(%softmax, %V : tensor<4x128xf32>, tensor<128x64xf32>) outs(%O_init : tensor<4x64xf32>) -> tensor<4x64xf32>
  return %O, %softmax : tensor<4x64xf32>, tensor<4x128xf32>
}
