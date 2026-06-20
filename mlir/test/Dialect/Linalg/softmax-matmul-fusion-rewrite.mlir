// RUN: mlir-opt %s -test-linalg-transform-patterns="test-softmax-matmul-fusion-rewrite softmax-matmul-fusion-tile-size=32" -split-input-file | FileCheck %s

// Test basic softmax -> matmul pattern match and rewrite (generic-only, no local_softmax).

// CHECK-LABEL: func.func @softmax_matmul_basic
// After rewrite: expand_shape + local-softmax generics (m = max, num = exp(S-m)),
// then the second GEMM is emitted SPLIT into three ops so the matmul is a
// standalone contraction:
//   op1  pv   = sum_ts num * V    (ts reduction, tn parallel) -> vector.contract
//   op1b lsum = sum_ts num        (ts reduction)
//   op2  online recurrence over tn (running max + rescale; consumes pv, m, lsum)
//   op3  final divide O = O / L
// This deviates from Triton's dot(exp(qk - m_ij), v): op1 contracts the
// local-max-shifted num (loop-invariant), and the running-max correction beta is
// applied in op2 via sum_ts(num*beta*V) = beta*sum_ts(num*V). Same result.
// CHECK: linalg.matmul
// CHECK: tensor.expand_shape {{.*}} tensor<4x128xf32> into tensor<4x4x32xf32>
// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "parallel", "reduction"]
// CHECK:   arith.maxnumf
// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "parallel", "parallel"]
// CHECK:   math.exp
// CHECK: tensor.expand_shape {{.*}} tensor<128x64xf32> into tensor<4x32x64xf32>
// op1: per-tile matmul (pure contraction) — tn parallel, ts reduction.
// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "parallel", "reduction", "parallel"]
// CHECK:   arith.mulf
// CHECK:   arith.addf
// op1b: per-tile denominator sum over ts.
// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "parallel", "reduction"]
// CHECK:   arith.addf
// op2: online recurrence over tn (running max + rescale, no matmul).
// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "reduction", "parallel"]
// CHECK:   arith.maximumf
// op3: final divide O = O / L.
// CHECK: linalg.generic {{.*}} iterator_types = ["parallel", "parallel"]
// CHECK:   arith.divf
// CHECK-NOT: linalg.softmax
func.func @softmax_matmul_basic(%Q : tensor<4x16xf32>, %K_T : tensor<16x128xf32>, %V : tensor<128x64xf32>) -> tensor<4x64xf32> {
  %S_init = tensor.empty() : tensor<4x128xf32>
  %S = linalg.matmul ins(%Q, %K_T : tensor<4x16xf32>, tensor<16x128xf32>) outs(%S_init : tensor<4x128xf32>) -> tensor<4x128xf32>
  %softmax_init = tensor.empty() : tensor<4x128xf32>
  %softmax = linalg.softmax dimension(1) ins(%S : tensor<4x128xf32>) outs(%softmax_init : tensor<4x128xf32>) -> tensor<4x128xf32>
  %O_init = tensor.empty() : tensor<4x64xf32>
  %O = linalg.matmul ins(%softmax, %V : tensor<4x128xf32>, tensor<128x64xf32>) outs(%O_init : tensor<4x64xf32>) -> tensor<4x64xf32>
  return %O : tensor<4x64xf32>
}

// -----

// Negative test: softmax with no matmul user — should not transform.
// CHECK-LABEL: func.func @softmax_no_matmul_user
// CHECK: linalg.softmax
func.func @softmax_no_matmul_user(%input : tensor<4x128xf32>) -> tensor<4x128xf32> {
  %init = tensor.empty() : tensor<4x128xf32>
  %result = linalg.softmax dimension(1) ins(%input : tensor<4x128xf32>) outs(%init : tensor<4x128xf32>) -> tensor<4x128xf32>
  return %result : tensor<4x128xf32>
}

// -----

// Negative test: N not divisible by tile_size — should not transform.
// CHECK-LABEL: func.func @softmax_not_divisible
// CHECK: linalg.softmax
func.func @softmax_not_divisible(%input : tensor<4x100xf32>, %V : tensor<100x64xf32>) -> tensor<4x64xf32> {
  %softmax_init = tensor.empty() : tensor<4x100xf32>
  %softmax = linalg.softmax dimension(1) ins(%input : tensor<4x100xf32>) outs(%softmax_init : tensor<4x100xf32>) -> tensor<4x100xf32>
  %O_init = tensor.empty() : tensor<4x64xf32>
  %O = linalg.matmul ins(%softmax, %V : tensor<4x100xf32>, tensor<100x64xf32>) outs(%O_init : tensor<4x64xf32>) -> tensor<4x64xf32>
  return %O : tensor<4x64xf32>
}
