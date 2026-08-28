// RUN: mlir-opt --transform-interpreter -canonicalize -split-input-file --verify-diagnostics %s | FileCheck %s

// When padding values are not specified, `pad_tiling_interface` infers them.
// For an operand that is reduced, a zero pad value would corrupt the result,
// so the neutral element of the reduction combiner must be inferred instead.

// CHECK-LABEL: @pad_reduce_maximumf
func.func @pad_reduce_maximumf(%input: tensor<8x30xf32>, %init: tensor<8xf32>)
    -> tensor<8xf32> {
  // maximumf neutral element is -inf.
  // CHECK-DAG:   %[[NEUTRAL:.*]] = arith.constant 0xFF800000 : f32
  // CHECK:       %[[PAD:.*]] = tensor.pad %{{.*}} low[0, 0] high[0, 2]
  // CHECK:         tensor.yield %[[NEUTRAL]] : f32
  // CHECK:       } : tensor<8x30xf32> to tensor<8x32xf32>
  // CHECK:       linalg.reduce ins(%[[PAD]] : tensor<8x32xf32>)
  %0 = linalg.reduce ins(%input : tensor<8x30xf32>) outs(%init : tensor<8xf32>)
      dimensions = [1]
    (%in: f32, %out: f32) {
      %m = arith.maximumf %in, %out : f32
      linalg.yield %m : f32
    }
  return %0 : tensor<8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %red = transform.structured.match ops{["linalg.reduce"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    // No padding_values: they are inferred.
    %padded, %pad = transform.structured.pad_tiling_interface %red to padding_sizes [8, 32]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// An operand reduced along several dimensions is padded with the neutral on all
// of them (here d1 and d2); one neutral suffices since the combiner is the same.

#in3  = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#out3 = affine_map<(d0, d1, d2) -> (d0)>
// CHECK-LABEL: @pad_reduce_two_reduction_dims
func.func @pad_reduce_two_reduction_dims(%in: tensor<8x6x10xf32>, %init: tensor<8xf32>)
    -> tensor<8xf32> {
  // CHECK-DAG:   %[[NINF:.*]] = arith.constant 0xFF800000 : f32
  // CHECK:       tensor.pad %{{.*}} low[0, 0, 0] high[0, 2, 6]
  // CHECK:         tensor.yield %[[NINF]] : f32
  // CHECK:       } : tensor<8x6x10xf32> to tensor<8x8x16xf32>
  %0 = linalg.generic {indexing_maps = [#in3, #out3],
                       iterator_types = ["parallel", "reduction", "reduction"]}
      ins(%in : tensor<8x6x10xf32>) outs(%init : tensor<8xf32>) {
  ^bb0(%a: f32, %o: f32):
    %m = arith.maximumf %a, %o : f32
    linalg.yield %m : f32
  } -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %gen = transform.structured.match ops{["linalg.generic"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %padded, %pad = transform.structured.pad_tiling_interface %gen to padding_sizes [8, 8, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: @pad_reduce_mulf
func.func @pad_reduce_mulf(%input: tensor<8x30xf32>, %init: tensor<8xf32>)
    -> tensor<8xf32> {
  // mulf neutral element is 1.0 (a zero pad would zero out the product).
  // CHECK-DAG:   %[[NEUTRAL:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK:       %[[PAD:.*]] = tensor.pad %{{.*}} low[0, 0] high[0, 2]
  // CHECK:         tensor.yield %[[NEUTRAL]] : f32
  // CHECK:       } : tensor<8x30xf32> to tensor<8x32xf32>
  // CHECK:       linalg.reduce ins(%[[PAD]] : tensor<8x32xf32>)
  %0 = linalg.reduce ins(%input : tensor<8x30xf32>) outs(%init : tensor<8xf32>)
      dimensions = [1]
    (%in: f32, %out: f32) {
      %m = arith.mulf %in, %out : f32
      linalg.yield %m : f32
    }
  return %0 : tensor<8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %red = transform.structured.match ops{["linalg.reduce"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %padded, %pad = transform.structured.pad_tiling_interface %red to padding_sizes [8, 32]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// CHECK-LABEL: @pad_reduce_minimumf
func.func @pad_reduce_minimumf(%input: tensor<8x30xf32>, %init: tensor<8xf32>)
    -> tensor<8xf32> {
  // minimumf neutral element is +inf.
  // CHECK-DAG:   %[[NEUTRAL:.*]] = arith.constant 0x7F800000 : f32
  // CHECK:       %[[PAD:.*]] = tensor.pad %{{.*}} low[0, 0] high[0, 2]
  // CHECK:         tensor.yield %[[NEUTRAL]] : f32
  // CHECK:       } : tensor<8x30xf32> to tensor<8x32xf32>
  // CHECK:       linalg.reduce ins(%[[PAD]] : tensor<8x32xf32>)
  %0 = linalg.reduce ins(%input : tensor<8x30xf32>) outs(%init : tensor<8xf32>)
      dimensions = [1]
    (%in: f32, %out: f32) {
      %m = arith.minimumf %in, %out : f32
      linalg.yield %m : f32
    }
  return %0 : tensor<8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %red = transform.structured.match ops{["linalg.reduce"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %padded, %pad = transform.structured.pad_tiling_interface %red to padding_sizes [8, 32]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// maxnumf is the NaN-ignoring variant, so its neutral element is NaN.

// CHECK-LABEL: @pad_reduce_maxnumf
func.func @pad_reduce_maxnumf(%input: tensor<8x30xf32>, %init: tensor<8xf32>)
    -> tensor<8xf32> {
  // CHECK-DAG:   %[[NAN:.*]] = arith.constant 0xFFC00000 : f32
  // CHECK:       tensor.pad %{{.*}} low[0, 0] high[0, 2]
  // CHECK:         tensor.yield %[[NAN]] : f32
  // CHECK:       } : tensor<8x30xf32> to tensor<8x32xf32>
  %0 = linalg.reduce ins(%input : tensor<8x30xf32>) outs(%init : tensor<8xf32>) dimensions = [1]
    (%in: f32, %out: f32) {
      %m = arith.maxnumf %in, %out : f32
      linalg.yield %m : f32
    }
  return %0 : tensor<8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %red = transform.structured.match ops{["linalg.reduce"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %padded, %pad = transform.structured.pad_tiling_interface %red to padding_sizes [8, 32]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Negative: `nnan` declares that NaN operands do not occur, so the NaN neutral
// element of maxnumf is not a usable pad value and inference fails instead.

func.func @pad_reduce_maxnumf_nnan_fails(%input: tensor<8x30xf32>,
    %init: tensor<8xf32>) -> tensor<8xf32> {
  // expected-note @below {{target op}}
  %0 = linalg.reduce ins(%input : tensor<8x30xf32>) outs(%init : tensor<8xf32>)
      dimensions = [1]
    (%in: f32, %out: f32) {
      %m = arith.maxnumf %in, %out fastmath<nnan> : f32
      linalg.yield %m : f32
    }
  return %0 : tensor<8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %red = transform.structured.match ops{["linalg.reduce"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{failed to pad op}}
    %padded, %pad = transform.structured.pad_tiling_interface %red to padding_sizes [8, 32]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// A contraction is padded with zero: its reduced operands feed the multiply and
// zero annihilates it (0 * x = 0), so the add-reduction stays correct. The
// combiner (arith.addf) must NOT be mistaken for the operands' direct combiner
// (arith.mulf, whose neutral 1 would corrupt the result).

// CHECK-LABEL: @pad_matmul_uses_zero
func.func @pad_matmul_uses_zero(%A: tensor<8x10xf32>, %B: tensor<10x8xf32>,
                                %C: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-DAG:   %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK:       tensor.pad %{{.*}} low[0, 0] high[0, 6]
  // CHECK:         tensor.yield %[[ZERO]] : f32
  // CHECK:       } : tensor<8x10xf32> to tensor<8x16xf32>
  // CHECK:       tensor.pad %{{.*}} low[0, 0] high[6, 0]
  // CHECK:         tensor.yield %[[ZERO]] : f32
  // CHECK:       } : tensor<10x8xf32> to tensor<16x8xf32>
  // CHECK-NOT:   arith.constant 1.000000e+00
  %0 = linalg.matmul ins(%A, %B : tensor<8x10xf32>, tensor<10x8xf32>)
                     outs(%C : tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %padded, %pad = transform.structured.pad_tiling_interface %mm to padding_sizes [8, 8, 16]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Only the reduced input takes the neutral (-inf); the non-reduced init operand
// keeps zero. Both dims are padded so the init is padded along the parallel dim.

// CHECK-LABEL: @pad_reduce_init_stays_zero
func.func @pad_reduce_init_stays_zero(%in: tensor<8x30xf32>, %init: tensor<8xf32>)
    -> tensor<8xf32> {
  // CHECK-DAG:   %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG:   %[[NINF:.*]] = arith.constant 0xFF800000 : f32
  // Reduced input -> maximumf neutral (-inf).
  // CHECK:       tensor.pad %{{.*}}
  // CHECK:         tensor.yield %[[NINF]] : f32
  // CHECK:       } : tensor<8x30xf32> to tensor<16x32xf32>
  // Init (non-reduced) operand -> plain zero.
  // CHECK:       tensor.pad %{{.*}}
  // CHECK:         tensor.yield %[[ZERO]] : f32
  // CHECK:       } : tensor<8xf32> to tensor<16xf32>
  %0 = linalg.reduce ins(%in : tensor<8x30xf32>) outs(%init : tensor<8xf32>)
      dimensions = [1]
    (%a: f32, %o: f32) {
      %m = arith.maximumf %a, %o : f32
      linalg.yield %m : f32
    }
  return %0 : tensor<8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %red = transform.structured.match ops{["linalg.reduce"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %padded, %pad = transform.structured.pad_tiling_interface %red to padding_sizes [16, 32]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Negative: the reduced operand is not consumed directly by a recognized
// reduction combiner (it flows through math.exp first), so no safe neutral
// element can be inferred and padding fails conservatively rather than
// silently padding with a wrong value.

func.func @pad_reduce_indirect_combiner_fails(
    %input: tensor<8x30xf32>, %init: tensor<8xf32>) -> tensor<8xf32> {
  // expected-note @below {{target op}}
  %0 = linalg.reduce ins(%input : tensor<8x30xf32>) outs(%init : tensor<8xf32>)
      dimensions = [1]
    (%in: f32, %out: f32) {
      %e = math.exp %in : f32
      %a = arith.addf %e, %out : f32
      linalg.yield %a : f32
    }
  return %0 : tensor<8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %red = transform.structured.match ops{["linalg.reduce"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{failed to pad op}}
    %padded, %pad = transform.structured.pad_tiling_interface %red to padding_sizes [8, 32]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Negative: a fused op with two reductions (max and sum) over the same input
// has no single correct neutral for that input (-inf breaks the sum, 0 breaks
// the max), so inference fails instead of padding with a wrong value.

#map  = affine_map<(d0, d1) -> (d0, d1)>
#mapr = affine_map<(d0, d1) -> (d0)>
func.func @pad_multi_reduction_fails(%in: tensor<8x30xf32>, %m0: tensor<8xf32>,
                                     %s0: tensor<8xf32>)
    -> (tensor<8xf32>, tensor<8xf32>) {
  // expected-note @below {{target op}}
  %r:2 = linalg.generic {indexing_maps = [#map, #mapr, #mapr],
                         iterator_types = ["parallel", "reduction"]}
      ins(%in : tensor<8x30xf32>) outs(%m0, %s0 : tensor<8xf32>, tensor<8xf32>) {
  ^bb0(%x: f32, %mo: f32, %so: f32):
    %mx = arith.maximumf %x, %mo : f32
    %sm = arith.addf %x, %so : f32
    linalg.yield %mx, %sm : f32, f32
  } -> (tensor<8xf32>, tensor<8xf32>)
  return %r#0, %r#1 : tensor<8xf32>, tensor<8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %gen = transform.structured.match ops{["linalg.generic"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{failed to pad op}}
    %padded, %pad = transform.structured.pad_tiling_interface %gen to padding_sizes [8, 32]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
