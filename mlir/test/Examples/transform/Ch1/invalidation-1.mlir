// RUN: mlir-opt %s \
// RUN:   --pass-pipeline="builtin.module(test-transform-dialect-interpreter{ \
// RUN:        bind-first-extra-to-ops=linalg.matmul \
// RUN:        bind-second-extra-to-ops=linalg.elemwise_binary \
// RUN:        enable-expensive-checks},canonicalize,cse,symbol-dce)" \
// RUN:   --split-input-file --verify-diagnostics

// ****************************** IMPORTANT NOTE ******************************
//
// If you are changing this file, you may also need to change
// mlir/docs/Tutorials/Transform accordingly.
//
// ****************************************************************************

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op,
     // expected-note @below {{handle to invalidated ops}}
     %arg1: !transform.op<"linalg.matmul">,
     %arg2: !transform.op<"linalg.elemwise_binary">):
  // The actual tiling transformation takes tile sizes as attributes.
  // expected-note @below {{invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them}}
  %tiled, %loop = transform.structured.tile_using_forall %arg1 tile_sizes [4, 32]
      : (!transform.op<"linalg.matmul">) -> (!transform.any_op, !transform.any_op)

  // This is trying to use an invalidated handle leading to undefined behavior.
  // expected-error @below {{uses a handle invalidated by a previously executed transform op}}
  transform.debug.emit_remark_at %arg1, "remark" : !transform.op<"linalg.matmul">
  transform.yield
}

// Original function to optimize.
func.func @fc_relu(%lhs: tensor<512x512xf32>, %rhs: tensor<512x512xf32>,
                   %bias: tensor<512x512xf32>, %output: tensor<512x512xf32>)
                   -> tensor<512x512xf32> {
  // Matrix-matrix multiplication.
  // expected-note @below {{payload op}}
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
                          outs(%output: tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise addition.
  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
    ins(%matmul, %bias : tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise max with 0 (ReLU).
  %c0f = arith.constant 0.0 : f32
  %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
    ins(%biased, %c0f : tensor<512x512xf32>, f32)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>
  func.return %relued : tensor<512x512xf32>
}

// -----

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op,
     %arg1: !transform.op<"linalg.matmul">,
     %arg2: !transform.op<"linalg.elemwise_binary">):
  // We can cast one type to another as long as operations are compatible
  // with both types. This creates "aliasing" handles.
  // expected-note @below {{handle to invalidated ops}}
  %casted = transform.cast %arg1 : !transform.op<"linalg.matmul"> to
      !transform.any_op

  // The actual tiling transformation takes tile sizes as attributes.
  // expected-note @below {{invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them}}
  %tiled, %loop = transform.structured.tile_using_forall %arg1 tile_sizes [4, 32]
    : (!transform.op<"linalg.matmul">) -> (!transform.any_op, !transform.any_op)

  // Consuming an operand invalidates the consumed handle and any other handle that is
  // associated with the same payload operations, or payload operations nested in them.
  // expected-error @below {{uses a handle invalidated by a previously executed transform op}}
  transform.debug.emit_remark_at %casted, "remark"
    : !transform.any_op
  transform.yield
}

// Original function to optimize.
func.func @fc_relu(%lhs: tensor<512x512xf32>, %rhs: tensor<512x512xf32>,
                   %bias: tensor<512x512xf32>, %output: tensor<512x512xf32>)
                   -> tensor<512x512xf32> {
  // Matrix-matrix multiplication.
  // expected-note @below {{payload op}}
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
                          outs(%output: tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise addition.
  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
    ins(%matmul, %bias : tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>

  // Elementwise max with 0 (ReLU).
  %c0f = arith.constant 0.0 : f32
  %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
    ins(%biased, %c0f : tensor<512x512xf32>, f32)
    outs(%output : tensor<512x512xf32>) -> tensor<512x512xf32>
  func.return %relued : tensor<512x512xf32>
}
