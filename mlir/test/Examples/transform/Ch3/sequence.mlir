// RUN: transform-opt-ch2 %s \
// RUN:   --pass-pipeline="builtin.module(test-transform-dialect-interpreter{ \
// RUN:        bind-first-extra-to-ops=linalg.matmul \
// RUN:        bind-second-extra-to-ops=linalg.elemwise_binary \
// RUN:        enable-expensive-checks},canonicalize,cse,symbol-dce)" |\
// RUN: FileCheck %s

// ****************************** IMPORTANT NOTE ******************************
//
// If you are changing this file, you may also need to change
// mlir/docs/Tutorials/Transform accordingly.
//
// ****************************************************************************

// Original function to optimize.
func.func @fc_relu(%lhs: tensor<512x512xf32>, %rhs: tensor<512x512xf32>,
                   %bias: tensor<512x512xf32>, %output: tensor<512x512xf32>)
                   -> tensor<512x512xf32> {
  // Matrix-matrix multiplication.
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

// CHECK-LABEL: func @fc_relu
// CHECK: scf.forall
// CHECK:   scf.forall
// CHECK:     %[[SLICE4:.+]] = tensor.extract_slice
// CHECK:     %[[SLICE5:.+]] = tensor.extract_slice
// CHECK:     %[[SLICE6:.+]] = tensor.extract_slice
// CHECK:     %[[SLICE7:.+]] = tensor.extract_slice
// CHECK:     %[[SLICE8:.+]] = tensor.extract_slice
// CHECK:     func.call @microkernel(%[[SLICE4]], %[[SLICE5]], %[[SLICE6]], %[[SLICE7]], %[[SLICE8]])
// CHECK-NOT: linalg.matmul
// CHECK-NOT: linalg.elemwise_binary
// CHECK:     scf.forall.in_parallel
// CHECK:   linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>}
// CHECK:   scf.forall.in_parallel

// Declaration of the "microkernel" function that we will be targeting.
func.func private @microkernel(
    %lhs: tensor<4x512xf32>,
    %rhs: tensor<512x4xf32>,
    %bias: tensor<4x4xf32>,
    %init: tensor<4x4xf32>,
    %output: tensor<4x4xf32>) -> tensor<4x4xf32>

transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op,
     %arg1: !transform.op<"linalg.matmul">,
     %arg2: !transform.op<"linalg.elemwise_binary">):
  // Since the %arg2 handle is associated with both elementwise operations,
  // we need to split it into two handles so we can target only the second
  // elementwise operation.
  %add, %max = transform.split_handle %arg2 : (!transform.op<"linalg.elemwise_binary">)
      -> (!transform.any_op, !transform.any_op)

  // The actual tiling transformation takes tile sizes as attributes. It produces a
  // handle to the loop generated during tiling.
  %loop, %tiled = transform.structured.tile_to_forall_op %max tile_sizes [8, 32]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // We can now fuse the other operations into the loop. Here, we fuse
  // operations one-by-one. This requires the operation that is being fused
  // to define the value used within the loop, so the order of such fusions
  // is important. We could also use "transform.merge_handles" to obtain
  // a single handle to all operations and give it to `fuse_into_containing_op`
  // that would take care of the ordering in this case.
  %add_fused, %loop2 = transform.structured.fuse_into_containing_op %add into %loop
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
  %matmul_fused, %loop3 = transform.structured.fuse_into_containing_op %arg1 into %loop2
      : (!transform.op<"linalg.matmul">, !transform.any_op) -> (!transform.any_op, !transform.any_op)

  // Tile again to get the desired size. Note that this time this tiles the
  // "add" operation and fuses matmul into the loop, but doesn't affect the
  // "max" operation. This illustrates the precise targeting with the transform
  // dialect. Otherwise, it is difficult to differentiate "add" and "max", both
  // of which having the same kind.
  %loop_second, %tiled_second = transform.structured.tile_to_forall_op %add_fused tile_sizes [4, 4]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %matmul_fused_2, %loop_second_2 =
      transform.structured.fuse_into_containing_op %matmul_fused into %loop_second
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

  // Since outlining is currently only implemented for region-holding operations
  // such as loops, use tiling to size 1 to materialize the outer loop that is
  // going to be outlined.
  %loop_third, %_0 = transform.structured.tile_to_forall_op %tiled_second tile_sizes [1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %_1, %outline_target = transform.structured.fuse_into_containing_op %matmul_fused_2 into %loop_third
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
  %func, %call = transform.loop.outline %outline_target {func_name = "outlined"}
      : (!transform.any_op) -> (!transform.any_op, !transform.op<"func.call">)

  // Rewrite the call target.
  transform.my.change_call_target %call, "microkernel" : !transform.op<"func.call">

  transform.yield
}
