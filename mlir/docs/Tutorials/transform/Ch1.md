# Chapter 1: Combining Existing Transformations

## Introduction

The Transform dialect allows one to precisely target transformations at specific operations in the IR and to chain them, that is to apply a transformation to operations produced by the previous transformation. To achieve this, transformations are expressed as other operations in the IR. We call these the IR containing these operations transform IR. And we call the IR that is being transformed payload IR.

Transform IR operations operate on values that may be associated with payload IR operations, values or attributes. We call the first two kinds of values operation and value handles, respectively. We call the last kind of values parameters.

The application of transform IR always starts from one top-level operation. In the C++ API, this operation is passed to the `applyTransforms` function. This top-level operation specifies if other transformations should be performed and how. The most common top-level operation merely applies other transform operations listed in its body one after the other.

Let us illustrate this with a simple sequence of transformations on the common “fully connected + bias + ReLU” ML layer, which boils down to performing a matrix multiplication, followed by an (elementwise) matrix addition and taking an elementwise maximum with 0. This can be expressed using the following IR:

```mlir
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
```

## Top-Level Sequence Operation

For performance reasons, we would like to tile and fuse these operations to exploit cache locality. This is a sequence of transformations that need to be performed one after another, so we naturally start with the corresponding top-level transform operation.

```mlir
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op,
     %arg1: !transform.op<"linalg.matmul">,
     %arg2: !transform.op<"linalg.elemwise_binary">):
  transform.yield
}
```

There are several aspects worth noticing in this operation.

The first entry block argument is mandatory for top-level transform operations and is associated with the top-level payload operation that sequence is applied to, for example, a module or a function. This operation is specified when calling `applyTransforms`.

The remaining entry block arguments are optional and can be associated with payload attributes, operations or values that are useful in the sequence. These are also specified when calling `applyTransforms`. In our case, we are interested in the matrix multiplication and elementwise operations that we are going to tile and fuse.

All value handles have Transform dialect types. These types specify certain properties of the payload IR entities associated with them. In this example, `transform.any_op` indicates that the handle is associated with arbitrary payload operations. On the contrary, `transform.op<"X">` indicates that the handle is associated _only_ with payload operations of kind `X`. These constraints are verified when the handle/payload association is created. For entry block arguments of top-level transform operations, this happens early in the `applyTransforms` function. If the constraints are not satisfied, the transform application fails and produces diagnostics for the user.

## Failure Propagation

Speaking about diagnostics, the `sequence` operation itself has a mandatory attribute specifying the failure propagation mode. There are two options:

*   “propagate” makes the sequence transformation fail if any of the nested transformation fails;
*   “suppress” makes the sequence succeed even if one of the nested transformations fails, but without attempting to perform the transformations following the failed one in the sequence.

This latter allows the transformation to continue despite (recoverable) errors. As we are only building the transformation, it is preferable to propagate failures so we know when something did not apply.

To check or debug a transform sequence, it is possible to print various entities associated with the transform IR values. For example, we can print the operations associated with the handles:

```mlir
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op,
     %arg1: !transform.op<"linalg.matmul">,
     %arg2: !transform.op<"linalg.elemwise_binary">):
  transform.test_print_remark_at_operand %arg1, "matmul"
      : !transform.op<"linalg.matmul">
  transform.test_print_remark_at_operand %arg2, "elemwise_binaries"
      : !transform.op<"linalg.elemwise_binary">
  transform.yield
}
```

## Transform Dialect Interpreter

Since we don’t want to recompile the compiler every time we change a transformation, we can use a Transform dialect interpreter pass to apply this transformation sequence to the payload IR. As we will see in the next chapter, it is possible to define custom passes or even integrate the transform interpreter into a larger pass. For now, we can use the existing test pass:


```sh
$ mlir-opt matmul.mlir --pass-pipeline="
    builtin.module(test-transform-dialect-interpreter{
        bind-first-extra-to-ops=linalg.matmul
        bind-second-extra-to-ops=linalg.elemwise_binary})"
```

The `matmul.mlir` file contains _both_ the payload IR function _and_ the transform IR sequence nested in the same module. The transform interpreter will find the first top-level transform operation in the root operation of the pass (the module in our case) and apply it to that root operation. In our case, we also asked the interpreter pass to associate the two extra arguments of the top-level sequence with all `linalg.matmul` and `linalg.elemwise_binary` payload operations through the respective pass options. Running this pass results in the expected remarks:

```sh
matmul.mlir:7:13: remark: matmul
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
            ^
matmul.mlir:7:13: note: see current operation: %0 = linalg.matmul ins(%arg0, %arg1 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
matmul.mlir:10:13: remark: elemwise_binaries
  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
            ^
matmul.mlir:10:13: note: see current operation: %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>} ins(%0, %arg2 : tensor<512x512xf32>, tensor<512x512xf32>) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
matmul.mlir:14:13: remark: elemwise_binaries
  %relued = linalg.elemwise_binary { fun = #linalg.binary_fn<max_signed> }
            ^
matmul.mlir:14:13: note: see current operation: %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>} ins(%1, %cst : tensor<512x512xf32>, f32) outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
```

Note that `%arg2` is associated with both elementwise payload operations. Any handle is associated with a list of entities. Individual transformations may or may not care about the order of elements in that list.


## Specifying Transformations

Now that we have handles to the operations we want to transform, we are ready to apply the transformations. Let us first try tiling the matmul operation itself.

```mlir
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op,
     %arg1: !transform.op<"linalg.matmul">,
     %arg2: !transform.op<"linalg.elemwise_binary">):
  // The actual tiling transformation takes tile sizes as attributes.
  %loop, %tiled = transform.structured.tile_using_forall %arg1 tile_sizes [4, 32]
    : (!transform.op<"linalg.matmul">) -> (!transform.any_op, !transform.any_op)
  transform.yield
}
```

The transformation returns two handles, as indicated in its [documentation](https://mlir.llvm.org/docs/Dialects/Transform/#transformstructuredtile_using_forall-transformtiletoforallop):

*   A handle to the `scf.forall` “multi-for” loop around tensors.
*   A handle to `linalg.generic` operating on the subset of the original data.

Running this transformation with the same command as above expectedly produces the tiled code.

```mlir
func.func @fc_relu(%arg0: tensor<512x512xf32>, %arg1: tensor<512x512xf32>, %arg2: tensor<512x512xf32>, %arg3: tensor<512x512xf32>) -> tensor<512x512xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = scf.forall (%arg4, %arg5) in (128, 16) shared_outs(%arg6 = %arg3) -> (tensor<512x512xf32>) {
    %3 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg4)
    %4 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg5)
    %extracted_slice = tensor.extract_slice %arg0[%3, 0] [4, 512] [1, 1]
                     : tensor<512x512xf32> to tensor<4x512xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[0, %4] [512, 32] [1, 1]
                       : tensor<512x512xf32> to tensor<512x32xf32>
    %extracted_slice_1 = tensor.extract_slice %arg6[%3, %4] [4, 32] [1, 1]
                      : tensor<512x512xf32> to tensor<4x32xf32>
    %5 = linalg.matmul 
         ins(%extracted_slice, %extracted_slice_0
             : tensor<4x512xf32>, tensor<512x32xf32>)
         outs(%extracted_slice_1 : tensor<4x32xf32>) -> tensor<4x32xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %5 into %arg6[%3, %4] [4, 32] [1, 1]
          : tensor<4x32xf32> into tensor<512x512xf32>
    }
  }
  %1 = linalg.elemwise_binary {fun = #linalg.binary_fn<add>}
    ins(%0, %arg2 : tensor<512x512xf32>, tensor<512x512xf32>)
    outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
  %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<max_signed>}
    ins(%1, %cst : tensor<512x512xf32>, f32)
    outs(%arg3 : tensor<512x512xf32>) -> tensor<512x512xf32>
  return %2 : tensor<512x512xf32>
}
```

Besides producing new handles, the tiling transform operation _consumes_ the operand handle. This means that the handle is _invalidated_ after this operation, and is no longer supposed to be used. Transform operations are required to mark all their operands as either consumed or readonly. Transform operations usually consume the operand if the associated payload operations are erased or recreated (which means erased and created anew with similar structure). As handles are essentially references to payload operations, they would become dangling if the payload no longer exists.


## Handle Invalidation and Expensive Checks Mode

Undefined behavior is difficult to grapple with when it does happen, so the Transform dialect interpreter provides a set of additional expensive checks that detect most undefined behavior in the transform IR. For example, if we wanted to  use the `%arg1` handle after it is consumed, it would cause undefined behavior that manifests as an assertion in the debug build, and likely as a segmentation fault in the release mode.

```mlir
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op,
     %arg1: !transform.op<"linalg.matmul">,
     %arg2: !transform.op<"linalg.elemwise_binary">):
  // The actual tiling transformation takes tile sizes as attributes.
  %loop, %tiled = transform.structured.tile_using_forall %arg1 tile_sizes [4, 32]
      : (!transform.op<"linalg.matmul">) -> (!transform.any_op, !transform.any_op)

  // This is trying to use an invalidated handle leading to undefined behavior.
  transform.test_print_remark_at_operand %arg1, "remark" : !transform.op<"linalg.matmul">
  transform.yield
}
```

However, with the expensive checks enabled in the interpreter, a nice diagnostic is produced:

```sh
$ mlir-opt matmul.mlir --pass-pipeline="
    builtin.module(test-transform-dialect-interpreter{
        bind-first-extra-to-ops=linalg.matmul
        bind-second-extra-to-ops=linalg.elemwise_binary
        enable-expensive-checks})"
```

```sh
matmul.mlir:28:3: error: op uses a handle invalidated by a previously executed transform op
  transform.test_print_remark_at_operand %mm, "elemwise_binaries" : !transform.any_op
  ^
matmul.mlir:26:9: note: handle to invalidated ops
  %mm = transform.cast %matmul : !transform.op<"linalg.matmul"> to !transform.any_op
        ^
matmul.mlir:27:19: note: invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them
  %loop, %tiled = transform.structured.tile_using_forall %mm tile_sizes [4, 32]
```

One may observe that some operations such as `transform.cast` do not consume the operand (because they don’t erase the corresponding operation). So what would happen if we tried to use that operand instead? 

```mlir
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op,
     %arg1: !transform.op<"linalg.matmul">,
     %arg2: !transform.op<"linalg.elemwise_binary">):
  // We can cast one type to another as long as operations are compatible
  // with both types. This creates "aliasing" handles.
  %casted = transform.cast %arg1 : !transform.op<"linalg.matmul">
      to !transform.any_op

  // The actual tiling transformation takes tile sizes as attributes.
  %loop, %tiled = transform.structured.tile_using_forall %arg1 tile_sizes [4, 32]
    : (!transform.op<"linalg.matmul">) -> (!transform.any_op, !transform.any_op)

  // Consuming an operand invalidates the consumed handle and any other handle that is
  // associated with the same payload operations, or payload operations nested in them.
  transform.test_print_remark_at_operand %casted, "remark"
    : !transform.any_op
  transform.yield
}
```

Both `%arg1` and `%casted` reference the same payload operation. Extending the reference analogy, these references alias. Naturally, when the payload operation is erased, all references to it become dangling. This is also the case for handles. In fact, consuming an operand invalidates the operand handle as well as any other handle that is associated with any of the same payload operations. The payload IR consideration is recursive: a handle associated with a payload operation _nested_ in the erased one is also invalidated (because erasing the operation also erases its regions and all contained operations). The expensive-checks mode can also handle this case.

```sh
matmul.mlir:28:3: error: op uses a handle invalidated by a previously executed transform op
  transform.test_print_remark_at_operand %matmul, "elemwise_binaries" : !transform.op<"linalg.matmul">
  ^
matmul.mlir:21:29: note: handle to invalidated ops
^bb0(%root: !transform.any_op, %matmul: !transform.op<"linalg.matmul">, %elemwise: !transform.op<"linalg.elemwise_binary">):
                            ^
matmul.mlir:27:19: note: invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them
  %loop, %tiled = transform.structured.tile_using_forall %mm tile_sizes [4, 32]
```

## Chaining Transformations with Handles

Going back to the transformation sequence, we have tiled the matrix multiplication, but we also want to tile and fuse the elementwise operations. The typical way of doing in the structured operations paradigm is to tile the last operation in some acyclic dataflow graph, and then progressively fuse the operations that produce its operands. This removes the need to explicitly tile all operations as fusion can adapt their sizes and inject recomputation if desired. So instead of tiling the matmul operation, we are going to tile the last operation in the chain, and then fuse the preceding operations into the loops produced by tiling.

```mlir
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op,
     %arg1: !transform.op<"linalg.matmul">,
     %arg2: !transform.op<"linalg.elemwise_binary">):
  // Since the %arg2 handle is associated with both elementwise operations,
  // we need to split it into two handles so we can target only the second
  // elementwise operation.
  %add, %max = transform.split_handle %arg2
      : (!transform.op<"linalg.elemwise_binary">)
      -> (!transform.any_op, !transform.any_op)

  // The actual tiling transformation takes tile sizes as attributes. It
  // produces a handle to the loop generated during tiling.
  %tiled_max, %loop =
      transform.structured.tile_using_forall %max tile_sizes [8, 32]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // We can now fuse the other operations into the loop. Here, we fuse
  // operations one by one. This requires the operation that is being fused to
  // define the value used within the loop, so the order of such fusions is
  // important. We could also use "transform.merge_handles" to obtain a single
  // handle to all operations and give it to `fuse_into_containing_op` that
  // would take care of the ordering in this case.
  %add_fused, %loop_0 =
      transform.structured.fuse_into_containing_op %add into %loop
        : (!transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op)
  %matmul_fused, %loop_1 =
      transform.structured.fuse_into_containing_op %arg1 into %loop_0
        : (!transform.op<"linalg.matmul">, !transform.any_op)
          -> (!transform.any_op, !transform.any_op)

  transform.yield
}
```

This achieves the desired tiling and fusion.

## More Handle Invalidation

Finally, let us assume there exists an efficient microkernel, or a hardware instruction expressed as an intrinsic function, for a 4x4 matrix multiplication. For this purpose, we need to tile the fused operation to the desired size, and then outline it. The resulting function call can then be replaced with a call to the microkernel.

```mlir
transform.sequence failures(propagate) {
^bb0(%arg0: !transform.any_op,
     %arg1: !transform.op<"linalg.matmul">,
     %arg2: !transform.op<"linalg.elemwise_binary">):
  // Since the %arg2 handle is associated with both elementwise operations,
  // we need to split it into two handles so we can target only the second
  // elementwise operation.
  %add, %max = transform.split_handle %arg2
      : (!transform.op<"linalg.elemwise_binary">)
        -> (!transform.any_op, !transform.any_op)

  // The actual tiling transformation takes tile sizes as attributes. It
  // produces a handle to the loop generated during tiling.
  %tiled, %loop  = transform.structured.tile_using_forall %max tile_sizes [8, 32]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // We can now fuse the other operations into the loop. Here, we fuse
  // operations one by one. This requires the operation that is being fused to
  // define the value used within the loop, so the order of such fusions is
  // important. We could also use "transform.merge_handles" to obtain a single
  // handle to all operations and give it to `fuse_into_containing_op` that
  // would take care of the ordering in this case.
  %add_fused, %loop_0 =
      transform.structured.fuse_into_containing_op %add into %loop
        : (!transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op)
  %matmul_fused, %loop_1 =
      transform.structured.fuse_into_containing_op %arg1 into %loop_0
        : (!transform.op<"linalg.matmul">, !transform.any_op)
          -> (!transform.any_op, !transform.any_op)

  // Tile again to get the desired size. Note that this time this tiles the
  // "add" operation and fuses matmul into the loop, but doesn't affect the
  // "max" operation. This illustrates the precise targeting with the transform
  // dialect. Otherwise, it is difficult to differentiate "add" and "max", both
  // of which having the same kind.
  %tiled_2, %loop_2 =
      transform.structured.tile_using_forall %add_fused tile_sizes [4, 4]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %matmul_fused_2, %loop_3 =
      transform.structured.fuse_into_containing_op %matmul_fused into %loop_2
        : (!transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op)

  // Since outlining is currently only implemented for region-holding operations
  // such as loops, use tiling to size 1 to materialize the outer loop that is
  // going to be outlined.
  %_, %outline_target =
      transform.structured.tile_using_forall %tiled_2 tile_sizes [1]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.structured.fuse_into_containing_op %matmul_fused_2
      into %outline_target
        : (!transform.any_op, !transform.any_op)
          -> (!transform.any_op, !transform.any_op)
  %func, %call = transform.loop.outline %outline_target {func_name = "outlined"}
      : (!transform.any_op) -> (!transform.any_op, !transform.op<"func.call">)

  transform.yield
}
```

This additional transformation also illustrates handle invalidation for nested operations. The `transform.loop.outline` operation consumes the handle to the loop, which invalidates it and all handles to any operations nested in it, such as `%2`. Attempting to use this handle will cause undefined behavior. (Note that it isn’t strictly necessary for this specific form of the outlining to consume the operand as the implementation only _moves_ the region without recreating the operations, but the author of the transformation chose to invalidate the handle anyway.)

Attempting to access the fusion result after outlining produces the following error

```sh
test/Examples/transform/Ch1/invalidation-2.mlir:109:3: error: op uses a handle invalidated by a previously executed transform op
  transform.test_print_remark_at_operand %outline_target, "outlined loop" : !transform.any_op
  ^
test/Examples/transform/Ch1/invalidation-2.mlir:102:25: note: handle to invalidated ops
  %outline_target, %_ = transform.structured.tile_using_forall %tiled_2 tile_sizes [1]
                        ^
test/Examples/transform/Ch1/invalidation-2.mlir:106:18: note: invalidated by this transform op that consumes its operand #0 and invalidates all handles to payload IR entities associated with this operand and entities nested in them
  %func, %call = transform.loop.outline %outline_target {func_name = "outlined"}
                 ^
test/Examples/transform/Ch1/invalidation-2.mlir:24:13: note: ancestor payload op
  %biased = linalg.elemwise_binary { fun = #linalg.binary_fn<add> }
            ^
test/Examples/transform/Ch1/invalidation-2.mlir:24:13: note: nested payload op
  %matmul = linalg.matmul ins(%lhs, %rhs: tensor<512x512xf32>, tensor<512x512xf32>)
```

Note that the “add” elementwise operation is indicated as payload ancestor because it was used to produce the tile loop, and the loop therefore has its location.

Finally, we would like to replace the call to the outlined function with a call to the microkernel. Unfortunately, the Transform dialect doesn’t have support for this transformation (and cannot have if the call is rewritten to a custom, out-of-tree operation). Therefore, we need to define new transform operations. The next chapters will describe how this can be done.

## Tracking IR Modifications

The Transform dialect automatically tracks all IR changes that are made as part
of transform ops. (Implementations must use the provided rewriter to modify IR.)
If a payload op is erased, it is automatically removed from all handles that it
is currently associated with. If a payload op is replaced, the transform dialect
tries to find the replacement op and updates all handles accordingly. If a
multi-result op is replaced with values that are defined by multiple ops, or if
an op is replaced with an op of a different type, an error is produced. This is
because it is unclear whether the direct replacements actually represent the
computation of the original op. There are ways to customize this behavior. More
details can be found at the documentation of `transform::TrackingListener`.
