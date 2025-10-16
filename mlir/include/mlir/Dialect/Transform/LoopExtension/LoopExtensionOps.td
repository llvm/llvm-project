//===- LoopExtensionOps.td - Transform dialect operations --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_LOOPEXTENSION_LOOPEXTENSIONOPS
#define MLIR_DIALECT_TRANSFORM_LOOPEXTENSION_LOOPEXTENSIONOPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def HoistLoopInvariantSubsetsOp
    : TransformDialectOp<"loop.hoist_loop_invariant_subsets",
        [TransformOpInterface, TransformEachOpTrait,
         DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
         ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Hoist loop invariant subset ops";
  let description = [{
    This transform hoists loop-invariant subset ops out of the targeted
    loop-like op. It looks for matching subset extraction/insertion op pairs and
    hoists them. The loop body operates on a newly introduced region iter_arg.

    Subset ops are hoisted only from the targeted op. If subset ops should be
    hoisted from an entire loop nest, this transformation must be applied to
    each loop-like op of the loop nest, starting with the innermost loop and
    ending with the outermost loop.

    Example:
    ```
    %r = scf.for ... iter_args(%t = %a) -> (tensor<?xf32>) {
      %0 = tensor.extract_slice %t[0][5][1] : tensor<?xf32> to tensor<5xf32>
      %1 = "test.foo"(%0) : (tensor<5xf32>) -> (tensor<5xf32>)
      %2 = tensor.insert_slice %1 into %t[0][5][1]
          : tensor<5xf32> into tensor<?xf32>
      scf.yield %2 : tensor<?xf32>
    }
    ```
    Is transformed to:
    ```
    %0 = tensor.extract_slice %a[0][5][1] : tensor<?xf32> to tensor<5xf32>
    %new_loop:2 = scf.for ... iter_args(%t = %a, %h = %0) -> (tensor<?xf32>) {
      %1 = "test.foo"(%h) : (tensor<5xf32>) -> (tensor<5xf32>)
      scf.yield %t, %2 : tensor<?xf32>, tensor<5xf32>
    }
    %r = tensor.insert_slice %new_loop#1 into %new_loop#0
        : tensor<5xf32> into tensor<?xf32>
    ```

    Subset ops are hoisted only if there are no conflicting subset ops. E.g.,
    if there were a second overlapping extraction in the above example, no ops
    could be hoisted safely.

    This transform reads the target handle and modifies the payload. This
    transform does not invalidate any handles, but loop-like ops are replaced
    with new loop-like ops when a subset op is hoisted. The transform rewriter
    updates all handles accordingly.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);
  let assemblyFormat = "$target attr-dict `:` type($target)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
      ::mlir::transform::TransformRewriter &rewriter,
      ::mlir::LoopLikeOpInterface loopLikeOp,
      ::mlir::transform::ApplyToEachResultList &results,
      ::mlir::transform::TransformState &state);
  }];
}

def HoistLoopInvariantCodeMotionOp
    : TransformDialectOp<"loop.hoist_loop_invariant_code_motion",
        [TransformOpInterface, TransformEachOpTrait,
         DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
         ReportTrackingListenerFailuresOpTrait]> {
  let summary = "Hoist loop invariant instructions outside of the loop";
  let description = [{
    This transform hoists loop-invariant instructions out of the targeted
    loop-like op.

    Example:
    ```
    %m = memref.alloc() : memref<10xf32>
    %cf7 = arith.constant 7.0 : f32
    %cf8 = arith.constant 8.0 : f32

    affine.for %arg0 = 0 to 10 {
      affine.for %arg1 = 0 to 10 {
        %v0 = arith.addf %cf7, %cf8 : f32
      }
    }
    ```
    Is transformed to:
    ```
    %alloc = memref.alloc() : memref<10xf32>
    %cst = arith.constant 7.000000e+00 : f32
    %cst_0 = arith.constant 8.000000e+00 : f32
    %0 = arith.addf %cst, %cst_0 : f32
    affine.for %arg0 = 0 to 10 {
    }
    affine.for %arg0 = 0 to 10 {
    }
    ```

    loop-invariant instructions are hoisted only if they are pure ops and
    they are ancestors of the parent regionOp of all operands.

    This transform reads the target handle and modifies the payload. This
    transform does not invalidate any handles, but loop-like ops are replaced
    with new loop-like ops when a loop-invariant op is hoisted. The transform
    rewriter updates all handles accordingly.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);
  let assemblyFormat = "$target attr-dict `:` type($target)";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
      ::mlir::transform::TransformRewriter &rewriter,
      ::mlir::LoopLikeOpInterface loopLikeOp,
      ::mlir::transform::ApplyToEachResultList &results,
      ::mlir::transform::TransformState &state);
  }];
}

#endif // MLIR_DIALECT_TRANSFORM_LOOPEXTENSION_LOOPEXTENSIONOPS
