//===- AffineTransformOps.td - Affine transformation ops ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AFFINE_TRANSFORM_OPS
#define AFFINE_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def Transform_AffineForOp : Transform_ConcreteOpType<"affine.for">;

def SimplifyBoundedAffineOpsOp
    : Op<Transform_Dialect, "affine.simplify_bounded_affine_ops",
         [DeclareOpInterfaceMethods<TransformOpInterface>,
          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let description = [{
    Simplify the targeted affine.min / affine.max ops given the supplied
    lower and upper bounds for values that may be used as target op operands.

    Example:
    ```
    %0 = transform.structured.match ops{["affine.min", "affine.max"]} in %arg1
    %1 = transform.structured.match ops{["gpu.lane_id"]} in %arg1
    transform.affine.simplify_bounded_affine_ops %0 with [%1] within [0] and [32]

    // Multiple bounds can be specified.
    transform.affine.simplify_bounded_affine_ops %0 with [%1, %2] within [0, 5] and [32, 50]
    ```

    Bounded op handles (`%1` and `%2) must be mapped to ops that have a single
    result of index type. The sets of target ops and bounded ops must not
    overlap.

    #### Return modes

    Target ops must be affine.min or affine.max ops. This transform consumes the
    target handle and does not produce any handle. It reads the bounded op
    handles.

    TODO: Support affine.apply targets.
    TODO: Allow mixed PDL_Operation/int64_t for lower_bounds and upper_bounds.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       Variadic<TransformHandleTypeInterface>:$bounded_values,
                       DenseI64ArrayAttr:$lower_bounds,
                       DenseI64ArrayAttr:$upper_bounds);
  let results = (outs);
  let hasVerifier = 1;

  let assemblyFormat = [{
      $target `with` `[` ($bounded_values^ `:` type($bounded_values))? `]`
          `within` $lower_bounds `and` $upper_bounds attr-dict
          `:` type($target)
  }];
}

def SimplifyMinMaxAffineOpsOp :
  Op<Transform_Dialect, "affine.simplify_min_max_affine_ops", [
    DeclareOpInterfaceMethods<TransformOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let description = [{
    Simplify the targeted `affine.min` / `affine.max` ops using the
    `mlir::affine::simplifyAffineMinMaxOps` transform.

    Example:
    ```
    %0 = transform.structured.match ops{["affine.max"]} in %arg1
    transform.affine.simplify_min_max_affine_ops %0 : !transform.any_op
    ```

    #### Return modes

    This transform consumes the target handle and does not produce any results.
    This transforms definitely fails if any of the targeted operations is not an
    `affine.min` or `affine.max` operation, or if the canonicalization patterns
    failed to converge.
    This transform silently fails if none of the operations were simplified.
    Otherwise, it succeeds.
  }];
  let arguments = (ins TransformHandleTypeInterface:$target);
  let results = (outs);
  let assemblyFormat = [{
      $target attr-dict `:` type($target)
  }];
}

def SuperVectorizeOp
    : Op<Transform_Dialect, "affine.super_vectorize",
         [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
         DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let description = [{
    Vectorize to a target independent n-D vector abstraction.
    This operation is an exposition to the transform dialect of the affine-super-vectorize pass.
    To make its usage easier, it ignores inputs which are children of an affine.for op (itself excluded),
    this way the matcher can be simpler.

    This operation may generate operations from the vector dialect.

    Example:
    ```
    %0 = transform.structured.match ops{["affine.for"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.affine.super_vectorize %0 [8, 16] fastest_varying_pattern=[1,0] vectorize_reductions=true : !transform.any_op
    ```
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       DenseI64ArrayAttr:$vector_sizes,
                       OptionalAttr<DenseI64ArrayAttr>:$fastest_varying_pattern,
                       DefaultValuedAttr<BoolAttr, "false">:$vectorize_reductions);
  let results = (outs);

  let assemblyFormat = [{
      $target $vector_sizes
          (`fastest_varying_pattern` `=` $fastest_varying_pattern^)? 
          (`vectorize_reductions` `=` $vectorize_reductions^)? 
          attr-dict `:` type($target)
  }];
  let hasVerifier = 1;
}

#endif // Affine_TRANSFORM_OPS
