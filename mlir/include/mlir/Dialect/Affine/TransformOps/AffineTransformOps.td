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

#endif // Affine_TRANSFORM_OPS
