//===-- Passes.td - Transform dialect pass definitions -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_TRANSFORMS_PASSES
#define MLIR_DIALECT_TRANSFORM_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def CheckUsesPass : Pass<"transform-dialect-check-uses"> {
  let summary = "warn about potential use-after-free in the transform dialect";
  let description = [{
    This pass analyzes operations from the transform dialect and its extensions
    and warns if a transform IR value may be used by an operation after it was
    "freed" by some other operation, as described by side effects on the
    `TransformMappingResource`. This statically detects situations that lead to
    errors when interpreting the Transform IR.

    The pass is capable of handling branching control flow and reports all
    _potential_ use-after-free situations, e.g., a may-use-after-free is
    reported if at least one of the control flow paths between the definition of
    a value and its use contains an operation with a "free" effect on the
    `TransformMappingResource`. It does not currently perform an SCCP-style data
    flow analysis to prove that some branches are not taken, however, SCCP and
    other control flow simplifications can be performed on the transform IR
    prior to this pass provided that transform ops implement the relevant
    control flow interfaces.
  }];
}

#endif // MLIR_DIALECT_TRANSFORM_TRANSFORMS_PASSES
