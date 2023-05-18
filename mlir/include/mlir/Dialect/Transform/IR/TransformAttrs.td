//===- TransformAttrs.td - Transform dialect attributes ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_IR_TRANSFORMATTRS
#define MLIR_DIALECT_TRANSFORM_IR_TRANSFORMATTRS

include "mlir/IR/EnumAttr.td"

def PropagateFailuresCase : I32EnumAttrCase<"Propagate", 1, "propagate">;
def SuppressFailuresCase : I32EnumAttrCase<"Suppress", 2, "suppress">;

def FailurePropagationMode : I32EnumAttr<
    "FailurePropagationMode", "Silenceable error propagation policy",
    [PropagateFailuresCase, SuppressFailuresCase]> {
  let cppNamespace = "::mlir::transform";
}

def MatchCmpIPredicateAttr : I32EnumAttr<
    "MatchCmpIPredicate", "",
    [
      I32EnumAttrCase<"eq", 0>,
      I32EnumAttrCase<"ne", 1>,
      I32EnumAttrCase<"lt", 2>,
      I32EnumAttrCase<"le", 3>,
      I32EnumAttrCase<"gt", 4>,
      I32EnumAttrCase<"ge", 5>,
    ]> {
  let cppNamespace = "::mlir::transform";
}

#endif  // MLIR_DIALECT_TRANSFORM_IR_TRANSFORMATTRS
