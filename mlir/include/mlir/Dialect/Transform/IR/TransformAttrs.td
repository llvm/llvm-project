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
include "mlir/Dialect/Transform/IR/TransformDialect.td"

class Transform_Attr<string name, string attrMnemonic,
                     list<Trait> traits = [],
                     string baseCppClass = "::mlir::Attribute">
    : AttrDef<Transform_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}

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

def ParamOperandAttr : Transform_Attr<"ParamOperand", "param_operand"> {
  let description = [{
    Used to refer to a specific param-operand (via its index) from within an
    attribute on a transform operation.
  }];
  let parameters = (ins
    "IntegerAttr":$index
  );
  let assemblyFormat = "`<` `index` `=` $index `>`";
}

#endif  // MLIR_DIALECT_TRANSFORM_IR_TRANSFORMATTRS
