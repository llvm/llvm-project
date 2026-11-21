//===- UBOpsInterfaces.td - UB interfaces definitions ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_UB_IR_UBOPSINTERFACES_TD
#define MLIR_DIALECT_UB_IR_UBOPSINTERFACES_TD

include "mlir/IR/OpBase.td"

def AssumePredicateAttrInterface : AttrInterface<"AssumePredicateAttrInterface"> {
  let cppNamespace = "::mlir::ub";
  let cppIncludes = ["mlir/Dialect/UB/IR/UBOpsInterfaces.h"];
  let description = [{
    A predicate on the predicated value of `ub.assume`. The predicate holds for
    the result of `ub.assume`.

    Predicates describe properties of SSA values. A false predicate produces a
    poisoned result of `ub.assume`, not immediate undefined behavior.
    Each predicate defines how it handles poison in additional operands.
  }];
  let methods = [
    InterfaceMethod<[{
        Verify the predicated value type and additional operand types.
      }],
      "::mlir::LogicalResult", "verifyAssumption",
      (ins "::mlir::Type":$valueType,
           "::mlir::TypeRange":$predicateOperandTypes,
           "::llvm::function_ref<::mlir::InFlightDiagnostic()>":$emitError)>,
    InterfaceMethod<[{
        Evaluate the predicate on constant operands. Null attributes denote
        operands whose constant values are not known. Return true if the
        predicate holds, false if it is false or produces poison, and std::nullopt
        if neither can be determined. Fully poisoned predicated values are
        handled by `ub.assume` before calling this method. Poison attributes for
        additional operands are passed to this method and must be interpreted
        according to the predicate's semantics.
      }], "::std::optional<bool>", "evaluate",
      (ins "::mlir::Attribute":$value,
           "::llvm::ArrayRef<::mlir::Attribute>":$predicateOperands),
      /*methodBody=*/"", /*defaultImplementation=*/[{
        return ::std::nullopt;
      }]>
  ];
}

def PoisonAttrInterface : AttrInterface<"PoisonAttrInterface"> {
  let cppNamespace = "::mlir::ub";
  // No methods for now.

  // To make DefaultValuedAttr happy.
  let constBuilderCall = cppNamespace # "::" # "PoisonAttr" #
                         "::get($_builder.getContext())";
}

#endif // MLIR_DIALECT_UB_IR_UBOPSINTERFACES_TD
