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

def PoisonAttrInterface : AttrInterface<"PoisonAttrInterface"> {
  let cppNamespace = "::mlir::ub";
  // No methods for now.

  // To make DefaultValuedAttr happy.
  let constBuilderCall = cppNamespace # "::" # "PoisonAttr" #
                         "::get($_builder.getContext())";
}

#endif // MLIR_DIALECT_UB_IR_UBOPSINTERFACES_TD
