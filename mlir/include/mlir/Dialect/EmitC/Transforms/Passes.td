//===-- Passes.td - pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_EMITC_TRANSFORMS_PASSES
#define MLIR_DIALECT_EMITC_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def FormExpressionsPass : Pass<"form-expressions"> {
  let summary = "Form C-style expressions from C-operator ops";
  let description = [{
    The pass wraps emitc ops modelling C operators in emitc.expression ops and
    then folds single-use expressions into their users where possible.
  }];
  let dependentDialects = ["emitc::EmitCDialect"];
}

def WrapFuncInClassPass : Pass<"wrap-emitc-func-in-class"> {
  let summary = "Wrap functions in classes, using arguments as fields.";
  let description = [{
    This pass transforms `emitc.func` operations into `emitc.class` operations.
    Function arguments become fields of the class, and the function body is moved
    to a new `execute` method within the class.
  }];
  let dependentDialects = ["emitc::EmitCDialect"];
  let options = [Option<
      "namedAttribute", "named-attribute", "std::string",
      /*default=*/"",
      "Attribute key used to extract field names from function argument's "
      "dictionary attributes">];
}

#endif // MLIR_DIALECT_EMITC_TRANSFORMS_PASSES
