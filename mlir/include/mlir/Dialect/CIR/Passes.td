//===-- Passes.td - CIR pass definition file ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_CIR_PASSES
#define MLIR_DIALECT_CIR_PASSES

include "mlir/Pass/PassBase.td"

def LifetimeCheck : Pass<"cir-lifetime-check"> {
  let summary = "Check lifetime safety and generate diagnostics";
  let description = [{
    This pass relies on a lifetime analysis pass and uses the diagnostics
    mechanism to report to the user. It does not change any code.
  }];
  let constructor = "mlir::createLifetimeCheckPass()";
  let dependentDialects = ["cir::CIRDialect"];

  let options = [
    ListOption<"remarksList", "remarks", "std::string",
               "List of remark styles to enable as part of diagnostics."
               " Supported styles: {all|pset}", "llvm::cl::ZeroOrMore">
  ];
}

#endif // MLIR_DIALECT_CIR_PASSES
