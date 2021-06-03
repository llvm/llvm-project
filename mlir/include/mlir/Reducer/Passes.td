//===-- Passes.td - MLIR Reduce pass definition file -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions of the passes for the MLIR Reduce Tool.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_REDUCER_PASSES
#define MLIR_REDUCER_PASSES

include "mlir/Pass/PassBase.td"

def CommonReductionPassOptions {
  list<Option> options = [
    Option<"testerName", "test", "std::string", /* default */"",
           "The filename of the tester">,
    ListOption<"testerArgs", "test-arg", "std::string",
               "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">,
  ];
}

def ReductionTree : Pass<"reduction-tree"> {
  let summary = "A general reduction tree pass for the MLIR Reduce Tool";

  let constructor = "mlir::createReductionTreePass()";

  let options = [
    Option<"traversalModeId", "traversal-mode", "unsigned",
           /* default */"0", "The graph traversal mode">,
  ] # CommonReductionPassOptions.options;
}

def OptReduction : Pass<"opt-reduction-pass", "ModuleOp"> {
  let summary = "A reduction pass wrapper for optimization passes";

  let constructor = "mlir::createOptReductionPass()";

  let options = [
    Option<"optPass", "opt-pass", "std::string", /* default */"",
           "The optimization pass will be run dynamically in OptReductionPass">,
  ] # CommonReductionPassOptions.options;
}

#endif // MLIR_REDUCER_PASSES
