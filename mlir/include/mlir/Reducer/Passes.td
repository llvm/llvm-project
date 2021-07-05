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
           "The location of the tester which tests the file interestingness">,
    ListOption<"testerArgs", "test-arg", "std::string",
               "arguments of the tester",
               "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">,
  ];
}

def ReductionTree : Pass<"reduction-tree"> {
  let summary = "Reduce the input with reduction-tree algorithm";

  let constructor = "mlir::createReductionTreePass()";

  let options = [
    Option<"traversalModeId", "traversal-mode", "unsigned",
           /* default */"0",
           "The graph traversal mode, the default is single-path mode">,
  ] # CommonReductionPassOptions.options;
}

def OptReduction : Pass<"opt-reduction-pass", "ModuleOp"> {
  let summary = "A wrapper pass that reduces the file with optimization passes";

  let constructor = "mlir::createOptReductionPass()";

  let options = [
    Option<"optPass", "opt-pass", "std::string", /* default */"",
           "The optimization passes used for reduction, e.g., symbol-dce">,
  ] # CommonReductionPassOptions.options;
}

#endif // MLIR_REDUCER_PASSES
