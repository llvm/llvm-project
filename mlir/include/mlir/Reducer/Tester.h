//===- Tester.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Tester class used in the MLIR Reduce tool.
//
// A Tester object is passed as an argument to the reduction passes and it is
// used to keep track of the state of the reduction throughout the multiple
// passes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_REDUCER_TESTER_H
#define MLIR_REDUCER_TESTER_H

#include <vector>

#include "mlir/IR/Module.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"

namespace mlir {

/// This class is used to keep track of the state of the reduction. It contains
/// a method to run the interestingness testing script on MLIR test case files
/// and provides functionality to track the most reduced test case.
class Tester {
public:
  Tester(StringRef testScript, ArrayRef<std::string> testScriptArgs);

  /// Runs the interestingness testing script on a MLIR test case file. Returns
  /// true if the interesting behavior is present in the test case or false
  /// otherwise.
  bool isInteresting(StringRef testCase);

  /// Returns the most reduced MLIR test case module.
  ModuleOp getMostReduced() const { return mostReduced; }

  /// Updates the most reduced MLIR test case module. If a
  /// generated variant is found to be successful and shorter than the
  /// mostReduced module, the mostReduced module must be updated with the new
  /// variant.
  void setMostReduced(ModuleOp t) { mostReduced = t; }

private:
  StringRef testScript;
  ArrayRef<std::string> testScriptArgs;
  ModuleOp mostReduced;
};

} // end namespace mlir

#endif