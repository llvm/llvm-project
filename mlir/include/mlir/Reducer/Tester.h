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
// used to run the interestingness testing script on the different generated
// reduced variants of the test case.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_REDUCER_TESTER_H
#define MLIR_REDUCER_TESTER_H

#include <vector>

#include "mlir/IR/BuiltinDialect.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"

namespace mlir {

/// This class is used to keep track of the testing environment of the tool. It
/// contains a method to run the interestingness testing script on a MLIR test
/// case file.
class Tester {
public:
  Tester(StringRef testScript, ArrayRef<std::string> testScriptArgs);

  /// Runs the interestingness testing script on a MLIR test case file. Returns
  /// true if the interesting behavior is present in the test case or false
  /// otherwise.
  bool isInteresting(StringRef testCase) const;

private:
  StringRef testScript;
  ArrayRef<std::string> testScriptArgs;
};

} // end namespace mlir

#endif
