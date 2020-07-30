//===- Tester.cpp ---------------------------------------------------------===//
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

#include "mlir/Reducer/Tester.h"

using namespace mlir;

Tester::Tester(StringRef scriptName, ArrayRef<std::string> scriptArgs)
    : testScript(scriptName), testScriptArgs(scriptArgs) {}

/// Runs the interestingness testing script on a MLIR test case file. Returns
/// true if the interesting behavior is present in the test case or false
/// otherwise.
bool Tester::isInteresting(StringRef testCase) {

  std::vector<StringRef> testerArgs;
  testerArgs.push_back(testCase);

  for (const std::string &arg : testScriptArgs)
    testerArgs.push_back(arg);

  std::string errMsg;
  int result = llvm::sys::ExecuteAndWait(
      testScript, testerArgs, /*Env=*/None, /*Redirects=*/None,
      /*SecondsToWait=*/0, /*MemoryLimit=*/0, &errMsg);

  if (result < 0)
    llvm::report_fatal_error("Error running interestingness test: " + errMsg,
                             false);

  if (!result)
    return false;

  return true;
}
