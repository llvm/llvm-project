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
// used to run the interestingness testing script on the different generated
// reduced variants of the test case.
//
//===----------------------------------------------------------------------===//

#include "mlir/Reducer/Tester.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

Tester::Tester(StringRef scriptName, ArrayRef<std::string> scriptArgs)
    : testScript(scriptName), testScriptArgs(scriptArgs) {}

std::pair<Tester::Interestingness, size_t>
Tester::isInteresting(ModuleOp module) const {
  // The reduced module should always be vaild, or we may end up retaining the
  // error message by an invalid case. Besides, an invalid module may not be
  // able to print properly.
  if (failed(verify(module)))
    return std::make_pair(Interestingness::False, /*size=*/0);

  SmallString<128> filepath;
  int fd;

  // Print module to temporary file.
  std::error_code ec =
      llvm::sys::fs::createTemporaryFile("mlir-reduce", "mlir", fd, filepath);

  if (ec)
    llvm::report_fatal_error(llvm::Twine("Error making unique filename: ") +
                             ec.message());

  llvm::ToolOutputFile out(filepath, fd);
  module.print(out.os());
  out.os().close();

  if (out.os().has_error())
    llvm::report_fatal_error(llvm::Twine("Error emitting the IR to file '") +
                             filepath);

  size_t size = out.os().tell();
  return std::make_pair(isInteresting(filepath), size);
}

/// Runs the interestingness testing script on a MLIR test case file. Returns
/// true if the interesting behavior is present in the test case or false
/// otherwise.
Tester::Interestingness Tester::isInteresting(StringRef testCase) const {
  std::vector<StringRef> testerArgs;
  testerArgs.push_back(testCase);

  for (const std::string &arg : testScriptArgs)
    testerArgs.emplace_back(arg);

  testerArgs.push_back(testCase);

  std::string errMsg;
  int result = llvm::sys::ExecuteAndWait(
      testScript, testerArgs, /*Env=*/std::nullopt, /*Redirects=*/std::nullopt,
      /*SecondsToWait=*/0, /*MemoryLimit=*/0, &errMsg);

  if (result < 0)
    llvm::report_fatal_error(
        llvm::Twine("Error running interestingness test: ") + errMsg, false);

  if (!result)
    return Interestingness::False;

  return Interestingness::True;
}
