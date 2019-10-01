//===-- TestRunner.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestRunner.h"

using namespace llvm;

TestRunner::TestRunner(StringRef TestName, const std::vector<std::string> &TestArgs)
    : TestName(TestName), TestArgs(TestArgs) {
}

/// Runs the interestingness test, passes file to be tested as first argument
/// and other specified test arguments after that.
int TestRunner::run(StringRef Filename) {
  std::vector<StringRef> ProgramArgs;
  ProgramArgs.push_back(TestName);

  for (const auto &Arg : TestArgs)
    ProgramArgs.push_back(Arg);

  ProgramArgs.push_back(Filename);

  std::string ErrMsg;
  int Result = sys::ExecuteAndWait(
      TestName, ProgramArgs, /*Env=*/None, /*Redirects=*/None,
      /*SecondsToWait=*/0, /*MemoryLimit=*/0, &ErrMsg);

  if (Result < 0) {
    Error E = make_error<StringError>("Error running interesting-ness test: " +
                                          ErrMsg,
                                      inconvertibleErrorCode());
    errs() << toString(std::move(E));
    exit(1);
  }

  return !Result;
}
