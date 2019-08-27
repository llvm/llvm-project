//===-- TestRunner.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestRunner.h"

using namespace llvm;

/// Gets Current Working Directory and tries to create a Tmp Directory
static SmallString<128> initializeTmpDirectory() {
  SmallString<128> CWD;
  if (std::error_code EC = sys::fs::current_path(CWD)) {
    errs() << "Error getting current directory: " << EC.message() << "!\n";
    exit(1);
  }

  SmallString<128> TmpDirectory;
  sys::path::append(TmpDirectory, CWD, "tmp");
  if (std::error_code EC = sys::fs::create_directory(TmpDirectory))
    errs() << "Error creating tmp directory: " << EC.message() << "!\n";

  return TmpDirectory;
}

TestRunner::TestRunner(StringRef TestName, std::vector<std::string> TestArgs,
                       StringRef ReducedFilepath)
    : TestName(TestName), TestArgs(std::move(TestArgs)),
      ReducedFilepath(ReducedFilepath) {
  TmpDirectory = initializeTmpDirectory();
}

/// Runs the interestingness test, passes file to be tested as first argument
/// and other specified test arguments after that.
int TestRunner::run(StringRef Filename) {
  std::vector<StringRef> ProgramArgs;
  ProgramArgs.push_back(TestName);
  ProgramArgs.push_back(Filename);

  for (auto Arg : TestArgs)
    ProgramArgs.push_back(Arg.c_str());

  Optional<StringRef> Redirects[3]; // STDIN, STDOUT, STDERR
  int Result = sys::ExecuteAndWait(TestName, ProgramArgs, None, Redirects);

  if (Result < 0) {
    Error E = make_error<StringError>("Error running interesting-ness test\n",
                                      inconvertibleErrorCode());
    errs() << toString(std::move(E));
    exit(1);
  }

  return !Result;
}
