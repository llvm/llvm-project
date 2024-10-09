//===-- TestRunner.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestRunner.h"
#include "ReducerWorkItem.h"
#include "deltas/Utils.h"
#include "llvm/Support/WithColor.h"

using namespace llvm;

TestRunner::TestRunner(StringRef TestName, ArrayRef<std::string> RawTestArgs,
                       std::unique_ptr<ReducerWorkItem> Program,
                       std::unique_ptr<TargetMachine> TM, StringRef ToolName,
                       StringRef OutputName, bool InputIsBitcode,
                       bool OutputBitcode)
    : TestName(TestName), ToolName(ToolName), Program(std::move(Program)),
      TM(std::move(TM)), OutputFilename(OutputName),
      InputIsBitcode(InputIsBitcode), EmitBitcode(OutputBitcode) {
  assert(this->Program && "Initialized with null program?");

  TestArgs.push_back(TestName); // argv[0]
  TestArgs.append(RawTestArgs.begin(), RawTestArgs.end());
}

static constexpr std::array<std::optional<StringRef>, 3> DefaultRedirects = {
    StringRef()};
static constexpr std::array<std::optional<StringRef>, 3> NullRedirects;

/// Runs the interestingness test, passes file to be tested as first argument
/// and other specified test arguments after that.
int TestRunner::run(StringRef Filename) const {
  SmallVector<StringRef> ExecArgs(TestArgs);
  ExecArgs.push_back(Filename);

  std::string ErrMsg;

  int Result =
      sys::ExecuteAndWait(TestName, ExecArgs, /*Env=*/std::nullopt,
                          Verbose ? DefaultRedirects : NullRedirects,
                          /*SecondsToWait=*/0, /*MemoryLimit=*/0, &ErrMsg);

  if (Result < 0) {
    Error E = make_error<StringError>("Error running interesting-ness test: " +
                                          ErrMsg,
                                      inconvertibleErrorCode());
    WithColor::error(errs(), ToolName) << toString(std::move(E)) << '\n';
    exit(1);
  }

  return !Result;
}

void TestRunner::writeOutput(StringRef Message) {
  std::error_code EC;
  raw_fd_ostream Out(OutputFilename, EC,
                     EmitBitcode && !Program->isMIR() ? sys::fs::OF_None
                                                      : sys::fs::OF_Text);
  if (EC) {
    errs() << "Error opening output file: " << EC.message() << "!\n";
    exit(1);
  }

  Program->writeOutput(Out, EmitBitcode);
  errs() << Message << OutputFilename << '\n';
}
