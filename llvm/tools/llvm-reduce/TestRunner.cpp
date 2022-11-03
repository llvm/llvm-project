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
#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/IPO/ThinLTOBitcodeWriter.h"

using namespace llvm;

TestRunner::TestRunner(StringRef TestName,
                       const std::vector<std::string> &TestArgs,
                       std::unique_ptr<ReducerWorkItem> Program,
                       std::unique_ptr<TargetMachine> TM, const char *ToolName,
                       StringRef OutputName, bool InputIsBitcode,
                       bool OutputBitcode)
    : TestName(TestName), ToolName(ToolName), TestArgs(TestArgs),
      Program(std::move(Program)), TM(std::move(TM)),
      OutputFilename(OutputName), InputIsBitcode(InputIsBitcode),
      EmitBitcode(OutputBitcode) {
  assert(this->Program && "Initialized with null program?");
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
  SmallVector<Optional<StringRef>, 3> Redirects;
  Optional<StringRef> Empty = StringRef();
  if (!Verbose) {
    for (int i = 0; i < 3; ++i)
      Redirects.push_back(Empty);
  }
  int Result =
      sys::ExecuteAndWait(TestName, ProgramArgs, /*Env=*/None, Redirects,
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

void TestRunner::setProgram(std::unique_ptr<ReducerWorkItem> P) {
  assert(P && "Setting null program?");
  Program = std::move(P);
}

void writeBitcode(ReducerWorkItem &M, raw_ostream &OutStream) {
  if (M.LTOInfo && M.LTOInfo->IsThinLTO && M.LTOInfo->EnableSplitLTOUnit) {
    PassBuilder PB;
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
    ModulePassManager MPM;
    MPM.addPass(ThinLTOBitcodeWriterPass(OutStream, nullptr));
    MPM.run(*M.M, MAM);
  } else {
    std::unique_ptr<ModuleSummaryIndex> Index;
    if (M.LTOInfo && M.LTOInfo->HasSummary) {
      ProfileSummaryInfo PSI(M);
      Index = std::make_unique<ModuleSummaryIndex>(
          buildModuleSummaryIndex(M, nullptr, &PSI));
    }
    WriteBitcodeToFile(M, OutStream, Index.get());
  }
}

void TestRunner::writeOutput(StringRef Message) {
  std::error_code EC;
  raw_fd_ostream Out(OutputFilename, EC);
  if (EC) {
    errs() << "Error opening output file: " << EC.message() << "!\n";
    exit(1);
  }

  // Requesting bitcode emission with mir is nonsense, so just ignore it.
  if (EmitBitcode && !Program->isMIR())
    writeBitcode(*Program, Out);
  else
    Program->print(Out, /*AnnotationWriter=*/nullptr);

  errs() << Message << OutputFilename << '\n';
}
