//===- llvm-reduce.cpp - The LLVM Delta Reduction utility -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This program tries to reduce an IR test case for a given interesting-ness
// test. It runs multiple delta debugging passes in order to minimize the input
// file. It's worth noting that this is a part of the bugpoint redesign
// proposal, and thus a *temporary* tool that will eventually be integrated
// into the bugpoint tool itself.
//
//===----------------------------------------------------------------------===//

#include "DeltaManager.h"
#include "ReducerWorkItem.h"
#include "TestRunner.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO.h"
#include <system_error>
#include <vector>

using namespace llvm;

cl::OptionCategory LLVMReduceOptions("llvm-reduce options");

static cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden,
                          cl::cat(LLVMReduceOptions));
static cl::opt<bool> Version("v", cl::desc("Alias for -version"), cl::Hidden,
                             cl::cat(LLVMReduceOptions));

static cl::opt<bool>
    PrintDeltaPasses("print-delta-passes",
                     cl::desc("Print list of delta passes, passable to "
                              "--delta-passes as a comma separated list"),
                     cl::cat(LLVMReduceOptions));

static cl::opt<std::string> InputFilename(cl::Positional, cl::Required,
                                          cl::desc("<input llvm ll/bc file>"),
                                          cl::cat(LLVMReduceOptions));

static cl::opt<std::string>
    TestFilename("test", cl::Required,
                 cl::desc("Name of the interesting-ness test to be run"),
                 cl::cat(LLVMReduceOptions));

static cl::list<std::string>
    TestArguments("test-arg",
                  cl::desc("Arguments passed onto the interesting-ness test"),
                  cl::cat(LLVMReduceOptions));

static cl::opt<std::string> OutputFilename(
    "output", cl::desc("Specify the output file. default: reduced.ll|mir"));
static cl::alias OutputFileAlias("o", cl::desc("Alias for -output"),
                                 cl::aliasopt(OutputFilename),
                                 cl::cat(LLVMReduceOptions));

static cl::opt<bool>
    ReplaceInput("in-place",
                 cl::desc("WARNING: This option will replace your input file "
                          "with the reduced version!"),
                 cl::cat(LLVMReduceOptions));

enum class InputLanguages { None, IR, MIR };

static cl::opt<InputLanguages>
    InputLanguage("x", cl::ValueOptional,
                  cl::desc("Input language ('ir' or 'mir')"),
                  cl::init(InputLanguages::None),
                  cl::values(clEnumValN(InputLanguages::IR, "ir", ""),
                             clEnumValN(InputLanguages::MIR, "mir", "")),
                  cl::cat(LLVMReduceOptions));

static cl::opt<int>
    MaxPassIterations("max-pass-iterations",
                      cl::desc("Maximum number of times to run the full set "
                               "of delta passes (default=5)"),
                      cl::init(5), cl::cat(LLVMReduceOptions));

static codegen::RegisterCodeGenFlags CGF;

void writeOutput(ReducerWorkItem &M, StringRef Message) {
  if (ReplaceInput) // In-place
    OutputFilename = InputFilename.c_str();
  else if (OutputFilename.empty() || OutputFilename == "-")
    OutputFilename = M.isMIR() ? "reduced.mir" : "reduced.ll";
  std::error_code EC;
  raw_fd_ostream Out(OutputFilename, EC);
  if (EC) {
    errs() << "Error opening output file: " << EC.message() << "!\n";
    exit(1);
  }
  M.print(Out, /*AnnotationWriter=*/nullptr);
  errs() << Message << OutputFilename << "\n";
}

void writeBitcode(ReducerWorkItem &M, llvm::raw_ostream &OutStream) {
  if (M.LTOInfo && M.LTOInfo->IsThinLTO && M.LTOInfo->EnableSplitLTOUnit) {
    legacy::PassManager PM;
    PM.add(llvm::createWriteThinLTOBitcodePass(OutStream));
    PM.run(*(M.M));
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

void readBitcode(ReducerWorkItem &M, MemoryBufferRef Data, LLVMContext &Ctx, const char *ToolName) {
  Expected<BitcodeFileContents> IF = llvm::getBitcodeFileContents(Data);
  if (!IF) {
    WithColor::error(errs(), ToolName) << IF.takeError();
    exit(1);
  }
  BitcodeModule BM = IF->Mods[0];
  Expected<BitcodeLTOInfo> LI = BM.getLTOInfo();
  Expected<std::unique_ptr<Module>> MOrErr = BM.parseModule(Ctx);
  if (!LI || !MOrErr) {
    WithColor::error(errs(), ToolName) << IF.takeError();
    exit(1);
  }
  M.LTOInfo = std::make_unique<BitcodeLTOInfo>(*LI);
  M.M = std::move(MOrErr.get());
}

int main(int Argc, char **Argv) {
  InitLLVM X(Argc, Argv);

  cl::HideUnrelatedOptions({&LLVMReduceOptions, &getColorCategory()});
  cl::ParseCommandLineOptions(Argc, Argv, "LLVM automatic testcase reducer.\n");

  bool ReduceModeMIR = false;
  if (InputLanguage != InputLanguages::None) {
    if (InputLanguage == InputLanguages::MIR)
      ReduceModeMIR = true;
  } else if (StringRef(InputFilename).endswith(".mir")) {
    ReduceModeMIR = true;
  }

  if (PrintDeltaPasses) {
    printDeltaPasses(errs());
    return 0;
  }

  LLVMContext Context;
  std::unique_ptr<TargetMachine> TM;

  std::unique_ptr<ReducerWorkItem> OriginalProgram =
      parseReducerWorkItem(Argv[0], InputFilename, Context, TM, ReduceModeMIR);
  if (!OriginalProgram) {
    return 1;
  }

  // Initialize test environment
  TestRunner Tester(TestFilename, TestArguments, std::move(OriginalProgram),
                    std::move(TM), Argv[0]);

  // Try to reduce code
  runDeltaPasses(Tester, MaxPassIterations);

  // Print reduced file to STDOUT
  if (OutputFilename == "-")
    Tester.getProgram().print(outs(), nullptr);
  else
    writeOutput(Tester.getProgram(), "\nDone reducing! Reduced testcase: ");

  return 0;
}
