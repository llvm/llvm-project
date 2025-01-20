//===--- llvm-isel-fuzzer.cpp - Fuzzer for instruction selection ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tool to fuzz instruction selection using libFuzzer.
//
//===----------------------------------------------------------------------===//

#include "llvm/InitializePasses.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/Lint.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetVerifier.h"

#include "llvm/Target/TargetVerify/AMDGPUTargetVerifier.h"

#define DEBUG_TYPE "isel-fuzzer"

using namespace llvm;

static codegen::RegisterCodeGenFlags CGF;

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input bitcode>"), cl::init("-"));

static cl::opt<bool>
    StacktraceAbort("stacktrace-abort",
        cl::desc("Turn on stacktrace"), cl::init(false));

static cl::opt<bool>
    NoLint("no-lint",
        cl::desc("Turn off Lint"), cl::init(false));

static cl::opt<bool>
    NoVerify("no-verifier",
        cl::desc("Turn off Verifier"), cl::init(false));

static cl::opt<char>
    OptLevel("O",
             cl::desc("Optimization level. [-O0, -O1, -O2, or -O3] "
                      "(default = '-O2')"),
             cl::Prefix, cl::init('2'));

static cl::opt<std::string>
    TargetTriple("mtriple", cl::desc("Override target triple for module"));

static std::unique_ptr<TargetMachine> TM;

static void handleLLVMFatalError(void *, const char *Message, bool) {
  if (StacktraceAbort) {
    dbgs() << "LLVM ERROR: " << Message << "\n"
           << "Aborting.\n";
    abort();
  }
}

int main(int argc, char **argv) {
  StringRef ExecName = argv[0];
  InitLLVM X(argc, argv);

  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  PassRegistry *Registry = PassRegistry::getPassRegistry();
  initializeCore(*Registry);
  initializeCodeGen(*Registry);
  initializeAnalysis(*Registry);
  initializeTarget(*Registry);

  cl::ParseCommandLineOptions(argc, argv);

  if (TargetTriple.empty()) {
    errs() << ExecName << ": -mtriple must be specified\n";
    exit(1);
  }

  CodeGenOptLevel OLvl;
  if (auto Level = CodeGenOpt::parseLevel(OptLevel)) {
    OLvl = *Level;
  } else {
    errs() << ExecName << ": invalid optimization level.\n";
    return 1;
  }
  ExitOnError ExitOnErr(std::string(ExecName) + ": error:");
  TM = ExitOnErr(codegen::createTargetMachineForTriple(
      Triple::normalize(TargetTriple), OLvl));
  assert(TM && "Could not allocate target machine!");

  // Make sure we print the summary and the current unit when LLVM errors out.
  install_fatal_error_handler(handleLLVMFatalError, nullptr);

  LLVMContext Context;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseIRFile(InputFilename, Err, Context);
  if (!M) {
    errs() << "Invalid mod\n";
    return 1;
  }
  auto S = Triple::normalize(TargetTriple);
  M->setTargetTriple(S);

  PassInstrumentationCallbacks PIC;
  StandardInstrumentations SI(Context, false/*debug PM*/,
                              false);
  registerCodeGenCallback(PIC, *TM);

  ModulePassManager MPM;
  FunctionPassManager FPM;
  //TargetLibraryInfoImpl TLII(Triple(M->getTargetTriple()));

  MachineFunctionAnalysisManager MFAM;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  PassBuilder PB(TM.get(), PipelineTuningOptions(), std::nullopt, &PIC);
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.registerMachineFunctionAnalyses(MFAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM, &MFAM);

  SI.registerCallbacks(PIC, &MAM);

  //FAM.registerPass([&] { return TargetLibraryAnalysis(TLII); });

  Triple TT(M->getTargetTriple());
  if (!NoLint)
    FPM.addPass(LintPass());
  if (!NoVerify)
    MPM.addPass(VerifierPass());
  if (TT.isAMDGPU())
    FPM.addPass(AMDGPUTargetVerifierPass());
  else if (false) {} // ...
  else
    FPM.addPass(TargetVerifierPass());
  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));

  MPM.run(*M, MAM);

  if (!M->IsValid)
    return 1;

  return 0;
}
