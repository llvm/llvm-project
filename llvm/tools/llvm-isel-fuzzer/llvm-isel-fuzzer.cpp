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

#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.inc"
#include "llvm/FuzzMutate/FuzzerCLI.h"
#include "llvm/FuzzMutate/IRMutator.h"
#include "llvm/FuzzMutate/Operations.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "isel-fuzzer"

using namespace llvm;

static cl::opt<char>
OptLevel("O",
         cl::desc("Optimization level. [-O0, -O1, -O2, or -O3] "
                  "(default = '-O2')"),
         cl::Prefix,
         cl::ZeroOrMore,
         cl::init(' '));

static cl::opt<std::string>
TargetTriple("mtriple", cl::desc("Override target triple for module"));

static std::unique_ptr<TargetMachine> TM;
static std::unique_ptr<IRMutator> Mutator;

std::unique_ptr<IRMutator> createISelMutator() {
  std::vector<TypeGetter> Types{
      Type::getInt1Ty,  Type::getInt8Ty,  Type::getInt16Ty, Type::getInt32Ty,
      Type::getInt64Ty, Type::getFloatTy, Type::getDoubleTy};

  std::vector<std::unique_ptr<IRMutationStrategy>> Strategies;
  Strategies.emplace_back(
      new InjectorIRStrategy(InjectorIRStrategy::getDefaultOps()));
  Strategies.emplace_back(new InstDeleterIRStrategy());

  return std::make_unique<IRMutator>(std::move(Types), std::move(Strategies));
}

extern "C" LLVM_ATTRIBUTE_USED size_t LLVMFuzzerCustomMutator(
    uint8_t *Data, size_t Size, size_t MaxSize, unsigned int Seed) {
  LLVMContext Context;
  std::unique_ptr<Module> M;
  if (Size <= 1)
    // We get bogus data given an empty corpus - just create a new module.
    M.reset(new Module("M", Context));
  else
    M = parseModule(Data, Size, Context);

  Mutator->mutateModule(*M, Seed, Size, MaxSize);

  return writeModule(*M, Data, MaxSize);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size <= 1)
    // We get bogus data given an empty corpus - ignore it.
    return 0;

  LLVMContext Context;
  auto M = parseAndVerify(Data, Size, Context);
  if (!M) {
    errs() << "error: input module is broken!\n";
    return 0;
  }

  // Set up the module to build for our target.
  M->setTargetTriple(TM->getTargetTriple().normalize());
  M->setDataLayout(TM->createDataLayout());

  // Build up a PM to do instruction selection.
  legacy::PassManager PM;
  TargetLibraryInfoImpl TLII(TM->getTargetTriple());
  PM.add(new TargetLibraryInfoWrapperPass(TLII));
  raw_null_ostream OS;
  TM->addPassesToEmitFile(PM, OS, nullptr, CGFT_Null);
  PM.run(*M);

  return 0;
}

static void handleLLVMFatalError(void *, const std::string &Message, bool) {
  // TODO: Would it be better to call into the fuzzer internals directly?
  dbgs() << "LLVM ERROR: " << Message << "\n"
         << "Aborting to trigger fuzzer exit handling.\n";
  abort();
}

extern "C" LLVM_ATTRIBUTE_USED int LLVMFuzzerInitialize(int *argc,
                                                        char ***argv) {
  EnableDebugBuffering = true;

  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  handleExecNameEncodedBEOpts(*argv[0]);
  parseFuzzerCLOpts(*argc, *argv);

  if (TargetTriple.empty()) {
    errs() << *argv[0] << ": -mtriple must be specified\n";
    exit(1);
  }

  Triple TheTriple = Triple(Triple::normalize(TargetTriple));

  // Get the target specific parser.
  std::string Error;
  const Target *TheTarget =
      TargetRegistry::lookupTarget(MArch, TheTriple, Error);
  if (!TheTarget) {
    errs() << argv[0] << ": " << Error;
    return 1;
  }

  // Set up the pipeline like llc does.
  std::string CPUStr = getCPUStr(), FeaturesStr = getFeaturesStr();

  CodeGenOpt::Level OLvl = CodeGenOpt::Default;
  switch (OptLevel) {
  default:
    errs() << argv[0] << ": invalid optimization level.\n";
    return 1;
  case ' ': break;
  case '0': OLvl = CodeGenOpt::None; break;
  case '1': OLvl = CodeGenOpt::Less; break;
  case '2': OLvl = CodeGenOpt::Default; break;
  case '3': OLvl = CodeGenOpt::Aggressive; break;
  }

  TargetOptions Options = InitTargetOptionsFromCodeGenFlags();
  TM.reset(TheTarget->createTargetMachine(TheTriple.getTriple(), CPUStr,
                                          FeaturesStr, Options, getRelocModel(),
                                          getCodeModel(), OLvl));
  assert(TM && "Could not allocate target machine!");

  // Make sure we print the summary and the current unit when LLVM errors out.
  install_fatal_error_handler(handleLLVMFatalError, nullptr);

  // Finally, create our mutator.
  Mutator = createISelMutator();
  return 0;
}
