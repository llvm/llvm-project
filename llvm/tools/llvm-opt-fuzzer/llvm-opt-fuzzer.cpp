//===--- llvm-opt-fuzzer.cpp - Fuzzer for instruction selection ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Tool to fuzz optimization passes using libFuzzer.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/FuzzMutate/FuzzerCLI.h"
#include "llvm/FuzzMutate/IRMutator.h"
#include "llvm/FuzzMutate/Operations.h"
#include "llvm/FuzzMutate/Random.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

using namespace llvm;

static cl::opt<std::string>
    TargetTripleStr("mtriple", cl::desc("Override target triple for module"));

// Passes to run for this fuzzer instance. Expects new pass manager syntax.
static cl::opt<std::string> PassPipeline(
    "passes",
    cl::desc("A textual description of the pass pipeline for testing"));

static std::unique_ptr<IRMutator> Mutator;
static std::unique_ptr<TargetMachine> TM;

std::unique_ptr<IRMutator> createOptMutator() {
  std::vector<TypeGetter> Types{
      Type::getInt1Ty,  Type::getInt8Ty,  Type::getInt16Ty, Type::getInt32Ty,
      Type::getInt64Ty, Type::getFloatTy, Type::getDoubleTy};

  std::vector<std::unique_ptr<IRMutationStrategy>> Strategies;
  Strategies.push_back(
      llvm::make_unique<InjectorIRStrategy>(
          InjectorIRStrategy::getDefaultOps()));
  Strategies.push_back(
      llvm::make_unique<InstDeleterIRStrategy>());

  return llvm::make_unique<IRMutator>(std::move(Types), std::move(Strategies));
}

extern "C" LLVM_ATTRIBUTE_USED size_t LLVMFuzzerCustomMutator(
    uint8_t *Data, size_t Size, size_t MaxSize, unsigned int Seed) {

  assert(Mutator &&
      "IR mutator should have been created during fuzzer initialization");

  LLVMContext Context;
  auto M = parseModule(Data, Size, Context);
  if (!M || verifyModule(*M, &errs())) {
    errs() << "error: mutator input module is broken!\n";
    return 0;
  }

  Mutator->mutateModule(*M, Seed, Size, MaxSize);

#ifndef NDEBUG
  if (verifyModule(*M, &errs())) {
    errs() << "mutation result doesn't pass verification\n";
    M->dump();
    abort();
  }
#endif

  return writeModule(*M, Data, MaxSize);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  assert(TM && "Should have been created during fuzzer initialization");

  if (Size <= 1)
    // We get bogus data given an empty corpus - ignore it.
    return 0;

  // Parse module
  //

  LLVMContext Context;
  auto M = parseModule(Data, Size, Context);
  if (!M || verifyModule(*M, &errs())) {
    errs() << "error: input module is broken!\n";
    return 0;
  }

  // Set up target dependant options
  //

  M->setTargetTriple(TM->getTargetTriple().normalize());
  M->setDataLayout(TM->createDataLayout());
  setFunctionAttributes(TM->getTargetCPU(), TM->getTargetFeatureString(), *M);

  // Create pass pipeline
  //

  PassBuilder PB(TM.get());

  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModulePassManager MPM;
  ModuleAnalysisManager MAM;

  FAM.registerPass([&] { return PB.buildDefaultAAPipeline(); });
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  bool Ok = PB.parsePassPipeline(MPM, PassPipeline, false, false);
  assert(Ok && "Should have been checked during fuzzer initialization");
  (void)Ok; // silence unused variable warning on release builds

  // Run passes which we need to test
  //

  MPM.run(*M, MAM);

  // Check that passes resulted in a correct code
  if (verifyModule(*M, &errs())) {
    errs() << "Transformation resulted in an invalid module\n";
    abort();
  }

  return 0;
}

static void handleLLVMFatalError(void *, const std::string &Message, bool) {
  // TODO: Would it be better to call into the fuzzer internals directly?
  dbgs() << "LLVM ERROR: " << Message << "\n"
         << "Aborting to trigger fuzzer exit handling.\n";
  abort();
}

extern "C" LLVM_ATTRIBUTE_USED int LLVMFuzzerInitialize(
    int *argc, char ***argv) {
  EnableDebugBuffering = true;

  // Make sure we print the summary and the current unit when LLVM errors out.
  install_fatal_error_handler(handleLLVMFatalError, nullptr);

  // Initialize llvm
  //

  InitializeAllTargets();
  InitializeAllTargetMCs();

  PassRegistry &Registry = *PassRegistry::getPassRegistry();
  initializeCore(Registry);
  initializeCoroutines(Registry);
  initializeScalarOpts(Registry);
  initializeObjCARCOpts(Registry);
  initializeVectorization(Registry);
  initializeIPO(Registry);
  initializeAnalysis(Registry);
  initializeTransformUtils(Registry);
  initializeInstCombine(Registry);
  initializeInstrumentation(Registry);
  initializeTarget(Registry);

  // Parse input options
  //

  handleExecNameEncodedOptimizerOpts(*argv[0]);
  parseFuzzerCLOpts(*argc, *argv);

  // Create TargetMachine
  //

  if (TargetTripleStr.empty()) {
    errs() << *argv[0] << ": -mtriple must be specified\n";
    exit(1);
  }
  Triple TargetTriple = Triple(Triple::normalize(TargetTripleStr));

  std::string Error;
  const Target *TheTarget =
      TargetRegistry::lookupTarget(MArch, TargetTriple, Error);
  if (!TheTarget) {
    errs() << *argv[0] << ": " << Error;
    exit(1);
  }

  TargetOptions Options = InitTargetOptionsFromCodeGenFlags();
  TM.reset(TheTarget->createTargetMachine(
      TargetTriple.getTriple(), getCPUStr(), getFeaturesStr(),
     Options, getRelocModel(), getCodeModel(), CodeGenOpt::Default));
  assert(TM && "Could not allocate target machine!");

  // Check that pass pipeline is specified and correct
  //

  if (PassPipeline.empty()) {
    errs() << *argv[0] << ": at least one pass should be specified\n";
    exit(1);
  }

  PassBuilder PB(TM.get());
  ModulePassManager MPM;
  if (!PB.parsePassPipeline(MPM, PassPipeline, false, false)) {
    errs() << *argv[0] << ": can't parse pass pipeline\n";
    exit(1);
  }

  // Create mutator
  //

  Mutator = createOptMutator();

  return 0;
}
