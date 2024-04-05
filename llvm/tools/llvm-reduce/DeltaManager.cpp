//===- DeltaManager.cpp - Runs Delta Passes to reduce Input ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file calls each specialized Delta pass in order to reduce the input IR
// file.
//
//===----------------------------------------------------------------------===//

#include "DeltaManager.h"
#include "ReducerWorkItem.h"
#include "TestRunner.h"
#include "deltas/Delta.h"
#include "deltas/ReduceAliases.h"
#include "deltas/ReduceArguments.h"
#include "deltas/ReduceAttributes.h"
#include "deltas/ReduceBasicBlocks.h"
#include "deltas/ReduceDIMetadata.h"
#include "deltas/ReduceDPValues.h"
#include "deltas/ReduceFunctionBodies.h"
#include "deltas/ReduceFunctions.h"
#include "deltas/ReduceGlobalObjects.h"
#include "deltas/ReduceGlobalValues.h"
#include "deltas/ReduceGlobalVarInitializers.h"
#include "deltas/ReduceGlobalVars.h"
#include "deltas/ReduceIRReferences.h"
#include "deltas/ReduceInstructionFlags.h"
#include "deltas/ReduceInstructionFlagsMIR.h"
#include "deltas/ReduceInstructions.h"
#include "deltas/ReduceInstructionsMIR.h"
#include "deltas/ReduceInvokes.h"
#include "deltas/ReduceMemoryOperations.h"
#include "deltas/ReduceMetadata.h"
#include "deltas/ReduceModuleData.h"
#include "deltas/ReduceOpcodes.h"
#include "deltas/ReduceOperandBundles.h"
#include "deltas/ReduceOperands.h"
#include "deltas/ReduceOperandsSkip.h"
#include "deltas/ReduceOperandsToArgs.h"
#include "deltas/ReduceRegisterDefs.h"
#include "deltas/ReduceRegisterMasks.h"
#include "deltas/ReduceRegisterUses.h"
#include "deltas/ReduceSpecialGlobals.h"
#include "deltas/ReduceUsingSimplifyCFG.h"
#include "deltas/ReduceVirtualRegisters.h"
#include "deltas/RunIRPasses.h"
#include "deltas/SimplifyInstructions.h"
#include "deltas/StripDebugInfo.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CommandLine.h"
#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace llvm;

using SmallStringSet = SmallSet<StringRef, 8>;

extern cl::OptionCategory LLVMReduceOptions;
static cl::list<std::string>
    DeltaPasses("delta-passes",
                cl::desc("Delta passes to run, separated by commas. By "
                         "default, run all delta passes."),
                cl::cat(LLVMReduceOptions), cl::CommaSeparated);

static cl::list<std::string>
    SkipDeltaPasses("skip-delta-passes",
                    cl::desc("Delta passes to not run, separated by commas. By "
                             "default, run all delta passes."),
                    cl::cat(LLVMReduceOptions), cl::CommaSeparated);
static cl::opt<bool> RunEachDeltaPassInChildProcess(
    "run-delta-in-child", cl::desc("Run each delta pass in new child process."),
    cl::init(false), cl::cat(LLVMReduceOptions));

#define DELTA_PASSES                                                           \
  do {                                                                         \
    DELTA_PASS("strip-debug-info", stripDebugInfoDeltaPass)                    \
    DELTA_PASS("functions", reduceFunctionsDeltaPass)                          \
    DELTA_PASS("function-bodies", reduceFunctionBodiesDeltaPass)               \
    DELTA_PASS("special-globals", reduceSpecialGlobalsDeltaPass)               \
    DELTA_PASS("aliases", reduceAliasesDeltaPass)                              \
    DELTA_PASS("ifuncs", reduceIFuncsDeltaPass)                                \
    DELTA_PASS("simplify-conditionals-true", reduceConditionalsTrueDeltaPass)  \
    DELTA_PASS("simplify-conditionals-false",                                  \
               reduceConditionalsFalseDeltaPass)                               \
    DELTA_PASS("invokes", reduceInvokesDeltaPass)                              \
    DELTA_PASS("unreachable-basic-blocks",                                     \
               reduceUnreachableBasicBlocksDeltaPass)                          \
    DELTA_PASS("basic-blocks", reduceBasicBlocksDeltaPass)                     \
    DELTA_PASS("simplify-cfg", reduceUsingSimplifyCFGDeltaPass)                \
    DELTA_PASS("function-data", reduceFunctionDataDeltaPass)                   \
    DELTA_PASS("global-values", reduceGlobalValuesDeltaPass)                   \
    DELTA_PASS("global-objects", reduceGlobalObjectsDeltaPass)                 \
    DELTA_PASS("global-initializers", reduceGlobalsInitializersDeltaPass)      \
    DELTA_PASS("global-variables", reduceGlobalsDeltaPass)                     \
    DELTA_PASS("di-metadata", reduceDIMetadataDeltaPass)                       \
    DELTA_PASS("dpvalues", reduceDPValuesDeltaPass)                            \
    DELTA_PASS("metadata", reduceMetadataDeltaPass)                            \
    DELTA_PASS("named-metadata", reduceNamedMetadataDeltaPass)                 \
    DELTA_PASS("arguments", reduceArgumentsDeltaPass)                          \
    DELTA_PASS("instructions", reduceInstructionsDeltaPass)                    \
    DELTA_PASS("simplify-instructions", simplifyInstructionsDeltaPass)         \
    DELTA_PASS("ir-passes", runIRPassesDeltaPass)                              \
    DELTA_PASS("operands-zero", reduceOperandsZeroDeltaPass)                   \
    DELTA_PASS("operands-one", reduceOperandsOneDeltaPass)                     \
    DELTA_PASS("operands-nan", reduceOperandsNaNDeltaPass)                     \
    DELTA_PASS("operands-to-args", reduceOperandsToArgsDeltaPass)              \
    DELTA_PASS("operands-skip", reduceOperandsSkipDeltaPass)                   \
    DELTA_PASS("operand-bundles", reduceOperandBundesDeltaPass)                \
    DELTA_PASS("attributes", reduceAttributesDeltaPass)                        \
    DELTA_PASS("module-data", reduceModuleDataDeltaPass)                       \
    DELTA_PASS("opcodes", reduceOpcodesDeltaPass)                              \
    DELTA_PASS("volatile", reduceVolatileInstructionsDeltaPass)                \
    DELTA_PASS("atomic-ordering", reduceAtomicOrderingDeltaPass)               \
    DELTA_PASS("syncscopes", reduceAtomicSyncScopesDeltaPass)                  \
    DELTA_PASS("instruction-flags", reduceInstructionFlagsDeltaPass)           \
  } while (false)

#define DELTA_PASSES_MIR                                                       \
  do {                                                                         \
    DELTA_PASS("instructions", reduceInstructionsMIRDeltaPass)                 \
    DELTA_PASS("ir-instruction-references",                                    \
               reduceIRInstructionReferencesDeltaPass)                         \
    DELTA_PASS("ir-block-references", reduceIRBlockReferencesDeltaPass)        \
    DELTA_PASS("ir-function-references", reduceIRFunctionReferencesDeltaPass)  \
    DELTA_PASS("instruction-flags", reduceInstructionFlagsMIRDeltaPass)        \
    DELTA_PASS("register-uses", reduceRegisterUsesMIRDeltaPass)                \
    DELTA_PASS("register-defs", reduceRegisterDefsMIRDeltaPass)                \
    DELTA_PASS("register-hints", reduceVirtualRegisterHintsDeltaPass)          \
    DELTA_PASS("register-masks", reduceRegisterMasksMIRDeltaPass)              \
  } while (false)

static void runAllDeltaPasses(TestRunner &Tester,
                              const SmallStringSet &SkipPass) {
#define DELTA_PASS(NAME, FUNC)                                                 \
  if (!SkipPass.count(NAME)) {                                                 \
    FUNC(Tester);                                                              \
  }
  if (Tester.getProgram().isMIR()) {
    DELTA_PASSES_MIR;
  } else {
    DELTA_PASSES;
  }
#undef DELTA_PASS
}

static void runDeltaPassName(TestRunner &Tester, StringRef PassName) {
#define DELTA_PASS(NAME, FUNC)                                                 \
  if (PassName == NAME) {                                                      \
    FUNC(Tester);                                                              \
    return;                                                                    \
  }
  if (Tester.getProgram().isMIR()) {
    DELTA_PASSES_MIR;
  } else {
    DELTA_PASSES;
  }
#undef DELTA_PASS

  // We should have errored on unrecognized passes before trying to run
  // anything.
  llvm_unreachable("unknown delta pass");
}

static void runAllDeltaPassesInChild(TestRunner &Tester,
                                     const SmallStringSet &SkipPass) {
#ifdef _WIN32
  errs() << "runAllDeltaPassesInChild() is only available on POSIX systems. \n";
  return;
#endif

#ifndef _WIN32
#define DELTA_PASS(NAME, FUNC)                                                 \
  if (!SkipPass.count(NAME)) {                                                 \
    pid_t CPid = fork();                                                       \
    if (CPid == -1) {                                                          \
      errs() << "Could not create child process. \n";                          \
      return;                                                                  \
    }                                                                          \
    if (CPid == 0) {                                                           \
      FUNC(Tester);                                                            \
    } else {                                                                   \
      /*parent waits for child to finish.*/                                    \
      int ReturnStatus;                                                        \
      waitpid(CPid, &ReturnStatus, 0);                                         \
      if (ReturnStatus != 0) {                                                 \
        errs() << "Reduction " << NAME << " failed. \n";                       \
        exit(ReturnStatus);                                                    \
      }                                                                        \
      /* if child finishes fine we kill parent otherwise it will overwrite*/   \
      /* results to output files.*/                                            \
      exit(0);                                                                 \
    }                                                                          \
  }

  if (Tester.getProgram().isMIR()) {
    DELTA_PASSES_MIR;
  } else {
    DELTA_PASSES;
  }
#undef DELTA_PASS
#endif
}

static void runDeltaPassNameInChild(TestRunner &Tester, StringRef PassName) {
#ifdef _WIN32
  errs() << "runDeltaPassNameInChild() is only available on POSIX systems. \n";
  return;
#endif

#ifndef _WIN32
#define DELTA_PASS(NAME, FUNC)                                                 \
  if (PassName == NAME) {                                                      \
    pid_t CPid = fork();                                                       \
    if (CPid == -1) {                                                          \
      errs() << "Could not create child process.\n";                           \
      return;                                                                  \
    }                                                                          \
    if (CPid == 0) {                                                           \
      FUNC(Tester);                                                            \
      return;                                                                  \
    } else {                                                                   \
      /*parent waits for child to finish.*/                                    \
      int ReturnStatus;                                                        \
      waitpid(CPid, &ReturnStatus, 0);                                         \
      if (ReturnStatus != 0) {                                                 \
        errs() << "Reduction " << NAME << " failed. \n";                       \
        exit(ReturnStatus);                                                    \
      }                                                                        \
      /* if child finishes fine we kill parent otherwise it will overwrite*/   \
      /* results to output files.*/                                            \
      exit(0);                                                                 \
    }                                                                          \
  }

  if (Tester.getProgram().isMIR()) {
    DELTA_PASSES_MIR;
  } else {
    DELTA_PASSES;
  }
#undef DELTA_PASS
#endif

  // We should have errored on unrecognized passes before trying to run
  // anything.
  llvm_unreachable("unknown delta pass");
}

void llvm::printDeltaPasses(raw_ostream &OS) {
  OS << "Delta passes (pass to `--delta-passes=` as a comma separated list):\n";
#define DELTA_PASS(NAME, FUNC) OS << "  " << NAME << "\n";
  OS << " IR:\n";
  DELTA_PASSES;
  OS << " MIR:\n";
  DELTA_PASSES_MIR;
#undef DELTA_PASS
}

// Built a set of available delta passes.
static void collectPassNames(const TestRunner &Tester,
                             SmallStringSet &NameSet) {
#define DELTA_PASS(NAME, FUNC) NameSet.insert(NAME);
  if (Tester.getProgram().isMIR()) {
    DELTA_PASSES_MIR;
  } else {
    DELTA_PASSES;
  }
#undef DELTA_PASS
}

/// Verify all requested or skipped passes are valid names, and return them in a
/// set.
static SmallStringSet handlePassList(const TestRunner &Tester,
                                     const cl::list<std::string> &PassList) {
  SmallStringSet AllPasses;
  collectPassNames(Tester, AllPasses);

  SmallStringSet PassSet;
  for (StringRef PassName : PassList) {
    if (!AllPasses.count(PassName)) {
      errs() << "unknown pass \"" << PassName << "\"\n";
      exit(1);
    }

    PassSet.insert(PassName);
  }

  return PassSet;
}

void llvm::runDeltaPasses(TestRunner &Tester, int MaxPassIterations) {
  uint64_t OldComplexity = Tester.getProgram().getComplexityScore();

  SmallStringSet RunPassSet, SkipPassSet;

  if (!DeltaPasses.empty())
    RunPassSet = handlePassList(Tester, DeltaPasses);

  if (!SkipDeltaPasses.empty())
    SkipPassSet = handlePassList(Tester, SkipDeltaPasses);

  for (int Iter = 0; Iter < MaxPassIterations; ++Iter) {
    if (DeltaPasses.empty()) {
      if (!RunEachDeltaPassInChildProcess)
        runAllDeltaPasses(Tester, SkipPassSet);
      else
        runAllDeltaPassesInChild(Tester, SkipPassSet);
    } else {
      for (StringRef PassName : DeltaPasses) {
        if (!SkipPassSet.count(PassName)) {
          if (!RunEachDeltaPassInChildProcess)
            runDeltaPassName(Tester, PassName);
          else
            runDeltaPassNameInChild(Tester, PassName);
        }
      }
    }

    uint64_t NewComplexity = Tester.getProgram().getComplexityScore();
    if (NewComplexity >= OldComplexity)
      break;
    OldComplexity = NewComplexity;
  }
}
