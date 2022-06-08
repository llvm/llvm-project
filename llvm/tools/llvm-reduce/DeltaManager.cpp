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
#include "deltas/ReduceFunctionBodies.h"
#include "deltas/ReduceFunctions.h"
#include "deltas/ReduceGlobalObjects.h"
#include "deltas/ReduceGlobalValues.h"
#include "deltas/ReduceGlobalVarInitializers.h"
#include "deltas/ReduceGlobalVars.h"
#include "deltas/ReduceIRReferences.h"
#include "deltas/ReduceInstructionFlagsMIR.h"
#include "deltas/ReduceInstructions.h"
#include "deltas/ReduceInstructionsMIR.h"
#include "deltas/ReduceMetadata.h"
#include "deltas/ReduceModuleData.h"
#include "deltas/ReduceOperandBundles.h"
#include "deltas/ReduceOperands.h"
#include "deltas/ReduceOperandsSkip.h"
#include "deltas/ReduceOperandsToArgs.h"
#include "deltas/ReduceRegisterUses.h"
#include "deltas/ReduceSpecialGlobals.h"
#include "deltas/ReduceVirtualRegisters.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

extern cl::OptionCategory LLVMReduceOptions;
static cl::opt<std::string>
    DeltaPasses("delta-passes",
                cl::desc("Delta passes to run, separated by commas. By "
                         "default, run all delta passes."),
                cl::cat(LLVMReduceOptions));

#define DELTA_PASSES                                                           \
  DELTA_PASS("special-globals", reduceSpecialGlobalsDeltaPass)                 \
  DELTA_PASS("aliases", reduceAliasesDeltaPass)                                \
  DELTA_PASS("function-bodies", reduceFunctionBodiesDeltaPass)                 \
  DELTA_PASS("functions", reduceFunctionsDeltaPass)                            \
  DELTA_PASS("basic-blocks", reduceBasicBlocksDeltaPass)                       \
  DELTA_PASS("global-values", reduceGlobalValuesDeltaPass)                     \
  DELTA_PASS("global-objects", reduceGlobalObjectsDeltaPass)                   \
  DELTA_PASS("global-initializers", reduceGlobalsInitializersDeltaPass)        \
  DELTA_PASS("global-variables", reduceGlobalsDeltaPass)                       \
  DELTA_PASS("metadata", reduceMetadataDeltaPass)                              \
  DELTA_PASS("arguments", reduceArgumentsDeltaPass)                            \
  DELTA_PASS("instructions", reduceInstructionsDeltaPass)                      \
  DELTA_PASS("operands-zero", reduceOperandsZeroDeltaPass)                     \
  DELTA_PASS("operands-one", reduceOperandsOneDeltaPass)                       \
  DELTA_PASS("operands-undef", reduceOperandsUndefDeltaPass)                   \
  DELTA_PASS("operands-to-args", reduceOperandsToArgsDeltaPass)                \
  DELTA_PASS("operands-skip", reduceOperandsSkipDeltaPass)                     \
  DELTA_PASS("operand-bundles", reduceOperandBundesDeltaPass)                  \
  DELTA_PASS("attributes", reduceAttributesDeltaPass)                          \
  DELTA_PASS("module-data", reduceModuleDataDeltaPass)

#define DELTA_PASSES_MIR                                                       \
  DELTA_PASS("instructions", reduceInstructionsMIRDeltaPass)                   \
  DELTA_PASS("ir-instruction-references",                                      \
             reduceIRInstructionReferencesDeltaPass)                           \
  DELTA_PASS("ir-block-references", reduceIRBlockReferencesDeltaPass)          \
  DELTA_PASS("ir-function-references", reduceIRFunctionReferencesDeltaPass)    \
  DELTA_PASS("instruction-flags", reduceInstructionFlagsMIRDeltaPass)          \
  DELTA_PASS("register-uses", reduceRegisterUsesMIRDeltaPass)                  \
  DELTA_PASS("register-hints", reduceVirtualRegisterHintsDeltaPass)

static void runAllDeltaPasses(TestRunner &Tester) {
#define DELTA_PASS(NAME, FUNC) FUNC(Tester);
  if (Tester.getProgram().isMIR()) {
    DELTA_PASSES_MIR
  } else {
    DELTA_PASSES
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
    DELTA_PASSES_MIR
  } else {
    DELTA_PASSES
  }
#undef DELTA_PASS
  errs() << "unknown pass \"" << PassName << "\"\n";
  exit(1);
}

void llvm::printDeltaPasses(raw_ostream &OS) {
  OS << "Delta passes (pass to `--delta-passes=` as a comma separated list):\n";
#define DELTA_PASS(NAME, FUNC) OS << "  " << NAME << "\n";
  OS << " IR:\n";
  DELTA_PASSES
  OS << " MIR:\n";
  DELTA_PASSES_MIR
#undef DELTA_PASS
}

void llvm::runDeltaPasses(TestRunner &Tester, int MaxPassIterations) {
  uint64_t OldComplexity = Tester.getProgram().getComplexityScore();
  for (int Iter = 0; Iter < MaxPassIterations; ++Iter) {
    if (DeltaPasses.empty()) {
      runAllDeltaPasses(Tester);
    } else {
      StringRef Passes = DeltaPasses;
      while (!Passes.empty()) {
        auto Split = Passes.split(",");
        runDeltaPassName(Tester, Split.first);
        Passes = Split.second;
      }
    }
    uint64_t NewComplexity = Tester.getProgram().getComplexityScore();
    if (NewComplexity >= OldComplexity)
      break;
    OldComplexity = NewComplexity;
  }
}
