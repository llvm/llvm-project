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
#include "TestRunner.h"
#include "deltas/Delta.h"
#include "deltas/ReduceAliases.h"
#include "deltas/ReduceArguments.h"
#include "deltas/ReduceAttributes.h"
#include "deltas/ReduceBasicBlocks.h"
#include "deltas/ReduceFunctionBodies.h"
#include "deltas/ReduceFunctions.h"
#include "deltas/ReduceGlobalValues.h"
#include "deltas/ReduceGlobalVarInitializers.h"
#include "deltas/ReduceGlobalVars.h"
#include "deltas/ReduceInstructions.h"
#include "deltas/ReduceMetadata.h"
#include "deltas/ReduceModuleData.h"
#include "deltas/ReduceOperandBundles.h"
#include "deltas/ReduceOperands.h"
#include "deltas/ReduceOperandsToArgs.h"
#include "deltas/ReduceSpecialGlobals.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

static cl::opt<std::string>
    DeltaPasses("delta-passes",
                cl::desc("Delta passes to run, separated by commas. By "
                         "default, run all delta passes."));

#define DELTA_PASSES                                                           \
  DELTA_PASS("special-globals", reduceSpecialGlobalsDeltaPass)                 \
  DELTA_PASS("aliases", reduceAliasesDeltaPass)                                \
  DELTA_PASS("function-bodies", reduceFunctionBodiesDeltaPass)                 \
  DELTA_PASS("functions", reduceFunctionsDeltaPass)                            \
  DELTA_PASS("basic-blocks", reduceBasicBlocksDeltaPass)                       \
  DELTA_PASS("global-values", reduceGlobalValuesDeltaPass)                     \
  DELTA_PASS("global-initializers", reduceGlobalsInitializersDeltaPass)        \
  DELTA_PASS("global-variables", reduceGlobalsDeltaPass)                       \
  DELTA_PASS("metadata", reduceMetadataDeltaPass)                              \
  DELTA_PASS("arguments", reduceArgumentsDeltaPass)                            \
  DELTA_PASS("instructions", reduceInstructionsDeltaPass)                      \
  DELTA_PASS("operands", reduceOperandsDeltaPass)                              \
  DELTA_PASS("operands-to-args", reduceOperandsToArgsDeltaPass)                \
  DELTA_PASS("operand-bundles", reduceOperandBundesDeltaPass)                  \
  DELTA_PASS("attributes", reduceAttributesDeltaPass)                          \
  DELTA_PASS("module-data", reduceModuleDataDeltaPass)

static void runAllDeltaPasses(TestRunner &Tester) {
#define DELTA_PASS(NAME, FUNC) FUNC(Tester);
  DELTA_PASSES
#undef DELTA_PASS
}

static void runDeltaPassName(TestRunner &Tester, StringRef PassName) {
#define DELTA_PASS(NAME, FUNC)                                                 \
  if (PassName == NAME) {                                                      \
    FUNC(Tester);                                                              \
    return;                                                                    \
  }
  DELTA_PASSES
#undef DELTA_PASS
  errs() << "unknown pass \"" << PassName << "\"";
  exit(1);
}

void llvm::printDeltaPasses(raw_ostream &OS) {
  OS << "Delta passes (pass to `--delta-passes=` as a comma separated list):\n";
#define DELTA_PASS(NAME, FUNC) OS << "  " << NAME << "\n";
  DELTA_PASSES
#undef DELTA_PASS
}

void llvm::runDeltaPasses(TestRunner &Tester) {
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
}
