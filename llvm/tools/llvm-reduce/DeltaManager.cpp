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
#include "DeltaPass.h"
#include "TestRunner.h"
#include "deltas/ReduceAliases.h"
#include "deltas/ReduceArguments.h"
#include "deltas/ReduceAttributes.h"
#include "deltas/ReduceBasicBlocks.h"
#include "deltas/ReduceDIMetadata.h"
#include "deltas/ReduceDbgRecords.h"
#include "deltas/ReduceDistinctMetadata.h"
#include "deltas/ReduceFunctionBodies.h"
#include "deltas/ReduceFunctions.h"
#include "deltas/ReduceGlobalObjects.h"
#include "deltas/ReduceGlobalValues.h"
#include "deltas/ReduceGlobalVarInitializers.h"
#include "deltas/ReduceGlobalVars.h"
#include "deltas/ReduceIRReferences.h"
#include "deltas/ReduceInlineCallSites.h"
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
#include "deltas/ReduceTargetFeaturesAttr.h"
#include "deltas/ReduceUsingSimplifyCFG.h"
#include "deltas/ReduceValuesToReturn.h"
#include "deltas/ReduceVirtualRegisters.h"
#include "deltas/RunIRPasses.h"
#include "deltas/SimplifyInstructions.h"
#include "deltas/StripDebugInfo.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CommandLine.h"

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

// Generate two separate Pass lists: IR_Passes and MIR_Passes
static const DeltaPass IR_Passes[] = {
#undef DELTA_PASS_IR
#undef DELTA_PASS_MIR
#define DELTA_PASS_IR(NAME, FUNC, DESC) {NAME, FUNC, DESC},
#include "DeltaPasses.def"
#undef DELTA_PASS_IR
};

static const DeltaPass MIR_Passes[] = {
#undef DELTA_PASS_IR
#undef DELTA_PASS_MIR
#define DELTA_PASS_MIR(NAME, FUNC, DESC) {NAME, FUNC, DESC},
#include "DeltaPasses.def"
#undef DELTA_PASS_MIR
};

static void runAllDeltaPasses(TestRunner &Tester,
                              const SmallStringSet &SkipPass) {
  if (Tester.getProgram().isMIR()) {
    for (const DeltaPass &Pass : MIR_Passes) {
      if (!SkipPass.count(Pass.Name)) {
        runDeltaPass(Tester, Pass);
      }
    }
  } else {
    for (const DeltaPass &Pass : IR_Passes) {
      if (!SkipPass.count(Pass.Name)) {
        runDeltaPass(Tester, Pass);
      }
    }
  }
}

static void runDeltaPassName(TestRunner &Tester, StringRef PassName) {
  if (Tester.getProgram().isMIR()) {
    for (const DeltaPass &Pass : MIR_Passes) {
      if (PassName == Pass.Name) {
        runDeltaPass(Tester, Pass);
        return;
      }
    }
  } else {
    for (const DeltaPass &Pass : IR_Passes) {
      if (PassName == Pass.Name) {
        runDeltaPass(Tester, Pass);
        return;
      }
    }
  }

  // We should have errored on unrecognized passes before trying to run
  // anything.
  llvm_unreachable("unknown delta pass");
}

void llvm::printDeltaPasses(raw_ostream &OS) {
  OS << "Delta passes (pass to `--delta-passes=` as a comma separated list):\n";
  OS << " IR:\n";
  for (const DeltaPass &Pass : IR_Passes) {
    OS << "  " << Pass.Name << '\n';
  }
  OS << " MIR:\n";
  for (const DeltaPass &Pass : MIR_Passes) {
    OS << "  " << Pass.Name << '\n';
  }
}

// Built a set of available delta passes.
static void collectPassNames(const TestRunner &Tester,
                             SmallStringSet &NameSet) {
  for (const DeltaPass &Pass : MIR_Passes) {
    NameSet.insert(Pass.Name);
  }
  for (const DeltaPass &Pass : IR_Passes) {
    NameSet.insert(Pass.Name);
  }
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
      runAllDeltaPasses(Tester, SkipPassSet);
    } else {
      for (StringRef PassName : DeltaPasses) {
        if (!SkipPassSet.count(PassName))
          runDeltaPassName(Tester, PassName);
      }
    }

    uint64_t NewComplexity = Tester.getProgram().getComplexityScore();
    if (NewComplexity >= OldComplexity)
      break;
    OldComplexity = NewComplexity;
  }
}
