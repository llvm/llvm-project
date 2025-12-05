//===- RippleTestBase.h ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines a RippleTestBase class, which provides helpers to create a
/// Ripple object and LLVM Functions for testing.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UNITTESTS_TRANSFORMS_RIPPLE_RIPPLETESTBASE_H
#define LLVM_UNITTESTS_TRANSFORMS_RIPPLE_RIPPLETESTBASE_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Ripple/Ripple.h"
#include "gtest/gtest.h"

namespace llvm {

class RippleTestBase : public testing::Test {
protected:
  LLVMContext C;
  RippleTestBase() {}
};

class RippleFunctionTest : public RippleTestBase {
protected:
  std::unique_ptr<TargetMachine> TM;
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  // Ripple constructor parameters that are reset for each test
  SmallVector<std::pair<Ripple::PEIdentifier, Ripple::DimType>, 4>
      DimensionTypes;
  Ripple::ProcessingStatus PS;
  DenseSet<AssertingVH<Function>> SpecializationsPending;
  DenseSet<AssertingVH<Function>> SpecializationsAvailable;

  RippleFunctionTest() {
    // Initialize LLVM targets for TargetMachine creation
    InitializeAllTargets();
    InitializeAllTargetMCs();
    InitializeAllAsmPrinters();
    InitializeAllAsmParsers();

    // Create a default TargetMachine (using host triple)
    std::string Error;
    Triple TheTriple(sys::getDefaultTargetTriple());
    const Target *TheTarget = TargetRegistry::lookupTarget(TheTriple, Error);

    if (TheTarget) {
      TargetOptions Options;
      TM.reset(TheTarget->createTargetMachine(TheTriple, "generic", "", Options,
                                              std::nullopt));
    }

    // Create FunctionAnalysisManager
    {
      PassBuilder PB(TM.get());
      PB.registerModuleAnalyses(MAM);
      PB.registerCGSCCAnalyses(CGAM);
      PB.registerFunctionAnalyses(FAM);
      PB.registerLoopAnalyses(LAM);
      PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
    }

    DimensionTypes.push_back({0, Ripple::VectorDimension});
  }

  void SetUp() override {
    // Reset per-test state
    SpecializationsPending.clear();
    SpecializationsAvailable.clear();
    FAM.clear();
    MAM.clear();
    LAM.clear();
    CGAM.clear();
  }

  /// Helper to create a Ripple object for testing
  /// @param F The function to process
  /// @return A unique_ptr to the created Ripple object
  std::unique_ptr<Ripple> createRipple(Function &F) {
    if (!TM) {
      return nullptr;
    }

    return std::make_unique<Ripple>(TM.get(), F, FAM, DimensionTypes, PS,
                                    SpecializationsPending,
                                    SpecializationsAvailable);
  }
};

} // namespace llvm

#endif // LLVM_UNITTESTS_TRANSFORMS_RIPPLE_RIPPLETESTBASE_H
