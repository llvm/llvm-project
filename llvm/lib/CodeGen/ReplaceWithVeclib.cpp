//=== ReplaceWithVeclib.cpp - Replace vector intrinsics with veclib calls -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Replaces calls to LLVM vector intrinsics (i.e., calls to LLVM intrinsics
// with vector operands) with matching calls to functions from a vector
// library (e.g., libmvec, SVML) according to TargetLibraryInfo.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ReplaceWithVeclib.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/DemandedBits.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "replace-with-veclib"

STATISTIC(NumCallsReplaced,
          "Number of calls to intrinsics that have been replaced.");

STATISTIC(NumTLIFuncDeclAdded,
          "Number of vector library function declarations added.");

STATISTIC(NumFuncUsedAdded,
          "Number of functions added to `llvm.compiler.used`");

static bool replaceWithTLIFunction(CallInst &CI, const StringRef TLIName) {
  Module *M = CI.getModule();

  Function *OldFunc = CI.getCalledFunction();

  // Check if the vector library function is already declared in this module,
  // otherwise insert it.
  Function *TLIFunc = M->getFunction(TLIName);
  if (!TLIFunc) {
    TLIFunc = Function::Create(OldFunc->getFunctionType(),
                               Function::ExternalLinkage, TLIName, *M);
    TLIFunc->copyAttributesFrom(OldFunc);

    LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Added vector library function `"
                      << TLIName << "` of type `" << *(TLIFunc->getType())
                      << "` to module.\n");

    ++NumTLIFuncDeclAdded;

    // Add the freshly created function to llvm.compiler.used,
    // similar to as it is done in InjectTLIMappings
    appendToCompilerUsed(*M, {TLIFunc});

    LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Adding `" << TLIName
                      << "` to `@llvm.compiler.used`.\n");
    ++NumFuncUsedAdded;
  }

  // Replace the call to the vector intrinsic with a call
  // to the corresponding function from the vector library.
  IRBuilder<> IRBuilder(&CI);
  SmallVector<Value *> Args(CI.args());
  // Preserve the operand bundles.
  SmallVector<OperandBundleDef, 1> OpBundles;
  CI.getOperandBundlesAsDefs(OpBundles);
  CallInst *Replacement = IRBuilder.CreateCall(TLIFunc, Args, OpBundles);
  assert(OldFunc->getFunctionType() == TLIFunc->getFunctionType() &&
         "Expecting function types to be identical");
  CI.replaceAllUsesWith(Replacement);
  if (isa<FPMathOperator>(Replacement)) {
    // Preserve fast math flags for FP math.
    Replacement->copyFastMathFlags(&CI);
  }

  LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Replaced call to `"
                    << OldFunc->getName() << "` with call to `" << TLIName
                    << "`.\n");
  ++NumCallsReplaced;
  return true;
}

static bool replaceWithCallToVeclib(const TargetLibraryInfo &TLI,
                                    CallInst &CI) {
  if (!CI.getCalledFunction()) {
    return false;
  }

  auto IntrinsicID = CI.getCalledFunction()->getIntrinsicID();
  if (IntrinsicID == Intrinsic::not_intrinsic) {
    // Replacement is only performed for intrinsic functions
    return false;
  }

  // Convert vector arguments to scalar type and check that
  // all vector operands have identical vector width.
  ElementCount VF = ElementCount::getFixed(0);
  SmallVector<Type *> ScalarTypes;
  for (auto Arg : enumerate(CI.args())) {
    auto *ArgType = Arg.value()->getType();
    // Vector calls to intrinsics can still have
    // scalar operands for specific arguments.
    if (hasVectorIntrinsicScalarOpd(IntrinsicID, Arg.index())) {
      ScalarTypes.push_back(ArgType);
    } else {
      // The argument in this place should be a vector if
      // this is a call to a vector intrinsic.
      auto *VectorArgTy = dyn_cast<VectorType>(ArgType);
      if (!VectorArgTy) {
        // The argument is not a vector, do not perform
        // the replacement.
        return false;
      }
      ElementCount NumElements = VectorArgTy->getElementCount();
      if (NumElements.isScalable()) {
        // The current implementation does not support
        // scalable vectors.
        return false;
      }
      if (VF.isNonZero() && VF != NumElements) {
        // The different arguments differ in vector size.
        return false;
      } else {
        VF = NumElements;
      }
      ScalarTypes.push_back(VectorArgTy->getElementType());
    }
  }

  // Try to reconstruct the name for the scalar version of this
  // intrinsic using the intrinsic ID and the argument types
  // converted to scalar above.
  std::string ScalarName;
  if (Intrinsic::isOverloaded(IntrinsicID)) {
    ScalarName = Intrinsic::getName(IntrinsicID, ScalarTypes, CI.getModule());
  } else {
    ScalarName = Intrinsic::getName(IntrinsicID).str();
  }

  if (!TLI.isFunctionVectorizable(ScalarName)) {
    // The TargetLibraryInfo does not contain a vectorized version of
    // the scalar function.
    return false;
  }

  // Try to find the mapping for the scalar version of this intrinsic
  // and the exact vector width of the call operands in the
  // TargetLibraryInfo.
  const std::string TLIName =
      std::string(TLI.getVectorizedFunction(ScalarName, VF));

  LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Looking up TLI mapping for `"
                    << ScalarName << "` and vector width " << VF << ".\n");

  if (!TLIName.empty()) {
    // Found the correct mapping in the TargetLibraryInfo,
    // replace the call to the intrinsic with a call to
    // the vector library function.
    LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Found TLI function `" << TLIName
                      << "`.\n");
    return replaceWithTLIFunction(CI, TLIName);
  }

  return false;
}

static bool runImpl(const TargetLibraryInfo &TLI, Function &F) {
  bool Changed = false;
  SmallVector<CallInst *> ReplacedCalls;
  for (auto &I : instructions(F)) {
    if (auto *CI = dyn_cast<CallInst>(&I)) {
      if (replaceWithCallToVeclib(TLI, *CI)) {
        ReplacedCalls.push_back(CI);
        Changed = true;
      }
    }
  }
  // Erase the calls to the intrinsics that have been replaced
  // with calls to the vector library.
  for (auto *CI : ReplacedCalls) {
    CI->eraseFromParent();
  }
  return Changed;
}

////////////////////////////////////////////////////////////////////////////////
// New pass manager implementation.
////////////////////////////////////////////////////////////////////////////////
PreservedAnalyses ReplaceWithVeclib::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  const TargetLibraryInfo &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  auto Changed = runImpl(TLI, F);
  if (Changed) {
    PreservedAnalyses PA;
    PA.preserveSet<CFGAnalyses>();
    PA.preserve<TargetLibraryAnalysis>();
    PA.preserve<ScalarEvolutionAnalysis>();
    PA.preserve<LoopAccessAnalysis>();
    PA.preserve<DemandedBitsAnalysis>();
    PA.preserve<OptimizationRemarkEmitterAnalysis>();
    return PA;
  } else {
    // The pass did not replace any calls, hence it preserves all analyses.
    return PreservedAnalyses::all();
  }
}

////////////////////////////////////////////////////////////////////////////////
// Legacy PM Implementation.
////////////////////////////////////////////////////////////////////////////////
bool ReplaceWithVeclibLegacy::runOnFunction(Function &F) {
  const TargetLibraryInfo &TLI =
      getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
  return runImpl(TLI, F);
}

void ReplaceWithVeclibLegacy::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<TargetLibraryInfoWrapperPass>();
  AU.addPreserved<TargetLibraryInfoWrapperPass>();
  AU.addPreserved<ScalarEvolutionWrapperPass>();
  AU.addPreserved<AAResultsWrapperPass>();
  AU.addPreserved<LoopAccessLegacyAnalysis>();
  AU.addPreserved<DemandedBitsWrapperPass>();
  AU.addPreserved<OptimizationRemarkEmitterWrapperPass>();
  AU.addPreserved<GlobalsAAWrapperPass>();
}

////////////////////////////////////////////////////////////////////////////////
// Legacy Pass manager initialization
////////////////////////////////////////////////////////////////////////////////
char ReplaceWithVeclibLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(ReplaceWithVeclibLegacy, DEBUG_TYPE,
                      "Replace intrinsics with calls to vector library", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(ReplaceWithVeclibLegacy, DEBUG_TYPE,
                    "Replace intrinsics with calls to vector library", false,
                    false)

FunctionPass *llvm::createReplaceWithVeclibLegacyPass() {
  return new ReplaceWithVeclibLegacy();
}
