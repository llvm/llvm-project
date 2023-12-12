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
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/DemandedBits.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "replace-with-veclib"

STATISTIC(NumCallsReplaced,
          "Number of calls to intrinsics that have been replaced.");

STATISTIC(NumTLIFuncDeclAdded,
          "Number of vector library function declarations added.");

STATISTIC(NumFuncUsedAdded,
          "Number of functions added to `llvm.compiler.used`");

/// Returns a vector Function that it adds to the Module \p M. When an \p
/// OptOldFunc is given, it copies its attributes to the newly created Function.
Function *getTLIFunction(Module *M, FunctionType *VectorFTy,
                         std::optional<Function *> OptOldFunc,
                         const StringRef TLIName) {
  Function *TLIFunc = M->getFunction(TLIName);
  if (!TLIFunc) {
    TLIFunc =
        Function::Create(VectorFTy, Function::ExternalLinkage, TLIName, *M);
    if (OptOldFunc)
      TLIFunc->copyAttributesFrom(*OptOldFunc);

    LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Added vector library function `"
                      << TLIName << "` of type `" << *(TLIFunc->getType())
                      << "` to module.\n");

    ++NumTLIFuncDeclAdded;
    // Add the freshly created function to llvm.compiler.used, similar to as it
    // is done in InjectTLIMappings
    appendToCompilerUsed(*M, {TLIFunc});
    LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Adding `" << TLIName
                      << "` to `@llvm.compiler.used`.\n");
    ++NumFuncUsedAdded;
  }
  return TLIFunc;
}

/// Replace the call to the vector intrinsic ( \p OldFunc ) with a call to the
/// corresponding function from the vector library ( \p TLIFunc ).
static bool replaceWithTLIFunction(const Module *M, CallInst &CI,
                                   const ElementCount &VecVF, Function *OldFunc,
                                   Function *TLIFunc, FunctionType *VecFTy,
                                   bool IsMasked) {
  IRBuilder<> IRBuilder(&CI);
  SmallVector<Value *> Args(CI.args());
  if (IsMasked) {
    if (Args.size() == VecFTy->getNumParams())
      static_assert(true && "mask was already in place");

    auto *MaskTy = VectorType::get(Type::getInt1Ty(M->getContext()), VecVF);
    Args.push_back(Constant::getAllOnesValue(MaskTy));
  }

  // Preserve the operand bundles.
  SmallVector<OperandBundleDef, 1> OpBundles;
  CI.getOperandBundlesAsDefs(OpBundles);
  CallInst *Replacement = IRBuilder.CreateCall(TLIFunc, Args, OpBundles);
  assert(VecFTy == TLIFunc->getFunctionType() &&
         "Expecting function types to be identical");
  CI.replaceAllUsesWith(Replacement);
  // Preserve fast math flags for FP math.
  if (isa<FPMathOperator>(Replacement))
    Replacement->copyFastMathFlags(&CI);

  LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Replaced call to `"
                    << OldFunc->getName() << "` with call to `"
                    << TLIFunc->getName() << "`.\n");
  ++NumCallsReplaced;
  return true;
}

/// Utility method to get the VecDesc, depending on whether there is a TLI
/// mapping, either with or without a mask.
static std::optional<const VecDesc *> getVecDesc(const TargetLibraryInfo &TLI,
                                                 const StringRef &ScalarName,
                                                 const ElementCount &VF) {
  const VecDesc *VDMasked = TLI.getVectorMappingInfo(ScalarName, VF, true);
  const VecDesc *VDNoMask = TLI.getVectorMappingInfo(ScalarName, VF, false);
  // Invalid when there are both variants (ie masked and unmasked), or none
  if ((VDMasked == nullptr) == (VDNoMask == nullptr))
    return std::nullopt;

  return {VDMasked != nullptr ? VDMasked : VDNoMask};
}

/// Returns whether it is able to replace a call to the intrinsic \p CI with a
/// TLI mapped call.
static bool replaceWithCallToVeclib(const TargetLibraryInfo &TLI,
                                    CallInst &CI) {
  if (!CI.getCalledFunction())
    return false;

  auto IntrinsicID = CI.getCalledFunction()->getIntrinsicID();
  // Replacement is only performed for intrinsic functions
  if (IntrinsicID == Intrinsic::not_intrinsic)
    return false;

  // Convert vector arguments to scalar type and check that all vector operands
  // have identical vector width.
  ElementCount VF = ElementCount::getFixed(0);
  SmallVector<Type *> ScalarTypes;
  for (auto Arg : enumerate(CI.args())) {
    auto *ArgTy = Arg.value()->getType();
    if (isVectorIntrinsicWithScalarOpAtArg(IntrinsicID, Arg.index())) {
      ScalarTypes.push_back(ArgTy);
    } else if (auto *VectorArgTy = dyn_cast<VectorType>(ArgTy)) {
      ScalarTypes.push_back(ArgTy->getScalarType());
      // Disallow vector arguments with different VFs. When processing the first
      // vector argument, store it's VF, and for the rest ensure that they match
      // it.
      if (VF.isZero())
        VF = VectorArgTy->getElementCount();
      else if (VF != VectorArgTy->getElementCount())
        return false;
    } else {
      // enters when it is supposed to be a vector argument but it isn't.
      return false;
    }
  }

  // Try to reconstruct the name for the scalar version of this intrinsic using
  // the intrinsic ID and the argument types converted to scalar above.
  std::string ScalarName =
      (Intrinsic::isOverloaded(IntrinsicID)
           ? Intrinsic::getName(IntrinsicID, ScalarTypes, CI.getModule())
           : Intrinsic::getName(IntrinsicID).str());

  // The TargetLibraryInfo does not contain a vectorized version of the scalar
  // function.
  if (!TLI.isFunctionVectorizable(ScalarName))
    return false;

  auto OptVD = getVecDesc(TLI, ScalarName, VF);
  if (!OptVD)
    return false;

  const VecDesc *VD = *OptVD;
  // Try to find the mapping for the scalar version of this intrinsic and the
  // exact vector width of the call operands in the TargetLibraryInfo.
  StringRef TLIName = TLI.getVectorizedFunction(ScalarName, VF, VD->isMasked());
  LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Looking up TLI mapping for `"
                    << ScalarName << "` and vector width " << VF << ".\n");

  // TLI failed to find a correct mapping.
  if (TLIName.empty())
    return false;

  // Find the vector Function and replace the call to the intrinsic with a call
  // to the vector library function.
  LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Found TLI function `" << TLIName
                    << "`.\n");

  Type *ScalarRetTy = CI.getType()->getScalarType();
  FunctionType *ScalarFTy = FunctionType::get(ScalarRetTy, ScalarTypes, false);
  const std::string MangledName = VD->getVectorFunctionABIVariantString();
  auto OptInfo = VFABI::tryDemangleForVFABI(MangledName, ScalarFTy);
  if (!OptInfo)
    return false;

  // get the vector FunctionType
  Module *M = CI.getModule();
  auto OptFTy = VFABI::createFunctionType(*OptInfo, ScalarFTy);
  if (!OptFTy)
    return false;

  Function *OldFunc = CI.getCalledFunction();
  FunctionType *VectorFTy = *OptFTy;
  Function *TLIFunc = getTLIFunction(M, VectorFTy, OldFunc, TLIName);
  return replaceWithTLIFunction(M, CI, OptInfo->Shape.VF, OldFunc, TLIFunc,
                                VectorFTy, VD->isMasked());
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
  for (auto *CI : ReplacedCalls)
    CI->eraseFromParent();
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
  }

  // The pass did not replace any calls, hence it preserves all analyses.
  return PreservedAnalyses::all();
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
