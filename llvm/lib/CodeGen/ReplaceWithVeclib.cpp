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

using namespace llvm;

#define DEBUG_TYPE "replace-with-veclib"

STATISTIC(NumCallsReplaced,
          "Number of calls to intrinsics that have been replaced.");

STATISTIC(NumTLIFuncDeclAdded,
          "Number of vector library function declarations added.");

STATISTIC(NumFuncUsedAdded,
          "Number of functions added to `llvm.compiler.used`");

/// Returns a vector Function that it adds to the Module \p M. When an \p
/// ScalarFunc is not null, it copies its attributes to the newly created
/// Function.
Function *getTLIFunction(Module *M, FunctionType *VectorFTy,
                         const StringRef TLIName,
                         Function *ScalarFunc = nullptr) {
  Function *TLIFunc = M->getFunction(TLIName);
  if (!TLIFunc) {
    TLIFunc =
        Function::Create(VectorFTy, Function::ExternalLinkage, TLIName, *M);
    if (ScalarFunc)
      TLIFunc->copyAttributesFrom(ScalarFunc);

    LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Added vector library function `"
                      << TLIName << "` of type `" << *(TLIFunc->getType())
                      << "` to module.\n");

    ++NumTLIFuncDeclAdded;
    // Add the freshly created function to llvm.compiler.used, similar to as it
    // is done in InjectTLIMappings.
    appendToCompilerUsed(*M, {TLIFunc});
    LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Adding `" << TLIName
                      << "` to `@llvm.compiler.used`.\n");
    ++NumFuncUsedAdded;
  }
  return TLIFunc;
}

/// Replace the call to the vector intrinsic ( \p CalltoReplace ) with a call to
/// the corresponding function from the vector library ( \p TLIVecFunc ).
static void replaceWithTLIFunction(CallInst &CalltoReplace, VFInfo &Info,
                                   Function *TLIVecFunc) {
  IRBuilder<> IRBuilder(&CalltoReplace);
  SmallVector<Value *> Args(CalltoReplace.args());
  if (auto OptMaskpos = Info.getParamIndexForOptionalMask()) {
    auto *MaskTy = VectorType::get(Type::getInt1Ty(CalltoReplace.getContext()),
                                   Info.Shape.VF);
    Args.insert(Args.begin() + OptMaskpos.value(),
                Constant::getAllOnesValue(MaskTy));
  }

  // Preserve the operand bundles.
  SmallVector<OperandBundleDef, 1> OpBundles;
  CalltoReplace.getOperandBundlesAsDefs(OpBundles);
  CallInst *Replacement = IRBuilder.CreateCall(TLIVecFunc, Args, OpBundles);
  CalltoReplace.replaceAllUsesWith(Replacement);
  // Preserve fast math flags for FP math.
  if (isa<FPMathOperator>(Replacement))
    Replacement->copyFastMathFlags(&CalltoReplace);
}

/// Returns true when successfully replaced \p CallToReplace with a suitable
/// function taking vector arguments, based on available mappings in the \p TLI.
/// Currently only works when \p CallToReplace is a call to vectorized
/// intrinsic.
static bool replaceWithCallToVeclib(const TargetLibraryInfo &TLI,
                                    CallInst &CallToReplace) {
  if (!CallToReplace.getCalledFunction())
    return false;

  auto IntrinsicID = CallToReplace.getCalledFunction()->getIntrinsicID();
  // Replacement is only performed for intrinsic functions.
  if (IntrinsicID == Intrinsic::not_intrinsic)
    return false;

  // Compute arguments types of the corresponding scalar call. Additionally
  // checks if in the vector call, all vector operands have the same EC.
  ElementCount VF = ElementCount::getFixed(0);
  SmallVector<Type *> ScalarArgTypes;
  for (auto Arg : enumerate(CallToReplace.args())) {
    auto *ArgTy = Arg.value()->getType();
    if (isVectorIntrinsicWithScalarOpAtArg(IntrinsicID, Arg.index())) {
      ScalarArgTypes.push_back(ArgTy);
    } else if (auto *VectorArgTy = dyn_cast<VectorType>(ArgTy)) {
      ScalarArgTypes.push_back(ArgTy->getScalarType());
      // Disallow vector arguments with different VFs. When processing the first
      // vector argument, store it's VF, and for the rest ensure that they match
      // it.
      if (VF.isZero())
        VF = VectorArgTy->getElementCount();
      else if (VF != VectorArgTy->getElementCount())
        return false;
    } else
      // Exit when it is supposed to be a vector argument but it isn't.
      return false;
  }

  // Try to reconstruct the name for the scalar version of this intrinsic using
  // the intrinsic ID and the argument types converted to scalar above.
  std::string ScalarName =
      (Intrinsic::isOverloaded(IntrinsicID)
           ? Intrinsic::getName(IntrinsicID, ScalarArgTypes,
                                CallToReplace.getModule())
           : Intrinsic::getName(IntrinsicID).str());

  // Try to find the mapping for the scalar version of this intrinsic and the
  // exact vector width of the call operands in the TargetLibraryInfo. First,
  // check with a non-masked variant, and if that fails try with a masked one.
  const VecDesc *VD =
      TLI.getVectorMappingInfo(ScalarName, VF, /*Masked*/ false);
  if (!VD && !(VD = TLI.getVectorMappingInfo(ScalarName, VF, /*Masked*/ true)))
    return false;

  LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Found TLI mapping from: `" << ScalarName
                    << "` and vector width " << VF << " to: `"
                    << VD->getVectorFnName() << "`.\n");

  // Replace the call to the intrinsic with a call to the vector library
  // function.
  Type *ScalarRetTy = CallToReplace.getType()->getScalarType();
  FunctionType *ScalarFTy =
      FunctionType::get(ScalarRetTy, ScalarArgTypes, /*isVarArg*/ false);
  const std::string MangledName = VD->getVectorFunctionABIVariantString();
  auto OptInfo = VFABI::tryDemangleForVFABI(MangledName, ScalarFTy);
  if (!OptInfo)
    return false;

  FunctionType *VectorFTy = VFABI::createFunctionType(*OptInfo, ScalarFTy);
  if (!VectorFTy)
    return false;

  Function *FuncToReplace = CallToReplace.getCalledFunction();
  Function *TLIFunc = getTLIFunction(CallToReplace.getModule(), VectorFTy,
                                     VD->getVectorFnName(), FuncToReplace);
  replaceWithTLIFunction(CallToReplace, *OptInfo, TLIFunc);

  LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Replaced call to `"
                    << FuncToReplace->getName() << "` with call to `"
                    << TLIFunc->getName() << "`.\n");
  ++NumCallsReplaced;
  return true;
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
