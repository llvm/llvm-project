//=== ReplaceWithVeclib.cpp - Replace vector intrinsics with veclib calls -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Replaces LLVM IR instructions with vector operands (i.e., the frem
// instruction or calls to LLVM intrinsics) with matching calls to functions
// from a vector library (e.g libmvec, SVML) using TargetLibraryInfo interface.
// This happens only when the cost of calling the vector library is not found to
// be more than the cost of the original instruction.
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
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/VFABIDemangler.h"
#include "llvm/Support/InstructionCost.h"
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

/// Replace the instruction \p I with a call to the corresponding function from
/// the vector library (\p TLIVecFunc).
static void replaceWithTLIFunction(Instruction &I, VFInfo &Info,
                                   Function *TLIVecFunc) {
  IRBuilder<> IRBuilder(&I);
  auto *CI = dyn_cast<CallInst>(&I);
  SmallVector<Value *> Args(CI ? CI->args() : I.operands());
  if (auto OptMaskpos = Info.getParamIndexForOptionalMask()) {
    auto *MaskTy =
        VectorType::get(Type::getInt1Ty(I.getContext()), Info.Shape.VF);
    Args.insert(Args.begin() + OptMaskpos.value(),
                Constant::getAllOnesValue(MaskTy));
  }

  // If it is a call instruction, preserve the operand bundles.
  SmallVector<OperandBundleDef, 1> OpBundles;
  if (CI)
    CI->getOperandBundlesAsDefs(OpBundles);

  auto *Replacement = IRBuilder.CreateCall(TLIVecFunc, Args, OpBundles);
  I.replaceAllUsesWith(Replacement);
  // Preserve fast math flags for FP math.
  if (isa<FPMathOperator>(Replacement))
    Replacement->copyFastMathFlags(&I);
}

/// Returns whether the vector library call \p TLIFunc costs more than the
/// original instruction \p I.
static bool isVeclibCallSlower(const TargetLibraryInfo &TLI,
                               const TargetTransformInfo &TTI, Instruction &I,
                               VectorType *VectorTy, CallInst *CI,
                               Function *TLIFunc) {
  SmallVector<Type *, 4> OpTypes;
  for (auto &Op : CI ? CI->args() : I.operands())
    OpTypes.push_back(Op->getType());

  TTI::TargetCostKind CostKind = TTI::TCK_RecipThroughput;
  InstructionCost DefaultCost;
  if (CI) {
    FastMathFlags FMF;
    if (auto *FPMO = dyn_cast<FPMathOperator>(CI))
      FMF = FPMO->getFastMathFlags();

    SmallVector<const Value *> Args(CI->args());
    IntrinsicCostAttributes CostAttrs(CI->getIntrinsicID(), VectorTy, Args,
                                      OpTypes, FMF,
                                      dyn_cast<IntrinsicInst>(CI));
    DefaultCost = TTI.getIntrinsicInstrCost(CostAttrs, CostKind);
  } else {
    assert((I.getOpcode() == Instruction::FRem) && "Only FRem is supported");
    auto Op2Info = TTI.getOperandInfo(I.getOperand(1));
    SmallVector<const Value *, 4> OpValues(I.operand_values());
    DefaultCost = TTI.getArithmeticInstrCost(
        I.getOpcode(), VectorTy, CostKind,
        {TargetTransformInfo::OK_AnyValue, TargetTransformInfo::OP_None},
        Op2Info, OpValues, &I);
  }

  InstructionCost VecLibCost =
      TTI.getCallInstrCost(TLIFunc, VectorTy, OpTypes, CostKind);
  return VecLibCost > DefaultCost;
}

/// Returns true when successfully replaced \p I with a suitable function taking
/// vector arguments, based on available mappings in the \p TLI and costs.
/// Currently only works when \p I is a call to vectorized intrinsic or the frem
/// instruction.
static bool replaceWithCallToVeclib(const TargetLibraryInfo &TLI,
                                    const TargetTransformInfo &TTI,
                                    Instruction &I) {
  // At the moment VFABI assumes the return type is always widened unless it is
  // a void type.
  auto *VectorTy = dyn_cast<VectorType>(I.getType());
  ElementCount EC(VectorTy ? VectorTy->getElementCount()
                           : ElementCount::getFixed(0));

  // Compute the argument types of the corresponding scalar call and the scalar
  // function name. For calls, it additionally finds the function to replace
  // and checks that all vector operands match the previously found EC.
  SmallVector<Type *, 8> ScalarArgTypes;
  std::string ScalarName;
  Function *FuncToReplace = nullptr;
  auto *CI = dyn_cast<CallInst>(&I);
  if (CI) {
    FuncToReplace = CI->getCalledFunction();
    Intrinsic::ID IID = FuncToReplace->getIntrinsicID();
    assert(IID != Intrinsic::not_intrinsic && "Not an intrinsic");
    for (auto Arg : enumerate(CI->args())) {
      auto *ArgTy = Arg.value()->getType();
      if (isVectorIntrinsicWithScalarOpAtArg(IID, Arg.index())) {
        ScalarArgTypes.push_back(ArgTy);
      } else if (auto *VectorArgTy = dyn_cast<VectorType>(ArgTy)) {
        ScalarArgTypes.push_back(VectorArgTy->getElementType());
        // When return type is void, set EC to the first vector argument, and
        // disallow vector arguments with different ECs.
        if (EC.isZero()) {
          EC = VectorArgTy->getElementCount();
          VectorTy = VectorArgTy;
        } else if (EC != VectorArgTy->getElementCount())
          return false;
      } else
        // Exit when it is supposed to be a vector argument but it isn't.
        return false;
    }
    // Try to reconstruct the name for the scalar version of the instruction,
    // using scalar argument types.
    ScalarName = Intrinsic::isOverloaded(IID)
                     ? Intrinsic::getName(IID, ScalarArgTypes, I.getModule())
                     : Intrinsic::getName(IID).str();
  } else {
    assert(VectorTy && "Return type must be a vector");
    auto *ScalarTy = VectorTy->getScalarType();
    LibFunc Func;
    if (!TLI.getLibFunc(I.getOpcode(), ScalarTy, Func))
      return false;
    ScalarName = TLI.getName(Func);
    ScalarArgTypes = {ScalarTy, ScalarTy};
  }

  // Try to find the mapping for the scalar version of this intrinsic and the
  // exact vector width of the call operands in the TargetLibraryInfo. First,
  // check with a non-masked variant, and if that fails try with a masked one.
  const VecDesc *VD =
      TLI.getVectorMappingInfo(ScalarName, EC, /*Masked*/ false);
  if (!VD && !(VD = TLI.getVectorMappingInfo(ScalarName, EC, /*Masked*/ true)))
    return false;

  LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Found TLI mapping from: `" << ScalarName
                    << "` and vector width " << EC << " to: `"
                    << VD->getVectorFnName() << "`.\n");

  // Replace the call to the intrinsic with a call to the vector library
  // function.
  Type *ScalarRetTy = I.getType()->getScalarType();
  FunctionType *ScalarFTy =
      FunctionType::get(ScalarRetTy, ScalarArgTypes, /*isVarArg*/ false);
  const std::string MangledName = VD->getVectorFunctionABIVariantString();
  auto OptInfo = VFABI::tryDemangleForVFABI(MangledName, ScalarFTy);
  if (!OptInfo)
    return false;

  // There is no guarantee that the vectorized instructions followed the VFABI
  // specification when being created, this is why we need to add extra check to
  // make sure that the operands of the vector function obtained via VFABI match
  // the operands of the original vector instruction.
  if (CI) {
    for (auto &VFParam : OptInfo->Shape.Parameters) {
      if (VFParam.ParamKind == VFParamKind::GlobalPredicate)
        continue;

      // tryDemangleForVFABI must return valid ParamPos, otherwise it could be
      // a bug in the VFABI parser.
      assert(VFParam.ParamPos < CI->arg_size() &&
             "ParamPos has invalid range.");
      Type *OrigTy = CI->getArgOperand(VFParam.ParamPos)->getType();
      if (OrigTy->isVectorTy() != (VFParam.ParamKind == VFParamKind::Vector)) {
        LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Will not replace: " << ScalarName
                          << ". Wrong type at index " << VFParam.ParamPos
                          << ": " << *OrigTy << "\n");
        return false;
      }
    }
  }

  FunctionType *VectorFTy = VFABI::createFunctionType(*OptInfo, ScalarFTy);
  if (!VectorFTy)
    return false;

  Function *TLIFunc = getTLIFunction(I.getModule(), VectorFTy,
                                     VD->getVectorFnName(), FuncToReplace);

  if (isVeclibCallSlower(TLI, TTI, I, VectorTy, CI, TLIFunc))
    return false;

  replaceWithTLIFunction(I, *OptInfo, TLIFunc);
  LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Replaced call to `" << ScalarName
                    << "` with call to `" << TLIFunc->getName() << "`.\n");
  ++NumCallsReplaced;
  return true;
}

/// Supported instruction \p I must be a vectorized frem or a call to an
/// intrinsic that returns either void or a vector.
static bool isSupportedInstruction(Instruction *I) {
  Type *Ty = I->getType();
  if (auto *CI = dyn_cast<CallInst>(I))
    return (Ty->isVectorTy() || Ty->isVoidTy()) && CI->getCalledFunction() &&
           CI->getCalledFunction()->getIntrinsicID() !=
               Intrinsic::not_intrinsic;
  if (I->getOpcode() == Instruction::FRem && Ty->isVectorTy())
    return true;
  return false;
}

static bool runImpl(const TargetLibraryInfo &TLI,
                    const TargetTransformInfo &TTI, Function &F) {
  bool Changed = false;
  SmallVector<Instruction *> ReplacedCalls;
  for (auto &I : instructions(F)) {
    if (!isSupportedInstruction(&I))
      continue;
    if (replaceWithCallToVeclib(TLI, TTI, I)) {
      ReplacedCalls.push_back(&I);
      Changed = true;
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
  const TargetTransformInfo &TTI = AM.getResult<TargetIRAnalysis>(F);
  auto Changed = runImpl(TLI, TTI, F);
  if (Changed) {
    LLVM_DEBUG(dbgs() << "Instructions replaced with vector libraries: "
                      << NumCallsReplaced << "\n");

    PreservedAnalyses PA;
    PA.preserveSet<CFGAnalyses>();
    PA.preserve<TargetLibraryAnalysis>();
    PA.preserve<TargetIRAnalysis>();
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
  const TargetTransformInfo &TTI =
      getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  return runImpl(TLI, TTI, F);
}

void ReplaceWithVeclibLegacy::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<TargetLibraryInfoWrapperPass>();
  AU.addRequired<TargetTransformInfoWrapperPass>();
  AU.addPreserved<TargetLibraryInfoWrapperPass>();
  AU.addPreserved<TargetTransformInfoWrapperPass>();
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
