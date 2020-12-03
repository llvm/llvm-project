//===- InjectTLIMAppings.cpp - TLI to VFABI attribute injection  ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Populates the VFABI attribute with the scalar-to-vector mappings
// from the TargetLibraryInfo.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/InjectTLIMappings.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/DemandedBits.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "inject-tli-mappings"

STATISTIC(NumCallInjected,
          "Number of calls in which the mappings have been injected.");

STATISTIC(NumVFDeclAdded,
          "Number of function declarations that have been added.");
STATISTIC(NumCompUsedAdded,
          "Number of `@llvm.compiler.used` operands that have been added.");

/// A helper function that adds the vector function declaration that
/// vectorizes the CallInst CI with a vectorization factor of VF
/// lanes. The TLI assumes that all parameters and the return type of
/// CI (other than void) need to be widened to a VectorType of VF
/// lanes.
static void addVariantDeclaration(CallInst &CI, const unsigned VF,
                                  const StringRef VFName) {
  Module *M = CI.getModule();

  // Add function declaration.
  Type *RetTy = ToVectorTy(CI.getType(), VF);
  SmallVector<Type *, 4> Tys;
  for (Value *ArgOperand : CI.arg_operands())
    Tys.push_back(ToVectorTy(ArgOperand->getType(), VF));
  assert(!CI.getFunctionType()->isVarArg() &&
         "VarArg functions are not supported.");
  FunctionType *FTy = FunctionType::get(RetTy, Tys, /*isVarArg=*/false);
  Function *VectorF =
      Function::Create(FTy, Function::ExternalLinkage, VFName, M);
  VectorF->copyAttributesFrom(CI.getCalledFunction());
  ++NumVFDeclAdded;
  LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Added to the module: `" << VFName
                    << "` of type " << *(VectorF->getType()) << "\n");

  // Make function declaration (without a body) "sticky" in the IR by
  // listing it in the @llvm.compiler.used intrinsic.
  assert(!VectorF->size() && "VFABI attribute requires `@llvm.compiler.used` "
                             "only on declarations.");
  appendToCompilerUsed(*M, {VectorF});
  LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Adding `" << VFName
                    << "` to `@llvm.compiler.used`.\n");
  ++NumCompUsedAdded;
}

static void addMappingsFromTLI(const TargetLibraryInfo &TLI, CallInst &CI) {
  // This is needed to make sure we don't query the TLI for calls to
  // bitcast of function pointers, like `%call = call i32 (i32*, ...)
  // bitcast (i32 (...)* @goo to i32 (i32*, ...)*)(i32* nonnull %i)`,
  // as such calls make the `isFunctionVectorizable` raise an
  // exception.
  if (CI.isNoBuiltin() || !CI.getCalledFunction())
    return;

  StringRef ScalarName = CI.getCalledFunction()->getName();

  // Nothing to be done if the TLI thinks the function is not
  // vectorizable.
  if (!TLI.isFunctionVectorizable(ScalarName))
    return;
  SmallVector<std::string, 8> Mappings;
  VFABI::getVectorVariantNames(CI, Mappings);
  Module *M = CI.getModule();
  const SetVector<StringRef> OriginalSetOfMappings(Mappings.begin(),
                                                   Mappings.end());
  //  All VFs in the TLI are powers of 2.
  for (unsigned VF = 2, WidestVF = TLI.getWidestVF(ScalarName); VF <= WidestVF;
       VF *= 2) {
    const std::string TLIName =
        std::string(TLI.getVectorizedFunction(ScalarName, VF));
    if (!TLIName.empty()) {
      std::string MangledName = VFABI::mangleTLIVectorName(
          TLIName, ScalarName, CI.getNumArgOperands(), VF);
      if (!OriginalSetOfMappings.count(MangledName)) {
        Mappings.push_back(MangledName);
        ++NumCallInjected;
      }
      Function *VariantF = M->getFunction(TLIName);
      if (!VariantF)
        addVariantDeclaration(CI, VF, TLIName);
    }
  }

  VFABI::setVectorVariantNames(&CI, Mappings);
}

static bool runImpl(const TargetLibraryInfo &TLI, Function &F) {
  for (auto &I : instructions(F))
    if (auto CI = dyn_cast<CallInst>(&I))
      addMappingsFromTLI(TLI, *CI);
  // Even if the pass adds IR attributes, the analyses are preserved.
  return false;
}

////////////////////////////////////////////////////////////////////////////////
// New pass manager implementation.
////////////////////////////////////////////////////////////////////////////////
PreservedAnalyses InjectTLIMappings::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  const TargetLibraryInfo &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  runImpl(TLI, F);
  // Even if the pass adds IR attributes, the analyses are preserved.
  return PreservedAnalyses::all();
}

////////////////////////////////////////////////////////////////////////////////
// Legacy PM Implementation.
////////////////////////////////////////////////////////////////////////////////
bool InjectTLIMappingsLegacy::runOnFunction(Function &F) {
  const TargetLibraryInfo &TLI =
      getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
  return runImpl(TLI, F);
}

void InjectTLIMappingsLegacy::getAnalysisUsage(AnalysisUsage &AU) const {
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
char InjectTLIMappingsLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(InjectTLIMappingsLegacy, DEBUG_TYPE,
                      "Inject TLI Mappings", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(InjectTLIMappingsLegacy, DEBUG_TYPE, "Inject TLI Mappings",
                    false, false)

FunctionPass *llvm::createInjectTLIMappingsLegacyPass() {
  return new InjectTLIMappingsLegacy();
}
