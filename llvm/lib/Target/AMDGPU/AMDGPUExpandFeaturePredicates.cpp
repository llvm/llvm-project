//===- AMDGPUExpandFeaturePredicates.cpp - Feature Predicate Expander Pass ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file implements a pass that deals with expanding AMDGCN generic feature
// predicates into target specific quantities / sequences. In this context, a
// generic feature predicate is an implementation detail global variable that
// is inserted by the FE as a consequence of using either the __builtin_cpu_is
// or the __builtin_amdgcn_is_invocable special builtins on an abstract target
// (AMDGCNSPIRV). These placeholder globals are used to guide target specific
// lowering, once the concrete target is known, by way of constant folding their
// value all the way into a terminator (i.e. a controlled block) or into a no
// live use scenario. We hard fail if the folding fails, to avoid obtuse BE
// errors or opaque run time errors. This pass should run as early as possible /
// immediately after Clang CodeGen, so that the optimisation pipeline and the BE
// operate with concrete target data.
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUTargetMachine.h"
#include "GCNSubtarget.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Passes/CodeGenPassBuilder.h"
#include "llvm/Transforms/Utils/Local.h"

#include <string>
#include <utility>

using namespace llvm;

namespace {
template <typename C> void collectUsers(Value *V, C &Container) {
  assert(V && "Must pass an existing Value!");

  for (auto &&U : V->users())
    if (auto *I = dyn_cast<Instruction>(U))
      Container.insert(Container.end(), I);
}

inline void setPredicate(const GCNSubtarget &ST, GlobalVariable *P) {
  const bool IsFeature = P->getName().starts_with("llvm.amdgcn.has");
  const size_t Offset =
      IsFeature ? sizeof("llvm.amdgcn.has") : sizeof("llvm.amdgcn.is");

  std::string PV = P->getName().substr(Offset).str();
  if (IsFeature) {
    size_t Dx = PV.find(',');
    while (Dx != std::string::npos) {
      PV.insert(++Dx, {'+'});

      Dx = PV.find(',', Dx);
    }
    PV.insert(PV.cbegin(), '+');
  }

  Type *PTy = P->getValueType();
  P->setLinkage(GlobalValue::PrivateLinkage);
  P->setExternallyInitialized(false);

  if (IsFeature)
    P->setInitializer(ConstantInt::getBool(PTy, ST.checkFeatures(PV)));
  else
    P->setInitializer(ConstantInt::getBool(PTy, PV == ST.getCPU()));
}

std::pair<PreservedAnalyses, bool>
unfoldableFound(Function *Caller, GlobalVariable *P, Instruction *NoFold) {
  std::string W;
  raw_string_ostream OS(W);

  OS << "Impossible to constant fold feature predicate: " << *P << " used by "
     << *NoFold << ", please simplify.\n";

  Caller->getContext().diagnose(
      DiagnosticInfoUnsupported(*Caller, W, NoFold->getDebugLoc(), DS_Error));

  return {PreservedAnalyses::none(), false};
}

std::pair<PreservedAnalyses, bool>
handlePredicate(const GCNSubtarget &ST, FunctionAnalysisManager &FAM,
                SmallPtrSet<Function *, 32> &Predicated, GlobalVariable *P) {
  setPredicate(ST, P);

  SmallPtrSet<Instruction *, 32> ToFold;
  collectUsers(P, ToFold);

  if (ToFold.empty())
    return {PreservedAnalyses::all(), true};

  do {
    Instruction *I = *ToFold.begin();
    ToFold.erase(I);

    I->dropDroppableUses();

    Function *F = I->getParent()->getParent();
    auto &DT = FAM.getResult<DominatorTreeAnalysis>(*F);
    DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Lazy);

    if (auto *C = ConstantFoldInstruction(I, P->getDataLayout())) {
      collectUsers(I, ToFold);
      I->replaceAllUsesWith(C);
      I->eraseFromParent();
      continue;
    } else if (I->isTerminator() &&
               ConstantFoldTerminator(I->getParent(), true, nullptr, &DTU)) {
        Predicated.insert(F);

        continue;
    }

    return unfoldableFound(I->getParent()->getParent(), P, I);
  } while (!ToFold.empty());

  return {PreservedAnalyses::none(), true};
}
} // Unnamed namespace.

PreservedAnalyses
AMDGPUExpandFeaturePredicatesPass::run(Module &M, ModuleAnalysisManager &MAM) {
  if (M.empty())
    return PreservedAnalyses::all();

  SmallVector<GlobalVariable *> Predicates;
  for (auto &&G : M.globals()) {
    if (!G.isDeclaration() || !G.hasName())
      continue;
    if (G.getName().starts_with("llvm.amdgcn."))
      Predicates.push_back(&G);
  }

  if (Predicates.empty())
    return PreservedAnalyses::all();

  const auto &ST = TM.getSubtarget<GCNSubtarget>(
      *find_if(M, [](auto &&F) { return !F.isIntrinsic(); }));

  auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  SmallPtrSet<Function *, 32> Predicated;
  auto Ret = PreservedAnalyses::all();
  for (auto &&P : Predicates) {
    auto R = handlePredicate(ST, FAM, Predicated, P);

    if (!R.second)
      break;

    Ret.intersect(R.first);
  }

  for (auto &&P : Predicates)
    P->eraseFromParent();
  for (auto &&F : Predicated)
    removeUnreachableBlocks(*F);

  return Ret;
}
