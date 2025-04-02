//===- AMDGPUExpandPseudoIntrinsics.cpp - Pseudo Intrinsic Expander Pass --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file implements a pass that deals with expanding AMDGCN generic pseudo-
// intrinsics into target specific quantities / sequences. In this context, a
// pseudo-intrinsic is an AMDGCN intrinsic that does not directly map to a
// specific instruction, but rather is intended as a mechanism for abstractly
// conveying target specific info to a HLL / the FE, without concretely
// impacting the AST. An example of such an intrinsic is amdgcn.wavefrontsize.
// This pass should run as early as possible / immediately after Clang CodeGen,
// so that the optimisation pipeline and the BE operate with concrete target
// data.
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUTargetMachine.h"
#include "GCNSubtarget.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetIntrinsicInfo.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"

#include <string>
#include <utility>

using namespace llvm;

namespace {
inline Function *getCloneForInlining(Function *OldF) {
  assert(OldF && "Must pass an existing Function!");

  // TODO - Alias Value to clone arg.
  ValueToValueMapTy VMap;

  auto NewF = CloneFunction(OldF, VMap);

  NewF->removeFnAttr(Attribute::OptimizeNone);
  NewF->removeFnAttr(Attribute::NoInline);
  NewF->addFnAttr(Attribute::AlwaysInline);

  return NewF;
}

template <typename C>
inline void collectUsers(Value *V, ModulePassManager &AlwaysInliner,
                         ModuleAnalysisManager &MAM,
                         SmallDenseMap<Function *, Function *> &InlinableClones,
                         C &Container) {
  assert(V && "Must pass an existing Value!");

  auto A = PreservedAnalyses::all();

  constexpr auto IsValidCall = [](auto &&U) {
    if (auto CB = dyn_cast<CallBase>(U))
      if (auto F = CB->getCalledFunction())
        if (!F->isIntrinsic() && !F->isDeclaration())
          return true;
    return false;
  };

  SmallVector<User *> Calls{};
  copy_if(V->users(), std::back_inserter(Calls), IsValidCall);

  while (!Calls.empty()) {
    for (auto &&Call : Calls) {
      auto CB = cast<CallBase>(Call);
      auto &TempF = InlinableClones[CB->getCalledFunction()];

      if (!TempF)
        TempF = getCloneForInlining(CB->getCalledFunction());

      CB->setCalledFunction(TempF);
      CB->removeFnAttr(Attribute::NoInline);
      CB->addFnAttr(Attribute::AlwaysInline);

      AlwaysInliner.run(*TempF->getParent(), MAM);
    }

    Calls.clear();

    copy_if(V->users(), std::back_inserter(Calls), IsValidCall);
  }

  for (auto &&U : V->users())
    if (auto I = dyn_cast<Instruction>(U)) {
      if (auto CB = dyn_cast<CallBase>(I)) {
        if (CB->getCalledFunction() && !CB->getCalledFunction()->isIntrinsic())
          Container.insert(Container.end(), I);
      } else {
        Container.insert(Container.end(), I);
      }
    }
}

std::pair<PreservedAnalyses, bool>
handlePredicate(const GCNSubtarget &ST, ModuleAnalysisManager &MAM,
                SmallDenseMap<Function *, Function *>& InlinableClones,
                GlobalVariable *P) {
  auto PV = P->getName().substr(P->getName().rfind('.') + 1).str();
  auto Dx = PV.find(',');
  while (Dx != std::string::npos) {
    PV.insert(++Dx, {'+'});

    Dx = PV.find(',', Dx);
  }

  auto PTy = P->getValueType();
  P->setLinkage(GlobalValue::PrivateLinkage);
  P->setExternallyInitialized(false);

  if (P->getName().starts_with("llvm.amdgcn.is"))
    P->setInitializer(ConstantInt::getBool(PTy, PV == ST.getCPU()));
  else
    P->setInitializer(ConstantInt::getBool(PTy, ST.checkFeatures('+' + PV)));

  ModulePassManager MPM;
  MPM.addPass(AlwaysInlinerPass());

  SmallPtrSet<Instruction *, 32> ToFold;
  collectUsers(P, MPM, MAM, InlinableClones, ToFold);

  if (ToFold.empty())
    return {PreservedAnalyses::all(), true};

  do {
    auto I = *ToFold.begin();
    ToFold.erase(I);

    if (auto C = ConstantFoldInstruction(I, P->getDataLayout())) {
      collectUsers(I, MPM, MAM, InlinableClones, ToFold);
      I->replaceAllUsesWith(C);
      I->eraseFromParent();
      continue;
    } else if (I->isTerminator() && ConstantFoldTerminator(I->getParent())) {
      continue;
    } else if (I->users().empty()) {
      continue;
    }

    std::string W;
    raw_string_ostream OS(W);

    auto Caller = I->getParent()->getParent();

    OS << "Impossible to constant fold feature predicate: " << P->getName()
       << ", please simplify.\n";

    Caller->getContext().diagnose(
        DiagnosticInfoUnsupported(*Caller, W, I->getDebugLoc(), DS_Error));

    return {PreservedAnalyses::none(), false};
  } while (!ToFold.empty());

  return {PreservedAnalyses::none(), true};
}
} // Unnamed namespace.

PreservedAnalyses
AMDGPUExpandPseudoIntrinsicsPass::run(Module &M, ModuleAnalysisManager &MAM) {
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

  PreservedAnalyses Ret = PreservedAnalyses::all();

  SmallDenseMap<Function *, Function *> InlinableClones;
  const auto &ST = TM.getSubtarget<GCNSubtarget>(
      *find_if(M, [](auto &&F) { return !F.isIntrinsic(); }));

  for (auto &&P : Predicates) {
    auto R = handlePredicate(ST, MAM, InlinableClones, P);

    if (!R.second)
      return PreservedAnalyses::none();

    Ret.intersect(R.first);
  }

  for (auto &&C : InlinableClones)
    C.second->eraseFromParent();

  return Ret;
}
