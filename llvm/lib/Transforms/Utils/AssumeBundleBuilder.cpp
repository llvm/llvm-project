//===- AssumeBundleBuilder.cpp - tools to preserve informations -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/AssumeBundleBuilder.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Analysis/AssumeBundleQueries.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

cl::opt<bool> ShouldPreserveAllAttributes(
    "assume-preserve-all", cl::init(false), cl::Hidden,
    cl::desc("enable preservation of all attrbitues. even those that are "
             "unlikely to be usefull"));

cl::opt<bool> EnableKnowledgeRetention(
    "enable-knowledge-retention", cl::init(false), cl::Hidden,
    cl::desc(
        "enable preservation of attributes throughout code transformation"));

namespace {

bool isUsefullToPreserve(Attribute::AttrKind Kind) {
  switch (Kind) {
    case Attribute::NonNull:
    case Attribute::Alignment:
    case Attribute::Dereferenceable:
    case Attribute::DereferenceableOrNull:
    case Attribute::Cold:
      return true;
    default:
      return false;
  }
}

/// This class contain all knowledge that have been gather while building an
/// llvm.assume and the function to manipulate it.
struct AssumeBuilderState {
  Module *M;

  using MapKey = std::pair<Value *, Attribute::AttrKind>;
  SmallMapVector<MapKey, unsigned, 8> AssumedKnowledgeMap;
  Instruction *InsertBeforeInstruction = nullptr;
  AssumptionCache* AC = nullptr;
  DominatorTree* DT = nullptr;

  AssumeBuilderState(Module *M, Instruction *I = nullptr,
                     AssumptionCache *AC = nullptr, DominatorTree *DT = nullptr)
      : M(M), InsertBeforeInstruction(I), AC(AC), DT(DT) {}

  bool tryToPreserveWithoutAddingAssume(RetainedKnowledge RK) {
    if (!InsertBeforeInstruction || !AC || !RK.WasOn)
      return false;
    bool HasBeenPreserved = false;
    Use* ToUpdate = nullptr;
    getKnowledgeForValue(
        RK.WasOn, {RK.AttrKind}, AC,
        [&](RetainedKnowledge RKOther, Instruction *Assume,
            const CallInst::BundleOpInfo *Bundle) {
          if (!isValidAssumeForContext(Assume, InsertBeforeInstruction, DT))
            return false;
          if (RKOther.ArgValue >= RK.ArgValue) {
            HasBeenPreserved = true;
            return true;
          } else if (isValidAssumeForContext(InsertBeforeInstruction, Assume,
                                             DT)) {
            HasBeenPreserved = true;
            IntrinsicInst *Intr = cast<IntrinsicInst>(Assume);
            ToUpdate = &Intr->op_begin()[Bundle->Begin + ABA_Argument];
            return true;
          }
          return false;
        });
    if (ToUpdate)
      ToUpdate->set(
          ConstantInt::get(Type::getInt64Ty(M->getContext()), RK.ArgValue));
    return HasBeenPreserved;
  }

  void addKnowledge(RetainedKnowledge RK) {
    if (tryToPreserveWithoutAddingAssume(RK))
      return;
    MapKey Key{RK.WasOn, RK.AttrKind};
    auto Lookup = AssumedKnowledgeMap.find(Key);
    if (Lookup == AssumedKnowledgeMap.end()) {
      AssumedKnowledgeMap[Key] = RK.ArgValue;
      return;
    }
    assert(((Lookup->second == 0 && RK.ArgValue == 0) ||
            (Lookup->second != 0 && RK.ArgValue != 0)) &&
           "inconsistent argument value");

    /// This is only desirable because for all attributes taking an argument
    /// higher is better.
    Lookup->second = std::max(Lookup->second, RK.ArgValue);
  }

  void addAttribute(Attribute Attr, Value *WasOn) {
    if (Attr.isTypeAttribute() || Attr.isStringAttribute() ||
        (!ShouldPreserveAllAttributes &&
         !isUsefullToPreserve(Attr.getKindAsEnum())))
      return;
    unsigned AttrArg = 0;
    if (Attr.isIntAttribute())
      AttrArg = Attr.getValueAsInt();
    addKnowledge({Attr.getKindAsEnum(), AttrArg, WasOn});
  }

  void addCall(const CallBase *Call) {
    auto addAttrList = [&](AttributeList AttrList) {
      for (unsigned Idx = AttributeList::FirstArgIndex;
           Idx < AttrList.getNumAttrSets(); Idx++)
        for (Attribute Attr : AttrList.getAttributes(Idx))
          addAttribute(Attr, Call->getArgOperand(Idx - 1));
      for (Attribute Attr : AttrList.getFnAttributes())
        addAttribute(Attr, nullptr);
    };
    addAttrList(Call->getAttributes());
    if (Function *Fn = Call->getCalledFunction())
      addAttrList(Fn->getAttributes());
  }

  IntrinsicInst *build() {
    if (AssumedKnowledgeMap.empty())
      return nullptr;
    Function *FnAssume = Intrinsic::getDeclaration(M, Intrinsic::assume);
    LLVMContext &C = M->getContext();
    SmallVector<OperandBundleDef, 8> OpBundle;
    for (auto &MapElem : AssumedKnowledgeMap) {
      SmallVector<Value *, 2> Args;
      if (MapElem.first.first)
        Args.push_back(MapElem.first.first);

      /// This is only valid because for all attribute that currently exist a
      /// value of 0 is useless. and should not be preserved.
      if (MapElem.second)
        Args.push_back(ConstantInt::get(Type::getInt64Ty(M->getContext()),
                                        MapElem.second));
      OpBundle.push_back(OperandBundleDefT<Value *>(
          std::string(Attribute::getNameFromAttrKind(MapElem.first.second)),
          Args));
    }
    return cast<IntrinsicInst>(CallInst::Create(
        FnAssume, ArrayRef<Value *>({ConstantInt::getTrue(C)}), OpBundle));
  }

  void addAccessedPtr(Instruction *MemInst, Value *Pointer, Type *AccType,
                      MaybeAlign MA) {
    unsigned DerefSize = MemInst->getModule()
                             ->getDataLayout()
                             .getTypeStoreSize(AccType)
                             .getKnownMinSize();
    if (DerefSize != 0) {
      addKnowledge({Attribute::Dereferenceable, DerefSize, Pointer});
      if (!NullPointerIsDefined(MemInst->getFunction(),
                                Pointer->getType()->getPointerAddressSpace()))
        addKnowledge({Attribute::NonNull, 0u, Pointer});
    }
    if (MA.valueOrOne() > 1)
      addKnowledge(
          {Attribute::Alignment, unsigned(MA.valueOrOne().value()), Pointer});
  }

  void addInstruction(Instruction *I) {
    if (auto *Call = dyn_cast<CallBase>(I))
      return addCall(Call);
    if (auto *Load = dyn_cast<LoadInst>(I))
      return addAccessedPtr(I, Load->getPointerOperand(), Load->getType(),
                            Load->getAlign());
    if (auto *Store = dyn_cast<StoreInst>(I))
      return addAccessedPtr(I, Store->getPointerOperand(),
                            Store->getValueOperand()->getType(),
                            Store->getAlign());
    // TODO: Add support for the other Instructions.
    // TODO: Maybe we should look around and merge with other llvm.assume.
  }
};

} // namespace

IntrinsicInst *llvm::buildAssumeFromInst(Instruction *I) {
  if (!EnableKnowledgeRetention)
    return nullptr;
  AssumeBuilderState Builder(I->getModule());
  Builder.addInstruction(I);
  return Builder.build();
}

void llvm::salvageKnowledge(Instruction *I, AssumptionCache *AC, DominatorTree* DT) {
  if (!EnableKnowledgeRetention)
    return;
  AssumeBuilderState Builder(I->getModule(), I, AC, DT);
  Builder.addInstruction(I);
  if (IntrinsicInst *Intr = Builder.build()) {
    Intr->insertBefore(I);
    if (AC)
      AC->registerAssumption(Intr);
  }
}

PreservedAnalyses AssumeBuilderPass::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  AssumptionCache* AC = AM.getCachedResult<AssumptionAnalysis>(F);
  DominatorTree* DT = AM.getCachedResult<DominatorTreeAnalysis>(F);
  for (Instruction &I : instructions(F))
    salvageKnowledge(&I, AC, DT);
  return PreservedAnalyses::all();
}
