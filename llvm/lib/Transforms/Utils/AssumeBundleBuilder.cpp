//===- AssumeBundleBuilder.cpp - tools to preserve informations -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/AssumeBundleBuilder.h"
#include "llvm/Analysis/AssumeBundleQueries.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/ADT/DenseSet.h"
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

/// Deterministically compare OperandBundleDef.
/// The ordering is:
/// - by the attribute's name aka operand bundle tag, (doesn't change)
/// - then by the numeric Value of the argument, (doesn't change)
/// - lastly by the Name of the current Value it WasOn. (may change)
/// This order is deterministic and allows looking for the right kind of
/// attribute with binary search. However finding the right WasOn needs to be
/// done via linear search because values can get replaced.
bool isLowerOpBundle(const OperandBundleDef &LHS, const OperandBundleDef &RHS) {
  auto getTuple = [](const OperandBundleDef &Op) {
    return std::make_tuple(
        Op.getTag(),
        Op.input_size() <= ABA_Argument
            ? 0
            : cast<ConstantInt>(*(Op.input_begin() + ABA_Argument))
                  ->getZExtValue(),
        Op.input_size() <= ABA_WasOn
            ? StringRef("")
            : (*(Op.input_begin() + ABA_WasOn))->getName());
  };
  return getTuple(LHS) < getTuple(RHS);
}

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
  SmallDenseMap<MapKey, unsigned, 8> AssumedKnowledgeMap;
  Instruction *InsertBeforeInstruction = nullptr;

  AssumeBuilderState(Module *M) : M(M) {}

  void addKnowledge(RetainedKnowledge RK) {
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
    llvm::sort(OpBundle, isLowerOpBundle);
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

void llvm::salvageKnowledge(Instruction *I, AssumptionCache *AC) {
  if (!EnableKnowledgeRetention)
    return;
  AssumeBuilderState Builder(I->getModule());
  Builder.InsertBeforeInstruction = I;
  Builder.addInstruction(I);
  if (IntrinsicInst *Intr = Builder.build()) {
    Intr->insertBefore(I);
    if (AC)
      AC->registerAssumption(Intr);
  }
}

PreservedAnalyses AssumeBuilderPass::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  for (Instruction &I : instructions(F))
    if (Instruction *Assume = buildAssumeFromInst(&I))
      Assume->insertBefore(&I);
  return PreservedAnalyses::all();
}
