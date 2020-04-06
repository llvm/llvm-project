//===- AssumeBundleBuilder.cpp - tools to preserve informations -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/AssumeBundleBuilder.h"
#include "llvm/Analysis/AssumeBundleQueries.h"
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

struct AssumedKnowledge {
  const char *Name;
  Value *Argument;
  enum {
    None,
    Empty,
    Tombstone,
  };
  /// Contain the argument and a flag if needed.
  llvm::PointerIntPair<Value *, 2> WasOn;
};

} // namespace

namespace llvm {

template <> struct DenseMapInfo<AssumedKnowledge> {
  static AssumedKnowledge getEmptyKey() {
    return {nullptr, nullptr, {nullptr, AssumedKnowledge::Empty}};
  }
  static AssumedKnowledge getTombstoneKey() {
    return {nullptr, nullptr, {nullptr, AssumedKnowledge::Tombstone}};
  }
  static unsigned getHashValue(const AssumedKnowledge &AK) {
    return hash_combine(AK.Name, AK.Argument, AK.WasOn.getPointer());
  }
  static bool isEqual(const AssumedKnowledge &LHS,
                      const AssumedKnowledge &RHS) {
    return LHS.WasOn == RHS.WasOn && LHS.Name == RHS.Name &&
           LHS.Argument == RHS.Argument;
  }
};

} // namespace llvm

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

  SmallDenseSet<AssumedKnowledge, 8> AssumedKnowledgeSet;

  AssumeBuilderState(Module *M) : M(M) {}

  void addAttribute(Attribute Attr, Value *WasOn) {
    if (!ShouldPreserveAllAttributes &&
        (Attr.isTypeAttribute() || Attr.isStringAttribute() ||
         !isUsefullToPreserve(Attr.getKindAsEnum())))
      return;
    StringRef Name;
    Value *AttrArg = nullptr;
    if (Attr.isStringAttribute())
      Name = Attr.getKindAsString();
    else
      Name = Attribute::getNameFromAttrKind(Attr.getKindAsEnum());
    if (Attr.isIntAttribute())
      AttrArg = ConstantInt::get(Type::getInt64Ty(M->getContext()),
                                 Attr.getValueAsInt());
    AssumedKnowledgeSet.insert(
        {Name.data(), AttrArg, {WasOn, AssumedKnowledge::None}});
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
    if (AssumedKnowledgeSet.empty())
      return nullptr;
    Function *FnAssume = Intrinsic::getDeclaration(M, Intrinsic::assume);
    LLVMContext &C = M->getContext();
    SmallVector<OperandBundleDef, 8> OpBundle;
    for (const AssumedKnowledge &Elem : AssumedKnowledgeSet) {
      SmallVector<Value *, 2> Args;
      assert(Attribute::getAttrKindFromName(Elem.Name) ==
                 Attribute::AttrKind::None ||
             static_cast<bool>(Elem.Argument) ==
                 Attribute::doesAttrKindHaveArgument(
                     Attribute::getAttrKindFromName(Elem.Name)));
      if (Elem.WasOn.getPointer())
        Args.push_back(Elem.WasOn.getPointer());
      if (Elem.Argument)
        Args.push_back(Elem.Argument);
      OpBundle.push_back(OperandBundleDefT<Value *>(Elem.Name, Args));
    }
    llvm::sort(OpBundle, isLowerOpBundle);
    return cast<IntrinsicInst>(CallInst::Create(
        FnAssume, ArrayRef<Value *>({ConstantInt::getTrue(C)}), OpBundle));
  }

  void addAttr(Attribute::AttrKind Kind, Value *Val, unsigned Argument = 0) {
    AssumedKnowledge AK;
    AK.Name = Attribute::getNameFromAttrKind(Kind).data();
    AK.WasOn.setPointer(Val);
    if (Attribute::doesAttrKindHaveArgument(Kind)) {
      AK.Argument =
          ConstantInt::get(Type::getInt64Ty(M->getContext()), Argument);
    } else {
      AK.Argument = nullptr;
      assert(Argument == 0 && "there should be no argument");
    }
    AssumedKnowledgeSet.insert(AK);
  };

  void addAccessedPtr(Instruction *MemInst, Value *Pointer, Type *AccType,
                      MaybeAlign MA) {
    uint64_t DerefSize = MemInst->getModule()
                             ->getDataLayout()
                             .getTypeStoreSize(AccType)
                             .getKnownMinSize();
    if (DerefSize != 0) {
      addAttr(Attribute::Dereferenceable, Pointer, DerefSize);
      if (!NullPointerIsDefined(MemInst->getFunction(),
                                Pointer->getType()->getPointerAddressSpace()))
        addAttr(Attribute::NonNull, Pointer);
    }
    if (MA.valueOrOne() > 1)
      addAttr(Attribute::Alignment, Pointer, MA.valueOrOne().value());
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

void llvm::salvageKnowledge(Instruction *I) {
  if (Instruction *Intr = buildAssumeFromInst(I))
    Intr->insertBefore(I);
}

PreservedAnalyses AssumeBuilderPass::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  for (Instruction &I : instructions(F))
    if (Instruction *Assume = buildAssumeFromInst(&I))
      Assume->insertBefore(&I);
  return PreservedAnalyses::all();
}
