//===- KnowledgeRetention.h - utilities to preserve informations *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/KnowledgeRetention.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

namespace {

cl::opt<bool> ShouldPreserveAllAttributes(
    "assume-preserve-all", cl::init(false), cl::Hidden,
    cl::desc("enable preservation of all attrbitues. even those that are "
             "unlikely to be usefull"));

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
/// - by the name of the attribute, (doesn't change)
/// - then by the Value of the argument, (doesn't change)
/// - lastly by the Name of the current Value it WasOn. (may change)
/// This order is deterministic and allows looking for the right kind of
/// attribute with binary search. However finding the right WasOn needs to be
/// done via linear search because values can get remplaced.
bool isLowerOpBundle(const OperandBundleDef &LHS, const OperandBundleDef &RHS) {
  auto getTuple = [](const OperandBundleDef &Op) {
    return std::make_tuple(
        Op.getTag(),
        Op.input_size() < 2
            ? 0
            : cast<ConstantInt>(*std::next(Op.input_begin()))->getZExtValue(),
        Op.input_size() < 1 ? StringRef("") : (*Op.input_begin())->getName());
  };
  return getTuple(LHS) < getTuple(RHS);
}

/// This class contain all knowledge that have been gather while building an
/// llvm.assume and the function to manipulate it.
struct AssumeBuilderState {
  Module *M;

  SmallDenseSet<AssumedKnowledge, 8> AssumedKnowledgeSet;

  AssumeBuilderState(Module *M) : M(M) {}

  void addAttribute(Attribute Attr, Value *WasOn) {
    StringRef Name;
    Value *AttrArg = nullptr;
    if (Attr.isStringAttribute())
      if (ShouldPreserveAllAttributes)
        Name = Attr.getKindAsString();
      else
        return;
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
      if (ShouldPreserveAllAttributes)
        for (Attribute Attr : AttrList.getFnAttributes())
          addAttribute(Attr, nullptr);
    };
    addAttrList(Call->getAttributes());
    if (Function *Fn = Call->getCalledFunction())
      addAttrList(Fn->getAttributes());
  }

  CallInst *build() {
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
    return CallInst::Create(
        FnAssume, ArrayRef<Value *>({ConstantInt::getTrue(C)}), OpBundle);
  }

  void addInstruction(const Instruction *I) {
    if (auto *Call = dyn_cast<CallBase>(I))
      addCall(Call);
    // TODO: Add support for the other Instructions.
    // TODO: Maybe we should look around and merge with other llvm.assume.
  }
};

} // namespace

CallInst *llvm::BuildAssumeFromInst(const Instruction *I, Module *M) {
  AssumeBuilderState Builder(M);
  Builder.addInstruction(I);
  return Builder.build();
}

PreservedAnalyses AssumeBuilderPass::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  for (Instruction &I : instructions(F))
    if (Instruction *Assume = BuildAssumeFromInst(&I))
      Assume->insertBefore(&I);
  return PreservedAnalyses::all();
}
