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

cl::opt<bool> ShouldPreserveAllAttributes(
    "assume-preserve-all", cl::init(false), cl::Hidden,
    cl::desc("enable preservation of all attrbitues. even those that are "
             "unlikely to be usefull"));

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

/// Index of elements in the operand bundle.
/// If the element exist it is guaranteed to be what is specified in this enum
/// but it may not exist.
enum BundleOpInfoElem {
  BOIE_WasOn = 0,
  BOIE_Argument = 1,
};

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
        Op.input_size() <= BOIE_Argument
            ? 0
            : cast<ConstantInt>(*(Op.input_begin() + BOIE_Argument))
                  ->getZExtValue(),
         Op.input_size() <= BOIE_WasOn
            ? StringRef("")
            : (*(Op.input_begin() + BOIE_WasOn))->getName());
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

#ifndef NDEBUG

static bool isExistingAttribute(StringRef Name) {
  return StringSwitch<bool>(Name)
#define GET_ATTR_NAMES
#define ATTRIBUTE_ALL(ENUM_NAME, DISPLAY_NAME) .Case(#DISPLAY_NAME, true)
#include "llvm/IR/Attributes.inc"
      .Default(false);
}

#endif

bool llvm::hasAttributeInAssume(CallInst &AssumeCI, Value *IsOn,
                                StringRef AttrName, uint64_t *ArgVal,
                                AssumeQuery AQR) {
  IntrinsicInst &Assume = cast<IntrinsicInst>(AssumeCI);
  assert(Assume.getIntrinsicID() == Intrinsic::assume &&
         "this function is intended to be used on llvm.assume");
  assert(isExistingAttribute(AttrName) && "this attribute doesn't exist");
  assert((ArgVal == nullptr || Attribute::doesAttrKindHaveArgument(
                                   Attribute::getAttrKindFromName(AttrName))) &&
         "requested value for an attribute that has no argument");
  if (Assume.bundle_op_infos().empty())
    return false;

  CallInst::bundle_op_iterator Lookup;

  /// The right attribute can be found by binary search. After this finding the
  /// right WasOn needs to be done via linear search.
  /// Element have been ordered by argument value so the first we find is the
  /// one we need.
  if (AQR == AssumeQuery::Lowest)
    Lookup =
        llvm::lower_bound(Assume.bundle_op_infos(), AttrName,
                          [](const CallBase::BundleOpInfo &BOI, StringRef RHS) {
                            assert(isExistingAttribute(BOI.Tag->getKey()) &&
                                   "this attribute doesn't exist");
                            return BOI.Tag->getKey() < RHS;
                          });
  else
    Lookup = std::prev(
        llvm::upper_bound(Assume.bundle_op_infos(), AttrName,
                          [](StringRef LHS, const CallBase::BundleOpInfo &BOI) {
                            assert(isExistingAttribute(BOI.Tag->getKey()) &&
                                   "this attribute doesn't exist");
                            return LHS < BOI.Tag->getKey();
                          }));

  auto getValueFromBundleOpInfo = [&Assume](const CallBase::BundleOpInfo &BOI,
                                            unsigned Idx) {
    assert(BOI.End - BOI.Begin > Idx && "index out of range");
    return (Assume.op_begin() + BOI.Begin + Idx)->get();
  };

  if (Lookup == Assume.bundle_op_info_end() ||
      Lookup->Tag->getKey() != AttrName)
    return false;
  if (IsOn) {
    assert((Lookup->End - Lookup->Begin > BOIE_WasOn) &&
           "missing argument of attribute");
    while (true) {
      if (Lookup == Assume.bundle_op_info_end() ||
          Lookup->Tag->getKey() != AttrName)
        return false;
      if (getValueFromBundleOpInfo(*Lookup, BOIE_WasOn) == IsOn)
        break;
      if (AQR == AssumeQuery::Highest &&
          Lookup == Assume.bundle_op_info_begin())
        return false;
      Lookup = Lookup + (AQR == AssumeQuery::Lowest ? 1 : -1);
    }
  }

  if (Lookup->End - Lookup->Begin < BOIE_Argument)
    return true;
  if (ArgVal)
    *ArgVal =
        cast<ConstantInt>(getValueFromBundleOpInfo(*Lookup, BOIE_Argument))
            ->getZExtValue();
  return true;
}

PreservedAnalyses AssumeBuilderPass::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  for (Instruction &I : instructions(F))
    if (Instruction *Assume = BuildAssumeFromInst(&I))
      Assume->insertBefore(&I);
  return PreservedAnalyses::all();
}
