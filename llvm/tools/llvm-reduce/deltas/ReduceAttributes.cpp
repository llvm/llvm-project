//===- ReduceAttributes.cpp - Specialized Delta Pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function which calls the Generic Delta pass in order
// to reduce uninteresting attributes.
//
//===----------------------------------------------------------------------===//

#include "ReduceAttributes.h"
#include "Delta.h"
#include "TestRunner.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <iterator>
#include <utility>
#include <vector>

namespace llvm {
class LLVMContext;
} // namespace llvm

using namespace llvm;

namespace {

using AttrPtrVecTy = std::vector<Attribute>;
using AttrPtrIdxVecVecTy = std::pair<unsigned, AttrPtrVecTy>;
using AttrPtrVecVecTy = SmallVector<AttrPtrIdxVecVecTy, 3>;

/// Given ChunksToKeep, produce a map of global variables/functions/calls
/// and indexes of attributes to be preserved for each of them.
class AttributeRemapper : public InstVisitor<AttributeRemapper> {
  Oracle &O;

public:
  DenseMap<GlobalVariable *, AttrPtrVecTy> GlobalVariablesToRefine;
  DenseMap<Function *, AttrPtrVecVecTy> FunctionsToRefine;
  DenseMap<CallBase *, AttrPtrVecVecTy> CallsToRefine;

  explicit AttributeRemapper(Oracle &O) : O(O) {}

  void visitModule(Module &M) {
    for (GlobalVariable &GV : M.getGlobalList())
      visitGlobalVariable(GV);
  }

  void visitGlobalVariable(GlobalVariable &GV) {
    // Global variables only have one attribute set.
    const AttributeSet &AS = GV.getAttributes();
    if (AS.hasAttributes())
      visitAttributeSet(AS, GlobalVariablesToRefine[&GV]);
  }

  void visitFunction(Function &F) {
    if (F.getIntrinsicID() != Intrinsic::not_intrinsic)
      return; // We can neither add nor remove attributes from intrinsics.
    visitAttributeList(F.getAttributes(), FunctionsToRefine[&F]);
  }

  void visitCallBase(CallBase &I) {
    visitAttributeList(I.getAttributes(), CallsToRefine[&I]);
  }

  void visitAttributeList(const AttributeList &AL,
                          AttrPtrVecVecTy &AttributeSetsToPreserve) {
    assert(AttributeSetsToPreserve.empty() && "Should not be sharing vectors.");
    AttributeSetsToPreserve.reserve(AL.getNumAttrSets());
    for (unsigned SetIdx : AL.indexes()) {
      AttrPtrIdxVecVecTy AttributesToPreserve;
      AttributesToPreserve.first = SetIdx;
      visitAttributeSet(AL.getAttributes(AttributesToPreserve.first),
                        AttributesToPreserve.second);
      if (!AttributesToPreserve.second.empty())
        AttributeSetsToPreserve.emplace_back(std::move(AttributesToPreserve));
    }
  }

  // FIXME: Should just directly use AttrBuilder instead of going through
  // AttrPtrVecTy
  void visitAttributeSet(const AttributeSet &AS,
                         AttrPtrVecTy &AttrsToPreserve) {
    assert(AttrsToPreserve.empty() && "Should not be sharing vectors.");
    AttrsToPreserve.reserve(AS.getNumAttributes());

    // Optnone requires noinline, so removing noinline requires removing the
    // pair.
    Attribute NoInline = AS.getAttribute(Attribute::NoInline);
    bool RemoveNoInline = false;
    if (NoInline.isValid()) {
      RemoveNoInline = !O.shouldKeep();
      if (!RemoveNoInline)
        AttrsToPreserve.emplace_back(NoInline);
    }

    for (Attribute A : AS) {
      if (A.isEnumAttribute()) {
        Attribute::AttrKind Kind = A.getKindAsEnum();
        if (Kind == Attribute::NoInline)
          continue;

        if (RemoveNoInline && Kind == Attribute::OptimizeNone)
          continue;

        // TODO: Could only remove this if there are no constrained calls in the
        // function.
        if (Kind == Attribute::StrictFP) {
          AttrsToPreserve.emplace_back(A);
          continue;
        }
      }

      if (O.shouldKeep())
        AttrsToPreserve.emplace_back(A);
    }
  }
};

} // namespace

AttributeSet convertAttributeRefToAttributeSet(LLVMContext &C,
                                               ArrayRef<Attribute> Attributes) {
  AttrBuilder B(C);
  for (Attribute A : Attributes)
    B.addAttribute(A);
  return AttributeSet::get(C, B);
}

AttributeList convertAttributeRefVecToAttributeList(
    LLVMContext &C, ArrayRef<AttrPtrIdxVecVecTy> AttributeSets) {
  std::vector<std::pair<unsigned, AttributeSet>> SetVec;
  SetVec.reserve(AttributeSets.size());

  transform(AttributeSets, std::back_inserter(SetVec),
            [&C](const AttrPtrIdxVecVecTy &V) {
              return std::make_pair(
                  V.first, convertAttributeRefToAttributeSet(C, V.second));
            });

  llvm::sort(SetVec, llvm::less_first()); // All values are unique.

  return AttributeList::get(C, SetVec);
}

/// Removes out-of-chunk attributes from module.
static void extractAttributesFromModule(Oracle &O, Module &Program) {
  AttributeRemapper R(O);
  R.visit(Program);

  LLVMContext &C = Program.getContext();
  for (const auto &I : R.GlobalVariablesToRefine)
    I.first->setAttributes(convertAttributeRefToAttributeSet(C, I.second));
  for (const auto &I : R.FunctionsToRefine)
    I.first->setAttributes(convertAttributeRefVecToAttributeList(C, I.second));
  for (const auto &I : R.CallsToRefine)
    I.first->setAttributes(convertAttributeRefVecToAttributeList(C, I.second));
}

void llvm::reduceAttributesDeltaPass(TestRunner &Test) {
  runDeltaPass(Test, extractAttributesFromModule, "Reducing Attributes");
}
