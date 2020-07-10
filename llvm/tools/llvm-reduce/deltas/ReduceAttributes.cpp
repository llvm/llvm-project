//===- ReduceAttributes.cpp - Specialized Delta Pass -------------------===//
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

using AttrPtrVecTy = std::vector<const Attribute *>;
using AttrPtrIdxVecVecTy = std::pair<unsigned, AttrPtrVecTy>;
using AttrPtrVecVecTy = SmallVector<AttrPtrIdxVecVecTy, 3>;

/// Given ChunksToKeep, produce a map of global variables/functions/calls
/// and indexes of attributes to be preserved for each of them.
class AttributeRemapper : public InstVisitor<AttributeRemapper> {
  Oracle O;

public:
  DenseMap<GlobalVariable *, AttrPtrVecTy> GlobalVariablesToRefine;
  DenseMap<Function *, AttrPtrVecVecTy> FunctionsToRefine;
  DenseMap<CallBase *, AttrPtrVecVecTy> CallsToRefine;

  explicit AttributeRemapper(ArrayRef<Chunk> ChunksToKeep) : O(ChunksToKeep) {}

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
    for (unsigned SetIdx : seq(AL.index_begin(), AL.index_end())) {
      AttrPtrIdxVecVecTy AttributesToPreserve;
      AttributesToPreserve.first = SetIdx;
      visitAttributeSet(AL.getAttributes(AttributesToPreserve.first),
                        AttributesToPreserve.second);
      if (!AttributesToPreserve.second.empty())
        AttributeSetsToPreserve.emplace_back(std::move(AttributesToPreserve));
    }
  }

  void visitAttributeSet(const AttributeSet &AS,
                         AttrPtrVecTy &AttrsToPreserve) {
    assert(AttrsToPreserve.empty() && "Should not be sharing vectors.");
    AttrsToPreserve.reserve(AS.getNumAttributes());
    for (const Attribute &A : AS)
      if (O.shouldKeep())
        AttrsToPreserve.emplace_back(&A);
  }
};

struct AttributeCounter : public InstVisitor<AttributeCounter> {
  /// How many features (in this case, attributes) did we count, total?
  int AttributeCount = 0;

  void visitModule(Module &M) {
    for (GlobalVariable &GV : M.getGlobalList())
      visitGlobalVariable(GV);
  }

  void visitGlobalVariable(GlobalVariable &GV) {
    // Global variables only have one attribute set.
    visitAttributeSet(GV.getAttributes());
  }

  void visitFunction(Function &F) {
    if (F.getIntrinsicID() != Intrinsic::not_intrinsic)
      return; // We can neither add nor remove attributes from intrinsics.
    visitAttributeList(F.getAttributes());
  }

  void visitCallBase(CallBase &I) { visitAttributeList(I.getAttributes()); }

  void visitAttributeList(const AttributeList &AL) {
    for (const AttributeSet &AS : AL)
      visitAttributeSet(AS);
  }

  void visitAttributeSet(const AttributeSet &AS) {
    AttributeCount += AS.getNumAttributes();
  }
};

} // namespace

AttributeSet
convertAttributeRefToAttributeSet(LLVMContext &C,
                                  ArrayRef<const Attribute *> Attributes) {
  AttrBuilder B;
  for (const Attribute *A : Attributes)
    B.addAttribute(*A);
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

  sort(SetVec, [](const std::pair<unsigned, AttributeSet> &LHS,
                  const std::pair<unsigned, AttributeSet> &RHS) {
    return LHS.first < RHS.first; // All values are unique.
  });

  return AttributeList::get(C, SetVec);
}

/// Removes out-of-chunk attributes from module.
static void extractAttributesFromModule(std::vector<Chunk> ChunksToKeep,
                                        Module *Program) {
  AttributeRemapper R(ChunksToKeep);
  R.visit(Program);

  LLVMContext &C = Program->getContext();
  for (const auto &I : R.GlobalVariablesToRefine)
    I.first->setAttributes(convertAttributeRefToAttributeSet(C, I.second));
  for (const auto &I : R.FunctionsToRefine)
    I.first->setAttributes(convertAttributeRefVecToAttributeList(C, I.second));
  for (const auto &I : R.CallsToRefine)
    I.first->setAttributes(convertAttributeRefVecToAttributeList(C, I.second));
}

/// Counts the amount of attributes.
static int countAttributes(Module *Program) {
  AttributeCounter C;

  // TODO: Silence index with --quiet flag
  outs() << "----------------------------\n";
  C.visit(Program);
  outs() << "Number of attributes: " << C.AttributeCount << "\n";

  return C.AttributeCount;
}

void llvm::reduceAttributesDeltaPass(TestRunner &Test) {
  outs() << "*** Reducing Attributes...\n";
  int AttributeCount = countAttributes(Test.getProgram());
  runDeltaPass(Test, AttributeCount, extractAttributesFromModule);
}
