//===------ PollyIRBuilder.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The Polly IRBuilder file contains Polly specific extensions for the IRBuilder
// that are used e.g. to emit the llvm.loop.parallel metadata.
//
//===----------------------------------------------------------------------===//

#include "polly/CodeGen/IRBuilder.h"
#include "polly/ScopInfo.h"
#include "polly/Support/ScopHelper.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Metadata.h"

using namespace llvm;
using namespace polly;

static const int MaxArraysInAliasScops = 10;

/// Get a self referencing id metadata node.
///
/// The MDNode looks like this (if arg0/arg1 are not null):
///
///    '!n = distinct !{!n, arg0, arg1}'
///
/// @return The self referencing id metadata node.
static MDNode *getID(LLVMContext &Ctx, Metadata *arg0 = nullptr,
                     Metadata *arg1 = nullptr) {
  MDNode *ID;
  SmallVector<Metadata *, 3> Args;
  // Reserve operand 0 for loop id self reference.
  Args.push_back(nullptr);

  if (arg0)
    Args.push_back(arg0);
  if (arg1)
    Args.push_back(arg1);

  ID = MDNode::getDistinct(Ctx, Args);
  ID->replaceOperandWith(0, ID);
  return ID;
}

ScopAnnotator::ScopAnnotator() : SE(nullptr), AliasScopeDomain(nullptr) {}

void ScopAnnotator::buildAliasScopes(Scop &S) {
  SE = S.getSE();

  LLVMContext &Ctx = SE->getContext();
  AliasScopeDomain = getID(Ctx, MDString::get(Ctx, "polly.alias.scope.domain"));

  AliasScopeMap.clear();
  OtherAliasScopeListMap.clear();

  // We are only interested in arrays, but no scalar references. Scalars should
  // be handled easily by basicaa.
  SmallVector<ScopArrayInfo *, 10> Arrays;
  for (ScopArrayInfo *Array : S.arrays())
    if (Array->isArrayKind())
      Arrays.push_back(Array);

  // The construction of alias scopes is quadratic in the number of arrays
  // involved. In case of too many arrays, skip the construction of alias
  // information to avoid quadratic increases in compile time and code size.
  if (Arrays.size() > MaxArraysInAliasScops)
    return;

  std::string AliasScopeStr = "polly.alias.scope.";
  for (const ScopArrayInfo *Array : Arrays) {
    assert(Array->getBasePtr() && "Base pointer must be present");
    AliasScopeMap[Array->getBasePtr()] =
        getID(Ctx, AliasScopeDomain,
              MDString::get(Ctx, (AliasScopeStr + Array->getName()).c_str()));
  }

  for (const ScopArrayInfo *Array : Arrays) {
    MDNode *AliasScopeList = MDNode::get(Ctx, {});
    for (const auto &AliasScopePair : AliasScopeMap) {
      if (Array->getBasePtr() == AliasScopePair.first)
        continue;

      Metadata *Args = {AliasScopePair.second};
      AliasScopeList =
          MDNode::concatenate(AliasScopeList, MDNode::get(Ctx, Args));
    }

    OtherAliasScopeListMap[Array->getBasePtr()] = AliasScopeList;
  }
}

void ScopAnnotator::pushLoop(Loop *L, bool IsParallel) {

  ActiveLoops.push_back(L);
  if (!IsParallel)
    return;

  BasicBlock *Header = L->getHeader();
  MDNode *Id = getID(Header->getContext());
  assert(Id->getOperand(0) == Id && "Expected Id to be a self-reference");
  assert(Id->getNumOperands() == 1 && "Unexpected extra operands in Id");
  MDNode *Ids = ParallelLoops.empty()
                    ? Id
                    : MDNode::concatenate(ParallelLoops.back(), Id);
  ParallelLoops.push_back(Ids);
}

void ScopAnnotator::popLoop(bool IsParallel) {
  ActiveLoops.pop_back();
  if (!IsParallel)
    return;

  assert(!ParallelLoops.empty() && "Expected a parallel loop to pop");
  ParallelLoops.pop_back();
}

void ScopAnnotator::annotateLoopLatch(BranchInst *B, Loop *L, bool IsParallel,
                                      bool IsLoopVectorizerDisabled) const {
  MDNode *MData = nullptr;

  if (IsLoopVectorizerDisabled) {
    SmallVector<Metadata *, 3> Args;
    LLVMContext &Ctx = SE->getContext();
    Args.push_back(MDString::get(Ctx, "llvm.loop.vectorize.enable"));
    auto *FalseValue = ConstantInt::get(Type::getInt1Ty(Ctx), 0);
    Args.push_back(ValueAsMetadata::get(FalseValue));
    MData = MDNode::concatenate(MData, getID(Ctx, MDNode::get(Ctx, Args)));
  }

  if (IsParallel) {
    assert(!ParallelLoops.empty() && "Expected a parallel loop to annotate");
    MDNode *Ids = ParallelLoops.back();
    MDNode *Id = cast<MDNode>(Ids->getOperand(Ids->getNumOperands() - 1));
    MData = MDNode::concatenate(MData, Id);
  }

  B->setMetadata("llvm.loop", MData);
}

/// Get the pointer operand
///
/// @param Inst The instruction to be analyzed.
/// @return the pointer operand in case @p Inst is a memory access
///         instruction and nullptr otherwise.
static llvm::Value *getMemAccInstPointerOperand(Instruction *Inst) {
  auto MemInst = MemAccInst::dyn_cast(Inst);
  if (!MemInst)
    return nullptr;

  return MemInst.getPointerOperand();
}

void ScopAnnotator::annotateSecondLevel(llvm::Instruction *Inst,
                                        llvm::Value *BasePtr) {
  Value *Ptr = getMemAccInstPointerOperand(Inst);
  if (!Ptr)
    return;

  auto *PtrSCEV = SE->getSCEV(Ptr);
  auto *BasePtrSCEV = SE->getPointerBase(PtrSCEV);

  auto SecondLevelAliasScope = SecondLevelAliasScopeMap.lookup(PtrSCEV);
  auto SecondLevelOtherAliasScopeList =
      SecondLevelOtherAliasScopeListMap.lookup(PtrSCEV);
  if (!SecondLevelAliasScope) {
    auto AliasScope = AliasScopeMap.lookup(BasePtr);
    if (!AliasScope)
      return;
    LLVMContext &Ctx = SE->getContext();
    SecondLevelAliasScope = getID(
        Ctx, AliasScope, MDString::get(Ctx, "second level alias metadata"));
    SecondLevelAliasScopeMap[PtrSCEV] = SecondLevelAliasScope;
    Metadata *Args = {SecondLevelAliasScope};
    auto SecondLevelBasePtrAliasScopeList =
        SecondLevelAliasScopeMap.lookup(BasePtrSCEV);
    SecondLevelAliasScopeMap[BasePtrSCEV] = MDNode::concatenate(
        SecondLevelBasePtrAliasScopeList, MDNode::get(Ctx, Args));
    auto OtherAliasScopeList = OtherAliasScopeListMap.lookup(BasePtr);
    SecondLevelOtherAliasScopeList = MDNode::concatenate(
        OtherAliasScopeList, SecondLevelBasePtrAliasScopeList);
    SecondLevelOtherAliasScopeListMap[PtrSCEV] = SecondLevelOtherAliasScopeList;
  }
  Inst->setMetadata("alias.scope", SecondLevelAliasScope);
  Inst->setMetadata("noalias", SecondLevelOtherAliasScopeList);
}

void ScopAnnotator::annotate(Instruction *Inst) {
  if (!Inst->mayReadOrWriteMemory())
    return;

  if (!ParallelLoops.empty())
    Inst->setMetadata("llvm.mem.parallel_loop_access", ParallelLoops.back());

  // TODO: Use the ScopArrayInfo once available here.
  if (!AliasScopeDomain)
    return;

  // Do not apply annotations on memory operations that take more than one
  // pointer. It would be ambiguous to which pointer the annotation applies.
  // FIXME: How can we specify annotations for all pointer arguments?
  if (isa<CallInst>(Inst) && !isa<MemSetInst>(Inst))
    return;

  auto *Ptr = getMemAccInstPointerOperand(Inst);
  if (!Ptr)
    return;

  auto *PtrSCEV = SE->getSCEV(Ptr);
  auto *BaseSCEV = SE->getPointerBase(PtrSCEV);
  auto *SU = dyn_cast<SCEVUnknown>(BaseSCEV);

  if (!SU)
    return;

  auto *BasePtr = SU->getValue();

  if (!BasePtr)
    return;

  auto AliasScope = AliasScopeMap.lookup(BasePtr);

  if (!AliasScope) {
    BasePtr = AlternativeAliasBases.lookup(BasePtr);
    if (!BasePtr)
      return;

    AliasScope = AliasScopeMap.lookup(BasePtr);
    if (!AliasScope)
      return;
  }

  assert(OtherAliasScopeListMap.count(BasePtr) &&
         "BasePtr either expected in AliasScopeMap and OtherAlias...Map");
  auto *OtherAliasScopeList = OtherAliasScopeListMap[BasePtr];

  if (InterIterationAliasFreeBasePtrs.count(BasePtr)) {
    annotateSecondLevel(Inst, BasePtr);
    return;
  }

  Inst->setMetadata("alias.scope", AliasScope);
  Inst->setMetadata("noalias", OtherAliasScopeList);
}

void ScopAnnotator::addInterIterationAliasFreeBasePtr(llvm::Value *BasePtr) {
  if (!BasePtr)
    return;

  InterIterationAliasFreeBasePtrs.insert(BasePtr);
}
