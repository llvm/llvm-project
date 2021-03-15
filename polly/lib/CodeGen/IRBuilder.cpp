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

ScopAnnotator::ScopAnnotator() : SE(nullptr), AliasScopeDomain(nullptr) {
  // Push an empty staging BandAttr.
  LoopAttrEnv.emplace_back();
}

ScopAnnotator::~ScopAnnotator() {
  assert(LoopAttrEnv.size() == 1 && "Loop stack imbalance");
  assert(!getStagingAttrEnv() && "Forgot to clear staging attr env");
}

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

  if (IsParallel) {
    LLVMContext &Ctx = SE->getContext();
    MDNode *AccessGroup = MDNode::getDistinct(Ctx, {});
    ParallelLoops.push_back(AccessGroup);
  }

  // Open an empty BandAttr context for loops nested in this one.
  LoopAttrEnv.emplace_back();
}

void ScopAnnotator::popLoop(bool IsParallel) {
  ActiveLoops.pop_back();

  if (IsParallel) {
    assert(!ParallelLoops.empty() && "Expected a parallel loop to pop");
    ParallelLoops.pop_back();
  }

  // Exit the subloop context.
  assert(!getStagingAttrEnv() && "Forgot to clear staging attr env");
  assert(LoopAttrEnv.size() >= 2 && "Popped too many");
  LoopAttrEnv.pop_back();
}

void ScopAnnotator::annotateLoopLatch(BranchInst *B, Loop *L, bool IsParallel,
                                      bool IsLoopVectorizerDisabled) const {
  LLVMContext &Ctx = SE->getContext();
  SmallVector<Metadata *, 3> Args;

  // For the LoopID self-reference.
  Args.push_back(nullptr);

  // Add the user-defined loop properties to the annotation, if any. Any
  // additional properties are appended.
  // FIXME: What to do if these conflict?
  MDNode *MData = nullptr;
  if (BandAttr *AttrEnv = getActiveAttrEnv()) {
    MData = AttrEnv->Metadata;
    if (MData)
      llvm::append_range(Args, drop_begin(MData->operands(), 1));
  }

  if (IsLoopVectorizerDisabled) {
    MDString *PropName = MDString::get(Ctx, "llvm.loop.vectorize.enable");
    ConstantInt *FalseValue = ConstantInt::get(Type::getInt1Ty(Ctx), 0);
    ValueAsMetadata *PropValue = ValueAsMetadata::get(FalseValue);
    Args.push_back(MDNode::get(Ctx, {PropName, PropValue}));
  }

  if (IsParallel) {
    MDString *PropName = MDString::get(Ctx, "llvm.loop.parallel_accesses");
    MDNode *AccGroup = ParallelLoops.back();
    Args.push_back(MDNode::get(Ctx, {PropName, AccGroup}));
  }

  // No metadata to annotate.
  if (!MData && Args.size() <= 1)
    return;

  // Reuse the MData node if possible, this will avoid having to create another
  // one that cannot be merged because LoopIDs are 'distinct'. However, we have
  // to create a new one if we add properties.
  if (!MData || Args.size() > MData->getNumOperands()) {
    MData = MDNode::getDistinct(Ctx, Args);
    MData->replaceOperandWith(0, MData);
  }
  B->setMetadata(LLVMContext::MD_loop, MData);
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

/// Find the base pointer of an array access.
///
/// This should be equivalent to ScalarEvolution::getPointerBase, which we
/// cannot use here the IR is still under construction which ScalarEvolution
/// assumes to not be modified.
static Value *findBasePtr(Value *Val) {
  while (true) {
    if (auto *Gep = dyn_cast<GEPOperator>(Val)) {
      Val = Gep->getPointerOperand();
      continue;
    }
    if (auto *Cast = dyn_cast<BitCastOperator>(Val)) {
      Val = Cast->getOperand(0);
      continue;
    }

    break;
  }

  return Val;
}

void ScopAnnotator::annotate(Instruction *Inst) {
  if (!Inst->mayReadOrWriteMemory())
    return;

  switch (ParallelLoops.size()) {
  case 0:
    // Not parallel to anything: no access group needed.
    break;
  case 1:
    // Single parallel loop: use directly.
    Inst->setMetadata(LLVMContext::MD_access_group,
                      cast<MDNode>(ParallelLoops.front()));
    break;
  default:
    // Parallel to multiple loops: refer to list of access groups.
    Inst->setMetadata(LLVMContext::MD_access_group,
                      MDNode::get(SE->getContext(),
                                  ArrayRef<Metadata *>(
                                      (Metadata *const *)ParallelLoops.data(),
                                      ParallelLoops.size())));
    break;
  }

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

  Value *BasePtr = findBasePtr(Ptr);
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
