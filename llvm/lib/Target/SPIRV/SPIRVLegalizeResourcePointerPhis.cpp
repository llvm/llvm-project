//===- SPIRVLegalizeResourcePointerPhis.cpp - Index phis of resources -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The SPIR-V backend lowers a resource access together with the
// spv.resource.getpointer call that defines its pointer, for example as one
// OpImageRead on a texel buffer. A phi or select between the two breaks that
// pairing:
//
//   %p = phi ptr addrspace(11) [ %gp0, %a ], [ %gp1, %b ]
//   %v = load i32, ptr addrspace(11) %p
//
// getpointer does not access memory, so this pass moves the phi to the indices
// and calls getpointer once on the handle %h of %gp0 and %gp1:
//
//   %i = phi i32 [ 0, %a ], [ 1, %b ]
//   %p = call ptr addrspace(11) @llvm.spv.resource.getpointer(%h, i32 %i)
//   %v = load i32, ptr addrspace(11) %p
//
// The pass leaves phis over different handles or struct members unchanged.
//
//===----------------------------------------------------------------------===//

#include "SPIRV.h"
#include "SPIRVUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

#define DEBUG_TYPE "spirv-legalize-resource-pointer-phis"

using namespace llvm;

/// Returns the pointer inputs of a phi or select.
static iterator_range<Use *> getPointerInputs(Instruction *I) {
  return isa<SelectInst>(I) ? drop_begin(I->operands()) : I->operands();
}

/// Returns true if the getpointer index on HandleTy selects a texel or an
/// array element. A struct member index must stay constant.
static bool hasElementIndex(Type *HandleTy) {
  auto *TET = cast<TargetExtType>(HandleTy);
  return TET->getName() != "spirv.VulkanBuffer" ||
         isa<ArrayType>(TET->getTypeParameter(0));
}

/// Returns the pointer operand use of a memory access, or null when I is not
/// one.
static Use *getAccessPointerUse(Instruction &I) {
  if (isa<LoadInst>(I))
    return &I.getOperandUse(LoadInst::getPointerOperandIndex());
  if (isa<StoreInst>(I))
    return &I.getOperandUse(StoreInst::getPointerOperandIndex());
  if (isa<AtomicRMWInst>(I))
    return &I.getOperandUse(AtomicRMWInst::getPointerOperandIndex());
  if (isa<AtomicCmpXchgInst>(I))
    return &I.getOperandUse(AtomicCmpXchgInst::getPointerOperandIndex());
  return nullptr;
}

/// Returns true if every use of Ptr outside Phis is the pointer operand of a
/// memory access, directly or through GEPs.
static bool isOnlyAccessed(Value *Ptr, const SetVector<Instruction *> &Phis) {
  return all_of(Ptr->uses(), [&](Use &U) {
    auto *I = cast<Instruction>(U.getUser());
    if (Phis.contains(I))
      return true;
    if (isa<GetElementPtrInst>(I))
      return isOnlyAccessed(I, Phis);
    return getAccessPointerUse(*I) == &U;
  });
}

/// Collects into Phis the phis and selects reachable from Root through their
/// inputs, and into GetPointers the getpointer calls among those inputs.
/// Returns true if rewriteAsIndexPhis can rewrite them.
static bool collectPointerPhis(Instruction *Root, const DominatorTree &DT,
                               SetVector<Instruction *> &Phis,
                               SetVector<IntrinsicInst *> &GetPointers) {
  bool HasUndef = false;
  SmallVector<Instruction *> Worklist = {Root};
  while (!Worklist.empty()) {
    Instruction *I = Worklist.pop_back_val();
    if (!Phis.insert(I))
      continue;
    for (Value *In : getPointerInputs(I)) {
      if (isa<UndefValue>(In)) {
        HasUndef |= !isa<PoisonValue>(In);
        continue;
      }
      if (isa<PHINode, SelectInst>(In)) {
        Worklist.push_back(cast<Instruction>(In));
        continue;
      }
      // Skip calls with operand bundles, such as deoptimization state.
      // The new getpointer call does not carry them.
      IntrinsicInst *GP = getResourceGetPointer(In);
      if (!GP || GP->hasOperandBundles())
        return false;
      // The intrinsic is overloaded on its result, handle and index types, so
      // the same callee and handle mean the same types.
      if (!GetPointers.empty() &&
          (GP->getCalledFunction() != GetPointers[0]->getCalledFunction() ||
           GP->getArgOperand(0) != GetPointers[0]->getArgOperand(0)))
        return false;
      GetPointers.insert(GP);
    }
  }
  if (GetPointers.empty())
    return false;

  Value *Handle = GetPointers.front()->getArgOperand(0);
  if (!hasElementIndex(Handle->getType()))
    return false;

  auto *HandleDef = dyn_cast<Instruction>(Handle);
  return all_of(Phis, [&](Instruction *I) {
    // With an undef input anywhere in Phis, skip a member with a use other
    // than an access, such as a comparison, since the handle may be poison on
    // the undef edge. An access through undef is already undefined.
    if (HasUndef && !isOnlyAccessed(I, Phis))
      return false;
    // A catchswitch block does not have a place for the new call.
    std::optional<BasicBlock::iterator> InsertPt =
        I->getInsertionPointAfterDef();
    if (!InsertPt)
      return false;
    // Require the handle to be defined before a phi's block, so a loop through
    // the phi uses one handle on every iteration.
    if (isa<PHINode>(I) && HandleDef &&
        HandleDef->getParent() == I->getParent())
      return false;
    return DT.dominates(Handle, &**InsertPt);
  });
}

/// Creates a phi or select of getpointer indices for each member of Phis, and
/// returns the map from each member to it.
static DenseMap<Instruction *, Instruction *>
createIndexPhis(const SetVector<Instruction *> &Phis, Type *IndexTy) {
  Value *PoisonIndex = PoisonValue::get(IndexTy);

  // A phi can refer to a later member of Phis through a loop, so create every
  // index phi and select before filling in the operands.
  DenseMap<Instruction *, Instruction *> IndexOf;
  for (Instruction *I : Phis) {
    if (auto *Phi = dyn_cast<PHINode>(I)) {
      auto *IndexPhi =
          PHINode::Create(IndexTy, Phi->getNumIncomingValues(),
                          Phi->getName() + ".idx", Phi->getIterator());
      for (BasicBlock *BB : Phi->blocks())
        IndexPhi->addIncoming(PoisonIndex, BB);
      IndexOf[I] = IndexPhi;
    } else {
      IndexOf[I] = SelectInst::Create(cast<SelectInst>(I)->getCondition(),
                                      PoisonIndex, PoisonIndex,
                                      I->getName() + ".idx", I->getIterator());
    }
  }

  // An undef input becomes index 0, one of the values undef may take.
  auto GetIndex = [&](Value *In) -> Value * {
    if (isa<PoisonValue>(In))
      return PoisonIndex;
    if (isa<UndefValue>(In))
      return Constant::getNullValue(IndexTy);
    if (IntrinsicInst *GP = getResourceGetPointer(In))
      return GP->getArgOperand(1);
    return IndexOf.lookup(cast<Instruction>(In));
  };
  // An index phi or select has the operand numbering of its pointer phi or
  // select.
  for (Instruction *I : Phis)
    for (Use &In : getPointerInputs(I))
      IndexOf[I]->setOperand(In.getOperandNo(), GetIndex(In));
  return IndexOf;
}

/// Replaces Phis with phis and selects of the getpointer indices, and each use
/// outside Phis with a getpointer on the index.
static void rewriteAsIndexPhis(const SetVector<Instruction *> &Phis,
                               const SetVector<IntrinsicInst *> &GetPointers) {
  IntrinsicInst *First = GetPointers.front();
  Value *Handle = First->getArgOperand(0);
  DenseMap<Instruction *, Instruction *> IndexOf =
      createIndexPhis(Phis, First->getArgOperand(1)->getType());

  for (Instruction *I : Phis) {
    if (all_of(I->users(),
               [&](User *U) { return Phis.contains(cast<Instruction>(U)); }))
      continue;
    // Create a new call instead of cloning the first one, so call-site
    // attributes that hold only for its index, such as range, do not carry
    // over.
    IRBuilder<> B(*I->getInsertionPointAfterDef());
    B.SetCurrentDebugLocation(I->getDebugLoc());
    CallInst *GP =
        B.CreateCall(First->getCalledFunction(), {Handle, IndexOf[I]});
    GP->takeName(I);
    // Replace the uses in Phis too, which are erased below, so that debug
    // records also move to the new call.
    I->replaceAllUsesWith(GP);
  }

  // The members of Phis are now used only by each other.
  for (Instruction *I : Phis)
    I->replaceAllUsesWith(PoisonValue::get(I->getType()));
  for (Instruction *I : Phis)
    I->eraseFromParent();
  for (IntrinsicInst *GP : GetPointers)
    if (GP->use_empty())
      GP->eraseFromParent();
}

static bool runImpl(Function &F, const DominatorTree &DT) {
  SmallVector<WeakVH> Roots;
  DenseSet<Value *> SeenRoots;
  for (Instruction &I : instructions(F)) {
    Use *PtrUse = getAccessPointerUse(I);
    if (!PtrUse)
      continue;
    Value *Ptr = PtrUse->get();
    // A GEP of the phi keeps its meaning when the phi becomes a getpointer, so
    // an access to a member or component of the element is a root too.
    while (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr))
      Ptr = GEP->getPointerOperand();
    if (isa<PHINode, SelectInst>(Ptr) && SeenRoots.insert(Ptr).second)
      Roots.push_back(Ptr);
  }

  bool Changed = false;
  for (Value *Root : Roots) {
    // A rewrite for an earlier root erases this root if it was one of its phis.
    if (!Root)
      continue;
    SetVector<Instruction *> Phis;
    SetVector<IntrinsicInst *> GetPointers;
    if (!collectPointerPhis(cast<Instruction>(Root), DT, Phis, GetPointers))
      continue;
    rewriteAsIndexPhis(Phis, GetPointers);
    Changed = true;
  }
  return Changed;
}

namespace {
class SPIRVLegalizeResourcePointerPhisLegacy : public FunctionPass {
public:
  static char ID;
  SPIRVLegalizeResourcePointerPhisLegacy() : FunctionPass(ID) {}
  bool runOnFunction(Function &F) override {
    return runImpl(F, getAnalysis<DominatorTreeWrapperPass>().getDomTree());
  }
  StringRef getPassName() const override {
    return "SPIRV legalize resource pointer phis";
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.setPreservesCFG();
  }
};
} // namespace

char SPIRVLegalizeResourcePointerPhisLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(SPIRVLegalizeResourcePointerPhisLegacy, DEBUG_TYPE,
                      "SPIRV legalize resource pointer phis", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(SPIRVLegalizeResourcePointerPhisLegacy, DEBUG_TYPE,
                    "SPIRV legalize resource pointer phis", false, false)

PreservedAnalyses
SPIRVLegalizeResourcePointerPhisPass::run(Function &F,
                                          FunctionAnalysisManager &AM) {
  if (!runImpl(F, AM.getResult<DominatorTreeAnalysis>(F)))
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

FunctionPass *llvm::createSPIRVLegalizeResourcePointerPhisPass() {
  return new SPIRVLegalizeResourcePointerPhisLegacy();
}
