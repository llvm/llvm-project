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
#include "llvm/ADT/DenseMap.h"
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
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

#define DEBUG_TYPE "spirv-legalize-resource-pointer-phis"

using namespace llvm;

static IntrinsicInst *getResourceGetPointer(Value *V) {
  auto *II = dyn_cast<IntrinsicInst>(V);
  if (II && II->getIntrinsicID() == Intrinsic::spv_resource_getpointer)
    return II;
  return nullptr;
}

static SmallVector<Value *> getPointerInputs(Instruction *I) {
  if (auto *Sel = dyn_cast<SelectInst>(I))
    return {Sel->getTrueValue(), Sel->getFalseValue()};
  return SmallVector<Value *>(cast<PHINode>(I)->incoming_values());
}

/// Returns true if the getpointer index on HandleTy selects a texel or an
/// array element. A struct member index must stay constant.
static bool hasElementIndex(Type *HandleTy) {
  auto *TET = dyn_cast<TargetExtType>(HandleTy);
  if (!TET)
    return false;
  StringRef Name = TET->getName();
  if (Name == "spirv.Image" || Name == "spirv.SignedImage")
    return true;
  return Name == "spirv.VulkanBuffer" && TET->getNumTypeParameters() > 0 &&
         isa<ArrayType>(TET->getTypeParameter(0));
}

/// Returns where the getpointer that replaces the phi or select I goes.
static BasicBlock::iterator getReplacementPoint(Instruction *I) {
  if (isa<PHINode>(I))
    return I->getParent()->getFirstInsertionPt();
  return std::next(I->getIterator());
}

/// Returns the pointer operand of a memory access, or null when I is not one.
static Value *getAccessPointerOperand(Instruction &I) {
  if (auto *RMW = dyn_cast<AtomicRMWInst>(&I))
    return RMW->getPointerOperand();
  if (auto *CmpXchg = dyn_cast<AtomicCmpXchgInst>(&I))
    return CmpXchg->getPointerOperand();
  return getLoadStorePointerOperand(&I);
}

/// Returns true if every use of Ptr outside Phis is the pointer of a memory
/// access, directly or through GEPs.
static bool isOnlyAccessed(Value *Ptr, const SetVector<Instruction *> &Phis) {
  return all_of(Ptr->users(), [&](User *U) {
    auto *I = cast<Instruction>(U);
    if (Phis.contains(I))
      return true;
    if (isa<GetElementPtrInst>(I))
      return isOnlyAccessed(I, Phis);
    return getAccessPointerOperand(*I) == Ptr;
  });
}

/// Collects into Phis the phis and selects reachable from Root through their
/// inputs, and into GetPointers the getpointer calls among those inputs.
/// Returns true if every other input is undef or poison and the getpointer
/// calls share one handle.
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
      if (!GetPointers.empty()) {
        IntrinsicInst *First = GetPointers.front();
        if (GP->getArgOperand(0) != First->getArgOperand(0) ||
            GP->getArgOperand(1)->getType() !=
                First->getArgOperand(1)->getType() ||
            GP->getType() != First->getType())
          return false;
      }
      GetPointers.insert(GP);
    }
  }
  if (GetPointers.empty())
    return false;

  Value *Handle = GetPointers.front()->getArgOperand(0);
  if (!hasElementIndex(Handle->getType()))
    return false;

  // Skip phis with an undef input whose pointer has a use other than an
  // access, such as a comparison, since the handle may be poison on the undef
  // edge. An access through undef is already undefined.
  if (HasUndef &&
      !all_of(Phis, [&](Instruction *I) { return isOnlyAccessed(I, Phis); }))
    return false;

  auto *HandleDef = dyn_cast<Instruction>(Handle);
  return all_of(Phis, [&](Instruction *I) {
    auto *Phi = dyn_cast<PHINode>(I);
    if (!Phi)
      return !HandleDef || DT.dominates(HandleDef, I);
    // A block whose only non-phi instruction is an EH pad, such as
    // catchswitch, cannot hold the new call.
    BasicBlock *BB = Phi->getParent();
    if (BB->getFirstInsertionPt() == BB->end())
      return false;
    // Require the handle to be defined before the phi's block, so a loop
    // through the phi uses one handle on every iteration.
    return !HandleDef || DT.properlyDominates(HandleDef->getParent(), BB);
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
    if (auto *Phi = dyn_cast<PHINode>(I))
      IndexOf[I] = PHINode::Create(IndexTy, Phi->getNumIncomingValues(),
                                   Phi->getName() + ".idx", Phi->getIterator());
    else
      IndexOf[I] = SelectInst::Create(cast<SelectInst>(I)->getCondition(),
                                      PoisonIndex, PoisonIndex,
                                      I->getName() + ".idx", I->getIterator());
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
  for (Instruction *I : Phis) {
    if (auto *Phi = dyn_cast<PHINode>(I)) {
      auto *IndexPhi = cast<PHINode>(IndexOf[I]);
      for (unsigned K = 0, E = Phi->getNumIncomingValues(); K != E; ++K)
        IndexPhi->addIncoming(GetIndex(Phi->getIncomingValue(K)),
                              Phi->getIncomingBlock(K));
    } else {
      auto *Sel = cast<SelectInst>(I);
      auto *IndexSel = cast<SelectInst>(IndexOf[I]);
      IndexSel->setTrueValue(GetIndex(Sel->getTrueValue()));
      IndexSel->setFalseValue(GetIndex(Sel->getFalseValue()));
    }
  }
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

  auto IsUseInPhis = [&](const Use &U) {
    return Phis.contains(cast<Instruction>(U.getUser()));
  };
  for (Instruction *I : Phis) {
    if (all_of(I->uses(), IsUseInPhis))
      continue;
    // Create a new call instead of cloning the first one, so call-site
    // attributes that hold only for its index, such as range, do not carry
    // over.
    IRBuilder<> B(getReplacementPoint(I));
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
  for (Instruction &I : instructions(F)) {
    Value *Ptr = getAccessPointerOperand(I);
    if (!Ptr)
      continue;
    // A GEP of the phi keeps its meaning when the phi becomes a getpointer, so
    // an access to a member or component of the element is a root too.
    while (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr))
      Ptr = GEP->getPointerOperand();
    if (isa<PHINode, SelectInst>(Ptr))
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
