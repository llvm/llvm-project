//===- AArch64StackTagging.cpp - Stack tagging in IR --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64InstrInfo.h"
#include "AArch64Subtarget.h"
#include "AArch64TargetMachine.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/LiveRegUnits.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Local.h"
#include <cassert>
#include <iterator>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "stack-tagging"

static constexpr unsigned kTagGranuleSize = 16;

namespace {
class AArch64StackTagging : public FunctionPass {
  struct AllocaInfo {
    AllocaInst *AI;
    SmallVector<IntrinsicInst *, 2> LifetimeStart;
    SmallVector<IntrinsicInst *, 2> LifetimeEnd;
    SmallVector<DbgVariableIntrinsic *, 2> DbgVariableIntrinsics;
    int Tag; // -1 for non-tagged allocations
  };

public:
  static char ID; // Pass ID, replacement for typeid

  AArch64StackTagging() : FunctionPass(ID) {
    initializeAArch64StackTaggingPass(*PassRegistry::getPassRegistry());
  }

  bool isInterestingAlloca(const AllocaInst &AI);
  void alignAndPadAlloca(AllocaInfo &Info);

  void tagAlloca(AllocaInst *AI, Instruction *InsertBefore, Value *Ptr,
                 uint64_t Size);
  void untagAlloca(AllocaInst *AI, Instruction *InsertBefore, uint64_t Size);

  Instruction *
  insertBaseTaggedPointer(const MapVector<AllocaInst *, AllocaInfo> &Allocas,
                          const DominatorTree *DT);
  bool runOnFunction(Function &F) override;

  StringRef getPassName() const override { return "AArch64 Stack Tagging"; }

private:
  Function *F;
  Function *SetTagFunc;
  const DataLayout *DL;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }
};

} // end anonymous namespace

char AArch64StackTagging::ID = 0;

INITIALIZE_PASS_BEGIN(AArch64StackTagging, DEBUG_TYPE, "AArch64 Stack Tagging",
                      false, false)
INITIALIZE_PASS_END(AArch64StackTagging, DEBUG_TYPE, "AArch64 Stack Tagging",
                    false, false)

FunctionPass *llvm::createAArch64StackTaggingPass() {
  return new AArch64StackTagging();
}

bool AArch64StackTagging::isInterestingAlloca(const AllocaInst &AI) {
  // FIXME: support dynamic allocas
  bool IsInteresting =
      AI.getAllocatedType()->isSized() && AI.isStaticAlloca() &&
      // alloca() may be called with 0 size, ignore it.
      AI.getAllocationSizeInBits(*DL).getValue() > 0 &&
      // inalloca allocas are not treated as static, and we don't want
      // dynamic alloca instrumentation for them as well.
      !AI.isUsedWithInAlloca() &&
      // swifterror allocas are register promoted by ISel
      !AI.isSwiftError();
  return IsInteresting;
}

void AArch64StackTagging::tagAlloca(AllocaInst *AI, Instruction *InsertBefore,
                                    Value *Ptr, uint64_t Size) {
  IRBuilder<> IRB(InsertBefore);
  IRB.CreateCall(SetTagFunc, {Ptr, ConstantInt::get(IRB.getInt64Ty(), Size)});
}

void AArch64StackTagging::untagAlloca(AllocaInst *AI, Instruction *InsertBefore,
                                      uint64_t Size) {
  IRBuilder<> IRB(InsertBefore);
  IRB.CreateCall(SetTagFunc, {IRB.CreatePointerCast(AI, IRB.getInt8PtrTy()),
                              ConstantInt::get(IRB.getInt64Ty(), Size)});
}

Instruction *AArch64StackTagging::insertBaseTaggedPointer(
    const MapVector<AllocaInst *, AllocaInfo> &Allocas,
    const DominatorTree *DT) {
  BasicBlock *PrologueBB = nullptr;
  // Try sinking IRG as deep as possible to avoid hurting shrink wrap.
  for (auto &I : Allocas) {
    const AllocaInfo &Info = I.second;
    AllocaInst *AI = Info.AI;
    if (Info.Tag < 0)
      continue;
    if (!PrologueBB) {
      PrologueBB = AI->getParent();
      continue;
    }
    PrologueBB = DT->findNearestCommonDominator(PrologueBB, AI->getParent());
  }
  assert(PrologueBB);

  IRBuilder<> IRB(&PrologueBB->front());
  Function *IRG_SP =
      Intrinsic::getDeclaration(F->getParent(), Intrinsic::aarch64_irg_sp);
  Instruction *Base =
      IRB.CreateCall(IRG_SP, {Constant::getNullValue(IRB.getInt64Ty())});
  Base->setName("basetag");
  return Base;
}

void AArch64StackTagging::alignAndPadAlloca(AllocaInfo &Info) {
  unsigned NewAlignment = std::max(Info.AI->getAlignment(), kTagGranuleSize);
  Info.AI->setAlignment(NewAlignment);

  uint64_t Size = Info.AI->getAllocationSizeInBits(*DL).getValue() / 8;
  uint64_t AlignedSize = alignTo(Size, kTagGranuleSize);
  if (Size == AlignedSize)
    return;

  // Add padding to the alloca.
  Type *AllocatedType =
      Info.AI->isArrayAllocation()
          ? ArrayType::get(
                Info.AI->getAllocatedType(),
                dyn_cast<ConstantInt>(Info.AI->getArraySize())->getZExtValue())
          : Info.AI->getAllocatedType();
  Type *PaddingType =
      ArrayType::get(Type::getInt8Ty(F->getContext()), AlignedSize - Size);
  Type *TypeWithPadding = StructType::get(AllocatedType, PaddingType);
  auto *NewAI = new AllocaInst(
      TypeWithPadding, Info.AI->getType()->getAddressSpace(), nullptr, "", Info.AI);
  NewAI->takeName(Info.AI);
  NewAI->setAlignment(Info.AI->getAlignment());
  NewAI->setUsedWithInAlloca(Info.AI->isUsedWithInAlloca());
  NewAI->setSwiftError(Info.AI->isSwiftError());
  NewAI->copyMetadata(*Info.AI);

  auto *NewPtr = new BitCastInst(NewAI, Info.AI->getType(), "", Info.AI);
  Info.AI->replaceAllUsesWith(NewPtr);
  Info.AI->eraseFromParent();
  Info.AI = NewAI;
}

// FIXME: check for MTE extension
bool AArch64StackTagging::runOnFunction(Function &Fn) {
  if (!Fn.hasFnAttribute(Attribute::SanitizeMemTag))
    return false;

  F = &Fn;
  DL = &Fn.getParent()->getDataLayout();

  MapVector<AllocaInst *, AllocaInfo> Allocas; // need stable iteration order
  SmallVector<Instruction *, 8> RetVec;
  DenseMap<Value *, AllocaInst *> AllocaForValue;
  SmallVector<Instruction *, 4> UnrecognizedLifetimes;

  for (auto &BB : *F) {
    for (BasicBlock::iterator IT = BB.begin(); IT != BB.end(); ++IT) {
      Instruction *I = &*IT;
      if (auto *AI = dyn_cast<AllocaInst>(I)) {
        Allocas[AI].AI = AI;
        continue;
      }

      if (auto *DVI = dyn_cast<DbgVariableIntrinsic>(I)) {
        if (auto *AI =
                dyn_cast_or_null<AllocaInst>(DVI->getVariableLocation())) {
          Allocas[AI].DbgVariableIntrinsics.push_back(DVI);
        }
        continue;
      }

      auto *II = dyn_cast<IntrinsicInst>(I);
      if (II && (II->getIntrinsicID() == Intrinsic::lifetime_start ||
                 II->getIntrinsicID() == Intrinsic::lifetime_end)) {
        AllocaInst *AI =
            llvm::findAllocaForValue(II->getArgOperand(1), AllocaForValue);
        if (!AI) {
          UnrecognizedLifetimes.push_back(I);
          continue;
        }
        if (II->getIntrinsicID() == Intrinsic::lifetime_start)
          Allocas[AI].LifetimeStart.push_back(II);
        else
          Allocas[AI].LifetimeEnd.push_back(II);
      }

      if (isa<ReturnInst>(I) || isa<ResumeInst>(I) || isa<CleanupReturnInst>(I))
        RetVec.push_back(I);
    }
  }

  if (Allocas.empty())
    return false;

  int NextTag = 0;
  int NumInterestingAllocas = 0;
  for (auto &I : Allocas) {
    AllocaInfo &Info = I.second;
    assert(Info.AI);

    if (!isInterestingAlloca(*Info.AI)) {
      Info.Tag = -1;
      continue;
    }

    alignAndPadAlloca(Info);
    NumInterestingAllocas++;
    Info.Tag = NextTag;
    NextTag = (NextTag + 1) % 16;
  }

  if (NumInterestingAllocas == 0)
    return true;

  SetTagFunc =
      Intrinsic::getDeclaration(F->getParent(), Intrinsic::aarch64_settag);

  // Compute DT only if the function has the attribute, there are more than 1
  // interesting allocas, and it is not available for free.
  Instruction *Base;
  if (NumInterestingAllocas > 1) {
    auto *DTWP = getAnalysisIfAvailable<DominatorTreeWrapperPass>();
    if (DTWP) {
      Base = insertBaseTaggedPointer(Allocas, &DTWP->getDomTree());
    } else {
      DominatorTree DT(*F);
      Base = insertBaseTaggedPointer(Allocas, &DT);
    }
  } else {
    Base = insertBaseTaggedPointer(Allocas, nullptr);
  }

  for (auto &I : Allocas) {
    const AllocaInfo &Info = I.second;
    AllocaInst *AI = Info.AI;
    if (Info.Tag < 0)
      continue;

    // Replace alloca with tagp(alloca).
    IRBuilder<> IRB(Info.AI->getNextNode());
    Function *TagP = Intrinsic::getDeclaration(
        F->getParent(), Intrinsic::aarch64_tagp, {Info.AI->getType()});
    Instruction *TagPCall =
        IRB.CreateCall(TagP, {Constant::getNullValue(Info.AI->getType()), Base,
                              ConstantInt::get(IRB.getInt64Ty(), Info.Tag)});
    if (Info.AI->hasName())
      TagPCall->setName(Info.AI->getName() + ".tag");
    Info.AI->replaceAllUsesWith(TagPCall);
    TagPCall->setOperand(0, Info.AI);

    if (UnrecognizedLifetimes.empty() && Info.LifetimeStart.size() == 1 &&
        Info.LifetimeEnd.size() == 1) {
      IntrinsicInst *Start = Info.LifetimeStart[0];
      uint64_t Size =
          dyn_cast<ConstantInt>(Start->getArgOperand(0))->getZExtValue();
      Size = alignTo(Size, kTagGranuleSize);
      tagAlloca(AI, Start->getNextNode(), Start->getArgOperand(1), Size);
      untagAlloca(AI, Info.LifetimeEnd[0], Size);
    } else {
      uint64_t Size = Info.AI->getAllocationSizeInBits(*DL).getValue() / 8;
      Value *Ptr = IRB.CreatePointerCast(TagPCall, IRB.getInt8PtrTy());
      tagAlloca(AI, &*IRB.GetInsertPoint(), Ptr, Size);
      for (auto &RI : RetVec) {
        untagAlloca(AI, RI, Size);
      }
      // We may have inserted tag/untag outside of any lifetime interval.
      // Remove all lifetime intrinsics for this alloca.
      for (auto &II : Info.LifetimeStart)
        II->eraseFromParent();
      for (auto &II : Info.LifetimeEnd)
        II->eraseFromParent();
    }

    // Fixup debug intrinsics to point to the new alloca.
    for (auto DVI : Info.DbgVariableIntrinsics)
      DVI->setArgOperand(
          0,
          MetadataAsValue::get(F->getContext(), LocalAsMetadata::get(Info.AI)));
  }

  // If we have instrumented at least one alloca, all unrecognized lifetime
  // instrinsics have to go.
  for (auto &I : UnrecognizedLifetimes)
    I->eraseFromParent();

  return true;
}
