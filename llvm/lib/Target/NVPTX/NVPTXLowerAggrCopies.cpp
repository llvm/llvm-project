//===- NVPTXLowerAggrCopies.cpp - ------------------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// Lower aggregate copies, memset, memcpy, memmov intrinsics into loops when
// the size is large or is not a compile-time constant.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/StackProtector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"

#define DEBUG_TYPE "nvptx"

using namespace llvm;

static const unsigned MaxAggrCopySize = 128;

// Copy a small memcpy or memmove whose source and destination are both 16 byte
// aligned with <4 x i32> loads and stores, all loads before the stores. Left to
// SelectionDAG, the copy is split into i64 accesses, the widest legal integer,
// and each load waits on the previous store. Issuing every load first also
// keeps memmove correct when the two ranges overlap.
static bool expandAlignedSmallCopy(MemTransferInst *MT) {
  auto *LenCI = dyn_cast<ConstantInt>(MT->getLength());
  if (!LenCI || MT->isVolatile())
    return false;
  uint64_t Len = LenCI->getZExtValue();
  if (Len < 16 || Len >= MaxAggrCopySize)
    return false;
  const Align VecAlign(16);
  if (MT->getDestAlign().valueOrOne() < VecAlign ||
      MT->getSourceAlign().valueOrOne() < VecAlign)
    return false;
  uint64_t NumVecs = Len / 16;
  uint64_t Tail = Len % 16;
  if (Tail && isa<MemMoveInst>(MT))
    return false;

  IRBuilder<> Builder(MT);
  Type *VecTy = FixedVectorType::get(Builder.getInt32Ty(), 4);
  Value *Src = MT->getRawSource();
  Value *Dst = MT->getRawDest();
  SmallVector<Value *, 8> Loads;
  for (uint64_t I = 0; I < NumVecs; ++I) {
    Value *Ptr = Builder.CreateConstInBoundsGEP1_64(Builder.getInt8Ty(), Src,
                                                    I * 16);
    Loads.push_back(Builder.CreateAlignedLoad(VecTy, Ptr, VecAlign));
  }
  for (uint64_t I = 0; I < NumVecs; ++I) {
    Value *Ptr = Builder.CreateConstInBoundsGEP1_64(Builder.getInt8Ty(), Dst,
                                                    I * 16);
    Builder.CreateAlignedStore(Loads[I], Ptr, VecAlign);
  }
  if (Tail) {
    uint64_t Off = NumVecs * 16;
    Builder.CreateMemCpy(
        Builder.CreateConstInBoundsGEP1_64(Builder.getInt8Ty(), Dst, Off),
        VecAlign,
        Builder.CreateConstInBoundsGEP1_64(Builder.getInt8Ty(), Src, Off),
        VecAlign, Tail);
  }
  return true;
}

static bool lowerAggrCopies(Function &F, const TargetTransformInfo &TTI,
                            AAResults &AA) {
  SmallVector<LoadInst *, 4> AggrLoads;
  SmallVector<MemIntrinsic *, 4> MemCalls;
  SmallVector<MemTransferInst *, 4> SmallCopies;

  const DataLayout &DL = F.getDataLayout();
  LLVMContext &Context = F.getParent()->getContext();

  // Collect all aggregate loads and mem* calls.
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (LoadInst *LI = dyn_cast<LoadInst>(&I)) {
        if (!LI->hasOneUse())
          continue;

        if (DL.getTypeStoreSize(LI->getType()) < MaxAggrCopySize)
          continue;

        if (StoreInst *SI = dyn_cast<StoreInst>(LI->user_back())) {
          if (SI->getOperand(0) != LI)
            continue;
          AggrLoads.push_back(LI);
        }
      } else if (MemIntrinsic *IntrCall = dyn_cast<MemIntrinsic>(&I)) {
        // Convert intrinsic calls with variable size or with constant size
        // larger than the MaxAggrCopySize threshold.
        if (ConstantInt *LenCI = dyn_cast<ConstantInt>(IntrCall->getLength())) {
          if (LenCI->getZExtValue() >= MaxAggrCopySize) {
            MemCalls.push_back(IntrCall);
          } else if (auto *MT = dyn_cast<MemTransferInst>(IntrCall)) {
            SmallCopies.push_back(MT);
          }
        } else {
          MemCalls.push_back(IntrCall);
        }
      }
    }
  }

  bool Changed = false;
  for (MemTransferInst *MT : SmallCopies) {
    if (expandAlignedSmallCopy(MT)) {
      MT->eraseFromParent();
      Changed = true;
    }
  }

  if (AggrLoads.size() == 0 && MemCalls.size() == 0) {
    return Changed;
  }

  //
  // Do the transformation of an aggr load/copy/set to a loop
  //
  for (LoadInst *LI : AggrLoads) {
    auto *SI = cast<StoreInst>(*LI->user_begin());
    Value *SrcAddr = LI->getOperand(0);
    Value *DstAddr = SI->getOperand(1);
    unsigned NumLoads = DL.getTypeStoreSize(LI->getType());
    ConstantInt *CopyLen =
        ConstantInt::get(Type::getInt32Ty(Context), NumLoads);

    LocationSize Size = LocationSize::precise(NumLoads);
    if (AA.isNoAlias(MemoryLocation(SrcAddr, Size),
                     MemoryLocation(DstAddr, Size))) {
      // No overlap: emit a plain memcpy loop. Expand the loop here (rather
      // than emitting a memcpy intrinsic and letting the code below expand it)
      // so we can pass CanOverlap = false; expandMemCpyAsLoop would
      // conservatively assume overlap.
      createMemCpyLoopKnownSize(/* ConvertedInst */ SI,
                                /* SrcAddr */ SrcAddr, /* DstAddr */ DstAddr,
                                /* CopyLen */ CopyLen,
                                /* SrcAlign */ LI->getAlign(),
                                /* DestAlign */ SI->getAlign(),
                                /* SrcIsVolatile */ LI->isVolatile(),
                                /* DstIsVolatile */ SI->isVolatile(),
                                /* CanOverlap */ false, TTI);
    } else {
      // May alias: lower as a memmove, which picks the copy direction at
      // runtime. Emit the intrinsic here and let the loop below expand it.
      //
      // The pointers may alias even if they're in different address spaces
      // (e.g. the generic addrspace may alias global).  If they're in
      // different addrspaces, cast to the generic space first, because
      // expandMemMoveAsLoop needs to compare the pointer values to determine
      // the copy direction.
      IRBuilder<> Builder(SI);
      unsigned SrcAS = LI->getPointerAddressSpace();
      unsigned DstAS = SI->getPointerAddressSpace();
      if (SrcAS != DstAS) {
        PointerType *GenericPtrTy =
            PointerType::get(Context, NVPTXAS::ADDRESS_SPACE_GENERIC);
        SrcAddr = Builder.CreateAddrSpaceCast(SrcAddr, GenericPtrTy);
        DstAddr = Builder.CreateAddrSpaceCast(DstAddr, GenericPtrTy);
      }
      MemCalls.push_back(cast<MemMoveInst>(Builder.CreateMemMove(
          DstAddr, SI->getAlign(), SrcAddr, LI->getAlign(), CopyLen,
          LI->isVolatile() || SI->isVolatile())));
    }

    SI->eraseFromParent();
    LI->eraseFromParent();
  }

  // Transform mem* intrinsic calls.
  for (MemIntrinsic *MemCall : MemCalls) {
    bool Expanded = true;
    if (MemCpyInst *Memcpy = dyn_cast<MemCpyInst>(MemCall)) {
      expandMemCpyAsLoop(Memcpy, TTI);
    } else if (MemMoveInst *Memmove = dyn_cast<MemMoveInst>(MemCall)) {
      Expanded = expandMemMoveAsLoop(Memmove, TTI);
    } else if (MemSetInst *Memset = dyn_cast<MemSetInst>(MemCall)) {
      expandMemSetAsLoop(Memset, TTI);
    }
    if (Expanded)
      MemCall->eraseFromParent();
  }

  return true;
}

namespace {

struct NVPTXLowerAggrCopiesLegacyPass : public FunctionPass {
  static char ID;

  NVPTXLowerAggrCopiesLegacyPass() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addPreserved<StackProtector>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.addRequired<AAResultsWrapperPass>();
  }

  bool runOnFunction(Function &F) override {
    return lowerAggrCopies(
        F, getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F),
        getAnalysis<AAResultsWrapperPass>().getAAResults());
  }

  StringRef getPassName() const override {
    return "Lower aggregate copies/intrinsics into loops";
  }
};

char NVPTXLowerAggrCopiesLegacyPass::ID = 0;

} // namespace

INITIALIZE_PASS_BEGIN(
    NVPTXLowerAggrCopiesLegacyPass, "nvptx-lower-aggr-copies",
    "Lower aggregate copies, and llvm.mem* intrinsics into loops", false, false)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(
    NVPTXLowerAggrCopiesLegacyPass, "nvptx-lower-aggr-copies",
    "Lower aggregate copies, and llvm.mem* intrinsics into loops", false, false)

FunctionPass *llvm::createNVPTXLowerAggrCopiesLegacyPass() {
  return new NVPTXLowerAggrCopiesLegacyPass();
}

PreservedAnalyses NVPTXLowerAggrCopiesPass::run(Function &F,
                                                FunctionAnalysisManager &FAM) {
  if (!lowerAggrCopies(F, FAM.getResult<TargetIRAnalysis>(F),
                       FAM.getResult<AAManager>(F)))
    return PreservedAnalyses::all();
  // Copies are expanded into loops, so the CFG is not preserved.
  PreservedAnalyses PA;
  PA.preserve<SSPLayoutAnalysis>();
  return PA;
}
