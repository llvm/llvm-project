//===-- AMDGPULDSBuffering.cpp - Per-thread LDS buffering -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass buffers selected per-thread global memory accesses through LDS
// (addrspace(3)) to improve performance in memory-bound kernels.
//
// This is intended to help cases where a value is loaded from global memory and
// later stored back to (the same) global location, but intervening memory
// operations (e.g. potentially-aliasing accesses) inhibit load/store
// forwarding, keeping the load and store "live" and increasing global traffic
// and cache pressure.
//
// The pass runs late in the pipeline, after SROA and AMDGPUPromoteAlloca,
// using only leftover LDS budget to avoid interfering with other LDS
// optimizations. It respects the same LDS budget constraints as
// AMDGPUPromoteAlloca, ensuring that LDS usage remains within occupancy
// tier limits.
//
// Current implementation handles the simplest pattern: a load from global
// memory whose only use is a store back to the same pointer. This pattern is
// transformed into a pair of memcpy operations (global->LDS and LDS->global),
// effectively moving the value through LDS instead of accessing global memory
// directly.
//
// This pass was inspired by finding that some rocrand performance tests
// show better performance when global memory is buffered through LDS
// instead of being loaded/stored to registers directly. This optimization
// is experimental and must be enabled in the default pipeline via the
// -amdgpu-enable-lds-buffering flag (or explicitly scheduled via
// -passes='amdgpu-lds-buffering<...>').
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUTargetMachine.h"
#include "GCNSubtarget.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "Utils/AMDGPULDSUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Debug.h"
#include <algorithm>

#define DEBUG_TYPE "amdgpu-lds-buffering"

using namespace llvm;

namespace {

class AMDGPULDSBufferingImpl {
  const AMDGPUTargetMachine &TM;
  unsigned MaxBytes;
  Module *Mod = nullptr;
  const DataLayout *DL = nullptr;

public:
  AMDGPULDSBufferingImpl(const AMDGPUTargetMachine &TM, unsigned MaxBytes)
      : TM(TM), MaxBytes(MaxBytes) {}

  bool run(Function &F) {
    LLVM_DEBUG(dbgs() << "[LDSBuffer] Visit function: " << F.getName() << '\n');
    if (!AMDGPU::isEntryFunctionCC(F.getCallingConv()))
      return false;

    Mod = F.getParent();
    DL = &Mod->getDataLayout();

    auto Budget = AMDGPU::computeLDSBudget(F, TM);
    if (!Budget.promotable)
      return false;
    uint32_t localUsage = Budget.currentUsage;
    uint32_t localLimit = Budget.limit;

    const AMDGPUSubtarget &ST = AMDGPUSubtarget::get(TM, F);
    unsigned WorkGroupSize = ST.getFlatWorkGroupSizes(F).second;

    bool Changed = false;
    unsigned NumTransformed = 0;

    // Minimal pattern: a load from AS(1) whose only use is a store back to the
    // exact same pointer later. Replace with global<->LDS memcpy pair to
    // shorten the live range and free VGPRs.
    SmallVector<Instruction *> ToErase;
    for (BasicBlock &BB : F) {
      for (Instruction &I : llvm::make_early_inc_range(BB)) {
        auto *LI = dyn_cast<LoadInst>(&I);
        if (!LI || LI->isVolatile() || LI->isAtomic())
          continue;

        Type *ValTy = LI->getType();
        if (!ValTy->isFirstClassType())
          continue;

        Value *Ptr = LI->getPointerOperand();
        auto *PtrTy = cast<PointerType>(Ptr->getType());
        if (PtrTy->getAddressSpace() != AMDGPUAS::GLOBAL_ADDRESS)
          continue;

        if (!LI->hasOneUse())
          continue;
        auto *SI = dyn_cast<StoreInst>(LI->user_back());
        if (!SI || SI->isVolatile() || SI->isAtomic())
          continue;
        if (SI->getValueOperand() != LI)
          continue;

        Value *SPtr = SI->getPointerOperand();
        if (SPtr != Ptr)
          continue;

        TypeSize TS = DL->getTypeStoreSize(ValTy);
        if (TS.isScalable())
          continue;
        uint64_t Size = TS.getFixedValue();
        if (Size == 0 || Size > MaxBytes)
          continue;
        Align MinAlign = Align(16);
        Align LoadAlign = LI->getAlign();
        Align StoreAlign = SI->getAlign();
        Align Alignment = std::min(LoadAlign, StoreAlign);
        if (Alignment < MinAlign)
          continue;

        // Create LDS slot near the load and emit memcpy global->LDS.
        LLVM_DEBUG({
          dbgs() << "[LDSBuffer] Candidate found: load->store same ptr in "
                 << F.getName() << '\n';
          dbgs() << "            size=" << Size
                 << "B, loadAlign=" << LoadAlign.value()
                 << ", storeAlign=" << StoreAlign.value()
                 << ", chosenAlign=" << Alignment.value()
                 << ", ptr AS=" << PtrTy->getAddressSpace() << "\n";
        });
        IRBuilder<> BLoad(LI);

        // Ensure LDS budget allows allocating a per-thread slot.
        uint32_t NewSize = alignTo(localUsage, Alignment);
        NewSize += WorkGroupSize * static_cast<uint32_t>(Size);
        if (NewSize > localLimit)
          continue;
        localUsage = NewSize;
        auto [GV, SlotPtr] =
            createLDSGlobalAndThreadSlot(F, ValTy, Alignment, "ldsbuf", BLoad);
        // memcpy p3 <- p1
        LLVM_DEBUG(dbgs() << "[LDSBuffer] Insert memcpy global->LDS: "
                          << GV->getName() << ", bytes=" << Size
                          << ", align=" << Alignment.value() << '\n');
        BLoad.CreateMemCpy(SlotPtr, Alignment, Ptr, LoadAlign, TS);

        // Replace the final store with memcpy LDS->global.
        IRBuilder<> BStore(SI);
        LLVM_DEBUG(dbgs() << "[LDSBuffer] Insert memcpy LDS->global: "
                          << GV->getName() << ", bytes=" << Size
                          << ", align=" << Alignment.value() << '\n');
        BStore.CreateMemCpy(SPtr, StoreAlign, SlotPtr, Alignment, TS);

        ToErase.push_back(SI);
        ToErase.push_back(LI);
        LLVM_DEBUG(dbgs() << "[LDSBuffer] Erase original load/store pair\n");
        Changed = true;
        ++NumTransformed;
      }
    }

    for (Instruction *E : ToErase)
      E->eraseFromParent();

    LLVM_DEBUG(dbgs() << "[LDSBuffer] Transformations applied: "
                      << NumTransformed << "\n");

    return Changed;
  }

private:
  // Create an LDS array [WGSize x ElemTy] and return pointer to per-thread
  // slot.
  std::pair<GlobalVariable *, Value *>
  createLDSGlobalAndThreadSlot(Function &F, Type *ElemTy, Align Alignment,
                               StringRef BaseName, IRBuilder<> &Builder) {
    const AMDGPUSubtarget &ST = AMDGPUSubtarget::get(TM, F);
    unsigned WorkGroupSize = ST.getFlatWorkGroupSizes(F).second;
    Type *ArrTy = ArrayType::get(ElemTy, WorkGroupSize);
    GlobalVariable *GV = new GlobalVariable(
        *Mod, ArrTy, /*isConstant=*/false, GlobalValue::InternalLinkage,
        PoisonValue::get(ArrTy), (F.getName() + "." + BaseName).str(), nullptr,
        GlobalVariable::NotThreadLocal, AMDGPUAS::LOCAL_ADDRESS);
    GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    GV->setAlignment(Alignment);

    LLVM_DEBUG({
      dbgs() << "[LDSBuffer] Create LDS global: name=" << GV->getName()
             << ", elemTy=" << *ElemTy << ", WGSize=" << WorkGroupSize
             << ", align=" << Alignment.value() << '\n';
    });

    Value *LinearTID = AMDGPU::buildLinearThreadId(Builder, *Mod, ST);
    LLVMContext &Ctx = Mod->getContext();
    Value *Indices[] = {Constant::getNullValue(Type::getInt32Ty(Ctx)),
                        LinearTID};
    Value *SlotPtr = Builder.CreateInBoundsGEP(ArrTy, GV, Indices);
    return {GV, SlotPtr};
  }
};

} // end anonymous namespace

PreservedAnalyses AMDGPULDSBufferingPass::run(Function &F,
                                              FunctionAnalysisManager &AM) {
  bool Changed = AMDGPULDSBufferingImpl(TM, MaxBytes).run(F);
  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

//===----------------------------------------------------------------------===//
// Legacy PM wrapper
//===----------------------------------------------------------------------===//

namespace {

class AMDGPULDSBufferingLegacy : public FunctionPass {
public:
  static char ID;
  AMDGPULDSBufferingLegacy() : FunctionPass(ID) {}

  StringRef getPassName() const override { return "AMDGPU LDS Buffering"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    FunctionPass::getAnalysisUsage(AU);
  }

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;
    if (auto *TPC = getAnalysisIfAvailable<TargetPassConfig>())
      return AMDGPULDSBufferingImpl(TPC->getTM<AMDGPUTargetMachine>(),
                                    /*MaxBytes=*/64)
          .run(F);
    return false;
  }
};

} // end anonymous namespace

char AMDGPULDSBufferingLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(AMDGPULDSBufferingLegacy, DEBUG_TYPE,
                      "AMDGPU per-thread LDS buffering", false, false)
INITIALIZE_PASS_END(AMDGPULDSBufferingLegacy, DEBUG_TYPE,
                    "AMDGPU per-thread LDS buffering", false, false)

FunctionPass *llvm::createAMDGPULDSBufferingLegacyPass() {
  return new AMDGPULDSBufferingLegacy();
}
