//===- LowFatSanitizer.cpp - LowFat Pointer Bounds Checking ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the LowFat Sanitizer instrumentation pass.
//
// LowFat pointers encode allocation bounds information directly in the pointer
// value through careful memory layout. This pass instruments memory accesses
// to call runtime functions that verify bounds using this encoded information.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/LowFatSanitizer.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ModRef.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "lowfat"

STATISTIC(NumInstrumentedLoads, "Number of loads instrumented");
STATISTIC(NumInstrumentedStores, "Number of stores instrumented");
STATISTIC(NumInstrumentedAtomics, "Number of atomic operations instrumented");
STATISTIC(NumInstrumentedMemIntrinsics, "Number of mem intrinsics instrumented");

namespace {

/// Helper class to instrument a module with LowFat bounds checks.
class LowFatSanitizer {
public:
  LowFatSanitizer(Module &M, const LowFatSanitizerOptions &Options)
      : M(M), Options(Options), DL(M.getDataLayout()),
        IntptrTy(DL.getIntPtrType(M.getContext())) {}

  bool run();

private:
  Module &M;
  const LowFatSanitizerOptions &Options;
  const DataLayout &DL;
  Type *IntptrTy;

  FunctionCallee ReportOobFn = nullptr;
  FunctionCallee WarnOobFn = nullptr;

  FunctionCallee getReportOobFn();
  FunctionCallee getWarnOobFn();

  bool instrumentFunction(Function &F);
  bool instrumentMemoryAccess(Instruction *I, Value *Ptr, Type *AccessTy);
  bool instrumentMemoryRange(Instruction *I, Value *Ptr, Value *Size,
                             bool IsWrite);

  // Constants (must match lf_config.h)
  static constexpr uint64_t RegionBase = 0x100000000000ULL;
  static constexpr uint64_t RegionSizeLog = 32;
  static constexpr uint64_t NumSizeClasses = 27; // kMaxSizeLog(30) - kMinSizeLog(4) + 1
  static constexpr uint64_t MinSizeLog = 4;
};

FunctionCallee LowFatSanitizer::getReportOobFn() {
  if (!ReportOobFn) {
    // void __lf_report_oob(uptr ptr, uptr base, uptr size, i8 is_write)
    Type *VoidTy = Type::getVoidTy(M.getContext());
    Type *I8Ty = Type::getInt8Ty(M.getContext());
    ReportOobFn = M.getOrInsertFunction(
        "__lf_report_oob",
        FunctionType::get(VoidTy, {IntptrTy, IntptrTy, IntptrTy, I8Ty}, false));
    if (auto *F = dyn_cast<Function>(ReportOobFn.getCallee())) {
      F->addFnAttr(Attribute::NoReturn);
      // inaccessibleMemOnly: prevents the branch from being eliminated
      // as "dead" if the optimizer can't prove OOB is impossible.
      F->setMemoryEffects(MemoryEffects::inaccessibleMemOnly());
    }
  }
  return ReportOobFn;
}

FunctionCallee LowFatSanitizer::getWarnOobFn() {
  if (!WarnOobFn) {
    // void __lf_warn_oob(uptr ptr, uptr base, uptr size, i8 is_write)
    Type *VoidTy = Type::getVoidTy(M.getContext());
    Type *I8Ty = Type::getInt8Ty(M.getContext());
    WarnOobFn = M.getOrInsertFunction(
        "__lf_warn_oob",
        FunctionType::get(VoidTy, {IntptrTy, IntptrTy, IntptrTy, I8Ty}, false));
    if (auto *F = dyn_cast<Function>(WarnOobFn.getCallee())) {
      F->addFnAttr(Attribute::NoUnwind);
      F->setMemoryEffects(MemoryEffects::inaccessibleMemOnly());
    }
  }
  return WarnOobFn;
}

bool LowFatSanitizer::instrumentMemoryAccess(Instruction *I, Value *Ptr,
                                              Type *AccessTy) {
  TypeSize AccessSize = DL.getTypeStoreSize(AccessTy);
  if (AccessSize.isScalable())
    return false;

  IRBuilder<> IRB(I);
  Value *PtrInt = IRB.CreatePtrToInt(Ptr, IntptrTy);

  // 1. Get region index: (Ptr - RegionBase) >> RegionSizeLog
  Value *RegionBaseVal = ConstantInt::get(IntptrTy, RegionBase);
  Value *RegionOffset = IRB.CreateSub(PtrInt, RegionBaseVal);
  Value *RegionIndex = IRB.CreateLShr(RegionOffset, RegionSizeLog);

  // 2. Check if LowFat pointer: Region < NumSizeClasses
  Value *MaxRegion = ConstantInt::get(IntptrTy, NumSizeClasses);
  Value *IsLowFat = IRB.CreateICmpULT(RegionIndex, MaxRegion);

  Instruction *ThenTerm = SplitBlockAndInsertIfThen(IsLowFat, I, false);
  IRBuilder<> ThenIRB(ThenTerm);

  // 3. Compute bounds inside the 'then' block
  Value *MinSizeLogVal = ConstantInt::get(IntptrTy, MinSizeLog);
  Value *ShiftAmount = ThenIRB.CreateAdd(RegionIndex, MinSizeLogVal);
  Value *SizeOne = ConstantInt::get(IntptrTy, 1);
  Value *AllocSize = ThenIRB.CreateShl(SizeOne, ShiftAmount);
  Value *SizeMinusOne = ThenIRB.CreateSub(AllocSize, SizeOne);
  Value *Mask = ThenIRB.CreateNot(SizeMinusOne);
  Value *Base = ThenIRB.CreateAnd(PtrInt, Mask);
  Value *End = ThenIRB.CreateAdd(Base, AllocSize);

  // 4. Check access: Ptr + AccessSize <= End
  Value *AccessSizeVal = ConstantInt::get(IntptrTy, AccessSize.getFixedValue());
  Value *AccessEnd = ThenIRB.CreateAdd(PtrInt, AccessSizeVal);
  Value *IsOOB = ThenIRB.CreateICmpUGT(AccessEnd, End);

  Instruction *OobTerm = SplitBlockAndInsertIfThen(IsOOB, ThenTerm, false);
  IRBuilder<> OobIRB(OobTerm);
  
  FunctionCallee OobFn = Options.Recover ? getWarnOobFn() : getReportOobFn();
  Type *I8Ty = Type::getInt8Ty(M.getContext());
  bool IsWrite = isa<StoreInst>(I) || isa<AtomicRMWInst>(I) ||
                 isa<AtomicCmpXchgInst>(I);
  Value *IsWriteVal = ConstantInt::get(I8Ty, IsWrite ? 1 : 0);
  OobIRB.CreateCall(OobFn, {PtrInt, Base, AllocSize, IsWriteVal});

  if (isa<LoadInst>(I)) NumInstrumentedLoads++;
  else if (isa<StoreInst>(I)) NumInstrumentedStores++;
  else NumInstrumentedAtomics++;

  return true;
}

bool LowFatSanitizer::instrumentMemoryRange(Instruction *I, Value *Ptr,
                                             Value *Size, bool IsWrite) {
  IRBuilder<> IRB(I);
  Value *PtrInt = IRB.CreatePtrToInt(Ptr, IntptrTy);
  Value *SizeInt = IRB.CreateZExtOrTrunc(Size, IntptrTy);

  Value *RegionBaseVal = ConstantInt::get(IntptrTy, RegionBase);
  Value *RegionOffset = IRB.CreateSub(PtrInt, RegionBaseVal);
  Value *RegionIndex = IRB.CreateLShr(RegionOffset, RegionSizeLog);

  Value *MaxRegion = ConstantInt::get(IntptrTy, NumSizeClasses);
  Value *IsLowFat = IRB.CreateICmpULT(RegionIndex, MaxRegion);

  Instruction *ThenTerm = SplitBlockAndInsertIfThen(IsLowFat, I, false);
  IRBuilder<> ThenIRB(ThenTerm);

  Value *MinSizeLogVal = ConstantInt::get(IntptrTy, MinSizeLog);
  Value *ShiftAmount = ThenIRB.CreateAdd(RegionIndex, MinSizeLogVal);
  Value *SizeOne = ConstantInt::get(IntptrTy, 1);
  Value *AllocSize = ThenIRB.CreateShl(SizeOne, ShiftAmount);
  Value *SizeMinusOne = ThenIRB.CreateSub(AllocSize, SizeOne);
  Value *Mask = ThenIRB.CreateNot(SizeMinusOne);
  Value *Base = ThenIRB.CreateAnd(PtrInt, Mask);
  Value *End = ThenIRB.CreateAdd(Base, AllocSize);

  Value *AccessEnd = ThenIRB.CreateAdd(PtrInt, SizeInt);
  Value *IsOOB = ThenIRB.CreateICmpUGT(AccessEnd, End);

  Instruction *OobTerm = SplitBlockAndInsertIfThen(IsOOB, ThenTerm, false);
  IRBuilder<> OobIRB(OobTerm);
  
  FunctionCallee OobFn = Options.Recover ? getWarnOobFn() : getReportOobFn();
  Type *I8Ty = Type::getInt8Ty(M.getContext());
  Value *IsWriteVal = ConstantInt::get(I8Ty, IsWrite ? 1 : 0);
  OobIRB.CreateCall(OobFn, {PtrInt, Base, AllocSize, IsWriteVal});

  NumInstrumentedMemIntrinsics++;
  return true;
}

bool LowFatSanitizer::instrumentFunction(Function &F) {
  bool Modified = false;
  SmallVector<Instruction *, 16> ToInstrument;

  for (auto &BB : F) {
    for (auto &I : BB) {
      if (isa<LoadInst>(&I) || isa<StoreInst>(&I) || isa<AtomicRMWInst>(&I) ||
          isa<AtomicCmpXchgInst>(&I))
        ToInstrument.push_back(&I);
      else if (isa<MemIntrinsic>(&I))
        ToInstrument.push_back(&I);
    }
  }

  for (Instruction *I : ToInstrument) {
    if (auto *LI = dyn_cast<LoadInst>(I))
      Modified |= instrumentMemoryAccess(I, LI->getPointerOperand(), LI->getType());
    else if (auto *SI = dyn_cast<StoreInst>(I))
      Modified |= instrumentMemoryAccess(I, SI->getPointerOperand(), SI->getValueOperand()->getType());
    else if (auto *RMW = dyn_cast<AtomicRMWInst>(I))
      Modified |= instrumentMemoryAccess(I, RMW->getPointerOperand(), RMW->getValOperand()->getType());
    else if (auto *CmpXchg = dyn_cast<AtomicCmpXchgInst>(I))
      Modified |= instrumentMemoryAccess(I, CmpXchg->getPointerOperand(), CmpXchg->getNewValOperand()->getType());
    else if (auto *MS = dyn_cast<MemSetInst>(I))
      Modified |= instrumentMemoryRange(I, MS->getDest(), MS->getLength(), true);
    else if (auto *MT = dyn_cast<MemTransferInst>(I)) {
      Modified |= instrumentMemoryRange(I, MT->getDest(), MT->getLength(), true);
      Modified |= instrumentMemoryRange(I, MT->getSource(), MT->getLength(), false);
    }
  }
  return Modified;
}

bool LowFatSanitizer::run() {
  LLVM_DEBUG(dbgs() << "[LowFat] run() Mode=" << (int)Options.Mode
                   << " BarrierOnly=" << Options.InternalBarrierOnly_ << "\n");
  LLVM_DEBUG(dbgs() << "[LowFat] Running on module: " << M.getName() << "\n");

  // Safe mode: InternalBarrierOnly_ is set for the PipelineStartEP pass.
  if (Options.InternalBarrierOnly_) {
    LLVM_DEBUG(dbgs() << "[LowFat] Inserting barriers (Safe mode)\n");
    bool Modified = false;
    // Declare llvm.sideeffect and llvm.fake.use once for the module.
    Function *SideEffectFn =
        Intrinsic::getOrInsertDeclaration(&M, Intrinsic::sideeffect);
    Function *FakeUseFn =
        Intrinsic::getOrInsertDeclaration(&M, Intrinsic::fake_use);
    for (Function &F : M) {
      if (F.isDeclaration() || F.empty())
        continue;

      // Insert @llvm.sideeffect() at function entry to prevent FunctionAttrs
      // from inferring memory(none) on callers, blocking call-level DCE.
      IRBuilder<> IRB(&*F.getEntryBlock().getFirstInsertionPt());
      IRB.CreateCall(SideEffectFn, {});
      LLVM_DEBUG(dbgs() << "    [LowFat] Inserted sideeffect barrier in: "
                       << F.getName() << "\n");

      // Insert @llvm.fake.use(loaded_val) immediately after every load.
      // Without this, Dead Argument Elimination (DAE) can prove that a
      // function's return value is unused at all call sites and rewrite
      //   ret %loaded_val  →  ret undef
      // making the load itself dead, which is then DCE'd before the LowFat
      // pass at OptimizerLastEP ever sees it.
      SmallVector<LoadInst *, 8> Loads;
      for (BasicBlock &BB : F)
        for (Instruction &I : BB)
          if (auto *LI = dyn_cast<LoadInst>(&I))
            Loads.push_back(LI);
      for (LoadInst *LI : Loads) {
        IRBuilder<> LIRB(LI->getNextNode());
        LIRB.CreateCall(FakeUseFn, {LI});
        LLVM_DEBUG(dbgs() << "    [LowFat] Inserted fake.use for load in: "
                         << F.getName() << "\n");
      }

      Modified = true;
    }
    return Modified;
  }

  bool Modified = false;
  for (Function &F : M) {
    if (F.isDeclaration() || F.empty())
      continue;
    Modified |= instrumentFunction(F);
  }

  // Emit a module constructor that calls __lf_set_recover(Recover) so the
  // runtime interceptors (memset/memcpy/memmove) know whether to warn or abort.
  // This runs before main() via .init_array / __mod_init_func.
  if (Options.Recover) {
    LLVMContext &Ctx = M.getContext();
    FunctionType *SetRecoverTy =
        FunctionType::get(Type::getVoidTy(Ctx), {Type::getInt32Ty(Ctx)}, false);
    FunctionCallee SetRecoverFn =
        M.getOrInsertFunction("__lf_set_recover", SetRecoverTy);
    Function *Ctor = Function::Create(
        FunctionType::get(Type::getVoidTy(Ctx), false),
        GlobalValue::InternalLinkage, "__lowfat_set_recover_ctor", &M);
    BasicBlock *BB = BasicBlock::Create(Ctx, "entry", Ctor);
    IRBuilder<> CtorBuilder(BB);
    CtorBuilder.CreateCall(SetRecoverFn,
                           {ConstantInt::get(Type::getInt32Ty(Ctx), 1)});
    CtorBuilder.CreateRetVoid();
    appendToGlobalCtors(M, Ctor, /*Priority=*/0);
    Modified = true;
  }

  return Modified;
}

} // anonymous namespace

LowFatSanitizerPass::LowFatSanitizerPass(const LowFatSanitizerOptions &Options)
    : Options(Options) {}

PreservedAnalyses LowFatSanitizerPass::run(Module &M,
                                           ModuleAnalysisManager &AM) {
  LowFatSanitizer Sanitizer(M, Options);
  if (!Sanitizer.run())
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}
