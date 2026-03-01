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
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "lowfat"

STATISTIC(NumInstrumentedLoads, "Number of loads instrumented");
STATISTIC(NumInstrumentedStores, "Number of stores instrumented");
STATISTIC(NumInstrumentedAtomics, "Number of atomic operations instrumented");

namespace {

/// Helper class to instrument a module with LowFat bounds checks.
class LowFatSanitizer {
public:
  LowFatSanitizer(Module &M, const LowFatSanitizerOptions &Options)
      : M(M), Options(Options), DL(M.getDataLayout()),
        IntptrTy(DL.getIntPtrType(M.getContext())) {}

  bool run();

private:
  /// Get or create the declaration for __lf_report_oob (fatal).
  FunctionCallee getReportOobFn();

  /// Get or create the declaration for __lf_warn_oob (non-fatal).
  FunctionCallee getWarnOobFn();

  /// Instrument a single function.
  bool instrumentFunction(Function &F);

  /// Instrument a memory access instruction.
  /// Returns true if instrumentation was inserted.
  bool instrumentMemoryAccess(Instruction *I, Value *Ptr, Type *AccessTy);

  Module &M;
  const LowFatSanitizerOptions &Options;
  const DataLayout &DL;
  Type *IntptrTy;
  FunctionCallee ReportOobFn;
  FunctionCallee WarnOobFn;
};

// LowFat Constants (must match lf_config.h)
static const uint64_t RegionBase = 0x100000000000ULL;
static const uint64_t RegionSizeLog = 32;
static const uint64_t MinSizeLog = 4;
static const uint64_t NumSizeClasses = 27;

FunctionCallee LowFatSanitizer::getReportOobFn() {
  if (!ReportOobFn) {
    // void __lf_report_oob(uptr ptr, uptr base, uptr size)
    Type *VoidTy = Type::getVoidTy(M.getContext());
    ReportOobFn = M.getOrInsertFunction(
        "__lf_report_oob",
        FunctionType::get(VoidTy, {IntptrTy, IntptrTy, IntptrTy}, false));
    if (auto *F = dyn_cast<Function>(ReportOobFn.getCallee()))
      F->addFnAttr(Attribute::NoReturn);
  }
  return ReportOobFn;
}

FunctionCallee LowFatSanitizer::getWarnOobFn() {
  if (!WarnOobFn) {
    // void __lf_warn_oob(uptr ptr, uptr base, uptr size)
    Type *VoidTy = Type::getVoidTy(M.getContext());
    WarnOobFn = M.getOrInsertFunction(
        "__lf_warn_oob",
        FunctionType::get(VoidTy, {IntptrTy, IntptrTy, IntptrTy}, false));
    if (auto *F = dyn_cast<Function>(WarnOobFn.getCallee()))
      F->addFnAttr(Attribute::NoUnwind);
  }
  return WarnOobFn;
}

bool LowFatSanitizer::instrumentMemoryAccess(Instruction *I, Value *Ptr,
                                              Type *AccessTy) {
  // Skip if the access type size is not known at compile time
  TypeSize AccessSize = DL.getTypeStoreSize(AccessTy);
  if (AccessSize.isScalable()) {
    LLVM_DEBUG(dbgs() << "[LowFat] Skipping scalable type access\n");
    return false;
  }

  IRBuilder<> IRB(I);

  // Convert pointer to integer
  Value *PtrInt = IRB.CreatePtrToInt(Ptr, IntptrTy);

  // 1. Get region index: (Ptr - RegionBase) >> RegionSizeLog
  Value *RegionBaseVal = ConstantInt::get(IntptrTy, RegionBase);
  Value *RegionOffset = IRB.CreateSub(PtrInt, RegionBaseVal);
  Value *RegionIndex = IRB.CreateLShr(RegionOffset, RegionSizeLog);

  // 2. Check if LowFat pointer: Region < NumSizeClasses
  Value *MaxRegion = ConstantInt::get(IntptrTy, NumSizeClasses);
  Value *IsLowFat = IRB.CreateICmpULT(RegionIndex, MaxRegion);

  // Split block for the slow path (if LowFat)
  Instruction *ThenTerm = SplitBlockAndInsertIfThen(IsLowFat, I, false);
  IRBuilder<> ThenIRB(ThenTerm);

  // 3. Compute bounds inside the 'then' block
  // Size = 1 << (Region + MinSizeLog)
  Value *MinSizeLogVal = ConstantInt::get(IntptrTy, MinSizeLog);
  Value *ShiftAmount = ThenIRB.CreateAdd(RegionIndex, MinSizeLogVal);
  Value *SizeOne = ConstantInt::get(IntptrTy, 1);
  Value *AllocSize = ThenIRB.CreateShl(SizeOne, ShiftAmount);

  // Mask = ~(AllocSize - 1)
  Value *SizeMinusOne = ThenIRB.CreateSub(AllocSize, SizeOne);
  Value *Mask = ThenIRB.CreateNot(SizeMinusOne);

  // Base = Ptr & Mask
  Value *Base = ThenIRB.CreateAnd(PtrInt, Mask);

  // End = Base + AllocSize
  Value *End = ThenIRB.CreateAdd(Base, AllocSize);

  // 4. Check access: Ptr + AccessSize <= End
  // Equivalent OOB check: (Ptr + AccessSize) > End
  Value *AccessSizeVal = ConstantInt::get(IntptrTy, AccessSize.getFixedValue());
  Value *AccessEnd = ThenIRB.CreateAdd(PtrInt, AccessSizeVal);
  Value *IsOOB = ThenIRB.CreateICmpUGT(AccessEnd, End);

  // Split again for OOB reporting
  Instruction *OobTerm = SplitBlockAndInsertIfThen(IsOOB, ThenTerm, false);
  IRBuilder<> OobIRB(OobTerm);
  
  // 5. Report OOB (slow path)
  FunctionCallee OobFn = Options.Recover ? getWarnOobFn() : getReportOobFn();
  OobIRB.CreateCall(OobFn, {PtrInt, Base, AllocSize});

  LLVM_DEBUG(dbgs() << "[LowFat] Instrumented (inline, "
                    << (Options.Recover ? "recover" : "fatal")
                    << "): " << *I << "\n");
  return true;
}

bool LowFatSanitizer::instrumentFunction(Function &F) {
  // Skip functions that shouldn't be instrumented
  if (F.isDeclaration())
    return false;

  // Skip the runtime library functions themselves
  if (F.getName().starts_with("__lf_"))
    return false;

  // Skip functions with nosanitize attribute
  if (F.hasFnAttribute(Attribute::NoSanitizeBounds))
    return false;

  LLVM_DEBUG(dbgs() << "[LowFat] Instrumenting function: " << F.getName() << "\n");

  bool Modified = false;

  // Collect instructions to instrument first to avoid iterator invalidation
  SmallVector<std::pair<Instruction *, std::pair<Value *, Type *>>, 16> ToInstrument;

  for (Instruction &I : instructions(F)) {
    Value *Ptr = nullptr;
    Type *AccessTy = nullptr;

    if (auto *LI = dyn_cast<LoadInst>(&I)) {
      if (!LI->isVolatile()) {
        Ptr = LI->getPointerOperand();
        AccessTy = LI->getType();
      }
    } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
      if (!SI->isVolatile()) {
        Ptr = SI->getPointerOperand();
        AccessTy = SI->getValueOperand()->getType();
      }
    } else if (auto *AI = dyn_cast<AtomicRMWInst>(&I)) {
      if (!AI->isVolatile()) {
        Ptr = AI->getPointerOperand();
        AccessTy = AI->getValOperand()->getType();
      }
    } else if (auto *AI = dyn_cast<AtomicCmpXchgInst>(&I)) {
      if (!AI->isVolatile()) {
        Ptr = AI->getPointerOperand();
        AccessTy = AI->getCompareOperand()->getType();
      }
    }

    if (Ptr && AccessTy) {
      ToInstrument.push_back({&I, {Ptr, AccessTy}});
    }
  }

  // Now instrument collected instructions
  for (auto &Entry : ToInstrument) {
    Instruction *I = Entry.first;
    Value *Ptr = Entry.second.first;
    Type *AccessTy = Entry.second.second;

    if (instrumentMemoryAccess(I, Ptr, AccessTy)) {
      Modified = true;
      if (isa<LoadInst>(I))
        ++NumInstrumentedLoads;
      else if (isa<StoreInst>(I))
        ++NumInstrumentedStores;
      else
        ++NumInstrumentedAtomics;
    }
  }

  return Modified;
}

bool LowFatSanitizer::run() {
  LLVM_DEBUG(dbgs() << "[LowFat] Running on module: " << M.getName() << "\n");

  bool Modified = false;

  for (Function &F : M) {
    Modified |= instrumentFunction(F);
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
