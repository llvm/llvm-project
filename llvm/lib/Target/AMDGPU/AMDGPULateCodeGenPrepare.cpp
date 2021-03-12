//===-- AMDGPUCodeGenPrepare.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass does misc. AMDGPU optimizations on IR *just* before instruction
/// selection.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LegacyDivergenceAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Transforms/Utils/Local.h"

#define DEBUG_TYPE "amdgpu-late-codegenprepare"

using namespace llvm;

// Scalar load widening needs running after load-store-vectorizer as that pass
// doesn't handle overlapping cases. In addition, this pass enhances the
// widening to handle cases where scalar sub-dword loads are naturally aligned
// only but not dword aligned.
static cl::opt<bool>
    WidenLoads("amdgpu-late-codegenprepare-widen-constant-loads",
               cl::desc("Widen sub-dword constant address space loads in "
                        "AMDGPULateCodeGenPrepare"),
               cl::ReallyHidden, cl::init(true));

namespace {

class AMDGPULateCodeGenPrepare
    : public FunctionPass,
      public InstVisitor<AMDGPULateCodeGenPrepare, bool> {
  Module *Mod = nullptr;
  const DataLayout *DL = nullptr;

  AssumptionCache *AC = nullptr;
  LegacyDivergenceAnalysis *DA = nullptr;

public:
  static char ID;

  AMDGPULateCodeGenPrepare() : FunctionPass(ID) {}

  StringRef getPassName() const override {
    return "AMDGPU IR late optimizations";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<LegacyDivergenceAnalysis>();
    AU.setPreservesAll();
  }

  bool doInitialization(Module &M) override;
  bool runOnFunction(Function &F) override;

  bool visitInstruction(Instruction &) { return false; }

  // Check if the specified value is at least DWORD aligned.
  bool isDWORDAligned(const Value *V) const {
    KnownBits Known = computeKnownBits(V, *DL, 0, AC);
    return Known.countMinTrailingZeros() >= 2;
  }

  bool canWidenScalarExtLoad(LoadInst &LI) const;
  bool visitLoadInst(LoadInst &LI);
};

} // end anonymous namespace

bool AMDGPULateCodeGenPrepare::doInitialization(Module &M) {
  Mod = &M;
  DL = &Mod->getDataLayout();
  return false;
}

bool AMDGPULateCodeGenPrepare::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  AC = &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  DA = &getAnalysis<LegacyDivergenceAnalysis>();

  bool Changed = false;
  for (auto &BB : F)
    for (auto BI = BB.begin(), BE = BB.end(); BI != BE; /*EMPTY*/) {
      Instruction *I = &*BI++;
      Changed |= visit(*I);
    }

  return Changed;
}

bool AMDGPULateCodeGenPrepare::canWidenScalarExtLoad(LoadInst &LI) const {
  unsigned AS = LI.getPointerAddressSpace();
  // Skip non-constant address space.
  if (AS != AMDGPUAS::CONSTANT_ADDRESS &&
      AS != AMDGPUAS::CONSTANT_ADDRESS_32BIT)
    return false;
  // Skip non-simple loads.
  if (!LI.isSimple())
    return false;
  auto *Ty = LI.getType();
  // Skip aggregate types.
  if (Ty->isAggregateType())
    return false;
  unsigned TySize = DL->getTypeStoreSize(Ty);
  // Only handle sub-DWORD loads.
  if (TySize >= 4)
    return false;
  // That load must be at least naturally aligned.
  if (LI.getAlign() < DL->getABITypeAlign(Ty))
    return false;
  // It should be uniform, i.e. a scalar load.
  return DA->isUniform(&LI);
}

bool AMDGPULateCodeGenPrepare::visitLoadInst(LoadInst &LI) {
  if (!WidenLoads)
    return false;

  // Skip if that load is already aligned on DWORD at least as it's handled in
  // SDAG.
  if (LI.getAlign() >= 4)
    return false;

  if (!canWidenScalarExtLoad(LI))
    return false;

  int64_t Offset = 0;
  auto *Base =
      GetPointerBaseWithConstantOffset(LI.getPointerOperand(), Offset, *DL);
  // If that base is not DWORD aligned, it's not safe to perform the following
  // transforms.
  if (!isDWORDAligned(Base))
    return false;

  int64_t Adjust = Offset & 0x3;
  if (Adjust == 0) {
    // With a zero adjust, the original alignment could be promoted with a
    // better one.
    LI.setAlignment(Align(4));
    return true;
  }

  IRBuilder<> IRB(&LI);
  IRB.SetCurrentDebugLocation(LI.getDebugLoc());

  unsigned AS = LI.getPointerAddressSpace();
  unsigned LdBits = DL->getTypeStoreSize(LI.getType()) * 8;
  auto IntNTy = Type::getIntNTy(LI.getContext(), LdBits);

  PointerType *Int32PtrTy = Type::getInt32PtrTy(LI.getContext(), AS);
  PointerType *Int8PtrTy = Type::getInt8PtrTy(LI.getContext(), AS);
  auto *NewPtr = IRB.CreateBitCast(
      IRB.CreateConstGEP1_64(IRB.CreateBitCast(Base, Int8PtrTy),
                             Offset - Adjust),
      Int32PtrTy);
  LoadInst *NewLd = IRB.CreateAlignedLoad(IRB.getInt32Ty(), NewPtr, Align(4));
  NewLd->copyMetadata(LI);
  NewLd->setMetadata(LLVMContext::MD_range, nullptr);

  unsigned ShAmt = Adjust * 8;
  auto *NewVal = IRB.CreateBitCast(
      IRB.CreateTrunc(IRB.CreateLShr(NewLd, ShAmt), IntNTy), LI.getType());
  LI.replaceAllUsesWith(NewVal);
  RecursivelyDeleteTriviallyDeadInstructions(&LI);

  return true;
}

INITIALIZE_PASS_BEGIN(AMDGPULateCodeGenPrepare, DEBUG_TYPE,
                      "AMDGPU IR late optimizations", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(LegacyDivergenceAnalysis)
INITIALIZE_PASS_END(AMDGPULateCodeGenPrepare, DEBUG_TYPE,
                    "AMDGPU IR late optimizations", false, false)

char AMDGPULateCodeGenPrepare::ID = 0;

FunctionPass *llvm::createAMDGPULateCodeGenPreparePass() {
  return new AMDGPULateCodeGenPrepare();
}
