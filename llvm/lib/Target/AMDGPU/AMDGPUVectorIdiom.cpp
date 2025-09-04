//===- AMDGPUVectorIdiom.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AMDGPU-specific vector idiom canonicalizations to unblock SROA and
// subsequent scalarization/vectorization.
//
// Motivation:
// - HIP vector types are often modeled as structs and copied with memcpy.
//   Address-level selects on such copies block SROA. Converting to value-level
//   operations or splitting the CFG enables SROA to break aggregates, which
//   unlocks scalarization/vectorization on AMDGPU.
//
// Example pattern:
//   %src = select i1 %c, ptr %A, ptr %B
//   call void @llvm.memcpy(ptr %dst, ptr %src, i32 16, i1 false)
//
// Objectives:
// - Canonicalize small memcpy patterns where source or destination is a select
// of pointers.
// - Prefer value-level selects (on loaded values) over address-level selects
// when safe.
// - When speculation is unsafe, split the CFG to isolate each arm.
//
// Assumptions:
// - Only handles non-volatile memcpy with constant length N where 0 < N <=
// MaxBytes (default 32).
// - Source and destination must be in the same address space.
// - Speculative loads are allowed only if a conservative alignment check
// passes.
// - No speculative stores are introduced.
//
// Transformations:
// - Source-select memcpy: attempt speculative loads -> value select -> single
// store.
//   Fallback is CFG split with two memcpy calls.
// - Destination-select memcpy: always CFG split to avoid speculative stores.
//
// Run this pass early, before SROA.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUVectorIdiom.h"
#include "AMDGPU.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "amdgpu-vector-idiom"

namespace {

// Default to 32 bytes since the largest HIP vector types are double4 or long4.
static cl::opt<unsigned> AMDGPUVectorIdiomMaxBytes(
    "amdgpu-vector-idiom-max-bytes",
    cl::desc("Max memcpy size (in bytes) to transform in AMDGPUVectorIdiom "
             "(default 32)"),
    cl::init(32));

// Selects an integer or integer-vector element type matching NBytes, using the
// minimum proven alignment to decide the widest safe element width.
// Assumptions:
// - Pointee types are opaque; the element choice is based solely on size and
// alignment.
// - Falls back to <N x i8> if wider lanes are not safe/aligned.
static Type *getIntOrVecTypeForSize(uint64_t NBytes, LLVMContext &Ctx,
                                    Align MinProvenAlign = Align(1)) {
  auto CanUseI64 = [&]() { return MinProvenAlign >= Align(8); };
  auto CanUseI32 = [&]() { return MinProvenAlign >= Align(4); };
  auto CanUseI16 = [&]() { return MinProvenAlign >= Align(2); };

  if (NBytes == 32 && CanUseI64())
    return FixedVectorType::get(Type::getInt64Ty(Ctx), 4);

  if ((NBytes % 4) == 0 && CanUseI32())
    return FixedVectorType::get(Type::getInt32Ty(Ctx), NBytes / 4);

  if ((NBytes % 2) == 0 && CanUseI16())
    return FixedVectorType::get(Type::getInt16Ty(Ctx), NBytes / 2);

  return FixedVectorType::get(Type::getInt8Ty(Ctx), NBytes);
}

static Align minAlign(Align A, Align B) { return A < B ? A : B; }

// Checks if both pointer operands can be speculatively loaded for N bytes and
// computes the minimum alignment to use.
// Notes:
// - Intentionally conservative: relies on getOrEnforceKnownAlignment.
// - AA/TLI are not used for deeper reasoning here.
static bool bothArmsSafeToSpeculateLoads(Value *A, Value *B, Align &OutAlign,
                                         const DataLayout &DL,
                                         AssumptionCache *AC,
                                         const DominatorTree *DT) {
  Align AlignA =
      llvm::getOrEnforceKnownAlignment(A, Align(1), DL, nullptr, AC, DT);
  Align AlignB =
      llvm::getOrEnforceKnownAlignment(B, Align(1), DL, nullptr, AC, DT);

  if (AlignA.value() < 1 || AlignB.value() < 1)
    return false;

  OutAlign = minAlign(AlignA, AlignB);
  return true;
}

// With opaque pointers, ensure address spaces match and otherwise return Ptr.
// Assumes the address space is the only property to validate for this cast.
static Value *castPtrTo(Value *Ptr, unsigned ExpectedAS) {
  auto *FromPTy = cast<PointerType>(Ptr->getType());
  unsigned AS = FromPTy->getAddressSpace();
  (void)ExpectedAS;
  assert(AS == ExpectedAS && "Address space mismatch for castPtrTo");
  return Ptr;
}

struct AMDGPUVectorIdiomImpl {
  unsigned MaxBytes;

  AMDGPUVectorIdiomImpl(unsigned MaxBytes) : MaxBytes(MaxBytes) {}

  // Rewrites memcpy when the source is a select of pointers. Prefers a
  // value-level select (two loads + select + one store) if speculative loads
  // are safe. Otherwise, falls back to a guarded CFG split with two memcpy
  // calls. Assumptions:
  // - Non-volatile, constant length, within MaxBytes.
  // - Source and destination in the same address space.
  bool transformSelectMemcpySource(MemTransferInst &MT, SelectInst &Sel,
                                   const DataLayout &DL,
                                   const DominatorTree *DT,
                                   AssumptionCache *AC) {
    IRBuilder<> B(&MT);
    Value *Dst = MT.getRawDest();
    Value *A = Sel.getTrueValue();
    Value *Bv = Sel.getFalseValue();
    if (!A->getType()->isPointerTy() || !Bv->getType()->isPointerTy())
      return false;

    ConstantInt *LenCI = cast<ConstantInt>(MT.getLength());
    uint64_t N = LenCI->getLimitedValue();

    Align DstAlign = MaybeAlign(MT.getDestAlign()).valueOrOne();
    Align AlignAB;
    bool CanSpeculate =
        bothArmsSafeToSpeculateLoads(A, Bv, AlignAB, DL, AC, DT);

    unsigned AS = cast<PointerType>(A->getType())->getAddressSpace();
    assert(AS == cast<PointerType>(Bv->getType())->getAddressSpace() &&
           "Expected same AS");

    if (CanSpeculate) {
      Align MinAlign = std::min(AlignAB, DstAlign);
      Type *Ty = getIntOrVecTypeForSize(N, B.getContext(), MinAlign);

      Value *PA = castPtrTo(A, AS);
      Value *PB = castPtrTo(Bv, AS);
      LoadInst *LA = B.CreateAlignedLoad(Ty, PA, MinAlign);
      LoadInst *LB = B.CreateAlignedLoad(Ty, PB, MinAlign);
      Value *V = B.CreateSelect(Sel.getCondition(), LA, LB);

      Value *PDst =
          castPtrTo(Dst, cast<PointerType>(Dst->getType())->getAddressSpace());
      (void)B.CreateAlignedStore(V, PDst, DstAlign);

      LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Rewrote memcpy(select-src) to "
                           "value-select loads/stores: "
                        << MT << "\n");
      MT.eraseFromParent();
      return true;
    }

    splitCFGForMemcpy(MT, Sel.getCondition(), A, Bv, true);
    LLVM_DEBUG(
        dbgs()
        << "[AMDGPUVectorIdiom] Rewrote memcpy(select-src) by CFG split\n");
    return true;
  }

  // Rewrites memcpy when the destination is a select of pointers. To avoid
  // speculative stores, always splits the CFG and emits a memcpy per branch.
  // Assumptions mirror the source case.
  bool transformSelectMemcpyDest(MemTransferInst &MT, SelectInst &Sel) {
    Value *DA = Sel.getTrueValue();
    Value *DB = Sel.getFalseValue();
    if (!DA->getType()->isPointerTy() || !DB->getType()->isPointerTy())
      return false;

    splitCFGForMemcpy(MT, Sel.getCondition(), DA, DB, false);
    LLVM_DEBUG(
        dbgs()
        << "[AMDGPUVectorIdiom] Rewrote memcpy(select-dst) by CFG split\n");
    return true;
  }

  // Splits the CFG around a memcpy whose source or destination depends on a
  // condition. Clones memcpy in then/else using TruePtr/FalsePtr and rejoins.
  // Assumptions:
  // - MT has constant length and is non-volatile.
  // - TruePtr/FalsePtr are correct replacements for the selected operand.
  void splitCFGForMemcpy(MemTransferInst &MT, Value *Cond, Value *TruePtr,
                         Value *FalsePtr, bool IsSource) {
    Function *F = MT.getFunction();
    BasicBlock *Cur = MT.getParent();
    BasicBlock *ThenBB = BasicBlock::Create(F->getContext(), "memcpy.then", F);
    BasicBlock *ElseBB = BasicBlock::Create(F->getContext(), "memcpy.else", F);
    BasicBlock *JoinBB =
        Cur->splitBasicBlock(BasicBlock::iterator(&MT), "memcpy.join");

    Cur->getTerminator()->eraseFromParent();
    IRBuilder<> B(Cur);
    B.CreateCondBr(Cond, ThenBB, ElseBB);

    ConstantInt *LenCI = cast<ConstantInt>(MT.getLength());

    IRBuilder<> BT(ThenBB);
    if (IsSource) {
      (void)BT.CreateMemCpy(MT.getRawDest(), MT.getDestAlign(), TruePtr,
                            MT.getSourceAlign(), LenCI, MT.isVolatile());
    } else {
      (void)BT.CreateMemCpy(TruePtr, MT.getDestAlign(), MT.getRawSource(),
                            MT.getSourceAlign(), LenCI, MT.isVolatile());
    }
    BT.CreateBr(JoinBB);

    IRBuilder<> BE(ElseBB);
    if (IsSource) {
      (void)BE.CreateMemCpy(MT.getRawDest(), MT.getDestAlign(), FalsePtr,
                            MT.getSourceAlign(), LenCI, MT.isVolatile());
    } else {
      (void)BE.CreateMemCpy(FalsePtr, MT.getDestAlign(), MT.getRawSource(),
                            MT.getSourceAlign(), LenCI, MT.isVolatile());
    }
    BE.CreateBr(JoinBB);

    MT.eraseFromParent();
  }
};

} // end anonymous namespace

AMDGPUVectorIdiomCombinePass::AMDGPUVectorIdiomCombinePass()
    : MaxBytes(AMDGPUVectorIdiomMaxBytes) {}

// Pass driver that locates small, constant-size, non-volatile memcpy calls
// where source or destination is a select in the same address space. Applies
// the source/destination transforms described above. Intended to run early to
// maximize SROA and subsequent optimizations.
PreservedAnalyses
AMDGPUVectorIdiomCombinePass::run(Function &F, FunctionAnalysisManager &FAM) {
  const DataLayout &DL = F.getParent()->getDataLayout();
  auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  auto &AC = FAM.getResult<AssumptionAnalysis>(F);

  SmallVector<CallInst *, 8> Worklist;
  for (Instruction &I : instructions(F)) {
    if (auto *CI = dyn_cast<CallInst>(&I)) {
      if (isa<MemTransferInst>(CI))
        Worklist.push_back(CI);
    }
  }

  bool Changed = false;
  AMDGPUVectorIdiomImpl Impl(MaxBytes);

  for (CallInst *CI : Worklist) {
    auto *MT = cast<MemTransferInst>(CI);
    if (MT->isVolatile())
      continue;

    ConstantInt *LenCI = dyn_cast<ConstantInt>(MT->getLength());
    if (!LenCI)
      continue;

    uint64_t N = LenCI->getLimitedValue();
    if (N == 0 || N > MaxBytes)
      continue;

    Value *Dst = MT->getRawDest();
    Value *Src = MT->getRawSource();

    unsigned DstAS = cast<PointerType>(Dst->getType())->getAddressSpace();
    unsigned SrcAS = cast<PointerType>(Src->getType())->getAddressSpace();
    if (DstAS != SrcAS)
      continue;

    if (auto *Sel = dyn_cast<SelectInst>(Src)) {
      Changed |= Impl.transformSelectMemcpySource(*MT, *Sel, DL, &DT, &AC);
      continue;
    }
    if (auto *Sel = dyn_cast<SelectInst>(Dst)) {
      Changed |= Impl.transformSelectMemcpyDest(*MT, *Sel);
      continue;
    }
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
