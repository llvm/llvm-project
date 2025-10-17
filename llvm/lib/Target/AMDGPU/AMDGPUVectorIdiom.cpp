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
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
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

static cl::opt<bool>
    AMDGPUVectorIdiomEnable("amdgpu-vector-idiom-enable",
                            cl::desc("Enable pass AMDGPUVectorIdiom"),
                            cl::init(true));

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

// Checks if the underlying object of a memcpy operand is an alloca.
// This helps focus on scratch memory optimizations by filtering out
// memcpy operations that don't involve stack-allocated memory.
static bool hasAllocaUnderlyingObject(Value *V) {
  Value *Underlying = getUnderlyingObject(V);
  return isa<AllocaInst>(Underlying);
}

// Checks if both pointer operands can be speculatively loaded for N bytes and
// computes the minimum alignment to use.
// Notes:
// - Intentionally conservative: relies on isDereferenceablePointer and
//   getOrEnforceKnownAlignment.
// - AA/TLI are not used for deeper reasoning here.
// Emits verbose LLVM_DEBUG logs explaining why speculation is disallowed.
// Return false reasons include: either arm not dereferenceable or computed
// known alignment < 1.
static bool bothArmsSafeToSpeculateLoads(Value *A, Value *B, uint64_t Size,
                                         Align &OutAlign, const DataLayout &DL,
                                         AssumptionCache *AC,
                                         const DominatorTree *DT,
                                         Instruction *CtxI) {
  APInt SizeAPInt(DL.getIndexTypeSizeInBits(A->getType()), Size);
  if (!isDereferenceableAndAlignedPointer(B, Align(1), SizeAPInt, DL, CtxI, AC,
                                          DT, nullptr)) {
    LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Not speculating loads: false arm "
                      << "(B) not dereferenceable for " << Size
                      << " bytes at align(1)\n");
    LLVM_DEBUG(dbgs() << "    false arm (B) value: " << *B << '\n');
    return false;
  }

  Align AlignB =
      llvm::getOrEnforceKnownAlignment(B, Align(1), DL, nullptr, AC, DT);

  if (AlignB < Align(1)) {
    LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Not speculating loads: known "
                      << "alignment of false arm (B) < 1: " << AlignB.value()
                      << '\n');
    return false;
  }

  if (!isDereferenceableAndAlignedPointer(A, Align(1), SizeAPInt, DL, CtxI, AC,
                                          DT, nullptr)) {
    LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Not speculating loads: true arm "
                      << "(A) not dereferenceable for " << Size
                      << " bytes at align(1)\n");
    LLVM_DEBUG(dbgs() << "    true arm (A) value: " << *A << '\n');
    return false;
  }

  Align AlignA =
      llvm::getOrEnforceKnownAlignment(A, Align(1), DL, nullptr, AC, DT);

  if (AlignA < Align(1)) {
    LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Not speculating loads: known "
                      << "alignment of true arm (A) < 1: " << AlignA.value()
                      << '\n');
    return false;
  }

  OutAlign = minAlign(AlignA, AlignB);
  LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Speculative loads allowed: "
                    << "minAlign=" << OutAlign.value() << '\n');
  return true;
}

struct AMDGPUVectorIdiomImpl {
  const unsigned MaxBytes;
  bool CFGChanged = false;

  AMDGPUVectorIdiomImpl(unsigned MaxBytes) : MaxBytes(MaxBytes) {}

  // Rewrites memcpy when the source is a select of pointers. Prefers a
  // value-level select (two loads + select + one store) if speculative loads
  // are safe. Otherwise, falls back to a guarded CFG split with two memcpy
  // calls. Assumptions:
  // - Non-volatile, constant length, within MaxBytes.
  // - Source and destination in the same address space.
  bool transformSelectMemcpySource(MemCpyInst &MT, SelectInst &Sel,
                                   const DataLayout &DL,
                                   const DominatorTree *DT,
                                   AssumptionCache *AC) {
    LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Considering memcpy(select-src): "
                      << MT << '\n');
    IRBuilder<> B(&MT);
    Value *Dst = MT.getRawDest();
    Value *A = Sel.getTrueValue();
    Value *Bv = Sel.getFalseValue();

    ConstantInt *LenCI = cast<ConstantInt>(MT.getLength());
    uint64_t N = LenCI->getLimitedValue();

    if (Sel.isVolatile()) {
      LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Not rewriting: Select marked "
                        << "volatile (unexpected) in memcpy source\n");
      return false;
    }

    // This is a null check - always use CFG split
    Value *Cond = Sel.getCondition();
    ICmpInst *ICmp = dyn_cast<ICmpInst>(Cond);
    if (ICmp && ICmp->isEquality() &&
        (isa<ConstantPointerNull>(ICmp->getOperand(0)) ||
         isa<ConstantPointerNull>(ICmp->getOperand(1)))) {
      splitCFGForMemcpy(MT, Sel.getCondition(), A, Bv, true);
      LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Null check pattern - "
                           "using CFG split\n");
      return true;
    }

    Align DstAlign = MaybeAlign(MT.getDestAlign()).valueOrOne();
    Align AlignAB;
    bool CanSpeculate = false;

    const CallBase &CB = MT;
    const unsigned SrcArgIdx = 1;
    uint64_t DerefBytes = CB.getParamDereferenceableBytes(SrcArgIdx);
    bool HasDerefOrNull =
        CB.paramHasAttr(SrcArgIdx, Attribute::DereferenceableOrNull);
    bool HasNonNull = CB.paramHasAttr(SrcArgIdx, Attribute::NonNull);
    MaybeAlign SrcParamAlign = CB.getParamAlign(SrcArgIdx);
    Align ProvenSrcAlign =
        SrcParamAlign.value_or(MaybeAlign(MT.getSourceAlign()).valueOrOne());

    if (DerefBytes > 0) {
      LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] memcpy source param attrs: "
                        << "dereferenceable(" << DerefBytes << ")"
                        << (HasDerefOrNull ? " (or null)" : "")
                        << (HasNonNull ? ", nonnull" : "") << ", align "
                        << ProvenSrcAlign.value() << '\n');
      if (DerefBytes >= N && (!HasDerefOrNull || HasNonNull)) {
        LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Using memcpy source operand "
                          << "attributes at this use; accepting speculation\n");
        CanSpeculate = true;
        AlignAB = ProvenSrcAlign;
      } else {
        LLVM_DEBUG(
            dbgs() << "[AMDGPUVectorIdiom] Source param attrs not strong "
                   << "enough for speculation: need dereferenceable(" << N
                   << ") and nonnull; got dereferenceable(" << DerefBytes << ")"
                   << (HasDerefOrNull ? " (or null)" : "")
                   << (HasNonNull ? ", nonnull" : "") << '\n');
      }
    } else {
      LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] memcpy source param has no "
                        << "dereferenceable bytes attribute; align "
                        << ProvenSrcAlign.value() << '\n');
    }
    if (!CanSpeculate)
      CanSpeculate =
          bothArmsSafeToSpeculateLoads(A, Bv, N, AlignAB, DL, AC, DT, &MT);

    if (CanSpeculate) {
      Align MinAlign = std::min(AlignAB, DstAlign);
      LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Rewriting memcpy(select-src) "
                        << "with value-level select; N=" << N
                        << " minAlign=" << MinAlign.value() << '\n');

      Type *Ty = getIntOrVecTypeForSize(N, B.getContext(), MinAlign);

      LoadInst *LA = B.CreateAlignedLoad(Ty, A, MinAlign);
      LoadInst *LB = B.CreateAlignedLoad(Ty, Bv, MinAlign);
      Value *V = B.CreateSelect(Sel.getCondition(), LA, LB);

      (void)B.CreateAlignedStore(V, Dst, DstAlign);

      LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Rewrote memcpy(select-src) to "
                           "value-select loads/stores: "
                        << MT << '\n');
      MT.eraseFromParent();
      return true;
    }

    LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Falling back to CFG split for "
                      << "memcpy(select-src); speculation unsafe\n");
    splitCFGForMemcpy(MT, Sel.getCondition(), A, Bv, true);
    LLVM_DEBUG(
        dbgs()
        << "[AMDGPUVectorIdiom] Rewrote memcpy(select-src) by CFG split\n");
    return true;
  }

  // Rewrites memcpy when the destination is a select of pointers. To avoid
  // speculative stores, always splits the CFG and emits a memcpy per branch.
  // Assumptions mirror the source case.
  bool transformSelectMemcpyDest(MemCpyInst &MT, SelectInst &Sel) {
    Value *DA = Sel.getTrueValue();
    Value *DB = Sel.getFalseValue();
    LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Rewriting memcpy(select-dst) via "
                      << "CFG split to avoid speculative stores: " << MT
                      << '\n');

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
  void splitCFGForMemcpy(MemCpyInst &MT, Value *Cond, Value *TruePtr,
                         Value *FalsePtr, bool IsSource) {
    CFGChanged = true;

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

AMDGPUVectorIdiomCombinePass::AMDGPUVectorIdiomCombinePass(unsigned MaxBytes)
    : MaxBytes(MaxBytes) {}

// Pass driver that locates small, constant-size, non-volatile memcpy calls
// where source or destination is a select in the same address space. Applies
// the source/destination transforms described above. Intended to run early to
// maximize SROA and subsequent optimizations.
PreservedAnalyses
AMDGPUVectorIdiomCombinePass::run(Function &F, FunctionAnalysisManager &FAM) {
  const DataLayout &DL = F.getParent()->getDataLayout();
  auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  auto &AC = FAM.getResult<AssumptionAnalysis>(F);

  if (!AMDGPUVectorIdiomEnable)
    return PreservedAnalyses::all();

  SmallVector<MemCpyInst *, 8> Worklist;
  for (Instruction &I : instructions(F)) {
    if (auto *MC = dyn_cast<MemCpyInst>(&I))
      Worklist.push_back(MC);
  }

  bool Changed = false;
  AMDGPUVectorIdiomImpl Impl(MaxBytes);

  for (MemCpyInst *MT : Worklist) {
    Value *Dst = MT->getRawDest();
    Value *Src = MT->getRawSource();
    if (!isa<SelectInst>(Src) && !isa<SelectInst>(Dst))
      continue;

    LLVM_DEBUG({
      Value *DstV = MT->getRawDest();
      Value *SrcV = MT->getRawSource();
      unsigned DstAS = cast<PointerType>(DstV->getType())->getAddressSpace();
      unsigned SrcAS = cast<PointerType>(SrcV->getType())->getAddressSpace();
      Value *LenV = MT->getLength();

      auto dumpPtrForms = [&](StringRef Label, Value *V) {
        dbgs() << "      " << Label << ": " << *V << '\n';

        Value *StripCasts = V->stripPointerCasts();
        if (StripCasts != V)
          dbgs() << "        - stripCasts: " << *StripCasts << '\n';
        else
          dbgs() << "        - stripCasts: (no change)\n";

        Value *Underlying = getUnderlyingObject(V);
        if (Underlying != V)
          dbgs() << "        - underlying: " << *Underlying << '\n';
        else
          dbgs() << "        - underlying: (no change)\n";
      };

      auto dumpSelect = [&](StringRef Which, Value *V) {
        if (auto *SI = dyn_cast<SelectInst>(V)) {
          dbgs() << "  - " << Which << " is Select: " << *SI << '\n';
          dbgs() << "      cond: " << *SI->getCondition() << '\n';
          Value *T = SI->getTrueValue();
          Value *Fv = SI->getFalseValue();
          dumpPtrForms("true", T);
          dumpPtrForms("false", Fv);
          dbgs() << "      trueIsAlloca=" << (hasAllocaUnderlyingObject(T) ? "true" : "false") << '\n';
          dbgs() << "      falseIsAlloca=" << (hasAllocaUnderlyingObject(Fv) ? "true" : "false") << '\n';
        }
      };

      dbgs() << "[AMDGPUVectorIdiom] Found memcpy: " << *MT << '\n'
             << "  in function: " << F.getName() << '\n'
             << "  - volatile=" << (MT->isVolatile() ? "true" : "false") << '\n'
             << "  - sameAS=" << (DstAS == SrcAS ? "true" : "false")
             << " (dstAS=" << DstAS << ", srcAS=" << SrcAS << ")\n"
             << "  - constLen=" << (isa<ConstantInt>(LenV) ? "true" : "false");
      if (auto *LCI = dyn_cast<ConstantInt>(LenV))
        dbgs() << " (N=" << LCI->getLimitedValue() << ")";
      dbgs() << '\n'
             << "  - srcIsSelect=" << (isa<SelectInst>(SrcV) ? "true" : "false")
             << '\n'
             << "  - dstIsSelect=" << (isa<SelectInst>(DstV) ? "true" : "false")
             << '\n'
             << "  - srcIsAlloca=" << (hasAllocaUnderlyingObject(SrcV) ? "true" : "false")
             << '\n'
             << "  - dstIsAlloca=" << (hasAllocaUnderlyingObject(DstV) ? "true" : "false")
             << '\n';

      dumpSelect("src", SrcV);
      dumpSelect("dst", DstV);
    });

    if (MT->isVolatile()) {
      LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Skip: memcpy is volatile\n");
      continue;
    }

    ConstantInt *LenCI = dyn_cast<ConstantInt>(MT->getLength());
    if (!LenCI) {
      LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Skip: memcpy length is not a "
                        << "constant integer\n");
      continue;
    }

    uint64_t N = LenCI->getLimitedValue();
    if (N == 0 || N > MaxBytes) {
      LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Skip: memcpy size out of range "
                        << "(N=" << N << ", MaxBytes=" << MaxBytes << ")\n");
      continue;
    }

    unsigned DstAS = cast<PointerType>(Dst->getType())->getAddressSpace();
    unsigned SrcAS = cast<PointerType>(Src->getType())->getAddressSpace();
    if (DstAS != SrcAS) {
      LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Skip: address space mismatch "
                        << "(dstAS=" << DstAS << ", srcAS=" << SrcAS << ")\n");
      continue;
    }

    // Check if we have select instructions and if their operands are alloca-based
    bool ShouldTransform = false;
    if (auto *Sel = dyn_cast<SelectInst>(Src)) {
      bool TrueIsAlloca = hasAllocaUnderlyingObject(Sel->getTrueValue());
      bool FalseIsAlloca = hasAllocaUnderlyingObject(Sel->getFalseValue());
      if (TrueIsAlloca || FalseIsAlloca) {
        ShouldTransform = true;
        Changed |= Impl.transformSelectMemcpySource(*MT, *Sel, DL, &DT, &AC);
      } else {
        LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Skip: select source operands "
                          << "are not alloca-based\n");
      }
      continue;
    }
    if (auto *Sel = dyn_cast<SelectInst>(Dst)) {
      bool TrueIsAlloca = hasAllocaUnderlyingObject(Sel->getTrueValue());
      bool FalseIsAlloca = hasAllocaUnderlyingObject(Sel->getFalseValue());
      if (TrueIsAlloca || FalseIsAlloca) {
        ShouldTransform = true;
        Changed |= Impl.transformSelectMemcpyDest(*MT, *Sel);
      } else {
        LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Skip: select destination operands "
                          << "are not alloca-based\n");
      }
      continue;
    }

    LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Skip: neither source nor "
                      << "destination is a select of pointers\n");
  }

  if (!Changed)
    return PreservedAnalyses::all();

  // Be conservative: preserve only analyses we know remain valid.
  PreservedAnalyses PA;
  PA.preserve<AssumptionAnalysis>();
  PA.preserve<TargetLibraryAnalysis>();
  PA.preserve<TargetIRAnalysis>();

  // If we didn't change the CFG, we can keep DT/LI/PostDT.
  if (!Impl.CFGChanged) {
    PA.preserve<DominatorTreeAnalysis>();
    PA.preserve<LoopAnalysis>();
    PA.preserve<PostDominatorTreeAnalysis>();
  }

  return PA;
}
