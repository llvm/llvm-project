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
#include "llvm/ADT/SmallPtrSet.h"
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
#include "llvm/IR/ValueHandle.h"
#include <atomic>
#include <cstdlib>

using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "amdgpu-vector-idiom"

namespace {

static cl::opt<bool>
    AMDGPUVectorIdiomEnable("amdgpu-vector-idiom-enable",
                            cl::desc("Enable pass AMDGPUVectorIdiom"),
                            cl::init(true));

// Static counter to track transformations performed across all instances
static std::atomic<unsigned> TransformationCounter{0};

// Get maximum transformations from environment variable
static unsigned getMaxTransformationsFromEnv() {
  const char *envVar = std::getenv("AMDGPU_VECTOR_IDIOM_MAX_TRANSFORMATIONS");
  if (!envVar)
    return 0; // Default: unlimited
  
  char *endPtr;
  unsigned long value = std::strtoul(envVar, &endPtr, 10);
  
  // Check for conversion errors
  if (endPtr == envVar || *endPtr != '\0') {
    LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Invalid AMDGPU_VECTOR_IDIOM_MAX_TRANSFORMATIONS value: " 
                      << envVar << ", using unlimited\n");
    return 0;
  }
  
  return static_cast<unsigned>(value);
}

// Helper function to check if transformations should be performed
static bool shouldPerformTransformation() {
  unsigned maxTransformations = getMaxTransformationsFromEnv();
  if (maxTransformations == 0)
    return true; // Unlimited transformations
  
  return TransformationCounter.load() < maxTransformations;
}

// Helper function to increment transformation counter
static void incrementTransformationCounter() {
  TransformationCounter.fetch_add(1);
}

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
  const DataLayout *DL;
  bool CFGChanged = false;

  AMDGPUVectorIdiomImpl(unsigned MaxBytes, const DataLayout *DL) : MaxBytes(MaxBytes), DL(DL) {}

  // Returns true if the given intrinsic is an allowed lifetime marker.
  static bool isAllowedLifetimeIntrinsic(Instruction *I) {
    if (auto *II = dyn_cast<IntrinsicInst>(I)) {
      return II->getIntrinsicID() == Intrinsic::lifetime_start ||
             II->getIntrinsicID() == Intrinsic::lifetime_end;
    }
    return false;
  }

  // Explore pointer casts/GEPs reachable from BasePtr, collecting all
  // derived pointers. This is a small, bounded exploration since we only
  // follow casts/GEPs.
  static void collectDerivedPointers(Value *BasePtr,
                                     SmallVectorImpl<Value *> &Derived) {
    SmallVector<Value *, 16> Worklist;
    SmallPtrSet<Value *, 32> Visited;
    Worklist.push_back(BasePtr);

    while (!Worklist.empty()) {
      Value *Cur = Worklist.pop_back_val();
      if (!Visited.insert(Cur).second)
        continue;
      Derived.push_back(Cur);

      for (User *U : Cur->users()) {
        if (auto *BC = dyn_cast<BitCastInst>(U)) {
          Worklist.push_back(BC);
          continue;
        }
        if (auto *ASC = dyn_cast<AddrSpaceCastInst>(U)) {
          Worklist.push_back(ASC);
          continue;
        }
        if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
          // Only consider in-bounds GEPs to be conservative.
          if (GEP->isInBounds())
            Worklist.push_back(GEP);
          continue;
        }
      }
    }
  }

  // Attempts to find a simple memcpy chain where this memcpy (Producer) writes
  // to a temporary (TmpPtr), and a later memcpy (Consumer) immediately copies
  // from the same temporary to some final destination. The chain is considered
  // simple if all uses of TmpPtr (and its derived bitcasts/GEPs) are limited to
  // exactly these two memcpy operations (the Producer writing TmpPtr, and a
  // single Consumer reading TmpPtr), plus optional lifetime intrinsics and
  // further pointer casts/GEPs. If found and Producer dominates Consumer, the
  // Consumer is returned. Otherwise returns null.
  MemCpyInst *findMemcpyChainConsumer(MemCpyInst &Producer, Value *TmpPtr,
                                      uint64_t N,
                                      const DominatorTree *DT) {
    Value *Base = TmpPtr->stripPointerCasts();
    SmallVector<Value *, 16> Ptrs;
    collectDerivedPointers(Base, Ptrs);

    SmallVector<MemCpyInst *, 2> MemCpyUsers;
    for (Value *P : Ptrs) {
      for (User *U : P->users()) {
        if (auto *I = dyn_cast<Instruction>(U)) {
          if (isAllowedLifetimeIntrinsic(I))
            continue;
        }

        if (auto *MC = dyn_cast<MemCpyInst>(U)) {
          MemCpyUsers.push_back(MC);
          continue;
        }

        if (isa<BitCastInst>(U) || isa<AddrSpaceCastInst>(U) || isa<GetElementPtrInst>(U))
          continue; // Covered by Derived pointers

        // Any other use (loads/stores/calls/etc.) makes the chain non-simple.
        return nullptr;
      }
    }

    // We expect exactly two memcpys in the simple chain: the producer 'Producer'
    // that writes TmpPtr, and a consumer that reads TmpPtr.
    MemCpyInst *Consumer = nullptr;
    for (MemCpyInst *MC : MemCpyUsers) {
      // Length must be constant and match N to be a simple forward copy.
      auto *LenCI = dyn_cast<ConstantInt>(MC->getLength());
      if (!LenCI || LenCI->getLimitedValue() != N)
        return nullptr;

      Value *SrcMC = MC->getRawSource();
      Value *DstMC = MC->getRawDest();

      bool SrcFromTmp = SrcMC->stripPointerCasts() == Base;
      bool DstIsTmp = DstMC->stripPointerCasts() == Base;

      if (MC == &Producer) {
        // Producer must be the one writing to TmpPtr.
        if (!DstIsTmp)
          return nullptr;
        continue;
      }

      // Any other memcpy must be the consumer reading from TmpPtr.
      if (!SrcFromTmp)
        return nullptr;

      if (Consumer)
        return nullptr; // More than one consumer
      Consumer = MC;
    }

    if (!Consumer)
      return nullptr;

    // Producer must dominate Consumer so that values computed at Producer
    // (loads/select) can be used at the Consumer insertion point.
    if (DT && !DT->dominates(&Producer, Consumer))
      return nullptr;

    return Consumer;
  }

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
    
    if (!shouldPerformTransformation()) {
      unsigned maxTransformations = getMaxTransformationsFromEnv();
      LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Skip: transformation limit reached ("
                        << TransformationCounter.load() << "/" 
                        << maxTransformations << ")\n");
      return false;
    }
    
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
      incrementTransformationCounter();
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
      // First, check if this memcpy writes to a temporary that is immediately
      // copied again by a following memcpy to the final destination. If so,
      // form the value at the current location (to preserve load timing) and
      // emit a single store at the consumer location, erasing both memcpys.
      if (hasAllocaUnderlyingObject(Dst)) {
        if (MemCpyInst *Consumer =
                findMemcpyChainConsumer(MT, Dst, N, DT)) {
          Align ConsumerDstAlign =
              MaybeAlign(Consumer->getDestAlign()).valueOrOne();
          Align MinAlign = std::min(AlignAB, ConsumerDstAlign);

          LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Folding memcpy chain: "
                            << "memcpy(tmp <- select), memcpy(dst <- tmp). "
                            << "Emitting value-select and single store to final dst; "
                            << "N=" << N << " minAlign=" << MinAlign.value()
                            << '\n');

          // Compute the selected value at the original memcpy to preserve
          // the timing of the loads from A/B.
          Type *Ty = getIntOrVecTypeForSize(N, B.getContext(), MinAlign);
          LoadInst *LA = B.CreateAlignedLoad(Ty, A, MinAlign);
          LoadInst *LB = B.CreateAlignedLoad(Ty, Bv, MinAlign);
          Value *V = B.CreateSelect(Sel.getCondition(), LA, LB);

          // Insert the final store right before the consumer memcpy.
          IRBuilder<> BC(Consumer);
          (void)BC.CreateAlignedStore(V, Consumer->getRawDest(),
                                      ConsumerDstAlign);

          LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Erasing memcpy chain: \n  - "
                            << MT << "\n  - " << *Consumer << '\n');

          incrementTransformationCounter();
          Consumer->eraseFromParent();
          MT.eraseFromParent();
          return true;
        }
      }

      // No chain detected. Do the normal value-level select and store directly
      // to the memcpy destination.
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
      incrementTransformationCounter();
      MT.eraseFromParent();
      return true;
    }

    LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Falling back to CFG split for "
                      << "memcpy(select-src); speculation unsafe\n");
    splitCFGForMemcpy(MT, Sel.getCondition(), A, Bv, true);
    incrementTransformationCounter();
    LLVM_DEBUG(
        dbgs()
        << "[AMDGPUVectorIdiom] Rewrote memcpy(select-src) by CFG split\n");
    return true;
  }

  // Rewrites memcpy when the destination is a select of pointers. To avoid
  // speculative stores, always splits the CFG and emits a memcpy per branch.
  // Assumptions mirror the source case.
  bool transformSelectMemcpyDest(MemCpyInst &MT, SelectInst &Sel) {
    if (!shouldPerformTransformation()) {
      unsigned maxTransformations = getMaxTransformationsFromEnv();
      LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Skip: transformation limit reached ("
                        << TransformationCounter.load() << "/" 
                        << maxTransformations << ")\n");
      return false;
    }
    
    Value *DA = Sel.getTrueValue();
    Value *DB = Sel.getFalseValue();
    LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Rewriting memcpy(select-dst) via "
                      << "CFG split to avoid speculative stores: " << MT
                      << '\n');

    splitCFGForMemcpy(MT, Sel.getCondition(), DA, DB, false);
    incrementTransformationCounter();
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

  // Accumulator vectorization methods
  
  // Represents a group of consecutive memory operations that can be vectorized
  struct AccumulatorGroup {
    SmallVector<LoadInst*, 4> Loads;
    SmallVector<StoreInst*, 4> Stores;
    Value *BasePtr;
    uint64_t StartOffset;
    uint64_t EndOffset;
    Align MinAlign;
    
    AccumulatorGroup(Value *Ptr, uint64_t Start, uint64_t End, Align Align)
        : BasePtr(Ptr), StartOffset(Start), EndOffset(End), MinAlign(Align) {}
  };
  
  // Check if a load/store instruction is part of an accumulator pattern
  bool isAccumulatorAccess(Instruction *I, Value *&BasePtr, uint64_t &Offset) {
    if (auto *LI = dyn_cast<LoadInst>(I)) {
      if (!LI->getType()->isFloatTy())
        return false;
      
      Value *Ptr = LI->getPointerOperand();
      if (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
        // Check for GEPs with multiple indices where the last index is constant
        if (GEP->getNumIndices() >= 1 && isa<ConstantInt>(GEP->getOperand(GEP->getNumOperands() - 1))) {
          BasePtr = GEP->getPointerOperand();
          // Check if this is a byte offset GEP (i8 type)
          if (GEP->getSourceElementType()->isIntegerTy(8)) {
            // Byte offset GEP - use the offset directly
            Offset = cast<ConstantInt>(GEP->getOperand(GEP->getNumOperands() - 1))->getZExtValue();
          } else {
            // Array index GEP - multiply by element size
            Offset = cast<ConstantInt>(GEP->getOperand(GEP->getNumOperands() - 1))->getZExtValue() * 4; // float = 4 bytes
          }
          return true;
        }
      } else {
        // Direct load from base pointer (offset 0)
        BasePtr = Ptr;
        Offset = 0;
        return true;
      }
    } else if (auto *SI = dyn_cast<StoreInst>(I)) {
      if (!SI->getValueOperand()->getType()->isFloatTy())
        return false;
      
      Value *Ptr = SI->getPointerOperand();
      if (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
        // Check for GEPs with multiple indices where the last index is constant
        if (GEP->getNumIndices() >= 1 && isa<ConstantInt>(GEP->getOperand(GEP->getNumOperands() - 1))) {
          BasePtr = GEP->getPointerOperand();
          // Check if this is a byte offset GEP (i8 type)
          if (GEP->getSourceElementType()->isIntegerTy(8)) {
            // Byte offset GEP - use the offset directly
            Offset = cast<ConstantInt>(GEP->getOperand(GEP->getNumOperands() - 1))->getZExtValue();
          } else {
            // Array index GEP - multiply by element size
            Offset = cast<ConstantInt>(GEP->getOperand(GEP->getNumOperands() - 1))->getZExtValue() * 4; // float = 4 bytes
          }
          return true;
        }
      } else {
        // Direct store to base pointer (offset 0)
        BasePtr = Ptr;
        Offset = 0;
        return true;
      }
    }
    return false;
  }
  
  // Normalize base pointers across different instructions to find common groups
  void normalizeBasePointers(SmallVectorImpl<std::pair<uint64_t, Instruction*>> &Ops, 
                           DenseMap<Value*, SmallVector<std::pair<uint64_t, Instruction*>, 8>> &BaseToOps) {
    // Find the most common GEP base that is used for byte-offset GEPs
    DenseMap<Value*, unsigned> BaseFreq;
    Value *CommonGEPBase = nullptr;
    unsigned MaxFreq = 0;
    
    for (auto &Op : Ops) {
      Value *BasePtr;
      uint64_t Offset;
      if (isAccumulatorAccess(Op.second, BasePtr, Offset)) {
        // Look for GEP bases that are used in byte-offset patterns
        bool HasByteOffsetUsers = false;
        for (User *U : BasePtr->users()) {
          if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
            if (GEP->getSourceElementType()->isIntegerTy(8)) {
              HasByteOffsetUsers = true;
              break;
            }
          }
        }
        if (HasByteOffsetUsers) {
          BaseFreq[BasePtr]++;
          LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Found potential common base: " 
                            << *BasePtr << " (freq=" << BaseFreq[BasePtr] << ")\n");
          if (BaseFreq[BasePtr] > MaxFreq) {
            MaxFreq = BaseFreq[BasePtr];
            CommonGEPBase = BasePtr;
          }
        }
      }
    }
    
    LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Selected common GEP base: " 
                      << (CommonGEPBase ? CommonGEPBase->getName().str() : "nullptr") 
                      << " (freq=" << MaxFreq << ")\n");
    
    // Re-categorize all operations using the common GEP base
    for (auto &Op : Ops) {
      Value *BasePtr;
      uint64_t Offset;
      if (isAccumulatorAccess(Op.second, BasePtr, Offset)) {
        if (CommonGEPBase) {
          // All operations should use the common base, preserving their byte offsets
          LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Using common base: " 
                            << *Op.second << " -> offset " << Offset << "\n");
          BaseToOps[CommonGEPBase].push_back({Offset, Op.second});
        } else {
          // No common base found, use original grouping
          LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Using original base for: " 
                            << *Op.second << " -> base=" << *BasePtr << ", offset=" << Offset << "\n");
          BaseToOps[BasePtr].push_back({Offset, Op.second});
        }
      }
    }
  }
  
  // Collect accumulator groups from a basic block
  void collectAccumulatorGroups(BasicBlock *BB, SmallVectorImpl<AccumulatorGroup> &Groups) {
    SmallVector<Instruction*, 32> AccumulatorOps;
    
    // Find all accumulator-related load/store instructions
    for (Instruction &I : *BB) {
      Value *BasePtr;
      uint64_t Offset;
      if (isAccumulatorAccess(&I, BasePtr, Offset)) {
        LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Found accumulator access: " << I 
                          << " (base=" << *BasePtr << ", offset=" << Offset << ")\n");
        AccumulatorOps.push_back(&I);
      }
    }
    
    LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Found " << AccumulatorOps.size() 
                      << " accumulator operations in basic block\n");
    
    // Convert to operations with initial base/offset pairs
    SmallVector<std::pair<uint64_t, Instruction*>, 32> InitialOps;
    for (Instruction *I : AccumulatorOps) {
      Value *BasePtr;
      uint64_t Offset;
      if (isAccumulatorAccess(I, BasePtr, Offset)) {
        InitialOps.push_back({Offset, I});
      }
    }
    
    // Group operations by normalized base pointer
    DenseMap<Value*, SmallVector<std::pair<uint64_t, Instruction*>, 8>> BaseToOps;
    normalizeBasePointers(InitialOps, BaseToOps);
    
    // Create groups for each base pointer
    for (auto &Pair : BaseToOps) {
      Value *BasePtr = Pair.first;
      auto &Ops = Pair.second;
      
      // Separate loads and stores first
      SmallVector<std::pair<uint64_t, LoadInst*>, 8> Loads;
      SmallVector<std::pair<uint64_t, StoreInst*>, 8> Stores;
      
      for (auto &Op : Ops) {
        if (auto *LI = dyn_cast<LoadInst>(Op.second)) {
          Loads.push_back({Op.first, LI});
        } else if (auto *SI = dyn_cast<StoreInst>(Op.second)) {
          Stores.push_back({Op.first, SI});
        }
      }
      
      // Sort loads and stores by offset
      llvm::sort(Loads, [](const auto &A, const auto &B) {
        return A.first < B.first;
      });
      llvm::sort(Stores, [](const auto &A, const auto &B) {
        return A.first < B.first;
      });
      
      // Group consecutive loads
      for (size_t i = 0; i < Loads.size(); i++) {
        uint64_t StartOffset = Loads[i].first;
        
        // Look for consecutive loads starting from this offset
        SmallVector<LoadInst*, 4> ConsecutiveLoads;
        uint64_t ExpectedOffset = StartOffset;
        Align MinAlign = Align(4);
        
        for (size_t j = i; j < Loads.size() && ConsecutiveLoads.size() < 4; j++) {
          if (Loads[j].first == ExpectedOffset) {
            ConsecutiveLoads.push_back(Loads[j].second);
            MinAlign = std::min(MinAlign, Loads[j].second->getAlign());
            ExpectedOffset += 4;
          } else {
            break; // Not consecutive, stop here
          }
        }
        
        // Only create groups with at least 2 loads
        if (ConsecutiveLoads.size() >= 2) {
          AccumulatorGroup Group(BasePtr, StartOffset, ExpectedOffset, MinAlign);
          Group.Loads = ConsecutiveLoads;
          Groups.push_back(Group);
          
          // Skip the loads we've already grouped
          i += ConsecutiveLoads.size() - 1;
        }
      }
      
      // Group consecutive stores
      for (size_t i = 0; i < Stores.size(); i++) {
        uint64_t StartOffset = Stores[i].first;
        
        // Look for consecutive stores starting from this offset
        SmallVector<StoreInst*, 4> ConsecutiveStores;
        uint64_t ExpectedOffset = StartOffset;
        Align MinAlign = Align(4);
        
        for (size_t j = i; j < Stores.size() && ConsecutiveStores.size() < 4; j++) {
          if (Stores[j].first == ExpectedOffset) {
            ConsecutiveStores.push_back(Stores[j].second);
            MinAlign = std::min(MinAlign, Stores[j].second->getAlign());
            ExpectedOffset += 4;
          } else {
            break; // Not consecutive, stop here
          }
        }
        
        // Only create groups with at least 2 stores
        if (ConsecutiveStores.size() >= 2) {
          AccumulatorGroup Group(BasePtr, StartOffset, ExpectedOffset, MinAlign);
          Group.Stores = ConsecutiveStores;
          Groups.push_back(Group);
          
          // Skip the stores we've already grouped
          i += ConsecutiveStores.size() - 1;
        }
      }
    }
  }
  
  // Vectorize an accumulator group
  bool vectorizeAccumulatorGroup(AccumulatorGroup &Group, IRBuilder<> &B) {
    if (!shouldPerformTransformation()) {
      unsigned maxTransformations = getMaxTransformationsFromEnv();
      LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Skip: transformation limit reached ("
                        << TransformationCounter.load() << "/" 
                        << maxTransformations << ")\n");
      return false;
    }
    
    LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Vectorizing accumulator group with "
                      << Group.Loads.size() << " loads and " << Group.Stores.size() 
                      << " stores\n");
    
    bool Changed = false;
    
    // Vectorize loads
    if (Group.Loads.size() >= 2) {
      // Create vector load pointer accounting for start offset
      Value *LoadPtr = Group.BasePtr;
      if (Group.StartOffset > 0) {
        // Try to reuse an existing GEP if possible
        bool FoundExistingGEP = false;
        for (auto *LI : Group.Loads) {
          Value *Ptr = LI->getPointerOperand();
          if (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
            if (GEP->getPointerOperand() == Group.BasePtr &&
                GEP->getNumIndices() == 1 &&
                GEP->getSourceElementType()->isIntegerTy(8)) {
              if (auto *OffsetCI = dyn_cast<ConstantInt>(GEP->getOperand(1))) {
                if (OffsetCI->getZExtValue() == Group.StartOffset) {
                  LoadPtr = GEP;
                  FoundExistingGEP = true;
                  break;
                }
              }
            }
          }
        }
        if (!FoundExistingGEP) {
          LoadPtr = B.CreateGEP(B.getInt8Ty(), Group.BasePtr, B.getInt64(Group.StartOffset));
        }
      }

      // Find the earliest load to position the vector load before it
      LoadInst *EarliestLoad = Group.Loads[0];
      for (auto *LI : Group.Loads) {
        if (LI->comesBefore(EarliestLoad)) {
          EarliestLoad = LI;
        }
      }

      // Create vector load
      Type *VecType = FixedVectorType::get(Type::getFloatTy(B.getContext()), Group.Loads.size());
      // Use the alignment of the first load in the group (at StartOffset)
      Align LoadAlign = Group.MinAlign;
      if (Group.StartOffset == 0 && !Group.Loads.empty()) {
        LoadAlign = Group.Loads[0]->getAlign();
      }
      Value *VecLoad = B.CreateAlignedLoad(VecType, LoadPtr, LoadAlign);
      
      // Move GEP and vector load to just before the earliest load
      if (auto *LoadPtrInst = dyn_cast<Instruction>(LoadPtr)) {
        LoadPtrInst->moveBefore(EarliestLoad);
      }
      cast<Instruction>(VecLoad)->moveBefore(EarliestLoad);

      // Replace individual loads with extractelement
      for (size_t i = 0; i < Group.Loads.size(); i++) {
        Value *Extract = B.CreateExtractElement(VecLoad, B.getInt32(i));
        // Move extractelement right after the vector load
        cast<Instruction>(Extract)->moveAfter(cast<Instruction>(VecLoad));
        Group.Loads[i]->replaceAllUsesWith(Extract);
        if (!Group.Loads[i]->isTerminator()) {
          Group.Loads[i]->eraseFromParent();
          Changed = true;
        }
      }
    }
    
    // Vectorize stores
    if (Group.Stores.size() >= 2) {
      // Create vector store pointer accounting for start offset
      Value *StorePtr = Group.BasePtr;
      if (Group.StartOffset > 0) {
        // Try to reuse an existing GEP if possible
        bool FoundExistingGEP = false;
        for (auto *SI : Group.Stores) {
          Value *Ptr = SI->getPointerOperand();
          if (auto *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
            if (GEP->getPointerOperand() == Group.BasePtr &&
                GEP->getNumIndices() == 1 &&
                GEP->getSourceElementType()->isIntegerTy(8)) {
              if (auto *OffsetCI = dyn_cast<ConstantInt>(GEP->getOperand(1))) {
                if (OffsetCI->getZExtValue() == Group.StartOffset) {
                  StorePtr = GEP;
                  FoundExistingGEP = true;
                  break;
                }
              }
            }
          }
        }
        if (!FoundExistingGEP) {
          StorePtr = B.CreateGEP(B.getInt8Ty(), Group.BasePtr, B.getInt64(Group.StartOffset));
        }
      }

      // Find the latest store to position the vector store after it
      StoreInst *LatestStore = Group.Stores[0];
      for (auto *SI : Group.Stores) {
        if (LatestStore->comesBefore(SI)) {
          LatestStore = SI;
        }
      }

      // Collect store values
      SmallVector<Value*, 4> StoreValues;
      for (auto *SI : Group.Stores) {
        StoreValues.push_back(SI->getValueOperand());
      }

      // Move the StorePtr GEP if it was created, right after the latest store
      Instruction *PrevInst = LatestStore;
      if (auto *StorePtrInst = dyn_cast<Instruction>(StorePtr)) {
        StorePtrInst->moveAfter(LatestStore);
        PrevInst = StorePtrInst;
      }

      // Create vector from individual values
      Type *VecType = FixedVectorType::get(Type::getFloatTy(B.getContext()), StoreValues.size());
      Value *VecValue = UndefValue::get(VecType);
      for (size_t i = 0; i < StoreValues.size(); i++) {
        VecValue = B.CreateInsertElement(VecValue, StoreValues[i], B.getInt32(i));
        // Move insertElement after the previous instruction to maintain order
        if (auto *InsertInst = dyn_cast<Instruction>(VecValue)) {
          InsertInst->moveAfter(PrevInst);
          PrevInst = InsertInst;
        }
      }

      // Create vector store
      Instruction *VecStore = B.CreateAlignedStore(VecValue, StorePtr, Group.MinAlign);
      // Move vector store after the vector creation
      VecStore->moveAfter(PrevInst);

      // Remove individual stores (but not terminators)
      for (auto *SI : Group.Stores) {
        if (!SI->isTerminator()) {
          SI->eraseFromParent();
          Changed = true;
        }
      }
    }
    
    if (Changed) {
      incrementTransformationCounter();
      LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Successfully vectorized accumulator group\n");
    }
    
    return Changed;
  }
  
  // Main accumulator vectorization method
  bool vectorizeAccumulators(Function &F) {
    bool Changed = false;
    
    LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Starting accumulator vectorization for function "
                      << F.getName() << "\n");
    
    for (BasicBlock &BB : F) {
      SmallVector<AccumulatorGroup, 8> Groups;
      collectAccumulatorGroups(&BB, Groups);
      
      LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Found " << Groups.size() 
                        << " accumulator groups in basic block\n");
      
      // Process groups from the end of the basic block to avoid ordering issues
      // when instructions are erased
      for (auto it = Groups.rbegin(); it != Groups.rend(); ++it) {
        AccumulatorGroup &Group = *it;
        // Create IRBuilder at the end of the basic block to avoid issues with instruction erasure
        IRBuilder<> B(BB.getTerminator());
        Changed |= vectorizeAccumulatorGroup(Group, B);
      }
    }
    
    LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Accumulator vectorization completed for function "
                      << F.getName() << " (changed=" << Changed << ")\n");
    
    return Changed;
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

  LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Starting pass on function " << F.getName() << "\n");

  if (!AMDGPUVectorIdiomEnable) {
    LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Pass disabled, returning\n");
    return PreservedAnalyses::all();
  }

  LLVM_DEBUG({
    unsigned currentCount = TransformationCounter.load();
    unsigned maxTransformations = getMaxTransformationsFromEnv();
    if (maxTransformations > 0) {
      dbgs() << "[AMDGPUVectorIdiom] Starting pass on function " << F.getName() 
             << " (transformations: " << currentCount << "/" 
             << maxTransformations << ")\n";
    } else {
      dbgs() << "[AMDGPUVectorIdiom] Starting pass on function " << F.getName() 
             << " (transformations: " << currentCount << "/unlimited)\n";
    }
  });

  SmallVector<WeakTrackingVH, 8> Worklist;
  for (Instruction &I : instructions(F)) {
    if (auto *MC = dyn_cast<MemCpyInst>(&I))
      Worklist.emplace_back(MC);
  }

  bool Changed = false;
  AMDGPUVectorIdiomImpl Impl(MaxBytes, &DL);

  // First, try accumulator vectorization
  LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Calling accumulator vectorization for function "
                    << F.getName() << "\n");
  bool AccChanged = Impl.vectorizeAccumulators(F);
  Changed |= AccChanged;
  LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Accumulator vectorization result: " 
                    << (AccChanged ? "changed" : "no change") << "\n");

  for (WeakTrackingVH &WH : Worklist) {
    auto *MT = dyn_cast_or_null<MemCpyInst>(WH);
    if (!MT)
      continue; // Was deleted by a previous transform
    
    Value *Dst = MT->getRawDest();
    Value *Src = MT->getRawSource();
    
    // Add null checks for safety
    if (!Dst || !Src) {
      LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Skip: null dst or src\n");
      continue;
    }
    
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

    auto *DstPTy = dyn_cast<PointerType>(Dst->getType());
    auto *SrcPTy = dyn_cast<PointerType>(Src->getType());
    if (!DstPTy || !SrcPTy) {
      LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Skip: non-pointer dst or src\n");
      continue;
    }
    
    unsigned DstAS = DstPTy->getAddressSpace();
    unsigned SrcAS = SrcPTy->getAddressSpace();
    if (DstAS != SrcAS) {
      LLVM_DEBUG(dbgs() << "[AMDGPUVectorIdiom] Skip: address space mismatch "
                        << "(dstAS=" << DstAS << ", srcAS=" << SrcAS << ")\n");
      continue;
    }

    // Check if we have select instructions and if their operands are alloca-based
    if (auto *Sel = dyn_cast<SelectInst>(Src)) {
      bool TrueIsAlloca = hasAllocaUnderlyingObject(Sel->getTrueValue());
      bool FalseIsAlloca = hasAllocaUnderlyingObject(Sel->getFalseValue());
      if (TrueIsAlloca || FalseIsAlloca) {
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
