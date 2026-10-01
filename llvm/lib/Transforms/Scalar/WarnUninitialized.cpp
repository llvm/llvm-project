//===- WarnUninitialized.cpp - Warn about uninitialized loads -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// These passes classify scalar field loads from local records as initialized,
/// uninitialized, conditionally uninitialized, or unknown by walking MemorySSA.
/// The late pass also handles fresh heap allocations exposed by inlining. Calls
/// are barriers unless a bounded scan of a visible callee proves that it does
/// not modify the queried byte range.
///
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/WarnUninitialized.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/MathExtras.h"
#include <limits>
#include <optional>

using namespace llvm;

class llvm::WarnUninitializedDiagnosticState {
  SmallVector<WeakTrackingVH, 8> DiagnosedLoads;

public:
  bool contains(const Instruction *I) const {
    for (const WeakTrackingVH &VH : DiagnosedLoads)
      if (static_cast<Value *>(VH) == I)
        return true;
    return false;
  }

  void insert(Instruction *I) { DiagnosedLoads.emplace_back(I); }
};

std::shared_ptr<WarnUninitializedDiagnosticState>
llvm::createWarnUninitializedDiagnosticState() {
  return std::make_shared<WarnUninitializedDiagnosticState>();
}

namespace {

struct ByteRange {
  int64_t Offset;
  uint64_t Size;
};

static std::optional<int64_t> getEnd(ByteRange Range) {
  if (Range.Size > uint64_t(std::numeric_limits<int64_t>::max()))
    return std::nullopt;
  int64_t End;
  if (AddOverflow(Range.Offset, int64_t(Range.Size), End))
    return std::nullopt;
  return End;
}

static bool rangesOverlap(ByteRange LHS, ByteRange RHS) {
  std::optional<int64_t> LHSEnd = getEnd(LHS);
  std::optional<int64_t> RHSEnd = getEnd(RHS);
  return !LHSEnd || !RHSEnd || (LHS.Offset < *RHSEnd && RHS.Offset < *LHSEnd);
}

static bool rangeContains(ByteRange Outer, ByteRange Inner) {
  std::optional<int64_t> OuterEnd = getEnd(Outer);
  std::optional<int64_t> InnerEnd = getEnd(Inner);
  return OuterEnd && InnerEnd && Outer.Offset <= Inner.Offset &&
         *InnerEnd <= *OuterEnd;
}

static bool isMaskedReadModifyWrite(const LoadInst &LI) {
  const Value *Current = &LI;
  bool SawMask = false;

  while (Current->hasOneUse()) {
    const User *OnlyUser = *Current->user_begin();
    if (const auto *SI = dyn_cast<StoreInst>(OnlyUser))
      return SawMask && SI->isSimple() && SI->getValueOperand() == Current &&
             SI->getPointerOperand()->stripPointerCasts() ==
                 LI.getPointerOperand()->stripPointerCasts();

    const auto *BO = dyn_cast<BinaryOperator>(OnlyUser);
    if (!BO)
      return false;

    const Value *Other =
        BO->getOperand(0) == Current ? BO->getOperand(1) : BO->getOperand(0);
    if (BO->getOpcode() == Instruction::And) {
      const auto *Mask = dyn_cast<ConstantInt>(Other);
      if (SawMask || !Mask || Mask->isMinusOne())
        return false;
      SawMask = true;
    } else if (BO->getOpcode() != Instruction::Or || !SawMask) {
      return false;
    }
    Current = BO;
  }
  return false;
}

struct Query {
  const Value *Object;
  ByteRange Range;
  MemoryLocation Location;
};

struct RootAndOffset {
  const Value *Root;
  int64_t Offset;
};

enum class AnalysisStage { Early, Late };

enum class InitializationState {
  Initialized,
  Uninitialized,
  MaybeUninitialized,
  Unknown
};

struct SummaryKey {
  const Function *F;
  unsigned ArgNo;
  ByteRange Range;
};

class UninitializedUseAnalyzer {
  static constexpr unsigned MaxMemoryAccesses = 128;
  static constexpr unsigned MaxSummaryDepth = 8;
  static constexpr unsigned MaxSummaryInstructions = 1024;

  const DataLayout &DL;
  MemorySSA &MSSA;
  MemorySSAWalker *Walker;
  BatchAAResults BatchAA;
  const TargetLibraryInfo *TLI;
  AnalysisStage Stage;
  SmallVector<SummaryKey, 8> SummaryStack;

  std::optional<int64_t>
  getOffsetFromRootImpl(const Value *Ptr, const Value *Root,
                        SmallPtrSetImpl<const Value *> &Visited) const {
    if (!Ptr->getType()->isPointerTy() || !Visited.insert(Ptr).second)
      return std::nullopt;
    scope_exit RemoveVisited([&] { Visited.erase(Ptr); });

    int64_t OuterOffset = 0;
    const Value *Base = GetPointerBaseWithConstantOffset(Ptr, OuterOffset, DL);
    if (Base == Root)
      return OuterOffset;

    if (const auto *LI = dyn_cast<LoadInst>(Base)) {
      int64_t SpillOffset = 0;
      const auto *Spill = dyn_cast<AllocaInst>(GetPointerBaseWithConstantOffset(
          LI->getPointerOperand(), SpillOffset, DL));
      if (!Spill || SpillOffset != 0 ||
          !Spill->getAllocatedType()->isPointerTy())
        return std::nullopt;

      const StoreInst *Def = nullptr;
      for (const User *U : Spill->users()) {
        if (const auto *SI = dyn_cast<StoreInst>(U)) {
          int64_t Offset = 0;
          if (GetPointerBaseWithConstantOffset(SI->getPointerOperand(), Offset,
                                               DL) != Spill ||
              Offset != 0)
            return std::nullopt;
          if (Def)
            return std::nullopt;
          Def = SI;
          continue;
        }
        if (isa<LoadInst>(U))
          continue;
        const auto *I = dyn_cast<Instruction>(U);
        if (!I || (!I->isLifetimeStartOrEnd() && !I->isDroppable()))
          return std::nullopt;
      }

      if (!Def || Def->getParent() != LI->getParent() || !Def->comesBefore(LI))
        return std::nullopt;
      std::optional<int64_t> StoredOffset =
          getOffsetFromRootImpl(Def->getValueOperand(), Root, Visited);
      if (!StoredOffset)
        return std::nullopt;
      int64_t Result;
      if (AddOverflow(*StoredOffset, OuterOffset, Result))
        return std::nullopt;
      return Result;
    }

    auto MergeOffset = [&](auto Values) -> std::optional<int64_t> {
      std::optional<int64_t> Common;
      for (const Value *V : Values) {
        std::optional<int64_t> Offset = getOffsetFromRootImpl(V, Root, Visited);
        if (!Offset)
          return std::nullopt;
        if (Common && *Common != *Offset)
          return std::nullopt;
        Common = Offset;
      }
      if (!Common)
        return std::nullopt;
      int64_t Result;
      if (AddOverflow(*Common, OuterOffset, Result))
        return std::nullopt;
      return Result;
    };

    if (const auto *PN = dyn_cast<PHINode>(Base))
      return MergeOffset(PN->incoming_values());
    if (const auto *SI = dyn_cast<SelectInst>(Base))
      return MergeOffset(
          ArrayRef<const Value *>{SI->getTrueValue(), SI->getFalseValue()});
    return std::nullopt;
  }

  std::optional<int64_t> getOffsetFromRoot(const Value *Ptr,
                                           const Value *Root) const {
    SmallPtrSet<const Value *, 16> Visited;
    return getOffsetFromRootImpl(Ptr, Root, Visited);
  }

  std::optional<RootAndOffset>
  getRootAndOffsetImpl(const Value *Ptr,
                       SmallPtrSetImpl<const Value *> &Visited) {
    if (!Ptr->getType()->isPointerTy() || !Visited.insert(Ptr).second)
      return std::nullopt;
    scope_exit RemoveVisited([&] { Visited.erase(Ptr); });

    int64_t OuterOffset = 0;
    const Value *Base = GetPointerBaseWithConstantOffset(Ptr, OuterOffset, DL);
    if (isa<AllocaInst, CallBase>(Base))
      return RootAndOffset{Base, OuterOffset};

    const auto *LI = dyn_cast<LoadInst>(Base);
    if (!LI || !LI->isSimple())
      return std::nullopt;

    MemoryAccess *Use = MSSA.getMemoryAccess(LI);
    if (!Use)
      return std::nullopt;
    auto *Def =
        dyn_cast<MemoryDef>(Walker->getClobberingMemoryAccess(Use, BatchAA));
    if (!Def || !Def->getMemoryInst())
      return std::nullopt;
    const auto *SI = dyn_cast<StoreInst>(Def->getMemoryInst());
    if (!SI || !SI->getValueOperand()->getType()->isPointerTy() ||
        BatchAA.alias(MemoryLocation::get(LI), MemoryLocation::get(SI)) !=
            AliasResult::MustAlias)
      return std::nullopt;

    std::optional<RootAndOffset> Stored =
        getRootAndOffsetImpl(SI->getValueOperand(), Visited);
    if (!Stored)
      return std::nullopt;
    int64_t Offset;
    if (AddOverflow(Stored->Offset, OuterOffset, Offset))
      return std::nullopt;
    Stored->Offset = Offset;
    return Stored;
  }

  std::optional<RootAndOffset> getRootAndOffset(const Value *Ptr) {
    SmallPtrSet<const Value *, 16> Visited;
    return getRootAndOffsetImpl(Ptr, Visited);
  }

  bool isSeparateObject(const Value *Ptr, const Value *Root) const {
    const Value *Object = getUnderlyingObject(Ptr);
    return Object != Root &&
           (isa<AllocaInst>(Object) || isa<GlobalValue>(Object));
  }

  bool isBenignPointerSpill(const Value *Ptr) const {
    int64_t Offset = 0;
    const auto *AI =
        dyn_cast<AllocaInst>(GetPointerBaseWithConstantOffset(Ptr, Offset, DL));
    if (!AI || Offset != 0 || !AI->getAllocatedType()->isPointerTy())
      return false;

    for (const User *U : AI->users()) {
      if (const auto *SI = dyn_cast<StoreInst>(U)) {
        int64_t StoreOffset = 0;
        if (GetPointerBaseWithConstantOffset(SI->getPointerOperand(),
                                             StoreOffset, DL) != AI ||
            StoreOffset != 0)
          return false;
        continue;
      }
      if (isa<LoadInst>(U))
        continue;
      const auto *I = dyn_cast<Instruction>(U);
      if (!I || (!I->isLifetimeStartOrEnd() && !I->isDroppable()))
        return false;
    }
    return true;
  }

  bool hasNonCallEscape(const Value *Root) const {
    SmallVector<const Value *, 16> Worklist(1, Root);
    SmallPtrSet<const Value *, 16> Visited;
    while (!Worklist.empty()) {
      const Value *V = Worklist.pop_back_val();
      if (!Visited.insert(V).second)
        continue;

      for (const User *U : V->users()) {
        const auto *I = dyn_cast<Instruction>(U);
        if (!I)
          return true;
        if (isa<CallBase>(I) || I->isDroppable())
          continue;
        if (const auto *SI = dyn_cast<StoreInst>(I)) {
          if (SI->getValueOperand() == V)
            return true;
          continue;
        }
        if (isa<LoadInst, ICmpInst>(I))
          continue;
        if (I->getType()->isPointerTy() &&
            isa<GetElementPtrInst, BitCastInst, AddrSpaceCastInst, PHINode,
                SelectInst, FreezeInst>(I)) {
          Worklist.push_back(I);
          continue;
        }
        return true;
      }
    }
    return false;
  }

  bool writeDoesNotOverlap(const Value *Ptr, uint64_t Size, const Value *Root,
                           ByteRange Target) const {
    if (std::optional<int64_t> Offset = getOffsetFromRoot(Ptr, Root))
      return !rangesOverlap({*Offset, Size}, Target);
    return isSeparateObject(Ptr, Root);
  }

  std::optional<InitializationState> classifyWrite(const Value *Ptr,
                                                   uint64_t Size,
                                                   const Query &Q,
                                                   bool Initializes) const {
    std::optional<int64_t> Offset = getOffsetFromRoot(Ptr, Q.Object);
    if (!Offset) {
      if (isSeparateObject(Ptr, Q.Object))
        return std::nullopt;
      return InitializationState::Unknown;
    }

    ByteRange WriteRange{*Offset, Size};
    if (!rangesOverlap(WriteRange, Q.Range))
      return std::nullopt;
    if (Initializes && rangeContains(WriteRange, Q.Range))
      return InitializationState::Initialized;
    return InitializationState::Unknown;
  }

  bool calleeDoesNotModify(const Function &Callee, unsigned ArgNo,
                           ByteRange Target, unsigned Depth) {
    if (Callee.isDeclaration() || ArgNo >= Callee.arg_size() ||
        Depth >= MaxSummaryDepth ||
        Callee.getInstructionCount() > MaxSummaryInstructions)
      return false;

    for (const SummaryKey &Key : SummaryStack)
      if (Key.F == &Callee && Key.ArgNo == ArgNo &&
          Key.Range.Offset == Target.Offset && Key.Range.Size == Target.Size)
        return false;

    SummaryStack.push_back({&Callee, ArgNo, Target});
    scope_exit PopStack([&] { SummaryStack.pop_back(); });
    const Argument *Root = Callee.getArg(ArgNo);

    for (const Instruction &I : instructions(Callee)) {
      if (const auto *SI = dyn_cast<StoreInst>(&I)) {
        TypeSize Size = DL.getTypeStoreSize(SI->getValueOperand()->getType());
        if (Size.isScalable() ||
            !writeDoesNotOverlap(SI->getPointerOperand(), Size.getFixedValue(),
                                 Root, Target))
          return false;

        if (SI->getValueOperand()->getType()->isPointerTy() &&
            getOffsetFromRoot(SI->getValueOperand(), Root) &&
            !isBenignPointerSpill(SI->getPointerOperand()))
          return false;
        continue;
      }

      if (const auto *MI = dyn_cast<MemIntrinsic>(&I)) {
        const auto *Length = dyn_cast<ConstantInt>(MI->getLength());
        if (!Length || !writeDoesNotOverlap(
                           MI->getDest(), Length->getZExtValue(), Root, Target))
          return false;
        continue;
      }

      if (const auto *RMW = dyn_cast<AtomicRMWInst>(&I)) {
        TypeSize Size = DL.getTypeStoreSize(RMW->getValOperand()->getType());
        if (Size.isScalable() ||
            !writeDoesNotOverlap(RMW->getPointerOperand(), Size.getFixedValue(),
                                 Root, Target))
          return false;
        continue;
      }

      if (const auto *CX = dyn_cast<AtomicCmpXchgInst>(&I)) {
        TypeSize Size = DL.getTypeStoreSize(CX->getCompareOperand()->getType());
        if (Size.isScalable() ||
            !writeDoesNotOverlap(CX->getPointerOperand(), Size.getFixedValue(),
                                 Root, Target))
          return false;
        continue;
      }

      if (const auto *CB = dyn_cast<CallBase>(&I)) {
        if (CB->isLifetimeStartOrEnd() || CB->onlyReadsMemory())
          continue;

        const Function *Nested = CB->getCalledFunction();
        for (unsigned I = 0; I < CB->arg_size(); ++I) {
          const Value *Arg = CB->getArgOperand(I);
          if (!Arg->getType()->isPointerTy())
            continue;
          std::optional<int64_t> Offset = getOffsetFromRoot(Arg, Root);
          if (!Offset) {
            if (!isa<ConstantPointerNull>(Arg) && !isSeparateObject(Arg, Root))
              return false;
            continue;
          }
          int64_t RelativeOffset;
          if (!Nested || I >= Nested->arg_size() ||
              SubOverflow(Target.Offset, *Offset, RelativeOffset) ||
              !calleeDoesNotModify(*Nested, I, {RelativeOffset, Target.Size},
                                   Depth + 1))
            return false;
        }
        continue;
      }

      if (const auto *PTI = dyn_cast<PtrToIntInst>(&I)) {
        if (getOffsetFromRoot(PTI->getPointerOperand(), Root))
          return false;
      }

      if (const auto *RI = dyn_cast<ReturnInst>(&I)) {
        const Value *ReturnValue = RI->getReturnValue();
        if (ReturnValue && ReturnValue->getType()->isPointerTy() &&
            getOffsetFromRoot(ReturnValue, Root))
          return false;
      }

      if (I.mayWriteToMemory())
        return false;
    }
    return true;
  }

  bool callDoesNotModify(const CallBase &CB, const Query &Q) {
    if (CB.onlyReadsMemory())
      return true;
    const Function *Callee = CB.getCalledFunction();
    if (!Callee)
      return false;

    bool FoundObjectArgument = false;
    for (unsigned I = 0; I < CB.arg_size(); ++I) {
      const Value *Arg = CB.getArgOperand(I);
      if (!Arg->getType()->isPointerTy())
        continue;
      std::optional<int64_t> ArgOffset = getOffsetFromRoot(Arg, Q.Object);
      if (!ArgOffset)
        continue;
      FoundObjectArgument = true;
      int64_t RelativeOffset;
      if (I >= Callee->arg_size() ||
          SubOverflow(Q.Range.Offset, *ArgOffset, RelativeOffset) ||
          !calleeDoesNotModify(*Callee, I, {RelativeOffset, Q.Range.Size}, 0))
        return false;
    }
    return FoundObjectArgument;
  }

  std::optional<Query> getMemcpySourceQuery(const MemCpyInst &Copy,
                                            const Query &Q) const {
    const auto *Length = dyn_cast<ConstantInt>(Copy.getLength());
    std::optional<int64_t> DestOffset =
        getOffsetFromRoot(Copy.getDest(), Q.Object);
    if (!Length || !DestOffset ||
        !rangeContains({*DestOffset, Length->getZExtValue()}, Q.Range))
      return std::nullopt;

    int64_t OffsetInCopy;
    if (SubOverflow(Q.Range.Offset, *DestOffset, OffsetInCopy))
      return std::nullopt;

    int64_t SourceBaseOffset = 0;
    auto *SourceObject = dyn_cast<AllocaInst>(GetPointerBaseWithConstantOffset(
        Copy.getSource(), SourceBaseOffset, DL));
    if (!SourceObject || !SourceObject->isStaticAlloca() ||
        SourceObject->isArrayAllocation() ||
        !SourceObject->getAllocatedType()->isStructTy() ||
        hasNonCallEscape(SourceObject))
      return std::nullopt;

    int64_t SourceOffset;
    if (AddOverflow(SourceBaseOffset, OffsetInCopy, SourceOffset))
      return std::nullopt;
    TypeSize ObjectSize = DL.getTypeAllocSize(SourceObject->getAllocatedType());
    if (ObjectSize.isScalable() ||
        !rangeContains({0, ObjectSize.getFixedValue()},
                       {SourceOffset, Q.Range.Size}))
      return std::nullopt;

    return Query{SourceObject,
                 {SourceOffset, Q.Range.Size},
                 MemoryLocation(SourceObject, ObjectSize)};
  }

  std::optional<InitializationState> getMemoryDefState(const Instruction &I,
                                                       const Query &Q) {
    if (I.isLifetimeStartOrEnd())
      return std::nullopt;

    if (const auto *SI = dyn_cast<StoreInst>(&I)) {
      TypeSize Size = DL.getTypeStoreSize(SI->getValueOperand()->getType());
      if (Size.isScalable())
        return InitializationState::Unknown;
      return classifyWrite(SI->getPointerOperand(), Size.getFixedValue(), Q,
                           /*Initializes=*/true);
    }

    if (const auto *MI = dyn_cast<MemIntrinsic>(&I)) {
      const auto *Length = dyn_cast<ConstantInt>(MI->getLength());
      if (!Length)
        return InitializationState::Unknown;
      return classifyWrite(MI->getDest(), Length->getZExtValue(), Q,
                           /*Initializes=*/isa<MemSetInst>(MI));
    }

    if (const auto *CB = dyn_cast<CallBase>(&I)) {
      if (callDoesNotModify(*CB, Q))
        return std::nullopt;
      return InitializationState::Unknown;
    }

    return InitializationState::Unknown;
  }

  InitializationState
  getInitializationState(MemoryAccess *Access, const Query &Q,
                         SmallPtrSetImpl<MemoryAccess *> &Active,
                         unsigned &NumAccesses) {
    if (++NumAccesses > MaxMemoryAccesses || !Active.insert(Access).second)
      return InitializationState::Unknown;
    scope_exit RemoveActive([&] { Active.erase(Access); });

    if (MSSA.isLiveOnEntryDef(Access))
      return InitializationState::Uninitialized;

    if (auto *Phi = dyn_cast<MemoryPhi>(Access)) {
      if (Phi->getNumIncomingValues() == 0)
        return InitializationState::Unknown;
      std::optional<InitializationState> Merged;
      for (unsigned I = 0; I < Phi->getNumIncomingValues(); ++I) {
        MemoryAccess *Incoming = Phi->getIncomingValue(I);
        MemoryAccess *Clobber =
            Walker->getClobberingMemoryAccess(Incoming, Q.Location, BatchAA);
        InitializationState State =
            getInitializationState(Clobber, Q, Active, NumAccesses);
        if (State == InitializationState::Unknown)
          return State;
        if (!Merged)
          Merged = State;
        else if (*Merged != State)
          Merged = InitializationState::MaybeUninitialized;
      }
      return *Merged;
    }

    auto *Def = dyn_cast<MemoryDef>(Access);
    if (!Def)
      return InitializationState::Unknown;

    if (Def->getMemoryInst() == Q.Object)
      return InitializationState::Uninitialized;

    if (const auto *Copy = dyn_cast<MemCpyInst>(Def->getMemoryInst())) {
      if (std::optional<Query> Source = getMemcpySourceQuery(*Copy, Q)) {
        MemoryAccess *SourceClobber = Walker->getClobberingMemoryAccess(
            Def->getDefiningAccess(), Source->Location, BatchAA);
        return getInitializationState(SourceClobber, *Source, Active,
                                      NumAccesses);
      }
    }

    std::optional<InitializationState> State =
        getMemoryDefState(*Def->getMemoryInst(), Q);
    if (State)
      return *State;

    MemoryAccess *Clobber = Walker->getClobberingMemoryAccess(
        Def->getDefiningAccess(), Q.Location, BatchAA);
    return getInitializationState(Clobber, Q, Active, NumAccesses);
  }

public:
  UninitializedUseAnalyzer(Function &F, AAResults &AA, MemorySSA &MSSA,
                           const TargetLibraryInfo *TLI, AnalysisStage Stage)
      : DL(F.getDataLayout()), MSSA(MSSA), Walker(MSSA.getWalker()),
        BatchAA(AA), TLI(TLI), Stage(Stage) {}

  InitializationState getInitializationState(LoadInst &LI) {
    if (!LI.isSimple() ||
        !(LI.getType()->isIntegerTy() || LI.getType()->isFloatingPointTy() ||
          LI.getType()->isPointerTy()))
      return InitializationState::Unknown;

    if (Stage == AnalysisStage::Early && isMaskedReadModifyWrite(LI))
      return InitializationState::Unknown;

    TypeSize Size = DL.getTypeStoreSize(LI.getType());
    if (Size.isScalable())
      return InitializationState::Unknown;

    const Value *Object;
    int64_t Offset;
    if (Stage == AnalysisStage::Early) {
      const Value *AccessPtr = LI.getPointerOperand();
      if (!isa<GetElementPtrInst>(AccessPtr))
        return InitializationState::Unknown;

      auto *AI = dyn_cast<AllocaInst>(
          GetPointerBaseWithConstantOffset(AccessPtr, Offset, DL));
      if (!AI || !AI->isStaticAlloca() || AI->isArrayAllocation() ||
          !AI->getAllocatedType()->isStructTy() || hasNonCallEscape(AI))
        return InitializationState::Unknown;
      Object = AI;
    } else {
      if (LI.getDebugLoc() && !LI.getDebugLoc().getInlinedAt())
        return InitializationState::Unknown;

      std::optional<RootAndOffset> Root =
          getRootAndOffset(LI.getPointerOperand());
      if (!Root)
        return InitializationState::Unknown;
      Object = Root->Root;
      Offset = Root->Offset;

      uint64_t ObjectSize;
      if (const auto *AI = dyn_cast<AllocaInst>(Object)) {
        TypeSize Size = DL.getTypeAllocSize(AI->getAllocatedType());
        if (!AI->isStaticAlloca() || AI->isArrayAllocation() ||
            !AI->getAllocatedType()->isStructTy() || Size.isScalable() ||
            hasNonCallEscape(AI))
          return InitializationState::Unknown;
        ObjectSize = Size.getFixedValue();
      } else {
        const auto *Alloc = dyn_cast<CallBase>(Object);
        if (!Alloc || !TLI ||
            !isa_and_nonnull<UndefValue>(
                getInitialValueOfAllocation(Alloc, TLI, LI.getType())))
          return InitializationState::Unknown;
        std::optional<APInt> Size = getAllocSize(Alloc, TLI);
        std::optional<uint64_t> FixedSize =
            Size ? Size->tryZExtValue() : std::nullopt;
        if (!FixedSize)
          return InitializationState::Unknown;
        ObjectSize = *FixedSize;
      }

      if (!rangeContains({0, ObjectSize}, {Offset, Size.getFixedValue()}))
        return InitializationState::Unknown;
    }

    MemoryAccess *Use = MSSA.getMemoryAccess(&LI);
    if (!Use)
      return InitializationState::Unknown;
    Query Q{Object, {Offset, Size.getFixedValue()}, MemoryLocation::get(&LI)};
    MemoryAccess *Clobber = Walker->getClobberingMemoryAccess(Use, BatchAA);
    SmallPtrSet<MemoryAccess *, 16> Active;
    unsigned NumAccesses = 0;
    return getInitializationState(Clobber, Q, Active, NumAccesses);
  }
};

} // namespace

static PreservedAnalyses
runUninitializedAnalysis(Function &F, FunctionAnalysisManager &AM,
                         AnalysisStage Stage,
                         WarnUninitializedDiagnosticState *DiagnosticState) {
  if (F.isDeclaration())
    return PreservedAnalyses::all();

  AAResults &AA = AM.getResult<AAManager>(F);
  DominatorTree &DT = AM.getResult<DominatorTreeAnalysis>(F);
  MemorySSA &MSSA = AM.getResult<MemorySSAAnalysis>(F).getMSSA();
  const TargetLibraryInfo *TLI = Stage == AnalysisStage::Late
                                     ? &AM.getResult<TargetLibraryAnalysis>(F)
                                     : nullptr;
  UninitializedUseAnalyzer Analyzer(F, AA, MSSA, TLI, Stage);
  for (BasicBlock &BB : F) {
    if (!DT.isReachableFromEntry(&BB))
      continue;
    for (Instruction &I : BB) {
      auto *LI = dyn_cast<LoadInst>(&I);
      // IR generation and optimization may introduce loads that only move an
      // indeterminate representation without using its value.
      if (!LI || (DiagnosticState && DiagnosticState->contains(LI)) ||
          !programUndefinedIfUndefOrPoison(LI))
        continue;
      InitializationState State = Analyzer.getInitializationState(*LI);
      if (State == InitializationState::Uninitialized ||
          State == InitializationState::MaybeUninitialized) {
        if (DiagnosticState)
          DiagnosticState->insert(LI);
        F.getContext().diagnose(DiagnosticInfoUninitialized(
            LI, State == InitializationState::MaybeUninitialized));
      }
    }
  }

  return PreservedAnalyses::all();
}

PreservedAnalyses WarnUninitializedEarlyPass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {
  return runUninitializedAnalysis(F, AM, AnalysisStage::Early, State.get());
}

PreservedAnalyses WarnUninitializedLatePass::run(Function &F,
                                                 FunctionAnalysisManager &AM) {
  return runUninitializedAnalysis(F, AM, AnalysisStage::Late, State.get());
}
