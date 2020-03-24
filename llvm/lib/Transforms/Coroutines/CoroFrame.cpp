//===- CoroFrame.cpp - Builds and manipulates coroutine frame -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file contains classes used to discover if for a particular value
// there from sue to definition that crosses a suspend block.
//
// Using the information discovered we form a Coroutine Frame structure to
// contain those values. All uses of those values are replaced with appropriate
// GEP + load from the coroutine frame. At the point of the definition we spill
// the value into the coroutine frame.
//
// TODO: pack values tightly using liveness info.
//===----------------------------------------------------------------------===//

#include "CoroInternal.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/PtrUseVisitor.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/circular_raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include <algorithm>

using namespace llvm;

// The "coro-suspend-crossing" flag is very noisy. There is another debug type,
// "coro-frame", which results in leaner debug spew.
#define DEBUG_TYPE "coro-suspend-crossing"

enum { SmallVectorThreshold = 32 };

// Provides two way mapping between the blocks and numbers.
namespace {
class BlockToIndexMapping {
  SmallVector<BasicBlock *, SmallVectorThreshold> V;

public:
  size_t size() const { return V.size(); }

  BlockToIndexMapping(Function &F) {
    for (BasicBlock &BB : F)
      V.push_back(&BB);
    llvm::sort(V);
  }

  size_t blockToIndex(BasicBlock *BB) const {
    auto *I = llvm::lower_bound(V, BB);
    assert(I != V.end() && *I == BB && "BasicBlockNumberng: Unknown block");
    return I - V.begin();
  }

  BasicBlock *indexToBlock(unsigned Index) const { return V[Index]; }
};
} // end anonymous namespace

// The SuspendCrossingInfo maintains data that allows to answer a question
// whether given two BasicBlocks A and B there is a path from A to B that
// passes through a suspend point.
//
// For every basic block 'i' it maintains a BlockData that consists of:
//   Consumes:  a bit vector which contains a set of indices of blocks that can
//              reach block 'i'
//   Kills: a bit vector which contains a set of indices of blocks that can
//          reach block 'i', but one of the path will cross a suspend point
//   Suspend: a boolean indicating whether block 'i' contains a suspend point.
//   End: a boolean indicating whether block 'i' contains a coro.end intrinsic.
//
namespace {
struct SuspendCrossingInfo {
  BlockToIndexMapping Mapping;

  struct BlockData {
    BitVector Consumes;
    BitVector Kills;
    bool Suspend = false;
    bool End = false;
  };
  SmallVector<BlockData, SmallVectorThreshold> Block;

  iterator_range<succ_iterator> successors(BlockData const &BD) const {
    BasicBlock *BB = Mapping.indexToBlock(&BD - &Block[0]);
    return llvm::successors(BB);
  }

  BlockData &getBlockData(BasicBlock *BB) {
    return Block[Mapping.blockToIndex(BB)];
  }

  void dump() const;
  void dump(StringRef Label, BitVector const &BV) const;

  SuspendCrossingInfo(Function &F, coro::Shape &Shape);

  bool hasPathCrossingSuspendPoint(BasicBlock *DefBB, BasicBlock *UseBB) const {
    size_t const DefIndex = Mapping.blockToIndex(DefBB);
    size_t const UseIndex = Mapping.blockToIndex(UseBB);

    bool const Result = Block[UseIndex].Kills[DefIndex];
    LLVM_DEBUG(dbgs() << UseBB->getName() << " => " << DefBB->getName()
                      << " answer is " << Result << "\n");
    return Result;
  }

  bool isDefinitionAcrossSuspend(BasicBlock *DefBB, User *U) const {
    auto *I = cast<Instruction>(U);

    // We rewrote PHINodes, so that only the ones with exactly one incoming
    // value need to be analyzed.
    if (auto *PN = dyn_cast<PHINode>(I))
      if (PN->getNumIncomingValues() > 1)
        return false;

    BasicBlock *UseBB = I->getParent();

    // As a special case, treat uses by an llvm.coro.suspend.retcon
    // as if they were uses in the suspend's single predecessor: the
    // uses conceptually occur before the suspend.
    if (isa<CoroSuspendRetconInst>(I)) {
      UseBB = UseBB->getSinglePredecessor();
      assert(UseBB && "should have split coro.suspend into its own block");
    }

    return hasPathCrossingSuspendPoint(DefBB, UseBB);
  }

  bool isDefinitionAcrossSuspend(Argument &A, User *U) const {
    return isDefinitionAcrossSuspend(&A.getParent()->getEntryBlock(), U);
  }

  bool isDefinitionAcrossSuspend(Instruction &I, User *U) const {
    auto *DefBB = I.getParent();

    // As a special case, treat values produced by an llvm.coro.suspend.*
    // as if they were defined in the single successor: the uses
    // conceptually occur after the suspend.
    if (isa<AnyCoroSuspendInst>(I)) {
      DefBB = DefBB->getSingleSuccessor();
      assert(DefBB && "should have split coro.suspend into its own block");
    }

    return isDefinitionAcrossSuspend(DefBB, U);
  }
};
} // end anonymous namespace

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void SuspendCrossingInfo::dump(StringRef Label,
                                                BitVector const &BV) const {
  dbgs() << Label << ":";
  for (size_t I = 0, N = BV.size(); I < N; ++I)
    if (BV[I])
      dbgs() << " " << Mapping.indexToBlock(I)->getName();
  dbgs() << "\n";
}

LLVM_DUMP_METHOD void SuspendCrossingInfo::dump() const {
  for (size_t I = 0, N = Block.size(); I < N; ++I) {
    BasicBlock *const B = Mapping.indexToBlock(I);
    dbgs() << B->getName() << ":\n";
    dump("   Consumes", Block[I].Consumes);
    dump("      Kills", Block[I].Kills);
  }
  dbgs() << "\n";
}
#endif

SuspendCrossingInfo::SuspendCrossingInfo(Function &F, coro::Shape &Shape)
    : Mapping(F) {
  const size_t N = Mapping.size();
  Block.resize(N);

  // Initialize every block so that it consumes itself
  for (size_t I = 0; I < N; ++I) {
    auto &B = Block[I];
    B.Consumes.resize(N);
    B.Kills.resize(N);
    B.Consumes.set(I);
  }

  // Mark all CoroEnd Blocks. We do not propagate Kills beyond coro.ends as
  // the code beyond coro.end is reachable during initial invocation of the
  // coroutine.
  for (auto *CE : Shape.CoroEnds)
    getBlockData(CE->getParent()).End = true;

  // Mark all suspend blocks and indicate that they kill everything they
  // consume. Note, that crossing coro.save also requires a spill, as any code
  // between coro.save and coro.suspend may resume the coroutine and all of the
  // state needs to be saved by that time.
  auto markSuspendBlock = [&](IntrinsicInst *BarrierInst) {
    BasicBlock *SuspendBlock = BarrierInst->getParent();
    auto &B = getBlockData(SuspendBlock);
    B.Suspend = true;
    B.Kills |= B.Consumes;
  };
  for (auto *CSI : Shape.CoroSuspends) {
    markSuspendBlock(CSI);
    if (auto *Save = CSI->getCoroSave())
      markSuspendBlock(Save);
  }

  // Iterate propagating consumes and kills until they stop changing.
  int Iteration = 0;
  (void)Iteration;

  bool Changed;
  do {
    LLVM_DEBUG(dbgs() << "iteration " << ++Iteration);
    LLVM_DEBUG(dbgs() << "==============\n");

    Changed = false;
    for (size_t I = 0; I < N; ++I) {
      auto &B = Block[I];
      for (BasicBlock *SI : successors(B)) {

        auto SuccNo = Mapping.blockToIndex(SI);

        // Saved Consumes and Kills bitsets so that it is easy to see
        // if anything changed after propagation.
        auto &S = Block[SuccNo];
        auto SavedConsumes = S.Consumes;
        auto SavedKills = S.Kills;

        // Propagate Kills and Consumes from block B into its successor S.
        S.Consumes |= B.Consumes;
        S.Kills |= B.Kills;

        // If block B is a suspend block, it should propagate kills into the
        // its successor for every block B consumes.
        if (B.Suspend) {
          S.Kills |= B.Consumes;
        }
        if (S.Suspend) {
          // If block S is a suspend block, it should kill all of the blocks it
          // consumes.
          S.Kills |= S.Consumes;
        } else if (S.End) {
          // If block S is an end block, it should not propagate kills as the
          // blocks following coro.end() are reached during initial invocation
          // of the coroutine while all the data are still available on the
          // stack or in the registers.
          S.Kills.reset();
        } else {
          // This is reached when S block it not Suspend nor coro.end and it
          // need to make sure that it is not in the kill set.
          S.Kills.reset(SuccNo);
        }

        // See if anything changed.
        Changed |= (S.Kills != SavedKills) || (S.Consumes != SavedConsumes);

        if (S.Kills != SavedKills) {
          LLVM_DEBUG(dbgs() << "\nblock " << I << " follower " << SI->getName()
                            << "\n");
          LLVM_DEBUG(dump("S.Kills", S.Kills));
          LLVM_DEBUG(dump("SavedKills", SavedKills));
        }
        if (S.Consumes != SavedConsumes) {
          LLVM_DEBUG(dbgs() << "\nblock " << I << " follower " << SI << "\n");
          LLVM_DEBUG(dump("S.Consume", S.Consumes));
          LLVM_DEBUG(dump("SavedCons", SavedConsumes));
        }
      }
    }
  } while (Changed);
  LLVM_DEBUG(dump());
}

#undef DEBUG_TYPE // "coro-suspend-crossing"
#define DEBUG_TYPE "coro-frame"

// We build up the list of spills for every case where a use is separated
// from the definition by a suspend point.

static const unsigned InvalidFieldIndex = ~0U;

namespace {
class Spill {
  Value *Def = nullptr;
  Instruction *User = nullptr;
  unsigned FieldNo = InvalidFieldIndex;

public:
  Spill(Value *Def, llvm::User *U) : Def(Def), User(cast<Instruction>(U)) {}

  Value *def() const { return Def; }
  Instruction *user() const { return User; }
  BasicBlock *userBlock() const { return User->getParent(); }

  // Note that field index is stored in the first SpillEntry for a particular
  // definition. Subsequent mentions of a defintion do not have fieldNo
  // assigned. This works out fine as the users of Spills capture the info about
  // the definition the first time they encounter it. Consider refactoring
  // SpillInfo into two arrays to normalize the spill representation.
  unsigned fieldIndex() const {
    assert(FieldNo != InvalidFieldIndex && "Accessing unassigned field");
    return FieldNo;
  }
  void setFieldIndex(unsigned FieldNumber) {
    assert(FieldNo == InvalidFieldIndex && "Reassigning field number");
    FieldNo = FieldNumber;
  }
};
} // namespace

// Note that there may be more than one record with the same value of Def in
// the SpillInfo vector.
using SpillInfo = SmallVector<Spill, 8>;

#ifndef NDEBUG
static void dump(StringRef Title, SpillInfo const &Spills) {
  dbgs() << "------------- " << Title << "--------------\n";
  Value *CurrentValue = nullptr;
  for (auto const &E : Spills) {
    if (CurrentValue != E.def()) {
      CurrentValue = E.def();
      CurrentValue->dump();
    }
    dbgs() << "   user: ";
    E.user()->dump();
  }
}
#endif

namespace {
// We cannot rely solely on natural alignment of a type when building a
// coroutine frame and if the alignment specified on the Alloca instruction
// differs from the natural alignment of the alloca type we will need to insert
// padding.
struct PaddingCalculator {
  const DataLayout &DL;
  LLVMContext &Context;
  unsigned StructSize = 0;

  PaddingCalculator(LLVMContext &Context, DataLayout const &DL)
      : DL(DL), Context(Context) {}

  // Replicate the logic from IR/DataLayout.cpp to match field offset
  // computation for LLVM structs.
  void addType(Type *Ty) {
    unsigned TyAlign = DL.getABITypeAlignment(Ty);
    if ((StructSize & (TyAlign - 1)) != 0)
      StructSize = alignTo(StructSize, TyAlign);

    StructSize += DL.getTypeAllocSize(Ty); // Consume space for this data item.
  }

  void addTypes(SmallVectorImpl<Type *> const &Types) {
    for (auto *Ty : Types)
      addType(Ty);
  }

  unsigned computePadding(Type *Ty, unsigned ForcedAlignment) {
    unsigned TyAlign = DL.getABITypeAlignment(Ty);
    auto Natural = alignTo(StructSize, TyAlign);
    auto Forced = alignTo(StructSize, ForcedAlignment);

    // Return how many bytes of padding we need to insert.
    if (Natural != Forced)
      return std::max(Natural, Forced) - StructSize;

    // Rely on natural alignment.
    return 0;
  }

  // If padding required, return the padding field type to insert.
  ArrayType *getPaddingType(Type *Ty, unsigned ForcedAlignment) {
    if (auto Padding = computePadding(Ty, ForcedAlignment))
      return ArrayType::get(Type::getInt8Ty(Context), Padding);

    return nullptr;
  }
};
} // namespace

// Build a struct that will keep state for an active coroutine.
//   struct f.frame {
//     ResumeFnTy ResumeFnAddr;
//     ResumeFnTy DestroyFnAddr;
//     int ResumeIndex;
//     ... promise (if present) ...
//     ... spills ...
//   };
static StructType *buildFrameType(Function &F, coro::Shape &Shape,
                                  SpillInfo &Spills) {
  LLVMContext &C = F.getContext();
  const DataLayout &DL = F.getParent()->getDataLayout();
  PaddingCalculator Padder(C, DL);
  SmallString<32> Name(F.getName());
  Name.append(".Frame");
  StructType *FrameTy = StructType::create(C, Name);
  SmallVector<Type *, 8> Types;

  AllocaInst *PromiseAlloca = Shape.getPromiseAlloca();

  if (Shape.ABI == coro::ABI::Switch) {
    auto *FramePtrTy = FrameTy->getPointerTo();
    auto *FnTy = FunctionType::get(Type::getVoidTy(C), FramePtrTy,
                                   /*IsVarArg=*/false);
    auto *FnPtrTy = FnTy->getPointerTo();

    // Figure out how wide should be an integer type storing the suspend index.
    unsigned IndexBits = std::max(1U, Log2_64_Ceil(Shape.CoroSuspends.size()));
    Type *PromiseType = PromiseAlloca
                            ? PromiseAlloca->getType()->getElementType()
                            : Type::getInt1Ty(C);
    Type *IndexType = Type::getIntNTy(C, IndexBits);
    Types.push_back(FnPtrTy);
    Types.push_back(FnPtrTy);
    Types.push_back(PromiseType);
    Types.push_back(IndexType);
  } else {
    assert(PromiseAlloca == nullptr && "lowering doesn't support promises");
  }

  Value *CurrentDef = nullptr;

  Padder.addTypes(Types);

  // Create an entry for every spilled value.
  for (auto &S : Spills) {
    if (CurrentDef == S.def())
      continue;

    CurrentDef = S.def();
    // PromiseAlloca was already added to Types array earlier.
    if (CurrentDef == PromiseAlloca)
      continue;

    uint64_t Count = 1;
    Type *Ty = nullptr;
    if (auto *AI = dyn_cast<AllocaInst>(CurrentDef)) {
      Ty = AI->getAllocatedType();
      if (unsigned AllocaAlignment = AI->getAlignment()) {
        // If alignment is specified in alloca, see if we need to insert extra
        // padding.
        if (auto PaddingTy = Padder.getPaddingType(Ty, AllocaAlignment)) {
          Types.push_back(PaddingTy);
          Padder.addType(PaddingTy);
        }
      }
      if (auto *CI = dyn_cast<ConstantInt>(AI->getArraySize()))
        Count = CI->getValue().getZExtValue();
      else
        report_fatal_error("Coroutines cannot handle non static allocas yet");
    } else {
      Ty = CurrentDef->getType();
    }
    S.setFieldIndex(Types.size());
    if (Count == 1)
      Types.push_back(Ty);
    else
      Types.push_back(ArrayType::get(Ty, Count));
    Padder.addType(Ty);
  }
  FrameTy->setBody(Types);

  switch (Shape.ABI) {
  case coro::ABI::Switch:
    break;

  // Remember whether the frame is inline in the storage.
  case coro::ABI::Retcon:
  case coro::ABI::RetconOnce: {
    auto &Layout = F.getParent()->getDataLayout();
    auto Id = Shape.getRetconCoroId();
    Shape.RetconLowering.IsFrameInlineInStorage
      = (Layout.getTypeAllocSize(FrameTy) <= Id->getStorageSize() &&
         Layout.getABITypeAlignment(FrameTy) <= Id->getStorageAlignment());
    break;
  }
  }

  return FrameTy;
}

// We use a pointer use visitor to discover if there are any writes into an
// alloca that dominates CoroBegin. If that is the case, insertSpills will copy
// the value from the alloca into the coroutine frame spill slot corresponding
// to that alloca.
namespace {
struct AllocaUseVisitor : PtrUseVisitor<AllocaUseVisitor> {
  using Base = PtrUseVisitor<AllocaUseVisitor>;
  AllocaUseVisitor(const DataLayout &DL, const DominatorTree &DT,
                   const CoroBeginInst &CB)
      : PtrUseVisitor(DL), DT(DT), CoroBegin(CB) {}

  // We are only interested in uses that dominate coro.begin.
  void visit(Instruction &I) {
    if (DT.dominates(&I, &CoroBegin))
      Base::visit(I);
  }
  // We need to provide this overload as PtrUseVisitor uses a pointer based
  // visiting function.
  void visit(Instruction *I) { return visit(*I); }

  void visitLoadInst(LoadInst &) {} // Good. Nothing to do.

  // If the use is an operand, the pointer escaped and anything can write into
  // that memory. If the use is the pointer, we are definitely writing into the
  // alloca and therefore we need to copy.
  void visitStoreInst(StoreInst &SI) { PI.setAborted(&SI); }

  // Any other instruction that is not filtered out by PtrUseVisitor, will
  // result in the copy.
  void visitInstruction(Instruction &I) { PI.setAborted(&I); }

private:
  const DominatorTree &DT;
  const CoroBeginInst &CoroBegin;
};
} // namespace
static bool mightWriteIntoAllocaPtr(AllocaInst &A, const DominatorTree &DT,
                                    const CoroBeginInst &CB) {
  const DataLayout &DL = A.getModule()->getDataLayout();
  AllocaUseVisitor Visitor(DL, DT, CB);
  auto PtrI = Visitor.visitPtr(A);
  if (PtrI.isEscaped() || PtrI.isAborted()) {
    auto *PointerEscapingInstr = PtrI.getEscapingInst()
                                     ? PtrI.getEscapingInst()
                                     : PtrI.getAbortingInst();
    if (PointerEscapingInstr) {
      LLVM_DEBUG(
          dbgs() << "AllocaInst copy was triggered by instruction: "
                 << *PointerEscapingInstr << "\n");
    }
    return true;
  }
  return false;
}

// We need to make room to insert a spill after initial PHIs, but before
// catchswitch instruction. Placing it before violates the requirement that
// catchswitch, like all other EHPads must be the first nonPHI in a block.
//
// Split away catchswitch into a separate block and insert in its place:
//
//   cleanuppad <InsertPt> cleanupret.
//
// cleanupret instruction will act as an insert point for the spill.
static Instruction *splitBeforeCatchSwitch(CatchSwitchInst *CatchSwitch) {
  BasicBlock *CurrentBlock = CatchSwitch->getParent();
  BasicBlock *NewBlock = CurrentBlock->splitBasicBlock(CatchSwitch);
  CurrentBlock->getTerminator()->eraseFromParent();

  auto *CleanupPad =
      CleanupPadInst::Create(CatchSwitch->getParentPad(), {}, "", CurrentBlock);
  auto *CleanupRet =
      CleanupReturnInst::Create(CleanupPad, NewBlock, CurrentBlock);
  return CleanupRet;
}

// Replace all alloca and SSA values that are accessed across suspend points
// with GetElementPointer from coroutine frame + loads and stores. Create an
// AllocaSpillBB that will become the new entry block for the resume parts of
// the coroutine:
//
//    %hdl = coro.begin(...)
//    whatever
//
// becomes:
//
//    %hdl = coro.begin(...)
//    %FramePtr = bitcast i8* hdl to %f.frame*
//    br label %AllocaSpillBB
//
//  AllocaSpillBB:
//    ; geps corresponding to allocas that were moved to coroutine frame
//    br label PostSpill
//
//  PostSpill:
//    whatever
//
//
static Instruction *insertSpills(const SpillInfo &Spills, coro::Shape &Shape) {
  auto *CB = Shape.CoroBegin;
  LLVMContext &C = CB->getContext();
  IRBuilder<> Builder(CB->getNextNode());
  StructType *FrameTy = Shape.FrameTy;
  PointerType *FramePtrTy = FrameTy->getPointerTo();
  auto *FramePtr =
      cast<Instruction>(Builder.CreateBitCast(CB, FramePtrTy, "FramePtr"));
  DominatorTree DT(*CB->getFunction());

  Value *CurrentValue = nullptr;
  BasicBlock *CurrentBlock = nullptr;
  Value *CurrentReload = nullptr;

  // Proper field number will be read from field definition.
  unsigned Index = InvalidFieldIndex;

  // We need to keep track of any allocas that need "spilling"
  // since they will live in the coroutine frame now, all access to them
  // need to be changed, not just the access across suspend points
  // we remember allocas and their indices to be handled once we processed
  // all the spills.
  SmallVector<std::pair<AllocaInst *, unsigned>, 4> Allocas;
  // Promise alloca (if present) has a fixed field number.
  if (auto *PromiseAlloca = Shape.getPromiseAlloca()) {
    assert(Shape.ABI == coro::ABI::Switch);
    Allocas.emplace_back(PromiseAlloca, coro::Shape::SwitchFieldIndex::Promise);
  }

  // Create a GEP with the given index into the coroutine frame for the original
  // value Orig. Appends an extra 0 index for array-allocas, preserving the
  // original type.
  auto GetFramePointer = [&](uint32_t Index, Value *Orig) -> Value * {
    SmallVector<Value *, 3> Indices = {
        ConstantInt::get(Type::getInt32Ty(C), 0),
        ConstantInt::get(Type::getInt32Ty(C), Index),
    };

    if (auto *AI = dyn_cast<AllocaInst>(Orig)) {
      if (auto *CI = dyn_cast<ConstantInt>(AI->getArraySize())) {
        auto Count = CI->getValue().getZExtValue();
        if (Count > 1) {
          Indices.push_back(ConstantInt::get(Type::getInt32Ty(C), 0));
        }
      } else {
        report_fatal_error("Coroutines cannot handle non static allocas yet");
      }
    }

    return Builder.CreateInBoundsGEP(FrameTy, FramePtr, Indices);
  };

  // Create a load instruction to reload the spilled value from the coroutine
  // frame. Populates the Value pointer reference provided with the frame GEP.
  auto CreateReload = [&](Instruction *InsertBefore, Value *&G) {
    assert(Index != InvalidFieldIndex && "accessing unassigned field number");
    Builder.SetInsertPoint(InsertBefore);

    G = GetFramePointer(Index, CurrentValue);
    G->setName(CurrentValue->getName() + Twine(".reload.addr"));

    return isa<AllocaInst>(CurrentValue)
               ? G
               : Builder.CreateLoad(FrameTy->getElementType(Index), G,
                                    CurrentValue->getName() + Twine(".reload"));
  };

  Value *GEP = nullptr, *CurrentGEP = nullptr;
  for (auto const &E : Spills) {
    // If we have not seen the value, generate a spill.
    if (CurrentValue != E.def()) {
      CurrentValue = E.def();
      CurrentBlock = nullptr;
      CurrentReload = nullptr;

      Index = E.fieldIndex();

      if (auto *AI = dyn_cast<AllocaInst>(CurrentValue)) {
        // Spilled AllocaInst will be replaced with GEP from the coroutine frame
        // there is no spill required.
        Allocas.emplace_back(AI, Index);
        if (!AI->isStaticAlloca())
          report_fatal_error("Coroutines cannot handle non static allocas yet");
      } else {
        // Otherwise, create a store instruction storing the value into the
        // coroutine frame.

        Instruction *InsertPt = nullptr;
        if (auto Arg = dyn_cast<Argument>(CurrentValue)) {
          // For arguments, we will place the store instruction right after
          // the coroutine frame pointer instruction, i.e. bitcast of
          // coro.begin from i8* to %f.frame*.
          InsertPt = FramePtr->getNextNode();

          // If we're spilling an Argument, make sure we clear 'nocapture'
          // from the coroutine function.
          Arg->getParent()->removeParamAttr(Arg->getArgNo(),
                                            Attribute::NoCapture);

        } else if (auto *II = dyn_cast<InvokeInst>(CurrentValue)) {
          // If we are spilling the result of the invoke instruction, split the
          // normal edge and insert the spill in the new block.
          auto NewBB = SplitEdge(II->getParent(), II->getNormalDest());
          InsertPt = NewBB->getTerminator();
        } else if (isa<PHINode>(CurrentValue)) {
          // Skip the PHINodes and EH pads instructions.
          BasicBlock *DefBlock = cast<Instruction>(E.def())->getParent();
          if (auto *CSI = dyn_cast<CatchSwitchInst>(DefBlock->getTerminator()))
            InsertPt = splitBeforeCatchSwitch(CSI);
          else
            InsertPt = &*DefBlock->getFirstInsertionPt();
        } else if (auto CSI = dyn_cast<AnyCoroSuspendInst>(CurrentValue)) {
          // Don't spill immediately after a suspend; splitting assumes
          // that the suspend will be followed by a branch.
          InsertPt = CSI->getParent()->getSingleSuccessor()->getFirstNonPHI();
        } else {
          auto *I = cast<Instruction>(E.def());
          assert(!I->isTerminator() && "unexpected terminator");
          // For all other values, the spill is placed immediately after
          // the definition.
          if (DT.dominates(CB, I)) {
            InsertPt = I->getNextNode();
          } else {
            // Unless, it is not dominated by CoroBegin, then it will be
            // inserted immediately after CoroFrame is computed.
            InsertPt = FramePtr->getNextNode();
          }
        }

        Builder.SetInsertPoint(InsertPt);
        auto *G = Builder.CreateConstInBoundsGEP2_32(
            FrameTy, FramePtr, 0, Index,
            CurrentValue->getName() + Twine(".spill.addr"));
        Builder.CreateStore(CurrentValue, G);
      }
    }

    // If we have not seen the use block, generate a reload in it.
    if (CurrentBlock != E.userBlock()) {
      CurrentBlock = E.userBlock();
      CurrentReload = CreateReload(&*CurrentBlock->getFirstInsertionPt(), GEP);
    }

    // If we have a single edge PHINode, remove it and replace it with a reload
    // from the coroutine frame. (We already took care of multi edge PHINodes
    // by rewriting them in the rewritePHIs function).
    if (auto *PN = dyn_cast<PHINode>(E.user())) {
      assert(PN->getNumIncomingValues() == 1 && "unexpected number of incoming "
                                                "values in the PHINode");
      PN->replaceAllUsesWith(CurrentReload);
      PN->eraseFromParent();
      continue;
    }

    // If we have not seen this GEP instruction, migrate any dbg.declare from
    // the alloca to it.
    if (CurrentGEP != GEP) {
      CurrentGEP = GEP;
      TinyPtrVector<DbgDeclareInst *> DIs = FindDbgDeclareUses(CurrentValue);
      if (!DIs.empty())
        DIBuilder(*CurrentBlock->getParent()->getParent(),
                  /*AllowUnresolved*/ false)
            .insertDeclare(CurrentGEP, DIs.front()->getVariable(),
                           DIs.front()->getExpression(),
                           DIs.front()->getDebugLoc(), DIs.front());
    }

    // Replace all uses of CurrentValue in the current instruction with reload.
    E.user()->replaceUsesOfWith(CurrentValue, CurrentReload);
  }

  BasicBlock *FramePtrBB = FramePtr->getParent();

  auto SpillBlock =
    FramePtrBB->splitBasicBlock(FramePtr->getNextNode(), "AllocaSpillBB");      
  SpillBlock->splitBasicBlock(&SpillBlock->front(), "PostSpill");
  Shape.AllocaSpillBlock = SpillBlock;
  // If we found any alloca, replace all of their remaining uses with GEP
  // instructions. Because new dbg.declare have been created for these alloca,
  // we also delete the original dbg.declare and replace other uses with undef.
  // Note: We cannot replace the alloca with GEP instructions indiscriminately,
  // as some of the uses may not be dominated by CoroBegin.
  bool MightNeedToCopy = false;
  Builder.SetInsertPoint(&Shape.AllocaSpillBlock->front());
  SmallVector<Instruction *, 4> UsersToUpdate;
  for (auto &P : Allocas) {
    AllocaInst *const A = P.first;

    for (auto *DI : FindDbgDeclareUses(A))
      DI->eraseFromParent();
    replaceDbgUsesWithUndef(A);

    UsersToUpdate.clear();
    for (User *U : A->users()) {
      auto *I = cast<Instruction>(U);
      if (DT.dominates(CB, I))
        UsersToUpdate.push_back(I);
      else
        MightNeedToCopy = true;
    }
    if (!UsersToUpdate.empty()) {
      auto *G = GetFramePointer(P.second, A);
      G->takeName(A);
      for (Instruction *I : UsersToUpdate)
        I->replaceUsesOfWith(A, G);
    }
  }
  // If we discovered such uses not dominated by CoroBegin, see if any of them
  // preceed coro begin and have instructions that can modify the
  // value of the alloca and therefore would require a copying the value into
  // the spill slot in the coroutine frame.
  if (MightNeedToCopy) {
    Builder.SetInsertPoint(FramePtr->getNextNode());

    for (auto &P : Allocas) {
      AllocaInst *const A = P.first;
      if (mightWriteIntoAllocaPtr(*A, DT, *CB)) {
        if (A->isArrayAllocation())
          report_fatal_error(
              "Coroutines cannot handle copying of array allocas yet");

        auto *G = GetFramePointer(P.second, A);
        auto *Value = Builder.CreateLoad(A->getAllocatedType(), A);
        Builder.CreateStore(Value, G);
      }
    }
  }
  return FramePtr;
}

// Sets the unwind edge of an instruction to a particular successor.
static void setUnwindEdgeTo(Instruction *TI, BasicBlock *Succ) {
  if (auto *II = dyn_cast<InvokeInst>(TI))
    II->setUnwindDest(Succ);
  else if (auto *CS = dyn_cast<CatchSwitchInst>(TI))
    CS->setUnwindDest(Succ);
  else if (auto *CR = dyn_cast<CleanupReturnInst>(TI))
    CR->setUnwindDest(Succ);
  else
    llvm_unreachable("unexpected terminator instruction");
}

// Replaces all uses of OldPred with the NewPred block in all PHINodes in a
// block.
static void updatePhiNodes(BasicBlock *DestBB, BasicBlock *OldPred,
                           BasicBlock *NewPred,
                           PHINode *LandingPadReplacement) {
  unsigned BBIdx = 0;
  for (BasicBlock::iterator I = DestBB->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);

    // We manually update the LandingPadReplacement PHINode and it is the last
    // PHI Node. So, if we find it, we are done.
    if (LandingPadReplacement == PN)
      break;

    // Reuse the previous value of BBIdx if it lines up.  In cases where we
    // have multiple phi nodes with *lots* of predecessors, this is a speed
    // win because we don't have to scan the PHI looking for TIBB.  This
    // happens because the BB list of PHI nodes are usually in the same
    // order.
    if (PN->getIncomingBlock(BBIdx) != OldPred)
      BBIdx = PN->getBasicBlockIndex(OldPred);

    assert(BBIdx != (unsigned)-1 && "Invalid PHI Index!");
    PN->setIncomingBlock(BBIdx, NewPred);
  }
}

// Uses SplitEdge unless the successor block is an EHPad, in which case do EH
// specific handling.
static BasicBlock *ehAwareSplitEdge(BasicBlock *BB, BasicBlock *Succ,
                                    LandingPadInst *OriginalPad,
                                    PHINode *LandingPadReplacement) {
  auto *PadInst = Succ->getFirstNonPHI();
  if (!LandingPadReplacement && !PadInst->isEHPad())
    return SplitEdge(BB, Succ);

  auto *NewBB = BasicBlock::Create(BB->getContext(), "", BB->getParent(), Succ);
  setUnwindEdgeTo(BB->getTerminator(), NewBB);
  updatePhiNodes(Succ, BB, NewBB, LandingPadReplacement);

  if (LandingPadReplacement) {
    auto *NewLP = OriginalPad->clone();
    auto *Terminator = BranchInst::Create(Succ, NewBB);
    NewLP->insertBefore(Terminator);
    LandingPadReplacement->addIncoming(NewLP, NewBB);
    return NewBB;
  }
  Value *ParentPad = nullptr;
  if (auto *FuncletPad = dyn_cast<FuncletPadInst>(PadInst))
    ParentPad = FuncletPad->getParentPad();
  else if (auto *CatchSwitch = dyn_cast<CatchSwitchInst>(PadInst))
    ParentPad = CatchSwitch->getParentPad();
  else
    llvm_unreachable("handling for other EHPads not implemented yet");

  auto *NewCleanupPad = CleanupPadInst::Create(ParentPad, {}, "", NewBB);
  CleanupReturnInst::Create(NewCleanupPad, Succ, NewBB);
  return NewBB;
}

static void rewritePHIs(BasicBlock &BB) {
  // For every incoming edge we will create a block holding all
  // incoming values in a single PHI nodes.
  //
  // loop:
  //    %n.val = phi i32[%n, %entry], [%inc, %loop]
  //
  // It will create:
  //
  // loop.from.entry:
  //    %n.loop.pre = phi i32 [%n, %entry]
  //    br %label loop
  // loop.from.loop:
  //    %inc.loop.pre = phi i32 [%inc, %loop]
  //    br %label loop
  //
  // After this rewrite, further analysis will ignore any phi nodes with more
  // than one incoming edge.

  // TODO: Simplify PHINodes in the basic block to remove duplicate
  // predecessors.

  LandingPadInst *LandingPad = nullptr;
  PHINode *ReplPHI = nullptr;
  if ((LandingPad = dyn_cast_or_null<LandingPadInst>(BB.getFirstNonPHI()))) {
    // ehAwareSplitEdge will clone the LandingPad in all the edge blocks.
    // We replace the original landing pad with a PHINode that will collect the
    // results from all of them.
    ReplPHI = PHINode::Create(LandingPad->getType(), 1, "", LandingPad);
    ReplPHI->takeName(LandingPad);
    LandingPad->replaceAllUsesWith(ReplPHI);
    // We will erase the original landing pad at the end of this function after
    // ehAwareSplitEdge cloned it in the transition blocks.
  }

  SmallVector<BasicBlock *, 8> Preds(pred_begin(&BB), pred_end(&BB));
  for (BasicBlock *Pred : Preds) {
    auto *IncomingBB = ehAwareSplitEdge(Pred, &BB, LandingPad, ReplPHI);
    IncomingBB->setName(BB.getName() + Twine(".from.") + Pred->getName());
    auto *PN = cast<PHINode>(&BB.front());
    do {
      int Index = PN->getBasicBlockIndex(IncomingBB);
      Value *V = PN->getIncomingValue(Index);
      PHINode *InputV = PHINode::Create(
          V->getType(), 1, V->getName() + Twine(".") + BB.getName(),
          &IncomingBB->front());
      InputV->addIncoming(V, Pred);
      PN->setIncomingValue(Index, InputV);
      PN = dyn_cast<PHINode>(PN->getNextNode());
    } while (PN != ReplPHI); // ReplPHI is either null or the PHI that replaced
                             // the landing pad.
  }

  if (LandingPad) {
    // Calls to ehAwareSplitEdge function cloned the original lading pad.
    // No longer need it.
    LandingPad->eraseFromParent();
  }
}

static void rewritePHIs(Function &F) {
  SmallVector<BasicBlock *, 8> WorkList;

  for (BasicBlock &BB : F)
    if (auto *PN = dyn_cast<PHINode>(&BB.front()))
      if (PN->getNumIncomingValues() > 1)
        WorkList.push_back(&BB);

  for (BasicBlock *BB : WorkList)
    rewritePHIs(*BB);
}

// Check for instructions that we can recreate on resume as opposed to spill
// the result into a coroutine frame.
static bool materializable(Instruction &V) {
  return isa<CastInst>(&V) || isa<GetElementPtrInst>(&V) ||
         isa<BinaryOperator>(&V) || isa<CmpInst>(&V) || isa<SelectInst>(&V);
}

// Check for structural coroutine intrinsics that should not be spilled into
// the coroutine frame.
static bool isCoroutineStructureIntrinsic(Instruction &I) {
  return isa<CoroIdInst>(&I) || isa<CoroSaveInst>(&I) ||
         isa<CoroSuspendInst>(&I);
}

// For every use of the value that is across suspend point, recreate that value
// after a suspend point.
static void rewriteMaterializableInstructions(IRBuilder<> &IRB,
                                              SpillInfo const &Spills) {
  BasicBlock *CurrentBlock = nullptr;
  Instruction *CurrentMaterialization = nullptr;
  Instruction *CurrentDef = nullptr;

  for (auto const &E : Spills) {
    // If it is a new definition, update CurrentXXX variables.
    if (CurrentDef != E.def()) {
      CurrentDef = cast<Instruction>(E.def());
      CurrentBlock = nullptr;
      CurrentMaterialization = nullptr;
    }

    // If we have not seen this block, materialize the value.
    if (CurrentBlock != E.userBlock()) {
      CurrentBlock = E.userBlock();
      CurrentMaterialization = cast<Instruction>(CurrentDef)->clone();
      CurrentMaterialization->setName(CurrentDef->getName());
      CurrentMaterialization->insertBefore(
          &*CurrentBlock->getFirstInsertionPt());
    }

    if (auto *PN = dyn_cast<PHINode>(E.user())) {
      assert(PN->getNumIncomingValues() == 1 && "unexpected number of incoming "
                                                "values in the PHINode");
      PN->replaceAllUsesWith(CurrentMaterialization);
      PN->eraseFromParent();
      continue;
    }

    // Replace all uses of CurrentDef in the current instruction with the
    // CurrentMaterialization for the block.
    E.user()->replaceUsesOfWith(CurrentDef, CurrentMaterialization);
  }
}

// Splits the block at a particular instruction unless it is the first
// instruction in the block with a single predecessor.
static BasicBlock *splitBlockIfNotFirst(Instruction *I, const Twine &Name) {
  auto *BB = I->getParent();
  if (&BB->front() == I) {
    if (BB->getSinglePredecessor()) {
      BB->setName(Name);
      return BB;
    }
  }
  return BB->splitBasicBlock(I, Name);
}

// Split above and below a particular instruction so that it
// will be all alone by itself in a block.
static void splitAround(Instruction *I, const Twine &Name) {
  splitBlockIfNotFirst(I, Name);
  splitBlockIfNotFirst(I->getNextNode(), "After" + Name);
}

static bool isSuspendBlock(BasicBlock *BB) {
  return isa<AnyCoroSuspendInst>(BB->front());
}

typedef SmallPtrSet<BasicBlock*, 8> VisitedBlocksSet;

/// Does control flow starting at the given block ever reach a suspend
/// instruction before reaching a block in VisitedOrFreeBBs?
static bool isSuspendReachableFrom(BasicBlock *From,
                                   VisitedBlocksSet &VisitedOrFreeBBs) {
  // Eagerly try to add this block to the visited set.  If it's already
  // there, stop recursing; this path doesn't reach a suspend before
  // either looping or reaching a freeing block.
  if (!VisitedOrFreeBBs.insert(From).second)
    return false;

  // We assume that we'll already have split suspends into their own blocks.
  if (isSuspendBlock(From))
    return true;

  // Recurse on the successors.
  for (auto Succ : successors(From)) {
    if (isSuspendReachableFrom(Succ, VisitedOrFreeBBs))
      return true;
  }

  return false;
}

/// Is the given alloca "local", i.e. bounded in lifetime to not cross a
/// suspend point?
static bool isLocalAlloca(CoroAllocaAllocInst *AI) {
  // Seed the visited set with all the basic blocks containing a free
  // so that we won't pass them up.
  VisitedBlocksSet VisitedOrFreeBBs;
  for (auto User : AI->users()) {
    if (auto FI = dyn_cast<CoroAllocaFreeInst>(User))
      VisitedOrFreeBBs.insert(FI->getParent());
  }

  return !isSuspendReachableFrom(AI->getParent(), VisitedOrFreeBBs);
}

/// After we split the coroutine, will the given basic block be along
/// an obvious exit path for the resumption function?
static bool willLeaveFunctionImmediatelyAfter(BasicBlock *BB,
                                              unsigned depth = 3) {
  // If we've bottomed out our depth count, stop searching and assume
  // that the path might loop back.
  if (depth == 0) return false;

  // If this is a suspend block, we're about to exit the resumption function.
  if (isSuspendBlock(BB)) return true;

  // Recurse into the successors.
  for (auto Succ : successors(BB)) {
    if (!willLeaveFunctionImmediatelyAfter(Succ, depth - 1))
      return false;
  }

  // If none of the successors leads back in a loop, we're on an exit/abort.
  return true;
}

static bool localAllocaNeedsStackSave(CoroAllocaAllocInst *AI) {
  // Look for a free that isn't sufficiently obviously followed by
  // either a suspend or a termination, i.e. something that will leave
  // the coro resumption frame.
  for (auto U : AI->users()) {
    auto FI = dyn_cast<CoroAllocaFreeInst>(U);
    if (!FI) continue;

    if (!willLeaveFunctionImmediatelyAfter(FI->getParent()))
      return true;
  }

  // If we never found one, we don't need a stack save.
  return false;
}

/// Turn each of the given local allocas into a normal (dynamic) alloca
/// instruction.
static void lowerLocalAllocas(ArrayRef<CoroAllocaAllocInst*> LocalAllocas,
                              SmallVectorImpl<Instruction*> &DeadInsts) {
  for (auto AI : LocalAllocas) {
    auto M = AI->getModule();
    IRBuilder<> Builder(AI);

    // Save the stack depth.  Try to avoid doing this if the stackrestore
    // is going to immediately precede a return or something.
    Value *StackSave = nullptr;
    if (localAllocaNeedsStackSave(AI))
      StackSave = Builder.CreateCall(
                            Intrinsic::getDeclaration(M, Intrinsic::stacksave));

    // Allocate memory.
    auto Alloca = Builder.CreateAlloca(Builder.getInt8Ty(), AI->getSize());
    Alloca->setAlignment(MaybeAlign(AI->getAlignment()));

    for (auto U : AI->users()) {
      // Replace gets with the allocation.
      if (isa<CoroAllocaGetInst>(U)) {
        U->replaceAllUsesWith(Alloca);

      // Replace frees with stackrestores.  This is safe because
      // alloca.alloc is required to obey a stack discipline, although we
      // don't enforce that structurally.
      } else {
        auto FI = cast<CoroAllocaFreeInst>(U);
        if (StackSave) {
          Builder.SetInsertPoint(FI);
          Builder.CreateCall(
                    Intrinsic::getDeclaration(M, Intrinsic::stackrestore),
                             StackSave);
        }
      }
      DeadInsts.push_back(cast<Instruction>(U));
    }

    DeadInsts.push_back(AI);
  }
}

/// Turn the given coro.alloca.alloc call into a dynamic allocation.
/// This happens during the all-instructions iteration, so it must not
/// delete the call.
static Instruction *lowerNonLocalAlloca(CoroAllocaAllocInst *AI,
                                        coro::Shape &Shape,
                                   SmallVectorImpl<Instruction*> &DeadInsts) {
  IRBuilder<> Builder(AI);
  auto Alloc = Shape.emitAlloc(Builder, AI->getSize(), nullptr);

  for (User *U : AI->users()) {
    if (isa<CoroAllocaGetInst>(U)) {
      U->replaceAllUsesWith(Alloc);
    } else {
      auto FI = cast<CoroAllocaFreeInst>(U);
      Builder.SetInsertPoint(FI);
      Shape.emitDealloc(Builder, Alloc, nullptr);
    }
    DeadInsts.push_back(cast<Instruction>(U));
  }

  // Push this on last so that it gets deleted after all the others.
  DeadInsts.push_back(AI);

  // Return the new allocation value so that we can check for needed spills.
  return cast<Instruction>(Alloc);
}

/// Get the current swifterror value.
static Value *emitGetSwiftErrorValue(IRBuilder<> &Builder, Type *ValueTy,
                                     coro::Shape &Shape) {
  // Make a fake function pointer as a sort of intrinsic.
  auto FnTy = FunctionType::get(ValueTy, {}, false);
  auto Fn = ConstantPointerNull::get(FnTy->getPointerTo());

  auto Call = Builder.CreateCall(FnTy, Fn, {});
  Shape.SwiftErrorOps.push_back(Call);

  return Call;
}

/// Set the given value as the current swifterror value.
///
/// Returns a slot that can be used as a swifterror slot.
static Value *emitSetSwiftErrorValue(IRBuilder<> &Builder, Value *V,
                                     coro::Shape &Shape) {
  // Make a fake function pointer as a sort of intrinsic.
  auto FnTy = FunctionType::get(V->getType()->getPointerTo(),
                                {V->getType()}, false);
  auto Fn = ConstantPointerNull::get(FnTy->getPointerTo());

  auto Call = Builder.CreateCall(FnTy, Fn, { V });
  Shape.SwiftErrorOps.push_back(Call);

  return Call;
}

/// Set the swifterror value from the given alloca before a call,
/// then put in back in the alloca afterwards.
///
/// Returns an address that will stand in for the swifterror slot
/// until splitting.
static Value *emitSetAndGetSwiftErrorValueAround(Instruction *Call,
                                                 AllocaInst *Alloca,
                                                 coro::Shape &Shape) {
  auto ValueTy = Alloca->getAllocatedType();
  IRBuilder<> Builder(Call);

  // Load the current value from the alloca and set it as the
  // swifterror value.
  auto ValueBeforeCall = Builder.CreateLoad(ValueTy, Alloca);
  auto Addr = emitSetSwiftErrorValue(Builder, ValueBeforeCall, Shape);

  // Move to after the call.  Since swifterror only has a guaranteed
  // value on normal exits, we can ignore implicit and explicit unwind
  // edges.
  if (isa<CallInst>(Call)) {
    Builder.SetInsertPoint(Call->getNextNode());
  } else {
    auto Invoke = cast<InvokeInst>(Call);
    Builder.SetInsertPoint(Invoke->getNormalDest()->getFirstNonPHIOrDbg());
  }

  // Get the current swifterror value and store it to the alloca.
  auto ValueAfterCall = emitGetSwiftErrorValue(Builder, ValueTy, Shape);
  Builder.CreateStore(ValueAfterCall, Alloca);

  return Addr;
}

/// Eliminate a formerly-swifterror alloca by inserting the get/set
/// intrinsics and attempting to MemToReg the alloca away.
static void eliminateSwiftErrorAlloca(Function &F, AllocaInst *Alloca,
                                      coro::Shape &Shape) {
  for (auto UI = Alloca->use_begin(), UE = Alloca->use_end(); UI != UE; ) {
    // We're likely changing the use list, so use a mutation-safe
    // iteration pattern.
    auto &Use = *UI;
    ++UI;

    // swifterror values can only be used in very specific ways.
    // We take advantage of that here.
    auto User = Use.getUser();
    if (isa<LoadInst>(User) || isa<StoreInst>(User))
      continue;

    assert(isa<CallInst>(User) || isa<InvokeInst>(User));
    auto Call = cast<Instruction>(User);

    auto Addr = emitSetAndGetSwiftErrorValueAround(Call, Alloca, Shape);

    // Use the returned slot address as the call argument.
    Use.set(Addr);
  }

  // All the uses should be loads and stores now.
  assert(isAllocaPromotable(Alloca));
}

/// "Eliminate" a swifterror argument by reducing it to the alloca case
/// and then loading and storing in the prologue and epilog.
///
/// The argument keeps the swifterror flag.
static void eliminateSwiftErrorArgument(Function &F, Argument &Arg,
                                        coro::Shape &Shape,
                             SmallVectorImpl<AllocaInst*> &AllocasToPromote) {
  IRBuilder<> Builder(F.getEntryBlock().getFirstNonPHIOrDbg());

  auto ArgTy = cast<PointerType>(Arg.getType());
  auto ValueTy = ArgTy->getElementType();

  // Reduce to the alloca case:

  // Create an alloca and replace all uses of the arg with it.
  auto Alloca = Builder.CreateAlloca(ValueTy, ArgTy->getAddressSpace());
  Arg.replaceAllUsesWith(Alloca);

  // Set an initial value in the alloca.  swifterror is always null on entry.
  auto InitialValue = Constant::getNullValue(ValueTy);
  Builder.CreateStore(InitialValue, Alloca);

  // Find all the suspends in the function and save and restore around them.
  for (auto Suspend : Shape.CoroSuspends) {
    (void) emitSetAndGetSwiftErrorValueAround(Suspend, Alloca, Shape);
  }

  // Find all the coro.ends in the function and restore the error value.
  for (auto End : Shape.CoroEnds) {
    Builder.SetInsertPoint(End);
    auto FinalValue = Builder.CreateLoad(ValueTy, Alloca);
    (void) emitSetSwiftErrorValue(Builder, FinalValue, Shape);
  }

  // Now we can use the alloca logic.
  AllocasToPromote.push_back(Alloca);
  eliminateSwiftErrorAlloca(F, Alloca, Shape);
}

/// Eliminate all problematic uses of swifterror arguments and allocas
/// from the function.  We'll fix them up later when splitting the function.
static void eliminateSwiftError(Function &F, coro::Shape &Shape) {
  SmallVector<AllocaInst*, 4> AllocasToPromote;

  // Look for a swifterror argument.
  for (auto &Arg : F.args()) {
    if (!Arg.hasSwiftErrorAttr()) continue;

    eliminateSwiftErrorArgument(F, Arg, Shape, AllocasToPromote);
    break;
  }

  // Look for swifterror allocas.
  for (auto &Inst : F.getEntryBlock()) {
    auto Alloca = dyn_cast<AllocaInst>(&Inst);
    if (!Alloca || !Alloca->isSwiftError()) continue;

    // Clear the swifterror flag.
    Alloca->setSwiftError(false);

    AllocasToPromote.push_back(Alloca);
    eliminateSwiftErrorAlloca(F, Alloca, Shape);
  }

  // If we have any allocas to promote, compute a dominator tree and
  // promote them en masse.
  if (!AllocasToPromote.empty()) {
    DominatorTree DT(F);
    PromoteMemToReg(AllocasToPromote, DT);
  }
}

void coro::buildCoroutineFrame(Function &F, Shape &Shape) {
  eliminateSwiftError(F, Shape);

  if (Shape.ABI == coro::ABI::Switch &&
      Shape.SwitchLowering.PromiseAlloca) {
    Shape.getSwitchCoroId()->clearPromise();
  }

  // Make sure that all coro.save, coro.suspend and the fallthrough coro.end
  // intrinsics are in their own blocks to simplify the logic of building up
  // SuspendCrossing data.
  for (auto *CSI : Shape.CoroSuspends) {
    if (auto *Save = CSI->getCoroSave())
      splitAround(Save, "CoroSave");
    splitAround(CSI, "CoroSuspend");
  }

  // Put CoroEnds into their own blocks.
  for (CoroEndInst *CE : Shape.CoroEnds)
    splitAround(CE, "CoroEnd");

  // Transforms multi-edge PHI Nodes, so that any value feeding into a PHI will
  // never has its definition separated from the PHI by the suspend point.
  rewritePHIs(F);

  // Build suspend crossing info.
  SuspendCrossingInfo Checker(F, Shape);

  IRBuilder<> Builder(F.getContext());
  SpillInfo Spills;
  SmallVector<CoroAllocaAllocInst*, 4> LocalAllocas;
  SmallVector<Instruction*, 4> DeadInstructions;

  for (int Repeat = 0; Repeat < 4; ++Repeat) {
    // See if there are materializable instructions across suspend points.
    for (Instruction &I : instructions(F))
      if (materializable(I))
        for (User *U : I.users())
          if (Checker.isDefinitionAcrossSuspend(I, U))
            Spills.emplace_back(&I, U);

    if (Spills.empty())
      break;

    // Rewrite materializable instructions to be materialized at the use point.
    LLVM_DEBUG(dump("Materializations", Spills));
    rewriteMaterializableInstructions(Builder, Spills);
    Spills.clear();
  }

  // Collect lifetime.start info for each alloca.
  using LifetimeStart = SmallPtrSet<Instruction *, 2>;
  llvm::DenseMap<Instruction *, std::unique_ptr<LifetimeStart>> LifetimeMap;
  for (Instruction &I : instructions(F)) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (!II || II->getIntrinsicID() != Intrinsic::lifetime_start)
      continue;

    if (auto *OpInst = dyn_cast<BitCastInst>(I.getOperand(1)))
      if (auto *AI = dyn_cast<AllocaInst>(OpInst->getOperand(0))) {

        if (LifetimeMap.find(AI) == LifetimeMap.end())
          LifetimeMap[AI] = std::make_unique<LifetimeStart>();

        LifetimeMap[AI]->insert(OpInst);
      }
  }

  // Collect the spills for arguments and other not-materializable values.
  for (Argument &A : F.args())
    for (User *U : A.users())
      if (Checker.isDefinitionAcrossSuspend(A, U))
        Spills.emplace_back(&A, U);

  for (Instruction &I : instructions(F)) {
    // Values returned from coroutine structure intrinsics should not be part
    // of the Coroutine Frame.
    if (isCoroutineStructureIntrinsic(I) || &I == Shape.CoroBegin)
      continue;

    // The Coroutine Promise always included into coroutine frame, no need to
    // check for suspend crossing.
    if (Shape.ABI == coro::ABI::Switch &&
        Shape.SwitchLowering.PromiseAlloca == &I)
      continue;

    // Handle alloca.alloc specially here.
    if (auto AI = dyn_cast<CoroAllocaAllocInst>(&I)) {
      // Check whether the alloca's lifetime is bounded by suspend points.
      if (isLocalAlloca(AI)) {
        LocalAllocas.push_back(AI);
        continue;
      }

      // If not, do a quick rewrite of the alloca and then add spills of
      // the rewritten value.  The rewrite doesn't invalidate anything in
      // Spills because the other alloca intrinsics have no other operands
      // besides AI, and it doesn't invalidate the iteration because we delay
      // erasing AI.
      auto Alloc = lowerNonLocalAlloca(AI, Shape, DeadInstructions);

      for (User *U : Alloc->users()) {
        if (Checker.isDefinitionAcrossSuspend(*Alloc, U))
          Spills.emplace_back(Alloc, U);
      }
      continue;
    }

    // Ignore alloca.get; we process this as part of coro.alloca.alloc.
    if (isa<CoroAllocaGetInst>(I)) {
      continue;
    }

    auto Iter = LifetimeMap.find(&I);
    for (User *U : I.users()) {
      bool NeedSpill = false;

      // Check against lifetime.start if the instruction has the info.
      if (Iter != LifetimeMap.end())
        for (auto *S : *Iter->second) {
          if ((NeedSpill = Checker.isDefinitionAcrossSuspend(*S, U)))
            break;
        }
      else
        NeedSpill = Checker.isDefinitionAcrossSuspend(I, U);

      if (NeedSpill) {
        // We cannot spill a token.
        if (I.getType()->isTokenTy())
          report_fatal_error(
              "token definition is separated from the use by a suspend point");
        Spills.emplace_back(&I, U);
      }
    }
  }
  LLVM_DEBUG(dump("Spills", Spills));
  Shape.FrameTy = buildFrameType(F, Shape, Spills);
  Shape.FramePtr = insertSpills(Spills, Shape);
  lowerLocalAllocas(LocalAllocas, DeadInstructions);

  for (auto I : DeadInstructions)
    I->eraseFromParent();
}
