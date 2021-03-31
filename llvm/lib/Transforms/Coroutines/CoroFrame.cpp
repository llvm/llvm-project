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
//===----------------------------------------------------------------------===//

#include "CoroInternal.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Analysis/PtrUseVisitor.h"
#include "llvm/Analysis/StackLifetime.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/OptimizedStructLayout.h"
#include "llvm/Support/circular_raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include <algorithm>

using namespace llvm;

// The "coro-suspend-crossing" flag is very noisy. There is another debug type,
// "coro-frame", which results in leaner debug spew.
#define DEBUG_TYPE "coro-suspend-crossing"

static cl::opt<bool> EnableReuseStorageInFrame(
    "reuse-storage-in-coroutine-frame", cl::Hidden,
    cl::desc(
        "Enable the optimization which would reuse the storage in the coroutine \
         frame for allocas whose liferanges are not overlapped, for testing purposes"),
    llvm::cl::init(false));

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

    // As a special case, treat uses by an llvm.coro.suspend.retcon or an
    // llvm.coro.suspend.async as if they were uses in the suspend's single
    // predecessor: the uses conceptually occur before the suspend.
    if (isa<CoroSuspendRetconInst>(I) || isa<CoroSuspendAsyncInst>(I)) {
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

namespace {
class FrameTypeBuilder;
// Mapping from the to-be-spilled value to all the users that need reload.
using SpillInfo = SmallMapVector<Value *, SmallVector<Instruction *, 2>, 8>;
struct AllocaInfo {
  AllocaInst *Alloca;
  DenseMap<Instruction *, llvm::Optional<APInt>> Aliases;
  bool MayWriteBeforeCoroBegin;
  AllocaInfo(AllocaInst *Alloca,
             DenseMap<Instruction *, llvm::Optional<APInt>> Aliases,
             bool MayWriteBeforeCoroBegin)
      : Alloca(Alloca), Aliases(std::move(Aliases)),
        MayWriteBeforeCoroBegin(MayWriteBeforeCoroBegin) {}
};
struct FrameDataInfo {
  // All the values (that are not allocas) that needs to be spilled to the
  // frame.
  SpillInfo Spills;
  // Allocas contains all values defined as allocas that need to live in the
  // frame.
  SmallVector<AllocaInfo, 8> Allocas;

  SmallVector<Value *, 8> getAllDefs() const {
    SmallVector<Value *, 8> Defs;
    for (const auto &P : Spills)
      Defs.push_back(P.first);
    for (const auto &A : Allocas)
      Defs.push_back(A.Alloca);
    return Defs;
  }

  uint32_t getFieldIndex(Value *V) const {
    auto Itr = FieldIndexMap.find(V);
    assert(Itr != FieldIndexMap.end() &&
           "Value does not have a frame field index");
    return Itr->second;
  }

  void setFieldIndex(Value *V, uint32_t Index) {
    assert((LayoutIndexUpdateStarted || FieldIndexMap.count(V) == 0) &&
           "Cannot set the index for the same field twice.");
    FieldIndexMap[V] = Index;
  }

  // Remap the index of every field in the frame, using the final layout index.
  void updateLayoutIndex(FrameTypeBuilder &B);

private:
  // LayoutIndexUpdateStarted is used to avoid updating the index of any field
  // twice by mistake.
  bool LayoutIndexUpdateStarted = false;
  // Map from values to their slot indexes on the frame. They will be first set
  // with their original insertion field index. After the frame is built, their
  // indexes will be updated into the final layout index.
  DenseMap<Value *, uint32_t> FieldIndexMap;
};
} // namespace

#ifndef NDEBUG
static void dumpSpills(StringRef Title, const SpillInfo &Spills) {
  dbgs() << "------------- " << Title << "--------------\n";
  for (const auto &E : Spills) {
    E.first->dump();
    dbgs() << "   user: ";
    for (auto *I : E.second)
      I->dump();
  }
}

static void dumpAllocas(const SmallVectorImpl<AllocaInfo> &Allocas) {
  dbgs() << "------------- Allocas --------------\n";
  for (const auto &A : Allocas) {
    A.Alloca->dump();
  }
}
#endif

namespace {
using FieldIDType = size_t;
// We cannot rely solely on natural alignment of a type when building a
// coroutine frame and if the alignment specified on the Alloca instruction
// differs from the natural alignment of the alloca type we will need to insert
// padding.
class FrameTypeBuilder {
private:
  struct Field {
    uint64_t Size;
    uint64_t Offset;
    Type *Ty;
    FieldIDType LayoutFieldIndex;
    Align Alignment;
    Align TyAlignment;
  };

  const DataLayout &DL;
  LLVMContext &Context;
  uint64_t StructSize = 0;
  Align StructAlign;
  bool IsFinished = false;

  SmallVector<Field, 8> Fields;
  DenseMap<Value*, unsigned> FieldIndexByKey;

public:
  FrameTypeBuilder(LLVMContext &Context, DataLayout const &DL)
      : DL(DL), Context(Context) {}

  /// Add a field to this structure for the storage of an `alloca`
  /// instruction.
  LLVM_NODISCARD FieldIDType addFieldForAlloca(AllocaInst *AI,
                                               bool IsHeader = false) {
    Type *Ty = AI->getAllocatedType();

    // Make an array type if this is a static array allocation.
    if (AI->isArrayAllocation()) {
      if (auto *CI = dyn_cast<ConstantInt>(AI->getArraySize()))
        Ty = ArrayType::get(Ty, CI->getValue().getZExtValue());
      else
        report_fatal_error("Coroutines cannot handle non static allocas yet");
    }

    return addField(Ty, AI->getAlign(), IsHeader);
  }

  /// We want to put the allocas whose lifetime-ranges are not overlapped
  /// into one slot of coroutine frame.
  /// Consider the example at:https://bugs.llvm.org/show_bug.cgi?id=45566
  ///
  ///     cppcoro::task<void> alternative_paths(bool cond) {
  ///         if (cond) {
  ///             big_structure a;
  ///             process(a);
  ///             co_await something();
  ///         } else {
  ///             big_structure b;
  ///             process2(b);
  ///             co_await something();
  ///         }
  ///     }
  ///
  /// We want to put variable a and variable b in the same slot to
  /// reduce the size of coroutine frame.
  ///
  /// This function use StackLifetime algorithm to partition the AllocaInsts in
  /// Spills to non-overlapped sets in order to put Alloca in the same
  /// non-overlapped set into the same slot in the Coroutine Frame. Then add
  /// field for the allocas in the same non-overlapped set by using the largest
  /// type as the field type.
  ///
  /// Side Effects: Because We sort the allocas, the order of allocas in the
  /// frame may be different with the order in the source code.
  void addFieldForAllocas(const Function &F, FrameDataInfo &FrameData,
                          coro::Shape &Shape);

  /// Add a field to this structure.
  LLVM_NODISCARD FieldIDType addField(Type *Ty, MaybeAlign FieldAlignment,
                                      bool IsHeader = false) {
    assert(!IsFinished && "adding fields to a finished builder");
    assert(Ty && "must provide a type for a field");

    // The field size is always the alloc size of the type.
    uint64_t FieldSize = DL.getTypeAllocSize(Ty);

    // The field alignment might not be the type alignment, but we need
    // to remember the type alignment anyway to build the type.
    Align TyAlignment = DL.getABITypeAlign(Ty);
    if (!FieldAlignment) FieldAlignment = TyAlignment;

    // Lay out header fields immediately.
    uint64_t Offset;
    if (IsHeader) {
      Offset = alignTo(StructSize, FieldAlignment);
      StructSize = Offset + FieldSize;

    // Everything else has a flexible offset.
    } else {
      Offset = OptimizedStructLayoutField::FlexibleOffset;
    }

    Fields.push_back({FieldSize, Offset, Ty, 0, *FieldAlignment, TyAlignment});
    return Fields.size() - 1;
  }

  /// Finish the layout and set the body on the given type.
  void finish(StructType *Ty);

  uint64_t getStructSize() const {
    assert(IsFinished && "not yet finished!");
    return StructSize;
  }

  Align getStructAlign() const {
    assert(IsFinished && "not yet finished!");
    return StructAlign;
  }

  FieldIDType getLayoutFieldIndex(FieldIDType Id) const {
    assert(IsFinished && "not yet finished!");
    return Fields[Id].LayoutFieldIndex;
  }
};
} // namespace

void FrameDataInfo::updateLayoutIndex(FrameTypeBuilder &B) {
  auto Updater = [&](Value *I) {
    setFieldIndex(I, B.getLayoutFieldIndex(getFieldIndex(I)));
  };
  LayoutIndexUpdateStarted = true;
  for (auto &S : Spills)
    Updater(S.first);
  for (const auto &A : Allocas)
    Updater(A.Alloca);
  LayoutIndexUpdateStarted = false;
}

void FrameTypeBuilder::addFieldForAllocas(const Function &F,
                                          FrameDataInfo &FrameData,
                                          coro::Shape &Shape) {
  using AllocaSetType = SmallVector<AllocaInst *, 4>;
  SmallVector<AllocaSetType, 4> NonOverlapedAllocas;

  // We need to add field for allocas at the end of this function. However, this
  // function has multiple exits, so we use this helper to avoid redundant code.
  struct RTTIHelper {
    std::function<void()> func;
    RTTIHelper(std::function<void()> &&func) : func(func) {}
    ~RTTIHelper() { func(); }
  } Helper([&]() {
    for (auto AllocaList : NonOverlapedAllocas) {
      auto *LargestAI = *AllocaList.begin();
      FieldIDType Id = addFieldForAlloca(LargestAI);
      for (auto *Alloca : AllocaList)
        FrameData.setFieldIndex(Alloca, Id);
    }
  });

  if (!Shape.ReuseFrameSlot && !EnableReuseStorageInFrame) {
    for (const auto &A : FrameData.Allocas) {
      AllocaInst *Alloca = A.Alloca;
      NonOverlapedAllocas.emplace_back(AllocaSetType(1, Alloca));
    }
    return;
  }

  // Because there are pathes from the lifetime.start to coro.end
  // for each alloca, the liferanges for every alloca is overlaped
  // in the blocks who contain coro.end and the successor blocks.
  // So we choose to skip there blocks when we calculates the liferange
  // for each alloca. It should be reasonable since there shouldn't be uses
  // in these blocks and the coroutine frame shouldn't be used outside the
  // coroutine body.
  //
  // Note that the user of coro.suspend may not be SwitchInst. However, this
  // case seems too complex to handle. And it is harmless to skip these
  // patterns since it just prevend putting the allocas to live in the same
  // slot.
  DenseMap<SwitchInst *, BasicBlock *> DefaultSuspendDest;
  for (auto CoroSuspendInst : Shape.CoroSuspends) {
    for (auto U : CoroSuspendInst->users()) {
      if (auto *ConstSWI = dyn_cast<SwitchInst>(U)) {
        auto *SWI = const_cast<SwitchInst *>(ConstSWI);
        DefaultSuspendDest[SWI] = SWI->getDefaultDest();
        SWI->setDefaultDest(SWI->getSuccessor(1));
      }
    }
  }

  auto ExtractAllocas = [&]() {
    AllocaSetType Allocas;
    Allocas.reserve(FrameData.Allocas.size());
    for (const auto &A : FrameData.Allocas)
      Allocas.push_back(A.Alloca);
    return Allocas;
  };
  StackLifetime StackLifetimeAnalyzer(F, ExtractAllocas(),
                                      StackLifetime::LivenessType::May);
  StackLifetimeAnalyzer.run();
  auto IsAllocaInferenre = [&](const AllocaInst *AI1, const AllocaInst *AI2) {
    return StackLifetimeAnalyzer.getLiveRange(AI1).overlaps(
        StackLifetimeAnalyzer.getLiveRange(AI2));
  };
  auto GetAllocaSize = [&](const AllocaInfo &A) {
    Optional<TypeSize> RetSize = A.Alloca->getAllocationSizeInBits(DL);
    assert(RetSize && "Variable Length Arrays (VLA) are not supported.\n");
    assert(!RetSize->isScalable() && "Scalable vectors are not yet supported");
    return RetSize->getFixedSize();
  };
  // Put larger allocas in the front. So the larger allocas have higher
  // priority to merge, which can save more space potentially. Also each
  // AllocaSet would be ordered. So we can get the largest Alloca in one
  // AllocaSet easily.
  sort(FrameData.Allocas, [&](const auto &Iter1, const auto &Iter2) {
    return GetAllocaSize(Iter1) > GetAllocaSize(Iter2);
  });
  for (const auto &A : FrameData.Allocas) {
    AllocaInst *Alloca = A.Alloca;
    bool Merged = false;
    // Try to find if the Alloca is not inferenced with any existing
    // NonOverlappedAllocaSet. If it is true, insert the alloca to that
    // NonOverlappedAllocaSet.
    for (auto &AllocaSet : NonOverlapedAllocas) {
      assert(!AllocaSet.empty() && "Processing Alloca Set is not empty.\n");
      bool NoInference = none_of(AllocaSet, [&](auto Iter) {
        return IsAllocaInferenre(Alloca, Iter);
      });
      // If the alignment of A is multiple of the alignment of B, the address
      // of A should satisfy the requirement for aligning for B.
      //
      // There may be other more fine-grained strategies to handle the alignment
      // infomation during the merging process. But it seems hard to handle
      // these strategies and benefit little.
      bool Alignable = [&]() -> bool {
        auto *LargestAlloca = *AllocaSet.begin();
        return LargestAlloca->getAlign().value() % Alloca->getAlign().value() ==
               0;
      }();
      bool CouldMerge = NoInference && Alignable;
      if (!CouldMerge)
        continue;
      AllocaSet.push_back(Alloca);
      Merged = true;
      break;
    }
    if (!Merged) {
      NonOverlapedAllocas.emplace_back(AllocaSetType(1, Alloca));
    }
  }
  // Recover the default target destination for each Switch statement
  // reserved.
  for (auto SwitchAndDefaultDest : DefaultSuspendDest) {
    SwitchInst *SWI = SwitchAndDefaultDest.first;
    BasicBlock *DestBB = SwitchAndDefaultDest.second;
    SWI->setDefaultDest(DestBB);
  }
  // This Debug Info could tell us which allocas are merged into one slot.
  LLVM_DEBUG(for (auto &AllocaSet
                  : NonOverlapedAllocas) {
    if (AllocaSet.size() > 1) {
      dbgs() << "In Function:" << F.getName() << "\n";
      dbgs() << "Find Union Set "
             << "\n";
      dbgs() << "\tAllocas are \n";
      for (auto Alloca : AllocaSet)
        dbgs() << "\t\t" << *Alloca << "\n";
    }
  });
}

void FrameTypeBuilder::finish(StructType *Ty) {
  assert(!IsFinished && "already finished!");

  // Prepare the optimal-layout field array.
  // The Id in the layout field is a pointer to our Field for it.
  SmallVector<OptimizedStructLayoutField, 8> LayoutFields;
  LayoutFields.reserve(Fields.size());
  for (auto &Field : Fields) {
    LayoutFields.emplace_back(&Field, Field.Size, Field.Alignment,
                              Field.Offset);
  }

  // Perform layout.
  auto SizeAndAlign = performOptimizedStructLayout(LayoutFields);
  StructSize = SizeAndAlign.first;
  StructAlign = SizeAndAlign.second;

  auto getField = [](const OptimizedStructLayoutField &LayoutField) -> Field & {
    return *static_cast<Field *>(const_cast<void*>(LayoutField.Id));
  };

  // We need to produce a packed struct type if there's a field whose
  // assigned offset isn't a multiple of its natural type alignment.
  bool Packed = [&] {
    for (auto &LayoutField : LayoutFields) {
      auto &F = getField(LayoutField);
      if (!isAligned(F.TyAlignment, LayoutField.Offset))
        return true;
    }
    return false;
  }();

  // Build the struct body.
  SmallVector<Type*, 16> FieldTypes;
  FieldTypes.reserve(LayoutFields.size() * 3 / 2);
  uint64_t LastOffset = 0;
  for (auto &LayoutField : LayoutFields) {
    auto &F = getField(LayoutField);

    auto Offset = LayoutField.Offset;

    // Add a padding field if there's a padding gap and we're either
    // building a packed struct or the padding gap is more than we'd
    // get from aligning to the field type's natural alignment.
    assert(Offset >= LastOffset);
    if (Offset != LastOffset) {
      if (Packed || alignTo(LastOffset, F.TyAlignment) != Offset)
        FieldTypes.push_back(ArrayType::get(Type::getInt8Ty(Context),
                                            Offset - LastOffset));
    }

    F.Offset = Offset;
    F.LayoutFieldIndex = FieldTypes.size();

    FieldTypes.push_back(F.Ty);
    LastOffset = Offset + F.Size;
  }

  Ty->setBody(FieldTypes, Packed);

#ifndef NDEBUG
  // Check that the IR layout matches the offsets we expect.
  auto Layout = DL.getStructLayout(Ty);
  for (auto &F : Fields) {
    assert(Ty->getElementType(F.LayoutFieldIndex) == F.Ty);
    assert(Layout->getElementOffset(F.LayoutFieldIndex) == F.Offset);
  }
#endif

  IsFinished = true;
}

// Build a struct that will keep state for an active coroutine.
//   struct f.frame {
//     ResumeFnTy ResumeFnAddr;
//     ResumeFnTy DestroyFnAddr;
//     int ResumeIndex;
//     ... promise (if present) ...
//     ... spills ...
//   };
static StructType *buildFrameType(Function &F, coro::Shape &Shape,
                                  FrameDataInfo &FrameData) {
  LLVMContext &C = F.getContext();
  const DataLayout &DL = F.getParent()->getDataLayout();
  StructType *FrameTy = [&] {
    SmallString<32> Name(F.getName());
    Name.append(".Frame");
    return StructType::create(C, Name);
  }();

  FrameTypeBuilder B(C, DL);

  AllocaInst *PromiseAlloca = Shape.getPromiseAlloca();
  Optional<FieldIDType> SwitchIndexFieldId;

  if (Shape.ABI == coro::ABI::Switch) {
    auto *FramePtrTy = FrameTy->getPointerTo();
    auto *FnTy = FunctionType::get(Type::getVoidTy(C), FramePtrTy,
                                   /*IsVarArg=*/false);
    auto *FnPtrTy = FnTy->getPointerTo();

    // Add header fields for the resume and destroy functions.
    // We can rely on these being perfectly packed.
    (void)B.addField(FnPtrTy, None, /*header*/ true);
    (void)B.addField(FnPtrTy, None, /*header*/ true);

    // PromiseAlloca field needs to be explicitly added here because it's
    // a header field with a fixed offset based on its alignment. Hence it
    // needs special handling and cannot be added to FrameData.Allocas.
    if (PromiseAlloca)
      FrameData.setFieldIndex(
          PromiseAlloca, B.addFieldForAlloca(PromiseAlloca, /*header*/ true));

    // Add a field to store the suspend index.  This doesn't need to
    // be in the header.
    unsigned IndexBits = std::max(1U, Log2_64_Ceil(Shape.CoroSuspends.size()));
    Type *IndexType = Type::getIntNTy(C, IndexBits);

    SwitchIndexFieldId = B.addField(IndexType, None);
  } else {
    assert(PromiseAlloca == nullptr && "lowering doesn't support promises");
  }

  // Because multiple allocas may own the same field slot,
  // we add allocas to field here.
  B.addFieldForAllocas(F, FrameData, Shape);
  // Add PromiseAlloca to Allocas list so that
  // 1. updateLayoutIndex could update its index after
  // `performOptimizedStructLayout`
  // 2. it is processed in insertSpills.
  if (Shape.ABI == coro::ABI::Switch && PromiseAlloca)
    // We assume that the promise alloca won't be modified before
    // CoroBegin and no alias will be create before CoroBegin.
    FrameData.Allocas.emplace_back(
        PromiseAlloca, DenseMap<Instruction *, llvm::Optional<APInt>>{}, false);
  // Create an entry for every spilled value.
  for (auto &S : FrameData.Spills) {
    FieldIDType Id = B.addField(S.first->getType(), None);
    FrameData.setFieldIndex(S.first, Id);
  }

  B.finish(FrameTy);
  FrameData.updateLayoutIndex(B);
  Shape.FrameAlign = B.getStructAlign();
  Shape.FrameSize = B.getStructSize();

  switch (Shape.ABI) {
  case coro::ABI::Switch:
    // In the switch ABI, remember the switch-index field.
    Shape.SwitchLowering.IndexField =
        B.getLayoutFieldIndex(*SwitchIndexFieldId);

    // Also round the frame size up to a multiple of its alignment, as is
    // generally expected in C/C++.
    Shape.FrameSize = alignTo(Shape.FrameSize, Shape.FrameAlign);
    break;

  // In the retcon ABI, remember whether the frame is inline in the storage.
  case coro::ABI::Retcon:
  case coro::ABI::RetconOnce: {
    auto Id = Shape.getRetconCoroId();
    Shape.RetconLowering.IsFrameInlineInStorage
      = (B.getStructSize() <= Id->getStorageSize() &&
         B.getStructAlign() <= Id->getStorageAlignment());
    break;
  }
  case coro::ABI::Async: {
    Shape.AsyncLowering.FrameOffset =
        alignTo(Shape.AsyncLowering.ContextHeaderSize, Shape.FrameAlign);
    // Also make the final context size a multiple of the context alignment to
    // make allocation easier for allocators.
    Shape.AsyncLowering.ContextSize =
        alignTo(Shape.AsyncLowering.FrameOffset + Shape.FrameSize,
                Shape.AsyncLowering.getContextAlignment());
    if (Shape.AsyncLowering.getContextAlignment() < Shape.FrameAlign) {
      report_fatal_error(
          "The alignment requirment of frame variables cannot be higher than "
          "the alignment of the async function context");
    }
    break;
  }
  }

  return FrameTy;
}

// We use a pointer use visitor to track how an alloca is being used.
// The goal is to be able to answer the following three questions:
// 1. Should this alloca be allocated on the frame instead.
// 2. Could the content of the alloca be modified prior to CoroBegn, which would
// require copying the data from alloca to the frame after CoroBegin.
// 3. Is there any alias created for this alloca prior to CoroBegin, but used
// after CoroBegin. In that case, we will need to recreate the alias after
// CoroBegin based off the frame. To answer question 1, we track two things:
//   a. List of all BasicBlocks that use this alloca or any of the aliases of
//   the alloca. In the end, we check if there exists any two basic blocks that
//   cross suspension points. If so, this alloca must be put on the frame. b.
//   Whether the alloca or any alias of the alloca is escaped at some point,
//   either by storing the address somewhere, or the address is used in a
//   function call that might capture. If it's ever escaped, this alloca must be
//   put on the frame conservatively.
// To answer quetion 2, we track through the variable MayWriteBeforeCoroBegin.
// Whenever a potential write happens, either through a store instruction, a
// function call or any of the memory intrinsics, we check whether this
// instruction is prior to CoroBegin. To answer question 3, we track the offsets
// of all aliases created for the alloca prior to CoroBegin but used after
// CoroBegin. llvm::Optional is used to be able to represent the case when the
// offset is unknown (e.g. when you have a PHINode that takes in different
// offset values). We cannot handle unknown offsets and will assert. This is the
// potential issue left out. An ideal solution would likely require a
// significant redesign.
namespace {
struct AllocaUseVisitor : PtrUseVisitor<AllocaUseVisitor> {
  using Base = PtrUseVisitor<AllocaUseVisitor>;
  AllocaUseVisitor(const DataLayout &DL, const DominatorTree &DT,
                   const CoroBeginInst &CB, const SuspendCrossingInfo &Checker)
      : PtrUseVisitor(DL), DT(DT), CoroBegin(CB), Checker(Checker) {}

  void visit(Instruction &I) {
    Users.insert(&I);
    Base::visit(I);
    // If the pointer is escaped prior to CoroBegin, we have to assume it would
    // be written into before CoroBegin as well.
    if (PI.isEscaped() && !DT.dominates(&CoroBegin, PI.getEscapingInst())) {
      MayWriteBeforeCoroBegin = true;
    }
  }
  // We need to provide this overload as PtrUseVisitor uses a pointer based
  // visiting function.
  void visit(Instruction *I) { return visit(*I); }

  void visitPHINode(PHINode &I) {
    enqueueUsers(I);
    handleAlias(I);
  }

  void visitSelectInst(SelectInst &I) {
    enqueueUsers(I);
    handleAlias(I);
  }

  void visitStoreInst(StoreInst &SI) {
    // Regardless whether the alias of the alloca is the value operand or the
    // pointer operand, we need to assume the alloca is been written.
    handleMayWrite(SI);

    if (SI.getValueOperand() != U->get())
      return;

    // We are storing the pointer into a memory location, potentially escaping.
    // As an optimization, we try to detect simple cases where it doesn't
    // actually escape, for example:
    //   %ptr = alloca ..
    //   %addr = alloca ..
    //   store %ptr, %addr
    //   %x = load %addr
    //   ..
    // If %addr is only used by loading from it, we could simply treat %x as
    // another alias of %ptr, and not considering %ptr being escaped.
    auto IsSimpleStoreThenLoad = [&]() {
      auto *AI = dyn_cast<AllocaInst>(SI.getPointerOperand());
      // If the memory location we are storing to is not an alloca, it
      // could be an alias of some other memory locations, which is difficult
      // to analyze.
      if (!AI)
        return false;
      // StoreAliases contains aliases of the memory location stored into.
      SmallVector<Instruction *, 4> StoreAliases = {AI};
      while (!StoreAliases.empty()) {
        Instruction *I = StoreAliases.pop_back_val();
        for (User *U : I->users()) {
          // If we are loading from the memory location, we are creating an
          // alias of the original pointer.
          if (auto *LI = dyn_cast<LoadInst>(U)) {
            enqueueUsers(*LI);
            handleAlias(*LI);
            continue;
          }
          // If we are overriding the memory location, the pointer certainly
          // won't escape.
          if (auto *S = dyn_cast<StoreInst>(U))
            if (S->getPointerOperand() == I)
              continue;
          if (auto *II = dyn_cast<IntrinsicInst>(U))
            if (II->isLifetimeStartOrEnd())
              continue;
          // BitCastInst creats aliases of the memory location being stored
          // into.
          if (auto *BI = dyn_cast<BitCastInst>(U)) {
            StoreAliases.push_back(BI);
            continue;
          }
          return false;
        }
      }

      return true;
    };

    if (!IsSimpleStoreThenLoad())
      PI.setEscaped(&SI);
  }

  // All mem intrinsics modify the data.
  void visitMemIntrinsic(MemIntrinsic &MI) { handleMayWrite(MI); }

  void visitBitCastInst(BitCastInst &BC) {
    Base::visitBitCastInst(BC);
    handleAlias(BC);
  }

  void visitAddrSpaceCastInst(AddrSpaceCastInst &ASC) {
    Base::visitAddrSpaceCastInst(ASC);
    handleAlias(ASC);
  }

  void visitGetElementPtrInst(GetElementPtrInst &GEPI) {
    // The base visitor will adjust Offset accordingly.
    Base::visitGetElementPtrInst(GEPI);
    handleAlias(GEPI);
  }

  void visitIntrinsicInst(IntrinsicInst &II) {
    if (II.getIntrinsicID() != Intrinsic::lifetime_start)
      return Base::visitIntrinsicInst(II);
    LifetimeStarts.insert(&II);
  }

  void visitCallBase(CallBase &CB) {
    for (unsigned Op = 0, OpCount = CB.getNumArgOperands(); Op < OpCount; ++Op)
      if (U->get() == CB.getArgOperand(Op) && !CB.doesNotCapture(Op))
        PI.setEscaped(&CB);
    handleMayWrite(CB);
  }

  bool getShouldLiveOnFrame() const {
    if (!ShouldLiveOnFrame)
      ShouldLiveOnFrame = computeShouldLiveOnFrame();
    return ShouldLiveOnFrame.getValue();
  }

  bool getMayWriteBeforeCoroBegin() const { return MayWriteBeforeCoroBegin; }

  DenseMap<Instruction *, llvm::Optional<APInt>> getAliasesCopy() const {
    assert(getShouldLiveOnFrame() && "This method should only be called if the "
                                     "alloca needs to live on the frame.");
    for (const auto &P : AliasOffetMap)
      if (!P.second)
        report_fatal_error("Unable to handle an alias with unknown offset "
                           "created before CoroBegin.");
    return AliasOffetMap;
  }

private:
  const DominatorTree &DT;
  const CoroBeginInst &CoroBegin;
  const SuspendCrossingInfo &Checker;
  // All alias to the original AllocaInst, created before CoroBegin and used
  // after CoroBegin. Each entry contains the instruction and the offset in the
  // original Alloca. They need to be recreated after CoroBegin off the frame.
  DenseMap<Instruction *, llvm::Optional<APInt>> AliasOffetMap{};
  SmallPtrSet<Instruction *, 4> Users{};
  SmallPtrSet<IntrinsicInst *, 2> LifetimeStarts{};
  bool MayWriteBeforeCoroBegin{false};

  mutable llvm::Optional<bool> ShouldLiveOnFrame{};

  bool computeShouldLiveOnFrame() const {
    // If lifetime information is available, we check it first since it's
    // more precise. We look at every pair of lifetime.start intrinsic and
    // every basic block that uses the pointer to see if they cross suspension
    // points. The uses cover both direct uses as well as indirect uses.
    if (!LifetimeStarts.empty()) {
      for (auto *I : Users)
        for (auto *S : LifetimeStarts)
          if (Checker.isDefinitionAcrossSuspend(*S, I))
            return true;
      return false;
    }
    // FIXME: Ideally the isEscaped check should come at the beginning.
    // However there are a few loose ends that need to be fixed first before
    // we can do that. We need to make sure we are not over-conservative, so
    // that the data accessed in-between await_suspend and symmetric transfer
    // is always put on the stack, and also data accessed after coro.end is
    // always put on the stack (esp the return object). To fix that, we need
    // to:
    //  1) Potentially treat sret as nocapture in calls
    //  2) Special handle the return object and put it on the stack
    //  3) Utilize lifetime.end intrinsic
    if (PI.isEscaped())
      return true;

    for (auto *U1 : Users)
      for (auto *U2 : Users)
        if (Checker.isDefinitionAcrossSuspend(*U1, U2))
          return true;

    return false;
  }

  void handleMayWrite(const Instruction &I) {
    if (!DT.dominates(&CoroBegin, &I))
      MayWriteBeforeCoroBegin = true;
  }

  bool usedAfterCoroBegin(Instruction &I) {
    for (auto &U : I.uses())
      if (DT.dominates(&CoroBegin, U))
        return true;
    return false;
  }

  void handleAlias(Instruction &I) {
    // We track all aliases created prior to CoroBegin but used after.
    // These aliases may need to be recreated after CoroBegin if the alloca
    // need to live on the frame.
    if (DT.dominates(&CoroBegin, &I) || !usedAfterCoroBegin(I))
      return;

    if (!IsOffsetKnown) {
      AliasOffetMap[&I].reset();
    } else {
      auto Itr = AliasOffetMap.find(&I);
      if (Itr == AliasOffetMap.end()) {
        AliasOffetMap[&I] = Offset;
      } else if (Itr->second.hasValue() && Itr->second.getValue() != Offset) {
        // If we have seen two different possible values for this alias, we set
        // it to empty.
        AliasOffetMap[&I].reset();
      }
    }
  }
};
} // namespace

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
static Instruction *insertSpills(const FrameDataInfo &FrameData,
                                 coro::Shape &Shape) {
  auto *CB = Shape.CoroBegin;
  LLVMContext &C = CB->getContext();
  IRBuilder<> Builder(CB->getNextNode());
  StructType *FrameTy = Shape.FrameTy;
  PointerType *FramePtrTy = FrameTy->getPointerTo();
  auto *FramePtr =
      cast<Instruction>(Builder.CreateBitCast(CB, FramePtrTy, "FramePtr"));
  DominatorTree DT(*CB->getFunction());
  SmallDenseMap<llvm::Value *, llvm::AllocaInst *, 4> DbgPtrAllocaCache;

  // Create a GEP with the given index into the coroutine frame for the original
  // value Orig. Appends an extra 0 index for array-allocas, preserving the
  // original type.
  auto GetFramePointer = [&](Value *Orig) -> Value * {
    FieldIDType Index = FrameData.getFieldIndex(Orig);
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

    auto GEP = cast<GetElementPtrInst>(
        Builder.CreateInBoundsGEP(FrameTy, FramePtr, Indices));
    if (isa<AllocaInst>(Orig)) {
      // If the type of GEP is not equal to the type of AllocaInst, it implies
      // that the AllocaInst may be reused in the Frame slot of other
      // AllocaInst. So We cast GEP to the AllocaInst here to re-use
      // the Frame storage.
      //
      // Note: If we change the strategy dealing with alignment, we need to refine
      // this casting.
      if (GEP->getResultElementType() != Orig->getType())
        return Builder.CreateBitCast(GEP, Orig->getType(),
                                     Orig->getName() + Twine(".cast"));
    }
    return GEP;
  };

  for (auto const &E : FrameData.Spills) {
    Value *Def = E.first;
    // Create a store instruction storing the value into the
    // coroutine frame.
    Instruction *InsertPt = nullptr;
    if (auto *Arg = dyn_cast<Argument>(Def)) {
      // For arguments, we will place the store instruction right after
      // the coroutine frame pointer instruction, i.e. bitcast of
      // coro.begin from i8* to %f.frame*.
      InsertPt = FramePtr->getNextNode();

      // If we're spilling an Argument, make sure we clear 'nocapture'
      // from the coroutine function.
      Arg->getParent()->removeParamAttr(Arg->getArgNo(), Attribute::NoCapture);

    } else if (auto *CSI = dyn_cast<AnyCoroSuspendInst>(Def)) {
      // Don't spill immediately after a suspend; splitting assumes
      // that the suspend will be followed by a branch.
      InsertPt = CSI->getParent()->getSingleSuccessor()->getFirstNonPHI();
    } else {
      auto *I = cast<Instruction>(Def);
      if (!DT.dominates(CB, I)) {
        // If it is not dominated by CoroBegin, then spill should be
        // inserted immediately after CoroFrame is computed.
        InsertPt = FramePtr->getNextNode();
      } else if (auto *II = dyn_cast<InvokeInst>(I)) {
        // If we are spilling the result of the invoke instruction, split
        // the normal edge and insert the spill in the new block.
        auto *NewBB = SplitEdge(II->getParent(), II->getNormalDest());
        InsertPt = NewBB->getTerminator();
      } else if (isa<PHINode>(I)) {
        // Skip the PHINodes and EH pads instructions.
        BasicBlock *DefBlock = I->getParent();
        if (auto *CSI = dyn_cast<CatchSwitchInst>(DefBlock->getTerminator()))
          InsertPt = splitBeforeCatchSwitch(CSI);
        else
          InsertPt = &*DefBlock->getFirstInsertionPt();
      } else {
        assert(!I->isTerminator() && "unexpected terminator");
        // For all other values, the spill is placed immediately after
        // the definition.
        InsertPt = I->getNextNode();
      }
    }

    auto Index = FrameData.getFieldIndex(Def);
    Builder.SetInsertPoint(InsertPt);
    auto *G = Builder.CreateConstInBoundsGEP2_32(
        FrameTy, FramePtr, 0, Index, Def->getName() + Twine(".spill.addr"));
    Builder.CreateStore(Def, G);

    BasicBlock *CurrentBlock = nullptr;
    Value *CurrentReload = nullptr;
    for (auto *U : E.second) {
      // If we have not seen the use block, create a load instruction to reload
      // the spilled value from the coroutine frame. Populates the Value pointer
      // reference provided with the frame GEP.
      if (CurrentBlock != U->getParent()) {
        CurrentBlock = U->getParent();
        Builder.SetInsertPoint(&*CurrentBlock->getFirstInsertionPt());

        auto *GEP = GetFramePointer(E.first);
        GEP->setName(E.first->getName() + Twine(".reload.addr"));
        CurrentReload = Builder.CreateLoad(
            FrameTy->getElementType(FrameData.getFieldIndex(E.first)), GEP,
            E.first->getName() + Twine(".reload"));

        TinyPtrVector<DbgDeclareInst *> DIs = FindDbgDeclareUses(Def);
        for (DbgDeclareInst *DDI : DIs) {
          bool AllowUnresolved = false;
          // This dbg.declare is preserved for all coro-split function
          // fragments. It will be unreachable in the main function, and
          // processed by coro::salvageDebugInfo() by CoroCloner.
          DIBuilder(*CurrentBlock->getParent()->getParent(), AllowUnresolved)
              .insertDeclare(CurrentReload, DDI->getVariable(),
                             DDI->getExpression(), DDI->getDebugLoc(),
                             &*Builder.GetInsertPoint());
          // This dbg.declare is for the main function entry point.  It
          // will be deleted in all coro-split functions.
          coro::salvageDebugInfo(DbgPtrAllocaCache, DDI, Shape.ReuseFrameSlot);
        }
      }

      // If we have a single edge PHINode, remove it and replace it with a
      // reload from the coroutine frame. (We already took care of multi edge
      // PHINodes by rewriting them in the rewritePHIs function).
      if (auto *PN = dyn_cast<PHINode>(U)) {
        assert(PN->getNumIncomingValues() == 1 &&
               "unexpected number of incoming "
               "values in the PHINode");
        PN->replaceAllUsesWith(CurrentReload);
        PN->eraseFromParent();
        continue;
      }

      // Replace all uses of CurrentValue in the current instruction with
      // reload.
      U->replaceUsesOfWith(Def, CurrentReload);
    }
  }

  BasicBlock *FramePtrBB = FramePtr->getParent();

  auto SpillBlock =
      FramePtrBB->splitBasicBlock(FramePtr->getNextNode(), "AllocaSpillBB");
  SpillBlock->splitBasicBlock(&SpillBlock->front(), "PostSpill");
  Shape.AllocaSpillBlock = SpillBlock;

  // retcon and retcon.once lowering assumes all uses have been sunk.
  if (Shape.ABI == coro::ABI::Retcon || Shape.ABI == coro::ABI::RetconOnce ||
      Shape.ABI == coro::ABI::Async) {
    // If we found any allocas, replace all of their remaining uses with Geps.
    Builder.SetInsertPoint(&SpillBlock->front());
    for (const auto &P : FrameData.Allocas) {
      AllocaInst *Alloca = P.Alloca;
      auto *G = GetFramePointer(Alloca);

      // We are not using ReplaceInstWithInst(P.first, cast<Instruction>(G))
      // here, as we are changing location of the instruction.
      G->takeName(Alloca);
      Alloca->replaceAllUsesWith(G);
      Alloca->eraseFromParent();
    }
    return FramePtr;
  }

  // If we found any alloca, replace all of their remaining uses with GEP
  // instructions. Because new dbg.declare have been created for these alloca,
  // we also delete the original dbg.declare and replace other uses with undef.
  // Note: We cannot replace the alloca with GEP instructions indiscriminately,
  // as some of the uses may not be dominated by CoroBegin.
  Builder.SetInsertPoint(&Shape.AllocaSpillBlock->front());
  SmallVector<Instruction *, 4> UsersToUpdate;
  for (const auto &A : FrameData.Allocas) {
    AllocaInst *Alloca = A.Alloca;
    UsersToUpdate.clear();
    for (User *U : Alloca->users()) {
      auto *I = cast<Instruction>(U);
      if (DT.dominates(CB, I))
        UsersToUpdate.push_back(I);
    }
    if (UsersToUpdate.empty())
      continue;
    auto *G = GetFramePointer(Alloca);
    G->setName(Alloca->getName() + Twine(".reload.addr"));

    TinyPtrVector<DbgDeclareInst *> DIs = FindDbgDeclareUses(Alloca);
    if (!DIs.empty())
      DIBuilder(*Alloca->getModule(),
                /*AllowUnresolved*/ false)
          .insertDeclare(G, DIs.front()->getVariable(),
                         DIs.front()->getExpression(),
                         DIs.front()->getDebugLoc(), DIs.front());
    for (auto *DI : FindDbgDeclareUses(Alloca))
      DI->eraseFromParent();
    replaceDbgUsesWithUndef(Alloca);

    for (Instruction *I : UsersToUpdate)
      I->replaceUsesOfWith(Alloca, G);
  }
  Builder.SetInsertPoint(FramePtr->getNextNode());
  for (const auto &A : FrameData.Allocas) {
    AllocaInst *Alloca = A.Alloca;
    if (A.MayWriteBeforeCoroBegin) {
      // isEscaped really means potentially modified before CoroBegin.
      if (Alloca->isArrayAllocation())
        report_fatal_error(
            "Coroutines cannot handle copying of array allocas yet");

      auto *G = GetFramePointer(Alloca);
      auto *Value = Builder.CreateLoad(Alloca->getAllocatedType(), Alloca);
      Builder.CreateStore(Value, G);
    }
    // For each alias to Alloca created before CoroBegin but used after
    // CoroBegin, we recreate them after CoroBegin by appplying the offset
    // to the pointer in the frame.
    for (const auto &Alias : A.Aliases) {
      auto *FramePtr = GetFramePointer(Alloca);
      auto *FramePtrRaw =
          Builder.CreateBitCast(FramePtr, Type::getInt8PtrTy(C));
      auto *AliasPtr = Builder.CreateGEP(
          FramePtrRaw,
          ConstantInt::get(Type::getInt64Ty(C), Alias.second.getValue()));
      auto *AliasPtrTyped =
          Builder.CreateBitCast(AliasPtr, Alias.first->getType());
      Alias.first->replaceUsesWithIf(
          AliasPtrTyped, [&](Use &U) { return DT.dominates(CB, U); });
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
                           BasicBlock *NewPred, PHINode *Until = nullptr) {
  unsigned BBIdx = 0;
  for (BasicBlock::iterator I = DestBB->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);

    // We manually update the LandingPadReplacement PHINode and it is the last
    // PHI Node. So, if we find it, we are done.
    if (Until == PN)
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

// Moves the values in the PHIs in SuccBB that correspong to PredBB into a new
// PHI in InsertedBB.
static void movePHIValuesToInsertedBlock(BasicBlock *SuccBB,
                                         BasicBlock *InsertedBB,
                                         BasicBlock *PredBB,
                                         PHINode *UntilPHI = nullptr) {
  auto *PN = cast<PHINode>(&SuccBB->front());
  do {
    int Index = PN->getBasicBlockIndex(InsertedBB);
    Value *V = PN->getIncomingValue(Index);
    PHINode *InputV = PHINode::Create(
        V->getType(), 1, V->getName() + Twine(".") + SuccBB->getName(),
        &InsertedBB->front());
    InputV->addIncoming(V, PredBB);
    PN->setIncomingValue(Index, InputV);
    PN = dyn_cast<PHINode>(PN->getNextNode());
  } while (PN != UntilPHI);
}

// Rewrites the PHI Nodes in a cleanuppad.
static void rewritePHIsForCleanupPad(BasicBlock *CleanupPadBB,
                                     CleanupPadInst *CleanupPad) {
  // For every incoming edge to a CleanupPad we will create a new block holding
  // all incoming values in single-value PHI nodes. We will then create another
  // block to act as a dispather (as all unwind edges for related EH blocks
  // must be the same).
  //
  // cleanuppad:
  //    %2 = phi i32[%0, %catchswitch], [%1, %catch.1]
  //    %3 = cleanuppad within none []
  //
  // It will create:
  //
  // cleanuppad.corodispatch
  //    %2 = phi i8[0, %catchswitch], [1, %catch.1]
  //    %3 = cleanuppad within none []
  //    switch i8 % 2, label %unreachable
  //            [i8 0, label %cleanuppad.from.catchswitch
  //             i8 1, label %cleanuppad.from.catch.1]
  // cleanuppad.from.catchswitch:
  //    %4 = phi i32 [%0, %catchswitch]
  //    br %label cleanuppad
  // cleanuppad.from.catch.1:
  //    %6 = phi i32 [%1, %catch.1]
  //    br %label cleanuppad
  // cleanuppad:
  //    %8 = phi i32 [%4, %cleanuppad.from.catchswitch],
  //                 [%6, %cleanuppad.from.catch.1]

  // Unreachable BB, in case switching on an invalid value in the dispatcher.
  auto *UnreachBB = BasicBlock::Create(
      CleanupPadBB->getContext(), "unreachable", CleanupPadBB->getParent());
  IRBuilder<> Builder(UnreachBB);
  Builder.CreateUnreachable();

  // Create a new cleanuppad which will be the dispatcher.
  auto *NewCleanupPadBB =
      BasicBlock::Create(CleanupPadBB->getContext(),
                         CleanupPadBB->getName() + Twine(".corodispatch"),
                         CleanupPadBB->getParent(), CleanupPadBB);
  Builder.SetInsertPoint(NewCleanupPadBB);
  auto *SwitchType = Builder.getInt8Ty();
  auto *SetDispatchValuePN =
      Builder.CreatePHI(SwitchType, pred_size(CleanupPadBB));
  CleanupPad->removeFromParent();
  CleanupPad->insertAfter(SetDispatchValuePN);
  auto *SwitchOnDispatch = Builder.CreateSwitch(SetDispatchValuePN, UnreachBB,
                                                pred_size(CleanupPadBB));

  int SwitchIndex = 0;
  SmallVector<BasicBlock *, 8> Preds(predecessors(CleanupPadBB));
  for (BasicBlock *Pred : Preds) {
    // Create a new cleanuppad and move the PHI values to there.
    auto *CaseBB = BasicBlock::Create(CleanupPadBB->getContext(),
                                      CleanupPadBB->getName() +
                                          Twine(".from.") + Pred->getName(),
                                      CleanupPadBB->getParent(), CleanupPadBB);
    updatePhiNodes(CleanupPadBB, Pred, CaseBB);
    CaseBB->setName(CleanupPadBB->getName() + Twine(".from.") +
                    Pred->getName());
    Builder.SetInsertPoint(CaseBB);
    Builder.CreateBr(CleanupPadBB);
    movePHIValuesToInsertedBlock(CleanupPadBB, CaseBB, NewCleanupPadBB);

    // Update this Pred to the new unwind point.
    setUnwindEdgeTo(Pred->getTerminator(), NewCleanupPadBB);

    // Setup the switch in the dispatcher.
    auto *SwitchConstant = ConstantInt::get(SwitchType, SwitchIndex);
    SetDispatchValuePN->addIncoming(SwitchConstant, Pred);
    SwitchOnDispatch->addCase(SwitchConstant, CaseBB);
    SwitchIndex++;
  }
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

  // Special case for CleanupPad: all EH blocks must have the same unwind edge
  // so we need to create an additional "dispatcher" block.
  if (auto *CleanupPad =
          dyn_cast_or_null<CleanupPadInst>(BB.getFirstNonPHI())) {
    SmallVector<BasicBlock *, 8> Preds(predecessors(&BB));
    for (BasicBlock *Pred : Preds) {
      if (CatchSwitchInst *CS =
              dyn_cast<CatchSwitchInst>(Pred->getTerminator())) {
        // CleanupPad with a CatchSwitch predecessor: therefore this is an
        // unwind destination that needs to be handle specially.
        assert(CS->getUnwindDest() == &BB);
        (void)CS;
        rewritePHIsForCleanupPad(&BB, CleanupPad);
        return;
      }
    }
  }

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

  SmallVector<BasicBlock *, 8> Preds(predecessors(&BB));
  for (BasicBlock *Pred : Preds) {
    auto *IncomingBB = ehAwareSplitEdge(Pred, &BB, LandingPad, ReplPHI);
    IncomingBB->setName(BB.getName() + Twine(".from.") + Pred->getName());

    // Stop the moving of values at ReplPHI, as this is either null or the PHI
    // that replaced the landing pad.
    movePHIValuesToInsertedBlock(&BB, IncomingBB, Pred, ReplPHI);
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
                                              const SpillInfo &Spills) {
  for (const auto &E : Spills) {
    Value *Def = E.first;
    BasicBlock *CurrentBlock = nullptr;
    Instruction *CurrentMaterialization = nullptr;
    for (Instruction *U : E.second) {
      // If we have not seen this block, materialize the value.
      if (CurrentBlock != U->getParent()) {
        CurrentBlock = U->getParent();
        CurrentMaterialization = cast<Instruction>(Def)->clone();
        CurrentMaterialization->setName(Def->getName());
        CurrentMaterialization->insertBefore(
            &*CurrentBlock->getFirstInsertionPt());
      }
      if (auto *PN = dyn_cast<PHINode>(U)) {
        assert(PN->getNumIncomingValues() == 1 &&
               "unexpected number of incoming "
               "values in the PHINode");
        PN->replaceAllUsesWith(CurrentMaterialization);
        PN->eraseFromParent();
        continue;
      }
      // Replace all uses of Def in the current instruction with the
      // CurrentMaterialization for the block.
      U->replaceUsesOfWith(Def, CurrentMaterialization);
    }
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
    Alloca->setAlignment(Align(AI->getAlignment()));

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

/// retcon and retcon.once conventions assume that all spill uses can be sunk
/// after the coro.begin intrinsic.
static void sinkSpillUsesAfterCoroBegin(Function &F,
                                        const FrameDataInfo &FrameData,
                                        CoroBeginInst *CoroBegin) {
  DominatorTree Dom(F);

  SmallSetVector<Instruction *, 32> ToMove;
  SmallVector<Instruction *, 32> Worklist;

  // Collect all users that precede coro.begin.
  for (auto *Def : FrameData.getAllDefs()) {
    for (User *U : Def->users()) {
      auto Inst = cast<Instruction>(U);
      if (Inst->getParent() != CoroBegin->getParent() ||
          Dom.dominates(CoroBegin, Inst))
        continue;
      if (ToMove.insert(Inst))
        Worklist.push_back(Inst);
    }
  }
  // Recursively collect users before coro.begin.
  while (!Worklist.empty()) {
    auto *Def = Worklist.pop_back_val();
    for (User *U : Def->users()) {
      auto Inst = cast<Instruction>(U);
      if (Dom.dominates(CoroBegin, Inst))
        continue;
      if (ToMove.insert(Inst))
        Worklist.push_back(Inst);
    }
  }

  // Sort by dominance.
  SmallVector<Instruction *, 64> InsertionList(ToMove.begin(), ToMove.end());
  llvm::sort(InsertionList, [&Dom](Instruction *A, Instruction *B) -> bool {
    // If a dominates b it should preceed (<) b.
    return Dom.dominates(A, B);
  });

  Instruction *InsertPt = CoroBegin->getNextNode();
  for (Instruction *Inst : InsertionList)
    Inst->moveBefore(InsertPt);
}

/// For each local variable that all of its user are only used inside one of
/// suspended region, we sink their lifetime.start markers to the place where
/// after the suspend block. Doing so minimizes the lifetime of each variable,
/// hence minimizing the amount of data we end up putting on the frame.
static void sinkLifetimeStartMarkers(Function &F, coro::Shape &Shape,
                                     SuspendCrossingInfo &Checker) {
  DominatorTree DT(F);

  // Collect all possible basic blocks which may dominate all uses of allocas.
  SmallPtrSet<BasicBlock *, 4> DomSet;
  DomSet.insert(&F.getEntryBlock());
  for (auto *CSI : Shape.CoroSuspends) {
    BasicBlock *SuspendBlock = CSI->getParent();
    assert(isSuspendBlock(SuspendBlock) && SuspendBlock->getSingleSuccessor() &&
           "should have split coro.suspend into its own block");
    DomSet.insert(SuspendBlock->getSingleSuccessor());
  }

  for (Instruction &I : instructions(F)) {
    AllocaInst* AI = dyn_cast<AllocaInst>(&I);
    if (!AI)
      continue;

    for (BasicBlock *DomBB : DomSet) {
      bool Valid = true;
      SmallVector<Instruction *, 1> Lifetimes;

      auto isLifetimeStart = [](Instruction* I) {
        if (auto* II = dyn_cast<IntrinsicInst>(I))
          return II->getIntrinsicID() == Intrinsic::lifetime_start;
        return false;
      };

      auto collectLifetimeStart = [&](Instruction *U, AllocaInst *AI) {
        if (isLifetimeStart(U)) {
          Lifetimes.push_back(U);
          return true;
        }
        if (!U->hasOneUse() || U->stripPointerCasts() != AI)
          return false;
        if (isLifetimeStart(U->user_back())) {
          Lifetimes.push_back(U->user_back());
          return true;
        }
        return false;
      };

      for (User *U : AI->users()) {
        Instruction *UI = cast<Instruction>(U);
        // For all users except lifetime.start markers, if they are all
        // dominated by one of the basic blocks and do not cross
        // suspend points as well, then there is no need to spill the
        // instruction.
        if (!DT.dominates(DomBB, UI->getParent()) ||
            Checker.isDefinitionAcrossSuspend(DomBB, UI)) {
          // Skip lifetime.start, GEP and bitcast used by lifetime.start
          // markers.
          if (collectLifetimeStart(UI, AI))
            continue;
          Valid = false;
          break;
        }
      }
      // Sink lifetime.start markers to dominate block when they are
      // only used outside the region.
      if (Valid && Lifetimes.size() != 0) {
        // May be AI itself, when the type of AI is i8*
        auto *NewBitCast = [&](AllocaInst *AI) -> Value* {
          if (isa<AllocaInst>(Lifetimes[0]->getOperand(1)))
            return AI;
          auto *Int8PtrTy = Type::getInt8PtrTy(F.getContext());
          return CastInst::Create(Instruction::BitCast, AI, Int8PtrTy, "",
                                  DomBB->getTerminator());
        }(AI);

        auto *NewLifetime = Lifetimes[0]->clone();
        NewLifetime->replaceUsesOfWith(NewLifetime->getOperand(1), NewBitCast);
        NewLifetime->insertBefore(DomBB->getTerminator());

        // All the outsided lifetime.start markers are no longer necessary.
        for (Instruction *S : Lifetimes)
          S->eraseFromParent();

        break;
      }
    }
  }
}

static void collectFrameAllocas(Function &F, coro::Shape &Shape,
                                const SuspendCrossingInfo &Checker,
                                SmallVectorImpl<AllocaInfo> &Allocas) {
  for (Instruction &I : instructions(F)) {
    auto *AI = dyn_cast<AllocaInst>(&I);
    if (!AI)
      continue;
    // The PromiseAlloca will be specially handled since it needs to be in a
    // fixed position in the frame.
    if (AI == Shape.SwitchLowering.PromiseAlloca) {
      continue;
    }
    DominatorTree DT(F);
    AllocaUseVisitor Visitor{F.getParent()->getDataLayout(), DT,
                             *Shape.CoroBegin, Checker};
    Visitor.visitPtr(*AI);
    if (!Visitor.getShouldLiveOnFrame())
      continue;
    Allocas.emplace_back(AI, Visitor.getAliasesCopy(),
                         Visitor.getMayWriteBeforeCoroBegin());
  }
}

void coro::salvageDebugInfo(
    SmallDenseMap<llvm::Value *, llvm::AllocaInst *, 4> &DbgPtrAllocaCache,
    DbgDeclareInst *DDI, bool ReuseFrameSlot) {
  Function *F = DDI->getFunction();
  IRBuilder<> Builder(F->getContext());
  auto InsertPt = F->getEntryBlock().getFirstInsertionPt();
  while (isa<IntrinsicInst>(InsertPt))
    ++InsertPt;
  Builder.SetInsertPoint(&F->getEntryBlock(), InsertPt);
  DIExpression *Expr = DDI->getExpression();
  // Follow the pointer arithmetic all the way to the incoming
  // function argument and convert into a DIExpression.
  bool OutermostLoad = true;
  Value *Storage = DDI->getAddress();
  Value *OriginalStorage = Storage;
  while (Storage) {
    if (auto *LdInst = dyn_cast<LoadInst>(Storage)) {
      Storage = LdInst->getOperand(0);
      // FIXME: This is a heuristic that works around the fact that
      // LLVM IR debug intrinsics cannot yet distinguish between
      // memory and value locations: Because a dbg.declare(alloca) is
      // implicitly a memory location no DW_OP_deref operation for the
      // last direct load from an alloca is necessary.  This condition
      // effectively drops the *last* DW_OP_deref in the expression.
      if (!OutermostLoad)
        Expr = DIExpression::prepend(Expr, DIExpression::DerefBefore);
      OutermostLoad = false;
    } else if (auto *StInst = dyn_cast<StoreInst>(Storage)) {
      Storage = StInst->getOperand(0);
    } else if (auto *GEPInst = dyn_cast<GetElementPtrInst>(Storage)) {
      Expr = llvm::salvageDebugInfoImpl(*GEPInst, Expr,
                                        /*WithStackValue=*/false, 0);
      if (!Expr)
        return;
      Storage = GEPInst->getOperand(0);
    } else if (auto *BCInst = dyn_cast<llvm::BitCastInst>(Storage))
      Storage = BCInst->getOperand(0);
    else
      break;
  }
  if (!Storage)
    return;

  // Store a pointer to the coroutine frame object in an alloca so it
  // is available throughout the function when producing unoptimized
  // code. Extending the lifetime this way is correct because the
  // variable has been declared by a dbg.declare intrinsic.
  //
  // Avoid to create the alloca would be eliminated by optimization
  // passes and the corresponding dbg.declares would be invalid.
  if (!ReuseFrameSlot && !EnableReuseStorageInFrame)
    if (auto *Arg = dyn_cast<llvm::Argument>(Storage)) {
      auto &Cached = DbgPtrAllocaCache[Storage];
      if (!Cached) {
        Cached = Builder.CreateAlloca(Storage->getType(), 0, nullptr,
                                      Arg->getName() + ".debug");
        Builder.CreateStore(Storage, Cached);
      }
      Storage = Cached;
      // FIXME: LLVM lacks nuanced semantics to differentiate between
      // memory and direct locations at the IR level. The backend will
      // turn a dbg.declare(alloca, ..., DIExpression()) into a memory
      // location. Thus, if there are deref and offset operations in the
      // expression, we need to add a DW_OP_deref at the *start* of the
      // expression to first load the contents of the alloca before
      // adjusting it with the expression.
      if (Expr && Expr->isComplex())
        Expr = DIExpression::prepend(Expr, DIExpression::DerefBefore);
    }
  DDI->replaceVariableLocationOp(OriginalStorage, Storage);
  DDI->setExpression(Expr);
  if (auto *InsertPt = dyn_cast<Instruction>(Storage))
    DDI->moveAfter(InsertPt);
  else if (isa<Argument>(Storage))
    DDI->moveAfter(F->getEntryBlock().getFirstNonPHI());
}

void coro::buildCoroutineFrame(Function &F, Shape &Shape) {
  // Don't eliminate swifterror in async functions that won't be split.
  if (Shape.ABI != coro::ABI::Async || !Shape.CoroSuspends.empty())
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
  for (AnyCoroEndInst *CE : Shape.CoroEnds) {
    splitAround(CE, "CoroEnd");

    // Emit the musttail call function in a new block before the CoroEnd.
    // We do this here so that the right suspend crossing info is computed for
    // the uses of the musttail call function call. (Arguments to the coro.end
    // instructions would be ignored)
    if (auto *AsyncEnd = dyn_cast<CoroAsyncEndInst>(CE)) {
      auto *MustTailCallFn = AsyncEnd->getMustTailCallFunction();
      if (!MustTailCallFn)
        continue;
      IRBuilder<> Builder(AsyncEnd);
      SmallVector<Value *, 8> Args(AsyncEnd->args());
      auto Arguments = ArrayRef<Value *>(Args).drop_front(3);
      auto *Call = createMustTailCall(AsyncEnd->getDebugLoc(), MustTailCallFn,
                                      Arguments, Builder);
      splitAround(Call, "MustTailCall.Before.CoroEnd");
    }
  }

  // Transforms multi-edge PHI Nodes, so that any value feeding into a PHI will
  // never has its definition separated from the PHI by the suspend point.
  rewritePHIs(F);

  // Build suspend crossing info.
  SuspendCrossingInfo Checker(F, Shape);

  IRBuilder<> Builder(F.getContext());
  FrameDataInfo FrameData;
  SmallVector<CoroAllocaAllocInst*, 4> LocalAllocas;
  SmallVector<Instruction*, 4> DeadInstructions;

  {
    SpillInfo Spills;
    for (int Repeat = 0; Repeat < 4; ++Repeat) {
      // See if there are materializable instructions across suspend points.
      for (Instruction &I : instructions(F))
        if (materializable(I))
          for (User *U : I.users())
            if (Checker.isDefinitionAcrossSuspend(I, U))
              Spills[&I].push_back(cast<Instruction>(U));

      if (Spills.empty())
        break;

      // Rewrite materializable instructions to be materialized at the use
      // point.
      LLVM_DEBUG(dumpSpills("Materializations", Spills));
      rewriteMaterializableInstructions(Builder, Spills);
      Spills.clear();
    }
  }

  sinkLifetimeStartMarkers(F, Shape, Checker);
  if (Shape.ABI != coro::ABI::Async || !Shape.CoroSuspends.empty())
    collectFrameAllocas(F, Shape, Checker, FrameData.Allocas);
  LLVM_DEBUG(dumpAllocas(FrameData.Allocas));

  // Collect the spills for arguments and other not-materializable values.
  for (Argument &A : F.args())
    for (User *U : A.users())
      if (Checker.isDefinitionAcrossSuspend(A, U))
        FrameData.Spills[&A].push_back(cast<Instruction>(U));

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
          FrameData.Spills[Alloc].push_back(cast<Instruction>(U));
      }
      continue;
    }

    // Ignore alloca.get; we process this as part of coro.alloca.alloc.
    if (isa<CoroAllocaGetInst>(I))
      continue;

    if (isa<AllocaInst>(I))
      continue;

    for (User *U : I.users())
      if (Checker.isDefinitionAcrossSuspend(I, U)) {
        // We cannot spill a token.
        if (I.getType()->isTokenTy())
          report_fatal_error(
              "token definition is separated from the use by a suspend point");
        FrameData.Spills[&I].push_back(cast<Instruction>(U));
      }
  }
  LLVM_DEBUG(dumpSpills("Spills", FrameData.Spills));
  if (Shape.ABI == coro::ABI::Retcon || Shape.ABI == coro::ABI::RetconOnce ||
      Shape.ABI == coro::ABI::Async)
    sinkSpillUsesAfterCoroBegin(F, FrameData, Shape.CoroBegin);
  Shape.FrameTy = buildFrameType(F, Shape, FrameData);
  Shape.FramePtr = insertSpills(FrameData, Shape);
  lowerLocalAllocas(LocalAllocas, DeadInstructions);

  for (auto I : DeadInstructions)
    I->eraseFromParent();
}
