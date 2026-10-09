//===- LoopDistribute.cpp - Loop Distribution Pass ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Loop Distribution Pass.  Its main focus is to
// distribute loops that cannot be vectorized due to dependence cycles.  It
// tries to isolate the offending dependences into a new loop allowing
// vectorization of the remaining parts.
//
// For dependence analysis, the pass uses the LoopVectorizer's
// LoopAccessAnalysis.  Because this analysis presumes no change in the order of
// memory operations, special care is taken to preserve the lexical order of
// these operations.
//
// Similarly to the Vectorizer, the pass also supports loop versioning to
// run-time disambiguate potentially overlapping arrays.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LoopDistribute.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/LoopVersioning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <cassert>
#include <list>
#include <tuple>

using namespace llvm;

#define LDIST_NAME "loop-distribute"
#define DEBUG_TYPE LDIST_NAME

/// @{
/// Metadata attribute names
static const char *const LLVMLoopDistributeFollowupAll =
    "llvm.loop.distribute.followup_all";
static const char *const LLVMLoopDistributeFollowupCoincident =
    "llvm.loop.distribute.followup_coincident";
static const char *const LLVMLoopDistributeFollowupSequential =
    "llvm.loop.distribute.followup_sequential";
static const char *const LLVMLoopDistributeFollowupFallback =
    "llvm.loop.distribute.followup_fallback";
/// @}

static cl::opt<bool>
    LDistVerify("loop-distribute-verify", cl::Hidden,
                cl::desc("Turn on DominatorTree and LoopInfo verification "
                         "after Loop Distribution"),
                cl::init(false));

static cl::opt<bool> DistributeNonIfConvertible(
    "loop-distribute-non-if-convertible", cl::Hidden,
    cl::desc("Whether to distribute into a loop that may not be "
             "if-convertible by the loop vectorizer"),
    cl::init(false));

static cl::opt<unsigned> DistributeSCEVCheckThreshold(
    "loop-distribute-scev-check-threshold", cl::init(8), cl::Hidden,
    cl::desc("The maximum number of SCEV checks allowed for Loop "
             "Distribution"));

static cl::opt<unsigned> PragmaDistributeSCEVCheckThreshold(
    "loop-distribute-scev-check-threshold-with-pragma", cl::init(128),
    cl::Hidden,
    cl::desc("The maximum number of SCEV checks allowed for Loop "
             "Distribution for loop marked with #pragma clang loop "
             "distribute(enable)"));

static cl::opt<bool> EnableLoopDistribute(
    "enable-loop-distribute", cl::Hidden,
    cl::desc("Enable the new, experimental LoopDistribution Pass"),
    cl::init(false));

// Experimental: seed partitions from each store's data-flow closure
// instead of from intervals delimited by unsafe LAA dependences.  The
// interval scheme over-merges when two stores share a backward sink
// but have independent feeders, defeating distribution.
static cl::opt<bool> StatementSeededPartitioning(
    "loop-distribute-statement-seeded", cl::Hidden,
    cl::desc("Form partitions from data-flow closures of stores"),
    cl::init(false));

static const char *DistributedMetaData = "llvm.loop.isdistributed";

STATISTIC(NumLoopsDistributed, "Number of loops distributed");

namespace {

/// Maintains the set of instructions of the loop for a partition before
/// cloning.  After cloning, it hosts the new loop.
class InstPartition {
  using InstructionSet = SmallSetVector<Instruction *, 8>;

public:
  InstPartition(Instruction *I, Loop *L, bool DepCycle = false)
      : DepCycle(DepCycle), OrigLoop(L) {
    Set.insert(I);
  }

  /// Returns whether this partition contains a dependence cycle.
  bool hasDepCycle() const { return DepCycle; }

  /// Adds an instruction to this partition.
  void add(Instruction *I) { Set.insert(I); }

  /// Collection accessors.
  InstructionSet::iterator begin() { return Set.begin(); }
  InstructionSet::iterator end() { return Set.end(); }
  InstructionSet::const_iterator begin() const { return Set.begin(); }
  InstructionSet::const_iterator end() const { return Set.end(); }
  bool empty() const { return Set.empty(); }

  /// Moves this partition into \p Other.  This partition becomes empty
  /// after this.
  void moveTo(InstPartition &Other) {
    Other.Set.insert_range(Set);
    Set.clear();
    Other.DepCycle |= DepCycle;
  }

  /// Populates the partition with a transitive closure of all the
  /// instructions that the seeded instructions dependent on.
  void populateUsedSet() {
    // FIXME: We currently don't use control-dependence but simply include all
    // blocks (possibly empty at the end) and let simplifycfg mostly clean this
    // up.
    for (auto *B : OrigLoop->getBlocks())
      Set.insert(B->getTerminator());

    // Follow the use-def chains to form a transitive closure of all the
    // instructions that the originally seeded instructions depend on.
    SmallVector<Instruction *, 8> Worklist(Set.begin(), Set.end());
    while (!Worklist.empty()) {
      Instruction *I = Worklist.pop_back_val();
      // Insert instructions from the loop that we depend on.
      for (Value *V : I->operand_values()) {
        auto *I = dyn_cast<Instruction>(V);
        if (I && OrigLoop->contains(I->getParent()) && Set.insert(I))
          Worklist.push_back(I);
      }
    }
  }

  /// Clones the original loop.
  ///
  /// Updates LoopInfo and DominatorTree using the information that block \p
  /// LoopDomBB dominates the loop.
  Loop *cloneLoopWithPreheader(BasicBlock *InsertBefore, BasicBlock *LoopDomBB,
                               unsigned Index, LoopInfo *LI,
                               DominatorTree *DT) {
    ClonedLoop = ::cloneLoopWithPreheader(InsertBefore, LoopDomBB, OrigLoop,
                                          VMap, Twine(".ldist") + Twine(Index),
                                          LI, DT, ClonedLoopBlocks);
    return ClonedLoop;
  }

  /// The cloned loop.  If this partition is mapped to the original loop,
  /// this is null.
  const Loop *getClonedLoop() const { return ClonedLoop; }

  /// Returns the loop where this partition ends up after distribution.
  /// If this partition is mapped to the original loop then use the block from
  /// the loop.
  Loop *getDistributedLoop() const {
    return ClonedLoop ? ClonedLoop : OrigLoop;
  }

  /// The VMap that is populated by cloning and then used in
  /// remapinstruction to remap the cloned instructions.
  ValueToValueMapTy &getVMap() { return VMap; }

  /// Remaps the cloned instructions using VMap.
  void remapInstructions() {
    remapInstructionsInBlocks(ClonedLoopBlocks, VMap);
  }

  /// Based on the set of instructions selected for this partition,
  /// removes the unnecessary ones.
  void removeUnusedInsts() {
    SmallVector<Instruction *, 8> Unused;

    for (auto *Block : OrigLoop->getBlocks())
      for (auto &Inst : *Block)
        if (!Set.count(&Inst)) {
          Instruction *NewInst = &Inst;
          if (!VMap.empty())
            NewInst = cast<Instruction>(VMap[NewInst]);

          assert((!isa<UncondBrInst, CondBrInst>(NewInst)) &&
                 "Branches are marked used early on");
          Unused.push_back(NewInst);
        }

    // Delete the instructions backwards, as it has a reduced likelihood of
    // having to update as many def-use and use-def chains.
    for (auto *Inst : reverse(Unused)) {
      salvageDebugInfo(*Inst);
      if (!Inst->use_empty())
        Inst->replaceAllUsesWith(PoisonValue::get(Inst->getType()));
      Inst->eraseFromParent();
    }
  }

  void print(raw_ostream &OS) const {
    OS << (DepCycle ? " (cycle)\n" : "\n");
    for (auto *I : Set)
      // Prefix with the block name.
      OS << "  " << I->getParent()->getName() << ":" << *I << "\n";
  }

  void printBlocks(raw_ostream &OS) const {
    for (auto *BB : getDistributedLoop()->getBlocks())
      OS << *BB;
  }

private:
  /// Instructions from OrigLoop selected for this partition.
  InstructionSet Set;

  /// Whether this partition contains a dependence cycle.
  bool DepCycle;

  /// The original loop.
  Loop *OrigLoop;

  /// The cloned loop.  If this partition is mapped to the original loop,
  /// this is null.
  Loop *ClonedLoop = nullptr;

  /// The blocks of ClonedLoop including the preheader.  If this
  /// partition is mapped to the original loop, this is empty.
  SmallVector<BasicBlock *, 8> ClonedLoopBlocks;

  /// These gets populated once the set of instructions have been
  /// finalized. If this partition is mapped to the original loop, these are not
  /// set.
  ValueToValueMapTy VMap;
};

/// Holds the set of Partitions.  It populates them, merges them and then
/// clones the loops.
class InstPartitionContainer {
  using InstToPartitionIdT = DenseMap<Instruction *, int>;

public:
  InstPartitionContainer(Loop *L, LoopInfo *LI, DominatorTree *DT)
      : L(L), LI(LI), DT(DT) {}

  /// Returns the number of partitions.
  unsigned getSize() const { return PartitionContainer.size(); }

  /// Adds \p Inst into the current partition if that is marked to
  /// contain cycles.  Otherwise start a new partition for it.
  void addToCyclicPartition(Instruction *Inst) {
    // If the current partition is non-cyclic.  Start a new one.
    if (PartitionContainer.empty() || !PartitionContainer.back().hasDepCycle())
      PartitionContainer.emplace_back(Inst, L, /*DepCycle=*/true);
    else
      PartitionContainer.back().add(Inst);
  }

  /// Adds \p Inst into a partition that is not marked to contain
  /// dependence cycles.
  ///
  //  Initially we isolate memory instructions into as many partitions as
  //  possible, then later we may merge them back together.
  void addToNewNonCyclicPartition(Instruction *Inst) {
    PartitionContainer.emplace_back(Inst, L);
  }

  /// Add a fresh partition pre-seeded with \p Insts.  Unlike
  /// addToCyclicPartition, this never extends the current trailing
  /// partition; the caller decides which seeds belong together.
  void addNewPartition(ArrayRef<Instruction *> Insts, bool DepCycle) {
    assert(!Insts.empty() && "empty partition");
    PartitionContainer.emplace_back(Insts.front(), L, DepCycle);
    for (Instruction *I : llvm::drop_begin(Insts))
      PartitionContainer.back().add(I);
  }

  /// Merges adjacent non-cyclic partitions.
  ///
  /// The idea is that we currently only want to isolate the non-vectorizable
  /// partition.  We could later allow more distribution among these partition
  /// too.
  void mergeAdjacentNonCyclic() {
    mergeAdjacentPartitionsIf(
        [](const InstPartition *P) { return !P->hasDepCycle(); });
  }

  /// If a partition contains only conditional stores, we won't vectorize
  /// it.  Try to merge it with a previous cyclic partition.
  void mergeNonIfConvertible() {
    mergeAdjacentPartitionsIf([&](const InstPartition *Partition) {
      if (Partition->hasDepCycle())
        return true;

      // Now, check if all stores are conditional in this partition.
      bool seenStore = false;

      for (auto *Inst : *Partition)
        if (isa<StoreInst>(Inst)) {
          seenStore = true;
          if (!LoopAccessInfo::blockNeedsPredication(Inst->getParent(), L, DT))
            return false;
        }
      return seenStore;
    });
  }

  /// Merges the partitions according to various heuristics.
  void mergeBeforePopulating() {
    mergeAdjacentNonCyclic();
    if (!DistributeNonIfConvertible)
      mergeNonIfConvertible();
  }

  /// Merges partitions in order to ensure that no loads are duplicated.
  ///
  /// We can't duplicate loads because that could potentially reorder them.
  /// LoopAccessAnalysis provides dependency information with the context that
  /// the order of memory operation is preserved.
  ///
  /// Return if any partitions were merged.
  bool mergeToAvoidDuplicatedLoads() {
    using LoadToPartitionT = DenseMap<Instruction *, InstPartition *>;
    using ToBeMergedT = EquivalenceClasses<InstPartition *>;

    LoadToPartitionT LoadToPartition;
    ToBeMergedT ToBeMerged;

    // Step through the partitions and create equivalence between partitions
    // that contain the same load.  Also put partitions in between them in the
    // same equivalence class to avoid reordering of memory operations.
    for (PartitionContainerT::iterator I = PartitionContainer.begin(),
                                       E = PartitionContainer.end();
         I != E; ++I) {
      auto *PartI = &*I;

      // If a load occurs in two partitions PartI and PartJ, merge all
      // partitions (PartI, PartJ] into PartI.
      for (Instruction *Inst : *PartI)
        if (isa<LoadInst>(Inst)) {
          bool NewElt;
          LoadToPartitionT::iterator LoadToPart;

          std::tie(LoadToPart, NewElt) =
              LoadToPartition.insert(std::make_pair(Inst, PartI));
          if (!NewElt) {
            LLVM_DEBUG(
                dbgs()
                << "LDist: Merging partitions due to this load in multiple "
                << "partitions: " << PartI << ", " << LoadToPart->second << "\n"
                << *Inst << "\n");

            auto PartJ = I;
            do {
              --PartJ;
              ToBeMerged.unionSets(PartI, &*PartJ);
            } while (&*PartJ != LoadToPart->second);
          }
        }
    }
    if (ToBeMerged.empty())
      return false;

    // Merge the member of an equivalence class into its class leader.  This
    // makes the members empty.
    for (const auto &C : ToBeMerged) {
      if (!C->isLeader())
        continue;

      auto PartI = C->getData();
      for (auto *PartJ : make_range(std::next(ToBeMerged.member_begin(*C)),
                                    ToBeMerged.member_end())) {
        PartJ->moveTo(*PartI);
      }
    }

    // Remove the empty partitions.
    PartitionContainer.remove_if(
        [](const InstPartition &P) { return P.empty(); });

    return true;
  }

  /// Sets up the mapping between instructions to partitions.  If the
  /// instruction is duplicated across multiple partitions, set the entry to -1.
  void setupPartitionIdOnInstructions() {
    int PartitionID = 0;
    for (const auto &Partition : PartitionContainer) {
      for (Instruction *Inst : Partition) {
        bool NewElt;
        InstToPartitionIdT::iterator Iter;

        std::tie(Iter, NewElt) =
            InstToPartitionId.insert(std::make_pair(Inst, PartitionID));
        if (!NewElt)
          Iter->second = -1;
      }
      ++PartitionID;
    }
  }

  /// Populates the partition with everything that the seeding
  /// instructions require.
  void populateUsedSet() {
    for (auto &P : PartitionContainer)
      P.populateUsedSet();
  }

  /// This performs the main chunk of the work of cloning the loops for
  /// the partitions.
  void cloneLoops() {
    BasicBlock *OrigPH = L->getLoopPreheader();
    // At this point the predecessor of the preheader is either the memcheck
    // block or the top part of the original preheader.
    BasicBlock *Pred = OrigPH->getSinglePredecessor();
    assert(Pred && "Preheader does not have a single predecessor");
    BasicBlock *ExitBlock = L->getExitBlock();
    assert(ExitBlock && "No single exit block");
    Loop *NewLoop;

    assert(!PartitionContainer.empty() && "at least two partitions expected");
    // We're cloning the preheader along with the loop so we already made sure
    // it was empty.
    assert(&*OrigPH->begin() == OrigPH->getTerminator() &&
           "preheader not empty");

    // Preserve the original loop ID for use after the transformation.
    MDNode *OrigLoopID = L->getLoopID();

    // Create a loop for each partition except the last.  Clone the original
    // loop before PH along with adding a preheader for the cloned loop.  Then
    // update PH to point to the newly added preheader.
    BasicBlock *TopPH = OrigPH;
    unsigned Index = getSize() - 1;
    for (auto &Part : llvm::drop_begin(llvm::reverse(PartitionContainer))) {
      NewLoop = Part.cloneLoopWithPreheader(TopPH, Pred, Index, LI, DT);

      Part.getVMap()[ExitBlock] = TopPH;
      Part.remapInstructions();
      setNewLoopID(OrigLoopID, &Part);
      --Index;
      TopPH = NewLoop->getLoopPreheader();
    }
    Pred->getTerminator()->replaceUsesOfWith(OrigPH, TopPH);

    // Also set a new loop ID for the last loop.
    setNewLoopID(OrigLoopID, &PartitionContainer.back());

    // Now go in forward order and update the immediate dominator for the
    // preheaders with the exiting block of the previous loop.  Dominance
    // within the loop is updated in cloneLoopWithPreheader.
    for (auto Curr = PartitionContainer.cbegin(),
              Next = std::next(PartitionContainer.cbegin()),
              E = PartitionContainer.cend();
         Next != E; ++Curr, ++Next)
      DT->changeImmediateDominator(
          Next->getDistributedLoop()->getLoopPreheader(),
          Curr->getDistributedLoop()->getExitingBlock());
  }

  /// Removes the dead instructions from the cloned loops.
  void removeUnusedInsts() {
    for (auto &Partition : PartitionContainer)
      Partition.removeUnusedInsts();
  }

  /// For each memory pointer, it computes the partitionId the pointer is
  /// used in.
  ///
  /// This returns an array of int where the I-th entry corresponds to I-th
  /// entry in LAI.getRuntimePointerCheck().  If the pointer is used in multiple
  /// partitions its entry is set to -1.
  SmallVector<int, 8>
  computePartitionSetForPointers(const LoopAccessInfo &LAI) {
    const RuntimePointerChecking *RtPtrCheck = LAI.getRuntimePointerChecking();

    unsigned N = RtPtrCheck->Pointers.size();
    SmallVector<int, 8> PtrToPartitions(N);
    for (unsigned I = 0; I < N; ++I) {
      Value *Ptr = RtPtrCheck->Pointers[I].PointerValue;
      auto Instructions = LAI.getInstructionsForAccess(Ptr, /* IsWrite */ true);
      auto ReadInstructions =
          LAI.getInstructionsForAccess(Ptr, /* IsWrite */ false);
      Instructions.append(ReadInstructions.begin(), ReadInstructions.end());

      int &Partition = PtrToPartitions[I];
      // First set it to uninitialized.
      Partition = -2;
      for (Instruction *Inst : Instructions) {
        // Note that this could be -1 if Inst is duplicated across multiple
        // partitions.
        int ThisPartition = this->InstToPartitionId[Inst];
        if (Partition == -2)
          Partition = ThisPartition;
        // -1 means belonging to multiple partitions.
        else if (Partition == -1)
          break;
        else if (Partition != ThisPartition)
          Partition = -1;
      }
      assert(Partition != -2 && "Pointer not belonging to any partition");
    }

    return PtrToPartitions;
  }

  void print(raw_ostream &OS) const {
    unsigned Index = 0;
    for (const auto &P : PartitionContainer) {
      OS << "LDist: Partition " << Index++ << ":";
      P.print(OS);
    }
  }

  void dump() const { print(dbgs()); }

#ifndef NDEBUG
  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const InstPartitionContainer &Partitions) {
    Partitions.print(OS);
    return OS;
  }
#endif

  void printBlocks(raw_ostream &OS) const {
    unsigned Index = 0;
    for (const auto &P : PartitionContainer) {
      OS << "LDist: Partition " << Index++ << ":";
      P.printBlocks(OS);
    }
  }

private:
  using PartitionContainerT = std::list<InstPartition>;

  /// List of partitions.
  PartitionContainerT PartitionContainer;

  /// Mapping from Instruction to partition Id.  If the instruction
  /// belongs to multiple partitions the entry contains -1.
  InstToPartitionIdT InstToPartitionId;

  Loop *L;
  LoopInfo *LI;
  DominatorTree *DT;

  /// The control structure to merge adjacent partitions if both satisfy
  /// the \p Predicate.
  template <class UnaryPredicate>
  void mergeAdjacentPartitionsIf(UnaryPredicate Predicate) {
    InstPartition *PrevMatch = nullptr;
    for (auto I = PartitionContainer.begin(); I != PartitionContainer.end();) {
      auto DoesMatch = Predicate(&*I);
      if (PrevMatch == nullptr && DoesMatch) {
        PrevMatch = &*I;
        ++I;
      } else if (PrevMatch != nullptr && DoesMatch) {
        I->moveTo(*PrevMatch);
        I = PartitionContainer.erase(I);
      } else {
        PrevMatch = nullptr;
        ++I;
      }
    }
  }

  /// Assign new LoopIDs for the partition's cloned loop.
  void setNewLoopID(MDNode *OrigLoopID, InstPartition *Part) {
    std::optional<MDNode *> PartitionID = makeFollowupLoopID(
        OrigLoopID,
        {LLVMLoopDistributeFollowupAll,
         Part->hasDepCycle() ? LLVMLoopDistributeFollowupSequential
                             : LLVMLoopDistributeFollowupCoincident});
    if (PartitionID) {
      Loop *NewLoop = Part->getDistributedLoop();
      NewLoop->setLoopID(*PartitionID);
    }
  }
};

/// For each memory instruction, this class maintains difference of the
/// number of unsafe dependences that start out from this instruction minus
/// those that end here.
///
/// By traversing the memory instructions in program order and accumulating this
/// number, we know whether any unsafe dependence crosses over a program point.
class MemoryInstructionDependences {
  using Dependence = MemoryDepChecker::Dependence;

public:
  struct Entry {
    Instruction *Inst;
    unsigned NumUnsafeDependencesStartOrEnd = 0;

    Entry(Instruction *Inst) : Inst(Inst) {}
  };

  using AccessesType = SmallVector<Entry, 8>;

  AccessesType::const_iterator begin() const { return Accesses.begin(); }
  AccessesType::const_iterator end() const { return Accesses.end(); }

  MemoryInstructionDependences(
      const SmallVectorImpl<Instruction *> &Instructions,
      const SmallVectorImpl<Dependence> &Dependences) {
    Accesses.append(Instructions.begin(), Instructions.end());

    LLVM_DEBUG(dbgs() << "LDist: Backward dependences:\n");
    for (const auto &Dep : Dependences)
      if (Dep.isPossiblyBackward()) {
        // Note that the designations source and destination follow the program
        // order, i.e. source is always first.  (The direction is given by the
        // DepType.)
        ++Accesses[Dep.Source].NumUnsafeDependencesStartOrEnd;
        --Accesses[Dep.Destination].NumUnsafeDependencesStartOrEnd;

        LLVM_DEBUG(Dep.print(dbgs(), 2, Instructions));
      }
  }

private:
  AccessesType Accesses;
};

/// The actual class performing the per-loop work.
class LoopDistributeForLoop {
public:
  LoopDistributeForLoop(Loop *L, Function *F, LoopInfo *LI, DominatorTree *DT,
                        ScalarEvolution *SE, LoopAccessInfoManager &LAIs,
                        OptimizationRemarkEmitter *ORE)
      : L(L), F(F), LI(LI), DT(DT), SE(SE), LAIs(LAIs), ORE(ORE) {
    setForced();
  }

  /// Try to distribute an inner-most loop.
  bool processLoop() {
    assert(L->isInnermost() && "Only process inner loops.");

    LLVM_DEBUG(dbgs() << "\nLDist: Checking a loop in '"
                      << L->getHeader()->getParent()->getName() << "' from "
                      << L->getLocStr() << "\n");

    // Having a single exit block implies there's also one exiting block.
    if (!L->getExitBlock())
      return fail("MultipleExitBlocks", "multiple exit blocks");
    if (!L->isLoopSimplifyForm())
      return fail("NotLoopSimplifyForm",
                  "loop is not in loop-simplify form");
    if (!L->isRotatedForm())
      return fail("NotBottomTested", "loop is not bottom tested");

    BasicBlock *PH = L->getLoopPreheader();

    LAI = &LAIs.getInfo(*L);

    // Currently, we only distribute to isolate the part of the loop with
    // dependence cycles to enable partial vectorization.
    if (LAI->canVectorizeMemory())
      return fail("MemOpsCanBeVectorized",
                  "memory operations are safe for vectorization");

    auto *Dependences = LAI->getDepChecker().getDependences();
    if (!Dependences || Dependences->empty())
      return fail("NoUnsafeDeps", "no unsafe dependences to isolate");

    LLVM_DEBUG(dbgs() << "LDist: Found a candidate loop: "
                      << L->getHeader()->getName() << "\n");

    InstPartitionContainer Partitions(L, LI, DT);

    // First, go through each memory operation and assign them to consecutive
    // partitions (the order of partitions follows program order).  Put those
    // with unsafe dependences into "cyclic" partition otherwise put each store
    // in its own "non-cyclic" partition (we'll merge these later).
    //
    // Note that a memory operation (e.g. Load2 below) at a program point that
    // has an unsafe dependence (Store3->Load1) spanning over it must be
    // included in the same cyclic partition as the dependent operations.  This
    // is to preserve the original program order after distribution.  E.g.:
    //
    //                NumUnsafeDependencesStartOrEnd  NumUnsafeDependencesActive
    //  Load1   -.                     1                       0->1
    //  Load2    | /Unsafe/            0                       1
    //  Store3  -'                    -1                       1->0
    //  Load4                          0                       0
    //
    // NumUnsafeDependencesActive > 0 indicates this situation and in this case
    // we just keep assigning to the same cyclic partition until
    // NumUnsafeDependencesActive reaches 0.
    const MemoryDepChecker &DepChecker = LAI->getDepChecker();
    MemoryInstructionDependences MID(DepChecker.getMemoryInstructions(),
                                     *Dependences);

    bool SeededByStatement = false;
    if (StatementSeededPartitioning) {
      // Every dependence constrains statement order, not just unsafe ones.
      // An unknown direction gets both orders, so splitting it is refused.
      SmallVector<OrderedMemPair, 4> OrderedPairs;
      for (const auto &Dep : *Dependences) {
        Instruction *Src = Dep.getSource(DepChecker);
        Instruction *Dst = Dep.getDestination(DepChecker);
        if (Dep.isForward()) {
          OrderedPairs.push_back({Src, Dst, /*Carried=*/false});
          continue;
        }
        OrderedPairs.push_back({Dst, Src, /*Carried=*/true});
        if (!Dep.isBackward())
          OrderedPairs.push_back({Src, Dst, /*Carried=*/true});
      }
      // LAA does not compare accesses through a pointer that is alone in
      // its alias set.  If the pointer moves past both accesses every
      // iteration they only overlap within an iteration.
      ArrayRef<Instruction *> MemInsts = DepChecker.getMemoryInstructions();
      const DataLayout &DL = L->getHeader()->getDataLayout();
      auto OverlapsOnlyInIteration = [&](Instruction *A, Instruction *B) {
        TypeSize SizeA = DL.getTypeStoreSize(getLoadStoreType(A));
        TypeSize SizeB = DL.getTypeStoreSize(getLoadStoreType(B));
        auto *AR = dyn_cast<SCEVAddRecExpr>(
            SE->getSCEV(getLoadStorePointerOperand(A)));
        if (SizeA.isScalable() || SizeB.isScalable() || !AR ||
            AR->getLoop() != L || AR->getNoWrapFlags() == SCEV::FlagNone)
          return false;
        auto *Step = dyn_cast<SCEVConstant>(AR->getStepRecurrence(*SE));
        return Step && Step->getAPInt().abs().uge(std::max(
                           SizeA.getFixedValue(), SizeB.getFixedValue()));
      };
      for (auto [Idx, A] : enumerate(MemInsts)) {
        for (Instruction *B : drop_begin(MemInsts, Idx + 1)) {
          if (getLoadStorePointerOperand(A) != getLoadStorePointerOperand(B) ||
              (isa<LoadInst>(A) && isa<LoadInst>(B)))
            continue;
          if (OverlapsOnlyInIteration(A, B)) {
            OrderedPairs.push_back({A, B, /*Carried=*/false});
            continue;
          }
          OrderedPairs.push_back({A, B, /*Carried=*/true});
          OrderedPairs.push_back({B, A, /*Carried=*/true});
        }
      }
      SeededByStatement =
          seedPartitionsFromPairs(Partitions, OrderedPairs, MemInsts);
    }

    if (!SeededByStatement) {
      int NumUnsafeDependencesActive = 0;
      for (const auto &InstDep : MID) {
        Instruction *I = InstDep.Inst;
        // We update NumUnsafeDependencesActive post-instruction, catch the
        // start of a dependence directly via NumUnsafeDependencesStartOrEnd.
        if (NumUnsafeDependencesActive ||
            InstDep.NumUnsafeDependencesStartOrEnd > 0)
          Partitions.addToCyclicPartition(I);
        else
          Partitions.addToNewNonCyclicPartition(I);
        NumUnsafeDependencesActive += InstDep.NumUnsafeDependencesStartOrEnd;
        assert(NumUnsafeDependencesActive >= 0 &&
               "Negative number of dependences active");
      }
    }

    // Add partitions for values used outside.  These partitions can be out of
    // order from the original program order.  This is OK because if the
    // partition uses a load we will merge this partition with the original
    // partition of the load that we set up in the previous loop (see
    // mergeToAvoidDuplicatedLoads).
    auto DefsUsedOutside = findDefsUsedOutsideOfLoop(L);
    for (auto *Inst : DefsUsedOutside)
      Partitions.addToNewNonCyclicPartition(Inst);

    LLVM_DEBUG(dbgs() << "LDist: Seeded partitions:\n" << Partitions);
    if (Partitions.getSize() < 2)
      return fail("CantIsolateUnsafeDeps",
                  "cannot isolate unsafe dependencies");

    // The statement-seeded path has already produced the partitions it
    // wants, including ordering edges between non-cyclic partitions, so
    // skip the merge heuristics that would coalesce them.
    if (!SeededByStatement) {
      // Run the merge heuristics: Merge non-cyclic adjacent partitions
      // since we should be able to vectorize these together.
      Partitions.mergeBeforePopulating();
      LLVM_DEBUG(dbgs() << "LDist: Merged partitions:\n" << Partitions);
      if (Partitions.getSize() < 2)
        return fail("CantIsolateUnsafeDeps",
                    "cannot isolate unsafe dependencies");
    }

    // Now, populate the partitions with non-memory operations.
    Partitions.populateUsedSet();
    LLVM_DEBUG(dbgs() << "LDist: Populated partitions:\n" << Partitions);

    // In order to preserve original lexical order for loads, keep them in the
    // partition that we set up in the MemoryInstructionDependences loop.
    // The statement-seeded path deliberately duplicates read-only shared
    // loads, so skip this merge in that mode.
    if (!SeededByStatement) {
      if (Partitions.mergeToAvoidDuplicatedLoads()) {
        LLVM_DEBUG(dbgs() << "LDist: Partitions merged to ensure unique "
                             "loads:\n"
                          << Partitions);
        if (Partitions.getSize() < 2)
          return fail("CantIsolateUnsafeDeps",
                      "cannot isolate unsafe dependencies");
      }
    }

    // Don't distribute the loop if we need too many SCEV run-time checks, or
    // any if it's illegal.
    const SCEVPredicate &Pred = LAI->getPSE().getPredicate();
    if (LAI->hasConvergentOp() && !Pred.isAlwaysTrue()) {
      return fail("RuntimeCheckWithConvergent",
                  "may not insert runtime check with convergent operation");
    }

    if (Pred.getComplexity() > (IsForced.value_or(false)
                                    ? PragmaDistributeSCEVCheckThreshold
                                    : DistributeSCEVCheckThreshold))
      return fail("TooManySCEVRuntimeChecks",
                  "too many SCEV run-time checks needed.\n");

    if (!IsForced.value_or(false) && hasDisableAllTransformsHint(L))
      return fail("HeuristicDisabled", "distribution heuristic disabled");

    LLVM_DEBUG(dbgs() << "LDist: Distributing loop: "
                      << L->getHeader()->getName() << "\n");
    // We're done forming the partitions set up the reverse mapping from
    // instructions to partitions.
    Partitions.setupPartitionIdOnInstructions();

    // If we need run-time checks, version the loop now.
    auto PtrToPartition = Partitions.computePartitionSetForPointers(*LAI);
    const auto *RtPtrChecking = LAI->getRuntimePointerChecking();
    const auto &AllChecks = RtPtrChecking->getChecks();
    auto Checks = includeOnlyCrossPartitionChecks(AllChecks, PtrToPartition,
                                                  RtPtrChecking);

    if (LAI->hasConvergentOp() && !Checks.empty()) {
      return fail("RuntimeCheckWithConvergent",
                  "may not insert runtime check with convergent operation");
    }

    // To keep things simple have an empty preheader before we version or clone
    // the loop.  (Also split if this has no predecessor, i.e. entry, because we
    // rely on PH having a predecessor.)
    if (!PH->getSinglePredecessor() || &*PH->begin() != PH->getTerminator())
      SplitBlock(PH, PH->getTerminator(), DT, LI);

    if (!Pred.isAlwaysTrue() || !Checks.empty()) {
      assert(!LAI->hasConvergentOp() && "inserting illegal loop versioning");

      MDNode *OrigLoopID = L->getLoopID();

      LLVM_DEBUG(dbgs() << "LDist: Pointers:\n");
      LLVM_DEBUG(LAI->getRuntimePointerChecking()->printChecks(dbgs(), Checks));
      // Forming LCSSA is a precondition of versioning.
      if (!L->isRecursivelyLCSSAForm(*DT, *LI))
        formLCSSARecursively(*L, *DT, LI, SE);
      LoopVersioning LVer(*LAI, Checks, L, LI, DT, SE);
      LVer.versionLoop(DefsUsedOutside);
      LVer.annotateLoopWithNoAlias();

      // The unversioned loop will not be changed, so we inherit all attributes
      // from the original loop, but remove the loop distribution metadata to
      // avoid to distribute it again.
      MDNode *UnversionedLoopID = *makeFollowupLoopID(
          OrigLoopID,
          {LLVMLoopDistributeFollowupAll, LLVMLoopDistributeFollowupFallback},
          "llvm.loop.distribute.", true);
      LVer.getNonVersionedLoop()->setLoopID(UnversionedLoopID);
      addStringMetadataToLoop(LVer.getNonVersionedLoop(), DistributedMetaData,
                              true);
    }

    // Create identical copies of the original loop for each partition and hook
    // them up sequentially.
    Partitions.cloneLoops();

    // Now, we remove the instruction from each loop that don't belong to that
    // partition.
    Partitions.removeUnusedInsts();
    LLVM_DEBUG(dbgs() << "LDist: After removing unused Instrs:\n");
    LLVM_DEBUG(Partitions.printBlocks(dbgs()));

    if (LDistVerify) {
      LI->verify();
      assert(DT->verify(DominatorTree::VerificationLevel::Fast));
    }

    ++NumLoopsDistributed;
    // Report the success.
    ORE->emit([&]() {
      return OptimizationRemark(LDIST_NAME, "Distribute", L->getStartLoc(),
                                L->getHeader())
             << "distributed loop";
    });
    return true;
  }

  /// Try to distribute a non-innermost loop using DependenceAnalysis.
  /// Gated by StatementSeededPartitioning.  Only accepts a perfectly-
  /// shaped nest (one inner loop + simple trailing block of memory
  /// ops); silently returns false otherwise so the upstream behaviour
  /// of LoopDistribute is unchanged on every other nest shape.
  bool processOuterLoop(DependenceInfo *DI) {
    assert(!L->isInnermost() && "Only call on non-innermost loops.");
    assert(DI && "DependenceInfo required for outer-loop distribution.");

    LLVM_DEBUG(dbgs() << "LDist: Outer-loop candidate '"
                      << L->getHeader()->getName() << "'\n");

    auto Subloops = L->getSubLoops();
    if (Subloops.size() != 1)
      return false;
    Loop *Inner = Subloops.front();
    if (!Inner->isInnermost())
      return false;
    if (!L->isLoopSimplifyForm() || !Inner->isLoopSimplifyForm())
      return false;
    if (!L->getExitBlock() || !Inner->getExitBlock())
      return false;

    // Accept reducible nests of the form
    //   header -> inner-preheader -> inner-loop... -> trailing -> latch
    // where Trailing is the inner-loop exit block (or coincides with the
    // latch).  After loop-rotate this is the doubly-nested shape with a
    // single statement after the inner loop.
    BasicBlock *Trailing = Inner->getExitBlock();
    if (!Trailing || !Trailing->getSinglePredecessor())
      return false;
    BasicBlock *Latch = L->getLoopLatch();
    if (!Latch)
      return false;
    if (Trailing != Latch && Trailing->getSingleSuccessor() != Latch)
      return false;

    // Refuse memory operations other than simple loads and stores in the
    // trailing block.
    SmallVector<StoreInst *, 4> TrailingStores;
    for (Instruction &I : *Trailing) {
      if (I.isTerminator())
        continue;
      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        if (!LI->isSimple())
          return false;
        continue;
      }
      if (auto *SI = dyn_cast<StoreInst>(&I)) {
        if (!SI->isSimple())
          return false;
        TrailingStores.push_back(SI);
        continue;
      }
      if (I.mayReadOrWriteMemory() && !isa<DbgInfoIntrinsic>(I))
        return false;
    }
    if (TrailingStores.empty())
      return false;

    // Refuse anything other than simple memory ops in the inner-loop body
    // as well.
    SmallVector<StoreInst *, 4> InnerStores;
    for (BasicBlock *BB : Inner->blocks()) {
      for (Instruction &I : *BB) {
        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          if (!LI->isSimple())
            return false;
          continue;
        }
        if (auto *SI = dyn_cast<StoreInst>(&I)) {
          if (!SI->isSimple())
            return false;
          InnerStores.push_back(SI);
          continue;
        }
        if (I.mayReadOrWriteMemory() && !isa<DbgInfoIntrinsic>(I))
          return false;
      }
    }
    if (InnerStores.empty())
      return false;

    // Memory instructions in program order: the inner loop, then the
    // trailing block.
    SmallVector<BasicBlock *, 8> Blocks(Inner->blocks());
    Blocks.push_back(Trailing);
    SmallVector<Instruction *, 16> MemInsts;
    for (BasicBlock *BB : Blocks)
      for (Instruction &I : *BB)
        if (isa<LoadInst>(I) || isa<StoreInst>(I))
          MemInsts.push_back(&I);

    // Order every DA dependence by the access that executes first, reading
    // the direction vector from the outer loop inwards.  When either access
    // may run first, both orders are recorded so that splitting them across
    // partitions is refused.
    SmallVector<OrderedMemPair, 4> Pairs;
    unsigned OuterDepth = L->getLoopDepth();
    for (auto [Idx, Src] : enumerate(MemInsts)) {
      for (Instruction *Dst : drop_begin(MemInsts, Idx + 1)) {
        std::unique_ptr<Dependence> Dep =
            DI->depends(Src, Dst, /*UnderRuntimeAssumptions=*/false);
        if (!Dep || !Dep->isOrdered())
          continue;
        bool SrcFirst = false;
        bool DstFirst = false;
        bool Tied = !Dep->isConfused() && Dep->getLevels() >= OuterDepth;
        if (Tied) {
          for (unsigned Level = OuterDepth; Tied && Level <= Dep->getLevels();
               ++Level) {
            unsigned Dir = Dep->getDirection(Level);
            SrcFirst |= (Dir & Dependence::DVEntry::LT) != 0;
            DstFirst |= (Dir & Dependence::DVEntry::GT) != 0;
            Tied = (Dir & Dependence::DVEntry::EQ) != 0;
          }
          // Same iteration of every loop: program order decides.
          if (Tied && DT->dominates(Src, Dst))
            SrcFirst = true;
          else if (Tied)
            SrcFirst = DstFirst = true;
        } else {
          SrcFirst = DstFirst = true;
        }
        if (SrcFirst)
          Pairs.push_back({Src, Dst, /*Carried=*/false});
        if (DstFirst)
          Pairs.push_back({Dst, Src, /*Carried=*/false});
      }
    }

    // Reuse the seeded core to form partitions.
    InstPartitionContainer Partitions(L, LI, DT);
    bool Seeded = seedPartitionsFromPairs(Partitions, Pairs, MemInsts);
    if (!Seeded || Partitions.getSize() < 2)
      return false;

    // Skip merge heuristics, mirroring the innermost seeded path.
    Partitions.populateUsedSet();
    LLVM_DEBUG(dbgs() << "LDist: Outer populated partitions:\n" << Partitions);

    Partitions.setupPartitionIdOnInstructions();

    // Empty preheader requirement before cloning.
    BasicBlock *PH = L->getLoopPreheader();
    if (!PH || !PH->getSinglePredecessor() ||
        &*PH->begin() != PH->getTerminator())
      SplitBlock(PH, PH->getTerminator(), DT, LI);

    Partitions.cloneLoops();
    Partitions.removeUnusedInsts();

    if (LDistVerify) {
      LI->verify();
      assert(DT->verify(DominatorTree::VerificationLevel::Fast));
    }

    ++NumLoopsDistributed;
    ORE->emit([&]() {
      return OptimizationRemark(LDIST_NAME, "Distribute", L->getStartLoc(),
                                L->getHeader())
             << "distributed loop (outer)";
    });
    return true;
  }

  /// Provide diagnostics then \return with false.
  bool fail(StringRef RemarkName, StringRef Message) {
    LLVMContext &Ctx = F->getContext();
    bool Forced = isForced().value_or(false);

    LLVM_DEBUG(dbgs() << "LDist: Skipping; " << Message << "\n");

    // With Rpass-missed report that distribution failed.
    ORE->emit([&]() {
      return OptimizationRemarkMissed(LDIST_NAME, "NotDistributed",
                                      L->getStartLoc(), L->getHeader())
             << "loop not distributed: use -Rpass-analysis=loop-distribute for "
                "more "
                "info";
    });

    // With Rpass-analysis report why.  This is on by default if distribution
    // was requested explicitly.
    ORE->emit(OptimizationRemarkAnalysis(
                  Forced ? OptimizationRemarkAnalysis::AlwaysPrint : LDIST_NAME,
                  RemarkName, L->getStartLoc(), L->getHeader())
              << "loop not distributed: " << Message);

    // Also issue a warning if distribution was requested explicitly but it
    // failed.
    if (Forced)
      Ctx.diagnose(DiagnosticInfoOptimizationFailure(
          *F, L->getStartLoc(), "loop not distributed: failed "
                                "explicitly specified loop distribution"));

    return false;
  }

  /// Return if distribution forced to be enabled/disabled for the loop.
  ///
  /// If the optional has a value, it indicates whether distribution was forced
  /// to be enabled (true) or disabled (false).  If the optional has no value
  /// distribution was not forced either way.
  const std::optional<bool> &isForced() const { return IsForced; }

private:
  /// A memory dependence whose First access executes before its Second
  /// access.  Statements containing First must run before statements
  /// containing Second; a Carried dependence inside one statement makes
  /// that statement cyclic.
  struct OrderedMemPair {
    Instruction *First;
    Instruction *Second;
    bool Carried;
  };

  /// Seed partitions from each store's data-flow closure (the in-loop
  /// transitive operand set of the store).  Each store becomes its own
  /// partition; self-loop pairs mark a partition cyclic; cross-
  /// statement pairs become topological ordering edges.  Returns true
  /// (and at least two partitions) when seeding succeeds; false means
  /// the caller falls back to interval-based seeding.
  bool seedPartitionsFromPairs(InstPartitionContainer &Partitions,
                               ArrayRef<OrderedMemPair> Pairs,
                               ArrayRef<Instruction *> MemInsts) {
    // Bail on anything that may invalidate naive closure: volatile or atomic
    // memory ops, multi-block bodies.
    SmallVector<StoreInst *, 8> Stores;
    for (BasicBlock *BB : L->blocks()) {
      for (Instruction &I : *BB) {
        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          if (!LI->isSimple())
            return false;
          continue;
        }
        if (auto *SI = dyn_cast<StoreInst>(&I)) {
          if (!SI->isSimple())
            return false;
          Stores.push_back(SI);
          continue;
        }
        if (I.mayReadOrWriteMemory() && !isa<DbgInfoIntrinsic>(I))
          return false;
      }
    }
    if (Stores.size() < 2)
      return false;

    // Build per-statement closures.  Each statement is keyed by its store's
    // index in `Stores`.  Walk store operands transitively, restricted to
    // the loop body.  A load shared by multiple statements is assigned to
    // each consuming statement (it will be replicated by cloning); only
    // stores must belong to a single statement.
    SmallVector<SmallSetVector<Instruction *, 8>, 4> Statements(Stores.size());
    DenseMap<Instruction *, int> StoreToStmt;
    DenseMap<Instruction *, SmallSet<int, 2>> InstToStmts;
    auto RecordInLoop = [&](Instruction *I, int StmtId) {
      if (!L->contains(I))
        return false;
      if (auto *SI = dyn_cast<StoreInst>(I)) {
        auto It = StoreToStmt.try_emplace(SI, StmtId);
        if (!It.second)
          return false;
      }
      auto &Set = InstToStmts[I];
      if (!Set.insert(StmtId).second)
        return false;
      Statements[StmtId].insert(I);
      return true;
    };
    for (auto [Idx, S] : llvm::enumerate(Stores)) {
      int StmtId = static_cast<int>(Idx);
      SmallVector<Instruction *, 16> Worklist;
      if (RecordInLoop(S, StmtId))
        Worklist.push_back(S);
      while (!Worklist.empty()) {
        Instruction *Cur = Worklist.pop_back_val();
        for (Value *Op : Cur->operands())
          if (auto *OpI = dyn_cast<Instruction>(Op))
            if (RecordInLoop(OpI, StmtId))
              Worklist.push_back(OpI);
      }
    }

    // Classify each dependence:
    //   - Self-loop (same statement on both sides): a carried one marks
    //     that statement's partition cyclic.
    //   - Inter-statement: emit a producer-before-consumer ordering
    //     edge; statements stay in separate partitions.
    SmallSet<int, 4> CyclicStmts;
    SmallVector<std::pair<int, int>, 4> OrderEdges;
    auto StmtsFor = [&](Instruction *I) -> SmallVector<int, 2> {
      if (auto *SI = dyn_cast<StoreInst>(I)) {
        auto It = StoreToStmt.find(SI);
        return It == StoreToStmt.end() ? SmallVector<int, 2>{}
                                       : SmallVector<int, 2>{It->second};
      }
      auto It = InstToStmts.find(I);
      if (It == InstToStmts.end())
        return {};
      return SmallVector<int, 2>(It->second.begin(), It->second.end());
    };
    for (const auto &[First, Second, Carried] : Pairs) {
      auto FirstStmts = StmtsFor(First);
      auto SecondStmts = StmtsFor(Second);
      if (FirstStmts.empty() || SecondStmts.empty())
        return false;
      // A load shared by several statements is cloned into each of them,
      // so every copy has to respect the dependence.
      for (int Before : FirstStmts)
        for (int After : SecondStmts) {
          if (Before != After)
            OrderEdges.emplace_back(Before, After);
          else if (Carried)
            CyclicStmts.insert(Before);
        }
    }

    // One statement = one partition.  Cyclic-ness is carried separately
    // by PartIsCyclic[] below; ordering between partitions comes from
    // OrderEdges and the topological pass that follows.
    int NumPartitions = static_cast<int>(Stores.size());
    if (NumPartitions < 2)
      return false;
    SmallVector<int, 4> StmtToPartId(NumPartitions);
    for (int Idx = 0; Idx < NumPartitions; ++Idx)
      StmtToPartId[Idx] = Idx;

    // Topologically order partitions to honour producer-before-consumer
    // edges.  If a true cycle exists in the partition graph, we cannot
    // safely split.
    SmallVector<SmallSet<int, 2>, 4> PartEdges(NumPartitions);
    SmallVector<unsigned, 4> InDegree(NumPartitions, 0);
    for (auto [Producer, Consumer] : OrderEdges) {
      int PProd = StmtToPartId[Producer];
      int PCons = StmtToPartId[Consumer];
      if (PProd == PCons)
        continue;
      if (PartEdges[PProd].insert(PCons).second)
        ++InDegree[PCons];
    }
    SmallVector<int, 4> NewOrder;
    SmallVector<int, 4> Ready;
    for (int P = 0; P < NumPartitions; ++P)
      if (InDegree[P] == 0)
        Ready.push_back(P);
    while (!Ready.empty()) {
      int P = Ready.pop_back_val();
      NewOrder.push_back(P);
      for (int Succ : PartEdges[P])
        if (--InDegree[Succ] == 0)
          Ready.push_back(Succ);
    }
    if (static_cast<int>(NewOrder.size()) != NumPartitions)
      return false;
    SmallVector<int, 4> Renumber(NumPartitions);
    for (int Pos = 0; Pos < NumPartitions; ++Pos)
      Renumber[NewOrder[Pos]] = Pos;
    for (int &Id : StmtToPartId)
      Id = Renumber[Id];

    // Bucket memory instructions by partition id, preserving program order
    // within each bucket via the supplied memory instruction list.  An
    // instruction that maps to multiple statements (shared load) is added
    // to each partition that consumes it; the partitions only have read
    // access to the duplicated value, which is safe.
    SmallVector<SmallVector<Instruction *, 8>, 4> Buckets(NumPartitions);
    SmallVector<bool, 4> PartIsCyclic(NumPartitions, false);
    for (Instruction *I : MemInsts) {
      auto Stmts = StmtsFor(I);
      if (Stmts.empty())
        return false;
      SmallSet<int, 2> Targeted;
      for (int Stmt : Stmts) {
        int PartId = StmtToPartId[Stmt];
        if (!Targeted.insert(PartId).second)
          continue;
        Buckets[PartId].push_back(I);
        if (CyclicStmts.contains(Stmt))
          PartIsCyclic[PartId] = true;
      }
    }

    // Emit partitions in topological order; each carries its full seed
    // bucket and its cyclic flag.
    for (int PartId = 0; PartId < NumPartitions; ++PartId) {
      if (Buckets[PartId].empty())
        continue;
      Partitions.addNewPartition(Buckets[PartId], PartIsCyclic[PartId]);
    }

    LLVM_DEBUG(dbgs() << "LDist: Statement-seeded into " << NumPartitions
                      << " partitions\n");
    return true;
  }

  /// Filter out checks between pointers from the same partition.
  ///
  /// \p PtrToPartition contains the partition number for pointers.  Partition
  /// number -1 means that the pointer is used in multiple partitions.  In this
  /// case we can't safely omit the check.
  SmallVector<RuntimePointerCheck, 4> includeOnlyCrossPartitionChecks(
      const SmallVectorImpl<RuntimePointerCheck> &AllChecks,
      const SmallVectorImpl<int> &PtrToPartition,
      const RuntimePointerChecking *RtPtrChecking) {
    SmallVector<RuntimePointerCheck, 4> Checks;

    copy_if(AllChecks, std::back_inserter(Checks),
            [&](const RuntimePointerCheck &Check) {
              for (unsigned PtrIdx1 : Check.first->Members)
                for (unsigned PtrIdx2 : Check.second->Members)
                  // Only include this check if there is a pair of pointers
                  // that require checking and the pointers fall into
                  // separate partitions.
                  //
                  // (Note that we already know at this point that the two
                  // pointer groups need checking but it doesn't follow
                  // that each pair of pointers within the two groups need
                  // checking as well.
                  //
                  // In other words we don't want to include a check just
                  // because there is a pair of pointers between the two
                  // pointer groups that require checks and a different
                  // pair whose pointers fall into different partitions.)
                  if (RtPtrChecking->needsChecking(PtrIdx1, PtrIdx2) &&
                      !RuntimePointerChecking::arePointersInSamePartition(
                          PtrToPartition, PtrIdx1, PtrIdx2))
                    return true;
              return false;
            });

    return Checks;
  }

  /// Check whether the loop metadata is forcing distribution to be
  /// enabled/disabled.
  void setForced() {
    if (getBooleanLoopAttribute(L, "llvm.loop.distribute.enable"))
      IsForced = true;
    else if (getBooleanLoopAttribute(L, "llvm.loop.distribute.disable"))
      IsForced = false;
  }

  Loop *L;
  Function *F;

  // Analyses used.
  LoopInfo *LI;
  const LoopAccessInfo *LAI = nullptr;
  DominatorTree *DT;
  ScalarEvolution *SE;
  LoopAccessInfoManager &LAIs;
  OptimizationRemarkEmitter *ORE;

  /// Indicates whether distribution is forced to be enabled/disabled for
  /// the loop.
  ///
  /// If the optional has a value, it indicates whether distribution was forced
  /// to be enabled (true) or disabled (false).  If the optional has no value
  /// distribution was not forced either way.
  std::optional<bool> IsForced;
};

} // end anonymous namespace

static bool runImpl(Function &F, LoopInfo *LI, DominatorTree *DT,
                    ScalarEvolution *SE, OptimizationRemarkEmitter *ORE,
                    LoopAccessInfoManager &LAIs, DependenceInfo *DI) {
  // Build up a worklist of inner-loops to distribute. This is necessary as the
  // act of distributing a loop creates new loops and can invalidate iterators
  // across the loops.
  SmallVector<Loop *, 8> InnerWorklist;
  // When the statement-seeded strategy is on, we additionally consider
  // non-innermost loops via DependenceAnalysis.  Enumerated separately so
  // that the existing innermost path runs first.
  SmallVector<Loop *, 4> OuterWorklist;

  for (Loop *TopLevelLoop : *LI)
    for (Loop *L : depth_first(TopLevelLoop)) {
      if (L->isInnermost())
        InnerWorklist.push_back(L);
      else if (StatementSeededPartitioning)
        OuterWorklist.push_back(L);
    }

  bool Changed = false;
  for (Loop *L : InnerWorklist) {
    LoopDistributeForLoop LDL(L, &F, LI, DT, SE, LAIs, ORE);

    // Do not reprocess loops we already distributed
    if (getOptionalBoolLoopAttribute(L, DistributedMetaData).value_or(false)) {
      LLVM_DEBUG(
          dbgs() << "LDist: Distributed loop guarded for reprocessing\n");
      continue;
    }

    // If distribution was forced for the specific loop to be
    // enabled/disabled, follow that.  Otherwise use the global flag.
    if (LDL.isForced().value_or(EnableLoopDistribute))
      Changed |= LDL.processLoop();
  }
  for (Loop *L : OuterWorklist) {
    if (getOptionalBoolLoopAttribute(L, DistributedMetaData).value_or(false))
      continue;
    LoopDistributeForLoop LDL(L, &F, LI, DT, SE, LAIs, ORE);
    if (LDL.isForced().value_or(EnableLoopDistribute))
      Changed |= LDL.processOuterLoop(DI);
  }

  // Process each loop nest in the function.
  return Changed;
}

PreservedAnalyses LoopDistributePass::run(Function &F,
                                          FunctionAnalysisManager &AM) {
  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  auto &ORE = AM.getResult<OptimizationRemarkEmitterAnalysis>(F);

  LoopAccessInfoManager &LAIs = AM.getResult<LoopAccessAnalysis>(F);
  // DependenceAnalysis is only consumed by the statement-seeded
  // outer-loop path, which is off by default.  Request it only when
  // that path is enabled so the default pipeline is unaffected.
  DependenceInfo *DI = StatementSeededPartitioning
                           ? &AM.getResult<DependenceAnalysis>(F)
                           : nullptr;
  bool Changed = runImpl(F, &LI, &DT, &SE, &ORE, LAIs, DI);
  if (!Changed)
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserve<LoopAnalysis>();
  PA.preserve<DominatorTreeAnalysis>();
  return PA;
}
