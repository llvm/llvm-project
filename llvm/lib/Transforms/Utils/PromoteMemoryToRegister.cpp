//===- PromoteMemoryToRegister.cpp - Convert allocas to registers ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file promotes memory references to be register references.  It promotes
// alloca instructions which only have loads and stores as uses.  An alloca is
// transformed by using iterated dominator frontiers to place PHI nodes, then
// traversing the function in depth-first order to rewrite loads and stores as
// appropriate.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/IteratedDominanceFrontier.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugProgramInstruction.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/User.h"
#include "llvm/Support/Casting.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include <algorithm>
#include <bitset>
#include <cassert>
#include <iterator>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "mem2reg"

STATISTIC(NumLocalPromoted, "Number of alloca's promoted within one block");
STATISTIC(NumSingleStore,   "Number of alloca's promoted with a single store");
STATISTIC(NumDeadAlloca,    "Number of dead alloca's removed");
STATISTIC(NumPHIInsert,     "Number of PHI nodes inserted");

static cl::opt<bool> UseVectorizedMem2Reg(
    "vectorized-mem2reg", cl::init(true), cl::Hidden,
    cl::desc("Use a vectorized algorithm to compute PHI nodes in batches."));

bool llvm::isAllocaPromotable(const AllocaInst *AI) {
  // Only allow direct and non-volatile loads and stores...
  for (const User *U : AI->users()) {
    if (const LoadInst *LI = dyn_cast<LoadInst>(U)) {
      // Note that atomic loads can be transformed; atomic semantics do
      // not have any meaning for a local alloca.
      if (LI->isVolatile() || LI->getType() != AI->getAllocatedType())
        return false;
    } else if (const StoreInst *SI = dyn_cast<StoreInst>(U)) {
      if (SI->getValueOperand() == AI ||
          SI->getValueOperand()->getType() != AI->getAllocatedType())
        return false; // Don't allow a store OF the AI, only INTO the AI.
      // Note that atomic stores can be transformed; atomic semantics do
      // not have any meaning for a local alloca.
      if (SI->isVolatile())
        return false;
    } else if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(U)) {
      if (!II->isLifetimeStartOrEnd() && !II->isDroppable() &&
          II->getIntrinsicID() != Intrinsic::fake_use)
        return false;
    } else if (const BitCastInst *BCI = dyn_cast<BitCastInst>(U)) {
      if (!onlyUsedByLifetimeMarkersOrDroppableInsts(BCI))
        return false;
    } else if (const GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(U)) {
      if (!GEPI->hasAllZeroIndices())
        return false;
      if (!onlyUsedByLifetimeMarkersOrDroppableInsts(GEPI))
        return false;
    } else if (const AddrSpaceCastInst *ASCI = dyn_cast<AddrSpaceCastInst>(U)) {
      if (!onlyUsedByLifetimeMarkers(ASCI))
        return false;
    } else {
      return false;
    }
  }

  return true;
}

namespace {

static void createDebugValue(DIBuilder &DIB, Value *NewValue,
                             DILocalVariable *Variable,
                             DIExpression *Expression, const DILocation *DI,
                             DbgVariableRecord *InsertBefore) {
  // FIXME: Merge these two functions now that DIBuilder supports
  // DbgVariableRecords. We neeed the API to accept DbgVariableRecords as an
  // insert point for that to work.
  (void)DIB;
  DbgVariableRecord::createDbgVariableRecord(NewValue, Variable, Expression, DI,
                                             *InsertBefore);
}

/// Helper for updating assignment tracking debug info when promoting allocas.
class AssignmentTrackingInfo {
  /// DbgAssignIntrinsics linked to the alloca with at most one per variable
  /// fragment. (i.e. not be a comprehensive set if there are multiple
  /// dbg.assigns for one variable fragment).
  SmallVector<DbgVariableRecord *> DVRAssigns;

public:
  void init(AllocaInst *AI) {
    SmallSet<DebugVariable, 2> Vars;
    for (DbgVariableRecord *DVR : at::getDVRAssignmentMarkers(AI)) {
      if (Vars.insert(DebugVariable(DVR)).second)
        DVRAssigns.push_back(DVR);
    }
  }

  /// Update assignment tracking debug info given for the to-be-deleted store
  /// \p ToDelete that stores to this alloca.
  void updateForDeletedStore(
      StoreInst *ToDelete, DIBuilder &DIB,
      SmallPtrSet<DbgVariableRecord *, 8> *DVRAssignsToDelete) const {
    // There's nothing to do if the alloca doesn't have any variables using
    // assignment tracking.
    if (DVRAssigns.empty())
      return;

    // Insert a dbg.value where the linked dbg.assign is and remember to delete
    // the dbg.assign later. Demoting to dbg.value isn't necessary for
    // correctness but does reduce compile time and memory usage by reducing
    // unnecessary function-local metadata. Remember that we've seen a
    // dbg.assign for each variable fragment for the untracked store handling
    // (after this loop).
    SmallSet<DebugVariableAggregate, 2> VarHasDbgAssignForStore;
    auto InsertValueForAssign = [&](auto *DbgAssign, auto *&AssignList) {
      VarHasDbgAssignForStore.insert(DebugVariableAggregate(DbgAssign));
      AssignList->insert(DbgAssign);
      createDebugValue(DIB, DbgAssign->getValue(), DbgAssign->getVariable(),
                       DbgAssign->getExpression(), DbgAssign->getDebugLoc(),
                       DbgAssign);
    };
    for (auto *Assign : at::getDVRAssignmentMarkers(ToDelete))
      InsertValueForAssign(Assign, DVRAssignsToDelete);

    // It's possible for variables using assignment tracking to have no
    // dbg.assign linked to this store. These are variables in DVRAssigns that
    // are missing from VarHasDbgAssignForStore. Since there isn't a dbg.assign
    // to mark the assignment - and the store is going to be deleted - insert a
    // dbg.value to do that now. An untracked store may be either one that
    // cannot be represented using assignment tracking (non-const offset or
    // size) or one that is trackable but has had its DIAssignID attachment
    // dropped accidentally.
    auto ConvertUnlinkedAssignToValue = [&](DbgVariableRecord *Assign) {
      if (VarHasDbgAssignForStore.contains(DebugVariableAggregate(Assign)))
        return;
      ConvertDebugDeclareToDebugValue(Assign, ToDelete, DIB);
    };
    for_each(DVRAssigns, ConvertUnlinkedAssignToValue);
  }

  /// Update assignment tracking debug info given for the newly inserted PHI \p
  /// NewPhi.
  void updateForNewPhi(PHINode *NewPhi, DIBuilder &DIB) const {
    // Regardless of the position of dbg.assigns relative to stores, the
    // incoming values into a new PHI should be the same for the (imaginary)
    // debug-phi.
    for (auto *DVR : DVRAssigns)
      ConvertDebugDeclareToDebugValue(DVR, NewPhi, DIB);
  }

  void clear() { DVRAssigns.clear(); }
  bool empty() { return DVRAssigns.empty(); }
};

struct AllocaInfo {
  using DPUserVec = SmallVector<DbgVariableRecord *, 1>;

  SmallVector<BasicBlock *, 32> DefiningBlocks;
  SmallVector<BasicBlock *, 32> UsingBlocks;

  StoreInst *OnlyStore;
  BasicBlock *OnlyBlock;
  bool OnlyUsedInOneBlock;

  /// Debug users of the alloca - does not include dbg.assign intrinsics.
  DPUserVec DPUsers;
  /// Helper to update assignment tracking debug info.
  AssignmentTrackingInfo AssignmentTracking;

  void clear() {
    DefiningBlocks.clear();
    UsingBlocks.clear();
    OnlyStore = nullptr;
    OnlyBlock = nullptr;
    OnlyUsedInOneBlock = true;
    DPUsers.clear();
    AssignmentTracking.clear();
  }

  /// Scan the uses of the specified alloca, filling in the AllocaInfo used
  /// by the rest of the pass to reason about the uses of this alloca.
  void AnalyzeAlloca(AllocaInst *AI) {
    clear();

    // As we scan the uses of the alloca instruction, keep track of stores,
    // and decide whether all of the loads and stores to the alloca are within
    // the same basic block.
    for (User *U : AI->users()) {
      Instruction *User = cast<Instruction>(U);

      if (StoreInst *SI = dyn_cast<StoreInst>(User)) {
        // Remember the basic blocks which define new values for the alloca
        DefiningBlocks.push_back(SI->getParent());
        OnlyStore = SI;
      } else {
        LoadInst *LI = cast<LoadInst>(User);
        // Otherwise it must be a load instruction, keep track of variable
        // reads.
        UsingBlocks.push_back(LI->getParent());
      }

      if (OnlyUsedInOneBlock) {
        if (!OnlyBlock)
          OnlyBlock = User->getParent();
        else if (OnlyBlock != User->getParent())
          OnlyUsedInOneBlock = false;
      }
    }
    SmallVector<DbgVariableRecord *> AllDPUsers;
    findDbgUsers(AI, AllDPUsers);
    std::copy_if(AllDPUsers.begin(), AllDPUsers.end(),
                 std::back_inserter(DPUsers),
                 [](DbgVariableRecord *DVR) { return !DVR->isDbgAssign(); });
    AssignmentTracking.init(AI);
  }
};

template <typename T> class VectorWithUndo {
  SmallVector<T, 8> Vals;
  SmallVector<std::pair<size_t, T>, 8> Undo;

public:
  void undo(size_t S) {
    assert(S <= Undo.size());
    while (S < Undo.size()) {
      Vals[Undo.back().first] = Undo.back().second;
      Undo.pop_back();
    }
  }

  void resize(size_t Sz) { Vals.resize(Sz); }

  size_t undoSize() const { return Undo.size(); }

  const T &operator[](size_t Idx) const { return Vals[Idx]; }

  void set(size_t Idx, const T &Val) {
    if (Vals[Idx] == Val)
      return;
    Undo.emplace_back(Idx, Vals[Idx]);
    Vals[Idx] = Val;
  }

  void init(size_t Idx, const T &Val) {
    assert(Undo.empty());
    Vals[Idx] = Val;
  }
};

/// Data package used by RenamePass().
struct RenamePassData {
  RenamePassData(BasicBlock *B, BasicBlock *P, size_t V, size_t L)
      : BB(B), Pred(P), UndoVals(V), UndoLocs(L) {}

  BasicBlock *BB;
  BasicBlock *Pred;

  size_t UndoVals;
  size_t UndoLocs;
};

/// This assigns and keeps a per-bb relative ordering of load/store
/// instructions in the block that directly load or store an alloca.
///
/// This functionality is important because it avoids scanning large basic
/// blocks multiple times when promoting many allocas in the same block.
class LargeBlockInfo {
  /// For each instruction that we track, keep the index of the
  /// instruction.
  ///
  /// The index starts out as the number of the instruction from the start of
  /// the block.
  DenseMap<const Instruction *, unsigned> InstNumbers;

public:

  /// This code only looks at accesses to allocas.
  static bool isInterestingInstruction(const Instruction *I) {
    return (isa<LoadInst>(I) && isa<AllocaInst>(I->getOperand(0))) ||
           (isa<StoreInst>(I) && isa<AllocaInst>(I->getOperand(1)));
  }

  /// Get or calculate the index of the specified instruction.
  unsigned getInstructionIndex(const Instruction *I) {
    assert(isInterestingInstruction(I) &&
           "Not a load/store to/from an alloca?");

    // If we already have this instruction number, return it.
    DenseMap<const Instruction *, unsigned>::iterator It = InstNumbers.find(I);
    if (It != InstNumbers.end())
      return It->second;

    // Scan the whole block to get the instruction.  This accumulates
    // information for every interesting instruction in the block, in order to
    // avoid gratuitus rescans.
    const BasicBlock *BB = I->getParent();
    unsigned InstNo = 0;
    for (const Instruction &BBI : *BB)
      if (isInterestingInstruction(&BBI))
        InstNumbers[&BBI] = InstNo++;
    It = InstNumbers.find(I);

    assert(It != InstNumbers.end() && "Didn't insert instruction?");
    return It->second;
  }

  void deleteValue(const Instruction *I) { InstNumbers.erase(I); }

  void clear() { InstNumbers.clear(); }
};

struct PromoteMem2Reg {
  /// The alloca instructions being promoted.
  std::vector<AllocaInst *> Allocas;

  DominatorTree &DT;
  DIBuilder DIB;

  /// A cache of @llvm.assume intrinsics used by SimplifyInstruction.
  AssumptionCache *AC;

  const SimplifyQuery SQ;

  /// Reverse mapping of Allocas.
  DenseMap<AllocaInst *, unsigned> AllocaLookup;

  /// The PhiNodes we're adding.
  ///
  /// That map is used to simplify some Phi nodes as we iterate over it, so
  /// it should have deterministic iterators.  We could use a MapVector, but
  /// since basic blocks have numbers, using these are more efficient.
  DenseMap<std::pair<unsigned, unsigned>, PHINode *> NewPhiNodes;

  /// For each PHI node, keep track of which entry in Allocas it corresponds
  /// to.
  DenseMap<PHINode *, unsigned> PhiToAllocaMap;

  /// For each alloca, we keep track of the dbg.declare record that
  /// describes it, if any, so that we can convert it to a dbg.value
  /// record if the alloca gets promoted.
  SmallVector<AllocaInfo::DPUserVec, 8> AllocaDPUsers;

  /// For each alloca, keep an instance of a helper class that gives us an easy
  /// way to update assignment tracking debug info if the alloca is promoted.
  SmallVector<AssignmentTrackingInfo, 8> AllocaATInfo;
  /// A set of dbg.assigns to delete because they've been demoted to
  /// dbg.values. Call cleanUpDbgAssigns to delete them.
  SmallPtrSet<DbgVariableRecord *, 8> DVRAssignsToDelete;

  /// The set of basic blocks the renamer has already visited.
  BitVector Visited;

  /// Lazily compute the number of predecessors a block has, indexed by block
  /// number.
  SmallVector<unsigned> BBNumPreds;

  /// The state of incoming values for the current DFS step.
  VectorWithUndo<Value *> IncomingVals;

  /// The state of incoming locations for the current DFS step.
  VectorWithUndo<DebugLoc> IncomingLocs;

  // DFS work stack.
  SmallVector<RenamePassData, 8> Worklist;

  /// Whether the function has the no-signed-zeros-fp-math attribute set.
  bool NoSignedZeros = false;

public:
  PromoteMem2Reg(ArrayRef<AllocaInst *> Allocas, DominatorTree &DT,
                 AssumptionCache *AC)
      : Allocas(Allocas.begin(), Allocas.end()), DT(DT),
        DIB(*DT.getRoot()->getParent()->getParent(), /*AllowUnresolved*/ false),
        AC(AC), SQ(DT.getRoot()->getDataLayout(),
                   nullptr, &DT, AC) {}

  void run();

private:
  void RemoveFromAllocasList(unsigned &AllocaIdx) {
    Allocas[AllocaIdx] = Allocas.back();
    Allocas.pop_back();
    --AllocaIdx;
  }

  unsigned getNumPreds(const BasicBlock *BB) {
    // BBNumPreds is resized to getMaxBlockNumber() at the beginning.
    unsigned &NP = BBNumPreds[BB->getNumber()];
    if (NP == 0)
      NP = pred_size(BB) + 1;
    return NP - 1;
  }

  void ComputeLiveInBlocks(AllocaInst *AI, AllocaInfo &Info,
                           const SmallPtrSetImpl<BasicBlock *> &DefBlocks,
                           SmallPtrSetImpl<BasicBlock *> &LiveInBlocks);
  void RenamePass(BasicBlock *BB, BasicBlock *Pred);
  bool QueuePhiNode(BasicBlock *BB, unsigned AllocaIdx, unsigned &Version);

  /// Delete dbg.assigns that have been demoted to dbg.values.
  void cleanUpDbgAssigns() {
    for (auto *DVR : DVRAssignsToDelete)
      DVR->eraseFromParent();
    DVRAssignsToDelete.clear();
  }

  void pushToWorklist(BasicBlock *BB, BasicBlock *Pred) {
    Worklist.emplace_back(BB, Pred, IncomingVals.undoSize(),
                          IncomingLocs.undoSize());
  }

  RenamePassData popFromWorklist() {
    RenamePassData R = Worklist.back();
    Worklist.pop_back();
    IncomingVals.undo(R.UndoVals);
    IncomingLocs.undo(R.UndoLocs);
    return R;
  }
};

/// This class computes liveness and dominance frontier (DF) of allocas in
/// batch. The scalar Mem2Reg algorithm processes alloca one by one, which is
/// not efficient in general because it repeatedly traverses the CFG and
/// enumerates predecessors or successors of basic blocks. Given that most
/// blocks are likely covered by the liveness of more than one values,
/// processing these values at one go can reduce the amount of CFG traversal.
/// This requires certain modifications to the algorithm computing liveness
/// range and DF of allocas. In general, previously we use a set to track
/// visited blocks, which is equivalent to have each block keeping track of a
/// "visited" state, and now the visited state becomes a bit vector
/// corresponding to each alloca to be process in one go. We OR the states when
/// visiting a block and if the state is changed we need to keep traversing
/// (because info carried by one alloca from a certain edge has not converged
/// yet) again, otherwise we have reached a fixed point and can stop there. The
/// number of repeated traversal is bounded by the bit vector's width because
/// each time at least a bit is set, so the overall complexity will be less than
/// the sum of each alloca being processed individually.
class VectorizedMem2Reg {
  // Max number of allocas per batch for each round of computation.
  static constexpr size_t MaxAllocaNum = 32;

  typedef decltype(std::declval<BasicBlock>().getNumber()) BBNumberTy;

  // Packed bits indicating the state of each alloca in Allocas.
  typedef std::bitset<MaxAllocaNum> AllocaState;

  // Each enum selects one of the several states used by this analysis.
  // Each basic block has an AllocaState for each enum value, and it can be
  // accessed by get<StateSelector>(BBNumber).
  enum StateSelector {

    // This is a transient state associated to the basic block inside Worklist
    // (for liveness analysis) or PQ (for IDF computation) to be processed.
    // Since a basic block can have multiple predecessors or successors, it is
    // necessary not to add the same block multiple times to Worklist or PQ,
    // which is done by checking whether this state of the block is zero. If
    // not, the block is already added and this state should be updated instead.
    // It is an invariant that get<UPDATE_STATE>(BB) != 0 iff BB is in Worklist
    // for liveness analysis or in PQ for IDF.
    UPDATE_STATE,

    // These states indicate for each alloca in Allocas, whether it is defined
    // (DEF), alive and will be used in a successor (ALIVE), or expanded by one
    // iteration of DF (IDF) in a block. The i-th position of the state bits
    // corresponds to the i-th alloca. For example get<DEF_STATE>[1][2] == true
    // means basic block #1 has a definition of Allocas[2].
    DEF_STATE,
    ALIVE_STATE,
    IDF_STATE,
  };

  // Encapsulating the states used by this analysis, making the data layout of
  // state bits opaque so that we can optimize it without affecting the
  // algorithm implementation.
  struct State {
    struct BlockState {
      AllocaState UpdateState;
      AllocaState DefState;
      AllocaState AliveState;
      AllocaState IDFState;
    };

    // A vector containing a state for each block in the function, indexed by
    // its BB Number.
    std::vector<BlockState> StateVector;

    State(size_t MaxBlockNumber) : StateVector(MaxBlockNumber) {}

    ~State() {
      assert(llvm::all_of(StateVector,
                          [](const BlockState &V) {
                            return V.UpdateState.none();
                          }));
    }

    // Select which kind of state to access. BN is the index of the basic block.
    template <enum StateSelector Kind> AllocaState &get(BBNumberTy BN) {
      if constexpr (Kind == UPDATE_STATE)
        return StateVector[BN].UpdateState;
      else if constexpr (Kind == DEF_STATE)
        return StateVector[BN].DefState;
      else if constexpr (Kind == ALIVE_STATE)
        return StateVector[BN].AliveState;
      else if constexpr (Kind == IDF_STATE)
        return StateVector[BN].IDFState;
      else
        static_assert(Kind != Kind, "Invalid StateSelector enum");
    }

    void Clear() {
      for (BlockState &State : StateVector) {
        assert(State.UpdateState.none());
        State.DefState.reset();
        State.AliveState.reset();
        State.IDFState.reset();
      }
    }
  };

  // Pointers to crucial objects of Mem2Reg.
  DominatorTree *DT;
  Function *F;

  // Reverse mapping from BB number to blocks.
  std::vector<BasicBlock *> BBList;

  // Predecessors and successors of basic blocks represented by block number.
  // They are materialized once at initialization since this pass does not
  // modify the CFG and usually almost every basic block contains at least one
  // value alive, so they are all needed.
  std::vector<llvm::SmallVector<BBNumberTy, 2>> Predecessors;
  std::vector<llvm::SmallVector<BBNumberTy, 2>> Successors;

  // Alloca instructions in the order they are gathered.
  llvm::SmallVector<AllocaInst *, MaxAllocaNum> Allocas;

  // The analysis states of every basic block.
  State BlockStates;

  // The output vectors of blocks to be added with PHI nodes for each alloca.
  std::vector<BBNumberTy> PHIBlocks[MaxAllocaNum];

  // Temporary storage for BB to be processed in liveness analysis, and in DT
  // subtree traversal.
  std::vector<BBNumberTy> Worklist;

  // See GenericIteratedDominanceFrontier.h, this is the vectorized version
  // using bit vectors indexed by BB number.
  // Use a priority queue keyed on dominator tree level so that inserted nodes
  // are handled from the bottom of the dominator tree upwards. We also augment
  // the level with a DFS number to ensure that the blocks are ordered in a
  // deterministic way.
  using DomTreeNodePair =
      std::pair<DomTreeNodeBase<BasicBlock> *, std::pair<unsigned, unsigned>>;
  using IDFPriorityQueue =
      std::priority_queue<DomTreeNodePair, std::vector<DomTreeNodePair>,
                          less_second>;
  IDFPriorityQueue PQ;

  // Add a block to the worklist for DFS traversal in liveless analysis.
  void PushWorkList(BBNumberTy BN, const AllocaState &UpdateState) {
    assert(UpdateState.any());
    // If UPDATE state is previously zero, this BB is newly encountered or
    // has been previously processed, add it to the worklist. Otherwise it is
    // already in the worklist, so we just need to merge existing UPDATE state.
    if (BlockStates.get<UPDATE_STATE>(BN).none())
      Worklist.push_back(BN);
    BlockStates.get<UPDATE_STATE>(BN) |= UpdateState;
  }

  // Get a block number and its UPDATE state from the worklist.
  std::pair<BBNumberTy, AllocaState> PopWorkList() {
    assert(!Worklist.empty());
    BBNumberTy BN = Worklist.back();
    Worklist.pop_back();
    AllocaState State = BlockStates.get<UPDATE_STATE>(BN);
    BlockStates.get<UPDATE_STATE>(BN).reset();
    return {BN, State};
  }

  // Add a node of the DT to the priority queue for IDF computation, also update
  // its state carrying info of new DEFs that is to be expanded into its DF
  // nodes.
  void PushPQ(DomTreeNodeBase<BasicBlock> *Node,
              const AllocaState &UpdateState) {
    assert(UpdateState.any());
    // If UPDATE state is previously zero, this node is newly encountered or
    // has been previously processed, add it to the PQ. Otherwise it is already
    // in the PQ, so we just need to merge existing UPDATE state.
    unsigned BN = Node->getBlock()->getNumber();
    if (BlockStates.get<UPDATE_STATE>(BN).none())
      PQ.push({Node, std::make_pair(Node->getLevel(), Node->getDFSNumIn())});
    BlockStates.get<UPDATE_STATE>(BN) |= UpdateState;
  }

  // Get the head node and its UPDATE state from the PQ.
  std::tuple<DomTreeNodeBase<BasicBlock> *, AllocaState> PopPQ() {
    assert(!PQ.empty());
    auto Node = PQ.top().first;
    BBNumberTy BN = Node->getBlock()->getNumber();
    PQ.pop();
    AllocaState State = BlockStates.get<UPDATE_STATE>(BN);
    BlockStates.get<UPDATE_STATE>(BN).reset();
    return {Node, State};
  }

  // Access the underlying container of PQ.
  std::vector<DomTreeNodePair> &GetPQContainer() {
    struct Getter : IDFPriorityQueue {
      static typename std::vector<DomTreeNodePair> &
      Get(IDFPriorityQueue &Object) {
        return Object.*&Getter::c;
      }
    };
    return Getter::Get(PQ);
  }

public:
  VectorizedMem2Reg(PromoteMem2Reg *Mem2Reg)
      : DT(&Mem2Reg->DT), F(DT->getRoot()->getParent()),
        BBList(F->getMaxBlockNumber()), Predecessors(F->getMaxBlockNumber()),
        Successors(F->getMaxBlockNumber()),
        BlockStates(F->getMaxBlockNumber()) {
    for (BasicBlock &BB : *F) {
      BBNumberTy BN = BB.getNumber();
      BBList[BN] = &BB;

      for (const BasicBlock *Predecessor : predecessors(&BB)) {
        Predecessors[BN].push_back(Predecessor->getNumber());
      }

      for (const BasicBlock *Successor : successors(&BB)) {
        Successors[BN].push_back(Successor->getNumber());
      }
    }
    Worklist.reserve(BBList.size());
    GetPQContainer().reserve(BBList.size());

    DT->updateDFSNumbers();
  }

  ~VectorizedMem2Reg() {
    assert(Worklist.empty() && PQ.empty() &&
           "Some nodes have not been processed.");
  }

  /// Clear allocas and their states for next round of computation.
  void clear() {
    assert(Worklist.empty() && PQ.empty() &&
           "Some nodes have not been processed.");
    Allocas.clear();
    BlockStates.Clear();
    for (size_t I = 0; I < MaxAllocaNum; ++I)
      PHIBlocks[I].clear();
  }

  size_t size() { return Allocas.size(); }

  /// Add an alloca to be processed, and perform some pre-processing for
  /// liveness analysis. Return true if we reached the max capacity of allocas.
  /// Currently we just gather enough allocas to fill up the bit vector. This
  /// can be improved by grouping allocas with similar liveness range in a
  /// batch.
  bool GatherAlloca(AllocaInst *AI, const AllocaInfo &Info);

  /// Calculate lists of basic blocks needing PHI nodes for the current batch
  /// of allocas if they are promoted to registers.
  void Calculate();

  /// After Calculate, retrieve the list of blocks needing PHI node for the
  /// alloca specified by the index.
  auto GetPHIBlocks(size_t AllocaIndex) {
    assert(AllocaIndex < Allocas.size());
    return llvm::map_range(PHIBlocks[AllocaIndex],
                           [this](BBNumberTy BB) { return BBList[BB]; });
  }
};

} // end anonymous namespace

/// Given a LoadInst LI this adds assume(LI != null) after it.
static void addAssumeNonNull(AssumptionCache *AC, LoadInst *LI) {
  Function *AssumeIntrinsic =
      Intrinsic::getOrInsertDeclaration(LI->getModule(), Intrinsic::assume);
  ICmpInst *LoadNotNull = new ICmpInst(ICmpInst::ICMP_NE, LI,
                                       Constant::getNullValue(LI->getType()));
  LoadNotNull->insertAfter(LI->getIterator());
  CallInst *CI = CallInst::Create(AssumeIntrinsic, {LoadNotNull});
  CI->insertAfter(LoadNotNull->getIterator());
  AC->registerAssumption(cast<AssumeInst>(CI));
}

static void convertMetadataToAssumes(LoadInst *LI, Value *Val,
                                     const DataLayout &DL, AssumptionCache *AC,
                                     const DominatorTree *DT) {
  if (isa<UndefValue>(Val) && LI->hasMetadata(LLVMContext::MD_noundef)) {
    // Insert non-terminator unreachable.
    LLVMContext &Ctx = LI->getContext();
    new StoreInst(ConstantInt::getTrue(Ctx),
                  PoisonValue::get(PointerType::getUnqual(Ctx)),
                  /*isVolatile=*/false, Align(1), LI->getIterator());
    return;
  }

  // If the load was marked as nonnull we don't want to lose that information
  // when we erase this Load. So we preserve it with an assume. As !nonnull
  // returns poison while assume violations are immediate undefined behavior,
  // we can only do this if the value is known non-poison.
  if (AC && LI->getMetadata(LLVMContext::MD_nonnull) &&
      LI->getMetadata(LLVMContext::MD_noundef) &&
      !isKnownNonZero(Val, SimplifyQuery(DL, DT, AC, LI)))
    addAssumeNonNull(AC, LI);
}

static void removeIntrinsicUsers(AllocaInst *AI) {
  // Knowing that this alloca is promotable, we know that it's safe to kill all
  // instructions except for load and store.

  for (Use &U : llvm::make_early_inc_range(AI->uses())) {
    Instruction *I = cast<Instruction>(U.getUser());
    if (isa<LoadInst>(I) || isa<StoreInst>(I))
      continue;

    // Drop the use of AI in droppable instructions.
    if (I->isDroppable()) {
      I->dropDroppableUse(U);
      continue;
    }

    if (!I->getType()->isVoidTy()) {
      // The only users of this bitcast/GEP instruction are lifetime intrinsics.
      // Follow the use/def chain to erase them now instead of leaving it for
      // dead code elimination later.
      for (Use &UU : llvm::make_early_inc_range(I->uses())) {
        Instruction *Inst = cast<Instruction>(UU.getUser());

        // Drop the use of I in droppable instructions.
        if (Inst->isDroppable()) {
          Inst->dropDroppableUse(UU);
          continue;
        }
        Inst->eraseFromParent();
      }
    }
    I->eraseFromParent();
  }
}

/// Rewrite as many loads as possible given a single store.
///
/// When there is only a single store, we can use the domtree to trivially
/// replace all of the dominated loads with the stored value. Do so, and return
/// true if this has successfully promoted the alloca entirely. If this returns
/// false there were some loads which were not dominated by the single store
/// and thus must be phi-ed with undef. We fall back to the standard alloca
/// promotion algorithm in that case.
static bool rewriteSingleStoreAlloca(
    AllocaInst *AI, AllocaInfo &Info, LargeBlockInfo &LBI, const DataLayout &DL,
    DominatorTree &DT, AssumptionCache *AC,
    SmallPtrSet<DbgVariableRecord *, 8> *DVRAssignsToDelete) {
  StoreInst *OnlyStore = Info.OnlyStore;
  Value *ReplVal = OnlyStore->getOperand(0);
  // Loads may either load the stored value or uninitialized memory (undef).
  // If the stored value may be poison, then replacing an uninitialized memory
  // load with it would be incorrect. If the store dominates the load, we know
  // it is always initialized.
  bool RequireDominatingStore =
      isa<Instruction>(ReplVal) || !isGuaranteedNotToBePoison(ReplVal);
  BasicBlock *StoreBB = OnlyStore->getParent();
  int StoreIndex = -1;

  // Clear out UsingBlocks.  We will reconstruct it here if needed.
  Info.UsingBlocks.clear();

  for (User *U : make_early_inc_range(AI->users())) {
    Instruction *UserInst = cast<Instruction>(U);
    if (UserInst == OnlyStore)
      continue;
    LoadInst *LI = cast<LoadInst>(UserInst);

    // Okay, if we have a load from the alloca, we want to replace it with the
    // only value stored to the alloca.  We can do this if the value is
    // dominated by the store.  If not, we use the rest of the mem2reg machinery
    // to insert the phi nodes as needed.
    if (RequireDominatingStore) {
      if (LI->getParent() == StoreBB) {
        // If we have a use that is in the same block as the store, compare the
        // indices of the two instructions to see which one came first.  If the
        // load came before the store, we can't handle it.
        if (StoreIndex == -1)
          StoreIndex = LBI.getInstructionIndex(OnlyStore);

        if (unsigned(StoreIndex) > LBI.getInstructionIndex(LI)) {
          // Can't handle this load, bail out.
          Info.UsingBlocks.push_back(StoreBB);
          continue;
        }
      } else if (!DT.dominates(StoreBB, LI->getParent())) {
        // If the load and store are in different blocks, use BB dominance to
        // check their relationships.  If the store doesn't dom the use, bail
        // out.
        Info.UsingBlocks.push_back(LI->getParent());
        continue;
      }
    }

    // Otherwise, we *can* safely rewrite this load.
    // If the replacement value is the load, this must occur in unreachable
    // code.
    if (ReplVal == LI)
      ReplVal = PoisonValue::get(LI->getType());

    convertMetadataToAssumes(LI, ReplVal, DL, AC, &DT);
    LI->replaceAllUsesWith(ReplVal);
    LI->eraseFromParent();
    LBI.deleteValue(LI);
  }

  // Finally, after the scan, check to see if the store is all that is left.
  if (!Info.UsingBlocks.empty())
    return false; // If not, we'll have to fall back for the remainder.

  DIBuilder DIB(*AI->getModule(), /*AllowUnresolved*/ false);
  // Update assignment tracking info for the store we're going to delete.
  Info.AssignmentTracking.updateForDeletedStore(Info.OnlyStore, DIB,
                                                DVRAssignsToDelete);

  // Record debuginfo for the store and remove the declaration's
  // debuginfo.
  for (DbgVariableRecord *DbgItem : Info.DPUsers) {
    if (DbgItem->isAddressOfVariable()) {
      ConvertDebugDeclareToDebugValue(DbgItem, Info.OnlyStore, DIB);
      DbgItem->eraseFromParent();
    } else if (DbgItem->isValueOfVariable() &&
               DbgItem->getExpression()->startsWithDeref()) {
      InsertDebugValueAtStoreLoc(DbgItem, Info.OnlyStore, DIB);
      DbgItem->eraseFromParent();
    } else if (DbgItem->getExpression()->startsWithDeref()) {
      DbgItem->eraseFromParent();
    }
  }

  // Remove dbg.assigns linked to the alloca as these are now redundant.
  at::deleteAssignmentMarkers(AI);

  // Remove the (now dead) store and alloca.
  Info.OnlyStore->eraseFromParent();
  LBI.deleteValue(Info.OnlyStore);

  AI->eraseFromParent();
  return true;
}

/// Many allocas are only used within a single basic block.  If this is the
/// case, avoid traversing the CFG and inserting a lot of potentially useless
/// PHI nodes by just performing a single linear pass over the basic block
/// using the Alloca.
///
/// If we cannot promote this alloca (because it is read before it is written),
/// return false.  This is necessary in cases where, due to control flow, the
/// alloca is undefined only on some control flow paths.  e.g. code like
/// this is correct in LLVM IR:
///  // A is an alloca with no stores so far
///  for (...) {
///    int t = *A;
///    if (!first_iteration)
///      use(t);
///    *A = 42;
///  }
static bool promoteSingleBlockAlloca(
    AllocaInst *AI, const AllocaInfo &Info, LargeBlockInfo &LBI,
    const DataLayout &DL, DominatorTree &DT, AssumptionCache *AC,
    SmallPtrSet<DbgVariableRecord *, 8> *DVRAssignsToDelete) {
  // The trickiest case to handle is when we have large blocks. Because of this,
  // this code is optimized assuming that large blocks happen.  This does not
  // significantly pessimize the small block case.  This uses LargeBlockInfo to
  // make it efficient to get the index of various operations in the block.

  // Walk the use-def list of the alloca, getting the locations of all stores.
  using StoresByIndexTy = SmallVector<std::pair<unsigned, StoreInst *>, 64>;
  StoresByIndexTy StoresByIndex;

  for (User *U : AI->users())
    if (StoreInst *SI = dyn_cast<StoreInst>(U))
      StoresByIndex.push_back(std::make_pair(LBI.getInstructionIndex(SI), SI));

  // Sort the stores by their index, making it efficient to do a lookup with a
  // binary search.
  llvm::sort(StoresByIndex, less_first());

  // Walk all of the loads from this alloca, replacing them with the nearest
  // store above them, if any.
  for (User *U : make_early_inc_range(AI->users())) {
    LoadInst *LI = dyn_cast<LoadInst>(U);
    if (!LI)
      continue;

    unsigned LoadIdx = LBI.getInstructionIndex(LI);

    // Find the nearest store that has a lower index than this load.
    StoresByIndexTy::iterator I = llvm::lower_bound(
        StoresByIndex,
        std::make_pair(LoadIdx, static_cast<StoreInst *>(nullptr)),
        less_first());
    Value *ReplVal;
    if (I == StoresByIndex.begin()) {
      if (StoresByIndex.empty())
        // If there are no stores, the load takes the undef value.
        ReplVal = UndefValue::get(LI->getType());
      else
        // There is no store before this load, bail out (load may be affected
        // by the following stores - see main comment).
        return false;
    } else {
      // Otherwise, there was a store before this load, the load takes its
      // value.
      ReplVal = std::prev(I)->second->getOperand(0);
    }

    convertMetadataToAssumes(LI, ReplVal, DL, AC, &DT);

    // If the replacement value is the load, this must occur in unreachable
    // code.
    if (ReplVal == LI)
      ReplVal = PoisonValue::get(LI->getType());

    LI->replaceAllUsesWith(ReplVal);
    LI->eraseFromParent();
    LBI.deleteValue(LI);
  }

  // Remove the (now dead) stores and alloca.
  DIBuilder DIB(*AI->getModule(), /*AllowUnresolved*/ false);
  while (!AI->use_empty()) {
    StoreInst *SI = cast<StoreInst>(AI->user_back());
    // Update assignment tracking info for the store we're going to delete.
    Info.AssignmentTracking.updateForDeletedStore(SI, DIB, DVRAssignsToDelete);
    // Record debuginfo for the store before removing it.
    for (DbgVariableRecord *DbgItem : Info.DPUsers) {
      if (DbgItem->isAddressOfVariable()) {
        ConvertDebugDeclareToDebugValue(DbgItem, SI, DIB);
      }
    }

    SI->eraseFromParent();
    LBI.deleteValue(SI);
  }

  // Remove dbg.assigns linked to the alloca as these are now redundant.
  at::deleteAssignmentMarkers(AI);
  AI->eraseFromParent();

  // The alloca's debuginfo can be removed as well.
  for (DbgVariableRecord *DbgItem : Info.DPUsers) {
    if (DbgItem->isAddressOfVariable() ||
        DbgItem->getExpression()->startsWithDeref())
      DbgItem->eraseFromParent();
  }

  ++NumLocalPromoted;
  return true;
}

void PromoteMem2Reg::run() {
  Function &F = *DT.getRoot()->getParent();

  AllocaATInfo.resize(Allocas.size());
  AllocaDPUsers.resize(Allocas.size());

  AllocaInfo Info;
  LargeBlockInfo LBI;
  ForwardIDFCalculator IDF(DT);

  NoSignedZeros = F.getFnAttribute("no-signed-zeros-fp-math").getValueAsBool();

  std::unique_ptr<VectorizedMem2Reg> VM2R;
  if (UseVectorizedMem2Reg)
    VM2R.reset(new VectorizedMem2Reg(this));

  llvm::SmallVector<unsigned, 64> AllocaNums;
  auto ProcessAllocaBatch = [&]() {
    VM2R->Calculate();
    for (size_t I = 0; I < VM2R->size(); ++I) {
      unsigned CurrentVersion = 0;
      for (BasicBlock *BB : VM2R->GetPHIBlocks(I))
        QueuePhiNode(BB, AllocaNums[I], CurrentVersion);
    }
    VM2R->clear();
    AllocaNums.clear();
  };

  for (unsigned AllocaNum = 0; AllocaNum != Allocas.size(); ++AllocaNum) {
    AllocaInst *AI = Allocas[AllocaNum];

    assert(isAllocaPromotable(AI) && "Cannot promote non-promotable alloca!");
    assert(AI->getParent()->getParent() == &F &&
           "All allocas should be in the same function, which is same as DF!");

    removeIntrinsicUsers(AI);

    if (AI->use_empty()) {
      // If there are no uses of the alloca, just delete it now.
      AI->eraseFromParent();

      // Remove the alloca from the Allocas list, since it has been processed
      RemoveFromAllocasList(AllocaNum);
      ++NumDeadAlloca;
      continue;
    }

    // Calculate the set of read and write-locations for each alloca.  This is
    // analogous to finding the 'uses' and 'definitions' of each variable.
    Info.AnalyzeAlloca(AI);

    // If there is only a single store to this value, replace any loads of
    // it that are directly dominated by the definition with the value stored.
    if (Info.DefiningBlocks.size() == 1) {
      if (rewriteSingleStoreAlloca(AI, Info, LBI, SQ.DL, DT, AC,
                                   &DVRAssignsToDelete)) {
        // The alloca has been processed, move on.
        RemoveFromAllocasList(AllocaNum);
        ++NumSingleStore;
        continue;
      }
    }

    // If the alloca is only read and written in one basic block, just perform a
    // linear sweep over the block to eliminate it.
    if (Info.OnlyUsedInOneBlock &&
        promoteSingleBlockAlloca(AI, Info, LBI, SQ.DL, DT, AC,
                                 &DVRAssignsToDelete)) {
      // The alloca has been processed, move on.
      RemoveFromAllocasList(AllocaNum);
      continue;
    }

    // Initialize BBNumPreds lazily
    if (BBNumPreds.empty())
      BBNumPreds.resize(F.getMaxBlockNumber());

    // Remember the dbg.declare record describing this alloca, if any.
    if (!Info.AssignmentTracking.empty())
      AllocaATInfo[AllocaNum] = Info.AssignmentTracking;
    if (!Info.DPUsers.empty())
      AllocaDPUsers[AllocaNum] = Info.DPUsers;

    // Keep the reverse mapping of the 'Allocas' array for the rename pass.
    AllocaLookup[Allocas[AllocaNum]] = AllocaNum;

    if (VM2R) {
      AllocaNums.push_back(AllocaNum);
      if (VM2R->GatherAlloca(AI, Info))
        ProcessAllocaBatch();
      continue;
    }

    // Unique the set of defining blocks for efficient lookup.
    SmallPtrSet<BasicBlock *, 32> DefBlocks(llvm::from_range,
                                            Info.DefiningBlocks);

    // Determine which blocks the value is live in.  These are blocks which lead
    // to uses.
    SmallPtrSet<BasicBlock *, 32> LiveInBlocks;
    ComputeLiveInBlocks(AI, Info, DefBlocks, LiveInBlocks);

    // At this point, we're committed to promoting the alloca using IDF's, and
    // the standard SSA construction algorithm.  Determine which blocks need phi
    // nodes and see if we can optimize out some work by avoiding insertion of
    // dead phi nodes.
    IDF.setLiveInBlocks(LiveInBlocks);
    IDF.setDefiningBlocks(DefBlocks);
    SmallVector<BasicBlock *, 32> PHIBlocks;
    IDF.calculate(PHIBlocks);
    llvm::sort(PHIBlocks, [](BasicBlock *A, BasicBlock *B) {
      return A->getNumber() < B->getNumber();
    });

    unsigned CurrentVersion = 0;
    for (BasicBlock *BB : PHIBlocks)
      QueuePhiNode(BB, AllocaNum, CurrentVersion);
  }
  // Process the last batch.
  if (VM2R && !AllocaNums.empty())
    ProcessAllocaBatch();

  if (Allocas.empty()) {
    cleanUpDbgAssigns();
    return; // All of the allocas must have been trivial!
  }
  LBI.clear();

  // Set the incoming values for the basic block to be null values for all of
  // the alloca's.  We do this in case there is a load of a value that has not
  // been stored yet.  In this case, it will get this null value.
  IncomingVals.resize(Allocas.size());
  for (unsigned i = 0, e = Allocas.size(); i != e; ++i)
    IncomingVals.init(i, UndefValue::get(Allocas[i]->getAllocatedType()));

  // When handling debug info, treat all incoming values as if they have
  // compiler-generated (empty) locations, representing the uninitialized
  // alloca, until proven otherwise.
  IncomingLocs.resize(Allocas.size());
  for (unsigned i = 0, e = Allocas.size(); i != e; ++i)
    IncomingLocs.init(i, DebugLoc::getCompilerGenerated());

  // The renamer uses the Visited set to avoid infinite loops.
  Visited.resize(F.getMaxBlockNumber(), false);

  // Add the entry block to the worklist, with a null predecessor.
  pushToWorklist(&F.front(), nullptr);

  do {
    RenamePassData RPD = popFromWorklist();
    RenamePass(RPD.BB, RPD.Pred);
  } while (!Worklist.empty());

  // Remove the allocas themselves from the function.
  for (Instruction *A : Allocas) {
    // Remove dbg.assigns linked to the alloca as these are now redundant.
    at::deleteAssignmentMarkers(A);
    // If there are any uses of the alloca instructions left, they must be in
    // unreachable basic blocks that were not processed by walking the dominator
    // tree. Just delete the users now.
    if (!A->use_empty())
      A->replaceAllUsesWith(PoisonValue::get(A->getType()));
    A->eraseFromParent();
  }

  // Remove alloca's dbg.declare intrinsics from the function.
  for (auto &DbgUsers : AllocaDPUsers) {
    for (DbgVariableRecord *DbgItem : DbgUsers)
      if (DbgItem->isAddressOfVariable() ||
          DbgItem->getExpression()->startsWithDeref())
        DbgItem->eraseFromParent();
  }

  // Loop over all of the PHI nodes and see if there are any that we can get
  // rid of because they merge all of the same incoming values.  This can
  // happen due to undef values coming into the PHI nodes.  This process is
  // iterative, because eliminating one PHI node can cause others to be removed.
  bool EliminatedAPHI = true;
  while (EliminatedAPHI) {
    EliminatedAPHI = false;

    // Iterating over NewPhiNodes is deterministic, so it is safe to try to
    // simplify and RAUW them as we go.  If it was not, we could add uses to
    // the values we replace with in a non-deterministic order, thus creating
    // non-deterministic def->use chains.
    for (DenseMap<std::pair<unsigned, unsigned>, PHINode *>::iterator
             I = NewPhiNodes.begin(),
             E = NewPhiNodes.end();
         I != E;) {
      PHINode *PN = I->second;

      // If this PHI node merges one value and/or undefs, get the value.
      if (Value *V = simplifyInstruction(PN, SQ)) {
        PN->replaceAllUsesWith(V);
        PN->eraseFromParent();
        NewPhiNodes.erase(I++);
        EliminatedAPHI = true;
        continue;
      }
      ++I;
    }
  }

  // At this point, the renamer has added entries to PHI nodes for all reachable
  // code.  Unfortunately, there may be unreachable blocks which the renamer
  // hasn't traversed.  If this is the case, the PHI nodes may not
  // have incoming values for all predecessors.  Loop over all PHI nodes we have
  // created, inserting poison values if they are missing any incoming values.
  for (const auto &PhiNode : NewPhiNodes) {
    // We want to do this once per basic block.  As such, only process a block
    // when we find the PHI that is the first entry in the block.
    PHINode *SomePHI = PhiNode.second;
    BasicBlock *BB = SomePHI->getParent();
    if (&BB->front() != SomePHI)
      continue;

    // Only do work here if there the PHI nodes are missing incoming values.  We
    // know that all PHI nodes that were inserted in a block will have the same
    // number of incoming values, so we can just check any of them.
    if (SomePHI->getNumIncomingValues() == getNumPreds(BB))
      continue;

    // Get the preds for BB.
    SmallVector<BasicBlock *, 16> Preds(predecessors(BB));

    // Ok, now we know that all of the PHI nodes are missing entries for some
    // basic blocks.  Start by sorting the incoming predecessors for efficient
    // access.
    auto CompareBBNumbers = [](BasicBlock *A, BasicBlock *B) {
      return A->getNumber() < B->getNumber();
    };
    llvm::sort(Preds, CompareBBNumbers);

    // Now we loop through all BB's which have entries in SomePHI and remove
    // them from the Preds list.
    for (unsigned i = 0, e = SomePHI->getNumIncomingValues(); i != e; ++i) {
      // Do a log(n) search of the Preds list for the entry we want.
      SmallVectorImpl<BasicBlock *>::iterator EntIt = llvm::lower_bound(
          Preds, SomePHI->getIncomingBlock(i), CompareBBNumbers);
      assert(EntIt != Preds.end() && *EntIt == SomePHI->getIncomingBlock(i) &&
             "PHI node has entry for a block which is not a predecessor!");

      // Remove the entry
      Preds.erase(EntIt);
    }

    // At this point, the blocks left in the preds list must have dummy
    // entries inserted into every PHI nodes for the block.  Update all the phi
    // nodes in this block that we are inserting (there could be phis before
    // mem2reg runs).
    unsigned NumBadPreds = SomePHI->getNumIncomingValues();
    BasicBlock::iterator BBI = BB->begin();
    while ((SomePHI = dyn_cast<PHINode>(BBI++)) &&
           SomePHI->getNumIncomingValues() == NumBadPreds) {
      Value *PoisonVal = PoisonValue::get(SomePHI->getType());
      for (BasicBlock *Pred : Preds)
        SomePHI->addIncoming(PoisonVal, Pred);
    }
  }

  NewPhiNodes.clear();
  cleanUpDbgAssigns();
}

/// Determine which blocks the value is live in.
///
/// These are blocks which lead to uses.  Knowing this allows us to avoid
/// inserting PHI nodes into blocks which don't lead to uses (thus, the
/// inserted phi nodes would be dead).
void PromoteMem2Reg::ComputeLiveInBlocks(
    AllocaInst *AI, AllocaInfo &Info,
    const SmallPtrSetImpl<BasicBlock *> &DefBlocks,
    SmallPtrSetImpl<BasicBlock *> &LiveInBlocks) {
  // To determine liveness, we must iterate through the predecessors of blocks
  // where the def is live.  Blocks are added to the worklist if we need to
  // check their predecessors.  Start with all the using blocks.
  SmallVector<BasicBlock *, 64> LiveInBlockWorklist(Info.UsingBlocks.begin(),
                                                    Info.UsingBlocks.end());

  // If any of the using blocks is also a definition block, check to see if the
  // definition occurs before or after the use.  If it happens before the use,
  // the value isn't really live-in.
  for (unsigned i = 0, e = LiveInBlockWorklist.size(); i != e; ++i) {
    BasicBlock *BB = LiveInBlockWorklist[i];
    if (!DefBlocks.count(BB))
      continue;

    // Okay, this is a block that both uses and defines the value.  If the first
    // reference to the alloca is a def (store), then we know it isn't live-in.
    for (BasicBlock::iterator I = BB->begin();; ++I) {
      if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
        if (SI->getOperand(1) != AI)
          continue;

        // We found a store to the alloca before a load.  The alloca is not
        // actually live-in here.
        LiveInBlockWorklist[i] = LiveInBlockWorklist.back();
        LiveInBlockWorklist.pop_back();
        --i;
        --e;
        break;
      }

      if (LoadInst *LI = dyn_cast<LoadInst>(I))
        // Okay, we found a load before a store to the alloca.  It is actually
        // live into this block.
        if (LI->getOperand(0) == AI)
          break;
    }
  }

  // Now that we have a set of blocks where the phi is live-in, recursively add
  // their predecessors until we find the full region the value is live.
  while (!LiveInBlockWorklist.empty()) {
    BasicBlock *BB = LiveInBlockWorklist.pop_back_val();

    // The block really is live in here, insert it into the set.  If already in
    // the set, then it has already been processed.
    if (!LiveInBlocks.insert(BB).second)
      continue;

    // Since the value is live into BB, it is either defined in a predecessor or
    // live into it to.  Add the preds to the worklist unless they are a
    // defining block.
    for (BasicBlock *P : predecessors(BB)) {
      // The value is not live into a predecessor if it defines the value.
      if (DefBlocks.count(P))
        continue;

      // Otherwise it is, add to the worklist.
      LiveInBlockWorklist.push_back(P);
    }
  }
}

bool VectorizedMem2Reg::GatherAlloca(AllocaInst *AI, const AllocaInfo &Info) {
  assert(Allocas.size() < MaxAllocaNum && "Allocas vector is full.");
  // Add new alloca to the current batch.
  size_t Index = Allocas.size();
  Allocas.push_back(AI);

  // Populate DEF states.
  for (BasicBlock* Def : Info.DefiningBlocks) {
    // We need to calculate IDF of every DEF block, adding them to the PQ here
    // so that a BB is only added once at most.
    if (BlockStates.get<DEF_STATE>(Def->getNumber()).none())
      if (DomTreeNodeBase<BasicBlock> *Node = DT->getNode(Def))
        PQ.push({Node, std::make_pair(Node->getLevel(), Node->getDFSNumIn())});
    BlockStates.get<DEF_STATE>(Def->getNumber()).set(Index);
  }

  // Initialize Worklist to compute ALIVE state. Find all uses of the value
  // where it is defined in another block and add them to Worklist.
  for (BasicBlock *Use : Info.UsingBlocks) {
    BBNumberTy BN = Use->getNumber();

    // If the use block is not the def block, the use block is live-in. It is
    // possible that a previous alloca lives in this block, so we should merge
    // their UPDATE states.
    if (!BlockStates.get<DEF_STATE>(BN)[Index]) {
      PushWorkList(BN, AllocaState().set(Index));
      continue;
    }

    // If use and def happens in the same block, check if the def occurs before
    // the use, in this case the value is not live-in at this block.
    for (BasicBlock::iterator I = Use->begin();; ++I) {
      if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
        if (SI->getOperand(1) != AI)
          continue;

        // We found a store to the alloca before a load. The alloca is not
        // actually live-in here.
        break;
      }

      if (LoadInst *LI = dyn_cast<LoadInst>(I))
        // Okay, we found a load before a store to the alloca.  It is actually
        // live into this block. Add it to the worklist.
        if (LI->getOperand(0) == AI) {
          PushWorkList(BN, AllocaState().set(Index));
          break;
        }
    }
  }

  return Allocas.size() == MaxAllocaNum;
}

void VectorizedMem2Reg::Calculate() {
  // @TODO: Assign BB number in a way such that the BB deepest in the CFG has
  // the largest number, and traverse it first using a priority queue. This
  // allows faster convergence to the fixed-point.

  // Compute ALIVE state for every block.
  // Now Worklist is initialized with blocks containing any live-in allocas. We
  // recursively add their predecessors to Worklist until we find the full
  // region where the value is alive. Whenever a block's ALIVE state is updated,
  // we need to check if the state value is actually modified, if so, we need
  // to iterate its predecessors again to propagate the new state until reaching
  // a fixed point.
  while (!Worklist.empty()) {
    auto [BN, State] = PopWorkList();

    // Update the ALIVE state of this block. If the state remains unchanged, we
    // have reached a fixed point, there is no more new liveness info to be
    // propagated to its predecessors.
    AllocaState OldState = BlockStates.get<ALIVE_STATE>(BN);
    AllocaState NewState = (BlockStates.get<ALIVE_STATE>(BN) |= State);
    if (NewState == OldState)
      continue;

    // If a fixed point is not reached, this is because either block BN is
    // visited for the first time, or a loop in the CFG brings new liveness info
    // back to this block. Either case, we add its predecessors to the worklist.
    for (BBNumberTy Pred : Predecessors[BN]) {
      // The value is not ALIVE in a predecessor if it contains a DEF, so we
      // need to exclude such values, and only find those values not defined in
      // this block.
      AllocaState UpdateState = NewState & ~BlockStates.get<DEF_STATE>(Pred);
      // Only add to Worklist if there is any value ALIVE at Pred.
      if (UpdateState.any())
        PushWorkList(Pred, UpdateState);
    }
  }

  // Initialize UPDATE states of blocks in PQ to maintain the invaraince. We
  // calculate IDF of every DEF, so the initial UPDATE state is DEF state.
  for (auto &Node : GetPQContainer()) {
    unsigned BN = Node.first->getBlock()->getNumber();
    BlockStates.get<UPDATE_STATE>(BN) = BlockStates.get<DEF_STATE>(BN);
  }

  // Compute IDF for every block containing alloca. Visiting blocks from the
  // largest to the smallest DT level number.
  while (!PQ.empty()) {
    // RootState is the values available at Root, which will be propagated to
    // the successors of its dominatees per the algorithm of IDF.
    auto [Root, RootState] = PopPQ();
    unsigned RootLevel = Root->getLevel();
    BBNumberTy RootBN = Root->getBlock()->getNumber();

    // Perform one iteration of dominance frontier computation on all blocks
    // dominated by root. Here Worklist is not associated with UPDATE state
    // because visited nodes are updated with the same RootState instead.
    Worklist.push_back(RootBN);
    while (!Worklist.empty()) {
      unsigned BN = Worklist.back();
      Worklist.pop_back();

      for (BBNumberTy Succ : Successors[BN]) {
        auto SuccNode = DT->getNode(Succ);
        unsigned SuccLevel = SuccNode->getLevel();

        // Successor node Succ with higher level in DT must be dominated by
        // current node BN, so PHI will not be placed in it.
        if (SuccLevel > RootLevel)
          continue;

        // Update IDF state of Succ by merging its previous state with available
        // values from Root. Values no longer alive need not to be propagated,
        // so that the algorithm converges faster.
        AllocaState AliveState = BlockStates.get<ALIVE_STATE>(Succ);
        AllocaState OldState = BlockStates.get<IDF_STATE>(Succ);
        AllocaState NewState =
            (BlockStates.get<IDF_STATE>(Succ) |= (RootState & AliveState));
        // If IDF state is unchanged, we reached a fixed point, and there will
        // be no more new value to propagate. This includes the case that no
        // value from Root is alive at Succ ((RootState & AliveState) == 0).
        if (NewState == OldState)
          continue;

        // We always filter UPDATE state with ALIVE state, so it is an invariant
        // that IDF values are a subset of ALIVE values.
        assert((AliveState | OldState) == AliveState);
        assert((AliveState | NewState) == AliveState);

        // Any newly set bit in IDF state represents inserted PHI, add it to the
        // output.
        AllocaState Inserted = NewState ^ OldState;
        do {
          size_t Index = 0;
          if constexpr (MaxAllocaNum <= sizeof(unsigned long long) * CHAR_BIT) {
            Index = llvm::countr_zero(Inserted.to_ullong());
          } else {
            while (!Inserted.test(Index))
              ++Index;
          }
          PHIBlocks[Index].push_back(Succ);
          Inserted.reset(Index);
        } while (Inserted.any());

        // If any new PHI is inserted at Succ, we need to iterate it too since
        // it will propagate the PHI to blocks it does not dominate. An existing
        // value is killed by DEF so the UPDATE state should exclude it.
        AllocaState UpdateState = NewState & ~BlockStates.get<DEF_STATE>(Succ);
        if (UpdateState.any())
          PushPQ(SuccNode, UpdateState);
      }

      // Visit every node in DT subtree.
      for (auto DomChild : *(DT->getNode(BN))) {
        BBNumberTy DomChildBN = DomChild->getBlock()->getNumber();
        // Since any value available at the dominator is available at the child
        // node, we merge the dominator's IDF state into it. If the child's IDF
        // state is unchanged, we reached a fixed point, so we do not need to
        // visit it. Values no longer alive need not to be propagated, so that
        // the algorithm converges faster.
        AllocaState OldState = BlockStates.get<IDF_STATE>(DomChildBN);
        AllocaState NewState = BlockStates.get<IDF_STATE>(DomChildBN) |=
            (RootState & BlockStates.get<ALIVE_STATE>(DomChildBN));
        // Since DT is a tree, there will be no dups in Worklist.
        if (OldState != NewState)
          Worklist.push_back(DomChildBN);
      }
    }
  }

  // Order inserted PHI nodes in a deterministic way.
  for (size_t I = 0; I < Allocas.size(); ++I)
    llvm::sort(PHIBlocks[I]);
}

/// Queue a phi-node to be added to a basic-block for a specific Alloca.
///
/// Returns true if there wasn't already a phi-node for that variable
bool PromoteMem2Reg::QueuePhiNode(BasicBlock *BB, unsigned AllocaNo,
                                  unsigned &Version) {
  // Look up the basic-block in question.
  PHINode *&PN = NewPhiNodes[std::make_pair(BB->getNumber(), AllocaNo)];

  // If the BB already has a phi node added for the i'th alloca then we're done!
  if (PN)
    return false;

  // Create a PhiNode using the dereferenced type... and add the phi-node to the
  // BasicBlock.
  PN = PHINode::Create(Allocas[AllocaNo]->getAllocatedType(), getNumPreds(BB),
                       Allocas[AllocaNo]->getName() + "." + Twine(Version++));
  PN->insertBefore(BB->begin());
  ++NumPHIInsert;
  PhiToAllocaMap[PN] = AllocaNo;
  return true;
}

/// Update the debug location of a phi. \p ApplyMergedLoc indicates whether to
/// create a merged location incorporating \p DL, or to set \p DL directly.
static void updateForIncomingValueLocation(PHINode *PN, DebugLoc DL,
                                           bool ApplyMergedLoc) {
  if (ApplyMergedLoc)
    PN->applyMergedLocation(PN->getDebugLoc(), DL);
  else
    PN->setDebugLoc(DL);
}

/// Recursively traverse the CFG of the function, renaming loads and
/// stores to the allocas which we are promoting.
///
/// IncomingVals indicates what value each Alloca contains on exit from the
/// predecessor block Pred.
void PromoteMem2Reg::RenamePass(BasicBlock *BB, BasicBlock *Pred) {
  // If we are inserting any phi nodes into this BB, they will already be in the
  // block.
  if (PHINode *APN = dyn_cast<PHINode>(BB->begin())) {
    // If we have PHI nodes to update, compute the number of edges from Pred to
    // BB.
    if (PhiToAllocaMap.count(APN)) {
      // We want to be able to distinguish between PHI nodes being inserted by
      // this invocation of mem2reg from those phi nodes that already existed in
      // the IR before mem2reg was run.  We determine that APN is being inserted
      // because it is missing incoming edges.  All other PHI nodes being
      // inserted by this pass of mem2reg will have the same number of incoming
      // operands so far.  Remember this count.
      unsigned NewPHINumOperands = APN->getNumOperands();

      unsigned NumEdges = llvm::count(successors(Pred), BB);
      assert(NumEdges && "Must be at least one edge from Pred to BB!");

      // Add entries for all the phis.
      BasicBlock::iterator PNI = BB->begin();
      do {
        unsigned AllocaNo = PhiToAllocaMap[APN];

        // Update the location of the phi node.
        updateForIncomingValueLocation(APN, IncomingLocs[AllocaNo],
                                       APN->getNumIncomingValues() > 0);

        // Add N incoming values to the PHI node.
        for (unsigned i = 0; i != NumEdges; ++i)
          APN->addIncoming(IncomingVals[AllocaNo], Pred);

        // For the  sequence `return X > 0.0 ? X : -X`, it is expected that this
        // results in fabs intrinsic. However, without no-signed-zeros(nsz) flag
        // on the phi node generated at this stage, fabs folding does not
        // happen. So, we try to infer nsz flag from the function attributes to
        // enable this fabs folding.
        if (isa<FPMathOperator>(APN) && NoSignedZeros)
          APN->setHasNoSignedZeros(true);

        // The currently active variable for this block is now the PHI.
        IncomingVals.set(AllocaNo, APN);
        AllocaATInfo[AllocaNo].updateForNewPhi(APN, DIB);
        for (DbgVariableRecord *DbgItem : AllocaDPUsers[AllocaNo])
          if (DbgItem->isAddressOfVariable())
            ConvertDebugDeclareToDebugValue(DbgItem, APN, DIB);

        // Get the next phi node.
        ++PNI;
        APN = dyn_cast<PHINode>(PNI);
        if (!APN)
          break;

        // Verify that it is missing entries.  If not, it is not being inserted
        // by this mem2reg invocation so we want to ignore it.
      } while (APN->getNumOperands() == NewPHINumOperands);
    }
  }

  // Don't revisit blocks.
  if (Visited.test(BB->getNumber()))
    return;
  Visited.set(BB->getNumber());

  for (BasicBlock::iterator II = BB->begin(); !II->isTerminator();) {
    Instruction *I = &*II++; // get the instruction, increment iterator

    if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
      AllocaInst *Src = dyn_cast<AllocaInst>(LI->getPointerOperand());
      if (!Src)
        continue;

      DenseMap<AllocaInst *, unsigned>::iterator AI = AllocaLookup.find(Src);
      if (AI == AllocaLookup.end())
        continue;

      Value *V = IncomingVals[AI->second];
      convertMetadataToAssumes(LI, V, SQ.DL, AC, &DT);

      // Anything using the load now uses the current value.
      LI->replaceAllUsesWith(V);
      LI->eraseFromParent();
    } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
      // Delete this instruction and mark the name as the current holder of the
      // value
      AllocaInst *Dest = dyn_cast<AllocaInst>(SI->getPointerOperand());
      if (!Dest)
        continue;

      DenseMap<AllocaInst *, unsigned>::iterator ai = AllocaLookup.find(Dest);
      if (ai == AllocaLookup.end())
        continue;

      // what value were we writing?
      unsigned AllocaNo = ai->second;
      IncomingVals.set(AllocaNo, SI->getOperand(0));

      // Record debuginfo for the store before removing it.
      IncomingLocs.set(AllocaNo, SI->getDebugLoc());
      AllocaATInfo[AllocaNo].updateForDeletedStore(SI, DIB,
                                                   &DVRAssignsToDelete);
      for (DbgVariableRecord *DbgItem : AllocaDPUsers[ai->second])
        if (DbgItem->isAddressOfVariable())
          ConvertDebugDeclareToDebugValue(DbgItem, SI, DIB);
      SI->eraseFromParent();
    }
  }

  // 'Recurse' to our successors.

  // Keep track of the successors so we don't visit the same successor twice
  SmallPtrSet<BasicBlock *, 8> VisitedSuccs;

  for (BasicBlock *S : reverse(successors(BB)))
    if (VisitedSuccs.insert(S).second)
      pushToWorklist(S, BB);
}

void llvm::PromoteMemToReg(ArrayRef<AllocaInst *> Allocas, DominatorTree &DT,
                           AssumptionCache *AC) {
  // If there is nothing to do, bail out...
  if (Allocas.empty())
    return;

  PromoteMem2Reg(Allocas, DT, AC).run();
}
