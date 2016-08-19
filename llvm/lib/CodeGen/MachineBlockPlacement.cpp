//===-- MachineBlockPlacement.cpp - Basic Block Code Layout optimization --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements basic block placement transformations using the CFG
// structure and branch probability estimates.
//
// The pass strives to preserve the structure of the CFG (that is, retain
// a topological ordering of basic blocks) in the absence of a *strong* signal
// to the contrary from probabilities. However, within the CFG structure, it
// attempts to choose an ordering which favors placing more likely sequences of
// blocks adjacent to each other.
//
// The algorithm works from the inner-most loop within a function outward, and
// at each stage walks through the basic blocks, trying to coalesce them into
// sequential chains where allowed by the CFG (or demanded by heavy
// probabilities). Finally, it walks the blocks in topological order, and the
// first time it reaches a chain of basic blocks, it schedules them in the
// function in-order.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "BranchFolding.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <algorithm>
using namespace llvm;

#define DEBUG_TYPE "block-placement"

STATISTIC(NumCondBranches, "Number of conditional branches");
STATISTIC(NumUncondBranches, "Number of unconditional branches");
STATISTIC(CondBranchTakenFreq,
          "Potential frequency of taking conditional branches");
STATISTIC(UncondBranchTakenFreq,
          "Potential frequency of taking unconditional branches");

static cl::opt<unsigned> AlignAllBlock("align-all-blocks",
                                       cl::desc("Force the alignment of all "
                                                "blocks in the function."),
                                       cl::init(0), cl::Hidden);

static cl::opt<unsigned> AlignAllNonFallThruBlocks(
    "align-all-nofallthru-blocks",
    cl::desc("Force the alignment of all "
             "blocks that have no fall-through predecessors (i.e. don't add "
             "nops that are executed)."),
    cl::init(0), cl::Hidden);

// FIXME: Find a good default for this flag and remove the flag.
static cl::opt<unsigned> ExitBlockBias(
    "block-placement-exit-block-bias",
    cl::desc("Block frequency percentage a loop exit block needs "
             "over the original exit to be considered the new exit."),
    cl::init(0), cl::Hidden);

// Definition:
// - Outlining: placement of a basic block outside the chain or hot path.

static cl::opt<bool> OutlineOptionalBranches(
    "outline-optional-branches",
    cl::desc("Outlining optional branches will place blocks that are optional "
              "branches, i.e. branches with a common post dominator, outside "
              "the hot path or chain"),
    cl::init(false), cl::Hidden);

static cl::opt<unsigned> OutlineOptionalThreshold(
    "outline-optional-threshold",
    cl::desc("Don't outline optional branches that are a single block with an "
             "instruction count below this threshold"),
    cl::init(4), cl::Hidden);

static cl::opt<unsigned> LoopToColdBlockRatio(
    "loop-to-cold-block-ratio",
    cl::desc("Outline loop blocks from loop chain if (frequency of loop) / "
             "(frequency of block) is greater than this ratio"),
    cl::init(5), cl::Hidden);

static cl::opt<bool>
    PreciseRotationCost("precise-rotation-cost",
                        cl::desc("Model the cost of loop rotation more "
                                 "precisely by using profile data."),
                        cl::init(false), cl::Hidden);
static cl::opt<bool>
    ForcePreciseRotationCost("force-precise-rotation-cost",
                             cl::desc("Force the use of precise cost "
                                      "loop rotation strategy."),
                             cl::init(false), cl::Hidden);

static cl::opt<unsigned> MisfetchCost(
    "misfetch-cost",
    cl::desc("Cost that models the probabilistic risk of an instruction "
             "misfetch due to a jump comparing to falling through, whose cost "
             "is zero."),
    cl::init(1), cl::Hidden);

static cl::opt<unsigned> JumpInstCost("jump-inst-cost",
                                      cl::desc("Cost of jump instructions."),
                                      cl::init(1), cl::Hidden);

static cl::opt<bool>
BranchFoldPlacement("branch-fold-placement",
              cl::desc("Perform branch folding during placement. "
                       "Reduces code size."),
              cl::init(true), cl::Hidden);

extern cl::opt<unsigned> StaticLikelyProb;
extern cl::opt<unsigned> ProfileLikelyProb;

namespace {
class BlockChain;
/// \brief Type for our function-wide basic block -> block chain mapping.
typedef DenseMap<MachineBasicBlock *, BlockChain *> BlockToChainMapType;
}

namespace {
/// \brief A chain of blocks which will be laid out contiguously.
///
/// This is the datastructure representing a chain of consecutive blocks that
/// are profitable to layout together in order to maximize fallthrough
/// probabilities and code locality. We also can use a block chain to represent
/// a sequence of basic blocks which have some external (correctness)
/// requirement for sequential layout.
///
/// Chains can be built around a single basic block and can be merged to grow
/// them. They participate in a block-to-chain mapping, which is updated
/// automatically as chains are merged together.
class BlockChain {
  /// \brief The sequence of blocks belonging to this chain.
  ///
  /// This is the sequence of blocks for a particular chain. These will be laid
  /// out in-order within the function.
  SmallVector<MachineBasicBlock *, 4> Blocks;

  /// \brief A handle to the function-wide basic block to block chain mapping.
  ///
  /// This is retained in each block chain to simplify the computation of child
  /// block chains for SCC-formation and iteration. We store the edges to child
  /// basic blocks, and map them back to their associated chains using this
  /// structure.
  BlockToChainMapType &BlockToChain;

public:
  /// \brief Construct a new BlockChain.
  ///
  /// This builds a new block chain representing a single basic block in the
  /// function. It also registers itself as the chain that block participates
  /// in with the BlockToChain mapping.
  BlockChain(BlockToChainMapType &BlockToChain, MachineBasicBlock *BB)
      : Blocks(1, BB), BlockToChain(BlockToChain), UnscheduledPredecessors(0) {
    assert(BB && "Cannot create a chain with a null basic block");
    BlockToChain[BB] = this;
  }

  /// \brief Iterator over blocks within the chain.
  typedef SmallVectorImpl<MachineBasicBlock *>::iterator iterator;

  /// \brief Beginning of blocks within the chain.
  iterator begin() { return Blocks.begin(); }

  /// \brief End of blocks within the chain.
  iterator end() { return Blocks.end(); }

  /// \brief Merge a block chain into this one.
  ///
  /// This routine merges a block chain into this one. It takes care of forming
  /// a contiguous sequence of basic blocks, updating the edge list, and
  /// updating the block -> chain mapping. It does not free or tear down the
  /// old chain, but the old chain's block list is no longer valid.
  void merge(MachineBasicBlock *BB, BlockChain *Chain) {
    assert(BB);
    assert(!Blocks.empty());

    // Fast path in case we don't have a chain already.
    if (!Chain) {
      assert(!BlockToChain[BB]);
      Blocks.push_back(BB);
      BlockToChain[BB] = this;
      return;
    }

    assert(BB == *Chain->begin());
    assert(Chain->begin() != Chain->end());

    // Update the incoming blocks to point to this chain, and add them to the
    // chain structure.
    for (MachineBasicBlock *ChainBB : *Chain) {
      Blocks.push_back(ChainBB);
      assert(BlockToChain[ChainBB] == Chain && "Incoming blocks not in chain");
      BlockToChain[ChainBB] = this;
    }
  }

#ifndef NDEBUG
  /// \brief Dump the blocks in this chain.
  LLVM_DUMP_METHOD void dump() {
    for (MachineBasicBlock *MBB : *this)
      MBB->dump();
  }
#endif // NDEBUG

  /// \brief Count of predecessors of any block within the chain which have not
  /// yet been scheduled.  In general, we will delay scheduling this chain
  /// until those predecessors are scheduled (or we find a sufficiently good
  /// reason to override this heuristic.)  Note that when forming loop chains,
  /// blocks outside the loop are ignored and treated as if they were already
  /// scheduled.
  ///
  /// Note: This field is reinitialized multiple times - once for each loop,
  /// and then once for the function as a whole.
  unsigned UnscheduledPredecessors;
};
}

namespace {
class MachineBlockPlacement : public MachineFunctionPass {
  /// \brief A typedef for a block filter set.
  typedef SmallPtrSet<MachineBasicBlock *, 16> BlockFilterSet;

  /// \brief work lists of blocks that are ready to be laid out
  SmallVector<MachineBasicBlock *, 16> BlockWorkList;
  SmallVector<MachineBasicBlock *, 16> EHPadWorkList;

  /// \brief Machine Function
  MachineFunction *F;

  /// \brief A handle to the branch probability pass.
  const MachineBranchProbabilityInfo *MBPI;

  /// \brief A handle to the function-wide block frequency pass.
  std::unique_ptr<BranchFolder::MBFIWrapper> MBFI;

  /// \brief A handle to the loop info.
  MachineLoopInfo *MLI;

  /// \brief A handle to the target's instruction info.
  const TargetInstrInfo *TII;

  /// \brief A handle to the target's lowering info.
  const TargetLoweringBase *TLI;

  /// \brief A handle to the post dominator tree.
  MachineDominatorTree *MDT;

  /// \brief A set of blocks that are unavoidably execute, i.e. they dominate
  /// all terminators of the MachineFunction.
  SmallPtrSet<MachineBasicBlock *, 4> UnavoidableBlocks;

  /// \brief Allocator and owner of BlockChain structures.
  ///
  /// We build BlockChains lazily while processing the loop structure of
  /// a function. To reduce malloc traffic, we allocate them using this
  /// slab-like allocator, and destroy them after the pass completes. An
  /// important guarantee is that this allocator produces stable pointers to
  /// the chains.
  SpecificBumpPtrAllocator<BlockChain> ChainAllocator;

  /// \brief Function wide BasicBlock to BlockChain mapping.
  ///
  /// This mapping allows efficiently moving from any given basic block to the
  /// BlockChain it participates in, if any. We use it to, among other things,
  /// allow implicitly defining edges between chains as the existing edges
  /// between basic blocks.
  DenseMap<MachineBasicBlock *, BlockChain *> BlockToChain;

  void markChainSuccessors(BlockChain &Chain, MachineBasicBlock *LoopHeaderBB,
                           const BlockFilterSet *BlockFilter = nullptr);
  BranchProbability
  collectViableSuccessors(MachineBasicBlock *BB, BlockChain &Chain,
                          const BlockFilterSet *BlockFilter,
                          SmallVector<MachineBasicBlock *, 4> &Successors);
  bool shouldPredBlockBeOutlined(MachineBasicBlock *BB, MachineBasicBlock *Succ,
                                 BlockChain &Chain,
                                 const BlockFilterSet *BlockFilter,
                                 BranchProbability SuccProb,
                                 BranchProbability HotProb);
  bool
  hasBetterLayoutPredecessor(MachineBasicBlock *BB, MachineBasicBlock *Succ,
                             BlockChain &SuccChain, BranchProbability SuccProb,
                             BranchProbability RealSuccProb, BlockChain &Chain,
                             const BlockFilterSet *BlockFilter);
  MachineBasicBlock *selectBestSuccessor(MachineBasicBlock *BB,
                                         BlockChain &Chain,
                                         const BlockFilterSet *BlockFilter);
  MachineBasicBlock *
  selectBestCandidateBlock(BlockChain &Chain,
                           SmallVectorImpl<MachineBasicBlock *> &WorkList);
  MachineBasicBlock *
  getFirstUnplacedBlock(const BlockChain &PlacedChain,
                        MachineFunction::iterator &PrevUnplacedBlockIt,
                        const BlockFilterSet *BlockFilter);

  /// \brief Add a basic block to the work list if it is appropriate.
  ///
  /// If the optional parameter BlockFilter is provided, only MBB
  /// present in the set will be added to the worklist. If nullptr
  /// is provided, no filtering occurs.
  void fillWorkLists(MachineBasicBlock *MBB,
                     SmallPtrSetImpl<BlockChain *> &UpdatedPreds,
                     const BlockFilterSet *BlockFilter);
  void buildChain(MachineBasicBlock *BB, BlockChain &Chain,
                  const BlockFilterSet *BlockFilter = nullptr);
  MachineBasicBlock *findBestLoopTop(MachineLoop &L,
                                     const BlockFilterSet &LoopBlockSet);
  MachineBasicBlock *findBestLoopExit(MachineLoop &L,
                                      const BlockFilterSet &LoopBlockSet);
  BlockFilterSet collectLoopBlockSet(MachineLoop &L);
  void buildLoopChains(MachineLoop &L);
  void rotateLoop(BlockChain &LoopChain, MachineBasicBlock *ExitingBB,
                  const BlockFilterSet &LoopBlockSet);
  void rotateLoopWithProfile(BlockChain &LoopChain, MachineLoop &L,
                             const BlockFilterSet &LoopBlockSet);
  void collectMustExecuteBBs();
  void buildCFGChains();
  void optimizeBranches();
  void alignBlocks();

public:
  static char ID; // Pass identification, replacement for typeid
  MachineBlockPlacement() : MachineFunctionPass(ID) {
    initializeMachineBlockPlacementPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineBranchProbabilityInfo>();
    AU.addRequired<MachineBlockFrequencyInfo>();
    AU.addRequired<MachineDominatorTree>();
    AU.addRequired<MachineLoopInfo>();
    AU.addRequired<TargetPassConfig>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
}

char MachineBlockPlacement::ID = 0;
char &llvm::MachineBlockPlacementID = MachineBlockPlacement::ID;
INITIALIZE_PASS_BEGIN(MachineBlockPlacement, "block-placement",
                      "Branch Probability Basic Block Placement", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineBranchProbabilityInfo)
INITIALIZE_PASS_DEPENDENCY(MachineBlockFrequencyInfo)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_END(MachineBlockPlacement, "block-placement",
                    "Branch Probability Basic Block Placement", false, false)

#ifndef NDEBUG
/// \brief Helper to print the name of a MBB.
///
/// Only used by debug logging.
static std::string getBlockName(MachineBasicBlock *BB) {
  std::string Result;
  raw_string_ostream OS(Result);
  OS << "BB#" << BB->getNumber();
  OS << " ('" << BB->getName() << "')";
  OS.flush();
  return Result;
}
#endif

/// \brief Mark a chain's successors as having one fewer preds.
///
/// When a chain is being merged into the "placed" chain, this routine will
/// quickly walk the successors of each block in the chain and mark them as
/// having one fewer active predecessor. It also adds any successors of this
/// chain which reach the zero-predecessor state to the worklist passed in.
void MachineBlockPlacement::markChainSuccessors(
    BlockChain &Chain, MachineBasicBlock *LoopHeaderBB,
    const BlockFilterSet *BlockFilter) {
  // Walk all the blocks in this chain, marking their successors as having
  // a predecessor placed.
  for (MachineBasicBlock *MBB : Chain) {
    // Add any successors for which this is the only un-placed in-loop
    // predecessor to the worklist as a viable candidate for CFG-neutral
    // placement. No subsequent placement of this block will violate the CFG
    // shape, so we get to use heuristics to choose a favorable placement.
    for (MachineBasicBlock *Succ : MBB->successors()) {
      if (BlockFilter && !BlockFilter->count(Succ))
        continue;
      BlockChain &SuccChain = *BlockToChain[Succ];
      // Disregard edges within a fixed chain, or edges to the loop header.
      if (&Chain == &SuccChain || Succ == LoopHeaderBB)
        continue;

      // This is a cross-chain edge that is within the loop, so decrement the
      // loop predecessor count of the destination chain.
      if (SuccChain.UnscheduledPredecessors == 0 ||
          --SuccChain.UnscheduledPredecessors > 0)
        continue;

      auto *MBB = *SuccChain.begin();
      if (MBB->isEHPad())
        EHPadWorkList.push_back(MBB);
      else
        BlockWorkList.push_back(MBB);
    }
  }
}

/// This helper function collects the set of successors of block
/// \p BB that are allowed to be its layout successors, and return
/// the total branch probability of edges from \p BB to those
/// blocks.
BranchProbability MachineBlockPlacement::collectViableSuccessors(
    MachineBasicBlock *BB, BlockChain &Chain, const BlockFilterSet *BlockFilter,
    SmallVector<MachineBasicBlock *, 4> &Successors) {
  // Adjust edge probabilities by excluding edges pointing to blocks that is
  // either not in BlockFilter or is already in the current chain. Consider the
  // following CFG:
  //
  //     --->A
  //     |  / \
  //     | B   C
  //     |  \ / \
  //     ----D   E
  //
  // Assume A->C is very hot (>90%), and C->D has a 50% probability, then after
  // A->C is chosen as a fall-through, D won't be selected as a successor of C
  // due to CFG constraint (the probability of C->D is not greater than
  // HotProb to break top-order). If we exclude E that is not in BlockFilter
  // when calculating the  probability of C->D, D will be selected and we
  // will get A C D B as the layout of this loop.
  auto AdjustedSumProb = BranchProbability::getOne();
  for (MachineBasicBlock *Succ : BB->successors()) {
    bool SkipSucc = false;
    if (Succ->isEHPad() || (BlockFilter && !BlockFilter->count(Succ))) {
      SkipSucc = true;
    } else {
      BlockChain *SuccChain = BlockToChain[Succ];
      if (SuccChain == &Chain) {
        SkipSucc = true;
      } else if (Succ != *SuccChain->begin()) {
        DEBUG(dbgs() << "    " << getBlockName(Succ) << " -> Mid chain!\n");
        continue;
      }
    }
    if (SkipSucc)
      AdjustedSumProb -= MBPI->getEdgeProbability(BB, Succ);
    else
      Successors.push_back(Succ);
  }

  return AdjustedSumProb;
}

/// The helper function returns the branch probability that is adjusted
/// or normalized over the new total \p AdjustedSumProb.
static BranchProbability
getAdjustedProbability(BranchProbability OrigProb,
                       BranchProbability AdjustedSumProb) {
  BranchProbability SuccProb;
  uint32_t SuccProbN = OrigProb.getNumerator();
  uint32_t SuccProbD = AdjustedSumProb.getNumerator();
  if (SuccProbN >= SuccProbD)
    SuccProb = BranchProbability::getOne();
  else
    SuccProb = BranchProbability(SuccProbN, SuccProbD);

  return SuccProb;
}

/// When the option OutlineOptionalBranches is on, this method
/// checks if the fallthrough candidate block \p Succ (of block
/// \p BB) also has other unscheduled predecessor blocks which
/// are also successors of \p BB (forming triangular shape CFG).
/// If none of such predecessors are small, it returns true.
/// The caller can choose to select \p Succ as the layout successors
/// so that \p Succ's predecessors (optional branches) can be
/// outlined.
/// FIXME: fold this with more general layout cost analysis.
bool MachineBlockPlacement::shouldPredBlockBeOutlined(
    MachineBasicBlock *BB, MachineBasicBlock *Succ, BlockChain &Chain,
    const BlockFilterSet *BlockFilter, BranchProbability SuccProb,
    BranchProbability HotProb) {
  if (!OutlineOptionalBranches)
    return false;
  // If we outline optional branches, look whether Succ is unavoidable, i.e.
  // dominates all terminators of the MachineFunction. If it does, other
  // successors must be optional. Don't do this for cold branches.
  if (SuccProb > HotProb.getCompl() && UnavoidableBlocks.count(Succ) > 0) {
    for (MachineBasicBlock *Pred : Succ->predecessors()) {
      // Check whether there is an unplaced optional branch.
      if (Pred == Succ || (BlockFilter && !BlockFilter->count(Pred)) ||
          BlockToChain[Pred] == &Chain)
        continue;
      // Check whether the optional branch has exactly one BB.
      if (Pred->pred_size() > 1 || *Pred->pred_begin() != BB)
        continue;
      // Check whether the optional branch is small.
      if (Pred->size() < OutlineOptionalThreshold)
        return false;
    }
    return true;
  } else
    return false;
}

// When profile is not present, return the StaticLikelyProb.
// When profile is available, we need to handle the triangle-shape CFG.
static BranchProbability getLayoutSuccessorProbThreshold(
      MachineBasicBlock *BB) {
  if (!BB->getParent()->getFunction()->getEntryCount())
    return BranchProbability(StaticLikelyProb, 100);
  if (BB->succ_size() == 2) {
    const MachineBasicBlock *Succ1 = *BB->succ_begin();
    const MachineBasicBlock *Succ2 = *(BB->succ_begin() + 1);
    if (Succ1->isSuccessor(Succ2) || Succ2->isSuccessor(Succ1)) {
      /* See case 1 below for the cost analysis. For BB->Succ to
       * be taken with smaller cost, the following needs to hold:
       *   Prob(BB->Succ) > 2* Prob(BB->Pred)
       *   So the threshold T
       *   T = 2 * (1-Prob(BB->Pred). Since T + Prob(BB->Pred) == 1,
       * We have  T + T/2 = 1, i.e. T = 2/3. Also adding user specified
       * branch bias, we have
       *   T = (2/3)*(ProfileLikelyProb/50)
       *     = (2*ProfileLikelyProb)/150)
       */
      return BranchProbability(2 * ProfileLikelyProb, 150);
    }
  }
  return BranchProbability(ProfileLikelyProb, 100);
}

/// Checks to see if the layout candidate block \p Succ has a better layout
/// predecessor than \c BB. If yes, returns true.
bool MachineBlockPlacement::hasBetterLayoutPredecessor(
    MachineBasicBlock *BB, MachineBasicBlock *Succ, BlockChain &SuccChain,
    BranchProbability SuccProb, BranchProbability RealSuccProb,
    BlockChain &Chain, const BlockFilterSet *BlockFilter) {

  // There isn't a better layout when there are no unscheduled predecessors.
  if (SuccChain.UnscheduledPredecessors == 0)
    return false;

  // There are two basic scenarios here:
  // -------------------------------------
  // Case 1: triangular shape CFG (if-then):
  //     BB
  //     | \
  //     |  \
  //     |   Pred
  //     |   /
  //     Succ
  // In this case, we are evaluating whether to select edge -> Succ, e.g.
  // set Succ as the layout successor of BB. Picking Succ as BB's
  // successor breaks the CFG constraints (FIXME: define these constraints).
  // With this layout, Pred BB
  // is forced to be outlined, so the overall cost will be cost of the
  // branch taken from BB to Pred, plus the cost of back taken branch
  // from Pred to Succ, as well as the additional cost associated
  // with the needed unconditional jump instruction from Pred To Succ.

  // The cost of the topological order layout is the taken branch cost
  // from BB to Succ, so to make BB->Succ a viable candidate, the following
  // must hold:
  //     2 * freq(BB->Pred) * taken_branch_cost + unconditional_jump_cost
  //      < freq(BB->Succ) *  taken_branch_cost.
  // Ignoring unconditional jump cost, we get
  //    freq(BB->Succ) > 2 * freq(BB->Pred), i.e.,
  //    prob(BB->Succ) > 2 * prob(BB->Pred)
  //
  // When real profile data is available, we can precisely compute the
  // probability threshold that is needed for edge BB->Succ to be considered.
  // Without profile data, the heuristic requires the branch bias to be
  // a lot larger to make sure the signal is very strong (e.g. 80% default).
  // -----------------------------------------------------------------
  // Case 2: diamond like CFG (if-then-else):
  //     S
  //    / \
  //   |   \
  //  BB    Pred
  //   \    /
  //    Succ
  //    ..
  //
  // The current block is BB and edge BB->Succ is now being evaluated.
  // Note that edge S->BB was previously already selected because
  // prob(S->BB) > prob(S->Pred).
  // At this point, 2 blocks can be placed after BB: Pred or Succ. If we
  // choose Pred, we will have a topological ordering as shown on the left
  // in the picture below. If we choose Succ, we have the solution as shown
  // on the right:
  //
  //   topo-order:
  //
  //       S-----                             ---S
  //       |    |                             |  |
  //    ---BB   |                             |  BB
  //    |       |                             |  |
  //    |  pred--                             |  Succ--
  //    |  |                                  |       |
  //    ---succ                               ---pred--
  //
  // cost = freq(S->Pred) + freq(BB->Succ)    cost = 2 * freq (S->Pred)
  //      = freq(S->Pred) + freq(S->BB)
  //
  // If we have profile data (i.e, branch probabilities can be trusted), the
  // cost (number of taken branches) with layout S->BB->Succ->Pred is 2 *
  // freq(S->Pred) while the cost of topo order is freq(S->Pred) + freq(S->BB).
  // We know Prob(S->BB) > Prob(S->Pred), so freq(S->BB) > freq(S->Pred), which
  // means the cost of topological order is greater.
  // When profile data is not available, however, we need to be more
  // conservative. If the branch prediction is wrong, breaking the topo-order
  // will actually yield a layout with large cost. For this reason, we need
  // strong biased branch at block S with Prob(S->BB) in order to select
  // BB->Succ. This is equivalent to looking the CFG backward with backward
  // edge: Prob(Succ->BB) needs to >= HotProb in order to be selected (without
  // profile data).
  // --------------------------------------------------------------------------
  // Case 3: forked diamond
  //       S
  //      / \
  //     /   \
  //   BB    Pred
  //   | \   / |
  //   |  \ /  |
  //   |   X   |
  //   |  / \  |
  //   | /   \ |
  //   S1     S2
  //
  // The current block is BB and edge BB->S1 is now being evaluated.
  // As above S->BB was already selected because
  // prob(S->BB) > prob(S->Pred). Assume that prob(BB->S1) >= prob(BB->S2).
  //
  // topo-order:
  //
  //     S-------|                     ---S
  //     |       |                     |  |
  //  ---BB      |                     |  BB
  //  |          |                     |  |
  //  |  Pred----|                     |  S1----
  //  |  |                             |       |
  //  --(S1 or S2)                     ---Pred--
  //
  // topo-cost = freq(S->Pred) + freq(BB->S1) + freq(BB->S2)
  //    + min(freq(Pred->S1), freq(Pred->S2))
  // Non-topo-order cost:
  // In the worst case, S2 will not get laid out after Pred.
  // non-topo-cost = 2 * freq(S->Pred) + freq(BB->S2).
  // To be conservative, we can assume that min(freq(Pred->S1), freq(Pred->S2))
  // is 0. Then the non topo layout is better when
  // freq(S->Pred) < freq(BB->S1).
  // This is exactly what is checked below.
  // Note there are other shapes that apply (Pred may not be a single block,
  // but they all fit this general pattern.)
  BranchProbability HotProb = getLayoutSuccessorProbThreshold(BB);

  // Make sure that a hot successor doesn't have a globally more
  // important predecessor.
  BlockFrequency CandidateEdgeFreq = MBFI->getBlockFreq(BB) * RealSuccProb;
  bool BadCFGConflict = false;

  for (MachineBasicBlock *Pred : Succ->predecessors()) {
    if (Pred == Succ || BlockToChain[Pred] == &SuccChain ||
        (BlockFilter && !BlockFilter->count(Pred)) ||
        BlockToChain[Pred] == &Chain)
      continue;
    // Do backward checking.
    // For all cases above, we need a backward checking to filter out edges that
    // are not 'strongly' biased. With profile data available, the check is
    // mostly redundant for case 2 (when threshold prob is set at 50%) unless S
    // has more than two successors.
    // BB  Pred
    //  \ /
    //  Succ
    // We select edge BB->Succ if
    //      freq(BB->Succ) > freq(Succ) * HotProb
    //      i.e. freq(BB->Succ) > freq(BB->Succ) * HotProb + freq(Pred->Succ) *
    //      HotProb
    //      i.e. freq((BB->Succ) * (1 - HotProb) > freq(Pred->Succ) * HotProb
    // Case 1 is covered too, because the first equation reduces to:
    // prob(BB->Succ) > HotProb. (freq(Succ) = freq(BB) for a triangle)
    BlockFrequency PredEdgeFreq =
        MBFI->getBlockFreq(Pred) * MBPI->getEdgeProbability(Pred, Succ);
    if (PredEdgeFreq * HotProb >= CandidateEdgeFreq * HotProb.getCompl()) {
      BadCFGConflict = true;
      break;
    }
  }

  if (BadCFGConflict) {
    DEBUG(dbgs() << "    Not a candidate: " << getBlockName(Succ) << " -> " << SuccProb
                 << " (prob) (non-cold CFG conflict)\n");
    return true;
  }

  return false;
}

/// \brief Select the best successor for a block.
///
/// This looks across all successors of a particular block and attempts to
/// select the "best" one to be the layout successor. It only considers direct
/// successors which also pass the block filter. It will attempt to avoid
/// breaking CFG structure, but cave and break such structures in the case of
/// very hot successor edges.
///
/// \returns The best successor block found, or null if none are viable.
MachineBasicBlock *
MachineBlockPlacement::selectBestSuccessor(MachineBasicBlock *BB,
                                           BlockChain &Chain,
                                           const BlockFilterSet *BlockFilter) {
  const BranchProbability HotProb(StaticLikelyProb, 100);

  MachineBasicBlock *BestSucc = nullptr;
  auto BestProb = BranchProbability::getZero();

  SmallVector<MachineBasicBlock *, 4> Successors;
  auto AdjustedSumProb =
      collectViableSuccessors(BB, Chain, BlockFilter, Successors);

  DEBUG(dbgs() << "Selecting best successor for: " << getBlockName(BB) << "\n");
  for (MachineBasicBlock *Succ : Successors) {
    auto RealSuccProb = MBPI->getEdgeProbability(BB, Succ);
    BranchProbability SuccProb =
        getAdjustedProbability(RealSuccProb, AdjustedSumProb);

    // This heuristic is off by default.
    if (shouldPredBlockBeOutlined(BB, Succ, Chain, BlockFilter, SuccProb,
                                  HotProb))
      return Succ;

    BlockChain &SuccChain = *BlockToChain[Succ];
    // Skip the edge \c BB->Succ if block \c Succ has a better layout
    // predecessor that yields lower global cost.
    if (hasBetterLayoutPredecessor(BB, Succ, SuccChain, SuccProb, RealSuccProb,
                                   Chain, BlockFilter))
      continue;

    DEBUG(
        dbgs() << "    Candidate: " << getBlockName(Succ) << ", probability: "
               << SuccProb
               << (SuccChain.UnscheduledPredecessors != 0 ? " (CFG break)" : "")
               << "\n");

    if (BestSucc && BestProb >= SuccProb) {
      DEBUG(dbgs() << "    Not the best candidate, continuing\n");
      continue;
    }

    DEBUG(dbgs() << "    Setting it as best candidate\n");
    BestSucc = Succ;
    BestProb = SuccProb;
  }
  if (BestSucc)
    DEBUG(dbgs() << "    Selected: " << getBlockName(BestSucc) << "\n");

  return BestSucc;
}

/// \brief Select the best block from a worklist.
///
/// This looks through the provided worklist as a list of candidate basic
/// blocks and select the most profitable one to place. The definition of
/// profitable only really makes sense in the context of a loop. This returns
/// the most frequently visited block in the worklist, which in the case of
/// a loop, is the one most desirable to be physically close to the rest of the
/// loop body in order to improve i-cache behavior.
///
/// \returns The best block found, or null if none are viable.
MachineBasicBlock *MachineBlockPlacement::selectBestCandidateBlock(
    BlockChain &Chain, SmallVectorImpl<MachineBasicBlock *> &WorkList) {
  // Once we need to walk the worklist looking for a candidate, cleanup the
  // worklist of already placed entries.
  // FIXME: If this shows up on profiles, it could be folded (at the cost of
  // some code complexity) into the loop below.
  WorkList.erase(remove_if(WorkList,
                           [&](MachineBasicBlock *BB) {
                             return BlockToChain.lookup(BB) == &Chain;
                           }),
                 WorkList.end());

  if (WorkList.empty())
    return nullptr;

  bool IsEHPad = WorkList[0]->isEHPad();

  MachineBasicBlock *BestBlock = nullptr;
  BlockFrequency BestFreq;
  for (MachineBasicBlock *MBB : WorkList) {
    assert(MBB->isEHPad() == IsEHPad);

    BlockChain &SuccChain = *BlockToChain[MBB];
    if (&SuccChain == &Chain)
      continue;

    assert(SuccChain.UnscheduledPredecessors == 0 && "Found CFG-violating block");

    BlockFrequency CandidateFreq = MBFI->getBlockFreq(MBB);
    DEBUG(dbgs() << "    " << getBlockName(MBB) << " -> ";
          MBFI->printBlockFreq(dbgs(), CandidateFreq) << " (freq)\n");

    // For ehpad, we layout the least probable first as to avoid jumping back
    // from least probable landingpads to more probable ones.
    //
    // FIXME: Using probability is probably (!) not the best way to achieve
    // this. We should probably have a more principled approach to layout
    // cleanup code.
    //
    // The goal is to get:
    //
    //                 +--------------------------+
    //                 |                          V
    // InnerLp -> InnerCleanup    OuterLp -> OuterCleanup -> Resume
    //
    // Rather than:
    //
    //                 +-------------------------------------+
    //                 V                                     |
    // OuterLp -> OuterCleanup -> Resume     InnerLp -> InnerCleanup
    if (BestBlock && (IsEHPad ^ (BestFreq >= CandidateFreq)))
      continue;

    BestBlock = MBB;
    BestFreq = CandidateFreq;
  }

  return BestBlock;
}

/// \brief Retrieve the first unplaced basic block.
///
/// This routine is called when we are unable to use the CFG to walk through
/// all of the basic blocks and form a chain due to unnatural loops in the CFG.
/// We walk through the function's blocks in order, starting from the
/// LastUnplacedBlockIt. We update this iterator on each call to avoid
/// re-scanning the entire sequence on repeated calls to this routine.
MachineBasicBlock *MachineBlockPlacement::getFirstUnplacedBlock(
    const BlockChain &PlacedChain,
    MachineFunction::iterator &PrevUnplacedBlockIt,
    const BlockFilterSet *BlockFilter) {
  for (MachineFunction::iterator I = PrevUnplacedBlockIt, E = F->end(); I != E;
       ++I) {
    if (BlockFilter && !BlockFilter->count(&*I))
      continue;
    if (BlockToChain[&*I] != &PlacedChain) {
      PrevUnplacedBlockIt = I;
      // Now select the head of the chain to which the unplaced block belongs
      // as the block to place. This will force the entire chain to be placed,
      // and satisfies the requirements of merging chains.
      return *BlockToChain[&*I]->begin();
    }
  }
  return nullptr;
}

void MachineBlockPlacement::fillWorkLists(
    MachineBasicBlock *MBB,
    SmallPtrSetImpl<BlockChain *> &UpdatedPreds,
    const BlockFilterSet *BlockFilter = nullptr) {
  BlockChain &Chain = *BlockToChain[MBB];
  if (!UpdatedPreds.insert(&Chain).second)
    return;

  assert(Chain.UnscheduledPredecessors == 0);
  for (MachineBasicBlock *ChainBB : Chain) {
    assert(BlockToChain[ChainBB] == &Chain);
    for (MachineBasicBlock *Pred : ChainBB->predecessors()) {
      if (BlockFilter && !BlockFilter->count(Pred))
        continue;
      if (BlockToChain[Pred] == &Chain)
        continue;
      ++Chain.UnscheduledPredecessors;
    }
  }

  if (Chain.UnscheduledPredecessors != 0)
    return;

  MBB = *Chain.begin();
  if (MBB->isEHPad())
    EHPadWorkList.push_back(MBB);
  else
    BlockWorkList.push_back(MBB);
}

void MachineBlockPlacement::buildChain(
    MachineBasicBlock *BB, BlockChain &Chain,
    const BlockFilterSet *BlockFilter) {
  assert(BB && "BB must not be null.\n");
  assert(BlockToChain[BB] == &Chain && "BlockToChainMap mis-match.\n");
  MachineFunction::iterator PrevUnplacedBlockIt = F->begin();

  MachineBasicBlock *LoopHeaderBB = BB;
  markChainSuccessors(Chain, LoopHeaderBB, BlockFilter);
  BB = *std::prev(Chain.end());
  for (;;) {
    assert(BB && "null block found at end of chain in loop.");
    assert(BlockToChain[BB] == &Chain && "BlockToChainMap mis-match in loop.");
    assert(*std::prev(Chain.end()) == BB && "BB Not found at end of chain.");


    // Look for the best viable successor if there is one to place immediately
    // after this block.
    MachineBasicBlock *BestSucc = selectBestSuccessor(BB, Chain, BlockFilter);

    // If an immediate successor isn't available, look for the best viable
    // block among those we've identified as not violating the loop's CFG at
    // this point. This won't be a fallthrough, but it will increase locality.
    if (!BestSucc)
      BestSucc = selectBestCandidateBlock(Chain, BlockWorkList);
    if (!BestSucc)
      BestSucc = selectBestCandidateBlock(Chain, EHPadWorkList);

    if (!BestSucc) {
      BestSucc = getFirstUnplacedBlock(Chain, PrevUnplacedBlockIt, BlockFilter);
      if (!BestSucc)
        break;

      DEBUG(dbgs() << "Unnatural loop CFG detected, forcibly merging the "
                      "layout successor until the CFG reduces\n");
    }

    // Place this block, updating the datastructures to reflect its placement.
    BlockChain &SuccChain = *BlockToChain[BestSucc];
    // Zero out UnscheduledPredecessors for the successor we're about to merge in case
    // we selected a successor that didn't fit naturally into the CFG.
    SuccChain.UnscheduledPredecessors = 0;
    DEBUG(dbgs() << "Merging from " << getBlockName(BB) << " to "
                 << getBlockName(BestSucc) << "\n");
    markChainSuccessors(SuccChain, LoopHeaderBB, BlockFilter);
    Chain.merge(BestSucc, &SuccChain);
    BB = *std::prev(Chain.end());
  }

  DEBUG(dbgs() << "Finished forming chain for header block "
               << getBlockName(*Chain.begin()) << "\n");
}

/// \brief Find the best loop top block for layout.
///
/// Look for a block which is strictly better than the loop header for laying
/// out at the top of the loop. This looks for one and only one pattern:
/// a latch block with no conditional exit. This block will cause a conditional
/// jump around it or will be the bottom of the loop if we lay it out in place,
/// but if it it doesn't end up at the bottom of the loop for any reason,
/// rotation alone won't fix it. Because such a block will always result in an
/// unconditional jump (for the backedge) rotating it in front of the loop
/// header is always profitable.
MachineBasicBlock *
MachineBlockPlacement::findBestLoopTop(MachineLoop &L,
                                       const BlockFilterSet &LoopBlockSet) {
  // Placing the latch block before the header may introduce an extra branch
  // that skips this block the first time the loop is executed, which we want
  // to avoid when optimising for size.
  // FIXME: in theory there is a case that does not introduce a new branch,
  // i.e. when the layout predecessor does not fallthrough to the loop header.
  // In practice this never happens though: there always seems to be a preheader
  // that can fallthrough and that is also placed before the header.
  if (F->getFunction()->optForSize())
    return L.getHeader();

  // Check that the header hasn't been fused with a preheader block due to
  // crazy branches. If it has, we need to start with the header at the top to
  // prevent pulling the preheader into the loop body.
  BlockChain &HeaderChain = *BlockToChain[L.getHeader()];
  if (!LoopBlockSet.count(*HeaderChain.begin()))
    return L.getHeader();

  DEBUG(dbgs() << "Finding best loop top for: " << getBlockName(L.getHeader())
               << "\n");

  BlockFrequency BestPredFreq;
  MachineBasicBlock *BestPred = nullptr;
  for (MachineBasicBlock *Pred : L.getHeader()->predecessors()) {
    if (!LoopBlockSet.count(Pred))
      continue;
    DEBUG(dbgs() << "    header pred: " << getBlockName(Pred) << ", has "
                 << Pred->succ_size() << " successors, ";
          MBFI->printBlockFreq(dbgs(), Pred) << " freq\n");
    if (Pred->succ_size() > 1)
      continue;

    BlockFrequency PredFreq = MBFI->getBlockFreq(Pred);
    if (!BestPred || PredFreq > BestPredFreq ||
        (!(PredFreq < BestPredFreq) &&
         Pred->isLayoutSuccessor(L.getHeader()))) {
      BestPred = Pred;
      BestPredFreq = PredFreq;
    }
  }

  // If no direct predecessor is fine, just use the loop header.
  if (!BestPred) {
    DEBUG(dbgs() << "    final top unchanged\n");
    return L.getHeader();
  }

  // Walk backwards through any straight line of predecessors.
  while (BestPred->pred_size() == 1 &&
         (*BestPred->pred_begin())->succ_size() == 1 &&
         *BestPred->pred_begin() != L.getHeader())
    BestPred = *BestPred->pred_begin();

  DEBUG(dbgs() << "    final top: " << getBlockName(BestPred) << "\n");
  return BestPred;
}

/// \brief Find the best loop exiting block for layout.
///
/// This routine implements the logic to analyze the loop looking for the best
/// block to layout at the top of the loop. Typically this is done to maximize
/// fallthrough opportunities.
MachineBasicBlock *
MachineBlockPlacement::findBestLoopExit(MachineLoop &L,
                                        const BlockFilterSet &LoopBlockSet) {
  // We don't want to layout the loop linearly in all cases. If the loop header
  // is just a normal basic block in the loop, we want to look for what block
  // within the loop is the best one to layout at the top. However, if the loop
  // header has be pre-merged into a chain due to predecessors not having
  // analyzable branches, *and* the predecessor it is merged with is *not* part
  // of the loop, rotating the header into the middle of the loop will create
  // a non-contiguous range of blocks which is Very Bad. So start with the
  // header and only rotate if safe.
  BlockChain &HeaderChain = *BlockToChain[L.getHeader()];
  if (!LoopBlockSet.count(*HeaderChain.begin()))
    return nullptr;

  BlockFrequency BestExitEdgeFreq;
  unsigned BestExitLoopDepth = 0;
  MachineBasicBlock *ExitingBB = nullptr;
  // If there are exits to outer loops, loop rotation can severely limit
  // fallthrough opportunities unless it selects such an exit. Keep a set of
  // blocks where rotating to exit with that block will reach an outer loop.
  SmallPtrSet<MachineBasicBlock *, 4> BlocksExitingToOuterLoop;

  DEBUG(dbgs() << "Finding best loop exit for: " << getBlockName(L.getHeader())
               << "\n");
  for (MachineBasicBlock *MBB : L.getBlocks()) {
    BlockChain &Chain = *BlockToChain[MBB];
    // Ensure that this block is at the end of a chain; otherwise it could be
    // mid-way through an inner loop or a successor of an unanalyzable branch.
    if (MBB != *std::prev(Chain.end()))
      continue;

    // Now walk the successors. We need to establish whether this has a viable
    // exiting successor and whether it has a viable non-exiting successor.
    // We store the old exiting state and restore it if a viable looping
    // successor isn't found.
    MachineBasicBlock *OldExitingBB = ExitingBB;
    BlockFrequency OldBestExitEdgeFreq = BestExitEdgeFreq;
    bool HasLoopingSucc = false;
    for (MachineBasicBlock *Succ : MBB->successors()) {
      if (Succ->isEHPad())
        continue;
      if (Succ == MBB)
        continue;
      BlockChain &SuccChain = *BlockToChain[Succ];
      // Don't split chains, either this chain or the successor's chain.
      if (&Chain == &SuccChain) {
        DEBUG(dbgs() << "    exiting: " << getBlockName(MBB) << " -> "
                     << getBlockName(Succ) << " (chain conflict)\n");
        continue;
      }

      auto SuccProb = MBPI->getEdgeProbability(MBB, Succ);
      if (LoopBlockSet.count(Succ)) {
        DEBUG(dbgs() << "    looping: " << getBlockName(MBB) << " -> "
                     << getBlockName(Succ) << " (" << SuccProb << ")\n");
        HasLoopingSucc = true;
        continue;
      }

      unsigned SuccLoopDepth = 0;
      if (MachineLoop *ExitLoop = MLI->getLoopFor(Succ)) {
        SuccLoopDepth = ExitLoop->getLoopDepth();
        if (ExitLoop->contains(&L))
          BlocksExitingToOuterLoop.insert(MBB);
      }

      BlockFrequency ExitEdgeFreq = MBFI->getBlockFreq(MBB) * SuccProb;
      DEBUG(dbgs() << "    exiting: " << getBlockName(MBB) << " -> "
                   << getBlockName(Succ) << " [L:" << SuccLoopDepth << "] (";
            MBFI->printBlockFreq(dbgs(), ExitEdgeFreq) << ")\n");
      // Note that we bias this toward an existing layout successor to retain
      // incoming order in the absence of better information. The exit must have
      // a frequency higher than the current exit before we consider breaking
      // the layout.
      BranchProbability Bias(100 - ExitBlockBias, 100);
      if (!ExitingBB || SuccLoopDepth > BestExitLoopDepth ||
          ExitEdgeFreq > BestExitEdgeFreq ||
          (MBB->isLayoutSuccessor(Succ) &&
           !(ExitEdgeFreq < BestExitEdgeFreq * Bias))) {
        BestExitEdgeFreq = ExitEdgeFreq;
        ExitingBB = MBB;
      }
    }

    if (!HasLoopingSucc) {
      // Restore the old exiting state, no viable looping successor was found.
      ExitingBB = OldExitingBB;
      BestExitEdgeFreq = OldBestExitEdgeFreq;
    }
  }
  // Without a candidate exiting block or with only a single block in the
  // loop, just use the loop header to layout the loop.
  if (!ExitingBB) {
    DEBUG(dbgs() << "    No other candidate exit blocks, using loop header\n");
    return nullptr;
  }
  if (L.getNumBlocks() == 1) {
    DEBUG(dbgs() << "    Loop has 1 block, using loop header as exit\n");
    return nullptr;
  }

  // Also, if we have exit blocks which lead to outer loops but didn't select
  // one of them as the exiting block we are rotating toward, disable loop
  // rotation altogether.
  if (!BlocksExitingToOuterLoop.empty() &&
      !BlocksExitingToOuterLoop.count(ExitingBB))
    return nullptr;

  DEBUG(dbgs() << "  Best exiting block: " << getBlockName(ExitingBB) << "\n");
  return ExitingBB;
}

/// \brief Attempt to rotate an exiting block to the bottom of the loop.
///
/// Once we have built a chain, try to rotate it to line up the hot exit block
/// with fallthrough out of the loop if doing so doesn't introduce unnecessary
/// branches. For example, if the loop has fallthrough into its header and out
/// of its bottom already, don't rotate it.
void MachineBlockPlacement::rotateLoop(BlockChain &LoopChain,
                                       MachineBasicBlock *ExitingBB,
                                       const BlockFilterSet &LoopBlockSet) {
  if (!ExitingBB)
    return;

  MachineBasicBlock *Top = *LoopChain.begin();
  bool ViableTopFallthrough = false;
  for (MachineBasicBlock *Pred : Top->predecessors()) {
    BlockChain *PredChain = BlockToChain[Pred];
    if (!LoopBlockSet.count(Pred) &&
        (!PredChain || Pred == *std::prev(PredChain->end()))) {
      ViableTopFallthrough = true;
      break;
    }
  }

  // If the header has viable fallthrough, check whether the current loop
  // bottom is a viable exiting block. If so, bail out as rotating will
  // introduce an unnecessary branch.
  if (ViableTopFallthrough) {
    MachineBasicBlock *Bottom = *std::prev(LoopChain.end());
    for (MachineBasicBlock *Succ : Bottom->successors()) {
      BlockChain *SuccChain = BlockToChain[Succ];
      if (!LoopBlockSet.count(Succ) &&
          (!SuccChain || Succ == *SuccChain->begin()))
        return;
    }
  }

  BlockChain::iterator ExitIt = find(LoopChain, ExitingBB);
  if (ExitIt == LoopChain.end())
    return;

  std::rotate(LoopChain.begin(), std::next(ExitIt), LoopChain.end());
}

/// \brief Attempt to rotate a loop based on profile data to reduce branch cost.
///
/// With profile data, we can determine the cost in terms of missed fall through
/// opportunities when rotating a loop chain and select the best rotation.
/// Basically, there are three kinds of cost to consider for each rotation:
///    1. The possibly missed fall through edge (if it exists) from BB out of
///    the loop to the loop header.
///    2. The possibly missed fall through edges (if they exist) from the loop
///    exits to BB out of the loop.
///    3. The missed fall through edge (if it exists) from the last BB to the
///    first BB in the loop chain.
///  Therefore, the cost for a given rotation is the sum of costs listed above.
///  We select the best rotation with the smallest cost.
void MachineBlockPlacement::rotateLoopWithProfile(
    BlockChain &LoopChain, MachineLoop &L, const BlockFilterSet &LoopBlockSet) {
  auto HeaderBB = L.getHeader();
  auto HeaderIter = find(LoopChain, HeaderBB);
  auto RotationPos = LoopChain.end();

  BlockFrequency SmallestRotationCost = BlockFrequency::getMaxFrequency();

  // A utility lambda that scales up a block frequency by dividing it by a
  // branch probability which is the reciprocal of the scale.
  auto ScaleBlockFrequency = [](BlockFrequency Freq,
                                unsigned Scale) -> BlockFrequency {
    if (Scale == 0)
      return 0;
    // Use operator / between BlockFrequency and BranchProbability to implement
    // saturating multiplication.
    return Freq / BranchProbability(1, Scale);
  };

  // Compute the cost of the missed fall-through edge to the loop header if the
  // chain head is not the loop header. As we only consider natural loops with
  // single header, this computation can be done only once.
  BlockFrequency HeaderFallThroughCost(0);
  for (auto *Pred : HeaderBB->predecessors()) {
    BlockChain *PredChain = BlockToChain[Pred];
    if (!LoopBlockSet.count(Pred) &&
        (!PredChain || Pred == *std::prev(PredChain->end()))) {
      auto EdgeFreq =
          MBFI->getBlockFreq(Pred) * MBPI->getEdgeProbability(Pred, HeaderBB);
      auto FallThruCost = ScaleBlockFrequency(EdgeFreq, MisfetchCost);
      // If the predecessor has only an unconditional jump to the header, we
      // need to consider the cost of this jump.
      if (Pred->succ_size() == 1)
        FallThruCost += ScaleBlockFrequency(EdgeFreq, JumpInstCost);
      HeaderFallThroughCost = std::max(HeaderFallThroughCost, FallThruCost);
    }
  }

  // Here we collect all exit blocks in the loop, and for each exit we find out
  // its hottest exit edge. For each loop rotation, we define the loop exit cost
  // as the sum of frequencies of exit edges we collect here, excluding the exit
  // edge from the tail of the loop chain.
  SmallVector<std::pair<MachineBasicBlock *, BlockFrequency>, 4> ExitsWithFreq;
  for (auto BB : LoopChain) {
    auto LargestExitEdgeProb = BranchProbability::getZero();
    for (auto *Succ : BB->successors()) {
      BlockChain *SuccChain = BlockToChain[Succ];
      if (!LoopBlockSet.count(Succ) &&
          (!SuccChain || Succ == *SuccChain->begin())) {
        auto SuccProb = MBPI->getEdgeProbability(BB, Succ);
        LargestExitEdgeProb = std::max(LargestExitEdgeProb, SuccProb);
      }
    }
    if (LargestExitEdgeProb > BranchProbability::getZero()) {
      auto ExitFreq = MBFI->getBlockFreq(BB) * LargestExitEdgeProb;
      ExitsWithFreq.emplace_back(BB, ExitFreq);
    }
  }

  // In this loop we iterate every block in the loop chain and calculate the
  // cost assuming the block is the head of the loop chain. When the loop ends,
  // we should have found the best candidate as the loop chain's head.
  for (auto Iter = LoopChain.begin(), TailIter = std::prev(LoopChain.end()),
            EndIter = LoopChain.end();
       Iter != EndIter; Iter++, TailIter++) {
    // TailIter is used to track the tail of the loop chain if the block we are
    // checking (pointed by Iter) is the head of the chain.
    if (TailIter == LoopChain.end())
      TailIter = LoopChain.begin();

    auto TailBB = *TailIter;

    // Calculate the cost by putting this BB to the top.
    BlockFrequency Cost = 0;

    // If the current BB is the loop header, we need to take into account the
    // cost of the missed fall through edge from outside of the loop to the
    // header.
    if (Iter != HeaderIter)
      Cost += HeaderFallThroughCost;

    // Collect the loop exit cost by summing up frequencies of all exit edges
    // except the one from the chain tail.
    for (auto &ExitWithFreq : ExitsWithFreq)
      if (TailBB != ExitWithFreq.first)
        Cost += ExitWithFreq.second;

    // The cost of breaking the once fall-through edge from the tail to the top
    // of the loop chain. Here we need to consider three cases:
    // 1. If the tail node has only one successor, then we will get an
    //    additional jmp instruction. So the cost here is (MisfetchCost +
    //    JumpInstCost) * tail node frequency.
    // 2. If the tail node has two successors, then we may still get an
    //    additional jmp instruction if the layout successor after the loop
    //    chain is not its CFG successor. Note that the more frequently executed
    //    jmp instruction will be put ahead of the other one. Assume the
    //    frequency of those two branches are x and y, where x is the frequency
    //    of the edge to the chain head, then the cost will be
    //    (x * MisfetechCost + min(x, y) * JumpInstCost) * tail node frequency.
    // 3. If the tail node has more than two successors (this rarely happens),
    //    we won't consider any additional cost.
    if (TailBB->isSuccessor(*Iter)) {
      auto TailBBFreq = MBFI->getBlockFreq(TailBB);
      if (TailBB->succ_size() == 1)
        Cost += ScaleBlockFrequency(TailBBFreq.getFrequency(),
                                    MisfetchCost + JumpInstCost);
      else if (TailBB->succ_size() == 2) {
        auto TailToHeadProb = MBPI->getEdgeProbability(TailBB, *Iter);
        auto TailToHeadFreq = TailBBFreq * TailToHeadProb;
        auto ColderEdgeFreq = TailToHeadProb > BranchProbability(1, 2)
                                  ? TailBBFreq * TailToHeadProb.getCompl()
                                  : TailToHeadFreq;
        Cost += ScaleBlockFrequency(TailToHeadFreq, MisfetchCost) +
                ScaleBlockFrequency(ColderEdgeFreq, JumpInstCost);
      }
    }

    DEBUG(dbgs() << "The cost of loop rotation by making " << getBlockName(*Iter)
                 << " to the top: " << Cost.getFrequency() << "\n");

    if (Cost < SmallestRotationCost) {
      SmallestRotationCost = Cost;
      RotationPos = Iter;
    }
  }

  if (RotationPos != LoopChain.end()) {
    DEBUG(dbgs() << "Rotate loop by making " << getBlockName(*RotationPos)
                 << " to the top\n");
    std::rotate(LoopChain.begin(), RotationPos, LoopChain.end());
  }
}

/// \brief Collect blocks in the given loop that are to be placed.
///
/// When profile data is available, exclude cold blocks from the returned set;
/// otherwise, collect all blocks in the loop.
MachineBlockPlacement::BlockFilterSet
MachineBlockPlacement::collectLoopBlockSet(MachineLoop &L) {
  BlockFilterSet LoopBlockSet;

  // Filter cold blocks off from LoopBlockSet when profile data is available.
  // Collect the sum of frequencies of incoming edges to the loop header from
  // outside. If we treat the loop as a super block, this is the frequency of
  // the loop. Then for each block in the loop, we calculate the ratio between
  // its frequency and the frequency of the loop block. When it is too small,
  // don't add it to the loop chain. If there are outer loops, then this block
  // will be merged into the first outer loop chain for which this block is not
  // cold anymore. This needs precise profile data and we only do this when
  // profile data is available.
  if (F->getFunction()->getEntryCount()) {
    BlockFrequency LoopFreq(0);
    for (auto LoopPred : L.getHeader()->predecessors())
      if (!L.contains(LoopPred))
        LoopFreq += MBFI->getBlockFreq(LoopPred) *
                    MBPI->getEdgeProbability(LoopPred, L.getHeader());

    for (MachineBasicBlock *LoopBB : L.getBlocks()) {
      auto Freq = MBFI->getBlockFreq(LoopBB).getFrequency();
      if (Freq == 0 || LoopFreq.getFrequency() / Freq > LoopToColdBlockRatio)
        continue;
      LoopBlockSet.insert(LoopBB);
    }
  } else
    LoopBlockSet.insert(L.block_begin(), L.block_end());

  return LoopBlockSet;
}

/// \brief Forms basic block chains from the natural loop structures.
///
/// These chains are designed to preserve the existing *structure* of the code
/// as much as possible. We can then stitch the chains together in a way which
/// both preserves the topological structure and minimizes taken conditional
/// branches.
void MachineBlockPlacement::buildLoopChains(MachineLoop &L) {
  // First recurse through any nested loops, building chains for those inner
  // loops.
  for (MachineLoop *InnerLoop : L)
    buildLoopChains(*InnerLoop);

  assert(BlockWorkList.empty());
  assert(EHPadWorkList.empty());
  BlockFilterSet LoopBlockSet = collectLoopBlockSet(L);

  // Check if we have profile data for this function. If yes, we will rotate
  // this loop by modeling costs more precisely which requires the profile data
  // for better layout.
  bool RotateLoopWithProfile =
      ForcePreciseRotationCost ||
      (PreciseRotationCost && F->getFunction()->getEntryCount());

  // First check to see if there is an obviously preferable top block for the
  // loop. This will default to the header, but may end up as one of the
  // predecessors to the header if there is one which will result in strictly
  // fewer branches in the loop body.
  // When we use profile data to rotate the loop, this is unnecessary.
  MachineBasicBlock *LoopTop =
      RotateLoopWithProfile ? L.getHeader() : findBestLoopTop(L, LoopBlockSet);

  // If we selected just the header for the loop top, look for a potentially
  // profitable exit block in the event that rotating the loop can eliminate
  // branches by placing an exit edge at the bottom.
  MachineBasicBlock *ExitingBB = nullptr;
  if (!RotateLoopWithProfile && LoopTop == L.getHeader())
    ExitingBB = findBestLoopExit(L, LoopBlockSet);

  BlockChain &LoopChain = *BlockToChain[LoopTop];

  // FIXME: This is a really lame way of walking the chains in the loop: we
  // walk the blocks, and use a set to prevent visiting a particular chain
  // twice.
  SmallPtrSet<BlockChain *, 4> UpdatedPreds;
  assert(LoopChain.UnscheduledPredecessors == 0);
  UpdatedPreds.insert(&LoopChain);

  for (MachineBasicBlock *LoopBB : LoopBlockSet)
    fillWorkLists(LoopBB, UpdatedPreds, &LoopBlockSet);

  buildChain(LoopTop, LoopChain, &LoopBlockSet);

  if (RotateLoopWithProfile)
    rotateLoopWithProfile(LoopChain, L, LoopBlockSet);
  else
    rotateLoop(LoopChain, ExitingBB, LoopBlockSet);

  DEBUG({
    // Crash at the end so we get all of the debugging output first.
    bool BadLoop = false;
    if (LoopChain.UnscheduledPredecessors) {
      BadLoop = true;
      dbgs() << "Loop chain contains a block without its preds placed!\n"
             << "  Loop header:  " << getBlockName(*L.block_begin()) << "\n"
             << "  Chain header: " << getBlockName(*LoopChain.begin()) << "\n";
    }
    for (MachineBasicBlock *ChainBB : LoopChain) {
      dbgs() << "          ... " << getBlockName(ChainBB) << "\n";
      if (!LoopBlockSet.erase(ChainBB)) {
        // We don't mark the loop as bad here because there are real situations
        // where this can occur. For example, with an unanalyzable fallthrough
        // from a loop block to a non-loop block or vice versa.
        dbgs() << "Loop chain contains a block not contained by the loop!\n"
               << "  Loop header:  " << getBlockName(*L.block_begin()) << "\n"
               << "  Chain header: " << getBlockName(*LoopChain.begin()) << "\n"
               << "  Bad block:    " << getBlockName(ChainBB) << "\n";
      }
    }

    if (!LoopBlockSet.empty()) {
      BadLoop = true;
      for (MachineBasicBlock *LoopBB : LoopBlockSet)
        dbgs() << "Loop contains blocks never placed into a chain!\n"
               << "  Loop header:  " << getBlockName(*L.block_begin()) << "\n"
               << "  Chain header: " << getBlockName(*LoopChain.begin()) << "\n"
               << "  Bad block:    " << getBlockName(LoopBB) << "\n";
    }
    assert(!BadLoop && "Detected problems with the placement of this loop.");
  });

  BlockWorkList.clear();
  EHPadWorkList.clear();
}

/// When OutlineOpitonalBranches is on, this method collects BBs that
/// dominates all terminator blocks of the function \p F.
void MachineBlockPlacement::collectMustExecuteBBs() {
  if (OutlineOptionalBranches) {
    // Find the nearest common dominator of all of F's terminators.
    MachineBasicBlock *Terminator = nullptr;
    for (MachineBasicBlock &MBB : *F) {
      if (MBB.succ_size() == 0) {
        if (Terminator == nullptr)
          Terminator = &MBB;
        else
          Terminator = MDT->findNearestCommonDominator(Terminator, &MBB);
      }
    }

    // MBBs dominating this common dominator are unavoidable.
    UnavoidableBlocks.clear();
    for (MachineBasicBlock &MBB : *F) {
      if (MDT->dominates(&MBB, Terminator)) {
        UnavoidableBlocks.insert(&MBB);
      }
    }
  }
}

void MachineBlockPlacement::buildCFGChains() {
  // Ensure that every BB in the function has an associated chain to simplify
  // the assumptions of the remaining algorithm.
  SmallVector<MachineOperand, 4> Cond; // For AnalyzeBranch.
  for (MachineFunction::iterator FI = F->begin(), FE = F->end(); FI != FE;
       ++FI) {
    MachineBasicBlock *BB = &*FI;
    BlockChain *Chain =
        new (ChainAllocator.Allocate()) BlockChain(BlockToChain, BB);
    // Also, merge any blocks which we cannot reason about and must preserve
    // the exact fallthrough behavior for.
    for (;;) {
      Cond.clear();
      MachineBasicBlock *TBB = nullptr, *FBB = nullptr; // For AnalyzeBranch.
      if (!TII->analyzeBranch(*BB, TBB, FBB, Cond) || !FI->canFallThrough())
        break;

      MachineFunction::iterator NextFI = std::next(FI);
      MachineBasicBlock *NextBB = &*NextFI;
      // Ensure that the layout successor is a viable block, as we know that
      // fallthrough is a possibility.
      assert(NextFI != FE && "Can't fallthrough past the last block.");
      DEBUG(dbgs() << "Pre-merging due to unanalyzable fallthrough: "
                   << getBlockName(BB) << " -> " << getBlockName(NextBB)
                   << "\n");
      Chain->merge(NextBB, nullptr);
      FI = NextFI;
      BB = NextBB;
    }
  }

  // Turned on with OutlineOptionalBranches option
  collectMustExecuteBBs();

  // Build any loop-based chains.
  for (MachineLoop *L : *MLI)
    buildLoopChains(*L);

  assert(BlockWorkList.empty());
  assert(EHPadWorkList.empty());

  SmallPtrSet<BlockChain *, 4> UpdatedPreds;
  for (MachineBasicBlock &MBB : *F)
    fillWorkLists(&MBB, UpdatedPreds);

  BlockChain &FunctionChain = *BlockToChain[&F->front()];
  buildChain(&F->front(), FunctionChain);

#ifndef NDEBUG
  typedef SmallPtrSet<MachineBasicBlock *, 16> FunctionBlockSetType;
#endif
  DEBUG({
    // Crash at the end so we get all of the debugging output first.
    bool BadFunc = false;
    FunctionBlockSetType FunctionBlockSet;
    for (MachineBasicBlock &MBB : *F)
      FunctionBlockSet.insert(&MBB);

    for (MachineBasicBlock *ChainBB : FunctionChain)
      if (!FunctionBlockSet.erase(ChainBB)) {
        BadFunc = true;
        dbgs() << "Function chain contains a block not in the function!\n"
               << "  Bad block:    " << getBlockName(ChainBB) << "\n";
      }

    if (!FunctionBlockSet.empty()) {
      BadFunc = true;
      for (MachineBasicBlock *RemainingBB : FunctionBlockSet)
        dbgs() << "Function contains blocks never placed into a chain!\n"
               << "  Bad block:    " << getBlockName(RemainingBB) << "\n";
    }
    assert(!BadFunc && "Detected problems with the block placement.");
  });

  // Splice the blocks into place.
  MachineFunction::iterator InsertPos = F->begin();
  DEBUG(dbgs() << "[MBP] Function: "<< F->getName() << "\n");
  for (MachineBasicBlock *ChainBB : FunctionChain) {
    DEBUG(dbgs() << (ChainBB == *FunctionChain.begin() ? "Placing chain "
                                                       : "          ... ")
                 << getBlockName(ChainBB) << "\n");
    if (InsertPos != MachineFunction::iterator(ChainBB))
      F->splice(InsertPos, ChainBB);
    else
      ++InsertPos;

    // Update the terminator of the previous block.
    if (ChainBB == *FunctionChain.begin())
      continue;
    MachineBasicBlock *PrevBB = &*std::prev(MachineFunction::iterator(ChainBB));

    // FIXME: It would be awesome of updateTerminator would just return rather
    // than assert when the branch cannot be analyzed in order to remove this
    // boiler plate.
    Cond.clear();
    MachineBasicBlock *TBB = nullptr, *FBB = nullptr; // For AnalyzeBranch.

    // The "PrevBB" is not yet updated to reflect current code layout, so,
    //   o. it may fall-through to a block without explicit "goto" instruction
    //      before layout, and no longer fall-through it after layout; or
    //   o. just opposite.
    //
    // analyzeBranch() may return erroneous value for FBB when these two
    // situations take place. For the first scenario FBB is mistakenly set NULL;
    // for the 2nd scenario, the FBB, which is expected to be NULL, is
    // mistakenly pointing to "*BI".
    // Thus, if the future change needs to use FBB before the layout is set, it
    // has to correct FBB first by using the code similar to the following:
    //
    // if (!Cond.empty() && (!FBB || FBB == ChainBB)) {
    //   PrevBB->updateTerminator();
    //   Cond.clear();
    //   TBB = FBB = nullptr;
    //   if (TII->analyzeBranch(*PrevBB, TBB, FBB, Cond)) {
    //     // FIXME: This should never take place.
    //     TBB = FBB = nullptr;
    //   }
    // }
    if (!TII->analyzeBranch(*PrevBB, TBB, FBB, Cond))
      PrevBB->updateTerminator();
  }

  // Fixup the last block.
  Cond.clear();
  MachineBasicBlock *TBB = nullptr, *FBB = nullptr; // For AnalyzeBranch.
  if (!TII->analyzeBranch(F->back(), TBB, FBB, Cond))
    F->back().updateTerminator();

  BlockWorkList.clear();
  EHPadWorkList.clear();
}

void MachineBlockPlacement::optimizeBranches() {
  BlockChain &FunctionChain = *BlockToChain[&F->front()];
  SmallVector<MachineOperand, 4> Cond; // For AnalyzeBranch.

  // Now that all the basic blocks in the chain have the proper layout,
  // make a final call to AnalyzeBranch with AllowModify set.
  // Indeed, the target may be able to optimize the branches in a way we
  // cannot because all branches may not be analyzable.
  // E.g., the target may be able to remove an unconditional branch to
  // a fallthrough when it occurs after predicated terminators.
  for (MachineBasicBlock *ChainBB : FunctionChain) {
    Cond.clear();
    MachineBasicBlock *TBB = nullptr, *FBB = nullptr; // For AnalyzeBranch.
    if (!TII->analyzeBranch(*ChainBB, TBB, FBB, Cond, /*AllowModify*/ true)) {
      // If PrevBB has a two-way branch, try to re-order the branches
      // such that we branch to the successor with higher probability first.
      if (TBB && !Cond.empty() && FBB &&
          MBPI->getEdgeProbability(ChainBB, FBB) >
              MBPI->getEdgeProbability(ChainBB, TBB) &&
          !TII->ReverseBranchCondition(Cond)) {
        DEBUG(dbgs() << "Reverse order of the two branches: "
                     << getBlockName(ChainBB) << "\n");
        DEBUG(dbgs() << "    Edge probability: "
                     << MBPI->getEdgeProbability(ChainBB, FBB) << " vs "
                     << MBPI->getEdgeProbability(ChainBB, TBB) << "\n");
        DebugLoc dl; // FIXME: this is nowhere
        TII->RemoveBranch(*ChainBB);
        TII->InsertBranch(*ChainBB, FBB, TBB, Cond, dl);
        ChainBB->updateTerminator();
      }
    }
  }
}

void MachineBlockPlacement::alignBlocks() {
  // Walk through the backedges of the function now that we have fully laid out
  // the basic blocks and align the destination of each backedge. We don't rely
  // exclusively on the loop info here so that we can align backedges in
  // unnatural CFGs and backedges that were introduced purely because of the
  // loop rotations done during this layout pass.
  if (F->getFunction()->optForSize())
    return;
  BlockChain &FunctionChain = *BlockToChain[&F->front()];
  if (FunctionChain.begin() == FunctionChain.end())
    return; // Empty chain.

  const BranchProbability ColdProb(1, 5); // 20%
  BlockFrequency EntryFreq = MBFI->getBlockFreq(&F->front());
  BlockFrequency WeightedEntryFreq = EntryFreq * ColdProb;
  for (MachineBasicBlock *ChainBB : FunctionChain) {
    if (ChainBB == *FunctionChain.begin())
      continue;

    // Don't align non-looping basic blocks. These are unlikely to execute
    // enough times to matter in practice. Note that we'll still handle
    // unnatural CFGs inside of a natural outer loop (the common case) and
    // rotated loops.
    MachineLoop *L = MLI->getLoopFor(ChainBB);
    if (!L)
      continue;

    unsigned Align = TLI->getPrefLoopAlignment(L);
    if (!Align)
      continue; // Don't care about loop alignment.

    // If the block is cold relative to the function entry don't waste space
    // aligning it.
    BlockFrequency Freq = MBFI->getBlockFreq(ChainBB);
    if (Freq < WeightedEntryFreq)
      continue;

    // If the block is cold relative to its loop header, don't align it
    // regardless of what edges into the block exist.
    MachineBasicBlock *LoopHeader = L->getHeader();
    BlockFrequency LoopHeaderFreq = MBFI->getBlockFreq(LoopHeader);
    if (Freq < (LoopHeaderFreq * ColdProb))
      continue;

    // Check for the existence of a non-layout predecessor which would benefit
    // from aligning this block.
    MachineBasicBlock *LayoutPred =
        &*std::prev(MachineFunction::iterator(ChainBB));

    // Force alignment if all the predecessors are jumps. We already checked
    // that the block isn't cold above.
    if (!LayoutPred->isSuccessor(ChainBB)) {
      ChainBB->setAlignment(Align);
      continue;
    }

    // Align this block if the layout predecessor's edge into this block is
    // cold relative to the block. When this is true, other predecessors make up
    // all of the hot entries into the block and thus alignment is likely to be
    // important.
    BranchProbability LayoutProb =
        MBPI->getEdgeProbability(LayoutPred, ChainBB);
    BlockFrequency LayoutEdgeFreq = MBFI->getBlockFreq(LayoutPred) * LayoutProb;
    if (LayoutEdgeFreq <= (Freq * ColdProb))
      ChainBB->setAlignment(Align);
  }
}

bool MachineBlockPlacement::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(*MF.getFunction()))
    return false;

  // Check for single-block functions and skip them.
  if (std::next(MF.begin()) == MF.end())
    return false;

  F = &MF;
  MBPI = &getAnalysis<MachineBranchProbabilityInfo>();
  MBFI = llvm::make_unique<BranchFolder::MBFIWrapper>(
      getAnalysis<MachineBlockFrequencyInfo>());
  MLI = &getAnalysis<MachineLoopInfo>();
  TII = MF.getSubtarget().getInstrInfo();
  TLI = MF.getSubtarget().getTargetLowering();
  MDT = &getAnalysis<MachineDominatorTree>();
  assert(BlockToChain.empty());

  buildCFGChains();

  // Changing the layout can create new tail merging opportunities.
  TargetPassConfig *PassConfig = &getAnalysis<TargetPassConfig>();
  // TailMerge can create jump into if branches that make CFG irreducible for
  // HW that requires structured CFG.
  bool EnableTailMerge = !MF.getTarget().requiresStructuredCFG() &&
                         PassConfig->getEnableTailMerge() &&
                         BranchFoldPlacement;
  // No tail merging opportunities if the block number is less than four.
  if (MF.size() > 3 && EnableTailMerge) {
    // Default to the standard tail-merge-size option.
    unsigned TailMergeSize = 0;
    BranchFolder BF(/*EnableTailMerge=*/true, /*CommonHoist=*/false, *MBFI,
                    *MBPI, TailMergeSize);

    if (BF.OptimizeFunction(MF, TII, MF.getSubtarget().getRegisterInfo(),
                            getAnalysisIfAvailable<MachineModuleInfo>(), MLI,
                            /*AfterBlockPlacement=*/true)) {
      // Redo the layout if tail merging creates/removes/moves blocks.
      BlockToChain.clear();
      ChainAllocator.DestroyAll();
      buildCFGChains();
    }
  }

  optimizeBranches();
  alignBlocks();

  BlockToChain.clear();
  ChainAllocator.DestroyAll();

  if (AlignAllBlock)
    // Align all of the blocks in the function to a specific alignment.
    for (MachineBasicBlock &MBB : MF)
      MBB.setAlignment(AlignAllBlock);
  else if (AlignAllNonFallThruBlocks) {
    // Align all of the blocks that have no fall-through predecessors to a
    // specific alignment.
    for (auto MBI = std::next(MF.begin()), MBE = MF.end(); MBI != MBE; ++MBI) {
      auto LayoutPred = std::prev(MBI);
      if (!LayoutPred->isSuccessor(&*MBI))
        MBI->setAlignment(AlignAllNonFallThruBlocks);
    }
  }

  // We always return true as we have no way to track whether the final order
  // differs from the original order.
  return true;
}

namespace {
/// \brief A pass to compute block placement statistics.
///
/// A separate pass to compute interesting statistics for evaluating block
/// placement. This is separate from the actual placement pass so that they can
/// be computed in the absence of any placement transformations or when using
/// alternative placement strategies.
class MachineBlockPlacementStats : public MachineFunctionPass {
  /// \brief A handle to the branch probability pass.
  const MachineBranchProbabilityInfo *MBPI;

  /// \brief A handle to the function-wide block frequency pass.
  const MachineBlockFrequencyInfo *MBFI;

public:
  static char ID; // Pass identification, replacement for typeid
  MachineBlockPlacementStats() : MachineFunctionPass(ID) {
    initializeMachineBlockPlacementStatsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineBranchProbabilityInfo>();
    AU.addRequired<MachineBlockFrequencyInfo>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
}

char MachineBlockPlacementStats::ID = 0;
char &llvm::MachineBlockPlacementStatsID = MachineBlockPlacementStats::ID;
INITIALIZE_PASS_BEGIN(MachineBlockPlacementStats, "block-placement-stats",
                      "Basic Block Placement Stats", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineBranchProbabilityInfo)
INITIALIZE_PASS_DEPENDENCY(MachineBlockFrequencyInfo)
INITIALIZE_PASS_END(MachineBlockPlacementStats, "block-placement-stats",
                    "Basic Block Placement Stats", false, false)

bool MachineBlockPlacementStats::runOnMachineFunction(MachineFunction &F) {
  // Check for single-block functions and skip them.
  if (std::next(F.begin()) == F.end())
    return false;

  MBPI = &getAnalysis<MachineBranchProbabilityInfo>();
  MBFI = &getAnalysis<MachineBlockFrequencyInfo>();

  for (MachineBasicBlock &MBB : F) {
    BlockFrequency BlockFreq = MBFI->getBlockFreq(&MBB);
    Statistic &NumBranches =
        (MBB.succ_size() > 1) ? NumCondBranches : NumUncondBranches;
    Statistic &BranchTakenFreq =
        (MBB.succ_size() > 1) ? CondBranchTakenFreq : UncondBranchTakenFreq;
    for (MachineBasicBlock *Succ : MBB.successors()) {
      // Skip if this successor is a fallthrough.
      if (MBB.isLayoutSuccessor(Succ))
        continue;

      BlockFrequency EdgeFreq =
          BlockFreq * MBPI->getEdgeProbability(&MBB, Succ);
      ++NumBranches;
      BranchTakenFreq += EdgeFreq.getFrequency();
    }
  }

  return false;
}
