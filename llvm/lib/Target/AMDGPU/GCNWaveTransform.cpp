//===- GCNWaveTransform.cpp - GCN Wave Transform ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Transform a function from thread-level to wave-level control flow
///
/// This pass is responsible for:
/// - Building the wave-level reconverging CFG and selecting corresponding
///   branch instructions.
/// - Constructing execmasks
///
/// TODO: In GlobalISel, this pass is additionally responsible for assigning
///       uniform vs. divergent register banks(?)
///
///
/// \section Reconvergence transform
///
/// The reconvergence transform establishes the "reconverging" property for the
/// CFG:
///
///   Every block with divergent terminator has exactly two successors, one of
///   which is a post-dominator.
///
/// The post-dominator is called "secondary" successor. During execution, the
/// wave of execution will first branch to the "primary" successor (if there
/// are any threads that want to go down that path), while adding the other
/// threads to a "rejoin mask" associated with the secondary successor. Since
/// it is a post-dominator, the wave is guaranteed to reach the secondary
/// successor eventually, at which point the threads from the "rejoin mask"
/// are added back to the wave.
///
/// The secondary successor will often be a newly introduced "flow block",
/// as in a simple hammock with divergent terminator at A:
///
///     A                 A
///    / \                |\
///   B   C     ===>      | B
///    \ /                |/
///     D                 X
///                       |\
///                       | C
///                       |/
///                       D
///
/// The reconvergence algorithm traverses blocks in heart-adjusted reverse post
/// order (HARPO), i.e. blocks of every cycle are contiguous, and the cycle's
/// heart is visited first (or the header, if there is no heart).
///
/// Flow blocks are inserted when a visited block has a predecessor with
/// divergent terminator that requires a flow block for the reconverging
/// property.
///
//
// TODO-NOW:
//  - uniform in cycle / divergent outside
//  - double-check order of successor nodes for divergent WaveNode
//
// TODO:
//  - _actually_ implement HARPO
//  - multiple function return blocks
//  - complex heart regions:
//  -- multiple backward edges from within the pre-heart region
//  -- multiple backward edges _into_ the pre-heart region
//  -- second pass of core transform with post-heart regions rotated to the
//     front
//  -- extra flow nodes for back edges in the pre-heart region?
//  -- problem of entry into the heart region: do the "second pass" of the
//     core transform first?
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNLaneMaskUtils.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/ADT/IntEqClasses.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/CodeGen/MachineCycleAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineUniformityAnalysis.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "gcn-wave-transform"

static cl::opt<bool>
    GCNWaveTransformPrintFinal("gcn-wave-transform-print-final",
                               cl::desc("Print the final wave CFG"),
                               cl::init(false));

namespace {

struct WaveNode;

/// Map a lane-level successor or predecessor to a wave-level successor or
/// predecessor.
struct LaneEdge {
  WaveNode *Lane = nullptr;
  WaveNode *Wave = nullptr;

  LaneEdge() = default;
  LaneEdge(WaveNode *Lane, WaveNode *Wave) : Lane(Lane), Wave(Wave) {}
};

/// \brief Representation of a node / basic block in the wave CFG.
struct WaveNode {
  MachineBasicBlock *Block = nullptr;
  MachineCycle *Cycle = nullptr;
  SmallVector<WaveNode *, 4> Predecessors;
  SmallVector<WaveNode *, 4> Successors;
  SmallVector<LaneEdge, 4> LanePredecessors;
  SmallVector<LaneEdge, 4> LaneSuccessors;

  bool IsDivergent = false;
  bool IsSecondary = false;
  unsigned OrderIndex = 0;
  unsigned FlowNum = 0; // non-zero if flow node

  /// Used during reconverging algorithm: track known post-dominators on the
  /// fly.
  WaveNode *LatestPostDom = nullptr;

  Register RejoinMask;

  WaveNode(MachineBasicBlock *Block, MachineCycle *Cycle)
      : Block(Block), Cycle(Cycle), LatestPostDom(this) {}
  WaveNode(MachineCycle *Cycle, unsigned FlowNum)
      : Cycle(Cycle), FlowNum(FlowNum), LatestPostDom(this) {}
  WaveNode(const WaveNode &) = delete;
  WaveNode(WaveNode &&) = delete;
  WaveNode &operator=(const WaveNode &) = delete;
  WaveNode &operator=(WaveNode &&) = delete;

  Printable printableName() const {
    return Printable([this](raw_ostream &Out) {
      if (Block) {
        // FIXME: remove the need to build a context everytime
        Out << MachineSSAContext().print(Block);
      }
      if (Block && FlowNum)
        Out << '.';
      if (FlowNum)
        Out << "<flow-" << FlowNum << '>';
    });
  }
};

/// \brief Helper class for making a CFG reconverging.
class ReconvergeCFGHelper {
private:
  // MachineConvergenceInfo &ConvergenceInfo;
  MachineCycleInfo &CycleInfo;
  MachineUniformityInfo &UniformInfo;
  MachineDominatorTree &DomTree;

  unsigned NumFlowNodes = 0;

  /// Current HAPO-ordered list of nodes.
  ///
  /// During individual transform steps, a prefix of this vector may have been
  /// moved to \ref NextNodes already.
  std::vector<std::unique_ptr<WaveNode>> Nodes;

  /// During individual transform steps, a prefix of the next \ref Nodes
  /// vector.
  std::vector<std::unique_ptr<WaveNode>> NextNodes;

  DenseMap<MachineBasicBlock *, WaveNode *> NodeForBlock;

  /// Temporary variables used by \ref appendOpenSet, persisted to reduce
  /// the number of temporary allocations.
  struct {
    SmallVector<WaveNode *, 8> Worklist;
    DenseSet<WaveNode *> Found;
  } OpenSetScan;

public:
  ReconvergeCFGHelper(MachineCycleInfo &CycleInfo,
                      MachineUniformityInfo &UniformInfo,
                      MachineDominatorTree &DomTree)
      : /*ConvergenceInfo(convergenceInfo),*/
        CycleInfo(CycleInfo), UniformInfo(UniformInfo), DomTree(DomTree) {}

  void run();

  WaveNode *rerouteViaNewNode(ArrayRef<WaveNode *> FromList, WaveNode *ToNode);

  WaveNode *nodeForBlock(MachineBasicBlock *Block) {
    return NodeForBlock.lookup(Block);
  }
  void setNodeForBlock(MachineBasicBlock *Block, WaveNode *Node) {
    assert(!NodeForBlock.count(Block));
    NodeForBlock.try_emplace(Block, Node);
  }

  template <typename WrappedIteratorT, typename WaveNodeT>
  struct node_iterator_impl;

  template <typename WrappedIteratorT, typename WaveNodeT>
  using node_iterator_impl_base = iterator_adaptor_base<
      node_iterator_impl<WrappedIteratorT, WaveNodeT>, WrappedIteratorT,
      typename std::iterator_traits<WrappedIteratorT>::iterator_category,
      WaveNodeT *, // value type
      typename std::iterator_traits<WrappedIteratorT>::difference_type,
      WaveNodeT **, // pointer type
      WaveNodeT *>; // reference type

  template <typename WrappedIteratorT, typename WaveNodeT>
  struct node_iterator_impl
      : node_iterator_impl_base<WrappedIteratorT, WaveNodeT> {
    node_iterator_impl() = default;
    explicit node_iterator_impl(WrappedIteratorT it)
        : node_iterator_impl_base<WrappedIteratorT, WaveNodeT>(it) {}

    WaveNodeT *operator*() const { return this->I->get(); }
  };

  using const_node_iterator =
      node_iterator_impl<std::vector<std::unique_ptr<WaveNode>>::const_iterator,
                         const WaveNode>;
  using node_iterator =
      node_iterator_impl<std::vector<std::unique_ptr<WaveNode>>::const_iterator,
                         WaveNode>;

  const_node_iterator nodes_begin() const {
    return const_node_iterator(Nodes.begin());
  }
  const_node_iterator nodes_end() const {
    return const_node_iterator(Nodes.end());
  }
  iterator_range<const_node_iterator> nodes() const {
    return {nodes_begin(), nodes_end()};
  }

  node_iterator nodes_begin() { return node_iterator(Nodes.begin()); }
  node_iterator nodes_end() { return node_iterator(Nodes.end()); }
  iterator_range<node_iterator> nodes() { return {nodes_begin(), nodes_end()}; }

  void printNodes(raw_ostream &out);
  void dumpNodes();

private:
  void cleanupSimpleFlowNodes();

  MachineBasicBlock *getEffectiveHeart(const MachineCycle *Cycle);
  void prepareNodesEnterCycle(WaveNode *HeaderNode);
  void prepareNodesExitCycle(MachineCycle *Cycle, WaveNode *NextNode);
  bool appendOpenSet(WaveNode *Grom, WaveNode *Nound,
                     SmallVectorImpl<WaveNode *> &OpenSet);
  void reroute(ArrayRef<WaveNode *> FromList, WaveNode *ToNode,
               WaveNode *ViaNode);
  void rerouteEdgesBeyond(ArrayRef<WaveNode *> from, WaveNode *ToBeyond,
                          WaveNode *ViaNode);
  void rerouteLane(WaveNode *FromNode, WaveNode *ToNode, WaveNode *ViaNode);

  void verifyNodes();
};

} // anonymous namespace

static MachineBasicBlock *getHeartBlock(const MachineCycle *Cycle) {
  return nullptr;
}

class HeartAdjustedPostOrder {
public:
  using CycleT = MachineCycle;
  using BlockT = MachineBasicBlock;
  using DominatorTreeT = MachineDominatorTree;
  using CycleInfoT = MachineCycleInfo;
  using const_iterator = typename std::vector<BlockT *>::const_iterator;

  bool empty() const { return Order.empty(); }
  size_t size() const { return Order.size(); }

  void clear() { Order.clear(); }
  // void compute(const ConvergenceInfoT &convergenceInfo,
  //              const CycleInfoT &CycleInfo, const DominatorTreeT &domTree);

  const_iterator begin() const { return Order.begin(); }
  const_iterator end() const { return Order.end(); }
  BlockT *operator[](size_t Idx) const { return Order[Idx]; }

private:
  std::vector<BlockT *> Order;

public:
  void compute(const CycleInfoT &CycleInfo, const DominatorTreeT &domTree) {
    // In our forward traversal, the modification bullets from the description
    // of heart-adjusted reverse post order happen in reverse: within each
    // cycle, we do a depth-first post-order traversal of only the blocks
    // belonging to the cycle, starting with the heart.
    //
    // The depth-first search mainly uses a stack of blocks, with a look-aside
    // stack of cycles. Cycles remain on the stack until their final post-order
    // visit, at which time their Blocks are added to the parent cycle's order.
    // We also maintain a linked list of cycles that are active in the sense
    // that we're currently visiting blocks inside them.
    struct HapoCycle {
      const CycleT *Cycle;
      BlockT *Heart;
      unsigned ParentStackIdx;
      std::vector<BlockT *> Order;
      SmallVector<BlockT *, 4> PostponedBlocks;

      explicit HapoCycle(const CycleT *Cycle, BlockT *Heart,
                         unsigned ParentStackIdx)
          : Cycle(Cycle), Heart(Heart), ParentStackIdx(ParentStackIdx) {}
    };

    DenseSet<BlockT *> VisitedBlocks;
    SmallVector<BlockT *, 32> BlockStack;
    // DoneIdxStack contains ((size of BlockStack before pop) << 1) |
    // isCycleHeart
    SmallVector<unsigned, 32> DoneIdxStack;
    SmallVector<HapoCycle, 8> CycleStack;
    unsigned CurrentCycleStackIdx = 0;

    BlockT *EntryBlock = domTree.getRootNode()->getBlock();
    CycleStack.emplace_back(nullptr, nullptr, 0);
    BlockStack.push_back(EntryBlock);

    // The entry block is not marked as a cycle header, so that we don't attempt
    // to pop the root cycle: it is handled at the very end after the loop.
    DoneIdxStack.push_back(BlockStack.size() << 1);
    llvm::append_range(BlockStack, successors(EntryBlock));

    do {
      MachineBasicBlock *Block = BlockStack.back();
      unsigned DoneBack = DoneIdxStack.back();

      if (BlockStack.size() == (DoneBack >> 1)) {
        if (!(DoneBack & 1)) {
          // Post-order visit of a regular Block.
          CycleStack[CurrentCycleStackIdx].Order.push_back(Block);
          BlockStack.pop_back();
          DoneIdxStack.pop_back();
          continue;
        }

        // This is the post-order visit of an effective Cycle heart.
        HapoCycle &Cycle = CycleStack.back();
        if (CurrentCycleStackIdx == CycleStack.size() - 1)
          CurrentCycleStackIdx = Cycle.ParentStackIdx;

        if (!Cycle.PostponedBlocks.empty()) {
          // Enqueue the Cycle's postponed exit Blocks if there are any. In this
          // case, we aren't actually at the post-order visit of the Cycle yet,
          // if we interpret it as a contracted node contained in its parent.
          for (BlockT *postponed : Cycle.PostponedBlocks) {
            assert(VisitedBlocks.count(postponed));
            VisitedBlocks.erase(postponed);
            BlockStack.push_back(postponed);
          }
          Cycle.PostponedBlocks.clear();
          continue;
        }

        // True post-order visit: collect all of the Cycle.
        Cycle.Order.push_back(Block);
        BlockStack.pop_back();
        DoneIdxStack.pop_back();

        auto &ParentOrder = CycleStack[Cycle.ParentStackIdx].Order;
        ParentOrder.insert(ParentOrder.end(), Cycle.Order.begin(),
                           Cycle.Order.end());
        CycleStack.pop_back();
        continue;
      }

      if (!VisitedBlocks.insert(Block).second) {
        BlockStack.pop_back();
        continue; // already visited this one
      }

      // Pre-order visit of the block.
      const CycleT *CurrentCycle = CycleStack[CurrentCycleStackIdx].Cycle;
      BlockT *CurrentHeart = CycleStack[CurrentCycleStackIdx].Heart;
      const CycleT *BlockCycle = CycleInfo.getCycle(Block);

      if (BlockCycle == CurrentCycle ||
          (CurrentHeart && CurrentHeart == getHeartBlock(BlockCycle))) {
        DoneIdxStack.push_back(BlockStack.size() << 1);
        llvm::append_range(BlockStack, successors(Block));
        continue;
      }

      if (!CurrentCycle || CurrentCycle->contains(BlockCycle)) {
        // Entering a child cycle. In the case of irreducible control flow,
        // BlockCycle might not be a direct child -- find it.
        while ((BlockCycle->getParentCycle() != CurrentCycle) &&
               (!CurrentHeart ||
                CurrentHeart != getHeartBlock(BlockCycle->getParentCycle())))
          BlockCycle = BlockCycle->getParentCycle();

        BlockT *Heart = getHeartBlock(BlockCycle);
        BlockT *EffectiveHeart = Heart ? Heart : BlockCycle->getHeader();

        CycleStack.emplace_back(BlockCycle, Heart, CurrentCycleStackIdx);
        CurrentCycleStackIdx = CycleStack.size() - 1;

        // Fixup state as-if we're visiting the effective heart.
        if (Block != EffectiveHeart) {
          BlockStack.pop_back();
          BlockStack.push_back(EffectiveHeart);
          VisitedBlocks.erase(Block);
          VisitedBlocks.insert(EffectiveHeart);
        }

        DoneIdxStack.push_back((BlockStack.size() << 1) | 1);
        llvm::append_range(BlockStack, successors(Block));
        continue;
      }

      // This Block is not contained in the current Cycle; we have to postpone
      // it.
      BlockStack.pop_back();

      HapoCycle *PostponeCycle = &CycleStack[CurrentCycleStackIdx];
      for (;;) {
        HapoCycle *parent = &CycleStack[PostponeCycle->ParentStackIdx];
        if (!parent->Cycle || parent->Cycle->contains(BlockCycle))
          break;
        PostponeCycle = parent;
      }
      PostponeCycle->PostponedBlocks.push_back(Block);
    } while (!BlockStack.empty());

    assert(CycleStack.size() == 1);
    Order = std::move(CycleStack[0].Order);
  }
};

void ReconvergeCFGHelper::run() {
  HeartAdjustedPostOrder hapo;
  hapo.compute(CycleInfo, DomTree);

  // Step 1: Create initial set of WaveNodes mirroring the thread-level CFG.
  Nodes.reserve(hapo.size());
  for (MachineBasicBlock *Block : llvm::reverse(hapo)) {
    Nodes.emplace_back(
        std::make_unique<WaveNode>(Block, CycleInfo.getCycle(Block)));
    WaveNode *WN = Nodes.back().get();
    WN->IsDivergent = UniformInfo.hasDivergentTerminator(*Block);
    NodeForBlock.insert(std::make_pair(Block, WN));
  }

  // Link up CFG edges. Note that we ignore unreachable predecessors.
  for (const auto &NodePtr : Nodes) {
    for (MachineBasicBlock *Succ : NodePtr->Block->successors()) {
      auto SuccNodeIt = NodeForBlock.find(Succ);
      assert(SuccNodeIt != NodeForBlock.end());
      NodePtr->Successors.push_back(SuccNodeIt->second);
      NodePtr->LaneSuccessors.emplace_back(SuccNodeIt->second,
                                           SuccNodeIt->second);
      SuccNodeIt->second->Predecessors.push_back(NodePtr.get());
      SuccNodeIt->second->LanePredecessors.emplace_back(NodePtr.get(),
                                                        NodePtr.get());
    }
  }

  LLVM_DEBUG(dbgs() << "CFG mirror:\n"; dumpNodes());

  // Step 2: Create helper nodes for cycles:
  //
  // At the end of every maximal cycle for a heart block, reroute every
  // backwards edge within the ordering span of the cycle (i.e., back edge to
  // the header of any cycle with the same heart, or edge from after heart block
  // to before) through a single flow node. (A single flow node
  for (unsigned Index = 0; Index != Nodes.size(); ++Index)
    Nodes[Index]->OrderIndex = Index;

  MachineCycle *CurrentCycle = nullptr;

  NextNodes.reserve(Nodes.size());

  for (auto &NodePtr : Nodes) {
    WaveNode *Node = NodePtr.get();

    if (Node->Cycle != CurrentCycle) {
      while (CurrentCycle && !CurrentCycle->contains(Node->Cycle)) {
        LLVM_DEBUG(dbgs() << "Prepare exit cycle: "
                          << CurrentCycle->print(CycleInfo.getSSAContext())
                          << '\n');

        prepareNodesExitCycle(CurrentCycle, Node);
        CurrentCycle = CurrentCycle->getParentCycle();

        LLVM_DEBUG(dumpNodes());
      }

      if (Node->Cycle != CurrentCycle) {
        assert(Node->Cycle->getParentCycle() == CurrentCycle);
        LLVM_DEBUG(dbgs() << "Prepare enter cycle: "
                          << Node->Cycle->print(CycleInfo.getSSAContext())
                          << '\n');

        prepareNodesEnterCycle(Node);
        CurrentCycle = Node->Cycle;

        LLVM_DEBUG(dumpNodes());
      }
    }

    NextNodes.push_back(std::move(NodePtr));
  }
  Nodes = std::move(NextNodes);
  NextNodes.clear();

  LLVM_DEBUG(dbgs() << "With helper nodes:\n"; dumpNodes());

  // Step 3: Run reconverging transform.
  for (unsigned Index = 0; Index != Nodes.size(); ++Index)
    Nodes[Index]->OrderIndex = Index;

  SmallVector<WaveNode *, 4> RerouteCandidates;
  IntEqClasses RerouteCandidateClasses;
  SmallVector<int, 4> PredClasses;
  SmallVector<WaveNode *, 4> RerouteNodes;
  SmallVector<WaveNode *, 4> RerouteRoots;
  SmallVector<WaveNode *, 4> TmpSet;
  for (auto &NodePtr : Nodes) {
    WaveNode *Node = NodePtr.get();

    LLVM_DEBUG(dbgs() << "Reconverging: " << Node->printableName() << '\n');

    int RerouteClass = -1;
    for (WaveNode *Pred : Node->Predecessors) {
      // Backward edge and predecessors without divergence don't need to
      // establish the reconverging property.
      if (Pred->OrderIndex >= Node->OrderIndex || !Pred->IsDivergent) {
        PredClasses.push_back(-1);
        continue;
      }

      bool HaveEarlierSuccessor = false;
      for (WaveNode *Succ : Pred->Successors) {
        assert(Succ->OrderIndex != Node->OrderIndex || Succ == Node);
        if (Succ->OrderIndex < Node->OrderIndex) {
          HaveEarlierSuccessor = true;
          break;
        }
      }
      if (!HaveEarlierSuccessor) {
        // The current node is going to be the primary successor.
        auto SelfIt = llvm::find(Pred->Successors, Node);
        std::rotate(Pred->Successors.begin(), SelfIt, SelfIt + 1);
        PredClasses.push_back(-1);
        continue;
      }

      bool AllEdgesToNode = appendOpenSet(Pred, Node, TmpSet);

      int PredClass = -1;
      for (WaveNode *reachableNode : TmpSet) {
        auto It = llvm::find(RerouteCandidates, reachableNode);
        int NodeClass;
        if (It != RerouteCandidates.end()) {
          NodeClass = std::distance(RerouteCandidates.begin(), It);
        } else {
          NodeClass = RerouteCandidates.size();
          RerouteCandidates.push_back(reachableNode);
          RerouteCandidateClasses.grow(RerouteCandidates.size());
        }

        if (PredClass == -1) {
          PredClass = NodeClass;
        } else {
          RerouteCandidateClasses.join(PredClass, NodeClass);
        }
      }

      TmpSet.clear();

      PredClasses.push_back(PredClass);

      if (!AllEdgesToNode) {
        // This predecessor reaches some "open" edge that bypasses the current
        // node and would contradict the reconverging property.
        //
        // The candidate nodes reachable from that predecessor must be rerouted,
        // as well as (transitively) all candidate nodes reachable from any
        // predecessor that can reach those candidate nodes.
        if (RerouteClass == -1) {
          RerouteClass = PredClass;
        } else {
          RerouteCandidateClasses.join(RerouteClass, PredClass);
        }
      }
    }
    assert(PredClasses.size() == Node->Predecessors.size());

    WaveNode *FlowNode = nullptr;
    if (RerouteClass != -1) {
      NextNodes.push_back(
          std::make_unique<WaveNode>(Node->Cycle, ++NumFlowNodes));
      FlowNode = NextNodes.back().get();
      FlowNode->OrderIndex = Node->OrderIndex;
      FlowNode->IsDivergent = true;
      FlowNode->IsSecondary = true;

      unsigned RerouteLeader = RerouteCandidateClasses.findLeader(RerouteClass);
      for (unsigned Idx = 0; Idx != RerouteCandidates.size(); ++Idx) {
        if (RerouteCandidateClasses.findLeader(Idx) == RerouteLeader)
          RerouteNodes.push_back(RerouteCandidates[Idx]);
      }
      for (unsigned Idx = 0; Idx != Node->Predecessors.size(); ++Idx) {
        if (PredClasses[Idx] == -1)
          continue;
        if (RerouteCandidateClasses.findLeader(PredClasses[Idx]) ==
            RerouteLeader)
          RerouteRoots.push_back(Node->Predecessors[Idx]);
      }

      rerouteEdgesBeyond(RerouteNodes, Node, FlowNode);

      // The current node is going to be the flow node's primary successor,
      // so rotate it to the front.
      auto SelfIt = llvm::find(FlowNode->Successors, Node);
      std::rotate(FlowNode->Successors.begin(), SelfIt, SelfIt + 1);

      // Compile-time optimization: record flow node as latest post-dominator
      // of all original predecessors for which we did rerouting.
      for (WaveNode *originalPredecessor : RerouteRoots)
        originalPredecessor->LatestPostDom = FlowNode;

      RerouteNodes.clear();
      RerouteRoots.clear();
    }

    RerouteCandidates.clear();
    RerouteCandidateClasses.clear();
    PredClasses.clear();

    for (WaveNode *Pred : Node->Predecessors) {
      if (Pred == FlowNode || !Pred->IsDivergent)
        continue;

      // TODO: handle the case where successors < 2
      // i.e. the same successor was listed multiple times and this is actually
      // a uniform unconditional branch.
      assert(Pred->Successors.size() == 2);

      WaveNode *Other;
      if (Node == Pred->Successors[0])
        Other = Pred->Successors[1];
      else
        Other = Pred->Successors[0];

      assert(Other->OrderIndex != Node->OrderIndex);
      if (Other->OrderIndex < Node->OrderIndex) {
        Node->IsSecondary = true;

        // Compile-time optimization: record this node as latest post-dominator
        // when possible.
        Pred->LatestPostDom = Node;
      }
    }

    NextNodes.push_back(std::move(NodePtr));

    LLVM_DEBUG(dumpNodes());
  }
  Nodes = std::move(NextNodes);
  NextNodes.clear();

  cleanupSimpleFlowNodes();
}

/// Short-circuit and remove flow nodes with a single wave successor.
void ReconvergeCFGHelper::cleanupSimpleFlowNodes() {
  bool Changed;

  do {
    Changed = false;

    for (auto &NodePtr : Nodes) {
      WaveNode *Node = NodePtr.get();
      if (!Node->FlowNum || Node->Successors.size() != 1) {
        NextNodes.push_back(std::move(NodePtr));
        continue;
      }

      WaveNode *Succ = Node->Successors[0];
      auto PredIt = llvm::find(Succ->Predecessors, Node);
      assert(PredIt != Succ->Predecessors.end());

      *PredIt = Succ->Predecessors.back();
      Succ->Predecessors.pop_back();
      assert(!is_contained(Succ->Predecessors, Node));

      // if flow node was a secondary target, copy the flag
      if (Node->IsSecondary)
        Succ->IsSecondary = true;

      for (WaveNode *Pred : Node->Predecessors) {
        if (!is_contained(Succ->Predecessors, Pred))
          Succ->Predecessors.push_back(Pred);

        // update LatestPostDom to avoid dangling pointers
        if (Pred->LatestPostDom == Node)
          Pred->LatestPostDom = Succ;

        bool HaveSucc = is_contained(Pred->Successors, Succ);
        auto SuccIt = llvm::find(Pred->Successors, Node);
        if (HaveSucc) {
          Pred->Successors.erase(SuccIt);
        } else {
          *SuccIt = Succ;
        }
        assert(!is_contained(Pred->Successors, Node));

        for (LaneEdge &LaneSucc : Pred->LaneSuccessors) {
          if (LaneSucc.Wave == Node)
            LaneSucc.Wave = Succ;
        }
      }

      for (LaneEdge &LanePred : Succ->LanePredecessors) {
        if (LanePred.Wave == Node) {
          auto PredIt =
              llvm::find_if(Node->LanePredecessors, [=](const LaneEdge &Pred) {
                return Pred.Lane == LanePred.Lane;
              });
          assert(PredIt != Node->LanePredecessors.end());
          LanePred.Wave = PredIt->Wave;
        }
      }

      Changed = true;
    }

    Nodes = std::move(NextNodes);
    NextNodes.clear();
  } while (Changed);

  LLVM_DEBUG(dbgs() << "After simplification:\n"; dumpNodes());
}

/// Return the given cycle's effective heart. If a cycle has no explicitly
/// specified heart, with use the cycle header as heart. This leads to a more
/// intuitive wave transform on natural loops with multiple back edges.
MachineBasicBlock *
ReconvergeCFGHelper::getEffectiveHeart(const MachineCycle *Cycle) {
  if (!Cycle)
    return nullptr;

  MachineBasicBlock *Heart = nullptr; // ConvergenceInfo.getHeartBlock(Cycle);
  if (Heart)
    return Heart;
  return Cycle->getHeader();
}

/// \brief Insert preparatory flow nodes for entering a cycle.
///
/// This method is called just before a cycle is entered, i.e. just before the
/// cycle's header is moved to \ref NextNodes.
///
/// The method unconditionally creates dedicated pre-entry nodes (i.e.,
/// pre-headers, but for all entry nodes in the case of irreducible cycles).
///
/// This ensures that any flow nodes that are required by the entry node
/// don't confound cycle and non-cycle control. Example:
///
///        |     |
///        |     v
///        |     A---->\
///        |    /      |
///        |   /       B
///        v  /        |
///      ^-H  |        |
///      |  \ |        |
///      |   \v        |
///      ^---<C        |
///           |        |
///          ...      ...
///
/// If A has a divergent branch and the main wave transform proceeds with
/// a top-down ordering, it proceeds to reroute the edges from A (incoming
/// to the cycle) and B (unrelated to the cycle) through a single flow
/// block, which unnecessarily causes the edge from B to pass through the
/// cycle. A pre-entry node for C that is processed before the cycle header
/// avoids this.
///
/// Dedicated, per-entry nodes are established to avoid triggering unneded
/// reroutes.
///
/// Pre-entry nodes are created unconditionally to guard against a situation
/// where the core transform creates a flow node that becomes a new entering
/// node with successors outside the cycle. We rely on a later cleanup to
/// remove unnecessary flow nodes in the end.
//
// TODO: Deal with nodes that are reachable without going through the cycle's
//       heart and that have two back edges. Keep various possible heart
//       structures in mind.
void ReconvergeCFGHelper::prepareNodesEnterCycle(WaveNode *headerNode) {
  MachineCycle *Cycle = headerNode->Cycle;
  SmallVector<WaveNode *, 4> Entering;

  assert(Cycle);
  for (unsigned Index = headerNode->OrderIndex;
       Index < Nodes.size() && Cycle->contains(Nodes[Index]->Cycle); ++Index) {
    // Check whether this is an entry block, and collect out-of-cycle
    // predecessors.
    WaveNode *Entry = Nodes[Index].get();
    for (WaveNode *Pred : Entry->Predecessors) {
      if (!Cycle->contains(Pred->Cycle))
        Entering.push_back(Pred);
    }

    if (!Entering.empty()) {
      NextNodes.push_back(
          std::make_unique<WaveNode>(Cycle->getParentCycle(), ++NumFlowNodes));
      WaveNode *FlowNode = NextNodes.back().get();
      FlowNode->OrderIndex = headerNode->OrderIndex;
      reroute(Entering, Entry, FlowNode);
      Entering.clear();
    }
  }
}

/// \brief Insert preparatory flow nodes for latches and cycle exits.
///
/// This method is called just after a cycle is left, i.e. just after the node
/// corresponding to the last block in the cycle is moved to \ref NextNodes.
///
/// If the cycle is the outer-most cycle for its heart, we reroute all backward
/// edges that cross the cycle's heart in the order (including backward edges of
/// a natural loop with heart at the header) through new flow nodes, with a
/// dedicated flow node per backwards target.
///
/// The purpose of these flow nodes is to ensure reconvergence before backwards
/// edges to satisfy the convergence rules of cycle hearts. Example (natural
/// loop with header as heart having a self-loop):
///
///     |           |
///  /-[A           A<---\
///  |  |     =>    |\   |
///  |  |           | \  |
///  ^-<B           B->X-^
///     |           |
///
/// If A has a divergent terminator, control flow will reconverge at X before
/// looping back to A. If A has no divergent terminator, the flow block is not
/// strictly needed. We rely on a post-reconverging cleanup to remove it either
/// way.
///
/// Additionally, these flow nodes ensure correct handling of the most common
/// case of nodes with multiple back edges. Example (A and B are hearts for the
/// cycles they head):
///
///      |                   |
///      A<---\              A<---\
///      |    |              |    |
///      |    |      =>      |    |
///      B<-\ |              B<-\ |
///     /|  | |             /|  | |
///    / |  | |            / |  | |
///   |  C--^ |           |  C--^ |
///   |   \   |           |  |    |
///   |    \--^           |  X----^
///   |                   |
///  ...                 ...
///
/// Flow block X is inserted when exiting the cycle headed by B.
///
/// If C has a divergent terminator, the core transform will reroute the
/// exiting edge from B through a new flow block when handling X to ensure
/// the reconverging condition at C.
///
/// Note: Nodes in the pre-heart region with multiple back edges need to be
///       handled separately!
///
void ReconvergeCFGHelper::prepareNodesExitCycle(MachineCycle *Cycle,
                                                WaveNode *nextNode) {
  SmallVector<WaveNode *, 4> FromNodes;
  SmallVector<WaveNode *, 4> ToNodes;

  assert(Cycle);
  MachineBasicBlock *Heart = getEffectiveHeart(Cycle);
  if (Heart && Heart != getEffectiveHeart(Cycle->getParentCycle())) {
    WaveNode *HeartNode = NodeForBlock.lookup(Heart);

    for (unsigned nextIndex = NextNodes.size() - 1;; nextIndex--) {
      WaveNode *Node = NextNodes[nextIndex].get();
      assert(Cycle->contains(Node->Cycle));

      bool isFromNode = false;
      for (WaveNode *Succ : Node->Successors) {
        if (Succ->OrderIndex <= HeartNode->OrderIndex) {
          isFromNode = true;
          if (!is_contained(ToNodes, Succ))
            ToNodes.push_back(Succ);
        }
      }

      if (isFromNode)
        FromNodes.push_back(Node);

      if (Node->Block == Heart)
        break;
    }

    // The sort should not be necessary for correctness, but it should help
    // generate a slightly cleaner wave CFG when there are multiple "to" nodes.
    llvm::sort(ToNodes, [](WaveNode *lhs, WaveNode *rhs) -> bool {
      return lhs->OrderIndex > rhs->OrderIndex;
    });

    for (WaveNode *ToNode : ToNodes) {
      MachineCycle *toCycle;
      if (Cycle->contains(ToNode->Cycle))
        toCycle = Cycle;
      else
        toCycle = Cycle->getParentCycle();

      NextNodes.push_back(std::make_unique<WaveNode>(toCycle, ++NumFlowNodes));
      WaveNode *FlowNode = NextNodes.back().get();
      FlowNode->OrderIndex = nextNode->OrderIndex;
      reroute(FromNodes, ToNode, FlowNode);
    }
  }
}

/// Compute the nodes that are reachable from \p from without going past
/// \p bound in the current node ordering, _and_ that have outgoing edges
/// to \p bound or later nodes ("open" edges).
///
/// Those nodes are appended to \p OpenSet.
///
/// Note: This method relies on WaveNode::LatestPostDom tracking to avoid
/// redundant scanning.
///
/// \return true if all found open edges go to \p bound
bool ReconvergeCFGHelper::appendOpenSet(WaveNode *from, WaveNode *bound,
                                        SmallVectorImpl<WaveNode *> &OpenSet) {
  while (from != from->LatestPostDom)
    from = from->LatestPostDom;
  assert(from->OrderIndex < bound->OrderIndex);

  OpenSetScan.Worklist.push_back(from);

  bool AllToBound = true;
  do {
    WaveNode *Node = OpenSetScan.Worklist.pop_back_val();
    if (Node != Node->LatestPostDom) {
      // Compress post-dom links on the fly
      while (Node->LatestPostDom != Node->LatestPostDom->LatestPostDom)
        Node->LatestPostDom = Node->LatestPostDom->LatestPostDom;
      Node = Node->LatestPostDom;
    }
    assert(Node->OrderIndex < bound->OrderIndex);

    if (!OpenSetScan.Found.insert(Node).second)
      continue;

    bool IsOpen = false;

    for (WaveNode *Succ : Node->Successors) {
      assert(Succ->OrderIndex != bound->OrderIndex || Succ == bound);
      if (Succ->OrderIndex >= bound->OrderIndex) {
        IsOpen = true;
        if (Succ != bound)
          AllToBound = false;
      } else {
        OpenSetScan.Worklist.push_back(Succ);
      }
    }

    if (IsOpen)
      OpenSet.push_back(Node);
  } while (!OpenSetScan.Worklist.empty());

  OpenSetScan.Found.clear();

  return AllToBound;
}

/// Reroute all edges going from any node in \p FromList to the \p ToNode
/// through a new flow node, and return that new node.
///
/// The new node will be appended to the \ref Nodes list.
WaveNode *ReconvergeCFGHelper::rerouteViaNewNode(ArrayRef<WaveNode *> FromList,
                                                 WaveNode *ToNode) {
  Nodes.push_back(std::make_unique<WaveNode>(ToNode->Cycle, ++NumFlowNodes));
  WaveNode *FlowNode = Nodes.back().get();
  reroute(FromList, ToNode, FlowNode);
  return FlowNode;
}

/// Reroute all edges going from any node in \p from to the \p to node via
/// the \p via node.
void ReconvergeCFGHelper::reroute(ArrayRef<WaveNode *> FromList,
                                  WaveNode *ToNode, WaveNode *ViaNode) {
  // In current use, we can assume that ViaNode is not connected to from or to.
  for (WaveNode *FromNode : FromList) {
    auto I = llvm::find(FromNode->Successors, ToNode);
    if (I == FromNode->Successors.end())
      continue;
    FromNode->Successors.erase(I);

    I = llvm::find(ToNode->Predecessors, FromNode);
    assert(I != ToNode->Predecessors.end());
    ToNode->Predecessors.erase(I);

    assert(!is_contained(FromNode->Successors, ViaNode));
    assert(!is_contained(ViaNode->Predecessors, FromNode));
    FromNode->Successors.push_back(ViaNode);
    ViaNode->Predecessors.push_back(FromNode);

    rerouteLane(FromNode, ToNode, ViaNode);
  }

  assert(!is_contained(ViaNode->Successors, ToNode));
  assert(!is_contained(ToNode->Predecessors, ViaNode));
  ViaNode->Successors.push_back(ToNode);
  ToNode->Predecessors.push_back(ViaNode);

  verifyNodes();
}

/// Collect all outgoing edges from nodes in \p FromList to \p ToBeyond or
/// later in the order, and reroute them via \p ViaNode.
void ReconvergeCFGHelper::rerouteEdgesBeyond(ArrayRef<WaveNode *> FromList,
                                             WaveNode *ToBeyond,
                                             WaveNode *ViaNode) {
  // In current use, we can assume that ViaNode is not connect to anything.
  for (WaveNode *FromNode : FromList) {
    assert(!is_contained(FromNode->Successors, ViaNode));

    auto RerouteBegin =
        llvm::partition(FromNode->Successors, [&](WaveNode *Succ) {
          assert(Succ->OrderIndex != ToBeyond->OrderIndex || Succ == ToBeyond);
          return Succ->OrderIndex < ToBeyond->OrderIndex;
        });

    for (WaveNode *Succ :
         llvm::make_range(RerouteBegin, FromNode->Successors.end())) {
      auto I = llvm::find(Succ->Predecessors, FromNode);
      assert(I != Succ->Predecessors.end());
      *I = Succ->Predecessors.back();
      Succ->Predecessors.pop_back();

      if (llvm::find(ViaNode->Successors, Succ) == ViaNode->Successors.end()) {
        ViaNode->Successors.push_back(Succ);

        assert(!is_contained(Succ->Predecessors, ViaNode));
        Succ->Predecessors.push_back(ViaNode);
      }

      rerouteLane(FromNode, Succ, ViaNode);
    }

    FromNode->Successors.erase(RerouteBegin, FromNode->Successors.end());
    FromNode->Successors.push_back(ViaNode);

    assert(!is_contained(ViaNode->Predecessors, FromNode));
    ViaNode->Predecessors.push_back(FromNode);
  }

  verifyNodes();
}

/// Helper for rerouting methods: update the WaveNode::LaneSuccessors and
/// LanePredecessors vectors based on a rerouting.
void ReconvergeCFGHelper::rerouteLane(WaveNode *FromNode, WaveNode *ToNode,
                                      WaveNode *ViaNode) {
  for (LaneEdge &FromLaneSucc : FromNode->LaneSuccessors) {
    if (FromLaneSucc.Wave != ToNode)
      continue;

    FromLaneSucc.Wave = ViaNode;

    bool found = false;
    for (const LaneEdge &ViaLaneSucc : ViaNode->LaneSuccessors) {
      if (ViaLaneSucc.Lane != FromLaneSucc.Lane)
        continue;
      assert(ViaLaneSucc.Wave == ToNode);
      found = true;
      break;
    }
    if (!found)
      ViaNode->LaneSuccessors.emplace_back(FromLaneSucc.Lane, ToNode);
  }

  for (LaneEdge &ToLanePred : ToNode->LanePredecessors) {
    if (ToLanePred.Wave != FromNode)
      continue;

    auto PredIt =
        llvm::find_if(ViaNode->LanePredecessors, [=](const LaneEdge &LanePred) {
          return LanePred.Lane == ToLanePred.Lane;
        });
    if (PredIt != ViaNode->LanePredecessors.end()) {
      assert(PredIt->Wave == FromNode);
    } else {
      ViaNode->LanePredecessors.emplace_back(ToLanePred.Lane, FromNode);
    }
    ToLanePred.Wave = ViaNode;
  }
}

/// Print all WaveNodes to the given stream.
void ReconvergeCFGHelper::printNodes(raw_ostream &Out) {
  auto printNode = [&](WaveNode *Node) {
    Out << "  " << Node->printableName() << " (#" << Node->OrderIndex << ")";

    if (!Node->Successors.empty()) {
      Out << " ->";
      for (WaveNode *Succ : Node->Successors) {
        Out << ' ' << Succ->printableName();

        bool Printed = false;
        for (const LaneEdge &LaneSucc : Node->LaneSuccessors) {
          if (LaneSucc.Wave != Succ)
            continue;

          if (!Printed) {
            Out << '(';
            Printed = true;
          } else {
            Out << ',';
          }

          if (LaneSucc.Lane == Succ)
            Out << '*';
          else
            Out << LaneSucc.Lane->printableName();
        }
        if (Printed)
          Out << ')';
      }
    }

    if (Node->LatestPostDom != Node)
      Out << " [LatestPostDom: " << Node->LatestPostDom->printableName() << ']';

    if (Node->IsDivergent)
      Out << " [divergent]";
    if (Node->IsSecondary)
      Out << " [secondary]";

    Out << '\n';
  };

  for (const auto &NodePtr : NextNodes)
    printNode(NodePtr.get());
  for (const auto &NodePtr : Nodes) {
    if (NodePtr)
      printNode(NodePtr.get());
  }
  Out << '\n';
}

/// Dump all WaveNodes to debug out.
LLVM_ATTRIBUTE_UNUSED
void ReconvergeCFGHelper::dumpNodes() {
  printNodes(dbgs());

  verifyNodes();
}

/// Verify some basic invariants on WaveNodes.
void ReconvergeCFGHelper::verifyNodes() {
  DenseSet<WaveNode *> SeenNodes;

  auto verifyNode = [&](WaveNode *Node) {
    LLVM_ATTRIBUTE_UNUSED
    bool Inserted = SeenNodes.insert(Node).second;
    assert(Inserted);
  };

  for (const auto &NodePtr : NextNodes)
    verifyNode(NodePtr.get());
  for (const auto &NodePtr : Nodes) {
    if (NodePtr)
      verifyNode(NodePtr.get());
  }

  DenseSet<WaveNode *> LanePreds;
  DenseSet<WaveNode *> LaneSuccs;

  // FIXME: these only contain assertions so are "unused variables" in release
  // build
  for (WaveNode *Node : SeenNodes) {
    for (WaveNode *Pred : Node->Predecessors) {
      assert(SeenNodes.count(Pred));
      assert(is_contained(Pred->Successors, Node));
    }
    for (WaveNode *Succ : Node->Successors) {
      assert(SeenNodes.count(Succ));
      assert(is_contained(Succ->Predecessors, Node));

      assert(llvm::any_of(Node->LaneSuccessors, [&](const auto &LaneSucc) {
        return LaneSucc.Wave == Succ;
      }));
    }

    for (const LaneEdge &LanePred : Node->LanePredecessors) {
      assert(SeenNodes.count(LanePred.Lane));
      assert(is_contained(Node->Predecessors, LanePred.Wave));
      bool Inserted = LanePreds.insert(LanePred.Lane).second;
      assert(Inserted);

      if (LanePred.Lane != LanePred.Wave) {
        assert(LanePred.Wave->FlowNum != 0);
        assert(!is_contained(Node->Predecessors, LanePred.Lane));
        assert(llvm::any_of(
            LanePred.Wave->LanePredecessors,
            [&](const auto &next) { return next.Lane == LanePred.Lane; }));
      }
    }
    LanePreds.clear();

    for (const LaneEdge &LaneSucc : Node->LaneSuccessors) {
      assert(SeenNodes.count(LaneSucc.Lane));
      assert(is_contained(Node->Successors, LaneSucc.Wave));
      bool Inserted = LaneSuccs.insert(LaneSucc.Lane).second;
      assert(Inserted);

      if (LaneSucc.Lane != LaneSucc.Wave) {
        assert(LaneSucc.Wave->FlowNum != 0);
        assert(!is_contained(Node->Successors, LaneSucc.Lane));
        assert(
            llvm::any_of(LaneSucc.Wave->LaneSuccessors, [&](const auto &next) {
              return next.Lane == LaneSucc.Lane;
            }));
      }
    }
    LaneSuccs.clear();
  }
}

namespace {

/// Helper class for reconstructing SSA form after transforming the MachineIR
/// CFG into the wave CFG.
class SSAReconstructor {
private:
  struct PhiIncoming {
    WaveNode *Node = nullptr;
    Register Reg;
    SmallVector<std::pair<WaveNode *, Register>, 4> Incoming;
  };

  MachineFunction &Function;
  MachineRegisterInfo &MRI;
  MachineDominatorTree &DomTree;
  ReconvergeCFGHelper &ReconvergeCfg;
  const SIInstrInfo &TII;
  const SIRegisterInfo &TRI;

public:
  SSAReconstructor(MachineFunction &function, MachineDominatorTree &domTree,
                   ReconvergeCFGHelper &ReconvergeCfg)
      : Function(function), MRI(function.getRegInfo()), DomTree(domTree),
        ReconvergeCfg(ReconvergeCfg),
        TII(*function.getSubtarget<GCNSubtarget>().getInstrInfo()),
        TRI(*static_cast<const SIRegisterInfo *>(MRI.getTargetRegisterInfo())) {
  }

  void run();
};

} // anonymous namespace

/// Run the SSA reconstruction algorithm.
//
// TODO: The currently implemented algorithm implicitly over-estimates the
//       liveness of values because it does not fully take lane predecessors /
//       successors into account. It's unclear whether we want to more
//       aggressively insert PHIs here, or preserve the thread-level CFG
//       in another way.
void SSAReconstructor::run() {
  // Step 1: Fix up original PHIs' predecessor blocks.
  std::vector<MachineInstr *> OriginalPhis;

  for (MachineBasicBlock &Block : Function) {
    bool FirstPhi = true;
    for (MachineInstr &Phi : Block.phis()) {
      if (FirstPhi) {
        // Compile-time optimization: if the block's predecessors haven't
        // changed, we don't have to do anything for the phis in this block.
        assert((Phi.getNumOperands() % 2) == 1);
        unsigned OrigNumPreds = Phi.getNumOperands() / 2;
        if (OrigNumPreds == Block.pred_size()) {
          bool FoundAll = true;
          for (unsigned OpIdx = 1; OpIdx < Phi.getNumOperands(); OpIdx += 2) {
            MachineBasicBlock *origPred = Phi.getOperand(OpIdx + 1).getMBB();
            if (!llvm::is_contained(Block.predecessors(), origPred)) {
              FoundAll = false;
              break;
            }
          }

          if (FoundAll)
            break;
        }

        FirstPhi = false;
      }

      OriginalPhis.push_back(&Phi);
    }
  }

#ifndef NDEBUG
  DenseSet<WaveNode *> PhiSeenNodes;
#endif
  SmallVector<PhiIncoming, 4> PhiWorklist;
  for (MachineInstr *OriginalPhi : OriginalPhis) {
    PhiIncoming Current;
    Current.Node = ReconvergeCfg.nodeForBlock(OriginalPhi->getParent());
    Current.Reg = OriginalPhi->getOperand(0).getReg();

    for (unsigned OpIdx = 1; OpIdx < OriginalPhi->getNumOperands();
         OpIdx += 2) {
      Register ValueReg = OriginalPhi->getOperand(OpIdx).getReg();
      MachineBasicBlock *OrigPred = OriginalPhi->getOperand(OpIdx + 1).getMBB();
      Current.Incoming.emplace_back(ReconvergeCfg.nodeForBlock(OrigPred),
                                    ValueReg);
    }

    const TargetRegisterClass *RC = MRI.getRegClass(Current.Reg);
    bool IsVector = TRI.isDivergentRegClass(RC);
    MachineInstr *Phi = OriginalPhi;
    for (;;) {
#ifndef NDEBUG
      assert(PhiSeenNodes.insert(Current.Node).second);
#endif

      // Skip nodes that are trivially dominated. We still end up inserting
      // PHIs that may seem unnecessary. Consider:
      //
      //     A   B
      //      \ /
      //       X
      //       |\
      //       | C
      //       |/
      //       Y
      //       |\
      //       | D
      //      ...
      //
      // If D has A and B, but not C, as original (lane) predecessors, we will
      // remove the original phi node from D and insert new ones in Y and X.
      //
      // The phi in Y will have an undef incoming from C. This phi is not
      // strictly required, as the (required) phi in X would be sufficient.
      //
      // As a consequence, the value is (arguably correctly) not considered
      // live during C. This has advantages and disadvantages:
      //  - Register allocation during C is less conservative, which is good
      //    when the value is in a VGPR anyway.
      //  - Register allocation can _incorrectly_ clobber the value during C
      //    when it is in an SGPR. We solve this by moving values to VGPRs
      //    whenever secondary phis are created.
      //
      // Note that the latter is correct even when the original phi is moved,
      // as in that case, the default liveness analysis prevents the value from
      // being clobbered between the "final" phi and its uses.
      while (Current.Node->Predecessors.size() == 1) {
        if (Phi) {
          assert(Phi == OriginalPhi);
          Phi->eraseFromParent();
          Phi = nullptr;
        }
        Current.Node = Current.Node->Predecessors[0];
      }

      if (!Phi) {
        Phi = BuildMI(*Current.Node->Block, Current.Node->Block->begin(), {},
                      TII.get(AMDGPU::PHI), Current.Reg);
      }

      unsigned OpIdx = 1;
      for (WaveNode *Pred : Current.Node->Predecessors) {
        PhiIncoming PredIncoming;
        Register Reg;
        bool RegConflict = false;

        for (const LaneEdge &LanePred : Current.Node->LanePredecessors) {
          if (LanePred.Wave != Pred)
            continue;

          auto IncomingIt =
              llvm::find_if(Current.Incoming, [=](const auto &Incoming) {
                return Incoming.first == LanePred.Lane;
              });
          if (IncomingIt != Current.Incoming.end()) {
            PredIncoming.Incoming.emplace_back(*IncomingIt);
            if (!Reg)
              Reg = IncomingIt->second;
            else if (Reg != IncomingIt->second)
              RegConflict = true;
          }
        }

        if (RegConflict) {
          // Multiple conflicting lane-predecessors arriving from the same
          // wave-predecessor. Need to insert a phi in the predecessor block
          // or its dominator.
          PredIncoming.Node = Pred;
          Reg = PredIncoming.Reg = MRI.createVirtualRegister(RC);
          PhiWorklist.emplace_back(std::move(PredIncoming));
        } else if (!Reg) {
          // No incoming value for this wave predecessor, fill in with undef.
          // This shouldn't happen for the initial phi, but it can happen
          // when inserting secondary phis into flow blocks.
          Reg = MRI.createVirtualRegister(RC);
          BuildMI(*Pred->Block, Pred->Block->getFirstNonPHI(), {},
                  TII.get(AMDGPU::IMPLICIT_DEF), Reg);
        }

        if (OpIdx == Phi->getNumOperands()) {
          Phi->addOperand(Function, MachineOperand::CreateReg(Reg, false));
          Phi->addOperand(Function, MachineOperand::CreateMBB(Pred->Block));
        } else {
          Phi->getOperand(OpIdx).setReg(Reg);
          Phi->getOperand(OpIdx + 1).setMBB(Pred->Block);
        }

        OpIdx += 2;
      }

      while (OpIdx < Phi->getNumOperands())
        Phi->removeOperand(Phi->getNumOperands() - 1);

      if (PhiWorklist.empty())
        break;

      if (!IsVector) {
        // Inserting a secondary phi. We must move values to vector registers
        // to prevent incorrect clobbers. There is only one relevant phi so
        // far, fix up its register class. Rely on later passes to legalize the
        // instructions that are using the destination register.
        RC = TRI.getEquivalentVGPRClass(RC);
        IsVector = true;

        MRI.setRegClass(Current.Reg, RC);
        OpIdx = 1;
        for (; OpIdx < Phi->getNumOperands(); OpIdx += 2) {
          Register ValueReg = Phi->getOperand(OpIdx).getReg();
          MachineInstr *DefInstr = MRI.getVRegDef(ValueReg);
          if (!DefInstr || DefInstr->getOpcode() == AMDGPU::IMPLICIT_DEF)
            MRI.setRegClass(ValueReg, RC);
        }
      }

      // Get the next secondary phi description.
      Current = PhiWorklist.pop_back_val();
      assert(Current.Incoming.size() >= 2);
      Phi = nullptr;
    }

#ifndef NDEBUG
    PhiSeenNodes.clear();
#endif
  }

  // Step 2: Re-establish dominance relation from defs to uses.
  unsigned NumVirtRegs = MRI.getNumVirtRegs();
  SmallVector<std::pair<MachineOperand *, MachineBasicBlock *>, 8> Rewrites;
  MachineSSAUpdater Updater(Function);

  for (unsigned VirtRegIndex = 0; VirtRegIndex < NumVirtRegs; ++VirtRegIndex) {
    Register Reg = Register::index2VirtReg(VirtRegIndex);
    MachineInstr *DefInstr = MRI.getVRegDef(Reg);
    if (!DefInstr)
      continue;

    MachineBasicBlock *DefBlock = DefInstr->getParent();
    MachineDomTreeNode *DefDomNode = DomTree.getNode(DefBlock);

    for (MachineOperand &Use : MRI.use_operands(Reg)) {
      MachineInstr *UseInstr = Use.getParent();
      MachineBasicBlock *UseBlock;
      if (UseInstr->isPHI()) {
        // Uses from PHIs are considered to occur inside the corresponding
        // predecessor basic block. Note that this is the non-adjusted
        // (original, pre-wave-transform) predecessor block.
        unsigned OpIdx = UseInstr->getOperandNo(&Use);
        UseBlock = UseInstr->getOperand(OpIdx + 1).getMBB();
      } else {
        UseBlock = UseInstr->getParent();
      }

      if (UseBlock != DefBlock &&
          !DomTree.dominates(DefDomNode, DomTree.getNode(UseBlock))) {
        Rewrites.emplace_back(&Use, UseBlock);
      }
    }

    if (!Rewrites.empty()) {
      Updater.Initialize(Reg);
      Updater.AddAvailableValue(DefBlock, Reg);

      for (const auto &rewrite : Rewrites)
        rewrite.first->setReg(Updater.GetValueAtEndOfBlock(rewrite.second));
      Rewrites.clear();
    }
  }
}

namespace {

/// Helper class for rewriting control-flow instruction after translation into
/// a wave CFG.
class ControlFlowRewriter {
private:
  /// For a given original target node, record information about where lanes
  /// for that target can come from.
  struct LaneOriginInfo {
    /// Node (original or flow) from which lanes can originate.
    WaveNode *Node;

    /// Condition under which lanes originate from that node (can be null,
    /// in which case EXEC / all active lanes should be used).
    Register CondReg;

    /// Whether the condition should be inverted.
    bool InvertCondition = false;

    explicit LaneOriginInfo(WaveNode *Node, Register CondReg = {},
                            bool InvertCondition = false)
        : Node(Node), CondReg(CondReg), InvertCondition(InvertCondition) {}
  };

  struct CFGNodeInfo {
    WaveNode *Node;

    bool OrigExit = false;

    /// Branch condition, if the block originally had a conditional branch.
    Register OrigCondition;

    /// Branch target if \ref condition is true.
    WaveNode *OrigSuccCond = nullptr;

    /// Final branch target, i.e. if there was no conditional branch or if
    /// \ref condition is false.
    WaveNode *OrigSuccFinal = nullptr;

    /// Information about nodes from which lanes targeting this node can
    /// originate.
    SmallVector<LaneOriginInfo, 4> origins;

    /// (origin, divergent) pairs of origin nodes that have a branch towards
    /// this node with the property that immediately after the corresponding
    /// branch, all active lanes target this node.
    SmallVector<PointerIntPair<WaveNode *, 1, bool>, 4> OriginBranch;

    Register PrimarySuccessorExec;

    explicit CFGNodeInfo(WaveNode *Node) : Node(Node) {}
  };

  /// Information required to synthesize divergent terminators with a common
  /// primary successor.
  struct DivergentTargetInfo {
    /// Nodes containing divergent terminators whose primary successor targets
    /// the node in question.
    SmallVector<WaveNode *, 2> BranchNodes;

    /// Flow nodes that are targeted by one or more of the terminators in
    /// \ref BranchNodes, but are themselves only intermediate steps to the
    /// targets in question.
    DenseSet<WaveNode *> FlowNodes;
  };

  MachineFunction &Function;
  ReconvergeCFGHelper &ReconvergeCfg;
  GCNLaneMaskUtils LMU;
  MachineRegisterInfo &MRI;
  const SIInstrInfo &TII;

  DenseMap<WaveNode *, CFGNodeInfo> NodeInfo;
  std::vector<WaveNode *> NodeOrder;

public:
  ControlFlowRewriter(MachineFunction &function,
                      ReconvergeCFGHelper &ReconvergeCfg)
      : Function(function), ReconvergeCfg(ReconvergeCfg), LMU(function),
        MRI(function.getRegInfo()),
        TII(*function.getSubtarget<GCNSubtarget>().getInstrInfo()) {}

  void prepareWaveCfg();
  void rewrite();
};

} // anonymous namespace

/// Collect information about original terminator instructions and prepare
/// the wave-level CFG without changing the MIR representation yet.
void ControlFlowRewriter::prepareWaveCfg() {
  // Pre-initialize the block-info map with all blocks, so that we can rely
  // on stable references for the next step.
  for (WaveNode *Node : ReconvergeCfg.nodes()) {
    if (NodeInfo.try_emplace(Node, Node).second)
      NodeOrder.push_back(Node);
  }

  // Step 1: Analyze original successors and branch conditions and record them
  // as well as related info that we will need to generate divergent branches.
  //
  // uniformCandidateEdges maps (ToNode, viaFlowNode) -> FromNodes for edges
  // _fro a node with uniform conditional terminator _to_ an original
  // predecessor _via_ a flow node with multiple successors.
  MapVector<std::pair<WaveNode *, WaveNode *>, SmallVector<WaveNode *, 2>>
      UniformSplitEdges;

  for (WaveNode *Node : ReconvergeCfg.nodes()) {
    CFGNodeInfo &Info = NodeInfo.find(Node)->second;

    if (Node->IsDivergent && Node->Successors.size() >= 2) {
      assert(Node->Successors.size() == 2);
      WaveNode *primaryWave = Node->Successors[0];
      WaveNode *primaryLane = nullptr;
      for (const LaneEdge &LaneSucc : Node->LaneSuccessors) {
        if (LaneSucc.Wave == primaryWave) {
          assert(!primaryLane);
          primaryLane = LaneSucc.Lane;
#ifdef NDEBUG
          // early-out when assertions are disabled: we don't check for
          // uniqueness in that case
          break;
#endif
        }
      }
      assert(primaryLane);

      NodeInfo.find(primaryLane)->second.OriginBranch.emplace_back(Node, true);
    }

    if (!Node->Block)
      continue;

    // Analyze original terminators.
    for (MachineInstr &Terminator : Node->Block->terminators()) {
      unsigned Opcode = Terminator.getOpcode();

      assert(!Info.OrigSuccFinal);
      if (Opcode == AMDGPU::SI_BRCOND) {
        assert(!Info.OrigCondition);
        Info.OrigCondition = Terminator.getOperand(0).getReg();
        Info.OrigSuccCond =
            ReconvergeCfg.nodeForBlock(Terminator.getOperand(1).getMBB());
      } else if (Opcode == AMDGPU::S_BRANCH) {
        Info.OrigSuccFinal =
            ReconvergeCfg.nodeForBlock(Terminator.getOperand(0).getMBB());
      } else {
        assert(!Info.OrigCondition);
        assert(Opcode == AMDGPU::S_ENDPGM || Opcode == AMDGPU::SI_RETURN ||
               Opcode == AMDGPU::SI_RETURN_TO_EPILOG);

        Info.OrigExit = true;
      }
    }

    if (!Info.OrigSuccFinal && !Info.OrigExit) {
      // Fall-through in the original code.
      auto BlockIt = Node->Block->getIterator();
      ++BlockIt;
      assert(BlockIt != Function.end());
      assert(is_contained(Node->Block->successors(), &*BlockIt));
      Info.OrigSuccFinal = ReconvergeCfg.nodeForBlock(&*BlockIt);
    }

    assert(Info.OrigExit || Node->FlowNum != 0 || Info.OrigSuccFinal);
    assert(!Info.OrigExit || !Info.OrigSuccFinal);
    assert(!Info.OrigSuccCond || Info.OrigSuccFinal);
    assert(Info.OrigExit == Node->Successors.empty() &&
           "TODO: exit unification");

    // Record information for reconstructing lane masks.
    if (!Info.OrigSuccCond) {
      if (Info.OrigSuccFinal) {
        NodeInfo.find(Info.OrigSuccFinal)->second.origins.emplace_back(Node);
      }
    } else {
      if (!Node->IsDivergent && Node->Successors.size() >= 2) {
        assert(Node->Successors.size() == 2);

        NodeInfo.find(Info.OrigSuccCond)
            ->second.OriginBranch.emplace_back(Node, false);
        NodeInfo.find(Info.OrigSuccFinal)
            ->second.OriginBranch.emplace_back(Node, false);

        for (const LaneEdge &LaneEdge : Node->LaneSuccessors) {
          assert(LaneEdge.Lane == Info.OrigSuccCond ||
                 LaneEdge.Lane == Info.OrigSuccFinal);

          if (LaneEdge.Lane == LaneEdge.Wave) {
            // If we directly branch to the Lane target, this edge will never
            // contribute to a divergent branch.
            continue;
          }

          // If the original edge was redirected through flow nodes, we are
          // likely going through a divergent branch at some point.
          if (LaneEdge.Wave->LaneSuccessors.size() > 1) {
            UniformSplitEdges[std::make_pair(LaneEdge.Lane, LaneEdge.Wave)]
                .emplace_back(Node);
          } else {
            CFGNodeInfo &succInfo = NodeInfo.find(LaneEdge.Lane)->second;
            if (!llvm::any_of(succInfo.origins,
                              [&](const LaneOriginInfo &origin) {
                                return origin.Node == LaneEdge.Wave;
                              }))
              succInfo.origins.emplace_back(LaneEdge.Wave);
          }
        }
      } else {
        NodeInfo.find(Info.OrigSuccCond)
            ->second.origins.emplace_back(Node, Info.OrigCondition);
        NodeInfo.find(Info.OrigSuccFinal)
            ->second.origins.emplace_back(Node, Info.OrigCondition, true);
      }
    }
  }

  // Step 2: Split certain critical edges after uniform branches.
  //
  // A uniform conditional branch can end up leading into a flow node with
  // multiple (lane) successors, which means the original target of the
  // conditional branch is ultimately reached via a divergent branch for which
  // we need to establish a corresponding lane mask. In this example, A has a
  // uniform branch to C that got rerouted through flow nodes X and Y for some
  // reason (e.g. part of loop control flow handling):
  //
  //     |
  //     A
  //    / \  ...
  //   ... \ /
  //        X
  //        |\
  //        | B
  //        |/
  //        Y
  //        |\
  //        | \
  //       ... C
  //           |
  //
  // In Y, we need a lane mask for the branch to C that takes into account
  // lanes from A as well as lanes from some potential other predecessors.
  //
  // To facilitate the construction of these lane masks, we split the edge from
  // A to X.
  for (const auto &UniformSplit : UniformSplitEdges) {
    WaveNode *FlowNode = ReconvergeCfg.rerouteViaNewNode(
        UniformSplit.second, UniformSplit.first.second);
    if (NodeInfo.try_emplace(FlowNode, FlowNode).second)
      NodeOrder.push_back(FlowNode);
    NodeInfo.find(UniformSplit.first.first)
        ->second.origins.emplace_back(FlowNode);
  }
}

/// Replace all original terminator instructions by the terminators for
/// establishing wave-level control flow and insert instructions for EXEC mask
/// manipulation.
void ControlFlowRewriter::rewrite() {
  GCNLaneMaskAnalysis LMA(Function);

  Register RegAllOnes;
  auto getAllOnes = [&]() {
    if (!RegAllOnes) {
      RegAllOnes = LMU.createLaneMaskReg();
      BuildMI(Function.front(), Function.front().getFirstTerminator(), {},
              TII.get(LMU.consts().OpMov), RegAllOnes)
          .addImm(-1);
    }
    return RegAllOnes;
  };

  // Step 1: Remove old terminators and insert new ones for uniform branches.
  for (WaveNode *Node : NodeOrder) {
    CFGNodeInfo &Info = NodeInfo.find(Node)->second;

    if (!Info.OrigExit) {
      // Remove original terminators.
      while (!Node->Block->empty() && Node->Block->back().isTerminator())
        Node->Block->back().eraseFromParent();
    }

    if (Node->Successors.size() == 0)
      continue;

    assert(!Info.OrigExit);

    if (Node->Successors.size() == 1) {
      BuildMI(*Node->Block, Node->Block->end(), {}, TII.get(AMDGPU::S_BRANCH))
          .addMBB(Node->Successors[0]->Block);
      continue;
    }

    assert(Node->Successors.size() == 2);

    if (!Node->IsDivergent) {
      // Uniform block with two successors: we must have had two original
      // successors, and one of the current successors leads to the original
      // conditional successor.
      assert(Info.OrigCondition);

      auto LaneSucc =
          llvm::find_if(Node->LaneSuccessors, [=](const auto &succ) {
            return succ.Lane == Info.OrigSuccCond;
          });
      assert(LaneSucc != Node->LaneSuccessors.end());

      unsigned Opcode;

      if (Info.OrigCondition == AMDGPU::SCC) {
        Opcode = AMDGPU::S_CBRANCH_SCC1;
      } else {
        Register CondReg = Info.OrigCondition;
        if (!LMA.isSubsetOfExec(CondReg, *Node->Block)) {
          CondReg = LMU.createLaneMaskReg();
          BuildMI(*Node->Block, Node->Block->end(), {},
                  TII.get(LMU.consts().OpAnd), CondReg)
              .addReg(LMU.consts().RegExec)
              .addReg(Info.OrigCondition);
        }
        BuildMI(*Node->Block, Node->Block->end(), {}, TII.get(AMDGPU::COPY),
                LMU.consts().RegVcc)
            .addReg(CondReg);

        Opcode = AMDGPU::S_CBRANCH_VCCNZ;
      }

      BuildMI(*Node->Block, Node->Block->end(), {}, TII.get(Opcode))
          .addMBB(LaneSucc->Wave->Block);

      // The _other_ successor may be a flow block instead of an original
      // successor.
      WaveNode *Other;
      if (Node->Successors[0] == LaneSucc->Wave)
        Other = Node->Successors[1];
      else
        Other = Node->Successors[0];
      BuildMI(*Node->Block, Node->Block->end(), {}, TII.get(AMDGPU::S_BRANCH))
          .addMBB(Other->Block);
    }
  }

  // Step 2: Insert lane masks and new terminators for divergent nodes.
  //
  // RegMap maps (block, register) -> (masked, inverted).
  DenseMap<std::pair<MachineBasicBlock *, Register>,
           std::pair<Register, Register>>
      RegMap;
  GCNLaneMaskUpdater Updater(Function);
  Updater.setLaneMaskAnalysis(&LMA);
  Updater.setAccumulating(true);

  for (WaveNode *LaneTarget : NodeOrder) {
    CFGNodeInfo &LaneTargetInfo = NodeInfo.find(LaneTarget)->second;

    if (!llvm::any_of(
            LaneTargetInfo.OriginBranch,
            [](const auto &OriginBranch) { return OriginBranch.getInt(); })) {
      // No divergent branches towards this node, nothing to be done.
      continue;
    }

    LLVM_DEBUG(dbgs() << "\nDivergent branches for "
                      << LaneTarget->printableName() << '\n');

    // Step 2.1: Add conditions branching to LaneTarget to the Lane mask
    // Updater.
    // FIXME: we are creating a register here only to initialize the updater
    Updater.init(LMU.createLaneMaskReg());
    Updater.addReset(*LaneTarget->Block, GCNLaneMaskUpdater::ResetInMiddle);
    for (const auto &NodeDivergentPair : LaneTargetInfo.OriginBranch) {
      Updater.addReset(*NodeDivergentPair.getPointer()->Block,
                       GCNLaneMaskUpdater::ResetAtEnd);
    }

    for (const LaneOriginInfo &LaneOrigin : LaneTargetInfo.origins) {
      Register CondReg;

      if (!LaneOrigin.CondReg) {
        assert(!LaneOrigin.InvertCondition);
        CondReg = getAllOnes();
      } else if (LaneOrigin.CondReg == AMDGPU::SCC) {
        assert(LaneOrigin.Node->Successors.size() == 1);

        // Subtle: We rely here on the fact that:
        //  1. No other instructions have been inserted at the end of the
        //     basic block since step 1, when the terminators were deleted --
        //     otherwise, SCC could have been clobbered.
        //  2. Later steps only insert instructions between the cselect here
        //     and the terminators, where SCC no longer matters.
        //
        // PHI nodes may have been inserted, but those are at the beginning
        // of the block.
        //
        // cond = SCC ? EXEC : 0; (or reverse)
        CondReg = LMU.createLaneMaskReg();
        if (!LaneOrigin.InvertCondition) {
          BuildMI(*LaneOrigin.Node->Block,
                  LaneOrigin.Node->Block->getFirstTerminator(), {},
                  TII.get(LMU.consts().OpCSelect), CondReg)
              .addReg(LMU.consts().RegExec)
              .addImm(0);
        } else {
          BuildMI(*LaneOrigin.Node->Block,
                  LaneOrigin.Node->Block->getFirstTerminator(), {},
                  TII.get(LMU.consts().OpCSelect), CondReg)
              .addImm(0)
              .addReg(LMU.consts().RegExec);
        }
      } else {
        CondReg = LaneOrigin.CondReg;
        if (!LMA.isSubsetOfExec(LaneOrigin.CondReg, *LaneOrigin.Node->Block)) {
          Register Prev = CondReg;
          CondReg = LMU.createLaneMaskReg();
          BuildMI(*LaneOrigin.Node->Block,
                  LaneOrigin.Node->Block->getFirstTerminator(), {},
                  TII.get(LMU.consts().OpAnd), CondReg)
              .addReg(LMU.consts().RegExec)
              .addReg(Prev);

          RegMap[std::make_pair(LaneOrigin.Node->Block, LaneOrigin.CondReg)]
              .first = CondReg;
        }

        if (LaneOrigin.InvertCondition) {
          // CondReg = EXEC ^ origCond;
          //
          // Ideally we would XOR with EXEC instead of -1 to avoid redundant
          // AND with EXEC later, but LICM can move the condition in ways which
          // violate such an optimisation.  Discard and demote operations
          // can also modify the value of EXEC requiring an AND.
          // TODO: We rely on later passes to clean up,
          // e.g. folding the XOR into the original V_CMP.
          Register Prev = CondReg;
          CondReg = LMU.createLaneMaskReg();
          BuildMI(*LaneOrigin.Node->Block,
                  LaneOrigin.Node->Block->getFirstTerminator(), {},
                  TII.get(LMU.consts().OpXor), CondReg)
              .addReg(LaneOrigin.CondReg)
              .addImm(-1);

          RegMap[std::make_pair(LaneOrigin.Node->Block, LaneOrigin.CondReg)]
              .second = CondReg;
          RegMap.try_emplace(std::make_pair(LaneOrigin.Node->Block, CondReg),
                             CondReg, Prev);
        }
      }

      LLVM_DEBUG(
          dbgs() << "  available @ " << LaneOrigin.Node->printableName() << ": "
                 << printReg(CondReg, MRI.getTargetRegisterInfo(), 0, &MRI)
                 << '\n');

      Updater.addAvailable(*LaneOrigin.Node->Block, CondReg);
    }

    // Step 2.2: Synthesize EXEC updates and branch instructions.
    for (const auto &NodeDivergentPair : LaneTargetInfo.OriginBranch) {
      if (!NodeDivergentPair.getInt())
        continue; // not a divergent branch

      WaveNode *OriginNode = NodeDivergentPair.getPointer();
      CFGNodeInfo &OriginCFGNodeInfo = NodeInfo.find(OriginNode)->second;
      OriginCFGNodeInfo.PrimarySuccessorExec =
          Updater.getValueAfterMerge(*OriginNode->Block);

      LLVM_DEBUG(dbgs() << "  " << OriginNode->printableName() << " -> "
                        << OriginNode->Successors[0]->printableName()
                        << " with EXEC="
                        << printReg(OriginCFGNodeInfo.PrimarySuccessorExec,
                                    MRI.getTargetRegisterInfo(), 0, &MRI)
                        << '\n');

      BuildMI(*OriginNode->Block, OriginNode->Block->end(), {},
              TII.get(LMU.consts().OpMovTerm), LMU.consts().RegExec)
          .addReg(OriginCFGNodeInfo.PrimarySuccessorExec);
      BuildMI(*OriginNode->Block, OriginNode->Block->end(), {},
              TII.get(AMDGPU::SI_WAVE_CF_EDGE));
      BuildMI(*OriginNode->Block, OriginNode->Block->end(), {},
              TII.get(AMDGPU::S_CBRANCH_EXECZ))
          .addMBB(OriginNode->Successors[1]->Block);
      BuildMI(*OriginNode->Block, OriginNode->Block->end(), {},
              TII.get(AMDGPU::S_BRANCH))
          .addMBB(OriginNode->Successors[0]->Block);
    }

    LLVM_DEBUG(Function.dump());
  }

  // Step 3: Insert rejoin masks.
  for (WaveNode *Secondary : ReconvergeCfg.nodes()) {
    if (!Secondary->IsSecondary)
      continue;

    LLVM_DEBUG(dbgs() << "\nRejoin @ " << Secondary->printableName() << '\n');

    // FIXME: we are creating a register here only to initialize the updater
    Updater.init(LMU.createLaneMaskReg());
    Updater.addReset(*Secondary->Block, GCNLaneMaskUpdater::ResetInMiddle);

    for (WaveNode *Pred : Secondary->Predecessors) {
      if (!Pred->IsDivergent || Pred->Successors.size() == 1)
        continue;

      CFGNodeInfo &PredInfo = NodeInfo.find(Pred)->second;
      Register PrimaryExec = PredInfo.PrimarySuccessorExec;

      MachineInstr *PrimaryExecDef;
      for (;;) {
        PrimaryExecDef = MRI.getVRegDef(PrimaryExec);
        if (PrimaryExecDef->getOpcode() != AMDGPU::COPY)
          break;
        PrimaryExec = PrimaryExecDef->getOperand(1).getReg();
      }

      // Rejoin = EXEC ^ PrimaryExec
      //
      // Fold immediately if PrimaryExec was obtained via XOR as well.
      Register Rejoin;

      if (PrimaryExecDef->getParent() == Pred->Block &&
          PrimaryExecDef->getOpcode() == LMU.consts().OpXor &&
          PrimaryExecDef->getOperand(1).isReg() &&
          PrimaryExecDef->getOperand(2).isReg()) {
        if (PrimaryExecDef->getOperand(1).getReg() == LMU.consts().RegExec)
          Rejoin = PrimaryExecDef->getOperand(2).getReg();
        else if (PrimaryExecDef->getOperand(2).getReg() == LMU.consts().RegExec)
          Rejoin = PrimaryExecDef->getOperand(1).getReg();
      }

      if (!Rejoin) {
        // Try to find a previously generated XOR (or merely masked) value
        // for reuse.
        auto MapIt = RegMap.find(std::make_pair(Pred->Block, PrimaryExec));
        if (MapIt != RegMap.end()) {
          Rejoin = MapIt->second.second;
          if (!Rejoin)
            PrimaryExec = MapIt->second.first;
        }
      }

      if (!Rejoin) {
        Rejoin = LMU.createLaneMaskReg();
        BuildMI(*Pred->Block, Pred->Block->getFirstTerminator(), {},
                TII.get(LMU.consts().OpXor), Rejoin)
            .addReg(LMU.consts().RegExec)
            .addReg(PrimaryExec);
      }

      LLVM_DEBUG(
          dbgs() << "  available @ " << Pred->printableName() << ": "
                 << printReg(Rejoin, MRI.getTargetRegisterInfo(), 0, &MRI)
                 << '\n');

      Updater.addAvailable(*Pred->Block, Rejoin);
    }

    Register Rejoin = Updater.getValueInMiddleOfBlock(*Secondary->Block);
    BuildMI(*Secondary->Block, Secondary->Block->getFirstNonPHI(), {},
            TII.get(LMU.consts().OpOr), LMU.consts().RegExec)
        .addReg(LMU.consts().RegExec)
        .addReg(Rejoin);

    LLVM_DEBUG(Function.dump());
  }

  Updater.cleanup();
}

namespace {

/// \brief Wave transform machine function pass.
class GCNWaveTransform : public MachineFunctionPass {
public:
  static char ID;

public:
  GCNWaveTransform() : MachineFunctionPass(ID) {
    initializeGCNWaveTransformPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &function) override;

  StringRef getPassName() const override {
    return "GCN Control Flow Wave Transform";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineCycleInfoWrapperPass>();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addUsedIfAvailable<MachineUniformityAnalysisPass>();
    AU.addPreserved<MachineDominatorTreeWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  MachineDominatorTree *DomTree = nullptr;
  // MachineConvergenceInfo ConvergenceInfo;
  MachineCycleInfo *CycleInfo;
  MachineUniformityInfo *UniformInfo = nullptr;
  GCNLaneMaskUtils LMU;
  const SIInstrInfo *TII;
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(GCNWaveTransform, DEBUG_TYPE, "GCN Wave Transform", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(MachineCycleInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(GCNWaveTransform, DEBUG_TYPE, "GCN Wave Transform", false,
                    false)

char GCNWaveTransform::ID = 0;

FunctionPass *llvm::createGCNWaveTransformPass() {
  return new GCNWaveTransform();
}

/// \brief Run the wave transform.
bool GCNWaveTransform::runOnMachineFunction(MachineFunction &MF) {
  if (MF.size() <= 1)
    return false; // skip MFs without control flow

  LLVM_DEBUG(dbgs() << "GCN Wave Transform: " << MF.getName() << '\n');

  DomTree = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  LMU.setFunction(MF);
  TII = MF.getSubtarget<GCNSubtarget>().getInstrInfo();

  // ConvergenceInfo = computeMachineConvergenceInfo(MF, *DomTree);
  CycleInfo = &getAnalysis<MachineCycleInfoWrapperPass>().getCycleInfo();
  UniformInfo = &getAnalysisIfAvailable<MachineUniformityAnalysisPass>()
                     ->getUniformityInfo();
  assert(UniformInfo && "wave transform needs MachineUniformityAnalysis");
  LLVM_DEBUG(UniformInfo->print(dbgs()));

  // Step 1: Compute reconverging Wave CFG
  ReconvergeCFGHelper ReconvergeHelper(*CycleInfo, *UniformInfo, *DomTree);
  ReconvergeHelper.run();

  ControlFlowRewriter CFRewriter(MF, ReconvergeHelper);
  CFRewriter.prepareWaveCfg();

  LLVM_DEBUG(dbgs() << "Final Wave CFG:\n"; ReconvergeHelper.dumpNodes());

  if (GCNWaveTransformPrintFinal) {
    dbgs() << "Wave CFG for " << MF.getName() << ":\n";
    ReconvergeHelper.printNodes(dbgs());
  }

  // Step 2: Create basic blocks for flow nodes and adjust MachineBasicBlock
  // successor and predecessor lists.
  MachineFunction::iterator insertIt = MF.end();
  for (auto *WN : llvm::reverse(ReconvergeHelper.nodes())) {
    if (!WN->Block) {
      WN->Block = MF.CreateMachineBasicBlock();
      MF.insert(insertIt, WN->Block);
      ReconvergeHelper.setNodeForBlock(WN->Block, WN);
    }

    insertIt = WN->Block->getIterator();
  }

  SmallVector<cfg::Update<MachineBasicBlock *>, 8> CFGUpdates;
  SmallVector<MachineBasicBlock *, 2> SuccToRemove;

  for (auto *WN : ReconvergeHelper.nodes()) {
    for (MachineBasicBlock *CurrentSucc : WN->Block->successors()) {
      if (llvm::find_if(WN->Successors, [=](WaveNode *Node) {
            return Node->Block == CurrentSucc;
          }) == WN->Successors.end())
        SuccToRemove.push_back(CurrentSucc);
    }
    for (MachineBasicBlock *Succ : SuccToRemove) {
      WN->Block->removeSuccessor(Succ);
      CFGUpdates.emplace_back(cfg::UpdateKind::Delete, WN->Block, Succ);
    }
    SuccToRemove.clear();

    for (auto *succ : WN->Successors) {
      if (!is_contained(WN->Block->successors(), succ->Block)) {
        WN->Block->addSuccessor(succ->Block);
        CFGUpdates.emplace_back(cfg::UpdateKind::Insert, WN->Block,
                                succ->Block);
      }
    }
  }

  DomTree->applyUpdates(CFGUpdates);
  CFGUpdates.clear();

  // Step 3: Re-establish SSA.
  SSAReconstructor SSAReconstruction(MF, *DomTree, ReconvergeHelper);
  SSAReconstruction.run();

  // Step 4: Fix up terminators and insert rejoin masks.
  CFRewriter.rewrite();

  // FIXME: restore the following 1 line:
  // UniformInfo.clear();
  // ConvergenceInfo.clear();
  DomTree = nullptr;

  // In some MIR tests, the MIR parser will set the NoPHIs property for the
  // test cases. We need to clear it here to avoid verifier errors.
  MF.getProperties().reset(MachineFunctionProperties::Property::NoPHIs);

  MF.getInfo<SIMachineFunctionInfo>()->setWholeWaveControlFlow(true);

  return true; // assume that we changed something
}
