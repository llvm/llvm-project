//===- CodeLayout.cpp - Implementation of code layout algorithms ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The file implements "cache-aware" layout algorithms of basic blocks and
// functions in a binary.
//
// The algorithm tries to find a layout of nodes (basic blocks) of a given CFG
// optimizing jump locality and thus processor I-cache utilization. This is
// achieved via increasing the number of fall-through jumps and co-locating
// frequently executed nodes together. The name follows the underlying
// optimization problem, Extended-TSP, which is a generalization of classical
// (maximum) Traveling Salesmen Problem.
//
// The algorithm is a greedy heuristic that works with chains (ordered lists)
// of basic blocks. Initially all chains are isolated basic blocks. On every
// iteration, we pick a pair of chains whose merging yields the biggest increase
// in the ExtTSP score, which models how i-cache "friendly" a specific chain is.
// A pair of chains giving the maximum gain is merged into a new chain. The
// procedure stops when there is only one chain left, or when merging does not
// increase ExtTSP. In the latter case, the remaining chains are sorted by
// density in the decreasing order.
//
// An important aspect is the way two chains are merged. Unlike earlier
// algorithms (e.g., based on the approach of Pettis-Hansen), two
// chains, X and Y, are first split into three, X1, X2, and Y. Then we
// consider all possible ways of gluing the three chains (e.g., X1YX2, X1X2Y,
// X2X1Y, X2YX1, YX1X2, YX2X1) and choose the one producing the largest score.
// This improves the quality of the final result (the search space is larger)
// while keeping the implementation sufficiently fast.
//
// Reference:
//   * A. Newell and S. Pupyrev, Improved Basic Block Reordering,
//     IEEE Transactions on Computers, 2020
//     https://arxiv.org/abs/1809.04676
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/CodeLayout.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include <cmath>

using namespace llvm;
#define DEBUG_TYPE "code-layout"

namespace llvm {
cl::opt<bool> EnableExtTspBlockPlacement(
    "enable-ext-tsp-block-placement", cl::Hidden, cl::init(false),
    cl::desc("Enable machine block placement based on the ext-tsp model, "
             "optimizing I-cache utilization."));

cl::opt<bool> ApplyExtTspWithoutProfile(
    "ext-tsp-apply-without-profile",
    cl::desc("Whether to apply ext-tsp placement for instances w/o profile"),
    cl::init(true), cl::Hidden);
} // namespace llvm

// Algorithm-specific params. The values are tuned for the best performance
// of large-scale front-end bound binaries.
static cl::opt<double> ForwardWeightCond(
    "ext-tsp-forward-weight-cond", cl::ReallyHidden, cl::init(0.1),
    cl::desc("The weight of conditional forward jumps for ExtTSP value"));

static cl::opt<double> ForwardWeightUncond(
    "ext-tsp-forward-weight-uncond", cl::ReallyHidden, cl::init(0.1),
    cl::desc("The weight of unconditional forward jumps for ExtTSP value"));

static cl::opt<double> BackwardWeightCond(
    "ext-tsp-backward-weight-cond", cl::ReallyHidden, cl::init(0.1),
    cl::desc("The weight of conditional backward jumps for ExtTSP value"));

static cl::opt<double> BackwardWeightUncond(
    "ext-tsp-backward-weight-uncond", cl::ReallyHidden, cl::init(0.1),
    cl::desc("The weight of unconditional backward jumps for ExtTSP value"));

static cl::opt<double> FallthroughWeightCond(
    "ext-tsp-fallthrough-weight-cond", cl::ReallyHidden, cl::init(1.0),
    cl::desc("The weight of conditional fallthrough jumps for ExtTSP value"));

static cl::opt<double> FallthroughWeightUncond(
    "ext-tsp-fallthrough-weight-uncond", cl::ReallyHidden, cl::init(1.05),
    cl::desc("The weight of unconditional fallthrough jumps for ExtTSP value"));

static cl::opt<unsigned> ForwardDistance(
    "ext-tsp-forward-distance", cl::ReallyHidden, cl::init(1024),
    cl::desc("The maximum distance (in bytes) of a forward jump for ExtTSP"));

static cl::opt<unsigned> BackwardDistance(
    "ext-tsp-backward-distance", cl::ReallyHidden, cl::init(640),
    cl::desc("The maximum distance (in bytes) of a backward jump for ExtTSP"));

// The maximum size of a chain created by the algorithm. The size is bounded
// so that the algorithm can efficiently process extremely large instance.
static cl::opt<unsigned>
    MaxChainSize("ext-tsp-max-chain-size", cl::ReallyHidden, cl::init(4096),
                 cl::desc("The maximum size of a chain to create."));

// The maximum size of a chain for splitting. Larger values of the threshold
// may yield better quality at the cost of worsen run-time.
static cl::opt<unsigned> ChainSplitThreshold(
    "ext-tsp-chain-split-threshold", cl::ReallyHidden, cl::init(128),
    cl::desc("The maximum size of a chain to apply splitting"));

// The option enables splitting (large) chains along in-coming and out-going
// jumps. This typically results in a better quality.
static cl::opt<bool> EnableChainSplitAlongJumps(
    "ext-tsp-enable-chain-split-along-jumps", cl::ReallyHidden, cl::init(true),
    cl::desc("The maximum size of a chain to apply splitting"));

namespace {

// Epsilon for comparison of doubles.
constexpr double EPS = 1e-8;

// Compute the Ext-TSP score for a given jump.
double jumpExtTSPScore(uint64_t JumpDist, uint64_t JumpMaxDist, uint64_t Count,
                       double Weight) {
  if (JumpDist > JumpMaxDist)
    return 0;
  double Prob = 1.0 - static_cast<double>(JumpDist) / JumpMaxDist;
  return Weight * Prob * Count;
}

// Compute the Ext-TSP score for a jump between a given pair of blocks,
// using their sizes, (estimated) addresses and the jump execution count.
double extTSPScore(uint64_t SrcAddr, uint64_t SrcSize, uint64_t DstAddr,
                   uint64_t Count, bool IsConditional) {
  // Fallthrough
  if (SrcAddr + SrcSize == DstAddr) {
    return jumpExtTSPScore(0, 1, Count,
                           IsConditional ? FallthroughWeightCond
                                         : FallthroughWeightUncond);
  }
  // Forward
  if (SrcAddr + SrcSize < DstAddr) {
    const uint64_t Dist = DstAddr - (SrcAddr + SrcSize);
    return jumpExtTSPScore(Dist, ForwardDistance, Count,
                           IsConditional ? ForwardWeightCond
                                         : ForwardWeightUncond);
  }
  // Backward
  const uint64_t Dist = SrcAddr + SrcSize - DstAddr;
  return jumpExtTSPScore(Dist, BackwardDistance, Count,
                         IsConditional ? BackwardWeightCond
                                       : BackwardWeightUncond);
}

/// A type of merging two chains, X and Y. The former chain is split into
/// X1 and X2 and then concatenated with Y in the order specified by the type.
enum class MergeTypeT : int { X_Y, Y_X, X1_Y_X2, Y_X2_X1, X2_X1_Y };

/// The gain of merging two chains, that is, the Ext-TSP score of the merge
/// together with the corresponding merge 'type' and 'offset'.
struct MergeGainT {
  explicit MergeGainT() = default;
  explicit MergeGainT(double Score, size_t MergeOffset, MergeTypeT MergeType)
      : Score(Score), MergeOffset(MergeOffset), MergeType(MergeType) {}

  double score() const { return Score; }

  size_t mergeOffset() const { return MergeOffset; }

  MergeTypeT mergeType() const { return MergeType; }

  void setMergeType(MergeTypeT Ty) { MergeType = Ty; }

  // Returns 'true' iff Other is preferred over this.
  bool operator<(const MergeGainT &Other) const {
    return (Other.Score > EPS && Other.Score > Score + EPS);
  }

  // Update the current gain if Other is preferred over this.
  void updateIfLessThan(const MergeGainT &Other) {
    if (*this < Other)
      *this = Other;
  }

private:
  double Score{-1.0};
  size_t MergeOffset{0};
  MergeTypeT MergeType{MergeTypeT::X_Y};
};

struct JumpT;
struct ChainT;
struct ChainEdge;

/// A node in the graph, typically corresponding to a basic block in the CFG or
/// a function in the call graph.
struct NodeT {
  NodeT(const NodeT &) = delete;
  NodeT(NodeT &&) = default;
  NodeT &operator=(const NodeT &) = delete;
  NodeT &operator=(NodeT &&) = default;

  explicit NodeT(size_t Index, uint64_t Size, uint64_t EC)
      : Index(Index), Size(Size), ExecutionCount(EC) {}

  bool isEntry() const { return Index == 0; }

  // The total execution count of outgoing jumps.
  uint64_t outCount() const;

  // The total execution count of incoming jumps.
  uint64_t inCount() const;

  // The original index of the node in graph.
  size_t Index{0};
  // The index of the node in the current chain.
  size_t CurIndex{0};
  // The size of the node in the binary.
  uint64_t Size{0};
  // The execution count of the node in the profile data.
  uint64_t ExecutionCount{0};
  // The current chain of the node.
  ChainT *CurChain{nullptr};
  // The offset of the node in the current chain.
  mutable uint64_t EstimatedAddr{0};
  // Forced successor of the node in the graph.
  NodeT *ForcedSucc{nullptr};
  // Forced predecessor of the node in the graph.
  NodeT *ForcedPred{nullptr};
  // Outgoing jumps from the node.
  std::vector<JumpT *> OutJumps;
  // Incoming jumps to the node.
  std::vector<JumpT *> InJumps;
};

/// An arc in the graph, typically corresponding to a jump between two nodes.
struct JumpT {
  JumpT(const JumpT &) = delete;
  JumpT(JumpT &&) = default;
  JumpT &operator=(const JumpT &) = delete;
  JumpT &operator=(JumpT &&) = default;

  explicit JumpT(NodeT *Source, NodeT *Target, uint64_t ExecutionCount)
      : Source(Source), Target(Target), ExecutionCount(ExecutionCount) {}

  // Source node of the jump.
  NodeT *Source;
  // Target node of the jump.
  NodeT *Target;
  // Execution count of the arc in the profile data.
  uint64_t ExecutionCount{0};
  // Whether the jump corresponds to a conditional branch.
  bool IsConditional{false};
  // The offset of the jump from the source node.
  uint64_t Offset{0};
};

/// A chain (ordered sequence) of nodes in the graph.
struct ChainT {
  ChainT(const ChainT &) = delete;
  ChainT(ChainT &&) = default;
  ChainT &operator=(const ChainT &) = delete;
  ChainT &operator=(ChainT &&) = default;

  explicit ChainT(uint64_t Id, NodeT *Node)
      : Id(Id), ExecutionCount(Node->ExecutionCount), Size(Node->Size),
        Nodes(1, Node) {}

  size_t numBlocks() const { return Nodes.size(); }

  double density() const { return static_cast<double>(ExecutionCount) / Size; }

  bool isEntry() const { return Nodes[0]->Index == 0; }

  bool isCold() const {
    for (NodeT *Node : Nodes) {
      if (Node->ExecutionCount > 0)
        return false;
    }
    return true;
  }

  ChainEdge *getEdge(ChainT *Other) const {
    for (auto It : Edges) {
      if (It.first == Other)
        return It.second;
    }
    return nullptr;
  }

  void removeEdge(ChainT *Other) {
    auto It = Edges.begin();
    while (It != Edges.end()) {
      if (It->first == Other) {
        Edges.erase(It);
        return;
      }
      It++;
    }
  }

  void addEdge(ChainT *Other, ChainEdge *Edge) {
    Edges.push_back(std::make_pair(Other, Edge));
  }

  void merge(ChainT *Other, const std::vector<NodeT *> &MergedBlocks) {
    Nodes = MergedBlocks;
    // Update the chain's data
    ExecutionCount += Other->ExecutionCount;
    Size += Other->Size;
    Id = Nodes[0]->Index;
    // Update the node's data
    for (size_t Idx = 0; Idx < Nodes.size(); Idx++) {
      Nodes[Idx]->CurChain = this;
      Nodes[Idx]->CurIndex = Idx;
    }
  }

  void mergeEdges(ChainT *Other);

  void clear() {
    Nodes.clear();
    Nodes.shrink_to_fit();
    Edges.clear();
    Edges.shrink_to_fit();
  }

  // Unique chain identifier.
  uint64_t Id;
  // Cached ext-tsp score for the chain.
  double Score{0};
  // The total execution count of the chain.
  uint64_t ExecutionCount{0};
  // The total size of the chain.
  uint64_t Size{0};
  // Nodes of the chain.
  std::vector<NodeT *> Nodes;
  // Adjacent chains and corresponding edges (lists of jumps).
  std::vector<std::pair<ChainT *, ChainEdge *>> Edges;
};

/// An edge in the graph representing jumps between two chains.
/// When nodes are merged into chains, the edges are combined too so that
/// there is always at most one edge between a pair of chains
struct ChainEdge {
  ChainEdge(const ChainEdge &) = delete;
  ChainEdge(ChainEdge &&) = default;
  ChainEdge &operator=(const ChainEdge &) = delete;
  ChainEdge &operator=(ChainEdge &&) = delete;

  explicit ChainEdge(JumpT *Jump)
      : SrcChain(Jump->Source->CurChain), DstChain(Jump->Target->CurChain),
        Jumps(1, Jump) {}

  ChainT *srcChain() const { return SrcChain; }

  ChainT *dstChain() const { return DstChain; }

  bool isSelfEdge() const { return SrcChain == DstChain; }

  const std::vector<JumpT *> &jumps() const { return Jumps; }

  void appendJump(JumpT *Jump) { Jumps.push_back(Jump); }

  void moveJumps(ChainEdge *Other) {
    Jumps.insert(Jumps.end(), Other->Jumps.begin(), Other->Jumps.end());
    Other->Jumps.clear();
    Other->Jumps.shrink_to_fit();
  }

  void changeEndpoint(ChainT *From, ChainT *To) {
    if (From == SrcChain)
      SrcChain = To;
    if (From == DstChain)
      DstChain = To;
  }

  bool hasCachedMergeGain(ChainT *Src, ChainT *Dst) const {
    return Src == SrcChain ? CacheValidForward : CacheValidBackward;
  }

  MergeGainT getCachedMergeGain(ChainT *Src, ChainT *Dst) const {
    return Src == SrcChain ? CachedGainForward : CachedGainBackward;
  }

  void setCachedMergeGain(ChainT *Src, ChainT *Dst, MergeGainT MergeGain) {
    if (Src == SrcChain) {
      CachedGainForward = MergeGain;
      CacheValidForward = true;
    } else {
      CachedGainBackward = MergeGain;
      CacheValidBackward = true;
    }
  }

  void invalidateCache() {
    CacheValidForward = false;
    CacheValidBackward = false;
  }

  void setMergeGain(MergeGainT Gain) { CachedGain = Gain; }

  MergeGainT getMergeGain() const { return CachedGain; }

  double gain() const { return CachedGain.score(); }

private:
  // Source chain.
  ChainT *SrcChain{nullptr};
  // Destination chain.
  ChainT *DstChain{nullptr};
  // Original jumps in the binary with corresponding execution counts.
  std::vector<JumpT *> Jumps;
  // Cached gain value for merging the pair of chains.
  MergeGainT CachedGain;

  // Cached gain values for merging the pair of chains. Since the gain of
  // merging (Src, Dst) and (Dst, Src) might be different, we store both values
  // here and a flag indicating which of the options results in a higher gain.
  // Cached gain values.
  MergeGainT CachedGainForward;
  MergeGainT CachedGainBackward;
  // Whether the cached value must be recomputed.
  bool CacheValidForward{false};
  bool CacheValidBackward{false};
};

uint64_t NodeT::outCount() const {
  uint64_t Count = 0;
  for (JumpT *Jump : OutJumps) {
    Count += Jump->ExecutionCount;
  }
  return Count;
}

uint64_t NodeT::inCount() const {
  uint64_t Count = 0;
  for (JumpT *Jump : InJumps) {
    Count += Jump->ExecutionCount;
  }
  return Count;
}

void ChainT::mergeEdges(ChainT *Other) {
  // Update edges adjacent to chain Other
  for (auto EdgeIt : Other->Edges) {
    ChainT *DstChain = EdgeIt.first;
    ChainEdge *DstEdge = EdgeIt.second;
    ChainT *TargetChain = DstChain == Other ? this : DstChain;
    ChainEdge *CurEdge = getEdge(TargetChain);
    if (CurEdge == nullptr) {
      DstEdge->changeEndpoint(Other, this);
      this->addEdge(TargetChain, DstEdge);
      if (DstChain != this && DstChain != Other) {
        DstChain->addEdge(this, DstEdge);
      }
    } else {
      CurEdge->moveJumps(DstEdge);
    }
    // Cleanup leftover edge
    if (DstChain != Other) {
      DstChain->removeEdge(Other);
    }
  }
}

using NodeIter = std::vector<NodeT *>::const_iterator;

/// A wrapper around three chains of nodes; it is used to avoid extra
/// instantiation of the vectors.
struct MergedChain {
  MergedChain(NodeIter Begin1, NodeIter End1, NodeIter Begin2 = NodeIter(),
              NodeIter End2 = NodeIter(), NodeIter Begin3 = NodeIter(),
              NodeIter End3 = NodeIter())
      : Begin1(Begin1), End1(End1), Begin2(Begin2), End2(End2), Begin3(Begin3),
        End3(End3) {}

  template <typename F> void forEach(const F &Func) const {
    for (auto It = Begin1; It != End1; It++)
      Func(*It);
    for (auto It = Begin2; It != End2; It++)
      Func(*It);
    for (auto It = Begin3; It != End3; It++)
      Func(*It);
  }

  std::vector<NodeT *> getNodes() const {
    std::vector<NodeT *> Result;
    Result.reserve(std::distance(Begin1, End1) + std::distance(Begin2, End2) +
                   std::distance(Begin3, End3));
    Result.insert(Result.end(), Begin1, End1);
    Result.insert(Result.end(), Begin2, End2);
    Result.insert(Result.end(), Begin3, End3);
    return Result;
  }

  const NodeT *getFirstNode() const { return *Begin1; }

private:
  NodeIter Begin1;
  NodeIter End1;
  NodeIter Begin2;
  NodeIter End2;
  NodeIter Begin3;
  NodeIter End3;
};

/// Merge two chains of nodes respecting a given 'type' and 'offset'.
///
/// If MergeType == 0, then the result is a concatenation of two chains.
/// Otherwise, the first chain is cut into two sub-chains at the offset,
/// and merged using all possible ways of concatenating three chains.
MergedChain mergeNodes(const std::vector<NodeT *> &X,
                       const std::vector<NodeT *> &Y, size_t MergeOffset,
                       MergeTypeT MergeType) {
  // Split the first chain, X, into X1 and X2
  NodeIter BeginX1 = X.begin();
  NodeIter EndX1 = X.begin() + MergeOffset;
  NodeIter BeginX2 = X.begin() + MergeOffset;
  NodeIter EndX2 = X.end();
  NodeIter BeginY = Y.begin();
  NodeIter EndY = Y.end();

  // Construct a new chain from the three existing ones
  switch (MergeType) {
  case MergeTypeT::X_Y:
    return MergedChain(BeginX1, EndX2, BeginY, EndY);
  case MergeTypeT::Y_X:
    return MergedChain(BeginY, EndY, BeginX1, EndX2);
  case MergeTypeT::X1_Y_X2:
    return MergedChain(BeginX1, EndX1, BeginY, EndY, BeginX2, EndX2);
  case MergeTypeT::Y_X2_X1:
    return MergedChain(BeginY, EndY, BeginX2, EndX2, BeginX1, EndX1);
  case MergeTypeT::X2_X1_Y:
    return MergedChain(BeginX2, EndX2, BeginX1, EndX1, BeginY, EndY);
  }
  llvm_unreachable("unexpected chain merge type");
}

/// The implementation of the ExtTSP algorithm.
class ExtTSPImpl {
public:
  ExtTSPImpl(const std::vector<uint64_t> &NodeSizes,
             const std::vector<uint64_t> &NodeCounts,
             const std::vector<EdgeCountT> &EdgeCounts)
      : NumNodes(NodeSizes.size()) {
    initialize(NodeSizes, NodeCounts, EdgeCounts);
  }

  /// Run the algorithm and return an optimized ordering of nodes.
  void run(std::vector<uint64_t> &Result) {
    // Pass 1: Merge nodes with their mutually forced successors
    mergeForcedPairs();

    // Pass 2: Merge pairs of chains while improving the ExtTSP objective
    mergeChainPairs();

    // Pass 3: Merge cold nodes to reduce code size
    mergeColdChains();

    // Collect nodes from all chains
    concatChains(Result);
  }

private:
  /// Initialize the algorithm's data structures.
  void initialize(const std::vector<uint64_t> &NodeSizes,
                  const std::vector<uint64_t> &NodeCounts,
                  const std::vector<EdgeCountT> &EdgeCounts) {
    // Initialize nodes
    AllNodes.reserve(NumNodes);
    for (uint64_t Idx = 0; Idx < NumNodes; Idx++) {
      uint64_t Size = std::max<uint64_t>(NodeSizes[Idx], 1ULL);
      uint64_t ExecutionCount = NodeCounts[Idx];
      // The execution count of the entry node is set to at least one
      if (Idx == 0 && ExecutionCount == 0)
        ExecutionCount = 1;
      AllNodes.emplace_back(Idx, Size, ExecutionCount);
    }

    // Initialize jumps between nodes
    SuccNodes.resize(NumNodes);
    PredNodes.resize(NumNodes);
    std::vector<uint64_t> OutDegree(NumNodes, 0);
    AllJumps.reserve(EdgeCounts.size());
    for (auto It : EdgeCounts) {
      uint64_t Pred = It.first.first;
      uint64_t Succ = It.first.second;
      OutDegree[Pred]++;
      // Ignore self-edges
      if (Pred == Succ)
        continue;

      SuccNodes[Pred].push_back(Succ);
      PredNodes[Succ].push_back(Pred);
      uint64_t ExecutionCount = It.second;
      if (ExecutionCount > 0) {
        NodeT &PredNode = AllNodes[Pred];
        NodeT &SuccNode = AllNodes[Succ];
        AllJumps.emplace_back(&PredNode, &SuccNode, ExecutionCount);
        SuccNode.InJumps.push_back(&AllJumps.back());
        PredNode.OutJumps.push_back(&AllJumps.back());
      }
    }
    for (JumpT &Jump : AllJumps) {
      assert(OutDegree[Jump.Source->Index] > 0);
      Jump.IsConditional = OutDegree[Jump.Source->Index] > 1;
    }

    // Initialize chains
    AllChains.reserve(NumNodes);
    HotChains.reserve(NumNodes);
    for (NodeT &Node : AllNodes) {
      AllChains.emplace_back(Node.Index, &Node);
      Node.CurChain = &AllChains.back();
      if (Node.ExecutionCount > 0) {
        HotChains.push_back(&AllChains.back());
      }
    }

    // Initialize chain edges
    AllEdges.reserve(AllJumps.size());
    for (NodeT &PredNode : AllNodes) {
      for (JumpT *Jump : PredNode.OutJumps) {
        NodeT *SuccNode = Jump->Target;
        ChainEdge *CurEdge = PredNode.CurChain->getEdge(SuccNode->CurChain);
        // this edge is already present in the graph
        if (CurEdge != nullptr) {
          assert(SuccNode->CurChain->getEdge(PredNode.CurChain) != nullptr);
          CurEdge->appendJump(Jump);
          continue;
        }
        // this is a new edge
        AllEdges.emplace_back(Jump);
        PredNode.CurChain->addEdge(SuccNode->CurChain, &AllEdges.back());
        SuccNode->CurChain->addEdge(PredNode.CurChain, &AllEdges.back());
      }
    }
  }

  /// For a pair of nodes, A and B, node B is the forced successor of A,
  /// if (i) all jumps (based on profile) from A goes to B and (ii) all jumps
  /// to B are from A. Such nodes should be adjacent in the optimal ordering;
  /// the method finds and merges such pairs of nodes.
  void mergeForcedPairs() {
    // Find fallthroughs based on edge weights
    for (NodeT &Node : AllNodes) {
      if (SuccNodes[Node.Index].size() == 1 &&
          PredNodes[SuccNodes[Node.Index][0]].size() == 1 &&
          SuccNodes[Node.Index][0] != 0) {
        size_t SuccIndex = SuccNodes[Node.Index][0];
        Node.ForcedSucc = &AllNodes[SuccIndex];
        AllNodes[SuccIndex].ForcedPred = &Node;
      }
    }

    // There might be 'cycles' in the forced dependencies, since profile
    // data isn't 100% accurate. Typically this is observed in loops, when the
    // loop edges are the hottest successors for the basic blocks of the loop.
    // Break the cycles by choosing the node with the smallest index as the
    // head. This helps to keep the original order of the loops, which likely
    // have already been rotated in the optimized manner.
    for (NodeT &Node : AllNodes) {
      if (Node.ForcedSucc == nullptr || Node.ForcedPred == nullptr)
        continue;

      NodeT *SuccNode = Node.ForcedSucc;
      while (SuccNode != nullptr && SuccNode != &Node) {
        SuccNode = SuccNode->ForcedSucc;
      }
      if (SuccNode == nullptr)
        continue;
      // Break the cycle
      AllNodes[Node.ForcedPred->Index].ForcedSucc = nullptr;
      Node.ForcedPred = nullptr;
    }

    // Merge nodes with their fallthrough successors
    for (NodeT &Node : AllNodes) {
      if (Node.ForcedPred == nullptr && Node.ForcedSucc != nullptr) {
        const NodeT *CurBlock = &Node;
        while (CurBlock->ForcedSucc != nullptr) {
          const NodeT *NextBlock = CurBlock->ForcedSucc;
          mergeChains(Node.CurChain, NextBlock->CurChain, 0, MergeTypeT::X_Y);
          CurBlock = NextBlock;
        }
      }
    }
  }

  /// Merge pairs of chains while improving the ExtTSP objective.
  void mergeChainPairs() {
    /// Deterministically compare pairs of chains
    auto compareChainPairs = [](const ChainT *A1, const ChainT *B1,
                                const ChainT *A2, const ChainT *B2) {
      if (A1 != A2)
        return A1->Id < A2->Id;
      return B1->Id < B2->Id;
    };

    while (HotChains.size() > 1) {
      ChainT *BestChainPred = nullptr;
      ChainT *BestChainSucc = nullptr;
      MergeGainT BestGain;
      // Iterate over all pairs of chains
      for (ChainT *ChainPred : HotChains) {
        // Get candidates for merging with the current chain
        for (auto EdgeIt : ChainPred->Edges) {
          ChainT *ChainSucc = EdgeIt.first;
          ChainEdge *Edge = EdgeIt.second;
          // Ignore loop edges
          if (ChainPred == ChainSucc)
            continue;

          // Stop early if the combined chain violates the maximum allowed size
          if (ChainPred->numBlocks() + ChainSucc->numBlocks() >= MaxChainSize)
            continue;

          // Compute the gain of merging the two chains
          MergeGainT CurGain = getBestMergeGain(ChainPred, ChainSucc, Edge);
          if (CurGain.score() <= EPS)
            continue;

          if (BestGain < CurGain ||
              (std::abs(CurGain.score() - BestGain.score()) < EPS &&
               compareChainPairs(ChainPred, ChainSucc, BestChainPred,
                                 BestChainSucc))) {
            BestGain = CurGain;
            BestChainPred = ChainPred;
            BestChainSucc = ChainSucc;
          }
        }
      }

      // Stop merging when there is no improvement
      if (BestGain.score() <= EPS)
        break;

      // Merge the best pair of chains
      mergeChains(BestChainPred, BestChainSucc, BestGain.mergeOffset(),
                  BestGain.mergeType());
    }
  }

  /// Merge remaining nodes into chains w/o taking jump counts into
  /// consideration. This allows to maintain the original node order in the
  /// absence of profile data
  void mergeColdChains() {
    for (size_t SrcBB = 0; SrcBB < NumNodes; SrcBB++) {
      // Iterating in reverse order to make sure original fallthrough jumps are
      // merged first; this might be beneficial for code size.
      size_t NumSuccs = SuccNodes[SrcBB].size();
      for (size_t Idx = 0; Idx < NumSuccs; Idx++) {
        size_t DstBB = SuccNodes[SrcBB][NumSuccs - Idx - 1];
        ChainT *SrcChain = AllNodes[SrcBB].CurChain;
        ChainT *DstChain = AllNodes[DstBB].CurChain;
        if (SrcChain != DstChain && !DstChain->isEntry() &&
            SrcChain->Nodes.back()->Index == SrcBB &&
            DstChain->Nodes.front()->Index == DstBB &&
            SrcChain->isCold() == DstChain->isCold()) {
          mergeChains(SrcChain, DstChain, 0, MergeTypeT::X_Y);
        }
      }
    }
  }

  /// Compute the Ext-TSP score for a given node order and a list of jumps.
  double extTSPScore(const MergedChain &MergedBlocks,
                     const std::vector<JumpT *> &Jumps) const {
    if (Jumps.empty())
      return 0.0;
    uint64_t CurAddr = 0;
    MergedBlocks.forEach([&](const NodeT *Node) {
      Node->EstimatedAddr = CurAddr;
      CurAddr += Node->Size;
    });

    double Score = 0;
    for (JumpT *Jump : Jumps) {
      const NodeT *SrcBlock = Jump->Source;
      const NodeT *DstBlock = Jump->Target;
      Score += ::extTSPScore(SrcBlock->EstimatedAddr, SrcBlock->Size,
                             DstBlock->EstimatedAddr, Jump->ExecutionCount,
                             Jump->IsConditional);
    }
    return Score;
  }

  /// Compute the gain of merging two chains.
  ///
  /// The function considers all possible ways of merging two chains and
  /// computes the one having the largest increase in ExtTSP objective. The
  /// result is a pair with the first element being the gain and the second
  /// element being the corresponding merging type.
  MergeGainT getBestMergeGain(ChainT *ChainPred, ChainT *ChainSucc,
                              ChainEdge *Edge) const {
    if (Edge->hasCachedMergeGain(ChainPred, ChainSucc)) {
      return Edge->getCachedMergeGain(ChainPred, ChainSucc);
    }

    // Precompute jumps between ChainPred and ChainSucc
    auto Jumps = Edge->jumps();
    ChainEdge *EdgePP = ChainPred->getEdge(ChainPred);
    if (EdgePP != nullptr) {
      Jumps.insert(Jumps.end(), EdgePP->jumps().begin(), EdgePP->jumps().end());
    }
    assert(!Jumps.empty() && "trying to merge chains w/o jumps");

    // The object holds the best currently chosen gain of merging the two chains
    MergeGainT Gain = MergeGainT();

    /// Given a merge offset and a list of merge types, try to merge two chains
    /// and update Gain with a better alternative
    auto tryChainMerging = [&](size_t Offset,
                               const std::vector<MergeTypeT> &MergeTypes) {
      // Skip merging corresponding to concatenation w/o splitting
      if (Offset == 0 || Offset == ChainPred->Nodes.size())
        return;
      // Skip merging if it breaks Forced successors
      NodeT *Node = ChainPred->Nodes[Offset - 1];
      if (Node->ForcedSucc != nullptr)
        return;
      // Apply the merge, compute the corresponding gain, and update the best
      // value, if the merge is beneficial
      for (const MergeTypeT &MergeType : MergeTypes) {
        Gain.updateIfLessThan(
            computeMergeGain(ChainPred, ChainSucc, Jumps, Offset, MergeType));
      }
    };

    // Try to concatenate two chains w/o splitting
    Gain.updateIfLessThan(
        computeMergeGain(ChainPred, ChainSucc, Jumps, 0, MergeTypeT::X_Y));

    if (EnableChainSplitAlongJumps) {
      // Attach (a part of) ChainPred before the first node of ChainSucc
      for (JumpT *Jump : ChainSucc->Nodes.front()->InJumps) {
        const NodeT *SrcBlock = Jump->Source;
        if (SrcBlock->CurChain != ChainPred)
          continue;
        size_t Offset = SrcBlock->CurIndex + 1;
        tryChainMerging(Offset, {MergeTypeT::X1_Y_X2, MergeTypeT::X2_X1_Y});
      }

      // Attach (a part of) ChainPred after the last node of ChainSucc
      for (JumpT *Jump : ChainSucc->Nodes.back()->OutJumps) {
        const NodeT *DstBlock = Jump->Source;
        if (DstBlock->CurChain != ChainPred)
          continue;
        size_t Offset = DstBlock->CurIndex;
        tryChainMerging(Offset, {MergeTypeT::X1_Y_X2, MergeTypeT::Y_X2_X1});
      }
    }

    // Try to break ChainPred in various ways and concatenate with ChainSucc
    if (ChainPred->Nodes.size() <= ChainSplitThreshold) {
      for (size_t Offset = 1; Offset < ChainPred->Nodes.size(); Offset++) {
        // Try to split the chain in different ways. In practice, applying
        // X2_Y_X1 merging is almost never provides benefits; thus, we exclude
        // it from consideration to reduce the search space
        tryChainMerging(Offset, {MergeTypeT::X1_Y_X2, MergeTypeT::Y_X2_X1,
                                 MergeTypeT::X2_X1_Y});
      }
    }
    Edge->setCachedMergeGain(ChainPred, ChainSucc, Gain);
    return Gain;
  }

  /// Compute the score gain of merging two chains, respecting a given
  /// merge 'type' and 'offset'.
  ///
  /// The two chains are not modified in the method.
  MergeGainT computeMergeGain(const ChainT *ChainPred, const ChainT *ChainSucc,
                              const std::vector<JumpT *> &Jumps,
                              size_t MergeOffset, MergeTypeT MergeType) const {
    auto MergedBlocks =
        mergeNodes(ChainPred->Nodes, ChainSucc->Nodes, MergeOffset, MergeType);

    // Do not allow a merge that does not preserve the original entry point
    if ((ChainPred->isEntry() || ChainSucc->isEntry()) &&
        !MergedBlocks.getFirstNode()->isEntry())
      return MergeGainT();

    // The gain for the new chain
    auto NewGainScore = extTSPScore(MergedBlocks, Jumps) - ChainPred->Score;
    return MergeGainT(NewGainScore, MergeOffset, MergeType);
  }

  /// Merge chain From into chain Into, update the list of active chains,
  /// adjacency information, and the corresponding cached values.
  void mergeChains(ChainT *Into, ChainT *From, size_t MergeOffset,
                   MergeTypeT MergeType) {
    assert(Into != From && "a chain cannot be merged with itself");

    // Merge the nodes
    MergedChain MergedNodes =
        mergeNodes(Into->Nodes, From->Nodes, MergeOffset, MergeType);
    Into->merge(From, MergedNodes.getNodes());

    // Merge the edges
    Into->mergeEdges(From);
    From->clear();

    // Update cached ext-tsp score for the new chain
    ChainEdge *SelfEdge = Into->getEdge(Into);
    if (SelfEdge != nullptr) {
      MergedNodes = MergedChain(Into->Nodes.begin(), Into->Nodes.end());
      Into->Score = extTSPScore(MergedNodes, SelfEdge->jumps());
    }

    // Remove the chain from the list of active chains
    llvm::erase_value(HotChains, From);

    // Invalidate caches
    for (auto EdgeIt : Into->Edges)
      EdgeIt.second->invalidateCache();
  }

  /// Concatenate all chains into the final order.
  void concatChains(std::vector<uint64_t> &Order) {
    // Collect chains and calculate density stats for their sorting
    std::vector<const ChainT *> SortedChains;
    DenseMap<const ChainT *, double> ChainDensity;
    for (ChainT &Chain : AllChains) {
      if (!Chain.Nodes.empty()) {
        SortedChains.push_back(&Chain);
        // Using doubles to avoid overflow of ExecutionCounts
        double Size = 0;
        double ExecutionCount = 0;
        for (NodeT *Node : Chain.Nodes) {
          Size += static_cast<double>(Node->Size);
          ExecutionCount += static_cast<double>(Node->ExecutionCount);
        }
        assert(Size > 0 && "a chain of zero size");
        ChainDensity[&Chain] = ExecutionCount / Size;
      }
    }

    // Sorting chains by density in the decreasing order
    std::stable_sort(SortedChains.begin(), SortedChains.end(),
                     [&](const ChainT *L, const ChainT *R) {
                       // Make sure the original entry point is at the
                       // beginning of the order
                       if (L->isEntry() != R->isEntry())
                         return L->isEntry();

                       const double DL = ChainDensity[L];
                       const double DR = ChainDensity[R];
                       // Compare by density and break ties by chain identifiers
                       return (DL != DR) ? (DL > DR) : (L->Id < R->Id);
                     });

    // Collect the nodes in the order specified by their chains
    Order.reserve(NumNodes);
    for (const ChainT *Chain : SortedChains) {
      for (NodeT *Node : Chain->Nodes) {
        Order.push_back(Node->Index);
      }
    }
  }

private:
  /// The number of nodes in the graph.
  const size_t NumNodes;

  /// Successors of each node.
  std::vector<std::vector<uint64_t>> SuccNodes;

  /// Predecessors of each node.
  std::vector<std::vector<uint64_t>> PredNodes;

  /// All nodes (basic blocks) in the graph.
  std::vector<NodeT> AllNodes;

  /// All jumps between the nodes.
  std::vector<JumpT> AllJumps;

  /// All chains of nodes.
  std::vector<ChainT> AllChains;

  /// All edges between the chains.
  std::vector<ChainEdge> AllEdges;

  /// Active chains. The vector gets updated at runtime when chains are merged.
  std::vector<ChainT *> HotChains;
};

} // end of anonymous namespace

std::vector<uint64_t>
llvm::applyExtTspLayout(const std::vector<uint64_t> &NodeSizes,
                        const std::vector<uint64_t> &NodeCounts,
                        const std::vector<EdgeCountT> &EdgeCounts) {
  // Verify correctness of the input data
  assert(NodeCounts.size() == NodeSizes.size() && "Incorrect input");
  assert(NodeSizes.size() > 2 && "Incorrect input");

  // Apply the reordering algorithm
  ExtTSPImpl Alg(NodeSizes, NodeCounts, EdgeCounts);
  std::vector<uint64_t> Result;
  Alg.run(Result);

  // Verify correctness of the output
  assert(Result.front() == 0 && "Original entry point is not preserved");
  assert(Result.size() == NodeSizes.size() && "Incorrect size of layout");
  return Result;
}

double llvm::calcExtTspScore(const std::vector<uint64_t> &Order,
                             const std::vector<uint64_t> &NodeSizes,
                             const std::vector<uint64_t> &NodeCounts,
                             const std::vector<EdgeCountT> &EdgeCounts) {
  // Estimate addresses of the blocks in memory
  std::vector<uint64_t> Addr(NodeSizes.size(), 0);
  for (size_t Idx = 1; Idx < Order.size(); Idx++) {
    Addr[Order[Idx]] = Addr[Order[Idx - 1]] + NodeSizes[Order[Idx - 1]];
  }
  std::vector<uint64_t> OutDegree(NodeSizes.size(), 0);
  for (auto It : EdgeCounts) {
    uint64_t Pred = It.first.first;
    OutDegree[Pred]++;
  }

  // Increase the score for each jump
  double Score = 0;
  for (auto It : EdgeCounts) {
    uint64_t Pred = It.first.first;
    uint64_t Succ = It.first.second;
    uint64_t Count = It.second;
    bool IsConditional = OutDegree[Pred] > 1;
    Score += ::extTSPScore(Addr[Pred], NodeSizes[Pred], Addr[Succ], Count,
                           IsConditional);
  }
  return Score;
}

double llvm::calcExtTspScore(const std::vector<uint64_t> &NodeSizes,
                             const std::vector<uint64_t> &NodeCounts,
                             const std::vector<EdgeCountT> &EdgeCounts) {
  std::vector<uint64_t> Order(NodeSizes.size());
  for (size_t Idx = 0; Idx < NodeSizes.size(); Idx++) {
    Order[Idx] = Idx;
  }
  return calcExtTspScore(Order, NodeSizes, NodeCounts, EdgeCounts);
}
