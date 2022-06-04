//===- CodeLayout.cpp - Implementation of code layout algorithms ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ExtTSP - layout of basic blocks with i-cache optimization.
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
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/CodeLayout.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;
#define DEBUG_TYPE "code-layout"

cl::opt<bool> EnableExtTspBlockPlacement(
    "enable-ext-tsp-block-placement", cl::Hidden, cl::init(false),
    cl::desc("Enable machine block placement based on the ext-tsp model, "
             "optimizing I-cache utilization."));

cl::opt<bool> ApplyExtTspWithoutProfile(
    "ext-tsp-apply-without-profile",
    cl::desc("Whether to apply ext-tsp placement for instances w/o profile"),
    cl::init(true), cl::Hidden);

// Algorithm-specific constants. The values are tuned for the best performance
// of large-scale front-end bound binaries.
static cl::opt<double>
    ForwardWeight("ext-tsp-forward-weight", cl::Hidden, cl::init(0.1),
                  cl::desc("The weight of forward jumps for ExtTSP value"));

static cl::opt<double>
    BackwardWeight("ext-tsp-backward-weight", cl::Hidden, cl::init(0.1),
                   cl::desc("The weight of backward jumps for ExtTSP value"));

static cl::opt<unsigned> ForwardDistance(
    "ext-tsp-forward-distance", cl::Hidden, cl::init(1024),
    cl::desc("The maximum distance (in bytes) of a forward jump for ExtTSP"));

static cl::opt<unsigned> BackwardDistance(
    "ext-tsp-backward-distance", cl::Hidden, cl::init(640),
    cl::desc("The maximum distance (in bytes) of a backward jump for ExtTSP"));

// The maximum size of a chain created by the algorithm. The size is bounded
// so that the algorithm can efficiently process extremely large instance.
static cl::opt<unsigned>
    MaxChainSize("ext-tsp-max-chain-size", cl::Hidden, cl::init(4096),
                 cl::desc("The maximum size of a chain to create."));

// The maximum size of a chain for splitting. Larger values of the threshold
// may yield better quality at the cost of worsen run-time.
static cl::opt<unsigned> ChainSplitThreshold(
    "ext-tsp-chain-split-threshold", cl::Hidden, cl::init(128),
    cl::desc("The maximum size of a chain to apply splitting"));

// The option enables splitting (large) chains along in-coming and out-going
// jumps. This typically results in a better quality.
static cl::opt<bool> EnableChainSplitAlongJumps(
    "ext-tsp-enable-chain-split-along-jumps", cl::Hidden, cl::init(true),
    cl::desc("The maximum size of a chain to apply splitting"));

namespace {

// Epsilon for comparison of doubles.
constexpr double EPS = 1e-8;

// Compute the Ext-TSP score for a jump between a given pair of blocks,
// using their sizes, (estimated) addresses and the jump execution count.
double extTSPScore(uint64_t SrcAddr, uint64_t SrcSize, uint64_t DstAddr,
                   uint64_t Count) {
  // Fallthrough
  if (SrcAddr + SrcSize == DstAddr) {
    // Assume that FallthroughWeight = 1.0 after normalization
    return static_cast<double>(Count);
  }
  // Forward
  if (SrcAddr + SrcSize < DstAddr) {
    const auto Dist = DstAddr - (SrcAddr + SrcSize);
    if (Dist <= ForwardDistance) {
      double Prob = 1.0 - static_cast<double>(Dist) / ForwardDistance;
      return ForwardWeight * Prob * Count;
    }
    return 0;
  }
  // Backward
  const auto Dist = SrcAddr + SrcSize - DstAddr;
  if (Dist <= BackwardDistance) {
    double Prob = 1.0 - static_cast<double>(Dist) / BackwardDistance;
    return BackwardWeight * Prob * Count;
  }
  return 0;
}

/// A type of merging two chains, X and Y. The former chain is split into
/// X1 and X2 and then concatenated with Y in the order specified by the type.
enum class MergeTypeTy : int { X_Y, X1_Y_X2, Y_X2_X1, X2_X1_Y };

/// The gain of merging two chains, that is, the Ext-TSP score of the merge
/// together with the corresponfiding merge 'type' and 'offset'.
class MergeGainTy {
public:
  explicit MergeGainTy() = default;
  explicit MergeGainTy(double Score, size_t MergeOffset, MergeTypeTy MergeType)
      : Score(Score), MergeOffset(MergeOffset), MergeType(MergeType) {}

  double score() const { return Score; }

  size_t mergeOffset() const { return MergeOffset; }

  MergeTypeTy mergeType() const { return MergeType; }

  // Returns 'true' iff Other is preferred over this.
  bool operator<(const MergeGainTy &Other) const {
    return (Other.Score > EPS && Other.Score > Score + EPS);
  }

  // Update the current gain if Other is preferred over this.
  void updateIfLessThan(const MergeGainTy &Other) {
    if (*this < Other)
      *this = Other;
  }

private:
  double Score{-1.0};
  size_t MergeOffset{0};
  MergeTypeTy MergeType{MergeTypeTy::X_Y};
};

class Jump;
class Chain;
class ChainEdge;

/// A node in the graph, typically corresponding to a basic block in CFG.
class Block {
public:
  Block(const Block &) = delete;
  Block(Block &&) = default;
  Block &operator=(const Block &) = delete;
  Block &operator=(Block &&) = default;

  // The original index of the block in CFG.
  size_t Index{0};
  // The index of the block in the current chain.
  size_t CurIndex{0};
  // Size of the block in the binary.
  uint64_t Size{0};
  // Execution count of the block in the profile data.
  uint64_t ExecutionCount{0};
  // Current chain of the node.
  Chain *CurChain{nullptr};
  // An offset of the block in the current chain.
  mutable uint64_t EstimatedAddr{0};
  // Forced successor of the block in CFG.
  Block *ForcedSucc{nullptr};
  // Forced predecessor of the block in CFG.
  Block *ForcedPred{nullptr};
  // Outgoing jumps from the block.
  std::vector<Jump *> OutJumps;
  // Incoming jumps to the block.
  std::vector<Jump *> InJumps;

public:
  explicit Block(size_t Index, uint64_t Size_, uint64_t EC)
      : Index(Index), Size(Size_), ExecutionCount(EC) {}
  bool isEntry() const { return Index == 0; }
};

/// An arc in the graph, typically corresponding to a jump between two blocks.
class Jump {
public:
  Jump(const Jump &) = delete;
  Jump(Jump &&) = default;
  Jump &operator=(const Jump &) = delete;
  Jump &operator=(Jump &&) = default;

  // Source block of the jump.
  Block *Source;
  // Target block of the jump.
  Block *Target;
  // Execution count of the arc in the profile data.
  uint64_t ExecutionCount{0};

public:
  explicit Jump(Block *Source, Block *Target, uint64_t ExecutionCount)
      : Source(Source), Target(Target), ExecutionCount(ExecutionCount) {}
};

/// A chain (ordered sequence) of blocks.
class Chain {
public:
  Chain(const Chain &) = delete;
  Chain(Chain &&) = default;
  Chain &operator=(const Chain &) = delete;
  Chain &operator=(Chain &&) = default;

  explicit Chain(uint64_t Id, Block *Block)
      : Id(Id), Score(0), Blocks(1, Block) {}

  uint64_t id() const { return Id; }

  bool isEntry() const { return Blocks[0]->Index == 0; }

  double score() const { return Score; }

  void setScore(double NewScore) { Score = NewScore; }

  const std::vector<Block *> &blocks() const { return Blocks; }

  size_t numBlocks() const { return Blocks.size(); }

  const std::vector<std::pair<Chain *, ChainEdge *>> &edges() const {
    return Edges;
  }

  ChainEdge *getEdge(Chain *Other) const {
    for (auto It : Edges) {
      if (It.first == Other)
        return It.second;
    }
    return nullptr;
  }

  void removeEdge(Chain *Other) {
    auto It = Edges.begin();
    while (It != Edges.end()) {
      if (It->first == Other) {
        Edges.erase(It);
        return;
      }
      It++;
    }
  }

  void addEdge(Chain *Other, ChainEdge *Edge) {
    Edges.push_back(std::make_pair(Other, Edge));
  }

  void merge(Chain *Other, const std::vector<Block *> &MergedBlocks) {
    Blocks = MergedBlocks;
    // Update the block's chains
    for (size_t Idx = 0; Idx < Blocks.size(); Idx++) {
      Blocks[Idx]->CurChain = this;
      Blocks[Idx]->CurIndex = Idx;
    }
  }

  void mergeEdges(Chain *Other);

  void clear() {
    Blocks.clear();
    Blocks.shrink_to_fit();
    Edges.clear();
    Edges.shrink_to_fit();
  }

private:
  // Unique chain identifier.
  uint64_t Id;
  // Cached ext-tsp score for the chain.
  double Score;
  // Blocks of the chain.
  std::vector<Block *> Blocks;
  // Adjacent chains and corresponding edges (lists of jumps).
  std::vector<std::pair<Chain *, ChainEdge *>> Edges;
};

/// An edge in CFG representing jumps between two chains.
/// When blocks are merged into chains, the edges are combined too so that
/// there is always at most one edge between a pair of chains
class ChainEdge {
public:
  ChainEdge(const ChainEdge &) = delete;
  ChainEdge(ChainEdge &&) = default;
  ChainEdge &operator=(const ChainEdge &) = delete;
  ChainEdge &operator=(ChainEdge &&) = default;

  explicit ChainEdge(Jump *Jump)
      : SrcChain(Jump->Source->CurChain), DstChain(Jump->Target->CurChain),
        Jumps(1, Jump) {}

  const std::vector<Jump *> &jumps() const { return Jumps; }

  void changeEndpoint(Chain *From, Chain *To) {
    if (From == SrcChain)
      SrcChain = To;
    if (From == DstChain)
      DstChain = To;
  }

  void appendJump(Jump *Jump) { Jumps.push_back(Jump); }

  void moveJumps(ChainEdge *Other) {
    Jumps.insert(Jumps.end(), Other->Jumps.begin(), Other->Jumps.end());
    Other->Jumps.clear();
    Other->Jumps.shrink_to_fit();
  }

  bool hasCachedMergeGain(Chain *Src, Chain *Dst) const {
    return Src == SrcChain ? CacheValidForward : CacheValidBackward;
  }

  MergeGainTy getCachedMergeGain(Chain *Src, Chain *Dst) const {
    return Src == SrcChain ? CachedGainForward : CachedGainBackward;
  }

  void setCachedMergeGain(Chain *Src, Chain *Dst, MergeGainTy MergeGain) {
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

private:
  // Source chain.
  Chain *SrcChain{nullptr};
  // Destination chain.
  Chain *DstChain{nullptr};
  // Original jumps in the binary with correspinding execution counts.
  std::vector<Jump *> Jumps;
  // Cached ext-tsp value for merging the pair of chains.
  // Since the gain of merging (Src, Dst) and (Dst, Src) might be different,
  // we store both values here.
  MergeGainTy CachedGainForward;
  MergeGainTy CachedGainBackward;
  // Whether the cached value must be recomputed.
  bool CacheValidForward{false};
  bool CacheValidBackward{false};
};

void Chain::mergeEdges(Chain *Other) {
  assert(this != Other && "cannot merge a chain with itself");

  // Update edges adjacent to chain Other
  for (auto EdgeIt : Other->Edges) {
    const auto DstChain = EdgeIt.first;
    const auto DstEdge = EdgeIt.second;
    const auto TargetChain = DstChain == Other ? this : DstChain;
    auto CurEdge = getEdge(TargetChain);
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

using BlockIter = std::vector<Block *>::const_iterator;

/// A wrapper around three chains of blocks; it is used to avoid extra
/// instantiation of the vectors.
class MergedChain {
public:
  MergedChain(BlockIter Begin1, BlockIter End1, BlockIter Begin2 = BlockIter(),
              BlockIter End2 = BlockIter(), BlockIter Begin3 = BlockIter(),
              BlockIter End3 = BlockIter())
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

  std::vector<Block *> getBlocks() const {
    std::vector<Block *> Result;
    Result.reserve(std::distance(Begin1, End1) + std::distance(Begin2, End2) +
                   std::distance(Begin3, End3));
    Result.insert(Result.end(), Begin1, End1);
    Result.insert(Result.end(), Begin2, End2);
    Result.insert(Result.end(), Begin3, End3);
    return Result;
  }

  const Block *getFirstBlock() const { return *Begin1; }

private:
  BlockIter Begin1;
  BlockIter End1;
  BlockIter Begin2;
  BlockIter End2;
  BlockIter Begin3;
  BlockIter End3;
};

/// The implementation of the ExtTSP algorithm.
class ExtTSPImpl {
  using EdgeT = std::pair<uint64_t, uint64_t>;
  using EdgeCountMap = DenseMap<EdgeT, uint64_t>;

public:
  ExtTSPImpl(size_t NumNodes, const std::vector<uint64_t> &NodeSizes,
             const std::vector<uint64_t> &NodeCounts,
             const EdgeCountMap &EdgeCounts)
      : NumNodes(NumNodes) {
    initialize(NodeSizes, NodeCounts, EdgeCounts);
  }

  /// Run the algorithm and return an optimized ordering of blocks.
  void run(std::vector<uint64_t> &Result) {
    // Pass 1: Merge blocks with their mutually forced successors
    mergeForcedPairs();

    // Pass 2: Merge pairs of chains while improving the ExtTSP objective
    mergeChainPairs();

    // Pass 3: Merge cold blocks to reduce code size
    mergeColdChains();

    // Collect blocks from all chains
    concatChains(Result);
  }

private:
  /// Initialize the algorithm's data structures.
  void initialize(const std::vector<uint64_t> &NodeSizes,
                  const std::vector<uint64_t> &NodeCounts,
                  const EdgeCountMap &EdgeCounts) {
    // Initialize blocks
    AllBlocks.reserve(NumNodes);
    for (uint64_t Node = 0; Node < NumNodes; Node++) {
      uint64_t Size = std::max<uint64_t>(NodeSizes[Node], 1ULL);
      uint64_t ExecutionCount = NodeCounts[Node];
      // The execution count of the entry block is set to at least 1
      if (Node == 0 && ExecutionCount == 0)
        ExecutionCount = 1;
      AllBlocks.emplace_back(Node, Size, ExecutionCount);
    }

    // Initialize jumps between blocks
    SuccNodes = std::vector<std::vector<uint64_t>>(NumNodes);
    PredNodes = std::vector<std::vector<uint64_t>>(NumNodes);
    AllJumps.reserve(EdgeCounts.size());
    for (auto It : EdgeCounts) {
      auto Pred = It.first.first;
      auto Succ = It.first.second;
      // Ignore self-edges
      if (Pred == Succ)
        continue;

      SuccNodes[Pred].push_back(Succ);
      PredNodes[Succ].push_back(Pred);
      auto ExecutionCount = It.second;
      if (ExecutionCount > 0) {
        auto &Block = AllBlocks[Pred];
        auto &SuccBlock = AllBlocks[Succ];
        AllJumps.emplace_back(&Block, &SuccBlock, ExecutionCount);
        SuccBlock.InJumps.push_back(&AllJumps.back());
        Block.OutJumps.push_back(&AllJumps.back());
      }
    }

    // Initialize chains
    AllChains.reserve(NumNodes);
    HotChains.reserve(NumNodes);
    for (auto &Block : AllBlocks) {
      AllChains.emplace_back(Block.Index, &Block);
      Block.CurChain = &AllChains.back();
      if (Block.ExecutionCount > 0) {
        HotChains.push_back(&AllChains.back());
      }
    }

    // Initialize chain edges
    AllEdges.reserve(AllJumps.size());
    for (auto &Block : AllBlocks) {
      for (auto &Jump : Block.OutJumps) {
        auto SuccBlock = Jump->Target;
        auto CurEdge = Block.CurChain->getEdge(SuccBlock->CurChain);
        // this edge is already present in the graph
        if (CurEdge != nullptr) {
          assert(SuccBlock->CurChain->getEdge(Block.CurChain) != nullptr);
          CurEdge->appendJump(Jump);
          continue;
        }
        // this is a new edge
        AllEdges.emplace_back(Jump);
        Block.CurChain->addEdge(SuccBlock->CurChain, &AllEdges.back());
        SuccBlock->CurChain->addEdge(Block.CurChain, &AllEdges.back());
      }
    }
  }

  /// For a pair of blocks, A and B, block B is the forced successor of A,
  /// if (i) all jumps (based on profile) from A goes to B and (ii) all jumps
  /// to B are from A. Such blocks should be adjacent in the optimal ordering;
  /// the method finds and merges such pairs of blocks.
  void mergeForcedPairs() {
    // Find fallthroughs based on edge weights
    for (auto &Block : AllBlocks) {
      if (SuccNodes[Block.Index].size() == 1 &&
          PredNodes[SuccNodes[Block.Index][0]].size() == 1 &&
          SuccNodes[Block.Index][0] != 0) {
        size_t SuccIndex = SuccNodes[Block.Index][0];
        Block.ForcedSucc = &AllBlocks[SuccIndex];
        AllBlocks[SuccIndex].ForcedPred = &Block;
      }
    }

    // There might be 'cycles' in the forced dependencies, since profile
    // data isn't 100% accurate. Typically this is observed in loops, when the
    // loop edges are the hottest successors for the basic blocks of the loop.
    // Break the cycles by choosing the block with the smallest index as the
    // head. This helps to keep the original order of the loops, which likely
    // have already been rotated in the optimized manner.
    for (auto &Block : AllBlocks) {
      if (Block.ForcedSucc == nullptr || Block.ForcedPred == nullptr)
        continue;

      auto SuccBlock = Block.ForcedSucc;
      while (SuccBlock != nullptr && SuccBlock != &Block) {
        SuccBlock = SuccBlock->ForcedSucc;
      }
      if (SuccBlock == nullptr)
        continue;
      // Break the cycle
      AllBlocks[Block.ForcedPred->Index].ForcedSucc = nullptr;
      Block.ForcedPred = nullptr;
    }

    // Merge blocks with their fallthrough successors
    for (auto &Block : AllBlocks) {
      if (Block.ForcedPred == nullptr && Block.ForcedSucc != nullptr) {
        auto CurBlock = &Block;
        while (CurBlock->ForcedSucc != nullptr) {
          const auto NextBlock = CurBlock->ForcedSucc;
          mergeChains(Block.CurChain, NextBlock->CurChain, 0, MergeTypeTy::X_Y);
          CurBlock = NextBlock;
        }
      }
    }
  }

  /// Merge pairs of chains while improving the ExtTSP objective.
  void mergeChainPairs() {
    /// Deterministically compare pairs of chains
    auto compareChainPairs = [](const Chain *A1, const Chain *B1,
                                const Chain *A2, const Chain *B2) {
      if (A1 != A2)
        return A1->id() < A2->id();
      return B1->id() < B2->id();
    };

    while (HotChains.size() > 1) {
      Chain *BestChainPred = nullptr;
      Chain *BestChainSucc = nullptr;
      auto BestGain = MergeGainTy();
      // Iterate over all pairs of chains
      for (auto ChainPred : HotChains) {
        // Get candidates for merging with the current chain
        for (auto EdgeIter : ChainPred->edges()) {
          auto ChainSucc = EdgeIter.first;
          auto ChainEdge = EdgeIter.second;
          // Ignore loop edges
          if (ChainPred == ChainSucc)
            continue;

          // Stop early if the combined chain violates the maximum allowed size
          if (ChainPred->numBlocks() + ChainSucc->numBlocks() >= MaxChainSize)
            continue;

          // Compute the gain of merging the two chains
          auto CurGain = getBestMergeGain(ChainPred, ChainSucc, ChainEdge);
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

  /// Merge cold blocks to reduce code size.
  void mergeColdChains() {
    for (size_t SrcBB = 0; SrcBB < NumNodes; SrcBB++) {
      // Iterating over neighbors in the reverse order to make sure original
      // fallthrough jumps are merged first
      size_t NumSuccs = SuccNodes[SrcBB].size();
      for (size_t Idx = 0; Idx < NumSuccs; Idx++) {
        auto DstBB = SuccNodes[SrcBB][NumSuccs - Idx - 1];
        auto SrcChain = AllBlocks[SrcBB].CurChain;
        auto DstChain = AllBlocks[DstBB].CurChain;
        if (SrcChain != DstChain && !DstChain->isEntry() &&
            SrcChain->blocks().back()->Index == SrcBB &&
            DstChain->blocks().front()->Index == DstBB) {
          mergeChains(SrcChain, DstChain, 0, MergeTypeTy::X_Y);
        }
      }
    }
  }

  /// Compute the Ext-TSP score for a given block order and a list of jumps.
  double extTSPScore(const MergedChain &MergedBlocks,
                     const std::vector<Jump *> &Jumps) const {
    if (Jumps.empty())
      return 0.0;
    uint64_t CurAddr = 0;
    MergedBlocks.forEach([&](const Block *BB) {
      BB->EstimatedAddr = CurAddr;
      CurAddr += BB->Size;
    });

    double Score = 0;
    for (auto &Jump : Jumps) {
      const auto SrcBlock = Jump->Source;
      const auto DstBlock = Jump->Target;
      Score += ::extTSPScore(SrcBlock->EstimatedAddr, SrcBlock->Size,
                             DstBlock->EstimatedAddr, Jump->ExecutionCount);
    }
    return Score;
  }

  /// Compute the gain of merging two chains.
  ///
  /// The function considers all possible ways of merging two chains and
  /// computes the one having the largest increase in ExtTSP objective. The
  /// result is a pair with the first element being the gain and the second
  /// element being the corresponding merging type.
  MergeGainTy getBestMergeGain(Chain *ChainPred, Chain *ChainSucc,
                               ChainEdge *Edge) const {
    if (Edge->hasCachedMergeGain(ChainPred, ChainSucc)) {
      return Edge->getCachedMergeGain(ChainPred, ChainSucc);
    }

    // Precompute jumps between ChainPred and ChainSucc
    auto Jumps = Edge->jumps();
    auto EdgePP = ChainPred->getEdge(ChainPred);
    if (EdgePP != nullptr) {
      Jumps.insert(Jumps.end(), EdgePP->jumps().begin(), EdgePP->jumps().end());
    }
    assert(!Jumps.empty() && "trying to merge chains w/o jumps");

    // The object holds the best currently chosen gain of merging the two chains
    MergeGainTy Gain = MergeGainTy();

    /// Given a merge offset and a list of merge types, try to merge two chains
    /// and update Gain with a better alternative
    auto tryChainMerging = [&](size_t Offset,
                               const std::vector<MergeTypeTy> &MergeTypes) {
      // Skip merging corresponding to concatenation w/o splitting
      if (Offset == 0 || Offset == ChainPred->blocks().size())
        return;
      // Skip merging if it breaks Forced successors
      auto BB = ChainPred->blocks()[Offset - 1];
      if (BB->ForcedSucc != nullptr)
        return;
      // Apply the merge, compute the corresponding gain, and update the best
      // value, if the merge is beneficial
      for (auto &MergeType : MergeTypes) {
        Gain.updateIfLessThan(
            computeMergeGain(ChainPred, ChainSucc, Jumps, Offset, MergeType));
      }
    };

    // Try to concatenate two chains w/o splitting
    Gain.updateIfLessThan(
        computeMergeGain(ChainPred, ChainSucc, Jumps, 0, MergeTypeTy::X_Y));

    if (EnableChainSplitAlongJumps) {
      // Attach (a part of) ChainPred before the first block of ChainSucc
      for (auto &Jump : ChainSucc->blocks().front()->InJumps) {
        const auto SrcBlock = Jump->Source;
        if (SrcBlock->CurChain != ChainPred)
          continue;
        size_t Offset = SrcBlock->CurIndex + 1;
        tryChainMerging(Offset, {MergeTypeTy::X1_Y_X2, MergeTypeTy::X2_X1_Y});
      }

      // Attach (a part of) ChainPred after the last block of ChainSucc
      for (auto &Jump : ChainSucc->blocks().back()->OutJumps) {
        const auto DstBlock = Jump->Source;
        if (DstBlock->CurChain != ChainPred)
          continue;
        size_t Offset = DstBlock->CurIndex;
        tryChainMerging(Offset, {MergeTypeTy::X1_Y_X2, MergeTypeTy::Y_X2_X1});
      }
    }

    // Try to break ChainPred in various ways and concatenate with ChainSucc
    if (ChainPred->blocks().size() <= ChainSplitThreshold) {
      for (size_t Offset = 1; Offset < ChainPred->blocks().size(); Offset++) {
        // Try to split the chain in different ways. In practice, applying
        // X2_Y_X1 merging is almost never provides benefits; thus, we exclude
        // it from consideration to reduce the search space
        tryChainMerging(Offset, {MergeTypeTy::X1_Y_X2, MergeTypeTy::Y_X2_X1,
                                 MergeTypeTy::X2_X1_Y});
      }
    }
    Edge->setCachedMergeGain(ChainPred, ChainSucc, Gain);
    return Gain;
  }

  /// Compute the score gain of merging two chains, respecting a given
  /// merge 'type' and 'offset'.
  ///
  /// The two chains are not modified in the method.
  MergeGainTy computeMergeGain(const Chain *ChainPred, const Chain *ChainSucc,
                               const std::vector<Jump *> &Jumps,
                               size_t MergeOffset,
                               MergeTypeTy MergeType) const {
    auto MergedBlocks = mergeBlocks(ChainPred->blocks(), ChainSucc->blocks(),
                                    MergeOffset, MergeType);

    // Do not allow a merge that does not preserve the original entry block
    if ((ChainPred->isEntry() || ChainSucc->isEntry()) &&
        !MergedBlocks.getFirstBlock()->isEntry())
      return MergeGainTy();

    // The gain for the new chain
    auto NewGainScore = extTSPScore(MergedBlocks, Jumps) - ChainPred->score();
    return MergeGainTy(NewGainScore, MergeOffset, MergeType);
  }

  /// Merge two chains of blocks respecting a given merge 'type' and 'offset'.
  ///
  /// If MergeType == 0, then the result is a concatentation of two chains.
  /// Otherwise, the first chain is cut into two sub-chains at the offset,
  /// and merged using all possible ways of concatenating three chains.
  MergedChain mergeBlocks(const std::vector<Block *> &X,
                          const std::vector<Block *> &Y, size_t MergeOffset,
                          MergeTypeTy MergeType) const {
    // Split the first chain, X, into X1 and X2
    BlockIter BeginX1 = X.begin();
    BlockIter EndX1 = X.begin() + MergeOffset;
    BlockIter BeginX2 = X.begin() + MergeOffset;
    BlockIter EndX2 = X.end();
    BlockIter BeginY = Y.begin();
    BlockIter EndY = Y.end();

    // Construct a new chain from the three existing ones
    switch (MergeType) {
    case MergeTypeTy::X_Y:
      return MergedChain(BeginX1, EndX2, BeginY, EndY);
    case MergeTypeTy::X1_Y_X2:
      return MergedChain(BeginX1, EndX1, BeginY, EndY, BeginX2, EndX2);
    case MergeTypeTy::Y_X2_X1:
      return MergedChain(BeginY, EndY, BeginX2, EndX2, BeginX1, EndX1);
    case MergeTypeTy::X2_X1_Y:
      return MergedChain(BeginX2, EndX2, BeginX1, EndX1, BeginY, EndY);
    }
    llvm_unreachable("unexpected chain merge type");
  }

  /// Merge chain From into chain Into, update the list of active chains,
  /// adjacency information, and the corresponding cached values.
  void mergeChains(Chain *Into, Chain *From, size_t MergeOffset,
                   MergeTypeTy MergeType) {
    assert(Into != From && "a chain cannot be merged with itself");

    // Merge the blocks
    auto MergedBlocks =
        mergeBlocks(Into->blocks(), From->blocks(), MergeOffset, MergeType);
    Into->merge(From, MergedBlocks.getBlocks());
    Into->mergeEdges(From);
    From->clear();

    // Update cached ext-tsp score for the new chain
    auto SelfEdge = Into->getEdge(Into);
    if (SelfEdge != nullptr) {
      MergedBlocks = MergedChain(Into->blocks().begin(), Into->blocks().end());
      Into->setScore(extTSPScore(MergedBlocks, SelfEdge->jumps()));
    }

    // Remove chain From from the list of active chains
    auto Iter = std::remove(HotChains.begin(), HotChains.end(), From);
    HotChains.erase(Iter, HotChains.end());

    // Invalidate caches
    for (auto EdgeIter : Into->edges()) {
      EdgeIter.second->invalidateCache();
    }
  }

  /// Concatenate all chains into a final order of blocks.
  void concatChains(std::vector<uint64_t> &Order) {
    // Collect chains and calculate some stats for their sorting
    std::vector<Chain *> SortedChains;
    DenseMap<const Chain *, double> ChainDensity;
    for (auto &Chain : AllChains) {
      if (!Chain.blocks().empty()) {
        SortedChains.push_back(&Chain);
        // Using doubles to avoid overflow of ExecutionCount
        double Size = 0;
        double ExecutionCount = 0;
        for (auto Block : Chain.blocks()) {
          Size += static_cast<double>(Block->Size);
          ExecutionCount += static_cast<double>(Block->ExecutionCount);
        }
        assert(Size > 0 && "a chain of zero size");
        ChainDensity[&Chain] = ExecutionCount / Size;
      }
    }

    // Sorting chains by density in the decreasing order
    std::stable_sort(SortedChains.begin(), SortedChains.end(),
                     [&](const Chain *C1, const Chain *C2) {
                       // Makre sure the original entry block is at the
                       // beginning of the order
                       if (C1->isEntry() != C2->isEntry()) {
                         return C1->isEntry();
                       }

                       const double D1 = ChainDensity[C1];
                       const double D2 = ChainDensity[C2];
                       // Compare by density and break ties by chain identifiers
                       return (D1 != D2) ? (D1 > D2) : (C1->id() < C2->id());
                     });

    // Collect the blocks in the order specified by their chains
    Order.reserve(NumNodes);
    for (auto Chain : SortedChains) {
      for (auto Block : Chain->blocks()) {
        Order.push_back(Block->Index);
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

  /// All basic blocks.
  std::vector<Block> AllBlocks;

  /// All jumps between blocks.
  std::vector<Jump> AllJumps;

  /// All chains of basic blocks.
  std::vector<Chain> AllChains;

  /// All edges between chains.
  std::vector<ChainEdge> AllEdges;

  /// Active chains. The vector gets updated at runtime when chains are merged.
  std::vector<Chain *> HotChains;
};

} // end of anonymous namespace

std::vector<uint64_t> llvm::applyExtTspLayout(
    const std::vector<uint64_t> &NodeSizes,
    const std::vector<uint64_t> &NodeCounts,
    const DenseMap<std::pair<uint64_t, uint64_t>, uint64_t> &EdgeCounts) {
  size_t NumNodes = NodeSizes.size();

  // Verify correctness of the input data.
  assert(NodeCounts.size() == NodeSizes.size() && "Incorrect input");
  assert(NumNodes > 2 && "Incorrect input");

  // Apply the reordering algorithm.
  auto Alg = ExtTSPImpl(NumNodes, NodeSizes, NodeCounts, EdgeCounts);
  std::vector<uint64_t> Result;
  Alg.run(Result);

  // Verify correctness of the output.
  assert(Result.front() == 0 && "Original entry point is not preserved");
  assert(Result.size() == NumNodes && "Incorrect size of reordered layout");
  return Result;
}

double llvm::calcExtTspScore(
    const std::vector<uint64_t> &Order, const std::vector<uint64_t> &NodeSizes,
    const std::vector<uint64_t> &NodeCounts,
    const DenseMap<std::pair<uint64_t, uint64_t>, uint64_t> &EdgeCounts) {
  // Estimate addresses of the blocks in memory
  auto Addr = std::vector<uint64_t>(NodeSizes.size(), 0);
  for (size_t Idx = 1; Idx < Order.size(); Idx++) {
    Addr[Order[Idx]] = Addr[Order[Idx - 1]] + NodeSizes[Order[Idx - 1]];
  }

  // Increase the score for each jump
  double Score = 0;
  for (auto It : EdgeCounts) {
    auto Pred = It.first.first;
    auto Succ = It.first.second;
    uint64_t Count = It.second;
    Score += extTSPScore(Addr[Pred], NodeSizes[Pred], Addr[Succ], Count);
  }
  return Score;
}

double llvm::calcExtTspScore(
    const std::vector<uint64_t> &NodeSizes,
    const std::vector<uint64_t> &NodeCounts,
    const DenseMap<std::pair<uint64_t, uint64_t>, uint64_t> &EdgeCounts) {
  auto Order = std::vector<uint64_t>(NodeSizes.size());
  for (size_t Idx = 0; Idx < NodeSizes.size(); Idx++) {
    Order[Idx] = Idx;
  }
  return calcExtTspScore(Order, NodeSizes, NodeCounts, EdgeCounts);
}
