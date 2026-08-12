//==- BlockFrequencyInfoImpl.h - Block Frequency Implementation --*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared implementation of BlockFrequency for IR and Machine Instructions.
// See the documentation below for BlockFrequencyInfoImpl for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_BLOCKFREQUENCYINFOIMPL_H
#define LLVM_ANALYSIS_BLOCKFREQUENCYINFOIMPL_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/GenericCycleInfo.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/BlockFrequency.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ScaledNumber.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iterator>
#include <limits>
#include <list>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#define DEBUG_TYPE "block-freq"

namespace llvm {
extern LLVM_ABI llvm::cl::opt<bool> CheckBFIUnknownBlockQueries;

extern LLVM_ABI llvm::cl::opt<bool> UseIterativeBFIInference;
extern LLVM_ABI llvm::cl::opt<unsigned> IterativeBFIMaxIterationsPerBlock;
extern LLVM_ABI llvm::cl::opt<double> IterativeBFIPrecision;

class BranchProbabilityInfo;
class CycleInfo;
class Function;
class MachineBasicBlock;
class MachineBranchProbabilityInfo;
class MachineCycleInfo;
class MachineFunction;

namespace bfi_detail {

struct IrreducibleGraph;

/// Mass of a block.
///
/// This class implements a sort of fixed-point fraction always between 0.0 and
/// 1.0.  getMass() == std::numeric_limits<uint64_t>::max() indicates a value of
/// 1.0.
///
/// Masses can be added and subtracted.  Simple saturation arithmetic is used,
/// so arithmetic operations never overflow or underflow.
///
/// Masses can be multiplied.  Multiplication treats full mass as 1.0 and uses
/// an inexpensive floating-point algorithm that's off-by-one (almost, but not
/// quite, maximum precision).
///
/// Masses can be scaled by \a BranchProbability at maximum precision.
class BlockMass {
  uint64_t Mass = 0;

public:
  BlockMass() = default;
  explicit BlockMass(uint64_t Mass) : Mass(Mass) {}

  static BlockMass getEmpty() { return BlockMass(); }

  static BlockMass getFull() {
    return BlockMass(std::numeric_limits<uint64_t>::max());
  }

  uint64_t getMass() const { return Mass; }

  bool isFull() const { return Mass == std::numeric_limits<uint64_t>::max(); }
  bool isEmpty() const { return !Mass; }

  bool operator!() const { return isEmpty(); }

  /// Add another mass.
  ///
  /// Adds another mass, saturating at \a isFull() rather than overflowing.
  BlockMass &operator+=(BlockMass X) {
    uint64_t Sum = Mass + X.Mass;
    Mass = Sum < Mass ? std::numeric_limits<uint64_t>::max() : Sum;
    return *this;
  }

  /// Subtract another mass.
  ///
  /// Subtracts another mass, saturating at \a isEmpty() rather than
  /// undeflowing.
  BlockMass &operator-=(BlockMass X) {
    uint64_t Diff = Mass - X.Mass;
    Mass = Diff > Mass ? 0 : Diff;
    return *this;
  }

  BlockMass &operator*=(BranchProbability P) {
    Mass = P.scale(Mass);
    return *this;
  }

  bool operator==(BlockMass X) const { return Mass == X.Mass; }
  bool operator!=(BlockMass X) const { return Mass != X.Mass; }
  bool operator<=(BlockMass X) const { return Mass <= X.Mass; }
  bool operator>=(BlockMass X) const { return Mass >= X.Mass; }
  bool operator<(BlockMass X) const { return Mass < X.Mass; }
  bool operator>(BlockMass X) const { return Mass > X.Mass; }

  /// Convert to scaled number.
  ///
  /// Convert to \a ScaledNumber.  \a isFull() gives 1.0, while \a isEmpty()
  /// gives slightly above 0.0.
  LLVM_ABI ScaledNumber<uint64_t> toScaled() const;

  LLVM_ABI void dump() const;
  LLVM_ABI raw_ostream &print(raw_ostream &OS) const;
};

inline BlockMass operator+(BlockMass L, BlockMass R) {
  return BlockMass(L) += R;
}
inline BlockMass operator-(BlockMass L, BlockMass R) {
  return BlockMass(L) -= R;
}
inline BlockMass operator*(BlockMass L, BranchProbability R) {
  return BlockMass(L) *= R;
}
inline BlockMass operator*(BranchProbability L, BlockMass R) {
  return BlockMass(R) *= L;
}

inline raw_ostream &operator<<(raw_ostream &OS, BlockMass X) {
  return X.print(OS);
}

} // end namespace bfi_detail

/// Base class for BlockFrequencyInfoImpl
///
/// BlockFrequencyInfoImplBase has supporting data structures and some
/// algorithms for BlockFrequencyInfoImplBase.  Only algorithms that depend on
/// the block type (or that call such algorithms) are skipped here.
///
/// Nevertheless, the majority of the overall algorithm documentation lives with
/// BlockFrequencyInfoImpl.  See there for details.
class LLVM_ABI BlockFrequencyInfoImplBase {
public:
  using Scaled64 = ScaledNumber<uint64_t>;
  using BlockMass = bfi_detail::BlockMass;

  /// Representative of a block.
  ///
  /// This is a simple wrapper around an index into the reverse-post-order
  /// traversal of the blocks.
  ///
  /// Unlike a block pointer, its order has meaning (location in the
  /// topological sort) and it's class is the same regardless of block type.
  struct BlockNode {
    using IndexType = uint32_t;

    IndexType Index;

    BlockNode() : Index(std::numeric_limits<uint32_t>::max()) {}
    BlockNode(IndexType Index) : Index(Index) {}

    bool operator==(const BlockNode &X) const { return Index == X.Index; }
    bool operator!=(const BlockNode &X) const { return Index != X.Index; }
    bool operator<=(const BlockNode &X) const { return Index <= X.Index; }
    bool operator>=(const BlockNode &X) const { return Index >= X.Index; }
    bool operator<(const BlockNode &X) const { return Index < X.Index; }
    bool operator>(const BlockNode &X) const { return Index > X.Index; }

    bool isValid() const { return Index <= getMaxIndex(); }

    static size_t getMaxIndex() {
       return std::numeric_limits<uint32_t>::max() - 1;
    }
  };

  /// Stats about a block itself.
  struct FrequencyData {
    Scaled64 Scaled;
    uint64_t Integer;
  };

  /// Data about a loop.
  ///
  /// Contains the data necessary to represent a loop as a pseudo-node once it's
  /// packaged.
  struct LoopData {
    using ExitMap = SmallVector<std::pair<BlockNode, BlockMass>, 4>;
    using NodeList = SmallVector<BlockNode, 4>;
    using HeaderMassList = SmallVector<BlockMass, 1>;

    LoopData *Parent;            ///< The parent loop.
    bool IsPackaged = false;     ///< Whether this has been packaged.
    // Has an irreducible SCC in its own nodes; sub-loops package theirs first.
    bool ContainsIrreducible = false;
    // Headers are Nodes[0, NumHeaders), sorted. For an irreducible loop, the
    // SCC's entries and the nodes a retreating edge from a non-entry reaches.
    uint32_t NumHeaders = 1;
    ExitMap Exits;               ///< Successor edges (and weights).
    NodeList Nodes;              ///< Header and the members of the loop.
    HeaderMassList BackedgeMass; ///< Mass returned to each loop header.
    BlockMass Mass;
    Scaled64 Scale;

    LoopData(LoopData *Parent, const BlockNode &Header)
      : Parent(Parent), Nodes(1, Header), BackedgeMass(1) {}

    template <class It>
    LoopData(LoopData *Parent, It FirstHeader, It LastHeader)
        : Parent(Parent), Nodes(FirstHeader, LastHeader) {
      NumHeaders = Nodes.size();
      BackedgeMass.resize(NumHeaders);
    }

    template <class It1, class It2>
    LoopData(LoopData *Parent, It1 FirstHeader, It1 LastHeader, It2 FirstOther,
             It2 LastOther)
        : Parent(Parent), Nodes(FirstHeader, LastHeader) {
      NumHeaders = Nodes.size();
      Nodes.insert(Nodes.end(), FirstOther, LastOther);
      BackedgeMass.resize(NumHeaders);
    }

    bool isHeader(const BlockNode &Node) const {
      if (isIrreducible())
        return std::binary_search(Nodes.begin(), Nodes.begin() + NumHeaders,
                                  Node);
      return Node == Nodes[0];
    }

    BlockNode getHeader() const { return Nodes[0]; }
    bool isIrreducible() const { return NumHeaders > 1; }

    HeaderMassList::difference_type getHeaderIndex(const BlockNode &B) {
      assert(isHeader(B) && "this is only valid on loop header blocks");
      if (isIrreducible())
        return std::lower_bound(Nodes.begin(), Nodes.begin() + NumHeaders, B) -
               Nodes.begin();
      return 0;
    }

    NodeList::const_iterator members_begin() const {
      return Nodes.begin() + NumHeaders;
    }

    NodeList::const_iterator members_end() const { return Nodes.end(); }
    iterator_range<NodeList::const_iterator> members() const {
      return make_range(members_begin(), members_end());
    }
  };

  /// Index of loop information.
  struct WorkingData {
    BlockNode Node;           ///< This node.
    LoopData *Loop = nullptr; ///< The loop this block is inside.
    BlockMass Mass;           ///< Mass distribution from the entry block.

    WorkingData(const BlockNode &Node) : Node(Node) {}

    bool isLoopHeader() const { return Loop && Loop->isHeader(Node); }

    /// The innermost loop containing Node that Node does not head.
    ///
    /// A block can head several nested loops: createIrreducibleLoop() reuses
    /// an SCC's entry blocks as the irreducible loop's headers.
    LoopData *getContainingLoop() const {
      LoopData *L = Loop;
      while (L && L->isHeader(Node))
        L = L->Parent;
      return L;
    }

    /// Resolve a node to its representative.
    ///
    /// Get the node currently representing Node, which could be a containing
    /// loop.
    ///
    /// This function should only be called when distributing mass.  As long as
    /// there are no irreducible edges to Node, then it will have complexity
    /// O(1) in this context.
    ///
    /// In general, the complexity is O(L), where L is the number of loop
    /// headers Node has been packaged into.  Since this method is called in
    /// the context of distributing mass, L will be the number of loop headers
    /// an early exit edge jumps out of.
    BlockNode getResolvedNode() const {
      auto *L = getPackagedLoop();
      return L ? L->getHeader() : Node;
    }

    /// The outermost loop containing Node that is currently packaged, if any.
    ///
    /// Packaging is transient state: this answers what represents Node at the
    /// level being processed, not where Node sits in the loop nest.
    LoopData *getPackagedLoop() const {
      if (!Loop || !Loop->IsPackaged)
        return nullptr;
      auto *L = Loop;
      while (L->Parent && L->Parent->IsPackaged)
        L = L->Parent;
      return L;
    }

    /// The mass slot for Node: its own, or that of the outermost packaged
    /// loop it heads.
    BlockMass &getMass() {
      BlockMass *M = &Mass;
      for (LoopData *L = Loop; L && L->IsPackaged && L->isHeader(Node);
           L = L->Parent)
        M = &L->Mass;
      return *M;
    }

    /// Has ContainingLoop been packaged up?
    bool isPackaged() const { return getResolvedNode() != Node; }

    /// Has Loop been packaged up?
    bool isAPackage() const { return isLoopHeader() && Loop->IsPackaged; }
  };

  /// Unscaled probability weight.
  ///
  /// Probability weight for an edge in the graph (including the
  /// successor/target node).
  ///
  /// All edges in the original function are 32-bit.  However, exit edges from
  /// loop packages are taken from 64-bit exit masses, so we need 64-bits of
  /// space in general.
  ///
  /// In addition to the raw weight amount, Weight stores the type of the edge
  /// in the current context (i.e., the context of the loop being processed).
  /// Is this a local edge within the loop, an exit from the loop, or a
  /// backedge to the loop header?
  struct Weight {
    enum DistType { Local, Exit, Backedge };
    DistType Type = Local;
    BlockNode TargetNode;
    uint64_t Amount = 0;

    Weight() = default;
    Weight(DistType Type, BlockNode TargetNode, uint64_t Amount)
        : Type(Type), TargetNode(TargetNode), Amount(Amount) {}
  };

  /// Distribution of unscaled probability weight.
  ///
  /// Distribution of unscaled probability weight to a set of successors.
  ///
  /// This class collates the successor edge weights for later processing.
  ///
  /// \a DidOverflow indicates whether \a Total did overflow while adding to
  /// the distribution.  It should never overflow twice.
  struct Distribution {
    using WeightList = SmallVector<Weight, 4>;

    WeightList Weights;       ///< Individual successor weights.
    uint64_t Total = 0;       ///< Sum of all weights.
    bool DidOverflow = false; ///< Whether \a Total did overflow.

    Distribution() = default;

    void addLocal(const BlockNode &Node, uint64_t Amount) {
      add(Node, Amount, Weight::Local);
    }

    void addExit(const BlockNode &Node, uint64_t Amount) {
      add(Node, Amount, Weight::Exit);
    }

    void addBackedge(const BlockNode &Node, uint64_t Amount) {
      add(Node, Amount, Weight::Backedge);
    }

    /// Normalize the distribution.
    ///
    /// Combines multiple edges to the same \a Weight::TargetNode and scales
    /// down so that \a Total fits into 32-bits.
    ///
    /// This is linear in the size of \a Weights.  For the vast majority of
    /// cases, adjacent edge weights are combined by sorting WeightList and
    /// combining adjacent weights.  However, for very large edge lists an
    /// auxiliary hash table is used.
    LLVM_ABI void normalize();

  private:
    LLVM_ABI void add(const BlockNode &Node, uint64_t Amount,
                      Weight::DistType Type);
  };

  /// Data about each block.  This is used downstream.
  std::vector<FrequencyData> Freqs;

  /// Whether each block is an irreducible loop header.
  /// This is used downstream.
  SparseBitVector<> IsIrrLoopHeader;

  /// Loop data: see initializeLoops().
  std::vector<WorkingData> Working;

  /// Indexed information about loops.
  std::list<LoopData> Loops;

  /// Has an irreducible SCC outside every loop.
  bool TopContainsIrreducible = false;

  /// Virtual destructor.
  ///
  /// Need a virtual destructor to mask the compiler warning about
  /// getBlockName().
  virtual ~BlockFrequencyInfoImplBase() = default;

  /// Add all edges out of a packaged loop to the distribution.
  ///
  /// Adds all edges from LocalLoopHead to Dist.  Calls addToDist() to add each
  /// successor edge.
  void addLoopSuccessorsToDist(const LoopData *OuterLoop, LoopData &Loop,
                               Distribution &Dist);

  /// Add an edge to the distribution.
  ///
  /// Adds an edge to Succ to Dist.  If \c LoopHead.isValid(), then whether the
  /// edge is local/exit/backedge is in the context of LoopHead.  Otherwise,
  /// every edge should be a local edge (since all the loops are packaged up).
  void addToDist(Distribution &Dist, const LoopData *OuterLoop,
                 const BlockNode &Pred, const BlockNode &Succ, uint64_t Weight);

  /// Analyze irreducible SCCs.
  ///
  /// Separate irreducible SCCs from \c G, which is an explicit graph of \c
  /// OuterLoop (or the top-level function, if \c OuterLoop is \c nullptr).
  /// Insert them into \a Loops before \c Insert.
  ///
  /// \return the \c LoopData nodes representing the irreducible SCCs.
  iterator_range<std::list<LoopData>::iterator>
  analyzeIrreducible(const bfi_detail::IrreducibleGraph &G, LoopData *OuterLoop,
                     std::list<LoopData>::iterator Insert);

  /// Distribute mass according to a distribution.
  ///
  /// Distributes the mass in Source according to Dist.  If LoopHead.isValid(),
  /// backedges and exits are stored in its entry in Loops.
  ///
  /// Mass is distributed in parallel from two copies of the source mass.
  void distributeMass(const BlockNode &Source, LoopData *OuterLoop,
                      Distribution &Dist);

  /// Compute the loop scale for a loop.
  void computeLoopScale(LoopData &Loop);

  /// Adjust the mass of all headers in an irreducible loop.
  ///
  /// Initially, irreducible loops are assumed to distribute their mass
  /// equally among its headers. This can lead to wrong frequency estimates
  /// since some headers may be executed more frequently than others.
  ///
  /// This adjusts header mass distribution so it matches the weights of
  /// the backedges going into each of the loop headers.
  void adjustLoopHeaderMass(LoopData &Loop);

  void distributeIrrLoopHeaderMass(Distribution &Dist);

  /// Package up a loop.
  void packageLoop(LoopData &Loop);

  /// Unwrap loops.
  void unwrapLoops();

  /// Finalize frequency metrics.
  ///
  /// Calculates final frequencies and cleans up no-longer-needed data
  /// structures.
  void finalizeMetrics();

  /// Clear all memory.
  void clear();

  virtual std::string getBlockName(const BlockNode &Node) const;
  std::string getLoopName(const LoopData &Loop) const;

  virtual raw_ostream &print(raw_ostream &OS) const { return OS; }
  void dump() const { print(dbgs()); }

  Scaled64 getFloatingBlockFreq(const BlockNode &Node) const;

  BlockFrequency getBlockFreq(const BlockNode &Node) const;
  std::optional<uint64_t> getBlockProfileCount(const Function &F,
                                               const BlockNode &Node) const;
  std::optional<uint64_t> getProfileCountFromFreq(const Function &F,
                                                  BlockFrequency Freq) const;
  bool isIrrLoopHeader(const BlockNode &Node);

  void setBlockFreq(const BlockNode &Node, BlockFrequency Freq);

  BlockFrequency getEntryFreq() const {
    assert(!Freqs.empty());
    return BlockFrequency(Freqs[0].Integer);
  }
};

namespace bfi_detail {

template <class BlockT> struct TypeMap {};
template <> struct TypeMap<BasicBlock> {
  using BlockT = BasicBlock;
  using FunctionT = Function;
  using BranchProbabilityInfoT = BranchProbabilityInfo;
  using CycleInfoT = CycleInfo;
};
template <> struct TypeMap<MachineBasicBlock> {
  using BlockT = MachineBasicBlock;
  using FunctionT = MachineFunction;
  using BranchProbabilityInfoT = MachineBranchProbabilityInfo;
  using CycleInfoT = MachineCycleInfo;
};

/// Get the name of a MachineBasicBlock.
///
/// Get the name of a MachineBasicBlock.  It's templated so that including from
/// CodeGen is unnecessary (that would be a layering issue).
///
/// This is used mainly for debug output.  The name is similar to
/// MachineBasicBlock::getFullName(), but skips the name of the function.
template <class BlockT> std::string getBlockName(const BlockT *BB) {
  assert(BB && "Unexpected nullptr");
  auto MachineName = "BB" + Twine(BB->getNumber());
  if (BB->getBasicBlock())
    return (MachineName + "[" + BB->getName() + "]").str();
  return MachineName.str();
}
/// Get the name of a BasicBlock.
template <> inline std::string getBlockName(const BasicBlock *BB) {
  assert(BB && "Unexpected nullptr");
  return BB->getName().str();
}

/// Graph of irreducible control flow.
///
/// This graph is used for determining the SCCs in a loop (or top-level
/// function) that has irreducible control flow.
///
/// During the block frequency algorithm, the local graphs are defined in a
/// light-weight way, deferring to the \a BasicBlock or \a MachineBasicBlock
/// graphs for most edges, but getting others from \a LoopData::ExitMap.  The
/// latter only has successor information.
///
/// \a IrreducibleGraph makes this graph explicit.  It's in a form that can use
/// \a GraphTraits (so that \a analyzeIrreducible() can use \a scc_iterator),
/// and it explicitly lists predecessors and successors.  The initialization
/// that relies on \c MachineBasicBlock is defined in the header.
struct IrreducibleGraph {
  using BFIBase = BlockFrequencyInfoImplBase;

  BFIBase &BFI;

  using BlockNode = BFIBase::BlockNode;
  struct IrrNode {
    BlockNode Node;
    SmallVector<const IrrNode *, 4> Succs;

    IrrNode(const BlockNode &Node) : Node(Node) {}

    using iterator = SmallVectorImpl<const IrrNode *>::const_iterator;

    iterator succ_begin() const { return Succs.begin(); }
    iterator succ_end() const { return Succs.end(); }
  };
  BlockNode Start;
  const IrrNode *StartIrr = nullptr;
  std::vector<IrrNode> Nodes;
  SmallDenseMap<uint32_t, IrrNode *, 4> Lookup;

  /// The position of \p N in \a Nodes, for indexing side tables.
  unsigned getIndex(const IrrNode *N) const { return N - Nodes.data(); }

  /// Construct an explicit graph containing irreducible control flow.
  ///
  /// Construct an explicit graph of the control flow in \c OuterLoop (or the
  /// top-level function, if \c OuterLoop is \c nullptr).  Uses \c
  /// addBlockEdges to add block successors that have not been packaged into
  /// loops.
  ///
  /// \a BlockFrequencyInfoImpl::computeIrreducibleMass() is the only expected
  /// user of this.
  template <class BlockEdgesAdder>
  IrreducibleGraph(BFIBase &BFI, const BFIBase::LoopData *OuterLoop,
                   BlockEdgesAdder addBlockEdges) : BFI(BFI) {
    initialize(OuterLoop, addBlockEdges);
  }

  template <class BlockEdgesAdder>
  void initialize(const BFIBase::LoopData *OuterLoop,
                  BlockEdgesAdder addBlockEdges);
  LLVM_ABI void addNodesInLoop(const BFIBase::LoopData &OuterLoop);
  LLVM_ABI void addNodesInFunction();

  void addNode(const BlockNode &Node) {
    Nodes.emplace_back(Node);
    assert(BFI.Working[Node.Index].getMass().isEmpty() &&
           "mass distributed before the region was packaged");
  }

  LLVM_ABI void indexNodes();
  template <class BlockEdgesAdder>
  void addEdges(const BlockNode &Node, const BFIBase::LoopData *OuterLoop,
                BlockEdgesAdder addBlockEdges);
  LLVM_ABI void addEdge(IrrNode &Irr, const BlockNode &Succ,
                        const BFIBase::LoopData *OuterLoop);
};

template <class BlockEdgesAdder>
void IrreducibleGraph::initialize(const BFIBase::LoopData *OuterLoop,
                                  BlockEdgesAdder addBlockEdges) {
  if (OuterLoop) {
    addNodesInLoop(*OuterLoop);
    for (auto N : OuterLoop->Nodes)
      addEdges(N, OuterLoop, addBlockEdges);
  } else {
    addNodesInFunction();
    for (uint32_t Index = 0; Index < BFI.Working.size(); ++Index)
      addEdges(Index, OuterLoop, addBlockEdges);
  }
  StartIrr = Lookup[Start.Index];
}

template <class BlockEdgesAdder>
void IrreducibleGraph::addEdges(const BlockNode &Node,
                                const BFIBase::LoopData *OuterLoop,
                                BlockEdgesAdder addBlockEdges) {
  auto L = Lookup.find(Node.Index);
  if (L == Lookup.end())
    return;
  IrrNode &Irr = *L->second;
  const auto &Working = BFI.Working[Node.Index];

  if (Working.isAPackage())
    for (const auto &I : Working.Loop->Exits)
      addEdge(Irr, I.first, OuterLoop);
  else
    addBlockEdges(*this, Irr, OuterLoop);
}

} // end namespace bfi_detail

/// Shared implementation for block frequency analysis.
///
/// This is a shared implementation of BlockFrequencyInfo and
/// MachineBlockFrequencyInfo, and calculates the relative frequencies of
/// blocks.
///
/// LoopInfo defines a loop as a "non-trivial" SCC dominated by a single block,
/// which is called the header.  A given loop, L, can have sub-loops, which are
/// loops within the subgraph of L that exclude its header.  (A "trivial" SCC
/// consists of a single block that does not have a self-edge.)
///
/// In addition to loops, this algorithm has limited support for irreducible
/// SCCs, which are SCCs with multiple entry blocks.  Irreducible SCCs are
/// found from CycleInfo before any mass is distributed, and modelled as loops
/// with multiple headers.
///
/// The headers of irreducible sub-SCCs consist of its entry blocks and all
/// nodes that are targets of a backedge within it (excluding backedges within
/// true sub-loops).  Block frequency calculations act as if a block is
/// inserted that intercepts all the edges to the headers.  All backedges and
/// entries point to this block.  Its successors are the headers, which split
/// the frequency evenly.
///
/// This algorithm leverages BlockMass and ScaledNumber to maintain precision,
/// separates mass distribution from loop scaling, and dithers to eliminate
/// probability mass loss.
///
/// The implementation is split between BlockFrequencyInfoImpl, which knows the
/// type of graph being modelled (BasicBlock vs. MachineBasicBlock), and
/// BlockFrequencyInfoImplBase, which doesn't.  The base class uses \a
/// BlockNode, a wrapper around a uint32_t.  BlockNode is numbered from 0 in
/// reverse-post order.  This gives two advantages:  it's easy to compare the
/// relative ordering of two nodes, and maps keyed on BlockT can be represented
/// by vectors.
///
/// This algorithm is O(V+E), unless there is irreducible control flow, in
/// which case it's O(V*E) in the worst case.
///
/// These are the main stages:
///
///  0. Reverse post-order traversal (\a initializeRPOT()).
///
///     Run a single post-order traversal and save it (in reverse) in RPOT.
///     All other stages make use of this ordering.  Save a lookup from BlockT
///     to BlockNode (the index into RPOT) in Nodes.
///
///  1. Loop initialization (\a initializeLoops()).
///
///     Translate LoopInfo/MachineLoopInfo into a form suitable for the rest of
///     the algorithm.  In particular, store the immediate members of each loop
///     in reverse post-order.
///
///  2. Calculate mass and scale in loops (\a computeMassInLoops()).
///
///     For each loop (bottom-up), distribute mass through the DAG resulting
///     from ignoring backedges and treating sub-loops as a single pseudo-node.
///     Track the backedge mass distributed to the loop header, and use it to
///     calculate the loop scale (number of loop iterations).  Immediate
///     members that represent sub-loops will already have been visited and
///     packaged into a pseudo-node.
///
///     Distributing mass in a loop is a reverse-post-order traversal through
///     the loop.  Start by assigning full mass to the Loop header.  For each
///     node in the loop:
///
///         - Fetch and categorize the weight distribution for its successors.
///           If this is a packaged-subloop, the weight distribution is stored
///           in \a LoopData::Exits.  Otherwise, fetch it from
///           BranchProbabilityInfo.
///
///         - Each successor is categorized as \a Weight::Local, a local edge
///           within the current loop, \a Weight::Backedge, a backedge to the
///           loop header, or \a Weight::Exit, any successor outside the loop.
///           The weight, the successor, and its category are stored in \a
///           Distribution.  There can be multiple edges to each successor.
///           \a computeIrreducibleMass() has packaged up every irreducible SCC
///           by this point, so no backedge here targets a non-header.
///
///         - Normalize the distribution:  scale weights down so that their sum
///           is 32-bits, and coalesce multiple edges to the same node.
///
///         - Distribute the mass accordingly, dithering to minimize mass loss,
///           as described in \a distributeMass().
///
///     In the case of irreducible loops, instead of a single loop header,
///     there will be several. The computation of backedge masses is similar
///     but instead of having a single backedge mass, there will be one
///     backedge per loop header. In these cases, each backedge will carry
///     a mass proportional to the edge weights along the corresponding
///     path.
///
///     At the end of propagation, the full mass assigned to the loop will be
///     distributed among the loop headers proportionally according to the
///     mass flowing through their backedges.
///
///     Finally, calculate the loop scale from the accumulated backedge mass.
///
///  3. Distribute mass in the function (\a computeMassInFunction()).
///
///     Finally, distribute mass through the DAG resulting from packaging all
///     loops in the function.  This uses the same algorithm as distributing
///     mass in a loop, except that there are no exit or backedge edges.
///
///  4. Unpackage loops (\a unwrapLoops()).
///
///     Initialize each block's frequency to a floating point representation of
///     its mass.
///
///     Visit loops top-down, scaling the frequencies of its immediate members
///     by the loop's pseudo-node's frequency.
///
///  5. Convert frequencies to a 64-bit range (\a finalizeMetrics()).
///
///     Using the min and max frequencies as a guide, translate floating point
///     frequencies to an appropriate range in uint64_t.
///
/// It has some known flaws.
///
///   - The model of irreducible control flow is a rough approximation.
///
///     Modelling irreducible control flow exactly involves setting up and
///     solving a group of infinite geometric series.  Such precision is
///     unlikely to be worthwhile, since most of our algorithms give up on
///     irreducible control flow anyway.
///
///     Nevertheless, we might find that we need to get closer.  Here's a sort
///     of TODO list for the model with diminishing returns, to be completed as
///     necessary.
///
///       - The headers for the \a LoopData representing an irreducible SCC
///         include non-entry blocks.  When these extra blocks exist, they
///         indicate a self-contained irreducible sub-SCC.  We could treat them
///         as sub-loops, rather than arbitrarily shoving the problematic
///         blocks into the headers of the main irreducible SCC.
///
///       - Entry frequencies are assumed to be evenly split between the
///         headers of a given irreducible SCC, which is the only option if we
///         need to compute mass in the SCC before its parent loop.  Instead,
///         we could partially compute mass in the parent loop, and stop when
///         we get to the SCC.  Here, we have the correct ratio of entry
///         masses, which we can use to adjust their relative frequencies.
///         Compute mass in the SCC, and then continue propagation in the
///         parent.
///
///       - We can propagate mass iteratively through the SCC, for some fixed
///         number of iterations.  Each iteration starts by assigning the entry
///         blocks their backedge mass from the prior iteration.  The final
///         mass for each block (and each exit, and the total backedge mass
///         used for computing loop scale) is the sum of all iterations.
///         (Running this until fixed point would "solve" the geometric
///         series by simulation.)
template <class BT> class BlockFrequencyInfoImpl : BlockFrequencyInfoImplBase {
  using BlockT = typename bfi_detail::TypeMap<BT>::BlockT;
  using FunctionT = typename bfi_detail::TypeMap<BT>::FunctionT;
  using BranchProbabilityInfoT =
      typename bfi_detail::TypeMap<BT>::BranchProbabilityInfoT;
  using CycleInfoT = typename bfi_detail::TypeMap<BT>::CycleInfoT;
  using Successor = GraphTraits<const BlockT *>;
  using Predecessor = GraphTraits<Inverse<const BlockT *>>;

  const BranchProbabilityInfoT *BPI = nullptr;
  const CycleInfoT *CI = nullptr;
  const FunctionT *F = nullptr;

  // All blocks in reverse postorder.
  std::vector<const BlockT *> RPOT;
  /// Map from block number to number on RPOT/Freqs.
  SmallVector<BlockNode, 0> Nodes;
  unsigned BlockNumberEpoch;

  BlockNode getNode(const BlockT *BB) const {
    assert(BlockNumberEpoch ==
           GraphTraits<const FunctionT *>::getNumberEpoch(F));
    unsigned BlockNumber = GraphTraits<const BlockT *>::getNumber(BB);
    return BlockNumber < Nodes.size() ? Nodes[BlockNumber] : BlockNode();
  }

  const BlockT *getBlock(const BlockNode &Node) const {
    assert(Node.Index < RPOT.size());
    return RPOT[Node.Index];
  }

  /// Save a reverse post-order traversal of all the nodes.
  void initializeRPOT();

  /// Initialize loop data.
  ///
  /// Build up \a Loops using \a LoopInfo.  \a LoopInfo gives us a mapping from
  /// each block to the deepest loop it's in, but we need the inverse.  For each
  /// loop, we store in reverse post-order its "immediate" members, defined as
  /// the header, the headers of immediate sub-loops, and all other blocks in
  /// the loop that are not in sub-loops.
  void initializeLoops();

  /// Propagate to a block's successors.
  ///
  /// In the context of distributing mass through \c OuterLoop, divide the mass
  /// currently assigned to \c Node between its successors.
  void propagateMassToSuccessors(LoopData *OuterLoop, const BlockNode &Node);

  /// Compute mass in a particular loop.
  ///
  /// Assign mass to \c Loop's header, and then for each block in \c Loop in
  /// reverse post-order, distribute mass to its successors.  Only visits nodes
  /// that have not been packaged into sub-loops.
  ///
  /// \pre \a computeMassInLoop() has been called for each subloop of \c Loop,
  /// and \a computeIrreducibleMass() for \c Loop if it contains irreducible
  /// control flow.
  void computeMassInLoop(LoopData &Loop);

  /// Distribute mass in a multi-header loop, seeding the headers from
  /// irr_loop_header_weight metadata and marking them in IsIrrLoopHeader.
  void computeMassInIrreducibleLoop(LoopData &Loop);

  /// Compute mass in (and package up) irreducible SCCs.
  ///
  /// Find the irreducible SCCs in \c OuterLoop, add them to \a Loops (in front
  /// of \c Insert), and call \a computeMassInLoop() on each of them.
  ///
  /// If \c OuterLoop is \c nullptr, it refers to the top-level function.
  ///
  /// \pre \a computeMassInLoop() has been called for each subloop of \c
  /// OuterLoop.
  /// \pre \c OuterLoop has irreducible SCCs.
  void computeIrreducibleMass(LoopData *OuterLoop,
                              std::list<LoopData>::iterator Insert);

  /// Compute mass in all loops.
  ///
  /// For each loop bottom-up, call \a computeMassInLoop(), packaging
  /// irreducible SCCs first via \a computeIrreducibleMass() where \a
  /// initializeLoops() found them.
  void computeMassInLoops();

  /// Compute mass in the top-level function.
  ///
  /// Package up any top-level irreducible SCCs, assign mass to the entry
  /// block, and then for each block in reverse post-order, distribute mass to
  /// its successors.  Skips nodes that have been packaged into loops.
  ///
  /// \pre \a computeMassInLoops() has been called.
  void computeMassInFunction();

  std::string getBlockName(const BlockNode &Node) const override {
    return bfi_detail::getBlockName(getBlock(Node));
  }

  /// The current implementation for computing relative block frequencies does
  /// not handle correctly control-flow graphs containing irreducible loops. To
  /// resolve the problem, we apply a post-processing step, which iteratively
  /// updates block frequencies based on the frequencies of their predesessors.
  /// This corresponds to finding the stationary point of the Markov chain by
  /// an iterative method aka "PageRank computation".
  /// The algorithm takes at most O(|E| * IterativeBFIMaxIterations) steps but
  /// typically converges faster.
  ///
  /// Decide whether we want to apply iterative inference for a given function.
  bool needIterativeInference() const;

  /// Apply an iterative post-processing to infer correct counts for irr loops.
  void applyIterativeInference();

  using ProbMatrixType = std::vector<std::vector<std::pair<size_t, Scaled64>>>;

  /// Run iterative inference for a probability matrix and initial frequencies.
  void iterativeInference(const ProbMatrixType &ProbMatrix,
                          const BitVector &Blocks,
                          std::vector<Scaled64> &Freq) const;

  /// Find all blocks to apply inference on, that is, reachable from the entry
  /// and backward reachable from exits along edges with positive probability.
  void findReachableBlocks(BitVector &Blocks) const;

  /// Build a matrix of probabilities with transitions (edges) between the
  /// blocks: ProbMatrix[I] holds pairs (J, P), where Pr[J -> I | J] = P
  void initTransitionProbabilities(const BitVector &Blocks,
                                   ProbMatrixType &ProbMatrix) const;

#ifndef NDEBUG
  /// Compute the discrepancy between current block frequencies and the
  /// probability matrix.
  Scaled64 discrepancy(const ProbMatrixType &ProbMatrix,
                       const std::vector<Scaled64> &Freq) const;
#endif

public:
  BlockFrequencyInfoImpl() = default;

  const FunctionT *getFunction() const { return F; }

  void calculate(const FunctionT &F, const BranchProbabilityInfoT &BPI,
                 const CycleInfoT &CI);

  using BlockFrequencyInfoImplBase::getEntryFreq;

  BlockFrequency getBlockFreq(const BlockT *BB) const {
    return BlockFrequencyInfoImplBase::getBlockFreq(getNode(BB));
  }

  std::optional<uint64_t> getBlockProfileCount(const Function &F,
                                               const BlockT *BB) const {
    return BlockFrequencyInfoImplBase::getBlockProfileCount(F, getNode(BB));
  }

  std::optional<uint64_t> getProfileCountFromFreq(const Function &F,
                                                  BlockFrequency Freq) const {
    return BlockFrequencyInfoImplBase::getProfileCountFromFreq(F, Freq);
  }

  bool isIrrLoopHeader(const BlockT *BB) {
    return BlockFrequencyInfoImplBase::isIrrLoopHeader(getNode(BB));
  }

  void setBlockFreq(const BlockT *BB, BlockFrequency Freq);

  Scaled64 getFloatingBlockFreq(const BlockT *BB) const {
    return BlockFrequencyInfoImplBase::getFloatingBlockFreq(getNode(BB));
  }

  const BranchProbabilityInfoT &getBPI() const { return *BPI; }

  /// Print the frequencies for the current function.
  ///
  /// Prints the frequencies for the blocks in the current function.
  ///
  /// Blocks are printed in the natural iteration order of the function, rather
  /// than reverse post-order.  This provides two advantages:  writing -analyze
  /// tests is easier (since blocks come out in source order), and even
  /// unreachable blocks are printed.
  ///
  /// \a BlockFrequencyInfoImplBase::print() only knows reverse post-order, so
  /// we need to override it here.
  raw_ostream &print(raw_ostream &OS) const override;

  using BlockFrequencyInfoImplBase::dump;

  void verifyMatch(BlockFrequencyInfoImpl<BT> &Other) const;
};

template <class BT>
void BlockFrequencyInfoImpl<BT>::calculate(const FunctionT &F,
                                           const BranchProbabilityInfoT &BPI,
                                           const CycleInfoT &CI) {
  // Save the parameters.
  this->BPI = &BPI;
  this->CI = &CI;
  this->F = &F;

  // Clean up left-over data structures.
  BlockFrequencyInfoImplBase::clear();
  RPOT.clear();
  Nodes.clear();

  LLVM_DEBUG(dbgs() << "\nblock-frequency: " << F.getName()
                    << "\n================="
                    << std::string(F.getName().size(), '=') << "\n");

  // Mass flows over a DAG: loops are packaged into pseudo-nodes, and backedges
  // accumulate as loop mass instead of being followed.

  // Number blocks in reverse post-order; BlockNode comparisons use it.
  initializeRPOT();
  // Group blocks into the loops BFI represents, marking irreducible regions.
  initializeLoops();

  // Deepest loop first, so each is packaged before its parent needs it.
  computeMassInLoops();
  computeMassInFunction();
  // Unpackage, scaling members by the loop's iterations and package mass.
  unwrapLoops();
  // Apply a post-processing step improving computed frequencies for functions
  // with irreducible loops.
  if (needIterativeInference())
    applyIterativeInference();
  finalizeMetrics();

  if (CheckBFIUnknownBlockQueries) {
    // To detect BFI queries for unknown blocks, add entries for unreachable
    // blocks, if any. This is to distinguish between known/existing unreachable
    // blocks and unknown blocks.
    for (const BlockT &BB : F)
      if (!getNode(&BB).isValid())
        setBlockFreq(&BB, BlockFrequency());
  }

  RPOT.clear();
}

template <class BT>
void BlockFrequencyInfoImpl<BT>::setBlockFreq(const BlockT *BB,
                                              BlockFrequency Freq) {
  assert(BlockNumberEpoch == GraphTraits<const FunctionT *>::getNumberEpoch(F));
  unsigned BlockNumber = GraphTraits<const BlockT *>::getNumber(BB);
  if (Nodes.size() <= BlockNumber)
    Nodes.resize(GraphTraits<const FunctionT *>::getMaxNumber(F));
  BlockNode &Node = Nodes[BlockNumber];
  if (!Node.isValid()) {
    // If BB is a newly added block after BFI is done, we need to create a new
    // BlockNode for it assigned with a new index. The index can be determined
    // by the size of Freqs.
    Node = BlockNode(Freqs.size());
    Freqs.emplace_back();
  }
  BlockFrequencyInfoImplBase::setBlockFreq(Node, Freq);
}

template <class BT> void BlockFrequencyInfoImpl<BT>::initializeRPOT() {
  const BlockT *Entry = &F->front();
  RPOT.reserve(F->size());
  for (const BlockT *BB : post_order(Entry))
    RPOT.emplace_back(BB);
  std::reverse(RPOT.begin(), RPOT.end());

  assert(RPOT.size() - 1 <= BlockNode::getMaxIndex() &&
         "More nodes in function than Block Frequency Info supports");

  LLVM_DEBUG(dbgs() << "reverse-post-order-traversal\n");
  Nodes.resize(GraphTraits<const FunctionT *>::getMaxNumber(F));
  BlockNumberEpoch = GraphTraits<const FunctionT *>::getNumberEpoch(F);
  for (auto [Idx, Block] : enumerate(RPOT)) {
    BlockNode Node = BlockNode(Idx);
    LLVM_DEBUG(dbgs() << " - " << Idx << ": " << getBlockName(Node) << "\n");
    Nodes[GraphTraits<const BlockT *>::getNumber(Block)] = Node;
  }

  Working.reserve(RPOT.size());
  for (size_t Index = 0; Index < RPOT.size(); ++Index)
    Working.emplace_back(Index);
  Freqs.resize(RPOT.size());
}

template <class BT> void BlockFrequencyInfoImpl<BT>::initializeLoops() {
  LLVM_DEBUG(dbgs() << "loop-detection\n");

  LLVM_DEBUG(CI->print(dbgs()));

  // Whether \p C describes a loop for BFI. An entry of a cycle an edge
  // re-enters heads a loop the forest does not represent, because the cycle
  // absorbed it; which entry that is depends on the order the search found
  // them in. Represent none of them, so that equal entries stay equal, and
  // leave the region to the packaging computeIrreducibleMass does.
  auto hasLoop = [&](CycleRef C) {
    if (!CI->isReducible(C))
      return false;
    for (CycleRef A = CI->getParentCycle(C); A; A = CI->getParentCycle(A))
      if (!CI->isReducible(A) && CI->isEntry(A, CI->getHeader(C)))
        return false;
    return true;
  };

  // Visit loops top down and assign them an index.
  std::deque<std::pair<CycleRef, LoopData *>> Q;
  for (CycleRef C : CI->toplevel_cycles())
    Q.emplace_back(C, nullptr);
  if (Q.empty())
    return; // Early exit if there are no cycles.
  while (!Q.empty()) {
    CycleRef Cycle = Q.front().first;
    LoopData *Parent = Q.front().second;
    Q.pop_front();

    if (hasLoop(Cycle)) {
      BlockNode Header = getNode(CI->getHeader(Cycle));
      Loops.emplace_back(Parent, Header);

      Working[Header.Index].Loop = &Loops.back();
      LLVM_DEBUG(dbgs() << " - loop = " << getBlockName(Header) << "\n");
      Parent = &Loops.back();
    } else if (!CI->isReducible(Cycle)) {
      // No LoopData yet; ask computeIrreducibleMass to package the SCC
      // that contains this cycle.
      if (Parent)
        Parent->ContainsIrreducible = true;
      else
        TopContainsIrreducible = true;
    }

    for (CycleRef C : CI->children(Cycle))
      Q.emplace_back(C, Parent);
  }

  // Visit nodes in reverse post-order and add them to their deepest containing
  // loop.
  for (size_t Index = 0; Index < RPOT.size(); ++Index) {
    // Loop headers have already been mostly mapped.
    if (Working[Index].isLoopHeader()) {
      LoopData *ContainingLoop = Working[Index].getContainingLoop();
      if (ContainingLoop)
        ContainingLoop->Nodes.push_back(Index);
      continue;
    }

    CycleRef Cycle = CI->getCycle(RPOT[Index]);
    while (Cycle && !hasLoop(Cycle))
      Cycle = CI->getParentCycle(Cycle);
    if (!Cycle)
      continue;

    // Add this node to its containing loop's member list.
    BlockNode Header = getNode(CI->getHeader(Cycle));
    assert(Header.isValid());
    const auto &HeaderData = Working[Header.Index];
    assert(HeaderData.isLoopHeader());

    Working[Index].Loop = HeaderData.Loop;
    HeaderData.Loop->Nodes.push_back(Index);
    LLVM_DEBUG(dbgs() << " - loop = " << getBlockName(Header)
                      << ": member = " << getBlockName(Index) << "\n");
  }
}

template <class BT> void BlockFrequencyInfoImpl<BT>::computeMassInLoops() {
  // Visit loops with the deepest first, and the top-level loops last.
  // computeIrreducibleMass inserts each new loop immediately after *L.
  for (auto L = Loops.end(), B = Loops.begin(); L != B;) {
    --L;
    if (L->ContainsIrreducible)
      computeIrreducibleMass(&*L, std::next(L));
    computeMassInLoop(*L);
  }
}

template <class BT>
void BlockFrequencyInfoImpl<BT>::computeMassInLoop(LoopData &Loop) {
  LLVM_DEBUG(dbgs() << "compute-mass-in-loop: " << getLoopName(Loop) << "\n");

  if (Loop.isIrreducible()) {
    LLVM_DEBUG(dbgs() << "isIrreducible = true\n");
    computeMassInIrreducibleLoop(Loop);
  } else {
    Working[Loop.getHeader().Index].getMass() = BlockMass::getFull();
    propagateMassToSuccessors(&Loop, Loop.getHeader());
    for (const BlockNode &M : Loop.members())
      propagateMassToSuccessors(&Loop, M);
  }

  computeLoopScale(Loop);
  packageLoop(Loop);
}

template <class BT>
void BlockFrequencyInfoImpl<BT>::computeMassInIrreducibleLoop(LoopData &Loop) {
  Distribution Dist;
  unsigned NumHeadersWithWeight = 0;
  std::optional<uint64_t> MinHeaderWeight;
  DenseSet<uint32_t> HeadersWithoutWeight;
  HeadersWithoutWeight.reserve(Loop.NumHeaders);
  for (uint32_t H = 0; H < Loop.NumHeaders; ++H) {
    auto &HeaderNode = Loop.Nodes[H];
    const BlockT *Block = getBlock(HeaderNode);
    IsIrrLoopHeader.set(Loop.Nodes[H].Index);
    std::optional<uint64_t> HeaderWeight = Block->getIrrLoopHeaderWeight();
    if (!HeaderWeight) {
      LLVM_DEBUG(dbgs() << "Missing irr loop header metadata on "
                        << getBlockName(HeaderNode) << "\n");
      HeadersWithoutWeight.insert(H);
      continue;
    }
    LLVM_DEBUG(dbgs() << getBlockName(HeaderNode)
                      << " has irr loop header weight " << *HeaderWeight
                      << "\n");
    NumHeadersWithWeight++;
    uint64_t HeaderWeightValue = *HeaderWeight;
    if (!MinHeaderWeight || HeaderWeightValue < MinHeaderWeight)
      MinHeaderWeight = HeaderWeightValue;
    if (HeaderWeightValue) {
      Dist.addLocal(HeaderNode, HeaderWeightValue);
    }
  }
  // As a heuristic, if some headers don't have a weight, give them the
  // minimum weight seen (not to disrupt the existing trends too much by
  // using a weight that's in the general range of the other headers' weights,
  // and the minimum seems to perform better than the average.)
  // FIXME: better update in the passes that drop the header weight.
  // If no headers have a weight, give them even weight (use weight 1).
  if (!MinHeaderWeight)
    MinHeaderWeight = 1;
  for (uint32_t H : HeadersWithoutWeight) {
    auto &HeaderNode = Loop.Nodes[H];
    assert(!getBlock(HeaderNode)->getIrrLoopHeaderWeight() &&
           "Shouldn't have a weight metadata");
    uint64_t MinWeight = *MinHeaderWeight;
    LLVM_DEBUG(dbgs() << "Giving weight " << MinWeight << " to "
                      << getBlockName(HeaderNode) << "\n");
    if (MinWeight)
      Dist.addLocal(HeaderNode, MinWeight);
  }
  distributeIrrLoopHeaderMass(Dist);
  // Seeded headers are ordered first. Any retreating edge from a non-header
  // targets a header.
  for (const BlockNode &M : Loop.Nodes)
    propagateMassToSuccessors(&Loop, M);
  if (NumHeadersWithWeight == 0)
    // No headers have a metadata. Adjust header mass.
    adjustLoopHeaderMass(Loop);
}

template <class BT> void BlockFrequencyInfoImpl<BT>::computeMassInFunction() {
  if (TopContainsIrreducible)
    computeIrreducibleMass(nullptr, Loops.begin());

  LLVM_DEBUG(dbgs() << "compute-mass-in-function\n");
  assert(!Working.empty() && "no blocks in function");
  assert(!Working[0].isLoopHeader() && "entry block is a loop header");

  Working[0].getMass() = BlockMass::getFull();
  for (size_t i = 0, n = RPOT.size(); i != n; ++i) {
    // Check for nodes that have been packaged.
    if (Working[i].isPackaged())
      continue;

    propagateMassToSuccessors(nullptr, BlockNode(i));
  }
}

template <class BT>
bool BlockFrequencyInfoImpl<BT>::needIterativeInference() const {
  if (!UseIterativeBFIInference)
    return false;
  if (!F->getFunction().hasProfileData())
    return false;
  // Apply iterative inference only if the function contains irreducible loops;
  // otherwise, computed block frequencies are reasonably correct.
  for (auto L = Loops.rbegin(), E = Loops.rend(); L != E; ++L) {
    if (L->isIrreducible())
      return true;
  }
  return false;
}

template <class BT> void BlockFrequencyInfoImpl<BT>::applyIterativeInference() {
  // Extract blocks for processing: a block is considered for inference iff it
  // can be reached from the entry by edges with a positive probability.
  // Non-processed blocks are assigned with the zero frequency and are ignored
  // in the computation
  BitVector ReachableBlocks;
  findReachableBlocks(ReachableBlocks);
  if (ReachableBlocks.none())
    return;

  // Extract initial frequencies for the reachable blocks
  auto Freq = std::vector<Scaled64>(ReachableBlocks.size());
  Scaled64 SumFreq;
  for (const BlockT &BB : *F) {
    unsigned Number = GraphTraits<const BlockT *>::getNumber(&BB);
    if (!ReachableBlocks[Number])
      continue;
    Freq[Number] = getFloatingBlockFreq(&BB);
    SumFreq += Freq[Number];
  }
  assert(!SumFreq.isZero() && "empty initial block frequencies");

  LLVM_DEBUG(dbgs() << "Applying iterative inference for " << F->getName()
                    << " with " << ReachableBlocks.count() << " blocks\n");

  // Normalizing frequencies so they sum up to 1.0
  for (auto &Value : Freq) {
    Value /= SumFreq;
  }

  // Setting up edge probabilities using sparse matrix representation:
  // ProbMatrix[I] holds a vector of pairs (J, P) where Pr[J -> I | J] = P
  ProbMatrixType ProbMatrix;
  initTransitionProbabilities(ReachableBlocks, ProbMatrix);

  // Run the propagation
  iterativeInference(ProbMatrix, ReachableBlocks, Freq);

  // Assign computed frequency values
  for (const BlockT &BB : *F) {
    auto Node = getNode(&BB);
    if (!Node.isValid())
      continue;
    unsigned Number = GraphTraits<const BlockT *>::getNumber(&BB);
    Freqs[Node.Index].Scaled =
        ReachableBlocks[Number] ? Freq[Number] : Scaled64::getZero();
  }
}

template <class BT>
void BlockFrequencyInfoImpl<BT>::iterativeInference(
    const ProbMatrixType &ProbMatrix, const BitVector &Blocks,
    std::vector<Scaled64> &Freq) const {
  assert(0.0 < IterativeBFIPrecision && IterativeBFIPrecision < 1.0 &&
         "incorrectly specified precision");
  // Convert double precision to Scaled64
  const auto Precision =
      Scaled64::getInverse(static_cast<uint64_t>(1.0 / IterativeBFIPrecision));
  const size_t MaxIterations =
      IterativeBFIMaxIterationsPerBlock * Blocks.count();

#ifndef NDEBUG
  LLVM_DEBUG(dbgs() << "  Initial discrepancy = "
                    << discrepancy(ProbMatrix, Freq).toString() << "\n");
#endif

  // Successors[I] holds unique sucessors of the I-th block
  auto Successors = std::vector<std::vector<size_t>>(Freq.size());
  for (size_t I = 0; I < Freq.size(); I++) {
    for (const auto &Jump : ProbMatrix[I]) {
      Successors[Jump.first].push_back(I);
    }
  }

  // To speedup computation, we maintain a set of "active" blocks whose
  // frequencies need to be updated based on the incoming edges.
  // The set is dynamic and changes after every update. Initially all blocks
  // with a positive frequency are active
  auto IsActive = BitVector(Freq.size(), false);
  std::queue<size_t> ActiveSet;
  for (unsigned I : Blocks.set_bits()) {
    if (Freq[I] > 0) {
      ActiveSet.push(I);
      IsActive[I] = true;
    }
  }

  // Iterate over the blocks propagating frequencies
  size_t It = 0;
  while (It++ < MaxIterations && !ActiveSet.empty()) {
    size_t I = ActiveSet.front();
    ActiveSet.pop();
    IsActive[I] = false;

    // Compute a new frequency for the block: NewFreq := Freq \times ProbMatrix.
    // A special care is taken for self-edges that needs to be scaled by
    // (1.0 - SelfProb), where SelfProb is the sum of probabilities on the edges
    Scaled64 NewFreq;
    Scaled64 OneMinusSelfProb = Scaled64::getOne();
    for (const auto &Jump : ProbMatrix[I]) {
      if (Jump.first == I) {
        OneMinusSelfProb -= Jump.second;
      } else {
        NewFreq += Freq[Jump.first] * Jump.second;
      }
    }
    if (OneMinusSelfProb != Scaled64::getOne())
      NewFreq /= OneMinusSelfProb;

    // If the block's frequency has changed enough, then
    // make sure the block and its successors are in the active set
    auto Change = Freq[I] >= NewFreq ? Freq[I] - NewFreq : NewFreq - Freq[I];
    if (Change > Precision) {
      ActiveSet.push(I);
      IsActive[I] = true;
      for (size_t Succ : Successors[I]) {
        if (!IsActive[Succ]) {
          ActiveSet.push(Succ);
          IsActive[Succ] = true;
        }
      }
    }

    // Update the frequency for the block
    Freq[I] = NewFreq;
  }

  LLVM_DEBUG(dbgs() << "  Completed " << It << " inference iterations"
                    << format(" (%0.0f per block)", double(It) / Freq.size())
                    << "\n");
#ifndef NDEBUG
  LLVM_DEBUG(dbgs() << "  Final   discrepancy = "
                    << discrepancy(ProbMatrix, Freq).toString() << "\n");
#endif
}

template <class BT>
void BlockFrequencyInfoImpl<BT>::findReachableBlocks(BitVector &Blocks) const {
  unsigned MaxNumber = GraphTraits<const FunctionT *>::getMaxNumber(F);
  auto number = [](const BlockT *BB) {
    return GraphTraits<const BlockT *>::getNumber(BB);
  };

  // Find all blocks to apply inference on, that is, reachable from the entry
  // along edges with non-zero probablities
  std::queue<const BlockT *> Queue;
  BitVector Reachable(MaxNumber);
  const BlockT *Entry = &F->front();
  Queue.push(Entry);
  Reachable.set(number(Entry));
  while (!Queue.empty()) {
    const BlockT *SrcBB = Queue.front();
    Queue.pop();
    for (auto It : enumerate(children<const BlockT *>(SrcBB))) {
      auto EP = BPI->getEdgeProbability(SrcBB, It.index());
      if (EP.isZero())
        continue;
      unsigned Number = number(It.value());
      if (!Reachable.test(Number)) {
        Reachable.set(Number);
        Queue.push(It.value());
      }
    }
  }

  // Find all blocks to apply inference on, that is, backward reachable from
  // the entry along (backward) edges with non-zero probablities
  BitVector InverseReachable(MaxNumber);
  for (const BlockT &BB : *F) {
    // An exit block is a block without any successors
    bool HasSucc = !llvm::children<const BlockT *>(&BB).empty();
    if (!HasSucc && Reachable.test(number(&BB))) {
      Queue.push(&BB);
      InverseReachable.set(number(&BB));
    }
  }
  while (!Queue.empty()) {
    const BlockT *SrcBB = Queue.front();
    Queue.pop();
    for (const BlockT *DstBB : inverse_children<const BlockT *>(SrcBB)) {
      auto EP = BPI->getEdgeProbability(DstBB, SrcBB);
      if (EP.isZero())
        continue;
      unsigned Number = number(DstBB);
      if (!InverseReachable.test(Number)) {
        InverseReachable.set(Number);
        Queue.push(DstBB);
      }
    }
  }

  // Collect the result
  Reachable &= InverseReachable;
  Blocks = std::move(Reachable);
}

template <class BT>
void BlockFrequencyInfoImpl<BT>::initTransitionProbabilities(
    const BitVector &Blocks, ProbMatrixType &ProbMatrix) const {
  const size_t NumBlocks = Blocks.size();
  auto Succs = std::vector<std::vector<std::pair<size_t, Scaled64>>>(NumBlocks);
  auto SumProb = std::vector<Scaled64>(NumBlocks);

  // Find unique successors and corresponding probabilities for every block
  for (const BlockT &BB : *F) {
    size_t Src = GraphTraits<const BlockT *>::getNumber(&BB);
    if (!Blocks[Src])
      continue;
    SmallPtrSet<const BlockT *, 2> UniqueSuccs;
    for (auto It : enumerate(children<const BlockT *>(&BB))) {
      const BlockT *SI = It.value();
      size_t Dst = GraphTraits<const BlockT *>::getNumber(SI);
      // Ignore cold blocks
      if (!Blocks[Dst])
        continue;
      // Ignore parallel edges between BB and SI blocks
      if (!UniqueSuccs.insert(SI).second)
        continue;
      // Ignore jumps with zero probability
      auto EP = BPI->getEdgeProbability(&BB, It.index());
      if (EP.isZero())
        continue;

      auto EdgeProb =
          Scaled64::getFraction(EP.getNumerator(), EP.getDenominator());
      Succs[Src].push_back(std::make_pair(Dst, EdgeProb));
      SumProb[Src] += EdgeProb;
    }
  }

  // Add transitions for every jump with positive branch probability
  ProbMatrix = ProbMatrixType(NumBlocks);
  for (size_t Src = 0; Src < NumBlocks; Src++) {
    // Ignore blocks w/o successors
    if (Succs[Src].empty())
      continue;

    assert(!SumProb[Src].isZero() && "Zero sum probability of non-exit block");
    for (auto &Jump : Succs[Src]) {
      size_t Dst = Jump.first;
      Scaled64 Prob = Jump.second;
      ProbMatrix[Dst].push_back(std::make_pair(Src, Prob / SumProb[Src]));
    }
  }

  // Add transitions from sinks to the source
  size_t EntryIdx = GraphTraits<const BlockT *>::getNumber(&F->front());
  for (size_t Src = 0; Src < NumBlocks; Src++) {
    if (Blocks[Src] && Succs[Src].empty()) {
      ProbMatrix[EntryIdx].push_back(std::make_pair(Src, Scaled64::getOne()));
    }
  }
}

#ifndef NDEBUG
template <class BT>
BlockFrequencyInfoImplBase::Scaled64 BlockFrequencyInfoImpl<BT>::discrepancy(
    const ProbMatrixType &ProbMatrix, const std::vector<Scaled64> &Freq) const {
  size_t EntryIdx = GraphTraits<const BlockT *>::getNumber(&F->front());
  assert(Freq[EntryIdx] > 0 &&
         "Incorrectly computed frequency of the entry block");
  Scaled64 Discrepancy;
  for (size_t I = 0; I < ProbMatrix.size(); I++) {
    Scaled64 Sum;
    for (const auto &Jump : ProbMatrix[I]) {
      Sum += Freq[Jump.first] * Jump.second;
    }
    Discrepancy += Freq[I] >= Sum ? Freq[I] - Sum : Sum - Freq[I];
  }
  // Normalizing by the frequency of the entry block
  return Discrepancy / Freq[EntryIdx];
}
#endif

template <class BT>
void BlockFrequencyInfoImpl<BT>::computeIrreducibleMass(
    LoopData *OuterLoop, std::list<LoopData>::iterator Insert) {
  LLVM_DEBUG(dbgs() << "analyze-irreducible-in-";
             if (OuterLoop) dbgs()
             << "loop: " << getLoopName(*OuterLoop) << "\n";
             else dbgs() << "function\n");

  using namespace bfi_detail;

  auto addBlockEdges = [&](IrreducibleGraph &G, IrreducibleGraph::IrrNode &Irr,
                           const LoopData *OuterLoop) {
    const BlockT *BB = RPOT[Irr.Node.Index];
    for (const auto *Succ : children<const BlockT *>(BB))
      G.addEdge(Irr, getNode(Succ), OuterLoop);
  };
  IrreducibleGraph G(*this, OuterLoop, addBlockEdges);

  for (auto &L : analyzeIrreducible(G, OuterLoop, Insert))
    computeMassInLoop(L);

  if (!OuterLoop)
    return;

  // Drop the nodes the new packages absorbed.
  assert(OuterLoop->Exits.empty() && "unexpected exits before distribution");
  assert(llvm::all_of(OuterLoop->BackedgeMass,
                      [](BlockMass M) { return M.isEmpty(); }) &&
         "unexpected backedge mass before distribution");
  auto O = OuterLoop->Nodes.begin() + 1;
  for (auto I = O, E = OuterLoop->Nodes.end(); I != E; ++I)
    if (!Working[I->Index].isPackaged())
      *O++ = *I;
  OuterLoop->Nodes.erase(O, OuterLoop->Nodes.end());
}

// A helper function that converts a branch probability into weight.
inline uint32_t getWeightFromBranchProb(const BranchProbability Prob) {
  return Prob.getNumerator();
}

template <class BT>
void BlockFrequencyInfoImpl<BT>::propagateMassToSuccessors(
    LoopData *OuterLoop, const BlockNode &Node) {
  LLVM_DEBUG(dbgs() << " - node: " << getBlockName(Node) << "\n");
  // Calculate probability for successors.
  Distribution Dist;
  if (auto *Loop = Working[Node.Index].getPackagedLoop()) {
    assert(Loop != OuterLoop && "Cannot propagate mass in a packaged loop");
    addLoopSuccessorsToDist(OuterLoop, *Loop, Dist);
  } else {
    const BlockT *BB = getBlock(Node);
    for (auto It : enumerate(children<const BlockT *>(BB)))
      addToDist(
          Dist, OuterLoop, Node, getNode(It.value()),
          getWeightFromBranchProb(BPI->getEdgeProbability(BB, It.index())));
  }

  // Distribute mass to successors, saving exit and backedge data in the
  // loop header.
  distributeMass(Node, OuterLoop, Dist);
}

template <class BT>
raw_ostream &BlockFrequencyInfoImpl<BT>::print(raw_ostream &OS) const {
  if (!F)
    return OS;
  OS << "block-frequency-info: " << F->getName() << "\n";
  for (const BlockT &BB : *F) {
    OS << " - " << bfi_detail::getBlockName(&BB) << ": float = ";
    getFloatingBlockFreq(&BB).print(OS, 5)
        << ", int = " << getBlockFreq(&BB).getFrequency();
    if (std::optional<uint64_t> ProfileCount =
        BlockFrequencyInfoImplBase::getBlockProfileCount(
            F->getFunction(), getNode(&BB)))
      OS << ", count = " << *ProfileCount;
    if (std::optional<uint64_t> IrrLoopHeaderWeight =
            BB.getIrrLoopHeaderWeight())
      OS << ", irr_loop_header_weight = " << *IrrLoopHeaderWeight;
    OS << "\n";
  }

  // Add an extra newline for readability.
  OS << "\n";
  return OS;
}

template <class BT>
void BlockFrequencyInfoImpl<BT>::verifyMatch(
    BlockFrequencyInfoImpl<BT> &Other) const {
  bool Match = true;
  // Gather blocks for numbers so that we can print names and determine whether
  // they still exist.
  SmallVector<const BlockT *> Blocks;
  Blocks.resize(GraphTraits<const FunctionT *>::getMaxNumber(F));
  for (const auto &BB : *F)
    Blocks[GraphTraits<const BlockT *>::getNumber(&BB)] = &BB;

  size_t MinSize = std::min(Nodes.size(), Other.Nodes.size());
  for (size_t i = 0; i < MinSize; ++i) {
    if (!Blocks[i])
      continue; // Block got deleted in the mean time, ignore.
    if (Nodes[i].isValid() != Other.Nodes[i].isValid()) {
      Match = false;
      dbgs() << "Block " << bfi_detail::getBlockName(Blocks[i])
             << " existence mismatch.\n";
    } else if (Nodes[i].isValid()) {
      const auto &Freq = Freqs[Nodes[i].Index];
      const auto &OtherFreq = Other.Freqs[Other.Nodes[i].Index];
      if (Freq.Integer != OtherFreq.Integer) {
        Match = false;
        dbgs() << "Freq mismatch: " << bfi_detail::getBlockName(Blocks[i])
               << " " << Freq.Integer << " vs " << OtherFreq.Integer << "\n";
      }
    }
  }
  // Block with higher numbers must not exist in either state.
  for (size_t i = MinSize; i < Nodes.size(); ++i) {
    if (Nodes[i].isValid()) {
      Match = false;
      dbgs() << "Block " << bfi_detail::getBlockName(Blocks[i])
             << " existence mismatch.\n";
    }
  }
  for (size_t i = MinSize; i < Other.Nodes.size(); ++i) {
    if (Other.Nodes[i].isValid()) {
      Match = false;
      dbgs() << "Block " << bfi_detail::getBlockName(Blocks[i])
             << " existence mismatch.\n";
    }
  }

  if (!Match) {
    dbgs() << "This\n";
    print(dbgs());
    dbgs() << "Other\n";
    Other.print(dbgs());
  }
  assert(Match && "BFI mismatch");
}

// Graph trait base class for block frequency information graph
// viewer.

enum GVDAGType { GVDT_None, GVDT_Fraction, GVDT_Integer, GVDT_Count };

template <class BlockFrequencyInfoT, class BranchProbabilityInfoT>
struct BFIDOTGraphTraitsBase : public DefaultDOTGraphTraits {
  using GTraits = GraphTraits<BlockFrequencyInfoT *>;
  using NodeRef = typename GTraits::NodeRef;
  using EdgeIter = typename GTraits::ChildIteratorType;
  using NodeIter = typename GTraits::nodes_iterator;

  uint64_t MaxFrequency = 0;

  explicit BFIDOTGraphTraitsBase(bool isSimple = false)
      : DefaultDOTGraphTraits(isSimple) {}

  static StringRef getGraphName(const BlockFrequencyInfoT *G) {
    return G->getFunction()->getName();
  }

  std::string getNodeAttributes(NodeRef Node, const BlockFrequencyInfoT *Graph,
                                unsigned HotPercentThreshold = 0) {
    std::string Result;
    if (!HotPercentThreshold)
      return Result;

    // Compute MaxFrequency on the fly:
    if (!MaxFrequency) {
      for (NodeIter I = GTraits::nodes_begin(Graph),
                    E = GTraits::nodes_end(Graph);
           I != E; ++I) {
        NodeRef N = *I;
        MaxFrequency =
            std::max(MaxFrequency, Graph->getBlockFreq(N).getFrequency());
      }
    }
    BlockFrequency Freq = Graph->getBlockFreq(Node);
    BlockFrequency HotFreq =
        (BlockFrequency(MaxFrequency) *
         BranchProbability::getBranchProbability(HotPercentThreshold, 100));

    if (Freq < HotFreq)
      return Result;

    raw_string_ostream(Result) << "color=\"red\"";
    return Result;
  }

  std::string getNodeLabel(NodeRef Node, const BlockFrequencyInfoT *Graph,
                           GVDAGType GType, int layout_order = -1) {
    std::string Result;
    raw_string_ostream OS(Result);

    if (layout_order != -1)
      OS << Node->getName() << "[" << layout_order << "] : ";
    else
      OS << Node->getName() << " : ";
    switch (GType) {
    case GVDT_Fraction:
      OS << printBlockFreq(*Graph, *Node);
      break;
    case GVDT_Integer:
      OS << Graph->getBlockFreq(Node).getFrequency();
      break;
    case GVDT_Count: {
      auto Count = Graph->getBlockProfileCount(Node);
      if (Count)
        OS << *Count;
      else
        OS << "Unknown";
      break;
    }
    case GVDT_None:
      llvm_unreachable("If we are not supposed to render a graph we should "
                       "never reach this point.");
    }
    return Result;
  }

  std::string getEdgeAttributes(NodeRef Node, EdgeIter EI,
                                const BlockFrequencyInfoT *BFI,
                                const BranchProbabilityInfoT *BPI,
                                unsigned HotPercentThreshold = 0) {
    std::string Str;
    if (!BPI)
      return Str;

    unsigned SuccIdx = std::distance(succ_begin(Node), EI);
    BranchProbability BP = BPI->getEdgeProbability(Node, SuccIdx);
    uint32_t N = BP.getNumerator();
    uint32_t D = BP.getDenominator();
    double Percent = 100.0 * N / D;
    raw_string_ostream OS(Str);
    OS << format("label=\"%.1f%%\"", Percent);

    if (HotPercentThreshold) {
      BlockFrequency EFreq = BFI->getBlockFreq(Node) * BP;
      BlockFrequency HotFreq = BlockFrequency(MaxFrequency) *
                               BranchProbability(HotPercentThreshold, 100);

      if (EFreq >= HotFreq)
        OS << ",color=\"red\"";
    }
    return Str;
  }
};

} // end namespace llvm

#undef DEBUG_TYPE

#endif // LLVM_ANALYSIS_BLOCKFREQUENCYINFOIMPL_H
