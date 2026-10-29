//===- SplitModuleUtils.h - Shared module splitting utilities ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines policy-neutral infrastructure shared by module splitting
// implementations. Callers supply graph construction, cost, partitioning, and
// definition selection policies. This file provides graph storage, dependency
// traversal, partition bookkeeping, and module cloning.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_SPLITMODULEUTILS_H
#define LLVM_TRANSFORMS_UTILS_SPLITMODULEUTILS_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/InstructionCost.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>

namespace llvm {

class Function;
class CallGraph;
class GlobalValue;
class Module;

namespace splitmodule {

using FunctionCost = InstructionCost::CostType;
using FunctionCostMap = DenseMap<const Function *, FunctionCost>;
using CostType = FunctionCost;
using FunctionsCostMap = FunctionCostMap;

/// A view of the source Module, as a graph of all components
/// that can be split into different modules.
///
/// The most trivial instance of this graph is just the CallGraph of the module,
/// but it is not guaranteed that the graph is strictly equal to the CG. It
/// currently always is but it's designed in a way that would eventually allow
/// us to create abstract nodes, or nodes for different entities such as global
/// variables or any other meaningful constraint we must consider.
///
/// The graph is only mutable by this class and its builders, and is not
/// modified after \ref SplitGraph::buildGraph runs. No consumers of the graph
/// can mutate it.
class LLVM_ABI SplitGraph {
public:
  class Node;

  enum class EdgeKind : uint8_t {
    /// The nodes are related through a direct call. This is a "strong" edge as
    /// it means the Src will directly reference the Dst.
    DirectCall,
    /// The nodes are related through an indirect call.
    /// This is a "weaker" edge and is only considered when traversing the graph
    /// starting from a kernel. We need this edge for resource usage analysis.
    ///
    /// The reason why we have this edge in the first place is due to how
    /// AMDGPUResourceUsageAnalysis works. In the presence of an indirect call,
    /// the resource usage of the kernel containing the indirect call is the
    /// max resource usage of all functions that can be indirectly called.
    IndirectCall,
  };

  /// An edge between two nodes. Edges are directional, and tagged with a
  /// "kind".
  struct Edge {
    Edge(Node *Src, Node *Dst, EdgeKind Kind)
        : Src(Src), Dst(Dst), Kind(Kind) {}

    Node *Src; ///< Source
    Node *Dst; ///< Destination
    EdgeKind Kind;
  };

  using EdgesVec = SmallVector<const Edge *, 0>;
  using edges_iterator = EdgesVec::const_iterator;
  using nodes_iterator = const Node *const *;

  SplitGraph(const Module &M, const FunctionsCostMap &CostMap,
             CostType ModuleCost)
      : M(M), CostMap(CostMap), ModuleCost(ModuleCost) {}

  virtual ~SplitGraph();

  virtual void buildGraph(CallGraph &CG) = 0;

#ifndef NDEBUG
  virtual bool verifyGraph() const;
#endif

  bool empty() const { return Nodes.empty(); }
  iterator_range<nodes_iterator> nodes() const { return Nodes; }
  const Node &getNode(unsigned ID) const { return *Nodes[ID]; }

  /// Mark entries while constructing a graph or its partitioning worklist.
  void markAsGraphEntry(Node &N);

  unsigned getNumNodes() const { return Nodes.size(); }
  BitVector createNodesBitVector() const { return BitVector(Nodes.size()); }

  const Module &getModule() const { return M; }

  CostType getModuleCost() const { return ModuleCost; }
  CostType getCost(const Function &F) const { return CostMap.at(&F); }

  /// \returns the aggregated cost of all nodes in \p BV (bits set to 1 = node
  /// IDs).
  CostType calculateCost(const BitVector &BV) const;

protected:
  /// Derived builders provide target-specific properties. These are evaluated
  /// only when a node is first created, never when its cache entry is reused.
  virtual bool isNonCopyable(const Function &F) const = 0;
  virtual bool isEntryFunctionCC(const Function &F) const = 0;

  /// Retrieves the node for \p GV in \p Cache, or creates a new node for it and
  /// updates \p Cache.
  Node &getNode(DenseMap<const GlobalValue *, Node *> &Cache,
                const GlobalValue &GV);

  // Create a new edge between two nodes and add it to both nodes.
  virtual const Edge &createEdge(Node &Src, Node &Dst, EdgeKind EK);

  const Module &M;
  const FunctionsCostMap &CostMap;
  CostType ModuleCost;

  // Final list of nodes with stable ordering.
  SmallVector<Node *> Nodes;

  SpecificBumpPtrAllocator<Node> NodesPool;

  // Edges are trivially destructible objects, so as a small optimization we
  // use a BumpPtrAllocator which avoids destructor calls but also makes
  // allocation faster.
  static_assert(
      std::is_trivially_destructible_v<Edge>,
      "Edge must be trivially destructible to use the BumpPtrAllocator");
  BumpPtrAllocator EdgesPool;
};

/// Nodes in the SplitGraph contain both incoming, and outgoing edges.
/// Incoming edges have this node as their Dst, and Outgoing ones have this node
/// as their Src.
///
/// Edge objects are shared by both nodes in Src/Dst. They provide immediate
/// feedback on how two nodes are related, and in which direction they are
/// related, which is valuable information to make splitting decisions.
///
/// Nodes are fundamentally abstract, and any consumers of the graph should
/// treat them as such. While a node will be a function most of the time, we
/// could also create nodes for any other reason. In the future, we could have
/// single nodes for multiple functions, or nodes for GVs, etc.
class SplitGraph::Node {
  friend class SplitGraph;

public:
  Node(unsigned ID, const GlobalValue &GV, CostType IndividualCost,
       bool IsNonCopyable, bool IsEntryFnCC)
      : ID(ID), GV(GV), IndividualCost(IndividualCost),
        IsNonCopyable(IsNonCopyable), IsEntryFnCC(IsEntryFnCC),
        IsGraphEntry(false) {}

  /// An 0-indexed ID for the node. The maximum ID (exclusive) is the number of
  /// nodes in the graph. This ID can be used as an index in a BitVector.
  unsigned getID() const { return ID; }

  const Function &getFunction() const { return cast<Function>(GV); }

  /// \returns the cost to import this component into a given module, not
  /// accounting for any dependencies that may need to be imported as well.
  CostType getIndividualCost() const { return IndividualCost; }

  bool isNonCopyable() const { return IsNonCopyable; }
  /// Cached by the graph when this node is first created.
  bool isEntryFunctionCC() const { return IsEntryFnCC; }

  /// \returns whether this is an entry point in the graph. Entry points are
  /// defined as follows: if you take all entry points in the graph, and iterate
  /// their dependencies, you are guaranteed to visit all nodes in the graph at
  /// least once.
  bool isGraphEntryPoint() const { return IsGraphEntry; }

  StringRef getName() const { return GV.getName(); }

  bool hasAnyIncomingEdges() const { return IncomingEdges.size(); }
  bool hasAnyIncomingEdgesOfKind(EdgeKind EK) const {
    return any_of(IncomingEdges, [&](const auto *E) { return E->Kind == EK; });
  }

  iterator_range<edges_iterator> outgoing_edges() const {
    return OutgoingEdges;
  }

  bool shouldFollowIndirectCalls() const { return isEntryFunctionCC(); }

  /// Visit all children of this node in a recursive fashion. Also visits Self.
  /// If \ref shouldFollowIndirectCalls returns false, then this only follows
  /// DirectCall edges.
  ///
  /// \param Visitor Visitor Function.
  LLVM_ABI void
  visitAllDependencies(std::function<void(const Node &)> Visitor) const;

  /// Adds the depedencies of this node in \p BV by setting the bit
  /// corresponding to each node.
  ///
  /// Implemented using \ref visitAllDependencies, hence it follows the same
  /// rules regarding dependencies traversal.
  ///
  /// \param[out] BV The bitvector where the bits should be set.
  void getDependencies(BitVector &BV) const {
    visitAllDependencies([&](const Node &N) { BV.set(N.getID()); });
  }

private:
  void markAsGraphEntry() { IsGraphEntry = true; }

  unsigned ID;
  const GlobalValue &GV;
  CostType IndividualCost;
  bool IsNonCopyable : 1;
  bool IsEntryFnCC : 1;
  bool IsGraphEntry : 1;

  // TODO: Use a single sorted vector (with all incoming/outgoing edges grouped
  // together)
  EdgesVec IncomingEdges;
  EdgesVec OutgoingEdges;
};

LLVM_ABI FunctionCost calculateFunctionCosts(
    Module &M, function_ref<FunctionCost(Function &F)> CalculateFunctionCost,
    FunctionCostMap &FunctionCosts);

/// Preserve module-level inline assembly in all partitions or only partition 0.
enum class ModuleAsmPolicy { PreserveAll, PreserveFirst };

/// Clone one partition and synchronously invoke ModuleCallback. Both callbacks
/// receive PartitionId; the caller controls partition iteration and cleanup.
/// PreserveFirst retains inline assembly only when PartitionId is zero.
LLVM_ABI void materializePartitions(
    const Module &M, unsigned PartitionId,
    function_ref<bool(const GlobalValue *GV, unsigned PartitionId)>
        ShouldCloneDefinition,
    function_ref<void(std::unique_ptr<Module> MPart, unsigned PartitionId)>
        ModuleCallback,
    ModuleAsmPolicy AsmPolicy);

} // namespace splitmodule
} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_SPLITMODULEUTILS_H
