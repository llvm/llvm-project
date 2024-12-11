//===- AMDGPUSplitModule.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Implements a module splitting algorithm designed to support the
/// FullLTO --lto-partitions option for parallel codegen.
///
/// The role of this module splitting pass is the same as
/// lib/Transforms/Utils/SplitModule.cpp: load-balance the module's functions
/// across a set of N partitions to allow for parallel codegen.
///
/// The similarities mostly end here, as this pass achieves load-balancing in a
/// more elaborate fashion which is targeted towards AMDGPU modules. It can take
/// advantage of the structure of AMDGPU modules (which are mostly
/// self-contained) to allow for more efficient splitting without affecting
/// codegen negatively, or causing innaccurate resource usage analysis.
///
/// High-level pass overview:
///   - SplitGraph & associated classes
///      - Graph representation of the module and of the dependencies that
///      matter for splitting.
///   - RecursiveSearchSplitting
///     - Core splitting algorithm.
///   - SplitProposal
///     - Represents a suggested solution for splitting the input module. These
///     solutions can be scored to determine the best one when multiple
///     solutions are available.
///   - Driver/pass "run" function glues everything together.

#include "AMDGPUSplitModule.h"
#include "AMDGPUTargetMachine.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <cassert>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#ifndef NDEBUG
#include "llvm/Support/LockFileManager.h"
#endif

#define DEBUG_TYPE "amdgpu-split-module"

namespace llvm {
namespace {

static cl::opt<unsigned> MaxDepth(
    "amdgpu-module-splitting-max-depth",
    cl::desc(
        "maximum search depth. 0 forces a greedy approach. "
        "warning: the algorithm is up to O(2^N), where N is the max depth."),
    cl::init(8));

static cl::opt<float> LargeFnFactor(
    "amdgpu-module-splitting-large-threshold", cl::init(2.0f), cl::Hidden,
    cl::desc(
        "when max depth is reached and we can no longer branch out, this "
        "value determines if a function is worth merging into an already "
        "existing partition to reduce code duplication. This is a factor "
        "of the ideal partition size, e.g. 2.0 means we consider the "
        "function for merging if its cost (including its callees) is 2x the "
        "size of an ideal partition."));

static cl::opt<float> LargeFnOverlapForMerge(
    "amdgpu-module-splitting-merge-threshold", cl::init(0.7f), cl::Hidden,
    cl::desc("when a function is considered for merging into a partition that "
             "already contains some of its callees, do the merge if at least "
             "n% of the code it can reach is already present inside the "
             "partition; e.g. 0.7 means only merge >70%"));

static cl::opt<bool> NoExternalizeGlobals(
    "amdgpu-module-splitting-no-externalize-globals", cl::Hidden,
    cl::desc("disables externalization of global variable with local linkage; "
             "may cause globals to be duplicated which increases binary size"));

static cl::opt<bool> NoExternalizeOnAddrTaken(
    "amdgpu-module-splitting-no-externalize-address-taken", cl::Hidden,
    cl::desc(
        "disables externalization of functions whose addresses are taken"));

static cl::opt<std::string>
    ModuleDotCfgOutput("amdgpu-module-splitting-print-module-dotcfg",
                       cl::Hidden,
                       cl::desc("output file to write out the dotgraph "
                                "representation of the input module"));

static cl::opt<std::string> PartitionSummariesOutput(
    "amdgpu-module-splitting-print-partition-summaries", cl::Hidden,
    cl::desc("output file to write out a summary of "
             "the partitions created for each module"));

#ifndef NDEBUG
static cl::opt<bool>
    UseLockFile("amdgpu-module-splitting-serial-execution", cl::Hidden,
                cl::desc("use a lock file so only one process in the system "
                         "can run this pass at once. useful to avoid mangled "
                         "debug output in multithreaded environments."));

static cl::opt<bool>
    DebugProposalSearch("amdgpu-module-splitting-debug-proposal-search",
                        cl::Hidden,
                        cl::desc("print all proposals received and whether "
                                 "they were rejected or accepted"));
#endif

struct SplitModuleTimer : NamedRegionTimer {
  SplitModuleTimer(StringRef Name, StringRef Desc)
      : NamedRegionTimer(Name, Desc, DEBUG_TYPE, "AMDGPU Module Splitting",
                         TimePassesIsEnabled) {}
};

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

using CostType = InstructionCost::CostType;
using FunctionsCostMap = DenseMap<const Function *, CostType>;
using GetTTIFn = function_ref<const TargetTransformInfo &(Function &)>;
static constexpr unsigned InvalidPID = -1;

/// \param Num numerator
/// \param Dem denominator
/// \returns a printable object to print (Num/Dem) using "%0.2f".
static auto formatRatioOf(CostType Num, CostType Dem) {
  CostType DemOr1 = Dem ? Dem : 1;
  return format("%0.2f", (static_cast<double>(Num) / DemOr1) * 100);
}

/// Checks whether a given function is non-copyable.
///
/// Non-copyable functions cannot be cloned into multiple partitions, and only
/// one copy of the function can be present across all partitions.
///
/// Kernel functions and external functions fall into this category. If we were
/// to clone them, we would end up with multiple symbol definitions and a very
/// unhappy linker.
static bool isNonCopyable(const Function &F) {
  return F.hasExternalLinkage() || !F.isDefinitionExact() ||
         AMDGPU::isEntryFunctionCC(F.getCallingConv());
}

/// If \p GV has local linkage, make it external + hidden.
static void externalize(GlobalValue &GV) {
  if (GV.hasLocalLinkage()) {
    GV.setLinkage(GlobalValue::ExternalLinkage);
    GV.setVisibility(GlobalValue::HiddenVisibility);
  }

  // Unnamed entities must be named consistently between modules. setName will
  // give a distinct name to each such entity.
  if (!GV.hasName())
    GV.setName("__llvmsplit_unnamed");
}

/// Cost analysis function. Calculates the cost of each function in \p M
///
/// \param GetTTI Abstract getter for TargetTransformInfo.
/// \param M Module to analyze.
/// \param CostMap[out] Resulting Function -> Cost map.
/// \return The module's total cost.
static CostType calculateFunctionCosts(GetTTIFn GetTTI, Module &M,
                                       FunctionsCostMap &CostMap) {
  SplitModuleTimer SMT("calculateFunctionCosts", "cost analysis");

  LLVM_DEBUG(dbgs() << "[cost analysis] calculating function costs\n");
  CostType ModuleCost = 0;
  [[maybe_unused]] CostType KernelCost = 0;

  for (auto &Fn : M) {
    if (Fn.isDeclaration())
      continue;

    CostType FnCost = 0;
    const auto &TTI = GetTTI(Fn);
    for (const auto &BB : Fn) {
      for (const auto &I : BB) {
        auto Cost =
            TTI.getInstructionCost(&I, TargetTransformInfo::TCK_CodeSize);
        assert(Cost != InstructionCost::getMax());
        // Assume expensive if we can't tell the cost of an instruction.
        CostType CostVal =
            Cost.getValue().value_or(TargetTransformInfo::TCC_Expensive);
        assert((FnCost + CostVal) >= FnCost && "Overflow!");
        FnCost += CostVal;
      }
    }

    assert(FnCost != 0);

    CostMap[&Fn] = FnCost;
    assert((ModuleCost + FnCost) >= ModuleCost && "Overflow!");
    ModuleCost += FnCost;

    if (AMDGPU::isEntryFunctionCC(Fn.getCallingConv()))
      KernelCost += FnCost;
  }

  if (CostMap.empty())
    return 0;

  assert(ModuleCost);
  LLVM_DEBUG({
    const CostType FnCost = ModuleCost - KernelCost;
    dbgs() << " - total module cost is " << ModuleCost << ". kernels cost "
           << "" << KernelCost << " ("
           << format("%0.2f", (float(KernelCost) / ModuleCost) * 100)
           << "% of the module), functions cost " << FnCost << " ("
           << format("%0.2f", (float(FnCost) / ModuleCost) * 100)
           << "% of the module)\n";
  });

  return ModuleCost;
}

/// \return true if \p F can be indirectly called
static bool canBeIndirectlyCalled(const Function &F) {
  if (F.isDeclaration() || AMDGPU::isEntryFunctionCC(F.getCallingConv()))
    return false;
  return !F.hasLocalLinkage() ||
         F.hasAddressTaken(/*PutOffender=*/nullptr,
                           /*IgnoreCallbackUses=*/false,
                           /*IgnoreAssumeLikeCalls=*/true,
                           /*IgnoreLLVMUsed=*/true,
                           /*IgnoreARCAttachedCall=*/false,
                           /*IgnoreCastedDirectCall=*/true);
}

//===----------------------------------------------------------------------===//
// Graph-based Module Representation
//===----------------------------------------------------------------------===//

/// AMDGPUSplitModule's view of the source Module, as a graph of all components
/// that can be split into different modules.
///
/// The most trivial instance of this graph is just the CallGraph of the module,
/// but it is not guaranteed that the graph is strictly equal to the CG. It
/// currently always is but it's designed in a way that would eventually allow
/// us to create abstract nodes, or nodes for different entities such as global
/// variables or any other meaningful constraint we must consider.
///
/// The graph is only mutable by this class, and is generally not modified
/// after \ref SplitGraph::buildGraph runs. No consumers of the graph can
/// mutate it.
class SplitGraph {
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

  void buildGraph(CallGraph &CG);

#ifndef NDEBUG
  bool verifyGraph() const;
#endif

  bool empty() const { return Nodes.empty(); }
  const iterator_range<nodes_iterator> nodes() const {
    return {Nodes.begin(), Nodes.end()};
  }
  const Node &getNode(unsigned ID) const { return *Nodes[ID]; }

  unsigned getNumNodes() const { return Nodes.size(); }
  BitVector createNodesBitVector() const { return BitVector(Nodes.size()); }

  const Module &getModule() const { return M; }

  CostType getModuleCost() const { return ModuleCost; }
  CostType getCost(const Function &F) const { return CostMap.at(&F); }

  /// \returns the aggregated cost of all nodes in \p BV (bits set to 1 = node
  /// IDs).
  CostType calculateCost(const BitVector &BV) const;

private:
  /// Retrieves the node for \p GV in \p Cache, or creates a new node for it and
  /// updates \p Cache.
  Node &getNode(DenseMap<const GlobalValue *, Node *> &Cache,
                const GlobalValue &GV);

  // Create a new edge between two nodes and add it to both nodes.
  const Edge &createEdge(Node &Src, Node &Dst, EdgeKind EK);

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
       bool IsNonCopyable)
      : ID(ID), GV(GV), IndividualCost(IndividualCost),
        IsNonCopyable(IsNonCopyable), IsEntryFnCC(false), IsGraphEntry(false) {
    if (auto *Fn = dyn_cast<Function>(&GV))
      IsEntryFnCC = AMDGPU::isEntryFunctionCC(Fn->getCallingConv());
  }

  /// An 0-indexed ID for the node. The maximum ID (exclusive) is the number of
  /// nodes in the graph. This ID can be used as an index in a BitVector.
  unsigned getID() const { return ID; }

  const Function &getFunction() const { return cast<Function>(GV); }

  /// \returns the cost to import this component into a given module, not
  /// accounting for any dependencies that may need to be imported as well.
  CostType getIndividualCost() const { return IndividualCost; }

  bool isNonCopyable() const { return IsNonCopyable; }
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

  bool hasAnyOutgoingEdges() const { return OutgoingEdges.size(); }
  bool hasAnyOutgoingEdgesOfKind(EdgeKind EK) const {
    return any_of(OutgoingEdges, [&](const auto *E) { return E->Kind == EK; });
  }

  iterator_range<edges_iterator> incoming_edges() const {
    return IncomingEdges;
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
  void visitAllDependencies(std::function<void(const Node &)> Visitor) const;

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

void SplitGraph::Node::visitAllDependencies(
    std::function<void(const Node &)> Visitor) const {
  const bool FollowIndirect = shouldFollowIndirectCalls();
  // FIXME: If this can access SplitGraph in the future, use a BitVector
  // instead.
  DenseSet<const Node *> Seen;
  SmallVector<const Node *, 8> WorkList({this});
  while (!WorkList.empty()) {
    const Node *CurN = WorkList.pop_back_val();
    if (auto [It, Inserted] = Seen.insert(CurN); !Inserted)
      continue;

    Visitor(*CurN);

    for (const Edge *E : CurN->outgoing_edges()) {
      if (!FollowIndirect && E->Kind == EdgeKind::IndirectCall)
        continue;
      WorkList.push_back(E->Dst);
    }
  }
}

/// Checks if \p I has MD_callees and if it does, parse it and put the function
/// in \p Callees.
///
/// \returns true if there was metadata and it was parsed correctly. false if
/// there was no MD or if it contained unknown entries and parsing failed.
/// If this returns false, \p Callees will contain incomplete information
/// and must not be used.
static bool handleCalleesMD(const Instruction &I,
                            SetVector<Function *> &Callees) {
  auto *MD = I.getMetadata(LLVMContext::MD_callees);
  if (!MD)
    return false;

  for (const auto &Op : MD->operands()) {
    Function *Callee = mdconst::extract_or_null<Function>(Op);
    if (!Callee)
      return false;
    Callees.insert(Callee);
  }

  return true;
}

void SplitGraph::buildGraph(CallGraph &CG) {
  SplitModuleTimer SMT("buildGraph", "graph construction");
  LLVM_DEBUG(
      dbgs()
      << "[build graph] constructing graph representation of the input\n");

  // FIXME(?): Is the callgraph really worth using if we have to iterate the
  // function again whenever it fails to give us enough information?

  // We build the graph by just iterating all functions in the module and
  // working on their direct callees. At the end, all nodes should be linked
  // together as expected.
  DenseMap<const GlobalValue *, Node *> Cache;
  SmallVector<const Function *> FnsWithIndirectCalls, IndirectlyCallableFns;
  for (const Function &Fn : M) {
    if (Fn.isDeclaration())
      continue;

    // Look at direct callees and create the necessary edges in the graph.
    SetVector<const Function *> DirectCallees;
    bool CallsExternal = false;
    for (auto &CGEntry : *CG[&Fn]) {
      auto *CGNode = CGEntry.second;
      if (auto *Callee = CGNode->getFunction()) {
        if (!Callee->isDeclaration())
          DirectCallees.insert(Callee);
      } else if (CGNode == CG.getCallsExternalNode())
        CallsExternal = true;
    }

    // Keep track of this function if it contains an indirect call and/or if it
    // can be indirectly called.
    if (CallsExternal) {
      LLVM_DEBUG(dbgs() << "  [!] callgraph is incomplete for ";
                 Fn.printAsOperand(dbgs());
                 dbgs() << " - analyzing function\n");

      SetVector<Function *> KnownCallees;
      bool HasUnknownIndirectCall = false;
      for (const auto &Inst : instructions(Fn)) {
        // look at all calls without a direct callee.
        const auto *CB = dyn_cast<CallBase>(&Inst);
        if (!CB || CB->getCalledFunction())
          continue;

        // inline assembly can be ignored, unless InlineAsmIsIndirectCall is
        // true.
        if (CB->isInlineAsm()) {
          LLVM_DEBUG(dbgs() << "    found inline assembly\n");
          continue;
        }

        if (handleCalleesMD(Inst, KnownCallees))
          continue;
        // If we failed to parse any !callees MD, or some was missing,
        // the entire KnownCallees list is now unreliable.
        KnownCallees.clear();

        // Everything else is handled conservatively. If we fall into the
        // conservative case don't bother analyzing further.
        HasUnknownIndirectCall = true;
        break;
      }

      if (HasUnknownIndirectCall) {
        LLVM_DEBUG(dbgs() << "    indirect call found\n");
        FnsWithIndirectCalls.push_back(&Fn);
      } else if (!KnownCallees.empty())
        DirectCallees.insert(KnownCallees.begin(), KnownCallees.end());
    }

    Node &N = getNode(Cache, Fn);
    for (const auto *Callee : DirectCallees)
      createEdge(N, getNode(Cache, *Callee), EdgeKind::DirectCall);

    if (canBeIndirectlyCalled(Fn))
      IndirectlyCallableFns.push_back(&Fn);
  }

  // Post-process functions with indirect calls.
  for (const Function *Fn : FnsWithIndirectCalls) {
    for (const Function *Candidate : IndirectlyCallableFns) {
      Node &Src = getNode(Cache, *Fn);
      Node &Dst = getNode(Cache, *Candidate);
      createEdge(Src, Dst, EdgeKind::IndirectCall);
    }
  }

  // Now, find all entry points.
  SmallVector<Node *, 16> CandidateEntryPoints;
  BitVector NodesReachableByKernels = createNodesBitVector();
  for (Node *N : Nodes) {
    // Functions with an Entry CC are always graph entry points too.
    if (N->isEntryFunctionCC()) {
      N->markAsGraphEntry();
      N->getDependencies(NodesReachableByKernels);
    } else if (!N->hasAnyIncomingEdgesOfKind(EdgeKind::DirectCall))
      CandidateEntryPoints.push_back(N);
  }

  for (Node *N : CandidateEntryPoints) {
    // This can be another entry point if it's not reachable by a kernel
    // TODO: We could sort all of the possible new entries in a stable order
    // (e.g. by cost), then consume them one by one until
    // NodesReachableByKernels is all 1s. It'd allow us to avoid
    // considering some nodes as non-entries in some specific cases.
    if (!NodesReachableByKernels.test(N->getID()))
      N->markAsGraphEntry();
  }

#ifndef NDEBUG
  assert(verifyGraph());
#endif
}

#ifndef NDEBUG
bool SplitGraph::verifyGraph() const {
  unsigned ExpectedID = 0;
  // Exceptionally using a set here in case IDs are messed up.
  DenseSet<const Node *> SeenNodes;
  DenseSet<const Function *> SeenFunctionNodes;
  for (const Node *N : Nodes) {
    if (N->getID() != (ExpectedID++)) {
      errs() << "Node IDs are incorrect!\n";
      return false;
    }

    if (!SeenNodes.insert(N).second) {
      errs() << "Node seen more than once!\n";
      return false;
    }

    if (&getNode(N->getID()) != N) {
      errs() << "getNode doesn't return the right node\n";
      return false;
    }

    for (const Edge *E : N->IncomingEdges) {
      if (!E->Src || !E->Dst || (E->Dst != N) ||
          (find(E->Src->OutgoingEdges, E) == E->Src->OutgoingEdges.end())) {
        errs() << "ill-formed incoming edges\n";
        return false;
      }
    }

    for (const Edge *E : N->OutgoingEdges) {
      if (!E->Src || !E->Dst || (E->Src != N) ||
          (find(E->Dst->IncomingEdges, E) == E->Dst->IncomingEdges.end())) {
        errs() << "ill-formed outgoing edges\n";
        return false;
      }
    }

    const Function &Fn = N->getFunction();
    if (AMDGPU::isEntryFunctionCC(Fn.getCallingConv())) {
      if (N->hasAnyIncomingEdges()) {
        errs() << "Kernels cannot have incoming edges\n";
        return false;
      }
    }

    if (Fn.isDeclaration()) {
      errs() << "declarations shouldn't have nodes!\n";
      return false;
    }

    auto [It, Inserted] = SeenFunctionNodes.insert(&Fn);
    if (!Inserted) {
      errs() << "one function has multiple nodes!\n";
      return false;
    }
  }

  if (ExpectedID != Nodes.size()) {
    errs() << "Node IDs out of sync!\n";
    return false;
  }

  if (createNodesBitVector().size() != getNumNodes()) {
    errs() << "nodes bit vector doesn't have the right size!\n";
    return false;
  }

  // Check we respect the promise of Node::isKernel
  BitVector BV = createNodesBitVector();
  for (const Node *N : nodes()) {
    if (N->isGraphEntryPoint())
      N->getDependencies(BV);
  }

  // Ensure each function in the module has an associated node.
  for (const auto &Fn : M) {
    if (!Fn.isDeclaration()) {
      if (!SeenFunctionNodes.contains(&Fn)) {
        errs() << "Fn has no associated node in the graph!\n";
        return false;
      }
    }
  }

  if (!BV.all()) {
    errs() << "not all nodes are reachable through the graph's entry points!\n";
    return false;
  }

  return true;
}
#endif

CostType SplitGraph::calculateCost(const BitVector &BV) const {
  CostType Cost = 0;
  for (unsigned NodeID : BV.set_bits())
    Cost += getNode(NodeID).getIndividualCost();
  return Cost;
}

SplitGraph::Node &
SplitGraph::getNode(DenseMap<const GlobalValue *, Node *> &Cache,
                    const GlobalValue &GV) {
  auto &N = Cache[&GV];
  if (N)
    return *N;

  CostType Cost = 0;
  bool NonCopyable = false;
  if (const Function *Fn = dyn_cast<Function>(&GV)) {
    NonCopyable = isNonCopyable(*Fn);
    Cost = CostMap.at(Fn);
  }
  N = new (NodesPool.Allocate()) Node(Nodes.size(), GV, Cost, NonCopyable);
  Nodes.push_back(N);
  assert(&getNode(N->getID()) == N);
  return *N;
}

const SplitGraph::Edge &SplitGraph::createEdge(Node &Src, Node &Dst,
                                               EdgeKind EK) {
  const Edge *E = new (EdgesPool.Allocate<Edge>(1)) Edge(&Src, &Dst, EK);
  Src.OutgoingEdges.push_back(E);
  Dst.IncomingEdges.push_back(E);
  return *E;
}

//===----------------------------------------------------------------------===//
// Split Proposals
//===----------------------------------------------------------------------===//

/// Represents a module splitting proposal.
///
/// Proposals are made of N BitVectors, one for each partition, where each bit
/// set indicates that the node is present and should be copied inside that
/// partition.
///
/// Proposals have several metrics attached so they can be compared/sorted,
/// which the driver to try multiple strategies resultings in multiple proposals
/// and choose the best one out of them.
class SplitProposal {
public:
  SplitProposal(const SplitGraph &SG, unsigned MaxPartitions) : SG(&SG) {
    Partitions.resize(MaxPartitions, {0, SG.createNodesBitVector()});
  }

  void setName(StringRef NewName) { Name = NewName; }
  StringRef getName() const { return Name; }

  const BitVector &operator[](unsigned PID) const {
    return Partitions[PID].second;
  }

  void add(unsigned PID, const BitVector &BV) {
    Partitions[PID].second |= BV;
    updateScore(PID);
  }

  void print(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const { print(dbgs()); }

  // Find the cheapest partition (lowest cost). In case of ties, always returns
  // the highest partition number.
  unsigned findCheapestPartition() const;

  /// Calculate the CodeSize and Bottleneck scores.
  void calculateScores();

#ifndef NDEBUG
  void verifyCompleteness() const;
#endif

  /// Only available after \ref calculateScores is called.
  ///
  /// A positive number indicating the % of code duplication that this proposal
  /// creates. e.g. 0.2 means this proposal adds roughly 20% code size by
  /// duplicating some functions across partitions.
  ///
  /// Value is always rounded up to 3 decimal places.
  ///
  /// A perfect score would be 0.0, and anything approaching 1.0 is very bad.
  double getCodeSizeScore() const { return CodeSizeScore; }

  /// Only available after \ref calculateScores is called.
  ///
  /// A number between [0, 1] which indicates how big of a bottleneck is
  /// expected from the largest partition.
  ///
  /// A score of 1.0 means the biggest partition is as big as the source module,
  /// so build time will be equal to or greater than the build time of the
  /// initial input.
  ///
  /// Value is always rounded up to 3 decimal places.
  ///
  /// This is one of the metrics used to estimate this proposal's build time.
  double getBottleneckScore() const { return BottleneckScore; }

private:
  void updateScore(unsigned PID) {
    assert(SG);
    for (auto &[PCost, Nodes] : Partitions) {
      TotalCost -= PCost;
      PCost = SG->calculateCost(Nodes);
      TotalCost += PCost;
    }
  }

  /// \see getCodeSizeScore
  double CodeSizeScore = 0.0;
  /// \see getBottleneckScore
  double BottleneckScore = 0.0;
  /// Aggregated cost of all partitions
  CostType TotalCost = 0;

  const SplitGraph *SG = nullptr;
  std::string Name;

  std::vector<std::pair<CostType, BitVector>> Partitions;
};

void SplitProposal::print(raw_ostream &OS) const {
  assert(SG);

  OS << "[proposal] " << Name << ", total cost:" << TotalCost
     << ", code size score:" << format("%0.3f", CodeSizeScore)
     << ", bottleneck score:" << format("%0.3f", BottleneckScore) << '\n';
  for (const auto &[PID, Part] : enumerate(Partitions)) {
    const auto &[Cost, NodeIDs] = Part;
    OS << "  - P" << PID << " nodes:" << NodeIDs.count() << " cost: " << Cost
       << '|' << formatRatioOf(Cost, SG->getModuleCost()) << "%\n";
  }
}

unsigned SplitProposal::findCheapestPartition() const {
  assert(!Partitions.empty());
  CostType CurCost = std::numeric_limits<CostType>::max();
  unsigned CurPID = InvalidPID;
  for (const auto &[Idx, Part] : enumerate(Partitions)) {
    if (Part.first <= CurCost) {
      CurPID = Idx;
      CurCost = Part.first;
    }
  }
  assert(CurPID != InvalidPID);
  return CurPID;
}

void SplitProposal::calculateScores() {
  if (Partitions.empty())
    return;

  assert(SG);
  CostType LargestPCost = 0;
  for (auto &[PCost, Nodes] : Partitions) {
    if (PCost > LargestPCost)
      LargestPCost = PCost;
  }

  CostType ModuleCost = SG->getModuleCost();
  CodeSizeScore = double(TotalCost) / ModuleCost;
  assert(CodeSizeScore >= 0.0);

  BottleneckScore = double(LargestPCost) / ModuleCost;

  CodeSizeScore = std::ceil(CodeSizeScore * 100.0) / 100.0;
  BottleneckScore = std::ceil(BottleneckScore * 100.0) / 100.0;
}

#ifndef NDEBUG
void SplitProposal::verifyCompleteness() const {
  if (Partitions.empty())
    return;

  BitVector Result = Partitions[0].second;
  for (const auto &P : drop_begin(Partitions))
    Result |= P.second;
  assert(Result.all() && "some nodes are missing from this proposal!");
}
#endif

//===-- RecursiveSearchStrategy -------------------------------------------===//

/// Partitioning algorithm.
///
/// This is a recursive search algorithm that can explore multiple possiblities.
///
/// When a cluster of nodes can go into more than one partition, and we haven't
/// reached maximum search depth, we recurse and explore both options and their
/// consequences. Both branches will yield a proposal, and the driver will grade
/// both and choose the best one.
///
/// If max depth is reached, we will use some heuristics to make a choice. Most
/// of the time we will just use the least-pressured (cheapest) partition, but
/// if a cluster is particularly big and there is a good amount of overlap with
/// an existing partition, we will choose that partition instead.
class RecursiveSearchSplitting {
public:
  using SubmitProposalFn = function_ref<void(SplitProposal)>;

  RecursiveSearchSplitting(const SplitGraph &SG, unsigned NumParts,
                           SubmitProposalFn SubmitProposal);

  void run();

private:
  struct WorkListEntry {
    WorkListEntry(const BitVector &BV) : Cluster(BV) {}

    unsigned NumNonEntryNodes = 0;
    CostType TotalCost = 0;
    CostType CostExcludingGraphEntryPoints = 0;
    BitVector Cluster;
  };

  /// Collects all graph entry points's clusters and sort them so the most
  /// expensive clusters are viewed first. This will merge clusters together if
  /// they share a non-copyable dependency.
  void setupWorkList();

  /// Recursive function that assigns the worklist item at \p Idx into a
  /// partition of \p SP.
  ///
  /// \p Depth is the current search depth. When this value is equal to
  /// \ref MaxDepth, we can no longer recurse.
  ///
  /// This function only recurses if there is more than one possible assignment,
  /// otherwise it is iterative to avoid creating a call stack that is as big as
  /// \ref WorkList.
  void pickPartition(unsigned Depth, unsigned Idx, SplitProposal SP);

  /// \return A pair: first element is the PID of the partition that has the
  /// most similarities with \p Entry, or \ref InvalidPID if no partition was
  /// found with at least one element in common. The second element is the
  /// aggregated cost of all dependencies in common between \p Entry and that
  /// partition.
  std::pair<unsigned, CostType>
  findMostSimilarPartition(const WorkListEntry &Entry, const SplitProposal &SP);

  const SplitGraph &SG;
  unsigned NumParts;
  SubmitProposalFn SubmitProposal;

  // A Cluster is considered large when its cost, excluding entry points,
  // exceeds this value.
  CostType LargeClusterThreshold = 0;
  unsigned NumProposalsSubmitted = 0;
  SmallVector<WorkListEntry> WorkList;
};

RecursiveSearchSplitting::RecursiveSearchSplitting(
    const SplitGraph &SG, unsigned NumParts, SubmitProposalFn SubmitProposal)
    : SG(SG), NumParts(NumParts), SubmitProposal(SubmitProposal) {
  // arbitrary max value as a safeguard. Anything above 10 will already be
  // slow, this is just a max value to prevent extreme resource exhaustion or
  // unbounded run time.
  if (MaxDepth > 16)
    report_fatal_error("[amdgpu-split-module] search depth of " +
                       Twine(MaxDepth) + " is too high!");
  LargeClusterThreshold =
      (LargeFnFactor != 0.0)
          ? CostType(((SG.getModuleCost() / NumParts) * LargeFnFactor))
          : std::numeric_limits<CostType>::max();
  LLVM_DEBUG(dbgs() << "[recursive search] large cluster threshold set at "
                    << LargeClusterThreshold << "\n");
}

void RecursiveSearchSplitting::run() {
  {
    SplitModuleTimer SMT("recursive_search_prepare", "preparing worklist");
    setupWorkList();
  }

  {
    SplitModuleTimer SMT("recursive_search_pick", "partitioning");
    SplitProposal SP(SG, NumParts);
    pickPartition(/*BranchDepth=*/0, /*Idx=*/0, SP);
  }
}

void RecursiveSearchSplitting::setupWorkList() {
  // e.g. if A and B are two worklist item, and they both call a non copyable
  // dependency C, this does:
  //    A=C
  //    B=C
  // => NodeEC will create a single group (A, B, C) and we create a new
  // WorkList entry for that group.

  EquivalenceClasses<unsigned> NodeEC;
  for (const SplitGraph::Node *N : SG.nodes()) {
    if (!N->isGraphEntryPoint())
      continue;

    NodeEC.insert(N->getID());
    N->visitAllDependencies([&](const SplitGraph::Node &Dep) {
      if (&Dep != N && Dep.isNonCopyable())
        NodeEC.unionSets(N->getID(), Dep.getID());
    });
  }

  for (auto I = NodeEC.begin(), E = NodeEC.end(); I != E; ++I) {
    if (!I->isLeader())
      continue;

    BitVector Cluster = SG.createNodesBitVector();
    for (auto MI = NodeEC.member_begin(I); MI != NodeEC.member_end(); ++MI) {
      const SplitGraph::Node &N = SG.getNode(*MI);
      if (N.isGraphEntryPoint())
        N.getDependencies(Cluster);
    }
    WorkList.emplace_back(std::move(Cluster));
  }

  // Calculate costs and other useful information.
  for (WorkListEntry &Entry : WorkList) {
    for (unsigned NodeID : Entry.Cluster.set_bits()) {
      const SplitGraph::Node &N = SG.getNode(NodeID);
      const CostType Cost = N.getIndividualCost();

      Entry.TotalCost += Cost;
      if (!N.isGraphEntryPoint()) {
        Entry.CostExcludingGraphEntryPoints += Cost;
        ++Entry.NumNonEntryNodes;
      }
    }
  }

  stable_sort(WorkList, [](const WorkListEntry &A, const WorkListEntry &B) {
    if (A.TotalCost != B.TotalCost)
      return A.TotalCost > B.TotalCost;

    if (A.CostExcludingGraphEntryPoints != B.CostExcludingGraphEntryPoints)
      return A.CostExcludingGraphEntryPoints > B.CostExcludingGraphEntryPoints;

    if (A.NumNonEntryNodes != B.NumNonEntryNodes)
      return A.NumNonEntryNodes > B.NumNonEntryNodes;

    return A.Cluster.count() > B.Cluster.count();
  });

  LLVM_DEBUG({
    dbgs() << "[recursive search] worklist:\n";
    for (const auto &[Idx, Entry] : enumerate(WorkList)) {
      dbgs() << "  - [" << Idx << "]: ";
      for (unsigned NodeID : Entry.Cluster.set_bits())
        dbgs() << NodeID << " ";
      dbgs() << "(total_cost:" << Entry.TotalCost
             << ", cost_excl_entries:" << Entry.CostExcludingGraphEntryPoints
             << ")\n";
    }
  });
}

void RecursiveSearchSplitting::pickPartition(unsigned Depth, unsigned Idx,
                                             SplitProposal SP) {
  while (Idx < WorkList.size()) {
    // Step 1: Determine candidate PIDs.
    //
    const WorkListEntry &Entry = WorkList[Idx];
    const BitVector &Cluster = Entry.Cluster;

    // Default option is to do load-balancing, AKA assign to least pressured
    // partition.
    const unsigned CheapestPID = SP.findCheapestPartition();
    assert(CheapestPID != InvalidPID);

    // Explore assigning to the kernel that contains the most dependencies in
    // common.
    const auto [MostSimilarPID, SimilarDepsCost] =
        findMostSimilarPartition(Entry, SP);

    // We can chose to explore only one path if we only have one valid path, or
    // if we reached maximum search depth and can no longer branch out.
    unsigned SinglePIDToTry = InvalidPID;
    if (MostSimilarPID == InvalidPID) // no similar PID found
      SinglePIDToTry = CheapestPID;
    else if (MostSimilarPID == CheapestPID) // both landed on the same PID
      SinglePIDToTry = CheapestPID;
    else if (Depth >= MaxDepth) {
      // We have to choose one path. Use a heuristic to guess which one will be
      // more appropriate.
      if (Entry.CostExcludingGraphEntryPoints > LargeClusterThreshold) {
        // Check if the amount of code in common makes it worth it.
        assert(SimilarDepsCost && Entry.CostExcludingGraphEntryPoints);
        const double Ratio = static_cast<double>(SimilarDepsCost) /
                             Entry.CostExcludingGraphEntryPoints;
        assert(Ratio >= 0.0 && Ratio <= 1.0);
        if (Ratio > LargeFnOverlapForMerge) {
          // For debug, just print "L", so we'll see "L3=P3" for instance, which
          // will mean we reached max depth and chose P3 based on this
          // heuristic.
          LLVM_DEBUG(dbgs() << 'L');
          SinglePIDToTry = MostSimilarPID;
        }
      } else
        SinglePIDToTry = CheapestPID;
    }

    // Step 2: Explore candidates.

    // When we only explore one possible path, and thus branch depth doesn't
    // increase, do not recurse, iterate instead.
    if (SinglePIDToTry != InvalidPID) {
      LLVM_DEBUG(dbgs() << Idx << "=P" << SinglePIDToTry << ' ');
      // Only one path to explore, don't clone SP, don't increase depth.
      SP.add(SinglePIDToTry, Cluster);
      ++Idx;
      continue;
    }

    assert(MostSimilarPID != InvalidPID);

    // We explore multiple paths: recurse at increased depth, then stop this
    // function.

    LLVM_DEBUG(dbgs() << '\n');

    // lb = load balancing = put in cheapest partition
    {
      SplitProposal BranchSP = SP;
      LLVM_DEBUG(dbgs().indent(Depth)
                 << " [lb] " << Idx << "=P" << CheapestPID << "? ");
      BranchSP.add(CheapestPID, Cluster);
      pickPartition(Depth + 1, Idx + 1, BranchSP);
    }

    // ms = most similar = put in partition with the most in common
    {
      SplitProposal BranchSP = SP;
      LLVM_DEBUG(dbgs().indent(Depth)
                 << " [ms] " << Idx << "=P" << MostSimilarPID << "? ");
      BranchSP.add(MostSimilarPID, Cluster);
      pickPartition(Depth + 1, Idx + 1, BranchSP);
    }

    return;
  }

  // Step 3: If we assigned all WorkList items, submit the proposal.

  assert(Idx == WorkList.size());
  assert(NumProposalsSubmitted <= (2u << MaxDepth) &&
         "Search got out of bounds?");
  SP.setName("recursive_search (depth=" + std::to_string(Depth) + ") #" +
             std::to_string(NumProposalsSubmitted++));
  LLVM_DEBUG(dbgs() << '\n');
  SubmitProposal(SP);
}

std::pair<unsigned, CostType>
RecursiveSearchSplitting::findMostSimilarPartition(const WorkListEntry &Entry,
                                                   const SplitProposal &SP) {
  if (!Entry.NumNonEntryNodes)
    return {InvalidPID, 0};

  // We take the partition that is the most similar using Cost as a metric.
  // So we take the set of nodes in common, compute their aggregated cost, and
  // pick the partition with the highest cost in common.
  unsigned ChosenPID = InvalidPID;
  CostType ChosenCost = 0;
  for (unsigned PID = 0; PID < NumParts; ++PID) {
    BitVector BV = SP[PID];
    BV &= Entry.Cluster; // FIXME: & doesn't work between BVs?!

    if (BV.none())
      continue;

    const CostType Cost = SG.calculateCost(BV);

    if (ChosenPID == InvalidPID || ChosenCost < Cost ||
        (ChosenCost == Cost && PID > ChosenPID)) {
      ChosenPID = PID;
      ChosenCost = Cost;
    }
  }

  return {ChosenPID, ChosenCost};
}

//===----------------------------------------------------------------------===//
// DOTGraph Printing Support
//===----------------------------------------------------------------------===//

const SplitGraph::Node *mapEdgeToDst(const SplitGraph::Edge *E) {
  return E->Dst;
}

using SplitGraphEdgeDstIterator =
    mapped_iterator<SplitGraph::edges_iterator, decltype(&mapEdgeToDst)>;

} // namespace

template <> struct GraphTraits<SplitGraph> {
  using NodeRef = const SplitGraph::Node *;
  using nodes_iterator = SplitGraph::nodes_iterator;
  using ChildIteratorType = SplitGraphEdgeDstIterator;

  using EdgeRef = const SplitGraph::Edge *;
  using ChildEdgeIteratorType = SplitGraph::edges_iterator;

  static NodeRef getEntryNode(NodeRef N) { return N; }

  static ChildIteratorType child_begin(NodeRef Ref) {
    return {Ref->outgoing_edges().begin(), mapEdgeToDst};
  }
  static ChildIteratorType child_end(NodeRef Ref) {
    return {Ref->outgoing_edges().end(), mapEdgeToDst};
  }

  static nodes_iterator nodes_begin(const SplitGraph &G) {
    return G.nodes().begin();
  }
  static nodes_iterator nodes_end(const SplitGraph &G) {
    return G.nodes().end();
  }
};

template <> struct DOTGraphTraits<SplitGraph> : public DefaultDOTGraphTraits {
  DOTGraphTraits(bool IsSimple = false) : DefaultDOTGraphTraits(IsSimple) {}

  static std::string getGraphName(const SplitGraph &SG) {
    return SG.getModule().getName().str();
  }

  std::string getNodeLabel(const SplitGraph::Node *N, const SplitGraph &SG) {
    return N->getName().str();
  }

  static std::string getNodeDescription(const SplitGraph::Node *N,
                                        const SplitGraph &SG) {
    std::string Result;
    if (N->isEntryFunctionCC())
      Result += "entry-fn-cc ";
    if (N->isNonCopyable())
      Result += "non-copyable ";
    Result += "cost:" + std::to_string(N->getIndividualCost());
    return Result;
  }

  static std::string getNodeAttributes(const SplitGraph::Node *N,
                                       const SplitGraph &SG) {
    return N->hasAnyIncomingEdges() ? "" : "color=\"red\"";
  }

  static std::string getEdgeAttributes(const SplitGraph::Node *N,
                                       SplitGraphEdgeDstIterator EI,
                                       const SplitGraph &SG) {

    switch ((*EI.getCurrent())->Kind) {
    case SplitGraph::EdgeKind::DirectCall:
      return "";
    case SplitGraph::EdgeKind::IndirectCall:
      return "style=\"dashed\"";
    }
    llvm_unreachable("Unknown SplitGraph::EdgeKind enum");
  }
};

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

namespace {

// If we didn't externalize GVs, then local GVs need to be conservatively
// imported into every module (including their initializers), and then cleaned
// up afterwards.
static bool needsConservativeImport(const GlobalValue *GV) {
  if (const auto *Var = dyn_cast<GlobalVariable>(GV))
    return Var->hasLocalLinkage();
  return isa<GlobalAlias>(GV);
}

/// Prints a summary of the partition \p N, represented by module \p M, to \p
/// OS.
static void printPartitionSummary(raw_ostream &OS, unsigned N, const Module &M,
                                  unsigned PartCost, unsigned ModuleCost) {
  OS << "*** Partition P" << N << " ***\n";

  for (const auto &Fn : M) {
    if (!Fn.isDeclaration())
      OS << " - [function] " << Fn.getName() << "\n";
  }

  for (const auto &GV : M.globals()) {
    if (GV.hasInitializer())
      OS << " - [global] " << GV.getName() << "\n";
  }

  OS << "Partition contains " << formatRatioOf(PartCost, ModuleCost)
     << "% of the source\n";
}

static void evaluateProposal(SplitProposal &Best, SplitProposal New) {
  SplitModuleTimer SMT("proposal_evaluation", "proposal ranking algorithm");

  LLVM_DEBUG({
    New.verifyCompleteness();
    if (DebugProposalSearch)
      New.print(dbgs());
  });

  const double CurBScore = Best.getBottleneckScore();
  const double CurCSScore = Best.getCodeSizeScore();
  const double NewBScore = New.getBottleneckScore();
  const double NewCSScore = New.getCodeSizeScore();

  // TODO: Improve this
  //    We can probably lower the precision of the comparison at first
  //    e.g. if we have
  //      - (Current): BScore: 0.489 CSCore 1.105
  //      - (New): BScore: 0.475 CSCore 1.305
  //    Currently we'd choose the new one because the bottleneck score is
  //    lower, but the new one duplicates more code. It may be worth it to
  //    discard the new proposal as the impact on build time is negligible.

  // Compare them
  bool IsBest = false;
  if (NewBScore < CurBScore)
    IsBest = true;
  else if (NewBScore == CurBScore)
    IsBest = (NewCSScore < CurCSScore); // Use code size as tie breaker.

  if (IsBest)
    Best = std::move(New);

  LLVM_DEBUG(if (DebugProposalSearch) {
    if (IsBest)
      dbgs() << "[search] new best proposal!\n";
    else
      dbgs() << "[search] discarding - not profitable\n";
  });
}

/// Trivial helper to create an identical copy of \p M.
static std::unique_ptr<Module> cloneAll(const Module &M) {
  ValueToValueMapTy VMap;
  return CloneModule(M, VMap, [&](const GlobalValue *GV) { return true; });
}

/// Writes \p SG as a DOTGraph to \ref ModuleDotCfgDir if requested.
static void writeDOTGraph(const SplitGraph &SG) {
  if (ModuleDotCfgOutput.empty())
    return;

  std::error_code EC;
  raw_fd_ostream OS(ModuleDotCfgOutput, EC);
  if (EC) {
    errs() << "[" DEBUG_TYPE "]: cannot open '" << ModuleDotCfgOutput
           << "' - DOTGraph will not be printed\n";
  }
  WriteGraph(OS, SG, /*ShortName=*/false,
             /*Title=*/SG.getModule().getName());
}

static void splitAMDGPUModule(
    GetTTIFn GetTTI, Module &M, unsigned NumParts,
    function_ref<void(std::unique_ptr<Module> MPart)> ModuleCallback) {
  CallGraph CG(M);

  // Externalize functions whose address are taken.
  //
  // This is needed because partitioning is purely based on calls, but sometimes
  // a kernel/function may just look at the address of another local function
  // and not do anything (no calls). After partitioning, that local function may
  // end up in a different module (so it's just a declaration in the module
  // where its address is taken), which emits a "undefined hidden symbol" linker
  // error.
  //
  // Additionally, it guides partitioning to not duplicate this function if it's
  // called directly at some point.
  //
  // TODO: Could we be smarter about this ? This makes all functions whose
  // addresses are taken non-copyable. We should probably model this type of
  // constraint in the graph and use it to guide splitting, instead of
  // externalizing like this. Maybe non-copyable should really mean "keep one
  // visible copy, then internalize all other copies" for some functions?
  if (!NoExternalizeOnAddrTaken) {
    for (auto &Fn : M) {
      // TODO: Should aliases count? Probably not but they're so rare I'm not
      // sure it's worth fixing.
      if (Fn.hasLocalLinkage() && Fn.hasAddressTaken()) {
        LLVM_DEBUG(dbgs() << "[externalize] "; Fn.printAsOperand(dbgs());
                   dbgs() << " because its address is taken\n");
        externalize(Fn);
      }
    }
  }

  // Externalize local GVs, which avoids duplicating their initializers, which
  // in turns helps keep code size in check.
  if (!NoExternalizeGlobals) {
    for (auto &GV : M.globals()) {
      if (GV.hasLocalLinkage())
        LLVM_DEBUG(dbgs() << "[externalize] GV " << GV.getName() << '\n');
      externalize(GV);
    }
  }

  // Start by calculating the cost of every function in the module, as well as
  // the module's overall cost.
  FunctionsCostMap FnCosts;
  const CostType ModuleCost = calculateFunctionCosts(GetTTI, M, FnCosts);

  // Build the SplitGraph, which represents the module's functions and models
  // their dependencies accurately.
  SplitGraph SG(M, FnCosts, ModuleCost);
  SG.buildGraph(CG);

  if (SG.empty()) {
    LLVM_DEBUG(
        dbgs()
        << "[!] no nodes in graph, input is empty - no splitting possible\n");
    ModuleCallback(cloneAll(M));
    return;
  }

  LLVM_DEBUG({
    dbgs() << "[graph] nodes:\n";
    for (const SplitGraph::Node *N : SG.nodes()) {
      dbgs() << "  - [" << N->getID() << "]: " << N->getName() << " "
             << (N->isGraphEntryPoint() ? "(entry)" : "") << " "
             << (N->isNonCopyable() ? "(noncopyable)" : "") << "\n";
    }
  });

  writeDOTGraph(SG);

  LLVM_DEBUG(dbgs() << "[search] testing splitting strategies\n");

  std::optional<SplitProposal> Proposal;
  const auto EvaluateProposal = [&](SplitProposal SP) {
    SP.calculateScores();
    if (!Proposal)
      Proposal = std::move(SP);
    else
      evaluateProposal(*Proposal, std::move(SP));
  };

  // TODO: It would be very easy to create new strategies by just adding a base
  // class to RecursiveSearchSplitting and abstracting it away.
  RecursiveSearchSplitting(SG, NumParts, EvaluateProposal).run();
  LLVM_DEBUG(if (Proposal) dbgs() << "[search done] selected proposal: "
                                  << Proposal->getName() << "\n";);

  if (!Proposal) {
    LLVM_DEBUG(dbgs() << "[!] no proposal made, no splitting possible!\n");
    ModuleCallback(cloneAll(M));
    return;
  }

  LLVM_DEBUG(Proposal->print(dbgs()););

  std::optional<raw_fd_ostream> SummariesOS;
  if (!PartitionSummariesOutput.empty()) {
    std::error_code EC;
    SummariesOS.emplace(PartitionSummariesOutput, EC);
    if (EC)
      errs() << "[" DEBUG_TYPE "]: cannot open '" << PartitionSummariesOutput
             << "' - Partition summaries will not be printed\n";
  }

  for (unsigned PID = 0; PID < NumParts; ++PID) {
    SplitModuleTimer SMT2("modules_creation",
                          "creating modules for each partition");
    LLVM_DEBUG(dbgs() << "[split] creating new modules\n");

    DenseSet<const Function *> FnsInPart;
    for (unsigned NodeID : (*Proposal)[PID].set_bits())
      FnsInPart.insert(&SG.getNode(NodeID).getFunction());

    ValueToValueMapTy VMap;
    CostType PartCost = 0;
    std::unique_ptr<Module> MPart(
        CloneModule(M, VMap, [&](const GlobalValue *GV) {
          // Functions go in their assigned partition.
          if (const auto *Fn = dyn_cast<Function>(GV)) {
            if (FnsInPart.contains(Fn)) {
              PartCost += SG.getCost(*Fn);
              return true;
            }
            return false;
          }

          // Everything else goes in the first partition.
          return needsConservativeImport(GV) || PID == 0;
        }));

    // FIXME: Aliases aren't seen often, and their handling isn't perfect so
    // bugs are possible.

    // Clean-up conservatively imported GVs without any users.
    for (auto &GV : make_early_inc_range(MPart->global_values())) {
      if (needsConservativeImport(&GV) && GV.use_empty())
        GV.eraseFromParent();
    }

    if (SummariesOS)
      printPartitionSummary(*SummariesOS, PID, *MPart, PartCost, ModuleCost);

    LLVM_DEBUG(
        printPartitionSummary(dbgs(), PID, *MPart, PartCost, ModuleCost));

    ModuleCallback(std::move(MPart));
  }
}
} // namespace

PreservedAnalyses AMDGPUSplitModulePass::run(Module &M,
                                             ModuleAnalysisManager &MAM) {
  SplitModuleTimer SMT(
      "total", "total pass runtime (incl. potentially waiting for lockfile)");

  FunctionAnalysisManager &FAM =
      MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  const auto TTIGetter = [&FAM](Function &F) -> const TargetTransformInfo & {
    return FAM.getResult<TargetIRAnalysis>(F);
  };

  bool Done = false;
#ifndef NDEBUG
  if (UseLockFile) {
    SmallString<128> LockFilePath;
    sys::path::system_temp_directory(/*ErasedOnReboot=*/true, LockFilePath);
    sys::path::append(LockFilePath, "amdgpu-split-module-debug");
    LLVM_DEBUG(dbgs() << DEBUG_TYPE " using lockfile '" << LockFilePath
                      << "'\n");

    while (true) {
      llvm::LockFileManager Locked(LockFilePath.str());
      switch (Locked) {
      case LockFileManager::LFS_Error:
        LLVM_DEBUG(
            dbgs() << "[amdgpu-split-module] unable to acquire lockfile, debug "
                      "output may be mangled by other processes\n");
        Locked.unsafeRemoveLockFile();
        break;
      case LockFileManager::LFS_Owned:
        break;
      case LockFileManager::LFS_Shared: {
        switch (Locked.waitForUnlock()) {
        case LockFileManager::Res_Success:
          break;
        case LockFileManager::Res_OwnerDied:
          continue; // try again to get the lock.
        case LockFileManager::Res_Timeout:
          LLVM_DEBUG(
              dbgs()
              << "[amdgpu-split-module] unable to acquire lockfile, debug "
                 "output may be mangled by other processes\n");
          Locked.unsafeRemoveLockFile();
          break; // give up
        }
        break;
      }
      }

      splitAMDGPUModule(TTIGetter, M, N, ModuleCallback);
      Done = true;
      break;
    }
  }
#endif

  if (!Done)
    splitAMDGPUModule(TTIGetter, M, N, ModuleCallback);

  // We can change linkage/visibilities in the input, consider that nothing is
  // preserved just to be safe. This pass runs last anyway.
  return PreservedAnalyses::none();
}
} // namespace llvm
