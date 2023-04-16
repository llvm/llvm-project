//==-- MemProfContextDisambiguation.cpp - Disambiguate contexts -------------=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements support for context disambiguation of allocation
// calls for profile guided heap optimization. Specifically, it uses Memprof
// profiles which indicate context specific allocation behavior (currently
// distinguishing cold vs hot memory allocations). Cloning is performed to
// expose the cold allocation call contexts, and the allocation calls are
// subsequently annotated with an attribute for later transformation.
//
// The transformations can be performed either directly on IR (regular LTO), or
// on a ThinLTO index (and later applied to the IR during the ThinLTO backend).
// Both types of LTO operate on a the same base graph representation, which
// uses CRTP to support either IR or Index formats.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/MemProfContextDisambiguation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/MemoryProfileInfo.h"
#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO.h"
#include <sstream>
#include <vector>
using namespace llvm;
using namespace llvm::memprof;

#define DEBUG_TYPE "memprof-context-disambiguation"

static cl::opt<std::string> DotFilePathPrefix(
    "memprof-dot-file-path-prefix", cl::init(""), cl::Hidden,
    cl::value_desc("filename"),
    cl::desc("Specify the path prefix of the MemProf dot files."));

static cl::opt<bool> ExportToDot("memprof-export-to-dot", cl::init(false),
                                 cl::Hidden,
                                 cl::desc("Export graph to dot files."));

static cl::opt<bool>
    DumpCCG("memprof-dump-ccg", cl::init(false), cl::Hidden,
            cl::desc("Dump CallingContextGraph to stdout after each stage."));

static cl::opt<bool>
    VerifyCCG("memprof-verify-ccg", cl::init(false), cl::Hidden,
              cl::desc("Perform verification checks on CallingContextGraph."));

static cl::opt<bool>
    VerifyNodes("memprof-verify-nodes", cl::init(false), cl::Hidden,
                cl::desc("Perform frequent verification checks on nodes."));

inline bool hasSingleAllocType(uint8_t AllocTypes) {
  switch (AllocTypes) {
  case (uint8_t)AllocationType::Cold:
  case (uint8_t)AllocationType::NotCold:
    return true;
    break;
  case (uint8_t)AllocationType::None:
    assert(false);
    break;
  default:
    return false;
    break;
  }
  llvm_unreachable("invalid alloc type");
}

/// CRTP base for graphs built from either IR or ThinLTO summary index.
///
/// The graph represents the call contexts in all memprof metadata on allocation
/// calls, with nodes for the allocations themselves, as well as for the calls
/// in each context. The graph is initially built from the allocation memprof
/// metadata (or summary) MIBs. It is then updated to match calls with callsite
/// metadata onto the nodes, updating it to reflect any inlining performed on
/// those calls.
///
/// Each MIB (representing an allocation's call context with allocation
/// behavior) is assigned a unique context id during the graph build. The edges
/// and nodes in the graph are decorated with the context ids they carry. This
/// is used to correctly update the graph when cloning is performed so that we
/// can uniquify the context for a single (possibly cloned) allocation.
template <typename DerivedCCG, typename FuncTy, typename CallTy>
class CallsiteContextGraph {
public:
  CallsiteContextGraph() = default;
  CallsiteContextGraph(const CallsiteContextGraph &) = default;
  CallsiteContextGraph(CallsiteContextGraph &&) = default;

  /// Main entry point to perform analysis and transformations on graph.
  bool process();

  void dump() const;
  void print(raw_ostream &OS) const;

  friend raw_ostream &operator<<(raw_ostream &OS,
                                 const CallsiteContextGraph &CCG) {
    CCG.print(OS);
    return OS;
  }

  friend struct GraphTraits<
      const CallsiteContextGraph<DerivedCCG, FuncTy, CallTy> *>;
  friend struct DOTGraphTraits<
      const CallsiteContextGraph<DerivedCCG, FuncTy, CallTy> *>;

  void exportToDot(std::string Label) const;

  /// Represents a function clone via FuncTy pointer and clone number pair.
  struct FuncInfo final
      : public std::pair<FuncTy *, unsigned /*Clone number*/> {
    using Base = std::pair<FuncTy *, unsigned>;
    FuncInfo(const Base &B) : Base(B) {}
    FuncInfo(FuncTy *F = nullptr, unsigned CloneNo = 0) : Base(F, CloneNo) {}
    explicit operator bool() const { return this->first != nullptr; }
    FuncTy *func() const { return this->first; }
    unsigned cloneNo() const { return this->second; }
  };

  /// Represents a callsite clone via CallTy and clone number pair.
  struct CallInfo final : public std::pair<CallTy, unsigned /*Clone number*/> {
    using Base = std::pair<CallTy, unsigned>;
    CallInfo(const Base &B) : Base(B) {}
    CallInfo(CallTy Call = nullptr, unsigned CloneNo = 0)
        : Base(Call, CloneNo) {}
    explicit operator bool() const { return (bool)this->first; }
    CallTy call() const { return this->first; }
    unsigned cloneNo() const { return this->second; }
    void setCloneNo(unsigned N) { this->second = N; }
    void print(raw_ostream &OS) const {
      if (!operator bool()) {
        assert(!cloneNo());
        OS << "null Call";
        return;
      }
      call()->print(OS);
      OS << "\t(clone " << cloneNo() << ")";
    }
    void dump() const {
      print(dbgs());
      dbgs() << "\n";
    }
    friend raw_ostream &operator<<(raw_ostream &OS, const CallInfo &Call) {
      Call.print(OS);
      return OS;
    }
  };

  struct ContextEdge;

  /// Node in the Callsite Context Graph
  struct ContextNode {
    // Keep this for now since in the IR case where we have an Instruction* it
    // is not as immediately discoverable. Used for printing richer information
    // when dumping graph.
    bool IsAllocation;

    // Keeps track of when the Call was reset to null because there was
    // recursion.
    bool Recursive = false;

    // The corresponding allocation or interior call.
    CallInfo Call;

    // For alloc nodes this is a unique id assigned when constructed, and for
    // callsite stack nodes it is the original stack id when the node is
    // constructed from the memprof MIB metadata on the alloc nodes. Note that
    // this is only used when matching callsite metadata onto the stack nodes
    // created when processing the allocation memprof MIBs, and for labeling
    // nodes in the dot graph. Therefore we don't bother to assign a value for
    // clones.
    uint64_t OrigStackOrAllocId = 0;

    // This will be formed by ORing together the AllocationType enum values
    // for contexts including this node.
    uint8_t AllocTypes = 0;

    // Edges to all callees in the profiled call stacks.
    // TODO: Should this be a map (from Callee node) for more efficient lookup?
    std::vector<std::shared_ptr<ContextEdge>> CalleeEdges;

    // Edges to all callers in the profiled call stacks.
    // TODO: Should this be a map (from Caller node) for more efficient lookup?
    std::vector<std::shared_ptr<ContextEdge>> CallerEdges;

    // The set of IDs for contexts including this node.
    DenseSet<uint32_t> ContextIds;

    // List of clones of this ContextNode, initially empty.
    std::vector<ContextNode *> Clones;

    // If a clone, points to the original uncloned node.
    ContextNode *CloneOf = nullptr;

    ContextNode(bool IsAllocation) : IsAllocation(IsAllocation), Call() {}

    ContextNode(bool IsAllocation, CallInfo C)
        : IsAllocation(IsAllocation), Call(C) {}

    std::unique_ptr<ContextNode> clone() {
      auto Clone = std::make_unique<ContextNode>(IsAllocation, Call);
      if (CloneOf) {
        CloneOf->Clones.push_back(Clone.get());
        Clone->CloneOf = CloneOf;
      } else {
        Clones.push_back(Clone.get());
        Clone->CloneOf = this;
      }
      return Clone;
    }

    ContextNode *getOrigNode() {
      if (!CloneOf)
        return this;
      return CloneOf;
    }

    void addOrUpdateCallerEdge(ContextNode *Caller, AllocationType AllocType,
                               unsigned int ContextId);

    ContextEdge *findEdgeFromCallee(const ContextNode *Callee);
    ContextEdge *findEdgeFromCaller(const ContextNode *Caller);
    void eraseCalleeEdge(const ContextEdge *Edge);
    void eraseCallerEdge(const ContextEdge *Edge);

    void setCall(CallInfo C) { Call = C; }

    bool hasCall() const { return (bool)Call.call(); }

    void printCall(raw_ostream &OS) const { Call.print(OS); }

    // True if this node was effectively removed from the graph, in which case
    // its context id set, caller edges, and callee edges should all be empty.
    bool isRemoved() const {
      assert(ContextIds.empty() ==
             (CalleeEdges.empty() && CallerEdges.empty()));
      return ContextIds.empty();
    }

    void dump() const;
    void print(raw_ostream &OS) const;

    friend raw_ostream &operator<<(raw_ostream &OS, const ContextNode &Node) {
      Node.print(OS);
      return OS;
    }
  };

  /// Edge in the Callsite Context Graph from a ContextNode N to a caller or
  /// callee.
  struct ContextEdge {
    ContextNode *Callee;
    ContextNode *Caller;

    // This will be formed by ORing together the AllocationType enum values
    // for contexts including this edge.
    uint8_t AllocTypes = 0;

    // The set of IDs for contexts including this edge.
    DenseSet<uint32_t> ContextIds;

    ContextEdge(ContextNode *Callee, ContextNode *Caller, uint8_t AllocType,
                DenseSet<uint32_t> ContextIds)
        : Callee(Callee), Caller(Caller), AllocTypes(AllocType),
          ContextIds(ContextIds) {}

    DenseSet<uint32_t> &getContextIds() { return ContextIds; }

    void dump() const;
    void print(raw_ostream &OS) const;

    friend raw_ostream &operator<<(raw_ostream &OS, const ContextEdge &Edge) {
      Edge.print(OS);
      return OS;
    }
  };

protected:
  /// Get a list of nodes corresponding to the stack ids in the given callsite
  /// context.
  template <class NodeT, class IteratorT>
  std::vector<uint64_t>
  getStackIdsWithContextNodes(CallStack<NodeT, IteratorT> &CallsiteContext);

  /// Adds nodes for the given allocation and any stack ids on its memprof MIB
  /// metadata (or summary).
  ContextNode *addAllocNode(CallInfo Call, const FuncTy *F);

  /// Adds nodes for the given MIB stack ids.
  template <class NodeT, class IteratorT>
  void addStackNodesForMIB(ContextNode *AllocNode,
                           CallStack<NodeT, IteratorT> &StackContext,
                           CallStack<NodeT, IteratorT> &CallsiteContext,
                           AllocationType AllocType);

  /// Matches all callsite metadata (or summary) to the nodes created for
  /// allocation memprof MIB metadata, synthesizing new nodes to reflect any
  /// inlining performed on those callsite instructions.
  void updateStackNodes();

  /// Update graph to conservatively handle any callsite stack nodes that target
  /// multiple different callee target functions.
  void handleCallsitesWithMultipleTargets();

  /// Save lists of calls with MemProf metadata in each function, for faster
  /// iteration.
  std::vector<std::pair<FuncTy *, std::vector<CallInfo>>>
      FuncToCallsWithMetadata;

  /// Map from callsite node to the enclosing caller function.
  std::map<const ContextNode *, const FuncTy *> NodeToCallingFunc;

private:
  using EdgeIter = typename std::vector<std::shared_ptr<ContextEdge>>::iterator;

  using CallContextInfo = std::tuple<CallTy, std::vector<uint64_t>,
                                     const FuncTy *, DenseSet<uint32_t>>;

  /// Assigns the given Node to calls at or inlined into the location with
  /// the Node's stack id, after post order traversing and processing its
  /// caller nodes. Uses the call information recorded in the given
  /// StackIdToMatchingCalls map, and creates new nodes for inlined sequences
  /// as needed. Called by updateStackNodes which sets up the given
  /// StackIdToMatchingCalls map.
  void assignStackNodesPostOrder(
      ContextNode *Node, DenseSet<const ContextNode *> &Visited,
      DenseMap<uint64_t, std::vector<CallContextInfo>> &StackIdToMatchingCalls);

  /// Duplicates the given set of context ids, updating the provided
  /// map from each original id with the newly generated context ids,
  /// and returning the new duplicated id set.
  DenseSet<uint32_t> duplicateContextIds(
      const DenseSet<uint32_t> &StackSequenceContextIds,
      DenseMap<uint32_t, DenseSet<uint32_t>> &OldToNewContextIds);

  /// Propagates all duplicated context ids across the graph.
  void propagateDuplicateContextIds(
      const DenseMap<uint32_t, DenseSet<uint32_t>> &OldToNewContextIds);

  /// Connect the NewNode to OrigNode's callees if TowardsCallee is true,
  /// else to its callers. Also updates OrigNode's edges to remove any context
  /// ids moved to the newly created edge.
  void connectNewNode(ContextNode *NewNode, ContextNode *OrigNode,
                      bool TowardsCallee);

  /// Get the stack id corresponding to the given Id or Index (for IR this will
  /// return itself, for a summary index this will return the id recorded in the
  /// index for that stack id index value).
  uint64_t getStackId(uint64_t IdOrIndex) const {
    return static_cast<const DerivedCCG *>(this)->getStackId(IdOrIndex);
  }

  /// Returns true if the given call targets the given function.
  bool calleeMatchesFunc(CallTy Call, const FuncTy *Func) {
    return static_cast<DerivedCCG *>(this)->calleeMatchesFunc(Call, Func);
  }

  /// Get a list of nodes corresponding to the stack ids in the given
  /// callsite's context.
  std::vector<uint64_t> getStackIdsWithContextNodesForCall(CallTy Call) {
    return static_cast<DerivedCCG *>(this)->getStackIdsWithContextNodesForCall(
        Call);
  }

  /// Get the last stack id in the context for callsite.
  uint64_t getLastStackId(CallTy Call) {
    return static_cast<DerivedCCG *>(this)->getLastStackId(Call);
  }

  /// Gets a label to use in the dot graph for the given call clone in the given
  /// function.
  std::string getLabel(const FuncTy *Func, const CallTy Call,
                       unsigned CloneNo) const {
    return static_cast<const DerivedCCG *>(this)->getLabel(Func, Call, CloneNo);
  }

  /// Helpers to find the node corresponding to the given call or stackid.
  ContextNode *getNodeForInst(const CallInfo &C);
  ContextNode *getNodeForAlloc(const CallInfo &C);
  ContextNode *getNodeForStackId(uint64_t StackId);

  /// Removes the node information recorded for the given call.
  void unsetNodeForInst(const CallInfo &C);

  /// Computes the alloc type corresponding to the given context ids, by
  /// unioning their recorded alloc types.
  uint8_t computeAllocType(DenseSet<uint32_t> &ContextIds);

  /// Map from each context ID to the AllocationType assigned to that context.
  std::map<uint32_t, AllocationType> ContextIdToAllocationType;

  /// Identifies the context node created for a stack id when adding the MIB
  /// contexts to the graph. This is used to locate the context nodes when
  /// trying to assign the corresponding callsites with those stack ids to these
  /// nodes.
  std::map<uint64_t, ContextNode *> StackEntryIdToContextNodeMap;

  /// Maps to track the calls to their corresponding nodes in the graph.
  std::map<const CallInfo, ContextNode *> AllocationCallToContextNodeMap;
  std::map<const CallInfo, ContextNode *> NonAllocationCallToContextNodeMap;

  /// Owner of all ContextNode unique_ptrs.
  std::vector<std::unique_ptr<ContextNode>> NodeOwner;

  /// Perform sanity checks on graph when requested.
  void check() const;

  /// Keeps track of the last unique context id assigned.
  unsigned int LastContextId = 0;
};

template <typename DerivedCCG, typename FuncTy, typename CallTy>
using ContextNode =
    typename CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::ContextNode;
template <typename DerivedCCG, typename FuncTy, typename CallTy>
using ContextEdge =
    typename CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::ContextEdge;
template <typename DerivedCCG, typename FuncTy, typename CallTy>
using FuncInfo =
    typename CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::FuncInfo;
template <typename DerivedCCG, typename FuncTy, typename CallTy>
using CallInfo =
    typename CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::CallInfo;

/// CRTP derived class for graphs built from IR (regular LTO).
class ModuleCallsiteContextGraph
    : public CallsiteContextGraph<ModuleCallsiteContextGraph, Function,
                                  Instruction *> {
public:
  ModuleCallsiteContextGraph(Module &M);

private:
  friend CallsiteContextGraph<ModuleCallsiteContextGraph, Function,
                              Instruction *>;

  uint64_t getStackId(uint64_t IdOrIndex) const;
  bool calleeMatchesFunc(Instruction *Call, const Function *Func);
  uint64_t getLastStackId(Instruction *Call);
  std::vector<uint64_t> getStackIdsWithContextNodesForCall(Instruction *Call);
  std::string getLabel(const Function *Func, const Instruction *Call,
                       unsigned CloneNo) const;

  const Module &Mod;
};

/// Represents a call in the summary index graph, which can either be an
/// allocation or an interior callsite node in an allocation's context.
/// Holds a pointer to the corresponding data structure in the index.
struct IndexCall : public PointerUnion<CallsiteInfo *, AllocInfo *> {
  IndexCall() : PointerUnion() {}
  IndexCall(std::nullptr_t) : IndexCall() {}
  IndexCall(CallsiteInfo *StackNode) : PointerUnion(StackNode) {}
  IndexCall(AllocInfo *AllocNode) : PointerUnion(AllocNode) {}

  IndexCall *operator->() { return this; }

  void print(raw_ostream &OS) const {
    if (auto *AI = dyn_cast<AllocInfo *>())
      OS << *AI;
    else {
      auto *CI = dyn_cast<CallsiteInfo *>();
      assert(CI);
      OS << *CI;
    }
  }
};

/// CRTP derived class for graphs built from summary index (ThinLTO).
class IndexCallsiteContextGraph
    : public CallsiteContextGraph<IndexCallsiteContextGraph, FunctionSummary,
                                  IndexCall> {
public:
  IndexCallsiteContextGraph(
      ModuleSummaryIndex &Index,
      function_ref<bool(GlobalValue::GUID, const GlobalValueSummary *)>
          isPrevailing);

private:
  friend CallsiteContextGraph<IndexCallsiteContextGraph, FunctionSummary,
                              IndexCall>;

  uint64_t getStackId(uint64_t IdOrIndex) const;
  bool calleeMatchesFunc(IndexCall &Call, const FunctionSummary *Func);
  uint64_t getLastStackId(IndexCall &Call);
  std::vector<uint64_t> getStackIdsWithContextNodesForCall(IndexCall &Call);
  std::string getLabel(const FunctionSummary *Func, const IndexCall &Call,
                       unsigned CloneNo) const;

  // Saves mapping from function summaries containing memprof records back to
  // its VI, for use in checking and debugging.
  std::map<const FunctionSummary *, ValueInfo> FSToVIMap;

  const ModuleSummaryIndex &Index;
};

namespace {

struct FieldSeparator {
  bool Skip = true;
  const char *Sep;

  FieldSeparator(const char *Sep = ", ") : Sep(Sep) {}
};

raw_ostream &operator<<(raw_ostream &OS, FieldSeparator &FS) {
  if (FS.Skip) {
    FS.Skip = false;
    return OS;
  }
  return OS << FS.Sep;
}

// Map the uint8_t alloc types (which may contain NotCold|Cold) to the alloc
// type we should actually use on the corresponding allocation.
// If we can't clone a node that has NotCold+Cold alloc type, we will fall
// back to using NotCold. So don't bother cloning to distinguish NotCold+Cold
// from NotCold.
AllocationType allocTypeToUse(uint8_t AllocTypes) {
  assert(AllocTypes != (uint8_t)AllocationType::None);
  if (AllocTypes ==
      ((uint8_t)AllocationType::NotCold | (uint8_t)AllocationType::Cold))
    return AllocationType::NotCold;
  else
    return (AllocationType)AllocTypes;
}

} // end anonymous namespace

template <typename DerivedCCG, typename FuncTy, typename CallTy>
typename CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::ContextNode *
CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::getNodeForInst(
    const CallInfo &C) {
  ContextNode *Node = getNodeForAlloc(C);
  if (Node)
    return Node;

  auto NonAllocCallNode = NonAllocationCallToContextNodeMap.find(C);
  if (NonAllocCallNode != NonAllocationCallToContextNodeMap.end()) {
    return NonAllocCallNode->second;
  }
  return nullptr;
}

template <typename DerivedCCG, typename FuncTy, typename CallTy>
typename CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::ContextNode *
CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::getNodeForAlloc(
    const CallInfo &C) {
  auto AllocCallNode = AllocationCallToContextNodeMap.find(C);
  if (AllocCallNode != AllocationCallToContextNodeMap.end()) {
    return AllocCallNode->second;
  }
  return nullptr;
}

template <typename DerivedCCG, typename FuncTy, typename CallTy>
typename CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::ContextNode *
CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::getNodeForStackId(
    uint64_t StackId) {
  auto StackEntryNode = StackEntryIdToContextNodeMap.find(StackId);
  if (StackEntryNode != StackEntryIdToContextNodeMap.end())
    return StackEntryNode->second;
  return nullptr;
}

template <typename DerivedCCG, typename FuncTy, typename CallTy>
void CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::unsetNodeForInst(
    const CallInfo &C) {
  AllocationCallToContextNodeMap.erase(C) ||
      NonAllocationCallToContextNodeMap.erase(C);
  assert(!AllocationCallToContextNodeMap.count(C) &&
         !NonAllocationCallToContextNodeMap.count(C));
}

template <typename DerivedCCG, typename FuncTy, typename CallTy>
void CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::ContextNode::
    addOrUpdateCallerEdge(ContextNode *Caller, AllocationType AllocType,
                          unsigned int ContextId) {
  for (auto &Edge : CallerEdges) {
    if (Edge->Caller == Caller) {
      Edge->AllocTypes |= (uint8_t)AllocType;
      Edge->getContextIds().insert(ContextId);
      return;
    }
  }
  std::shared_ptr<ContextEdge> Edge = std::make_shared<ContextEdge>(
      this, Caller, (uint8_t)AllocType, DenseSet<uint32_t>({ContextId}));
  CallerEdges.push_back(Edge);
  Caller->CalleeEdges.push_back(Edge);
}

template <typename DerivedCCG, typename FuncTy, typename CallTy>
typename CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::ContextEdge *
CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::ContextNode::
    findEdgeFromCallee(const ContextNode *Callee) {
  for (const auto &Edge : CalleeEdges)
    if (Edge->Callee == Callee)
      return Edge.get();
  return nullptr;
}

template <typename DerivedCCG, typename FuncTy, typename CallTy>
typename CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::ContextEdge *
CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::ContextNode::
    findEdgeFromCaller(const ContextNode *Caller) {
  for (const auto &Edge : CallerEdges)
    if (Edge->Caller == Caller)
      return Edge.get();
  return nullptr;
}

template <typename DerivedCCG, typename FuncTy, typename CallTy>
void CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::ContextNode::
    eraseCalleeEdge(const ContextEdge *Edge) {
  auto EI =
      std::find_if(CalleeEdges.begin(), CalleeEdges.end(),
                   [Edge](const std::shared_ptr<ContextEdge> &CalleeEdge) {
                     return CalleeEdge.get() == Edge;
                   });
  assert(EI != CalleeEdges.end());
  CalleeEdges.erase(EI);
}

template <typename DerivedCCG, typename FuncTy, typename CallTy>
void CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::ContextNode::
    eraseCallerEdge(const ContextEdge *Edge) {
  auto EI =
      std::find_if(CallerEdges.begin(), CallerEdges.end(),
                   [Edge](const std::shared_ptr<ContextEdge> &CallerEdge) {
                     return CallerEdge.get() == Edge;
                   });
  assert(EI != CallerEdges.end());
  CallerEdges.erase(EI);
}

template <typename DerivedCCG, typename FuncTy, typename CallTy>
uint8_t CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::computeAllocType(
    DenseSet<uint32_t> &ContextIds) {
  uint8_t BothTypes =
      (uint8_t)AllocationType::Cold | (uint8_t)AllocationType::NotCold;
  uint8_t AllocType = (uint8_t)AllocationType::None;
  for (auto Id : ContextIds) {
    AllocType |= (uint8_t)ContextIdToAllocationType[Id];
    // Bail early if alloc type reached both, no further refinement.
    if (AllocType == BothTypes)
      return AllocType;
  }
  return AllocType;
}

template <typename DerivedCCG, typename FuncTy, typename CallTy>
typename CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::ContextNode *
CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::addAllocNode(
    CallInfo Call, const FuncTy *F) {
  assert(!getNodeForAlloc(Call));
  NodeOwner.push_back(
      std::make_unique<ContextNode>(/*IsAllocation=*/true, Call));
  ContextNode *AllocNode = NodeOwner.back().get();
  AllocationCallToContextNodeMap[Call] = AllocNode;
  NodeToCallingFunc[AllocNode] = F;
  // Use LastContextId as a uniq id for MIB allocation nodes.
  AllocNode->OrigStackOrAllocId = LastContextId;
  // Alloc type should be updated as we add in the MIBs. We should assert
  // afterwards that it is not still None.
  AllocNode->AllocTypes = (uint8_t)AllocationType::None;

  return AllocNode;
}

template <typename DerivedCCG, typename FuncTy, typename CallTy>
template <class NodeT, class IteratorT>
void CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::addStackNodesForMIB(
    ContextNode *AllocNode, CallStack<NodeT, IteratorT> &StackContext,
    CallStack<NodeT, IteratorT> &CallsiteContext, AllocationType AllocType) {
  ContextIdToAllocationType[++LastContextId] = AllocType;

  // Update alloc type and context ids for this MIB.
  AllocNode->AllocTypes |= (uint8_t)AllocType;
  AllocNode->ContextIds.insert(LastContextId);

  // Now add or update nodes for each stack id in alloc's context.
  // Later when processing the stack ids on non-alloc callsites we will adjust
  // for any inlining in the context.
  ContextNode *PrevNode = AllocNode;
  // Look for recursion (direct recursion should have been collapsed by
  // module summary analysis, here we should just be detecting mutual
  // recursion). Mark these nodes so we don't try to clone.
  SmallSet<uint64_t, 8> StackIdSet;
  // Skip any on the allocation call (inlining).
  for (auto ContextIter = StackContext.beginAfterSharedPrefix(CallsiteContext);
       ContextIter != StackContext.end(); ++ContextIter) {
    auto StackId = getStackId(*ContextIter);
    ContextNode *StackNode = getNodeForStackId(StackId);
    if (!StackNode) {
      NodeOwner.push_back(
          std::make_unique<ContextNode>(/*IsAllocation=*/false));
      StackNode = NodeOwner.back().get();
      StackEntryIdToContextNodeMap[StackId] = StackNode;
      StackNode->OrigStackOrAllocId = StackId;
    }
    auto Ins = StackIdSet.insert(StackId);
    if (!Ins.second)
      StackNode->Recursive = true;
    StackNode->ContextIds.insert(LastContextId);
    StackNode->AllocTypes |= (uint8_t)AllocType;
    PrevNode->addOrUpdateCallerEdge(StackNode, AllocType, LastContextId);
    PrevNode = StackNode;
  }
}

template <typename DerivedCCG, typename FuncTy, typename CallTy>
DenseSet<uint32_t>
CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::duplicateContextIds(
    const DenseSet<uint32_t> &StackSequenceContextIds,
    DenseMap<uint32_t, DenseSet<uint32_t>> &OldToNewContextIds) {
  DenseSet<uint32_t> NewContextIds;
  for (auto OldId : StackSequenceContextIds) {
    NewContextIds.insert(++LastContextId);
    OldToNewContextIds[OldId].insert(LastContextId);
    assert(ContextIdToAllocationType.count(OldId));
    // The new context has the same allocation type as original.
    ContextIdToAllocationType[LastContextId] = ContextIdToAllocationType[OldId];
  }
  return NewContextIds;
}

template <typename DerivedCCG, typename FuncTy, typename CallTy>
void CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::
    propagateDuplicateContextIds(
        const DenseMap<uint32_t, DenseSet<uint32_t>> &OldToNewContextIds) {
  // Build a set of duplicated context ids corresponding to the input id set.
  auto GetNewIds = [&OldToNewContextIds](const DenseSet<uint32_t> &ContextIds) {
    DenseSet<uint32_t> NewIds;
    for (auto Id : ContextIds)
      if (auto NewId = OldToNewContextIds.find(Id);
          NewId != OldToNewContextIds.end())
        NewIds.insert(NewId->second.begin(), NewId->second.end());
    return NewIds;
  };

  // Recursively update context ids sets along caller edges.
  auto UpdateCallers = [&](ContextNode *Node,
                           DenseSet<const ContextEdge *> &Visited,
                           auto &&UpdateCallers) -> void {
    for (const auto &Edge : Node->CallerEdges) {
      auto Inserted = Visited.insert(Edge.get());
      if (!Inserted.second)
        continue;
      ContextNode *NextNode = Edge->Caller;
      DenseSet<uint32_t> NewIdsToAdd = GetNewIds(Edge->getContextIds());
      // Only need to recursively iterate to NextNode via this caller edge if
      // it resulted in any added ids to NextNode.
      if (!NewIdsToAdd.empty()) {
        Edge->getContextIds().insert(NewIdsToAdd.begin(), NewIdsToAdd.end());
        NextNode->ContextIds.insert(NewIdsToAdd.begin(), NewIdsToAdd.end());
        UpdateCallers(NextNode, Visited, UpdateCallers);
      }
    }
  };

  DenseSet<const ContextEdge *> Visited;
  for (auto &Entry : AllocationCallToContextNodeMap) {
    auto *Node = Entry.second;
    // Update ids on the allocation nodes before calling the recursive
    // update along caller edges, since this simplifies the logic during
    // that traversal.
    DenseSet<uint32_t> NewIdsToAdd = GetNewIds(Node->ContextIds);
    Node->ContextIds.insert(NewIdsToAdd.begin(), NewIdsToAdd.end());
    UpdateCallers(Node, Visited, UpdateCallers);
  }
}

template <typename DerivedCCG, typename FuncTy, typename CallTy>
void CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::connectNewNode(
    ContextNode *NewNode, ContextNode *OrigNode, bool TowardsCallee) {
  // Make a copy of the context ids, since this will be adjusted below as they
  // are moved.
  DenseSet<uint32_t> RemainingContextIds = NewNode->ContextIds;
  auto &OrigEdges =
      TowardsCallee ? OrigNode->CalleeEdges : OrigNode->CallerEdges;
  // Increment iterator in loop so that we can remove edges as needed.
  for (auto EI = OrigEdges.begin(); EI != OrigEdges.end();) {
    auto Edge = *EI;
    // Remove any matching context ids from Edge, return set that were found and
    // removed, these are the new edge's context ids. Also update the remaining
    // (not found ids).
    DenseSet<uint32_t> NewEdgeContextIds, NotFoundContextIds;
    set_subtract(Edge->getContextIds(), RemainingContextIds, NewEdgeContextIds,
                 NotFoundContextIds);
    RemainingContextIds.swap(NotFoundContextIds);
    // If no matching context ids for this edge, skip it.
    if (NewEdgeContextIds.empty()) {
      ++EI;
      continue;
    }
    if (TowardsCallee) {
      auto NewEdge = std::make_shared<ContextEdge>(
          Edge->Callee, NewNode, computeAllocType(NewEdgeContextIds),
          NewEdgeContextIds);
      NewNode->CalleeEdges.push_back(NewEdge);
      NewEdge->Callee->CallerEdges.push_back(NewEdge);
    } else {
      auto NewEdge = std::make_shared<ContextEdge>(
          NewNode, Edge->Caller, computeAllocType(NewEdgeContextIds),
          NewEdgeContextIds);
      NewNode->CallerEdges.push_back(NewEdge);
      NewEdge->Caller->CalleeEdges.push_back(NewEdge);
    }
    // Remove old edge if context ids empty.
    if (Edge->getContextIds().empty()) {
      if (TowardsCallee) {
        Edge->Callee->eraseCallerEdge(Edge.get());
        EI = OrigNode->CalleeEdges.erase(EI);
      } else {
        Edge->Caller->eraseCalleeEdge(Edge.get());
        EI = OrigNode->CallerEdges.erase(EI);
      }
      continue;
    }
    ++EI;
  }
}

template <typename DerivedCCG, typename FuncTy, typename CallTy>
void CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::
    assignStackNodesPostOrder(ContextNode *Node,
                              DenseSet<const ContextNode *> &Visited,
                              DenseMap<uint64_t, std::vector<CallContextInfo>>
                                  &StackIdToMatchingCalls) {
  auto Inserted = Visited.insert(Node);
  if (!Inserted.second)
    return;
  // Post order traversal. Iterate over a copy since we may add nodes and
  // therefore new callers during the recursive call, invalidating any
  // iterator over the original edge vector. We don't need to process these
  // new nodes as they were already processed on creation.
  auto CallerEdges = Node->CallerEdges;
  for (auto &Edge : CallerEdges) {
    // Skip any that have been removed during the recursion.
    if (!Edge)
      continue;
    assignStackNodesPostOrder(Edge->Caller, Visited, StackIdToMatchingCalls);
  }

  // If this node's stack id is in the map, update the graph to contain new
  // nodes representing any inlining at interior callsites. Note we move the
  // associated context ids over to the new nodes.

  // Ignore this node if it is for an allocation or we didn't record any
  // stack id lists ending at it.
  if (Node->IsAllocation ||
      !StackIdToMatchingCalls.count(Node->OrigStackOrAllocId))
    return;

  auto &Calls = StackIdToMatchingCalls[Node->OrigStackOrAllocId];
  // Handle the simple case first. A single call with a single stack id.
  // In this case there is no need to create any new context nodes, simply
  // assign the context node for stack id to this Call.
  if (Calls.size() == 1) {
    auto &[Call, Ids, Func, SavedContextIds] = Calls[0];
    if (Ids.size() == 1) {
      assert(SavedContextIds.empty());
      // It should be this Node
      assert(Node == getNodeForStackId(Ids[0]));
      if (Node->Recursive)
        return;
      Node->setCall(Call);
      NonAllocationCallToContextNodeMap[Call] = Node;
      NodeToCallingFunc[Node] = Func;
      return;
    }
  }

  // Find the node for the last stack id, which should be the same
  // across all calls recorded for this id, and is this node's id.
  uint64_t LastId = Node->OrigStackOrAllocId;
  ContextNode *LastNode = getNodeForStackId(LastId);
  // We should only have kept stack ids that had nodes.
  assert(LastNode);

  for (unsigned I = 0; I < Calls.size(); I++) {
    auto &[Call, Ids, Func, SavedContextIds] = Calls[I];
    // Skip any for which we didn't assign any ids, these don't get a node in
    // the graph.
    if (SavedContextIds.empty())
      continue;

    assert(LastId == Ids.back());

    ContextNode *FirstNode = getNodeForStackId(Ids[0]);
    assert(FirstNode);

    // Recompute the context ids for this stack id sequence (the
    // intersection of the context ids of the corresponding nodes).
    // Start with the ids we saved in the map for this call, which could be
    // duplicated context ids. We have to recompute as we might have overlap
    // overlap between the saved context ids for different last nodes, and
    // removed them already during the post order traversal.
    set_intersect(SavedContextIds, FirstNode->ContextIds);
    ContextNode *PrevNode = nullptr;
    for (auto Id : Ids) {
      ContextNode *CurNode = getNodeForStackId(Id);
      // We should only have kept stack ids that had nodes and weren't
      // recursive.
      assert(CurNode);
      assert(!CurNode->Recursive);
      if (!PrevNode) {
        PrevNode = CurNode;
        continue;
      }
      auto *Edge = CurNode->findEdgeFromCallee(PrevNode);
      if (!Edge) {
        SavedContextIds.clear();
        break;
      }
      PrevNode = CurNode;
      set_intersect(SavedContextIds, Edge->getContextIds());

      // If we now have no context ids for clone, skip this call.
      if (SavedContextIds.empty())
        break;
    }
    if (SavedContextIds.empty())
      continue;

    // Create new context node.
    NodeOwner.push_back(
        std::make_unique<ContextNode>(/*IsAllocation=*/false, Call));
    ContextNode *NewNode = NodeOwner.back().get();
    NodeToCallingFunc[NewNode] = Func;
    NonAllocationCallToContextNodeMap[Call] = NewNode;
    NewNode->ContextIds = SavedContextIds;
    NewNode->AllocTypes = computeAllocType(NewNode->ContextIds);

    // Connect to callees of innermost stack frame in inlined call chain.
    // This updates context ids for FirstNode's callee's to reflect those
    // moved to NewNode.
    connectNewNode(NewNode, FirstNode, /*TowardsCallee=*/true);

    // Connect to callers of outermost stack frame in inlined call chain.
    // This updates context ids for FirstNode's caller's to reflect those
    // moved to NewNode.
    connectNewNode(NewNode, LastNode, /*TowardsCallee=*/false);

    // Now we need to remove context ids from edges/nodes between First and
    // Last Node.
    PrevNode = nullptr;
    for (auto Id : Ids) {
      ContextNode *CurNode = getNodeForStackId(Id);
      // We should only have kept stack ids that had nodes.
      assert(CurNode);

      // Remove the context ids moved to NewNode from CurNode, and the
      // edge from the prior node.
      set_subtract(CurNode->ContextIds, NewNode->ContextIds);
      if (PrevNode) {
        auto *PrevEdge = CurNode->findEdgeFromCallee(PrevNode);
        assert(PrevEdge);
        set_subtract(PrevEdge->getContextIds(), NewNode->ContextIds);
        if (PrevEdge->getContextIds().empty()) {
          PrevNode->eraseCallerEdge(PrevEdge);
          CurNode->eraseCalleeEdge(PrevEdge);
        }
      }
      PrevNode = CurNode;
    }
  }
}

template <typename DerivedCCG, typename FuncTy, typename CallTy>
void CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::updateStackNodes() {
  // Map of stack id to all calls with that as the last (outermost caller)
  // callsite id that has a context node (some might not due to pruning
  // performed during matching of the allocation profile contexts).
  // The CallContextInfo contains the Call and a list of its stack ids with
  // ContextNodes, the function containing Call, and the set of context ids
  // the analysis will eventually identify for use in any new node created
  // for that callsite.
  DenseMap<uint64_t, std::vector<CallContextInfo>> StackIdToMatchingCalls;
  for (auto &[Func, CallsWithMetadata] : FuncToCallsWithMetadata) {
    for (auto &Call : CallsWithMetadata) {
      // Ignore allocations, already handled.
      if (AllocationCallToContextNodeMap.count(Call))
        continue;
      auto StackIdsWithContextNodes =
          getStackIdsWithContextNodesForCall(Call.call());
      // If there were no nodes created for MIBs on allocs (maybe this was in
      // the unambiguous part of the MIB stack that was pruned), ignore.
      if (StackIdsWithContextNodes.empty())
        continue;
      // Otherwise, record this Call along with the list of ids for the last
      // (outermost caller) stack id with a node.
      StackIdToMatchingCalls[StackIdsWithContextNodes.back()].push_back(
          {Call.call(), StackIdsWithContextNodes, Func, {}});
    }
  }

  // First make a pass through all stack ids that correspond to a call,
  // as identified in the above loop. Compute the context ids corresponding to
  // each of these calls when they correspond to multiple stack ids due to
  // due to inlining. Perform any duplication of context ids required when
  // there is more than one call with the same stack ids. Their (possibly newly
  // duplicated) context ids are saved in the StackIdToMatchingCalls map.
  DenseMap<uint32_t, DenseSet<uint32_t>> OldToNewContextIds;
  for (auto &It : StackIdToMatchingCalls) {
    auto &Calls = It.getSecond();
    // Skip single calls with a single stack id. These don't need a new node.
    if (Calls.size() == 1) {
      auto &Ids = std::get<1>(Calls[0]);
      if (Ids.size() == 1)
        continue;
    }
    // In order to do the best and maximal matching of inlined calls to context
    // node sequences we will sort the vectors of stack ids in descending order
    // of length, and within each length, lexicographically by stack id. The
    // latter is so that we can specially handle calls that have identical stack
    // id sequences (either due to cloning or artificially because of the MIB
    // context pruning).
    std::stable_sort(Calls.begin(), Calls.end(),
                     [](const CallContextInfo &A, const CallContextInfo &B) {
                       auto &IdsA = std::get<1>(A);
                       auto &IdsB = std::get<1>(B);
                       return IdsA.size() > IdsB.size() ||
                              (IdsA.size() == IdsB.size() && IdsA < IdsB);
                     });

    // Find the node for the last stack id, which should be the same
    // across all calls recorded for this id, and is the id for this
    // entry in the StackIdToMatchingCalls map.
    uint64_t LastId = It.getFirst();
    ContextNode *LastNode = getNodeForStackId(LastId);
    // We should only have kept stack ids that had nodes.
    assert(LastNode);

    if (LastNode->Recursive)
      continue;

    // Initialize the context ids with the last node's. We will subsequently
    // refine the context ids by computing the intersection along all edges.
    DenseSet<uint32_t> LastNodeContextIds = LastNode->ContextIds;
    assert(!LastNodeContextIds.empty());

    for (unsigned I = 0; I < Calls.size(); I++) {
      auto &[Call, Ids, Func, SavedContextIds] = Calls[I];
      assert(SavedContextIds.empty());
      assert(LastId == Ids.back());

      // First compute the context ids for this stack id sequence (the
      // intersection of the context ids of the corresponding nodes).
      // Start with the remaining saved ids for the last node.
      assert(!LastNodeContextIds.empty());
      DenseSet<uint32_t> StackSequenceContextIds = LastNodeContextIds;

      ContextNode *PrevNode = LastNode;
      ContextNode *CurNode = LastNode;
      bool Skip = false;

      // Iterate backwards through the stack Ids, starting after the last Id
      // in the list, which was handled once outside for all Calls.
      for (auto IdIter = Ids.rbegin() + 1; IdIter != Ids.rend(); IdIter++) {
        auto Id = *IdIter;
        CurNode = getNodeForStackId(Id);
        // We should only have kept stack ids that had nodes.
        assert(CurNode);

        if (CurNode->Recursive) {
          Skip = true;
          break;
        }

        auto *Edge = CurNode->findEdgeFromCaller(PrevNode);
        // If there is no edge then the nodes belong to different MIB contexts,
        // and we should skip this inlined context sequence. For example, this
        // particular inlined context may include stack ids A->B, and we may
        // indeed have nodes for both A and B, but it is possible that they were
        // never profiled in sequence in a single MIB for any allocation (i.e.
        // we might have profiled an allocation that involves the callsite A,
        // but through a different one of its callee callsites, and we might
        // have profiled an allocation that involves callsite B, but reached
        // from a different caller callsite).
        if (!Edge) {
          Skip = true;
          break;
        }
        PrevNode = CurNode;

        // Update the context ids, which is the intersection of the ids along
        // all edges in the sequence.
        set_intersect(StackSequenceContextIds, Edge->getContextIds());

        // If we now have no context ids for clone, skip this call.
        if (StackSequenceContextIds.empty()) {
          Skip = true;
          break;
        }
      }
      if (Skip)
        continue;

      // If some of this call's stack ids did not have corresponding nodes (due
      // to pruning), don't include any context ids for contexts that extend
      // beyond these nodes. Otherwise we would be matching part of unrelated /
      // not fully matching stack contexts. To do this, subtract any context ids
      // found in caller nodes of the last node found above.
      if (Ids.back() != getLastStackId(Call)) {
        for (const auto &PE : LastNode->CallerEdges) {
          set_subtract(StackSequenceContextIds, PE->getContextIds());
          if (StackSequenceContextIds.empty())
            break;
        }
        // If we now have no context ids for clone, skip this call.
        if (StackSequenceContextIds.empty())
          continue;
      }

      // Check if the next set of stack ids is the same (since the Calls vector
      // of tuples is sorted by the stack ids we can just look at the next one).
      bool DuplicateContextIds = false;
      if (I + 1 < Calls.size()) {
        auto NextIds = std::get<1>(Calls[I + 1]);
        DuplicateContextIds = Ids == NextIds;
      }

      // If we don't have duplicate context ids, then we can assign all the
      // context ids computed for the original node sequence to this call.
      // If there are duplicate calls with the same stack ids then we synthesize
      // new context ids that are duplicates of the originals. These are
      // assigned to SavedContextIds, which is a reference into the map entry
      // for this call, allowing us to access these ids later on.
      OldToNewContextIds.reserve(OldToNewContextIds.size() +
                                 StackSequenceContextIds.size());
      SavedContextIds =
          DuplicateContextIds
              ? duplicateContextIds(StackSequenceContextIds, OldToNewContextIds)
              : StackSequenceContextIds;
      assert(!SavedContextIds.empty());

      if (!DuplicateContextIds) {
        // Update saved last node's context ids to remove those that are
        // assigned to other calls, so that it is ready for the next call at
        // this stack id.
        set_subtract(LastNodeContextIds, StackSequenceContextIds);
        if (LastNodeContextIds.empty())
          break;
      }
    }
  }

  // Propagate the duplicate context ids over the graph.
  propagateDuplicateContextIds(OldToNewContextIds);

  if (VerifyCCG)
    check();

  // Now perform a post-order traversal over the graph, starting with the
  // allocation nodes, essentially processing nodes from callers to callees.
  // For any that contains an id in the map, update the graph to contain new
  // nodes representing any inlining at interior callsites. Note we move the
  // associated context ids over to the new nodes.
  DenseSet<const ContextNode *> Visited;
  for (auto &Entry : AllocationCallToContextNodeMap)
    assignStackNodesPostOrder(Entry.second, Visited, StackIdToMatchingCalls);
}

uint64_t ModuleCallsiteContextGraph::getLastStackId(Instruction *Call) {
  CallStack<MDNode, MDNode::op_iterator> CallsiteContext(
      Call->getMetadata(LLVMContext::MD_callsite));
  return CallsiteContext.back();
}

uint64_t IndexCallsiteContextGraph::getLastStackId(IndexCall &Call) {
  assert(Call.is<CallsiteInfo *>());
  CallStack<CallsiteInfo, SmallVector<unsigned>::const_iterator>
      CallsiteContext(Call.dyn_cast<CallsiteInfo *>());
  // Need to convert index into stack id.
  return Index.getStackIdAtIndex(CallsiteContext.back());
}

static std::string getMemProfFuncName(Twine Base, unsigned CloneNo) {
  if (!CloneNo)
    return Base.str();
  return (Base + ".memprof." + Twine(CloneNo)).str();
}

std::string ModuleCallsiteContextGraph::getLabel(const Function *Func,
                                                 const Instruction *Call,
                                                 unsigned CloneNo) const {
  return (Twine(Call->getFunction()->getName()) + " -> " +
          cast<CallBase>(Call)->getCalledFunction()->getName())
      .str();
}

std::string IndexCallsiteContextGraph::getLabel(const FunctionSummary *Func,
                                                const IndexCall &Call,
                                                unsigned CloneNo) const {
  auto VI = FSToVIMap.find(Func);
  assert(VI != FSToVIMap.end());
  if (Call.is<AllocInfo *>())
    return (VI->second.name() + " -> alloc").str();
  else {
    auto *Callsite = Call.dyn_cast<CallsiteInfo *>();
    return (VI->second.name() + " -> " +
            getMemProfFuncName(Callsite->Callee.name(),
                               Callsite->Clones[CloneNo]))
        .str();
  }
}

std::vector<uint64_t>
ModuleCallsiteContextGraph::getStackIdsWithContextNodesForCall(
    Instruction *Call) {
  CallStack<MDNode, MDNode::op_iterator> CallsiteContext(
      Call->getMetadata(LLVMContext::MD_callsite));
  return getStackIdsWithContextNodes<MDNode, MDNode::op_iterator>(
      CallsiteContext);
}

std::vector<uint64_t>
IndexCallsiteContextGraph::getStackIdsWithContextNodesForCall(IndexCall &Call) {
  assert(Call.is<CallsiteInfo *>());
  CallStack<CallsiteInfo, SmallVector<unsigned>::const_iterator>
      CallsiteContext(Call.dyn_cast<CallsiteInfo *>());
  return getStackIdsWithContextNodes<CallsiteInfo,
                                     SmallVector<unsigned>::const_iterator>(
      CallsiteContext);
}

template <typename DerivedCCG, typename FuncTy, typename CallTy>
template <class NodeT, class IteratorT>
std::vector<uint64_t>
CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::getStackIdsWithContextNodes(
    CallStack<NodeT, IteratorT> &CallsiteContext) {
  std::vector<uint64_t> StackIds;
  for (auto IdOrIndex : CallsiteContext) {
    auto StackId = getStackId(IdOrIndex);
    ContextNode *Node = getNodeForStackId(StackId);
    if (!Node)
      break;
    StackIds.push_back(StackId);
  }
  return StackIds;
}

ModuleCallsiteContextGraph::ModuleCallsiteContextGraph(Module &M) : Mod(M) {
  for (auto &F : M) {
    std::vector<CallInfo> CallsWithMetadata;
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (!isa<CallBase>(I))
          continue;
        if (auto *MemProfMD = I.getMetadata(LLVMContext::MD_memprof)) {
          CallsWithMetadata.push_back(&I);
          auto *AllocNode = addAllocNode(&I, &F);
          auto *CallsiteMD = I.getMetadata(LLVMContext::MD_callsite);
          assert(CallsiteMD);
          CallStack<MDNode, MDNode::op_iterator> CallsiteContext(CallsiteMD);
          // Add all of the MIBs and their stack nodes.
          for (auto &MDOp : MemProfMD->operands()) {
            auto *MIBMD = cast<const MDNode>(MDOp);
            MDNode *StackNode = getMIBStackNode(MIBMD);
            assert(StackNode);
            CallStack<MDNode, MDNode::op_iterator> StackContext(StackNode);
            addStackNodesForMIB<MDNode, MDNode::op_iterator>(
                AllocNode, StackContext, CallsiteContext,
                getMIBAllocType(MIBMD));
          }
          assert(AllocNode->AllocTypes != (uint8_t)AllocationType::None);
          // Memprof and callsite metadata on memory allocations no longer
          // needed.
          I.setMetadata(LLVMContext::MD_memprof, nullptr);
          I.setMetadata(LLVMContext::MD_callsite, nullptr);
        }
        // For callsite metadata, add to list for this function for later use.
        else if (I.getMetadata(LLVMContext::MD_callsite))
          CallsWithMetadata.push_back(&I);
      }
    }
    if (!CallsWithMetadata.empty())
      FuncToCallsWithMetadata.push_back({&F, CallsWithMetadata});
  }

  if (DumpCCG) {
    dbgs() << "CCG before updating call stack chains:\n";
    dbgs() << *this;
  }

  if (ExportToDot)
    exportToDot("prestackupdate");

  updateStackNodes();

  handleCallsitesWithMultipleTargets();

  // Strip off remaining callsite metadata, no longer needed.
  for (auto &FuncEntry : FuncToCallsWithMetadata)
    for (auto &Call : FuncEntry.second)
      Call.call()->setMetadata(LLVMContext::MD_callsite, nullptr);
}

IndexCallsiteContextGraph::IndexCallsiteContextGraph(
    ModuleSummaryIndex &Index,
    function_ref<bool(GlobalValue::GUID, const GlobalValueSummary *)>
        isPrevailing)
    : Index(Index) {
  for (auto &I : Index) {
    auto VI = Index.getValueInfo(I);
    for (auto &S : VI.getSummaryList()) {
      // We should only add the prevailing nodes. Otherwise we may try to clone
      // in a weak copy that won't be linked (and may be different than the
      // prevailing version).
      // We only keep the memprof summary on the prevailing copy now when
      // building the combined index, as a space optimization, however don't
      // rely on this optimization. The linker doesn't resolve local linkage
      // values so don't check whether those are prevailing.
      if (!GlobalValue::isLocalLinkage(S->linkage()) &&
          !isPrevailing(VI.getGUID(), S.get()))
        continue;
      auto *FS = dyn_cast<FunctionSummary>(S.get());
      if (!FS)
        continue;
      std::vector<CallInfo> CallsWithMetadata;
      if (!FS->allocs().empty()) {
        for (auto &AN : FS->mutableAllocs()) {
          // This can happen because of recursion elimination handling that
          // currently exists in ModuleSummaryAnalysis. Skip these for now.
          // We still added them to the summary because we need to be able to
          // correlate properly in applyImport in the backends.
          if (AN.MIBs.empty())
            continue;
          CallsWithMetadata.push_back({&AN});
          auto *AllocNode = addAllocNode({&AN}, FS);
          // Pass an empty CallStack to the CallsiteContext (second)
          // parameter, since for ThinLTO we already collapsed out the inlined
          // stack ids on the allocation call during ModuleSummaryAnalysis.
          CallStack<MIBInfo, SmallVector<unsigned>::const_iterator>
              EmptyContext;
          // Now add all of the MIBs and their stack nodes.
          for (auto &MIB : AN.MIBs) {
            CallStack<MIBInfo, SmallVector<unsigned>::const_iterator>
                StackContext(&MIB);
            addStackNodesForMIB<MIBInfo, SmallVector<unsigned>::const_iterator>(
                AllocNode, StackContext, EmptyContext, MIB.AllocType);
          }
          assert(AllocNode->AllocTypes != (uint8_t)AllocationType::None);
          // Initialize version 0 on the summary alloc node to the current alloc
          // type, unless it has both types in which case make it default, so
          // that in the case where we aren't able to clone the original version
          // always ends up with the default allocation behavior.
          AN.Versions[0] = (uint8_t)allocTypeToUse(AllocNode->AllocTypes);
        }
      }
      // For callsite metadata, add to list for this function for later use.
      if (!FS->callsites().empty())
        for (auto &SN : FS->mutableCallsites())
          CallsWithMetadata.push_back({&SN});

      if (!CallsWithMetadata.empty())
        FuncToCallsWithMetadata.push_back({FS, CallsWithMetadata});

      if (!FS->allocs().empty() || !FS->callsites().empty())
        FSToVIMap[FS] = VI;
    }
  }

  if (DumpCCG) {
    dbgs() << "CCG before updating call stack chains:\n";
    dbgs() << *this;
  }

  if (ExportToDot)
    exportToDot("prestackupdate");

  updateStackNodes();

  handleCallsitesWithMultipleTargets();
}

template <typename DerivedCCG, typename FuncTy, typename CallTy>
void CallsiteContextGraph<DerivedCCG, FuncTy,
                          CallTy>::handleCallsitesWithMultipleTargets() {
  // Look for and workaround callsites that call multiple functions.
  // This can happen for indirect calls, which needs better handling, and in
  // more rare cases (e.g. macro expansion).
  // TODO: To fix this for indirect calls we will want to perform speculative
  // devirtualization using either the normal PGO info with ICP, or using the
  // information in the profiled MemProf contexts. We can do this prior to
  // this transformation for regular LTO, and for ThinLTO we can simulate that
  // effect in the summary and perform the actual speculative devirtualization
  // while cloning in the ThinLTO backend.
  for (auto Entry = NonAllocationCallToContextNodeMap.begin();
       Entry != NonAllocationCallToContextNodeMap.end();) {
    auto *Node = Entry->second;
    assert(Node->Clones.empty());
    // Check all node callees and see if in the same function.
    bool Removed = false;
    auto Call = Node->Call.call();
    for (auto &Edge : Node->CalleeEdges) {
      if (!Edge->Callee->hasCall())
        continue;
      assert(NodeToCallingFunc.count(Edge->Callee));
      // Check if the called function matches that of the callee node.
      if (calleeMatchesFunc(Call, NodeToCallingFunc[Edge->Callee]))
        continue;
      // Work around by setting Node to have a null call, so it gets
      // skipped during cloning. Otherwise assignFunctions will assert
      // because its data structures are not designed to handle this case.
      Entry = NonAllocationCallToContextNodeMap.erase(Entry);
      Node->setCall(CallInfo());
      Removed = true;
      break;
    }
    if (!Removed)
      Entry++;
  }
}

uint64_t ModuleCallsiteContextGraph::getStackId(uint64_t IdOrIndex) const {
  // In the Module (IR) case this is already the Id.
  return IdOrIndex;
}

uint64_t IndexCallsiteContextGraph::getStackId(uint64_t IdOrIndex) const {
  // In the Index case this is an index into the stack id list in the summary
  // index, convert it to an Id.
  return Index.getStackIdAtIndex(IdOrIndex);
}

bool ModuleCallsiteContextGraph::calleeMatchesFunc(Instruction *Call,
                                                   const Function *Func) {
  auto *CB = dyn_cast<CallBase>(Call);
  if (!CB->getCalledOperand())
    return false;
  auto *CalleeVal = CB->getCalledOperand()->stripPointerCasts();
  auto *CalleeFunc = dyn_cast<Function>(CalleeVal);
  if (CalleeFunc == Func)
    return true;
  auto *Alias = dyn_cast<GlobalAlias>(CalleeVal);
  return Alias && Alias->getAliasee() == Func;
}

bool IndexCallsiteContextGraph::calleeMatchesFunc(IndexCall &Call,
                                                  const FunctionSummary *Func) {
  ValueInfo Callee = Call.dyn_cast<CallsiteInfo *>()->Callee;
  // If there is no summary list then this is a call to an externally defined
  // symbol.
  AliasSummary *Alias =
      Callee.getSummaryList().empty()
          ? nullptr
          : dyn_cast<AliasSummary>(Callee.getSummaryList()[0].get());
  assert(FSToVIMap.count(Func));
  return Callee == FSToVIMap[Func] ||
         // If callee is an alias, check the aliasee, since only function
         // summary base objects will contain the stack node summaries and thus
         // get a context node.
         (Alias && Alias->getAliaseeVI() == FSToVIMap[Func]);
}

static std::string getAllocTypeString(uint8_t AllocTypes) {
  if (!AllocTypes)
    return "None";
  std::string Str;
  if (AllocTypes & (uint8_t)AllocationType::NotCold)
    Str += "NotCold";
  if (AllocTypes & (uint8_t)AllocationType::Cold)
    Str += "Cold";
  return Str;
}

template <typename DerivedCCG, typename FuncTy, typename CallTy>
void CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::ContextNode::dump()
    const {
  print(dbgs());
  dbgs() << "\n";
}

template <typename DerivedCCG, typename FuncTy, typename CallTy>
void CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::ContextNode::print(
    raw_ostream &OS) const {
  OS << "Node " << this << "\n";
  OS << "\t";
  printCall(OS);
  if (Recursive)
    OS << " (recursive)";
  OS << "\n";
  OS << "\tAllocTypes: " << getAllocTypeString(AllocTypes) << "\n";
  OS << "\tContextIds:";
  std::vector<uint32_t> SortedIds(ContextIds.begin(), ContextIds.end());
  std::sort(SortedIds.begin(), SortedIds.end());
  for (auto Id : SortedIds)
    OS << " " << Id;
  OS << "\n";
  OS << "\tCalleeEdges:\n";
  for (auto &Edge : CalleeEdges)
    OS << "\t\t" << *Edge << "\n";
  OS << "\tCallerEdges:\n";
  for (auto &Edge : CallerEdges)
    OS << "\t\t" << *Edge << "\n";
  if (!Clones.empty()) {
    OS << "\tClones: ";
    FieldSeparator FS;
    for (auto *Clone : Clones)
      OS << FS << Clone;
    OS << "\n";
  } else if (CloneOf) {
    OS << "\tClone of " << CloneOf << "\n";
  }
}

template <typename DerivedCCG, typename FuncTy, typename CallTy>
void CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::ContextEdge::dump()
    const {
  print(dbgs());
  dbgs() << "\n";
}

template <typename DerivedCCG, typename FuncTy, typename CallTy>
void CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::ContextEdge::print(
    raw_ostream &OS) const {
  OS << "Edge from Callee " << Callee << " to Caller: " << Caller
     << " AllocTypes: " << getAllocTypeString(AllocTypes);
  OS << " ContextIds:";
  std::vector<uint32_t> SortedIds(ContextIds.begin(), ContextIds.end());
  std::sort(SortedIds.begin(), SortedIds.end());
  for (auto Id : SortedIds)
    OS << " " << Id;
}

template <typename DerivedCCG, typename FuncTy, typename CallTy>
void CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::dump() const {
  print(dbgs());
}

template <typename DerivedCCG, typename FuncTy, typename CallTy>
void CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::print(
    raw_ostream &OS) const {
  OS << "Callsite Context Graph:\n";
  using GraphType = const CallsiteContextGraph<DerivedCCG, FuncTy, CallTy> *;
  for (const auto Node : nodes<GraphType>(this)) {
    if (Node->isRemoved())
      continue;
    Node->print(OS);
    OS << "\n";
  }
}

template <typename DerivedCCG, typename FuncTy, typename CallTy>
static void checkEdge(
    const std::shared_ptr<ContextEdge<DerivedCCG, FuncTy, CallTy>> &Edge) {
  // Confirm that alloc type is not None and that we have at least one context
  // id.
  assert(Edge->AllocTypes != (uint8_t)AllocationType::None);
  assert(!Edge->ContextIds.empty());
}

template <typename DerivedCCG, typename FuncTy, typename CallTy>
static void checkNode(const ContextNode<DerivedCCG, FuncTy, CallTy> *Node) {
  if (Node->isRemoved())
    return;
  // Node's context ids should be the union of both its callee and caller edge
  // context ids.
  if (Node->CallerEdges.size()) {
    auto EI = Node->CallerEdges.begin();
    auto &FirstEdge = *EI;
    EI++;
    DenseSet<uint32_t> CallerEdgeContextIds(FirstEdge->ContextIds);
    for (; EI != Node->CallerEdges.end(); EI++) {
      const auto &Edge = *EI;
      set_union(CallerEdgeContextIds, Edge->ContextIds);
    }
    // Node can have more context ids than callers if some contexts terminate at
    // node and some are longer.
    assert(Node->ContextIds == CallerEdgeContextIds ||
           set_is_subset(CallerEdgeContextIds, Node->ContextIds));
  }
  if (Node->CalleeEdges.size()) {
    auto EI = Node->CalleeEdges.begin();
    auto &FirstEdge = *EI;
    EI++;
    DenseSet<uint32_t> CalleeEdgeContextIds(FirstEdge->ContextIds);
    for (; EI != Node->CalleeEdges.end(); EI++) {
      const auto &Edge = *EI;
      set_union(CalleeEdgeContextIds, Edge->ContextIds);
    }
    assert(Node->ContextIds == CalleeEdgeContextIds);
  }
}

template <typename DerivedCCG, typename FuncTy, typename CallTy>
void CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::check() const {
  using GraphType = const CallsiteContextGraph<DerivedCCG, FuncTy, CallTy> *;
  for (const auto Node : nodes<GraphType>(this)) {
    checkNode<DerivedCCG, FuncTy, CallTy>(Node);
    for (auto &Edge : Node->CallerEdges)
      checkEdge<DerivedCCG, FuncTy, CallTy>(Edge);
  }
}

template <typename DerivedCCG, typename FuncTy, typename CallTy>
struct GraphTraits<const CallsiteContextGraph<DerivedCCG, FuncTy, CallTy> *> {
  using GraphType = const CallsiteContextGraph<DerivedCCG, FuncTy, CallTy> *;
  using NodeRef = const ContextNode<DerivedCCG, FuncTy, CallTy> *;

  using NodePtrTy = std::unique_ptr<ContextNode<DerivedCCG, FuncTy, CallTy>>;
  static NodeRef getNode(const NodePtrTy &P) { return P.get(); }

  using nodes_iterator =
      mapped_iterator<typename std::vector<NodePtrTy>::const_iterator,
                      decltype(&getNode)>;

  static nodes_iterator nodes_begin(GraphType G) {
    return nodes_iterator(G->NodeOwner.begin(), &getNode);
  }

  static nodes_iterator nodes_end(GraphType G) {
    return nodes_iterator(G->NodeOwner.end(), &getNode);
  }

  static NodeRef getEntryNode(GraphType G) {
    return G->NodeOwner.begin()->get();
  }

  using EdgePtrTy = std::shared_ptr<ContextEdge<DerivedCCG, FuncTy, CallTy>>;
  static const ContextNode<DerivedCCG, FuncTy, CallTy> *
  GetCallee(const EdgePtrTy &P) {
    return P->Callee;
  }

  using ChildIteratorType =
      mapped_iterator<typename std::vector<std::shared_ptr<ContextEdge<
                          DerivedCCG, FuncTy, CallTy>>>::const_iterator,
                      decltype(&GetCallee)>;

  static ChildIteratorType child_begin(NodeRef N) {
    return ChildIteratorType(N->CalleeEdges.begin(), &GetCallee);
  }

  static ChildIteratorType child_end(NodeRef N) {
    return ChildIteratorType(N->CalleeEdges.end(), &GetCallee);
  }
};

template <typename DerivedCCG, typename FuncTy, typename CallTy>
struct DOTGraphTraits<const CallsiteContextGraph<DerivedCCG, FuncTy, CallTy> *>
    : public DefaultDOTGraphTraits {
  DOTGraphTraits(bool IsSimple = false) : DefaultDOTGraphTraits(IsSimple) {}

  using GraphType = const CallsiteContextGraph<DerivedCCG, FuncTy, CallTy> *;
  using GTraits = GraphTraits<GraphType>;
  using NodeRef = typename GTraits::NodeRef;
  using ChildIteratorType = typename GTraits::ChildIteratorType;

  static std::string getNodeLabel(NodeRef Node, GraphType G) {
    std::string LabelString =
        (Twine("OrigId: ") + (Node->IsAllocation ? "Alloc" : "") +
         Twine(Node->OrigStackOrAllocId))
            .str();
    LabelString += "\n";
    if (Node->hasCall()) {
      auto Func = G->NodeToCallingFunc.find(Node);
      assert(Func != G->NodeToCallingFunc.end());
      LabelString +=
          G->getLabel(Func->second, Node->Call.call(), Node->Call.cloneNo());
    } else {
      LabelString += "null call";
      if (Node->Recursive)
        LabelString += " (recursive)";
      else
        LabelString += " (external)";
    }
    return LabelString;
  }

  static std::string getNodeAttributes(NodeRef Node, GraphType) {
    std::string AttributeString = (Twine("tooltip=\"") + getNodeId(Node) + " " +
                                   getContextIds(Node->ContextIds) + "\"")
                                      .str();
    AttributeString +=
        (Twine(",fillcolor=\"") + getColor(Node->AllocTypes) + "\"").str();
    AttributeString += ",style=\"filled\"";
    if (Node->CloneOf) {
      AttributeString += ",color=\"blue\"";
      AttributeString += ",style=\"filled,bold,dashed\"";
    } else
      AttributeString += ",style=\"filled\"";
    return AttributeString;
  }

  static std::string getEdgeAttributes(NodeRef, ChildIteratorType ChildIter,
                                       GraphType) {
    auto &Edge = *(ChildIter.getCurrent());
    return (Twine("tooltip=\"") + getContextIds(Edge->ContextIds) + "\"" +
            Twine(",fillcolor=\"") + getColor(Edge->AllocTypes) + "\"")
        .str();
  }

  // Since the NodeOwners list includes nodes that are no longer connected to
  // the graph, skip them here.
  static bool isNodeHidden(NodeRef Node, GraphType) {
    return Node->isRemoved();
  }

private:
  static std::string getContextIds(const DenseSet<uint32_t> &ContextIds) {
    std::string IdString = "ContextIds:";
    if (ContextIds.size() < 100) {
      std::vector<uint32_t> SortedIds(ContextIds.begin(), ContextIds.end());
      std::sort(SortedIds.begin(), SortedIds.end());
      for (auto Id : SortedIds)
        IdString += (" " + Twine(Id)).str();
    } else {
      IdString += (" (" + Twine(ContextIds.size()) + " ids)").str();
    }
    return IdString;
  }

  static std::string getColor(uint8_t AllocTypes) {
    if (AllocTypes == (uint8_t)AllocationType::NotCold)
      // Color "brown1" actually looks like a lighter red.
      return "brown1";
    if (AllocTypes == (uint8_t)AllocationType::Cold)
      return "cyan";
    if (AllocTypes ==
        ((uint8_t)AllocationType::NotCold | (uint8_t)AllocationType::Cold))
      // Lighter purple.
      return "mediumorchid1";
    return "gray";
  }

  static std::string getNodeId(NodeRef Node) {
    std::stringstream SStream;
    SStream << std::hex << "N0x" << (unsigned long long)Node;
    std::string Result = SStream.str();
    return Result;
  }
};

template <typename DerivedCCG, typename FuncTy, typename CallTy>
void CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::exportToDot(
    std::string Label) const {
  WriteGraph(this, "", false, Label,
             DotFilePathPrefix + "ccg." + Label + ".dot");
}

template <typename DerivedCCG, typename FuncTy, typename CallTy>
bool CallsiteContextGraph<DerivedCCG, FuncTy, CallTy>::process() {
  if (DumpCCG) {
    dbgs() << "CCG before cloning:\n";
    dbgs() << *this;
  }
  if (ExportToDot)
    exportToDot("postbuild");

  if (VerifyCCG) {
    check();
  }

  return false;
}

bool MemProfContextDisambiguation::processModule(Module &M) {
  bool Changed = false;

  ModuleCallsiteContextGraph CCG(M);
  Changed = CCG.process();

  return Changed;
}

PreservedAnalyses MemProfContextDisambiguation::run(Module &M,
                                                    ModuleAnalysisManager &AM) {
  if (!processModule(M))
    return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}

void MemProfContextDisambiguation::run(
    ModuleSummaryIndex &Index,
    function_ref<bool(GlobalValue::GUID, const GlobalValueSummary *)>
        isPrevailing) {
  IndexCallsiteContextGraph CCG(Index, isPrevailing);
  CCG.process();
}
