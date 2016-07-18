//- CFLAndersAliasAnalysis.cpp - Unification-based Alias Analysis ---*- C++-*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a CFL-based, summary-based alias analysis algorithm. It
// differs from CFLSteensAliasAnalysis in its inclusion-based nature while
// CFLSteensAliasAnalysis is unification-based. This pass has worse performance
// than CFLSteensAliasAnalysis (the worst case complexity of
// CFLAndersAliasAnalysis is cubic, while the worst case complexity of
// CFLSteensAliasAnalysis is almost linear), but it is able to yield more
// precise analysis result. The precision of this analysis is roughly the same
// as that of an one level context-sensitive Andersen's algorithm.
//
// The algorithm used here is based on recursive state machine matching scheme
// proposed in "Demand-driven alias analysis for C" by Xin Zheng and Radu
// Rugina. The general idea is to extend the tranditional transitive closure
// algorithm to perform CFL matching along the way: instead of recording
// "whether X is reachable from Y", we keep track of "whether X is reachable
// from Y at state Z", where the "state" field indicates where we are in the CFL
// matching process. To understand the matching better, it is advisable to have
// the state machine shown in Figure 3 of the paper available when reading the
// codes: all we do here is to selectively expand the transitive closure by
// discarding edges that are not recognized by the state machine.
//
// There is one difference between our current implementation and the one
// described in the paper: out algorithm eagerly computes all alias pairs after
// the CFLGraph is built, while in the paper the authors did the computation in
// a demand-driven fashion. We did not implement the demand-driven algorithm due
// to the additional coding complexity and higher memory profile, but if we
// found it necessary we may switch to it eventually.
//
//===----------------------------------------------------------------------===//

// N.B. AliasAnalysis as a whole is phrased as a FunctionPass at the moment, and
// CFLAndersAA is interprocedural. This is *technically* A Bad Thing, because
// FunctionPasses are only allowed to inspect the Function that they're being
// run on. Realistically, this likely isn't a problem until we allow
// FunctionPasses to run concurrently.

#include "llvm/Analysis/CFLAndersAliasAnalysis.h"
#include "CFLGraph.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Pass.h"

using namespace llvm;
using namespace llvm::cflaa;

#define DEBUG_TYPE "cfl-anders-aa"

CFLAndersAAResult::CFLAndersAAResult(const TargetLibraryInfo &TLI) : TLI(TLI) {}
CFLAndersAAResult::CFLAndersAAResult(CFLAndersAAResult &&RHS)
    : AAResultBase(std::move(RHS)), TLI(RHS.TLI) {}
CFLAndersAAResult::~CFLAndersAAResult() {}

static const Function *parentFunctionOfValue(const Value *Val) {
  if (auto *Inst = dyn_cast<Instruction>(Val)) {
    auto *Bb = Inst->getParent();
    return Bb->getParent();
  }

  if (auto *Arg = dyn_cast<Argument>(Val))
    return Arg->getParent();
  return nullptr;
}

namespace {

enum class MatchState : uint8_t {
  FlowFrom = 0,     // S1 in the paper
  FlowFromMemAlias, // S2 in the paper
  FlowTo,           // S3 in the paper
  FlowToMemAlias    // S4 in the paper
};

// We use ReachabilitySet to keep track of value aliases (The nonterminal "V" in
// the paper) during the analysis.
class ReachabilitySet {
  typedef std::bitset<4> StateSet;
  typedef DenseMap<InstantiatedValue, StateSet> ValueStateMap;
  typedef DenseMap<InstantiatedValue, ValueStateMap> ValueReachMap;
  ValueReachMap ReachMap;

public:
  typedef ValueStateMap::const_iterator const_valuestate_iterator;
  typedef ValueReachMap::const_iterator const_value_iterator;

  // Insert edge 'From->To' at state 'State'
  bool insert(InstantiatedValue From, InstantiatedValue To, MatchState State) {
    auto &States = ReachMap[To][From];
    auto Idx = static_cast<size_t>(State);
    if (!States.test(Idx)) {
      States.set(Idx);
      return true;
    }
    return false;
  }

  // Return the set of all ('From', 'State') pair for a given node 'To'
  iterator_range<const_valuestate_iterator>
  reachableValueAliases(InstantiatedValue V) const {
    auto Itr = ReachMap.find(V);
    if (Itr == ReachMap.end())
      return make_range<const_valuestate_iterator>(const_valuestate_iterator(),
                                                   const_valuestate_iterator());
    return make_range<const_valuestate_iterator>(Itr->second.begin(),
                                                 Itr->second.end());
  }

  iterator_range<const_value_iterator> value_mappings() const {
    return make_range<const_value_iterator>(ReachMap.begin(), ReachMap.end());
  }
};

// We use AliasMemSet to keep track of all memory aliases (the nonterminal "M"
// in the paper) during the analysis.
class AliasMemSet {
  typedef DenseSet<InstantiatedValue> MemSet;
  typedef DenseMap<InstantiatedValue, MemSet> MemMapType;
  MemMapType MemMap;

public:
  typedef MemSet::const_iterator const_mem_iterator;

  bool insert(InstantiatedValue LHS, InstantiatedValue RHS) {
    // Top-level values can never be memory aliases because one cannot take the
    // addresses of them
    assert(LHS.DerefLevel > 0 && RHS.DerefLevel > 0);
    return MemMap[LHS].insert(RHS).second;
  }

  const MemSet *getMemoryAliases(InstantiatedValue V) const {
    auto Itr = MemMap.find(V);
    if (Itr == MemMap.end())
      return nullptr;
    return &Itr->second;
  }
};

// We use AliasAttrMap to keep track of the AliasAttr of each node.
class AliasAttrMap {
  typedef DenseMap<InstantiatedValue, AliasAttrs> MapType;
  MapType AttrMap;

public:
  typedef MapType::const_iterator const_iterator;

  bool add(InstantiatedValue V, AliasAttrs Attr) {
    if (Attr.none())
      return false;
    auto &OldAttr = AttrMap[V];
    auto NewAttr = OldAttr | Attr;
    if (OldAttr == NewAttr)
      return false;
    OldAttr = NewAttr;
    return true;
  }

  AliasAttrs getAttrs(InstantiatedValue V) const {
    AliasAttrs Attr;
    auto Itr = AttrMap.find(V);
    if (Itr != AttrMap.end())
      Attr = Itr->second;
    return Attr;
  }

  iterator_range<const_iterator> mappings() const {
    return make_range<const_iterator>(AttrMap.begin(), AttrMap.end());
  }
};

struct WorkListItem {
  InstantiatedValue From;
  InstantiatedValue To;
  MatchState State;
};
}

class CFLAndersAAResult::FunctionInfo {
  /// Map a value to other values that may alias it
  /// Since the alias relation is symmetric, to save some space we assume values
  /// are properly ordered: if a and b alias each other, and a < b, then b is in
  /// AliasMap[a] but not vice versa.
  DenseMap<const Value *, std::vector<const Value *>> AliasMap;

  /// Map a value to its corresponding AliasAttrs
  DenseMap<const Value *, AliasAttrs> AttrMap;

  /// Summary of externally visible effects.
  AliasSummary Summary;

  AliasAttrs getAttrs(const Value *) const;

public:
  FunctionInfo(const ReachabilitySet &, AliasAttrMap);

  bool mayAlias(const Value *LHS, const Value *RHS) const;
  const AliasSummary &getAliasSummary() const { return Summary; }
};

CFLAndersAAResult::FunctionInfo::FunctionInfo(const ReachabilitySet &ReachSet,
                                              AliasAttrMap AMap) {
  // Populate AttrMap
  for (const auto &Mapping : AMap.mappings()) {
    auto IVal = Mapping.first;

    // AttrMap only cares about top-level values
    if (IVal.DerefLevel == 0)
      AttrMap[IVal.Val] = Mapping.second;
  }

  // Populate AliasMap
  for (const auto &OuterMapping : ReachSet.value_mappings()) {
    // AliasMap only cares about top-level values
    if (OuterMapping.first.DerefLevel > 0)
      continue;

    auto Val = OuterMapping.first.Val;
    auto &AliasList = AliasMap[Val];
    for (const auto &InnerMapping : OuterMapping.second) {
      // Again, AliasMap only cares about top-level values
      if (InnerMapping.first.DerefLevel == 0)
        AliasList.push_back(InnerMapping.first.Val);
    }

    // Sort AliasList for faster lookup
    std::sort(AliasList.begin(), AliasList.end(), std::less<const Value *>());
  }

  // TODO: Populate function summary here
}

AliasAttrs CFLAndersAAResult::FunctionInfo::getAttrs(const Value *V) const {
  assert(V != nullptr);

  AliasAttrs Attr;
  auto Itr = AttrMap.find(V);
  if (Itr != AttrMap.end())
    Attr = Itr->second;
  return Attr;
}

bool CFLAndersAAResult::FunctionInfo::mayAlias(const Value *LHS,
                                               const Value *RHS) const {
  assert(LHS && RHS);

  auto Itr = AliasMap.find(LHS);
  if (Itr != AliasMap.end()) {
    if (std::binary_search(Itr->second.begin(), Itr->second.end(), RHS,
                           std::less<const Value *>()))
      return true;
  }

  // Even if LHS and RHS are not reachable, they may still alias due to their
  // AliasAttrs
  auto AttrsA = getAttrs(LHS);
  auto AttrsB = getAttrs(RHS);

  if (AttrsA.none() || AttrsB.none())
    return false;
  if (hasUnknownOrCallerAttr(AttrsA) || hasUnknownOrCallerAttr(AttrsB))
    return true;
  if (isGlobalOrArgAttr(AttrsA) && isGlobalOrArgAttr(AttrsB))
    return true;
  return false;
}

static void propagate(InstantiatedValue From, InstantiatedValue To,
                      MatchState State, ReachabilitySet &ReachSet,
                      std::vector<WorkListItem> &WorkList) {
  if (From == To)
    return;
  if (ReachSet.insert(From, To, State))
    WorkList.push_back(WorkListItem{From, To, State});
}

static void initializeWorkList(std::vector<WorkListItem> &WorkList,
                               ReachabilitySet &ReachSet,
                               const CFLGraph &Graph) {
  for (const auto &Mapping : Graph.value_mappings()) {
    auto Val = Mapping.first;
    auto &ValueInfo = Mapping.second;
    assert(ValueInfo.getNumLevels() > 0);

    // Insert all immediate assignment neighbors to the worklist
    for (unsigned I = 0, E = ValueInfo.getNumLevels(); I < E; ++I) {
      auto Src = InstantiatedValue{Val, I};
      // If there's an assignment edge from X to Y, it means Y is reachable from
      // X at S2 and X is reachable from Y at S1
      for (auto &Edge : ValueInfo.getNodeInfoAtLevel(I).Edges) {
        propagate(Edge.Other, Src, MatchState::FlowFrom, ReachSet, WorkList);
        propagate(Src, Edge.Other, MatchState::FlowTo, ReachSet, WorkList);
      }
    }
  }
}

static Optional<InstantiatedValue> getNodeBelow(const CFLGraph &Graph,
                                                InstantiatedValue V) {
  auto NodeBelow = InstantiatedValue{V.Val, V.DerefLevel + 1};
  if (Graph.getNode(NodeBelow))
    return NodeBelow;
  return None;
}

static void processWorkListItem(const WorkListItem &Item, const CFLGraph &Graph,
                                ReachabilitySet &ReachSet, AliasMemSet &MemSet,
                                std::vector<WorkListItem> &WorkList) {
  auto FromNode = Item.From;
  auto ToNode = Item.To;

  auto NodeInfo = Graph.getNode(ToNode);
  assert(NodeInfo != nullptr);

  // TODO: propagate field offsets

  // FIXME: Here is a neat trick we can do: since both ReachSet and MemSet holds
  // relations that are symmetric, we could actually cut the storage by half by
  // sorting FromNode and ToNode before insertion happens.

  // The newly added value alias pair may pontentially generate more memory
  // alias pairs. Check for them here.
  auto FromNodeBelow = getNodeBelow(Graph, FromNode);
  auto ToNodeBelow = getNodeBelow(Graph, ToNode);
  if (FromNodeBelow && ToNodeBelow &&
      MemSet.insert(*FromNodeBelow, *ToNodeBelow)) {
    propagate(*FromNodeBelow, *ToNodeBelow, MatchState::FlowFromMemAlias,
              ReachSet, WorkList);
    for (const auto &Mapping : ReachSet.reachableValueAliases(*FromNodeBelow)) {
      auto Src = Mapping.first;
      if (Mapping.second.test(static_cast<size_t>(MatchState::FlowFrom)))
        propagate(Src, *ToNodeBelow, MatchState::FlowFromMemAlias, ReachSet,
                  WorkList);
      if (Mapping.second.test(static_cast<size_t>(MatchState::FlowTo)))
        propagate(Src, *ToNodeBelow, MatchState::FlowToMemAlias, ReachSet,
                  WorkList);
    }
  }

  // This is the core of the state machine walking algorithm. We expand ReachSet
  // based on which state we are at (which in turn dictates what edges we
  // should examine)
  // From a high-level point of view, the state machine here guarantees two
  // properties:
  // - If *X and *Y are memory aliases, then X and Y are value aliases
  // - If Y is an alias of X, then reverse assignment edges (if there is any)
  // should precede any assignment edges on the path from X to Y.
  switch (Item.State) {
  case MatchState::FlowFrom: {
    for (const auto &RevAssignEdge : NodeInfo->ReverseEdges)
      propagate(FromNode, RevAssignEdge.Other, MatchState::FlowFrom, ReachSet,
                WorkList);
    for (const auto &AssignEdge : NodeInfo->Edges)
      propagate(FromNode, AssignEdge.Other, MatchState::FlowTo, ReachSet,
                WorkList);
    if (auto AliasSet = MemSet.getMemoryAliases(ToNode)) {
      for (const auto &MemAlias : *AliasSet)
        propagate(FromNode, MemAlias, MatchState::FlowFromMemAlias, ReachSet,
                  WorkList);
    }
    break;
  }
  case MatchState::FlowFromMemAlias: {
    for (const auto &RevAssignEdge : NodeInfo->ReverseEdges)
      propagate(FromNode, RevAssignEdge.Other, MatchState::FlowFrom, ReachSet,
                WorkList);
    for (const auto &AssignEdge : NodeInfo->Edges)
      propagate(FromNode, AssignEdge.Other, MatchState::FlowTo, ReachSet,
                WorkList);
    break;
  }
  case MatchState::FlowTo: {
    for (const auto &AssignEdge : NodeInfo->Edges)
      propagate(FromNode, AssignEdge.Other, MatchState::FlowTo, ReachSet,
                WorkList);
    if (auto AliasSet = MemSet.getMemoryAliases(ToNode)) {
      for (const auto &MemAlias : *AliasSet)
        propagate(FromNode, MemAlias, MatchState::FlowToMemAlias, ReachSet,
                  WorkList);
    }
    break;
  }
  case MatchState::FlowToMemAlias: {
    for (const auto &AssignEdge : NodeInfo->Edges)
      propagate(FromNode, AssignEdge.Other, MatchState::FlowTo, ReachSet,
                WorkList);
    break;
  }
  }
}

static AliasAttrMap buildAttrMap(const CFLGraph &Graph,
                                 const ReachabilitySet &ReachSet) {
  AliasAttrMap AttrMap;
  std::vector<InstantiatedValue> WorkList, NextList;

  // Initialize each node with its original AliasAttrs in CFLGraph
  for (const auto &Mapping : Graph.value_mappings()) {
    auto Val = Mapping.first;
    auto &ValueInfo = Mapping.second;
    for (unsigned I = 0, E = ValueInfo.getNumLevels(); I < E; ++I) {
      auto Node = InstantiatedValue{Val, I};
      AttrMap.add(Node, ValueInfo.getNodeInfoAtLevel(I).Attr);
      WorkList.push_back(Node);
    }
  }

  while (!WorkList.empty()) {
    for (const auto &Dst : WorkList) {
      auto DstAttr = AttrMap.getAttrs(Dst);
      if (DstAttr.none())
        continue;

      // Propagate attr on the same level
      for (const auto &Mapping : ReachSet.reachableValueAliases(Dst)) {
        auto Src = Mapping.first;
        if (AttrMap.add(Src, DstAttr))
          NextList.push_back(Src);
      }

      // Propagate attr to the levels below
      auto DstBelow = getNodeBelow(Graph, Dst);
      while (DstBelow) {
        if (AttrMap.add(*DstBelow, DstAttr)) {
          NextList.push_back(*DstBelow);
          break;
        }
        DstBelow = getNodeBelow(Graph, *DstBelow);
      }
    }
    WorkList.swap(NextList);
    NextList.clear();
  }

  return AttrMap;
}

CFLAndersAAResult::FunctionInfo
CFLAndersAAResult::buildInfoFrom(const Function &Fn) {
  CFLGraphBuilder<CFLAndersAAResult> GraphBuilder(
      *this, TLI,
      // Cast away the constness here due to GraphBuilder's API requirement
      const_cast<Function &>(Fn));
  auto &Graph = GraphBuilder.getCFLGraph();

  ReachabilitySet ReachSet;
  AliasMemSet MemSet;

  std::vector<WorkListItem> WorkList, NextList;
  initializeWorkList(WorkList, ReachSet, Graph);
  // TODO: make sure we don't stop before the fix point is reached
  while (!WorkList.empty()) {
    for (const auto &Item : WorkList)
      processWorkListItem(Item, Graph, ReachSet, MemSet, NextList);

    NextList.swap(WorkList);
    NextList.clear();
  }

  // Now that we have all the reachability info, propagate AliasAttrs according
  // to it
  auto IValueAttrMap = buildAttrMap(Graph, ReachSet);

  return FunctionInfo(ReachSet, std::move(IValueAttrMap));
}

void CFLAndersAAResult::scan(const Function &Fn) {
  auto InsertPair = Cache.insert(std::make_pair(&Fn, Optional<FunctionInfo>()));
  (void)InsertPair;
  assert(InsertPair.second &&
         "Trying to scan a function that has already been cached");

  // Note that we can't do Cache[Fn] = buildSetsFrom(Fn) here: the function call
  // may get evaluated after operator[], potentially triggering a DenseMap
  // resize and invalidating the reference returned by operator[]
  auto FunInfo = buildInfoFrom(Fn);
  Cache[&Fn] = std::move(FunInfo);
  Handles.push_front(FunctionHandle(const_cast<Function *>(&Fn), this));
}

void CFLAndersAAResult::evict(const Function &Fn) { Cache.erase(&Fn); }

const Optional<CFLAndersAAResult::FunctionInfo> &
CFLAndersAAResult::ensureCached(const Function &Fn) {
  auto Iter = Cache.find(&Fn);
  if (Iter == Cache.end()) {
    scan(Fn);
    Iter = Cache.find(&Fn);
    assert(Iter != Cache.end());
    assert(Iter->second.hasValue());
  }
  return Iter->second;
}

const AliasSummary *CFLAndersAAResult::getAliasSummary(const Function &Fn) {
  auto &FunInfo = ensureCached(Fn);
  if (FunInfo.hasValue())
    return &FunInfo->getAliasSummary();
  else
    return nullptr;
}

AliasResult CFLAndersAAResult::query(const MemoryLocation &LocA,
                                     const MemoryLocation &LocB) {
  auto *ValA = LocA.Ptr;
  auto *ValB = LocB.Ptr;

  if (!ValA->getType()->isPointerTy() || !ValB->getType()->isPointerTy())
    return NoAlias;

  auto *Fn = parentFunctionOfValue(ValA);
  if (!Fn) {
    Fn = parentFunctionOfValue(ValB);
    if (!Fn) {
      // The only times this is known to happen are when globals + InlineAsm are
      // involved
      DEBUG(dbgs()
            << "CFLAndersAA: could not extract parent function information.\n");
      return MayAlias;
    }
  } else {
    assert(!parentFunctionOfValue(ValB) || parentFunctionOfValue(ValB) == Fn);
  }

  assert(Fn != nullptr);
  auto &FunInfo = ensureCached(*Fn);

  // AliasMap lookup
  if (FunInfo->mayAlias(ValA, ValB))
    return MayAlias;
  return NoAlias;
}

AliasResult CFLAndersAAResult::alias(const MemoryLocation &LocA,
                                     const MemoryLocation &LocB) {
  if (LocA.Ptr == LocB.Ptr)
    return LocA.Size == LocB.Size ? MustAlias : PartialAlias;

  // Comparisons between global variables and other constants should be
  // handled by BasicAA.
  // CFLAndersAA may report NoAlias when comparing a GlobalValue and
  // ConstantExpr, but every query needs to have at least one Value tied to a
  // Function, and neither GlobalValues nor ConstantExprs are.
  if (isa<Constant>(LocA.Ptr) && isa<Constant>(LocB.Ptr))
    return AAResultBase::alias(LocA, LocB);

  AliasResult QueryResult = query(LocA, LocB);
  if (QueryResult == MayAlias)
    return AAResultBase::alias(LocA, LocB);

  return QueryResult;
}

char CFLAndersAA::PassID;

CFLAndersAAResult CFLAndersAA::run(Function &F, AnalysisManager<Function> &AM) {
  return CFLAndersAAResult(AM.getResult<TargetLibraryAnalysis>(F));
}

char CFLAndersAAWrapperPass::ID = 0;
INITIALIZE_PASS(CFLAndersAAWrapperPass, "cfl-anders-aa",
                "Inclusion-Based CFL Alias Analysis", false, true)

ImmutablePass *llvm::createCFLAndersAAWrapperPass() {
  return new CFLAndersAAWrapperPass();
}

CFLAndersAAWrapperPass::CFLAndersAAWrapperPass() : ImmutablePass(ID) {
  initializeCFLAndersAAWrapperPassPass(*PassRegistry::getPassRegistry());
}

void CFLAndersAAWrapperPass::initializePass() {
  auto &TLIWP = getAnalysis<TargetLibraryInfoWrapperPass>();
  Result.reset(new CFLAndersAAResult(TLIWP.getTLI()));
}

void CFLAndersAAWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<TargetLibraryInfoWrapperPass>();
}
