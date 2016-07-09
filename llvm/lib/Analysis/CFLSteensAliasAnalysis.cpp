//- CFLSteensAliasAnalysis.cpp - Unification-based Alias Analysis ---*- C++-*-//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a CFL-base, summary-based alias analysis algorithm. It
// does not depend on types. The algorithm is a mixture of the one described in
// "Demand-driven alias analysis for C" by Xin Zheng and Radu Rugina, and "Fast
// algorithms for Dyck-CFL-reachability with applications to Alias Analysis" by
// Zhang Q, Lyu M R, Yuan H, and Su Z. -- to summarize the papers, we build a
// graph of the uses of a variable, where each node is a memory location, and
// each edge is an action that happened on that memory location.  The "actions"
// can be one of Dereference, Reference, or Assign. The precision of this
// analysis is roughly the same as that of an one level context-sensitive
// Steensgaard's algorithm.
//
// Two variables are considered as aliasing iff you can reach one value's node
// from the other value's node and the language formed by concatenating all of
// the edge labels (actions) conforms to a context-free grammar.
//
// Because this algorithm requires a graph search on each query, we execute the
// algorithm outlined in "Fast algorithms..." (mentioned above)
// in order to transform the graph into sets of variables that may alias in
// ~nlogn time (n = number of variables), which makes queries take constant
// time.
//===----------------------------------------------------------------------===//

// N.B. AliasAnalysis as a whole is phrased as a FunctionPass at the moment, and
// CFLSteensAA is interprocedural. This is *technically* A Bad Thing, because
// FunctionPasses are only allowed to inspect the Function that they're being
// run on. Realistically, this likely isn't a problem until we allow
// FunctionPasses to run concurrently.

#include "llvm/Analysis/CFLSteensAliasAnalysis.h"
#include "CFLGraph.h"
#include "StratifiedSets.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <memory>
#include <tuple>

using namespace llvm;
using namespace llvm::cflaa;

#define DEBUG_TYPE "cfl-steens-aa"

CFLSteensAAResult::CFLSteensAAResult(const TargetLibraryInfo &TLI)
    : AAResultBase(), TLI(TLI) {}
CFLSteensAAResult::CFLSteensAAResult(CFLSteensAAResult &&Arg)
    : AAResultBase(std::move(Arg)), TLI(Arg.TLI) {}
CFLSteensAAResult::~CFLSteensAAResult() {}

/// Information we have about a function and would like to keep around.
class CFLSteensAAResult::FunctionInfo {
  StratifiedSets<Value *> Sets;
  AliasSummary Summary;

public:
  FunctionInfo(Function &Fn, const SmallVectorImpl<Value *> &RetVals,
               StratifiedSets<Value *> S);

  const StratifiedSets<Value *> &getStratifiedSets() const { return Sets; }
  const AliasSummary &getAliasSummary() const { return Summary; }
};

/// Try to go from a Value* to a Function*. Never returns nullptr.
static Optional<Function *> parentFunctionOfValue(Value *);

const StratifiedIndex StratifiedLink::SetSentinel =
    std::numeric_limits<StratifiedIndex>::max();

namespace {

/// StratifiedSets call for knowledge of "direction", so this is how we
/// represent that locally.
enum class Level { Same, Above, Below };
}

//===----------------------------------------------------------------------===//
// Function declarations that require types defined in the namespace above
//===----------------------------------------------------------------------===//

/// Gets the "Level" that one should travel in StratifiedSets
/// given an EdgeType.
static Level directionOfEdgeType(EdgeType);

/// Determines whether it would be pointless to add the given Value to our sets.
static bool canSkipAddingToSets(Value *Val);

static Optional<Function *> parentFunctionOfValue(Value *Val) {
  if (auto *Inst = dyn_cast<Instruction>(Val)) {
    auto *Bb = Inst->getParent();
    return Bb->getParent();
  }

  if (auto *Arg = dyn_cast<Argument>(Val))
    return Arg->getParent();
  return None;
}

static Level directionOfEdgeType(EdgeType Weight) {
  switch (Weight) {
  case EdgeType::Reference:
    return Level::Above;
  case EdgeType::Dereference:
    return Level::Below;
  case EdgeType::Assign:
    return Level::Same;
  }
  llvm_unreachable("Incomplete switch coverage");
}

static bool canSkipAddingToSets(Value *Val) {
  // Constants can share instances, which may falsely unify multiple
  // sets, e.g. in
  // store i32* null, i32** %ptr1
  // store i32* null, i32** %ptr2
  // clearly ptr1 and ptr2 should not be unified into the same set, so
  // we should filter out the (potentially shared) instance to
  // i32* null.
  if (isa<Constant>(Val)) {
    // TODO: Because all of these things are constant, we can determine whether
    // the data is *actually* mutable at graph building time. This will probably
    // come for free/cheap with offset awareness.
    bool CanStoreMutableData = isa<GlobalValue>(Val) ||
                               isa<ConstantExpr>(Val) ||
                               isa<ConstantAggregate>(Val);
    return !CanStoreMutableData;
  }

  return false;
}

CFLSteensAAResult::FunctionInfo::FunctionInfo(
    Function &Fn, const SmallVectorImpl<Value *> &RetVals,
    StratifiedSets<Value *> S)
    : Sets(std::move(S)) {
  // Historically, an arbitrary upper-bound of 50 args was selected. We may want
  // to remove this if it doesn't really matter in practice.
  if (Fn.arg_size() > MaxSupportedArgsInSummary)
    return;

  DenseMap<StratifiedIndex, InterfaceValue> InterfaceMap;

  // Our intention here is to record all InterfaceValues that share the same
  // StratifiedIndex in RetParamRelations. For each valid InterfaceValue, we
  // have its StratifiedIndex scanned here and check if the index is presented
  // in InterfaceMap: if it is not, we add the correspondence to the map;
  // otherwise, an aliasing relation is found and we add it to
  // RetParamRelations.

  auto AddToRetParamRelations = [&](unsigned InterfaceIndex,
                                    StratifiedIndex SetIndex) {
    unsigned Level = 0;
    while (true) {
      InterfaceValue CurrValue{InterfaceIndex, Level};

      auto Itr = InterfaceMap.find(SetIndex);
      if (Itr != InterfaceMap.end()) {
        if (CurrValue != Itr->second)
          Summary.RetParamRelations.push_back(
              ExternalRelation{CurrValue, Itr->second});
        break;
      }

      auto &Link = Sets.getLink(SetIndex);
      InterfaceMap.insert(std::make_pair(SetIndex, CurrValue));
      auto ExternalAttrs = getExternallyVisibleAttrs(Link.Attrs);
      if (ExternalAttrs.any())
        Summary.RetParamAttributes.push_back(
            ExternalAttribute{CurrValue, ExternalAttrs});

      if (!Link.hasBelow())
        break;

      ++Level;
      SetIndex = Link.Below;
    }
  };

  // Populate RetParamRelations for return values
  for (auto *RetVal : RetVals) {
    assert(RetVal != nullptr);
    assert(RetVal->getType()->isPointerTy());
    auto RetInfo = Sets.find(RetVal);
    if (RetInfo.hasValue())
      AddToRetParamRelations(0, RetInfo->Index);
  }

  // Populate RetParamRelations for parameters
  unsigned I = 0;
  for (auto &Param : Fn.args()) {
    if (Param.getType()->isPointerTy()) {
      auto ParamInfo = Sets.find(&Param);
      if (ParamInfo.hasValue())
        AddToRetParamRelations(I + 1, ParamInfo->Index);
    }
    ++I;
  }
}

// Builds the graph + StratifiedSets for a function.
CFLSteensAAResult::FunctionInfo CFLSteensAAResult::buildSetsFrom(Function *Fn) {
  CFLGraphBuilder<CFLSteensAAResult> GraphBuilder(*this, TLI, *Fn);
  StratifiedSetsBuilder<Value *> SetBuilder;

  auto &Graph = GraphBuilder.getCFLGraph();
  SmallVector<Value *, 16> Worklist;
  for (auto Node : Graph.nodes())
    Worklist.push_back(Node);

  while (!Worklist.empty()) {
    auto *CurValue = Worklist.pop_back_val();
    SetBuilder.add(CurValue);
    if (canSkipAddingToSets(CurValue))
      continue;

    auto Attr = Graph.attrFor(CurValue);
    SetBuilder.noteAttributes(CurValue, Attr);

    for (const auto &Edge : Graph.edgesFor(CurValue)) {
      auto Label = Edge.Type;
      auto *OtherValue = Edge.Other;

      if (canSkipAddingToSets(OtherValue))
        continue;

      bool Added;
      switch (directionOfEdgeType(Label)) {
      case Level::Above:
        Added = SetBuilder.addAbove(CurValue, OtherValue);
        break;
      case Level::Below:
        Added = SetBuilder.addBelow(CurValue, OtherValue);
        break;
      case Level::Same:
        Added = SetBuilder.addWith(CurValue, OtherValue);
        break;
      }

      if (Added)
        Worklist.push_back(OtherValue);
    }
  }

  // Special handling for interprocedural aliases
  for (auto &Edge : GraphBuilder.getInstantiatedRelations()) {
    auto FromVal = Edge.From.Val;
    auto ToVal = Edge.To.Val;
    SetBuilder.add(FromVal);
    SetBuilder.add(ToVal);
    SetBuilder.addBelowWith(FromVal, Edge.From.DerefLevel, ToVal,
                            Edge.To.DerefLevel);
  }

  // Special handling for interprocedural attributes
  for (auto &IPAttr : GraphBuilder.getInstantiatedAttrs()) {
    auto Val = IPAttr.IValue.Val;
    SetBuilder.add(Val);
    SetBuilder.addAttributesBelow(Val, IPAttr.IValue.DerefLevel, IPAttr.Attr);
  }

  return FunctionInfo(*Fn, GraphBuilder.getReturnValues(), SetBuilder.build());
}

void CFLSteensAAResult::scan(Function *Fn) {
  auto InsertPair = Cache.insert(std::make_pair(Fn, Optional<FunctionInfo>()));
  (void)InsertPair;
  assert(InsertPair.second &&
         "Trying to scan a function that has already been cached");

  // Note that we can't do Cache[Fn] = buildSetsFrom(Fn) here: the function call
  // may get evaluated after operator[], potentially triggering a DenseMap
  // resize and invalidating the reference returned by operator[]
  auto FunInfo = buildSetsFrom(Fn);
  Cache[Fn] = std::move(FunInfo);

  Handles.push_front(FunctionHandle(Fn, this));
}

void CFLSteensAAResult::evict(Function *Fn) { Cache.erase(Fn); }

/// Ensures that the given function is available in the cache, and returns the
/// entry.
const Optional<CFLSteensAAResult::FunctionInfo> &
CFLSteensAAResult::ensureCached(Function *Fn) {
  auto Iter = Cache.find(Fn);
  if (Iter == Cache.end()) {
    scan(Fn);
    Iter = Cache.find(Fn);
    assert(Iter != Cache.end());
    assert(Iter->second.hasValue());
  }
  return Iter->second;
}

const AliasSummary *CFLSteensAAResult::getAliasSummary(Function &Fn) {
  auto &FunInfo = ensureCached(&Fn);
  if (FunInfo.hasValue())
    return &FunInfo->getAliasSummary();
  else
    return nullptr;
}

AliasResult CFLSteensAAResult::query(const MemoryLocation &LocA,
                                     const MemoryLocation &LocB) {
  auto *ValA = const_cast<Value *>(LocA.Ptr);
  auto *ValB = const_cast<Value *>(LocB.Ptr);

  if (!ValA->getType()->isPointerTy() || !ValB->getType()->isPointerTy())
    return NoAlias;

  Function *Fn = nullptr;
  auto MaybeFnA = parentFunctionOfValue(ValA);
  auto MaybeFnB = parentFunctionOfValue(ValB);
  if (!MaybeFnA.hasValue() && !MaybeFnB.hasValue()) {
    // The only times this is known to happen are when globals + InlineAsm are
    // involved
    DEBUG(dbgs()
          << "CFLSteensAA: could not extract parent function information.\n");
    return MayAlias;
  }

  if (MaybeFnA.hasValue()) {
    Fn = *MaybeFnA;
    assert((!MaybeFnB.hasValue() || *MaybeFnB == *MaybeFnA) &&
           "Interprocedural queries not supported");
  } else {
    Fn = *MaybeFnB;
  }

  assert(Fn != nullptr);
  auto &MaybeInfo = ensureCached(Fn);
  assert(MaybeInfo.hasValue());

  auto &Sets = MaybeInfo->getStratifiedSets();
  auto MaybeA = Sets.find(ValA);
  if (!MaybeA.hasValue())
    return MayAlias;

  auto MaybeB = Sets.find(ValB);
  if (!MaybeB.hasValue())
    return MayAlias;

  auto SetA = *MaybeA;
  auto SetB = *MaybeB;
  auto AttrsA = Sets.getLink(SetA.Index).Attrs;
  auto AttrsB = Sets.getLink(SetB.Index).Attrs;

  // If both values are local (meaning the corresponding set has attribute
  // AttrNone or AttrEscaped), then we know that CFLSteensAA fully models them:
  // they may-alias each other if and only if they are in the same set.
  // If at least one value is non-local (meaning it either is global/argument or
  // it comes from unknown sources like integer cast), the situation becomes a
  // bit more interesting. We follow three general rules described below:
  // - Non-local values may alias each other
  // - AttrNone values do not alias any non-local values
  // - AttrEscaped do not alias globals/arguments, but they may alias
  // AttrUnknown values
  if (SetA.Index == SetB.Index)
    return MayAlias;
  if (AttrsA.none() || AttrsB.none())
    return NoAlias;
  if (hasUnknownOrCallerAttr(AttrsA) || hasUnknownOrCallerAttr(AttrsB))
    return MayAlias;
  if (isGlobalOrArgAttr(AttrsA) && isGlobalOrArgAttr(AttrsB))
    return MayAlias;
  return NoAlias;
}

ModRefInfo CFLSteensAAResult::getArgModRefInfo(ImmutableCallSite CS,
                                               unsigned ArgIdx) {
  if (auto CalledFunc = CS.getCalledFunction()) {
    auto &MaybeInfo = ensureCached(const_cast<Function *>(CalledFunc));
    if (!MaybeInfo.hasValue())
      return MRI_ModRef;
    auto &RetParamAttributes = MaybeInfo->getAliasSummary().RetParamAttributes;
    auto &RetParamRelations = MaybeInfo->getAliasSummary().RetParamRelations;

    bool ArgAttributeIsWritten =
        std::any_of(RetParamAttributes.begin(), RetParamAttributes.end(),
                    [ArgIdx](const ExternalAttribute &ExtAttr) {
                      return ExtAttr.IValue.Index == ArgIdx + 1;
                    });
    bool ArgIsAccessed =
        std::any_of(RetParamRelations.begin(), RetParamRelations.end(),
                    [ArgIdx](const ExternalRelation &ExtRelation) {
                      return ExtRelation.To.Index == ArgIdx + 1 ||
                             ExtRelation.From.Index == ArgIdx + 1;
                    });

    return (!ArgIsAccessed && !ArgAttributeIsWritten) ? MRI_NoModRef
                                                      : MRI_ModRef;
  }

  return MRI_ModRef;
}

FunctionModRefBehavior
CFLSteensAAResult::getModRefBehavior(ImmutableCallSite CS) {
  // If we know the callee, try analyzing it
  if (auto CalledFunc = CS.getCalledFunction())
    return getModRefBehavior(CalledFunc);

  // Otherwise, be conservative
  return FMRB_UnknownModRefBehavior;
}

FunctionModRefBehavior CFLSteensAAResult::getModRefBehavior(const Function *F) {
  assert(F != nullptr);

  // TODO: Remove the const_cast
  auto &MaybeInfo = ensureCached(const_cast<Function *>(F));
  if (!MaybeInfo.hasValue())
    return FMRB_UnknownModRefBehavior;
  auto &RetParamAttributes = MaybeInfo->getAliasSummary().RetParamAttributes;
  auto &RetParamRelations = MaybeInfo->getAliasSummary().RetParamRelations;

  // First, if any argument is marked Escpaed, Unknown or Global, anything may
  // happen to them and thus we can't draw any conclusion.
  if (!RetParamAttributes.empty())
    return FMRB_UnknownModRefBehavior;

  // Currently we don't (and can't) distinguish reads from writes in
  // RetParamRelations. All we can say is whether there may be memory access or
  // not.
  if (RetParamRelations.empty())
    return FMRB_DoesNotAccessMemory;

  // Check if something beyond argmem gets touched.
  bool AccessArgMemoryOnly =
      std::all_of(RetParamRelations.begin(), RetParamRelations.end(),
                  [](const ExternalRelation &ExtRelation) {
                    // Both DerefLevels has to be 0, since we don't know which
                    // one is a read and which is a write.
                    return ExtRelation.From.DerefLevel == 0 &&
                           ExtRelation.To.DerefLevel == 0;
                  });
  return AccessArgMemoryOnly ? FMRB_OnlyAccessesArgumentPointees
                             : FMRB_UnknownModRefBehavior;
}

char CFLSteensAA::PassID;

CFLSteensAAResult CFLSteensAA::run(Function &F, AnalysisManager<Function> &AM) {
  return CFLSteensAAResult(AM.getResult<TargetLibraryAnalysis>(F));
}

char CFLSteensAAWrapperPass::ID = 0;
INITIALIZE_PASS(CFLSteensAAWrapperPass, "cfl-steens-aa",
                "Unification-Based CFL Alias Analysis", false, true)

ImmutablePass *llvm::createCFLSteensAAWrapperPass() {
  return new CFLSteensAAWrapperPass();
}

CFLSteensAAWrapperPass::CFLSteensAAWrapperPass() : ImmutablePass(ID) {
  initializeCFLSteensAAWrapperPassPass(*PassRegistry::getPassRegistry());
}

void CFLSteensAAWrapperPass::initializePass() {
  auto &TLIWP = getAnalysis<TargetLibraryInfoWrapperPass>();
  Result.reset(new CFLSteensAAResult(TLIWP.getTLI()));
}

void CFLSteensAAWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<TargetLibraryInfoWrapperPass>();
}
