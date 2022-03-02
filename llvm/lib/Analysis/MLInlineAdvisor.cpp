//===- MLInlineAdvisor.cpp - machine learned InlineAdvisor ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the interface between the inliner and a learned model.
// It delegates model evaluation to either the AOT compiled model (the
// 'release' mode) or a runtime-loaded model (the 'development' case).
//
//===----------------------------------------------------------------------===//
#include "llvm/Analysis/MLInlineAdvisor.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/FunctionPropertiesAnalysis.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/InlineModelFeatureMaps.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/MLModelRunner.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#if defined(LLVM_HAVE_TF_AOT_INLINERSIZEMODEL)
#include "llvm/Analysis/ReleaseModeModelRunner.h"
// codegen-ed file
#include "InlinerSizeModel.h" // NOLINT

std::unique_ptr<InlineAdvisor>
llvm::getReleaseModeAdvisor(Module &M, ModuleAnalysisManager &MAM) {
  auto AOTRunner =
      std::make_unique<ReleaseModeModelRunner<llvm::InlinerSizeModel>>(
          M.getContext(), FeatureNameMap, DecisionName);
  return std::make_unique<MLInlineAdvisor>(M, MAM, std::move(AOTRunner));
}
#endif

#define DEBUG_TYPE "inline-ml"

static cl::opt<float> SizeIncreaseThreshold(
    "ml-advisor-size-increase-threshold", cl::Hidden,
    cl::desc("Maximum factor by which expected native size may increase before "
             "blocking any further inlining."),
    cl::init(2.0));

// clang-format off
const std::array<std::string, NumberOfFeatures> llvm::FeatureNameMap{
// InlineCost features - these must come first
#define POPULATE_NAMES(INDEX_NAME, NAME) NAME,
  INLINE_COST_FEATURE_ITERATOR(POPULATE_NAMES)
#undef POPULATE_NAMES

// Non-cost features
#define POPULATE_NAMES(INDEX_NAME, NAME, COMMENT) NAME,
  INLINE_FEATURE_ITERATOR(POPULATE_NAMES)
#undef POPULATE_NAMES
};
// clang-format on

const char *const llvm::DecisionName = "inlining_decision";
const char *const llvm::DefaultDecisionName = "inlining_default";
const char *const llvm::RewardName = "delta_size";

CallBase *getInlinableCS(Instruction &I) {
  if (auto *CS = dyn_cast<CallBase>(&I))
    if (Function *Callee = CS->getCalledFunction()) {
      if (!Callee->isDeclaration()) {
        return CS;
      }
    }
  return nullptr;
}

MLInlineAdvisor::MLInlineAdvisor(Module &M, ModuleAnalysisManager &MAM,
                                 std::unique_ptr<MLModelRunner> Runner)
    : InlineAdvisor(
          M, MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager()),
      ModelRunner(std::move(Runner)),
      CG(MAM.getResult<LazyCallGraphAnalysis>(M)),
      InitialIRSize(getModuleIRSize()), CurrentIRSize(InitialIRSize) {
  assert(ModelRunner);

  // Extract the 'call site height' feature - the position of a call site
  // relative to the farthest statically reachable SCC node. We don't mutate
  // this value while inlining happens. Empirically, this feature proved
  // critical in behavioral cloning - i.e. training a model to mimic the manual
  // heuristic's decisions - and, thus, equally important for training for
  // improvement.
  CallGraph CGraph(M);
  for (auto I = scc_begin(&CGraph); !I.isAtEnd(); ++I) {
    const std::vector<CallGraphNode *> &CGNodes = *I;
    unsigned Level = 0;
    for (auto *CGNode : CGNodes) {
      Function *F = CGNode->getFunction();
      if (!F || F->isDeclaration())
        continue;
      for (auto &I : instructions(F)) {
        if (auto *CS = getInlinableCS(I)) {
          auto *Called = CS->getCalledFunction();
          auto Pos = FunctionLevels.find(&CG.get(*Called));
          // In bottom up traversal, an inlinable callee is either in the
          // same SCC, or to a function in a visited SCC. So not finding its
          // level means we haven't visited it yet, meaning it's in this SCC.
          if (Pos == FunctionLevels.end())
            continue;
          Level = std::max(Level, Pos->second + 1);
        }
      }
    }
    for (auto *CGNode : CGNodes) {
      Function *F = CGNode->getFunction();
      if (F && !F->isDeclaration())
        FunctionLevels[&CG.get(*F)] = Level;
    }
  }
  for (auto KVP : FunctionLevels) {
    AllNodes.insert(KVP.first);
    EdgeCount += getLocalCalls(KVP.first->getFunction());
  }
  NodeCount = AllNodes.size();
}

unsigned MLInlineAdvisor::getInitialFunctionLevel(const Function &F) const {
  return CG.lookup(F) ? FunctionLevels.at(CG.lookup(F)) : 0;
}

void MLInlineAdvisor::onPassEntry() {
  // Function passes executed between InlinerPass runs may have changed the
  // module-wide features.
  // The cgscc pass manager rules are such that:
  // - if a pass leads to merging SCCs, then the pipeline is restarted on the
  // merged SCC
  // - if a pass leads to splitting the SCC, then we continue with one of the
  // splits
  // This means that the NodesInLastSCC is a superset (not strict) of the nodes
  // that subsequent passes would have processed
  // - in addition, if new Nodes were created by a pass (e.g. CoroSplit),
  // they'd be adjacent to Nodes in the last SCC. So we just need to check the
  // boundary of Nodes in NodesInLastSCC for Nodes we haven't seen. We don't
  // care about the nature of the Edge (call or ref).
  NodeCount -= static_cast<int64_t>(NodesInLastSCC.size());
  while (!NodesInLastSCC.empty()) {
    const auto *N = NodesInLastSCC.front();
    NodesInLastSCC.pop_front();
    // The Function wrapped by N could have been deleted since we last saw it.
    if (N->isDead()) {
      assert(!N->getFunction().isDeclaration());
      continue;
    }
    ++NodeCount;
    EdgeCount += getLocalCalls(N->getFunction());
    for (const auto &E : *(*N)) {
      const auto *AdjNode = &E.getNode();
      assert(!AdjNode->isDead() && !AdjNode->getFunction().isDeclaration());
      auto I = AllNodes.insert(AdjNode);
      if (I.second)
        NodesInLastSCC.push_back(AdjNode);
    }
  }

  EdgeCount -= EdgesOfLastSeenNodes;
  EdgesOfLastSeenNodes = 0;
}

void MLInlineAdvisor::onPassExit(LazyCallGraph::SCC *LastSCC) {
  if (!LastSCC)
    return;
  // Keep track of the nodes and edges we last saw. Then, in onPassEntry,
  // we update the node count and edge count from the subset of these nodes that
  // survived.
  assert(NodesInLastSCC.empty());
  assert(NodeCount >= LastSCC->size());
  EdgesOfLastSeenNodes = 0;
  for (const auto &N : *LastSCC) {
    assert(!N.isDead());
    EdgesOfLastSeenNodes += getLocalCalls(N.getFunction());
    NodesInLastSCC.push_back(&N);
  }
  assert(EdgeCount >= EdgesOfLastSeenNodes);
}

int64_t MLInlineAdvisor::getLocalCalls(Function &F) {
  return FAM.getResult<FunctionPropertiesAnalysis>(F)
      .DirectCallsToDefinedFunctions;
}

// Update the internal state of the advisor, and force invalidate feature
// analysis. Currently, we maintain minimal (and very simple) global state - the
// number of functions and the number of static calls. We also keep track of the
// total IR size in this module, to stop misbehaving policies at a certain bloat
// factor (SizeIncreaseThreshold)
void MLInlineAdvisor::onSuccessfulInlining(const MLInlineAdvice &Advice,
                                           bool CalleeWasDeleted) {
  assert(!ForceStop);
  Function *Caller = Advice.getCaller();
  Function *Callee = Advice.getCallee();

  // The caller features aren't valid anymore.
  {
    PreservedAnalyses PA = PreservedAnalyses::all();
    PA.abandon<FunctionPropertiesAnalysis>();
    FAM.invalidate(*Caller, PA);
  }
  int64_t IRSizeAfter =
      getIRSize(*Caller) + (CalleeWasDeleted ? 0 : Advice.CalleeIRSize);
  CurrentIRSize += IRSizeAfter - (Advice.CallerIRSize + Advice.CalleeIRSize);
  if (CurrentIRSize > SizeIncreaseThreshold * InitialIRSize)
    ForceStop = true;

  // We can delta-update module-wide features. We know the inlining only changed
  // the caller, and maybe the callee (by deleting the latter).
  // Nodes are simple to update.
  // For edges, we 'forget' the edges that the caller and callee used to have
  // before inlining, and add back what they currently have together.
  int64_t NewCallerAndCalleeEdges =
      FAM.getResult<FunctionPropertiesAnalysis>(*Caller)
          .DirectCallsToDefinedFunctions;

  if (CalleeWasDeleted)
    --NodeCount;
  else
    NewCallerAndCalleeEdges +=
        FAM.getResult<FunctionPropertiesAnalysis>(*Callee)
            .DirectCallsToDefinedFunctions;
  EdgeCount += (NewCallerAndCalleeEdges - Advice.CallerAndCalleeEdges);
  assert(CurrentIRSize >= 0 && EdgeCount >= 0 && NodeCount >= 0);
}

int64_t MLInlineAdvisor::getModuleIRSize() const {
  int64_t Ret = 0;
  for (auto &F : M)
    if (!F.isDeclaration())
      Ret += getIRSize(F);
  return Ret;
}

std::unique_ptr<InlineAdvice> MLInlineAdvisor::getAdviceImpl(CallBase &CB) {
  auto &Caller = *CB.getCaller();
  auto &Callee = *CB.getCalledFunction();

  auto GetAssumptionCache = [&](Function &F) -> AssumptionCache & {
    return FAM.getResult<AssumptionAnalysis>(F);
  };
  auto &TIR = FAM.getResult<TargetIRAnalysis>(Callee);
  auto &ORE = FAM.getResult<OptimizationRemarkEmitterAnalysis>(Caller);

  auto MandatoryKind = InlineAdvisor::getMandatoryKind(CB, FAM, ORE);
  // If this is a "never inline" case, there won't be any changes to internal
  // state we need to track, so we can just return the base InlineAdvice, which
  // will do nothing interesting.
  // Same thing if this is a recursive case.
  if (MandatoryKind == InlineAdvisor::MandatoryInliningKind::Never ||
      &Caller == &Callee)
    return getMandatoryAdvice(CB, false);

  bool Mandatory =
      MandatoryKind == InlineAdvisor::MandatoryInliningKind::Always;

  // If we need to stop, we won't want to track anymore any state changes, so
  // we just return the base InlineAdvice, which acts as a noop.
  if (ForceStop) {
    ORE.emit([&] {
      return OptimizationRemarkMissed(DEBUG_TYPE, "ForceStop", &CB)
             << "Won't attempt inlining because module size grew too much.";
    });
    return std::make_unique<InlineAdvice>(this, CB, ORE, Mandatory);
  }

  int CostEstimate = 0;
  if (!Mandatory) {
    auto IsCallSiteInlinable =
        llvm::getInliningCostEstimate(CB, TIR, GetAssumptionCache);
    if (!IsCallSiteInlinable) {
      // We can't inline this for correctness reasons, so return the base
      // InlineAdvice, as we don't care about tracking any state changes (which
      // won't happen).
      return std::make_unique<InlineAdvice>(this, CB, ORE, false);
    }
    CostEstimate = *IsCallSiteInlinable;
  }

  const auto CostFeatures =
      llvm::getInliningCostFeatures(CB, TIR, GetAssumptionCache);
  if (!CostFeatures) {
    return std::make_unique<InlineAdvice>(this, CB, ORE, false);
  }

  if (Mandatory)
    return getMandatoryAdvice(CB, true);

  auto NrCtantParams = 0;
  for (auto I = CB.arg_begin(), E = CB.arg_end(); I != E; ++I) {
    NrCtantParams += (isa<Constant>(*I));
  }

  auto &CallerBefore = FAM.getResult<FunctionPropertiesAnalysis>(Caller);
  auto &CalleeBefore = FAM.getResult<FunctionPropertiesAnalysis>(Callee);

  *ModelRunner->getTensor<int64_t>(FeatureIndex::CalleeBasicBlockCount) =
      CalleeBefore.BasicBlockCount;
  *ModelRunner->getTensor<int64_t>(FeatureIndex::CallSiteHeight) =
      getInitialFunctionLevel(Caller);
  *ModelRunner->getTensor<int64_t>(FeatureIndex::NodeCount) = NodeCount;
  *ModelRunner->getTensor<int64_t>(FeatureIndex::NrCtantParams) = NrCtantParams;
  *ModelRunner->getTensor<int64_t>(FeatureIndex::EdgeCount) = EdgeCount;
  *ModelRunner->getTensor<int64_t>(FeatureIndex::CallerUsers) =
      CallerBefore.Uses;
  *ModelRunner->getTensor<int64_t>(
      FeatureIndex::CallerConditionallyExecutedBlocks) =
      CallerBefore.BlocksReachedFromConditionalInstruction;
  *ModelRunner->getTensor<int64_t>(FeatureIndex::CallerBasicBlockCount) =
      CallerBefore.BasicBlockCount;
  *ModelRunner->getTensor<int64_t>(
      FeatureIndex::CalleeConditionallyExecutedBlocks) =
      CalleeBefore.BlocksReachedFromConditionalInstruction;
  *ModelRunner->getTensor<int64_t>(FeatureIndex::CalleeUsers) =
      CalleeBefore.Uses;
  *ModelRunner->getTensor<int64_t>(FeatureIndex::CostEstimate) = CostEstimate;

  // Add the cost features
  for (size_t I = 0;
       I < static_cast<size_t>(InlineCostFeatureIndex::NumberOfFeatures); ++I) {
    *ModelRunner->getTensor<int64_t>(inlineCostFeatureToMlFeature(
        static_cast<InlineCostFeatureIndex>(I))) = CostFeatures->at(I);
  }

  return getAdviceFromModel(CB, ORE);
}

std::unique_ptr<MLInlineAdvice>
MLInlineAdvisor::getAdviceFromModel(CallBase &CB,
                                    OptimizationRemarkEmitter &ORE) {
  return std::make_unique<MLInlineAdvice>(
      this, CB, ORE, static_cast<bool>(ModelRunner->evaluate<int64_t>()));
}

std::unique_ptr<InlineAdvice> MLInlineAdvisor::getMandatoryAdvice(CallBase &CB,
                                                                  bool Advice) {
  // Make sure we track inlinings in all cases - mandatory or not.
  if (Advice && !ForceStop)
    return getMandatoryAdviceImpl(CB);

  // If this is a "never inline" case, there won't be any changes to internal
  // state we need to track, so we can just return the base InlineAdvice, which
  // will do nothing interesting.
  // Same if we are forced to stop - we don't track anymore.
  return std::make_unique<InlineAdvice>(this, CB, getCallerORE(CB), Advice);
}

std::unique_ptr<MLInlineAdvice>
MLInlineAdvisor::getMandatoryAdviceImpl(CallBase &CB) {
  return std::make_unique<MLInlineAdvice>(this, CB, getCallerORE(CB), true);
}

void MLInlineAdvice::reportContextForRemark(
    DiagnosticInfoOptimizationBase &OR) {
  using namespace ore;
  OR << NV("Callee", Callee->getName());
  for (size_t I = 0; I < NumberOfFeatures; ++I)
    OR << NV(FeatureNameMap[I],
             *getAdvisor()->getModelRunner().getTensor<int64_t>(I));
  OR << NV("ShouldInline", isInliningRecommended());
}

void MLInlineAdvice::recordInliningImpl() {
  ORE.emit([&]() {
    OptimizationRemark R(DEBUG_TYPE, "InliningSuccess", DLoc, Block);
    reportContextForRemark(R);
    return R;
  });
  getAdvisor()->onSuccessfulInlining(*this, /*CalleeWasDeleted*/ false);
}

void MLInlineAdvice::recordInliningWithCalleeDeletedImpl() {
  ORE.emit([&]() {
    OptimizationRemark R(DEBUG_TYPE, "InliningSuccessWithCalleeDeleted", DLoc,
                         Block);
    reportContextForRemark(R);
    return R;
  });
  getAdvisor()->onSuccessfulInlining(*this, /*CalleeWasDeleted*/ true);
}

void MLInlineAdvice::recordUnsuccessfulInliningImpl(
    const InlineResult &Result) {
  ORE.emit([&]() {
    OptimizationRemarkMissed R(DEBUG_TYPE, "InliningAttemptedAndUnsuccessful",
                               DLoc, Block);
    reportContextForRemark(R);
    return R;
  });
}
void MLInlineAdvice::recordUnattemptedInliningImpl() {
  ORE.emit([&]() {
    OptimizationRemarkMissed R(DEBUG_TYPE, "IniningNotAttempted", DLoc, Block);
    reportContextForRemark(R);
    return R;
  });
}
