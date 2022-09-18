//===- InlineOrder.cpp - Inlining order abstraction -*- C++ ---*-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/InlineOrder.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/InlineAdvisor.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "inline-order"

enum class InlinePriorityMode : int { Size, Cost, OptRatio };

static cl::opt<InlinePriorityMode> UseInlinePriority(
    "inline-priority-mode", cl::init(InlinePriorityMode::Size), cl::Hidden,
    cl::desc("Choose the priority mode to use in module inline"),
    cl::values(clEnumValN(InlinePriorityMode::Size, "size",
                          "Use callee size priority."),
               clEnumValN(InlinePriorityMode::Cost, "cost",
                          "Use inline cost priority.")));

namespace {

llvm::InlineCost getInlineCostWrapper(CallBase &CB,
                                      FunctionAnalysisManager &FAM,
                                      const InlineParams &Params) {
  Function &Caller = *CB.getCaller();
  ProfileSummaryInfo *PSI =
      FAM.getResult<ModuleAnalysisManagerFunctionProxy>(Caller)
          .getCachedResult<ProfileSummaryAnalysis>(
              *CB.getParent()->getParent()->getParent());

  auto &ORE = FAM.getResult<OptimizationRemarkEmitterAnalysis>(Caller);
  auto GetAssumptionCache = [&](Function &F) -> AssumptionCache & {
    return FAM.getResult<AssumptionAnalysis>(F);
  };
  auto GetBFI = [&](Function &F) -> BlockFrequencyInfo & {
    return FAM.getResult<BlockFrequencyAnalysis>(F);
  };
  auto GetTLI = [&](Function &F) -> const TargetLibraryInfo & {
    return FAM.getResult<TargetLibraryAnalysis>(F);
  };

  Function &Callee = *CB.getCalledFunction();
  auto &CalleeTTI = FAM.getResult<TargetIRAnalysis>(Callee);
  bool RemarksEnabled =
      Callee.getContext().getDiagHandlerPtr()->isMissedOptRemarkEnabled(
          DEBUG_TYPE);
  return getInlineCost(CB, Params, CalleeTTI, GetAssumptionCache, GetTLI,
                       GetBFI, PSI, RemarksEnabled ? &ORE : nullptr);
}

class InlinePriority {
public:
  virtual ~InlinePriority() = default;
  virtual bool hasLowerPriority(const CallBase *L, const CallBase *R) const = 0;
  virtual void update(const CallBase *CB) = 0;
  virtual bool updateAndCheckDecreased(const CallBase *CB) = 0;
};

class SizePriority : public InlinePriority {
  using PriorityT = unsigned;
  DenseMap<const CallBase *, PriorityT> Priorities;

  PriorityT evaluate(const CallBase *CB) {
    Function *Callee = CB->getCalledFunction();
    return Callee->getInstructionCount();
  }

  bool isMoreDesirable(const PriorityT &P1, const PriorityT &P2) const {
    return P1 < P2;
  }

public:
  bool hasLowerPriority(const CallBase *L, const CallBase *R) const override {
    const auto I1 = Priorities.find(L);
    const auto I2 = Priorities.find(R);
    assert(I1 != Priorities.end() && I2 != Priorities.end());
    return isMoreDesirable(I2->second, I1->second);
  }

  // Update the priority associated with CB.
  void update(const CallBase *CB) override { Priorities[CB] = evaluate(CB); };

  bool updateAndCheckDecreased(const CallBase *CB) override {
    auto It = Priorities.find(CB);
    const auto OldPriority = It->second;
    It->second = evaluate(CB);
    const auto NewPriority = It->second;
    return isMoreDesirable(OldPriority, NewPriority);
  }
};

class CostPriority : public InlinePriority {
  using PriorityT = int;
  DenseMap<const CallBase *, PriorityT> Priorities;
  std::function<InlineCost(const CallBase *)> getInlineCost;

  PriorityT evaluate(const CallBase *CB) {
    auto IC = getInlineCost(CB);
    int cost = 0;
    if (IC.isVariable())
      cost = IC.getCost();
    else
      cost = IC.isNever() ? INT_MAX : INT_MIN;
    return cost;
  }

  bool isMoreDesirable(const PriorityT &P1, const PriorityT &P2) const {
    return P1 < P2;
  }

public:
  CostPriority() = delete;
  CostPriority(std::function<InlineCost(const CallBase *)> getInlineCost)
      : getInlineCost(getInlineCost){};

  bool hasLowerPriority(const CallBase *L, const CallBase *R) const override {
    const auto I1 = Priorities.find(L);
    const auto I2 = Priorities.find(R);
    assert(I1 != Priorities.end() && I2 != Priorities.end());
    return isMoreDesirable(I2->second, I1->second);
  }

  // Update the priority associated with CB.
  void update(const CallBase *CB) override { Priorities[CB] = evaluate(CB); };

  bool updateAndCheckDecreased(const CallBase *CB) override {
    auto It = Priorities.find(CB);
    const auto OldPriority = It->second;
    It->second = evaluate(CB);
    const auto NewPriority = It->second;
    return isMoreDesirable(OldPriority, NewPriority);
  }
};

class PriorityInlineOrder : public InlineOrder<std::pair<CallBase *, int>> {
  using T = std::pair<CallBase *, int>;
  using reference = T &;
  using const_reference = const T &;

  // A call site could become less desirable for inlining because of the size
  // growth from prior inlining into the callee. This method is used to lazily
  // update the desirability of a call site if it's decreasing. It is only
  // called on pop() or front(), not every time the desirability changes. When
  // the desirability of the front call site decreases, an updated one would be
  // pushed right back into the heap. For simplicity, those cases where
  // the desirability of a call site increases are ignored here.
  void adjust() {
    while (PriorityPtr->updateAndCheckDecreased(Heap.front())) {
      std::pop_heap(Heap.begin(), Heap.end(), isLess);
      std::push_heap(Heap.begin(), Heap.end(), isLess);
    }
  }

public:
  PriorityInlineOrder(std::unique_ptr<InlinePriority> PriorityPtr)
      : PriorityPtr(std::move(PriorityPtr)) {
    isLess = [this](const CallBase *L, const CallBase *R) {
      return this->PriorityPtr->hasLowerPriority(L, R);
    };
  }

  size_t size() override { return Heap.size(); }

  void push(const T &Elt) override {
    CallBase *CB = Elt.first;
    const int InlineHistoryID = Elt.second;

    Heap.push_back(CB);
    PriorityPtr->update(CB);
    std::push_heap(Heap.begin(), Heap.end(), isLess);
    InlineHistoryMap[CB] = InlineHistoryID;
  }

  T pop() override {
    assert(size() > 0);
    adjust();

    CallBase *CB = Heap.front();
    T Result = std::make_pair(CB, InlineHistoryMap[CB]);
    InlineHistoryMap.erase(CB);
    std::pop_heap(Heap.begin(), Heap.end(), isLess);
    Heap.pop_back();
    return Result;
  }

  void erase_if(function_ref<bool(T)> Pred) override {
    auto PredWrapper = [=](CallBase *CB) -> bool {
      return Pred(std::make_pair(CB, 0));
    };
    llvm::erase_if(Heap, PredWrapper);
    std::make_heap(Heap.begin(), Heap.end(), isLess);
  }

private:
  SmallVector<CallBase *, 16> Heap;
  std::function<bool(const CallBase *L, const CallBase *R)> isLess;
  DenseMap<CallBase *, int> InlineHistoryMap;
  std::unique_ptr<InlinePriority> PriorityPtr;
};

} // namespace

std::unique_ptr<InlineOrder<std::pair<CallBase *, int>>>
llvm::getInlineOrder(FunctionAnalysisManager &FAM, const InlineParams &Params) {
  switch (UseInlinePriority) {
  case InlinePriorityMode::Size:
    LLVM_DEBUG(dbgs() << "    Current used priority: Size priority ---- \n");
    return std::make_unique<PriorityInlineOrder>(
        std::make_unique<SizePriority>());

  case InlinePriorityMode::Cost:
    LLVM_DEBUG(dbgs() << "    Current used priority: Cost priority ---- \n");
    return std::make_unique<PriorityInlineOrder>(
        std::make_unique<CostPriority>([&](const CallBase *CB) -> InlineCost {
          return getInlineCostWrapper(const_cast<CallBase &>(*CB), FAM, Params);
        }));

  default:
    llvm_unreachable("Unsupported Inline Priority Mode");
    break;
  }
  return nullptr;
}
