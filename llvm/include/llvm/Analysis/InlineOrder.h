//===- InlineOrder.h - Inlining order abstraction -*- C++ ---*-------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
#ifndef LLVM_ANALYSIS_INLINEORDER_H
#define LLVM_ANALYSIS_INLINEORDER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/InstrTypes.h"
#include <algorithm>
#include <utility>

namespace llvm {
class CallBase;
class Function;

template <typename T> class InlineOrder {
public:
  using reference = T &;
  using const_reference = const T &;

  virtual ~InlineOrder() = default;

  virtual size_t size() = 0;

  virtual void push(const T &Elt) = 0;

  virtual T pop() = 0;

  virtual const_reference front() = 0;

  virtual void erase_if(function_ref<bool(T)> Pred) = 0;

  bool empty() { return !size(); }
};

template <typename T, typename Container = SmallVector<T, 16>>
class DefaultInlineOrder : public InlineOrder<T> {
  using reference = T &;
  using const_reference = const T &;

public:
  size_t size() override { return Calls.size() - FirstIndex; }

  void push(const T &Elt) override { Calls.push_back(Elt); }

  T pop() override {
    assert(size() > 0);
    return Calls[FirstIndex++];
  }

  const_reference front() override {
    assert(size() > 0);
    return Calls[FirstIndex];
  }

  void erase_if(function_ref<bool(T)> Pred) override {
    Calls.erase(std::remove_if(Calls.begin() + FirstIndex, Calls.end(), Pred),
                Calls.end());
  }

private:
  Container Calls;
  size_t FirstIndex = 0;
};

class InlinePriority {
public:
  virtual bool hasLowerPriority(const CallBase *L, const CallBase *R) const = 0;
  virtual void update(const CallBase *CB) = 0;
  virtual bool updateAndCheckDecreased(const CallBase *CB) = 0;
};

class SizePriority : public InlinePriority {
  using PriorityT = unsigned;
  DenseMap<const CallBase *, PriorityT> Priorities;

  static PriorityT evaluate(const CallBase *CB) {
    Function *Callee = CB->getCalledFunction();
    return Callee->getInstructionCount();
  }

  static bool isMoreDesirable(const PriorityT &P1, const PriorityT &P2) {
    return P1 < P2;
  }

  bool hasLowerPriority(const CallBase *L, const CallBase *R) const override {
    const auto I1 = Priorities.find(L);
    const auto I2 = Priorities.find(R);
    assert(I1 != Priorities.end() && I2 != Priorities.end());
    return isMoreDesirable(I2->second, I1->second);
  }

public:
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

  const_reference front() override {
    assert(size() > 0);
    adjust();

    CallBase *CB = Heap.front();
    return *InlineHistoryMap.find(CB);
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
} // namespace llvm
#endif // LLVM_ANALYSIS_INLINEORDER_H
