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

class InlineSizePriority {
public:
  InlineSizePriority(int Size) : Size(Size) {}

  static bool isMoreDesirable(const InlineSizePriority &S1,
                              const InlineSizePriority &S2) {
    return S1.Size < S2.Size;
  }

  static InlineSizePriority evaluate(CallBase *CB) {
    Function *Callee = CB->getCalledFunction();
    return InlineSizePriority(Callee->getInstructionCount());
  }

  int Size;
};

template <typename PriorityT>
class PriorityInlineOrder : public InlineOrder<std::pair<CallBase *, int>> {
  using T = std::pair<CallBase *, int>;
  using HeapT = std::pair<CallBase *, PriorityT>;
  using reference = T &;
  using const_reference = const T &;

  static bool cmp(const HeapT &P1, const HeapT &P2) {
    return PriorityT::isMoreDesirable(P2.second, P1.second);
  }

  // A call site could become less desirable for inlining because of the size
  // growth from prior inlining into the callee. This method is used to lazily
  // update the desirability of a call site if it's decreasing. It is only
  // called on pop() or front(), not every time the desirability changes. When
  // the desirability of the front call site decreases, an updated one would be
  // pushed right back into the heap. For simplicity, those cases where
  // the desirability of a call site increases are ignored here.
  void adjust() {
    bool Changed = false;
    do {
      CallBase *CB = Heap.front().first;
      const PriorityT PreviousGoodness = Heap.front().second;
      const PriorityT CurrentGoodness = PriorityT::evaluate(CB);
      Changed = PriorityT::isMoreDesirable(PreviousGoodness, CurrentGoodness);
      if (Changed) {
        std::pop_heap(Heap.begin(), Heap.end(), cmp);
        Heap.pop_back();
        Heap.push_back({CB, CurrentGoodness});
        std::push_heap(Heap.begin(), Heap.end(), cmp);
      }
    } while (Changed);
  }

public:
  size_t size() override { return Heap.size(); }

  void push(const T &Elt) override {
    CallBase *CB = Elt.first;
    const int InlineHistoryID = Elt.second;
    const PriorityT Goodness = PriorityT::evaluate(CB);

    Heap.push_back({CB, Goodness});
    std::push_heap(Heap.begin(), Heap.end(), cmp);
    InlineHistoryMap[CB] = InlineHistoryID;
  }

  T pop() override {
    assert(size() > 0);
    adjust();

    CallBase *CB = Heap.front().first;
    T Result = std::make_pair(CB, InlineHistoryMap[CB]);
    InlineHistoryMap.erase(CB);
    std::pop_heap(Heap.begin(), Heap.end(), cmp);
    Heap.pop_back();
    return Result;
  }

  const_reference front() override {
    assert(size() > 0);
    adjust();

    CallBase *CB = Heap.front().first;
    return *InlineHistoryMap.find(CB);
  }

  void erase_if(function_ref<bool(T)> Pred) override {
    auto PredWrapper = [=](HeapT P) -> bool {
      return Pred(std::make_pair(P.first, 0));
    };
    llvm::erase_if(Heap, PredWrapper);
    std::make_heap(Heap.begin(), Heap.end(), cmp);
  }

private:
  SmallVector<HeapT, 16> Heap;
  DenseMap<CallBase *, int> InlineHistoryMap;
};
} // namespace llvm
#endif // LLVM_ANALYSIS_INLINEORDER_H
