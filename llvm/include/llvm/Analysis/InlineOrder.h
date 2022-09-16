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
#include "llvm/Analysis/InlineCost.h"
#include "llvm/IR/InstrTypes.h"
#include <algorithm>
#include <utility>

namespace llvm {
class CallBase;
class Function;

enum class InlinePriorityMode : int { NoPriority, Size, Cost, OptRatio };

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

std::unique_ptr<InlineOrder<std::pair<CallBase *, int>>>
getInlineOrder(InlinePriorityMode UseInlinePriority,
               FunctionAnalysisManager &FAM, const InlineParams &Params);

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

} // namespace llvm
#endif // LLVM_ANALYSIS_INLINEORDER_H
