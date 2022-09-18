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

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Analysis/InlineCost.h"
#include <utility>

namespace llvm {
class CallBase;

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
getInlineOrder(FunctionAnalysisManager &FAM, const InlineParams &Params);

} // namespace llvm
#endif // LLVM_ANALYSIS_INLINEORDER_H
