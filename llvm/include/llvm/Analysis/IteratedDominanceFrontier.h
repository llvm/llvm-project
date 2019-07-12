//===- IteratedDominanceFrontier.h - Calculate IDF --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_IDF_H
#define LLVM_ANALYSIS_IDF_H

#include "llvm/IR/CFGDiff.h"
#include "llvm/Support/GenericIteratedDominanceFrontier.h"

namespace llvm {

class BasicBlock;

namespace IDFCalculatorDetail {

/// Specialization for BasicBlock for the optional use of GraphDiff.
template <bool IsPostDom> struct ChildrenGetterTy<BasicBlock, IsPostDom> {
  using NodeRef = BasicBlock *;
  using ChildrenTy = SmallVector<BasicBlock *, 8>;

  ChildrenGetterTy() = default;
  ChildrenGetterTy(const GraphDiff<BasicBlock *, IsPostDom> *GD) : GD(GD) {
    assert(GD);
  }

  ChildrenTy get(const NodeRef &N);

  const GraphDiff<BasicBlock *, IsPostDom> *GD = nullptr;
};

} // end of namespace IDFCalculatorDetail

template <bool IsPostDom>
class IDFCalculator final : public IDFCalculatorBase<BasicBlock, IsPostDom> {
public:
  using IDFCalculatorBase =
      typename llvm::IDFCalculatorBase<BasicBlock, IsPostDom>;
  using ChildrenGetterTy = typename IDFCalculatorBase::ChildrenGetterTy;

  IDFCalculator(DominatorTreeBase<BasicBlock, IsPostDom> &DT)
      : IDFCalculatorBase(DT) {}

  IDFCalculator(DominatorTreeBase<BasicBlock, IsPostDom> &DT,
                const GraphDiff<BasicBlock *, IsPostDom> *GD)
      : IDFCalculatorBase(DT, ChildrenGetterTy(GD)) {
    assert(GD);
  }
};

using ForwardIDFCalculator = IDFCalculator<false>;
using ReverseIDFCalculator = IDFCalculator<true>;

//===----------------------------------------------------------------------===//
// Implementation.
//===----------------------------------------------------------------------===//

namespace IDFCalculatorDetail {

template <bool IsPostDom>
typename ChildrenGetterTy<BasicBlock, IsPostDom>::ChildrenTy
ChildrenGetterTy<BasicBlock, IsPostDom>::get(const NodeRef &N) {

  using OrderedNodeTy =
      typename IDFCalculatorBase<BasicBlock, IsPostDom>::OrderedNodeTy;

  if (!GD) {
    auto Children = children<OrderedNodeTy>(N);
    return {Children.begin(), Children.end()};
  }

  using SnapShotBBPairTy =
      std::pair<const GraphDiff<BasicBlock *, IsPostDom> *, OrderedNodeTy>;

  ChildrenTy Ret;
  for (const auto &SnapShotBBPair : children<SnapShotBBPairTy>({GD, N}))
    Ret.emplace_back(SnapShotBBPair.second);
  return Ret;
}

} // end of namespace IDFCalculatorDetail

} // end of namespace llvm

#endif
