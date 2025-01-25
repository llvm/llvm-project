//===- ExtraMatchers.h - Various common matchers ---------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides extra matchers that are very useful for mlir-query
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_EXTRAMATCHERS_H
#define MLIR_IR_EXTRAMATCHERS_H

#include "MatchFinder.h"
#include "MatchersInternal.h"
#include "mlir/IR/Region.h"
#include "mlir/Query/Query.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

namespace query {

namespace extramatcher {

namespace detail {

class BackwardSliceMatcher {
public:
  BackwardSliceMatcher(matcher::DynMatcher &&innerMatcher, unsigned hops)
      : innerMatcher(std::move(innerMatcher)), hops(hops) {}

private:
  bool matches(Operation *op, SetVector<Operation *> &backwardSlice,
               QueryOptions &options, unsigned tempHops) {

    if (op->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
      return false;
    }

    auto processValue = [&](Value value) {
      if (tempHops == 0) {
        return;
      }
      if (auto *definingOp = value.getDefiningOp()) {
        if (backwardSlice.count(definingOp) == 0)
          matches(definingOp, backwardSlice, options, tempHops - 1);
      } else if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        if (options.omitBlockArguments)
          return;
        Block *block = blockArg.getOwner();

        Operation *parentOp = block->getParentOp();

        if (parentOp && backwardSlice.count(parentOp) == 0) {
          if (parentOp->getNumRegions() != 1 &&
              parentOp->getRegion(0).getBlocks().size() != 1) {
            llvm::errs()
                << "Error: Expected parentOp to have exactly one region and "
                << "exactly one block, but found " << parentOp->getNumRegions()
                << " regions and "
                << (parentOp->getRegion(0).getBlocks().size()) << " blocks.\n";
          };
          matches(parentOp, backwardSlice, options, tempHops - 1);
        }
      } else {
        llvm::errs() << "No definingOp and not a block argument\n";
        return;
      }
    };

    if (!options.omitUsesFromAbove) {
      llvm::for_each(op->getRegions(), [&](Region &region) {
        SmallPtrSet<Region *, 4> descendents;
        region.walk(
            [&](Region *childRegion) { descendents.insert(childRegion); });
        region.walk([&](Operation *op) {
          for (OpOperand &operand : op->getOpOperands()) {
            if (!descendents.contains(operand.get().getParentRegion()))
              processValue(operand.get());
          }
        });
      });
    }

    llvm::for_each(op->getOperands(), processValue);
    backwardSlice.insert(op);
    return true;
  }

public:
  bool match(Operation *op, SetVector<Operation *> &backwardSlice,
             QueryOptions &options) {

    if (innerMatcher.match(op) && matches(op, backwardSlice, options, hops)) {
      if (!options.inclusive) {
        backwardSlice.remove(op);
      }
      return true;
    }
    return false;
  }

private:
  matcher::DynMatcher innerMatcher;
  unsigned hops;
};

class ForwardSliceMatcher {
public:
  ForwardSliceMatcher(matcher::DynMatcher &&innerMatcher, unsigned hops)
      : innerMatcher(std::move(innerMatcher)), hops(hops) {}

private:
  bool matches(Operation *op, SetVector<Operation *> &forwardSlice,
               QueryOptions &options, unsigned tempHops) {

    if (tempHops == 0) {
      forwardSlice.insert(op);
      return true;
    }

    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (Operation &blockOp : block)
          if (forwardSlice.count(&blockOp) == 0)
            matches(&blockOp, forwardSlice, options, tempHops - 1);
    for (Value result : op->getResults()) {
      for (Operation *userOp : result.getUsers())
        if (forwardSlice.count(userOp) == 0)
          matches(userOp, forwardSlice, options, tempHops - 1);
    }

    forwardSlice.insert(op);
    return true;
  }

public:
  bool match(Operation *op, SetVector<Operation *> &forwardSlice,
             QueryOptions &options) {
    if (innerMatcher.match(op) && matches(op, forwardSlice, options, hops)) {
      if (!options.inclusive) {
        forwardSlice.remove(op);
      }
      SmallVector<Operation *, 0> v(forwardSlice.takeVector());
      forwardSlice.insert(v.rbegin(), v.rend());
      return true;
    }
    return false;
  }

private:
  matcher::DynMatcher innerMatcher;
  unsigned hops;
};

} // namespace detail

inline detail::BackwardSliceMatcher
definedBy(mlir::query::matcher::DynMatcher innerMatcher) {
  return detail::BackwardSliceMatcher(std::move(innerMatcher), 1);
}

inline detail::BackwardSliceMatcher
getDefinitions(mlir::query::matcher::DynMatcher innerMatcher, unsigned hops) {
  return detail::BackwardSliceMatcher(std::move(innerMatcher), hops);
}

inline detail::ForwardSliceMatcher
usedBy(mlir::query::matcher::DynMatcher innerMatcher) {
  return detail::ForwardSliceMatcher(std::move(innerMatcher), 1);
}

inline detail::ForwardSliceMatcher
getUses(mlir::query::matcher::DynMatcher innerMatcher, unsigned hops) {
  return detail::ForwardSliceMatcher(std::move(innerMatcher), hops);
}

} // namespace extramatcher

} // namespace query

} // namespace mlir

#endif // MLIR_IR_EXTRAMATCHERS_H
