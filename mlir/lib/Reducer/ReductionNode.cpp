//===- ReductionNode.cpp - Reduction Node Implementation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the reduction nodes which are used to track of the
// metadata for a specific generated variant within a reduction pass and are the
// building blocks of the reduction tree structure. A reduction tree is used to
// keep track of the different generated variants throughout a reduction pass in
// the MLIR Reduce tool.
//
//===----------------------------------------------------------------------===//

#include "mlir/Reducer/ReductionNode.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>

using namespace mlir;

ReductionNode::ReductionNode(
    ReductionNode *parentNode, std::vector<Range> ranges,
    llvm::SpecificBumpPtrAllocator<ReductionNode> &allocator)
    /// Root node will have the parent pointer point to themselves.
    : parent(parentNode == nullptr ? this : parentNode),
      size(std::numeric_limits<size_t>::max()),
      interesting(Tester::Interestingness::Untested), ranges(ranges),
      startRanges(ranges), allocator(allocator) {
  if (parent != this)
    if (failed(initialize(parent->getModule(), parent->getRegion())))
      llvm_unreachable("unexpected initialization failure");
}

LogicalResult ReductionNode::initialize(ModuleOp parentModule,
                                        Region &targetRegion) {
  // Use the mapper help us find the corresponding region after module clone.
  BlockAndValueMapping mapper;
  module = cast<ModuleOp>(parentModule->clone(mapper));
  // Use the first block of targetRegion to locate the cloned region.
  Block *block = mapper.lookup(&*targetRegion.begin());
  region = block->getParent();
  return success();
}

/// If we haven't explored any variants from this node, we will create N
/// variants, N is the length of `ranges` if N > 1. Otherwise, we will split the
/// max element in `ranges` and create 2 new variants for each call.
ArrayRef<ReductionNode *> ReductionNode::generateNewVariants() {
  int oldNumVariant = getVariants().size();

  auto createNewNode = [this](std::vector<Range> ranges) {
    return new (allocator.Allocate())
        ReductionNode(this, std::move(ranges), allocator);
  };

  // If we haven't created new variant, then we can create varients by removing
  // each of them respectively. For example, given {{1, 3}, {4, 9}}, we can
  // produce variants with range {{1, 3}} and {{4, 9}}.
  if (variants.size() == 0 && getRanges().size() > 1) {
    for (const Range &range : getRanges()) {
      std::vector<Range> subRanges = getRanges();
      llvm::erase_value(subRanges, range);
      variants.push_back(createNewNode(std::move(subRanges)));
    }

    return getVariants().drop_front(oldNumVariant);
  }

  // At here, we have created the type of variants mentioned above. We would
  // like to split the max range into 2 to create 2 new variants. Continue on
  // the above example, we split the range {4, 9} into {4, 6}, {6, 9}, and
  // create two variants with range {{1, 3}, {4, 6}} and {{1, 3}, {6, 9}}. The
  // final ranges vector will be {{1, 3}, {4, 6}, {6, 9}}.
  auto maxElement = std::max_element(
      ranges.begin(), ranges.end(), [](const Range &lhs, const Range &rhs) {
        return (lhs.second - lhs.first) > (rhs.second - rhs.first);
      });

  // The length of range is less than 1, we can't split it to create new
  // variant.
  if (maxElement->second - maxElement->first <= 1)
    return {};

  Range maxRange = *maxElement;
  std::vector<Range> subRanges = getRanges();
  auto subRangesIter = subRanges.begin() + (maxElement - ranges.begin());
  int half = (maxRange.first + maxRange.second) / 2;
  *subRangesIter = std::make_pair(maxRange.first, half);
  variants.push_back(createNewNode(subRanges));
  *subRangesIter = std::make_pair(half, maxRange.second);
  variants.push_back(createNewNode(std::move(subRanges)));

  auto it = ranges.insert(maxElement, std::make_pair(half, maxRange.second));
  it = ranges.insert(it, std::make_pair(maxRange.first, half));
  // Remove the range that has been split.
  ranges.erase(it + 2);

  return getVariants().drop_front(oldNumVariant);
}

void ReductionNode::update(std::pair<Tester::Interestingness, size_t> result) {
  std::tie(interesting, size) = result;
  // After applying reduction, the number of operation in the region may have
  // changed. Non-interesting case won't be explored thus it's safe to keep it
  // in a stale status.
  if (interesting == Tester::Interestingness::True) {
    // This module may has been updated. Reset the range.
    ranges.clear();
    ranges.push_back({0, std::distance(region->op_begin(), region->op_end())});
  }
}

ArrayRef<ReductionNode *>
ReductionNode::iterator<SinglePath>::getNeighbors(ReductionNode *node) {
  // Single Path: Traverses the smallest successful variant at each level until
  // no new successful variants can be created at that level.
  ArrayRef<ReductionNode *> variantsFromParent =
      node->getParent()->getVariants();

  // The parent node created several variants and they may be waiting for
  // examing interestingness. In Single Path approach, we will select the
  // smallest variant to continue our exploration. Thus we should wait until the
  // last variant to be examed then do the following traversal decision.
  if (!llvm::all_of(variantsFromParent, [](ReductionNode *node) {
        return node->isInteresting() != Tester::Interestingness::Untested;
      })) {
    return {};
  }

  ReductionNode *smallest = nullptr;
  for (ReductionNode *node : variantsFromParent) {
    if (node->isInteresting() != Tester::Interestingness::True)
      continue;
    if (smallest == nullptr || node->getSize() < smallest->getSize())
      smallest = node;
  }

  if (smallest != nullptr &&
      smallest->getSize() < node->getParent()->getSize()) {
    // We got a smallest one, keep traversing from this node.
    node = smallest;
  } else {
    // None of these variants is interesting, let the parent node to generate
    // more variants.
    node = node->getParent();
  }

  return node->generateNewVariants();
}
