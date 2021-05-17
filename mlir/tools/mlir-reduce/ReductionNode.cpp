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
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <limits>

using namespace mlir;

ReductionNode::ReductionNode(
    ReductionNode *parent, std::vector<Range> ranges,
    llvm::SpecificBumpPtrAllocator<ReductionNode> &allocator)
    : size(std::numeric_limits<size_t>::max()),
      interesting(Tester::Interestingness::Untested),
      /// Root node will have the parent pointer point to themselves.
      parent(parent == nullptr ? this : parent), ranges(ranges),
      allocator(allocator) {}

/// Returns the size in bytes of the module.
size_t ReductionNode::getSize() const { return size; }

ReductionNode *ReductionNode::getParent() const { return parent; }

/// Returns true if the module exhibits the interesting behavior.
Tester::Interestingness ReductionNode::isInteresting() const {
  return interesting;
}

std::vector<ReductionNode::Range> ReductionNode::getRanges() const {
  return ranges;
}

std::vector<ReductionNode *> &ReductionNode::getVariants() { return variants; }

#include <iostream>

/// If we haven't explored any variants from this node, we will create N
/// variants, N is the length of `ranges` if N > 1. Otherwise, we will split the
/// max element in `ranges` and create 2 new variants for each call.
std::vector<ReductionNode *> ReductionNode::generateNewVariants() {
  std::vector<ReductionNode *> newNodes;

  // If we haven't created new variant, then we can create varients by removing
  // each of them respectively. For example, given {{1, 3}, {4, 9}}, we can
  // produce variants with range {{1, 3}} and {{4, 9}}.
  if (variants.size() == 0 && ranges.size() != 1) {
    for (const Range &range : ranges) {
      std::vector<Range> subRanges = ranges;
      llvm::erase_value(subRanges, range);
      ReductionNode *newNode = allocator.Allocate();
      new (newNode) ReductionNode(this, subRanges, allocator);
      newNodes.push_back(newNode);
      variants.push_back(newNode);
    }

    return newNodes;
  }

  // At here, we have created the type of variants mentioned above. We would
  // like to split the max range into 2 to create 2 new variants. Continue on
  // the above example, we split the range {4, 9} into {4, 6}, {6, 9}, and
  // create two variants with range {{1, 3}, {4, 6}} and {{1, 3}, {6, 9}}. The
  // result ranges vector will be {{1, 3}, {4, 6}, {6, 9}}.
  auto maxElement = std::max_element(
      ranges.begin(), ranges.end(), [](const Range &lhs, const Range &rhs) {
        return (lhs.second - lhs.first) > (rhs.second - rhs.first);
      });

  // We can't split range with lenght 1, which means we can't produce new
  // variant.
  if (maxElement->second - maxElement->first == 1)
    return {};

  auto createNewNode = [this](const std::vector<Range> &ranges) {
    ReductionNode *newNode = allocator.Allocate();
    new (newNode) ReductionNode(this, ranges, allocator);
    return newNode;
  };

  Range maxRange = *maxElement;
  std::vector<Range> subRanges = ranges;
  auto subRangesIter = subRanges.begin() + (maxElement - ranges.begin());
  int half = (maxRange.first + maxRange.second) / 2;
  *subRangesIter = std::make_pair(maxRange.first, half);
  newNodes.push_back(createNewNode(subRanges));
  *subRangesIter = std::make_pair(half, maxRange.second);
  newNodes.push_back(createNewNode(subRanges));

  variants.insert(variants.end(), newNodes.begin(), newNodes.end());
  auto it = ranges.insert(maxElement, std::make_pair(half, maxRange.second));
  it = ranges.insert(it, std::make_pair(maxRange.first, half));
  // Remove the range that has been split.
  ranges.erase(it + 2);

  return newNodes;
}

void ReductionNode::update(std::pair<Tester::Interestingness, size_t> result) {
  std::tie(interesting, size) = result;
}

std::vector<ReductionNode *>
ReductionNode::iterator<SinglePath>::getNeighbors(ReductionNode *node) {
  // Single Path: Traverses the smallest successful variant at each level until
  // no new successful variants can be created at that level.
  llvm::ArrayRef<ReductionNode *> variantsFromParent =
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

  if (smallest != nullptr) {
    // We got a smallest one, keep traversing from this node.
    node = smallest;
  } else {
    // None of these variants is interesting, let the parent node to generate
    // more variants.
    node = node->getParent();
  }

  return node->generateNewVariants();
}
