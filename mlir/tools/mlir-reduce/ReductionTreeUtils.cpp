//===- ReductionTreeUtils.cpp - Reduction Tree Utilities ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Reduction Tree Utilities. It defines pass independent
// methods that help in a reduction pass of the MLIR Reduce tool.
//
//===----------------------------------------------------------------------===//

#include "mlir/Reducer/ReductionTreeUtils.h"

#define DEBUG_TYPE "mlir-reduce"

using namespace mlir;

/// Update the golden module's content with that of the reduced module.
void ReductionTreeUtils::updateGoldenModule(ModuleOp &golden,
                                            ModuleOp reduced) {
  golden.getBody()->clear();

  golden.getBody()->getOperations().splice(golden.getBody()->begin(),
                                           reduced.getBody()->getOperations());
}

/// Update the smallest node traversed so far in the reduction tree and
/// print the debugging information for the currNode being traversed.
void ReductionTreeUtils::updateSmallestNode(ReductionNode *currNode,
                                            ReductionNode *&smallestNode,
                                            std::vector<int> path) {
  LLVM_DEBUG(llvm::dbgs() << "\nTree Path: root");
  #ifndef NDEBUG
  for (int nodeIndex : path)
    LLVM_DEBUG(llvm::dbgs() << " -> " << nodeIndex);
  #endif

  LLVM_DEBUG(llvm::dbgs() << "\nSize (chars): " << currNode->getSize());
  if (currNode->getSize() < smallestNode->getSize()) {
    LLVM_DEBUG(llvm::dbgs() << " - new smallest node!");
    smallestNode = currNode;
  }
}

/// Create a transform space index vector based on the specified number of
/// indices.
std::vector<bool> ReductionTreeUtils::createTransformSpace(ModuleOp module,
                                                           int numIndices) {
  std::vector<bool> transformSpace;
  for (int i = 0; i < numIndices; ++i)
    transformSpace.push_back(false);

  return transformSpace;
}

/// Translate section start and end into a vector of ranges specifying the
/// section in the non transformed indices in the transform space.
static std::vector<std::tuple<int, int>> getRanges(std::vector<bool> tSpace,
                                                   int start, int end) {
  std::vector<std::tuple<int, int>> ranges;
  int rangeStart = 0;
  int rangeEnd = 0;
  bool inside = false;
  int transformableCount = 0;

  for (auto element : llvm::enumerate(tSpace)) {
    int index = element.index();
    bool value = element.value();

    if (start <= transformableCount && transformableCount < end) {
      if (!value && !inside) {
        inside = true;
        rangeStart = index;
      }
      if (value && inside) {
        rangeEnd = index;
        ranges.push_back(std::make_tuple(rangeStart, rangeEnd));
        inside = false;
      }
    }

    if (!value)
      transformableCount++;

    if (transformableCount == end && inside) {
      ranges.push_back(std::make_tuple(rangeStart, index + 1));
      inside = false;
      break;
    }
  }

  return ranges;
}

/// Create the specified number of variants by applying the transform method
/// to different ranges of indices in the parent module. The isDeletion boolean
/// specifies if the transformation is the deletion of indices.
void ReductionTreeUtils::createVariants(
    ReductionNode *parent, const Tester &test, int numVariants,
    llvm::function_ref<void(ModuleOp, int, int)> transform, bool isDeletion) {
  std::vector<bool> newTSpace;
  ModuleOp module = parent->getModule();

  std::vector<bool> parentTSpace = parent->getTransformSpace();
  int indexCount = parent->transformSpaceSize();
  std::vector<std::tuple<int, int>> ranges;

  // No new variants can be created.
  if (indexCount == 0)
    return;

  // Create a single variant by transforming the unique index.
  if (indexCount == 1) {
    ModuleOp variantModule = module.clone();
    if (isDeletion) {
      transform(variantModule, 0, 1);
    } else {
      ranges = getRanges(parentTSpace, 0, parentTSpace.size());
      transform(variantModule, std::get<0>(ranges[0]), std::get<1>(ranges[0]));
    }

    new ReductionNode(variantModule, parent, newTSpace);

    return;
  }

  // Create the specified number of variants.
  for (int i = 0; i < numVariants; ++i) {
    ModuleOp variantModule = module.clone();
    newTSpace = parent->getTransformSpace();
    int sectionSize = indexCount / numVariants;
    int sectionStart = sectionSize * i;
    int sectionEnd = sectionSize * (i + 1);

    if (i == numVariants - 1)
      sectionEnd = indexCount;

    if (isDeletion)
      transform(variantModule, sectionStart, sectionEnd);

    ranges = getRanges(parentTSpace, sectionStart, sectionEnd);

    for (auto range : ranges) {
      int rangeStart = std::get<0>(range);
      int rangeEnd = std::get<1>(range);

      for (int x = rangeStart; x < rangeEnd; ++x)
        newTSpace[x] = true;

      if (!isDeletion)
        transform(variantModule, rangeStart, rangeEnd);
    }

    // Create Reduction Node in the Reduction tree
    new ReductionNode(variantModule, parent, newTSpace);
  }
}
