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

using namespace mlir;

/// Sets up the metadata and links the node to its parent.
ReductionNode::ReductionNode(ModuleOp module, ReductionNode *parent)
    : module(module), evaluated(false) {

  if (parent != nullptr)
    parent->linkVariant(this);
}

ReductionNode::ReductionNode(ModuleOp module, ReductionNode *parent,
                             std::vector<bool> transformSpace)
    : module(module), evaluated(false), transformSpace(transformSpace) {

  if (parent != nullptr)
    parent->linkVariant(this);
}

/// Calculates and updates the size and interesting values of the module.
void ReductionNode::measureAndTest(const Tester &test) {
  SmallString<128> filepath;
  int fd;

  // Print module to temporary file.
  std::error_code ec =
      llvm::sys::fs::createTemporaryFile("mlir-reduce", "mlir", fd, filepath);

  if (ec)
    llvm::report_fatal_error("Error making unique filename: " + ec.message());

  llvm::ToolOutputFile out(filepath, fd);
  module.print(out.os());
  out.os().close();

  if (out.os().has_error())
    llvm::report_fatal_error("Error emitting bitcode to file '" + filepath);

  size = out.os().tell();
  interesting = test.isInteresting(filepath);
  evaluated = true;
}

/// Returns true if the size and interestingness have been calculated.
bool ReductionNode::isEvaluated() const { return evaluated; }

/// Returns the size in bytes of the module.
int ReductionNode::getSize() const { return size; }

/// Returns true if the module exhibits the interesting behavior.
bool ReductionNode::isInteresting() const { return interesting; }

/// Returns the pointers to the child variants.
ReductionNode *ReductionNode::getVariant(unsigned long index) const {
  if (index < variants.size())
    return variants[index].get();

  return nullptr;
}

/// Returns the number of child variants.
int ReductionNode::variantsSize() const { return variants.size(); }

/// Returns true if the child variants vector is empty.
bool ReductionNode::variantsEmpty() const { return variants.empty(); }

/// Link a child variant node.
void ReductionNode::linkVariant(ReductionNode *newVariant) {
  std::unique_ptr<ReductionNode> ptrVariant(newVariant);
  variants.push_back(std::move(ptrVariant));
}

/// Sort the child variants and remove the uninteresting ones.
void ReductionNode::organizeVariants(const Tester &test) {
  // Ensure all variants are evaluated.
  for (auto &var : variants)
    if (!var->isEvaluated())
      var->measureAndTest(test);

  // Sort variants by interestingness and size.
  llvm::array_pod_sort(
      variants.begin(), variants.end(), [](const auto *lhs, const auto *rhs) {
        if (lhs->get()->isInteresting() && !rhs->get()->isInteresting())
          return 0;

        if (!lhs->get()->isInteresting() && rhs->get()->isInteresting())
          return 1;

        return (lhs->get()->getSize(), rhs->get()->getSize());
      });

  int interestingCount = 0;
  for (auto &var : variants) {
    if (var->isInteresting()) {
      ++interestingCount;
    } else {
      break;
    }
  }

  // Remove uninteresting variants.
  variants.resize(interestingCount);
}

/// Returns the number of non transformed indices.
int ReductionNode::transformSpaceSize() {
  return std::count(transformSpace.begin(), transformSpace.end(), false);
}

/// Returns a vector of the transformable indices in the Module.
const std::vector<bool> ReductionNode::getTransformSpace() {
  return transformSpace;
}
