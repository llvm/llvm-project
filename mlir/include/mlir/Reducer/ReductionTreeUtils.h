//===- ReductionTreeUtils.h - Reduction Tree utilities ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Reduction Tree Utilities. It defines pass independent
// methods that help in the reduction passes of the MLIR Reduce tool.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_REDUCER_REDUCTIONTREEUTILS_H
#define MLIR_REDUCER_REDUCTIONTREEUTILS_H

#include <tuple>

#include "PassDetail.h"
#include "ReductionNode.h"
#include "mlir/Reducer/Tester.h"
#include "llvm/Support/Debug.h"

namespace mlir {

// Defines the utilities for the implementation of custom reduction
// passes using the ReductionTreePass framework.
namespace ReductionTreeUtils {

/// Update the golden module's content with that of the reduced module.
void updateGoldenModule(ModuleOp &golden, ModuleOp reduced);

/// Update the the smallest node traversed so far in the reduction tree and
/// print the debugging information for the currNode being traversed.
void updateSmallestNode(ReductionNode *currNode, ReductionNode *&smallestNode,
                        std::vector<int> path);

/// Create a transform space index vector based on the specified number of
/// indices.
std::vector<bool> createTransformSpace(ModuleOp module, int numIndices);

/// Create the specified number of variants by applying the transform method
/// to different ranges of indices in the parent module. The isDeletion bolean
/// specifies if the transformation is the deletion of indices.
void createVariants(ReductionNode *parent, const Tester &test, int numVariants,
                    llvm::function_ref<void(ModuleOp, int, int)> transform,
                    bool isDeletion);

} // namespace ReductionTreeUtils

} // end namespace mlir

#endif
