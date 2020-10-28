//===- ReductionNode.h - Reduction Node Implementation ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the reduction nodes which are used to track of the metadata
// for a specific generated variant within a reduction pass and are the building
// blocks of the reduction tree structure. A reduction tree is used to keep
// track of the different generated variants throughout a reduction pass in the
// MLIR Reduce tool.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_REDUCER_REDUCTIONNODE_H
#define MLIR_REDUCER_REDUCTIONNODE_H

#include <vector>

#include "mlir/Reducer/Tester.h"
#include "llvm/Support/ToolOutputFile.h"

namespace mlir {

/// This class defines the ReductionNode which is used to wrap the module of
/// a generated variant and keep track of the necessary metadata for the
/// reduction pass. The nodes are linked together in a reduction tree stucture
/// which defines the relationship between all the different generated variants.
class ReductionNode {
public:
  ReductionNode(ModuleOp module, ReductionNode *parent);

  ReductionNode(ModuleOp module, ReductionNode *parent,
                std::vector<bool> transformSpace);

  /// Calculates and initializes the size and interesting values of the node.
  void measureAndTest(const Tester &test);

  /// Returns the module.
  ModuleOp getModule() const { return module; }

  /// Returns true if the size and interestingness have been calculated.
  bool isEvaluated() const;

  /// Returns the size in bytes of the module.
  int getSize() const;

  /// Returns true if the module exhibits the interesting behavior.
  bool isInteresting() const;

  /// Returns the pointer to a child variant by index.
  ReductionNode *getVariant(unsigned long index) const;

  /// Returns the number of child variants.
  int variantsSize() const;

  /// Returns true if the vector containing the child variants is empty.
  bool variantsEmpty() const;

  /// Sort the child variants and remove the uninteresting ones.
  void organizeVariants(const Tester &test);

  /// Returns the number of child variants.
  int transformSpaceSize();

  /// Returns a vector indicating the transformed indices as true.
  const std::vector<bool> getTransformSpace();

private:
  /// Link a child variant node.
  void linkVariant(ReductionNode *newVariant);

  // This is the MLIR module of this variant.
  ModuleOp module;

  // This is true if the module has been evaluated and it exhibits the
  // interesting behavior.
  bool interesting;

  // This indicates the number of characters in the printed module if the module
  // has been evaluated.
  int size;

  // This indicates if the module has been evalueated (measured and tested).
  bool evaluated;

  // Indicates the indices in the node that have been transformed in previous
  // levels of the reduction tree.
  std::vector<bool> transformSpace;

  // This points to the child variants that were created using this node as a
  // starting point.
  std::vector<std::unique_ptr<ReductionNode>> variants;
};

} // end namespace mlir

#endif
