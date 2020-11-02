//===- OpReducer.h - MLIR Reduce Operation Reducer ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the OpReducer class. It defines a variant generator method
// with the purpose of producing different variants by eliminating a
// parameterizable type of operations from the  parent module.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_REDUCER_PASSES_OPREDUCER_H
#define MLIR_REDUCER_PASSES_OPREDUCER_H

#include "mlir/IR/Region.h"
#include "mlir/Reducer/ReductionNode.h"
#include "mlir/Reducer/ReductionTreeUtils.h"
#include "mlir/Reducer/Tester.h"

namespace mlir {

class OpReducerImpl {
public:
  OpReducerImpl(
      llvm::function_ref<std::vector<Operation *>(ModuleOp)> getSpecificOps);

  /// Return the name of this reducer class.
  StringRef getName();

  /// Return the initial transformSpace containing the transformable indices.
  std::vector<bool> initTransformSpace(ModuleOp module);

  /// Generate variants by removing OpType operations from the module in the
  /// parent and link the variants as childs in the Reduction Tree Pass.
  void generateVariants(ReductionNode *parent, const Tester &test,
                        int numVariants);

  /// Generate variants by removing OpType operations from the module in the
  /// parent and link the variants as childs in the Reduction Tree Pass. The
  /// transform argument defines the function used to remove the OpTpye
  /// operations in range of indexed OpType operations.
  void generateVariants(ReductionNode *parent, const Tester &test,
                        int numVariants,
                        llvm::function_ref<void(ModuleOp, int, int)> transform);

private:
  llvm::function_ref<std::vector<Operation *>(ModuleOp)> getSpecificOps;
};

/// The OpReducer class defines a variant generator method that produces
/// multiple variants by eliminating different OpType operations from the
/// parent module.
template <typename OpType>
class OpReducer {
public:
  OpReducer() : impl(new OpReducerImpl(getSpecificOps)) {}

  /// Returns the vector of pointer to the OpType operations in the module.
  static std::vector<Operation *> getSpecificOps(ModuleOp module) {
    std::vector<Operation *> ops;
    for (auto op : module.getOps<OpType>()) {
      ops.push_back(op);
    }
    return ops;
  }

  /// Deletes the OpType operations in the module in the specified index.
  static void deleteOps(ModuleOp module, int start, int end) {
    std::vector<Operation *> opsToRemove;

    for (auto op : enumerate(getSpecificOps(module))) {
      int index = op.index();
      if (index >= start && index < end)
        opsToRemove.push_back(op.value());
    }

    for (Operation *o : opsToRemove) {
      o->dropAllUses();
      o->erase();
    }
  }

  /// Return the name of this reducer class.
  StringRef getName() { return impl->getName(); }

  /// Return the initial transformSpace containing the transformable indices.
  std::vector<bool> initTransformSpace(ModuleOp module) {
    return impl->initTransformSpace(module);
  }

  /// Generate variants by removing OpType operations from the module in the
  /// parent and link the variants as childs in the Reduction Tree Pass.
  void generateVariants(ReductionNode *parent, const Tester &test,
                        int numVariants) {
    impl->generateVariants(parent, test, numVariants, deleteOps);
  }

private:
  std::unique_ptr<OpReducerImpl> impl;
};

} // end namespace mlir

#endif
