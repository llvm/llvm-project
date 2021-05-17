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

#include <limits>

#include "mlir/Reducer/ReductionNode.h"
#include "mlir/Reducer/Tester.h"

namespace mlir {

class OpReducer {
public:
  virtual ~OpReducer() = default;
  /// According to rangeToKeep, try to reduce the given module. We implicitly
  /// number each interesting operation and rangeToKeep indicates that if an
  /// operation's number falls into certain range, then we will not try to
  /// reduce that operation.
  virtual void reduce(ModuleOp module,
                      ArrayRef<ReductionNode::Range> rangeToKeep) = 0;
  /// Return the number of certain kind of operations that we would like to
  /// reduce. This can be used to build a range map to exclude uninterested
  /// operations.
  virtual int getNumTargetOps(ModuleOp module) const = 0;
};

/// Reducer is a helper class to remove potential uninteresting operations from
/// module.
template <typename OpType>
class Reducer : public OpReducer {
public:
  ~Reducer() override = default;

  int getNumTargetOps(ModuleOp module) const override {
    return std::distance(module.getOps<OpType>().begin(),
                         module.getOps<OpType>().end());
  }

  void reduce(ModuleOp module,
              ArrayRef<ReductionNode::Range> rangeToKeep) override {
    std::vector<Operation *> opsToRemove;
    size_t keepIndex = 0;

    for (auto op : enumerate(module.getOps<OpType>())) {
      int index = op.index();
      if (keepIndex < rangeToKeep.size() &&
          index == rangeToKeep[keepIndex].second)
        ++keepIndex;
      if (keepIndex == rangeToKeep.size() ||
          index < rangeToKeep[keepIndex].first)
        opsToRemove.push_back(op.value());
    }

    for (Operation *o : opsToRemove) {
      o->dropAllUses();
      o->erase();
    }
  }
};

} // end namespace mlir

#endif
