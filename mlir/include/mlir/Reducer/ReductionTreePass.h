//===- ReductionTreePass.h - Reduction Tree Pass Implementation -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Reduction Tree Pass class. It provides a framework for
// the implementation of different reduction passes in the MLIR Reduce tool. It
// allows for custom specification of the variant generation behavior. It
// implements methods that define the different possible traversals of the
// reduction tree.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_REDUCER_REDUCTIONTREEPASS_H
#define MLIR_REDUCER_REDUCTIONTREEPASS_H

#include <vector>

#include "PassDetail.h"
#include "ReductionNode.h"
#include "mlir/Reducer/Passes/OpReducer.h"
#include "mlir/Reducer/Tester.h"

#define DEBUG_TYPE "mlir-reduce"

namespace mlir {

/// This class defines the Reduction Tree Pass. It provides a framework to
/// to implement a reduction pass using a tree structure to keep track of the
/// generated reduced variants.
class ReductionTreePass : public ReductionTreeBase<ReductionTreePass> {
public:
  ReductionTreePass(const ReductionTreePass &pass)
      : ReductionTreeBase<ReductionTreePass>(pass), opType(pass.opType),
        mode(pass.mode), test(pass.test) {}

  ReductionTreePass(StringRef opType, TraversalMode mode, const Tester &test)
      : opType(opType), mode(mode), test(test) {}

  /// Runs the pass instance in the pass pipeline.
  void runOnOperation() override;

private:
  template <typename IteratorType>
  ModuleOp findOptimal(ModuleOp module, std::unique_ptr<OpReducer> reducer,
                       ReductionNode *node);

  /// The name of operation that we will try to remove.
  StringRef opType;

  TraversalMode mode;

  /// This is used to test the interesting behavior of the reduction nodes in
  /// the tree.
  const Tester &test;
};

} // end namespace mlir

#endif
