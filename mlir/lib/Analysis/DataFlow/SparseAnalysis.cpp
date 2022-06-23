//===- SparseAnalysis.cpp - Sparse data-flow analysis ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"

using namespace mlir;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// AbstractSparseLattice
//===----------------------------------------------------------------------===//

void AbstractSparseLattice::onUpdate(DataFlowSolver *solver) const {
  // Push all users of the value to the queue.
  for (Operation *user : point.get<Value>().getUsers())
    for (DataFlowAnalysis *analysis : useDefSubscribers)
      solver->enqueue({user, analysis});
}
