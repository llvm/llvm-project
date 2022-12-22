//===- CodegenEnv.h - Code generation environment class ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines the code generation environment class.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_CODEGENENV_H_
#define MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_CODEGENENV_H_

#include "CodegenUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/SparseTensor/Utils/Merger.h"

namespace mlir {
namespace sparse_tensor {

/// The code generation environment class aggregates a number of data
/// structures that are needed during the code generation phase of
/// sparsification. This environment simplifies passing around such
/// data during sparsification (rather than passing around all the
/// individual compoments where needed). Furthermore, it provides
/// a number of delegate and convience methods that keep some of the
/// implementation details transparent to sparsification.
class CodegenEnv {
public:
  CodegenEnv(linalg::GenericOp linop, SparsificationOptions opts,
             unsigned numTensors, unsigned numLoops, unsigned numFilterLoops);

  // Start emitting.
  void startEmit(SparseTensorLoopEmitter *le);

  // Delegate methods to merger.
  TensorExp &exp(unsigned e) { return merger.exp(e); }
  LatPoint &lat(unsigned l) { return merger.lat(l); }
  SmallVector<unsigned> &set(unsigned s) { return merger.set(s); }
  DimLevelType dimLevelType(unsigned t, unsigned i) const {
    return merger.getDimLevelType(t, i);
  }
  DimLevelType dimLevelType(unsigned b) const {
    return merger.getDimLevelType(b);
  }
  bool isFilterLoop(unsigned i) const { return merger.isFilterLoop(i); }

  // Delegate methods to loop emitter.
  Value getLoopIV(unsigned i) const { return loopEmitter->getLoopIV(i); }
  const std::vector<Value> &getValBuffer() const {
    return loopEmitter->getValBuffer();
  }

  // Convenience method to slice topsort.
  ArrayRef<unsigned> getTopSortSlice(size_t n, size_t m) const {
    return ArrayRef<unsigned>(topSort).slice(n, m);
  }

  // Convenience method to get current loop stack.
  ArrayRef<unsigned> getLoopCurStack() const {
    return getTopSortSlice(0, loopEmitter->getCurrentDepth());
  }

  // Convenience method to get the IV of the given loop index.
  Value getLoopIdxValue(size_t loopIdx) const {
    for (unsigned lv = 0, lve = topSort.size(); lv < lve; lv++)
      if (topSort[lv] == loopIdx)
        return getLoopIV(lv);
    llvm_unreachable("invalid loop index");
  }

  //
  // Reductions.
  //

  void startReduc(unsigned exp, Value val);
  void updateReduc(Value val);
  bool isReduc() const { return redExp != -1u; }
  Value getReduc() const { return redVal; }
  Value endReduc();

  void startCustomReduc(unsigned exp);
  bool isCustomReduc() const { return redCustom != -1u; }
  Value getCustomRedId();
  void endCustomReduc();

public:
  //
  // TODO make this section private too, using similar refactoring as for reduc
  //

  // Linalg operation.
  linalg::GenericOp linalgOp;

  // Sparsification options.
  SparsificationOptions options;

  // Topological sort.
  std::vector<unsigned> topSort;

  // Merger helper class.
  Merger merger;

  // Loop emitter helper class (keep reference in scope!).
  // TODO: move emitter constructor up in time?
  SparseTensorLoopEmitter *loopEmitter;

  // Sparse tensor as output. Implemented either through direct injective
  // insertion in lexicographic index order or through access pattern expansion
  // in the innermost loop nest (`expValues` through `expCount`).
  OpOperand *sparseOut;
  unsigned outerParNest;
  Value insChain; // bookkeeping for insertion chain
  Value expValues;
  Value expFilled;
  Value expAdded;
  Value expCount;

private:
  // Bookkeeping for reductions (up-to-date value of the reduction, and indices
  // into the merger's expression tree. When the indices of a tensor reduction
  // expression are exhausted, all inner loops can use a scalarized reduction.
  Value redVal;
  unsigned redExp;
  unsigned redCustom;
};

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_CODEGENENV_H_
