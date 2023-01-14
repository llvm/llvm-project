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
#include "LoopEmitter.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/SparseTensor/Transforms/Passes.h"
#include "mlir/Dialect/SparseTensor/Utils/Merger.h"
#include <optional>

namespace mlir {
namespace sparse_tensor {

/// The code generation environment class aggregates a number of data
/// structures that are needed during the code generation phase of
/// sparsification. This environment simplifies passing around such
/// data during sparsification (rather than passing around all the
/// individual compoments where needed). Furthermore, it provides
/// convience methods that keep implementation details transparent
/// to sparsification while asserting on internal consistency.
class CodegenEnv {
public:
  /// Constructs a code generation environment which can be
  /// passed around during sparsification for bookkeeping
  /// together with some consistency asserts.
  CodegenEnv(linalg::GenericOp linop, SparsificationOptions opts,
             unsigned numTensors, unsigned numLoops, unsigned numFilterLoops);

  //
  // General methods.
  //

  linalg::GenericOp op() const { return linalgOp; }
  const SparsificationOptions &options() const { return sparseOptions; }
  Merger &merger() { return latticeMerger; }
  LoopEmitter &emitter() { return loopEmitter; }

  void startEmit(OpOperand *so, unsigned lv);

  /// Generates loop boundary statements (entering/exiting loops). The function
  /// passes and updates the passed-in parameters.
  Optional<Operation *> genLoopBoundary(
      function_ref<Optional<Operation *>(MutableArrayRef<Value> parameters)>
          callback);

  //
  // Merger delegates.
  //

  TensorExp &exp(unsigned e) { return latticeMerger.exp(e); }
  LatPoint &lat(unsigned l) { return latticeMerger.lat(l); }
  SmallVector<unsigned> &set(unsigned s) { return latticeMerger.set(s); }
  DimLevelType dlt(unsigned t, unsigned i) const {
    return latticeMerger.getDimLevelType(t, i);
  }
  DimLevelType dlt(unsigned b) const {
    return latticeMerger.getDimLevelType(b);
  }

  //
  // Topological delegate and sort methods.
  //

  size_t topSortSize() const { return topSort.size(); }
  unsigned topSortAt(unsigned i) const { return topSort.at(i); }
  void topSortPushBack(unsigned i) { topSort.push_back(i); }
  void topSortClear(unsigned capacity = 0) {
    topSort.clear();
    topSort.reserve(capacity);
  }

  ArrayRef<unsigned> getTopSortSlice(size_t n, size_t m) const;
  ArrayRef<unsigned> getLoopCurStack() const;
  Value getLoopIdxValue(size_t loopIdx) const;

  //
  // Sparse tensor output and expansion methods.
  //

  bool hasSparseOutput() const { return sparseOut != nullptr; }
  bool isSparseOutput(OpOperand *o) const { return sparseOut == o; }

  Value getInsertionChain() const { return insChain; }
  void updateInsertionChain(Value chain);

  bool atExpandLevel(OpOperand *o, unsigned rank, unsigned lv) const;
  void startExpand(Value values, Value filled, Value added, Value count);
  bool isExpand() const { return expValues != nullptr; }
  void updateExpandCount(Value count);
  Value getExpandValues() const { return expValues; }
  Value getExpandFilled() const { return expFilled; }
  Value getExpandAdded() const { return expAdded; }
  Value getExpandCount() const { return expCount; }
  void endExpand();

  //
  // Reduction methods.
  //

  void startReduc(unsigned exp, Value val);
  bool isReduc() const { return redExp != -1u; }
  void updateReduc(Value val);
  Value getReduc() const { return redVal; }
  Value endReduc();

  void startCustomReduc(unsigned exp);
  bool isCustomReduc() const { return redCustom != -1u; }
  Value getCustomRedId();
  void endCustomReduc();

private:
  // Linalg operation.
  linalg::GenericOp linalgOp;

  // Sparsification options.
  SparsificationOptions sparseOptions;

  // Merger helper class.
  Merger latticeMerger;

  // Loop emitter helper class.
  LoopEmitter loopEmitter;

  // Topological sort.
  std::vector<unsigned> topSort;

  // Sparse tensor as output. Implemented either through direct injective
  // insertion in lexicographic index order or through access pattern
  // expansion in the innermost loop nest (`expValues` through `expCount`).
  OpOperand *sparseOut;
  unsigned outerParNest;
  Value insChain;
  Value expValues;
  Value expFilled;
  Value expAdded;
  Value expCount;

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
