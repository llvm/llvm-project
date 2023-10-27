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
             unsigned numTensors, unsigned numLoops, unsigned numFilterLoops,
             unsigned maxRank);

  //
  // General methods.
  //

  LogicalResult initTensorExp();
  ExprId getExprId() const { return tensorExp; }

  linalg::GenericOp op() const { return linalgOp; }
  const SparsificationOptions &options() const { return sparseOptions; }
  Merger &merger() { return latticeMerger; }
  LoopEmitter &emitter() { return loopEmitter; }

  void startEmit();

  /// Generates loop boundary statements (entering/exiting loops). The function
  /// passes and updates the passed-in parameters.
  std::optional<Operation *>
  genLoopBoundary(function_ref<
                  std::optional<Operation *>(MutableArrayRef<Value> parameters)>
                      callback);

  //
  // Merger delegates.
  //

  constexpr TensorId makeTensorId(unsigned t) const {
    return latticeMerger.makeTensorId(t);
  }
  constexpr LoopId makeLoopId(unsigned i) const {
    return latticeMerger.makeLoopId(i);
  }
  constexpr TensorLoopId makeTensorLoopId(unsigned t, unsigned i) const {
    return latticeMerger.makeTensorLoopId(t, i);
  }
  const TensorExp &exp(ExprId e) const { return latticeMerger.exp(e); }
  const LatPoint &lat(LatPointId l) const { return latticeMerger.lat(l); }
  ArrayRef<LatPointId> set(LatSetId s) const { return latticeMerger.set(s); }
  DimLevelType dlt(TensorId t, LoopId i) const {
    return latticeMerger.getLvlType(t, i);
  }
  DimLevelType dlt(TensorLoopId b) const { return latticeMerger.getLvlType(b); }

  //
  // LoopEmitter delegates.
  //

  TensorLevel makeTensorLevel(TensorId t, Level l) const {
    // Make sure LoopEmitter, GenericOp, and Merger agree on the number of
    // tensors.
    assert(loopEmitter.getNumManifestTensors() == linalgOp->getNumOperands() &&
           loopEmitter.getNumTensors() == latticeMerger.getNumTensors() &&
           loopEmitter.getOutTensorId() == latticeMerger.getOutTensorID() &&
           loopEmitter.getSynTensorId() == latticeMerger.getSynTensorID());
    return loopEmitter.makeTensorLevel(t, l);
  }
  TensorLevel makeTensorLevel(std::pair<TensorId, Level> tlPair) const {
    return makeTensorLevel(tlPair.first, tlPair.second);
  }
  std::pair<TensorId, Level> unpackTensorLevel(TensorLevel tl) const {
    return loopEmitter.unpackTensorLevel(tl);
  }
  template <class ContainerTy>
  auto unpackTensorLevelRange(ContainerTy &&c) const {
    return loopEmitter.unpackTensorLevelRange(std::forward<ContainerTy>(c));
  }

  //
  // Code generation environment verify functions.
  //

  /// Whether the tensor expression is admissible for codegen.
  /// It also sets the sparseOut if the output tensor is sparse.
  bool isAdmissibleTensorExp(ExprId e);

  /// Whether the iteration graph is sorted in admissible topoOrder.
  /// Sets outerParNest on success with sparse output
  bool isAdmissibleTopoOrder();

  //
  // Topological delegate and sort methods.
  //

  LoopOrd topSortSize() const { return topSort.size(); }
  LoopId topSortAt(LoopOrd n) const { return topSort.at(n); }
  void topSortPushBack(LoopId i) { topSort.push_back(i); }
  void topSortClear(size_t capacity = 0) {
    topSort.clear();
    topSort.reserve(capacity);
  }

  ArrayRef<LoopId> getTopSortSlice(LoopOrd n, LoopOrd m) const;
  ArrayRef<LoopId> getLoopStackUpTo(LoopOrd n) const;
  ArrayRef<LoopId> getCurrentLoopStack() const;
  /// Returns the induction-variable for the loop identified by the given
  /// `LoopId`.  This method handles application of the topological sort
  /// in order to convert the `LoopId` into the corresponding `LoopOrd`.
  Value getLoopVar(LoopId i) const;

  //
  // Sparse tensor output and expansion methods.
  //

  bool hasSparseOutput() const { return sparseOut != nullptr; }
  bool isSparseOutput(OpOperand *o) const { return sparseOut == o; }

  Value getInsertionChain() const { return insChain; }
  void updateInsertionChain(Value chain);

  // FIXME: clarify what this "rank" is really supposed to mean/be.
  bool atExpandLevel(OpOperand *o, unsigned rank, LoopOrd n) const;
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

  void startReduc(ExprId exp, Value val);
  bool isReduc() const { return redExp != detail::kInvalidId; }
  void updateReduc(Value val);
  Value getReduc() const { return redVal; }
  Value endReduc();
  void setValidLexInsert(Value val);
  void clearValidLexInsert();
  Value getValidLexInsert() const { return redValidLexInsert; }

  void startCustomReduc(ExprId exp);
  bool isCustomReduc() const { return redCustom != detail::kInvalidId; }
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

  // Topological sort.  This serves as a mapping from `LoopOrd` to `LoopId`
  // (cf., `getLoopVar` and `topSortAt`).
  std::vector<LoopId> topSort;

  // Sparse tensor as output. Implemented either through direct injective
  // insertion in lexicographic index order or through access pattern
  // expansion in the innermost loop nest (`expValues` through `expCount`).
  OpOperand *sparseOut;
  // The count of outer non-filter loops, as defined by `isAdmissibleTopoOrder`.
  LoopOrd outerParNest;
  Value insChain;
  Value expValues;
  Value expFilled;
  Value expAdded;
  Value expCount;

  // Bookkeeping for reductions (up-to-date value of the reduction, and indices
  // into the merger's expression tree. When the indices of a tensor reduction
  // expression are exhausted, all inner loops can use a scalarized reduction.
  Value redVal;
  ExprId redExp;
  ExprId redCustom;

  // Bookkeeping for lex insertion during reductions. Holds the runtime boolean
  // value of whether any reduction occurred. This is only set during a
  // reduction and cleared once the reduction is finished.
  Value redValidLexInsert;

  // The root tensor expression of the kernel.
  ExprId tensorExp;
};

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_CODEGENENV_H_
