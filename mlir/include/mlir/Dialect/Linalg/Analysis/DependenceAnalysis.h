//===- DependenceAnalysis.h - Dependence analysis on SSA views --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_ANALYSIS_DEPENDENCEANALYSIS_H_
#define MLIR_DIALECT_LINALG_ANALYSIS_DEPENDENCEANALYSIS_H_

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
class FuncOp;

namespace linalg {

class LinalgOp;

/// A very primitive alias analysis which just records for each view, either:
///   1. The base buffer, or
///   2. The block argument view
/// that it indexes into.
/// This does not perform inter-block or inter-procedural analysis and assumes
/// that different block argument views do not alias.
class Aliases {
public:
  /// Returns true if v1 and v2 alias.
  bool alias(Value v1, Value v2) { return find(v1) == find(v2); }

private:
  /// Returns the base buffer or block argument into which the view `v` aliases.
  /// This lazily records the new aliases discovered while walking back the
  /// use-def chain.
  Value find(Value v);

  DenseMap<Value, Value> aliases;
};

/// Data structure for holding a dependence graph that operates on LinalgOp and
/// views as SSA values.
class LinalgDependenceGraph {
public:
  enum DependenceType { RAR = 0, RAW, WAR, WAW, NumTypes };
  // TODO: OpOperand tracks dependencies on buffer operands. Tensor result will
  // need an extension to use OpResult.
  struct LinalgDependenceGraphElem {
    // dependentOpView may be either:
    //   1. src in the case of dependencesIntoGraphs.
    //   2. dst in the case of dependencesFromDstGraphs.
    OpOperand *dependentOpView;
    // View in the op that is used to index in the graph:
    //   1. src in the case of dependencesFromDstGraphs.
    //   2. dst in the case of dependencesIntoGraphs.
    OpOperand *indexingOpView;
    // Type of the dependence.
    DependenceType dependenceType;
  };
  using LinalgDependences = SmallVector<LinalgDependenceGraphElem, 8>;
  using DependenceGraph = DenseMap<Operation *, LinalgDependences>;
  using dependence_iterator = LinalgDependences::const_iterator;
  using dependence_range = iterator_range<dependence_iterator>;

  static StringRef getDependenceTypeStr(DependenceType depType);

  // Builds a linalg dependence graph for the ops of type LinalgOp under `f`.
  static LinalgDependenceGraph buildDependenceGraph(Aliases &aliases, FuncOp f);
  LinalgDependenceGraph(Aliases &aliases, ArrayRef<LinalgOp> ops);

  /// Returns the X such that op -> X is a dependence of type dt.
  dependence_range getDependencesFrom(Operation *src, DependenceType dt) const;
  dependence_range getDependencesFrom(LinalgOp src, DependenceType dt) const;

  /// Returns the X such that X -> op is a dependence of type dt.
  dependence_range getDependencesInto(Operation *dst, DependenceType dt) const;
  dependence_range getDependencesInto(LinalgOp dst, DependenceType dt) const;

  /// Returns the operations that are interleaved between `srcLinalgOp` and
  /// `dstLinalgOp` and that are involved in any RAW, WAR or WAW dependence
  /// relation with `srcLinalgOp`, on any view.
  /// Any such operation prevents reordering.
  SmallVector<Operation *, 8>
  findCoveringDependences(LinalgOp srcLinalgOp, LinalgOp dstLinalgOp) const;

  /// Returns the operations that are interleaved between `srcLinalgOp` and
  /// `dstLinalgOp` and that are involved in a RAR or RAW with `srcLinalgOp`.
  /// Dependences are restricted to views aliasing `view`.
  SmallVector<Operation *, 8> findCoveringReads(LinalgOp srcLinalgOp,
                                                LinalgOp dstLinalgOp,
                                                Value view) const;

  /// Returns the operations that are interleaved between `srcLinalgOp` and
  /// `dstLinalgOp` and that are involved in a WAR or WAW with `srcLinalgOp`.
  /// Dependences are restricted to views aliasing `view`.
  SmallVector<Operation *, 8> findCoveringWrites(LinalgOp srcLinalgOp,
                                                 LinalgOp dstLinalgOp,
                                                 Value view) const;

  /// Returns true if the two operations have the specified dependence from
  /// `srcLinalgOp` to `dstLinalgOp`.
  bool hasDependenceFrom(LinalgOp srcLinalgOp, LinalgOp dstLinalgOp,
                         ArrayRef<DependenceType> depTypes = {
                             DependenceType::RAW, DependenceType::WAW}) const;

  /// Returns true if the `linalgOp` has dependences into it.
  bool hasDependentOperationsInto(LinalgOp linalgOp,
                                  ArrayRef<DependenceType> depTypes = {
                                      DependenceType::RAW,
                                      DependenceType::WAW}) const;

  /// Returns true if the `linalgOp` has dependences from it.
  bool hasDependentOperationsFrom(LinalgOp linalgOp,
                                  ArrayRef<DependenceType> depTypes = {
                                      DependenceType::RAW,
                                      DependenceType::WAW}) const;

  /// Returns true if the `linalgOp` has dependences into or from it.
  bool hasDependentOperations(LinalgOp linalgOp,
                              ArrayRef<DependenceType> depTypes = {
                                  DependenceType::RAW,
                                  DependenceType::WAW}) const;

  /// Returns all operations that have a dependence into `linalgOp` of types
  /// listed in `depTypes`.
  SmallVector<LinalgDependenceGraphElem, 2> getDependentOperationsInto(
      LinalgOp linalgOp, ArrayRef<DependenceType> depTypes = {
                             DependenceType::RAW, DependenceType::WAW}) const;

  /// Returns all operations that have a dependence from `linalgOp` of types
  /// listed in `depTypes`.
  SmallVector<LinalgDependenceGraphElem, 2> getDependentOperationsFrom(
      LinalgOp linalgOp, ArrayRef<DependenceType> depTypes = {
                             DependenceType::RAW, DependenceType::WAW}) const;

  /// Returns all dependent operations (into and from) given `operation`.
  SmallVector<LinalgDependenceGraphElem, 2>
  getDependentOperations(LinalgOp linalgOp,
                         ArrayRef<DependenceType> depTypes = {
                             DependenceType::RAW, DependenceType::WAW}) const;

private:
  // Keep dependences in both directions, this is not just a performance gain
  // but it also reduces usage errors.
  // Dependence information is stored as a map of:
  //   (source operation -> LinalgDependenceGraphElem)
  DependenceGraph dependencesFromGraphs[DependenceType::NumTypes];
  // Reverse dependence information is stored as a map of:
  //   (destination operation -> LinalgDependenceGraphElem)
  DependenceGraph dependencesIntoGraphs[DependenceType::NumTypes];

  /// Analyses the aliasing views between `src` and `dst` and inserts the proper
  /// dependences in the graph.
  void addDependencesBetween(LinalgOp src, LinalgOp dst);

  // Adds an new dependence unit in the proper graph.
  // Uses std::pair to keep operations and view together and avoid usage errors
  // related to src/dst and producer/consumer terminology in the context of
  // dependences.
  void addDependenceElem(DependenceType dt, OpOperand *indexingOpView,
                         OpOperand *dependentOpView);

  /// Implementation detail for findCoveringxxx.
  SmallVector<Operation *, 8>
  findOperationsWithCoveringDependences(LinalgOp srcLinalgOp,
                                        LinalgOp dstLinalgOp, Value view,
                                        ArrayRef<DependenceType> types) const;

  Aliases &aliases;
  SmallVector<LinalgOp, 8> linalgOps;
  DenseMap<Operation *, unsigned> linalgOpPositions;
};
} // namespace linalg
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_ANALYSIS_DEPENDENCEANALYSIS_H_
