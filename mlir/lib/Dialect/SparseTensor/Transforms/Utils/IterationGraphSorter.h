//===- IterationGraphSorter.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines the iteration graph sorter (top-sort scheduling).
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_UTILS_ITERATIONGRAPHSORTER_H_
#define MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_UTILS_ITERATIONGRAPHSORTER_H_

#include "mlir/IR/AffineMap.h"

namespace mlir {

// Forward declarations.
class Value;
namespace utils {
enum class IteratorType : uint32_t;
} // namespace utils
namespace linalg {
class GenericOp;
} // namespace linalg

namespace sparse_tensor {

/// Iteration graph sorting mask,
enum class SortMask : unsigned {
  // The individual mask bits.
  kIncludeDenseOutput = 0x1, // b001
  kIncludeDenseInput = 0x2,  // b010
  // The subsets of mask bits.
  kIncludeAll = 0x7,   // b111
  kIncludeDense = 0x3, // b011
  kSparseOnly = 0x0,   // b000
};

class IterationGraphSorter {
public:
  /// Factory method that construct an iteration graph sorter
  /// for the given linalg.generic operation.
  static IterationGraphSorter fromGenericOp(linalg::GenericOp genericOp);

  /// Returns a permutation that represents the scheduled loop order.
  /// Note that the returned AffineMap could be null if the kernel
  /// cannot be scheduled due to cyclic iteration graph.
  [[nodiscard]] AffineMap sort(SortMask mask, Value ignored = nullptr);

  /// Returns the number of loops in the iteration graph.
  unsigned getNumLoops() const { return loop2OutLvl.getNumDims(); }

private:
  // Private constructor.
  IterationGraphSorter(SmallVector<Value> &&ins,
                       SmallVector<AffineMap> &&loop2InsLvl, Value out,
                       AffineMap loop2OutLvl,
                       SmallVector<utils::IteratorType> &&iterTypes);

  // Adds all the constraints in the given loop to level map.
  void addConstraints(Value t, AffineMap loop2LvlMap);

  /// A helper to compute a topological sort. The method has an
  /// O(n^2) time complexity since we use an adjacency matrix
  /// representation for the iteration graph.
  AffineMap topoSort();

  // Input tensors and associated loop to level maps.
  SmallVector<Value> ins;
  SmallVector<AffineMap> loop2InsLvl;

  // Output tensor and associated loop to level map.
  Value out;
  AffineMap loop2OutLvl;

  // Loop itation types;
  SmallVector<utils::IteratorType> iterTypes;

  // Adjacency matrix that represents the iteration graph.
  std::vector<std::vector<bool>> itGraph;

  // InDegree used for topo sort.
  std::vector<unsigned> inDegree;
};

} // namespace sparse_tensor
} // namespace mlir

#endif // MLIR_DIALECT_SPARSETENSOR_TRANSFORMS_UTILS_ITERATIONGRAPHSORTER_H_
