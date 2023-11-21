//===- LoopScheduler.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

/// Iteration graph sorting.
enum class SortMask : unsigned {
  // The individual mask bits.
  kIncludeDenseOutput = 0x1, // b001
  kIncludeDenseInput = 0x2,  // b010
  kIncludeUndef = 0x4,       // b100
  // The subsets of mask bits.
  kIncludeAll = 0x7,   // b111
  kIncludeDense = 0x3, // b011
  kSparseOnly = 0x0,   // b000
};

class IterationGraphSorter {
public:
  // Constructs a scheduler from linalg.generic
  // Maybe reuses the class to schedule foreach as well (to address
  // non-permutation, e.g, traverse CSR in BSR order).
  static IterationGraphSorter fromGenericOp(linalg::GenericOp genericOp);

  // Returns a permutation that represents the scheduled loop order.
  // Note that the returned AffineMap could be null if the kernel can not be
  // schedule due to cycles in the iteration graph.
  [[nodiscard]] AffineMap sort(SortMask mask, Value ignored = nullptr);
  unsigned getNumLoops() const { return loop2OutLvl.getNumDims(); }

private:
  IterationGraphSorter(SmallVector<Value> &&ins,
                       SmallVector<AffineMap> &&loop2InsLvl, Value out,
                       AffineMap loop2OutLvl,
                       SmallVector<utils::IteratorType> &&iterTypes);

  void addConstraints(Value t, AffineMap loop2LvlMap);
  AffineMap topoSort();

  // Input tensors and associated loop to level maps.
  SmallVector<Value> ins;
  SmallVector<AffineMap> loop2InsLvl;
  // Output tensor and associated loop to level map.
  Value out;
  AffineMap loop2OutLvl;
  // Loop type;
  SmallVector<utils::IteratorType> iterTypes;

  // Adjacent matrix that represents the iteration graph.
  std::vector<std::vector<bool>> itGraph;
  // InDegree used for topo sort.
  std::vector<unsigned> inDegree;
};

} // namespace sparse_tensor
} // namespace mlir
