//===- InferStridedMetadataInterface.h - Strided Metadata Inference -C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions of the strided metadata inference interface
// defined in `InferStridedMetadataInterface.td`
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_INFERSTRIDEDMETADATAINTERFACE_H
#define MLIR_INTERFACES_INFERSTRIDEDMETADATAINTERFACE_H

#include "mlir/Interfaces/InferIntRangeInterface.h"

namespace mlir {
/// A class that represents the strided metadata range information, including
/// offsets, sizes, and strides as integer ranges.
class StridedMetadataRange {
public:
  /// Default constructor creates uninitialized ranges.
  StridedMetadataRange() = default;

  /// Returns a ranked strided metadata range.
  static StridedMetadataRange
  getRanked(SmallVectorImpl<ConstantIntRanges> &&offsets,
            SmallVectorImpl<ConstantIntRanges> &&sizes,
            SmallVectorImpl<ConstantIntRanges> &&strides) {
    return StridedMetadataRange(std::move(offsets), std::move(sizes),
                                std::move(strides));
  }

  /// Returns a strided metadata range with maximum ranges.
  static StridedMetadataRange getMaxRanges(int32_t indexBitwidth,
                                           int32_t offsetsRank,
                                           int32_t sizeRank,
                                           int32_t stridedRank) {
    return StridedMetadataRange(
        SmallVector<ConstantIntRanges>(
            offsetsRank, ConstantIntRanges::maxRange(indexBitwidth)),
        SmallVector<ConstantIntRanges>(
            sizeRank, ConstantIntRanges::maxRange(indexBitwidth)),
        SmallVector<ConstantIntRanges>(
            stridedRank, ConstantIntRanges::maxRange(indexBitwidth)));
  }

  static StridedMetadataRange getMaxRanges(int32_t indexBitwidth,
                                           int32_t rank) {
    return getMaxRanges(indexBitwidth, 1, rank, rank);
  }

  /// Returns whether the metadata is uninitialized.
  bool isUninitialized() const { return !offsets.has_value(); }

  /// Get the offsets range.
  ArrayRef<ConstantIntRanges> getOffsets() const {
    return offsets ? *offsets : ArrayRef<ConstantIntRanges>();
  }
  MutableArrayRef<ConstantIntRanges> getOffsets() {
    return offsets ? *offsets : MutableArrayRef<ConstantIntRanges>();
  }

  /// Get the sizes ranges.
  ArrayRef<ConstantIntRanges> getSizes() const { return sizes; }
  MutableArrayRef<ConstantIntRanges> getSizes() { return sizes; }

  /// Get the strides ranges.
  ArrayRef<ConstantIntRanges> getStrides() const { return strides; }
  MutableArrayRef<ConstantIntRanges> getStrides() { return strides; }

  /// Compare two strided metadata ranges.
  bool operator==(const StridedMetadataRange &other) const {
    return offsets == other.offsets && sizes == other.sizes &&
           strides == other.strides;
  }

  /// Print the strided metadata range.
  void print(raw_ostream &os) const;

  /// Join two strided metadata ranges, by taking the element-wise union of the
  /// metadata.
  static StridedMetadataRange join(const StridedMetadataRange &lhs,
                                   const StridedMetadataRange &rhs) {
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;

    // Helper fuction to compute the range union of constant ranges.
    auto rangeUnion =
        +[](const std::tuple<ConstantIntRanges, ConstantIntRanges> &lhsRhs)
        -> ConstantIntRanges {
      return std::get<0>(lhsRhs).rangeUnion(std::get<1>(lhsRhs));
    };

    // Get the elementwise range union. Note, that `zip_equal` will assert if
    // sizes are not equal.
    SmallVector<ConstantIntRanges> offsets = llvm::map_to_vector(
        llvm::zip_equal(*lhs.offsets, *rhs.offsets), rangeUnion);
    SmallVector<ConstantIntRanges> sizes =
        llvm::map_to_vector(llvm::zip_equal(lhs.sizes, rhs.sizes), rangeUnion);
    SmallVector<ConstantIntRanges> strides = llvm::map_to_vector(
        llvm::zip_equal(lhs.strides, rhs.strides), rangeUnion);

    // Return the joined metadata.
    return StridedMetadataRange(std::move(offsets), std::move(sizes),
                                std::move(strides));
  }

private:
  /// Create a strided metadata range with the given offset, sizes, and strides.
  StridedMetadataRange(SmallVectorImpl<ConstantIntRanges> &&offsets,
                       SmallVectorImpl<ConstantIntRanges> &&sizes,
                       SmallVectorImpl<ConstantIntRanges> &&strides)
      : offsets(std::move(offsets)), sizes(std::move(sizes)),
        strides(std::move(strides)) {}

  /// The offsets range.
  std::optional<SmallVector<ConstantIntRanges>> offsets;

  /// The sizes ranges.
  SmallVector<ConstantIntRanges> sizes;

  /// The strides ranges.
  SmallVector<ConstantIntRanges> strides;
};

/// Print the strided metadata to `os`.
inline raw_ostream &operator<<(raw_ostream &os,
                               const StridedMetadataRange &range) {
  range.print(os);
  return os;
}

/// Callback function type for setting the strided metadata of a value.
using SetStridedMetadataRangeFn =
    function_ref<void(Value, const StridedMetadataRange &)>;
} // end namespace mlir

#include "mlir/Interfaces/InferStridedMetadataInterface.h.inc"

#endif // MLIR_INTERFACES_INFERSTRIDEDMETADATAINTERFACE_H
