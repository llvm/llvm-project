//===- GPUHeuristics.h - GPU heuristics for Linalg transforms ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_TRANSFORMOPS_GPUHEURISTICS_H
#define MLIR_DIALECT_LINALG_TRANSFORMOPS_GPUHEURISTICS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir {
namespace transform {
namespace gpu {

/// Base struct to hold GPU mapping information for a given operation.
struct MappingInfo {
  /// Number of threads to use for the mapping.
  /// Note: When the number of threads used is smaller than the total number of
  /// available threads, predication ensues. It is often useful to use more
  /// threads and saturate memory bandwidth for some operations, even if others
  /// end up being predicated.
  SmallVector<int64_t> numThreads;

  /// Thread mapping attributes, one per entry of `numThreads`.
  SmallVector<Attribute> threadMapping;
};

struct CopyMappingInfo : public MappingInfo {
  /// Status of the mapping computation, invalid usually means too many threads
  /// are required and we fail to map. This usually happens when the copy is too
  /// large compared to the number of threads.
  enum class Status { Success = 0, RequiresPredication, Invalid };

  /// Greedily compute the MappingInfo to use to perform a copy of `sizes`
  /// elements of bitwidth `elementalBitwidth`.
  /// The `desiredBitAlignment` is the number of elements by which the most
  /// minor dimension of the copy is expected to be aligned.
  /// This is an approximation of the final alignment, for each row of the copy.
  /// This is used to restrict the size of copied vector so that they match
  /// potential subsequent cp.async.
  /// If the alignment does not match the required alignment for a cp.async down
  /// the line, the conversion to cp.async will be eventually skipped, possibly
  /// degrading performance.
  /// When `favorPredication` is false, the mapping is computed to fill all
  /// threads with an equal amount of data to copy, so as to avoid predication.
  /// Predication ends up requiring a split epilogue in current pipelining
  /// implementations and is better avoided when possible.
  CopyMappingInfo(MLIRContext *ctx, int totalNumThreads,
                  int64_t desiredBitAlignment, ArrayRef<int64_t> sizes,
                  bool favorPredication = false,
                  int64_t elementalBitwidth = 32);

private:
  /// Determine the maximal vector size to use to copy a contiguous array of
  /// `numContiguousElements`, each of bitwidth `elementalBitwidth`.
  /// The `alignment` is the number of elements by which the most minor
  /// dimension of the copy is aligned. This is an approximation of actual
  /// memory alignment after bufferization, for each row of the copy. This is
  /// used to restrict the of the copied vector so that it is properly aligned
  /// with the requirements of cp.async. If the copy alignment does not match
  /// the required aligned for a cp.async, thae conversion to cp.async will be
  /// skipped.
  /// Asserts that `elementalBitwidth` divides `numContiguousElements`.
  static int64_t
  maxContiguousElementsToTransfer(int64_t alignment,
                                  int64_t numContiguousElements,
                                  int64_t elementalBitwidth = 32);

  /// Compute the number of threads to use to perform a copy of `sizes`
  /// elements of `elementalBitwidth`.
  /// The `alignment` is the number of elements by which the most minor
  /// dimension of the copy is aligned. This is an approximation of actual
  /// memory alignment after bufferization, for each row of the copy. This is
  /// used to restrict the of the copied vector so that it is properly aligned
  /// with the requirements of cp.async. If the copy alignment does not match
  /// the required aligned for a cp.async, the conversion to cp.async will be
  /// skipped.
  /// When `favorPredication` is false, the implementation avoids predication
  /// in the copy, even if it means reducing the granularity of the transfer.
  /// Otherwise, the implementation will come up with a maximal assignment of
  /// the remaining threads to sizes of interest, using a DP implementation.
  Status inferNumThreads(int64_t totalNumThreads, ArrayRef<int64_t> sizes,
                         int64_t desiredVectorSize, bool favorPredication);
  Status inferNumThreadsImpl(int64_t totalNumThreads, ArrayRef<int64_t> sizes,
                             int64_t desiredVectorSize);

public:
  // Pretty-printing and diagnostic methods.
  void print(llvm::raw_ostream &os) const;
  LLVM_DUMP_METHOD void dump() const;

  /// Static quantity determining the number of bits to target in an individual
  /// copy. Assumes that smaller increments of 64, 32, 16, 8 are also valid
  /// transfer sizes. In the future we should have more hardware pluggability
  /// here, especially when we want sub-byte granularity
  static constexpr int64_t kMaxVectorLoadBitWidth = 128;

  /// Most minor vector size (i.e. 1-D), in number of elements, used in a copy.
  int64_t vectorSize;

  /// Number of threads to use for the copy mapping, from most major to most
  /// minor dims (i.e. numThreads.back() should be mapped to contiguous threads
  /// for best coalescing).
  using MappingInfo::numThreads;

  /// Explicit computation / injection of the smallest bounding tile sizes after
  /// mapping to `numThreads`. This is useful in masked scenarios.
  SmallVector<int64_t> smallestBoundingTileSizes;

  /// Thread mapping attributes, one per entry of `numThreads`.
  using MappingInfo::threadMapping;

  /// The status of a particular copy mapping. Must be checked before applying
  /// transformations.
  Status status;
};

} // namespace gpu
} // namespace transform
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_TRANSFORMOPS_GPUHEURISTICS_H
