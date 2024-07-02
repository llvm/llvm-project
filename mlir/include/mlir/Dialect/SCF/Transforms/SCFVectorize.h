//===- SCFVectorize.h - ------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_SCFVECTORIZE_H_
#define MLIR_TRANSFORMS_SCFVECTORIZE_H_

#include <optional>

namespace mlir {
class DataLayout;
struct LogicalResult;
namespace scf {
class ParallelOp;
}
namespace scf {

/// Loop vectorization info
struct SCFVectorizeInfo {
  /// Loop dimension on which to vectorize.
  unsigned dim = 0;

  /// Biggest vector width, in elements.
  unsigned factor = 0;

  /// Number of ops, which will be vectorized.
  unsigned count = 0;

  /// Can use masked vector ops for our of bounds memory accesses.
  bool masked = false;
};

/// Collect vectorization statistics on specified `scf.parallel` dimension.
/// Return `SCFVectorizeInfo` or `std::nullopt` if loop cannot be vectorized on
/// specified dimension.
///
/// `vectorBitwidth` - maximum vector size, in bits.
std::optional<SCFVectorizeInfo>
getLoopVectorizeInfo(mlir::scf::ParallelOp loop, unsigned dim,
                     unsigned vectorBitwidth, const DataLayout *DL = nullptr);

/// Vectorization params
struct SCFVectorizeParams {
  /// Loop dimension on which to vectorize.
  unsigned dim = 0;

  /// Desired vector length, in elements
  unsigned factor = 0;

  /// Use masked vector ops for memory access outside loop bounds.
  bool masked = false;
};

/// Vectorize loop on specified dimension with specified factor.
///
/// If `masked` is `true` and loop bound is not divisible by `factor`, instead
/// of generating second loop to process remainig iterations, extend loop count
/// and generate masked vector ops to handle out-of bounds memory accesses.
mlir::LogicalResult vectorizeLoop(mlir::scf::ParallelOp loop,
                                  const SCFVectorizeParams &params,
                                  const DataLayout *DL = nullptr);
} // namespace scf
} // namespace mlir

#endif // MLIR_TRANSFORMS_SCFVECTORIZE_H_
