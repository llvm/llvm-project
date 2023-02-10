//===-- MMAUtils.h - MLIR NVGPU dialect utilities for MMA operations-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utilities to assist in the lowering of other dialects
// (e.g. Vector) to `nvgpu.mma.*` dialect operations.
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_NVGPU_UTILS_MMAUTILS_H
#define MLIR_DIALECT_NVGPU_UTILS_MMAUTILS_H

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace nvgpu {

/// Represents the role of an operand in an MMA instruction:
/// `result := matmul(A, B) + C`
enum class MatMulOperandRole : int32_t { A = 0, B, C };

/// Returns the first user of the `op` that is vector.contract. If no
/// vector.contract user exists, return failure.
FailureOr<vector::ContractionOp> getUserContract(Operation *op);

/// Collects information about a warp-level matrix operand represented by a
/// VectorType.
struct WarpMatrixInfo {
  VectorType vectorType;
  MatMulOperandRole operandRole;
};

/// If `op` is a `vector.transfer_write`, return the `WarpMatrixInfo` for the
/// vector operand. If op is a `vector.transfer_read`, `vector.contraction`, or
/// `arith.constant`, return the `WarpMatrixInfo` corresponding to the result.
/// Otherwise, return failure.
FailureOr<WarpMatrixInfo> getWarpMatrixInfo(Operation *op);

/// Returns the number of bits in a single tile row. It is either 128, 256, or
/// 512 bits depending on the data type and` whether the operand is an
/// accumulator/result operand
int64_t inferTileWidthInBits(const WarpMatrixInfo &type);

/// Specifies information about the registers which compose a matrix fragment
/// according to the PTX documentation.
struct FragmentElementInfo {
  Type registerLLVMType;
  int64_t elementsPerRegister;
  int64_t registerWidthBits;
  int64_t numRegistersPerFragment;
};

/// Returns a FragmentElementInfo struct describing the register types for the
/// given matrix fragment type.
FailureOr<FragmentElementInfo>
getMmaSyncRegisterType(const WarpMatrixInfo &type);

/// Returns an AffineMap which maps a two dimensions representing (laneId,
/// logicalValueId) and returns two results representing offsets within a
/// matrix operand. The offsets point to the values the thread is responsible
/// for (AKA the matrix fragment values) during a warp-collective matrix
/// operation. For a visual reference of this LaneId -> (row, col) mapping,
/// please see NVIDIA's PTX documentation:
/// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-mma
FailureOr<AffineMap>
getLaneIdAndValueIdToOperandCoord(OpBuilder &builder, Location loc,
                                  const WarpMatrixInfo &fragmentType);

/// Encapsulates the parameters needed to lower a `nvgpu.ldmatrix` operation to
/// `nvvm.ldmatrix`.
struct LdMatrixParams {
  VectorType fragmentType;
  bool isAccum;
  int64_t numTiles;
  vector::IteratorType contiguousDimType;
  NVVM::MMALayout targetLayout;
};

/// Given `type` that contains info for a warp-matrix operand and whether or not
/// the load is a transposed load, return the LdMatrixParams.
FailureOr<LdMatrixParams> getLdMatrixParams(const WarpMatrixInfo &type,
                                            bool transpose);
/// Returns an AffineMap which maps a single dimension representing the laneId
/// to two results representing offsets within the matrix operand that should
/// be the pointer locations a thread should pass to the ldmatrix instruction.
FailureOr<AffineMap>
getLaneIdToLdMatrixMatrixCoord(OpBuilder &builder, Location loc,
                               const LdMatrixParams &params);

/// Transform `vector.contract` into (m,k)x(n,k)x(m,n) form so that it can be
/// converted to `nvgpu.mma.sync`. This specific form is meant to indicate that
/// the vector operands are organized such that the reduction dimension is
/// contiguous.
struct PrepareContractToGPUMMASync
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace nvgpu
} // namespace mlir

#endif // MLIR_DIALECT_NVGPU_UTILS_MMAUTILS_H
