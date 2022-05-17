//===- NvvmMMASupport.h - MLIR Vector to GPU lowering support --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utilities to assist in the lowering of Vector operations
// to GPU dialect MMA operations.
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_VECTORTOGPU_NVGPUSUPPORT_H
#define MLIR_CONVERSION_VECTORTOGPU_NVGPUSUPPORT_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace nvgpu {

enum class MatMulOperandRole : int32_t { A = 0, B, C };

/// Collects information about a warp-level matrix operand represented by a
/// VectorType.
struct WarpMatrixInfo {
  VectorType vectorType;
  MatMulOperandRole operandRole;
};

/// Given an op that operates on a VectorType representing a warp-level matrix
/// operand, the function returns a struct containing relevant type information.
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
getLaneIdAndValueIdToOperandCoord(Location loc, OpBuilder &builder,
                                  const WarpMatrixInfo &fragmentType);

struct LdMatrixParams {
  VectorType fragmentType;
  bool isAccum;
  int64_t numTiles;
  IteratorType contiguousDimType;
  NVVM::MMALayout targetLayout;
};

FailureOr<LdMatrixParams> getLdMatrixParams(const WarpMatrixInfo &type,
                                            bool transpose);
/// Returns an AffineMap which maps a single dimension representing the laneId
/// to two results representing offsets within the matrix operand that should
/// be the pointer locations a thread should pass to the ldmatrix instruction.
FailureOr<AffineMap>
getLaneIdToLdMatrixMatrixCoord(Location loc, OpBuilder &builder,
                               const LdMatrixParams &params);

// Transform contract into (m, k)x(n, k)x(m, n) form so that it can be converted
// to MMA matmul.
struct PrepareContractToGPUMMASync
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override;
};

} // namespace nvgpu
} // namespace mlir

#endif // MLIR_CONVERSION_VECTORTOGPU_NVGPUSUPPORT_H
