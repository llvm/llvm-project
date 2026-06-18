//===- OpenACCParMapping.h - OpenACC Parallelism Mapping -------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for mapping OpenACC parallelism levels
// (gang, worker, vector) to target-specific parallel dimension attributes.
//
// Users can provide custom implementations of ACCParMappingPolicy to
// support different mapping strategies and target attributes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENACC_OPENACCPARMAPPING_H_
#define MLIR_DIALECT_OPENACC_OPENACCPARMAPPING_H_

#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace acc {

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Convert a gang dimension value (1, 2, or 3) to the corresponding ParLevel.
/// Asserts if the value is not a valid gang dimension.
inline ParLevel getGangParLevel(int64_t gangDimValue) {
  assert((gangDimValue >= 1 && gangDimValue <= 3) &&
         "gang dimension must be 1, 2, or 3");
  switch (gangDimValue) {
  case 1:
    return ParLevel::gang_dim1;
  case 2:
    return ParLevel::gang_dim2;
  case 3:
    return ParLevel::gang_dim3;
  }
  llvm_unreachable("validated gang dimension");
}

//===----------------------------------------------------------------------===//
// ACCParMappingPolicy
//===----------------------------------------------------------------------===//

/// Policy class that defines how OpenACC parallelism levels map to
/// target-specific parallel dimension attributes. Implementations provide the
/// actual mapping.
///
/// Template parameter ParDimAttrT specifies the attribute type returned by
/// the mapping functions (e.g., mlir::acc::GPUParallelDimAttr for GPU targets).
///
/// This policy allows different mapping strategies:
/// - Standard GPU mapping (gang->block, worker->threadY, vector->threadX)
/// - Custom mappings for specific targets or optimization strategies
///
/// Pass an implementation to functions that need to perform the mapping.
template <typename ParDimAttrT>
class ACCParMappingPolicy {
public:
  virtual ~ACCParMappingPolicy() = default;

  /// Map an OpenACC parallelism level to target dimension.
  /// @param ctx The MLIR context
  /// @param level The OpenACC parallelism level (gang_dim1, gang_dim2,
  ///              gang_dim3, worker, vector, or seq)
  /// @return The corresponding parallel dimension attribute
  virtual ParDimAttrT map(MLIRContext *ctx, ParLevel level) const = 0;

  /// Convenience methods for specific parallelism levels.
  ParDimAttrT gangDim(MLIRContext *ctx, ParLevel level) const {
    assert((level == ParLevel::gang_dim1 || level == ParLevel::gang_dim2 ||
            level == ParLevel::gang_dim3) &&
           "gangDim requires a gang parallelism level");
    return map(ctx, level);
  }
  ParDimAttrT workerDim(MLIRContext *ctx) const {
    return map(ctx, ParLevel::worker);
  }
  ParDimAttrT vectorDim(MLIRContext *ctx) const {
    return map(ctx, ParLevel::vector);
  }
  ParDimAttrT seqDim(MLIRContext *ctx) const { return map(ctx, ParLevel::seq); }

  //===--------------------------------------------------------------------===//
  // Predicate methods - check if an attribute matches a parallelism level
  //===--------------------------------------------------------------------===//

  /// Check if the attribute represents vector parallelism.
  virtual bool isVector(ParDimAttrT attr) const = 0;

  /// Check if the attribute represents worker parallelism.
  virtual bool isWorker(ParDimAttrT attr) const = 0;

  /// Check if the attribute represents gang parallelism (any gang dimension).
  virtual bool isGang(ParDimAttrT attr) const = 0;

  /// Check if the attribute represents sequential execution.
  virtual bool isSeq(ParDimAttrT attr) const = 0;
};

//===----------------------------------------------------------------------===//
// DefaultACCToGPUMappingPolicy
//===----------------------------------------------------------------------===//

/// Default policy that provides the standard GPU mapping:
///   gang(dim:1) -> BlockX (gridDim.x / blockIdx.x)
///   gang(dim:2) -> BlockY (gridDim.y / blockIdx.y)
///   gang(dim:3) -> BlockZ (gridDim.z / blockIdx.z)
///   worker      -> ThreadY (blockDim.y / threadIdx.y)
///   vector      -> ThreadX (blockDim.x / threadIdx.x)
///   seq         -> Sequential
class DefaultACCToGPUMappingPolicy
    : public ACCParMappingPolicy<mlir::acc::GPUParallelDimAttr> {
public:
  mlir::acc::GPUParallelDimAttr map(MLIRContext *ctx,
                                    ParLevel level) const override {
    switch (level) {
    case ParLevel::gang_dim1:
      return mlir::acc::GPUParallelDimAttr::blockXDim(ctx);
    case ParLevel::gang_dim2:
      return mlir::acc::GPUParallelDimAttr::blockYDim(ctx);
    case ParLevel::gang_dim3:
      return mlir::acc::GPUParallelDimAttr::blockZDim(ctx);
    case ParLevel::worker:
      return mlir::acc::GPUParallelDimAttr::threadYDim(ctx);
    case ParLevel::vector:
      return mlir::acc::GPUParallelDimAttr::threadXDim(ctx);
    case ParLevel::seq:
      return mlir::acc::GPUParallelDimAttr::seqDim(ctx);
    }
    llvm_unreachable("Unknown ParLevel");
  }

  bool isVector(mlir::acc::GPUParallelDimAttr attr) const override {
    return attr.isThreadX();
  }

  bool isWorker(mlir::acc::GPUParallelDimAttr attr) const override {
    return attr.isThreadY();
  }

  bool isGang(mlir::acc::GPUParallelDimAttr attr) const override {
    return attr.isAnyBlock();
  }

  bool isSeq(mlir::acc::GPUParallelDimAttr attr) const override {
    return attr.isSeq();
  }
};

/// Type alias for the GPU-specific mapping policy
using ACCToGPUMappingPolicy =
    ACCParMappingPolicy<mlir::acc::GPUParallelDimAttr>;

} // namespace acc
} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_OPENACCPARMAPPING_H_
