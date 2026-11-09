//===- VectorToGPUUtils.h - Shared vector-to-MMA helpers -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_VECTORTOGPUCOMMON_VECTORTOGPUUTILS_H
#define MLIR_CONVERSION_VECTORTOGPUCOMMON_VECTORTOGPUUTILS_H

#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include <cstdint>
#include <optional>

namespace mlir {
class Operation;
class RewriterBase;
class RewritePatternSet;

namespace arith {
class ConstantOp;
class ExtFOp;
class ExtSIOp;
class ExtUIOp;
class TruncFOp;
} // namespace arith

namespace gpu {
enum class MMAElementwiseOp : uint32_t;
} // namespace gpu

namespace scf {
class ForOp;
class YieldOp;
} // namespace scf

namespace vector {
class BroadcastOp;
} // namespace vector

/// Return true if the constant is a splat to a 2D vector so that it can be
/// converted to an MMA constant matrix op.
bool constantSupportsMMAMatrixType(arith::ConstantOp constantOp);

/// Return true if this is a broadcast from scalar to a 2D vector.
bool broadcastSupportsMMAMatrixType(vector::BroadcastOp broadcastOp);

/// Return true if this integer extend op can be folded into a contract op.
/// The operation must be arith.extsi or arith.extui.
template <typename ExtOpTy>
bool integerExtendSupportsMMAMatrixType(ExtOpTy extOp);

extern template bool
    integerExtendSupportsMMAMatrixType<arith::ExtSIOp>(arith::ExtSIOp);

extern template bool
    integerExtendSupportsMMAMatrixType<arith::ExtUIOp>(arith::ExtUIOp);

/// Return true if the floating-point extension is supported.
bool fpExtendSupportsMMAMatrixType(arith::ExtFOp extOp);

/// Return true if the floating-point truncation is supported.
bool fpTruncSupportsMMAMatrixType(arith::TruncFOp extOp);

/// Return the MMA elementwise enum associated with the operation if supported.
/// Return std::nullopt otherwise.
std::optional<gpu::MMAElementwiseOp> convertElementwiseOpToMMA(Operation *op);

/// Return true if the operation is supported as elementwise on an MMA matrix.
bool elementwiseSupportsMMAMatrixType(Operation *op);

/// Extend loop-carried values with their converted counterparts.
LogicalResult convertForOp(RewriterBase &rewriter, scf::ForOp op,
                           llvm::DenseMap<Value, Value> &valueMapping);

/// Update a yield operation to yield the converted values.
LogicalResult convertYieldOp(RewriterBase &rewriter, scf::YieldOp op,
                             llvm::DenseMap<Value, Value> &valueMapping);

/// Populate patterns that fold vector.transpose into vector.transfer_read,
/// including through arithmetic extension operations.
void populateVectorTransferReadTransposePatterns(RewritePatternSet &patterns);
} // namespace mlir

#endif // MLIR_CONVERSION_VECTORTOGPUCOMMON_VECTORTOGPUUTILS_H