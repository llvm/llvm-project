//===- Bufferize.h - Bufferization Utilities --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// We use the term "bufferize" to mean conversion from tensor types to
// memref types.
//
// Generally speaking, for each op that operates on tensor types, the
// `BufferizableOpInterface` needs to be implemented. This file contains the
// bufferization driver that is responsible for bufferizing the ops in the right
// order, etc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_BUFFERIZE_H
#define MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_BUFFERIZE_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace bufferization {

class AnalysisState;
struct BufferizationState;
struct BufferizationOptions;
class OpFilter;

/// A helper type converter class that automatically populates the relevant
/// materializations and type conversions for bufferization.
class BufferizeTypeConverter : public TypeConverter {
public:
  BufferizeTypeConverter();
};

/// Marks ops used by bufferization for type conversion materializations as
/// "legal" in the given ConversionTarget.
///
/// This function should be called by all bufferization passes using
/// BufferizeTypeConverter so that materializations work properly. One exception
/// is bufferization passes doing "full" conversions, where it can be desirable
/// for even the materializations to remain illegal so that they are eliminated,
/// such as via the patterns in
/// populateEliminateBufferizeMaterializationsPatterns.
void populateBufferizeMaterializationLegality(ConversionTarget &target);

/// Populate patterns to eliminate bufferize materializations.
///
/// In particular, these are the tensor_load/buffer_cast ops.
void populateEliminateBufferizeMaterializationsPatterns(
    BufferizeTypeConverter &typeConverter, RewritePatternSet &patterns);

/// Bufferize `op` and its nested ops that implement `BufferizableOpInterface`.
/// Whether buffer copies are needed or not is queried from `state`.
///
/// Note: If `allowUnknownOps` is set to false, bufferization fails when an
/// unknown op (that does not implement `BufferizableOpInterface`) is found. No
/// to_tensor/to_memref ops are inserted in that case.
///
/// Note: The layout map chosen to bufferize is the most dynamic canonical
/// strided layout of the proper rank. This ensures compatibility with expected
/// layouts after transformations. Combinations of memref.cast +
/// canonicalization are responsible for clean ups.
// TODO: Extract `options` from `state` and pass as separate argument.
LogicalResult bufferizeOp(Operation *op, const AnalysisState &analysisState);

/// Bufferize `op` and its nested ops that implement `BufferizableOpInterface`.
/// Buffers are duplicated and copied before any tensor use that bufferizes to
/// a memory write.
///
/// Note: This function bufferizes ops without utilizing analysis results. It
/// can be used to implement partial bufferization passes.
LogicalResult bufferizeOp(Operation *op, const BufferizationOptions &options);

BufferizationOptions getPartialBufferizationOptions();

//===----------------------------------------------------------------------===//
// Helper functions for extending Bufferization
//===----------------------------------------------------------------------===//

/// Bufferize `op` and its nested ops that implement `BufferizableOpInterface`.
/// Reuse an existing `BufferizationState`.
///
/// Note: This function overload is useful for extending the bufferization.
LogicalResult bufferizeOp(Operation *op, BufferizationState &bufferizationState,
                          const OpFilter *opFilter = nullptr);

} // namespace bufferization
} // namespace mlir

#endif // MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_BUFFERIZE_H
