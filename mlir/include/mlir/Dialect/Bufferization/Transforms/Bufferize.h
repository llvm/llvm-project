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
/// If `copyBeforeWrite`, buffers are duplicated and copied before any tensor
/// use that bufferizes to a memory write.
///
/// Note: In the general case, it unsafe to run with `copyBeforeWrite = false`
/// because read-after-write conflicts may materialize during bufferization.
/// `copyBeforeWrite = false` is safe only if the input IR is guaranteed to
/// *not* require any out-of-place bufferization.
///
/// Note: This function bufferizes ops without utilizing analysis results. It
/// can be used to implement partial bufferization passes.
LogicalResult bufferizeOp(Operation *op, const BufferizationOptions &options,
                          bool copyBeforeWrite = true,
                          const OpFilter *opFilter = nullptr);

BufferizationOptions getPartialBufferizationOptions();

} // namespace bufferization
} // namespace mlir

#endif // MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_BUFFERIZE_H
