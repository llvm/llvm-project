//===- Bufferize.h - Bufferization utilities --------------------*- C++ -*-===//
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
// Generally speaking, for each op that operates on tensor types, a conversion
// pattern needs to be written. The infrastructure in this file assists in
// defining these conversion patterns in a composable way.
//
// Bufferization conversion patterns should generally use the ordinary
// conversion pattern classes (e.g. OpConversionPattern). A TypeConverter
// (accessible with getTypeConverter()) is available if needed for converting
// types.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_BUFFERIZE_H
#define MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_BUFFERIZE_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace bufferization {

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

} // namespace bufferization
} // namespace mlir

#endif // MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_BUFFERIZE_H
