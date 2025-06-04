//===- BufferizationConversionInterface.h - Dialect Interface ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_IR_BUFFERIZATIONCONVERSIONINTERFACE_H_
#define MLIR_DIALECT_BUFFERIZATION_IR_BUFFERIZATIONCONVERSIONINTERFACE_H_

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizationTypeInterfaces.h"
#include "mlir/IR/DialectInterface.h"

namespace mlir {
namespace bufferization {

/// This class defines a virtual interface for conversions between tensor-like
/// and buffer-like types.
struct ConversionDialectInterface
    : DialectInterface::Base<ConversionDialectInterface> {
  using Base::Base;

  /// Hook to customize tensor-like -> buffer-like conversion within a given
  /// dialect. Returns a buffer-like type for the specific tensor-like type.
  virtual FailureOr<BufferLikeType> getBufferType(
      Value value, const BufferizationOptions &options,
      const BufferizationState &state,
      function_ref<InFlightDiagnostic(const Twine &)> emitError) const = 0;

  /// Hook to customize type checking between tensor-like and buffer-like types.
  /// Given tensor `T` and buffer `B = getBufferType(T, ...)`, the call to
  /// `typesMatch(T, B)` must return true.
  virtual LogicalResult typesMatch(
      TensorLikeType tensor, BufferLikeType buffer,
      function_ref<InFlightDiagnostic(const Twine &)> emitError) const = 0;

  /// Hook to customize buffer-like -> tensor-like conversion, which is the
  /// opposite of bufferization.
  virtual TensorLikeType getTensorFromBuffer(BufferLikeType buffer) const = 0;
};

/// Interface collection for conversion between tensor-like and buffer-like
/// types, dispatches to a concrete interface implementation based on the
/// dialect to which the given type belongs.
struct ConversionInterface
    : DialectInterfaceCollection<ConversionDialectInterface> {
  using Base::Base;

  /// Dispatches to ConversionDialectInterface::getBufferType() of the dialect
  /// associated with the value type.
  FailureOr<BufferLikeType> getBufferType(
      Value value, const BufferizationOptions &options,
      const BufferizationState &state,
      function_ref<InFlightDiagnostic(const Twine &)> emitError) const;

  /// Dispatches to ConversionDialectInterface::typesMatch() of the dialect
  /// associated with the value type.
  LogicalResult
  typesMatch(TensorLikeType tensor, BufferLikeType buffer,
             function_ref<InFlightDiagnostic(const Twine &)> emitError) const;

  /// Dispatches to ConversionDialectInterface::getTensorFromBuffer() of the
  /// dialect associated with the value type.
  TensorLikeType getTensorFromBuffer(BufferLikeType buffer) const;
};

} // namespace bufferization
} // namespace mlir

#endif // MLIR_DIALECT_BUFFERIZATION_IR_BUFFERIZATIONTYPEINTERFACES_H_
