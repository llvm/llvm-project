//===- BufferizationConversionInterface.cpp - Dialect Interface  ---=------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/IR/BufferizationConversionInterface.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h" // getTensorTypeFromMemRefType

namespace mlir {
namespace bufferization {

FailureOr<BufferLikeType> ConversionInterface::getBufferType(
    Value value, const BufferizationOptions &options,
    const BufferizationState &state,
    function_ref<InFlightDiagnostic(const Twine &)> emitError) const {
  Dialect *dialect = &value.getType().getDialect();
  if (const ConversionDialectInterface *iface = getInterfaceFor(dialect))
    return iface->getBufferType(value, options, state, emitError);

  // Fall back to tensor -> memref conversion.
  auto memSpace =
      options.defaultMemorySpaceFn(cast<TensorType>(value.getType()));
  if (!memSpace.has_value())
    return emitError("could not infer memory space");

  return cast<BufferLikeType>(
      getMemRefType(value, options, /*layout=*/{}, *memSpace));
}

LogicalResult ConversionInterface::typesMatch(
    TensorLikeType tensor, BufferLikeType buffer,
    function_ref<InFlightDiagnostic(const Twine &)> emitError) const {
  Dialect *dialect = &tensor.getDialect();
  if (const ConversionDialectInterface *iface = getInterfaceFor(dialect))
    return iface->typesMatch(tensor, buffer, emitError);

  // Fall back to tensor, memref checking.
  assert(isa<TensorType>(tensor) && "expected tensor type");
  assert(isa<BaseMemRefType>(buffer) && "expected memref type");

  if (cast<ShapedType>(tensor).getShape() !=
      cast<ShapedType>(buffer).getShape()) {
    return emitError("shapes do not match");
  }

  if (cast<ShapedType>(tensor).getElementType() !=
      cast<ShapedType>(buffer).getElementType()) {
    return emitError("element types do not match");
  }

  return success();
}

} // namespace bufferization
} // namespace mlir
