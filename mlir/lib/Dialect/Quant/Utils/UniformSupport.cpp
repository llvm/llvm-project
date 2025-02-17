//===- UniformSupport.cpp - Support utilities for uniform quant -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Quant/Utils/UniformSupport.h"
#include "mlir/IR/BuiltinTypes.h"
#include <numeric>

using namespace mlir;
using namespace mlir::quant;

static bool isQuantizablePrimitiveType(Type inputType) {
  return isa<FloatType>(inputType);
}

ExpressedToQuantizedConverter
ExpressedToQuantizedConverter::forInputType(Type inputType) {
  if (isa<TensorType, VectorType>(inputType)) {
    Type elementType = cast<ShapedType>(inputType).getElementType();
    if (!isQuantizablePrimitiveType(elementType))
      return ExpressedToQuantizedConverter{inputType, nullptr};
    return ExpressedToQuantizedConverter{inputType, elementType};
  }
  // Supported primitive type (which just is the expressed type).
  if (isQuantizablePrimitiveType(inputType))
    return ExpressedToQuantizedConverter{inputType, inputType};
  // Unsupported.
  return ExpressedToQuantizedConverter{inputType, nullptr};
}

Type ExpressedToQuantizedConverter::convert(QuantizedType elementalType) const {
  assert(expressedType && "convert() on unsupported conversion");
  if (auto tensorType = dyn_cast<RankedTensorType>(inputType))
    return RankedTensorType::get(tensorType.getShape(), elementalType);
  if (dyn_cast<UnrankedTensorType>(inputType))
    return UnrankedTensorType::get(elementalType);
  if (auto vectorType = dyn_cast<VectorType>(inputType))
    return VectorType::get(vectorType.getShape(), elementalType);

  // If the expressed types match, just use the new elemental type.
  if (elementalType.getExpressedType() == expressedType)
    return elementalType;
  // Unsupported.
  return nullptr;
}

ElementsAttr
UniformQuantizedPerAxisValueConverter::convert(Attribute realValue) {
  if (auto attr = dyn_cast<DenseFPElementsAttr>(realValue)) {
    return convert(attr);
  }
  // TODO: handles sparse elements attribute
  return nullptr;
}

DenseElementsAttr
UniformQuantizedPerAxisValueConverter::convert(DenseFPElementsAttr attr) {
  // Creates the converter for each chunk. Normally the size of the
  // quantization dim is 3, so we can cache all the converters.
  ShapedType type = attr.getType();
  size_t dimSize = type.getDimSize(quantizationDim);
  if (dimSize != scales.size()) {
    return {};
  }
  SmallVector<UniformQuantizedValueConverter, 4> converters;
  converters.reserve(dimSize);
  for (int i = 0, e = dimSize; i != e; ++i) {
    converters.push_back(getPerChunkConverter(i));
  }

  // Scan the elements of the dense elements attributes and quantize them by
  // using the right quantization parameters.
  int64_t flattenIndex = 0;
  auto shape = type.getShape();
  int64_t chunkSize =
      std::accumulate(std::next(shape.begin(), quantizationDim + 1),
                      shape.end(), 1, std::multiplies<int64_t>());
  Type newElementType = IntegerType::get(attr.getContext(), storageBitWidth);
  return attr.mapValues(newElementType, [&](const APFloat &old) {
    int chunkIndex = (flattenIndex++) / chunkSize;
    return converters[chunkIndex % dimSize].quantizeFloatToInt(old);
  });
}
