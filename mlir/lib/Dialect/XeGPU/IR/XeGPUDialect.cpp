//===- XeGPUDialect.cpp - MLIR XeGPU dialect implementation -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace xegpu {

void XeGPUDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include <mlir/Dialect/XeGPU/IR/XeGPUTypes.cpp.inc>
      >();
  addOperations<
#define GET_OP_LIST
#include <mlir/Dialect/XeGPU/IR/XeGPU.cpp.inc>
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include <mlir/Dialect/XeGPU/IR/XeGPUAttrs.cpp.inc>
      >();
}

//===----------------------------------------------------------------------===//
// XeGPU_BlockTensorDescAttr
//===----------------------------------------------------------------------===//
BlockTensorDescAttr BlockTensorDescAttr::get(mlir::MLIRContext *context,
                                             xegpu::MemorySpace memory_space,
                                             int array_length,
                                             bool boundary_check) {
  auto scopeAttr = MemorySpaceAttr::get(context, memory_space);
  auto lengthAttr =
      IntegerAttr::get(IntegerType::get(context, 64), array_length);
  auto boundaryAttr = BoolAttr::get(context, boundary_check);
  return Base::get(context, scopeAttr, lengthAttr, boundaryAttr);
}

//===----------------------------------------------------------------------===//
// XeGPU_ScatterTensorDescAttr
//===----------------------------------------------------------------------===//
ScatterTensorDescAttr
ScatterTensorDescAttr::get(mlir::MLIRContext *context,
                           xegpu::MemorySpace memory_space, int chunk_size) {
  auto scopeAttr = MemorySpaceAttr::get(context, memory_space);
  auto chunkSizeAttr =
      IntegerAttr::get(IntegerType::get(context, 64), chunk_size);
  return Base::get(context, scopeAttr, chunkSizeAttr);
}

//===----------------------------------------------------------------------===//
// XeGPU_TensorDescType
//===----------------------------------------------------------------------===//

mlir::Type TensorDescType::parse(::mlir::AsmParser &parser) {
  llvm::SmallVector<int64_t> shape;
  mlir::Type elementType;
  mlir::FailureOr<mlir::Attribute> encoding;

  // Parse literal '<'
  if (parser.parseLess())
    return {};

  auto shapeLoc = parser.getCurrentLocation();
  if (mlir::failed(parser.parseDimensionList(shape))) {
    parser.emitError(shapeLoc, "failed to parse parameter 'shape'");
    return {};
  }

  auto elemTypeLoc = parser.getCurrentLocation();
  if (mlir::failed(parser.parseType(elementType))) {
    parser.emitError(elemTypeLoc, "failed to parse parameter 'elementType'");
    return {};
  }

  // parse optional attributes
  if (mlir::succeeded(parser.parseOptionalComma())) {
    encoding = mlir::FieldParser<mlir::Attribute>::parse(parser);
    if (mlir::failed(encoding)) {
      parser.emitError(
          parser.getCurrentLocation(),
          "Failed to parse the attribute field for TensorDescType.\n");
      return {};
    }
  }

  // Parse literal '>'
  if (parser.parseGreater())
    return {};

  return TensorDescType::get(parser.getContext(), shape, elementType,
                             encoding.value_or(mlir::Attribute()));
}

void TensorDescType::print(::mlir::AsmPrinter &printer) const {
  printer << "<";

  auto shape = getShape();
  for (int64_t dim : shape) {
    if (mlir::ShapedType::isDynamic(dim))
      printer << '?';
    else
      printer << dim;
    printer << 'x';
  }

  printer << getElementType();

  if (auto encoding = getEncoding())
    printer << ", " << encoding;

  printer << ">";
}

TensorDescType TensorDescType::get(llvm::ArrayRef<int64_t> shape,
                                   mlir::Type elementType, int array_length,
                                   bool boundary_check,
                                   MemorySpace memory_space) {
  auto context = elementType.getContext();
  auto attr = BlockTensorDescAttr::get(context, memory_space, array_length,
                                       boundary_check);
  return Base::get(context, shape, elementType, attr);
}

TensorDescType TensorDescType::get(llvm::ArrayRef<int64_t> shape,
                                   mlir::Type elementType, int chunk_size,
                                   MemorySpace memory_space) {
  auto context = elementType.getContext();
  auto attr = ScatterTensorDescAttr::get(context, memory_space, chunk_size);
  return Base::get(context, shape, elementType, attr);
}

} // namespace xegpu
} // namespace mlir

#include <mlir/Dialect/XeGPU/IR/XeGPUDialect.cpp.inc>
#define GET_ATTRDEF_CLASSES
#include <mlir/Dialect/XeGPU/IR/XeGPUAttrs.cpp.inc>
#define GET_TYPEDEF_CLASSES
#include <mlir/Dialect/XeGPU/IR/XeGPUTypes.cpp.inc>
