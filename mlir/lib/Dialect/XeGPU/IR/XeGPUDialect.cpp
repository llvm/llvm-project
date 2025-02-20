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

LogicalResult ScatterTensorDescAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    MemorySpaceAttr memory_space, IntegerAttr chunk_size) {
  int64_t chunkSize = chunk_size.getInt();
  SmallVector<int64_t> supportedChunkSizes = {1,  2,  3,  4,   8,
                                              16, 32, 64, 128, 256};
  if (!llvm::is_contained(supportedChunkSizes, chunkSize))
    return emitError() << "invalid chunk size";

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_SGMapAttr
//===----------------------------------------------------------------------===//
namespace {
template <typename T, unsigned N>
LogicalResult parseIntArrayField(::mlir::AsmParser &parser,
                                 llvm::SmallVector<T, N> &result,
                                 llvm::StringRef fieldName) {
  if (failed(parser.parseKeyword(fieldName))) {
    parser.emitError(parser.getCurrentLocation(),
                     "unexpected field name. Expected " + fieldName + ".");
    return failure();
  }

  if (failed(parser.parseEqual())) {
    parser.emitError(parser.getCurrentLocation(), "expected '=' sign.");
    return failure();
  }

  auto elemParser = [&]() -> llvm::ParseResult {
    uint32_t elem = 0;
    auto res = parser.parseInteger(elem);
    result.push_back(elem);
    return res;
  };

  return parser.parseCommaSeparatedList(AsmParser::Delimiter::Square,
                                        elemParser, fieldName);
}
} // namespace

mlir::Attribute SGMapAttr::parse(::mlir::AsmParser &parser,
                                 ::mlir::Type attrType) {
  if (failed(parser.parseLess()))
    return {};

  llvm::SmallVector<uint32_t, 2> wi_layout, wi_data;
  if (failed(parseIntArrayField(parser, wi_layout, "wi_layout")))
    return {};

  if (failed(parser.parseComma()))
    return {};

  if (failed(parseIntArrayField(parser, wi_data, "wi_data")))
    return {};

  return SGMapAttr::getChecked(
      [&]() { return parser.emitError(parser.getNameLoc()); },
      parser.getContext(), wi_layout, wi_data);
}

void SGMapAttr::print(::mlir::AsmPrinter &printer) const {
  printer << "<";
  printer.printKeywordOrString("wi_layout");
  printer << " = [" << getWiLayout() << "], ";
  printer.printKeywordOrString("wi_data");
  printer << " = [" << getWiData() << "]";
  printer << ">";
}

LogicalResult
SGMapAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                  llvm::ArrayRef<uint32_t> wi_layout,
                  llvm::ArrayRef<uint32_t> wi_data) {
  if (wi_layout.size() != 2)
    return emitError() << "expected wi_layout of size 2";
  if (wi_data.size() != 2)
    return emitError() << "expected wi_data of size 2";
  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_TensorDescType
//===----------------------------------------------------------------------===//

mlir::Type TensorDescType::parse(::mlir::AsmParser &parser) {
  llvm::SmallVector<int64_t> shape;
  mlir::Type elementType;
  mlir::FailureOr<mlir::Attribute> encoding;
  mlir::FailureOr<mlir::Attribute> sg_map;

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
  while (mlir::succeeded(parser.parseOptionalComma())) {
    mlir::Attribute attr;
    ParseResult res = parser.parseAttribute(attr);
    if (mlir::succeeded(res)) {
      if (mlir::isa<SGMapAttr>(attr)) {
        sg_map = attr;
        continue;
      }
      if (mlir::isa<BlockTensorDescAttr, ScatterTensorDescAttr>(attr)) {
        encoding = attr;
        continue;
      }
    }
    return {};
  }

  // Parse literal '>'
  if (parser.parseGreater())
    return {};

  return TensorDescType::getChecked(
      [&]() { return parser.emitError(parser.getNameLoc()); },
      parser.getContext(), shape, elementType,
      encoding.value_or(mlir::Attribute()), sg_map.value_or(mlir::Attribute()));
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

  if (auto sg_map = getSgMap())
    printer << ", " << sg_map;

  printer << ">";
}

TensorDescType TensorDescType::get(llvm::ArrayRef<int64_t> shape,
                                   mlir::Type elementType, int array_length,
                                   bool boundary_check,
                                   MemorySpace memory_space,
                                   mlir::Attribute sg_map) {
  auto context = elementType.getContext();
  auto attr = BlockTensorDescAttr::get(context, memory_space, array_length,
                                       boundary_check);
  return Base::get(context, shape, elementType, attr, sg_map);
}

TensorDescType TensorDescType::get(llvm::ArrayRef<int64_t> shape,
                                   mlir::Type elementType, int chunk_size,
                                   MemorySpace memory_space,
                                   mlir::Attribute sg_map) {
  auto context = elementType.getContext();
  auto attr = ScatterTensorDescAttr::get(context, memory_space, chunk_size);
  return Base::get(context, shape, elementType, attr, sg_map);
}

LogicalResult TensorDescType::verify(
    llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape, mlir::Type elementType,
    mlir::Attribute encoding, mlir::Attribute sg_map) {
  size_t rank = shape.size();
  if (rank != 1 && rank != 2)
    return emitError() << "expected 1D or 2D tensor";

  auto scatterAttr = mlir::dyn_cast_if_present<ScatterTensorDescAttr>(encoding);
  if (scatterAttr) {
    // Expected tensor ranks for scattered data:
    //   - 1D tensor for fully non-contiguous elements (chunk size == 1)
    //   - 2D tensor for scattered blocks (chunk size > 1)
    unsigned chunkSize = scatterAttr.getChunkSize().getInt();
    if (rank == 1 && chunkSize != 1)
      return emitError() << "expected non-contiguous elements for 1D tensor";
    if (rank == 2 && chunkSize < 2)
      return emitError() << "expected chunk blocks for 2D tensor";
  }

  if (auto blockAttr =
          mlir::dyn_cast_if_present<BlockTensorDescAttr>(encoding)) {
    MemorySpaceAttr memorySpaceAttr = blockAttr.getMemorySpace();
    if (rank == 2 && memorySpaceAttr &&
        memorySpaceAttr.getValue() == MemorySpace::SLM)
      return emitError() << "SLM is not supported for 2D block tensor";
  }

  if (auto sgMapAttr = llvm::dyn_cast_if_present<SGMapAttr>(sg_map)) {
    ArrayRef<uint32_t> wiLayout = sgMapAttr.getWiLayout();
    ArrayRef<uint32_t> wiData = sgMapAttr.getWiData();

    if (rank == 1) {
      if (wiLayout[0] != 1 || wiData[0] != 1)
        return emitError()
               << "outer layout distribution and data mapping must be 1 "
                  "for 1D tensor";
    }

    if (scatterAttr) {
      // Validate subgroup mapping rules for scattered tensors.
      // A work-item's slice of the tensor with shape [sg_size] or
      // [sg_size, chunk_size] will be [1] or [1, chunks_size] respectively,
      // the mapping should reflect that.
      if (wiData[0] != 1)
        return emitError()
               << "cannot map over non-contiguous scattered row elements";

      unsigned chunkSize = scatterAttr.getChunkSize().getInt();
      if (wiData[1] != chunkSize)
        return emitError() << "work item data mapping must match the number of "
                              "contiguous elements";
    }

    // For 1D tensor, pad the shape with an outer unit dimension to allow common
    // validation logic.
    SmallVector<int64_t> tensorShape(shape.begin(), shape.end());
    if (rank == 1)
      tensorShape = {1, tensorShape.back()};

    size_t dims = tensorShape.size();
    for (size_t i = 0; i < dims; ++i) {
      uint32_t numElemPerWi = wiLayout[i] * wiData[i];
      if (tensorShape[i] < numElemPerWi || tensorShape[i] % numElemPerWi != 0)
        return emitError() << "cannot distribute " << tensorShape[i] << " over "
                           << wiLayout[i] << " work items with " << wiData[i]
                           << " elements each";
    }
  }

  return success();
}

} // namespace xegpu
} // namespace mlir

#include <mlir/Dialect/XeGPU/IR/XeGPUDialect.cpp.inc>
#define GET_ATTRDEF_CLASSES
#include <mlir/Dialect/XeGPU/IR/XeGPUAttrs.cpp.inc>
#define GET_TYPEDEF_CLASSES
#include <mlir/Dialect/XeGPU/IR/XeGPUTypes.cpp.inc>
