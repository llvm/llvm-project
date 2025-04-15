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
// XeGPU_LayoutAttr
//===----------------------------------------------------------------------===//
LogicalResult
LayoutAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                   DenseI32ArrayAttr sg_layout, DenseI32ArrayAttr sg_data,
                   DenseI32ArrayAttr inst_data, DenseI32ArrayAttr lane_layout,
                   DenseI32ArrayAttr lane_data, DenseI32ArrayAttr order) {

  // A valid layout must include at least one of sg_layout and lane_layout.
  // sg_layout is essential for Workgroup layout, while lane_layout is
  // required for Subgroup layout.
  if (!sg_layout && !inst_data && !lane_layout) {
    return emitError()
           << "expected at least one of sg_layout, inst_data or lane_layout";
  }

  // generate code to check sg_laout, inst_data and lane_layout having the same
  // rank if they are not null.

  if (sg_layout && inst_data && sg_layout.size() != inst_data.size()) {
    return emitError()
           << "expected sg_layout and inst_data to have the same rank";
  }

  if (sg_layout && lane_layout && sg_layout.size() != lane_layout.size()) {
    return emitError()
           << "expected sg_layout and lane_layout to have the same rank";
  }

  if (inst_data && lane_layout && inst_data.size() != lane_layout.size()) {
    return emitError()
           << "expected inst_data and lane_layout to have the same rank";
  }

  // sg_data is optional for Workgroup layout, but its presence requires
  // sg_layout.
  if (sg_data) {
    if (!sg_layout)
      return emitError() << "expected sg_layout being used with sg_data";
    if (sg_data.size() != sg_layout.size())
      return emitError()
             << "expected sg_data and sg_layout to have the same rank";
  }

  // lane_data is optional for Subgroup layout, but its presence requires
  // lane_layout.
  if (lane_data) {
    if (!lane_layout)
      return emitError() << "expected lane_layout being used with lane_data";
    if (lane_data.size() != lane_layout.size())
      return emitError()
             << "expected lane_data and lane_layout to have the same rank";
  }

  if (order) {
    if (!sg_layout && !lane_layout)
      return emitError()
             << "expected sg_layout/lane_layout being used with order";

    if (sg_layout && order.size() != sg_layout.size())
      return emitError()
             << "expected order and sg_layout to have the same rank";

    if (lane_layout && order.size() != lane_layout.size())
      return emitError()
             << "expected order and lane_layout to have the same rank";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_TensorDescType
//===----------------------------------------------------------------------===//

mlir::Type TensorDescType::parse(::mlir::AsmParser &parser) {
  llvm::SmallVector<int64_t> shape;
  mlir::Type elementType;
  mlir::FailureOr<mlir::Attribute> encoding;
  mlir::FailureOr<mlir::Attribute> layout;

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
      if (mlir::isa<LayoutAttr>(attr)) {
        layout = attr;
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
      encoding.value_or(mlir::Attribute()), layout.value_or(mlir::Attribute()));
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

  if (auto layout = getLayout())
    printer << ", " << layout;

  printer << ">";
}

TensorDescType TensorDescType::get(llvm::ArrayRef<int64_t> shape,
                                   mlir::Type elementType, int array_length,
                                   bool boundary_check,
                                   MemorySpace memory_space,
                                   mlir::Attribute layout) {
  auto context = elementType.getContext();
  auto attr = BlockTensorDescAttr::get(context, memory_space, array_length,
                                       boundary_check);
  return Base::get(context, shape, elementType, attr, layout);
}

TensorDescType TensorDescType::get(llvm::ArrayRef<int64_t> shape,
                                   mlir::Type elementType, int chunk_size,
                                   MemorySpace memory_space,
                                   mlir::Attribute layout) {
  auto context = elementType.getContext();
  auto attr = ScatterTensorDescAttr::get(context, memory_space, chunk_size);
  return Base::get(context, shape, elementType, attr, layout);
}

LogicalResult TensorDescType::verify(
    llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape, mlir::Type elementType,
    mlir::Attribute encoding, mlir::Attribute layout) {
  size_t rank = shape.size();
  // Low-pressure types are packed in 32-bit units.
  int32_t packingFactor = 32 / elementType.getIntOrFloatBitWidth();
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
    // If chunk size > 1, the second dimension of the tensor shape must be
    // equal to chunk size and it must be a multiple of the packing factor.
    if (chunkSize > 1) {
      if (shape.back() != chunkSize)
        return emitError() << "expected tensor shape[1] to match chunk size";
      if (shape.back() % packingFactor != 0)
        return emitError()
               << "expected tensor shape[1] to be a multiple of packing factor "
               << packingFactor;
    }
  }

  if (auto blockAttr =
          mlir::dyn_cast_if_present<BlockTensorDescAttr>(encoding)) {
    MemorySpaceAttr memorySpaceAttr = blockAttr.getMemorySpace();
    if (rank == 2 && memorySpaceAttr &&
        memorySpaceAttr.getValue() == MemorySpace::SLM)
      return emitError() << "SLM is not supported for 2D block tensor";
  }

  if (auto layoutAttr = llvm::dyn_cast_if_present<LayoutAttr>(layout)) {

    if (rank != (size_t)layoutAttr.getRank())
      return emitError() << "expected layout rank to match tensor rank";

    ArrayRef<int32_t> laneLayout = layoutAttr.getLaneLayout().asArrayRef();
    ArrayRef<int32_t> laneData = layoutAttr.getLaneData().asArrayRef();

    if (scatterAttr) {
      // Validate subgroup mapping rules for scattered tensors.
      // A work-item's slice of the tensor with shape [sg_size] or
      // [sg_size, chunk_size] will be [1] or [1, 32/element_ty_bit_width]
      // respectively, the mapping should reflect that. This is because each
      // work item access data in 32 bit granularity.

      if (rank > 1 && laneData[0] != 1)
        return emitError()
               << "cannot map over non-contiguous scattered row elements";
      if (laneData.back() != packingFactor)
        return emitError() << "work item data mapping must match the number of "
                              "contiguous elements";
    }

    for (size_t i = 0; i < shape.size(); ++i) {
      uint32_t numElemPerWi = laneLayout[i] * laneData[i];
      if (shape[i] < numElemPerWi || shape[i] % numElemPerWi != 0)
        return emitError() << "cannot distribute " << shape[i] << " over "
                           << laneLayout[i] << " work items with "
                           << laneData[i] << " elements each";
    }
  }

  return success();
}

// If tensor descriptor has a layout attribute it is used in SIMT mode.
// In this mode, the distributed vector shape is determined as follows:
// Definitions:
//        lane_data_size = lane_data[0] × lane_data[1]
//        subgroup_size = lane_layout[0] × lane_layout[1]
//        distribution_unit_size = subgroup_size × lane_data_size
// ---------------------------------------------------------------------
// Case 1: Regular loads/stores.
// ---------------------------------------------------------------------
// Distributed vector shape must be:
//        [chunk_size / lane_data_size, lane_data_size]
// If the tensor descriptor shape is 1D, first dimension is ignored (set to 1).
//        [lane_data_size]
// ---------------------------------------------------------------------
// Case 2: Block loads/stores
// ---------------------------------------------------------------------
// Additional definitions:
//        tensor_size = tensor_desc[0] * .. * tensor_desc[r-1] * array_length
//        n_distribution_units = tensor_size / distribution_unit_size
// Given above definitions, the following conditions must be met:
//        * tensor_desc[0] % (lane_layout[0] × lane_data[0]) == 0
//        * tensor_desc[1] % (lane_layout[1] × lane_data[1]) == 0
// Distributed vector shape must be:
//        [n_distribution_units, lane_data_size]
FailureOr<VectorType> TensorDescType::getDistributedVectorType() {
  auto layout = llvm::dyn_cast_if_present<LayoutAttr>(getLayout());
  // If no layout is provided, tensor desc is not used in SIMT mode.
  if (!layout)
    return failure();

  SmallVector<int64_t> laneData(layout.getLaneData().asArrayRef());
  SmallVector<int64_t> laneLayout(layout.getLaneLayout().asArrayRef());
  auto tdescShape = getShape();

  auto laneDataSize = 1, sgSize = 1;
  for (auto [laneDim, laneDataDim] : llvm::zip_equal(laneLayout, laneData)) {
    laneDataSize *= laneDataDim;
    sgSize *= laneDim;
  }

  // Case 1: regular loads/stores
  auto scatterAttr = getEncodingAsScatterTensorDescAttr();
  if (scatterAttr) {
    auto chunkSize = scatterAttr.getChunkSize().getInt();
    // Verify if the first dimension of the tensor descriptor shape is
    // distributable.
    assert(tdescShape[0] % (laneLayout[0]) == 0 &&
           "tensor descriptor shape is not distributable");
    if (chunkSize > 1)
      return VectorType::get({chunkSize / laneDataSize, laneDataSize},
                             getElementType());
    return VectorType::get({laneDataSize}, getElementType());
  }

  // Case 2: block loads/stores
  // Check if the tensor descriptor shape is distributable.
  int64_t tensorSize = 1;
  for (auto [tdescDim, laneDim, laneDataDim] :
       llvm::zip_equal(tdescShape, laneLayout, laneData)) {
    assert((tdescDim % (laneDim * laneDataDim) == 0) &&
           "tensor descriptor shape is not distributable");
    tensorSize *= tdescDim;
  }
  // tensorSize must be adjusted for array_length.
  tensorSize *= getArrayLength();

  return VectorType::get({tensorSize / (sgSize * laneDataSize), laneDataSize},
                         getElementType());
}

} // namespace xegpu
} // namespace mlir

#include <mlir/Dialect/XeGPU/IR/XeGPUDialect.cpp.inc>
#define GET_ATTRDEF_CLASSES
#include <mlir/Dialect/XeGPU/IR/XeGPUAttrs.cpp.inc>
#define GET_TYPEDEF_CLASSES
#include <mlir/Dialect/XeGPU/IR/XeGPUTypes.cpp.inc>
