//===- XeGPUDialect.cpp - MLIR XeGPU dialect implementation -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/Dialect/XeGPU/IR/XeGPUTargetInfo.h"
#include "mlir/Dialect/XeGPU/uArch/IntelGpuXe2.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

using std::optional;

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

/// Generates instructions to compute offsets for a subgroup identified by
/// its multidimensional indices (sgId), using the specified subgroup layout
/// (sgLayout), subgroup data dimensions (sizePerSg), and the overall data
/// dimensions (sizePerWg).
static SmallVector<SmallVector<Value>>
genOffsetsComputingInsts(OpBuilder &builder, Location loc,
                         SmallVector<Value> sgId, ArrayRef<int64_t> sgLayout,
                         ArrayRef<int64_t> sizePerSg,
                         ArrayRef<int64_t> sizePerWg) {

  SmallVector<SmallVector<Value>> offsets;

  // nd local offset, localOffset[i] = sgId[i] * sizePerSg[i]
  SmallVector<Value> localOffsets = llvm::map_to_vector(
      llvm::zip(sgId, sizePerSg), [&](const auto &t) -> Value {
        return builder.createOrFold<index::MulOp>(
            loc, std::get<0>(t),
            builder.createOrFold<arith::ConstantIndexOp>(loc, std::get<1>(t)));
      });

  // distUnit[i] is the minimum value between sizePerWg[i] and
  // sgLayout[i] * sizePerSg[i]
  SmallVector<int64_t> distUnit = llvm::map_to_vector(
      llvm::zip_equal(sizePerWg, computeElementwiseMul(sgLayout, sizePerSg)),
      [](const auto &t) { return std::min(std::get<0>(t), std::get<1>(t)); });

  for (SmallVector<int64_t> unitOffs :
       StaticTileOffsetRange(sizePerWg, distUnit)) {
    SmallVector<Value> base =
        llvm::map_to_vector(unitOffs, [&](int64_t d) -> Value {
          return arith::ConstantIndexOp::create(builder, loc, d);
        });

    SmallVector<Value> adds = llvm::map_to_vector(
        llvm::zip_equal(base, localOffsets), [&](const auto &t) -> Value {
          return builder.createOrFold<arith::AddIOp>(loc, std::get<0>(t),
                                                     std::get<1>(t));
        });

    SmallVector<Value> mods = llvm::map_to_vector(
        llvm::zip_equal(adds, sizePerWg), [&](const auto &t) -> Value {
          return builder.createOrFold<index::RemUOp>(
              loc, std::get<0>(t),
              arith::ConstantIndexOp::create(builder, loc, std::get<1>(t)));
        });

    offsets.push_back(mods);
  }
  return offsets;
}

// Checks if the given shape can be evenly distributed based on the layout
// and data factors provided by the LayoutAttr.
bool XeGPUDialect::isEvenlyDistributable(llvm::ArrayRef<int64_t> shape,
                                         xegpu::DistributeLayoutAttr attr) {
  assert(attr && "Layout attribute is missing.");

  // Checks whether the given shape can be evenly distributed using the
  // specified layout and data attributes. If successful, it returns the work
  // size for each compute unit; otherwise, it returns `std::nullopt`. The work
  // size per compute unit is calculated as follows:
  //   - If `data` is null: newShape[i] = shape[i] / layout[i]
  //   - If `data` is not null: newShape[i] = data[i]
  // When round-robin distribution (`rr`) is enabled, `shape[i]` can be
  // smaller than `layout[i] * data[i]`, allowing multiple compute units to
  // share the data.
  auto tryDistribute = [&](llvm::ArrayRef<int64_t> shape,
                           SmallVector<int64_t> layout,
                           SmallVector<int64_t> data,
                           bool rr = true) -> optional<SmallVector<int64_t>> {
    llvm::SmallVector<int64_t> newShape(shape);
    if (layout.size()) {
      if (layout.size() != shape.size())
        return std::nullopt;
      auto ratio = computeShapeRatio(shape, layout);
      if (!ratio.has_value())
        return std::nullopt;
      newShape = ratio.value();
    }

    if (data.size()) {
      if (data.size() != shape.size())
        return std::nullopt;
      auto ratio = computeShapeRatio(newShape, data);
      if (!ratio.has_value() && rr)
        ratio = computeShapeRatio(data, newShape);
      if (!ratio.has_value())
        return std::nullopt;

      // if data is not null, we always return it for next phase.
      newShape = data;
    }
    return newShape;
  };

  // check the sgLayout and sgData
  auto maybeSgShape = tryDistribute(shape, attr.getEffectiveSgLayoutAsInt(),
                                    attr.getEffectiveSgDataAsInt());
  if (!maybeSgShape)
    return false;
  auto sgShape = maybeSgShape.value();

  // check InstData, it neither have layout nor need round-robin
  auto maybeInstShape =
      tryDistribute(sgShape, {}, attr.getEffectiveInstDataAsInt(), false);
  if (!maybeInstShape)
    return false;
  auto instShape = maybeInstShape.value();

  // check LaneLayout and LaneData
  auto maybeLaneShape =
      tryDistribute(instShape, attr.getEffectiveLaneLayoutAsInt(),
                    attr.getEffectiveLaneDataAsInt(), false);
  return maybeLaneShape.has_value();
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

bool BlockTensorDescAttr::hasDefaultsOnly() {
  return getMemorySpace().getValue() == xegpu::MemorySpace::Global &&
         getArrayLength().getInt() == 1 && getBoundaryCheck().getValue();
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
  if (chunkSize <= 0)
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

FailureOr<SmallVector<Value>>
LayoutAttr::delinearizeSubgroupId(OpBuilder &builder, Location loc,
                                  Value linearId) {
  // delinearizeSubgroupId is only available for
  // workgroup-level layout attribute
  if (!isForWorkgroup())
    return failure();

  // TODO: handle order attribute
  auto hasDefaultOrder = [&]() {
    DenseI32ArrayAttr order = getOrder();
    return !order || isIdentityPermutation(llvm::to_vector_of<int64_t>(
                         llvm::reverse(order.asArrayRef())));
  };
  if (!hasDefaultOrder())
    return mlir::emitError(loc, "order attribute is currently not supported.");

  auto dims =
      llvm::map_to_vector(getEffectiveSgLayoutAsInt(), [&](int64_t d) -> Value {
        return builder.createOrFold<arith::ConstantIndexOp>(loc, d);
      });

  return affine::delinearizeIndex(builder, loc, linearId, dims);
}

/// Implements DistributeLayoutAttr::getOffsets to generate
/// instructions for computing multi-dimensional offsets when distributed by
/// LayoutAttr.
FailureOr<SmallVector<SmallVector<Value>>>
LayoutAttr::getOffsets(OpBuilder &builder, Location loc, Value linearId,
                       ArrayRef<int64_t> shape) {
  if (!isForWorkgroup())
    return failure();

  SmallVector<int64_t> sgLayout = getEffectiveSgLayoutAsInt();
  SmallVector<int64_t> sgShape = getEffectiveSgDataAsInt();
  if (sgShape.empty()) {
    if (auto derivedShape = computeShapeRatio(shape, sgLayout))
      sgShape = derivedShape.value();
    else
      return failure();
  }

  // delinearize Ids
  auto maybeIds = delinearizeSubgroupId(builder, loc, linearId);
  if (failed(maybeIds))
    return failure();
  SmallVector<Value> sgIds = *maybeIds;

  return genOffsetsComputingInsts(builder, loc, sgIds, sgLayout, sgShape,
                                  shape);
}

//===----------------------------------------------------------------------===//
// XeGPU_SliceAttr
//===----------------------------------------------------------------------===//
LogicalResult
SliceAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                  xegpu::DistributeLayoutAttr parent, DenseI64ArrayAttr dims) {
  if (!parent || !dims)
    return emitError() << "expected parent layout and dims attribute";

  int64_t rank = parent.getRank();

  // check every element in dims is unique and smaller than rank
  llvm::SmallDenseSet<int64_t> seen;
  for (int64_t dim : dims.asArrayRef()) {
    if (dim < 0 || dim >= rank)
      return emitError() << "invalid dim (" << dim << ") in slice attribute.";
    if (!seen.insert(dim).second)
      return emitError() << "repeated dim (" << dim << ") in slice attribute.";
  }
  return success();
}

SliceAttr SliceAttr::flatten() const {
  xegpu::DistributeLayoutAttr parent = getParent();
  SmallVector<DenseI64ArrayAttr> slicedDims({getDims()});

  while (auto sliceAttr = dyn_cast<xegpu::SliceAttr>(parent)) {
    parent = sliceAttr.getParent();
    slicedDims.push_back(sliceAttr.getDims());
  }

  auto layoutAttr = dyn_cast<xegpu::LayoutAttr>(parent);
  SmallVector<int64_t> indices =
      llvm::to_vector(llvm::seq<int64_t>(0, layoutAttr.getRank()));

  // get remaining dims (flattend) by applying slice ops with all slicedDims
  SmallVector<int64_t> remainingDims(indices);
  for (auto dim : llvm::reverse(slicedDims))
    remainingDims = XeGPUDialect::slice(llvm::ArrayRef<int64_t>(remainingDims),
                                        dim.asArrayRef());

  // get flattend sliced dims by applying slice ops with the remaining dims
  SmallVector<int64_t> flattendDims = XeGPUDialect::slice(
      llvm::ArrayRef<int64_t>(indices), llvm::ArrayRef<int64_t>(remainingDims));

  return xegpu::SliceAttr::get(
      getContext(), layoutAttr,
      DenseI64ArrayAttr::get(getContext(), flattendDims));
}

FailureOr<SmallVector<Value>>
SliceAttr::delinearizeSubgroupId(OpBuilder &builder, Location loc,
                                 Value linearId) {
  SliceAttr attr = flatten();
  auto parent = dyn_cast<LayoutAttr>(attr.getParent());
  return parent.delinearizeSubgroupId(builder, loc, linearId);
}

/// Implements DistributeLayoutAttr::getOffsets to generate
/// instructions for computing multi-dimensional offsets when distributed by
/// SliceAttr.
FailureOr<SmallVector<SmallVector<Value>>>
SliceAttr::getOffsets(OpBuilder &builder, Location loc, Value linearId,
                      ArrayRef<int64_t> shape) {
  assert(getRank() == static_cast<int64_t>(shape.size()) && "invalid shape.");
  if (!isForWorkgroup())
    return failure();

  SmallVector<int64_t> sgLayout = getEffectiveSgLayoutAsInt();
  SmallVector<int64_t> sgShape = getEffectiveSgDataAsInt();
  if (sgShape.empty()) {
    if (auto derivedShape = computeShapeRatio(shape, sgLayout))
      sgShape = derivedShape.value();
    else
      return failure();
  }

  // delinearize Ids
  auto maybeIds = delinearizeSubgroupId(builder, loc, linearId);
  if (failed(maybeIds))
    return failure();

  // The effective sgIds for offsets computing correspond
  // to the dims that are not sliced.
  ArrayRef<int64_t> dims = flatten().getDims().asArrayRef();
  SmallVector<Value> sgIds =
      XeGPUDialect::slice(ArrayRef<Value>(*maybeIds), dims);

  return genOffsetsComputingInsts(builder, loc, sgIds, sgLayout, sgShape,
                                  shape);
}

bool SliceAttr::isSliceOf(const xegpu::DistributeLayoutAttr &other) {
  auto flattenedThis = flatten();
  // If other is a LayoutAttr, just compare directly with parent of
  // flattenedThis.
  if (auto otherLayout = dyn_cast<xegpu::LayoutAttr>(other))
    return flattenedThis.getParent() == otherLayout;
  // If other is a SliceAttr, flatten it first before comparing.
  auto flattenedOther = dyn_cast<xegpu::SliceAttr>(other).flatten();
  // Both must have common parent LayoutAttr.
  if (flattenedThis.getParent() != flattenedOther.getParent())
    return false;
  // otherFlattened's sliced dims must be a subset of flattenedThis's sliced
  // dims.
  llvm::SmallDenseSet<int64_t> thisDims(
      flattenedThis.getDims().asArrayRef().begin(),
      flattenedThis.getDims().asArrayRef().end());
  return llvm::all_of(flattenedOther.getDims().asArrayRef(),
                      [&](int64_t dim) { return thisDims.contains(dim); });
}

//===----------------------------------------------------------------------===//
// XeGPU_RangeAttr
//===----------------------------------------------------------------------===//

LogicalResult
RangeAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                  IntegerAttr startOfRange, IntegerAttr endOfRange) {
  if (startOfRange.getInt() >= endOfRange.getInt())
    return emitError() << "'end' : " << endOfRange.getInt()
                       << " must be greater than 'start' : "
                       << startOfRange.getInt();

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_TensorDescType
//===----------------------------------------------------------------------===//

mlir::Type TensorDescType::parse(AsmParser &parser) {
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

  MLIRContext *ctxt = parser.getContext();
  return TensorDescType::getChecked(
      [&]() { return parser.emitError(parser.getNameLoc()); }, ctxt, shape,
      elementType, encoding.value_or(BlockTensorDescAttr::get(ctxt)),
      layout.value_or(mlir::Attribute()));
}

void TensorDescType::print(AsmPrinter &printer) const {
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

  auto encoding = getEncoding();
  auto blockAttr = llvm::dyn_cast_if_present<BlockTensorDescAttr>(encoding);
  if (encoding && (!blockAttr || !blockAttr.hasDefaultsOnly()))
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

LogicalResult
TensorDescType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                       llvm::ArrayRef<int64_t> shape, mlir::Type elementType,
                       mlir::Attribute encoding, mlir::Attribute layout) {
  size_t rank = shape.size();

  if (rank == 0)
    return emitError() << "expected non-zero rank tensor";

  auto blockAttr = mlir::dyn_cast_if_present<BlockTensorDescAttr>(encoding);
  if (blockAttr) {
    MemorySpaceAttr memorySpaceAttr = blockAttr.getMemorySpace();
    if (rank > 1 && memorySpaceAttr &&
        memorySpaceAttr.getValue() == MemorySpace::SLM)
      return emitError() << "SLM is only supported for 1D block tensor";
  }

  // for gather and scatter ops, Low-precision types are packed in 32-bit units.
  unsigned bitWidth = elementType.getIntOrFloatBitWidth();
  int chunkAlignmentFactor =
      bitWidth < targetinfo::packedSizeInBitsForGatherScatter
          ? targetinfo::packedSizeInBitsForGatherScatter / bitWidth
          : 1;
  auto scatterAttr = mlir::dyn_cast_if_present<ScatterTensorDescAttr>(encoding);
  if (scatterAttr) {
    int64_t chunkSize = scatterAttr.getChunkSizeAsInt();
    if (rank == 1 && chunkSize != 1)
      return emitError() << "expected non-contiguous elements for 1D tensor";

    // If chunk size > 1, the second dimension of the tensor shape must be
    // equal to chunk size and it must be a multiple of the
    // chunkAlignmentFactor.
    if (chunkSize > 1) {
      if (shape.back() != chunkSize)
        return emitError() << "expected last dim of tensor to match chunk size";
      if (shape.back() % chunkAlignmentFactor != 0)
        return emitError() << "expected last dim of tensor to be a multiple of "
                           << chunkAlignmentFactor;
    }
  }

  auto layoutAttr = llvm::dyn_cast_if_present<LayoutAttr>(layout);
  if (layoutAttr) {
    if (rank != (size_t)layoutAttr.getRank())
      return emitError() << "expected layout rank to match tensor rank";

    auto laneData = layoutAttr.getLaneData();
    if (scatterAttr && laneData) {
      // Validate subgroup mapping rules for scattered tensors.
      // if chunkSize > 1, the last dimension of the tensor should
      // be distributed in the units divisible by chunkAlignmentFactor.
      int64_t chunkSize = scatterAttr.getChunkSizeAsInt();
      if (chunkSize > 1 && laneData[rank - 1] % chunkAlignmentFactor)
        return emitError()
               << "expected last dim of lane_data to be a multiple of: "
               << chunkAlignmentFactor;
    }

    if (!XeGPUDialect::isEvenlyDistributable(shape, layoutAttr)) {
      std::string shapeStr;
      llvm::raw_string_ostream stream(shapeStr);
      llvm::interleaveComma(shape, stream);
      return emitError() << "cannot distribute [" << shapeStr << "] using "
                         << layoutAttr;
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_MemDescType
//===----------------------------------------------------------------------===//
mlir::Type MemDescType::parse(AsmParser &parser) {
  llvm::SmallVector<int64_t> shape;
  mlir::Type elementType;
  mlir::FailureOr<MemLayoutAttr> layout;

  // Parse literal '<'
  if (parser.parseLess())
    return {};

  auto shapeLoc = parser.getCurrentLocation();
  if (mlir::failed(parser.parseDimensionList(shape, false, true))) {
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
    MemLayoutAttr attr;
    ParseResult res = parser.parseAttribute(attr);
    if (mlir::failed(res))
      return {};
    layout = attr;
  }

  // Parse literal '>'
  if (parser.parseGreater())
    return {};

  MLIRContext *ctxt = parser.getContext();
  return MemDescType::getChecked(
      [&]() { return parser.emitError(parser.getNameLoc()); }, ctxt, shape,
      elementType, layout.value_or(MemLayoutAttr()));
}

void MemDescType::print(AsmPrinter &printer) const {
  printer << "<";

  printer.printDimensionList(getShape());
  printer << 'x';
  printer << getElementType();

  if (auto layout = getMemLayout())
    printer << ", " << layout;

  printer << ">";
}

//===----------------------------------------------------------------------===//
// XeGPU_MemDescType
//===----------------------------------------------------------------------===//

Attribute MemLayoutAttr::parse(AsmParser &parser, Type type) {

  auto context = parser.getContext();
  llvm::SMLoc loc = parser.getCurrentLocation();

  llvm::SmallDenseSet<StringRef> seenKeys;
  SmallVector<NamedAttribute> attributes;

  auto parseElt = [&]() -> ParseResult {
    StringRef nameId;
    if (failed(parser.parseKeyword(&nameId)))
      return parser.emitError(loc, "expected valid attribute name");

    if (!seenKeys.insert(nameId).second)
      return parser.emitError(loc, "duplicate key '")
             << nameId << " in mem layout attribute";

    if (failed(parser.parseEqual()))
      return failure();

    Attribute attr;
    if (failed(parser.parseAttribute(attr)))
      return failure();
    attributes.emplace_back(nameId, attr);
    return success();
  };

  // Parse literal '<'
  if (parser.parseLess())
    return {};

  if (failed(parser.parseCommaSeparatedList(parseElt)))
    return {};

  // Parse literal '>'
  if (parser.parseGreater())
    return {};

  return parser.getChecked<MemLayoutAttr>(
      loc, context, DictionaryAttr::get(context, attributes));
}

void MemLayoutAttr::print(AsmPrinter &printer) const {
  printer << "<";
  ArrayRef<NamedAttribute> attrs = getAttrs().getValue();
  for (size_t i = 0; i < attrs.size(); i++) {
    printer << attrs[i].getName().str() << " = " << attrs[i].getValue();
    if (i < attrs.size() - 1)
      printer << ", ";
  }
  printer << ">";
}
// a helper utility to perform binary operation on OpFoldResult.
// If both a and b are attributes, it will simply return the result.
// Otherwise, the corresponding arith op will be generated, and an
// contant op will be created if one of them is an attribute.
template <typename ArithOp>
OpFoldResult genBinOp(OpFoldResult a, OpFoldResult b, Location loc,
                      OpBuilder &builder) {
  auto aVal = getValueOrCreateConstantIndexOp(builder, loc, a);
  auto bVal = getValueOrCreateConstantIndexOp(builder, loc, b);
  return builder.create<ArithOp>(loc, aVal, bVal).getResult();
}

// a helper utility to perform division operation on OpFoldResult and int64_t.
#define div(a, b)                                                              \
  genBinOp<arith::DivSIOp>(a, builder.getIndexAttr(b), loc, builder)

// a helper utility to perform reminder operation on OpFoldResult and int64_t.
#define rem(a, b)                                                              \
  genBinOp<arith::RemSIOp>(a, builder.getIndexAttr(b), loc, builder)

// a helper utility to perform multiply operation on OpFoldResult and int64_t.
#define mul(a, b)                                                              \
  genBinOp<arith::MulIOp>(a, builder.getIndexAttr(b), loc, builder)

// a helper utility to perform addition operation on two OpFoldResult.
#define add(a, b) genBinOp<arith::AddIOp>(a, b, loc, builder)

// block the given offsets according to the block shape
// say the original offset is [y, x], and the block shape is [By, Bx],
// then the blocked offset is [y/By, x/Bx, y%By, x%Bx]
SmallVector<OpFoldResult> getBlockedOffsets(OpBuilder &builder, Location loc,
                                            ArrayRef<OpFoldResult> offsets,
                                            ArrayRef<int64_t> blockShape) {

  assert(offsets.size() == blockShape.size() &&
         "offsets and blockShape must have the same size");
  SmallVector<OpFoldResult> blockedOffsets;
  SmallVector<OpFoldResult> divs, rems;

  for (auto [offset, block] : llvm::zip(offsets, blockShape)) {
    divs.push_back(div(offset, block));
    rems.push_back(rem(offset, block));
  }
  blockedOffsets.append(divs.begin(), divs.end());
  blockedOffsets.append(rems.begin(), rems.end());

  return blockedOffsets;
}

// Get strides as vector of integer for MemDesc.
SmallVector<int64_t> MemDescType::getStrideShape() {

  SmallVector<int64_t> matrixShape(getShape().begin(), getShape().end());

  ArrayAttr strideAttr = getStrideAttr();
  SmallVector<int64_t> strides;
  for (Attribute attr : strideAttr.getValue()) {
    strides.push_back(cast<IntegerAttr>(attr).getInt());
  }

  SmallVector<int64_t> innerBlkShape = getBlockShape();

  // get perm from FCD to LCD
  // perm[i] = the dim with i-th smallest stride
  SmallVector<int, 4> perm =
      llvm::to_vector<4>(llvm::seq<int>(0, strides.size()));
  llvm::sort(perm, [&](int a, int b) { return strides[a] < strides[b]; });

  assert(strides[perm[0]] == 1 && "inner most dim must have stride 1");

  SmallVector<int64_t> innerBlkStride(innerBlkShape.size());
  innerBlkStride[perm[0]] = 1;
  for (size_t i = 1; i < perm.size(); ++i)
    innerBlkStride[perm[i]] =
        innerBlkStride[perm[i - 1]] * innerBlkShape[perm[i - 1]];

  // compute the original matrix shape using the stride info
  // and compute the number of blocks in each dimension
  // The shape of highest dim can't be derived from stride info,
  // but doesn't impact the stride computation for blocked layout.
  SmallVector<int64_t> matrixShapeOrig(matrixShape.size());
  SmallVector<int64_t> BlkShapeOrig(matrixShape.size());
  for (size_t i = 0; i < perm.size() - 1; ++i) {
    matrixShapeOrig[perm[i]] = strides[perm[i + 1]] / strides[perm[i]];
    BlkShapeOrig[perm[i]] = matrixShapeOrig[perm[i]] / innerBlkShape[perm[i]];
  }

  int64_t innerBlkSize = 1;
  for (auto s : innerBlkShape)
    innerBlkSize *= s;

  SmallVector<int64_t> outerBlkStride(matrixShape.size());
  outerBlkStride[perm[0]] = innerBlkSize;
  for (size_t i = 0; i < perm.size() - 1; ++i) {
    outerBlkStride[perm[i + 1]] =
        outerBlkStride[perm[i]] * BlkShapeOrig[perm[i]];
  }

  // combine the inner and outer strides
  SmallVector<int64_t> blockedStrides;
  blockedStrides.append(outerBlkStride.begin(), outerBlkStride.end());
  blockedStrides.append(innerBlkStride.begin(), innerBlkStride.end());

  return blockedStrides;
}

// Calculate the linear offset using the blocked offsets and stride
Value MemDescType::getLinearOffsets(OpBuilder &builder, Location loc,
                                    ArrayRef<OpFoldResult> offsets) {

  SmallVector<int64_t> matrixShape(getShape().begin(), getShape().end());
  SmallVector<int64_t> blockShape = getBlockShape();
  SmallVector<int64_t> strides = getStrideShape();

  // blockshape equal to matrixshape means no blocking
  if (llvm::equal(blockShape, matrixShape)) {
    // remove the outer dims from strides
    strides.erase(strides.begin(), strides.begin() + matrixShape.size());
  } else {
    assert(offsets.size() == blockShape.size() &&
           "offsets and blockShape must have the same size");
    // say the original offset is [y, x], and the block shape is [By, Bx],
    // then the blocked offset is [y/By, x/Bx, y%By, x%Bx]
    SmallVector<OpFoldResult> blockedOffsets;
    SmallVector<OpFoldResult> divs, rems;

    for (auto [offset, block] : llvm::zip(offsets, blockShape)) {
      divs.push_back(div(offset, block));
      rems.push_back(rem(offset, block));
    }
    blockedOffsets.append(divs.begin(), divs.end());
    blockedOffsets.append(rems.begin(), rems.end());

    offsets = blockedOffsets;
  }

  // Start with initial value as matrix descriptor's base offset.
  Value linearOffset = arith::ConstantIndexOp::create(builder, loc, 0);
  for (size_t i = 0; i < offsets.size(); ++i) {
    OpFoldResult mulResult = mul(offsets[i], strides[i]);
    Value mulVal = getValueOrCreateConstantIndexOp(builder, loc, mulResult);
    linearOffset = arith::AddIOp::create(builder, loc, mulVal, linearOffset);
  }

  return linearOffset;
}

} // namespace xegpu
} // namespace mlir

#include <mlir/Dialect/XeGPU/IR/XeGPUDialect.cpp.inc>
#define GET_ATTRDEF_CLASSES
#include <mlir/Dialect/XeGPU/IR/XeGPUAttrs.cpp.inc>
#define GET_TYPEDEF_CLASSES
#include <mlir/Dialect/XeGPU/IR/XeGPUTypes.cpp.inc>
