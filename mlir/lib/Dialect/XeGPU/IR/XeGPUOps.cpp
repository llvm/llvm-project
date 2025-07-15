//===- XeGPUOps.cpp - MLIR XeGPU ops implementation -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "xegpu"

namespace mlir {
namespace xegpu {

template <typename T>
static std::string makeString(T array, bool breakline = false) {
  std::string buf;
  buf.clear();
  llvm::raw_string_ostream os(buf);
  os << "[";
  for (size_t i = 1; i < array.size(); i++) {
    os << array[i - 1] << ", ";
    if (breakline)
      os << "\n\t\t";
  }
  os << array.back() << "]";
  return buf;
}

static SmallVector<int64_t> getShapeOf(Type type) {
  SmallVector<int64_t> shape;
  if (auto ty = llvm::dyn_cast<ShapedType>(type))
    shape = SmallVector<int64_t>(ty.getShape());
  else
    shape.push_back(1);
  return shape;
}

static int64_t getRankOf(Value val) {
  auto type = val.getType();
  if (auto ty = llvm::dyn_cast<ShapedType>(type))
    return ty.getRank();
  return 0;
}

static bool isReadHintOrNone(const CachePolicyAttr &attr) {
  if (!attr)
    return true;
  auto kind = attr.getValue();
  return kind == CachePolicy::CACHED || kind == CachePolicy::UNCACHED ||
         kind == CachePolicy::STREAMING || kind == CachePolicy::READ_INVALIDATE;
}

static bool isWriteHintOrNone(const CachePolicyAttr &attr) {
  if (!attr)
    return true;
  auto kind = attr.getValue();
  return kind == CachePolicy::CACHED || kind == CachePolicy::UNCACHED ||
         kind == CachePolicy::WRITE_BACK || kind == CachePolicy::WRITE_THROUGH;
}

static LogicalResult
isValidGatherScatterParams(Type maskTy, VectorType valueTy,
                           TensorDescType tdescTy,
                           function_ref<InFlightDiagnostic()> emitError) {

  if (!tdescTy.isScattered())
    return emitError() << "Expects a scattered TensorDesc.";

  if (!valueTy)
    return emitError() << "Expecting a vector type result.";

  auto maskShape = getShapeOf(maskTy);
  auto valueShape = getShapeOf(valueTy);
  auto tdescShape = getShapeOf(tdescTy);
  auto chunkSize = tdescTy.getChunkSizeAsInt();

  if (valueTy.getElementType() != tdescTy.getElementType())
    return emitError()
           << "Value should have the same element type as TensorDesc.";

  llvm::SmallVector<int64_t> expectedMaskShape(tdescShape);
  if (chunkSize > 1)
    expectedMaskShape.pop_back();
  if (expectedMaskShape != maskShape)
    return emitError()
           << "Mask should match TensorDesc except the chunk size dim.";

  // a valid shape for SIMT case
  if (valueTy.getRank() == 1 && valueTy.getNumElements() == chunkSize) {
    if (tdescTy.getLayoutAttr())
      return emitError() << "TensorDesc doesn't need LayoutAttr for SIMT code";
    return success();
  }

  if (tdescShape != valueShape)
    return emitError() << "Value shape " << makeString(valueShape)
                       << " is neither a valid distribution for SIMT nor "
                          "consistent with the tensor descriptor for SIMD "
                       << tdescTy;
  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_CreateNdDescOp
//===----------------------------------------------------------------------===//

void CreateNdDescOp::build(OpBuilder &builder, OperationState &state,
                           Type tdesc, TypedValue<MemRefType> source) {
  [[maybe_unused]] auto ty = source.getType();
  assert(ty.hasStaticShape());

  build(builder, state, tdesc, source, ValueRange({}) /* dynamic offsets */,
        ValueRange({}) /* empty dynamic shape */,
        ValueRange({}) /* empty dynamic strides */,
        builder.getDenseI64ArrayAttr({}) /* const offsets */,
        builder.getDenseI64ArrayAttr({}) /* empty const shape*/,
        builder.getDenseI64ArrayAttr({}) /* empty const strides*/);
}

void CreateNdDescOp::build(OpBuilder &builder, OperationState &state,
                           Type tdesc, TypedValue<MemRefType> source,
                           llvm::ArrayRef<OpFoldResult> shape,
                           llvm::ArrayRef<OpFoldResult> strides) {
  assert(shape.size() && strides.size() && shape.size() == strides.size());

  llvm::SmallVector<int64_t> staticShape;
  llvm::SmallVector<int64_t> staticStrides;
  llvm::SmallVector<Value> dynamicShape;
  llvm::SmallVector<Value> dynamicStrides;

  dispatchIndexOpFoldResults(shape, dynamicShape, staticShape);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);

  auto staticShapeAttr = builder.getDenseI64ArrayAttr(staticShape);
  auto staticStridesAttr = builder.getDenseI64ArrayAttr(staticStrides);

  build(builder, state, tdesc, source, ValueRange({}), dynamicShape,
        dynamicStrides, builder.getDenseI64ArrayAttr({}), staticShapeAttr,
        staticStridesAttr);
}

void CreateNdDescOp::build(OpBuilder &builder, OperationState &state,
                           Type tdesc, TypedValue<IntegerType> source,
                           llvm::ArrayRef<OpFoldResult> shape,
                           llvm::ArrayRef<OpFoldResult> strides) {
  assert(shape.size() && strides.size() && shape.size() == strides.size());

  llvm::SmallVector<int64_t> staticShape;
  llvm::SmallVector<int64_t> staticStrides;
  llvm::SmallVector<Value> dynamicShape;
  llvm::SmallVector<Value> dynamicStrides;

  dispatchIndexOpFoldResults(shape, dynamicShape, staticShape);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);

  auto staticShapeAttr = builder.getDenseI64ArrayAttr(staticShape);
  auto staticStridesAttr = builder.getDenseI64ArrayAttr(staticStrides);

  build(builder, state, tdesc, source, ValueRange({}), dynamicShape,
        dynamicStrides, builder.getDenseI64ArrayAttr({}), staticShapeAttr,
        staticStridesAttr);
}

void CreateNdDescOp::build(OpBuilder &builder, OperationState &state,
                           Type tdesc, TypedValue<MemRefType> source,
                           llvm::ArrayRef<OpFoldResult> offsets) {
  [[maybe_unused]] auto ty = source.getType();
  assert(ty.hasStaticShape() && offsets.size() == (size_t)ty.getRank());

  llvm::SmallVector<int64_t> staticOffsets;
  llvm::SmallVector<Value> dynamicOffsets;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);

  build(builder, state, tdesc, source, dynamicOffsets /* dynamic offsets */,
        ValueRange({}) /* empty dynamic shape */,
        ValueRange({}) /* empty dynamic strides */,
        builder.getDenseI64ArrayAttr(staticOffsets) /* const offsets */,
        {} /* empty const shape*/, {} /* empty const strides*/);
}

void CreateNdDescOp::build(OpBuilder &builder, OperationState &state,
                           Type tdesc, Value source,
                           llvm::ArrayRef<OpFoldResult> offsets,
                           llvm::ArrayRef<OpFoldResult> shape,
                           llvm::ArrayRef<OpFoldResult> strides) {
  assert(shape.size() && offsets.size() && strides.size() &&
         shape.size() == strides.size() && shape.size() == offsets.size());

  Type srcTy = source.getType();
  assert((isa<IntegerType, MemRefType>(srcTy)) &&
         "Source has to be either int or memref.");

  llvm::SmallVector<Value> dynamicOffsets;
  llvm::SmallVector<Value> dynamicShape;
  llvm::SmallVector<Value> dynamicStrides;

  llvm::SmallVector<int64_t> staticOffsets;
  llvm::SmallVector<int64_t> staticShape;
  llvm::SmallVector<int64_t> staticStrides;

  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(shape, dynamicShape, staticShape);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);

  auto staticOffsetsAttr = builder.getDenseI64ArrayAttr(staticOffsets);
  auto staticShapeAttr = builder.getDenseI64ArrayAttr(staticShape);
  auto staticStridesAttr = builder.getDenseI64ArrayAttr(staticStrides);

  if (auto memrefTy = dyn_cast<MemRefType>(srcTy)) {
    auto memrefShape = memrefTy.getShape();
    auto [memrefStrides, _] = memrefTy.getStridesAndOffset();

    // if shape and strides are from Memref, we don't need attributes for them
    // to keep the IR print clean.
    if (staticShape == memrefShape && staticStrides == memrefStrides) {
      staticShapeAttr = DenseI64ArrayAttr();
      staticStridesAttr = DenseI64ArrayAttr();
    }
  }

  build(builder, state, tdesc, source, dynamicOffsets, dynamicShape,
        dynamicStrides, staticOffsetsAttr, staticShapeAttr, staticStridesAttr);
}

LogicalResult CreateNdDescOp::verify() {
  auto rank = (int64_t)getMixedOffsets().size();
  bool invalidRank = false;
  bool invalidElemTy = false;

  // Memory space of created TensorDesc should match with the source.
  // Both source and TensorDesc are considered for global memory by default,
  // if the memory scope attr is not specified. If source is an integer,
  // it is considered as ptr to global memory.
  auto srcMemorySpace = getSourceMemorySpace();
  auto tdescMemorySpace = static_cast<unsigned>(getType().getMemorySpace());
  if (srcMemorySpace != tdescMemorySpace)
    return emitOpError("Memory space mismatch.")
           << " Source: " << srcMemorySpace
           << ", TensorDesc: " << tdescMemorySpace;

  // check source type matches the rank if it is a memref.
  // It also should have the same ElementType as TensorDesc.
  auto memrefTy = dyn_cast<MemRefType>(getSourceType());
  if (memrefTy) {
    invalidRank |= (memrefTy.getRank() != rank);
    invalidElemTy |= memrefTy.getElementType() != getElementType();
  }

  // mismatches among shape, strides, and offsets are
  // already handeled by OffsetSizeAndStrideOpInterface.
  // So they are not check here.
  if (invalidRank)
    return emitOpError(
        "Expecting the rank of shape, strides, offsets, and source (if source "
        "is a memref) should match with each other.");

  // check result TensorDesc rank
  if (getType().getRank() > rank)
    return emitOpError(
        "Expecting the TensorDesc rank is not greater than the "
        "ranks of shape, strides, offsets or the memref source.");

  if (invalidElemTy)
    return emitOpError("TensorDesc should have the same element "
                       "type with the source if it is a memref.\n");

  if (getType().isScattered())
    return emitOpError("Expects a non-scattered TensorDesc.\n");

  return success();
}

ParseResult parseOptionalDynamicIndexList(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &values,
    DenseI64ArrayAttr &integers, DenseBoolArrayAttr &scalableFlags,
    SmallVectorImpl<Type> *valueTypes = nullptr,
    AsmParser::Delimiter delimiter = AsmParser::Delimiter::Square) {

  SmallVector<int64_t, 4> integerVals;
  SmallVector<bool, 4> scalableVals;
  auto parseIntegerOrValue = [&]() {
    OpAsmParser::UnresolvedOperand operand;
    auto res = parser.parseOptionalOperand(operand);

    // When encountering `[`, assume that this is a scalable index.
    scalableVals.push_back(parser.parseOptionalLSquare().succeeded());

    if (res.has_value() && succeeded(res.value())) {
      values.push_back(operand);
      integerVals.push_back(ShapedType::kDynamic);
      if (valueTypes && parser.parseColonType(valueTypes->emplace_back()))
        return failure();
    } else {
      int64_t integer;
      if (failed(parser.parseInteger(integer)))
        return failure();
      integerVals.push_back(integer);
    }

    // If this is assumed to be a scalable index, verify that there's a closing
    // `]`.
    if (scalableVals.back() && parser.parseOptionalRSquare().failed())
      return failure();
    return success();
  };
  if (parser.parseOptionalLSquare().succeeded()) {
    if (parser.parseCommaSeparatedList(parseIntegerOrValue) ||
        parser.parseRSquare())
      return parser.emitError(parser.getNameLoc())
             << "expected SSA value or integer";
    integers = parser.getBuilder().getDenseI64ArrayAttr(integerVals);
    scalableFlags = parser.getBuilder().getDenseBoolArrayAttr(scalableVals);
    return success();
  }
  return success();
}

::mlir::ParseResult CreateNdDescOp::parse(::mlir::OpAsmParser &parser,
                                          ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand sourceRawOperand{};
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> sourceOperands(
      &sourceRawOperand, 1);
  ::llvm::SMLoc sourceOperandsLoc;

  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4>
      offsetsOperands;
  ::llvm::SMLoc offsetsOperandsLoc;
  ::mlir::DenseI64ArrayAttr const_offsetsAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> shapeOperands;
  ::llvm::SMLoc shapeOperandsLoc;
  ::mlir::DenseI64ArrayAttr const_shapeAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4>
      stridesOperands;
  ::llvm::SMLoc stridesOperandsLoc;
  ::mlir::DenseI64ArrayAttr const_stridesAttr;
  ::mlir::Type sourceRawType{};
  ::llvm::ArrayRef<::mlir::Type> sourceTypes(&sourceRawType, 1);
  ::mlir::Type TensorDescRawType{};
  ::llvm::ArrayRef<::mlir::Type> TensorDescTypes(&TensorDescRawType, 1);

  sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceRawOperand))
    return ::mlir::failure();

  offsetsOperandsLoc = parser.getCurrentLocation();

  DenseBoolArrayAttr scalableFlags;
  auto odsResult = parseOptionalDynamicIndexList(
      parser, offsetsOperands, const_offsetsAttr, scalableFlags);

  if (const_offsetsAttr) {
    if (odsResult)
      return ::mlir::failure();
    result.getOrAddProperties<CreateNdDescOp::Properties>().const_offsets =
        const_offsetsAttr;
  }

  if (::mlir::succeeded(parser.parseOptionalKeyword("shape"))) {
    if (parser.parseColon())
      return ::mlir::failure();
    {
      shapeOperandsLoc = parser.getCurrentLocation();
      auto odsResult =
          parseDynamicIndexList(parser, shapeOperands, const_shapeAttr);
      if (const_shapeAttr) {
        if (odsResult)
          return ::mlir::failure();
        result.getOrAddProperties<CreateNdDescOp::Properties>().const_shape =
            const_shapeAttr;
      }
    }

    if (parser.parseKeyword("strides"))
      return ::mlir::failure();
    if (parser.parseColon())
      return ::mlir::failure();
    {
      stridesOperandsLoc = parser.getCurrentLocation();
      auto odsResult =
          parseDynamicIndexList(parser, stridesOperands, const_stridesAttr);
      if (const_stridesAttr) {
        if (odsResult)
          return ::mlir::failure();
        result.getOrAddProperties<CreateNdDescOp::Properties>().const_strides =
            const_stridesAttr;
      }
    }
  }
  {
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
          return parser.emitError(loc)
                 << "'" << result.name.getStringRef() << "' op ";
        })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::Type type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    sourceRawType = type;
  }
  if (parser.parseArrow())
    return ::mlir::failure();

  if (parser.parseType(TensorDescRawType))
    return ::mlir::failure();

  ::llvm::copy(::llvm::ArrayRef<int32_t>(
                   {1, static_cast<int32_t>(offsetsOperands.size()),
                    static_cast<int32_t>(shapeOperands.size()),
                    static_cast<int32_t>(stridesOperands.size())}),
               result.getOrAddProperties<CreateNdDescOp::Properties>()
                   .operandSegmentSizes.begin());

  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(TensorDescTypes);

  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc,
                             result.operands))
    return ::mlir::failure();

  if (parser.resolveOperands(offsetsOperands, odsBuildableType0,
                             offsetsOperandsLoc, result.operands))
    return ::mlir::failure();

  if (parser.resolveOperands(shapeOperands, odsBuildableType0, shapeOperandsLoc,
                             result.operands))
    return ::mlir::failure();

  if (parser.resolveOperands(stridesOperands, odsBuildableType0,
                             stridesOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void CreateNdDescOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getSource();

  auto constOffsetsAttr = getConstOffsetsAttr();
  bool printOffsets = false;
  if (constOffsetsAttr && constOffsetsAttr.size() > 0) {
    auto firstVal = constOffsetsAttr.asArrayRef()[0];
    if (firstVal != std::numeric_limits<int64_t>::max()) {
      printOffsets = true;
    }
  }
  if (printOffsets) {

    printDynamicIndexList(_odsPrinter, *this, getOffsets(),
                          getConstOffsetsAttr());
  }
  if (((!getShape().empty()) || (getConstShapeAttr()))) {
    _odsPrinter << ' ' << "shape";
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    printDynamicIndexList(_odsPrinter, *this, getShape(), getConstShapeAttr());
    _odsPrinter << ' ' << "strides";
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    printDynamicIndexList(_odsPrinter, *this, getStrides(),
                          getConstStridesAttr());
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operandSegmentSizes");
  elidedAttrs.push_back("const_offsets");
  elidedAttrs.push_back("const_shape");
  elidedAttrs.push_back("const_strides");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getSource().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::Type>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
    else
      _odsPrinter << type;
  }
  _odsPrinter << ' ' << "->";
  _odsPrinter << ' ';
  // _odsPrinter << getTensorDesc().getType();

  _odsPrinter << "!xegpu.tensor_desc<";

  auto tDesc = getTensorDesc().getType();
  auto shape = tDesc.getShape();
  for (int64_t dim : shape) {
    if (mlir::ShapedType::isDynamic(dim))
      _odsPrinter << '?';
    else
      _odsPrinter << dim;
    _odsPrinter << 'x';
  }

  _odsPrinter << tDesc.getElementType();

  if (auto encoding = tDesc.getEncoding())
    _odsPrinter << ", " << encoding;

  if (auto layout = tDesc.getLayout())
    _odsPrinter << ", " << layout;

  _odsPrinter << ">";
}

//===----------------------------------------------------------------------===//
// XeGPU_PrefetchNdOp
//===----------------------------------------------------------------------===//
LogicalResult PrefetchNdOp::verify() {
  auto tdescTy = getTensorDescType();
  if (tdescTy.isScattered())
    return emitOpError("Expects a non-scattered TensorDesc.\n");

  if (!isReadHintOrNone(getL1HintAttr()))
    return emitOpError("invalid l1_hint: ") << getL1HintAttr();

  if (!isReadHintOrNone(getL2HintAttr()))
    return emitOpError("invalid l2_hint: ") << getL2HintAttr();

  if (!isReadHintOrNone(getL3HintAttr()))
    return emitOpError("invalid l3_hint: ") << getL3HintAttr();

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_LoadNdOp
//===----------------------------------------------------------------------===//
LogicalResult LoadNdOp::verify() {
  auto tdescTy = getTensorDescType();
  auto valueTy = getType();

  if (tdescTy.isScattered())
    return emitOpError("Expects a non-scattered TensorDesc.\n");

  if (tdescTy.getRank() > 2)
    return emitOpError("Expects a 1D or 2D TensorDesc.\n");

  if (!valueTy)
    return emitOpError("Invalid result, it should be a VectorType.\n");

  if (!isReadHintOrNone(getL1HintAttr()))
    return emitOpError("invalid l1_hint: ") << getL1HintAttr();

  if (!isReadHintOrNone(getL2HintAttr()))
    return emitOpError("invalid l2_hint: ") << getL2HintAttr();

  if (!isReadHintOrNone(getL3HintAttr()))
    return emitOpError("invalid l3_hint: ") << getL3HintAttr();

  int tdescElems = tdescTy.getNumElements() * tdescTy.getArrayLength();
  int valueElems = valueTy.getNumElements();

  // If the result vector is 1D and has less elements than the tensor
  // descriptor, it is supposed to be a SIMT op. The layout attribute in
  // tensor_desc is not needed.
  if (valueElems < tdescElems && valueTy.getRank() == 1) {
    // SIMT mode doesn't need LayoutAttr.
    if (tdescTy.getLayoutAttr())
      return emitOpError()
             << "TensorDesc doesn't need LayoutAttr for SIMT code";

    // For SIMT code, the load is evenly distributed across all lanes in a
    // subgroup. Since subgroup size is arch dependent, we only check even
    // distribution here.
    if (tdescElems % valueElems)
      return emitOpError()
             << "Result shape " << makeString(getShapeOf(valueTy))
             << " is not a valid distribution for tensor descriptor "
             << tdescTy;

    return success();
  }

  // Check SIMD mode.
  auto tdescShape = getShapeOf(tdescTy);
  auto valueShape = getShapeOf(valueTy);

  if (getTranspose()) {
    auto trans = getTranspose().value();
    // Make sure the transpose value is valid, and apply it
    if (llvm::all_of(trans, [&](size_t s) { return s < tdescShape.size(); }))
      tdescShape = applyPermutation(tdescShape, trans);
    else
      mlir::emitWarning(getLoc()) << "Invalid transpose attr. It is ignored.";
  }

  if (getPacked()) {
    if (tdescTy.getRank() == 2) {
      const int axis = 0;
      auto vnni_factor = valueShape.back();
      tdescShape[axis] /= vnni_factor;
      tdescShape.push_back(vnni_factor);
    } else {
      mlir::emitWarning(getLoc())
          << "Invalid Packed Attr. It is ignored (available for 2D "
             "TensorDesc only).";
    }
  }

  auto array_len = tdescTy.getArrayLength();
  if (array_len > 1)
    tdescShape.insert(tdescShape.begin(), array_len);

  if (tdescShape != valueShape)
    return emitOpError() << "Result shape " << makeString(valueShape)
                         << " is not consistent with tensor descriptor "
                         << tdescTy;

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_StoreNdOp
//===----------------------------------------------------------------------===//
LogicalResult StoreNdOp::verify() {
  auto dstTy = getTensorDescType(); // Tile
  auto valTy = getValueType();      // Vector

  if (dstTy.isScattered())
    return emitOpError("Expects a non-scattered TensorDesc.\n");

  if (dstTy.getRank() > 2)
    return emitOpError("Expects a 1D or 2D TensorDesc.\n");

  if (!valTy)
    return emitOpError("Expecting a VectorType result.\n");

  if (!isWriteHintOrNone(getL1HintAttr()))
    return emitOpError("invalid l1_hint: ") << getL1HintAttr();

  if (!isWriteHintOrNone(getL2HintAttr()))
    return emitOpError("invalid l2_hint: ") << getL2HintAttr();

  if (!isWriteHintOrNone(getL3HintAttr()))
    return emitOpError("invalid l3_hint: ") << getL3HintAttr();

  auto array_len = dstTy.getArrayLength();
  if (array_len > 1)
    return emitOpError("array length is not supported by store_nd.\n");

  auto tdescElems = dstTy.getNumElements();
  auto valueElems = valTy.getNumElements();

  // Similar to LoadNdOp, if the value vector is 1D and has less elements than
  // the tensor descriptor, it is supposed to be a SIMT op. The layout attribute
  // in tensor_desc is not needed.
  if (valTy.getRank() == 1 && valueElems < tdescElems) {
    // SIMT mode doesn't need LayoutAttr.
    if (dstTy.getLayoutAttr())
      return emitOpError()
             << "TensorDesc doesn't need LayoutAttr for SIMT code";

    if (tdescElems % valueElems)
      return emitOpError()
             << "Value shape " << makeString(getShapeOf(valTy))
             << " is not a valid distribution for tensor descriptor " << dstTy;

    return success();
  }

  // SIMD code should have the same shape as the tensor descriptor.
  auto tdescShape = getShapeOf(dstTy);
  auto valueShape = getShapeOf(valTy);
  if (tdescShape != valueShape)
    return emitOpError() << "Value shape " << makeString(valueShape)
                         << " is not consistent with tensor descriptor "
                         << dstTy;

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_UpdateNDOffsetOp
//===----------------------------------------------------------------------===//
LogicalResult UpdateNdOffsetOp::verify() {
  auto ty = getTensorDescType();
  if (ty.isScattered())
    return emitOpError("Expects a non-scattered TensorDesc.\n");

  // number of offsets specified must match the rank of the tensor descriptor
  if (ty.getRank() != (int64_t)getNumOffsets()) {
    return emitOpError("Invalid number of offsets.");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_CreateDescOp
//===----------------------------------------------------------------------===//

void CreateDescOp::build(OpBuilder &builder, OperationState &state,
                         TensorDescType TensorDesc, Value source,
                         llvm::ArrayRef<OpFoldResult> offsets) {
  auto loc = source.getLoc();
  int64_t size = static_cast<int64_t>(offsets.size());
  auto type = VectorType::get(size, builder.getIndexType());
  auto values = getValueOrCreateConstantIndexOp(builder, loc, offsets);
  auto offset = builder.create<vector::FromElementsOp>(loc, type, values);
  build(builder, state, TensorDesc, source, offset);
}

void CreateDescOp::build(OpBuilder &builder, OperationState &state,
                         TensorDescType TensorDesc, Value source,
                         llvm::ArrayRef<int64_t> offsets) {
  auto ofrs = getAsIndexOpFoldResult(builder.getContext(), offsets);
  build(builder, state, TensorDesc, source, ofrs);
}

LogicalResult CreateDescOp::verify() {
  auto tdescTy = getTensorDescType();

  if (getRankOf(getSource()) > 1)
    return emitOpError(
        "Expecting the source is a 1D memref or pointer (uint64_t).");

  if (!tdescTy.isScattered())
    return emitOpError("Expects a scattered TensorDesc.\n");

  // Memory space of created TensorDesc should match with the source.
  // Both source and TensorDesc are considered for global memory by default,
  // if the memory scope attr is not specified. If source is an integer,
  // it is considered as ptr to global memory.
  auto srcMemorySpace = getSourceMemorySpace();
  auto tdescMemorySpace = static_cast<unsigned>(tdescTy.getMemorySpace());
  if (srcMemorySpace != tdescMemorySpace)
    return emitOpError("Memory space mismatch.")
           << " Source: " << srcMemorySpace
           << ", TensorDesc: " << tdescMemorySpace;

  // check total size
  auto chunkSize = tdescTy.getChunkSizeAsInt();
  SmallVector<int64_t> shape(getOffsetsType().getShape());
  if (chunkSize != 1)
    shape.push_back(chunkSize);

  auto tdescShape = getShapeOf(tdescTy);
  if (shape != tdescShape)
    return emitOpError("Incorrect TensorDesc shape. ")
           << "Expected is " << makeString(shape) << "\n";

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_PrefetchOp
//===----------------------------------------------------------------------===//
LogicalResult PrefetchOp::verify() {
  auto tdescTy = getTensorDescType();
  if (!tdescTy.isScattered())
    return emitOpError("Expects a scattered TensorDesc.\n");

  if (!isReadHintOrNone(getL1HintAttr()))
    return emitOpError("invalid l1_hint: ") << getL1HintAttr();

  if (!isReadHintOrNone(getL2HintAttr()))
    return emitOpError("invalid l2_hint: ") << getL2HintAttr();

  if (!isReadHintOrNone(getL3HintAttr()))
    return emitOpError("invalid l3_hint: ") << getL3HintAttr();

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_LoadGatherOp
//===----------------------------------------------------------------------===//
LogicalResult LoadGatherOp::verify() {
  auto tdescTy = getTensorDescType();
  auto maskTy = getMaskType();
  auto valueTy = getValueType();

  if (!isReadHintOrNone(getL1HintAttr()))
    return emitOpError("invalid l1_hint: ") << getL1HintAttr();

  if (!isReadHintOrNone(getL2HintAttr()))
    return emitOpError("invalid l2_hint: ") << getL2HintAttr();

  if (!isReadHintOrNone(getL3HintAttr()))
    return emitOpError("invalid l3_hint: ") << getL3HintAttr();

  return isValidGatherScatterParams(maskTy, valueTy, tdescTy,
                                    [&]() { return emitOpError(); });
}

//===----------------------------------------------------------------------===//
// XeGPU_StoreScatterOp
//===----------------------------------------------------------------------===//
LogicalResult StoreScatterOp::verify() {
  auto tdescTy = getTensorDescType();
  auto maskTy = getMaskType();
  auto valueTy = getValueType();

  if (!isWriteHintOrNone(getL1HintAttr()))
    return emitOpError("invalid l1_hint: ") << getL1HintAttr();

  if (!isWriteHintOrNone(getL2HintAttr()))
    return emitOpError("invalid l2_hint: ") << getL2HintAttr();

  if (!isWriteHintOrNone(getL3HintAttr()))
    return emitOpError("invalid l3_hint: ") << getL3HintAttr();

  return isValidGatherScatterParams(maskTy, valueTy, tdescTy,
                                    [&]() { return emitOpError(); });
}

//===----------------------------------------------------------------------===//
// XeGPU_UpdateOffsetOp
//===----------------------------------------------------------------------===//
void UpdateOffsetOp::build(OpBuilder &builder, OperationState &state,
                           mlir::Value tensorDesc,
                           llvm::ArrayRef<OpFoldResult> offsets) {
  auto tdescTy = mlir::dyn_cast<TensorDescType>(tensorDesc.getType());
  assert(tdescTy && "Expecting the source is a TensorDescType value.");
  auto loc = tensorDesc.getLoc();
  int64_t size = static_cast<int64_t>(offsets.size());
  auto type = VectorType::get({size}, builder.getIndexType());
  auto values = getValueOrCreateConstantIndexOp(builder, loc, offsets);
  auto offset = builder.create<vector::FromElementsOp>(loc, type, values);
  build(builder, state, tdescTy, tensorDesc, offset);
}

void UpdateOffsetOp::build(OpBuilder &builder, OperationState &state,
                           Value tensorDesc, llvm::ArrayRef<int64_t> offsets) {
  auto ofrs = getAsIndexOpFoldResult(builder.getContext(), offsets);
  build(builder, state, tensorDesc, ofrs);
}

LogicalResult UpdateOffsetOp::verify() {
  auto tdescTy = getTensorDescType();
  if (!tdescTy.isScattered())
    return emitOpError("Expects a scattered TensorDesc.\n");

  SmallVector<int64_t> expectedOffsetShape = getShapeOf(tdescTy);
  SmallVector<int64_t> offsetShape = getShapeOf(getOffsetsType());
  if (tdescTy.getChunkSizeAsInt() > 1)
    expectedOffsetShape.pop_back();

  if (expectedOffsetShape != offsetShape)
    return emitOpError(
        "Offsets should match TensorDesc except the chunk size dim.");

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_DpasOp
//===----------------------------------------------------------------------===//
LogicalResult DpasOp::verify() {
  int64_t lhsRank = getLhsType().getRank();
  int64_t rhsRank = getRhsType().getRank();
  int64_t resRank = getResultType().getRank();
  auto lhsShape = getLhsType().getShape();
  auto rhsShape = getRhsType().getShape();
  auto resShape = getResultType().getShape();

  if (getAcc() && getAcc().getType() != getResultType())
    return emitOpError("Expecting the acc type to be the same as result.");

  // SIMT code: the size of the B operand has to be a multiple of 32 bits.
  // It skips the semantic check since lack of architecture information.
  // Users need to ensure the correctness.
  if (lhsRank == 1 && rhsRank == 1 && resRank == 1) {
    auto numElems = getRhsType().getNumElements();
    auto elemTy = getRhsType().getElementType();
    auto factor = 32 / elemTy.getIntOrFloatBitWidth();
    if (numElems % factor != 0)
      return emitOpError("Expecting B operand to be a multiple of 32 bits.");
    return success();
  }

  // SIMD code
  if (lhsRank != 2 || (rhsRank != 2 && rhsRank != 3) || resRank != 2)
    return emitOpError(
        "expecting lhs and result to be a 2D vector, and rhs to be either "
        "2D or 3D (packed) vector.");
  auto bK = rhsRank == 3 ? rhsShape[0] * rhsShape[2] : rhsShape[0];
  if (bK != lhsShape[1])
    return emitOpError("K-dimension mismatch.");
  if (lhsShape[0] != resShape[0])
    return emitOpError("M-dimension mismatch.");
  if (rhsShape[1] != resShape[1])
    return emitOpError("N-dimension mismatch.");

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_ConvertLayoutOp
//===----------------------------------------------------------------------===//
LogicalResult ConvertLayoutOp::verify() {
  auto srcMap = getSrcMapAttr();
  auto resMap = getResMapAttr();
  if (!srcMap)
    return emitOpError("expected srcMap.");
  if (!resMap)
    return emitOpError("expected resMap.");

  if (srcMap == resMap)
    return emitOpError("expected different srcMap and resMap.");

  // both srcMap and resMap should be WgLayout or SgLayout at the same time.
  if ((!srcMap.isWgLayout() || !resMap.isWgLayout()) &&
      (!srcMap.isSgLayout() || !resMap.isSgLayout()))
    return emitOpError(
        "expected srcMap and resMap be WgLayout or SgLayout at the same time.");

  auto shape = getSource().getType().getShape();
  if (!XeGPUDialect::isEvenlyDistributable(shape, srcMap))
    return emitOpError("invalid srcMap, data cannot be evenly distributed.");

  if (!XeGPUDialect::isEvenlyDistributable(shape, resMap))
    return emitOpError("invalid resMap, data cannot be evenly distributed.");

  return mlir::success();
}

} // namespace xegpu
} // namespace mlir

#include <mlir/Dialect/XeGPU/IR/XeGPUEnums.cpp.inc>
#define GET_OP_CLASSES
#include <mlir/Dialect/XeGPU/IR/XeGPU.cpp.inc>
