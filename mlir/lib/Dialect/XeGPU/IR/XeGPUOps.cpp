//===- XeGPUOps.cpp - MLIR XeGPU ops implementation -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/Dialect/XeGPU/IR/XeGPU.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/TypeUtilities.h>
#include <numeric>
#include <type_traits>

#define DEBUG_TYPE "xegpu"

namespace mlir {
class Token;

namespace xegpu {

extern bool printDefaultValues();

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
  os.flush();
  return buf;
}

static size_t getRankOf(Value value) {
  if (value.getType().isIntOrIndexOrFloat())
    return 0;
  if (auto ty = llvm::dyn_cast_if_present<MemRefType>(value.getType()))
    return ty.getRank();
  if (auto ty = llvm::dyn_cast_if_present<VectorType>(value.getType()))
    return ty.getRank();
  llvm_unreachable("Unsupported value for getRankOf");
}

static void transpose(llvm::ArrayRef<int64_t> trans,
                      std::vector<int64_t> &shape) {
  std::vector<int64_t> old = shape;
  for (size_t i = 0; i < trans.size(); i++)
    shape[i] = old[trans[i]];
}

static bool verifyAndInferShape(std::vector<int64_t> &shape,
                                SubGroupMapAttr sgMap) {
  if (sgMap) {
    auto wiLayout = sgMap.getWiLayout();
    auto wiData = sgMap.getWiData();

    if ((int64_t)shape.size() != wiData.size() ||
        (int64_t)shape.size() != wiLayout.size()) {
      return false;
    }

    for (size_t i = 0; i < shape.size(); i++) {

      if ((shape[i] % (wiLayout[i] * wiData[i]) != 0 &&
           (wiLayout[i] * wiData[i]) % shape[i] != 0) ||
          shape[i] % wiLayout[i] != 0 || shape[i] % wiData[i] != 0) {
        return false;
      }
      shape[i] /= wiLayout[i];
    }
  }

  return true;
}

static ParseResult
parseOptionalAttrDictWithCustomAttrs(OpAsmParser &parser,
                                     OperationState &result) {
  // no optional attributes, return success
  if (failed(parser.parseOptionalLBrace()))
    return success();

  llvm::SmallDenseSet<StringRef, 8> seenKeys;
  auto parseElt = [&]() -> ParseResult {
    // The name of an attribute can either be a keyword, or a string.
    // as compared to mlir::parseOptionalAttrList, the cases of using
    // TOken::bare_identifier and Token::inttype as key maybe not handlered
    std::string nameId;
    auto loc = parser.getCurrentLocation();
    if (parser.parseOptionalKeywordOrString(&nameId))
      return parser.emitError(loc, "invalid attribute name: ")
             << nameId << ".\n";

    if (nameId.empty())
      return parser.emitError(loc, "expected valid attribute name");

    if (!seenKeys.insert(nameId).second)
      return parser.emitError(loc, "duplicate key '")
             << nameId << "' in dictionary attribute.";

    // Lazy load a dialect in the context if there is a possible namespace.
    auto splitName = StringRef(nameId).split('.');
    if (!splitName.second.empty())
      parser.getContext()->getOrLoadDialect(splitName.first);

    // Try to parse the '=' for the attribute value.
    if (parser.parseEqual()) {
      // If there is no '=', it is treated as a unit attribute.
      result.addAttribute(nameId, parser.getBuilder().getUnitAttr());
      return success();
    }

    // for xegpu specific attributes
    if (nameId == "mode") {
      ModeKindAttr attr;
      return parser.parseCustomAttributeWithFallback(attr, Type{}, nameId,
                                                     result.attributes);
    } else if (nameId == "l1_hint" || nameId == "l2_hint" ||
               nameId == "l3_hint") {
      CacheKindAttr attr;
      return parser.parseCustomAttributeWithFallback(attr, Type{}, nameId,
                                                     result.attributes);
    } else if (nameId == "transpose") {
      // in form of [4, 5], acctually it is a copy of DenseI63ArrayAttr::parse()
      if (succeeded(parser.parseOptionalLSquare())) {
        Attribute attr;
        // handle empty list case
        if (succeeded(parser.parseOptionalRSquare())) {
          attr = DenseI64ArrayAttr::get(parser.getContext(), {});
        } else {
          attr = DenseI64ArrayAttr::parseWithoutBraces(parser, Type{});
          if (failed(parser.parseRSquare()))
            return failure();
        }
        if (!attr)
          return failure();
        result.addAttribute(nameId, attr);
        return success();
      } else {
        // in form of array<i64: 4, 5>
        DenseI64ArrayAttr attr;
        return parser.parseAttribute(attr, nameId, result.attributes);
      }
    } else {
      Attribute attr;
      return parser.parseAttribute(attr, nameId, result.attributes);
    }
  };

  if (parser.parseCommaSeparatedList(parseElt))
    return failure();

  return parser.parseRBrace();
}

//===----------------------------------------------------------------------===//
// XeGPU_CreateNdDescOp
//===----------------------------------------------------------------------===//
void CreateNdDescOp::build(OpBuilder &builder, OperationState &state,
                           Type TensorDesc, Value source, ValueRange offsets,
                           ValueRange shape, ValueRange strides,
                           llvm::ArrayRef<int64_t> static_offsets,
                           ModeKind mode) {
  auto offsetRank = static_offsets.size();
  auto shapeRank = shape.size() ? shape.size() : getRankOf(source);

  size_t dynOffsetRank =
      std::count_if(static_offsets.begin(), static_offsets.end(),
                    [](int64_t d) { return ShapedType::isDynamic(d); });

  // shape and strides should exists at the same time
  // and the final rank for shape and offset (dynamic + static)
  // should be the same
  assert(shape.size() == strides.size() && shapeRank == offsetRank &&
         offsets.size() == dynOffsetRank);

  state.addOperands(source);
  state.addOperands(offsets);
  state.addOperands(shape);
  state.addOperands(strides);
  state.addAttribute(
      getOperandSegmentSizesAttrName(state.name),
      builder.getDenseI32ArrayAttr({1, static_cast<int32_t>(offsets.size()),
                                    static_cast<int32_t>(shape.size()),
                                    static_cast<int32_t>(strides.size())}));
  state.addAttribute(getStaticOffsetsAttrName(state.name),
                     builder.getDenseI64ArrayAttr(static_offsets));
  state.addAttribute(getModeAttrName(state.name),
                     xegpu::ModeKindAttr::get(builder.getContext(), mode));
  state.addTypes(TensorDesc);
}

void CreateNdDescOp::build(OpBuilder &builder, OperationState &state,
                           Type tdesc, Value source,
                           llvm::ArrayRef<OpFoldResult> offsets,
                           ModeKind mode) {
  auto ty = llvm::dyn_cast_if_present<MemRefType>(source.getType());
  assert(ty && ty.hasStaticShape() && offsets.size() == getRankOf(source));

  llvm::SmallVector<int64_t> staticOffsets;
  llvm::SmallVector<Value> dynamicOffsets;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);

  build(builder, state, tdesc, source, dynamicOffsets /* dynamic offsets */,
        ValueRange({}) /* empty dynamic shape */,
        ValueRange({}) /* empty dynamic strides */,
        staticOffsets /* static offsets */, mode);
}

void CreateNdDescOp::build(OpBuilder &builder, OperationState &state,
                           Type tdesc, Value source,
                           llvm::ArrayRef<OpFoldResult> offsets,
                           ValueRange shape, ValueRange stride, ModeKind mode) {
  assert(shape.size() && offsets.size() && stride.size() &&
         shape.size() == stride.size() && shape.size() == offsets.size());

  llvm::SmallVector<int64_t> staticOffsets;
  llvm::SmallVector<Value> dynamicOffsets;

  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);

  build(builder, state, tdesc, source, dynamicOffsets /* dynamic offsets */,
        shape /* dynamic shape */, stride /* dynamic strides */,
        staticOffsets /* static offsets */, mode);
}

ParseResult CreateNdDescOp::parse(OpAsmParser &parser, OperationState &result) {
  // parse the source operand
  llvm::SmallVector<OpAsmParser::UnresolvedOperand> sourceOperands(1);
  llvm::SMLoc sourceOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(sourceOperands[0]))
    return failure();

  // parse the offset operand, in format of [x, y]
  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> offsetsOperands;
  DenseI64ArrayAttr static_offsetsAttr;
  llvm::SMLoc offsetsOperandsLoc = parser.getCurrentLocation();
  if (parseDynamicIndexList(parser, offsetsOperands, static_offsetsAttr))
    return failure();
  result.addAttribute("static_offsets", static_offsetsAttr);

  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> shapeOperands;
  llvm::SMLoc shapeOperandsLoc;

  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> stridesOperands;
  llvm::SMLoc stridesOperandsLoc;
  // parse optional shape and strides, shape and strides should always come
  // together
  if (succeeded(parser.parseOptionalComma())) {
    // parse shape part, in form of [x, y]
    if (parser.parseLSquare())
      return failure();
    shapeOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(shapeOperands))
      return failure();
    if (parser.parseRSquare())
      return failure();

    if (parser.parseComma())
      return failure();

    // parse stride part, in form of [x, y]
    if (parser.parseLSquare())
      return failure();
    stridesOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(stridesOperands))
      return failure();
    if (parser.parseRSquare())
      return failure();
  }

  auto loc = parser.getCurrentLocation();
  if (parseOptionalAttrDictWithCustomAttrs(parser, result))
    return failure();

  if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc)
               << "'" << result.name.getStringRef() << "' op ";
      })))
    return failure();

  if (parser.parseColon())
    return failure();

  llvm::SmallVector<Type> sourceTypes(1);
  if (parser.parseType(sourceTypes[0]))
    return failure();

  if (parser.parseArrow())
    return failure();

  llvm::SmallVector<Type> TensorDescTypes(1);
  if (parser.parseType(TensorDescTypes[0]))
    return failure();
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {1, static_cast<int32_t>(offsetsOperands.size()),
                           static_cast<int32_t>(shapeOperands.size()),
                           static_cast<int32_t>(stridesOperands.size())}));

  result.addTypes(TensorDescTypes);
  if (parser.resolveOperands(sourceOperands, sourceTypes, sourceOperandsLoc,
                             result.operands))
    return failure();

  Type indexType = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(offsetsOperands, indexType, offsetsOperandsLoc,
                             result.operands))
    return failure();
  if (parser.resolveOperands(shapeOperands, indexType, shapeOperandsLoc,
                             result.operands))
    return failure();
  if (parser.resolveOperands(stridesOperands, indexType, stridesOperandsLoc,
                             result.operands))
    return failure();
  return success();
}

void CreateNdDescOp::print(OpAsmPrinter &printer) {
  auto mode = getMode();
  auto printDefaults = printDefaultValues();

  printer << ' ';
  printer << getSource();
  printDynamicIndexList(printer, *this, getDynamicOffsets(),
                        getStaticOffsetsAttr());
  if (!getDynamicShape().empty()) {
    printer << ",";
    printer << ' ' << "[";
    printer << getDynamicShape();
    printer << "]";
  }

  if (!getDynamicStrides().empty()) {
    printer << ",";
    printer << ' ' << "[";
    printer << getDynamicStrides();
    printer << "]";
  }

  llvm::SmallVector<llvm::StringRef> elidedAttrs;
  elidedAttrs.push_back("static_offsets");
  elidedAttrs.push_back("operandSegmentSizes");
  if (!printDefaults && mode == xegpu::ModeKind::SIMT)
    elidedAttrs.push_back("mode");

  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);

  printer << ' ' << ":";
  printer << ' ';
  printer << getSourceType();
  printer << ' ' << "->";
  printer << ' ';
  printer << getTensorDescType();
}

LogicalResult CreateNdDescOp::verify() {
  auto mode = getMode();
  auto isScattered = getTensorDescType().getScattered();
  auto mapping = getTensorDescType().getMapping();

  if (isScattered) {
    return emitOpError("Encoding Attribute of TensorDesc is not expected for "
                       "non-scattered operators.\n");
  }

  if (mode == ModeKind::VC && mapping) {
    return emitOpError("Mapping attribute of TensorDesc is not expected "
                       "for VC mode operations.\n");
  }

  if (mode == ModeKind::SIMT && !mapping) {
    return emitOpError("Expecting SgMap attribute for SIMT mode operators.\n");
  }

  auto offsetRank = getOffsets().size();
  auto shapeRank = getShape().size();
  auto stridesRank = getStrides().size();
  auto baseRank = getRankOf(getSource()) ? getRankOf(getSource()) : 2;

  if (offsetRank != shapeRank || shapeRank != stridesRank ||
      shapeRank != baseRank)
    return emitOpError(
        "Expecting the rank of shape, strides, offsets and memref type "
        "should match with each other (they currently should be 2D).");

  return success();
}

xegpu::TensorDescType CreateNdDescOp::getTensorDescType() {
  return getTensorDesc().getType();
}

llvm::SmallVector<OpFoldResult> CreateNdDescOp::getOffsets() {
  llvm::SmallVector<OpFoldResult> offsets;
  auto dynamicOffsets = getDynamicOffsets(); // given by dynamic_offsets
                                             // variable
  auto staticOffsets = getStaticOffsets(); // given by static_offsets attribute

  // in case static_offsets is missing
  if (staticOffsets.size() == 0) {
    offsets.assign(dynamicOffsets.begin(), dynamicOffsets.end());
    return offsets;
  }

  for (size_t i = 0, j = 0; i < staticOffsets.size(); i++) {
    if (ShapedType::isDynamic(staticOffsets[i])) {
      assert(j < dynamicOffsets.size());
      offsets.push_back(dynamicOffsets[j++]);
    } else {
      auto ty = IndexType::get(getContext());
      auto attr = IntegerAttr::get(ty, staticOffsets[i]);
      offsets.push_back(attr);
    }
  }
  return offsets;
}

llvm::ArrayRef<int64_t> CreateNdDescOp::getStaticShape() {
  auto rank = getTensorDescType().getRank();
  static llvm::SmallVector<int64_t> dyn(rank, ShapedType::kDynamic);
  auto srcTy = llvm::dyn_cast_if_present<MemRefType>(getSourceType());
  if (srcTy)
    return srcTy.getShape();

  return dyn;
}

llvm::SmallVector<OpFoldResult> CreateNdDescOp::getShape() {
  llvm::SmallVector<OpFoldResult> shape;
  auto dynShape = getDynamicShape();
  if (dynShape.size()) {
    shape.append(dynShape.begin(), dynShape.end());
    return shape;
  }

  auto ty = llvm::dyn_cast_if_present<MemRefType>(getSourceType());
  if (ty && ty.hasStaticShape()) {
    for (auto dim : ty.getShape()) {
      auto attr = IntegerAttr::get(IndexType::get(getContext()), dim);
      shape.push_back(attr);
    }
    return shape;
  }

  llvm_unreachable("Unexpected error in CreateNdDescOp. "
                   "The shape information is missing.\n");
}

llvm::ArrayRef<int64_t> CreateNdDescOp::getStaticStrides() {
  auto rank = getTensorDescType().getRank();
  static llvm::SmallVector<int64_t> dyn(rank, ShapedType::kDynamic);
  auto srcTy = llvm::dyn_cast_if_present<MemRefType>(getSourceType());
  if (srcTy) {
    auto [strides, offset] = getStridesAndOffset(srcTy);
    return strides;
  }
  return dyn;
}

llvm::SmallVector<OpFoldResult> CreateNdDescOp::getStrides() {
  llvm::SmallVector<OpFoldResult> strides;

  auto dynStrides = getDynamicStrides();
  if (dynStrides.size()) {
    strides.append(dynStrides.begin(), dynStrides.end());
    return strides;
  }

  auto ty = llvm::dyn_cast_if_present<MemRefType>(getSourceType());
  if (ty && ty.hasStaticShape()) {
    auto [staticStrides, offset] = getStridesAndOffset(ty);
    for (auto dim : staticStrides) {
      auto attr = IntegerAttr::get(IndexType::get(getContext()), dim);
      strides.push_back(attr);
    }
    return strides;
  }
  llvm_unreachable("Unexpected error in CreateNdDescOp. The strides "
                   "information is missing.\n");
}

/// Return the element type of the TensorDesc
Type CreateNdDescOp::getElementType() {
  return getTensorDescType().getElementType();
}

/// Return the shape of the TensorDesc
llvm::ArrayRef<int64_t> CreateNdDescOp::getTensorDescShape() {
  return getTensorDescType().getShape();
}

//===----------------------------------------------------------------------===//
// XeGPU_LoadNDOp
//===----------------------------------------------------------------------===//

ParseResult LoadNDOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SmallVector<OpAsmParser::UnresolvedOperand> Operands(1);
  llvm::SMLoc OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(Operands[0]))
    return failure();

  auto loc = parser.getCurrentLocation();
  if (parseOptionalAttrDictWithCustomAttrs(parser, result))
    return failure();

  if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc)
               << "'" << result.name.getStringRef() << "' op ";
      })))
    return failure();

  if (parser.parseColon())
    return failure();

  llvm::SmallVector<Type> Types(1);
  if (parser.parseType(Types[0]))
    return failure();

  if (parser.parseArrow())
    return failure();

  llvm::SmallVector<Type> valueTypes(1);
  if (parser.parseType(valueTypes[0]))
    return failure();

  result.addTypes(valueTypes);
  if (parser.resolveOperands(Operands, Types, OperandsLoc, result.operands))
    return failure();

  return success();
}

void LoadNDOp::print(OpAsmPrinter &printer) {
  auto mode = getMode();
  auto printDefaults = printDefaultValues();

  printer << ' ';
  printer << getTensorDesc();

  llvm::SmallVector<llvm::StringRef> elidedAttrs;
  if (!printDefaults && mode == xegpu::ModeKind::SIMT)
    elidedAttrs.push_back("mode");

  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);

  printer << ' ' << ":";
  printer << ' ';
  printer << getTensorDesc().getType();
  printer << ' ' << "->";
  printer << ' ';
  printer << getValue().getType();
}

LogicalResult LoadNDOp::verify() {
  auto tdescTy = getTensorDescType();
  auto valueTy = getValueType();

  if (tdescTy.getRank() != 2)
    return emitOpError(
        "The TensorDesc for LoadNDOp should be a 2D TensorDesc.");

  if (!valueTy)
    return emitOpError("Invalid result, it should be a VectorType.\n");

  auto tdescElemTy = tdescTy.getElementType();
  auto valueElemTy = valueTy.getElementType();

  if (tdescElemTy != valueElemTy)
    return emitOpError(
        "Value should have the same element type as TensorDesc.");

  auto mode = getMode();
  auto tdescShape = tdescTy.getShape().vec();
  auto valueShape = valueTy.getShape().vec();
  auto array_len = tdescTy.getArrayLength();

  if (mode == ModeKind::SIMT) {
    auto sgMap = tdescTy.getMapping();
    if (!sgMap) {
      return emitOpError(
          "Expecting SgMap attribute for SIMT mode operators.\n");
    }

    if (!verifyAndInferShape(tdescShape, sgMap)) {
      return emitOpError("Failed to infer the shape.")
             << "The new shape[i] should meet the following condistions "
                "for SubGroupMapAttr: "
             << "\n\ttdescShape[i] % mma_block_size[i] == 0 (if it has) && "
             << "\n\ttdescShape[i] % wi_layout[i] == 0 && "
             << "\n\ttdescShape[i] % wi_data[i] == 0 && "
             << "\n\t(tdescShape[i] % (wi_layout[i] * wi_data[i]) == 0 || "
             << "\n\t (wi_layout[i] * wi_data[i]) % tdescShape[i] == 0).\n";
    }
  }

  if (getTranspose()) {
    auto trans = getTranspose().value();
    if (tdescShape.size() >= trans.size())
      transpose(trans, tdescShape);
    else
      emitWarning("Invalid transpose attr. It is ignored.");
  }

  if (getVnniAxis()) {
    auto axis = getVnniAxis().value();
    auto vnni_factor = valueShape.back();
    tdescShape[axis] /= vnni_factor;
    tdescShape.push_back(vnni_factor);
  }

  if (array_len > 1) {
    auto it = tdescShape.begin();
    tdescShape.insert(it, array_len);
  }

  if (tdescShape != valueShape)
    return emitOpError("Result shape doesn't match TensorDesc shape.")
           << "\nThe expected shape is " << makeString(tdescShape) << "."
           << "\nBut the given shape is " << makeString(valueShape) << "."
           << "\nIn VC mode, when VNNI is not enabled, the result should have "
           << "the same shape (or transposed shape if transpose is enabled) "
           << "as TensorDesc; \nwhen VNNI is enabled, the result should have "
           << "one more dimention than the TensorDesc, with last dimention "
           << "having vnni factor, \nbut having same number of total data "
           << "elements. The vnni factor are typically calculated as "
           << "simd_lane_width / elementTypeBitWidth. \nFor element type "
           << "having more than 32 bits, vnni shouldn't be used. \nIn SIMT "
           << "mode, the shape is derived from the mapping attributes.\n";
  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_StoreNDOp
//===----------------------------------------------------------------------===//
ParseResult StoreNDOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SmallVector<OpAsmParser::UnresolvedOperand> Operands(2);
  llvm::SMLoc OperandsLoc = parser.getCurrentLocation();
  // parse value
  if (parser.parseOperand(Operands[0]))
    return failure();

  if (parser.parseComma())
    return failure();

  // parse TensorDesc
  if (parser.parseOperand(Operands[1]))
    return failure();

  // parse optional attributes
  auto loc = parser.getCurrentLocation();
  if (parseOptionalAttrDictWithCustomAttrs(parser, result))
    return failure();

  if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc)
               << "'" << result.name.getStringRef() << "' op ";
      })))
    return failure();

  if (parser.parseColon())
    return failure();

  llvm::SmallVector<Type> Types;
  if (parser.parseTypeList(Types))
    return failure();

  if (parser.resolveOperands(Operands, Types, OperandsLoc, result.operands))
    return failure();

  return success();
}

void StoreNDOp::print(OpAsmPrinter &printer) {
  auto mode = getMode();
  auto printDefaults = printDefaultValues();

  printer << ' ';
  printer << getValue();
  printer << ",";
  printer << ' ';
  printer << getTensorDesc();

  llvm::SmallVector<llvm::StringRef> elidedAttrs;
  if (!printDefaults && mode == xegpu::ModeKind::SIMT)
    elidedAttrs.push_back("mode");
  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);

  printer << ' ' << ":";
  printer << ' ';
  printer << getValue().getType();
  printer << ",";
  printer << ' ';
  printer << getTensorDesc().getType();
}

LogicalResult StoreNDOp::verify() {
  auto dstTy = getTensorDesc().getType();                        // Tile
  auto valTy = llvm::dyn_cast<VectorType>(getValue().getType()); // Vector

  if (dstTy.getRank() != 2)
    return emitOpError(
        "The TensorDesc for StoreNdOp should be a 2D TensorDesc.");

  if (!valTy)
    return emitOpError("Invalid value operand, it should be a VectorType.\n");

  auto dstElemTy = dstTy.getElementType();
  auto valElemTy = valTy.getElementType();

  if (dstElemTy != valElemTy) {
    return emitOpError("The elem type of value (vector) shape doesn't match "
                       "the elem type of memory (dst) shape.\n");
  }

  auto mode = getMode();

  if (mode == ModeKind::VC) { // for VC mode, no attr attached
    if (dstTy.getShape() != valTy.getShape())
      return emitOpError("In VC mode, the value (vector) shape doesn't match "
                         "the memory (dst) shape.\n");
  } else {
    auto mapping = dstTy.getMapping();
    if (!mapping) {
      return emitOpError(
          "Expecting SgMap attribute for SIMT mode operators.\n");
    }

    SubGroupMapAttr sgMap;
    std::vector<int64_t> shape = dstTy.getShape().vec();

    sgMap = llvm::dyn_cast<SubGroupMapAttr>(mapping);

    if (!verifyAndInferShape(shape, sgMap)) {
      return emitOpError("Failed to infer the shape.")
             << "The new shape[i] should meet the following condistions "
                "for SubGroupMapAttr: "
             << "\n\ttdescShape[i] % mma_block_size[i] == 0 (if it has) && "
             << "\n\ttdescShape[i] % wi_layout[i] == 0 && "
             << "\n\ttdescShape[i] % wi_data[i] == 0 && "
             << "\n\t(tdescShape[i] % (wi_layout[i] * wi_data[i]) == 0 || "
             << "\n\t (wi_layout[i] * wi_data[i]) % tdescShape[i] == 0).\n";
    }

    if (shape != valTy.getShape().vec())
      return emitOpError(
          "In SIMT mode, the value (vector) shape doesn't match the memory"
          "(dst) shape as derived according to the mapping rule.\n");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_PrefetchNDOp
//===----------------------------------------------------------------------===//
ParseResult PrefetchNDOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SmallVector<OpAsmParser::UnresolvedOperand> TensorDescOperands(1);
  llvm::SmallVector<Type> TensorDescTypes(1);
  llvm::SMLoc TensorDescOperandsLoc;

  TensorDescOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(TensorDescOperands[0]))
    return failure();

  auto loc = parser.getCurrentLocation();
  if (parseOptionalAttrDictWithCustomAttrs(parser, result))
    return failure();

  if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc)
               << "'" << result.name.getStringRef() << "' op ";
      })))
    return failure();

  if (parser.parseColon())
    return failure();

  if (parser.parseType(TensorDescTypes[0]))
    return failure();
  if (parser.resolveOperands(TensorDescOperands, TensorDescTypes,
                             TensorDescOperandsLoc, result.operands))
    return failure();
  return success();
}

void PrefetchNDOp::print(OpAsmPrinter &printer) {
  auto mode = getMode();
  auto printDefaults = printDefaultValues();

  printer << ' ';
  printer << getTensorDesc();

  llvm::SmallVector<llvm::StringRef> elidedAttrs;
  if (!printDefaults && mode == xegpu::ModeKind::SIMT)
    elidedAttrs.push_back("mode");
  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);

  printer << ' ' << ":";
  printer << ' ';
  printer << getTensorDesc().getType();
}

//===----------------------------------------------------------------------===//
// XeGPU_UpdateNDOffsetOp
//===----------------------------------------------------------------------===//
ParseResult UpdateNDOffsetOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  llvm::SmallVector<OpAsmParser::UnresolvedOperand> TensorDescOperands(1);
  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 4> offsetsOperands;
  llvm::SmallVector<Type> TensorDescTypes(1);
  llvm::SmallVector<Type> resultTypes(1);
  llvm::SMLoc TensorDescOperandsLoc;
  llvm::SMLoc offsetsOperandsLoc;

  TensorDescOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(TensorDescOperands[0]))
    return failure();
  if (parser.parseComma())
    return failure();

  // parse offsets, e.g.,  [x, y]
  if (succeeded(parser.parseOptionalLSquare())) {
    offsetsOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(offsetsOperands))
      return failure();
    if (parser.parseRSquare())
      return failure();
  }

  if (parseOptionalAttrDictWithCustomAttrs(parser, result))
    return failure();

  auto loc = parser.getCurrentLocation();
  if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc)
               << "'" << result.name.getStringRef() << "' op ";
      })))
    return failure();

  if (parser.parseColon())
    return failure();

  if (parser.parseType(TensorDescTypes[0]))
    return failure();
  if (parser.parseArrow())
    return failure();

  if (parser.parseType(resultTypes[0]))
    return failure();
  result.addTypes(resultTypes);
  if (parser.resolveOperands(TensorDescOperands, TensorDescTypes,
                             TensorDescOperandsLoc, result.operands))
    return failure();

  Type indexType = parser.getBuilder().getIndexType();
  if (parser.resolveOperands(offsetsOperands, indexType, offsetsOperandsLoc,
                             result.operands))
    return failure();
  return success();
}

void UpdateNDOffsetOp::print(OpAsmPrinter &printer) {
  auto mode = getMode();
  auto printDefaults = printDefaultValues();

  printer << ' ';
  printer << getTensorDesc();
  printer << ",";
  if (!getOffsets().empty()) {
    printer << ' ' << "[";
    printer << getOffsets();
    printer << "]";
  }

  llvm::SmallVector<llvm::StringRef> elidedAttrs;
  if (!printDefaults && mode == xegpu::ModeKind::SIMT)
    elidedAttrs.push_back("mode");
  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);

  printer << ' ' << ":";
  printer << ' ';
  printer << getTensorDesc().getType();
  printer << ' ' << "->";
  printer << ' ';
  printer << getResult().getType();
}

LogicalResult UpdateNDOffsetOp::verify() {
  // number of offsets specified must match the rank of the tensor descriptor
  if (getTensorDesc().getType().getRank() != (int64_t)getOffsets().size()) {
    return emitOpError("Invalid number of offsets.");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_CreateDescOp
//===----------------------------------------------------------------------===//
void CreateDescOp::build(OpBuilder &builder, OperationState &state,
                         TensorDescType TensorDesc, Value source, Value offsets,
                         uint32_t chunk_size_per_lane) {
  state.addOperands(source);
  state.addOperands(offsets);
  state.getOrAddProperties<Properties>().chunk_size_per_lane =
      builder.getIntegerAttr(builder.getIntegerType(32), chunk_size_per_lane);
  state.getOrAddProperties<Properties>().mode =
      ModeKindAttr::get(builder.getContext(), ModeKind::VC);
  state.addTypes(TensorDesc);
}

void CreateDescOp::build(OpBuilder &builder, OperationState &state,
                         TensorDescType TensorDesc, Value source, Value offsets,
                         IntegerAttr chunk_size_per_lane) {
  state.addOperands(source);
  state.addOperands(offsets);
  if (chunk_size_per_lane)
    state.getOrAddProperties<Properties>().chunk_size_per_lane =
        chunk_size_per_lane;
  state.getOrAddProperties<Properties>().mode =
      ModeKindAttr::get(builder.getContext(), ModeKind::VC);
  state.addTypes(TensorDesc);
}

ParseResult CreateDescOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SmallVector<OpAsmParser::UnresolvedOperand> Operands(2);
  llvm::SmallVector<Type> Types(2);
  llvm::SMLoc operandsLoc = parser.getCurrentLocation();
  // parse the source operand
  if (parser.parseOperand(Operands[0]))
    return failure();

  if (parser.parseComma())
    return failure();

  // parse the offset operand
  if (parser.parseOperand(Operands[1]))
    return failure();

  // parse the optional attributes
  auto loc = parser.getCurrentLocation();
  if (parseOptionalAttrDictWithCustomAttrs(parser, result))
    return failure();

  if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc)
               << "'" << result.name.getStringRef() << "' op ";
      })))
    return failure();

  if (parser.parseColon())
    return failure();

  if (parser.parseType(Types[0]))
    return failure();
  if (parser.parseComma())
    return failure();

  if (parser.parseType(Types[1]))
    return failure();
  if (parser.parseArrow())
    return failure();

  llvm::SmallVector<Type> TensorDescTypes(1);
  if (parser.parseType(TensorDescTypes[0]))
    return failure();

  result.addTypes(TensorDescTypes);
  if (parser.resolveOperands(Operands, Types, operandsLoc, result.operands))
    return failure();
  return success();
}

void CreateDescOp::print(OpAsmPrinter &printer) {
  auto mode = getMode();
  auto chunk = getChunkSizePerLane();
  auto printDefaults = printDefaultValues();

  printer << ' ';
  printer << getSource();
  printer << ",";
  printer << ' ';
  printer << getOffsets();

  llvm::SmallVector<llvm::StringRef> elidedAttrs;
  if (!printDefaults) {
    if (mode == xegpu::ModeKind::SIMT)
      elidedAttrs.push_back("mode");
    if (chunk == 1)
      elidedAttrs.push_back("chunk_size_per_lane");
  }
  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);

  printer << ' ' << ":";
  printer << ' ';
  printer << getSource().getType();
  printer << ",";
  printer << ' ';
  printer << getOffsets().getType();
  printer << ' ' << "->";
  printer << ' ';
  printer << getTensorDesc().getType();
}

LogicalResult CreateDescOp::verify() {
  auto mode = getMode();
  auto mapping = getTensorDesc().getType().getMapping();
  auto offsetTy = getOffsets().getType();
  auto tdescTy = getTensorDesc().getType();
  auto chunkSize = getChunkSizePerLane();

  if (mode == ModeKind::SIMT || mapping) {
    return emitOpError("CreateDescOp only support VC mode and mapping "
                       "attribute of TensorDesc is not expected.\n");
  }

  if (getRankOf(getSource()) > 2)
    return emitOpError(
        "Expecting the source is a 1D/2D memref or pointer (uint64_t).");

  if (!tdescTy.getScattered())
    return emitOpError(
        "Expecting the presence of ScatteredAttr for tensor descriptor.");

  // Infer the TensorDesc shape
  std::vector<int64_t> shape;
  if (llvm::isa<VectorType>(offsetTy)) {
    shape = llvm::dyn_cast<VectorType>(offsetTy).getShape().vec();
    if (shape.size() != 1)
      return emitOpError("Expecting the offset is a 1D vector.");
  }

  if (chunkSize != 1) {
    shape.push_back(chunkSize);
  }

  auto tdescShape = tdescTy.getShape();
  if (shape != tdescShape.vec()) {
    return emitOpError("Expecting dimensions of offsets is the same as the "
                       "tensor descriptor, or one less than.");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_LoadGatherOp
//===----------------------------------------------------------------------===//
void LoadGatherOp::build(OpBuilder &builder, OperationState &state, Type value,
                         Value TensorDesc, Value mask, IntegerAttr vnni_axis,
                         DenseI64ArrayAttr transpose, CacheKindAttr l1_hint,
                         CacheKindAttr l2_hint, CacheKindAttr l3_hint) {
  state.addOperands(TensorDesc);
  state.addOperands(mask);
  if (vnni_axis)
    state.getOrAddProperties<Properties>().vnni_axis = vnni_axis;

  if (transpose)
    state.getOrAddProperties<Properties>().transpose = transpose;

  if (l1_hint)
    state.getOrAddProperties<Properties>().l1_hint = l1_hint;

  if (l2_hint)
    state.getOrAddProperties<Properties>().l2_hint = l2_hint;

  if (l3_hint)
    state.getOrAddProperties<Properties>().l3_hint = l3_hint;

  state.getOrAddProperties<Properties>().mode =
      ModeKindAttr::get(builder.getContext(), ModeKind::VC);
  state.addTypes(value);
}

void LoadGatherOp::build(OpBuilder &builder, OperationState &state, Type value,
                         Value TensorDesc, Value mask, IntegerAttr vnni_axis,
                         DenseI64ArrayAttr transpose, CacheKind l1_hint,
                         CacheKind l2_hint, CacheKind l3_hint) {
  state.addOperands(TensorDesc);
  state.addOperands(mask);
  if (vnni_axis)
    state.getOrAddProperties<Properties>().vnni_axis = vnni_axis;

  if (transpose)
    state.getOrAddProperties<Properties>().transpose = transpose;

  state.getOrAddProperties<Properties>().l1_hint =
      CacheKindAttr::get(builder.getContext(), l1_hint);
  state.getOrAddProperties<Properties>().l2_hint =
      CacheKindAttr::get(builder.getContext(), l2_hint);
  state.getOrAddProperties<Properties>().l3_hint =
      CacheKindAttr::get(builder.getContext(), l3_hint);
  state.getOrAddProperties<Properties>().mode =
      ModeKindAttr::get(builder.getContext(), ModeKind::VC);
  state.addTypes(value);
}

ParseResult LoadGatherOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SmallVector<OpAsmParser::UnresolvedOperand> Operands(2);
  llvm::SmallVector<Type> Types(2);
  llvm::SmallVector<Type> valueTypes(1);
  llvm::SMLoc OperandsLoc;

  OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(Operands[0]))
    return failure();

  if (parser.parseComma())
    return failure();

  if (parser.parseOperand(Operands[1]))
    return failure();

  auto loc = parser.getCurrentLocation();
  if (parseOptionalAttrDictWithCustomAttrs(parser, result))
    return failure();
  if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc)
               << "'" << result.name.getStringRef() << "' op ";
      })))
    return failure();

  if (parser.parseColon())
    return failure();

  if (parser.parseType(Types[0]))
    return failure();

  if (parser.parseComma())
    return failure();

  if (parser.parseType(Types[1]))
    return failure();

  if (parser.parseArrow())
    return failure();

  if (parser.parseType(valueTypes[0]))
    return failure();

  result.addTypes(valueTypes);

  if (parser.resolveOperands(Operands, Types, OperandsLoc, result.operands))
    return failure();

  return success();
}

void LoadGatherOp::print(OpAsmPrinter &printer) {
  auto mode = getMode();
  auto printDefaults = printDefaultValues();

  printer << ' ';
  printer << getTensorDesc();
  printer << ",";
  printer << ' ';
  printer << getMask();

  llvm::SmallVector<llvm::StringRef> elidedAttrs;
  if (!printDefaults && mode == xegpu::ModeKind::SIMT)
    elidedAttrs.push_back("mode");
  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);

  printer << ' ' << ":";
  printer << ' ';
  printer << getTensorDesc().getType();
  printer << ",";
  printer << ' ';
  printer << getMask().getType();
  printer << ' ' << "->";
  printer << ' ';
  printer << getValue().getType();
}

LogicalResult LoadGatherOp::verify() {
  auto tdescTy = getTensorDesc().getType();
  auto maskTy = getMask().getType();
  auto valueTy = getValue().getType();

  if (!tdescTy.getScattered())
    return emitOpError(
        "LoadGatherOp only works on TensorDesc with ScatteredAttr.");

  auto getElementType = [&](Type type) -> Type {
    if (type.isIntOrIndexOrFloat())
      return type;
    else if (llvm::isa<VectorType>(type))
      return llvm::dyn_cast<VectorType>(type).getElementType();
    else if (llvm::isa<TensorDescType>(type))
      return llvm::dyn_cast<TensorDescType>(type).getElementType();
    llvm_unreachable("Unsupported type.");
    return type;
  };

  auto tdescElemTy = getElementType(tdescTy);
  auto valueElemTy = getElementType(valueTy);
  if (tdescElemTy != valueElemTy)
    return emitOpError(
        "Value should have the same element type as TensorDesc.");

  auto getShape = [&](Type type) -> std::vector<int64_t> {
    std::vector<int64_t> shape;
    if (type.isIntOrIndexOrFloat())
      shape.push_back(1);
    else if (llvm::isa<VectorType>(type))
      shape = llvm::dyn_cast<VectorType>(type).getShape().vec();
    else
      llvm_unreachable("Unsupported type.");
    return shape;
  };

  std::vector<int64_t> maskShape = getShape(maskTy);
  std::vector<int64_t> valueShape = getShape(valueTy);
  std::vector<int64_t> tdescShape = tdescTy.getShape().vec();

  if (tdescShape != maskShape)
    return emitOpError("Mask should have the same shape as TensorDesc.");

  auto mode = getMode();
  auto mapping = tdescTy.getMapping();
  if (mode == ModeKind::SIMT || mapping) {
    return emitOpError("LoadGatherOp only supports VC mode and mapping "
                       "attribute of TensorDesc is not expected.\n");
  }

  if (getTransposeAttr()) {
    auto trans = getTranspose().value();
    if (tdescShape.size() < trans.size())
      return emitWarning("Invalid transpose attr. It is ignored.");
    transpose(trans, tdescShape);
  }

  if (getVnniAxis()) {
    auto axis = getVnniAxis().value();
    auto vnni_factor = valueShape.back();
    tdescShape[axis] /= vnni_factor;
    tdescShape.push_back(vnni_factor);
  }

  if (valueShape != tdescShape)
    return emitOpError(
        "Result shape doesn't match TensorDesc shape. when VNNI is not enabled,"
        "the result should have the same shape (or transposed shape if "
        "transpose is also enabled) as TensorDesc. When VNNI is enabled, "
        "the result should have one more dimention than the TensorDesc, "
        "with last dimention having vnni factor, but having same number of"
        "total data elements. The vnni factor are typically calculated as "
        "simd_lane_width/elementTypeBitWidth. For element type having "
        "more than 32 bits, vnni shouldn't be used.\n");

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_StoreScatterOp
//===----------------------------------------------------------------------===//
void StoreScatterOp::build(OpBuilder &builder, OperationState &state,
                           Value value, Value TensorDesc, Value mask,
                           CacheKindAttr l1_hint, CacheKindAttr l2_hint,
                           CacheKindAttr l3_hint) {
  state.addOperands(value);
  state.addOperands(TensorDesc);
  state.addOperands(mask);
  if (l1_hint)
    state.getOrAddProperties<Properties>().l1_hint = l1_hint;
  if (l2_hint)
    state.getOrAddProperties<Properties>().l2_hint = l2_hint;
  if (l3_hint)
    state.getOrAddProperties<Properties>().l3_hint = l3_hint;
  state.getOrAddProperties<Properties>().mode =
      ModeKindAttr::get(builder.getContext(), ModeKind::VC);
}

void StoreScatterOp::build(OpBuilder &builder, OperationState &state,
                           Value value, Value TensorDesc, Value mask,
                           CacheKind l1_hint, CacheKind l2_hint,
                           CacheKind l3_hint) {
  state.addOperands(value);
  state.addOperands(TensorDesc);
  state.addOperands(mask);
  state.getOrAddProperties<Properties>().l1_hint =
      CacheKindAttr::get(builder.getContext(), l1_hint);
  state.getOrAddProperties<Properties>().l2_hint =
      CacheKindAttr::get(builder.getContext(), l2_hint);
  ;
  state.getOrAddProperties<Properties>().l3_hint =
      CacheKindAttr::get(builder.getContext(), l3_hint);
  ;
  state.getOrAddProperties<Properties>().mode =
      ModeKindAttr::get(builder.getContext(), ModeKind::VC);
}

ParseResult StoreScatterOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SmallVector<OpAsmParser::UnresolvedOperand> Operands;
  llvm::SmallVector<Type> Types;
  llvm::SMLoc OperandsLoc;

  OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(Operands))
    return failure();

  auto loc = parser.getCurrentLocation();
  if (parseOptionalAttrDictWithCustomAttrs(parser, result))
    return failure();
  if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc)
               << "'" << result.name.getStringRef() << "' op ";
      })))
    return failure();

  if (parser.parseColon())
    return failure();

  if (parser.parseTypeList(Types))
    return failure();

  if (parser.resolveOperands(Operands, Types, OperandsLoc, result.operands))
    return failure();

  return success();
}

void StoreScatterOp::print(OpAsmPrinter &printer) {
  auto mode = getMode();
  auto printDefaults = printDefaultValues();

  printer << ' ';
  printer << getValue();
  printer << ",";
  printer << ' ';
  printer << getTensorDesc();
  printer << ",";
  printer << ' ';
  printer << getMask();

  llvm::SmallVector<llvm::StringRef> elidedAttrs;
  if (!printDefaults && mode == xegpu::ModeKind::SIMT)
    elidedAttrs.push_back("mode");
  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);

  printer << ' ' << ":";
  printer << ' ';
  printer << getValue().getType();
  printer << ",";
  printer << ' ';
  printer << getTensorDesc().getType();
  printer << ",";
  printer << ' ';
  printer << getMask().getType();
}

LogicalResult StoreScatterOp::verify() {
  auto tdescTy = getTensorDesc().getType();
  auto valueTy = getValue().getType();
  auto maskTy = getMask().getType();
  auto mode = getMode();
  auto mapping = tdescTy.getMapping();

  if (mode != ModeKind::VC || mapping)
    return emitOpError("StoreScatterOp only supports VC mode and mapping "
                       "attribute of TensorDesc is not expected.\n");

  if (!tdescTy.getScattered())
    return emitOpError("Invalid TensorDesc. StoreScatterOp only works on "
                       "TensorDescs with ScatteredAttr.");

  auto getShape = [&](Type type) -> std::vector<int64_t> {
    std::vector<int64_t> shape;
    if (type.isIntOrIndexOrFloat())
      shape.push_back(1);
    else if (llvm::isa<VectorType>(type))
      shape = llvm::dyn_cast<VectorType>(type).getShape().vec();
    else
      llvm_unreachable("Unsupported type.");
    return shape;
  };

  std::vector<int64_t> maskShape = getShape(maskTy);
  std::vector<int64_t> valueShape = getShape(valueTy);
  std::vector<int64_t> tdescShape = tdescTy.getShape().vec();

  if (valueShape != maskShape) {
    return emitOpError("Mask and value should have the same shape/size");
  }

  if (tdescShape != valueShape) {
    return emitOpError("TensorDesc shape and value shape doesn't match. ")
           << "The expected/derived value shape is: " << makeString(tdescShape)
           << ".\nMask and value should have the same shape/size as "
              "TensorDesc.\n";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_PrefetchOp
//===----------------------------------------------------------------------===//
void PrefetchOp::build(OpBuilder &builder, OperationState &state,
                       Value TensorDesc, CacheKindAttr l1_hint,
                       CacheKindAttr l2_hint, CacheKindAttr l3_hint) {
  state.addOperands(TensorDesc);
  if (l1_hint)
    state.getOrAddProperties<Properties>().l1_hint = l1_hint;

  if (l2_hint)
    state.getOrAddProperties<Properties>().l2_hint = l2_hint;

  if (l3_hint)
    state.getOrAddProperties<Properties>().l3_hint = l3_hint;

  state.getOrAddProperties<Properties>().mode =
      ModeKindAttr::get(builder.getContext(), ModeKind::VC);
}

void PrefetchOp::build(OpBuilder &builder, OperationState &state,
                       Value TensorDesc, CacheKind l1_hint, CacheKind l2_hint,
                       CacheKind l3_hint) {
  state.addOperands(TensorDesc);
  state.getOrAddProperties<Properties>().l1_hint =
      CacheKindAttr::get(builder.getContext(), l1_hint);
  state.getOrAddProperties<Properties>().l2_hint =
      CacheKindAttr::get(builder.getContext(), l2_hint);
  state.getOrAddProperties<Properties>().l3_hint =
      CacheKindAttr::get(builder.getContext(), l3_hint);
  state.getOrAddProperties<Properties>().mode =
      ModeKindAttr::get(builder.getContext(), ModeKind::VC);
}

ParseResult PrefetchOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SmallVector<OpAsmParser::UnresolvedOperand> TensorDescOperands(1);
  llvm::SmallVector<Type> TensorDescTypes(1);
  llvm::SMLoc TensorDescOperandsLoc;

  TensorDescOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(TensorDescOperands[0]))
    return failure();

  auto loc = parser.getCurrentLocation();
  if (parseOptionalAttrDictWithCustomAttrs(parser, result))
    return failure();
  if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc)
               << "'" << result.name.getStringRef() << "' op ";
      })))
    return failure();

  if (parser.parseColon())
    return failure();

  if (parser.parseType(TensorDescTypes[0]))
    return failure();

  if (parser.resolveOperands(TensorDescOperands, TensorDescTypes,
                             TensorDescOperandsLoc, result.operands))
    return failure();
  return success();
}

void PrefetchOp::print(OpAsmPrinter &printer) {
  auto mode = getMode();
  auto printDefaults = printDefaultValues();

  printer << ' ';
  printer << getTensorDesc();

  llvm::SmallVector<llvm::StringRef> elidedAttrs;
  if (!printDefaults && mode == xegpu::ModeKind::SIMT)
    elidedAttrs.push_back("mode");
  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);

  printer << ' ' << ":";
  printer << ' ';
  printer << getTensorDesc().getType();
}

LogicalResult PrefetchOp::verify() {
  auto mode = getMode();
  auto tdescTy = getTensorDesc().getType();
  auto mapping = tdescTy.getMapping();

  auto isValidHint = [&](CacheKindAttr attr) -> bool {
    if (!attr)
      return true;
    auto kind = attr.getValue();
    return kind == CacheKind::CACHED || kind == CacheKind::UNCACHED ||
           kind == CacheKind::STREAMING || kind == CacheKind::READ_INVALIDATE;
  };

  if (!isValidHint(getL1HintAttr()))
    return emitOpError("invlid l1_hint: ") << getL1HintAttr();

  if (!isValidHint(getL2HintAttr()))
    return emitOpError("invlid l2_hint: ") << getL2HintAttr();

  if (!isValidHint(getL3HintAttr()))
    return emitOpError("invlid l3_hint: ") << getL3HintAttr();

  if (!tdescTy.getScattered())
    return emitOpError("Invalid TensorDesc. PrefetchOp only works on "
                       "TensorDescs with ScatteredAttr.");

  if (mode != ModeKind::VC || mapping) {
    return emitOpError("PrefetchOp only supports VC mode, and mapping "
                       "attribute of TensorDesc is not expected.\n");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_UpdateOffsetOp
//===----------------------------------------------------------------------===//
void UpdateOffsetOp::build(OpBuilder &builder, OperationState &state,
                           Type result, Value TensorDesc, Value offsets) {
  state.addOperands(TensorDesc);
  state.addOperands(offsets);
  state.getOrAddProperties<Properties>().mode =
      xegpu::ModeKindAttr::get(builder.getContext(), xegpu::ModeKind::VC);
  state.addTypes(result);
}

ParseResult UpdateOffsetOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SmallVector<OpAsmParser::UnresolvedOperand> Operands;
  llvm::SmallVector<Type> Types;

  auto OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(Operands))
    return failure();

  auto loc = parser.getCurrentLocation();
  if (parseOptionalAttrDictWithCustomAttrs(parser, result))
    return failure();
  if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc)
               << "'" << result.name.getStringRef() << "' op ";
      })))
    return failure();

  if (parser.parseColon())
    return failure();

  if (parser.parseTypeList(Types))
    return failure();

  if (parser.parseArrow())
    return failure();

  llvm::SmallVector<Type> resultTypes(1);
  if (parser.parseType(resultTypes[0]))
    return failure();
  result.addTypes(resultTypes);

  if (parser.resolveOperands(Operands, Types, OperandsLoc, result.operands))
    return failure();
  return success();
}

void UpdateOffsetOp::print(OpAsmPrinter &printer) {
  auto mode = getMode();
  auto printDefaults = printDefaultValues();

  printer << ' ';
  printer << getTensorDesc();
  printer << ",";
  printer << ' ';
  printer << getOffsets();

  llvm::SmallVector<llvm::StringRef> elidedAttrs;
  if (!printDefaults && mode == xegpu::ModeKind::SIMT)
    elidedAttrs.push_back("mode");
  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  printer << ' ' << ":";
  printer << ' ';
  printer << getTensorDesc().getType();
  printer << ",";
  printer << ' ';
  printer << getOffsets().getType();
  printer << ' ' << "->";
  printer << ' ';
  printer << getResult().getType();
}

LogicalResult UpdateOffsetOp::verify() {
  auto mode = getMode();
  if (mode != ModeKind::VC)
    return emitOpError("UpdateOffsetOp only work on VC mode.\n");

  auto srcTy = getTensorDesc().getType();
  auto resTy = getResult().getType();
  if (srcTy != resTy)
    return emitOpError("The result should have the same type (shape and "
                       "encoding attribute) as the input TensorDesc.");

  if (!srcTy.getScattered()) {
    return emitOpError("Invalid TensorDesc. UpdateOffsetOp only works on "
                       "TensorDescs with ScatteredAttr.");
  }

  auto offTy = llvm::dyn_cast<VectorType>(getOffsets().getType());
  if (!offTy || offTy.getRank() != 1)
    return emitOpError("The offset should be an 1D vector.\n");

  auto shape = srcTy.getShape();
  if (shape[0] != offTy.getShape()[0])
    return emitOpError(
        "The offset should have same length as the dim-0 of TensorDesc.");

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_DpasOp
//===----------------------------------------------------------------------===//
ParseResult DpasOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SmallVector<OpAsmParser::UnresolvedOperand> Operands;
  llvm::SmallVector<Type> Types;

  llvm::SMLoc OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(Operands))
    return failure();

  auto loc = parser.getCurrentLocation();
  if (parseOptionalAttrDictWithCustomAttrs(parser, result))
    return failure();
  if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc)
               << "'" << result.name.getStringRef() << "' op ";
      })))
    return failure();

  if (parser.parseColon())
    return failure();

  if (parser.parseTypeList(Types))
    return failure();

  if (parser.parseArrow())
    return failure();

  llvm::SmallVector<Type> resultTypes(1);
  if (parser.parseType(resultTypes[0]))
    return failure();
  result.addTypes(resultTypes);

  if (parser.resolveOperands(Operands, Types, OperandsLoc, result.operands))
    return failure();

  return success();
}

void DpasOp::print(OpAsmPrinter &printer) {
  auto mode = getMode();
  auto printDefaults = printDefaultValues();

  printer << ' ';
  printer << getLhs();
  printer << ",";
  printer << ' ';
  printer << getRhs();
  if (Value value = getAcc())
    printer << ", " << value;

  llvm::SmallVector<llvm::StringRef, 2> elidedAttrs;
  if (!printDefaults && mode == xegpu::ModeKind::SIMT)
    elidedAttrs.push_back("mode");

  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  printer << ' ' << ":";
  printer << ' ';
  printer << getLhs().getType();
  printer << ",";
  printer << ' ';
  printer << getRhs().getType();
  if (getAcc()) {
    printer << ",";
    printer << ' ';
    printer << llvm::ArrayRef<Type>(getAcc().getType());
  }
  printer << ' ' << "->";
  printer << ' ';
  printer << getResult().getType();
}

LogicalResult DpasOp::verify() {
  int64_t lhsRank = getLhsType().getRank();
  int64_t rhsRank = getRhsType().getRank();
  Type lhsElemType = getLhsType().getElementType();
  Type rhsElemType = getRhsType().getElementType();

  if (lhsElemType != rhsElemType)
    return emitOpError("lhs and rhs element type does not match for dpas op");

  if (getAcc() && getAccType() != getResultType())
    return emitOpError("Accumulator and Result for dpas op should have the "
                       "same type (both shape and element type).");

  if (lhsRank != rhsRank || lhsRank != 3)
    return emitOpError(
        "lhs and rhs rank does not match for dpas op, or their rank is not 3.");

  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_InvokeSIMDOp
//===----------------------------------------------------------------------===//
void InvokeSIMDOp::build(OpBuilder &builder, OperationState &state,
                         SymbolRefAttr callee, TypeRange results,
                         ArgTypeKindAttr argType, ValueRange operands) {
  state.addOperands(operands);
  state.addAttribute("argType", argType);
  state.addAttribute("callee", callee);
  state.addTypes(results);
}

void InvokeSIMDOp::build(OpBuilder &builder, OperationState &state,
                         StringAttr callee, TypeRange results,
                         ArgTypeKindAttr argType, ValueRange operands) {
  build(builder, state, SymbolRefAttr::get(callee), results, argType, operands);
}

void InvokeSIMDOp::build(OpBuilder &builder, OperationState &state,
                         llvm::StringRef callee, TypeRange results,
                         ArgTypeKindAttr argType, ValueRange operands) {
  build(builder, state, StringAttr::get(builder.getContext(), callee), results,
        argType, operands);
}

//===----------------------------------------------------------------------===//
// XeGPU_AtomicRMWOp
//===----------------------------------------------------------------------===//
void AtomicRMWOp::build(OpBuilder &builder, OperationState &state, Type result,
                        AtomicRMWKindAttr kind, Value tensorDesc, Value mask,
                        Value value) {
  state.addOperands(tensorDesc);
  state.addOperands(mask);
  if (value)
    state.addOperands(value);
  state.getOrAddProperties<Properties>().kind = kind;
  state.getOrAddProperties<Properties>().mode =
      ModeKindAttr::get(builder.getContext(), ModeKind::VC);
  state.addTypes(result);
}

void AtomicRMWOp::build(OpBuilder &builder, OperationState &state, Type result,
                        AtomicRMWKind kind, Value tensorDesc, Value mask,
                        Value value) {
  state.addOperands(tensorDesc);
  state.addOperands(mask);
  if (value)
    state.addOperands(value);
  state.getOrAddProperties<Properties>().kind =
      AtomicRMWKindAttr::get(builder.getContext(), kind);
  state.getOrAddProperties<Properties>().mode =
      ModeKindAttr::get(builder.getContext(), ModeKind::VC);
  state.addTypes(result);
}

ParseResult AtomicRMWOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SmallVector<OpAsmParser::UnresolvedOperand> Operands;
  llvm::SmallVector<Type, 1> Types;
  llvm::SMLoc OperandsLoc;

  llvm::SmallVector<Type> resultTypes(1);

  xegpu::AtomicRMWKindAttr kindAttr;
  if (parser.parseCustomAttributeWithFallback(kindAttr, Type{}))
    return failure();
  if (kindAttr)
    result.getOrAddProperties<AtomicRMWOp::Properties>().kind = kindAttr;

  OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(Operands))
    return failure();

  auto loc = parser.getCurrentLocation();
  if (parseOptionalAttrDictWithCustomAttrs(parser, result))
    return failure();
  if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc)
               << "'" << result.name.getStringRef() << "' op ";
      })))
    return failure();

  if (parser.parseColon())
    return failure();

  if (parser.parseTypeList(Types))
    return failure();

  if (parser.parseArrow())
    return failure();

  if (parser.parseCustomTypeWithFallback(resultTypes[0]))
    return failure();
  result.addTypes(resultTypes);

  if (parser.resolveOperands(Operands, Types, OperandsLoc, result.operands))
    return failure();
  return success();
}

void AtomicRMWOp::print(OpAsmPrinter &printer) {
  auto mode = getMode();
  auto printDefaults = printDefaultValues();

  printer.printStrippedAttrOrType(getKindAttr());
  printer << ' ';
  printer << getTensorDesc();
  printer << ",";
  printer << ' ';
  printer << getMask();
  if (Value value = getValue())
    printer << ", " << value;

  llvm::SmallVector<llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("kind");
  if (!printDefaults && mode == xegpu::ModeKind::SIMT)
    elidedAttrs.push_back("mode");

  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  printer << ' ' << ":";
  printer << ' ';
  printer << getOperation()->getOperandTypes();
  printer << ' ' << "->";
  printer << ' ';
  printer << getResult().getType();
}

LogicalResult AtomicRMWOp::verify() {
  auto mode = getMode();
  if (mode != ModeKind::VC)
    return emitOpError("AtomicRMWOp only work on VC mode.\n");
  return success();
}

//===----------------------------------------------------------------------===//
// XeGPU_CreateNbarrierOp
//===----------------------------------------------------------------------===//
ParseResult CreateNbarrierOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  llvm::SmallVector<OpAsmParser::UnresolvedOperand, 2> Operands;
  llvm::SmallVector<Type> Types;
  llvm::SMLoc OperandsLoc;

  OperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(Operands))
    return failure();

  auto loc = parser.getCurrentLocation();
  if (parseOptionalAttrDictWithCustomAttrs(parser, result))
    return failure();

  if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc)
               << "'" << result.name.getStringRef() << "' op ";
      })))
    return failure();

  if (parser.parseColon())
    return failure();

  if (parser.parseLParen())
    return failure();

  if (parser.parseTypeList(Types))
    return failure();

  if (parser.parseRParen())
    return failure();

  if (parser.parseArrow())
    return failure();

  llvm::SmallVector<Type> resultTypes(1);
  if (parser.parseType(resultTypes[0]))
    return failure();

  result.addTypes(resultTypes);
  if (parser.resolveOperands(Operands, Types, OperandsLoc, result.operands))
    return failure();
  return success();
}

void CreateNbarrierOp::print(OpAsmPrinter &printer) {
  auto mode = getMode();
  auto printDefaults = printDefaultValues();
  llvm::SmallVector<llvm::StringRef, 2> elidedAttrs;
  if (!printDefaults && mode == xegpu::ModeKind::SIMT)
    elidedAttrs.push_back("mode");

  printer << ' ';
  printer << getNbarrierId();
  printer << ",";
  printer << ' ';
  printer << getNbarrierRole();
  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  printer << ' ' << ":";
  printer << ' ' << "(";
  printer << getNbarrierId().getType();
  printer << ",";
  printer << ' ';
  printer << getNbarrierRole().getType();
  printer << ")";
  printer << ' ' << "->";
  printer << ' ';
  printer << getResult().getType();
}

} // namespace xegpu
} // namespace mlir

#include <mlir/Dialect/XeGPU/IR/XeGPUEnums.cpp.inc>
#define GET_OP_CLASSES
#include <mlir/Dialect/XeGPU/IR/XeGPU.cpp.inc>
