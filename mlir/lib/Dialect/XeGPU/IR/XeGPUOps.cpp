//===- XeGPUOps.cpp - MLIR XeGPU ops implementation -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/Dialect/XeGPU/IR/XeGPU.h>
#include <mlir/Interfaces/ViewLikeInterface.h>
#include <mlir/IR/Builders.h>

#define DEBUG_TYPE "xegpu"

namespace mlir {
namespace xegpu {

bool printDefaultValues() {return false;}

static size_t getRankOf(Value value) {
  if (value.getType().isIntOrIndexOrFloat())
    return 0;
  if (auto ty = llvm::dyn_cast_if_present<MemRefType>(value.getType()))
    return ty.getRank();
  if (auto ty = llvm::dyn_cast_if_present<VectorType>(value.getType()))
    return ty.getRank();
  llvm_unreachable("Unsupported value for getRankOf");
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
                           llvm::ArrayRef<int64_t> static_offsets) {
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
  state.addTypes(TensorDesc);
}

void CreateNdDescOp::build(OpBuilder &builder, OperationState &state,
                           Type tdesc, Value source,
                           llvm::ArrayRef<OpFoldResult> offsets) {
  auto ty = llvm::dyn_cast_if_present<MemRefType>(source.getType());
  assert(ty && ty.hasStaticShape() && offsets.size() == getRankOf(source));

  llvm::SmallVector<int64_t> staticOffsets;
  llvm::SmallVector<Value> dynamicOffsets;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);

  build(builder, state, tdesc, source, dynamicOffsets /* dynamic offsets */,
        ValueRange({}) /* empty dynamic shape */,
        ValueRange({}) /* empty dynamic strides */,
        staticOffsets /* static offsets */);
}

void CreateNdDescOp::build(OpBuilder &builder, OperationState &state,
                           Type tdesc, Value source,
                           llvm::ArrayRef<OpFoldResult> offsets,
                           ValueRange shape, ValueRange stride) {
  assert(shape.size() && offsets.size() && stride.size() &&
         shape.size() == stride.size() && shape.size() == offsets.size());

  llvm::SmallVector<int64_t> staticOffsets;
  llvm::SmallVector<Value> dynamicOffsets;

  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);

  build(builder, state, tdesc, source, /* dynamic_offsets = */ dynamicOffsets,
        /* dynamic shape = */ shape , /* dynamic strides = */ stride,
        /* static offsets = */ staticOffsets);
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
  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  printer << ' ' << ":";
  printer << ' ';
  printer << getSourceType();
  printer << ' ' << "->";
  printer << ' ';
  printer << getType();
}

LogicalResult CreateNdDescOp::verify() {
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

// compute consolidated offsets from dynamic_offsets and static_offsets parameters
llvm::SmallVector<OpFoldResult> CreateNdDescOp::getOffsets() {
  llvm::SmallVector<OpFoldResult> offsets;
  auto dynamicOffsets = getDynamicOffsets(); // dynamic_offsets variable
  auto staticOffsets = getStaticOffsets();   // static_offsets attribute

  // in case static_offsets is missing, dynamic_offsets will be used
  if (staticOffsets.size() == 0) {
    offsets.assign(dynamicOffsets.begin(), dynamicOffsets.end());
    return offsets;
  }

  // use static offsets for each dim if it has valid value, 
  // othwise use the value from dynamic_offsets
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

// get the consolidated shape of the 2D memory region. 
// It prefer dynamic_shape than the static shape of 
// memref type.
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
  
  this->emitError("The shape information of the memory is missing.\n");
  return {};
}

// get the consolidated strides of the 2D memory region. 
// It prefer dynamic_stride than the static strides of 
// memref type.
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

  this->emitError("The strides information of the memory is missing.\n");
  return {};
}

} // namespace xegpu
} // namespace mlir

#include <mlir/Dialect/XeGPU/IR/XeGPUEnums.cpp.inc>
#define GET_OP_CLASSES
#include <mlir/Dialect/XeGPU/IR/XeGPU.cpp.inc>
