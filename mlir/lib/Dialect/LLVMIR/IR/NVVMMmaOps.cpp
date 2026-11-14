//===- NVVMMmaOps.cpp - NVVM dialect MMA ops ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ops of op partition `Mma` of the NVVM dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/NVVMMmaOps.h"

#include "IR/NVVMOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cassert>
#include <optional>
#include <string>
#include <utility>

using namespace mlir;
using namespace mlir::NVVM;

//===----------------------------------------------------------------------===//
// Helper/Utility methods
//===----------------------------------------------------------------------===//

template <typename AttrTy>
static ParseResult parseEnumKeyword(OpAsmParser &parser, AttrTy &attr) {
  SMLoc loc = parser.getCurrentLocation();
  std::string keyword;
  if (parser.parseKeywordOrString(&keyword))
    return failure();

  using EnumTy = decltype(attr.getValue());
  std::optional<EnumTy> value = NVVM::symbolizeEnum<EnumTy>(keyword);
  if (!value)
    return parser.emitError(loc) << "unknown enum value '" << keyword << "'";

  attr = AttrTy::get(parser.getContext(), *value);
  return success();
}

//===----------------------------------------------------------------------===//
// Verifier methods
//===----------------------------------------------------------------------===//

// Given the element type of an operand and whether or not it is an accumulator,
// this function returns the PTX type (`NVVM::MMATypes`) that corresponds to the
// operand's element type.
std::optional<mlir::NVVM::MMATypes>
MmaOp::inferOperandMMAType(Type operandElType, bool isAccumulator) {
  auto half2Type =
      VectorType::get(2, Float16Type::get(operandElType.getContext()));
  if (operandElType.isF64())
    return NVVM::MMATypes::f64;
  if (operandElType.isF16() || operandElType == half2Type)
    return NVVM::MMATypes::f16;
  if (operandElType.isF32() && isAccumulator)
    return NVVM::MMATypes::f32;
  if (operandElType.isF32() && !isAccumulator)
    return NVVM::MMATypes::tf32;
  if (llvm::isa<IntegerType>(operandElType)) {
    if (isAccumulator)
      return NVVM::MMATypes::s32;
    return std::nullopt;
  }

  if (auto structType = llvm::dyn_cast<LLVM::LLVMStructType>(operandElType)) {
    if (structType.getBody().empty())
      return std::nullopt;
    return inferOperandMMAType(structType.getBody()[0], isAccumulator);
  }

  return std::nullopt;
}

static bool isInt4PtxType(MMATypes type) {
  return (type == MMATypes::u4 || type == MMATypes::s4);
}

static bool isInt8PtxType(MMATypes type) {
  return (type == MMATypes::u8 || type == MMATypes::s8);
}

static bool isIntegerPtxType(MMATypes type) {
  return isInt4PtxType(type) || isInt8PtxType(type) || type == MMATypes::b1 ||
         type == MMATypes::s32;
}

MMATypes MmaOp::accumPtxType() {
  std::optional<mlir::NVVM::MMATypes> val = inferOperandMMAType(
      getODSOperands(2).getTypes().front(), /*isAccumulator=*/true);
  assert(val.has_value() && "accumulator PTX type should always be inferrable");
  return val.value();
}

MMATypes MmaOp::resultPtxType() {
  std::optional<mlir::NVVM::MMATypes> val =
      inferOperandMMAType(getResult().getType(), /*isAccumulator=*/true);
  assert(val.has_value() && "result PTX type should always be inferrable");
  return val.value();
}

template <typename AttrTy>
static void printMmaProperty(OpAsmPrinter &printer, bool &isFirst,
                             StringRef keyword, AttrTy value) {
  printer << (isFirst ? " " : ", ") << keyword << " = ";
  printer.printStrippedAttrOrType(value);
  isFirst = false;
}

template <typename AttrTy>
static void printMmaEnumProperty(OpAsmPrinter &printer, bool &isFirst,
                                 StringRef keyword, AttrTy value) {
  printer << (isFirst ? " " : ", ") << keyword << " = "
          << NVVM::stringifyEnum(value.getValue());
  isFirst = false;
}

static void printMmaUnitProperty(OpAsmPrinter &printer, bool &isFirst,
                                 StringRef keyword) {
  printer << (isFirst ? " " : ", ") << keyword;
  isFirst = false;
}

template <typename AttrTy>
static ParseResult parseMmaPropertyValue(OpAsmParser &parser,
                                         NamedAttrList &attributes,
                                         StringRef name) {
  if (attributes.get(name))
    return parser.emitError(parser.getCurrentLocation(),
                            "duplicate property '" + name + "'");
  AttrTy value;
  if (parser.parseEqual() || parser.parseCustomAttributeWithFallback(value))
    return failure();
  attributes.append(name, value);
  return success();
}

template <typename AttrTy>
static ParseResult parseMmaEnumPropertyValue(OpAsmParser &parser,
                                             NamedAttrList &attributes,
                                             StringRef name) {
  if (attributes.get(name))
    return parser.emitError(parser.getCurrentLocation(),
                            "duplicate property '" + name + "'");
  AttrTy value;
  if (parser.parseEqual() || parseEnumKeyword(parser, value))
    return failure();
  attributes.append(name, value);
  return success();
}

static bool isMmaPropertyName(StringRef name) {
  return llvm::is_contained(
      ArrayRef<StringRef>{
          "shape", "b1Op", "intOverflowBehavior", "layoutA", "layoutB",
          "multiplicandAPtxType", "multiplicandBPtxType", "orderedMetadata",
          "kind", "scaleVecSize", "blockScaleFormat", "operandSegmentSizes"},
      name);
}

static ParseResult parseMmaProperties(OpAsmParser &parser,
                                      NamedAttrList &attributes,
                                      ArrayRef<StringRef> allowedKeywords,
                                      ArrayRef<StringRef> requiredProperties) {
  while (true) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword))
      return failure();
    if (!llvm::is_contained(allowedKeywords, keyword))
      return parser.emitError(parser.getCurrentLocation(),
                              "unknown MMA property '" + keyword + "'");

    ParseResult parseResult = success();
    if (keyword == "shape")
      parseResult =
          parseMmaPropertyValue<MMAShapeAttr>(parser, attributes, "shape");
    else if (keyword == "b1_op")
      parseResult =
          parseMmaEnumPropertyValue<MMAB1OpAttr>(parser, attributes, "b1Op");
    else if (keyword == "int_overflow")
      parseResult = parseMmaEnumPropertyValue<MMAIntOverflowAttr>(
          parser, attributes, "intOverflowBehavior");
    else if (keyword == "layout_a")
      parseResult = parseMmaEnumPropertyValue<MMALayoutAttr>(parser, attributes,
                                                             "layoutA");
    else if (keyword == "layout_b")
      parseResult = parseMmaEnumPropertyValue<MMALayoutAttr>(parser, attributes,
                                                             "layoutB");
    else if (keyword == "multiplicand_a_ptx_type")
      parseResult = parseMmaEnumPropertyValue<MMATypesAttr>(
          parser, attributes, "multiplicandAPtxType");
    else if (keyword == "multiplicand_b_ptx_type")
      parseResult = parseMmaEnumPropertyValue<MMATypesAttr>(
          parser, attributes, "multiplicandBPtxType");
    else if (keyword == "kind") {
      if (llvm::is_contained(allowedKeywords, "block_scale_format"))
        parseResult = parseMmaEnumPropertyValue<MMABlockScaleKindAttr>(
            parser, attributes, "kind");
      else
        parseResult =
            parseMmaEnumPropertyValue<MMAKindAttr>(parser, attributes, "kind");
    } else if (keyword == "scale_vec_size")
      parseResult = parseMmaEnumPropertyValue<ScaleVecSizeAttr>(
          parser, attributes, "scaleVecSize");
    else if (keyword == "block_scale_format")
      parseResult = parseMmaEnumPropertyValue<BlockScaleFormatAttr>(
          parser, attributes, "blockScaleFormat");
    else if (keyword == "ordered_metadata") {
      if (attributes.get("orderedMetadata"))
        return parser.emitError(parser.getCurrentLocation(),
                                "duplicate property 'orderedMetadata'");
      attributes.append("orderedMetadata", parser.getBuilder().getUnitAttr());
    } else {
      return parser.emitError(parser.getCurrentLocation(),
                              "unknown MMA property '" + keyword + "'");
    }
    if (failed(parseResult))
      return failure();
    if (failed(parser.parseOptionalComma()))
      break;
  }

  for (StringRef property : requiredProperties) {
    if (!attributes.get(property))
      return parser.emitError(parser.getCurrentLocation(),
                              "missing required property '" + property + "'");
  }

  NamedAttrList discardableAttributes;
  if (parser.parseOptionalAttrDict(discardableAttributes))
    return failure();
  for (NamedAttribute attribute : discardableAttributes) {
    if (isMmaPropertyName(attribute.getName()))
      return parser.emitError(
          parser.getCurrentLocation(),
          "inherent property '" + attribute.getName().getValue() +
              "' must be spelled directly in the operation syntax");
    attributes.append(attribute);
  }
  return success();
}

void MmaOp::print(OpAsmPrinter &p) {
  SmallVector<Type, 4> regTypes;
  struct MMAOperandFragment {
    StringRef operandName;
    StringRef ptxTypeAttr;
    SmallVector<Value, 4> regs;
    explicit MMAOperandFragment(StringRef name, StringRef ptxTypeName)
        : operandName(name), ptxTypeAttr(ptxTypeName) {}
  };

  std::array<MMAOperandFragment, 3> frags{
      MMAOperandFragment("A", getMultiplicandAPtxTypeAttrName()),
      MMAOperandFragment("B", getMultiplicandBPtxTypeAttrName()),
      MMAOperandFragment("C", "")};
  SmallVector<StringRef, 4> ignoreAttrNames{
      mlir::NVVM::MmaOp::getOperandSegmentSizeAttr()};

  for (unsigned fragIdx = 0; fragIdx < frags.size(); fragIdx++) {
    auto &frag = frags[fragIdx];
    auto varOperandSpec = getODSOperandIndexAndLength(fragIdx);
    for (auto operandIdx = varOperandSpec.first;
         operandIdx < varOperandSpec.first + varOperandSpec.second;
         operandIdx++) {
      frag.regs.push_back(this->getOperand(operandIdx));
      if (operandIdx == 0) {
        regTypes.push_back(this->getOperand(operandIdx).getType());
      }
    }
    std::optional<MMATypes> inferredType = MmaOp::inferOperandMMAType(
        regTypes.back(), /*isAccumulator=*/fragIdx >= 2);
    if (inferredType)
      ignoreAttrNames.push_back(frag.ptxTypeAttr);
  }

  auto printMmaOperand = [&](const MMAOperandFragment &frag) -> void {
    p << " " << frag.operandName;
    p << "[";
    p.printOperands(frag.regs);
    p << "] ";
  };

  for (const auto &frag : frags) {
    printMmaOperand(frag);
  }

  bool isFirstProperty = true;
  printMmaProperty(p, isFirstProperty, "shape", getShapeAttr());
  if (getB1OpAttr())
    printMmaEnumProperty(p, isFirstProperty, "b1_op", getB1OpAttr());
  if (getIntOverflowBehaviorAttr())
    printMmaEnumProperty(p, isFirstProperty, "int_overflow",
                         getIntOverflowBehaviorAttr());
  printMmaEnumProperty(p, isFirstProperty, "layout_a", getLayoutAAttr());
  printMmaEnumProperty(p, isFirstProperty, "layout_b", getLayoutBAttr());
  if (getMultiplicandAPtxTypeAttr() &&
      !llvm::is_contained(ignoreAttrNames, getMultiplicandAPtxTypeAttrName()))
    printMmaEnumProperty(p, isFirstProperty, "multiplicand_a_ptx_type",
                         getMultiplicandAPtxTypeAttr());
  if (getMultiplicandBPtxTypeAttr() &&
      !llvm::is_contained(ignoreAttrNames, getMultiplicandBPtxTypeAttrName()))
    printMmaEnumProperty(p, isFirstProperty, "multiplicand_b_ptx_type",
                         getMultiplicandBPtxTypeAttr());
  llvm::append_range(ignoreAttrNames,
                     ArrayRef<StringRef>{getShapeAttrName(), getB1OpAttrName(),
                                         getIntOverflowBehaviorAttrName(),
                                         getLayoutAAttrName(),
                                         getLayoutBAttrName(),
                                         getMultiplicandAPtxTypeAttrName(),
                                         getMultiplicandBPtxTypeAttrName()});
  p.printOptionalAttrDict((*this)->getDiscardableAttrDictionary().getValue(),
                          ignoreAttrNames);

  // Print the types of the operands and result.
  p << " : " << "(";
  llvm::interleaveComma(SmallVector<Type, 3>{frags[0].regs[0].getType(),
                                             frags[1].regs[0].getType(),
                                             frags[2].regs[0].getType()},
                        p);
  p << ")";
  p.printArrowTypeList(TypeRange{this->getRes().getType()});
}

void MmaOp::build(OpBuilder &builder, OperationState &result, Type resultType,
                  ValueRange operandA, ValueRange operandB, ValueRange operandC,
                  ArrayRef<int64_t> shape, std::optional<MMAB1Op> b1Op,
                  std::optional<MMAIntOverflow> intOverflow,
                  std::optional<std::array<MMATypes, 2>> multiplicandPtxTypes,
                  std::optional<std::array<MMALayout, 2>> multiplicandLayouts) {

  assert(shape.size() == 3 && "expected shape to have size 3 (m, n, k)");
  MLIRContext *ctx = builder.getContext();
  result.addAttribute(
      "shape", builder.getAttr<MMAShapeAttr>(shape[0], shape[1], shape[2]));

  result.addOperands(operandA);
  result.addOperands(operandB);
  result.addOperands(operandC);

  if (multiplicandPtxTypes) {
    result.addAttribute("multiplicandAPtxType",
                        MMATypesAttr::get(ctx, (*multiplicandPtxTypes)[0]));
    result.addAttribute("multiplicandBPtxType",
                        MMATypesAttr::get(ctx, (*multiplicandPtxTypes)[1]));
  } else {
    if (auto res = inferOperandMMAType(operandA[0].getType(), false))
      result.addAttribute("multiplicandAPtxType", MMATypesAttr::get(ctx, *res));
    if (auto res = inferOperandMMAType(operandB[0].getType(), false))
      result.addAttribute("multiplicandBPtxType", MMATypesAttr::get(ctx, *res));
  }

  if (multiplicandLayouts) {
    result.addAttribute("layoutA",
                        MMALayoutAttr::get(ctx, (*multiplicandLayouts)[0]));
    result.addAttribute("layoutB",
                        MMALayoutAttr::get(ctx, (*multiplicandLayouts)[1]));
  } else {
    result.addAttribute("layoutA", MMALayoutAttr::get(ctx, MMALayout::row));
    result.addAttribute("layoutB", MMALayoutAttr::get(ctx, MMALayout::col));
  }

  if (intOverflow.has_value())
    result.addAttribute("intOverflowBehavior",
                        MMAIntOverflowAttr::get(ctx, *intOverflow));
  if (b1Op.has_value())
    result.addAttribute("b1Op", MMAB1OpAttr::get(ctx, *b1Op));

  result.addTypes(resultType);
  result.addAttribute(
      MmaOp::getOperandSegmentSizeAttr(),
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(operandA.size()),
                                    static_cast<int32_t>(operandB.size()),
                                    static_cast<int32_t>(operandC.size())}));
}

// <operation> :=
//   A `[` $operandA `]` B `[` $operandB `]` C `[` $operandC `]`
//   properties attr-dict
//   : (type($operandA[0]), type($operandB[0]), type($operandC[0]))
//     `->` type($res)
ParseResult MmaOp::parse(OpAsmParser &parser, OperationState &result) {
  struct MMAOperandFragment {
    std::optional<MMATypes> elemtype;
    SmallVector<OpAsmParser::UnresolvedOperand, 4> regs;
    SmallVector<Type> regTypes;
  };

  Builder &builder = parser.getBuilder();
  std::array<MMAOperandFragment, 4> frags;

  NamedAttrList namedAttributes;

  // A helper to parse the operand segments.
  auto parseMmaOperand = [&](StringRef operandName,
                             MMAOperandFragment &frag) -> LogicalResult {
    if (parser.parseKeyword(operandName).failed())
      return failure();
    if (parser
            .parseOperandList(frag.regs, OpAsmParser::Delimiter::OptionalSquare)
            .failed())
      return failure();
    return success();
  };

  // Parse the operand segments.
  if (parseMmaOperand("A", frags[0]).failed())
    return failure();
  if (parseMmaOperand("B", frags[1]).failed())
    return failure();
  if (parseMmaOperand("C", frags[2]).failed())
    return failure();

  if (parseMmaProperties(parser, namedAttributes,
                         {"shape", "b1_op", "int_overflow", "layout_a",
                          "layout_b", "multiplicand_a_ptx_type",
                          "multiplicand_b_ptx_type"},
                         {"shape", "layoutA", "layoutB"}))
    return failure();

  // Parse the type specification and resolve operands.
  SmallVector<Type, 3> operandTypes;
  if (failed(parser.parseColon()))
    return failure();
  if (failed(parser.parseLParen()))
    return failure();
  if (failed(parser.parseTypeList(operandTypes)))
    return failure();
  if (failed(parser.parseRParen()))
    if (operandTypes.size() != 3)
      return parser.emitError(
          parser.getNameLoc(),
          "expected one type for each operand segment but got " +
              Twine(operandTypes.size()) + " types");
  for (const auto &iter : llvm::enumerate(operandTypes)) {
    auto &frag = frags[iter.index()];
    frag.regTypes.resize(frag.regs.size(), iter.value());
    if (failed(parser.resolveOperands(frag.regs, frag.regTypes,
                                      parser.getNameLoc(), result.operands)))
      return failure();
    frag.elemtype = inferOperandMMAType(frag.regTypes[0],
                                        /*isAccumulator*/ iter.index() < 2);
  }

  Type resultType;
  if (parser.parseArrow() || parser.parseType(resultType))
    return failure();
  frags[3].elemtype = inferOperandMMAType(resultType, /*isAccumulator*/ true);

  std::array<StringRef, 2> names{"multiplicandAPtxType",
                                 "multiplicandBPtxType"};
  for (unsigned idx = 0; idx < names.size(); idx++) {
    const auto &frag = frags[idx];
    std::optional<NamedAttribute> attr = namedAttributes.getNamed(names[idx]);
    if (!frag.elemtype.has_value() && !attr.has_value()) {
      return parser.emitError(
          parser.getNameLoc(),
          "attribute " + names[idx] +
              " is not provided explicitly and cannot be inferred");
    }
    if (!attr.has_value())
      result.addAttribute(
          names[idx], MMATypesAttr::get(parser.getContext(), *frag.elemtype));
  }

  result.addTypes(resultType);
  if (!namedAttributes.empty())
    result.addAttributes(namedAttributes);
  result.addAttribute(MmaOp::getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr({
                          static_cast<int32_t>(frags[0].regs.size()),
                          static_cast<int32_t>(frags[1].regs.size()),
                          static_cast<int32_t>(frags[2].regs.size()),
                      }));
  return success();
}

LogicalResult MmaOp::verify() {
  MLIRContext *context = getContext();
  auto f16Ty = Float16Type::get(context);
  auto i32Ty = IntegerType::get(context, 32);
  auto f16x2Ty = VectorType::get(2, f16Ty);
  auto f32Ty = Float32Type::get(context);
  auto f16x2x4StructTy = LLVM::LLVMStructType::getLiteral(
      context, {f16x2Ty, f16x2Ty, f16x2Ty, f16x2Ty});

  auto s32x4StructTy =
      LLVM::LLVMStructType::getLiteral(context, {i32Ty, i32Ty, i32Ty, i32Ty});
  auto f32x8StructTy =
      LLVM::LLVMStructType::getLiteral(context, SmallVector<Type>(8, f32Ty));
  auto f16x2x2StructTy =
      LLVM::LLVMStructType::getLiteral(context, {f16x2Ty, f16x2Ty});
  auto f32x4StructTy =
      LLVM::LLVMStructType::getLiteral(context, {f32Ty, f32Ty, f32Ty, f32Ty});
  auto s32x2StructTy =
      LLVM::LLVMStructType::getLiteral(context, {i32Ty, i32Ty});

  std::array<int64_t, 3> mmaShape{getShapeAttr().getM(), getShapeAttr().getN(),
                                  getShapeAttr().getK()};

  // These variables define the set of allowed data types for matrices A, B, C,
  // and result.
  using AllowedShapes = SmallVector<std::array<int64_t, 3>, 2>;
  using AllowedTypes = SmallVector<SmallVector<Type, 4>, 2>;
  AllowedShapes allowedShapes;
  AllowedTypes expectedA;
  AllowedTypes expectedB;
  AllowedTypes expectedC;
  SmallVector<Type> expectedResult;

  // When M = 16, we just need to calculate the number of 8xk tiles, where
  // k is a factor that depends on the data type.
  if (mmaShape[0] == 16) {
    int64_t kFactor;
    Type multiplicandFragType;
    switch (*getMultiplicandAPtxType()) {
    case MMATypes::tf32:
      kFactor = 4;
      multiplicandFragType = i32Ty;
      expectedResult.push_back(LLVM::LLVMStructType::getLiteral(
          context, {f32Ty, f32Ty, f32Ty, f32Ty}));
      break;
    case MMATypes::bf16:
      kFactor = 8;
      multiplicandFragType = i32Ty;
      expectedResult.push_back(LLVM::LLVMStructType::getLiteral(
          context, {f32Ty, f32Ty, f32Ty, f32Ty}));
      break;
    case MMATypes::f16:
      kFactor = 8;
      multiplicandFragType = f16x2Ty;
      expectedResult.push_back(f16x2x2StructTy);
      expectedResult.push_back(f32x4StructTy);
      break;
    case MMATypes::e4m3:
    case MMATypes::e5m2:
      // FP8 (m16n8k16 / m16n8k32) packs 4 values per 32-bit register, same
      // as s8/u8, but the accumulator is f16 or f32 (not integer).
      kFactor = 16;
      multiplicandFragType = i32Ty;
      expectedResult.push_back(f16x2x2StructTy);
      expectedResult.push_back(f32x4StructTy);
      break;
    case MMATypes::s4:
    case MMATypes::u4:
      kFactor = 32;
      break;
    case MMATypes::b1:
      kFactor = 128;
      break;
    case MMATypes::s8:
    case MMATypes::u8:
      kFactor = 16;
      break;
    default:
      return emitError("invalid shape or multiplicand type: ")
             << getMultiplicandAPtxType().value();
    }

    if (isIntegerPtxType(getMultiplicandAPtxType().value())) {
      expectedResult.push_back(s32x4StructTy);
      expectedC.emplace_back(4, i32Ty);
      multiplicandFragType = i32Ty;
    } else {
      expectedC.emplace_back(2, f16x2Ty);
      expectedC.emplace_back(4, f32Ty);
    }

    int64_t unitA = (mmaShape[0] / 8) * (mmaShape[2] / kFactor);
    int64_t unitB = (mmaShape[1] / 8) * (mmaShape[2] / kFactor);
    expectedA.emplace_back(unitA, multiplicandFragType);
    expectedB.emplace_back(unitB, multiplicandFragType);
    allowedShapes.push_back({16, 8, kFactor});
    allowedShapes.push_back({16, 8, kFactor * 2});

    if (resultPtxType() != accumPtxType())
      return emitOpError("ctype does not match dtype");
  }

  // In the M=8 case, there is only 1 possible case per data type.
  if (mmaShape[0] == 8) {
    if (*getMultiplicandAPtxType() == MMATypes::f16) {
      expectedA.emplace_back(2, f16x2Ty);
      expectedB.emplace_back(2, f16x2Ty);
      expectedResult.push_back(f16x2x4StructTy);
      expectedResult.push_back(f32x8StructTy);
      expectedC.emplace_back(4, f16x2Ty);
      expectedC.emplace_back(8, f32Ty);
      allowedShapes.push_back({8, 8, 4});
    }
    if (*getMultiplicandAPtxType() == MMATypes::f64) {
      Type f64Ty = Float64Type::get(context);
      expectedA.emplace_back(1, f64Ty);
      expectedB.emplace_back(1, f64Ty);
      expectedC.emplace_back(2, f64Ty);
      expectedResult.emplace_back(LLVM::LLVMStructType::getLiteral(
          context, SmallVector<Type>(2, f64Ty)));
      allowedShapes.push_back({8, 8, 4});
    }
    if (isIntegerPtxType(getMultiplicandAPtxType().value())) {
      expectedA.push_back({i32Ty});
      expectedB.push_back({i32Ty});
      expectedC.push_back({i32Ty, i32Ty});
      expectedResult.push_back(s32x2StructTy);
      if (isInt4PtxType(getMultiplicandAPtxType().value()))
        allowedShapes.push_back({8, 8, 32});
      if (isInt8PtxType(getMultiplicandAPtxType().value()))
        allowedShapes.push_back({8, 8, 16});
      if (getMultiplicandAPtxType().value() == MMATypes::b1)
        allowedShapes.push_back({8, 8, 128});
    }
  }

  std::string errorMessage;
  llvm::raw_string_ostream errorStream(errorMessage);

  // Check that we matched an existing shape/dtype combination.
  if (expectedA.empty() || expectedB.empty() || expectedC.empty() ||
      !llvm::is_contained(allowedShapes, mmaShape)) {
    errorStream << "unimplemented variant for MMA shape <";
    llvm::interleaveComma(mmaShape, errorStream);
    errorStream << ">";
    return emitOpError(errorMessage);
  }

  // Verify the operand types for segments of A, B, and C operands.
  std::array<StringRef, 3> operandNames{"A", "B", "C"};
  for (const auto &iter : llvm::enumerate(
           std::array<AllowedTypes, 3>{std::move(expectedA),
                                       std::move(expectedB),
                                       std::move(expectedC)})) {
    auto spec = this->getODSOperandIndexAndLength(iter.index());
    SmallVector<Type, 4> operandTySeg(operand_type_begin() + spec.first,
                                      operand_type_begin() + spec.first +
                                          spec.second);
    bool match = llvm::is_contained(iter.value(), operandTySeg);

    if (!match) {
      errorStream << "Could not match types for the "
                  << operandNames[iter.index()]
                  << " operands; expected one of ";
      for (const auto &x : iter.value()) {
        errorStream << x.size() << "x" << x[0] << " ";
      }
      errorStream << "but got ";
      llvm::interleaveComma(operandTySeg, errorStream);
      return emitOpError(errorMessage);
    }
  }

  // Check the result type
  if (!llvm::any_of(expectedResult, [&](Type expectedResultType) {
        return expectedResultType == getResult().getType();
      })) {
    errorStream
        << "Could not match allowed types for the result; expected one of ";
    llvm::interleaveComma(expectedResult, errorStream);
    errorStream << " but got " << getResult().getType();
    return emitOpError(errorMessage);
  }

  // Ensure that binary MMA variants have a b1 MMA operation defined.
  if (getMultiplicandAPtxType() == MMATypes::b1 && !getB1Op()) {
    return emitOpError("op requires " + getB1OpAttrName().strref() +
                       " attribute");
  }

  // Ensure int4/int8 MMA variants specify the accum overflow behavior
  // attribute.
  if (isInt4PtxType(*getMultiplicandAPtxType()) ||
      isInt8PtxType(*getMultiplicandAPtxType())) {
    if (!getIntOverflowBehavior())
      return emitOpError("op requires " +
                         getIntOverflowBehaviorAttrName().strref() +
                         " attribute");
  }

  // Validate layout combinations. According to the operation description, most
  // MMA operations require layoutA=row and layoutB=col. Only m8n8k4 with f16
  // can use other layout combinations.
  bool isM8N8K4_F16 =
      (mmaShape[0] == 8 && mmaShape[1] == 8 && mmaShape[2] == 4 &&
       getMultiplicandAPtxType() == MMATypes::f16);

  if (!isM8N8K4_F16) {
    // For all other shapes/types, layoutA must be row and layoutB must be col
    if (getLayoutA() != MMALayout::row || getLayoutB() != MMALayout::col) {
      return emitOpError("requires layoutA = #nvvm.mma_layout<row> and "
                         "layoutB = #nvvm.mma_layout<col> for shape <")
             << mmaShape[0] << ", " << mmaShape[1] << ", " << mmaShape[2]
             << "> with element types " << *getMultiplicandAPtxType() << " and "
             << *getMultiplicandBPtxType()
             << ". Only m8n8k4 with f16 supports other layouts.";
    }
  }

  return success();
}

MMATypes MmaSpOp::accumPtxType() {
  std::optional<mlir::NVVM::MMATypes> val = MmaOp::inferOperandMMAType(
      getODSOperands(2).getTypes().front(), /*isAccumulator=*/true);
  assert(val.has_value() && "accumulator PTX type should always be inferrable");
  return val.value();
}

MMATypes MmaSpOp::resultPtxType() {
  std::optional<mlir::NVVM::MMATypes> val =
      MmaOp::inferOperandMMAType(getResult().getType(), /*isAccumulator=*/true);
  assert(val.has_value() && "result PTX type should always be inferrable");
  return val.value();
}

mlir::NVVM::IDArgPair
MmaSpOp::getIntrinsicIDAndArgs(Operation &op, LLVM::ModuleTranslation &mt,
                               llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::MmaSpOp>(op);

  // Get operands
  llvm::SmallVector<llvm::Value *> args;
  for (mlir::Value v : thisOp.getOperands())
    args.push_back(mt.lookupValue(v));

  // Get intrinsic ID using the existing getIntrinsicID method
  auto intId = MmaSpOp::getIntrinsicID(
      thisOp.getShape().getM(), thisOp.getShape().getN(),
      thisOp.getShape().getK(), thisOp.getIntOverflowBehavior(),
      thisOp.getOrderedMetadata(), thisOp.getKind(),
      *thisOp.getMultiplicandAPtxType(), *thisOp.getMultiplicandBPtxType(),
      thisOp.accumPtxType(), thisOp.resultPtxType());

  return {intId, args};
}

void MmaSpOp::print(OpAsmPrinter &p) {
  SmallVector<Type, 4> regTypes;
  struct MMAOperandFragment {
    StringRef operandName;
    StringRef ptxTypeAttr;
    SmallVector<Value, 4> regs;
    explicit MMAOperandFragment(StringRef name, StringRef ptxTypeName)
        : operandName(name), ptxTypeAttr(ptxTypeName) {}
  };

  std::array<MMAOperandFragment, 5> frags{
      MMAOperandFragment("A", getMultiplicandAPtxTypeAttrName()),
      MMAOperandFragment("B", getMultiplicandBPtxTypeAttrName()),
      MMAOperandFragment("C", ""), MMAOperandFragment("sparseMetadata", ""),
      MMAOperandFragment("selector", "")};
  SmallVector<StringRef, 4> ignoreAttrNames{
      mlir::NVVM::MmaSpOp::getOperandSegmentSizeAttr()};

  // Handle variadic operands A, B, C
  for (unsigned fragIdx = 0; fragIdx < 3; fragIdx++) {
    auto &frag = frags[fragIdx];
    auto varOperandSpec = getODSOperandIndexAndLength(fragIdx);
    for (auto operandIdx = varOperandSpec.first;
         operandIdx < varOperandSpec.first + varOperandSpec.second;
         operandIdx++) {
      frag.regs.push_back(this->getOperand(operandIdx));
      if (operandIdx == varOperandSpec.first) {
        regTypes.push_back(this->getOperand(operandIdx).getType());
      }
    }
    std::optional<MMATypes> inferredType = MmaOp::inferOperandMMAType(
        regTypes.back(), /*isAccumulator=*/fragIdx >= 2);
    if (inferredType)
      ignoreAttrNames.push_back(frag.ptxTypeAttr);
  }

  // Handle sparse metadata and selector (single operands)
  frags[3].regs.push_back(getSparseMetadata());
  frags[4].regs.push_back(getSparsitySelector());

  auto printMmaSpOperand = [&](const MMAOperandFragment &frag) -> void {
    p << " " << frag.operandName;
    p << "[";
    p.printOperands(frag.regs);
    p << "]";
  };

  for (const auto &frag : frags)
    printMmaSpOperand(frag);

  bool isFirstProperty = true;
  printMmaProperty(p, isFirstProperty, "shape", getShapeAttr());
  if (getIntOverflowBehaviorAttr())
    printMmaEnumProperty(p, isFirstProperty, "int_overflow",
                         getIntOverflowBehaviorAttr());
  if (getMultiplicandAPtxTypeAttr() &&
      !llvm::is_contained(ignoreAttrNames, getMultiplicandAPtxTypeAttrName()))
    printMmaEnumProperty(p, isFirstProperty, "multiplicand_a_ptx_type",
                         getMultiplicandAPtxTypeAttr());
  if (getMultiplicandBPtxTypeAttr() &&
      !llvm::is_contained(ignoreAttrNames, getMultiplicandBPtxTypeAttrName()))
    printMmaEnumProperty(p, isFirstProperty, "multiplicand_b_ptx_type",
                         getMultiplicandBPtxTypeAttr());
  if (getOrderedMetadata())
    printMmaUnitProperty(p, isFirstProperty, "ordered_metadata");
  if (getKindAttr())
    printMmaEnumProperty(p, isFirstProperty, "kind", getKindAttr());
  llvm::append_range(
      ignoreAttrNames,
      ArrayRef<StringRef>{getShapeAttrName(), getIntOverflowBehaviorAttrName(),
                          getMultiplicandAPtxTypeAttrName(),
                          getMultiplicandBPtxTypeAttrName(),
                          getOrderedMetadataAttrName(), getKindAttrName()});
  p.printOptionalAttrDict((*this)->getDiscardableAttrDictionary().getValue(),
                          ignoreAttrNames);
  p << " : ";
  p << "(";
  for (int i = 0; i < 3; ++i) {
    p << regTypes[i];
    if (i < 2)
      p << ", ";
  }
  p << ") -> " << getResult().getType();
}

void MmaSpOp::build(
    OpBuilder &builder, OperationState &result, Type resultType,
    ValueRange operandA, ValueRange operandB, ValueRange operandC,
    Value sparseMetadata, Value sparsitySelector, ArrayRef<int64_t> shape,
    std::optional<MMAIntOverflow> intOverflow,
    std::optional<std::array<MMATypes, 2>> multiplicandPtxTypes) {

  assert(shape.size() == 3 && "expected shape to have size 3 (m, n, k)");
  MLIRContext *ctx = builder.getContext();
  result.addAttribute(
      "shape", builder.getAttr<MMAShapeAttr>(shape[0], shape[1], shape[2]));

  result.addOperands(operandA);
  result.addOperands(operandB);
  result.addOperands(operandC);
  result.addOperands(sparseMetadata);
  result.addOperands(sparsitySelector);

  if (multiplicandPtxTypes) {
    result.addAttribute("multiplicandAPtxType",
                        MMATypesAttr::get(ctx, (*multiplicandPtxTypes)[0]));
    result.addAttribute("multiplicandBPtxType",
                        MMATypesAttr::get(ctx, (*multiplicandPtxTypes)[1]));
  } else {
    if (auto res = MmaOp::inferOperandMMAType(operandA[0].getType(), false))
      result.addAttribute("multiplicandAPtxType", MMATypesAttr::get(ctx, *res));
    if (auto res = MmaOp::inferOperandMMAType(operandB[0].getType(), false))
      result.addAttribute("multiplicandBPtxType", MMATypesAttr::get(ctx, *res));
  }

  if (intOverflow.has_value())
    result.addAttribute("intOverflowBehavior",
                        MMAIntOverflowAttr::get(ctx, *intOverflow));

  result.addTypes(resultType);
  result.addAttribute(
      MmaSpOp::getOperandSegmentSizeAttr(),
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(operandA.size()),
                                    static_cast<int32_t>(operandB.size()),
                                    static_cast<int32_t>(operandC.size()), 1,
                                    1})); // sparseMetadata and sparsitySelector
}

ParseResult MmaSpOp::parse(OpAsmParser &parser, OperationState &result) {
  struct MMAOperandFragment {
    std::optional<MMATypes> elemtype;
    SmallVector<OpAsmParser::UnresolvedOperand, 4> regs;
    SmallVector<Type> regTypes;
  };

  Builder &builder = parser.getBuilder();
  std::array<MMAOperandFragment, 6> frags; // A, B, C, sparseMetadata, selector

  NamedAttrList namedAttributes;

  // A helper to parse the operand segments.
  auto parseMmaSpOperand = [&](StringRef operandName,
                               MMAOperandFragment &frag) -> LogicalResult {
    if (parser.parseKeyword(operandName).failed())
      return failure();
    if (parser
            .parseOperandList(frag.regs, OpAsmParser::Delimiter::OptionalSquare)
            .failed())
      return failure();
    return success();
  };

  // Parse the operand segments.
  if (parseMmaSpOperand("A", frags[0]).failed())
    return failure();
  if (parseMmaSpOperand("B", frags[1]).failed())
    return failure();
  if (parseMmaSpOperand("C", frags[2]).failed())
    return failure();
  if (parseMmaSpOperand("sparseMetadata", frags[3]).failed())
    return failure();
  if (parseMmaSpOperand("selector", frags[4]).failed())
    return failure();

  if (parseMmaProperties(parser, namedAttributes,
                         {"shape", "int_overflow", "multiplicand_a_ptx_type",
                          "multiplicand_b_ptx_type", "ordered_metadata",
                          "kind"},
                         {"shape"}))
    return failure();

  // Parse the type specification and resolve operands.
  SmallVector<Type, 3> operandTypes;
  if (failed(parser.parseColon()))
    return failure();
  if (failed(parser.parseLParen()))
    return failure();
  if (failed(parser.parseTypeList(operandTypes)))
    return failure();
  if (failed(parser.parseRParen()))
    return failure();
  if (operandTypes.size() != 3)
    return parser.emitError(
        parser.getNameLoc(),
        "expected one type for each operand segment but got " +
            Twine(operandTypes.size()) + " types");
  for (const auto &iter : llvm::enumerate(operandTypes)) {
    auto &frag = frags[iter.index()];
    frag.regTypes.resize(frag.regs.size(), iter.value());
    if (failed(parser.resolveOperands(frag.regs, frag.regTypes,
                                      parser.getNameLoc(), result.operands)))
      return failure();
    frag.elemtype =
        MmaOp::inferOperandMMAType(frag.regTypes[0],
                                   /*isAccumulator*/ iter.index() >= 2);
  }

  Type resultType;
  if (parser.parseArrow() || parser.parseType(resultType))
    return failure();
  frags[5].elemtype =
      MmaOp::inferOperandMMAType(resultType, /*isAccumulator*/ true);

  // Resolve sparse metadata and selector (assume i32 type)
  Type i32Type = builder.getIntegerType(32);
  if (parser
          .resolveOperands(frags[3].regs, i32Type, parser.getCurrentLocation(),
                           result.operands)
          .failed())
    return failure();
  if (parser
          .resolveOperands(frags[4].regs, i32Type, parser.getCurrentLocation(),
                           result.operands)
          .failed())
    return failure();

  std::array<StringRef, 2> names{"multiplicandAPtxType",
                                 "multiplicandBPtxType"};
  for (unsigned idx = 0; idx < names.size(); idx++) {
    const auto &frag = frags[idx];
    std::optional<NamedAttribute> attr = namedAttributes.getNamed(names[idx]);
    if (!frag.elemtype.has_value() && !attr.has_value()) {
      return parser.emitError(
          parser.getNameLoc(),
          "attribute " + names[idx] +
              " is not provided explicitly and cannot be inferred");
    }
    if (!attr.has_value())
      result.addAttribute(
          names[idx], MMATypesAttr::get(parser.getContext(), *frag.elemtype));
  }

  result.addTypes(resultType);
  if (!namedAttributes.empty())
    result.addAttributes(namedAttributes);
  result.addAttribute(MmaSpOp::getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr({
                          static_cast<int32_t>(frags[0].regs.size()),
                          static_cast<int32_t>(frags[1].regs.size()),
                          static_cast<int32_t>(frags[2].regs.size()),
                          1, // sparseMetadata
                          1  // sparsitySelector
                      }));
  return success();
}

LogicalResult MmaSpOp::verify() {
  MLIRContext *context = getContext();
  auto f16Ty = Float16Type::get(context);
  auto i32Ty = IntegerType::get(context, 32);
  auto f16x2Ty = VectorType::get(2, f16Ty);
  auto f32Ty = Float32Type::get(context);
  auto f16x2x4StructTy = LLVM::LLVMStructType::getLiteral(
      context, {f16x2Ty, f16x2Ty, f16x2Ty, f16x2Ty});

  auto s32x4StructTy =
      LLVM::LLVMStructType::getLiteral(context, {i32Ty, i32Ty, i32Ty, i32Ty});
  auto f32x8StructTy =
      LLVM::LLVMStructType::getLiteral(context, SmallVector<Type>(8, f32Ty));
  auto f16x2x2StructTy =
      LLVM::LLVMStructType::getLiteral(context, {f16x2Ty, f16x2Ty});
  auto f32x4StructTy =
      LLVM::LLVMStructType::getLiteral(context, {f32Ty, f32Ty, f32Ty, f32Ty});
  auto s32x2StructTy =
      LLVM::LLVMStructType::getLiteral(context, {i32Ty, i32Ty});

  std::array<int64_t, 3> mmaShape{getShapeAttr().getM(), getShapeAttr().getN(),
                                  getShapeAttr().getK()};

  // These variables define the set of allowed data types for matrices A, B, C,
  // and result.
  using AllowedShapes = SmallVector<std::array<int64_t, 3>, 2>;
  using AllowedTypes = SmallVector<SmallVector<Type, 4>, 2>;
  AllowedShapes allowedShapes;
  AllowedTypes expectedA;
  AllowedTypes expectedB;
  AllowedTypes expectedC;
  SmallVector<Type> expectedResult;

  // When M = 16, we just need to calculate the number of 8xk tiles, where
  // k is a factor that depends on the data type.
  if (mmaShape[0] == 16) {
    int64_t kFactor;
    Type multiplicandFragType;
    switch (*getMultiplicandAPtxType()) {
    case MMATypes::tf32:
      kFactor = 4;
      multiplicandFragType = i32Ty;
      expectedResult.push_back(LLVM::LLVMStructType::getLiteral(
          context, {f32Ty, f32Ty, f32Ty, f32Ty}));
      // Sparse MMA supports m16n8k8 and m16n8k16 for tf32
      allowedShapes.push_back({16, 8, 8});
      allowedShapes.push_back({16, 8, 16});
      break;
    case MMATypes::bf16:
      kFactor = 8;
      multiplicandFragType = i32Ty;
      expectedResult.push_back(LLVM::LLVMStructType::getLiteral(
          context, {f32Ty, f32Ty, f32Ty, f32Ty}));
      // Sparse MMA supports m16n8k16 and m16n8k32 for bf16
      allowedShapes.push_back({16, 8, 16});
      allowedShapes.push_back({16, 8, 32});
      break;
    case MMATypes::f16:
      kFactor = 8;
      multiplicandFragType = f16x2Ty;
      expectedResult.push_back(f16x2x2StructTy);
      expectedResult.push_back(f32x4StructTy);
      // Sparse MMA supports m16n8k16 and m16n8k32 for f16
      allowedShapes.push_back({16, 8, 16});
      allowedShapes.push_back({16, 8, 32});
      break;
    case MMATypes::s4:
    case MMATypes::u4:
      kFactor = 32;
      // Sparse MMA supports m16n8k64 and m16n8k128 for s4/u4
      allowedShapes.push_back({16, 8, 64});
      allowedShapes.push_back({16, 8, 128});
      break;
    case MMATypes::s8:
    case MMATypes::u8:
      kFactor = 16;
      // Sparse MMA supports m16n8k32 and m16n8k64 for s8/u8
      allowedShapes.push_back({16, 8, 32});
      allowedShapes.push_back({16, 8, 64});
      break;
    case MMATypes::e4m3:
    case MMATypes::e5m2:
    case MMATypes::e3m2:
    case MMATypes::e2m3:
    case MMATypes::e2m1:
      kFactor = 16;
      multiplicandFragType = i32Ty;
      expectedResult.push_back(f16x2x2StructTy);
      expectedResult.push_back(f32x4StructTy);
      // Sparse MMA supports m16n8k64 for FP8 types
      allowedShapes.push_back({16, 8, 64});
      break;
    default:
      return emitError("invalid shape or multiplicand type: ")
             << getMultiplicandAPtxType().value();
    }

    if (isIntegerPtxType(getMultiplicandAPtxType().value())) {
      expectedResult.push_back(s32x4StructTy);
      expectedC.emplace_back(4, i32Ty);
      multiplicandFragType = i32Ty;
    } else if (*getMultiplicandAPtxType() >= MMATypes::e4m3 &&
               *getMultiplicandAPtxType() <= MMATypes::e2m1) {
      // FP8 types
      expectedC.emplace_back(2, f16x2Ty);
      expectedC.emplace_back(4, f32Ty);
    } else {
      expectedC.emplace_back(2, f16x2Ty);
      expectedC.emplace_back(4, f32Ty);
    }

    // For sparse MMA, A operand is compressed (2:4 sparsity means half the
    // elements)
    int64_t unitA = (mmaShape[0] / 8) * (mmaShape[2] / kFactor) / 2;
    int64_t unitB = (mmaShape[1] / 8) * (mmaShape[2] / kFactor);
    expectedA.emplace_back(unitA, multiplicandFragType);
    expectedB.emplace_back(unitB, multiplicandFragType);

    if (resultPtxType() != accumPtxType())
      return emitOpError("ctype does not match dtype");
  }

  // In the M=8 case, there is only 1 possible case per data type.
  if (mmaShape[0] == 8) {
    if (*getMultiplicandAPtxType() == MMATypes::f16) {
      expectedA.emplace_back(2, f16x2Ty);
      expectedB.emplace_back(2, f16x2Ty);
      expectedResult.push_back(f16x2x4StructTy);
      expectedResult.push_back(f32x8StructTy);
      expectedC.emplace_back(4, f16x2Ty);
      expectedC.emplace_back(8, f32Ty);
      allowedShapes.push_back({8, 8, 4});
    }
    if (*getMultiplicandAPtxType() == MMATypes::f64) {
      Type f64Ty = Float64Type::get(context);
      expectedA.emplace_back(1, f64Ty);
      expectedB.emplace_back(1, f64Ty);
      expectedC.emplace_back(2, f64Ty);
      expectedResult.emplace_back(LLVM::LLVMStructType::getLiteral(
          context, SmallVector<Type>(2, f64Ty)));
      allowedShapes.push_back({8, 8, 4});
    }
    if (isIntegerPtxType(getMultiplicandAPtxType().value())) {
      expectedA.push_back({i32Ty});
      expectedB.push_back({i32Ty});
      expectedC.push_back({i32Ty, i32Ty});
      expectedResult.push_back(s32x2StructTy);
      if (isInt4PtxType(getMultiplicandAPtxType().value()))
        allowedShapes.push_back({8, 8, 32});
      if (isInt8PtxType(getMultiplicandAPtxType().value()))
        allowedShapes.push_back({8, 8, 16});
    }
  }

  std::string errorMessage;
  llvm::raw_string_ostream errorStream(errorMessage);

  // Check that we matched an existing shape/dtype combination.
  if (expectedA.empty() || expectedB.empty() || expectedC.empty() ||
      !llvm::is_contained(allowedShapes, mmaShape)) {
    errorStream << "unimplemented variant for MMA shape <";
    llvm::interleaveComma(mmaShape, errorStream);
    errorStream << ">";
    return emitOpError(errorMessage);
  }

  // Verify the operand types for segments of A, B, and C operands.
  std::array<StringRef, 3> operandNames{"A", "B", "C"};
  for (const auto &iter : llvm::enumerate(
           std::array<AllowedTypes, 3>{std::move(expectedA),
                                       std::move(expectedB),
                                       std::move(expectedC)})) {
    auto spec = this->getODSOperandIndexAndLength(iter.index());
    SmallVector<Type, 4> operandTySeg(operand_type_begin() + spec.first,
                                      operand_type_begin() + spec.first +
                                          spec.second);
    bool match = llvm::is_contained(iter.value(), operandTySeg);

    if (!match) {
      errorStream << "Could not match types for the "
                  << operandNames[iter.index()]
                  << " operands; expected one of ";
      for (const auto &x : iter.value()) {
        errorStream << x.size() << "x" << x[0] << " ";
      }
      errorStream << "but got ";
      llvm::interleaveComma(operandTySeg, errorStream);
      return emitOpError(errorMessage);
    }
  }

  // Check the result type
  if (!llvm::any_of(expectedResult, [&](Type expectedResultType) {
        return expectedResultType == getResult().getType();
      })) {
    errorStream
        << "Could not match allowed types for the result; expected one of ";
    llvm::interleaveComma(expectedResult, errorStream);
    errorStream << " but got " << getResult().getType();
    return emitOpError(errorMessage);
  }

  // Ensure int4/int8 MMA variants specify the accum overflow behavior
  // attribute.
  if (isInt4PtxType(*getMultiplicandAPtxType()) ||
      isInt8PtxType(*getMultiplicandAPtxType())) {
    if (!getIntOverflowBehavior())
      return emitOpError("op requires " +
                         getIntOverflowBehaviorAttrName().strref() +
                         " attribute");
  }

  // Validate sparse metadata type (should be i32)
  if (!getSparseMetadata().getType().isInteger(32)) {
    return emitOpError() << "sparse metadata must be i32 type";
  }

  // Validate sparsity selector type (should be i32)
  if (!getSparsitySelector().getType().isInteger(32)) {
    return emitOpError() << "sparsity selector must be i32 type";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MMA Block Scale Operations - Shared Helpers
//===----------------------------------------------------------------------===//

namespace {
// Shared structure for MMA operand fragments (A, B, C)
struct MMAOperandFragment {
  StringRef operandName;
  StringRef ptxTypeAttr;
  SmallVector<Value, 4> regs;
  explicit MMAOperandFragment(StringRef name, StringRef ptxTypeName)
      : operandName(name), ptxTypeAttr(ptxTypeName) {}
};
} // namespace

// Helper to print operand list in the format: name[operands]
static void printOperandList(OpAsmPrinter &p, StringRef name,
                             ArrayRef<Value> operands) {
  p << " " << name << "[";
  p.printOperands(operands);
  p << "]";
}

// Helper to parse operand list in the format: name[operands]
static LogicalResult
parseMmaOperand(OpAsmParser &parser, StringRef operandName,
                SmallVectorImpl<OpAsmParser::UnresolvedOperand> &regs) {
  if (parser.parseKeyword(operandName).failed())
    return failure();
  if (parser.parseOperandList(regs, OpAsmParser::Delimiter::OptionalSquare)
          .failed())
    return failure();
  return success();
}

// Helper to process operand fragments and determine which attributes can be
// inferred
template <typename Op>
static void
processOperandFragments(Op &op, std::array<MMAOperandFragment, 3> &frags,
                        SmallVectorImpl<Type> &regTypes,
                        SmallVectorImpl<StringRef> &ignoreAttrNames) {
  for (unsigned fragIdx = 0; fragIdx < frags.size(); fragIdx++) {
    auto &frag = frags[fragIdx];
    auto varOperandSpec = op.getODSOperandIndexAndLength(fragIdx);
    for (auto operandIdx = varOperandSpec.first;
         operandIdx < varOperandSpec.first + varOperandSpec.second;
         operandIdx++) {
      frag.regs.push_back(op.getOperand(operandIdx));
      if (fragIdx == 0 && operandIdx == varOperandSpec.first) {
        regTypes.push_back(op.getOperand(operandIdx).getType());
      }
    }
    if (fragIdx < 2) {
      regTypes.push_back(frag.regs[0].getType());
    }
    std::optional<MMATypes> inferredType =
        MmaOp::inferOperandMMAType(regTypes.back(),
                                   /*isAccumulator=*/fragIdx >= 2);
    if (inferredType)
      ignoreAttrNames.push_back(frag.ptxTypeAttr);
  }
}

// Helper to parse type signature: (A_type, B_type, C_type)
static LogicalResult
parseMmaTypeSignature(OpAsmParser &parser,
                      SmallVectorImpl<Type> &operandTypes) {
  if (parser.parseColon().failed() || parser.parseLParen().failed())
    return failure();

  auto typeParser = [&]() {
    Type ty;
    if (parser.parseType(ty).failed())
      return failure();
    operandTypes.push_back(ty);
    return success();
  };
  if (parser.parseCommaSeparatedList(typeParser))
    return failure();

  if (operandTypes.size() != 3)
    return parser.emitError(parser.getCurrentLocation(),
                            "expected exactly 3 types");

  return parser.parseRParen();
}

// Helper to infer and set multiplicand PTX type attributes
static void
inferAndSetMultiplicandTypes(MLIRContext *ctx, NamedAttrList &attrs,
                             const SmallVectorImpl<Type> &operandTypes) {
  if (!attrs.get("multiplicandAPtxType")) {
    if (auto inferredType =
            MmaOp::inferOperandMMAType(operandTypes[0], false)) {
      attrs.set("multiplicandAPtxType", MMATypesAttr::get(ctx, *inferredType));
    }
  }
  if (!attrs.get("multiplicandBPtxType")) {
    if (auto inferredType =
            MmaOp::inferOperandMMAType(operandTypes[1], false)) {
      attrs.set("multiplicandBPtxType", MMATypesAttr::get(ctx, *inferredType));
    }
  }
}

// Helper to add common block scale properties
template <typename OpType>
static void addBlockScaleProperties(OpBuilder &builder, OperationState &result,
                                    ArrayRef<int64_t> shape,
                                    ScaleVecSize scaleVecSize,
                                    BlockScaleFormat blockScaleFormat,
                                    MMABlockScaleKind kind) {
  MLIRContext *ctx = builder.getContext();
  auto &properties = result.getOrAddProperties<typename OpType::Properties>();
  properties.setShape(
      builder.getAttr<MMAShapeAttr>(shape[0], shape[1], shape[2]));
  properties.setScaleVecSize(ScaleVecSizeAttr::get(ctx, scaleVecSize));
  properties.setBlockScaleFormat(
      BlockScaleFormatAttr::get(ctx, blockScaleFormat));
  properties.setKind(MMABlockScaleKindAttr::get(ctx, kind));
}

// Helper to infer and add multiplicand PTX types to builder
static void addInferredMultiplicandTypes(
    MLIRContext *ctx, OperationState &result, ValueRange operandA,
    ValueRange operandB,
    std::optional<std::array<MMATypes, 2>> multiplicandPtxTypes) {
  if (multiplicandPtxTypes) {
    result.addAttribute("multiplicandAPtxType",
                        MMATypesAttr::get(ctx, (*multiplicandPtxTypes)[0]));
    result.addAttribute("multiplicandBPtxType",
                        MMATypesAttr::get(ctx, (*multiplicandPtxTypes)[1]));
  } else {
    if (auto res = MmaOp::inferOperandMMAType(operandA[0].getType(), false))
      result.addAttribute("multiplicandAPtxType", MMATypesAttr::get(ctx, *res));
    if (auto res = MmaOp::inferOperandMMAType(operandB[0].getType(), false))
      result.addAttribute("multiplicandBPtxType", MMATypesAttr::get(ctx, *res));
  }
}

// Template helper for common accumPtxType/resultPtxType implementation
template <typename OpTy>
static MMATypes inferPtxTypeFromResult(OpTy op) {
  return *MmaOp::inferOperandMMAType(
      cast<LLVM::LLVMStructType>(op.getRes().getType()).getBody()[0],
      /*isAccumulator=*/true);
}

//===----------------------------------------------------------------------===//
// MmaBlockScaleOp
//===----------------------------------------------------------------------===//

void MmaBlockScaleOp::print(OpAsmPrinter &p) {
  SmallVector<Type, 4> regTypes;
  std::array<MMAOperandFragment, 3> frags{
      MMAOperandFragment("A", getMultiplicandAPtxTypeAttrName()),
      MMAOperandFragment("B", getMultiplicandBPtxTypeAttrName()),
      MMAOperandFragment("C", "")};
  SmallVector<StringRef, 4> ignoreAttrNames{
      mlir::NVVM::MmaBlockScaleOp::getOperandSegmentSizeAttr()};

  processOperandFragments(*this, frags, regTypes, ignoreAttrNames);

  // Print A, B, C operands
  for (const auto &frag : frags)
    printOperandList(p, frag.operandName, frag.regs);

  // Print scale operands
  printOperandList(p, "scaleA",
                   {getScaleAData(), getByteIdA(), getThreadIdA()});
  printOperandList(p, "scaleB",
                   {getScaleBData(), getByteIdB(), getThreadIdB()});

  bool isFirstProperty = true;
  printMmaProperty(p, isFirstProperty, "shape", getShapeAttr());
  if (getMultiplicandAPtxTypeAttr() &&
      !llvm::is_contained(ignoreAttrNames, getMultiplicandAPtxTypeAttrName()))
    printMmaEnumProperty(p, isFirstProperty, "multiplicand_a_ptx_type",
                         getMultiplicandAPtxTypeAttr());
  if (getMultiplicandBPtxTypeAttr() &&
      !llvm::is_contained(ignoreAttrNames, getMultiplicandBPtxTypeAttrName()))
    printMmaEnumProperty(p, isFirstProperty, "multiplicand_b_ptx_type",
                         getMultiplicandBPtxTypeAttr());
  printMmaEnumProperty(p, isFirstProperty, "scale_vec_size",
                       getScaleVecSizeAttr());
  printMmaEnumProperty(p, isFirstProperty, "block_scale_format",
                       getBlockScaleFormatAttr());
  printMmaEnumProperty(p, isFirstProperty, "kind", getKindAttr());
  llvm::append_range(
      ignoreAttrNames,
      ArrayRef<StringRef>{getShapeAttrName(), getMultiplicandAPtxTypeAttrName(),
                          getMultiplicandBPtxTypeAttrName(),
                          getScaleVecSizeAttrName(),
                          getBlockScaleFormatAttrName(), getKindAttrName()});
  p.printOptionalAttrDict((*this)->getDiscardableAttrDictionary().getValue(),
                          ignoreAttrNames);

  // Print type signature
  p << " : (";
  llvm::interleaveComma(SmallVector<Type, 3>{frags[0].regs[0].getType(),
                                             frags[1].regs[0].getType(),
                                             frags[2].regs[0].getType()},
                        p);
  p << ")";
  p.printArrowTypeList(TypeRange{this->getRes().getType()});
}

ParseResult MmaBlockScaleOp::parse(OpAsmParser &parser,
                                   OperationState &result) {
  struct LocalOperandFragment {
    std::optional<MMATypes> elemtype;
    SmallVector<OpAsmParser::UnresolvedOperand, 4> regs;
  };

  Builder &builder = parser.getBuilder();
  std::array<LocalOperandFragment, 3> frags;
  NamedAttrList namedAttributes;

  // Parse A[...] B[...] C[...]
  if (parseMmaOperand(parser, "A", frags[0].regs).failed() ||
      parseMmaOperand(parser, "B", frags[1].regs).failed() ||
      parseMmaOperand(parser, "C", frags[2].regs).failed())
    return failure();

  // Parse scale operands: scaleA[...] scaleB[...]
  SmallVector<OpAsmParser::UnresolvedOperand, 3> scaleAOperands, scaleBOperands;
  if (parseMmaOperand(parser, "scaleA", scaleAOperands).failed() ||
      parseMmaOperand(parser, "scaleB", scaleBOperands).failed())
    return failure();

  if (parseMmaProperties(parser, namedAttributes,
                         {"shape", "multiplicand_a_ptx_type",
                          "multiplicand_b_ptx_type", "scale_vec_size",
                          "block_scale_format", "kind"},
                         {"shape", "scaleVecSize", "blockScaleFormat", "kind"}))
    return failure();

  // Parse type signature
  SmallVector<Type, 3> operandTypes;
  if (parseMmaTypeSignature(parser, operandTypes).failed())
    return failure();

  // Parse result type
  SmallVector<Type, 1> resultTypes;
  if (parser.parseArrowTypeList(resultTypes).failed())
    return failure();

  // Infer element types and resolve operands
  for (const auto &[idx, frag] : llvm::enumerate(frags)) {
    frag.elemtype = MmaOp::inferOperandMMAType(operandTypes[idx],
                                               /*isAccumulator=*/idx >= 2);
    if (parser
            .resolveOperands(frag.regs, operandTypes[idx], parser.getNameLoc(),
                             result.operands)
            .failed())
      return failure();
  }

  // Resolve scale operands
  SmallVector<Type, 3> scaleTypes = {builder.getI32Type(), builder.getI16Type(),
                                     builder.getI16Type()};
  if (parser
          .resolveOperands(scaleAOperands, scaleTypes, parser.getNameLoc(),
                           result.operands)
          .failed() ||
      parser
          .resolveOperands(scaleBOperands, scaleTypes, parser.getNameLoc(),
                           result.operands)
          .failed())
    return failure();

  // Add attributes
  result.addAttributes(namedAttributes);
  inferAndSetMultiplicandTypes(parser.getContext(), result.attributes,
                               operandTypes);

  result.addTypes(resultTypes);
  result.addAttribute(MmaBlockScaleOp::getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr({
                          static_cast<int32_t>(frags[0].regs.size()),
                          static_cast<int32_t>(frags[1].regs.size()),
                          static_cast<int32_t>(frags[2].regs.size()),
                          1, // scaleAData
                          1, // byteIdA
                          1, // threadIdA
                          1, // scaleBData
                          1, // byteIdB
                          1  // threadIdB
                      }));
  return success();
}

void MmaBlockScaleOp::build(
    OpBuilder &builder, OperationState &result, Type resultType,
    ValueRange operandA, ValueRange operandB, ValueRange operandC,
    Value scaleAData, Value byteIdA, Value threadIdA, Value scaleBData,
    Value byteIdB, Value threadIdB, ArrayRef<int64_t> shape,
    std::optional<std::array<MMATypes, 2>> multiplicandPtxTypes,
    ScaleVecSize scaleVecSize, BlockScaleFormat blockScaleFormat,
    MMABlockScaleKind kind) {
  assert(shape.size() == 3 && "expected shape to have size 3 (m, n, k)");

  addBlockScaleProperties<MmaBlockScaleOp>(builder, result, shape, scaleVecSize,
                                           blockScaleFormat, kind);

  result.addOperands(operandA);
  result.addOperands(operandB);
  result.addOperands(operandC);
  result.addOperands(
      {scaleAData, byteIdA, threadIdA, scaleBData, byteIdB, threadIdB});

  addInferredMultiplicandTypes(builder.getContext(), result, operandA, operandB,
                               multiplicandPtxTypes);

  result.addTypes(resultType);
  result.addAttribute(MmaBlockScaleOp::getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr({
                          static_cast<int32_t>(operandA.size()),
                          static_cast<int32_t>(operandB.size()),
                          static_cast<int32_t>(operandC.size()),
                          1, // scaleAData
                          1, // byteIdA
                          1, // threadIdA
                          1, // scaleBData
                          1, // byteIdB
                          1  // threadIdB
                      }));
}

NVVM::IDArgPair MmaBlockScaleOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto curOp = cast<NVVM::MmaBlockScaleOp>(op);

  SmallVector<llvm::Value *> args;
  // Add A, B, C operands
  for (Value operand : curOp.getOperandA())
    args.push_back(mt.lookupValue(operand));
  for (Value operand : curOp.getOperandB())
    args.push_back(mt.lookupValue(operand));
  for (Value operand : curOp.getOperandC())
    args.push_back(mt.lookupValue(operand));

  // Add scale operands
  args.push_back(mt.lookupValue(curOp.getScaleAData()));
  args.push_back(mt.lookupValue(curOp.getByteIdA()));
  args.push_back(mt.lookupValue(curOp.getThreadIdA()));
  args.push_back(mt.lookupValue(curOp.getScaleBData()));
  args.push_back(mt.lookupValue(curOp.getByteIdB()));
  args.push_back(mt.lookupValue(curOp.getThreadIdB()));

  unsigned intId = MmaBlockScaleOp::getIntrinsicID(
      curOp.getShape().getM(), curOp.getShape().getN(), curOp.getShape().getK(),
      *curOp.getMultiplicandAPtxType(), *curOp.getMultiplicandBPtxType(),
      inferPtxTypeFromResult(curOp), curOp.getScaleVecSize(),
      curOp.getBlockScaleFormat(), curOp.getKind());

  return {intId, args};
}

LogicalResult MmaBlockScaleOp::verify() {
  LogicalResult result = success();
  int m = getShape().getM();
  int n = getShape().getN();
  int k = getShape().getK();

  if (m == 16 && n == 8 && k == 64) {
    if (getMultiplicandAPtxType() != NVVM::MMATypes::e2m1 ||
        getMultiplicandBPtxType() != NVVM::MMATypes::e2m1)
      result = emitOpError(
          "unsupported MMATypes attribute for mma.m16n8k64.(mxf4nvf4|mxf4)");
    if (getKind() == NVVM::MMABlockScaleKind::MXF4) {
      if (getScaleVecSize() != NVVM::ScaleVecSize::X2)
        result = emitOpError(
            "unsupported ScaleVecSize attribute for mma.m16n8k64.mxf4");
      if (getBlockScaleFormat() != NVVM::BlockScaleFormat::UE8M0)
        result = emitOpError(
            "unsupported BlockScaleFormat attribute for mma.m16n8k64.mxf4");
    } else if (getKind() == NVVM::MMABlockScaleKind::MXF4NVF4) {
      if (!((getScaleVecSize() == NVVM::ScaleVecSize::X2 &&
             getBlockScaleFormat() == NVVM::BlockScaleFormat::UE8M0) ||
            (getScaleVecSize() == NVVM::ScaleVecSize::X4 &&
             (getBlockScaleFormat() == NVVM::BlockScaleFormat::UE4M3 ||
              getBlockScaleFormat() == NVVM::BlockScaleFormat::UE8M0))))
        result = emitOpError("unsupported ScaleVecSize and BlockScaleFormat "
                             "attributes for mma.m16n8k64.mxf4nvf4");
    } else {
      result = emitOpError("unsupported Kind attribute for mma.m16n8k64");
    }
  } else if (m == 16 && n == 8 && k == 32) {
    if (!(getKind() == NVVM::MMABlockScaleKind::MXF8F6F4 &&
          getScaleVecSize() == NVVM::ScaleVecSize::X1 &&
          getBlockScaleFormat() == NVVM::BlockScaleFormat::UE8M0))
      result =
          emitOpError("unsupported Kind, ScaleVecSize and BlockScaleFormat "
                      "attributes for mma.m16n8k32");
  } else {
    result = emitOpError("unsupported Geom for mma with block scaling");
  }
  return result;
}

//===----------------------------------------------------------------------===//
// MmaSpBlockScaleOp
//===----------------------------------------------------------------------===//

void MmaSpBlockScaleOp::print(OpAsmPrinter &p) {
  SmallVector<Type, 4> regTypes;
  std::array<MMAOperandFragment, 3> frags{
      MMAOperandFragment("A", getMultiplicandAPtxTypeAttrName()),
      MMAOperandFragment("B", getMultiplicandBPtxTypeAttrName()),
      MMAOperandFragment("C", "")};
  SmallVector<StringRef, 4> ignoreAttrNames{
      mlir::NVVM::MmaSpBlockScaleOp::getOperandSegmentSizeAttr()};

  processOperandFragments(*this, frags, regTypes, ignoreAttrNames);

  // Print A, B, C operands
  for (const auto &frag : frags)
    printOperandList(p, frag.operandName, frag.regs);

  // Print sparse-specific operands
  printOperandList(p, "sparseMetadata", {getSparseMetadata()});
  printOperandList(p, "selector", {getSparsitySelector()});

  // Print scale operands
  printOperandList(p, "scaleA",
                   {getScaleAData(), getByteIdA(), getThreadIdA()});
  printOperandList(p, "scaleB",
                   {getScaleBData(), getByteIdB(), getThreadIdB()});

  bool isFirstProperty = true;
  printMmaProperty(p, isFirstProperty, "shape", getShapeAttr());
  if (getMultiplicandAPtxTypeAttr() &&
      !llvm::is_contained(ignoreAttrNames, getMultiplicandAPtxTypeAttrName()))
    printMmaEnumProperty(p, isFirstProperty, "multiplicand_a_ptx_type",
                         getMultiplicandAPtxTypeAttr());
  if (getMultiplicandBPtxTypeAttr() &&
      !llvm::is_contained(ignoreAttrNames, getMultiplicandBPtxTypeAttrName()))
    printMmaEnumProperty(p, isFirstProperty, "multiplicand_b_ptx_type",
                         getMultiplicandBPtxTypeAttr());
  printMmaUnitProperty(p, isFirstProperty, "ordered_metadata");
  printMmaEnumProperty(p, isFirstProperty, "scale_vec_size",
                       getScaleVecSizeAttr());
  printMmaEnumProperty(p, isFirstProperty, "block_scale_format",
                       getBlockScaleFormatAttr());
  printMmaEnumProperty(p, isFirstProperty, "kind", getKindAttr());
  llvm::append_range(
      ignoreAttrNames,
      ArrayRef<StringRef>{getShapeAttrName(), getMultiplicandAPtxTypeAttrName(),
                          getMultiplicandBPtxTypeAttrName(),
                          getOrderedMetadataAttrName(),
                          getScaleVecSizeAttrName(),
                          getBlockScaleFormatAttrName(), getKindAttrName()});
  p.printOptionalAttrDict((*this)->getDiscardableAttrDictionary().getValue(),
                          ignoreAttrNames);

  // Print type signature
  p << " : (";
  llvm::interleaveComma(SmallVector<Type, 3>{frags[0].regs[0].getType(),
                                             frags[1].regs[0].getType(),
                                             frags[2].regs[0].getType()},
                        p);
  p << ")";
  p.printArrowTypeList(TypeRange{this->getRes().getType()});
}

ParseResult MmaSpBlockScaleOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  struct LocalOperandFragment {
    std::optional<MMATypes> elemtype;
    SmallVector<OpAsmParser::UnresolvedOperand, 4> regs;
  };

  Builder &builder = parser.getBuilder();
  std::array<LocalOperandFragment, 3> frags;
  NamedAttrList namedAttributes;

  // Parse A[...] B[...] C[...]
  if (parseMmaOperand(parser, "A", frags[0].regs).failed() ||
      parseMmaOperand(parser, "B", frags[1].regs).failed() ||
      parseMmaOperand(parser, "C", frags[2].regs).failed())
    return failure();

  // Parse sparse-specific operands
  SmallVector<OpAsmParser::UnresolvedOperand, 1> metadataOperands,
      selectorOperands;
  if (parseMmaOperand(parser, "sparseMetadata", metadataOperands).failed() ||
      parseMmaOperand(parser, "selector", selectorOperands).failed())
    return failure();

  // Parse scale operands
  SmallVector<OpAsmParser::UnresolvedOperand, 3> scaleAOperands, scaleBOperands;
  if (parseMmaOperand(parser, "scaleA", scaleAOperands).failed() ||
      parseMmaOperand(parser, "scaleB", scaleBOperands).failed())
    return failure();

  if (parseMmaProperties(parser, namedAttributes,
                         {"shape", "multiplicand_a_ptx_type",
                          "multiplicand_b_ptx_type", "ordered_metadata",
                          "scale_vec_size", "block_scale_format", "kind"},
                         {"shape", "scaleVecSize", "blockScaleFormat", "kind"}))
    return failure();

  // Parse type signature
  SmallVector<Type, 3> operandTypes;
  if (parseMmaTypeSignature(parser, operandTypes).failed())
    return failure();

  // Parse result type
  SmallVector<Type, 1> resultTypes;
  if (parser.parseArrowTypeList(resultTypes).failed())
    return failure();

  // Infer element types and resolve operands
  for (const auto &[idx, frag] : llvm::enumerate(frags)) {
    frag.elemtype = MmaOp::inferOperandMMAType(operandTypes[idx],
                                               /*isAccumulator=*/idx >= 2);
    if (parser
            .resolveOperands(frag.regs, operandTypes[idx], parser.getNameLoc(),
                             result.operands)
            .failed())
      return failure();
  }

  // Resolve sparse metadata and selector
  Type i32Type = builder.getI32Type();
  if (parser
          .resolveOperands(metadataOperands, i32Type, parser.getNameLoc(),
                           result.operands)
          .failed() ||
      parser
          .resolveOperands(selectorOperands, i32Type, parser.getNameLoc(),
                           result.operands)
          .failed())
    return failure();

  // Resolve scale operands
  SmallVector<Type, 3> scaleTypes = {i32Type, builder.getI16Type(),
                                     builder.getI16Type()};
  if (parser
          .resolveOperands(scaleAOperands, scaleTypes, parser.getNameLoc(),
                           result.operands)
          .failed() ||
      parser
          .resolveOperands(scaleBOperands, scaleTypes, parser.getNameLoc(),
                           result.operands)
          .failed())
    return failure();

  // Add attributes
  result.addAttributes(namedAttributes);
  inferAndSetMultiplicandTypes(parser.getContext(), result.attributes,
                               operandTypes);

  // orderedMetadata is mandatory
  if (!result.attributes.get("orderedMetadata"))
    result.addAttribute("orderedMetadata", builder.getUnitAttr());

  result.addTypes(resultTypes);
  result.addAttribute(MmaSpBlockScaleOp::getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr({
                          static_cast<int32_t>(frags[0].regs.size()),
                          static_cast<int32_t>(frags[1].regs.size()),
                          static_cast<int32_t>(frags[2].regs.size()),
                          1, // sparseMetadata
                          1, // sparsitySelector
                          1, // scaleAData
                          1, // byteIdA
                          1, // threadIdA
                          1, // scaleBData
                          1, // byteIdB
                          1  // threadIdB
                      }));
  return success();
}

void MmaSpBlockScaleOp::build(
    OpBuilder &builder, OperationState &result, Type resultType,
    ValueRange operandA, ValueRange operandB, ValueRange operandC,
    Value sparseMetadata, Value sparsitySelector, Value scaleAData,
    Value byteIdA, Value threadIdA, Value scaleBData, Value byteIdB,
    Value threadIdB, ArrayRef<int64_t> shape,
    std::optional<std::array<MMATypes, 2>> multiplicandPtxTypes,
    ScaleVecSize scaleVecSize, BlockScaleFormat blockScaleFormat,
    MMABlockScaleKind kind) {
  assert(shape.size() == 3 && "expected shape to have size 3 (m, n, k)");

  addBlockScaleProperties<MmaSpBlockScaleOp>(
      builder, result, shape, scaleVecSize, blockScaleFormat, kind);
  result.addAttribute("orderedMetadata", builder.getUnitAttr());

  result.addOperands(operandA);
  result.addOperands(operandB);
  result.addOperands(operandC);
  result.addOperands({sparseMetadata, sparsitySelector, scaleAData, byteIdA,
                      threadIdA, scaleBData, byteIdB, threadIdB});

  addInferredMultiplicandTypes(builder.getContext(), result, operandA, operandB,
                               multiplicandPtxTypes);

  result.addTypes(resultType);
  result.addAttribute(MmaSpBlockScaleOp::getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr({
                          static_cast<int32_t>(operandA.size()),
                          static_cast<int32_t>(operandB.size()),
                          static_cast<int32_t>(operandC.size()),
                          1, // sparseMetadata
                          1, // sparsitySelector
                          1, // scaleAData
                          1, // byteIdA
                          1, // threadIdA
                          1, // scaleBData
                          1, // byteIdB
                          1  // threadIdB
                      }));
}

NVVM::IDArgPair MmaSpBlockScaleOp::getIntrinsicIDAndArgs(
    Operation &op, LLVM::ModuleTranslation &mt, llvm::IRBuilderBase &builder) {
  auto curOp = cast<NVVM::MmaSpBlockScaleOp>(op);

  SmallVector<llvm::Value *> args;
  // Add A, B, C operands
  for (Value operand : curOp.getOperandA())
    args.push_back(mt.lookupValue(operand));
  for (Value operand : curOp.getOperandB())
    args.push_back(mt.lookupValue(operand));
  for (Value operand : curOp.getOperandC())
    args.push_back(mt.lookupValue(operand));

  // Add sparse metadata and selector
  args.push_back(mt.lookupValue(curOp.getSparseMetadata()));
  args.push_back(mt.lookupValue(curOp.getSparsitySelector()));

  // Add scale operands
  args.push_back(mt.lookupValue(curOp.getScaleAData()));
  args.push_back(mt.lookupValue(curOp.getByteIdA()));
  args.push_back(mt.lookupValue(curOp.getThreadIdA()));
  args.push_back(mt.lookupValue(curOp.getScaleBData()));
  args.push_back(mt.lookupValue(curOp.getByteIdB()));
  args.push_back(mt.lookupValue(curOp.getThreadIdB()));

  unsigned intId = MmaSpBlockScaleOp::getIntrinsicID(
      curOp.getShape().getM(), curOp.getShape().getN(), curOp.getShape().getK(),
      *curOp.getMultiplicandAPtxType(), *curOp.getMultiplicandBPtxType(),
      inferPtxTypeFromResult(curOp), curOp.getScaleVecSize(),
      curOp.getBlockScaleFormat(), curOp.getKind());

  return {intId, args};
}

LogicalResult MmaSpBlockScaleOp::verify() {
  // Check that orderedMetadata is present
  if (!getOrderedMetadata()) {
    return emitOpError("'orderedMetadata' attribute is mandatory");
  }

  LogicalResult result = success();
  int m = getShape().getM();
  int n = getShape().getN();
  int k = getShape().getK();

  if (m == 16 && n == 8 && k == 128) {
    if (getMultiplicandAPtxType() != NVVM::MMATypes::e2m1 ||
        getMultiplicandBPtxType() != NVVM::MMATypes::e2m1)
      result = emitOpError(
          "unsupported MMATypes attribute for mma.m16n8k128.(mxf4nvf4|mxf4)");
    if (getKind() == NVVM::MMABlockScaleKind::MXF4) {
      if (getScaleVecSize() != NVVM::ScaleVecSize::X2)
        result = emitOpError(
            "unsupported ScaleVecSize attribute for mma.m16n8k128.mxf4");
      if (getBlockScaleFormat() != NVVM::BlockScaleFormat::UE8M0)
        result = emitOpError(
            "unsupported BlockScaleFormat attribute for mma.m16n8k128.mxf4");
    } else if (getKind() == NVVM::MMABlockScaleKind::MXF4NVF4) {
      if (!((getScaleVecSize() == NVVM::ScaleVecSize::X2 &&
             getBlockScaleFormat() == NVVM::BlockScaleFormat::UE8M0) ||
            (getScaleVecSize() == NVVM::ScaleVecSize::X4 &&
             (getBlockScaleFormat() == NVVM::BlockScaleFormat::UE4M3 ||
              getBlockScaleFormat() == NVVM::BlockScaleFormat::UE8M0))))
        result = emitOpError("unsupported ScaleVecSize and BlockScaleFormat "
                             "attributes for mma.m16n8k128.mxf4nvf4");
    } else {
      result = emitOpError("unsupported Kind attribute for mma.m16n8k128");
    }
  } else if (m == 16 && n == 8 && k == 64) {
    if (!(getKind() == NVVM::MMABlockScaleKind::MXF8F6F4 &&
          getScaleVecSize() == NVVM::ScaleVecSize::X1 &&
          getBlockScaleFormat() == NVVM::BlockScaleFormat::UE8M0))
      result =
          emitOpError("unsupported Kind, ScaleVecSize and BlockScaleFormat "
                      "attributes for mma.m16n8k64");
  } else {
    result = emitOpError("unsupported Geom for sparse mma with block scaling");
  }
  return result;
}

std::pair<mlir::Type, unsigned> NVVM::inferMMAType(NVVM::MMATypes type,
                                                   NVVM::MMAFrag frag, int nRow,
                                                   int nCol,
                                                   MLIRContext *context) {
  unsigned numberElements = 0;
  Type elementType;
  OpBuilder builder(context);
  Type f16x2 = VectorType::get(2, builder.getF16Type());
  if (type == NVVM::MMATypes::f16) {
    elementType = f16x2;
    if (frag == NVVM::MMAFrag::a || frag == NVVM::MMAFrag::b)
      numberElements = 8;
    else
      numberElements = 4;
  } else if (type == NVVM::MMATypes::f32) {
    elementType = builder.getF32Type();
    numberElements = 8;
  } else if (type == NVVM::MMATypes::f64) {
    elementType = builder.getF64Type();
    if (frag == NVVM::MMAFrag::a || frag == NVVM::MMAFrag::b)
      numberElements = 1;
    else
      numberElements = 2;
  } else if (type == NVVM::MMATypes::tf32) {
    elementType = builder.getI32Type();
    numberElements = 4;
  } else if (type == NVVM::MMATypes::s8 || type == NVVM::MMATypes::u8) {
    elementType = builder.getI32Type();
    int parallelSize = 0;
    if (frag == NVVM::MMAFrag::a)
      parallelSize = nRow;
    if (frag == NVVM::MMAFrag::b)
      parallelSize = nCol;

    // m == 16 && n == 16 && k == 16
    if (parallelSize == 16)
      numberElements = 2;
    // m == 8 && n == 32 && k == 16 or m == 32 && n == 8 && k == 16
    else if (parallelSize == 8)
      numberElements = 1;
    else if (parallelSize == 32)
      numberElements = 4;
  } else if (type == NVVM::MMATypes::s32) {
    elementType = builder.getI32Type();
    numberElements = 8;
  }
  assert(numberElements != 0 && elementType != nullptr);
  return std::make_pair(elementType, numberElements);
}

static std::pair<mlir::Type, unsigned>
inferMMATypeFromMNK(NVVM::MMATypes type, NVVM::MMAFrag frag, int m, int n,
                    int k, MLIRContext *context) {
  int nRow, nCol;
  if (frag == NVVM::MMAFrag::a) {
    nRow = m;
    nCol = k;
  } else if (frag == NVVM::MMAFrag::b) {
    nRow = k;
    nCol = n;
  } else {
    nRow = m;
    nCol = n;
  }
  assert(nRow && nCol);
  return inferMMAType(type, frag, nRow, nCol, context);
}

LogicalResult NVVM::WMMALoadOp::verify() {
  unsigned addressSpace =
      llvm::cast<LLVM::LLVMPointerType>(getPtr().getType()).getAddressSpace();
  if (addressSpace != 0 && addressSpace != NVVMMemorySpace::Global &&
      addressSpace != NVVMMemorySpace::Shared)
    return emitOpError("expected source pointer in memory "
                       "space 0, 1, 3");

  if (NVVM::WMMALoadOp::getIntrinsicID(getM(), getN(), getK(), getLayout(),
                                       getEltype(), getFrag()) == 0)
    return emitOpError() << "invalid attribute combination";
  std::pair<Type, unsigned> typeInfo = inferMMATypeFromMNK(
      getEltype(), getFrag(), getM(), getN(), getK(), getContext());
  // Special case for f64 fragments
  Type f64Ty = Float64Type::get(getContext());
  if (typeInfo.first == f64Ty && typeInfo.second == 1) {
    if (getType() != f64Ty)
      return emitOpError("expected destination type to be f64");
    return success();
  }
  // Everything else is a struct
  Type dstType = LLVM::LLVMStructType::getLiteral(
      getContext(), SmallVector<Type, 8>(typeInfo.second, typeInfo.first));
  if (getType() != dstType)
    return emitOpError("expected destination type is a structure of ")
           << typeInfo.second << " elements of type " << typeInfo.first;
  return success();
}

LogicalResult NVVM::WMMAStoreOp::verify() {
  unsigned addressSpace =
      llvm::cast<LLVM::LLVMPointerType>(getPtr().getType()).getAddressSpace();
  if (addressSpace != 0 && addressSpace != NVVMMemorySpace::Global &&
      addressSpace != NVVMMemorySpace::Shared)
    return emitOpError("expected operands to be a source pointer in memory "
                       "space 0, 1, 3");

  if (NVVM::WMMAStoreOp::getIntrinsicID(getM(), getN(), getK(), getLayout(),
                                        getEltype()) == 0)
    return emitOpError() << "invalid attribute combination";
  std::pair<Type, unsigned> typeInfo = inferMMATypeFromMNK(
      getEltype(), NVVM::MMAFrag::c, getM(), getN(), getK(), getContext());
  if (getArgs().size() != typeInfo.second)
    return emitOpError() << "expected " << typeInfo.second << " data operands";
  if (llvm::any_of(getArgs(), [&typeInfo](Value operands) {
        return operands.getType() != typeInfo.first;
      }))
    return emitOpError() << "expected data operands of type " << typeInfo.first;
  return success();
}

LogicalResult NVVM::WMMAMmaOp::verify() {
  if (NVVM::WMMAMmaOp::getIntrinsicID(getM(), getN(), getK(), getLayoutA(),
                                      getLayoutB(), getEltypeA(),
                                      getEltypeB()) == 0)
    return emitOpError() << "invalid attribute combination";
  std::pair<Type, unsigned> typeInfoA = inferMMATypeFromMNK(
      getEltypeA(), NVVM::MMAFrag::a, getM(), getN(), getK(), getContext());
  std::pair<Type, unsigned> typeInfoB = inferMMATypeFromMNK(
      getEltypeA(), NVVM::MMAFrag::b, getM(), getN(), getK(), getContext());
  std::pair<Type, unsigned> typeInfoC = inferMMATypeFromMNK(
      getEltypeB(), NVVM::MMAFrag::c, getM(), getN(), getK(), getContext());
  SmallVector<Type, 32> arguments;
  arguments.append(typeInfoA.second, typeInfoA.first);
  arguments.append(typeInfoB.second, typeInfoB.first);
  arguments.append(typeInfoC.second, typeInfoC.first);
  unsigned numArgs = arguments.size();
  if (getArgs().size() != numArgs)
    return emitOpError() << "expected " << numArgs << " arguments";
  for (unsigned i = 0; i < numArgs; i++) {
    if (getArgs()[i].getType() != arguments[i])
      return emitOpError() << "expected argument " << i << " to be of type "
                           << arguments[i];
  }
  Type dstType = LLVM::LLVMStructType::getLiteral(
      getContext(), SmallVector<Type, 8>(typeInfoC.second, typeInfoC.first));
  if (getType() != dstType)
    return emitOpError("expected destination type is a structure of ")
           << typeInfoC.second << " elements of type " << typeInfoC.first;
  return success();
}

LogicalResult NVVM::LdMatrixOp::verify() {
  uint32_t num = getNum(), m = getShape().getM(), n = getShape().getN();
  if (m == 8 && n == 8) {
    if (num != 1 && num != 2 && num != 4) {
      return emitOpError("expected num attribute to be 1, 2 or 4 for 8x8 "
                         "matrix");
    }
    if (getEltType() != LdStMatrixEltType::B16) {
      return emitOpError("expected element type to be b16 for 8x8 matrix");
    }
  } else if (m == 8 && n == 16) {
    if (num != 1 && num != 2 && num != 4) {
      return emitOpError("expected num attribute to be 1, 2 or 4 for 8x16 "
                         "matrix");
    }
    if (getLayout() != MMALayout::row) {
      return emitOpError("expected layout to be row for 8x16 matrix");
    }
    if (getEltType() != LdStMatrixEltType::B8X16_B4X16_P64 &&
        getEltType() != LdStMatrixEltType::B8X16_B6X16_P32 &&
        getEltType() != LdStMatrixEltType::S8_S4) {
      return emitOpError("expected element type to be b8x16.b4x16_p64, "
                         "b8x16.b6x16_p32 or s8.s4 for 8x16 matrix");
    }
  } else if (m == 16 && n == 16) {
    if (num != 1 && num != 2) {
      return emitOpError("expected num attribute to be 1 or 2 for 16x16 "
                         "matrix");
    }
    if (getLayout() != MMALayout::col) {
      return emitOpError("expected layout to be col for 16x16 matrix");
    }
    if (getEltType() != LdStMatrixEltType::B8 &&
        getEltType() != LdStMatrixEltType::B8X16_B4X16_P64 &&
        getEltType() != LdStMatrixEltType::B8X16_B6X16_P32) {
      return emitOpError("expected element type to be b8, b8x16.b4x16_p64 or "
                         "b8x16.b6x16_p32 for 16x16 matrix");
    }
  } else {
    return emitOpError("expected shape to be 8x8, 8x16 or 16x16");
  }

  Type i32 = IntegerType::get(getContext(), 32);
  uint32_t numElements = (m == 16 && n == 16 ? num * 2 : num);
  if (numElements == 1 && getType() != i32)
    return emitOpError("expected destination type is i32");
  if (numElements == 2 || numElements == 4) {
    Type dstType = LLVM::LLVMStructType::getLiteral(
        getContext(), SmallVector<Type>(numElements, i32));
    if (getType() != dstType)
      return emitOpError("expected destination type is a structure of ")
             << numElements << " elements of type i32";
  }

  return success();
}

LogicalResult LdMatrixOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    LdMatrixOp::Adaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {
  uint32_t num = adaptor.getNum();
  uint32_t m = adaptor.getShape().getM();
  uint32_t n = adaptor.getShape().getN();
  uint32_t numElements = (m == 16 && n == 16) ? num * 2 : num;

  Type i32 = IntegerType::get(context, 32);
  if (numElements == 1)
    inferredReturnTypes.push_back(i32);
  else
    inferredReturnTypes.push_back(LLVM::LLVMStructType::getLiteral(
        context, SmallVector<Type>(numElements, i32)));
  return success();
}

LogicalResult NVVM::StMatrixOp::verify() {
  int numMatrix = getSources().size();
  if (numMatrix != 1 && numMatrix != 2 && numMatrix != 4)
    return emitOpError("expected num attribute to be 1, 2 or 4");

  int m = getShape().getM(), n = getShape().getN();
  if (m == 8 && n == 8) {
    if (getEltType() != NVVM::LdStMatrixEltType::B16) {
      return emitOpError("expected element type to be B16 for 8x8 matrix");
    }
  } else if (m == 16 && n == 8) {
    if (getEltType() != NVVM::LdStMatrixEltType::B8) {
      return emitOpError("expected element type to be B8 for 16x8 matrix");
    }
    if (getLayout() != NVVM::MMALayout::col) {
      return emitOpError("expected layout to be col for 16x8 matrix");
    }
  } else {
    return emitOpError("expected shape to be 8x8 or 16x8");
  }

  return success();
}

LogicalResult NVVM::MovMatrixOp::verify() {
  int m = getShape().getM(), n = getShape().getN();
  if (m != 8 || n != 8)
    return emitOpError("expected shape to be 8x8");
  if (getLayout() != NVVM::MMALayout::col)
    return emitOpError("expected layout to be col");
  if (getEltType() != NVVM::LdStMatrixEltType::B16)
    return emitOpError("expected element type to be b16");
  return success();
}

static FailureOr<int> getAllowedSizeK(NVVM::WGMMATypes typeA) {
  if (typeA == NVVM::WGMMATypes::tf32)
    return 8;
  if (typeA == NVVM::WGMMATypes::f16 || typeA == NVVM::WGMMATypes::bf16)
    return 16;
  if (typeA == NVVM::WGMMATypes::s8 || typeA == NVVM::WGMMATypes::u8)
    return 32;
  if (typeA == NVVM::WGMMATypes::e4m3 || typeA == NVVM::WGMMATypes::e5m2)
    return 32;
  if (typeA == NVVM::WGMMATypes::b1)
    return 256;
  return failure();
}

static LogicalResult isAllowedWGMMADataType(NVVM::WGMMATypes typeD,
                                            NVVM::WGMMATypes typeA,
                                            NVVM::WGMMATypes typeB) {
  switch (typeA) {
  case NVVM::WGMMATypes::f16:
    if ((typeD == NVVM::WGMMATypes::f32 || typeD == NVVM::WGMMATypes::f16) &&
        typeB == NVVM::WGMMATypes::f16)
      return success();
    break;
  case NVVM::WGMMATypes::tf32:
    if (typeD == NVVM::WGMMATypes::f32 && typeB == NVVM::WGMMATypes::tf32)
      return success();
    break;
  case NVVM::WGMMATypes::u8:
  case NVVM::WGMMATypes::s8:
    if (typeD == NVVM::WGMMATypes::s32 &&
        (typeB == NVVM::WGMMATypes::u8 || typeB == NVVM::WGMMATypes::s8))
      return success();
    break;
  case NVVM::WGMMATypes::b1:
    if (typeD == NVVM::WGMMATypes::s32 && typeB == NVVM::WGMMATypes::b1)
      return success();
    break;
  case NVVM::WGMMATypes::bf16:
    if (typeD == NVVM::WGMMATypes::f32 && typeB == NVVM::WGMMATypes::bf16)
      return success();
    break;
  case NVVM::WGMMATypes::e4m3:
  case NVVM::WGMMATypes::e5m2:
    if ((typeD == NVVM::WGMMATypes::f32 || typeD == NVVM::WGMMATypes::f16) &&
        (typeB == NVVM::WGMMATypes::e5m2 || typeB == NVVM::WGMMATypes::e4m3))
      return success();
    break;
  case WGMMATypes::f32:
  case WGMMATypes::s32:
    llvm_unreachable("unsupported input types");
    break;
  }
  return failure();
}

static LogicalResult isAllowedSizeN(int sizeN, NVVM::WGMMATypes typeA) {
  SmallVector<int> allowedN = {8,   16,  24,  32,  40,  48,  56,  64,
                               72,  80,  88,  96,  104, 112, 120, 128,
                               136, 144, 152, 160, 168, 176, 184, 192,
                               200, 208, 216, 224, 232, 240, 248, 256};
  SmallVector<int> allowedNshort = {8,   16,  24,  32,  48,  64,
                                    80,  96,  112, 128, 144, 160,
                                    176, 192, 208, 224, 240, 256};
  switch (typeA) {
  case WGMMATypes::f16:
  case WGMMATypes::tf32:
  case WGMMATypes::bf16:
  case WGMMATypes::e4m3:
  case WGMMATypes::e5m2:
    if (llvm::is_contained(allowedN, sizeN))
      return success();
    break;
  case WGMMATypes::u8:
  case WGMMATypes::s8:
  case WGMMATypes::b1:
    if (llvm::is_contained(allowedNshort, sizeN))
      return success();
    break;
  case WGMMATypes::f32:
  case WGMMATypes::s32:
    llvm_unreachable("unsupported input types");
    break;
  }
  return failure();
}

LogicalResult NVVM::WgmmaMmaAsyncOp::verify() {
  Value outValue = getResults();
  auto stype = dyn_cast<LLVM::LLVMStructType>(outValue.getType());
  if (!stype)
    return emitOpError() << "expected results to be struct";
  int outputSize = stype.getBody().size();
  WGMMATypes typeD = getTypeD();
  WGMMATypes typeA = getTypeA();
  WGMMATypes typeB = getTypeB();

  for (Type t : stype.getBody()) {
    if (t != stype.getBody().front())
      return emitOpError()
             << "all elements in struct must be same type but there is " << t;
  }

  if (typeD != WGMMATypes::f32 && typeD != WGMMATypes::f16 &&
      typeD != WGMMATypes::s32) {
    return emitOpError() << "does not support the given output type " << typeD;
  }
  if (typeD == WGMMATypes::s32 &&
      (getScaleA() == WGMMAScaleIn::neg || getScaleB() == WGMMAScaleIn::neg)) {
    return emitOpError() << "has s32 output, scaleA and scaleB cannot be neg";
  }

  if (failed(isAllowedWGMMADataType(typeD, typeA, typeB))) {
    return emitOpError() << typeD << " += " << typeA << " * " << typeB
                         << ", it is not supported.";
  }

  // Check M
  if (getShape().getM() != 64)
    return emitOpError() << "shape 'm' must be 64";

  // Check K
  FailureOr<int> allowedK = getAllowedSizeK(typeA);
  if (failed(allowedK) || allowedK.value() != getShape().getK())
    return emitOpError() << "shape 'k' must be " << allowedK.value()
                         << " for input type " << typeA;

  // Check N
  if (failed(isAllowedSizeN(getShape().getN(), typeA))) {
    return emitOpError() << "has input type " << typeA << " n is set to "
                         << getShape().getN() << ", it is not supported.";
  }

  // Check transpose (only available for f16/bf16)
  // Matrices A should be stored in row-major and B in column-major.
  // Only f16/bf16 matrices can be stored in either column-major or row-major
  // by setting the transpose value(imm-trans-a,imm-trans-b) in PTX code.
  if ((typeA != WGMMATypes::f16 && typeA != WGMMATypes::bf16) &&
      (getLayoutA() == mlir::NVVM::MMALayout::col ||
       getLayoutB() == mlir::NVVM::MMALayout::row)) {
    return emitOpError()
           << "given layouts layout_a = " << getLayoutA()
           << " and layout_b = " << getLayoutB() << " for input types " << typeA
           << " and " << typeB
           << " requires transpose. However, this is only supported for: "
           << MMATypes::f16 << " and " << MMATypes::bf16;
  }

  // Check result registers
  int expectedOutput = 0;
  if (typeD == WGMMATypes::f32 || typeD == WGMMATypes::s32)
    expectedOutput = getShape().getN() / 2;
  if (typeD == WGMMATypes::f16)
    expectedOutput = getShape().getN() / 4;
  if (outputSize != expectedOutput) {
    return emitOpError() << "results " << expectedOutput
                         << ", however output struct has " << outputSize
                         << " elements";
  }
  // Check satfinite (only available for s32 accumulator)
  if (typeD != WGMMATypes::s32 &&
      getSatfinite().value_or(NVVM::MMAIntOverflow::wrapped) ==
          NVVM::MMAIntOverflow::satfinite) {
    return emitOpError()
           << " `satfinite` can be only used with s32 accumulator, however "
              "the current accumulator is "
           << typeD;
  }

  return success();
}

std::string NVVM::WgmmaMmaAsyncOp::getPtx() {

  int m = getShape().getM(), n = getShape().getN(), k = getShape().getK();
  bool isF16 = getTypeA() == WGMMATypes::f16 || getTypeA() == WGMMATypes::bf16;

  StringRef outputTypeName = stringifyWGMMATypes(getTypeD());

  int expectedOutputRegisters = 0;
  if (getTypeD() == WGMMATypes::f16)
    expectedOutputRegisters = getShape().getN() / 4;
  else
    expectedOutputRegisters = getShape().getN() / 2;

  std::string ptx;
  llvm::raw_string_ostream ss(ptx);

  ss << "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, $"
     << ((expectedOutputRegisters * 2) + 2)
     << ", 0;\n"
        "wgmma.mma_async.sync.aligned.m"
     << m << "n" << n << "k" << k << "." << outputTypeName << "." << getTypeA()
     << "." << getTypeB();
  if (getSatfinite().value_or(NVVM::MMAIntOverflow::wrapped) ==
      NVVM::MMAIntOverflow::satfinite)
    ss << ".satfinite";
  ss << " {";
  int regCnt = 0;
  for (; regCnt < expectedOutputRegisters; ++regCnt) {
    ss << "$" << regCnt;
    if (regCnt != expectedOutputRegisters - 1)
      ss << ", ";
  }

  ss << "},";
  // Need to map read/write registers correctly.
  regCnt = (regCnt * 2);
  ss << " $" << (regCnt) << "," << " $" << (regCnt + 1) << "," << " p";
  if (getTypeD() != WGMMATypes::s32) {
    ss << ", $" << (regCnt + 3) << ",  $" << (regCnt + 4);
  }
  // Don't add transpose parameters unless needed.
  if (isF16) {
    ss << ", $" << (regCnt + 5) << ",  $" << (regCnt + 6);
  }
  ss << ";\n"
     << "}\n";
  return ptx;
}

bool NVVM::WgmmaMmaAsyncOp::getAsmValues(
    RewriterBase &rewriter,
    llvm::SmallVectorImpl<std::pair<mlir::Value, mlir::NVVM::PTXRegisterMod>>
        &asmValues) {
  bool isF16 = getTypeA() == WGMMATypes::f16 || getTypeA() == WGMMATypes::bf16;
  if (getResults())
    asmValues.push_back({getResults(), mlir::NVVM::PTXRegisterMod::Write});
  if (getInouts())
    asmValues.push_back({getInouts(), mlir::NVVM::PTXRegisterMod::ReadWrite});
  asmValues.push_back({getDescriptorA(), mlir::NVVM::PTXRegisterMod::Read});
  asmValues.push_back({getDescriptorB(), mlir::NVVM::PTXRegisterMod::Read});
  asmValues.push_back({makeConstantI32(rewriter, static_cast<int>(getScaleD())),
                       mlir::NVVM::PTXRegisterMod::Read});
  if (getTypeD() != WGMMATypes::s32) {
    asmValues.push_back(
        {makeConstantI32(rewriter,
                         getScaleA() == NVVM::WGMMAScaleIn::neg ? -1 : 1),
         mlir::NVVM::PTXRegisterMod::Read});
    asmValues.push_back(
        {makeConstantI32(rewriter,
                         getScaleB() == NVVM::WGMMAScaleIn::neg ? -1 : 1),
         mlir::NVVM::PTXRegisterMod::Read});
  }
  if (isF16) {
    asmValues.push_back(
        {makeConstantI32(rewriter, static_cast<int>(getLayoutA())),
         mlir::NVVM::PTXRegisterMod::Read});
    asmValues.push_back(
        {makeConstantI32(rewriter, 1 - static_cast<int>(getLayoutB())),
         mlir::NVVM::PTXRegisterMod::Read});
  }
  return true; // Has manual mapping
}

//===----------------------------------------------------------------------===//
// getIntrinsicID/getIntrinsicIDAndArgs methods
//===----------------------------------------------------------------------===//

mlir::NVVM::IDArgPair
MovMatrixOp::getIntrinsicIDAndArgs(Operation &op, LLVM::ModuleTranslation &mt,
                                   llvm::IRBuilderBase &builder) {
  auto thisOp = cast<NVVM::MovMatrixOp>(op);
  return {llvm::Intrinsic::nvvm_movmatrix_sync_aligned_m8n8_trans_b16,
          {mt.lookupValue(thisOp.getSrc())}};
}

#define GET_OP_CLASSES
#include "mlir/Dialect/LLVMIR/NVVMMmaOps.cpp.inc"
