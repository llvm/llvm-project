//===- TosaOps.cpp - MLIR Dialect for TOSA --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// This file implements the TOSA Specification:
// https://www.mlplatform.org/tosa/tosa_spec.html
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Shard/Interfaces/ShardingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/Dialect/Tosa/Utils/ShapeUtils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/TypeSwitch.h"

#include <numeric>

using namespace mlir;
using namespace mlir::tosa;

#include "mlir/Dialect/Tosa/IR/TosaOpsDialect.cpp.inc"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"

//===----------------------------------------------------------------------===//
// Tosa dialect interface includes.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Tosa/IR/TosaAvailability.cpp.inc"
#include "mlir/Dialect/Tosa/IR/TosaEnums.cpp.inc"
#include "mlir/Dialect/Tosa/IR/TosaInterfaces.cpp.inc"
#include "mlir/Dialect/Tosa/IR/TosaOpAvailabilityImpl.inc"

namespace {
#include "mlir/Dialect/Tosa/IR/TosaDialectBytecode.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect Function Inliner Interface.
//===----------------------------------------------------------------------===//
struct TosaInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks.
  //===--------------------------------------------------------------------===//

  /// All operations can be inlined by default.
  bool isLegalToInline(Operation *op, Region *region, bool wouldBeCloned,
                       IRMapping &map) const final {
    return true;
  }

  /// All regions with If and While parent operators can be inlined.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &map) const final {
    return (isa<tosa::IfOp>(dest->getParentOp()) ||
            isa<tosa::WhileOp>(dest->getParentOp()));
  }
};

/// This class implements the bytecode interface for the Tosa dialect.
struct TosaDialectBytecodeInterface : public BytecodeDialectInterface {
  TosaDialectBytecodeInterface(Dialect *dialect)
      : BytecodeDialectInterface(dialect) {}

  //===--------------------------------------------------------------------===//
  // Attributes

  Attribute readAttribute(DialectBytecodeReader &reader) const override {
    return ::readAttribute(getContext(), reader);
  }

  LogicalResult writeAttribute(Attribute attr,
                               DialectBytecodeWriter &writer) const override {
    return ::writeAttribute(attr, writer);
  }

  //===--------------------------------------------------------------------===//
  // Types

  Type readType(DialectBytecodeReader &reader) const override {
    return ::readType(getContext(), reader);
  }

  LogicalResult writeType(Type type,
                          DialectBytecodeWriter &writer) const override {
    return ::writeType(type, writer);
  }

  void writeVersion(DialectBytecodeWriter &writer) const final {
    // TODO: Populate.
  }

  std::unique_ptr<DialectVersion>
  readVersion(DialectBytecodeReader &reader) const final {
    // TODO: Populate
    reader.emitError("Dialect does not support versioning");
    return nullptr;
  }

  LogicalResult upgradeFromVersion(Operation *topLevelOp,
                                   const DialectVersion &version) const final {
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// TOSA control flow support.
//===----------------------------------------------------------------------===//

/// Returns the while loop body.
SmallVector<Region *> tosa::WhileOp::getLoopRegions() {
  return {&getBodyGraph()};
}

//===----------------------------------------------------------------------===//
// TOSA variable operator support.
//===----------------------------------------------------------------------===//

static SmallVector<int64_t> convertToMlirShape(ArrayRef<int64_t> shape) {
  return to_vector(llvm::map_range(shape, [](int64_t dim) {
    return dim == -1 ? ShapedType::kDynamic : dim;
  }));
}

// returns type of variable op
RankedTensorType mlir::tosa::getVariableType(tosa::VariableOp variableOp) {
  Type elementType = variableOp.getType();
  DenseIntElementsAttr varShapeAttr = variableOp.getVarShape();
  auto shape = convertToMlirShape(to_vector(varShapeAttr.getValues<int64_t>()));
  return RankedTensorType::get(shape, elementType);
}

//===----------------------------------------------------------------------===//
// Tosa dialect initialization.
//===----------------------------------------------------------------------===//

void TosaDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Tosa/IR/TosaOpsTypesBase.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Tosa/IR/TosaOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Tosa/IR/TosaAttributes.cpp.inc"
      >();
  addInterfaces<TosaDialectBytecodeInterface, TosaInlinerInterface>();
  declarePromisedInterfaces<
      shard::ShardingInterface, ClampOp, SigmoidOp, TanhOp, AddOp,
      ArithmeticRightShiftOp, BitwiseAndOp, BitwiseOrOp, BitwiseXorOp, IntDivOp,
      LogicalAndOp, LogicalLeftShiftOp, LogicalRightShiftOp, LogicalOrOp,
      LogicalXorOp, MaximumOp, MinimumOp, MulOp, PowOp, SubOp, AbsOp,
      BitwiseNotOp, CeilOp, ClzOp, ExpOp, FloorOp, LogOp, LogicalNotOp,
      NegateOp, ReciprocalOp, RsqrtOp, SelectOp, EqualOp, GreaterOp,
      GreaterEqualOp, MatMulOp>();
}

Operation *TosaDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  // Tosa dialect constants only support ElementsAttr unlike standard dialect
  // constant which supports all attributes.
  if (llvm::isa<shapeType>(type) && llvm::isa<DenseIntElementsAttr>(value)) {
    return tosa::ConstShapeOp::create(builder, loc, type,
                                      llvm::cast<DenseIntElementsAttr>(value));
  }
  if (llvm::isa<ElementsAttr>(value))
    return tosa::ConstOp::create(builder, loc, type,
                                 llvm::cast<ElementsAttr>(value));
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Parsers and printers
//===----------------------------------------------------------------------===//

namespace {

ParseResult getShapeAndElementType(OpAsmParser &parser, Type parsedType,
                                   DenseElementsAttr &varShapeAttr,
                                   TypeAttr &typeAttr) {
  if (auto shapedType = dyn_cast<ShapedType>(parsedType)) {
    if (!shapedType.hasRank())
      return parser.emitError(parser.getCurrentLocation())
             << "expected ranked type";

    auto elementType = shapedType.getElementType();
    typeAttr = TypeAttr::get(elementType);
    ArrayRef<int64_t> shape = shapedType.getShape();
    Builder builder(parser.getContext());
    varShapeAttr = builder.getIndexTensorAttr(convertFromMlirShape(shape));
    return success();
  }
  return parser.emitError(parser.getCurrentLocation())
         << "expected shaped type";
}

} // namespace

// parses the optional initial value or type for a tosa variable
//  with initial value:
//    tosa.variable @name = dense<0.0> : tensor<1x8xf32>
//
//  without initial value:
//    tosa.variable @name : tensor<1x8xf32>
ParseResult mlir::tosa::parseVariableOpTypeOrInitialValue(
    OpAsmParser &parser, DenseElementsAttr &varShapeAttr, TypeAttr &typeAttr,
    Attribute &initialValueAttr) {
  if (succeeded(parser.parseOptionalEqual())) {
    if (failed(parser.parseAttribute(initialValueAttr))) {
      return parser.emitError(parser.getCurrentLocation())
             << "expected attribute";
    }
    if (auto typedAttr = dyn_cast<TypedAttr>(initialValueAttr)) {
      return getShapeAndElementType(parser, typedAttr.getType(), varShapeAttr,
                                    typeAttr);
    }
    return parser.emitError(parser.getCurrentLocation())
           << "expected Typed attr";
  }

  initialValueAttr = nullptr;
  Type parsedType;
  if (failed(parser.parseColonType(parsedType))) {
    return parser.emitError(parser.getCurrentLocation())
           << "expected type after colon";
  }
  return getShapeAndElementType(parser, parsedType, varShapeAttr, typeAttr);
}

void mlir::tosa::printVariableOpTypeOrInitialValue(
    OpAsmPrinter &p, Operation *op, DenseElementsAttr varShapeAttr,
    TypeAttr typeAttr, Attribute initialValueAttr) {
  bool needsSpace = false;
  if (!dyn_cast_or_null<TypedAttr>(initialValueAttr)) {
    auto shape =
        convertToMlirShape(to_vector(varShapeAttr.getValues<int64_t>()));
    Type elementType = typeAttr.getValue();
    RankedTensorType tensorType =
        RankedTensorType::get(ArrayRef<int64_t>(shape), elementType);
    auto tensorTypeAttr = TypeAttr::get(tensorType);
    p << ": ";
    p.printAttribute(tensorTypeAttr);
    needsSpace = true; // subsequent attr value needs a space separator
  }
  if (initialValueAttr) {
    if (needsSpace)
      p << ' ';
    p << "= ";
    p.printAttribute(initialValueAttr);
  }
}

namespace {

// parse attributes with special handling for tosa enum attributes
template <typename EnumType>
ParseResult parseAttrEntryWithEnumHandling(OpAsmParser &parser,
                                           NamedAttrList &outAttrs) {
  llvm::StringRef name;
  if (parser.parseOptionalKeyword(&name) || parser.parseEqual())
    return failure();

  // special handling: rounding_mode accepts a *bare* RoundingMode enum
  // keyword.
  llvm::StringRef kw;
  if constexpr (std::is_same_v<EnumType, tosa::RoundingMode>) {
    if (name == "rounding_mode" &&
        succeeded(parser.parseOptionalKeyword(&kw))) {
      auto sym = symbolizeRoundingMode(kw);
      if (!sym)
        return parser.emitError(parser.getCurrentLocation())
               << "invalid rounding_mode value: " << kw;
      auto attr = RoundingModeAttr::get(parser.getContext(), sym.value());
      outAttrs.push_back(NamedAttribute(name, attr));
      return success();
    }
  }
  // special handling: mode accepts a *bare* ResizeMode enum keyword.
  if constexpr (std::is_same_v<EnumType, tosa::ResizeMode>) {
    if (name == "mode" && succeeded(parser.parseOptionalKeyword(&kw))) {
      auto sym = symbolizeResizeMode(kw);
      if (!sym)
        return parser.emitError(parser.getCurrentLocation())
               << "invalid resize mode value: " << kw;
      auto attr = ResizeModeAttr::get(parser.getContext(), sym.value());
      outAttrs.push_back(NamedAttribute(name, attr));
      return success();
    }
  }
  // special handling: nan_mode accepts a *bare* NanPropagationMode enum
  // keyword.
  if constexpr (std::is_same_v<EnumType, tosa::NanPropagationMode>) {
    if (name == "nan_mode" && succeeded(parser.parseOptionalKeyword(&kw))) {
      auto sym = symbolizeNanPropagationMode(kw);
      if (!sym)
        return parser.emitError(parser.getCurrentLocation())
               << "invalid nan_mode value: " << kw;
      auto attr = NanPropagationModeAttr::get(parser.getContext(), sym.value());
      outAttrs.push_back(NamedAttribute(name, attr));
      return success();
    }
  }

  // Default path: parse any normal attribute literal, including fully qualified
  // enum keyword
  Attribute attr;
  return parser.parseAttribute(attr, name, outAttrs);
}

template <typename EnumType>
ParseResult parseWithEnumHandling(OpAsmParser &parser, OperationState &result) {
  // parse operands
  SmallVector<OpAsmParser::UnresolvedOperand, 5> operands;
  if (parser.parseCommaSeparatedList(
          [&]() { return parser.parseOperand(operands.emplace_back()); }))
    return failure();

  // Parse { attr-dict } with special handling for enum bare token
  NamedAttrList attrs;
  if (succeeded(parser.parseOptionalLBrace()) &&
      failed(parser.parseOptionalRBrace())) {
    do {
      if (parseAttrEntryWithEnumHandling<EnumType>(parser, attrs))
        return failure();
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRBrace())
      return failure();
  }

  FunctionType fnTy;
  if (parser.parseColonType(fnTy))
    return failure();

  // Resolve operands and types
  if (failed(parser.resolveOperands(operands, fnTy.getInputs(),
                                    parser.getCurrentLocation(),
                                    result.operands)))
    return failure();

  result.addTypes(fnTy.getResult(0));
  result.addAttributes(attrs);

  return success();
}

void printNamedAttr(OpAsmPrinter &parser, const NamedAttribute namedAttr) {
  parser << namedAttr.getName().strref() << " = ";
  auto attr = namedAttr.getValue();
  if (auto roundingModeAttr = dyn_cast<tosa::RoundingModeAttr>(attr)) {
    parser << roundingModeAttr.getValue();
  } else if (auto resizeModeAttr = dyn_cast<tosa::ResizeModeAttr>(attr)) {
    parser << resizeModeAttr.getValue();
  } else if (auto nanPropagationModeAttr =
                 dyn_cast<tosa::NanPropagationModeAttr>(attr)) {
    parser << nanPropagationModeAttr.getValue();
  } else {
    parser.printAttribute(attr);
  }
}

// print with special handling for default valued NanPropagationMode attribute
void printWithNanPropagationHandling(OpAsmPrinter &parser, Operation *op) {
  parser << " ";
  parser.printOperands(op->getOperands());

  NamedAttrList toPrint(op->getAttrs());
  // remove default NanPropagate attribute
  const auto kDefaultNanValue = NanPropagationMode::PROPAGATE;
  for (auto attr : op->getAttrs()) {
    if (auto nanAttr = dyn_cast<NanPropagationModeAttr>(attr.getValue())) {
      if (nanAttr.getValue() == kDefaultNanValue) {
        // elide from toPrint
        toPrint.erase(attr.getName());
        break;
      }
    }
  }

  if (!toPrint.empty()) {
    parser << " {";
    llvm::interleaveComma(toPrint, parser, [&](const NamedAttribute namedAttr) {
      printNamedAttr(parser, namedAttr);
    });
    parser << "}";
  }

  parser << " : ";
  parser.printFunctionalType(op);
}

// print with special handling for enums: RoundingMode, ResizeMode
void printWithEnumHandling(OpAsmPrinter &parser, Operation *op) {
  parser << " ";
  parser.printOperands(op->getOperands());

  if (!op->getAttrs().empty()) {
    parser << " {";
    llvm::interleaveComma(op->getAttrs(), parser,
                          [&](const NamedAttribute namedAttr) {
                            printNamedAttr(parser, namedAttr);
                          });
    parser << "}";
  }

  parser << " : ";
  parser.printFunctionalType(op);
}

} // namespace

ParseResult RescaleOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseWithEnumHandling<tosa::RoundingMode>(parser, result);
}

void RescaleOp::print(OpAsmPrinter &parser) {
  printWithEnumHandling(parser, *this);
}

ParseResult ApplyScaleOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseWithEnumHandling<tosa::RoundingMode>(parser, result);
}

void ApplyScaleOp::print(OpAsmPrinter &parser) {
  printWithEnumHandling(parser, *this);
}

ParseResult ResizeOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseWithEnumHandling<tosa::ResizeMode>(parser, result);
}

void ResizeOp::print(OpAsmPrinter &parser) {
  printWithEnumHandling(parser, *this);
}

ParseResult ArgMaxOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseWithEnumHandling<tosa::NanPropagationMode>(parser, result);
}

void ArgMaxOp::print(OpAsmPrinter &parser) {
  printWithNanPropagationHandling(parser, *this);
}

ParseResult MaxPool2dOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseWithEnumHandling<tosa::NanPropagationMode>(parser, result);
}

void MaxPool2dOp::print(OpAsmPrinter &parser) {
  printWithNanPropagationHandling(parser, *this);
}

ParseResult ClampOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseWithEnumHandling<tosa::NanPropagationMode>(parser, result);
}

void ClampOp::print(OpAsmPrinter &parser) {
  printWithNanPropagationHandling(parser, *this);
}

ParseResult MaximumOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseWithEnumHandling<tosa::NanPropagationMode>(parser, result);
}

void MaximumOp::print(OpAsmPrinter &parser) {
  printWithNanPropagationHandling(parser, *this);
}

ParseResult MinimumOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseWithEnumHandling<tosa::NanPropagationMode>(parser, result);
}

void MinimumOp::print(OpAsmPrinter &parser) {
  printWithNanPropagationHandling(parser, *this);
}

ParseResult ReduceMaxOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseWithEnumHandling<tosa::NanPropagationMode>(parser, result);
}

void ReduceMaxOp::print(OpAsmPrinter &parser) {
  printWithNanPropagationHandling(parser, *this);
}

ParseResult ReduceMinOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseWithEnumHandling<tosa::NanPropagationMode>(parser, result);
}

void ReduceMinOp::print(OpAsmPrinter &parser) {
  printWithNanPropagationHandling(parser, *this);
}

//===----------------------------------------------------------------------===//
// Tosa utilities.
//===----------------------------------------------------------------------===//

static std::optional<int64_t> idivCheck(const int64_t lhs, const int64_t rhs) {
  if (lhs % rhs != 0)
    return std::nullopt;
  return lhs / rhs;
}

static Type getStorageElementTypeOrSelf(Type type) {
  auto srcType = getElementTypeOrSelf(type);
  if (auto quantType = llvm::dyn_cast<mlir::quant::QuantizedType>(srcType))
    srcType = quantType.getStorageType();
  return srcType;
}

static Type getStorageElementTypeOrSelf(Value value) {
  return getStorageElementTypeOrSelf(value.getType());
}

static LogicalResult verifyRescaleValueAndZpTypes(Operation *op, Value val,
                                                  Value valZp, StringRef name) {
  Type eType = getStorageElementTypeOrSelf(val.getType());
  Type eZpType = getStorageElementTypeOrSelf(valZp.getType());

  bool bothInts =
      mlir::isa<IntegerType>(eType) && mlir::isa<IntegerType>(eZpType);
  bool sameBitWidth =
      (eType.getIntOrFloatBitWidth() == eZpType.getIntOrFloatBitWidth());

  if (!bothInts || !sameBitWidth) {
    return op->emitOpError()
           << "expected " << name << " and " << name
           << "_zp to both be integer of the same bitwidth, but got " << eType
           << " vs. " << eZpType;
  }
  return success();
}

// Create a pad-const const tensor with value of `val` of required data-type
Value mlir::tosa::createPadConstTensor(OpBuilder &builder, Location loc,
                                       Value src, int32_t val) {
  const auto srcType = getElementTypeOrSelf(src);
  const auto srcElemType = getStorageElementTypeOrSelf(src);
  const auto padConstType = mlir::RankedTensorType::get({1}, srcType);
  const auto padConstEType = mlir::RankedTensorType::get({1}, srcElemType);
  const auto padConstAttr{
      llvm::isa<FloatType>(srcElemType)
          ? DenseElementsAttr::get(padConstEType,
                                   builder.getFloatAttr(srcElemType, val))
          : DenseElementsAttr::get(padConstEType,
                                   builder.getIntegerAttr(srcElemType, val))};
  return tosa::ConstOp::create(builder, loc, padConstType, padConstAttr);
}

//===----------------------------------------------------------------------===//
// TOSA Operator Verifiers.
//===----------------------------------------------------------------------===//

template <typename T>
static LogicalResult verifyConvOp(T op) {
  const auto inputType = llvm::dyn_cast<TensorType>(op.getInput().getType());
  const auto weightType = llvm::dyn_cast<TensorType>(op.getWeight().getType());

  auto inputEType = inputType.getElementType();
  auto weightEType = weightType.getElementType();
  auto biasEType =
      llvm::cast<ShapedType>(op.getBias().getType()).getElementType();
  auto resultEType =
      llvm::cast<ShapedType>(op.getResult().getType()).getElementType();
  bool biasIsFloat = llvm::isa<FloatType>(biasEType);
  bool resultIsFloat = llvm::isa<FloatType>(resultEType);

  if (auto quantType = llvm::dyn_cast<mlir::quant::QuantizedType>(inputEType))
    inputEType = quantType.getStorageType();

  if (auto quantType = llvm::dyn_cast<mlir::quant::QuantizedType>(weightEType))
    weightEType = quantType.getStorageType();

  if (auto quantType = llvm::dyn_cast<mlir::quant::QuantizedType>(biasEType))
    biasEType = quantType.getStorageType();

  if (auto quantType = llvm::dyn_cast<mlir::quant::QuantizedType>(resultEType))
    resultEType = quantType.getStorageType();

  if (biasIsFloat && resultIsFloat && (biasEType != resultEType)) {
    // for now, only enforce bias element type == result element type for
    // float types.
    op.emitOpError(
        "expect both bias and result to have same element type, got ")
        << biasEType << " and " << resultEType;
    return failure();
  }

  if (isa<Float8E5M2Type>(inputEType) || isa<Float8E4M3FNType>(inputEType) ||
      isa<Float8E5M2Type>(weightEType) || isa<Float8E4M3FNType>(weightEType)) {
    if (inputEType != weightEType) {
      op.emitOpError(
          "expect both input and weight to have same element type, got ")
          << inputEType << " and " << weightEType;
      return failure();
    }
  }

  bool inputIsFloat = llvm::isa<FloatType>(inputEType);
  bool weightIsFloat = llvm::isa<FloatType>(weightEType);

  // Either both must be float or both non-float.
  if (inputIsFloat != weightIsFloat) {
    op.emitOpError(
        "expect both input and weight to be float or not together, got ")
        << inputEType << " and " << weightEType;
    return failure();
  }

  auto inputZpEType = getStorageElementTypeOrSelf(op.getInputZp().getType());
  if (inputEType != inputZpEType) {
    return op.emitOpError("expect both input and its zero point are the same "
                          "element type, got ")
           << inputEType << " and " << inputZpEType;
  }

  auto weightZpEType = getStorageElementTypeOrSelf(op.getWeightZp().getType());
  if (weightEType != weightZpEType) {
    return op.emitOpError("expect both weight and its zero point are the same "
                          "element type, got ")
           << weightEType << " and " << weightZpEType;
  }

  FailureOr<int64_t> maybeIZp = op.getInputZeroPoint();
  if (succeeded(maybeIZp) && op.verifyInputZeroPoint(*maybeIZp).failed())
    return failure();

  FailureOr<int64_t> maybeWZp = op.getWeightZeroPoint();
  if (succeeded(maybeWZp) && op.verifyWeightZeroPoint(*maybeWZp).failed())
    return failure();

  return success();
}

LogicalResult tosa::ConstOp::verify() {

  auto attrType = llvm::dyn_cast<TensorType>(getValuesAttr().getType());
  auto outputType = llvm::dyn_cast<TensorType>(getOutput().getType());

  if (!attrType || !outputType) {
    emitOpError("expected tensors for attr/result type");
    return failure();
  }

  if (auto result = llvm::dyn_cast<mlir::quant::QuantizedType>(
          outputType.getElementType())) {
    if (result.getStorageType() == attrType.getElementType())
      return success();
  }

  if (attrType.getElementType() != outputType.getElementType()) {
    emitOpError("expected same attr/result element types");
    return failure();
  }

  return success();
}

template <typename T>
static LogicalResult verifyConvOpModes(T op) {
  auto inputEType =
      llvm::cast<ShapedType>(op.getInput().getType()).getElementType();

  if (auto quantType = llvm::dyn_cast<mlir::quant::QuantizedType>(inputEType))
    inputEType = quantType.getStorageType();

  auto accType = op.getAccType();
  if (inputEType.isInteger(8) && !accType.isInteger(32))
    return op.emitOpError("accumulator type for i8 tensor is not i32");

  if (inputEType.isInteger(16) && !accType.isInteger(48))
    return op.emitOpError("accumulator type for i16 tensor is not i48");

  if (isa<Float8E5M2Type, Float8E4M3Type>(inputEType) && !accType.isF16())
    return op.emitOpError("accumulator type for f8 tensor is not f16");

  if (inputEType.isF16() && !(accType.isF16() || accType.isF32()))
    return op.emitOpError("accumulator type for f16 tensor is not f16/f32");

  if (inputEType.isBF16() && !accType.isF32())
    return op.emitOpError("accumulator type for bf16 tensor is not f32");

  if (inputEType.isF32() && !accType.isF32())
    return op.emitOpError("accumulator type for f32 tensor is not f32");

  auto resultEType =
      llvm::cast<ShapedType>(op.getResult().getType()).getElementType();

  if (auto quantType = llvm::dyn_cast<mlir::quant::QuantizedType>(resultEType))
    resultEType = quantType.getStorageType();

  return success();
}

//===----------------------------------------------------------------------===//
// ERROR_IF functions.
// ERROR_IF is a predicate that must set an error if the condition holds.
//===----------------------------------------------------------------------===//

template <typename T>
static LogicalResult verifyConvOpErrorIf(T op) {
  llvm::ArrayRef<int64_t> padding = op.getPad();
  if (llvm::any_of(padding, [](int64_t p) { return p < 0; }))
    return op.emitOpError("expect all padding values to be >= 0, got ")
           << padding;

  llvm::ArrayRef<int64_t> strides = op.getStride();
  if (llvm::any_of(strides, [](int64_t s) { return s < 1; }))
    return op.emitOpError("expect all stride values to be >= 1, got ")
           << strides;

  llvm::ArrayRef<int64_t> dilations = op.getDilation();
  if (llvm::any_of(dilations, [](int64_t d) { return d < 1; }))
    return op.emitOpError("expect all dilation values to be >= 1, got ")
           << dilations;

  const RankedTensorType outputType =
      llvm::dyn_cast<RankedTensorType>(op.getOutput().getType());
  if (!outputType)
    // Skip following checks if output is not ranked
    return success();

  const RankedTensorType inputType =
      llvm::dyn_cast<RankedTensorType>(op.getInput().getType());
  const RankedTensorType weightType =
      llvm::dyn_cast<RankedTensorType>(op.getWeight().getType());

  if (inputType && weightType) {
    const auto verifyOutputSize =
        [&op](const int64_t inputSize, const int64_t kernelSize,
              const int64_t outputSize, const int64_t padBefore,
              const int64_t padAfter, const int64_t stride,
              const int64_t dilation, const llvm::StringRef dimName,
              const llvm::StringRef dimAxis,
              const llvm::StringRef padBeforeName,
              const llvm::StringRef padAfterName) -> LogicalResult {
      if (inputSize == ShapedType::kDynamic ||
          kernelSize == ShapedType::kDynamic)
        return success();

      // ERROR_IF: O != idiv_check(I - 1 + pa + pb - (K - 1) * d, s) + 1

      const std::optional<int64_t> calculatedOutSizeMinusOne = idivCheck(
          inputSize - 1 + padBefore + padAfter - (kernelSize - 1) * dilation,
          stride);
      if (!calculatedOutSizeMinusOne.has_value())
        return op.emitOpError("expected input_")
               << dimName << " - 1 + pad_" << padBeforeName << " + pad_"
               << padAfterName << " - (kernel_" << dimName
               << " - 1) * dilation_" << dimAxis
               << " to be wholly divisible by stride_" << dimAxis << ", got ("
               << inputSize << " - 1 + " << padBefore << " + " << padAfter
               << " - (" << kernelSize << " - 1) * " << dilation << ") / "
               << stride;

      const int64_t calculatedOutSize = calculatedOutSizeMinusOne.value() + 1;
      if (outputSize != ShapedType::kDynamic && calculatedOutSize != outputSize)
        return op.emitOpError("calculated output ")
               << dimName << " did not match expected: "
               << "calculated=" << calculatedOutSize
               << ", expected=" << outputSize;

      return success();
    };

    // input = [_,IH,IW,_], weight = [_,KH,KW,_], output = [_,OH,OW,_]
    if constexpr (std::is_same<T, tosa::Conv2DOp>::value) {
      if (failed(verifyOutputSize(
              inputType.getDimSize(1), weightType.getDimSize(1),
              outputType.getDimSize(1), padding[0], padding[1], strides[0],
              dilations[0], "height", "y", "top", "bottom")))
        return failure();

      if (failed(verifyOutputSize(
              inputType.getDimSize(2), weightType.getDimSize(2),
              outputType.getDimSize(2), padding[2], padding[3], strides[1],
              dilations[1], "width", "x", "left", "right")))
        return failure();
    }

    // input = [_,IH,IW,_], weight = [KH,KW,_,_], output = [_,OH,OW,_]
    if constexpr (std::is_same<T, tosa::DepthwiseConv2DOp>::value) {
      if (failed(verifyOutputSize(
              inputType.getDimSize(1), weightType.getDimSize(0),
              outputType.getDimSize(1), padding[0], padding[1], strides[0],
              dilations[0], "height", "y", "top", "bottom")))
        return failure();

      if (failed(verifyOutputSize(
              inputType.getDimSize(2), weightType.getDimSize(1),
              outputType.getDimSize(2), padding[2], padding[3], strides[1],
              dilations[1], "width", "x", "left", "right")))
        return failure();
    }

    // input = [_,ID,IH,IW,_], weight = [_,KD,KH,KW,_], output = [_,OD,OH,OW,_]
    if constexpr (std::is_same<T, tosa::Conv3DOp>::value) {
      if (failed(verifyOutputSize(
              inputType.getDimSize(1), weightType.getDimSize(1),
              outputType.getDimSize(1), padding[0], padding[1], strides[0],
              dilations[0], "depth", "d", "front", "back")))
        return failure();

      if (failed(verifyOutputSize(
              inputType.getDimSize(2), weightType.getDimSize(2),
              outputType.getDimSize(2), padding[2], padding[3], strides[1],
              dilations[1], "height", "y", "top", "bottom")))
        return failure();

      if (failed(verifyOutputSize(
              inputType.getDimSize(3), weightType.getDimSize(3),
              outputType.getDimSize(3), padding[4], padding[5], strides[2],
              dilations[2], "width", "x", "left", "right")))
        return failure();
    }
  }

  const RankedTensorType biasType =
      llvm::dyn_cast<RankedTensorType>(op.getBias().getType());
  if (!biasType)
    // Skip following checks if bias is not ranked
    return success();

  const int64_t biasChannels = biasType.getDimSize(0);
  const int64_t outputChannels =
      outputType.getDimSize(outputType.getRank() - 1);
  if (biasChannels == ShapedType::kDynamic ||
      outputChannels == ShapedType::kDynamic)
    // Skip following checks if biasChannels or outputChannels is dynamic dim
    return success();

  if (biasChannels != outputChannels && biasChannels != 1)
    return op.emitOpError(
               "bias channels expected to be equal to output channels (")
           << outputChannels << ") or 1, got " << biasChannels;

  return success();
}

// Verify whether same type and shape of the given two types.
static LogicalResult errorIfTypeOrShapeMismatch(Operation *op, Type type1,
                                                StringRef name1, Type type2,
                                                StringRef name2) {
  auto shapeType1 = dyn_cast<ShapedType>(type1);
  auto shapeType2 = dyn_cast<ShapedType>(type2);
  if (!shapeType1 || !shapeType2)
    return failure();

  auto elemType1 = shapeType1.getElementType();
  auto elemType2 = shapeType2.getElementType();
  if (elemType1 != elemType2)
    return op->emitOpError()
           << "require same element type for " << name1 << " (" << elemType1
           << ") and " << name2 << " (" << elemType2 << ")";

  if (failed(verifyCompatibleShape(type1, type2)))
    return op->emitOpError()
           << "require same shapes for " << name1 << " (" << type1 << ") and "
           << name2 << " (" << type2 << ")";

  return success();
}

// Verify whether same length, type, and shape of the given two tensor lists.
static LogicalResult errorIfTypeOrShapeMismatch(Operation *op, ValueRange list1,
                                                StringRef name1,
                                                ValueRange list2,
                                                StringRef name2) {
  if (list1.size() != list2.size())
    return op->emitOpError()
           << "require same number of values in " << name1 << " ("
           << list1.size() << ") and " << name2 << " (" << list2.size() << ")";

  for (auto [type1, type2] :
       llvm::zip_equal(list1.getTypes(), list2.getTypes())) {
    if (errorIfTypeOrShapeMismatch(op, type1, name1, type2, name2).failed())
      return failure();
  }

  return success();
}

static inline LogicalResult errorIfShapeNotSizeOne(Operation *op, Type type) {
  ShapeAdaptor shapeAdaptor(type);
  if (!shapeAdaptor.hasRank() || !shapeAdaptor.hasStaticShape())
    return success();

  return shapeAdaptor.getNumElements() == 1 ? success() : failure();
}

// Returns the first declaration point prior to this operation or failure if
// not found.
static FailureOr<tosa::VariableOp> findVariableDecl(Operation *op,
                                                    StringRef symName) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  tosa::VariableOp varOp = nullptr;

  // TODO: Adopt SymbolTable trait to Varible ops.
  // Currently, the variable's definition point is searched via walk(),
  // starting from the top-level ModuleOp and stopping at the point of use. Once
  // TOSA control flow and variable extensions reach the complete state, may
  // leverage MLIR's Symbol Table functionality to look up symbol and enhance
  // the search to a TOSA specific graph traversal over the IR structure.
  module.walk([&](Operation *tempOp) {
    // Reach this op itself.
    if (tempOp == op) {
      return WalkResult::interrupt();
    }

    if (auto tosaOp = dyn_cast<tosa::VariableOp>(tempOp)) {
      if (symName == tosaOp.getName()) {
        varOp = tosaOp;
        return WalkResult::interrupt();
      }
    }

    return WalkResult::advance();
  });

  if (varOp)
    return varOp;

  return failure();
}

template <typename T>
static LogicalResult verifyVariableOpErrorIf(T op, Type type, StringRef name) {
  StringRef symName = op.getName();
  FailureOr<tosa::VariableOp> varOp = findVariableDecl(op, symName);
  if (failed(varOp))
    return op->emitOpError("'")
           << symName << "' has not been declared by 'tosa.variable'";

  // Verify type and shape
  auto variableType = getVariableType(varOp.value());
  if (errorIfTypeOrShapeMismatch(op, type, name, variableType,
                                 "the input tensor")
          .failed())
    return failure();

  return success();
}

// verify that inType and outType have same element types
template <typename T>
static LogicalResult verifySameElementTypes(T op, Type inType, Type outType) {
  auto inputType = llvm::dyn_cast<TensorType>(inType);
  auto outputType = llvm::dyn_cast<TensorType>(outType);
  if (!inputType) {
    op.emitOpError("expect shaped tensor for input, got ") << inType;
    return failure();
  }
  if (!outputType) {
    op.emitOpError("expect shaped tensor for output, got ") << outType;
    return failure();
  }
  auto inputElementType = inputType.getElementType();
  auto outputElementType = outputType.getElementType();
  auto inputQuantType =
      llvm::dyn_cast<mlir::quant::UniformQuantizedType>(inputElementType);
  auto outputQuantType =
      llvm::dyn_cast<mlir::quant::UniformQuantizedType>(outputElementType);
  if ((inputElementType.isIntOrIndexOrFloat() || inputQuantType) &&
      (outputElementType.isIntOrIndexOrFloat() || outputQuantType) &&
      inputElementType != outputElementType) {
    // only check if both element types are int/index/float/UniformQuantized
    // eg, not sure how to check quant::QuantizedType
    // this happens in test_conv2d_q_grouped_convolution in
    // tfl-to-tosa-pipeline.mlir
    op.emitOpError("expect input and output to have same element type, got ")
        << inputElementType << " and " << outputElementType;
    return failure();
  }
  return success();
}

LogicalResult tosa::ArgMaxOp::verify() {
  const ShapedType resultType = llvm::cast<ShapedType>(getType());

  // Ensure output is of 32-bit integer
  if (const auto resultETy = resultType.getElementType();
      !resultETy.isIntOrIndex())
    return emitOpError("result tensor is not of integer type");

  const auto inputType = llvm::cast<ShapedType>(getInput().getType());
  if (!inputType.hasRank())
    return success();

  // Ensure axis is within the tensor rank
  const int64_t axis = getAxisAttr().getInt();
  if (((axis < 0) || axis >= inputType.getRank()))
    return emitOpError("specified axis is outside the rank of the tensor");

  if (!resultType.hasRank())
    return success();

  const ArrayRef<int64_t> inputShape = inputType.getShape();
  const ArrayRef<int64_t> outputShape = resultType.getShape();
  llvm::SmallVector<int64_t> expectedOutputShape(inputShape);
  expectedOutputShape.erase(expectedOutputShape.begin() + axis);
  if (failed(verifyCompatibleShape(expectedOutputShape, outputShape)))
    return emitOpError("expected output shape '")
           << expectedOutputShape << "', got '" << outputShape << "'";

  return success();
}

template <typename T>
static LogicalResult verifyPoolingOp(T op) {
  const llvm::ArrayRef<int64_t> kernel = op.getKernel();
  if (llvm::any_of(kernel, [](int64_t s) { return s < 1; }))
    return op.emitOpError("expect all kernel values to be >= 1, got ")
           << kernel;

  const llvm::ArrayRef<int64_t> strides = op.getStride();
  if (llvm::any_of(strides, [](int64_t s) { return s < 1; }))
    return op.emitOpError("expect all stride values to be >= 1, got ")
           << strides;

  const llvm::ArrayRef<int64_t> padding = op.getPad();
  if (llvm::any_of(padding, [](int64_t p) { return p < 0; }))
    return op.emitOpError("expect all padding values to be >= 0, got ")
           << padding;

  // Padding must be less than kernel size to avoid a divide-by-zero
  const int64_t kernelX = kernel[1];
  const int64_t padLeft = padding[2];
  const int64_t padRight = padding[3];
  if (padRight >= kernelX || padLeft >= kernelX)
    return op.emitOpError("expected left/right padding to be less than the "
                          "width of the kernel, got pad_left=")
           << padLeft << ", pad_right=" << padRight << ", kernel_x=" << kernelX;

  const int64_t kernelY = kernel[0];
  const int64_t padTop = padding[0];
  const int64_t padBottom = padding[1];
  if (padTop >= kernelY || padBottom >= kernelY)
    return op.emitOpError("expected top/bottom padding to be less than the "
                          "height of the kernel, got pad_top=")
           << padTop << ", pad_bottom=" << padBottom
           << ", kernel_y=" << kernelY;

  const auto inputType =
      llvm::dyn_cast<RankedTensorType>(op.getInput().getType());
  const auto outputType =
      llvm::dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!inputType || !outputType)
    return success();

  const auto verifyOutputSize =
      [&op](const int64_t inputSize, const int64_t outputSize,
            const int64_t kernelSize, const int64_t strideSize,
            const int64_t padBefore, const int64_t padAfter,
            const llvm::StringRef dimName, const llvm::StringRef dimAxis,
            const llvm::StringRef padBeforeName,
            const llvm::StringRef padAfterName) -> LogicalResult {
    if (ShapedType::isDynamic(inputSize))
      return success();

    const std::optional<int64_t> calculatedOutSizeMinusOne =
        idivCheck(inputSize + padBefore + padAfter - kernelSize, strideSize);
    if (!calculatedOutSizeMinusOne.has_value())
      return op.emitOpError("expected input_")
             << dimName << " + pad_" << padBeforeName << " + pad_"
             << padAfterName << " - kernel_" << dimAxis
             << " to be wholly divisible by stride_" << dimAxis << ", got ("
             << inputSize << " + " << padBefore << " + " << padAfter << " - "
             << kernelSize << ") / " << strideSize;

    const int64_t calculatedOutSize = calculatedOutSizeMinusOne.value() + 1;
    if (ShapedType::isStatic(outputSize) && calculatedOutSize != outputSize)
      return op.emitOpError("calculated output ")
             << dimName << " did not match expected: "
             << "calculated=" << calculatedOutSize
             << ", expected=" << outputSize;

    return success();
  };

  if (failed(verifyOutputSize(inputType.getDimSize(1), outputType.getDimSize(1),
                              kernel[0], strides[0], padding[0], padding[1],
                              "height", "y", "top", "bottom")))
    return failure();

  if (failed(verifyOutputSize(inputType.getDimSize(2), outputType.getDimSize(2),
                              kernel[1], strides[1], padding[2], padding[3],
                              "width", "x", "left", "right")))
    return failure();

  return success();
}

LogicalResult tosa::AvgPool2dOp::verify() {
  if (failed(verifyPoolingOp(*this)))
    return failure();

  const Type inputETy = getStorageElementTypeOrSelf(getInput().getType());
  const Type resultETy = getStorageElementTypeOrSelf(getOutput().getType());
  const Type inputZpETy = getStorageElementTypeOrSelf(getInputZp().getType());
  const Type outputZpETy = getStorageElementTypeOrSelf(getOutputZp().getType());

  auto accType = getAccType();
  if (llvm::isa<IntegerType>(inputETy) && !accType.isInteger(32))
    return emitOpError("accumulator type for integer tensor is not i32");

  if (inputETy.isF16() && !(accType.isF16() || accType.isF32()))
    return emitOpError("accumulator type for f16 tensor is not f16/f32");

  if (inputETy.isBF16() && !accType.isF32())
    return emitOpError("accumulator type for bf16 tensor is not f32");

  if (inputETy.isF32() && !accType.isF32())
    return emitOpError("accumulator type for f32 tensor is not f32");

  if (inputETy != inputZpETy)
    return emitOpError("expect both input and its zero point are the same "
                       "element type, got ")
           << inputETy << " and " << inputZpETy;

  if (resultETy != outputZpETy)
    return emitOpError("expect both output and its zero point are the same "
                       "element type, got ")
           << resultETy << " and " << outputZpETy;

  FailureOr<int64_t> maybeIZp = getInputZeroPoint();
  if (succeeded(maybeIZp) && verifyInputZeroPoint(*maybeIZp).failed())
    return failure();

  FailureOr<int64_t> maybeOZp = getOutputZeroPoint();
  if (succeeded(maybeOZp) && verifyOutputZeroPoint(*maybeOZp).failed())
    return failure();

  return success();
}

LogicalResult tosa::ClampOp::verify() {
  mlir::Type inputETy =
      llvm::cast<ShapedType>(getInput().getType()).getElementType();
  if (auto quantType =
          llvm::dyn_cast<mlir::quant::UniformQuantizedType>(inputETy)) {
    inputETy = quantType.getStorageType();
  }
  mlir::Type outputETy =
      llvm::cast<ShapedType>(getOutput().getType()).getElementType();
  if (auto quantType =
          llvm::dyn_cast<mlir::quant::UniformQuantizedType>(outputETy)) {
    outputETy = quantType.getStorageType();
  }
  if (inputETy != outputETy)
    return emitOpError("input/output element types are incompatible.");

  auto maxValAttr = getMaxValAttr();
  auto minValAttr = getMinValAttr();

  unsigned dataTypeBitWidth = inputETy.getIntOrFloatBitWidth();

  if (inputETy.isInteger(dataTypeBitWidth)) {
    // if input datatype is integer, check that the min_val/max_val attributes
    // are integer attributes, and that their type is the same as the input's
    // datatype
    auto intMaxValAttr = mlir::dyn_cast<mlir::IntegerAttr>(maxValAttr);
    auto intMinValAttr = mlir::dyn_cast<mlir::IntegerAttr>(minValAttr);
    if (!intMaxValAttr || !intMinValAttr ||
        (intMaxValAttr.getType() != intMinValAttr.getType()) ||
        (intMaxValAttr.getType() != inputETy))
      return emitOpError("min/max attributes types are incompatible with "
                         "input/output element types.");

    const bool isUnsigned = inputETy.isUnsignedInteger();
    const bool isBoolean = inputETy.isInteger(1);
    const APInt minVal = intMinValAttr.getValue();
    const APInt maxVal = intMaxValAttr.getValue();
    if ((isUnsigned || isBoolean) ? maxVal.ult(minVal) : maxVal.slt(minVal))
      return emitOpError("expected min_val <= max_val, got min_val=")
             << minValAttr << ", max_val=" << maxValAttr;
  } else {
    // otherwise, input datatype is float, check that the min_val/max_val
    // attributes share the same type and that their type is the same as the
    // input's datatype
    auto floatMaxValAttr = mlir::dyn_cast<mlir::FloatAttr>(maxValAttr);
    auto floatMinValAttr = mlir::dyn_cast<mlir::FloatAttr>(minValAttr);
    if (!floatMaxValAttr || !floatMinValAttr ||
        (floatMaxValAttr.getType() != floatMinValAttr.getType()) ||
        (floatMaxValAttr.getType() != inputETy))
      return emitOpError("min/max attributes types are incompatible with "
                         "input/output element types.");

    const APFloat minVal = floatMinValAttr.getValue();
    const APFloat maxVal = floatMaxValAttr.getValue();
    if (minVal.isNaN() || maxVal.isNaN())
      return emitOpError("min/max attributes should not be 'NaN', got min_val=")
             << minValAttr << ", max_val=" << maxValAttr;

    if (maxVal < minVal)
      return emitOpError("expected min_val <= max_val, got min_val=")
             << minValAttr << ", max_val=" << maxValAttr;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TOSA Operator Quantization Builders.
//===----------------------------------------------------------------------===//

/// This builder is called on all convolution operators except TransposeConv,
/// which has specialized output shape semantics. The builder also defines the
/// bitwidth of the output given the bit width of the input & weight content.
static void buildConvOpWithQuantInfo(OpBuilder &builder, OperationState &result,
                                     Type outputType, Value input, Value weight,
                                     Value bias, DenseI64ArrayAttr pad,
                                     DenseI64ArrayAttr stride,
                                     DenseI64ArrayAttr dilation,
                                     TypeAttr accType) {
  auto zps = createZPsAsConst(builder, input, weight);
  result.addOperands({input, weight, bias, zps.first, zps.second});
  result.addAttribute("pad", pad);
  result.addAttribute("stride", stride);
  result.addAttribute("dilation", dilation);
  result.addAttribute("acc_type", accType);
  Type finalOutputType = outputType;
  auto quantAttr = buildConvOpQuantizationAttr(builder, input, weight);
  if (quantAttr) {
    finalOutputType =
        buildConvOpResultTypeInfo(builder, outputType, input, weight);
  }
  result.addTypes(finalOutputType);
}

/// Handles tosa.transpose_conv2d which has outpad and output shape
/// attributes.
static void
buildTransConvOpWithQuantInfo(OpBuilder &builder, OperationState &result,
                              Type outputType, Value input, Value weight,
                              Value bias, DenseI64ArrayAttr outpad,
                              DenseI64ArrayAttr stride, TypeAttr accType) {
  auto zps = createZPsAsConst(builder, input, weight);
  result.addOperands({input, weight, bias, zps.first, zps.second});
  result.addAttribute("out_pad", outpad);
  result.addAttribute("stride", stride);
  result.addAttribute("acc_type", accType);
  Type finalOutputType = outputType;
  auto quantAttr = buildConvOpQuantizationAttr(builder, input, weight);
  if (quantAttr) {
    finalOutputType =
        buildConvOpResultTypeInfo(builder, outputType, input, weight);
  }
  result.addTypes(finalOutputType);
}

/// The tosa.matmul op is also intended to be generated where a fully_connected
/// op must be constructed where the weight is not a constant. In this case,
/// the fully_connected op must be expressed using matmul.
/// TODO: Add link to the leglization document explaining this.
static void buildMatMulOpWithQuantInfo(OpBuilder &builder,
                                       OperationState &result, Type outputType,
                                       Value a, Value b) {
  auto zps = createZPsAsConst(builder, a, b);
  result.addOperands({a, b, zps.first, zps.second});

  Type finalOutputType{outputType};
  if (auto quantAttr = buildMatMulOpQuantizationAttr(builder, a, b)) {
    auto eType = getStorageElementTypeOrSelf(a.getType());
    auto inputBits = eType.getIntOrFloatBitWidth();

    auto outputShapedType = llvm::dyn_cast<ShapedType>(outputType);
    assert(outputShapedType && "Output must be a shaped type");

    IntegerType accElementType;
    if (inputBits == 16)
      accElementType = builder.getIntegerType(48);
    else
      accElementType = builder.getI32Type();

    finalOutputType = outputShapedType.clone(accElementType);
  }
  result.addTypes(finalOutputType);
}

/// Both the tosa.avg_pool2d and unary ops use the same
/// UnaryOpQuantizationAttr but avg_pool operator has its own builder as it
/// has additional parameters not part of the unary ops.
static void
buildAvgPool2dOpWithQuantInfo(OpBuilder &builder, OperationState &result,
                              Type outputType, Value input,
                              DenseArrayAttr kernel, DenseArrayAttr stride,
                              DenseArrayAttr pad, TypeAttr accType) {
  const Location loc{result.location};
  int64_t inputZp{0};
  int64_t outputZp{0};

  if (auto quantAttr =
          buildUnaryOpQuantizationAttr(builder, input, outputType)) {
    inputZp = quantAttr.getInputZp();
    outputZp = quantAttr.getOutputZp();
  }
  const std::optional<Value> inputZpOp =
      createZeroPointTensor(builder, loc, input.getType(), inputZp);
  if (!inputZpOp) {
    (void)emitError(
        loc,
        "Failed to create input zero point tensor for quantized AVG_POOL2D op");
  }
  const std::optional<Value> outputZpOp =
      createZeroPointTensor(builder, loc, outputType, outputZp);
  if (!outputZpOp) {
    (void)emitError(loc, "Failed to create output zero point tensor for "
                         "quantized AVG_POOL2D op");
  }

  if (inputZpOp && outputZpOp) {
    result.addOperands({input, inputZpOp.value(), outputZpOp.value()});
  } else {
    // failed to create one or more zero points above: just add input as
    // operands this will trigger error in building the op because of missing
    // zero points
    result.addOperands({input});
  }
  result.addAttribute("kernel", kernel);
  result.addAttribute("stride", stride);
  result.addAttribute("pad", pad);
  result.addAttribute("acc_type", accType);
  result.types.push_back(outputType);
}

/// This builder is called on single-parameter negate operator
/// to construct input and output zero points based on their
/// types.
static void buildNegateOpWithQuantInfo(OpBuilder &builder,
                                       OperationState &result, Type outputType,
                                       Value input) {
  const Location loc{result.location};
  int64_t input1Zp{0};
  int64_t outputZp{0};
  auto quantAttr = buildUnaryOpQuantizationAttr(builder, input, outputType);
  if (quantAttr) {
    input1Zp = quantAttr.getInputZp();
    outputZp = quantAttr.getOutputZp();
  }
  const std::optional<Value> input1ZpOp =
      createZeroPointTensor(builder, loc, input.getType(), input1Zp);
  if (!input1ZpOp) {
    (void)emitError(
        loc, "Failed to create input1 zero point for quantized NEGATE op");
  }

  const std::optional<Value> outputZpOp =
      createZeroPointTensor(builder, loc, input.getType(), outputZp);
  if (!outputZpOp) {
    (void)emitError(
        loc, "Failed to create output zero point for quantized NEGATE op");
  }

  if (input1ZpOp && outputZpOp) {
    result.addOperands({input, input1ZpOp.value(), outputZpOp.value()});
  } else {
    // failed to create one or more zero points above: just add input as
    // operands. This will trigger error in building the op because of
    // missing zero points
    result.addOperands({input});
  }

  result.types.push_back(outputType);
}

/// This builder is called on TOSA pad operator that needs to create its own
/// OptionalAttr quantization_attr parameter to scale the padding values
/// correctly. No pad_const is interpreted as zero-padding.
static void buildPadOpWithQuantInfo(OpBuilder &builder, OperationState &result,
                                    Type outputType, Value input,
                                    Value paddings) {
  const Location loc{result.location};
  int32_t zp{0};
  const auto quantAttr = buildPadOpQuantizationAttr(builder, input);
  if (quantAttr) {
    zp = static_cast<int32_t>(quantAttr.getInputZp());
  }
  const auto padConstOp{createPadConstTensor(builder, loc, input, zp)};
  result.addOperands({input, paddings, padConstOp});
  result.types.push_back(outputType);
}

static void buildVariableOp(OpBuilder &builder, OperationState &result,
                            StringRef name, Type variableType,
                            Attribute initialValue) {
  const Location loc{result.location};
  auto nameAttr = builder.getStringAttr(name);

  auto shapedType = dyn_cast<ShapedType>(variableType);
  if (!shapedType) {
    (void)emitError(loc, "variable type must be a shaped type");
    return;
  }
  if (!shapedType.hasRank()) {
    (void)emitError(loc, "variable type must be a ranked type");
    return;
  }

  auto elementType = shapedType.getElementType();
  auto elementTypeAttr = TypeAttr::get(elementType);
  ArrayRef<int64_t> shape = shapedType.getShape();
  auto varShapeAttr = builder.getIndexTensorAttr(convertFromMlirShape(shape));

  result.addAttribute("name", nameAttr);
  result.addAttribute("var_shape", varShapeAttr);
  result.addAttribute("type", elementTypeAttr);
  result.addAttribute("initial_value", initialValue);
}

//===----------------------------------------------------------------------===//
// TOSA Operator Return Type Inference.
//===----------------------------------------------------------------------===//

static LogicalResult resolveBroadcastShape(const ValueShapeRange &operands,
                                           SmallVector<int64_t> &outShape) {
  int64_t outRank = 0;
  for (int i = 0, e = operands.size(); i != e; ++i) {
    auto shape = operands.getShape(i);
    if (!shape.hasRank()) {
      // TODO(jennik): Update function to have better case handling for
      // invalid operands and for ranked tensors.
      return failure();
    }
    outRank = std::max<int64_t>(outRank, shape.getRank());
  }

  outShape.resize(outRank, 1);

  for (int i = 0, e = operands.size(); i != e; ++i) {
    auto shape = operands.getShape(i);
    auto rankDiff = outShape.size() - shape.getRank();

    for (size_t i = 0, e = shape.getRank(); i < e; ++i) {
      auto dim1 = outShape[i + rankDiff];
      auto dim2 = shape.getDimSize(i);
      auto resolvedDim = dim1;

      if (dim1 == 1) {
        resolvedDim = dim2;
      } else if (dim2 == 1) {
        resolvedDim = dim1;
      } else if (dim1 != dim2) {
        return failure();
      }
      outShape[i + rankDiff] = resolvedDim;
    }
  }

  return success();
}

LogicalResult tosa::ArgMaxOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ArgMaxOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape(adaptor.getInput().getType());
  IntegerAttr axis = adaptor.getProperties().axis;
  int32_t axisVal = axis.getValue().getSExtValue();

  if (!inputShape.hasRank()) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  SmallVector<int64_t> outShape;
  outShape.reserve(inputShape.getRank() - 1);
  for (int i = 0, s = inputShape.getRank(); i < s; i++) {
    if (i == axisVal)
      continue;
    outShape.push_back(inputShape.getDimSize(i));
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outShape));
  return success();
}

LogicalResult tosa::RFFT2dOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    RFFT2dOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape(adaptor.getInputReal().getType());

  if (!inputShape.hasRank())
    return failure();

  llvm::SmallVector<int64_t> outputShape;
  outputShape.resize(3, ShapedType::kDynamic);
  outputShape[0] = inputShape.getDimSize(0);
  outputShape[1] = inputShape.getDimSize(1);
  int64_t inWidth = inputShape.getDimSize(2);

  // Note that we can support this calculation symbolically
  // in the future e.g. [x, y, z] -> [x, y, z / 2 + 1]
  if (inWidth != ShapedType::kDynamic)
    outputShape[2] = inWidth / 2 + 1;

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));

  return success();
}

static LogicalResult verifyDimIsPowerOfTwo(Operation *op, const int64_t dimSize,
                                           const llvm::StringRef dimName) {
  const bool isPowerOfTwo = (dimSize & (dimSize - 1)) == 0 && dimSize > 0;
  if (!isPowerOfTwo)
    return op->emitOpError("expected ")
           << dimName << " to be a power of two, got " << dimSize;

  return success();
}

LogicalResult tosa::RFFT2dOp::verify() {
  const auto outputTypes = getResultTypes();
  if (failed(verifyCompatibleShapes(outputTypes)))
    return emitOpError("expected output shapes to match, got ") << outputTypes;

  const auto inputType =
      llvm::dyn_cast<RankedTensorType>(getInputReal().getType());
  if (!inputType)
    return success();

  const int64_t height = inputType.getDimSize(1);
  if (ShapedType::isStatic(height) &&
      failed(verifyDimIsPowerOfTwo(*this, height, "height")))
    return failure();

  const int64_t width = inputType.getDimSize(2);
  if (ShapedType::isStatic(width) &&
      failed(verifyDimIsPowerOfTwo(*this, width, "width")))
    return failure();

  const auto outputType = llvm::dyn_cast<RankedTensorType>(outputTypes[0]);
  if (!outputType)
    return success();

  // Batch and height input/output dimensions should match
  if (failed(verifyCompatibleShape(inputType.getShape().drop_back(),
                                   outputType.getShape().drop_back())))
    return emitOpError("expected batch and height dimensions of input/output "
                       "to match, got input=")
           << inputType << " output=" << outputType;

  // Output width dimension expected to be input_width / 2 + 1
  const int64_t outputWidth = outputType.getDimSize(2);
  if (ShapedType::isStatic(width) && ShapedType::isStatic(outputWidth) &&
      (outputWidth != (width / 2) + 1))
    return emitOpError(
               "expected output width to be equal to input_width / 2 + 1, got ")
           << outputWidth;

  return success();
}

LogicalResult tosa::FFT2dOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    FFT2dOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  inferredReturnShapes.push_back(
      ShapedTypeComponents(ShapeAdaptor(adaptor.getInputReal().getType())));
  inferredReturnShapes.push_back(
      ShapedTypeComponents(ShapeAdaptor(adaptor.getInputImag().getType())));
  return success();
}

LogicalResult tosa::FFT2dOp::verify() {
  const auto inputRealType =
      llvm::dyn_cast<RankedTensorType>(getInputReal().getType());
  const auto inputImagType =
      llvm::dyn_cast<RankedTensorType>(getInputImag().getType());
  if (!inputRealType || !inputImagType)
    return success();

  const auto trySelectStaticDim = [](const int64_t a, const int64_t b) {
    return ShapedType::isDynamic(a) ? a : b;
  };

  const int64_t height = trySelectStaticDim(inputRealType.getDimSize(1),
                                            inputImagType.getDimSize(1));
  if (ShapedType::isStatic(height) &&
      failed(verifyDimIsPowerOfTwo(*this, height, "height")))
    return failure();

  const int64_t width = trySelectStaticDim(inputRealType.getDimSize(2),
                                           inputImagType.getDimSize(2));
  if (ShapedType::isStatic(width) &&
      failed(verifyDimIsPowerOfTwo(*this, width, "width")))
    return failure();

  return success();
}

LogicalResult tosa::ConcatOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ConcatOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  // Infer all dimension sizes by reducing based on inputs.
  const Properties &prop = adaptor.getProperties();
  int32_t axis = prop.axis.getValue().getSExtValue();
  llvm::SmallVector<int64_t> outputShape;
  bool hasRankedInput = false;
  for (auto operand : adaptor.getOperands()) {
    ShapeAdaptor operandShape(operand.getType());
    if (!operandShape.hasRank())
      continue;

    // Copy the Operand's rank.
    if (!hasRankedInput)
      outputShape.resize(operandShape.getRank(), ShapedType::kDynamic);

    // Copy shapes until the dim is non-dynamic.
    for (int i = 0, s = operandShape.getRank(); i < s; i++) {
      if (i == axis || operandShape.isDynamicDim(i))
        continue;
      if (outputShape[i] == ShapedType::kDynamic)
        outputShape[i] = operandShape.getDimSize(i);
      if (outputShape[i] != operandShape.getDimSize(i))
        return emitOptionalError(location,
                                 "Cannot concat tensors with different sizes"
                                 " on the non-axis dimension ",
                                 i);
    }

    hasRankedInput = true;
  }

  if (adaptor.getInput1().empty())
    return failure();

  Type inputType =
      llvm::cast<TensorType>(adaptor.getInput1().getType()[0]).getElementType();
  if (!hasRankedInput) {
    inferredReturnShapes.push_back(ShapedTypeComponents(inputType));
    return success();
  }

  // Determine the dimension size along the concatenation axis.
  int64_t concatDimSize = 0;
  for (auto operand : adaptor.getOperands()) {
    ShapeAdaptor operandShape(operand.getType());

    // We need to know the length of the concatenation axis of all inputs to
    // determine the dimension size of the output shape.
    if (!operandShape.hasRank() || operandShape.isDynamicDim(axis)) {
      concatDimSize = ShapedType::kDynamic;
      break;
    }

    concatDimSize += operandShape.getDimSize(axis);
  }

  outputShape[axis] = concatDimSize;

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape, inputType));
  return success();
}

LogicalResult tosa::ConcatOp::verify() {
  // check that each input has same element type as output
  auto outType = getOutput().getType();
  const Operation::operand_range inputList = getInput1();

  // Check there is at least one input
  if (inputList.empty())
    return emitOpError("expect at least one input");

  if (!llvm::all_of(inputList, [&](auto input) {
        return succeeded(verifySameElementTypes(
            *this, /* inType = */ input.getType(), outType));
      })) {
    return failure();
  }

  const int32_t axis = getAxis();
  ShapeAdaptor firstRankedInputShape = nullptr;
  for (const auto &input : inputList) {
    const Type inputType = input.getType();
    ShapeAdaptor currShape(inputType);
    if (currShape.hasRank()) {
      firstRankedInputShape = currShape;
      // Check axis is in expected range
      if (axis < 0 || axis >= firstRankedInputShape.getRank())
        return emitOpError("expect axis to be within range 0 < axis < "
                           "rank(input1[firstRankedTensorIdx]), got ")
               << axis;
      break;
    }
  }

  const auto allOperandsHasRank = [](const Value input) {
    return ShapeAdaptor(input.getType()).hasRank();
  };
  if (llvm::all_of(inputList, allOperandsHasRank)) {
    const int64_t firstInputRank = firstRankedInputShape.getRank();

    for (const auto &[index, input] : llvm::enumerate(inputList.drop_front())) {
      const ShapeAdaptor inputShape(input.getType());
      const int64_t inputRank = inputShape.getRank();
      const size_t operandNum = index + 1;

      // Check that each operand has the same rank
      if (inputRank != firstInputRank)
        return emitOpError(
                   "expect all operands to have the same rank, but got ")
               << firstInputRank << " vs " << inputRank << " on operands 0 and "
               << operandNum;

      // Check non-axis dims match
      for (int i = 0; i < inputRank; i++) {
        const int64_t inputDim = inputShape.getDimSize(i);
        const int64_t firstInputDim = firstRankedInputShape.getDimSize(i);
        if (i == axis || firstRankedInputShape.isDynamicDim(i) ||
            inputShape.isDynamicDim(i))
          continue;
        if (inputDim != firstInputDim)
          return emitOpError("expect all operand shapes to have the same sizes "
                             "on non-axis dimensions, but got ")
                 << inputDim << " vs " << firstInputDim << " at index " << i
                 << " on operands 0 and " << operandNum;
      }
    }

    // ERROR_IF(axis_sum != shape[axis]);
    int64_t axisSum = 0;
    for (const auto &input : inputList) {
      const ShapeAdaptor inputShape(input.getType());
      if (inputShape.isDynamicDim(axis)) {
        // make axisSum negative to indicate invalid value
        axisSum = -1;
        break;
      }
      axisSum += inputShape.getDimSize(axis);
    }
    const ShapeAdaptor outputShape(outType);
    if (axisSum >= 0 && outputShape.hasRank() &&
        !outputShape.isDynamicDim(axis) &&
        axisSum != outputShape.getDimSize(axis))
      return emitOpError("requires sum of axis dimensions of input1 "
                         "equal to output axis dimension, got ")
             << axisSum << " and " << outputShape.getDimSize(axis);
  }

  return success();
}

LogicalResult tosa::EqualOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  auto elementType = IntegerType::get(context, /*width=*/1);

  llvm::SmallVector<int64_t> outShape;
  if (resolveBroadcastShape(operands, outShape).failed()) {
    inferredReturnShapes.push_back(ShapedTypeComponents(elementType));
    return success();
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outShape, elementType));
  return success();
}

bool tosa::EqualOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  if (l.size() != r.size() || l.size() != 1)
    return false;
  return succeeded(verifyCompatibleShape(l[0], r[0]));
}

LogicalResult tosa::MatMulOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    MatMulOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor lhsShape(adaptor.getA().getType());
  ShapeAdaptor rhsShape(adaptor.getB().getType());

  // All shapes are dynamic.
  SmallVector<int64_t> outShape;
  outShape.resize(3, ShapedType::kDynamic);

  if (lhsShape.hasRank()) {
    outShape[0] = lhsShape.getDimSize(0);
    outShape[1] = lhsShape.getDimSize(1);
  }

  if (rhsShape.hasRank()) {
    outShape[0] = outShape[0] == ShapedType::kDynamic ? rhsShape.getDimSize(0)
                                                      : outShape[0];
    outShape[2] = rhsShape.getDimSize(2);
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outShape));
  return success();
}

LogicalResult MatMulOp::verify() {
  auto aType = llvm::dyn_cast<ShapedType>(getA().getType());
  auto bType = llvm::dyn_cast<ShapedType>(getB().getType());

  // Must be shaped tensor types
  if (!aType)
    return emitOpError("expect a shaped tensor for input a, got ")
           << getA().getType();

  if (!bType)
    return emitOpError("expect a shaped tensor for input b, got ")
           << getB().getType();

  auto aElementType = aType.getElementType();
  auto bElementType = bType.getElementType();

  auto aQuantizedEType =
      llvm::dyn_cast<quant::UniformQuantizedType>(aElementType);
  auto bQuantizedEType =
      llvm::dyn_cast<quant::UniformQuantizedType>(bElementType);

  if (aQuantizedEType || bQuantizedEType) {
    if (!aQuantizedEType || !bQuantizedEType) {
      return emitOpError("expect operands to be both quantized or both not "
                         "quantized, got ")
             << aElementType << " and " << bElementType;
    }
    // both a and b have quantized element types
    auto aQuantWidth = aQuantizedEType.getStorageTypeIntegralWidth();
    auto bQuantWidth = bQuantizedEType.getStorageTypeIntegralWidth();
    if (aQuantWidth != bQuantWidth) {
      return emitOpError("expect quantized operands to have same widths, got ")
             << aQuantWidth << " and " << bQuantWidth;
    }
  } else {
    // non-quantized element types
    if (aElementType != bElementType) {
      return emitOpError("expect same element type for inputs a and b, got ")
             << aElementType << " and " << bElementType;
    }
  }

  // check a_zp and b_zp
  auto aEType = getStorageElementTypeOrSelf(aType);
  auto aZpEType = getStorageElementTypeOrSelf(getAZp().getType());
  if (aEType != aZpEType) {
    return emitOpError("expect input a and a_zp have the same "
                       "element type, got ")
           << aEType << " and " << aZpEType;
  }

  auto bEType = getStorageElementTypeOrSelf(bType);
  auto bZpEType = getStorageElementTypeOrSelf(getBZp().getType());
  if (bEType != bZpEType) {
    return emitOpError("expect input b and b_zp have the same "
                       "element type, got ")
           << bEType << " and " << bZpEType;
  }

  FailureOr<int64_t> maybeAZp = getAZeroPoint();
  if (succeeded(maybeAZp) && verifyAZeroPoint(*maybeAZp).failed())
    return failure();

  FailureOr<int64_t> maybeBZp = getBZeroPoint();
  if (succeeded(maybeBZp) && verifyBZeroPoint(*maybeBZp).failed())
    return failure();

  return success();
}

LogicalResult tosa::PadOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    PadOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape(adaptor.getInput1().getType());
  auto paddingRank =
      cast<tosa::shapeType>(adaptor.getPadding().getType()).getRank();
  SmallVector<int64_t> outputShape;

  // If the input rank is unknown, we can infer the output rank using the
  // padding shape's rank divided by 2.
  if (!inputShape.hasRank()) {
    outputShape.resize(paddingRank / 2, ShapedType::kDynamic);
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  SmallVector<int64_t> paddingValues;
  // If the paddings value is not a constant, all dimensions must be dynamic.
  if (!tosa::getConstShapeValues(adaptor.getPadding().getDefiningOp(),
                                 paddingValues)) {
    outputShape.resize(inputShape.getRank(), ShapedType::kDynamic);
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  outputShape.reserve(inputShape.getRank());
  for (int i = 0, s = inputShape.getRank(); i < s; i++) {
    if (inputShape.isDynamicDim(i)) {
      outputShape.push_back(ShapedType::kDynamic);
      continue;
    }
    auto padFront = paddingValues[i * 2];
    auto padBack = paddingValues[i * 2 + 1];
    if (padFront < 0 || padBack < 0) {
      // if either padding for dim i is -1, output dim is unknown
      outputShape.push_back(ShapedType::kDynamic);
      continue;
    }

    outputShape.push_back(inputShape.getDimSize(i) + padFront + padBack);
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult tosa::PadOp::verify() {
  if (verifySameElementTypes(*this, /* inType = */ getInput1().getType(),
                             /* outType = */ getOutput().getType())
          .failed()) {
    return failure();
  }

  if (auto padConst = getPadConst()) {
    if (verifySameElementTypes(*this, /* inType = */ padConst.getType(),
                               /* outType = */ getOutput().getType())
            .failed()) {
      return failure();
    }
  }

  RankedTensorType inputType =
      llvm::dyn_cast<RankedTensorType>(getInput1().getType());
  RankedTensorType outputType =
      llvm::dyn_cast<RankedTensorType>(getOutput().getType());
  if (!inputType || !outputType)
    return success();

  auto inputRank = inputType.getRank();
  auto outputRank = outputType.getRank();
  if (inputRank != outputRank)
    return emitOpError() << "expect same input and output tensor rank, but got "
                         << "inputRank: " << inputRank
                         << ", outputRank: " << outputRank;

  DenseIntElementsAttr paddingAttr;
  if (!matchPattern(getPadding(), m_Constant(&paddingAttr))) {
    return failure();
  }

  auto paddingValues = paddingAttr.getValues<APInt>();
  if (paddingValues.size() != static_cast<size_t>(inputRank * 2))
    return emitOpError() << "padding tensor must have " << inputRank
                         << " * 2 = " << inputRank * 2 << " elements, but got "
                         << paddingValues.size();

  auto inputShape = inputType.getShape();
  auto outputShape = outputType.getShape();

  for (int64_t i = 0; i < inputRank; ++i) {
    int64_t padStart = paddingValues[i * 2].getSExtValue();
    int64_t padEnd = paddingValues[i * 2 + 1].getSExtValue();

    if ((padStart < 0 && padStart != -1) || (padEnd < 0 && padEnd != -1)) {
      return emitOpError()
             << "invalid padding values at dimension " << i
             << ": values must be non-negative or -1 for dynamic padding, got ["
             << padStart << ", " << padEnd << "]";
    }

    // Skip shape verification for dynamic input/output
    if (inputShape[i] == ShapedType::kDynamic ||
        outputShape[i] == ShapedType::kDynamic)
      continue;

    if (outputShape[i] != inputShape[i] + padStart + padEnd) {
      return emitOpError() << "mismatch in output shape at dimension " << i
                           << ": expected " << inputShape[i] << " + "
                           << padStart << " + " << padEnd << " = "
                           << (inputShape[i] + padStart + padEnd)
                           << ", but got " << outputShape[i];
    }
  }

  return success();
}

LogicalResult tosa::SliceOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    SliceOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {

  Type inputType = getElementTypeOrSelf(adaptor.getInput1().getType());
  SmallVector<int64_t> start;
  SmallVector<int64_t> size;

  if (!tosa::getConstShapeValues(adaptor.getStart().getDefiningOp(), start) ||
      !tosa::getConstShapeValues(adaptor.getSize().getDefiningOp(), size)) {
    auto rank = cast<tosa::shapeType>(adaptor.getSize().getType()).getRank();
    SmallVector<int64_t> fallback(rank, ShapedType::kDynamic);
    inferredReturnShapes.push_back(ShapedTypeComponents(fallback, inputType));
    return success();
  }

  // if size[i] is -1, all remaining elements in dimension i are included
  // in the slice, similar to TF.
  ShapeAdaptor inputShape(adaptor.getInput1().getType());
  // initialize outputShape to all unknown
  SmallVector<int64_t> outputShape(size.size(), ShapedType::kDynamic);
  if (inputShape.hasRank()) {
    for (size_t i = 0; i < size.size(); i++) {
      if (size[i] != 0 && size[i] >= -1 && start[i] >= 0 &&
          (ShapedType::isDynamic(inputShape.getDimSize(i)) ||
           start[i] < inputShape.getDimSize(i))) {
        // size[i] is not 0 and not < -1, and start[i] is in valid range
        if (ShapedType::isDynamic(inputShape.getDimSize(i))) {
          // input shape has unknown dim[i] - only valid if size[i] > 0
          if (size[i] > 0) {
            outputShape[i] = size[i];
          }
        } else {
          // input shape has known dim[i]
          if (size[i] == -1) {
            outputShape[i] = inputShape.getDimSize(i) - start[i];
          } else if (start[i] + size[i] <= inputShape.getDimSize(i)) {
            // start[i] + size[i] is within bound of input shape's dim[i]
            outputShape[i] = size[i];
          }
        }
      }
    }
  } else {
    outputShape = convertToMlirShape(size);
  }
  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult tosa::SliceOp::verify() {
  if (verifySameElementTypes(*this, /* inType = */ getInput1().getType(),
                             /* outType = */ getOutput().getType())
          .failed())
    return failure();

  const ShapeAdaptor inputShape(getInput1().getType());
  if (inputShape.hasRank()) {
    const auto inputRank = inputShape.getRank();
    const ShapeAdaptor outputShape(getOutput().getType());
    if (outputShape.hasRank() && inputRank != outputShape.getRank())
      return emitOpError(
                 "expect input1 and output to have the same ranks, got ")
             << inputRank << " and " << outputShape.getRank();

    const auto startShapeRank =
        llvm::cast<tosa::shapeType>(getStart().getType()).getRank();
    if (inputRank != startShapeRank)
      return emitOpError("length of start is not equal to rank of input shape");

    const auto sizeShapeRank =
        llvm::cast<tosa::shapeType>(getSize().getType()).getRank();
    if (inputRank != sizeShapeRank)
      return emitOpError("length of size is not equal to rank of input shape");
  }

  return success();
}

LogicalResult tosa::MulOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ValueShapeRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  // mul op's output shape only depend on input1 and input2, not on shift
  ValueShapeRange twoInputs = operands.drop_back();
  llvm::SmallVector<int64_t> outShape;
  if (resolveBroadcastShape(twoInputs, outShape).failed()) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
  } else {
    inferredReturnShapes.push_back(ShapedTypeComponents(outShape));
  }
  return success();
}

LogicalResult tosa::MulOp::verify() {
  const Value output = getOutput();
  auto resElemType = getElementTypeOrSelf(output);

  // Verify if the element type among operands and result match tosa
  // specification.
  if (auto resIntType = dyn_cast<IntegerType>(resElemType)) {
    IntegerType lhsIntType =
        dyn_cast<IntegerType>(getElementTypeOrSelf(getInput1()));
    IntegerType rhsIntType =
        dyn_cast<IntegerType>(getElementTypeOrSelf(getInput2()));
    if (!lhsIntType || !rhsIntType || lhsIntType != rhsIntType)
      return emitOpError("requires the same element type for all operands");

    // Though the spec requires the element type of result to be i32, a more
    // relaxed way is provided at dialect level for easier cooperating with
    // other dialects.
    if (lhsIntType.getWidth() > resIntType.getWidth())
      return emitOpError("invalid data type size for operands or result");

  } else {
    // For other supported type, the spec requires requires the same element
    // type for all operands (excludes `shift` operand) and results.
    for (int i = 0; i < 2; ++i) {
      if (getElementTypeOrSelf(getOperand(i)) != resElemType)
        return emitOpError(
            "requires the same element type for all operands and results");
    }

    // verify shift has value 0 for non-integer types
    ElementsAttr shift_elem;
    if (matchPattern(getShift(), m_Constant(&shift_elem))) {
      int32_t shift = shift_elem.getValues<IntegerAttr>()[0].getInt();
      if (shift != 0) {
        return emitOpError() << "require shift to be 0 for float type";
      }
    }
  }

  // Verify the op has same ranks for all main operands (excludes extra operands
  // such as shift of mul op, so this is the only difference with the built-in
  // `SameOperandsAndResultRank` trait) and results types, if known.
  TypeRange operandTypes = getOperandTypes();
  ShapedType aType = cast<ShapedType>(operandTypes[0]);
  ShapedType bType = cast<ShapedType>(operandTypes[1]);

  const bool aHasRank = aType.hasRank();
  const bool bHasRank = bType.hasRank();
  if (aHasRank && bHasRank) {
    const int64_t aRank = aType.getRank();
    const int64_t bRank = bType.getRank();
    if (aRank != bRank)
      return emitOpError("a and b operands don't have matching ranks, got ")
             << aRank << " and " << bRank;

    // check for broadcast compatible shapes
    SmallVector<int64_t> resultShape;
    if (!mlir::OpTrait::util::getBroadcastedShape(
            aType.getShape(), bType.getShape(), resultShape))
      return emitOpError("a and b operands don't have broadcast-compatible "
                         "shapes, got ")
             << aType << " and " << bType;
  }

  ShapedType resultType = cast<ShapedType>(output.getType());
  if (!resultType.hasRank())
    return success();

  const int64_t resultRank = resultType.getRank();
  if (aHasRank && resultRank != aType.getRank())
    return emitOpError("result type has different rank than a, got ")
           << resultRank << " vs " << aType.getRank();
  if (bHasRank && resultRank != bType.getRank())
    return emitOpError("result type has different rank than b, got ")
           << resultRank << " vs " << bType.getRank();

  return success();
}

LogicalResult tosa::TableOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    TableOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape(adaptor.getInput1().getType());

  if (!inputShape.hasRank()) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  inferredReturnShapes.resize(1);
  inputShape.getDims(inferredReturnShapes[0]);
  return success();
}

LogicalResult tosa::TableOp::verify() {
  const TensorType inputType = getInput1().getType();
  const TensorType outputType = getOutput().getType();

  if (!inputType.hasRank() || !outputType.hasRank())
    return success();

  if (inputType.getRank() != outputType.getRank())
    return emitOpError()
           << "expected input tensor rank to equal result tensor rank";

  auto inputDims = inputType.getShape();
  auto outputDims = outputType.getShape();
  for (auto it : llvm::enumerate(llvm::zip(inputDims, outputDims))) {
    int64_t dim = it.index();
    auto [inputDim, outputDim] = it.value();
    if (ShapedType::isStatic(outputDim) && outputDim != inputDim) {
      return emitOpError() << "dim(result, " << dim << ") = " << outputDim
                           << " doesn't match dim(input, " << dim
                           << ") = " << inputDim;
    }
  }
  return success();
}

LogicalResult
tosa::TileOp::getConstantMultiples(SmallVector<int64_t> &multiples) {
  // Multiples must be constants.
  DenseIntElementsAttr multiplesAttr;
  if (!matchPattern(getMultiples(), m_Constant(&multiplesAttr)))
    return failure();
  multiples = llvm::to_vector(
      llvm::map_range(multiplesAttr.getValues<APInt>(),
                      [](const APInt &val) { return val.getSExtValue(); }));
  return success();
}

LogicalResult tosa::TileOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    TileOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  Type inputType = getElementTypeOrSelf(adaptor.getInput1().getType());
  SmallVector<int64_t> multiples;
  if (!tosa::getConstShapeValues(adaptor.getMultiples().getDefiningOp(),
                                 multiples)) {
    auto rank =
        cast<tosa::shapeType>(adaptor.getMultiples().getType()).getRank();
    SmallVector<int64_t> fallback(rank, ShapedType::kDynamic);
    inferredReturnShapes.push_back(ShapedTypeComponents(fallback, inputType));
    return success();
  } else {
    multiples = convertToMlirShape(multiples);
  }

  ShapeAdaptor inputShape(adaptor.getInput1().getType());
  SmallVector<int64_t> outputShape;
  if (!inputShape.hasRank()) {
    outputShape.resize(multiples.size(), ShapedType::kDynamic);
    inferredReturnShapes.push_back(
        ShapedTypeComponents(outputShape, inputType));
    return success();
  } else if (static_cast<size_t>(inputShape.getRank()) != multiples.size())
    return failure();

  // Any non dynamic dimension can be multiplied to a known size.
  outputShape.reserve(multiples.size());
  for (int i = 0, s = inputShape.getRank(); i < s; i++) {
    if (multiples[i] == ShapedType::kDynamic) {
      outputShape.push_back(ShapedType::kDynamic);
    } else {
      int64_t dim = inputShape.getDimSize(i);
      if (dim != ShapedType::kDynamic)
        dim *= multiples[i];
      outputShape.push_back(dim);
    }
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape, inputType));
  return success();
}

LogicalResult tosa::TileOp::verify() {
  if (verifySameElementTypes(*this, /* intype = */ getInput1().getType(),
                             /* outType = */ getOutput().getType())
          .failed()) {
    return failure();
  }
  ShapedType inputType = llvm::cast<ShapedType>(getInput1().getType());
  ShapedType outputType = llvm::cast<ShapedType>(getType());

  shapeType multiplesType =
      llvm::cast<tosa::shapeType>(getMultiples().getType());

  auto multiplesRank = multiplesType.getRank();

  if (inputType.hasRank()) {
    if (inputType.getRank() != multiplesRank)
      return emitOpError("expect 'multiples' to have rank ")
             << inputType.getRank() << " but got " << multiplesRank << ".";
    if (outputType.hasRank() && inputType.getRank() != outputType.getRank())
      return emitOpError("expect same input and output tensor rank.");
  } else if (outputType.hasRank() && outputType.getRank() != multiplesRank)
    return emitOpError("expect 'multiples' array to have length ")
           << outputType.getRank() << " but got " << multiplesRank << ".";

  SmallVector<int64_t> multiples;
  if (getConstantMultiples(multiples).succeeded() &&
      llvm::any_of(multiples, [](int64_t v) { return v <= 0 && v != -1; }))
    return emitOpError(
        "expect element of 'multiples' to be positive integer or -1.");

  return success();
}

bool tosa::ReshapeOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  if (l.size() != r.size() || l.size() != 1)
    return false;
  return getElementTypeOrSelf(l[0]) == getElementTypeOrSelf(r[0]);
}

LogicalResult tosa::ReshapeOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ReshapeOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape(adaptor.getInput1().getType());
  Type inputType = getElementTypeOrSelf(adaptor.getInput1().getType());
  llvm::SmallVector<int64_t> newShapeValue;
  if (!tosa::getConstShapeValues(adaptor.getShape().getDefiningOp(),
                                 newShapeValue)) {
    auto rank = cast<tosa::shapeType>(adaptor.getShape().getType()).getRank();
    SmallVector<int64_t> fallback(rank, ShapedType::kDynamic);
    inferredReturnShapes.push_back(ShapedTypeComponents(fallback, inputType));
    return success();
  } else {
    newShapeValue = convertToMlirShape(newShapeValue);
  }

  // We cannot infer from the total number of elements so we must take the
  // shape attribute as exact.
  if (!inputShape.hasRank() || !inputShape.hasStaticShape()) {
    inferredReturnShapes.push_back(
        ShapedTypeComponents(newShapeValue, inputType));
    return success();
  }

  // Determine the number of elements covered by the slice of all static
  // dimensions. This allows us to infer the length of the remaining dynamic
  // dimension.
  int64_t numElements = inputShape.getNumElements();
  int64_t staticMul = 1;
  for (auto val : newShapeValue) {
    if (ShapedType::isStatic(val)) {
      staticMul *= val;
    }
  }

  // Determine the length of the dynamic dimension.
  for (auto &val : newShapeValue) {
    if (ShapedType::isDynamic(val))
      val = numElements / staticMul;
  }

  inferredReturnShapes.push_back(
      ShapedTypeComponents(newShapeValue, inputType));
  return success();
}

llvm::LogicalResult tosa::ReshapeOp::verify() {
  if (verifySameElementTypes(*this, /* inType = */ getInput1().getType(),
                             /* outType = */ getOutput().getType())
          .failed()) {
    return failure();
  }
  TensorType inputType = getInput1().getType();

  SmallVector<int64_t> shapeValues;
  if (!tosa::getConstShapeValues(getShape().getDefiningOp(), shapeValues)) {
    // skip following checks if shape is not constant
    return mlir::success();
  }

  int missingDims = llvm::count(shapeValues, -1);
  if (missingDims > 1)
    return emitOpError() << "expected at most one target dimension to be -1";

  const auto outputType = dyn_cast<RankedTensorType>(getType());
  if (!outputType)
    return success();

  if ((int64_t)shapeValues.size() != outputType.getRank())
    return emitOpError() << "new shape does not match result rank";

  for (auto [newShapeDim, outputShapeDim] :
       zip(shapeValues, outputType.getShape())) {
    if (newShapeDim != -1 && newShapeDim != ShapedType::kDynamic &&
        outputShapeDim != ShapedType::kDynamic && newShapeDim != outputShapeDim)
      return emitOpError() << "new shape is inconsistent with result shape";

    if (newShapeDim != ShapedType::kDynamic && newShapeDim < -1)
      return emitOpError() << "new shape has invalid tensor dimension size "
                           << newShapeDim;
  }

  if (inputType.hasStaticShape()) {
    int64_t inputElementsNum = inputType.getNumElements();
    if (outputType.hasStaticShape()) {
      int64_t outputElementsNum = outputType.getNumElements();
      if (inputElementsNum != outputElementsNum) {
        return emitOpError() << "cannot reshape " << inputElementsNum
                             << " elements into " << outputElementsNum;
      }
    }

    int64_t newShapeElementsNum = std::accumulate(
        shapeValues.begin(), shapeValues.end(), 1LL,
        [](int64_t acc, int64_t dim) { return (dim > 0) ? acc * dim : acc; });
    bool isStaticNewShape =
        llvm::all_of(shapeValues, [](int64_t s) { return s > 0; });
    if ((isStaticNewShape && inputElementsNum != newShapeElementsNum) ||
        (!isStaticNewShape && newShapeElementsNum > inputElementsNum)) {
      return emitOpError() << "cannot reshape " << inputElementsNum
                           << " elements into " << newShapeElementsNum;
    }
  }

  return mlir::success();
}

// return failure if val is not a constant
// set zp to -1 if val is non-zero float or val is not integer nor float
// otherwise set zp to val's constant value
static FailureOr<int64_t> getZeroPoint(Value val, bool signExtend) {
  ElementsAttr zpAttr;
  if (!matchPattern(val, m_Constant(&zpAttr))) {
    return failure();
  }

  Type zpElemType = zpAttr.getElementType();

  if (llvm::isa<FloatType>(zpElemType)) {
    if (zpAttr.getValues<APFloat>()[0].isZero()) {
      return 0;
    }
    // return non-zero value to trigger error check
    return -1;
  }

  if (llvm::isa<IntegerType>(zpElemType)) {
    if (signExtend)
      return zpAttr.getValues<APInt>()[0].getSExtValue();
    else
      return zpAttr.getValues<APInt>()[0].getZExtValue();
  }

  // return non-zero value to trigger error check
  return -1;
}

template <typename T>
static LogicalResult verifyZeroPoint(T op, Value val, const int64_t &zp,
                                     const std::string &operand) {
  Type zpElemType = getElementTypeOrSelf(val);

  if (!zpElemType.isInteger(8) && zp != 0) {
    // convert operand to lower case for error message
    std::string lower = operand;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    return op.emitOpError()
           << lower << " zero point must be zero for non-int8 integer types";
  }

  return success();
}

static LogicalResult verifyZeroPoint(tosa::RescaleOp op, Value zpVal,
                                     const int64_t &zp,
                                     const std::string &operand) {
  bool isInputZp = (operand == "Input");

  bool tensorUnsigned =
      isInputZp ? op.getInputUnsigned() : op.getOutputUnsigned();
  StringRef tensorName = isInputZp ? "input" : "output";

  Type zpElemType = getElementTypeOrSelf(zpVal);

  if (zp != 0) {
    if (!zpElemType.isInteger(8) &&
        !(zpElemType.isInteger(16) && tensorUnsigned)) {
      return op.emitOpError()
             << "expect " << tensorName << "_zp of 0, got " << zp;
    }
    if (zpElemType.isInteger(16) && tensorUnsigned && zp != 32768) {
      return op.emitOpError() << "expect " << tensorName
                              << "_zp of 0 or 32768 for unsigned int16 "
                              << tensorName << ", got " << zp;
    }
  }

  return success();
}

#define ZERO_POINT_HELPER(OP, OPERAND_NAME, SIGN_EXTEND)                       \
  FailureOr<int64_t> tosa::OP::get##OPERAND_NAME##ZeroPoint() {                \
    return getZeroPoint(get##OPERAND_NAME##Zp(), SIGN_EXTEND);                 \
  }                                                                            \
  LogicalResult tosa::OP::verify##OPERAND_NAME##ZeroPoint(int64_t zp) {        \
    return verifyZeroPoint(*this, get##OPERAND_NAME##Zp(), zp, #OPERAND_NAME); \
  }

ZERO_POINT_HELPER(Conv2DOp, Input, true)
ZERO_POINT_HELPER(Conv2DOp, Weight, true)
ZERO_POINT_HELPER(Conv3DOp, Input, true)
ZERO_POINT_HELPER(Conv3DOp, Weight, true)
ZERO_POINT_HELPER(DepthwiseConv2DOp, Input, true)
ZERO_POINT_HELPER(DepthwiseConv2DOp, Weight, true)
ZERO_POINT_HELPER(TransposeConv2DOp, Input, true)
ZERO_POINT_HELPER(TransposeConv2DOp, Weight, true)
ZERO_POINT_HELPER(AvgPool2dOp, Input, true)
ZERO_POINT_HELPER(AvgPool2dOp, Output, true)
ZERO_POINT_HELPER(MatMulOp, A, true)
ZERO_POINT_HELPER(MatMulOp, B, true)
ZERO_POINT_HELPER(NegateOp, Input1, true)
ZERO_POINT_HELPER(NegateOp, Output, true)
ZERO_POINT_HELPER(RescaleOp, Input, !getInputUnsigned())
ZERO_POINT_HELPER(RescaleOp, Output, !getOutputUnsigned())
#undef ZERO_POINT_HELPER

LogicalResult tosa::TransposeOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    TransposeOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape(adaptor.getInput1().getType());

  // If input rank and permutation length is unknown, the output rank is
  // unknown.
  if (!inputShape.hasRank()) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
    return success();
  }

  const auto inputRank = inputShape.getRank();

  // This would imply the number of permutations does not match the rank of
  // the input which is illegal.
  if (adaptor.getPerms().size() != static_cast<size_t>(inputRank)) {
    return failure();
  }

  SmallVector<int64_t> outputShape;
  // Rank-0 means no permutations matter.
  if (inputRank == 0) {
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  // Check whether the input dimensions are all the same.
  bool allTheSame = true;
  for (int i = 1, s = inputRank; i < s; i++) {
    if (inputShape.getDimSize(0) != inputShape.getDimSize(i)) {
      allTheSame = false;
      break;
    }
  }

  // If all of the input dimensions are the same we don't care about the
  // permutation.
  if (allTheSame) {
    outputShape.resize(inputRank, inputShape.getDimSize(0));
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  outputShape.resize(inputRank, ShapedType::kDynamic);

  // Constant permutation values must be within the input rank.
  if (llvm::any_of(adaptor.getPerms(),
                   [inputRank](const auto i) { return i >= inputRank; }))
    return failure();

  outputShape.reserve(inputRank);
  for (int i = 0, s = inputRank; i < s; i++) {
    outputShape[i] = inputShape.getDimSize(adaptor.getPerms()[i]);
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult tosa::TransposeOp::verify() {
  if (verifySameElementTypes(*this, /* inType = */ getInput1().getType(),
                             /* outType = */ getOutput().getType())
          .failed()) {
    return failure();
  }

  const ShapeAdaptor inputShape(getInput1().getType());
  const ShapeAdaptor outputShape(getOutput().getType());

  const llvm::ArrayRef<int32_t> constantPerms = getPerms();

  if (inputShape.hasRank() &&
      constantPerms.size() != static_cast<size_t>(inputShape.getRank()))
    return emitOpError() << "expected perms attribute to have size "
                         << inputShape.getRank()
                         << " (input rank) but got size "
                         << constantPerms.size();

  if (inputShape.hasRank() && outputShape.hasRank() &&
      inputShape.getRank() != outputShape.getRank())
    return emitOpError()
           << "expected input tensor rank to equal result tensor rank";

  if (outputShape.hasRank() &&
      constantPerms.size() != static_cast<size_t>(outputShape.getRank()))
    return emitOpError() << "expected perms attribute to have size "
                         << outputShape.getRank()
                         << " (output rank) but got size "
                         << constantPerms.size();

  if (!llvm::all_of(constantPerms,
                    [&constantPerms](int32_t s) {
                      return s >= 0 &&
                             static_cast<size_t>(s) < constantPerms.size();
                    }) ||
      !isPermutationVector(llvm::to_vector(llvm::map_range(
          constantPerms, [](int32_t v) -> int64_t { return v; }))))
    return emitOpError() << "expected valid permutation indices";

  // ERROR_IF(tensor_size(shape1) != tensor_size(shape))
  if (inputShape.hasStaticShape() && outputShape.hasStaticShape() &&
      inputShape.getNumElements() != outputShape.getNumElements())
    return emitOpError() << "expected input1 and output to have same numbers "
                            "of elements, got "
                         << inputShape.getNumElements() << " and "
                         << outputShape.getNumElements();

  // Verify that the types of the input and output tensors are properly
  // permuted.
  if (inputShape.hasRank() && outputShape.hasRank()) {
    for (auto i = 0; i < outputShape.getRank(); i++) {
      if (inputShape.isDynamicDim(constantPerms[i]) ||
          outputShape.isDynamicDim(i))
        continue;

      if (inputShape.getDimSize(constantPerms[i]) != outputShape.getDimSize(i))
        return emitOpError()
               << "expected output tensor dim " << i << " to match "
               << "input dim " << constantPerms[i] << " with value of "
               << inputShape.getDimSize(constantPerms[i]);
    }
  }

  return success();
}

LogicalResult TransposeOp::reifyResultShapes(
    OpBuilder &builder, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {

  const llvm::ArrayRef<int32_t> transposePerms = getPerms();

  Value input = getInput1();
  auto inputType = cast<TensorType>(input.getType());

  SmallVector<OpFoldResult> returnedDims(inputType.getRank());
  for (auto dim : transposePerms) {
    int32_t dimInInput = transposePerms[dim];
    if (inputType.isDynamicDim(dimInInput))
      returnedDims[dim] =
          tensor::DimOp::create(builder, getLoc(), input, dimInInput)
              .getResult();
    else
      returnedDims[dim] =
          builder.getIndexAttr(inputType.getDimSize(dimInInput));
  }

  reifiedReturnShapes.emplace_back(std::move(returnedDims));
  return success();
}

LogicalResult tosa::GatherOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    GatherOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outputShape;
  outputShape.resize(3, ShapedType::kDynamic);

  ShapeAdaptor valuesShape(adaptor.getValues().getType());
  if (valuesShape.hasRank()) {
    outputShape[0] = valuesShape.getDimSize(0);
    outputShape[2] = valuesShape.getDimSize(2);
  }

  ShapeAdaptor indicesShape(adaptor.getIndices().getType());
  if (indicesShape.hasRank()) {
    if (outputShape[0] == ShapedType::kDynamic)
      outputShape[0] = indicesShape.getDimSize(0);
    if (outputShape[1] == ShapedType::kDynamic)
      outputShape[1] = indicesShape.getDimSize(1);
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult tosa::GatherOp::verify() {
  if (verifySameElementTypes(*this, /* inType = */ getValues().getType(),
                             /* outType = */ getOutput().getType())
          .failed()) {
    return failure();
  }

  const ShapeAdaptor valuesShape(getValues().getType());
  const ShapeAdaptor indicesShape(getIndices().getType());
  const ShapeAdaptor outputShape(getOutput().getType());

  int64_t N = ShapedType::kDynamic;
  int64_t W = ShapedType::kDynamic;
  int64_t C = ShapedType::kDynamic;

  if (valuesShape.hasRank()) {
    N = valuesShape.getDimSize(0);
    C = valuesShape.getDimSize(2);
  }
  if (indicesShape.hasRank()) {
    const int64_t indicesN = indicesShape.getDimSize(0);
    W = indicesShape.getDimSize(1);
    if (N == ShapedType::kDynamic)
      N = indicesN;
    else if (indicesN != ShapedType::kDynamic && N != indicesN)
      return emitOpError() << "requires indices dimension 0 to have size " << N
                           << ", got " << indicesN;
  }
  if (outputShape.hasRank()) {
    const int64_t outputN = outputShape.getDimSize(0);
    const int64_t outputW = outputShape.getDimSize(1);
    const int64_t outputC = outputShape.getDimSize(2);
    if (N != ShapedType::kDynamic && outputN != ShapedType::kDynamic &&
        N != outputN)
      return emitOpError() << "requires output dimension 0 to have size " << N
                           << ", got " << outputN;

    if (W != ShapedType::kDynamic && outputW != ShapedType::kDynamic &&
        W != outputW)
      return emitOpError() << "requires output dimension 1 to have size " << W
                           << ", got " << outputW;
    if (C != ShapedType::kDynamic && outputC != ShapedType::kDynamic &&
        C != outputC)
      return emitOpError() << "requires output dimension 2 to have size " << C
                           << ", got " << outputC;
  }
  return success();
}

LogicalResult tosa::ResizeOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ResizeOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t, 4> outputShape;
  outputShape.resize(4, ShapedType::kDynamic);

  ShapeAdaptor inputShape(adaptor.getInput().getType());
  if (!inputShape.hasRank())
    return failure();

  outputShape[0] = inputShape.getDimSize(0);
  outputShape[3] = inputShape.getDimSize(3);
  int64_t inputHeight = inputShape.getDimSize(1);
  int64_t inputWidth = inputShape.getDimSize(2);

  if ((inputHeight == ShapedType::kDynamic) ||
      (inputWidth == ShapedType::kDynamic))
    return failure();

  SmallVector<int64_t> scaleInt, offsetInt, borderInt;
  if (!tosa::getConstShapeValues(adaptor.getScale().getDefiningOp(),
                                 scaleInt) ||
      !tosa::getConstShapeValues(adaptor.getOffset().getDefiningOp(),
                                 offsetInt) ||
      !tosa::getConstShapeValues(adaptor.getBorder().getDefiningOp(),
                                 borderInt)) {
    return failure();
  }

  // Compute the output shape based on attributes: scale, offset, and border.
  const int64_t outputHeight =
      (((inputHeight - 1) * scaleInt[0] - offsetInt[0] + borderInt[0]) /
       scaleInt[1]) +
      1;

  const int64_t outputWidth =
      (((inputWidth - 1) * scaleInt[2] - offsetInt[1] + borderInt[1]) /
       scaleInt[3]) +
      1;

  if (outputHeight < 0 || outputWidth < 0) {
    return emitOptionalError(
        location,
        "calculated output height and width must be non-negative, "
        "got height = ",
        outputHeight, ", width = ", outputWidth);
  }

  outputShape[1] = outputHeight;
  outputShape[2] = outputWidth;
  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult tosa::ResizeOp::verify() {
  const Value input = getInput();
  const Value output = getOutput();
  const RankedTensorType inputType =
      llvm::dyn_cast<RankedTensorType>(input.getType());
  const RankedTensorType outputType =
      llvm::dyn_cast<RankedTensorType>(output.getType());

  SmallVector<int64_t> scaleValues;
  SmallVector<int64_t> offsetValues;
  SmallVector<int64_t> borderValues;
  if (!tosa::getConstShapeValues(getScale().getDefiningOp(), scaleValues) ||
      !tosa::getConstShapeValues(getOffset().getDefiningOp(), offsetValues) ||
      !tosa::getConstShapeValues(getBorder().getDefiningOp(), borderValues)) {
    // Skip following checks if shape is not constant
    return success();
  }

  if (llvm::any_of(scaleValues, [](int64_t s) { return s <= 0; }))
    return emitOpError("expect all scale values to be > 0, got ")
           << scaleValues;

  const int64_t scaleYN = scaleValues[0];
  const int64_t scaleYD = scaleValues[1];
  const int64_t scaleXN = scaleValues[2];
  const int64_t scaleXD = scaleValues[3];

  const int64_t offsetY = offsetValues[0];
  const int64_t offsetX = offsetValues[1];

  const int64_t borderY = borderValues[0];
  const int64_t borderX = borderValues[1];

  if (!inputType)
    return success();
  if (!outputType)
    return success();

  const int64_t oh = outputType.getDimSize(1);
  const int64_t ow = outputType.getDimSize(2);
  const int64_t ih = inputType.getDimSize(1);
  const int64_t iw = inputType.getDimSize(2);

  // Don't check with input height that could be broadcast (ih != 1)
  // since Linalg, a consumer of TOSA, expects broadcasting support
  // in resize to be available. Taking the cautious approach for now,
  // we can consider removing support for broadcasting later.
  if (ih != ShapedType::kDynamic && ih != 1) {
    const std::optional<int64_t> calculatedOutHeightMinusOne =
        idivCheck((ih - 1) * scaleYN - offsetY + borderY, scaleYD);
    if (!calculatedOutHeightMinusOne.has_value())
      return emitOpError("expected (input_height - 1) * scale_y_n - offset_y + "
                         "border_y ")
             << "to be wholly divisible by scale_y_d, got ((" << ih
             << " - 1) * " << scaleYN << " - " << offsetY << " + " << borderY
             << ") / " << scaleYD;
    const int64_t calculatedOutHeight = calculatedOutHeightMinusOne.value() + 1;
    if (oh != ShapedType::kDynamic && calculatedOutHeight != oh)
      return emitOpError("calculated output height did not match expected: ")
             << "calculated=" << calculatedOutHeight << ", expected=" << oh;
  }

  // Don't check with input width that could be broadcast (iw != 1)
  // since Linalg, a consumer of TOSA, expects broadcasting support
  // in resize to be available. Taking the cautious approach for now,
  // we can consider removing support for broadcasting later.
  if (iw != ShapedType::kDynamic && iw != 1) {
    const int64_t scaledInWidth = (iw - 1) * scaleXN - offsetX + borderX;
    const std::optional<int64_t> calculatedOutWidthMinusOne =
        idivCheck(scaledInWidth, scaleXD);
    if (!calculatedOutWidthMinusOne.has_value())
      return emitOpError("expected (input_width - 1) * scale_x_n - offset_x + "
                         "border_x ")
             << "to be wholly divisible by scale_x_d, got ((" << iw
             << " - 1) * " << scaleXN << " - " << offsetX << " + " << borderX
             << ") / " << scaleXD;
    const int64_t calculatedOutWidth = calculatedOutWidthMinusOne.value() + 1;
    if (ow != ShapedType::kDynamic && calculatedOutWidth != ow)
      return emitOpError("calculated output width did not match expected: ")
             << "calculated=" << calculatedOutWidth << ", expected=" << ow;
  }

  return success();
}

LogicalResult tosa::ScatterOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    ScatterOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outputShape;
  outputShape.resize(3, ShapedType::kDynamic);

  ShapeAdaptor valuesInShape(adaptor.getValuesIn().getType());
  if (valuesInShape.hasRank()) {
    outputShape[0] = valuesInShape.getDimSize(0);
    outputShape[1] = valuesInShape.getDimSize(1);
    outputShape[2] = valuesInShape.getDimSize(2);
  }

  ShapeAdaptor indicesShape(adaptor.getIndices().getType());
  if (indicesShape.hasRank()) {
    if (outputShape[0] == ShapedType::kDynamic)
      outputShape[0] = indicesShape.getDimSize(0);
  }

  ShapeAdaptor inputShape(adaptor.getInput().getType());
  if (inputShape.hasRank()) {
    if (outputShape[0] == ShapedType::kDynamic)
      outputShape[0] = inputShape.getDimSize(0);
    if (outputShape[2] == ShapedType::kDynamic)
      outputShape[2] = inputShape.getDimSize(2);
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult tosa::ScatterOp::verify() {
  if (verifySameElementTypes(*this, /* inType = */ getValuesIn().getType(),
                             /* outType = */ getValuesOut().getType())
          .failed() ||
      verifySameElementTypes(*this, /* inType = */ getInput().getType(),
                             /* outType = */ getValuesOut().getType())
          .failed()) {
    return failure();
  }

  const ShapeAdaptor valuesInShape(getValuesIn().getType());
  const ShapeAdaptor indicesShape(getIndices().getType());
  const ShapeAdaptor inputShape(getInput().getType());
  const ShapeAdaptor outputShape(getValuesOut().getType());

  int64_t N = ShapedType::kDynamic;
  int64_t K = ShapedType::kDynamic;
  int64_t W = ShapedType::kDynamic;
  int64_t C = ShapedType::kDynamic;
  if (valuesInShape.hasRank()) {
    N = valuesInShape.getDimSize(0);
    K = valuesInShape.getDimSize(1);
    C = valuesInShape.getDimSize(2);
  }
  if (indicesShape.hasRank()) {
    const int64_t indicesN = indicesShape.getDimSize(0);
    W = indicesShape.getDimSize(1);
    if (N == ShapedType::kDynamic)
      N = indicesN;
    else if (indicesN != ShapedType::kDynamic && N != indicesN)
      return emitOpError() << "requires indices dimension 0 to have size " << N
                           << ", got " << indicesN;
  }
  if (inputShape.hasRank()) {
    const int64_t inputN = inputShape.getDimSize(0);
    const int64_t inputW = inputShape.getDimSize(1);
    const int64_t inputC = inputShape.getDimSize(2);
    if (N == ShapedType::kDynamic)
      N = inputN;
    else if (inputN != ShapedType::kDynamic && N != inputN)
      return emitOpError() << "requires input dimension 0 to have size " << N
                           << ", got " << inputN;
    if (W == ShapedType::kDynamic)
      W = inputW;
    else if (inputW != ShapedType::kDynamic && W != inputW)
      return emitOpError() << "requires input dimension 1 to have size " << W
                           << ", got " << inputW;

    if (C == ShapedType::kDynamic)
      C = inputC;
    else if (inputC != ShapedType::kDynamic && C != inputC)
      return emitOpError() << "requires input dimension 2 to have size " << C
                           << ", got " << inputC;
  }
  if (outputShape.hasRank()) {
    const int64_t outputN = outputShape.getDimSize(0);
    const int64_t outputK = outputShape.getDimSize(1);
    const int64_t outputC = outputShape.getDimSize(2);
    if (N != ShapedType::kDynamic && outputN != ShapedType::kDynamic &&
        N != outputN)
      return emitOpError() << "requires values_out dimension 0 to have size "
                           << N << ", got " << outputN;
    if (K == ShapedType::kDynamic)
      K = outputK;
    else if (outputK != ShapedType::kDynamic && K != outputK)
      return emitOpError() << "requires values_out dimension 1 to have size "
                           << K << ", got " << outputK;
    if (C != ShapedType::kDynamic && outputC != ShapedType::kDynamic &&
        C != outputC)
      return emitOpError() << "requires values_out dimension 2 to have size "
                           << C << ", got " << outputC;
  }
  if (K != ShapedType::kDynamic && W != ShapedType::kDynamic && !(K >= W))
    return emitOpError() << "requires dimensions K >= W, got K=" << K
                         << " and W=" << W;

  return success();
}

static LogicalResult ReduceInferReturnTypes(
    ShapeAdaptor operandShape, Type inputType, IntegerAttr axis,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  int64_t axisVal = axis.getValue().getSExtValue();
  if (!operandShape.hasRank() || operandShape.getRank() <= axisVal) {
    inferredReturnShapes.push_back(ShapedTypeComponents(inputType));
    return success();
  }

  SmallVector<int64_t> outputShape;
  operandShape.getDims(outputShape);
  outputShape[axisVal] = 1;
  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape, inputType));
  return success();
}

#define COMPATIBLE_RETURN_TYPES(OP)                                            \
  bool OP::isCompatibleReturnTypes(TypeRange l, TypeRange r) {                 \
    if (l.size() != r.size() || l.size() != 1)                                 \
      return false;                                                            \
    if (getElementTypeOrSelf(l[0]) != getElementTypeOrSelf(r[0]))              \
      return false;                                                            \
    return succeeded(verifyCompatibleShape(l[0], r[0]));                       \
  }

#define REDUCE_SHAPE_INFER(OP)                                                 \
  LogicalResult OP::inferReturnTypeComponents(                                 \
      MLIRContext *context, ::std::optional<Location> location,                \
      OP::Adaptor adaptor,                                                     \
      SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {           \
    Type inputType =                                                           \
        llvm::cast<TensorType>(adaptor.getInput().getType()).getElementType(); \
    ShapeAdaptor inputShape(adaptor.getInput().getType());                     \
    const Properties &prop = adaptor.getProperties();                          \
    return ReduceInferReturnTypes(inputShape, inputType, prop.axis,            \
                                  inferredReturnShapes);                       \
  }                                                                            \
  COMPATIBLE_RETURN_TYPES(OP)

REDUCE_SHAPE_INFER(tosa::ReduceAllOp)
REDUCE_SHAPE_INFER(tosa::ReduceAnyOp)
REDUCE_SHAPE_INFER(tosa::ReduceMaxOp)
REDUCE_SHAPE_INFER(tosa::ReduceMinOp)
REDUCE_SHAPE_INFER(tosa::ReduceProductOp)
REDUCE_SHAPE_INFER(tosa::ReduceSumOp)
#undef REDUCE_SHAPE_INFER
COMPATIBLE_RETURN_TYPES(tosa::ConcatOp)
#undef COMPATIBLE_RETURN_TYPES

template <typename T>
static LogicalResult verifyReduceOp(T op) {
  // All TOSA reduce Ops have input, output and axis.
  TensorType inputType = op.getInput().getType();
  TensorType outputType = op.getOutput().getType();
  int32_t reduceAxis = op.getAxis();

  if (reduceAxis < 0) {
    op.emitOpError("reduce axis must not be negative");
    return failure();
  }
  if (inputType.hasRank()) {
    int64_t inputRank = inputType.getRank();
    // We allow for a special case where the input/output shape has rank 0 and
    // axis is also 0.
    if (reduceAxis >= inputRank && !(reduceAxis == 0 && inputRank == 0)) {
      op.emitOpError("expect input tensor rank (")
          << inputRank << ") to be larger than reduce axis (" << reduceAxis
          << ")";
      return failure();
    }
  }
  if (outputType.hasRank()) {
    int64_t outputRank = outputType.getRank();
    if (inputType.hasRank() && outputRank != inputType.getRank()) {
      op.emitOpError(
          "expect output tensor rank to be equal to input tensor rank");
      return failure();
    }
    if (reduceAxis >= outputRank && !(reduceAxis == 0 && outputRank == 0)) {
      op.emitOpError("expect output tensor rank (")
          << outputRank << ") to be larger than reduce axis (" << reduceAxis
          << ")";
      return failure();
    }
    // We can only verify the reduced dimension size to be 1 if this is not
    // the special case of output rank == 0.
    if (outputRank != 0) {
      auto outputShape = outputType.getShape();
      if (!outputType.isDynamicDim(reduceAxis) &&
          outputShape[reduceAxis] != 1) {
        op.emitOpError("expect reduced dimension size to be 1, got ")
            << outputShape[reduceAxis];
        return failure();
      }
    }
  }
  return success();
}

LogicalResult tosa::ReduceAllOp::verify() { return verifyReduceOp(*this); }
LogicalResult tosa::ReduceAnyOp::verify() { return verifyReduceOp(*this); }
LogicalResult tosa::ReduceMaxOp::verify() { return verifyReduceOp(*this); }
LogicalResult tosa::ReduceMinOp::verify() { return verifyReduceOp(*this); }
LogicalResult tosa::ReduceProductOp::verify() { return verifyReduceOp(*this); }
LogicalResult tosa::ReduceSumOp::verify() { return verifyReduceOp(*this); }

static LogicalResult NAryInferReturnTypes(
    const ValueShapeRange &operands,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outShape;
  if (resolveBroadcastShape(operands, outShape).failed()) {
    inferredReturnShapes.push_back(ShapedTypeComponents());
  } else {
    inferredReturnShapes.push_back(ShapedTypeComponents(outShape));
  }
  return success();
}

#define NARY_SHAPE_INFER(OP)                                                   \
  LogicalResult OP::inferReturnTypeComponents(                                 \
      MLIRContext *context, ::std::optional<Location> location,                \
      ValueShapeRange operands, DictionaryAttr attributes,                     \
      OpaqueProperties properties, RegionRange regions,                        \
      SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {           \
    return NAryInferReturnTypes(operands, inferredReturnShapes);               \
  }

NARY_SHAPE_INFER(tosa::AbsOp)
NARY_SHAPE_INFER(tosa::AddOp)
NARY_SHAPE_INFER(tosa::ArithmeticRightShiftOp)
NARY_SHAPE_INFER(tosa::BitwiseAndOp)
NARY_SHAPE_INFER(tosa::BitwiseOrOp)
NARY_SHAPE_INFER(tosa::BitwiseXorOp)
NARY_SHAPE_INFER(tosa::BitwiseNotOp)
NARY_SHAPE_INFER(tosa::CastOp)
NARY_SHAPE_INFER(tosa::CeilOp)
NARY_SHAPE_INFER(tosa::ClampOp)
NARY_SHAPE_INFER(tosa::ClzOp)
NARY_SHAPE_INFER(tosa::CosOp)
NARY_SHAPE_INFER(tosa::ExpOp)
NARY_SHAPE_INFER(tosa::FloorOp)
NARY_SHAPE_INFER(tosa::GreaterEqualOp)
NARY_SHAPE_INFER(tosa::GreaterOp)
NARY_SHAPE_INFER(tosa::IdentityOp)
NARY_SHAPE_INFER(tosa::IntDivOp)
NARY_SHAPE_INFER(tosa::LogOp)
NARY_SHAPE_INFER(tosa::LogicalAndOp)
NARY_SHAPE_INFER(tosa::LogicalLeftShiftOp)
NARY_SHAPE_INFER(tosa::LogicalNotOp)
NARY_SHAPE_INFER(tosa::LogicalOrOp)
NARY_SHAPE_INFER(tosa::LogicalRightShiftOp)
NARY_SHAPE_INFER(tosa::LogicalXorOp)
NARY_SHAPE_INFER(tosa::MaximumOp)
NARY_SHAPE_INFER(tosa::MinimumOp)
NARY_SHAPE_INFER(tosa::PowOp)
NARY_SHAPE_INFER(tosa::ReciprocalOp)
NARY_SHAPE_INFER(tosa::ReverseOp)
NARY_SHAPE_INFER(tosa::RsqrtOp)
NARY_SHAPE_INFER(tosa::SinOp)
NARY_SHAPE_INFER(tosa::SelectOp)
NARY_SHAPE_INFER(tosa::SubOp)
NARY_SHAPE_INFER(tosa::TanhOp)
NARY_SHAPE_INFER(tosa::ErfOp)
NARY_SHAPE_INFER(tosa::SigmoidOp)
#undef PRED_SHAPE_INFER

LogicalResult tosa::NegateOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    NegateOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape(adaptor.getInput1().getType());
  inferredReturnShapes.push_back(ShapedTypeComponents(inputShape));
  return success();
}

LogicalResult tosa::NegateOp::verify() {
  // Verify same element type
  const Type input1Type = getInput1().getType();
  const Type outputType = getOutput().getType();
  if (verifySameElementTypes(*this, input1Type, outputType).failed())
    return failure();

  // Verify same shape
  const SmallVector<Type, 2> types = {input1Type, outputType};
  if (failed(verifyCompatibleShapes(types)))
    return emitOpError() << "requires the same shape for input1 and output";

  const Type input1EType = getStorageElementTypeOrSelf(getInput1().getType());
  const Type input1ZpEType =
      getStorageElementTypeOrSelf(getInput1Zp().getType());
  if (input1EType != input1ZpEType) {
    return emitOpError("expect both input1 and its zero point are the same "
                       "element type, got ")
           << input1EType << " and " << input1ZpEType;
  }
  const Type outputEType = getStorageElementTypeOrSelf(getOutput().getType());
  const Type outputZpEType =
      getStorageElementTypeOrSelf(getOutputZp().getType());
  if (outputEType != outputZpEType) {
    return emitOpError("expect both output and its zero point are the same "
                       "element type, got ")
           << outputEType << " and " << outputZpEType;
  }

  FailureOr<int64_t> maybeIZp = getInput1ZeroPoint();
  if (succeeded(maybeIZp) && verifyInput1ZeroPoint(*maybeIZp).failed())
    return failure();

  FailureOr<int64_t> maybeOZp = getOutputZeroPoint();
  if (succeeded(maybeOZp) && verifyOutputZeroPoint(*maybeOZp).failed())
    return failure();

  return success();
}

static LogicalResult poolingInferReturnTypes(
    ShapeAdaptor inputShape, ArrayRef<int64_t> kernel, ArrayRef<int64_t> stride,
    ArrayRef<int64_t> pad,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outputShape;
  outputShape.resize(4, ShapedType::kDynamic);

  // We only know the rank if the input type is unranked.
  if (!inputShape) {
    inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
    return success();
  }

  // Batch and number of channels are identical for pooling layer.
  outputShape[0] = inputShape.getDimSize(0);
  outputShape[3] = inputShape.getDimSize(3);

  int64_t height = inputShape.getDimSize(1);
  int64_t width = inputShape.getDimSize(2);

  if (ShapedType::isStatic(height)) {
    int64_t padded = height + pad[0] + pad[1] - kernel[0];
    outputShape[1] = padded / stride[0] + 1;
  }

  if (ShapedType::isStatic(width)) {
    int64_t padded = width + pad[2] + pad[3] - kernel[1];
    outputShape[2] = padded / stride[1] + 1;
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult Conv2DOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    Conv2DOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outputShape(4, ShapedType::kDynamic);

  int64_t inputWidth = ShapedType::kDynamic;
  int64_t inputHeight = ShapedType::kDynamic;
  int64_t weightWidth = ShapedType::kDynamic;
  int64_t weightHeight = ShapedType::kDynamic;

  // Input shape describes input width/height and batch.

  ShapeAdaptor inputShape(adaptor.getInput().getType());
  if (inputShape.hasRank()) {
    outputShape[0] = inputShape.getDimSize(0);
    inputHeight = inputShape.getDimSize(1);
    inputWidth = inputShape.getDimSize(2);
  }

  // Weight shapes describes the filter width/height and the output channels.
  ShapeAdaptor weightShape(adaptor.getWeight().getType());
  if (weightShape.hasRank()) {
    outputShape[3] = weightShape.getDimSize(0);
    weightHeight = weightShape.getDimSize(1);
    weightWidth = weightShape.getDimSize(2);
  }

  // Bias shape can describe the output channels.
  ShapeAdaptor biasShape(adaptor.getBias().getType());
  if (biasShape.hasRank()) {
    outputShape[3] = ShapedType::isDynamic(outputShape[3])
                         ? biasShape.getDimSize(0)
                         : outputShape[3];
  }

  llvm::ArrayRef<int64_t> dilation = adaptor.getDilation();
  llvm::ArrayRef<int64_t> stride = adaptor.getStride();
  llvm::ArrayRef<int64_t> padding = adaptor.getPad();

  if (ShapedType::isStatic(inputHeight) && ShapedType::isStatic(weightHeight)) {
    int64_t inputSize = inputHeight + padding[0] + padding[1];
    int64_t filterSize = (weightHeight - 1) * dilation[0] + 1;
    int64_t unstridedResult = inputSize - filterSize + 1;
    outputShape[1] = (unstridedResult - 1) / stride[0] + 1;
  }

  if (ShapedType::isStatic(inputWidth) && ShapedType::isStatic(weightWidth)) {
    int64_t inputSize = inputWidth + padding[2] + padding[3];
    int64_t filterSize = (weightWidth - 1) * dilation[1] + 1;
    int64_t unstridedResult = inputSize - filterSize + 1;
    outputShape[2] = (unstridedResult - 1) / stride[1] + 1;
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult Conv2DOp::verify() {
  if (verifyConvOp(*this).failed() || verifyConvOpModes(*this).failed() ||
      verifyConvOpErrorIf(*this).failed())
    return failure();
  return success();
}

LogicalResult Conv3DOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    Conv3DOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outputShape(5, ShapedType::kDynamic);

  int64_t inputWidth = ShapedType::kDynamic;
  int64_t inputHeight = ShapedType::kDynamic;
  int64_t inputDepth = ShapedType::kDynamic;

  int64_t weightWidth = ShapedType::kDynamic;
  int64_t weightHeight = ShapedType::kDynamic;
  int64_t weightDepth = ShapedType::kDynamic;

  // Input shape describes input width/height and batch.
  ShapeAdaptor inputShape(adaptor.getInput().getType());
  if (inputShape.hasRank()) {
    outputShape[0] = inputShape.getDimSize(0);
    inputDepth = inputShape.getDimSize(1);
    inputHeight = inputShape.getDimSize(2);
    inputWidth = inputShape.getDimSize(3);
  }

  // Weight shapes describes the filter width/height and the output channels.
  ShapeAdaptor weightShape(adaptor.getWeight().getType());
  if (weightShape.hasRank()) {
    outputShape[4] = weightShape.getDimSize(0);
    weightDepth = weightShape.getDimSize(1);
    weightHeight = weightShape.getDimSize(2);
    weightWidth = weightShape.getDimSize(3);
  }

  // Bias shape can describe the output channels.
  ShapeAdaptor biasShape(adaptor.getBias().getType());
  if (biasShape.hasRank() && ShapedType::isDynamic(outputShape[4])) {
    outputShape[4] = biasShape.getDimSize(0);
  }

  llvm::ArrayRef<int64_t> dilation = adaptor.getDilation();
  llvm::ArrayRef<int64_t> stride = adaptor.getStride();
  llvm::ArrayRef<int64_t> pad = adaptor.getPad();

  if (ShapedType::isStatic(inputDepth) && ShapedType::isStatic(weightDepth)) {
    int32_t inputSize = inputDepth + pad[0] + pad[1];
    int32_t filterSize = (weightDepth - 1) * dilation[0] + 1;
    int32_t unstridedResult = inputSize - filterSize + 1;
    outputShape[1] = (unstridedResult - 1) / stride[0] + 1;
  }

  if (ShapedType::isStatic(inputHeight) && ShapedType::isStatic(weightHeight)) {
    int32_t inputSize = inputHeight + pad[2] + pad[3];
    int32_t filterSize = (weightHeight - 1) * dilation[1] + 1;
    int32_t unstridedResult = inputSize - filterSize + 1;
    outputShape[2] = (unstridedResult - 1) / stride[1] + 1;
  }

  if (ShapedType::isStatic(inputWidth) && ShapedType::isStatic(weightWidth)) {
    int32_t inputSize = inputWidth + pad[4] + pad[5];
    int32_t filterSize = (weightWidth - 1) * dilation[2] + 1;
    int32_t unstridedResult = inputSize - filterSize + 1;
    outputShape[3] = (unstridedResult - 1) / stride[2] + 1;
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult Conv3DOp::verify() {
  if (verifyConvOp(*this).failed() || verifyConvOpModes(*this).failed() ||
      verifyConvOpErrorIf(*this).failed())
    return failure();
  return success();
}

LogicalResult AvgPool2dOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    AvgPool2dOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape(adaptor.getInput().getType());
  const Properties &prop = adaptor.getProperties();
  return poolingInferReturnTypes(inputShape, prop.kernel, prop.stride, prop.pad,
                                 inferredReturnShapes);
}

LogicalResult MaxPool2dOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    MaxPool2dOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape(adaptor.getInput().getType());
  const Properties &prop = adaptor.getProperties();
  return poolingInferReturnTypes(inputShape, prop.kernel, prop.stride, prop.pad,
                                 inferredReturnShapes);
}

LogicalResult MaxPool2dOp::verify() {
  if (failed(verifySameElementTypes(*this, /* intype = */ getInput().getType(),
                                    /* outType = */ getOutput().getType())))
    return failure();

  if (failed(verifyPoolingOp(*this)))
    return failure();

  return success();
}

LogicalResult DepthwiseConv2DOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    DepthwiseConv2DOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outputShape(4, ShapedType::kDynamic);

  int64_t inputWidth = ShapedType::kDynamic;
  int64_t inputHeight = ShapedType::kDynamic;
  int64_t inputChannels = ShapedType::kDynamic;

  int64_t weightWidth = ShapedType::kDynamic;
  int64_t weightHeight = ShapedType::kDynamic;
  int64_t depthChannels = ShapedType::kDynamic;

  // Input shape describes input width/height and batch.
  ShapeAdaptor inputShape(adaptor.getInput().getType());
  if (inputShape.hasRank()) {
    outputShape[0] = inputShape.getDimSize(0);
    inputHeight = inputShape.getDimSize(1);
    inputWidth = inputShape.getDimSize(2);
    inputChannels = inputShape.getDimSize(3);
  }

  // Weight shapes describes the filter width/height and the output channels.
  ShapeAdaptor weightShape(adaptor.getWeight().getType());
  if (weightShape.hasRank()) {
    weightHeight = weightShape.getDimSize(0);
    weightWidth = weightShape.getDimSize(1);
    inputChannels = ShapedType::isDynamic(inputChannels)
                        ? weightShape.getDimSize(2)
                        : inputChannels;
    depthChannels = weightShape.getDimSize(3);
  }

  // If both inputChannels and depthChannels are available we can determine
  // the output channels.
  if (ShapedType::isStatic(inputChannels) &&
      ShapedType::isStatic(depthChannels)) {
    outputShape[3] = inputChannels * depthChannels;
  }

  // Bias shape can describe the output channels.
  ShapeAdaptor biasShape(adaptor.getBias().getType());
  if (biasShape.hasRank()) {
    outputShape[3] = ShapedType::isDynamic(outputShape[3])
                         ? biasShape.getDimSize(0)
                         : outputShape[3];
  }

  llvm::ArrayRef<int64_t> dilation = adaptor.getDilation();
  llvm::ArrayRef<int64_t> padding = adaptor.getPad();
  llvm::ArrayRef<int64_t> stride = adaptor.getStride();

  if (ShapedType::isStatic(inputHeight) && ShapedType::isStatic(weightHeight)) {
    int64_t inputSize = inputHeight + padding[0] + padding[1];
    int64_t filterSize = (weightHeight - 1) * dilation[0] + 1;
    int64_t unstridedResult = inputSize - filterSize + 1;
    outputShape[1] = (unstridedResult - 1) / stride[0] + 1;
  }

  if (ShapedType::isStatic(inputWidth) && ShapedType::isStatic(weightWidth)) {
    int64_t inputSize = inputWidth + padding[2] + padding[3];
    int64_t filterSize = (weightWidth - 1) * dilation[1] + 1;
    int64_t unstridedResult = inputSize - filterSize + 1;
    outputShape[2] = (unstridedResult - 1) / stride[1] + 1;
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult DepthwiseConv2DOp::verify() {
  if (verifyConvOp(*this).failed() || verifyConvOpModes(*this).failed() ||
      verifyConvOpErrorIf(*this).failed())
    return failure();
  return success();
}

LogicalResult TransposeConv2DOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    TransposeConv2DOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<int64_t> outputShape(4, ShapedType::kDynamic);

  int64_t inputWidth = ShapedType::kDynamic;
  int64_t inputHeight = ShapedType::kDynamic;
  int64_t weightWidth = ShapedType::kDynamic;
  int64_t weightHeight = ShapedType::kDynamic;

  // Input shape describes input width/height and batch.
  ShapeAdaptor inputShape(adaptor.getInput().getType());
  if (inputShape.hasRank()) {
    outputShape[0] = ShapedType::isDynamic(outputShape[0])
                         ? inputShape.getDimSize(0)
                         : outputShape[0];
    inputHeight = inputShape.getDimSize(1);
    inputWidth = inputShape.getDimSize(2);
  }

  // Weight shapes describes the filter width/height and the output channels.
  ShapeAdaptor weightShape(adaptor.getWeight().getType());
  if (weightShape.hasRank()) {
    outputShape[3] = ShapedType::isDynamic(outputShape[3])
                         ? weightShape.getDimSize(0)
                         : outputShape[3];
    weightHeight = weightShape.getDimSize(1);
    weightWidth = weightShape.getDimSize(2);
  }

  // Bias shape can describe the output channels.
  ShapeAdaptor biasShape(adaptor.getInput().getType());
  if (biasShape.hasRank()) {
    outputShape[3] = ShapedType::isDynamic(outputShape[3])
                         ? biasShape.getDimSize(0)
                         : outputShape[3];
  }

  llvm::ArrayRef<int64_t> padding = adaptor.getOutPad();
  llvm::ArrayRef<int64_t> stride = adaptor.getStride();

  if (ShapedType::isStatic(inputHeight) && ShapedType::isStatic(weightHeight)) {
    int64_t calculateSize =
        (inputHeight - 1) * stride[0] + padding[0] + padding[1] + weightHeight;
    outputShape[1] =
        ShapedType::isDynamic(outputShape[1]) ? calculateSize : outputShape[1];
  }

  if (ShapedType::isStatic(inputWidth) && ShapedType::isStatic(weightWidth)) {
    int64_t calculateSize =
        (inputWidth - 1) * stride[1] + padding[2] + padding[3] + weightWidth;
    outputShape[2] =
        ShapedType::isDynamic(outputShape[2]) ? calculateSize : outputShape[2];
  }

  inferredReturnShapes.push_back(ShapedTypeComponents(outputShape));
  return success();
}

LogicalResult TransposeConv2DOp::verify() {
  if (verifyConvOp(*this).failed() || verifyConvOpModes(*this).failed())
    return failure();

  const llvm::ArrayRef<int64_t> strides = getStride();
  const int64_t strideY = strides[0];
  const int64_t strideX = strides[1];

  if (strideY < 1 || strideX < 1)
    return emitOpError("expect all stride values to be >= 1, got [")
           << strides << "]";

  const auto checkPadAgainstKernelDim =
      [this](int64_t pad_value, int64_t kernel_dim_size,
             llvm::StringRef pad_name,
             llvm::StringRef kernel_dim_name) -> LogicalResult {
    if (pad_value <= -kernel_dim_size)
      return emitOpError("expected ")
             << pad_name << " > -" << kernel_dim_name
             << ", but got: " << pad_name << "=" << pad_value << " and "
             << kernel_dim_name << "=" << kernel_dim_size;
    return success();
  };

  const llvm::ArrayRef<int64_t> padding = getOutPad();
  const int64_t outPadTop = padding[0];
  const int64_t outPadBottom = padding[1];
  const int64_t outPadLeft = padding[2];
  const int64_t outPadRight = padding[3];

  const auto weightType =
      llvm::dyn_cast<RankedTensorType>(getWeight().getType());

  if (weightType) {
    const int64_t kernelHeight = weightType.getDimSize(1);
    if (ShapedType::isStatic(kernelHeight)) {
      if (failed(checkPadAgainstKernelDim(outPadTop, kernelHeight,
                                          "out_pad_top", "KH")))
        return failure();

      if (failed(checkPadAgainstKernelDim(outPadBottom, kernelHeight,
                                          "out_pad_bottom", "KH")))
        return failure();
    }

    const int64_t kernelWidth = weightType.getDimSize(2);
    if (ShapedType::isStatic(kernelWidth)) {
      if (failed(checkPadAgainstKernelDim(outPadLeft, kernelWidth,
                                          "out_pad_left", "KW")))
        return failure();

      if (failed(checkPadAgainstKernelDim(outPadRight, kernelWidth,
                                          "out_pad_right", "KW")))
        return failure();
    }
  }

  // Rest of the checks depend on the output type being a RankedTensorType
  const auto outputType =
      llvm::dyn_cast<RankedTensorType>(getOutput().getType());
  if (!outputType)
    return success();

  const auto inputType = llvm::dyn_cast<RankedTensorType>(getInput().getType());
  if (inputType && weightType) {
    const int64_t inputHeight = inputType.getDimSize(1);
    const int64_t kernelHeight = weightType.getDimSize(1);
    const int64_t outputHeight = outputType.getDimSize(1);

    if (ShapedType::isStatic(inputHeight) &&
        ShapedType::isStatic(outputHeight)) {
      if (outputHeight !=
          (inputHeight - 1) * strideY + outPadTop + outPadBottom + kernelHeight)
        return emitOpError(
                   "dimension mismatch: expected OH == (IH - 1) * stride_y "
                   "+ out_pad_top + out_pad_bottom + KH, but got ")
               << outputHeight << " != (" << inputHeight << " - 1) * "
               << strideY << " + " << outPadTop << " + " << outPadBottom
               << " + " << kernelHeight;
    }

    const int64_t inputWidth = inputType.getDimSize(2);
    const int64_t kernelWidth = weightType.getDimSize(2);
    const int64_t outputWidth = outputType.getDimSize(2);

    if (ShapedType::isStatic(inputWidth) && ShapedType::isStatic(outputWidth)) {
      if (outputWidth !=
          (inputWidth - 1) * strideX + outPadLeft + outPadRight + kernelWidth)
        return emitOpError(
                   "dimension mismatch: expected OW == (IW - 1) * stride_x "
                   "+ out_pad_left + out_pad_right + KW, but got ")
               << outputWidth << " != (" << inputWidth << " - 1) * " << strideX
               << " + " << outPadLeft << " + " << outPadRight << " + "
               << kernelWidth;
    }
  }

  const auto biasType = llvm::dyn_cast<RankedTensorType>(getBias().getType());

  if (!biasType)
    return success();

  const int64_t biasChannels = biasType.getDimSize(0);

  // Skip further checks if bias is dynamic
  if (biasChannels == ShapedType::kDynamic)
    return success();

  const int64_t outputChannels = outputType.getDimSize(3);
  if (!ShapedType::isDynamic(outputChannels) &&
      biasChannels != outputChannels && biasChannels != 1)
    return emitOpError(
               "bias channels expected to be equal to output channels (")
           << outputChannels << ") or 1, got " << biasChannels;

  return success();
}

LogicalResult RescaleOp::verify() {
  auto inputType = llvm::dyn_cast<ShapedType>(getInput().getType());
  if (!inputType) {
    emitOpError("expect shaped tensor for input, got ") << getInput().getType();
    return failure();
  }

  auto inputElementType =
      getStorageElementTypeOrSelf(inputType.getElementType());
  if (!mlir::isa<IntegerType>(inputElementType)) {
    emitOpError("expect input to have integer element type, got ")
        << inputElementType;
    return failure();
  }

  auto outputType = llvm::dyn_cast<ShapedType>(getOutput().getType());
  if (!outputType) {
    emitOpError("expect shaped tensor for output, got ")
        << getOutput().getType();
    return failure();
  }

  auto outputElementType =
      getStorageElementTypeOrSelf(outputType.getElementType());
  if (!mlir::isa<IntegerType>(outputElementType)) {
    emitOpError("expect output to have integer element type, got ")
        << outputElementType;
    return failure();
  }

  if (verifyRescaleValueAndZpTypes(*this, getInput(), getInputZp(), "input")
          .failed())
    return failure();

  if (verifyRescaleValueAndZpTypes(*this, getOutput(), getOutputZp(), "output")
          .failed())
    return failure();

  FailureOr<int64_t> maybeIZp = getInputZeroPoint();
  if (succeeded(maybeIZp) && verifyInputZeroPoint(*maybeIZp).failed())
    return failure();

  FailureOr<int64_t> maybeOZp = getOutputZeroPoint();
  if (succeeded(maybeOZp) && verifyOutputZeroPoint(*maybeOZp).failed())
    return failure();

  auto multiplierType = llvm::dyn_cast<ShapedType>(getMultiplier().getType());
  if (!multiplierType) {
    emitOpError("expect shaped tensor for multiplier, got ")
        << getMultiplier().getType();
    return failure();
  }

  auto shiftType = llvm::dyn_cast<ShapedType>(getShift().getType());
  if (!shiftType) {
    emitOpError("expect shaped tensor for shift, got ") << getShift().getType();
    return failure();
  }

  // multiplier element type must be i32 for scale32 = true
  if (getScale32() && !multiplierType.getElementType().isInteger(32)) {
    emitOpError("expect i32 element type for multiplier for scale32=true, got ")
        << multiplierType.getElementType();
    return failure();
  }

  // multiplier element type must be i16 for scale32 = false
  if (!getScale32() && !multiplierType.getElementType().isInteger(16)) {
    emitOpError(
        "expect i16 element type for multiplier for scale32=false, got ")
        << multiplierType.getElementType();
    return failure();
  }

  if (!inputType.hasRank())
    return success();

  // multiplier/shift must have shape = {numChannels},
  // where numChannel is 1 if per_channel = false
  // otherwise numChannel is dimension in input shape's last axis
  int64_t numChannels = 1;
  if (getPerChannel()) {
    if (inputType.getRank() < 1) {
      emitOpError("requires input to be at least rank 1 when per_channel is "
                  "true, but got rank ")
          << inputType.getRank();
      return failure();
    }
    numChannels = inputType.getDimSize(inputType.getRank() - 1);
  }

  if (!multiplierType.hasRank())
    return success();

  ArrayRef<int64_t> multiplierShape = multiplierType.getShape();
  // multiplier input has rank 1 by dialect definition
  if (multiplierShape[0] != ShapedType::kDynamic &&
      multiplierShape[0] != numChannels) {
    emitOpError("expect shape of { ")
        << numChannels << " } for multiplier input, got { "
        << multiplierShape[0] << " }";
    return failure();
  }

  if (!shiftType.hasRank())
    return success();

  ArrayRef<int64_t> shiftShape = shiftType.getShape();
  // shift input has rank 1 by dialect definition
  if (shiftShape[0] != ShapedType::kDynamic && shiftShape[0] != numChannels) {
    emitOpError("expect shape of { ")
        << numChannels << " } for shift input, got { " << shiftShape[0] << " }";
    return failure();
  }

  return success();
}

LogicalResult RescaleOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    RescaleOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  ShapeAdaptor inputShape(adaptor.getInput().getType());
  inferredReturnShapes.push_back(ShapedTypeComponents(inputShape));
  return success();
}

LogicalResult IfOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    IfOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<tosa::YieldOp> yieldOps;
  for (Region *region : adaptor.getRegions()) {
    for (auto &block : *region)
      if (auto returnOp = dyn_cast<tosa::YieldOp>(block.getTerminator()))
        yieldOps.push_back(returnOp);
  }

  if (yieldOps.empty())
    return failure();

  // Get the initial type information for the yield op.
  llvm::SmallVector<ValueKnowledge> resultKnowledge;
  resultKnowledge.reserve(yieldOps.front().getNumOperands());
  for (auto operand : yieldOps.front().getOperands()) {
    resultKnowledge.push_back(
        ValueKnowledge::getKnowledgeFromType(operand.getType()));
  }

  for (auto yieldOp : yieldOps) {
    if (resultKnowledge.size() != yieldOp.getNumOperands())
      return failure();

    for (const auto &it : llvm::enumerate(yieldOp.getOperands())) {
      int32_t index = it.index();
      auto meet = ValueKnowledge::meet(
          resultKnowledge[index],
          ValueKnowledge::getKnowledgeFromType(it.value().getType()));
      if (!meet)
        continue;
      resultKnowledge[index] = meet;
    }
  }

  for (const ValueKnowledge &result : resultKnowledge) {
    inferredReturnShapes.push_back(result.getShapedTypeComponents());
  }

  return success();
}

LogicalResult WhileOp::inferReturnTypeComponents(
    MLIRContext *context, ::std::optional<Location> location,
    WhileOp::Adaptor adaptor,
    SmallVectorImpl<ShapedTypeComponents> &inferredReturnShapes) {
  llvm::SmallVector<tosa::YieldOp> yieldOps;
  for (auto &block : adaptor.getBodyGraph())
    if (auto returnOp = dyn_cast<tosa::YieldOp>(block.getTerminator()))
      yieldOps.push_back(returnOp);

  // TOSA's while must have a tosa.yield as its terminator. If not found this
  // tosa.while is invalid.
  if (yieldOps.empty())
    return failure();

  // Get the initial type information from the operand types.
  llvm::SmallVector<ValueKnowledge> resultKnowledge;
  resultKnowledge.reserve(yieldOps.front().getNumOperands());
  for (auto operand : yieldOps.front().getOperands()) {
    resultKnowledge.push_back(
        ValueKnowledge::getKnowledgeFromType(operand.getType()));
  }

  for (auto yieldOp : yieldOps) {
    if (resultKnowledge.size() != yieldOp.getNumOperands())
      return failure();

    for (const auto &it : llvm::enumerate(yieldOp.getOperands())) {
      int32_t index = it.index();
      if (auto meet = ValueKnowledge::meet(
              resultKnowledge[index],
              ValueKnowledge::getKnowledgeFromType(it.value().getType()))) {
        resultKnowledge[index] = meet;
      }
    }
  }

  for (const ValueKnowledge &result : resultKnowledge) {
    inferredReturnShapes.push_back(result.getShapedTypeComponents());
  }

  return success();
}

std::optional<SmallVector<int64_t, 4>> ApplyScaleOp::getShapeForUnroll() {
  if (auto vt = llvm::dyn_cast<VectorType>(getType()))
    return llvm::to_vector<4>(vt.getShape());
  return std::nullopt;
}

static void printInitializationList(OpAsmPrinter &parser,
                                    Block::BlockArgListType blocksArgs,
                                    ValueRange initializers,
                                    StringRef prefix = "") {
  assert(blocksArgs.size() == initializers.size() &&
         "expected same length of arguments and initializers");
  if (initializers.empty())
    return;

  parser << prefix << '(';
  llvm::interleaveComma(
      llvm::zip(blocksArgs, initializers), parser,
      [&](auto it) { parser << std::get<0>(it) << " = " << std::get<1>(it); });
  parser << ")";
}

// parse and print of IfOp refer to the implementation of SCF dialect.
ParseResult IfOp::parse(OpAsmParser &parser, OperationState &result) {
  // Create the regions for 'then'.
  result.regions.reserve(2);
  Region *thenRegion = result.addRegion();
  Region *elseRegion = result.addRegion();

  OpAsmParser::UnresolvedOperand cond;

  if (parser.parseOperand(cond))
    return failure();

  SmallVector<OpAsmParser::Argument, 4> regionArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;

  // Parse the optional block arguments
  OptionalParseResult listResult =
      parser.parseOptionalAssignmentList(regionArgs, operands);
  if (listResult.has_value() && failed(listResult.value()))
    return failure();

  // Parse a colon.
  if (failed(parser.parseColon()))
    return parser.emitError(parser.getCurrentLocation(),
                            "expected type for condition operand");

  // Parse the type of the condition operand
  Type condType;
  if (failed(parser.parseType(condType)))
    return parser.emitError(parser.getCurrentLocation(),
                            "expected type for condition operand");

  // Resolve operand with provided type
  if (failed(parser.resolveOperand(cond, condType, result.operands)))
    return failure();

  // Parse optional block arg types
  if (listResult.has_value()) {
    FunctionType functionType;

    if (failed(parser.parseType(functionType)))
      return parser.emitError(parser.getCurrentLocation())
             << "expected list of types for block arguments "
             << "followed by arrow type and list of return types";

    result.addTypes(functionType.getResults());

    if (functionType.getNumInputs() != operands.size()) {
      return parser.emitError(parser.getCurrentLocation())
             << "expected as many input types as operands "
             << "(expected " << operands.size() << " got "
             << functionType.getNumInputs() << ")";
    }

    // Resolve input operands.
    if (failed(parser.resolveOperands(operands, functionType.getInputs(),
                                      parser.getCurrentLocation(),
                                      result.operands)))
      return failure();
  } else {
    // Parse optional results type list.
    if (parser.parseOptionalArrowTypeList(result.types))
      return failure();
  }

  // Parse the 'then' region.
  if (parser.parseRegion(*thenRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  // If we find an 'else' keyword then parse the 'else' region.
  if (!parser.parseOptionalKeyword("else")) {
    if (parser.parseRegion(*elseRegion, /*arguments=*/{}, /*argTypes=*/{}))
      return failure();
  }

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void IfOp::print(OpAsmPrinter &p) {
  p << " " << getCondition();

  printInitializationList(p, getThenGraph().front().getArguments(),
                          getInputList(), " ");
  p << " : ";
  p << getCondition().getType();

  if (!getInputList().empty()) {
    p << " (";
    llvm::interleaveComma(getInputList().getTypes(), p);
    p << ")";
  }
  p.printArrowTypeList(getResultTypes());
  p << " ";

  p.printRegion(getThenGraph());

  // Print the 'else' regions if it exists and has a block.
  auto &elseRegion = getElseGraph();
  if (!elseRegion.empty()) {
    p << " else ";
    p.printRegion(elseRegion);
  }

  p.printOptionalAttrDict((*this)->getAttrs());
}

LogicalResult IfOp::verify() {
  if (errorIfTypeOrShapeMismatch(*this, getThenGraph().front().getArguments(),
                                 "'then_graph' arguments", getInputList(),
                                 "'input_list'")
          .failed())
    return failure();

  if (errorIfTypeOrShapeMismatch(*this, getElseGraph().front().getArguments(),
                                 "'else_graph' arguments", getInputList(),
                                 "'input_list'")
          .failed())
    return failure();

  auto thenYield = cast<tosa::YieldOp>(getThenGraph().front().getTerminator());
  if (errorIfTypeOrShapeMismatch(*this, thenYield.getInputs(),
                                 "'then_graph' results", getOutputList(),
                                 "'output_list'")
          .failed())
    return failure();

  auto elseYield = cast<tosa::YieldOp>(getElseGraph().front().getTerminator());
  if (errorIfTypeOrShapeMismatch(*this, elseYield.getInputs(),
                                 "'else_graph' results", getOutputList(),
                                 "'output_list'")
          .failed())
    return failure();

  auto condType = getCondition().getType();
  if (errorIfShapeNotSizeOne(*this, condType).failed())
    return emitOpError() << "'condition' must be a size 1 tensor, got "
                         << condType;

  return success();
}

LogicalResult WhileOp::verify() {
  if (errorIfTypeOrShapeMismatch(*this, getInputList(), "'input_list'",
                                 getOutputList(), "'output_list'")
          .failed())
    return failure();

  if (errorIfTypeOrShapeMismatch(*this, getCondGraph().front().getArguments(),
                                 "'cond_graph' arguments", getInputList(),
                                 "'input_list'")
          .failed())
    return failure();

  if (errorIfTypeOrShapeMismatch(*this, getBodyGraph().front().getArguments(),
                                 "'body_graph' arguments", getInputList(),
                                 "'input_list'")
          .failed())
    return failure();

  auto bodyYield = cast<tosa::YieldOp>(getBodyGraph().front().getTerminator());
  if (errorIfTypeOrShapeMismatch(*this, bodyYield.getInputs(),
                                 "'body_graph' results", getInputList(),
                                 "'input_list'")
          .failed())
    return failure();

  // Condition block output must be a single element tensor with a single bool
  // value.
  auto condYield = cast<tosa::YieldOp>(getCondGraph().front().getTerminator());
  if (condYield.getInputs().size() != 1)
    return emitOpError() << "require 'cond_graph' only have one result";

  auto condOutType = condYield.getInputs()[0].getType();
  if (errorIfShapeNotSizeOne(*this, condOutType).failed())
    return emitOpError() << "'cond_graph' result must be a size 1 tensor, got "
                         << condOutType;

  if (!getElementTypeOrSelf(condOutType).isInteger(1))
    return emitOpError() << "'cond_graph' result must be a boolean tensor, got "
                         << condOutType;

  return success();
}

LogicalResult ReverseOp::verify() {
  if (verifySameElementTypes(*this, /* inType = */ getInput1().getType(),
                             /* outType = */ getOutput().getType())
          .failed())
    return failure();
  TensorType inputType = getInput1().getType();
  TensorType outputType = getOutput().getType();
  int32_t reverseAxis = getAxis();

  if (reverseAxis < 0)
    return emitOpError("expected non-negative reverse axis");
  if (inputType.hasRank()) {
    int64_t inputRank = inputType.getRank();
    // We allow for a special case where the input/output shape has rank 0 and
    // axis is also 0.
    if (reverseAxis >= inputRank && !(reverseAxis == 0 && inputRank == 0))
      return emitOpError("expect input tensor rank (")
             << inputRank << ") to be larger than reverse axis (" << reverseAxis
             << ")";
  }
  if (outputType.hasRank()) {
    int64_t outputRank = outputType.getRank();
    if (inputType.hasRank() && outputRank != inputType.getRank())
      return emitOpError(
          "expect output tensor rank to be equal to input tensor rank");
    if (reverseAxis >= outputRank && !(reverseAxis == 0 && outputRank == 0))
      return emitOpError("expect output tensor rank (")
             << outputRank << ") to be larger than reverse axis ("
             << reverseAxis << ")";
  }
  return success();
}

LogicalResult tosa::SelectOp::verify() {
  // verify input2 and input3 have same element type as output
  if (verifySameElementTypes(*this, /* inType = */ getOnTrue().getType(),
                             /* outType = */ getOutput().getType())
          .failed() ||
      verifySameElementTypes(*this, /* inType = */ getOnFalse().getType(),
                             /* outType = */ getOutput().getType())
          .failed()) {
    return failure();
  }
  // verify input1 has element type of bool
  auto predicateType = llvm::dyn_cast<ShapedType>(getPred().getType());
  if (!predicateType) {
    return emitOpError("expect shaped tensor for input1, got ")
           << getInput1().getType();
  }
  auto predicateElementType = predicateType.getElementType();
  if (!predicateElementType.isInteger(1)) {
    return emitOpError("expect element type of bool for input1, got ")
           << predicateElementType;
  }

  return success();
}

LogicalResult tosa::VariableOp::verify() {
  StringRef symName = getName();
  FailureOr<tosa::VariableOp> varOp = findVariableDecl(*this, symName);
  if (succeeded(varOp))
    return emitOpError("illegal to have multiple declaration of '")
           << symName << "'";

  return success();
}

LogicalResult tosa::VariableReadOp::verify() {
  if (verifyVariableOpErrorIf(*this, getOutput1().getType(), "'output1'")
          .failed())
    return failure();

  return success();
}

LogicalResult tosa::VariableWriteOp::verify() {
  if (verifyVariableOpErrorIf(*this, getInput1().getType(), "'input1'")
          .failed())
    return failure();

  return success();
}

// parse and print of WhileOp refer to the implementation of SCF dialect.
ParseResult WhileOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::Argument, 4> regionArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  Region *cond = result.addRegion();
  Region *body = result.addRegion();

  OptionalParseResult listResult =
      parser.parseOptionalAssignmentList(regionArgs, operands);
  if (listResult.has_value() && failed(listResult.value()))
    return failure();

  FunctionType functionType;
  SMLoc typeLoc = parser.getCurrentLocation();
  if (failed(parser.parseColonType(functionType)))
    return failure();

  result.addTypes(functionType.getResults());

  if (functionType.getNumInputs() != operands.size()) {
    return parser.emitError(typeLoc)
           << "expected as many input types as operands "
           << "(expected " << operands.size() << " got "
           << functionType.getNumInputs() << ")";
  }

  // Resolve input operands.
  if (failed(parser.resolveOperands(operands, functionType.getInputs(),
                                    parser.getCurrentLocation(),
                                    result.operands)))
    return failure();

  // Propagate the types into the region arguments.
  for (size_t i = 0, e = regionArgs.size(); i != e; ++i)
    regionArgs[i].type = functionType.getInput(i);

  return failure(parser.parseRegion(*cond, regionArgs) ||
                 parser.parseKeyword("do") || parser.parseRegion(*body) ||
                 parser.parseOptionalAttrDictWithKeyword(result.attributes));
}

void WhileOp::print(OpAsmPrinter &parser) {
  printInitializationList(parser, getCondGraph().front().getArguments(),
                          getInputList(), " ");
  parser << " : ";
  parser.printFunctionalType(getInputList().getTypes(),
                             getResults().getTypes());
  parser << ' ';
  parser.printRegion(getCondGraph(), /*printEntryBlockArgs=*/false);
  parser << " do ";
  parser.printRegion(getBodyGraph());
  parser.printOptionalAttrDictWithKeyword((*this)->getAttrs());
}

// Create a rank-1 const tensor for zero point of the source tensor.
std::optional<Value> mlir::tosa::createZeroPointTensor(OpBuilder &builder,
                                                       Location loc,
                                                       Type srcElemType,
                                                       int64_t zp) {
  srcElemType = getStorageElementTypeOrSelf(srcElemType);
  auto zpType = mlir::RankedTensorType::get({1}, srcElemType);
  if (llvm::isa<FloatType>(srcElemType)) {
    auto zpAttr = DenseElementsAttr::get(
        zpType, builder.getFloatAttr(srcElemType, static_cast<double>(zp)));
    return tosa::ConstOp::create(builder, loc, zpType, zpAttr);
  }
  if (llvm::isa<IntegerType>(srcElemType)) {
    auto zpAttr =
        DenseElementsAttr::get(zpType, builder.getIntegerAttr(srcElemType, zp));
    return tosa::ConstOp::create(builder, loc, zpType, zpAttr);
  }
  llvm::errs() << "zero point is not allowed for unsupported data types\n";
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// TOSA Shape and Shape Operators Helper functions.
//===----------------------------------------------------------------------===//

bool mlir::tosa::isa_tosa_shape_type(mlir::Type t) {
  return mlir::isa<tosa::shapeType>(t);
}

LogicalResult
mlir::tosa::shapeType::verify(function_ref<InFlightDiagnostic()> emitError,
                              int rank) {
  if (rank < 0)
    return emitError() << "invalid rank (must be >= 0): " << rank;
  return success();
}

LogicalResult OpTrait::tosa::verifyTosaResolvableShapeOperands(Operation *op) {
  for (auto v : op->getOperands()) {
    if (mlir::isa<::mlir::tosa::shapeType>(v.getType())) {
      Operation *definingOp = v.getDefiningOp();
      if (!definingOp || !definingOp->hasTrait<TosaShapeOperator>()) {
        return op->emitOpError("shape operand is not compile time resolvable");
      }
    }
  }
  return success();
}

LogicalResult OpTrait::tosa::verifyTosaShapeOperator(Operation *op) {
  for (auto type : op->getOperandTypes()) {
    if (!mlir::isa<mlir::tosa::shapeType>(type)) {
      return op->emitOpError("must have operands with tosa shape type");
    }
  }
  for (auto type : op->getResultTypes()) {
    if (!mlir::isa<mlir::tosa::shapeType>(type)) {
      return op->emitOpError("must have result with tosa shape type");
    }
  }
  return success();
}

LogicalResult
OpTrait::tosa::verifyTosaShapeOperatorWithSameRanks(Operation *op) {
  if (failed(OpTrait::impl::verifyAtLeastNOperands(op, 1)) ||
      failed(verifyTosaShapeOperator(op)))
    return failure();

  // delegate function that returns rank of shape type
  auto getRank = [](const Type type) {
    return mlir::cast<mlir::tosa::shapeType>(type).getRank();
  };
  auto operandTypes = op->getOperandTypes();
  auto resultTypes = op->getResultTypes();

  auto rank = getRank(*op->getOperandTypes().begin());
  for (auto type : operandTypes) {
    if (getRank(type) != rank) {
      return op->emitOpError("operands don't have matching ranks");
    }
  }
  for (auto type : resultTypes) {
    if (getRank(type) != rank) {
      return op->emitOpError("result shape has different rank than operands");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TOSA Shape Operators verify functions.
//===----------------------------------------------------------------------===//

LogicalResult tosa::ConstShapeOp::verify() {
  // check one dimensional rank
  auto valuesRank = getValues().getType().getRank();
  if (valuesRank != 1)
    return emitOpError("expect elements in attribute values with rank 1");
  // check that number of elements in values attr equal to rank of result shape
  auto count = getValues().getNumElements();
  auto rank = (cast<tosa::shapeType>(getResult().getType())).getRank();
  if (!(count == rank || (count == 1 && rank == 0))) {
    return emitOpError("expect number of elements in attribute values (")
           << count << ") to be equal to the rank (" << rank
           << ") for the result shape type";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// TOSA Attribute Definitions.
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Tosa/IR/TosaAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// TOSA Type Definitions.
//===----------------------------------------------------------------------===//
#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Tosa/IR/TosaOpsTypesBase.cpp.inc"

//===----------------------------------------------------------------------===//
// TOSA Operator Definitions.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Tosa/IR/TosaOps.cpp.inc"
