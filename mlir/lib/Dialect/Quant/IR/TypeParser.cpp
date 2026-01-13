//===- TypeParser.h - Quantization Type Parser ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/APFloat.h"

using namespace mlir;
using namespace quant;

static IntegerType parseStorageType(DialectAsmParser &parser, bool &isSigned) {
  auto typeLoc = parser.getCurrentLocation();
  IntegerType type;

  // Parse storage type (alpha_ident, integer_literal).
  StringRef identifier;
  unsigned storageTypeWidth = 0;
  OptionalParseResult result = parser.parseOptionalType(type);
  if (result.has_value()) {
    if (!succeeded(*result))
      return nullptr;
    isSigned = !type.isUnsigned();
    storageTypeWidth = type.getWidth();
  } else if (succeeded(parser.parseKeyword(&identifier))) {
    // Otherwise, this must be an unsigned integer (`u` integer-literal).
    if (!identifier.consume_front("u")) {
      parser.emitError(typeLoc, "illegal storage type prefix");
      return nullptr;
    }
    if (identifier.getAsInteger(10, storageTypeWidth)) {
      parser.emitError(typeLoc, "expected storage type width");
      return nullptr;
    }
    isSigned = false;
    type = parser.getBuilder().getIntegerType(storageTypeWidth);
  } else {
    return nullptr;
  }

  if (storageTypeWidth == 0 ||
      storageTypeWidth > QuantizedType::MaxStorageBits) {
    parser.emitError(typeLoc, "illegal storage type size: ")
        << storageTypeWidth;
    return nullptr;
  }

  return type;
}

static ParseResult parseStorageRange(DialectAsmParser &parser,
                                     IntegerType storageType, bool isSigned,
                                     int64_t &storageTypeMin,
                                     int64_t &storageTypeMax) {
  int64_t defaultIntegerMin = QuantizedType::getDefaultMinimumForInteger(
      isSigned, storageType.getWidth());
  int64_t defaultIntegerMax = QuantizedType::getDefaultMaximumForInteger(
      isSigned, storageType.getWidth());
  if (failed(parser.parseOptionalLess())) {
    storageTypeMin = defaultIntegerMin;
    storageTypeMax = defaultIntegerMax;
    return success();
  }

  // Explicit storage min and storage max.
  SMLoc minLoc = parser.getCurrentLocation(), maxLoc;
  if (parser.parseInteger(storageTypeMin) || parser.parseColon() ||
      parser.getCurrentLocation(&maxLoc) ||
      parser.parseInteger(storageTypeMax) || parser.parseGreater())
    return failure();
  if (storageTypeMin < defaultIntegerMin) {
    return parser.emitError(minLoc, "illegal storage type minimum: ")
           << storageTypeMin;
  }
  if (storageTypeMax > defaultIntegerMax) {
    return parser.emitError(maxLoc, "illegal storage type maximum: ")
           << storageTypeMax;
  }
  return success();
}

static FloatType parseExpressedTypeAndRange(DialectAsmParser &parser,
                                            double &min, double &max) {
  auto typeLoc = parser.getCurrentLocation();
  FloatType type;

  if (failed(parser.parseType(type))) {
    parser.emitError(typeLoc, "expecting float expressed type");
    return nullptr;
  }

  // Calibrated min and max values.
  if (parser.parseLess() || parser.parseFloat(min) || parser.parseColon() ||
      parser.parseFloat(max) || parser.parseGreater()) {
    parser.emitError(typeLoc, "calibrated values must be present");
    return nullptr;
  }
  return type;
}

/// Parses an AnyQuantizedType.
///
///   any ::= `any<` storage-spec (expressed-type-spec)?`>`
///   storage-spec ::= storage-type (`<` storage-range `>`)?
///   storage-range ::= integer-literal `:` integer-literal
///   storage-type ::= (`i` | `u`) integer-literal
///   expressed-type-spec ::= `:` `f` integer-literal
static Type parseAnyType(DialectAsmParser &parser) {
  IntegerType storageType;
  FloatType expressedType;
  unsigned typeFlags = 0;
  int64_t storageTypeMin;
  int64_t storageTypeMax;

  // Type specification.
  if (parser.parseLess())
    return nullptr;

  // Storage type.
  bool isSigned = false;
  storageType = parseStorageType(parser, isSigned);
  if (!storageType) {
    return nullptr;
  }
  if (isSigned) {
    typeFlags |= QuantizationFlags::Signed;
  }

  // Storage type range.
  if (parseStorageRange(parser, storageType, isSigned, storageTypeMin,
                        storageTypeMax)) {
    return nullptr;
  }

  // Optional expressed type.
  if (succeeded(parser.parseOptionalColon())) {
    if (parser.parseType(expressedType)) {
      return nullptr;
    }
  }

  if (parser.parseGreater()) {
    return nullptr;
  }

  return parser.getChecked<AnyQuantizedType>(
      typeFlags, storageType, expressedType, storageTypeMin, storageTypeMax);
}

/// Checks if the given scale value is within the valid range of the expressed
/// type. The `expressedType` argument is the floating-point type used for
/// expressing the quantized values, and `scale` is the double value to check.
static LogicalResult
isScaleInExpressedTypeRange(function_ref<InFlightDiagnostic()> emitError,
                            Type expressedType, double scale) {
  auto floatType = cast<FloatType>(expressedType);
  double minScale =
      APFloat::getSmallest(floatType.getFloatSemantics()).convertToDouble();
  double maxScale =
      APFloat::getLargest(floatType.getFloatSemantics()).convertToDouble();
  if (scale < minScale || scale > maxScale)
    return emitError() << "scale " << scale << " out of expressed type range ["
                       << minScale << ", " << maxScale << "]";
  return success();
}

/// Parses a quantization parameter, which is either a scale value (float) or a
/// scale-zero point pair (float:integer). `expressedType`, expressing the type
/// of scale values, is used to validate the scale. The parsed scale and zero
/// point (if any) are stored in `scale` and `zeroPoint`.
static ParseResult parseQuantParams(DialectAsmParser &parser,
                                    Type expressedType, double &scale,
                                    int64_t &zeroPoint) {

  if (parser.parseFloat(scale)) {
    return failure();
  }

  if (failed(isScaleInExpressedTypeRange(
          [&]() { return parser.emitError(parser.getCurrentLocation()); },
          expressedType, scale))) {
    return failure();
  }

  zeroPoint = 0;
  if (failed(parser.parseOptionalColon())) {
    return success();
  }

  return parser.parseInteger(zeroPoint);
}

/// Parses block size information for sub-channel quantization, assuming the
/// leading '{' has already been parsed. The block size information is provided
/// as a comma-separated list of "Axis:BlockSize" pairs, terminated by a '}'.
///
/// The parsed axis indices are stored in `quantizedDimensions`, and the
/// corresponding block sizes are stored in `blockSizes`.
static ParseResult
parseBlockSizeInfoUntilRBrace(DialectAsmParser &parser,
                              SmallVectorImpl<int32_t> &quantizedDimensions,
                              SmallVectorImpl<int64_t> &blockSizes) {
  // Empty block-sizes info.
  if (succeeded(parser.parseOptionalRBrace())) {
    return success();
  }

  auto parseBlockSizeElements = [&]() -> ParseResult {
    quantizedDimensions.resize(quantizedDimensions.size() + 1);
    blockSizes.resize(blockSizes.size() + 1);
    if (parser.parseInteger(quantizedDimensions.back()) ||
        parser.parseColon() || parser.parseInteger(blockSizes.back()))
      return failure();
    return success();
  };

  if (parser.parseCommaSeparatedList(parseBlockSizeElements) ||
      parser.parseRBrace()) {
    return failure();
  }

  return success();
}

/// Parses a bracketed list of quantization parameters, returning the dimensions
/// of the parsed sub-tensors in `dims`. The dimension of the list is prepended
/// to the dimensions of the sub-tensors. This function assumes that the initial
/// left brace has already been parsed. For example:
///
///   parseQuantParamListUntilRBrace(1.0:1, 2.0:4, 3.0:4}) -> Success,
///       dims = [3], scales = [1.0, 2.0, 3.0], zeroPoints = [1, 4, 4]
///
///   parseQuantParamListUntilRBrace({1.0, 2.0}, {3.0:1, 4.0:9}}) -> Success,
///       dims = [2, 2], scales = [1.0, 2.0, 3.0, 4.0], zeroPoints = [0, 0, 1,
///       9]
///
/// This function expects all sub-tensors to have the same rank.
static ParseResult
parseQuantParamListUntilRBrace(DialectAsmParser &parser, Type expressedType,
                               SmallVectorImpl<double> &scales,
                               SmallVectorImpl<int64_t> &zeroPoints,
                               SmallVectorImpl<int64_t> &dims) {
  auto checkDims = [&](const SmallVectorImpl<int64_t> &prevDims,
                       const SmallVectorImpl<int64_t> &newDims) -> ParseResult {
    if (prevDims == newDims)
      return success();
    return parser.emitError(parser.getCurrentLocation())
           << "tensor literal is invalid; ranks are not consistent "
              "between elements";
  };

  bool first = true;
  SmallVector<int64_t, 4> newDims;
  unsigned size = 0;

  auto parseOneElement = [&]() -> ParseResult {
    SmallVector<int64_t, 4> thisDims;
    if (succeeded(parser.parseOptionalLBrace())) {
      if (parseQuantParamListUntilRBrace(parser, expressedType, scales,
                                         zeroPoints, thisDims))
        return failure();
    } else {
      zeroPoints.resize(zeroPoints.size() + 1);
      scales.resize(scales.size() + 1);
      if (parseQuantParams(parser, expressedType, scales.back(),
                           zeroPoints.back())) {
        return failure();
      }
    }
    ++size;
    if (!first)
      return checkDims(newDims, thisDims);
    newDims = thisDims;
    first = false;
    return success();
  };

  if (parser.parseCommaSeparatedList(parseOneElement) || parser.parseRBrace()) {
    return failure();
  }

  // Return the sublists' dimensions with 'size' prepended.
  dims.clear();
  dims.push_back(size);
  dims.append(newDims.begin(), newDims.end());

  return success();
}

/// Parses a UniformQuantizedType.
///
///   uniform_type ::= uniform_per_layer
///                  | uniform_per_axis
///                  | uniform_sub_channel
///   uniform_per_layer ::= `uniform<` storage-spec expressed-type-spec
///                          `,` scale-zero `>`
///   uniform_per_axis ::= `uniform<` storage-spec expressed-type-spec
///                        axis-spec `,` `{` scale-zero-list `}` `>`
///   uniform_sub_channel ::= `uniform<` storage-spec expressed-type-spec
///                        block-size-info `,` scale-zero-tensor `>`
///   storage-spec ::= storage-type (`<` storage-range `>`)?
///   storage-range ::= integer-literal `:` integer-literal
///   storage-type ::= (`i` | `u`) integer-literal
///   expressed-type-spec ::= `:` `f` integer-literal
///   axis-spec ::= `:` integer-literal
///   scale-zero ::= scale (`:` zero-point)?
///   scale ::= float-literal
///   zero-point ::= integer-literal
///   scale-zero-list ::= scale-zero (`,` scale-zero)*
///   block-size-info ::= `{` `}` | `{` axis-block `:` (`,` axis-block)* `}`
///   axis-block ::= axis-spec `:` block-size-spec
///   block-size-spec ::= integer-literal
///   scale-zero-tensor ::= scale-zero-dense-exp | scale-zero-list
///   scale-zero-dense-exp ::= `{`
///     scale-zero-tensor (`,` scale-zero-tensor)*
///   `}`
static Type parseUniformType(DialectAsmParser &parser) {
  IntegerType storageType;
  FloatType expressedType;
  unsigned typeFlags = 0;
  int64_t storageTypeMin;
  int64_t storageTypeMax;
  bool isPerAxis = false;
  bool isSubChannel = false;
  SmallVector<int32_t, 1> quantizedDimensions;
  SmallVector<int64_t, 1> blockSizes;
  SmallVector<double, 1> scales;
  SmallVector<int64_t, 1> zeroPoints;

  // Type specification.
  if (parser.parseLess()) {
    return nullptr;
  }

  // Storage type.
  bool isSigned = false;
  storageType = parseStorageType(parser, isSigned);
  if (!storageType) {
    return nullptr;
  }
  if (isSigned) {
    typeFlags |= QuantizationFlags::Signed;
  }

  // Storage type range.
  if (parseStorageRange(parser, storageType, isSigned, storageTypeMin,
                        storageTypeMax)) {
    return nullptr;
  }

  // Expressed type.
  if (parser.parseColon() || parser.parseType(expressedType)) {
    return nullptr;
  }

  // Optionally parse quantized dimension for per-axis or sub-channel
  // quantization.
  if (succeeded(parser.parseOptionalColon())) {
    if (succeeded(parser.parseOptionalLBrace())) {
      isSubChannel = true;
      if (parseBlockSizeInfoUntilRBrace(parser, quantizedDimensions,
                                        blockSizes)) {
        return nullptr;
      }
    } else {
      isPerAxis = true;
      quantizedDimensions.resize(1);
      if (parser.parseInteger(quantizedDimensions.back())) {
        return nullptr;
      }
    }
  }

  // Comma leading into range_spec.
  if (parser.parseComma()) {
    return nullptr;
  }

  // Quantization parameter (scales/zeroPoints) specification.
  bool isPerTensor = !isPerAxis && !isSubChannel;
  SmallVector<int64_t> dims;
  if (isPerTensor) {
    zeroPoints.resize(zeroPoints.size() + 1);
    scales.resize(scales.size() + 1);
    if (parseQuantParams(parser, expressedType, scales.back(),
                         zeroPoints.back())) {
      return nullptr;
    }

  } else {
    if (parser.parseLBrace() ||
        parseQuantParamListUntilRBrace(parser, expressedType, scales,
                                       zeroPoints, dims)) {
      return nullptr;
    }
  }

  if (parser.parseGreater()) {
    return nullptr;
  }

  if (isPerAxis) {
    return parser.getChecked<UniformQuantizedPerAxisType>(
        typeFlags, storageType, expressedType, scales, zeroPoints,
        quantizedDimensions[0], storageTypeMin, storageTypeMax);
  }
  if (isSubChannel) {
    SmallVector<APFloat> apFloatScales =
        llvm::to_vector(llvm::map_range(scales, [&](double scale) -> APFloat {
          APFloat apFloatScale(scale);
          bool unused;
          apFloatScale.convert(expressedType.getFloatSemantics(),
                               APFloat::rmNearestTiesToEven, &unused);
          return apFloatScale;
        }));
    SmallVector<APInt> apIntZeroPoints = llvm::to_vector(
        llvm::map_range(zeroPoints, [&](int64_t zeroPoint) -> APInt {
          return APInt(storageType.getIntOrFloatBitWidth(), zeroPoint);
        }));
    auto scalesRef = mlir::DenseElementsAttr::get(
        RankedTensorType::get(dims, expressedType), apFloatScales);
    auto zeroPointsRef = mlir::DenseElementsAttr::get(
        RankedTensorType::get(dims, storageType), apIntZeroPoints);
    return parser.getChecked<UniformQuantizedSubChannelType>(
        typeFlags, storageType, expressedType, scalesRef, zeroPointsRef,
        quantizedDimensions, blockSizes, storageTypeMin, storageTypeMax);
  }

  return parser.getChecked<UniformQuantizedType>(
      typeFlags, storageType, expressedType, scales.front(), zeroPoints.front(),
      storageTypeMin, storageTypeMax);
}

/// Parses an CalibratedQuantizedType.
///
///   calibrated ::= `calibrated<` expressed-spec `>`
///   expressed-spec ::= expressed-type `<` calibrated-range `>`
///   expressed-type ::= `f` integer-literal
///   calibrated-range ::= float-literal `:` float-literal
static Type parseCalibratedType(DialectAsmParser &parser) {
  FloatType expressedType;
  double min;
  double max;

  // Type specification.
  if (parser.parseLess())
    return nullptr;

  // Expressed type.
  expressedType = parseExpressedTypeAndRange(parser, min, max);
  if (!expressedType) {
    return nullptr;
  }

  if (parser.parseGreater()) {
    return nullptr;
  }

  return parser.getChecked<CalibratedQuantizedType>(expressedType, min, max);
}

/// Parse a type registered to this dialect.
Type QuantDialect::parseType(DialectAsmParser &parser) const {
  // All types start with an identifier that we switch on.
  StringRef typeNameSpelling;
  if (failed(parser.parseKeyword(&typeNameSpelling)))
    return nullptr;

  if (typeNameSpelling == "uniform")
    return parseUniformType(parser);
  if (typeNameSpelling == "any")
    return parseAnyType(parser);
  if (typeNameSpelling == "calibrated")
    return parseCalibratedType(parser);

  parser.emitError(parser.getNameLoc(),
                   "unknown quantized type " + typeNameSpelling);
  return nullptr;
}

static void printStorageType(QuantizedType type, DialectAsmPrinter &out) {
  // storage type
  unsigned storageWidth = type.getStorageTypeIntegralWidth();
  bool isSigned = type.isSigned();
  if (isSigned) {
    out << "i" << storageWidth;
  } else {
    out << "u" << storageWidth;
  }

  // storageTypeMin and storageTypeMax if not default.
  if (type.hasStorageTypeBounds()) {
    out << "<" << type.getStorageTypeMin() << ":" << type.getStorageTypeMax()
        << ">";
  }
}

static void printQuantParams(double scale, int64_t zeroPoint,
                             DialectAsmPrinter &out) {
  out << scale;
  if (zeroPoint != 0) {
    out << ":" << zeroPoint;
  }
}

static void
printBlockSizeInfo(ArrayRef<std::pair<int32_t, int64_t>> blockSizeInfo,
                   DialectAsmPrinter &out) {
  out << "{";
  llvm::interleaveComma(
      llvm::seq<size_t>(0, blockSizeInfo.size()), out, [&](size_t index) {
        out << blockSizeInfo[index].first << ":" << blockSizeInfo[index].second;
      });
  out << "}";
}

/// Helper that prints a AnyQuantizedType.
static void printAnyQuantizedType(AnyQuantizedType type,
                                  DialectAsmPrinter &out) {
  out << "any<";
  printStorageType(type, out);
  if (Type expressedType = type.getExpressedType()) {
    out << ":" << expressedType;
  }
  out << ">";
}

/// Helper that prints a UniformQuantizedType.
static void printUniformQuantizedType(UniformQuantizedType type,
                                      DialectAsmPrinter &out) {
  out << "uniform<";
  printStorageType(type, out);
  out << ":" << type.getExpressedType() << ", ";

  // scheme specific parameters
  printQuantParams(type.getScale(), type.getZeroPoint(), out);
  out << ">";
}

/// Helper that prints a UniformQuantizedPerAxisType.
static void printUniformQuantizedPerAxisType(UniformQuantizedPerAxisType type,
                                             DialectAsmPrinter &out) {
  out << "uniform<";
  printStorageType(type, out);
  out << ":" << type.getExpressedType() << ":";
  out << type.getQuantizedDimension();
  out << ", ";

  // scheme specific parameters
  ArrayRef<double> scales = type.getScales();
  ArrayRef<int64_t> zeroPoints = type.getZeroPoints();
  out << "{";
  llvm::interleave(
      llvm::seq<size_t>(0, scales.size()), out,
      [&](size_t index) {
        printQuantParams(scales[index], zeroPoints[index], out);
      },
      ",");
  out << "}>";
}

/// Prints quantization parameters as a nested list of `scale`[:`zero_point`]
/// elements.  The nesting corresponds to the `shape` dimensions.
///
/// Elements are delimited by commas, and the inner dimensions are enclosed in
/// braces.  `zero_point` is only printed if it is non-zero.  For example:
///
///   printDenseQuantizationParameters(scales=[1.0, 2.0, 3.0, 4.0],
///                                   zeroPoints=[0, 0, 1, 9],
///                                   shape=[2, 2])
///
///   would print:
///
///     {{1.0, 2.0}, {3.0:1, 4.0:9}}
static void printDenseQuantizationParameters(ArrayRef<APFloat> scales,
                                             ArrayRef<APInt> zeroPoints,
                                             ArrayRef<int64_t> shape,
                                             DialectAsmPrinter &out) {
  int64_t rank = shape.size();
  SmallVector<unsigned, 4> counter(rank, 0);
  unsigned openBrackets = 0;

  auto incrementCounterAndDelimit = [&]() {
    ++counter[rank - 1];
    for (unsigned i = rank - 1; i > 0; --i) {
      if (counter[i] >= shape[i]) {
        counter[i] = 0;
        ++counter[i - 1];
        --openBrackets;
        out << '}';
      }
    }
  };

  for (unsigned idx = 0, e = scales.size(); idx < e; ++idx) {
    if (idx != 0)
      out << ", ";
    while (openBrackets++ < rank)
      out << '{';
    openBrackets = rank;
    out << scales[idx];
    if (zeroPoints[idx] != 0) {
      out << ":" << zeroPoints[idx];
    }
    incrementCounterAndDelimit();
  }
  while (openBrackets-- > 0)
    out << '}';
}

/// Helper that prints a UniformQuantizedSubChannelType.
static void
printUniformQuantizedSubChannelType(UniformQuantizedSubChannelType type,
                                    DialectAsmPrinter &out) {
  out << "uniform<";
  printStorageType(type, out);
  out << ":" << type.getExpressedType() << ":";
  printBlockSizeInfo(type.getBlockSizeInfo(), out);
  out << ", ";

  auto scalesItr = type.getScales().getValues<APFloat>();
  auto zeroPointsItr = type.getZeroPoints().getValues<APInt>();
  SmallVector<APFloat> scales(scalesItr.begin(), scalesItr.end());
  SmallVector<APInt> zeroPoints(zeroPointsItr.begin(), zeroPointsItr.end());
  printDenseQuantizationParameters(scales, zeroPoints,
                                   type.getScales().getType().getShape(), out);
  out << ">";
}

/// Helper that prints a CalibratedQuantizedType.
static void printCalibratedQuantizedType(CalibratedQuantizedType type,
                                         DialectAsmPrinter &out) {
  out << "calibrated<" << type.getExpressedType();
  out << "<" << type.getMin() << ":" << type.getMax() << ">";
  out << ">";
}

/// Print a type registered to this dialect.
void QuantDialect::printType(Type type, DialectAsmPrinter &os) const {
  if (auto anyType = llvm::dyn_cast<AnyQuantizedType>(type))
    printAnyQuantizedType(anyType, os);
  else if (auto uniformType = llvm::dyn_cast<UniformQuantizedType>(type))
    printUniformQuantizedType(uniformType, os);
  else if (auto perAxisType = llvm::dyn_cast<UniformQuantizedPerAxisType>(type))
    printUniformQuantizedPerAxisType(perAxisType, os);
  else if (auto perAxisType =
               llvm::dyn_cast<UniformQuantizedSubChannelType>(type))
    printUniformQuantizedSubChannelType(perAxisType, os);
  else if (auto calibratedType = llvm::dyn_cast<CalibratedQuantizedType>(type))
    printCalibratedQuantizedType(calibratedType, os);
  else
    llvm_unreachable("Unhandled quantized type");
}
