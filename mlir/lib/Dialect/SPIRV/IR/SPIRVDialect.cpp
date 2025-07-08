//===- LLVMDialect.cpp - MLIR SPIR-V dialect ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the SPIR-V dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"

#include "SPIRVParsingUtils.h"

#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::spirv;

#include "mlir/Dialect/SPIRV/IR/SPIRVOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// InlinerInterface
//===----------------------------------------------------------------------===//

/// Returns true if the given region contains spirv.Return or spirv.ReturnValue
/// ops.
static inline bool containsReturn(Region &region) {
  return llvm::any_of(region, [](Block &block) {
    Operation *terminator = block.getTerminator();
    return isa<spirv::ReturnOp, spirv::ReturnValueOp>(terminator);
  });
}

namespace {
/// This class defines the interface for inlining within the SPIR-V dialect.
struct SPIRVInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All call operations within SPIRV can be inlined.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    return true;
  }

  /// Returns true if the given region 'src' can be inlined into the region
  /// 'dest' that is attached to an operation registered to the current dialect.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &) const final {
    // Return true here when inlining into spirv.func, spirv.mlir.selection, and
    // spirv.mlir.loop operations.
    auto *op = dest->getParentOp();
    return isa<spirv::FuncOp, spirv::SelectionOp, spirv::LoopOp>(op);
  }

  /// Returns true if the given operation 'op', that is registered to this
  /// dialect, can be inlined into the region 'dest' that is attached to an
  /// operation registered to the current dialect.
  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &) const final {
    // TODO: Enable inlining structured control flows with return.
    if ((isa<spirv::SelectionOp, spirv::LoopOp>(op)) &&
        containsReturn(op->getRegion(0)))
      return false;
    // TODO: we need to filter OpKill here to avoid inlining it to
    // a loop continue construct:
    // https://github.com/KhronosGroup/SPIRV-Headers/issues/86
    // For now, we just disallow inlining OpKill anywhere in the code,
    // but this restriction should be relaxed, as pointed above.
    if (isa<spirv::KillOp>(op))
      return false;

    return true;
  }

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(Operation *op, Block *newDest) const final {
    if (auto returnOp = dyn_cast<spirv::ReturnOp>(op)) {
      OpBuilder(op).create<spirv::BranchOp>(op->getLoc(), newDest);
      op->erase();
    } else if (auto retValOp = dyn_cast<spirv::ReturnValueOp>(op)) {
      OpBuilder(op).create<spirv::BranchOp>(retValOp->getLoc(), newDest,
                                            retValOp->getOperands());
      op->erase();
    }
  }

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
    // Only spirv.ReturnValue needs to be handled here.
    auto retValOp = dyn_cast<spirv::ReturnValueOp>(op);
    if (!retValOp)
      return;

    // Replace the values directly with the return operands.
    assert(valuesToRepl.size() == 1 &&
           "spirv.ReturnValue expected to only handle one result");
    valuesToRepl.front().replaceAllUsesWith(retValOp.getValue());
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// SPIR-V Dialect
//===----------------------------------------------------------------------===//

void SPIRVDialect::initialize() {
  registerAttributes();
  registerTypes();

  // Add SPIR-V ops.
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.cpp.inc"
      >();

  addInterfaces<SPIRVInlinerInterface>();

  // Allow unknown operations because SPIR-V is extensible.
  allowUnknownOperations();
  declarePromisedInterface<gpu::TargetAttrInterface, TargetEnvAttr>();
}

std::string SPIRVDialect::getAttributeName(Decoration decoration) {
  return llvm::convertToSnakeFromCamelCase(stringifyDecoration(decoration));
}

//===----------------------------------------------------------------------===//
// Type Parsing
//===----------------------------------------------------------------------===//

// Forward declarations.
template <typename ValTy>
static std::optional<ValTy> parseAndVerify(SPIRVDialect const &dialect,
                                           DialectAsmParser &parser);
template <>
std::optional<Type> parseAndVerify<Type>(SPIRVDialect const &dialect,
                                         DialectAsmParser &parser);

template <>
std::optional<unsigned> parseAndVerify<unsigned>(SPIRVDialect const &dialect,
                                                 DialectAsmParser &parser);

static Type parseAndVerifyType(SPIRVDialect const &dialect,
                               DialectAsmParser &parser) {
  Type type;
  SMLoc typeLoc = parser.getCurrentLocation();
  if (parser.parseType(type))
    return Type();

  // Allow SPIR-V dialect types
  if (&type.getDialect() == &dialect)
    return type;

  // Check other allowed types
  if (auto t = llvm::dyn_cast<FloatType>(type)) {
    // TODO: All float types are allowed for now, but this should be fixed.
  } else if (auto t = llvm::dyn_cast<IntegerType>(type)) {
    if (!ScalarType::isValid(t)) {
      parser.emitError(typeLoc,
                       "only 1/8/16/32/64-bit integer type allowed but found ")
          << type;
      return Type();
    }
  } else if (auto t = llvm::dyn_cast<VectorType>(type)) {
    if (t.getRank() != 1) {
      parser.emitError(typeLoc, "only 1-D vector allowed but found ") << t;
      return Type();
    }
    if (t.getNumElements() > 4) {
      parser.emitError(
          typeLoc, "vector length has to be less than or equal to 4 but found ")
          << t.getNumElements();
      return Type();
    }
  } else if (auto t = dyn_cast<TensorArmType>(type)) {
    if (!isa<ScalarType>(t.getElementType())) {
      parser.emitError(
          typeLoc, "only scalar element type allowed in tensor type but found ")
          << t.getElementType();
      return Type();
    }
  } else {
    parser.emitError(typeLoc, "cannot use ")
        << type << " to compose SPIR-V types";
    return Type();
  }

  return type;
}

static Type parseAndVerifyMatrixType(SPIRVDialect const &dialect,
                                     DialectAsmParser &parser) {
  Type type;
  SMLoc typeLoc = parser.getCurrentLocation();
  if (parser.parseType(type))
    return Type();

  if (auto t = llvm::dyn_cast<VectorType>(type)) {
    if (t.getRank() != 1) {
      parser.emitError(typeLoc, "only 1-D vector allowed but found ") << t;
      return Type();
    }
    if (t.getNumElements() > 4 || t.getNumElements() < 2) {
      parser.emitError(typeLoc,
                       "matrix columns size has to be less than or equal "
                       "to 4 and greater than or equal 2, but found ")
          << t.getNumElements();
      return Type();
    }

    if (!llvm::isa<FloatType>(t.getElementType())) {
      parser.emitError(typeLoc, "matrix columns' elements must be of "
                                "Float type, got ")
          << t.getElementType();
      return Type();
    }
  } else {
    parser.emitError(typeLoc, "matrix must be composed using vector "
                              "type, got ")
        << type;
    return Type();
  }

  return type;
}

static Type parseAndVerifySampledImageType(SPIRVDialect const &dialect,
                                           DialectAsmParser &parser) {
  Type type;
  SMLoc typeLoc = parser.getCurrentLocation();
  if (parser.parseType(type))
    return Type();

  if (!llvm::isa<ImageType>(type)) {
    parser.emitError(typeLoc,
                     "sampled image must be composed using image type, got ")
        << type;
    return Type();
  }

  return type;
}

/// Parses an optional `, stride = N` assembly segment. If no parsing failure
/// occurs, writes `N` to `stride` if existing and writes 0 to `stride` if
/// missing.
static LogicalResult parseOptionalArrayStride(const SPIRVDialect &dialect,
                                              DialectAsmParser &parser,
                                              unsigned &stride) {
  if (failed(parser.parseOptionalComma())) {
    stride = 0;
    return success();
  }

  if (parser.parseKeyword("stride") || parser.parseEqual())
    return failure();

  SMLoc strideLoc = parser.getCurrentLocation();
  std::optional<unsigned> optStride = parseAndVerify<unsigned>(dialect, parser);
  if (!optStride)
    return failure();

  if (!(stride = *optStride)) {
    parser.emitError(strideLoc, "ArrayStride must be greater than zero");
    return failure();
  }
  return success();
}

// element-type ::= integer-type
//                | floating-point-type
//                | vector-type
//                | spirv-type
//
// array-type ::= `!spirv.array` `<` integer-literal `x` element-type
//                (`,` `stride` `=` integer-literal)? `>`
static Type parseArrayType(SPIRVDialect const &dialect,
                           DialectAsmParser &parser) {
  if (parser.parseLess())
    return Type();

  SmallVector<int64_t, 1> countDims;
  SMLoc countLoc = parser.getCurrentLocation();
  if (parser.parseDimensionList(countDims, /*allowDynamic=*/false))
    return Type();
  if (countDims.size() != 1) {
    parser.emitError(countLoc,
                     "expected single integer for array element count");
    return Type();
  }

  // According to the SPIR-V spec:
  // "Length is the number of elements in the array. It must be at least 1."
  int64_t count = countDims[0];
  if (count == 0) {
    parser.emitError(countLoc, "expected array length greater than 0");
    return Type();
  }

  Type elementType = parseAndVerifyType(dialect, parser);
  if (!elementType)
    return Type();

  unsigned stride = 0;
  if (failed(parseOptionalArrayStride(dialect, parser, stride)))
    return Type();

  if (parser.parseGreater())
    return Type();
  return ArrayType::get(elementType, count, stride);
}

// cooperative-matrix-type ::=
//   `!spirv.coopmatrix` `<` rows `x` columns `x` element-type `,`
//                           scope `,` use `>`
static Type parseCooperativeMatrixType(SPIRVDialect const &dialect,
                                       DialectAsmParser &parser) {
  if (parser.parseLess())
    return {};

  SmallVector<int64_t, 2> dims;
  SMLoc countLoc = parser.getCurrentLocation();
  if (parser.parseDimensionList(dims, /*allowDynamic=*/false))
    return {};

  if (dims.size() != 2) {
    parser.emitError(countLoc, "expected row and column count");
    return {};
  }

  auto elementTy = parseAndVerifyType(dialect, parser);
  if (!elementTy)
    return {};

  Scope scope;
  if (parser.parseComma() ||
      spirv::parseEnumKeywordAttr(scope, parser, "scope <id>"))
    return {};

  CooperativeMatrixUseKHR use;
  if (parser.parseComma() ||
      spirv::parseEnumKeywordAttr(use, parser, "use <id>"))
    return {};

  if (parser.parseGreater())
    return {};

  return CooperativeMatrixType::get(elementTy, dims[0], dims[1], scope, use);
}

// tensor-arm-type ::=
//   `!spirv.arm.tensor` `<` dim0 `x` dim1 `x` ... `x` dimN `x` element-type`>`
static Type parseTensorArmType(SPIRVDialect const &dialect,
                               DialectAsmParser &parser) {
  if (parser.parseLess())
    return {};

  bool unranked = false;
  SmallVector<int64_t, 4> dims;
  SMLoc countLoc = parser.getCurrentLocation();

  if (parser.parseOptionalStar().succeeded()) {
    unranked = true;
    if (parser.parseXInDimensionList())
      return {};
  } else if (parser.parseDimensionList(dims, /*allowDynamic=*/true)) {
    return {};
  }

  if (!unranked && dims.empty()) {
    parser.emitError(countLoc, "arm.tensors do not support rank zero");
    return {};
  }

  if (llvm::is_contained(dims, 0)) {
    parser.emitError(countLoc, "arm.tensors do not support zero dimensions");
    return {};
  }

  if (llvm::any_of(dims, [](int64_t dim) { return dim < 0; }) &&
      llvm::any_of(dims, [](int64_t dim) { return dim > 0; })) {
    parser.emitError(countLoc, "arm.tensor shape dimensions must be either "
                               "fully dynamic or completed shaped");
    return {};
  }

  auto elementTy = parseAndVerifyType(dialect, parser);
  if (!elementTy)
    return {};

  if (parser.parseGreater())
    return {};

  return TensorArmType::get(dims, elementTy);
}

// TODO: Reorder methods to be utilities first and parse*Type
// methods in alphabetical order
//
// storage-class ::= `UniformConstant`
//                 | `Uniform`
//                 | `Workgroup`
//                 | <and other storage classes...>
//
// pointer-type ::= `!spirv.ptr<` element-type `,` storage-class `>`
static Type parsePointerType(SPIRVDialect const &dialect,
                             DialectAsmParser &parser) {
  if (parser.parseLess())
    return Type();

  auto pointeeType = parseAndVerifyType(dialect, parser);
  if (!pointeeType)
    return Type();

  StringRef storageClassSpec;
  SMLoc storageClassLoc = parser.getCurrentLocation();
  if (parser.parseComma() || parser.parseKeyword(&storageClassSpec))
    return Type();

  auto storageClass = symbolizeStorageClass(storageClassSpec);
  if (!storageClass) {
    parser.emitError(storageClassLoc, "unknown storage class: ")
        << storageClassSpec;
    return Type();
  }
  if (parser.parseGreater())
    return Type();
  return PointerType::get(pointeeType, *storageClass);
}

// runtime-array-type ::= `!spirv.rtarray` `<` element-type
//                        (`,` `stride` `=` integer-literal)? `>`
static Type parseRuntimeArrayType(SPIRVDialect const &dialect,
                                  DialectAsmParser &parser) {
  if (parser.parseLess())
    return Type();

  Type elementType = parseAndVerifyType(dialect, parser);
  if (!elementType)
    return Type();

  unsigned stride = 0;
  if (failed(parseOptionalArrayStride(dialect, parser, stride)))
    return Type();

  if (parser.parseGreater())
    return Type();
  return RuntimeArrayType::get(elementType, stride);
}

// matrix-type ::= `!spirv.matrix` `<` integer-literal `x` element-type `>`
static Type parseMatrixType(SPIRVDialect const &dialect,
                            DialectAsmParser &parser) {
  if (parser.parseLess())
    return Type();

  SmallVector<int64_t, 1> countDims;
  SMLoc countLoc = parser.getCurrentLocation();
  if (parser.parseDimensionList(countDims, /*allowDynamic=*/false))
    return Type();
  if (countDims.size() != 1) {
    parser.emitError(countLoc, "expected single unsigned "
                               "integer for number of columns");
    return Type();
  }

  int64_t columnCount = countDims[0];
  // According to the specification, Matrices can have 2, 3, or 4 columns
  if (columnCount < 2 || columnCount > 4) {
    parser.emitError(countLoc, "matrix is expected to have 2, 3, or 4 "
                               "columns");
    return Type();
  }

  Type columnType = parseAndVerifyMatrixType(dialect, parser);
  if (!columnType)
    return Type();

  if (parser.parseGreater())
    return Type();

  return MatrixType::get(columnType, columnCount);
}

// Specialize this function to parse each of the parameters that define an
// ImageType. By default it assumes this is an enum type.
template <typename ValTy>
static std::optional<ValTy> parseAndVerify(SPIRVDialect const &dialect,
                                           DialectAsmParser &parser) {
  StringRef enumSpec;
  SMLoc enumLoc = parser.getCurrentLocation();
  if (parser.parseKeyword(&enumSpec)) {
    return std::nullopt;
  }

  auto val = spirv::symbolizeEnum<ValTy>(enumSpec);
  if (!val)
    parser.emitError(enumLoc, "unknown attribute: '") << enumSpec << "'";
  return val;
}

template <>
std::optional<Type> parseAndVerify<Type>(SPIRVDialect const &dialect,
                                         DialectAsmParser &parser) {
  // TODO: Further verify that the element type can be sampled
  auto ty = parseAndVerifyType(dialect, parser);
  if (!ty)
    return std::nullopt;
  return ty;
}

template <typename IntTy>
static std::optional<IntTy> parseAndVerifyInteger(SPIRVDialect const &dialect,
                                                  DialectAsmParser &parser) {
  IntTy offsetVal = std::numeric_limits<IntTy>::max();
  if (parser.parseInteger(offsetVal))
    return std::nullopt;
  return offsetVal;
}

template <>
std::optional<unsigned> parseAndVerify<unsigned>(SPIRVDialect const &dialect,
                                                 DialectAsmParser &parser) {
  return parseAndVerifyInteger<unsigned>(dialect, parser);
}

namespace {
// Functor object to parse a comma separated list of specs. The function
// parseAndVerify does the actual parsing and verification of individual
// elements. This is a functor since parsing the last element of the list
// (termination condition) needs partial specialization.
template <typename ParseType, typename... Args>
struct ParseCommaSeparatedList {
  std::optional<std::tuple<ParseType, Args...>>
  operator()(SPIRVDialect const &dialect, DialectAsmParser &parser) const {
    auto parseVal = parseAndVerify<ParseType>(dialect, parser);
    if (!parseVal)
      return std::nullopt;

    auto numArgs = std::tuple_size<std::tuple<Args...>>::value;
    if (numArgs != 0 && failed(parser.parseComma()))
      return std::nullopt;
    auto remainingValues = ParseCommaSeparatedList<Args...>{}(dialect, parser);
    if (!remainingValues)
      return std::nullopt;
    return std::tuple_cat(std::tuple<ParseType>(parseVal.value()),
                          remainingValues.value());
  }
};

// Partial specialization of the function to parse a comma separated list of
// specs to parse the last element of the list.
template <typename ParseType>
struct ParseCommaSeparatedList<ParseType> {
  std::optional<std::tuple<ParseType>>
  operator()(SPIRVDialect const &dialect, DialectAsmParser &parser) const {
    if (auto value = parseAndVerify<ParseType>(dialect, parser))
      return std::tuple<ParseType>(*value);
    return std::nullopt;
  }
};
} // namespace

// dim ::= `1D` | `2D` | `3D` | `Cube` | <and other SPIR-V Dim specifiers...>
//
// depth-info ::= `NoDepth` | `IsDepth` | `DepthUnknown`
//
// arrayed-info ::= `NonArrayed` | `Arrayed`
//
// sampling-info ::= `SingleSampled` | `MultiSampled`
//
// sampler-use-info ::= `SamplerUnknown` | `NeedSampler` |  `NoSampler`
//
// format ::= `Unknown` | `Rgba32f` | <and other SPIR-V Image formats...>
//
// image-type ::= `!spirv.image<` element-type `,` dim `,` depth-info `,`
//                              arrayed-info `,` sampling-info `,`
//                              sampler-use-info `,` format `>`
static Type parseImageType(SPIRVDialect const &dialect,
                           DialectAsmParser &parser) {
  if (parser.parseLess())
    return Type();

  auto value =
      ParseCommaSeparatedList<Type, Dim, ImageDepthInfo, ImageArrayedInfo,
                              ImageSamplingInfo, ImageSamplerUseInfo,
                              ImageFormat>{}(dialect, parser);
  if (!value)
    return Type();

  if (parser.parseGreater())
    return Type();
  return ImageType::get(*value);
}

// sampledImage-type :: = `!spirv.sampledImage<` image-type `>`
static Type parseSampledImageType(SPIRVDialect const &dialect,
                                  DialectAsmParser &parser) {
  if (parser.parseLess())
    return Type();

  Type parsedType = parseAndVerifySampledImageType(dialect, parser);
  if (!parsedType)
    return Type();

  if (parser.parseGreater())
    return Type();
  return SampledImageType::get(parsedType);
}

// Parse decorations associated with a member.
static ParseResult parseStructMemberDecorations(
    SPIRVDialect const &dialect, DialectAsmParser &parser,
    ArrayRef<Type> memberTypes,
    SmallVectorImpl<StructType::OffsetInfo> &offsetInfo,
    SmallVectorImpl<StructType::MemberDecorationInfo> &memberDecorationInfo) {

  // Check if the first element is offset.
  SMLoc offsetLoc = parser.getCurrentLocation();
  StructType::OffsetInfo offset = 0;
  OptionalParseResult offsetParseResult = parser.parseOptionalInteger(offset);
  if (offsetParseResult.has_value()) {
    if (failed(*offsetParseResult))
      return failure();

    if (offsetInfo.size() != memberTypes.size() - 1) {
      return parser.emitError(offsetLoc,
                              "offset specification must be given for "
                              "all members");
    }
    offsetInfo.push_back(offset);
  }

  // Check for no spirv::Decorations.
  if (succeeded(parser.parseOptionalRSquare()))
    return success();

  // If there was an offset, make sure to parse the comma.
  if (offsetParseResult.has_value() && parser.parseComma())
    return failure();

  // Check for spirv::Decorations.
  auto parseDecorations = [&]() {
    auto memberDecoration = parseAndVerify<spirv::Decoration>(dialect, parser);
    if (!memberDecoration)
      return failure();

    // Parse member decoration value if it exists.
    if (succeeded(parser.parseOptionalEqual())) {
      auto memberDecorationValue =
          parseAndVerifyInteger<uint32_t>(dialect, parser);

      if (!memberDecorationValue)
        return failure();

      memberDecorationInfo.emplace_back(
          static_cast<uint32_t>(memberTypes.size() - 1), 1,
          memberDecoration.value(), memberDecorationValue.value());
    } else {
      memberDecorationInfo.emplace_back(
          static_cast<uint32_t>(memberTypes.size() - 1), 0,
          memberDecoration.value(), 0);
    }
    return success();
  };
  if (failed(parser.parseCommaSeparatedList(parseDecorations)) ||
      failed(parser.parseRSquare()))
    return failure();

  return success();
}

// struct-member-decoration ::= integer-literal? spirv-decoration*
// struct-type ::=
//             `!spirv.struct<` (id `,`)?
//                          `(`
//                            (spirv-type (`[` struct-member-decoration `]`)?)*
//                          `)>`
static Type parseStructType(SPIRVDialect const &dialect,
                            DialectAsmParser &parser) {
  // TODO: This function is quite lengthy. Break it down into smaller chunks.

  if (parser.parseLess())
    return Type();

  StringRef identifier;
  FailureOr<DialectAsmParser::CyclicParseReset> cyclicParse;

  // Check if this is an identified struct type.
  if (succeeded(parser.parseOptionalKeyword(&identifier))) {
    // Check if this is a possible recursive reference.
    auto structType =
        StructType::getIdentified(dialect.getContext(), identifier);
    cyclicParse = parser.tryStartCyclicParse(structType);
    if (succeeded(parser.parseOptionalGreater())) {
      if (succeeded(cyclicParse)) {
        parser.emitError(
            parser.getNameLoc(),
            "recursive struct reference not nested in struct definition");

        return Type();
      }

      return structType;
    }

    if (failed(parser.parseComma()))
      return Type();

    if (failed(cyclicParse)) {
      parser.emitError(parser.getNameLoc(),
                       "identifier already used for an enclosing struct");
      return Type();
    }
  }

  if (failed(parser.parseLParen()))
    return Type();

  if (succeeded(parser.parseOptionalRParen()) &&
      succeeded(parser.parseOptionalGreater())) {
    return StructType::getEmpty(dialect.getContext(), identifier);
  }

  StructType idStructTy;

  if (!identifier.empty())
    idStructTy = StructType::getIdentified(dialect.getContext(), identifier);

  SmallVector<Type, 4> memberTypes;
  SmallVector<StructType::OffsetInfo, 4> offsetInfo;
  SmallVector<StructType::MemberDecorationInfo, 4> memberDecorationInfo;

  do {
    Type memberType;
    if (parser.parseType(memberType))
      return Type();
    memberTypes.push_back(memberType);

    if (succeeded(parser.parseOptionalLSquare()))
      if (parseStructMemberDecorations(dialect, parser, memberTypes, offsetInfo,
                                       memberDecorationInfo))
        return Type();
  } while (succeeded(parser.parseOptionalComma()));

  if (!offsetInfo.empty() && memberTypes.size() != offsetInfo.size()) {
    parser.emitError(parser.getNameLoc(),
                     "offset specification must be given for all members");
    return Type();
  }

  if (failed(parser.parseRParen()) || failed(parser.parseGreater()))
    return Type();

  if (!identifier.empty()) {
    if (failed(idStructTy.trySetBody(memberTypes, offsetInfo,
                                     memberDecorationInfo)))
      return Type();
    return idStructTy;
  }

  return StructType::get(memberTypes, offsetInfo, memberDecorationInfo);
}

// spirv-type ::= array-type
//              | element-type
//              | image-type
//              | pointer-type
//              | runtime-array-type
//              | sampled-image-type
//              | struct-type
Type SPIRVDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();

  if (keyword == "array")
    return parseArrayType(*this, parser);
  if (keyword == "coopmatrix")
    return parseCooperativeMatrixType(*this, parser);
  if (keyword == "image")
    return parseImageType(*this, parser);
  if (keyword == "ptr")
    return parsePointerType(*this, parser);
  if (keyword == "rtarray")
    return parseRuntimeArrayType(*this, parser);
  if (keyword == "sampled_image")
    return parseSampledImageType(*this, parser);
  if (keyword == "struct")
    return parseStructType(*this, parser);
  if (keyword == "matrix")
    return parseMatrixType(*this, parser);
  if (keyword == "arm.tensor")
    return parseTensorArmType(*this, parser);
  parser.emitError(parser.getNameLoc(), "unknown SPIR-V type: ") << keyword;
  return Type();
}

//===----------------------------------------------------------------------===//
// Type Printing
//===----------------------------------------------------------------------===//

static void print(ArrayType type, DialectAsmPrinter &os) {
  os << "array<" << type.getNumElements() << " x " << type.getElementType();
  if (unsigned stride = type.getArrayStride())
    os << ", stride=" << stride;
  os << ">";
}

static void print(RuntimeArrayType type, DialectAsmPrinter &os) {
  os << "rtarray<" << type.getElementType();
  if (unsigned stride = type.getArrayStride())
    os << ", stride=" << stride;
  os << ">";
}

static void print(PointerType type, DialectAsmPrinter &os) {
  os << "ptr<" << type.getPointeeType() << ", "
     << stringifyStorageClass(type.getStorageClass()) << ">";
}

static void print(ImageType type, DialectAsmPrinter &os) {
  os << "image<" << type.getElementType() << ", " << stringifyDim(type.getDim())
     << ", " << stringifyImageDepthInfo(type.getDepthInfo()) << ", "
     << stringifyImageArrayedInfo(type.getArrayedInfo()) << ", "
     << stringifyImageSamplingInfo(type.getSamplingInfo()) << ", "
     << stringifyImageSamplerUseInfo(type.getSamplerUseInfo()) << ", "
     << stringifyImageFormat(type.getImageFormat()) << ">";
}

static void print(SampledImageType type, DialectAsmPrinter &os) {
  os << "sampled_image<" << type.getImageType() << ">";
}

static void print(StructType type, DialectAsmPrinter &os) {
  FailureOr<AsmPrinter::CyclicPrintReset> cyclicPrint;

  os << "struct<";

  if (type.isIdentified()) {
    os << type.getIdentifier();

    cyclicPrint = os.tryStartCyclicPrint(type);
    if (failed(cyclicPrint)) {
      os << ">";
      return;
    }

    os << ", ";
  }

  os << "(";

  auto printMember = [&](unsigned i) {
    os << type.getElementType(i);
    SmallVector<spirv::StructType::MemberDecorationInfo, 0> decorations;
    type.getMemberDecorations(i, decorations);
    if (type.hasOffset() || !decorations.empty()) {
      os << " [";
      if (type.hasOffset()) {
        os << type.getMemberOffset(i);
        if (!decorations.empty())
          os << ", ";
      }
      auto eachFn = [&os](spirv::StructType::MemberDecorationInfo decoration) {
        os << stringifyDecoration(decoration.decoration);
        if (decoration.hasValue) {
          os << "=" << decoration.decorationValue;
        }
      };
      llvm::interleaveComma(decorations, os, eachFn);
      os << "]";
    }
  };
  llvm::interleaveComma(llvm::seq<unsigned>(0, type.getNumElements()), os,
                        printMember);
  os << ")>";
}

static void print(CooperativeMatrixType type, DialectAsmPrinter &os) {
  os << "coopmatrix<" << type.getRows() << "x" << type.getColumns() << "x"
     << type.getElementType() << ", " << type.getScope() << ", "
     << type.getUse() << ">";
}

static void print(MatrixType type, DialectAsmPrinter &os) {
  os << "matrix<" << type.getNumColumns() << " x " << type.getColumnType();
  os << ">";
}

static void print(TensorArmType type, DialectAsmPrinter &os) {
  os << "arm.tensor<";

  llvm::interleave(
      type.getShape(), os,
      [&](int64_t dim) {
        if (ShapedType::isDynamic(dim))
          os << '?';
        else
          os << dim;
      },
      "x");
  if (!type.hasRank()) {
    os << "*";
  }
  os << "x" << type.getElementType() << ">";
}

void SPIRVDialect::printType(Type type, DialectAsmPrinter &os) const {
  TypeSwitch<Type>(type)
      .Case<ArrayType, CooperativeMatrixType, PointerType, RuntimeArrayType,
            ImageType, SampledImageType, StructType, MatrixType, TensorArmType>(
          [&](auto type) { print(type, os); })
      .Default([](Type) { llvm_unreachable("unhandled SPIR-V type"); });
}

//===----------------------------------------------------------------------===//
// Constant
//===----------------------------------------------------------------------===//

Operation *SPIRVDialect::materializeConstant(OpBuilder &builder,
                                             Attribute value, Type type,
                                             Location loc) {
  if (auto poison = dyn_cast<ub::PoisonAttr>(value))
    return builder.create<ub::PoisonOp>(loc, type, poison);

  if (!spirv::ConstantOp::isBuildableWith(type))
    return nullptr;

  return builder.create<spirv::ConstantOp>(loc, type, value);
}

//===----------------------------------------------------------------------===//
// Shader Interface ABI
//===----------------------------------------------------------------------===//

LogicalResult SPIRVDialect::verifyOperationAttribute(Operation *op,
                                                     NamedAttribute attribute) {
  StringRef symbol = attribute.getName().strref();
  Attribute attr = attribute.getValue();

  if (symbol == spirv::getEntryPointABIAttrName()) {
    if (!llvm::isa<spirv::EntryPointABIAttr>(attr)) {
      return op->emitError("'")
             << symbol << "' attribute must be an entry point ABI attribute";
    }
  } else if (symbol == spirv::getTargetEnvAttrName()) {
    if (!llvm::isa<spirv::TargetEnvAttr>(attr))
      return op->emitError("'") << symbol << "' must be a spirv::TargetEnvAttr";
  } else {
    return op->emitError("found unsupported '")
           << symbol << "' attribute on operation";
  }

  return success();
}

/// Verifies the given SPIR-V `attribute` attached to a value of the given
/// `valueType` is valid.
static LogicalResult verifyRegionAttribute(Location loc, Type valueType,
                                           NamedAttribute attribute) {
  StringRef symbol = attribute.getName().strref();
  Attribute attr = attribute.getValue();

  if (symbol == spirv::getInterfaceVarABIAttrName()) {
    auto varABIAttr = llvm::dyn_cast<spirv::InterfaceVarABIAttr>(attr);
    if (!varABIAttr)
      return emitError(loc, "'")
             << symbol << "' must be a spirv::InterfaceVarABIAttr";

    if (varABIAttr.getStorageClass() && !valueType.isIntOrIndexOrFloat())
      return emitError(loc, "'") << symbol
                                 << "' attribute cannot specify storage class "
                                    "when attaching to a non-scalar value";
    return success();
  }
  if (symbol == spirv::DecorationAttr::name) {
    if (!isa<spirv::DecorationAttr>(attr))
      return emitError(loc, "'")
             << symbol << "' must be a spirv::DecorationAttr";
    return success();
  }

  return emitError(loc, "found unsupported '")
         << symbol << "' attribute on region argument";
}

LogicalResult SPIRVDialect::verifyRegionArgAttribute(Operation *op,
                                                     unsigned regionIndex,
                                                     unsigned argIndex,
                                                     NamedAttribute attribute) {
  auto funcOp = dyn_cast<FunctionOpInterface>(op);
  if (!funcOp)
    return success();
  Type argType = funcOp.getArgumentTypes()[argIndex];

  return verifyRegionAttribute(op->getLoc(), argType, attribute);
}

LogicalResult SPIRVDialect::verifyRegionResultAttribute(
    Operation *op, unsigned /*regionIndex*/, unsigned /*resultIndex*/,
    NamedAttribute attribute) {
  return op->emitError("cannot attach SPIR-V attributes to region result");
}
