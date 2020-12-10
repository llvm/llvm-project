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

#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/ParserUtils.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace spirv {
#include "mlir/Dialect/SPIRV/SPIRVOpUtils.inc"
} // namespace spirv
} // namespace mlir

using namespace mlir;
using namespace mlir::spirv;

//===----------------------------------------------------------------------===//
// InlinerInterface
//===----------------------------------------------------------------------===//

/// Returns true if the given region contains spv.Return or spv.ReturnValue ops.
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
                       BlockAndValueMapping &) const final {
    // Return true here when inlining into spv.func, spv.selection, and
    // spv.loop operations.
    auto *op = dest->getParentOp();
    return isa<spirv::FuncOp, spirv::SelectionOp, spirv::LoopOp>(op);
  }

  /// Returns true if the given operation 'op', that is registered to this
  /// dialect, can be inlined into the region 'dest' that is attached to an
  /// operation registered to the current dialect.
  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       BlockAndValueMapping &) const final {
    // TODO: Enable inlining structured control flows with return.
    if ((isa<spirv::SelectionOp, spirv::LoopOp>(op)) &&
        containsReturn(op->getRegion(0)))
      return false;
    // TODO: we need to filter OpKill here to avoid inlining it to
    // a loop continue construct:
    // https://github.com/KhronosGroup/SPIRV-Headers/issues/86
    // However OpKill is fragment shader specific and we don't support it yet.
    return true;
  }

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(Operation *op, Block *newDest) const final {
    if (auto returnOp = dyn_cast<spirv::ReturnOp>(op)) {
      OpBuilder(op).create<spirv::BranchOp>(op->getLoc(), newDest);
      op->erase();
    } else if (auto retValOp = dyn_cast<spirv::ReturnValueOp>(op)) {
      llvm_unreachable("unimplemented spv.ReturnValue in inliner");
    }
  }

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToRepl) const final {
    // Only spv.ReturnValue needs to be handled here.
    auto retValOp = dyn_cast<spirv::ReturnValueOp>(op);
    if (!retValOp)
      return;

    // Replace the values directly with the return operands.
    assert(valuesToRepl.size() == 1 &&
           "spv.ReturnValue expected to only handle one result");
    valuesToRepl.front().replaceAllUsesWith(retValOp.value());
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// SPIR-V Dialect
//===----------------------------------------------------------------------===//

void SPIRVDialect::initialize() {
  addTypes<ArrayType, CooperativeMatrixNVType, ImageType, MatrixType,
           PointerType, RuntimeArrayType, StructType>();

  addAttributes<InterfaceVarABIAttr, TargetEnvAttr, VerCapExtAttr>();

  // Add SPIR-V ops.
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/SPIRV/SPIRVOps.cpp.inc"
      >();

  addInterfaces<SPIRVInlinerInterface>();

  // Allow unknown operations because SPIR-V is extensible.
  allowUnknownOperations();
}

std::string SPIRVDialect::getAttributeName(Decoration decoration) {
  return llvm::convertToSnakeFromCamelCase(stringifyDecoration(decoration));
}

//===----------------------------------------------------------------------===//
// Type Parsing
//===----------------------------------------------------------------------===//

// Forward declarations.
template <typename ValTy>
static Optional<ValTy> parseAndVerify(SPIRVDialect const &dialect,
                                      DialectAsmParser &parser);
template <>
Optional<Type> parseAndVerify<Type>(SPIRVDialect const &dialect,
                                    DialectAsmParser &parser);

template <>
Optional<unsigned> parseAndVerify<unsigned>(SPIRVDialect const &dialect,
                                            DialectAsmParser &parser);

static Type parseAndVerifyType(SPIRVDialect const &dialect,
                               DialectAsmParser &parser) {
  Type type;
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  if (parser.parseType(type))
    return Type();

  // Allow SPIR-V dialect types
  if (&type.getDialect() == &dialect)
    return type;

  // Check other allowed types
  if (auto t = type.dyn_cast<FloatType>()) {
    if (type.isBF16()) {
      parser.emitError(typeLoc, "cannot use 'bf16' to compose SPIR-V types");
      return Type();
    }
  } else if (auto t = type.dyn_cast<IntegerType>()) {
    if (!ScalarType::isValid(t)) {
      parser.emitError(typeLoc,
                       "only 1/8/16/32/64-bit integer type allowed but found ")
          << type;
      return Type();
    }
  } else if (auto t = type.dyn_cast<VectorType>()) {
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
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  if (parser.parseType(type))
    return Type();

  if (auto t = type.dyn_cast<VectorType>()) {
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

    if (!t.getElementType().isa<FloatType>()) {
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

  llvm::SMLoc strideLoc = parser.getCurrentLocation();
  Optional<unsigned> optStride = parseAndVerify<unsigned>(dialect, parser);
  if (!optStride)
    return failure();

  if (!(stride = optStride.getValue())) {
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
// array-type ::= `!spv.array` `<` integer-literal `x` element-type
//                (`,` `stride` `=` integer-literal)? `>`
static Type parseArrayType(SPIRVDialect const &dialect,
                           DialectAsmParser &parser) {
  if (parser.parseLess())
    return Type();

  SmallVector<int64_t, 1> countDims;
  llvm::SMLoc countLoc = parser.getCurrentLocation();
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

// cooperative-matrix-type ::= `!spv.coopmatrix` `<` element-type ',' scope ','
//                                                   rows ',' columns>`
static Type parseCooperativeMatrixType(SPIRVDialect const &dialect,
                                       DialectAsmParser &parser) {
  if (parser.parseLess())
    return Type();

  SmallVector<int64_t, 2> dims;
  llvm::SMLoc countLoc = parser.getCurrentLocation();
  if (parser.parseDimensionList(dims, /*allowDynamic=*/false))
    return Type();

  if (dims.size() != 2) {
    parser.emitError(countLoc, "expected rows and columns size");
    return Type();
  }

  auto elementTy = parseAndVerifyType(dialect, parser);
  if (!elementTy)
    return Type();

  Scope scope;
  if (parser.parseComma() || parseEnumKeywordAttr(scope, parser, "scope <id>"))
    return Type();

  if (parser.parseGreater())
    return Type();
  return CooperativeMatrixNVType::get(elementTy, scope, dims[0], dims[1]);
}

// TODO: Reorder methods to be utilities first and parse*Type
// methods in alphabetical order
//
// storage-class ::= `UniformConstant`
//                 | `Uniform`
//                 | `Workgroup`
//                 | <and other storage classes...>
//
// pointer-type ::= `!spv.ptr<` element-type `,` storage-class `>`
static Type parsePointerType(SPIRVDialect const &dialect,
                             DialectAsmParser &parser) {
  if (parser.parseLess())
    return Type();

  auto pointeeType = parseAndVerifyType(dialect, parser);
  if (!pointeeType)
    return Type();

  StringRef storageClassSpec;
  llvm::SMLoc storageClassLoc = parser.getCurrentLocation();
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

// runtime-array-type ::= `!spv.rtarray` `<` element-type
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

// matrix-type ::= `!spv.matrix` `<` integer-literal `x` element-type `>`
static Type parseMatrixType(SPIRVDialect const &dialect,
                            DialectAsmParser &parser) {
  if (parser.parseLess())
    return Type();

  SmallVector<int64_t, 1> countDims;
  llvm::SMLoc countLoc = parser.getCurrentLocation();
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
static Optional<ValTy> parseAndVerify(SPIRVDialect const &dialect,
                                      DialectAsmParser &parser) {
  StringRef enumSpec;
  llvm::SMLoc enumLoc = parser.getCurrentLocation();
  if (parser.parseKeyword(&enumSpec)) {
    return llvm::None;
  }

  auto val = spirv::symbolizeEnum<ValTy>(enumSpec);
  if (!val)
    parser.emitError(enumLoc, "unknown attribute: '") << enumSpec << "'";
  return val;
}

template <>
Optional<Type> parseAndVerify<Type>(SPIRVDialect const &dialect,
                                    DialectAsmParser &parser) {
  // TODO: Further verify that the element type can be sampled
  auto ty = parseAndVerifyType(dialect, parser);
  if (!ty)
    return llvm::None;
  return ty;
}

template <typename IntTy>
static Optional<IntTy> parseAndVerifyInteger(SPIRVDialect const &dialect,
                                             DialectAsmParser &parser) {
  IntTy offsetVal = std::numeric_limits<IntTy>::max();
  if (parser.parseInteger(offsetVal))
    return llvm::None;
  return offsetVal;
}

template <>
Optional<unsigned> parseAndVerify<unsigned>(SPIRVDialect const &dialect,
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
  Optional<std::tuple<ParseType, Args...>>
  operator()(SPIRVDialect const &dialect, DialectAsmParser &parser) const {
    auto parseVal = parseAndVerify<ParseType>(dialect, parser);
    if (!parseVal)
      return llvm::None;

    auto numArgs = std::tuple_size<std::tuple<Args...>>::value;
    if (numArgs != 0 && failed(parser.parseComma()))
      return llvm::None;
    auto remainingValues = ParseCommaSeparatedList<Args...>{}(dialect, parser);
    if (!remainingValues)
      return llvm::None;
    return std::tuple_cat(std::tuple<ParseType>(parseVal.getValue()),
                          remainingValues.getValue());
  }
};

// Partial specialization of the function to parse a comma separated list of
// specs to parse the last element of the list.
template <typename ParseType>
struct ParseCommaSeparatedList<ParseType> {
  Optional<std::tuple<ParseType>> operator()(SPIRVDialect const &dialect,
                                             DialectAsmParser &parser) const {
    if (auto value = parseAndVerify<ParseType>(dialect, parser))
      return std::tuple<ParseType>(value.getValue());
    return llvm::None;
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
// image-type ::= `!spv.image<` element-type `,` dim `,` depth-info `,`
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
  return ImageType::get(value.getValue());
}

// Parse decorations associated with a member.
static ParseResult parseStructMemberDecorations(
    SPIRVDialect const &dialect, DialectAsmParser &parser,
    ArrayRef<Type> memberTypes,
    SmallVectorImpl<StructType::OffsetInfo> &offsetInfo,
    SmallVectorImpl<StructType::MemberDecorationInfo> &memberDecorationInfo) {

  // Check if the first element is offset.
  llvm::SMLoc offsetLoc = parser.getCurrentLocation();
  StructType::OffsetInfo offset = 0;
  OptionalParseResult offsetParseResult = parser.parseOptionalInteger(offset);
  if (offsetParseResult.hasValue()) {
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
  if (offsetParseResult.hasValue() && parser.parseComma())
    return failure();

  // Check for spirv::Decorations.
  do {
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
          memberDecoration.getValue(), memberDecorationValue.getValue());
    } else {
      memberDecorationInfo.emplace_back(
          static_cast<uint32_t>(memberTypes.size() - 1), 0,
          memberDecoration.getValue(), 0);
    }

  } while (succeeded(parser.parseOptionalComma()));

  return parser.parseRSquare();
}

// struct-member-decoration ::= integer-literal? spirv-decoration*
// struct-type ::=
//             `!spv.struct<` (id `,`)?
//                          `(`
//                            (spirv-type (`[` struct-member-decoration `]`)?)*
//                          `)>`
static Type parseStructType(SPIRVDialect const &dialect,
                            DialectAsmParser &parser) {
  // TODO: This function is quite lengthy. Break it down into smaller chunks.

  // To properly resolve recursive references while parsing recursive struct
  // types, we need to maintain a list of enclosing struct type names. This set
  // maintains the names of struct types in which the type we are about to parse
  // is nested.
  //
  // Note: This has to be thread_local to enable multiple threads to safely
  // parse concurrently.
  thread_local llvm::SetVector<StringRef> structContext;

  static auto removeIdentifierAndFail =
      [](llvm::SetVector<StringRef> &structContext, StringRef identifier) {
        if (!identifier.empty())
          structContext.remove(identifier);

        return Type();
      };

  if (parser.parseLess())
    return Type();

  StringRef identifier;

  // Check if this is an identified struct type.
  if (succeeded(parser.parseOptionalKeyword(&identifier))) {
    // Check if this is a possible recursive reference.
    if (succeeded(parser.parseOptionalGreater())) {
      if (structContext.count(identifier) == 0) {
        parser.emitError(
            parser.getNameLoc(),
            "recursive struct reference not nested in struct definition");

        return Type();
      }

      return StructType::getIdentified(dialect.getContext(), identifier);
    }

    if (failed(parser.parseComma()))
      return Type();

    if (structContext.count(identifier) != 0) {
      parser.emitError(parser.getNameLoc(),
                       "identifier already used for an enclosing struct");

      return removeIdentifierAndFail(structContext, identifier);
    }

    structContext.insert(identifier);
  }

  if (failed(parser.parseLParen()))
    return removeIdentifierAndFail(structContext, identifier);

  if (succeeded(parser.parseOptionalRParen()) &&
      succeeded(parser.parseOptionalGreater())) {
    if (!identifier.empty())
      structContext.remove(identifier);

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
      return removeIdentifierAndFail(structContext, identifier);
    memberTypes.push_back(memberType);

    if (succeeded(parser.parseOptionalLSquare()))
      if (parseStructMemberDecorations(dialect, parser, memberTypes, offsetInfo,
                                       memberDecorationInfo))
        return removeIdentifierAndFail(structContext, identifier);
  } while (succeeded(parser.parseOptionalComma()));

  if (!offsetInfo.empty() && memberTypes.size() != offsetInfo.size()) {
    parser.emitError(parser.getNameLoc(),
                     "offset specification must be given for all members");
    return removeIdentifierAndFail(structContext, identifier);
  }

  if (failed(parser.parseRParen()) || failed(parser.parseGreater()))
    return removeIdentifierAndFail(structContext, identifier);

  if (!identifier.empty()) {
    if (failed(idStructTy.trySetBody(memberTypes, offsetInfo,
                                     memberDecorationInfo)))
      return Type();

    structContext.remove(identifier);
    return idStructTy;
  }

  return StructType::get(memberTypes, offsetInfo, memberDecorationInfo);
}

// spirv-type ::= array-type
//              | element-type
//              | image-type
//              | pointer-type
//              | runtime-array-type
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
  if (keyword == "struct")
    return parseStructType(*this, parser);
  if (keyword == "matrix")
    return parseMatrixType(*this, parser);
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

static void print(StructType type, DialectAsmPrinter &os) {
  thread_local llvm::SetVector<StringRef> structContext;

  os << "struct<";

  if (type.isIdentified()) {
    os << type.getIdentifier();

    if (structContext.count(type.getIdentifier())) {
      os << ">";
      return;
    }

    os << ", ";
    structContext.insert(type.getIdentifier());
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

  if (type.isIdentified())
    structContext.remove(type.getIdentifier());
}

static void print(CooperativeMatrixNVType type, DialectAsmPrinter &os) {
  os << "coopmatrix<" << type.getRows() << "x" << type.getColumns() << "x";
  os << type.getElementType() << ", " << stringifyScope(type.getScope());
  os << ">";
}

static void print(MatrixType type, DialectAsmPrinter &os) {
  os << "matrix<" << type.getNumColumns() << " x " << type.getColumnType();
  os << ">";
}

void SPIRVDialect::printType(Type type, DialectAsmPrinter &os) const {
  TypeSwitch<Type>(type)
      .Case<ArrayType, CooperativeMatrixNVType, PointerType, RuntimeArrayType,
            ImageType, StructType, MatrixType>(
          [&](auto type) { print(type, os); })
      .Default([](Type) { llvm_unreachable("unhandled SPIR-V type"); });
}

//===----------------------------------------------------------------------===//
// Attribute Parsing
//===----------------------------------------------------------------------===//

/// Parses a comma-separated list of keywords, invokes `processKeyword` on each
/// of the parsed keyword, and returns failure if any error occurs.
static ParseResult parseKeywordList(
    DialectAsmParser &parser,
    function_ref<LogicalResult(llvm::SMLoc, StringRef)> processKeyword) {
  if (parser.parseLSquare())
    return failure();

  // Special case for empty list.
  if (succeeded(parser.parseOptionalRSquare()))
    return success();

  // Keep parsing the keyword and an optional comma following it. If the comma
  // is successfully parsed, then we have more keywords to parse.
  do {
    auto loc = parser.getCurrentLocation();
    StringRef keyword;
    if (parser.parseKeyword(&keyword) || failed(processKeyword(loc, keyword)))
      return failure();
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseRSquare())
    return failure();

  return success();
}

/// Parses a spirv::InterfaceVarABIAttr.
static Attribute parseInterfaceVarABIAttr(DialectAsmParser &parser) {
  if (parser.parseLess())
    return {};

  Builder &builder = parser.getBuilder();

  if (parser.parseLParen())
    return {};

  IntegerAttr descriptorSetAttr;
  {
    auto loc = parser.getCurrentLocation();
    uint32_t descriptorSet = 0;
    auto descriptorSetParseResult = parser.parseOptionalInteger(descriptorSet);

    if (!descriptorSetParseResult.hasValue() ||
        failed(*descriptorSetParseResult)) {
      parser.emitError(loc, "missing descriptor set");
      return {};
    }
    descriptorSetAttr = builder.getI32IntegerAttr(descriptorSet);
  }

  if (parser.parseComma())
    return {};

  IntegerAttr bindingAttr;
  {
    auto loc = parser.getCurrentLocation();
    uint32_t binding = 0;
    auto bindingParseResult = parser.parseOptionalInteger(binding);

    if (!bindingParseResult.hasValue() || failed(*bindingParseResult)) {
      parser.emitError(loc, "missing binding");
      return {};
    }
    bindingAttr = builder.getI32IntegerAttr(binding);
  }

  if (parser.parseRParen())
    return {};

  IntegerAttr storageClassAttr;
  {
    if (succeeded(parser.parseOptionalComma())) {
      auto loc = parser.getCurrentLocation();
      StringRef storageClass;
      if (parser.parseKeyword(&storageClass))
        return {};

      if (auto storageClassSymbol =
              spirv::symbolizeStorageClass(storageClass)) {
        storageClassAttr = builder.getI32IntegerAttr(
            static_cast<uint32_t>(*storageClassSymbol));
      } else {
        parser.emitError(loc, "unknown storage class: ") << storageClass;
        return {};
      }
    }
  }

  if (parser.parseGreater())
    return {};

  return spirv::InterfaceVarABIAttr::get(descriptorSetAttr, bindingAttr,
                                         storageClassAttr);
}

static Attribute parseVerCapExtAttr(DialectAsmParser &parser) {
  if (parser.parseLess())
    return {};

  Builder &builder = parser.getBuilder();

  IntegerAttr versionAttr;
  {
    auto loc = parser.getCurrentLocation();
    StringRef version;
    if (parser.parseKeyword(&version) || parser.parseComma())
      return {};

    if (auto versionSymbol = spirv::symbolizeVersion(version)) {
      versionAttr =
          builder.getI32IntegerAttr(static_cast<uint32_t>(*versionSymbol));
    } else {
      parser.emitError(loc, "unknown version: ") << version;
      return {};
    }
  }

  ArrayAttr capabilitiesAttr;
  {
    SmallVector<Attribute, 4> capabilities;
    llvm::SMLoc errorloc;
    StringRef errorKeyword;

    auto processCapability = [&](llvm::SMLoc loc, StringRef capability) {
      if (auto capSymbol = spirv::symbolizeCapability(capability)) {
        capabilities.push_back(
            builder.getI32IntegerAttr(static_cast<uint32_t>(*capSymbol)));
        return success();
      }
      return errorloc = loc, errorKeyword = capability, failure();
    };
    if (parseKeywordList(parser, processCapability) || parser.parseComma()) {
      if (!errorKeyword.empty())
        parser.emitError(errorloc, "unknown capability: ") << errorKeyword;
      return {};
    }

    capabilitiesAttr = builder.getArrayAttr(capabilities);
  }

  ArrayAttr extensionsAttr;
  {
    SmallVector<Attribute, 1> extensions;
    llvm::SMLoc errorloc;
    StringRef errorKeyword;

    auto processExtension = [&](llvm::SMLoc loc, StringRef extension) {
      if (spirv::symbolizeExtension(extension)) {
        extensions.push_back(builder.getStringAttr(extension));
        return success();
      }
      return errorloc = loc, errorKeyword = extension, failure();
    };
    if (parseKeywordList(parser, processExtension)) {
      if (!errorKeyword.empty())
        parser.emitError(errorloc, "unknown extension: ") << errorKeyword;
      return {};
    }

    extensionsAttr = builder.getArrayAttr(extensions);
  }

  if (parser.parseGreater())
    return {};

  return spirv::VerCapExtAttr::get(versionAttr, capabilitiesAttr,
                                   extensionsAttr);
}

/// Parses a spirv::TargetEnvAttr.
static Attribute parseTargetEnvAttr(DialectAsmParser &parser) {
  if (parser.parseLess())
    return {};

  spirv::VerCapExtAttr tripleAttr;
  if (parser.parseAttribute(tripleAttr) || parser.parseComma())
    return {};

  // Parse [vendor[:device-type[:device-id]]]
  Vendor vendorID = Vendor::Unknown;
  DeviceType deviceType = DeviceType::Unknown;
  uint32_t deviceID = spirv::TargetEnvAttr::kUnknownDeviceID;
  {
    auto loc = parser.getCurrentLocation();
    StringRef vendorStr;
    if (succeeded(parser.parseOptionalKeyword(&vendorStr))) {
      if (auto vendorSymbol = spirv::symbolizeVendor(vendorStr)) {
        vendorID = *vendorSymbol;
      } else {
        parser.emitError(loc, "unknown vendor: ") << vendorStr;
      }

      if (succeeded(parser.parseOptionalColon())) {
        loc = parser.getCurrentLocation();
        StringRef deviceTypeStr;
        if (parser.parseKeyword(&deviceTypeStr))
          return {};
        if (auto deviceTypeSymbol = spirv::symbolizeDeviceType(deviceTypeStr)) {
          deviceType = *deviceTypeSymbol;
        } else {
          parser.emitError(loc, "unknown device type: ") << deviceTypeStr;
        }

        if (succeeded(parser.parseOptionalColon())) {
          loc = parser.getCurrentLocation();
          if (parser.parseInteger(deviceID))
            return {};
        }
      }
      if (parser.parseComma())
        return {};
    }
  }

  DictionaryAttr limitsAttr;
  {
    auto loc = parser.getCurrentLocation();
    if (parser.parseAttribute(limitsAttr))
      return {};

    if (!limitsAttr.isa<spirv::ResourceLimitsAttr>()) {
      parser.emitError(
          loc,
          "limits must be a dictionary attribute containing two 32-bit integer "
          "attributes 'max_compute_workgroup_invocations' and "
          "'max_compute_workgroup_size'");
      return {};
    }
  }

  if (parser.parseGreater())
    return {};

  return spirv::TargetEnvAttr::get(tripleAttr, vendorID, deviceType, deviceID,
                                   limitsAttr);
}

Attribute SPIRVDialect::parseAttribute(DialectAsmParser &parser,
                                       Type type) const {
  // SPIR-V attributes are dictionaries so they do not have type.
  if (type) {
    parser.emitError(parser.getNameLoc(), "unexpected type");
    return {};
  }

  // Parse the kind keyword first.
  StringRef attrKind;
  if (parser.parseKeyword(&attrKind))
    return {};

  if (attrKind == spirv::TargetEnvAttr::getKindName())
    return parseTargetEnvAttr(parser);
  if (attrKind == spirv::VerCapExtAttr::getKindName())
    return parseVerCapExtAttr(parser);
  if (attrKind == spirv::InterfaceVarABIAttr::getKindName())
    return parseInterfaceVarABIAttr(parser);

  parser.emitError(parser.getNameLoc(), "unknown SPIR-V attribute kind: ")
      << attrKind;
  return {};
}

//===----------------------------------------------------------------------===//
// Attribute Printing
//===----------------------------------------------------------------------===//

static void print(spirv::VerCapExtAttr triple, DialectAsmPrinter &printer) {
  auto &os = printer.getStream();
  printer << spirv::VerCapExtAttr::getKindName() << "<"
          << spirv::stringifyVersion(triple.getVersion()) << ", [";
  llvm::interleaveComma(
      triple.getCapabilities(), os,
      [&](spirv::Capability cap) { os << spirv::stringifyCapability(cap); });
  printer << "], [";
  llvm::interleaveComma(triple.getExtensionsAttr(), os, [&](Attribute attr) {
    os << attr.cast<StringAttr>().getValue();
  });
  printer << "]>";
}

static void print(spirv::TargetEnvAttr targetEnv, DialectAsmPrinter &printer) {
  printer << spirv::TargetEnvAttr::getKindName() << "<#spv.";
  print(targetEnv.getTripleAttr(), printer);
  spirv::Vendor vendorID = targetEnv.getVendorID();
  spirv::DeviceType deviceType = targetEnv.getDeviceType();
  uint32_t deviceID = targetEnv.getDeviceID();
  if (vendorID != spirv::Vendor::Unknown) {
    printer << ", " << spirv::stringifyVendor(vendorID);
    if (deviceType != spirv::DeviceType::Unknown) {
      printer << ":" << spirv::stringifyDeviceType(deviceType);
      if (deviceID != spirv::TargetEnvAttr::kUnknownDeviceID)
        printer << ":" << deviceID;
    }
  }
  printer << ", " << targetEnv.getResourceLimits() << ">";
}

static void print(spirv::InterfaceVarABIAttr interfaceVarABIAttr,
                  DialectAsmPrinter &printer) {
  printer << spirv::InterfaceVarABIAttr::getKindName() << "<("
          << interfaceVarABIAttr.getDescriptorSet() << ", "
          << interfaceVarABIAttr.getBinding() << ")";
  auto storageClass = interfaceVarABIAttr.getStorageClass();
  if (storageClass)
    printer << ", " << spirv::stringifyStorageClass(*storageClass);
  printer << ">";
}

void SPIRVDialect::printAttribute(Attribute attr,
                                  DialectAsmPrinter &printer) const {
  if (auto targetEnv = attr.dyn_cast<TargetEnvAttr>())
    print(targetEnv, printer);
  else if (auto vceAttr = attr.dyn_cast<VerCapExtAttr>())
    print(vceAttr, printer);
  else if (auto interfaceVarABIAttr = attr.dyn_cast<InterfaceVarABIAttr>())
    print(interfaceVarABIAttr, printer);
  else
    llvm_unreachable("unhandled SPIR-V attribute kind");
}

//===----------------------------------------------------------------------===//
// Constant
//===----------------------------------------------------------------------===//

Operation *SPIRVDialect::materializeConstant(OpBuilder &builder,
                                             Attribute value, Type type,
                                             Location loc) {
  if (!spirv::ConstantOp::isBuildableWith(type))
    return nullptr;

  return builder.create<spirv::ConstantOp>(loc, type, value);
}

//===----------------------------------------------------------------------===//
// Shader Interface ABI
//===----------------------------------------------------------------------===//

LogicalResult SPIRVDialect::verifyOperationAttribute(Operation *op,
                                                     NamedAttribute attribute) {
  StringRef symbol = attribute.first.strref();
  Attribute attr = attribute.second;

  // TODO: figure out a way to generate the description from the
  // StructAttr definition.
  if (symbol == spirv::getEntryPointABIAttrName()) {
    if (!attr.isa<spirv::EntryPointABIAttr>())
      return op->emitError("'")
             << symbol
             << "' attribute must be a dictionary attribute containing one "
                "32-bit integer elements attribute: 'local_size'";
  } else if (symbol == spirv::getTargetEnvAttrName()) {
    if (!attr.isa<spirv::TargetEnvAttr>())
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
  StringRef symbol = attribute.first.strref();
  Attribute attr = attribute.second;

  if (symbol != spirv::getInterfaceVarABIAttrName())
    return emitError(loc, "found unsupported '")
           << symbol << "' attribute on region argument";

  auto varABIAttr = attr.dyn_cast<spirv::InterfaceVarABIAttr>();
  if (!varABIAttr)
    return emitError(loc, "'")
           << symbol << "' must be a spirv::InterfaceVarABIAttr";

  if (varABIAttr.getStorageClass() && !valueType.isIntOrIndexOrFloat())
    return emitError(loc, "'") << symbol
                               << "' attribute cannot specify storage class "
                                  "when attaching to a non-scalar value";

  return success();
}

LogicalResult SPIRVDialect::verifyRegionArgAttribute(Operation *op,
                                                     unsigned regionIndex,
                                                     unsigned argIndex,
                                                     NamedAttribute attribute) {
  return verifyRegionAttribute(
      op->getLoc(), op->getRegion(regionIndex).getArgument(argIndex).getType(),
      attribute);
}

LogicalResult SPIRVDialect::verifyRegionResultAttribute(
    Operation *op, unsigned /*regionIndex*/, unsigned /*resultIndex*/,
    NamedAttribute attribute) {
  return op->emitError("cannot attach SPIR-V attributes to region result");
}
