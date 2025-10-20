//===- TranslateFromWasm.cpp - Translating to WasmSSA dialect -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the WebAssembly importer.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/Wasm/WasmBinaryEncoding.h"
#include "mlir/Target/Wasm/WasmImporter.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/LogicalResult.h"

#include <cstddef>
#include <cstdint>
#include <variant>

#define DEBUG_TYPE "wasm-translate"

static_assert(CHAR_BIT == 8,
              "This code expects std::byte to be exactly 8 bits");

using namespace mlir;
using namespace mlir::wasm;
using namespace mlir::wasmssa;

namespace {
using section_id_t = uint8_t;
enum struct WasmSectionType : section_id_t {
  CUSTOM = 0,
  TYPE = 1,
  IMPORT = 2,
  FUNCTION = 3,
  TABLE = 4,
  MEMORY = 5,
  GLOBAL = 6,
  EXPORT = 7,
  START = 8,
  ELEMENT = 9,
  CODE = 10,
  DATA = 11,
  DATACOUNT = 12
};

constexpr section_id_t highestWasmSectionID{
    static_cast<section_id_t>(WasmSectionType::DATACOUNT)};

#define APPLY_WASM_SEC_TRANSFORM                                               \
  WASM_SEC_TRANSFORM(CUSTOM)                                                   \
  WASM_SEC_TRANSFORM(TYPE)                                                     \
  WASM_SEC_TRANSFORM(IMPORT)                                                   \
  WASM_SEC_TRANSFORM(FUNCTION)                                                 \
  WASM_SEC_TRANSFORM(TABLE)                                                    \
  WASM_SEC_TRANSFORM(MEMORY)                                                   \
  WASM_SEC_TRANSFORM(GLOBAL)                                                   \
  WASM_SEC_TRANSFORM(EXPORT)                                                   \
  WASM_SEC_TRANSFORM(START)                                                    \
  WASM_SEC_TRANSFORM(ELEMENT)                                                  \
  WASM_SEC_TRANSFORM(CODE)                                                     \
  WASM_SEC_TRANSFORM(DATA)                                                     \
  WASM_SEC_TRANSFORM(DATACOUNT)

template <WasmSectionType>
constexpr const char *wasmSectionName = "";

#define WASM_SEC_TRANSFORM(section)                                            \
  template <>                                                                  \
  [[maybe_unused]] constexpr const char                                        \
      *wasmSectionName<WasmSectionType::section> = #section;
APPLY_WASM_SEC_TRANSFORM
#undef WASM_SEC_TRANSFORM

constexpr bool sectionShouldBeUnique(WasmSectionType secType) {
  return secType != WasmSectionType::CUSTOM;
}

template <std::byte... Bytes>
struct ByteSequence {};

/// Template class for representing a byte sequence of only one byte
template <std::byte Byte>
struct UniqueByte : ByteSequence<Byte> {};

[[maybe_unused]] constexpr ByteSequence<
    WasmBinaryEncoding::Type::i32, WasmBinaryEncoding::Type::i64,
    WasmBinaryEncoding::Type::f32, WasmBinaryEncoding::Type::f64,
    WasmBinaryEncoding::Type::v128> valueTypesEncodings{};

template <std::byte... allowedFlags>
constexpr bool isValueOneOf(std::byte value,
                            ByteSequence<allowedFlags...> = {}) {
  return ((value == allowedFlags) | ... | false);
}

template <std::byte... flags>
constexpr bool isNotIn(std::byte value, ByteSequence<flags...> = {}) {
  return !isValueOneOf<flags...>(value);
}

struct GlobalTypeRecord {
  Type type;
  bool isMutable;
};

struct TypeIdxRecord {
  size_t id;
};

struct SymbolRefContainer {
  FlatSymbolRefAttr symbol;
};

struct GlobalSymbolRefContainer : SymbolRefContainer {
  Type globalType;
};

struct FunctionSymbolRefContainer : SymbolRefContainer {
  FunctionType functionType;
};

using ImportDesc =
    std::variant<TypeIdxRecord, TableType, LimitType, GlobalTypeRecord>;

using parsed_inst_t = FailureOr<SmallVector<Value>>;

struct EmptyBlockMarker {};
using BlockTypeParseResult =
    std::variant<EmptyBlockMarker, TypeIdxRecord, Type>;

struct WasmModuleSymbolTables {
  SmallVector<FunctionSymbolRefContainer> funcSymbols;
  SmallVector<GlobalSymbolRefContainer> globalSymbols;
  SmallVector<SymbolRefContainer> memSymbols;
  SmallVector<SymbolRefContainer> tableSymbols;
  SmallVector<FunctionType> moduleFuncTypes;

  std::string getNewSymbolName(StringRef prefix, size_t id) const {
    return (prefix + Twine{id}).str();
  }

  std::string getNewFuncSymbolName() const {
    size_t id = funcSymbols.size();
    return getNewSymbolName("func_", id);
  }

  std::string getNewGlobalSymbolName() const {
    size_t id = globalSymbols.size();
    return getNewSymbolName("global_", id);
  }

  std::string getNewMemorySymbolName() const {
    size_t id = memSymbols.size();
    return getNewSymbolName("mem_", id);
  }

  std::string getNewTableSymbolName() const {
    size_t id = tableSymbols.size();
    return getNewSymbolName("table_", id);
  }
};

class ParserHead;

/// Wrapper around SmallVector to only allow access as push and pop on the
/// stack. Makes sure that there are no "free accesses" on the stack to preserve
/// its state.
/// This class also keep tracks of the Wasm labels defined by different ops,
/// which can be targeted by control flow ops. This can be modeled as part of
/// the Value Stack as Wasm control flow ops can only target enclosing labels.
class ValueStack {
private:
  struct LabelLevel {
    size_t stackIdx;
    LabelLevelOpInterface levelOp;
  };

public:
  bool empty() const { return values.empty(); }

  size_t size() const { return values.size(); }

  /// Pops values from the stack because they are being used in an operation.
  /// @param operandTypes The list of expected types of the operation, used
  ///   to know how many values to pop and check if the types match the
  ///   expectation.
  /// @param opLoc Location of the caller, used to report accurately the
  /// location
  ///   if an error occurs.
  /// @return Failure or the vector of popped values.
  FailureOr<SmallVector<Value>> popOperands(TypeRange operandTypes,
                                            Location *opLoc);

  /// Push the results of an operation to the stack so they can be used in a
  /// following operation.
  /// @param results The list of results of the operation
  /// @param opLoc Location of the caller, used to report accurately the
  /// location
  ///   if an error occurs.
  LogicalResult pushResults(ValueRange results, Location *opLoc);

  void addLabelLevel(LabelLevelOpInterface levelOp) {
    labelLevel.push_back({values.size(), levelOp});
    LDBG() << "Adding a new frame context to ValueStack";
  }

  void dropLabelLevel() {
    assert(!labelLevel.empty() && "Trying to drop a frame from empty context");
    auto newSize = labelLevel.pop_back_val().stackIdx;
    values.truncate(newSize);
  }
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// A simple dump function for debugging.
  /// Writes output to llvm::dbgs().
  LLVM_DUMP_METHOD void dump() const;
#endif

private:
  SmallVector<Value> values;
  SmallVector<LabelLevel> labelLevel;
};

using local_val_t = TypedValue<wasmssa::LocalRefType>;

class ExpressionParser {
public:
  using locals_t = SmallVector<local_val_t>;
  ExpressionParser(ParserHead &parser, WasmModuleSymbolTables const &symbols,
                   ArrayRef<local_val_t> initLocal)
      : parser{parser}, symbols{symbols}, locals{initLocal} {}

private:
  template <std::byte opCode>
  inline parsed_inst_t parseSpecificInstruction(OpBuilder &builder);

  template <typename valueT>
  parsed_inst_t
  parseConstInst(OpBuilder &builder,
                 std::enable_if_t<std::is_arithmetic_v<valueT>> * = nullptr);

  /// Construct an operation with \p numOperands operands and a single result.
  /// Each operand must have the same type. Suitable for e.g. binops, unary
  /// ops, etc.
  ///
  /// \p opcode - The WASM opcode to build.
  /// \p valueType - The operand and result type for the built instruction.
  /// \p numOperands - The number of operands for the built operation.
  ///
  /// \returns The parsed instruction result, or failure.
  template <typename opcode, typename valueType, unsigned int numOperands>
  inline parsed_inst_t
  buildNumericOp(OpBuilder &builder,
                 std::enable_if_t<std::is_arithmetic_v<valueType>> * = nullptr);

  /// Construct a conversion operation of type \p opType that takes a value from
  /// type \p inputType on the stack and will produce a value of type
  /// \p outputType.
  ///
  /// \p opType - The WASM dialect operation to build.
  /// \p inputType - The operand type for the built instruction.
  /// \p outputType - The result type for the built instruction.
  ///
  /// \returns The parsed instruction result, or failure.
  template <typename opType, typename inputType, typename outputType,
            typename... extraArgsT>
  inline parsed_inst_t buildConvertOp(OpBuilder &builder, extraArgsT...);

  /// This function generates a dispatch tree to associate an opcode with a
  /// parser. Parsers are registered by specialising the
  /// `parseSpecificInstruction` function for the op code to handle.
  ///
  /// The dispatcher is generated by recursively creating all possible patterns
  /// for an opcode and calling the relevant parser on the leaf.
  ///
  /// @tparam patternBitSize is the first bit for which the pattern is not fixed
  ///
  /// @tparam highBitPattern is the fixed pattern that this instance handles for
  /// the 8-patternBitSize bits
  template <size_t patternBitSize = 0, std::byte highBitPattern = std::byte{0}>
  inline parsed_inst_t dispatchToInstParser(std::byte opCode,
                                            OpBuilder &builder) {
    static_assert(patternBitSize <= 8,
                  "PatternBitSize is outside of range of opcode space! "
                  "(expected at most 8 bits)");
    if constexpr (patternBitSize < 8) {
      constexpr std::byte bitSelect{1 << (7 - patternBitSize)};
      constexpr std::byte nextHighBitPatternStem = highBitPattern << 1;
      constexpr size_t nextPatternBitSize = patternBitSize + 1;
      if ((opCode & bitSelect) != std::byte{0})
        return dispatchToInstParser<nextPatternBitSize,
                                    nextHighBitPatternStem | std::byte{1}>(
            opCode, builder);
      return dispatchToInstParser<nextPatternBitSize, nextHighBitPatternStem>(
          opCode, builder);
    } else {
      return parseSpecificInstruction<highBitPattern>(builder);
    }
  }

  ///
  /// RAII guard class for creating a nesting level
  ///
  struct NestingContextGuard {
    NestingContextGuard(ExpressionParser &parser, LabelLevelOpInterface levelOp)
        : parser{parser} {
      parser.addNestingContextLevel(levelOp);
    }
    NestingContextGuard(NestingContextGuard &&other) : parser{other.parser} {
      other.shouldDropOnDestruct = false;
    }
    NestingContextGuard(NestingContextGuard const &) = delete;
    ~NestingContextGuard() {
      if (shouldDropOnDestruct)
        parser.dropNestingContextLevel();
    }
    ExpressionParser &parser;
    bool shouldDropOnDestruct = true;
  };

  void addNestingContextLevel(LabelLevelOpInterface levelOp) {
    valueStack.addLabelLevel(levelOp);
  }

  void dropNestingContextLevel() {
    // Should always succeed as we are droping the frame that was previously
    // created.
    valueStack.dropLabelLevel();
  }

  llvm::FailureOr<FunctionType> getFuncTypeFor(OpBuilder &builder,
                                               EmptyBlockMarker) {
    return builder.getFunctionType({}, {});
  }

  llvm::FailureOr<FunctionType> getFuncTypeFor(OpBuilder &builder,
                                               TypeIdxRecord type) {
    if (type.id >= symbols.moduleFuncTypes.size())
      return emitError(*currentOpLoc,
                       "type index references nonexistent type (")
             << type.id << "). Only " << symbols.moduleFuncTypes.size()
             << " types are registered";
    return symbols.moduleFuncTypes[type.id];
  }

  llvm::FailureOr<FunctionType> getFuncTypeFor(OpBuilder &builder,
                                               Type valType) {
    return builder.getFunctionType({}, {valType});
  }

  llvm::FailureOr<FunctionType>
  getFuncTypeFor(OpBuilder &builder, BlockTypeParseResult parseResult) {
    return std::visit(
        [this, &builder](auto value) { return getFuncTypeFor(builder, value); },
        parseResult);
  }

  llvm::FailureOr<FunctionType>
  getFuncTypeFor(OpBuilder &builder,
                 llvm::FailureOr<BlockTypeParseResult> parseResult) {
    if (llvm::failed(parseResult))
      return failure();
    return getFuncTypeFor(builder, *parseResult);
  }

  llvm::FailureOr<FunctionType> parseBlockFuncType(OpBuilder &builder);

  struct ParseResultWithInfo {
    SmallVector<Value> opResults;
    std::byte endingByte;
  };

  template <typename FilterT = ByteSequence<WasmBinaryEncoding::endByte>>
  /// @param blockToFill: the block which content will be populated
  /// @param resType: the type that this block is supposed to return
  llvm::FailureOr<std::byte>
  parseBlockContent(OpBuilder &builder, Block *blockToFill, TypeRange resTypes,
                    Location opLoc, LabelLevelOpInterface levelOp,
                    FilterT parseEndBytes = {}) {
    OpBuilder::InsertionGuard guard{builder};
    builder.setInsertionPointToStart(blockToFill);
    LDBG() << "parsing a block of type "
           << builder.getFunctionType(blockToFill->getArgumentTypes(),
                                      resTypes);
    auto nC = addNesting(levelOp);

    if (failed(pushResults(blockToFill->getArguments())))
      return failure();
    auto bodyParsingRes = parse(builder, parseEndBytes);
    if (failed(bodyParsingRes))
      return failure();
    auto returnOperands = popOperands(resTypes);
    if (failed(returnOperands))
      return failure();
    builder.create<BlockReturnOp>(opLoc, *returnOperands);
    LDBG() << "end of parsing of a block";
    return bodyParsingRes->endingByte;
  }

public:
  template <std::byte ParseEndByte = WasmBinaryEncoding::endByte>
  parsed_inst_t parse(OpBuilder &builder, UniqueByte<ParseEndByte> = {});

  template <std::byte... ExpressionParseEnd>
  FailureOr<ParseResultWithInfo>
  parse(OpBuilder &builder,
        ByteSequence<ExpressionParseEnd...> parsingEndFilters);

  NestingContextGuard addNesting(LabelLevelOpInterface levelOp) {
    return NestingContextGuard{*this, levelOp};
  }

  FailureOr<llvm::SmallVector<Value>> popOperands(TypeRange operandTypes) {
    return valueStack.popOperands(operandTypes, &currentOpLoc.value());
  }

  LogicalResult pushResults(ValueRange results) {
    return valueStack.pushResults(results, &currentOpLoc.value());
  }

  /// The local.set and local.tee operations behave similarly and only differ
  /// on their return value. This function factorizes the behavior of the two
  /// operations in one place.
  template <typename OpToCreate>
  parsed_inst_t parseSetOrTee(OpBuilder &);

  /// Blocks and Loops have a similar format and differ only in how their exit
  /// is handled which doesn´t matter at parsing time. Factorizes in one
  /// function.
  template <typename OpToCreate>
  parsed_inst_t parseBlockLikeOp(OpBuilder &);

private:
  std::optional<Location> currentOpLoc;
  ParserHead &parser;
  WasmModuleSymbolTables const &symbols;
  locals_t locals;
  ValueStack valueStack;
};

class ParserHead {
public:
  ParserHead(StringRef src, StringAttr name) : head{src}, locName{name} {}
  ParserHead(ParserHead &&) = default;

private:
  ParserHead(ParserHead const &other) = default;

public:
  auto getLocation() const {
    return FileLineColLoc::get(locName, 0, anchorOffset + offset);
  }

  FailureOr<StringRef> consumeNBytes(size_t nBytes) {
    LDBG() << "Consume " << nBytes << " bytes";
    LDBG() << "  Bytes remaining: " << size();
    LDBG() << "  Current offset: " << offset;
    if (nBytes > size())
      return emitError(getLocation(), "trying to extract ")
             << nBytes << "bytes when only " << size() << "are available";

    StringRef res = head.slice(offset, offset + nBytes);
    offset += nBytes;
    LDBG() << "  Updated offset (+" << nBytes << "): " << offset;
    return res;
  }

  FailureOr<std::byte> consumeByte() {
    FailureOr<StringRef> res = consumeNBytes(1);
    if (failed(res))
      return failure();
    return std::byte{*res->bytes_begin()};
  }

  template <typename T>
  FailureOr<T> parseLiteral();

  FailureOr<uint32_t> parseVectorSize();

private:
  // TODO: This is equivalent to parseLiteral<uint32_t> and could be removed
  // if parseLiteral specialization were moved here, but default GCC on Ubuntu
  // 22.04 has bug with template specialization in class declaration
  inline FailureOr<uint32_t> parseUI32();
  inline FailureOr<int64_t> parseI64();

public:
  FailureOr<StringRef> parseName() {
    FailureOr<uint32_t> size = parseVectorSize();
    if (failed(size))
      return failure();

    return consumeNBytes(*size);
  }

  FailureOr<WasmSectionType> parseWasmSectionType() {
    FailureOr<std::byte> id = consumeByte();
    if (failed(id))
      return failure();
    if (std::to_integer<unsigned>(*id) > highestWasmSectionID)
      return emitError(getLocation(), "invalid section ID: ")
             << static_cast<int>(*id);
    return static_cast<WasmSectionType>(*id);
  }

  FailureOr<LimitType> parseLimit(MLIRContext *ctx) {
    using WasmLimits = WasmBinaryEncoding::LimitHeader;
    FileLineColLoc limitLocation = getLocation();
    FailureOr<std::byte> limitHeader = consumeByte();
    if (failed(limitHeader))
      return failure();

    if (isNotIn<WasmLimits::bothLimits, WasmLimits::lowLimitOnly>(*limitHeader))
      return emitError(limitLocation, "invalid limit header: ")
             << static_cast<int>(*limitHeader);
    FailureOr<uint32_t> minParse = parseUI32();
    if (failed(minParse))
      return failure();
    std::optional<uint32_t> max{std::nullopt};
    if (*limitHeader == WasmLimits::bothLimits) {
      FailureOr<uint32_t> maxParse = parseUI32();
      if (failed(maxParse))
        return failure();
      max = *maxParse;
    }
    return LimitType::get(ctx, *minParse, max);
  }

  FailureOr<Type> parseValueType(MLIRContext *ctx) {
    FileLineColLoc typeLoc = getLocation();
    FailureOr<std::byte> typeEncoding = consumeByte();
    if (failed(typeEncoding))
      return failure();
    switch (*typeEncoding) {
    case WasmBinaryEncoding::Type::i32:
      return IntegerType::get(ctx, 32);
    case WasmBinaryEncoding::Type::i64:
      return IntegerType::get(ctx, 64);
    case WasmBinaryEncoding::Type::f32:
      return Float32Type::get(ctx);
    case WasmBinaryEncoding::Type::f64:
      return Float64Type::get(ctx);
    case WasmBinaryEncoding::Type::v128:
      return IntegerType::get(ctx, 128);
    case WasmBinaryEncoding::Type::funcRef:
      return wasmssa::FuncRefType::get(ctx);
    case WasmBinaryEncoding::Type::externRef:
      return wasmssa::ExternRefType::get(ctx);
    default:
      return emitError(typeLoc, "invalid value type encoding: ")
             << static_cast<int>(*typeEncoding);
    }
  }

  FailureOr<GlobalTypeRecord> parseGlobalType(MLIRContext *ctx) {
    using WasmGlobalMut = WasmBinaryEncoding::GlobalMutability;
    FailureOr<Type> typeParsed = parseValueType(ctx);
    if (failed(typeParsed))
      return failure();
    FileLineColLoc mutLoc = getLocation();
    FailureOr<std::byte> mutSpec = consumeByte();
    if (failed(mutSpec))
      return failure();
    if (isNotIn<WasmGlobalMut::isConst, WasmGlobalMut::isMutable>(*mutSpec))
      return emitError(mutLoc, "invalid global mutability specifier: ")
             << static_cast<int>(*mutSpec);
    return GlobalTypeRecord{*typeParsed, *mutSpec == WasmGlobalMut::isMutable};
  }

  FailureOr<TupleType> parseResultType(MLIRContext *ctx) {
    FailureOr<uint32_t> nParamsParsed = parseVectorSize();
    if (failed(nParamsParsed))
      return failure();
    uint32_t nParams = *nParamsParsed;
    SmallVector<Type> res{};
    res.reserve(nParams);
    for (size_t i = 0; i < nParams; ++i) {
      FailureOr<Type> parsedType = parseValueType(ctx);
      if (failed(parsedType))
        return failure();
      res.push_back(*parsedType);
    }
    return TupleType::get(ctx, res);
  }

  FailureOr<FunctionType> parseFunctionType(MLIRContext *ctx) {
    FileLineColLoc typeLoc = getLocation();
    FailureOr<std::byte> funcTypeHeader = consumeByte();
    if (failed(funcTypeHeader))
      return failure();
    if (*funcTypeHeader != WasmBinaryEncoding::Type::funcType)
      return emitError(typeLoc, "invalid function type header byte. Expecting ")
             << std::to_integer<unsigned>(WasmBinaryEncoding::Type::funcType)
             << " got " << std::to_integer<unsigned>(*funcTypeHeader);
    FailureOr<TupleType> inputTypes = parseResultType(ctx);
    if (failed(inputTypes))
      return failure();

    FailureOr<TupleType> resTypes = parseResultType(ctx);
    if (failed(resTypes))
      return failure();

    return FunctionType::get(ctx, inputTypes->getTypes(), resTypes->getTypes());
  }

  FailureOr<TypeIdxRecord> parseTypeIndex() {
    FailureOr<uint32_t> res = parseUI32();
    if (failed(res))
      return failure();
    return TypeIdxRecord{*res};
  }

  FailureOr<TableType> parseTableType(MLIRContext *ctx) {
    FailureOr<Type> elmTypeParse = parseValueType(ctx);
    if (failed(elmTypeParse))
      return failure();
    if (!isWasmRefType(*elmTypeParse))
      return emitError(getLocation(), "invalid element type for table");
    FailureOr<LimitType> limitParse = parseLimit(ctx);
    if (failed(limitParse))
      return failure();
    return TableType::get(ctx, *elmTypeParse, *limitParse);
  }

  FailureOr<ImportDesc> parseImportDesc(MLIRContext *ctx) {
    FileLineColLoc importLoc = getLocation();
    FailureOr<std::byte> importType = consumeByte();
    auto packager = [](auto parseResult) -> FailureOr<ImportDesc> {
      if (failed(parseResult))
        return failure();
      return {*parseResult};
    };
    if (failed(importType))
      return failure();
    switch (*importType) {
    case WasmBinaryEncoding::Import::typeID:
      return packager(parseTypeIndex());
    case WasmBinaryEncoding::Import::tableType:
      return packager(parseTableType(ctx));
    case WasmBinaryEncoding::Import::memType:
      return packager(parseLimit(ctx));
    case WasmBinaryEncoding::Import::globalType:
      return packager(parseGlobalType(ctx));
    default:
      return emitError(importLoc, "invalid import type descriptor: ")
             << static_cast<int>(*importType);
    }
  }

  parsed_inst_t parseExpression(OpBuilder &builder,
                                WasmModuleSymbolTables const &symbols,
                                ArrayRef<local_val_t> locals = {}) {
    auto eParser = ExpressionParser{*this, symbols, locals};
    return eParser.parse(builder);
  }

  LogicalResult parseCodeFor(FuncOp func,
                             WasmModuleSymbolTables const &symbols) {
    SmallVector<local_val_t> locals{};
    // Populating locals with function argument
    Block &block = func.getBody().front();
    // Delete temporary return argument which was only created for IR validity
    assert(func.getBody().getBlocks().size() == 1 &&
           "Function should only have its default created block at this point");
    assert(block.getOperations().size() == 1 &&
           "Only the placeholder return op should be present at this point");
    auto returnOp = cast<ReturnOp>(&block.back());
    assert(returnOp);

    FailureOr<uint32_t> codeSizeInBytes = parseUI32();
    if (failed(codeSizeInBytes))
      return failure();
    FailureOr<StringRef> codeContent = consumeNBytes(*codeSizeInBytes);
    if (failed(codeContent))
      return failure();
    auto name = StringAttr::get(func->getContext(),
                                locName.str() + "::" + func.getSymName());
    auto cParser = ParserHead{*codeContent, name};
    FailureOr<uint32_t> localVecSize = cParser.parseVectorSize();
    if (failed(localVecSize))
      return failure();
    OpBuilder builder{&func.getBody().front().back()};
    for (auto arg : block.getArguments())
      locals.push_back(cast<TypedValue<LocalRefType>>(arg));
    // Declare the local ops
    uint32_t nVarVec = *localVecSize;
    for (size_t i = 0; i < nVarVec; ++i) {
      FileLineColLoc varLoc = cParser.getLocation();
      FailureOr<uint32_t> nSubVar = cParser.parseUI32();
      if (failed(nSubVar))
        return failure();
      FailureOr<Type> varT = cParser.parseValueType(func->getContext());
      if (failed(varT))
        return failure();
      for (size_t j = 0; j < *nSubVar; ++j) {
        auto local = LocalOp::create(builder, varLoc, *varT);
        locals.push_back(local.getResult());
      }
    }
    parsed_inst_t res = cParser.parseExpression(builder, symbols, locals);
    if (failed(res))
      return failure();
    if (!cParser.end())
      return emitError(cParser.getLocation(),
                       "unparsed garbage remaining at end of code block");
    ReturnOp::create(builder, func->getLoc(), *res);
    returnOp->erase();
    return success();
  }

  llvm::FailureOr<BlockTypeParseResult> parseBlockType(MLIRContext *ctx) {
    auto loc = getLocation();
    auto blockIndicator = peek();
    if (failed(blockIndicator))
      return failure();
    if (*blockIndicator == WasmBinaryEncoding::Type::emptyBlockType) {
      offset += 1;
      return {EmptyBlockMarker{}};
    }
    if (isValueOneOf(*blockIndicator, valueTypesEncodings))
      return parseValueType(ctx);
    /// Block type idx is a 32 bit positive integer encoded as a 33 bit signed
    /// value
    auto typeIdx = parseI64();
    if (failed(typeIdx))
      return failure();
    if (*typeIdx < 0 || *typeIdx > std::numeric_limits<uint32_t>::max())
      return emitError(loc, "type ID should be representable with an unsigned "
                            "32 bits integer. Got ")
             << *typeIdx;
    return {TypeIdxRecord{static_cast<uint32_t>(*typeIdx)}};
  }

  bool end() const { return curHead().empty(); }

  ParserHead copy() const { return *this; }

private:
  StringRef curHead() const { return head.drop_front(offset); }

  FailureOr<std::byte> peek() const {
    if (end())
      return emitError(
          getLocation(),
          "trying to peek at next byte, but input stream is empty");
    return static_cast<std::byte>(curHead().front());
  }

  size_t size() const { return head.size() - offset; }

  StringRef head;
  StringAttr locName;
  unsigned anchorOffset{0};
  unsigned offset{0};
};

template <>
FailureOr<float> ParserHead::parseLiteral<float>() {
  FailureOr<StringRef> bytes = consumeNBytes(4);
  if (failed(bytes))
    return failure();
  return llvm::support::endian::read<float>(bytes->bytes_begin(),
                                            llvm::endianness::little);
}

template <>
FailureOr<double> ParserHead::parseLiteral<double>() {
  FailureOr<StringRef> bytes = consumeNBytes(8);
  if (failed(bytes))
    return failure();
  return llvm::support::endian::read<double>(bytes->bytes_begin(),
                                             llvm::endianness::little);
}

template <>
FailureOr<uint32_t> ParserHead::parseLiteral<uint32_t>() {
  char const *error = nullptr;
  uint32_t res{0};
  unsigned encodingSize{0};
  StringRef src = curHead();
  uint64_t decoded = llvm::decodeULEB128(src.bytes_begin(), &encodingSize,
                                         src.bytes_end(), &error);
  if (error)
    return emitError(getLocation(), error);

  if (std::isgreater(decoded, std::numeric_limits<uint32_t>::max()))
    return emitError(getLocation()) << "literal does not fit on 32 bits";

  res = static_cast<uint32_t>(decoded);
  offset += encodingSize;
  return res;
}

template <>
FailureOr<int32_t> ParserHead::parseLiteral<int32_t>() {
  char const *error = nullptr;
  int32_t res{0};
  unsigned encodingSize{0};
  StringRef src = curHead();
  int64_t decoded = llvm::decodeSLEB128(src.bytes_begin(), &encodingSize,
                                        src.bytes_end(), &error);
  if (error)
    return emitError(getLocation(), error);
  if (std::isgreater(decoded, std::numeric_limits<int32_t>::max()) ||
      std::isgreater(std::numeric_limits<int32_t>::min(), decoded))
    return emitError(getLocation()) << "literal does not fit on 32 bits";

  res = static_cast<int32_t>(decoded);
  offset += encodingSize;
  return res;
}

template <>
FailureOr<int64_t> ParserHead::parseLiteral<int64_t>() {
  char const *error = nullptr;
  unsigned encodingSize{0};
  StringRef src = curHead();
  int64_t res = llvm::decodeSLEB128(src.bytes_begin(), &encodingSize,
                                    src.bytes_end(), &error);
  if (error)
    return emitError(getLocation(), error);

  offset += encodingSize;
  return res;
}

FailureOr<uint32_t> ParserHead::parseVectorSize() {
  return parseLiteral<uint32_t>();
}

inline FailureOr<uint32_t> ParserHead::parseUI32() {
  return parseLiteral<uint32_t>();
}

inline FailureOr<int64_t> ParserHead::parseI64() {
  return parseLiteral<int64_t>();
}

template <std::byte opCode>
inline parsed_inst_t ExpressionParser::parseSpecificInstruction(OpBuilder &) {
  return emitError(*currentOpLoc, "unknown instruction opcode: ")
         << static_cast<int>(opCode);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void ValueStack::dump() const {
  llvm::dbgs() << "================= Wasm ValueStack =======================\n";
  llvm::dbgs() << "size: " << size() << "\n";
  llvm::dbgs() << "nbFrames: " << labelLevel.size() << '\n';
  llvm::dbgs() << "<Top>"
               << "\n";
  // Stack is pushed to via push_back. Therefore the top of the stack is the
  // end of the vector. Iterate in reverse so that the first thing we print
  // is the top of the stack.
  auto indexGetter = [this]() {
    size_t idx = labelLevel.size();
    return [this, idx]() mutable -> std::optional<std::pair<size_t, size_t>> {
      llvm::dbgs() << "IDX: " << idx << '\n';
      if (idx == 0)
        return std::nullopt;
      auto frameId = idx - 1;
      auto frameLimit = labelLevel[frameId].stackIdx;
      idx -= 1;
      return {{frameId, frameLimit}};
    };
  };
  auto getNextFrameIndex = indexGetter();
  auto nextFrameIdx = getNextFrameIndex();
  size_t stackSize = size();
  for (size_t idx = 0; idx < stackSize; ++idx) {
    size_t actualIdx = stackSize - 1 - idx;
    while (nextFrameIdx && (nextFrameIdx->second > actualIdx)) {
      llvm::dbgs() << "  --------------- Frame (" << nextFrameIdx->first
                   << ")\n";
      nextFrameIdx = getNextFrameIndex();
    }
    llvm::dbgs() << "  ";
    values[actualIdx].dump();
  }
  while (nextFrameIdx) {
    llvm::dbgs() << "  --------------- Frame (" << nextFrameIdx->first << ")\n";
    nextFrameIdx = getNextFrameIndex();
  }
  llvm::dbgs() << "<Bottom>"
               << "\n";
  llvm::dbgs() << "=========================================================\n";
}
#endif

parsed_inst_t ValueStack::popOperands(TypeRange operandTypes, Location *opLoc) {
  LDBG() << "Popping from ValueStack\n"
         << "  Elements(s) to pop: " << operandTypes.size() << "\n"
         << "  Current stack size: " << values.size();
  if (operandTypes.size() > values.size())
    return emitError(*opLoc,
                     "stack doesn't contain enough values. trying to get ")
           << operandTypes.size() << " operands on a stack containing only "
           << values.size() << " values";
  size_t stackIdxOffset = values.size() - operandTypes.size();
  SmallVector<Value> res{};
  res.reserve(operandTypes.size());
  for (size_t i{0}; i < operandTypes.size(); ++i) {
    Value operand = values[i + stackIdxOffset];
    Type stackType = operand.getType();
    if (stackType != operandTypes[i])
      return emitError(*opLoc, "invalid operand type on stack. expecting ")
             << operandTypes[i] << ", value on stack is of type " << stackType;
    LDBG() << "    POP: " << operand;
    res.push_back(operand);
  }
  values.resize(values.size() - operandTypes.size());
  LDBG() << "  Updated stack size: " << values.size();
  return res;
}

LogicalResult ValueStack::pushResults(ValueRange results, Location *opLoc) {
  LDBG() << "Pushing to ValueStack\n"
         << "  Elements(s) to push: " << results.size() << "\n"
         << "  Current stack size: " << values.size();
  for (Value val : results) {
    if (!isWasmValueType(val.getType()))
      return emitError(*opLoc, "invalid value type on stack: ")
             << val.getType();
    LDBG() << "    PUSH: " << val;
    values.push_back(val);
  }

  LDBG() << "  Updated stack size: " << values.size();
  return success();
}

template <std::byte EndParseByte>
parsed_inst_t ExpressionParser::parse(OpBuilder &builder,
                                      UniqueByte<EndParseByte> endByte) {
  auto res = parse(builder, ByteSequence<EndParseByte>{});
  if (failed(res))
    return failure();
  return res->opResults;
}

template <std::byte... ExpressionParseEnd>
FailureOr<ExpressionParser::ParseResultWithInfo>
ExpressionParser::parse(OpBuilder &builder,
                        ByteSequence<ExpressionParseEnd...> parsingEndFilters) {
  SmallVector<Value> res;
  for (;;) {
    currentOpLoc = parser.getLocation();
    FailureOr<std::byte> opCode = parser.consumeByte();
    if (failed(opCode))
      return failure();
    if (isValueOneOf(*opCode, parsingEndFilters))
      return {{res, *opCode}};
    parsed_inst_t resParsed;
    resParsed = dispatchToInstParser(*opCode, builder);
    if (failed(resParsed))
      return failure();
    std::swap(res, *resParsed);
    if (failed(pushResults(res)))
      return failure();
  }
}

llvm::FailureOr<FunctionType>
ExpressionParser::parseBlockFuncType(OpBuilder &builder) {
  return getFuncTypeFor(builder, parser.parseBlockType(builder.getContext()));
}

template <typename OpToCreate>
parsed_inst_t ExpressionParser::parseBlockLikeOp(OpBuilder &builder) {
  auto opLoc = currentOpLoc;
  auto funcType = parseBlockFuncType(builder);
  if (failed(funcType))
    return failure();

  auto inputTypes = funcType->getInputs();
  auto inputOps = popOperands(inputTypes);
  if (failed(inputOps))
    return failure();

  Block *curBlock = builder.getBlock();
  Region *curRegion = curBlock->getParent();
  auto resTypes = funcType->getResults();
  llvm::SmallVector<Location> locations{};
  locations.resize(resTypes.size(), *currentOpLoc);
  auto *successor =
      builder.createBlock(curRegion, curRegion->end(), resTypes, locations);
  builder.setInsertionPointToEnd(curBlock);
  auto blockOp =
      builder.create<OpToCreate>(*currentOpLoc, *inputOps, successor);
  auto *blockBody = blockOp.createBlock();
  if (failed(parseBlockContent(builder, blockBody, resTypes, *opLoc, blockOp)))
    return failure();
  builder.setInsertionPointToStart(successor);
  return {ValueRange{successor->getArguments()}};
}

template <>
inline parsed_inst_t
ExpressionParser::parseSpecificInstruction<WasmBinaryEncoding::OpCode::block>(
    OpBuilder &builder) {
  return parseBlockLikeOp<BlockOp>(builder);
}

template <>
inline parsed_inst_t
ExpressionParser::parseSpecificInstruction<WasmBinaryEncoding::OpCode::loop>(
    OpBuilder &builder) {
  return parseBlockLikeOp<LoopOp>(builder);
}

template <>
inline parsed_inst_t ExpressionParser::parseSpecificInstruction<
    WasmBinaryEncoding::OpCode::ifOpCode>(OpBuilder &builder) {
  auto opLoc = currentOpLoc;
  auto funcType = parseBlockFuncType(builder);
  if (failed(funcType))
    return failure();

  LDBG() << "Parsing an if instruction of type " << *funcType;
  auto inputTypes = funcType->getInputs();
  auto conditionValue = popOperands(builder.getI32Type());
  if (failed(conditionValue))
    return failure();
  auto inputOps = popOperands(inputTypes);
  if (failed(inputOps))
    return failure();

  Block *curBlock = builder.getBlock();
  Region *curRegion = curBlock->getParent();
  auto resTypes = funcType->getResults();
  llvm::SmallVector<Location> locations{};
  locations.resize(resTypes.size(), *currentOpLoc);
  auto *successor =
      builder.createBlock(curRegion, curRegion->end(), resTypes, locations);
  builder.setInsertionPointToEnd(curBlock);
  auto ifOp = builder.create<IfOp>(*currentOpLoc, conditionValue->front(),
                                   *inputOps, successor);
  auto *ifEntryBlock = ifOp.createIfBlock();
  constexpr auto ifElseFilter =
      ByteSequence<WasmBinaryEncoding::endByte,
                   WasmBinaryEncoding::OpCode::elseOpCode>{};
  auto parseIfRes = parseBlockContent(builder, ifEntryBlock, resTypes, *opLoc,
                                      ifOp, ifElseFilter);
  if (failed(parseIfRes))
    return failure();
  if (*parseIfRes == WasmBinaryEncoding::OpCode::elseOpCode) {
    LDBG() << "  else block is present.";
    Block *elseEntryBlock = ifOp.createElseBlock();
    auto parseElseRes =
        parseBlockContent(builder, elseEntryBlock, resTypes, *opLoc, ifOp);
    if (failed(parseElseRes))
      return failure();
  }
  builder.setInsertionPointToStart(successor);
  return {ValueRange{successor->getArguments()}};
}

template <>
inline parsed_inst_t ExpressionParser::parseSpecificInstruction<
    WasmBinaryEncoding::OpCode::branchIf>(OpBuilder &builder) {
  auto level = parser.parseLiteral<uint32_t>();
  if (failed(level))
    return failure();
  Block *curBlock = builder.getBlock();
  Region *curRegion = curBlock->getParent();
  auto sip = builder.saveInsertionPoint();
  Block *elseBlock = builder.createBlock(curRegion, curRegion->end());
  auto condition = popOperands(builder.getI32Type());
  if (failed(condition))
    return failure();
  builder.restoreInsertionPoint(sip);
  auto targetOp =
      LabelBranchingOpInterface::getTargetOpFromBlock(curBlock, *level);
  if (failed(targetOp))
    return failure();
  auto inputTypes = targetOp->getLabelTarget()->getArgumentTypes();
  auto branchArgs = popOperands(inputTypes);
  if (failed(branchArgs))
    return failure();
  builder.create<BranchIfOp>(*currentOpLoc, condition->front(),
                             builder.getUI32IntegerAttr(*level), *branchArgs,
                             elseBlock);
  builder.setInsertionPointToStart(elseBlock);
  return {*branchArgs};
}

template <>
inline parsed_inst_t
ExpressionParser::parseSpecificInstruction<WasmBinaryEncoding::OpCode::call>(
    OpBuilder &builder) {
  auto loc = *currentOpLoc;
  auto funcIdx = parser.parseLiteral<uint32_t>();
  if (failed(funcIdx))
    return failure();
  if (*funcIdx >= symbols.funcSymbols.size())
    return emitError(loc, "Invalid function index: ") << *funcIdx;
  auto callee = symbols.funcSymbols[*funcIdx];
  llvm::ArrayRef<Type> inTypes = callee.functionType.getInputs();
  llvm::ArrayRef<Type> resTypes = callee.functionType.getResults();
  parsed_inst_t inOperands = popOperands(inTypes);
  if (failed(inOperands))
    return failure();
  auto callOp =
      builder.create<FuncCallOp>(loc, resTypes, callee.symbol, *inOperands);
  return {callOp.getResults()};
}

template <>
inline parsed_inst_t ExpressionParser::parseSpecificInstruction<
    WasmBinaryEncoding::OpCode::localGet>(OpBuilder &builder) {
  FailureOr<uint32_t> id = parser.parseLiteral<uint32_t>();
  Location instLoc = *currentOpLoc;
  if (failed(id))
    return failure();
  if (*id >= locals.size())
    return emitError(instLoc, "invalid local index. function has ")
           << locals.size() << " accessible locals, received index " << *id;
  return {{LocalGetOp::create(builder, instLoc, locals[*id]).getResult()}};
}

template <>
inline parsed_inst_t ExpressionParser::parseSpecificInstruction<
    WasmBinaryEncoding::OpCode::globalGet>(OpBuilder &builder) {
  FailureOr<uint32_t> id = parser.parseLiteral<uint32_t>();
  Location instLoc = *currentOpLoc;
  if (failed(id))
    return failure();
  if (*id >= symbols.globalSymbols.size())
    return emitError(instLoc, "invalid global index. function has ")
           << symbols.globalSymbols.size()
           << " accessible globals, received index " << *id;
  GlobalSymbolRefContainer globalVar = symbols.globalSymbols[*id];
  auto globalOp = GlobalGetOp::create(builder, instLoc, globalVar.globalType,
                                      globalVar.symbol);

  return {{globalOp.getResult()}};
}

template <typename OpToCreate>
parsed_inst_t ExpressionParser::parseSetOrTee(OpBuilder &builder) {
  FailureOr<uint32_t> id = parser.parseLiteral<uint32_t>();
  if (failed(id))
    return failure();
  if (*id >= locals.size())
    return emitError(*currentOpLoc, "invalid local index. function has ")
           << locals.size() << " accessible locals, received index " << *id;
  if (valueStack.empty())
    return emitError(
        *currentOpLoc,
        "invalid stack access, trying to access a value on an empty stack");

  parsed_inst_t poppedOp = popOperands(locals[*id].getType().getElementType());
  if (failed(poppedOp))
    return failure();
  return {
      OpToCreate::create(builder, *currentOpLoc, locals[*id], poppedOp->front())
          ->getResults()};
}

template <>
inline parsed_inst_t ExpressionParser::parseSpecificInstruction<
    WasmBinaryEncoding::OpCode::localSet>(OpBuilder &builder) {
  return parseSetOrTee<LocalSetOp>(builder);
}

template <>
inline parsed_inst_t ExpressionParser::parseSpecificInstruction<
    WasmBinaryEncoding::OpCode::localTee>(OpBuilder &builder) {
  return parseSetOrTee<LocalTeeOp>(builder);
}

template <typename T>
inline Type buildLiteralType(OpBuilder &);

template <>
inline Type buildLiteralType<int32_t>(OpBuilder &builder) {
  return builder.getI32Type();
}

template <>
inline Type buildLiteralType<int64_t>(OpBuilder &builder) {
  return builder.getI64Type();
}

template <>
[[maybe_unused]] inline Type buildLiteralType<uint32_t>(OpBuilder &builder) {
  return builder.getI32Type();
}

template <>
[[maybe_unused]] inline Type buildLiteralType<uint64_t>(OpBuilder &builder) {
  return builder.getI64Type();
}

template <>
inline Type buildLiteralType<float>(OpBuilder &builder) {
  return builder.getF32Type();
}

template <>
inline Type buildLiteralType<double>(OpBuilder &builder) {
  return builder.getF64Type();
}

template <typename ValT,
          typename E = std::enable_if_t<std::is_arithmetic_v<ValT>>>
struct AttrHolder;

template <typename ValT>
struct AttrHolder<ValT, std::enable_if_t<std::is_integral_v<ValT>>> {
  using type = IntegerAttr;
};

template <typename ValT>
struct AttrHolder<ValT, std::enable_if_t<std::is_floating_point_v<ValT>>> {
  using type = FloatAttr;
};

template <typename ValT>
using attr_holder_t = typename AttrHolder<ValT>::type;

template <typename ValT,
          typename EnableT = std::enable_if_t<std::is_arithmetic_v<ValT>>>
attr_holder_t<ValT> buildLiteralAttr(OpBuilder &builder, ValT val) {
  return attr_holder_t<ValT>::get(buildLiteralType<ValT>(builder), val);
}

template <typename valueT>
parsed_inst_t ExpressionParser::parseConstInst(
    OpBuilder &builder, std::enable_if_t<std::is_arithmetic_v<valueT>> *) {
  auto parsedConstant = parser.parseLiteral<valueT>();
  if (failed(parsedConstant))
    return failure();
  auto constOp =
      ConstOp::create(builder, *currentOpLoc,
                      buildLiteralAttr<valueT>(builder, *parsedConstant));
  return {{constOp.getResult()}};
}

template <>
inline parsed_inst_t ExpressionParser::parseSpecificInstruction<
    WasmBinaryEncoding::OpCode::constI32>(OpBuilder &builder) {
  return parseConstInst<int32_t>(builder);
}

template <>
inline parsed_inst_t ExpressionParser::parseSpecificInstruction<
    WasmBinaryEncoding::OpCode::constI64>(OpBuilder &builder) {
  return parseConstInst<int64_t>(builder);
}

template <>
inline parsed_inst_t ExpressionParser::parseSpecificInstruction<
    WasmBinaryEncoding::OpCode::constFP32>(OpBuilder &builder) {
  return parseConstInst<float>(builder);
}

template <>
inline parsed_inst_t ExpressionParser::parseSpecificInstruction<
    WasmBinaryEncoding::OpCode::constFP64>(OpBuilder &builder) {
  return parseConstInst<double>(builder);
}

template <typename opcode, typename valueType, unsigned int numOperands>
inline parsed_inst_t ExpressionParser::buildNumericOp(
    OpBuilder &builder, std::enable_if_t<std::is_arithmetic_v<valueType>> *) {
  auto ty = buildLiteralType<valueType>(builder);
  LDBG() << "*** buildNumericOp: numOperands = " << numOperands
         << ", type = " << ty << " ***";
  auto tysToPop = SmallVector<Type, numOperands>();
  tysToPop.resize(numOperands);
  llvm::fill(tysToPop, ty);
  auto operands = popOperands(tysToPop);
  if (failed(operands))
    return failure();
  auto op = opcode::create(builder, *currentOpLoc, *operands).getResult();
  LDBG() << "Built operation: " << op;
  return {{op}};
}

// Convenience macro for generating numerical operations.
#define BUILD_NUMERIC_OP(OP_NAME, N_ARGS, PREFIX, SUFFIX, TYPE)                \
  template <>                                                                  \
  inline parsed_inst_t ExpressionParser::parseSpecificInstruction<             \
      WasmBinaryEncoding::OpCode::PREFIX##SUFFIX>(OpBuilder & builder) {       \
    return buildNumericOp<OP_NAME, TYPE, N_ARGS>(builder);                     \
  }

// Macro to define binops that only support integer types.
#define BUILD_NUMERIC_BINOP_INT(OP_NAME, PREFIX)                               \
  BUILD_NUMERIC_OP(OP_NAME, 2, PREFIX, I32, int32_t)                           \
  BUILD_NUMERIC_OP(OP_NAME, 2, PREFIX, I64, int64_t)

// Macro to define binops that only support floating point types.
#define BUILD_NUMERIC_BINOP_FP(OP_NAME, PREFIX)                                \
  BUILD_NUMERIC_OP(OP_NAME, 2, PREFIX, F32, float)                             \
  BUILD_NUMERIC_OP(OP_NAME, 2, PREFIX, F64, double)

// Macro to define binops that support both floating point and integer types.
#define BUILD_NUMERIC_BINOP_INTFP(OP_NAME, PREFIX)                             \
  BUILD_NUMERIC_BINOP_INT(OP_NAME, PREFIX)                                     \
  BUILD_NUMERIC_BINOP_FP(OP_NAME, PREFIX)

// Macro to implement unary ops that only support integers.
#define BUILD_NUMERIC_UNARY_OP_INT(OP_NAME, PREFIX)                            \
  BUILD_NUMERIC_OP(OP_NAME, 1, PREFIX, I32, int32_t)                           \
  BUILD_NUMERIC_OP(OP_NAME, 1, PREFIX, I64, int64_t)

// Macro to implement unary ops that support integer and floating point types.
#define BUILD_NUMERIC_UNARY_OP_FP(OP_NAME, PREFIX)                             \
  BUILD_NUMERIC_OP(OP_NAME, 1, PREFIX, F32, float)                             \
  BUILD_NUMERIC_OP(OP_NAME, 1, PREFIX, F64, double)

BUILD_NUMERIC_BINOP_FP(CopySignOp, copysign)
BUILD_NUMERIC_BINOP_FP(DivOp, div)
BUILD_NUMERIC_BINOP_FP(GeOp, ge)
BUILD_NUMERIC_BINOP_FP(GtOp, gt)
BUILD_NUMERIC_BINOP_FP(LeOp, le)
BUILD_NUMERIC_BINOP_FP(LtOp, lt)
BUILD_NUMERIC_BINOP_FP(MaxOp, max)
BUILD_NUMERIC_BINOP_FP(MinOp, min)
BUILD_NUMERIC_BINOP_INT(AndOp, and)
BUILD_NUMERIC_BINOP_INT(DivSIOp, divS)
BUILD_NUMERIC_BINOP_INT(DivUIOp, divU)
BUILD_NUMERIC_BINOP_INT(GeSIOp, geS)
BUILD_NUMERIC_BINOP_INT(GeUIOp, geU)
BUILD_NUMERIC_BINOP_INT(GtSIOp, gtS)
BUILD_NUMERIC_BINOP_INT(GtUIOp, gtU)
BUILD_NUMERIC_BINOP_INT(LeSIOp, leS)
BUILD_NUMERIC_BINOP_INT(LeUIOp, leU)
BUILD_NUMERIC_BINOP_INT(LtSIOp, ltS)
BUILD_NUMERIC_BINOP_INT(LtUIOp, ltU)
BUILD_NUMERIC_BINOP_INT(OrOp, or)
BUILD_NUMERIC_BINOP_INT(RemSIOp, remS)
BUILD_NUMERIC_BINOP_INT(RemUIOp, remU)
BUILD_NUMERIC_BINOP_INT(RotlOp, rotl)
BUILD_NUMERIC_BINOP_INT(RotrOp, rotr)
BUILD_NUMERIC_BINOP_INT(ShLOp, shl)
BUILD_NUMERIC_BINOP_INT(ShRSOp, shrS)
BUILD_NUMERIC_BINOP_INT(ShRUOp, shrU)
BUILD_NUMERIC_BINOP_INT(XOrOp, xor)
BUILD_NUMERIC_BINOP_INTFP(AddOp, add)
BUILD_NUMERIC_BINOP_INTFP(EqOp, eq)
BUILD_NUMERIC_BINOP_INTFP(MulOp, mul)
BUILD_NUMERIC_BINOP_INTFP(NeOp, ne)
BUILD_NUMERIC_BINOP_INTFP(SubOp, sub)
BUILD_NUMERIC_UNARY_OP_FP(AbsOp, abs)
BUILD_NUMERIC_UNARY_OP_FP(CeilOp, ceil)
BUILD_NUMERIC_UNARY_OP_FP(FloorOp, floor)
BUILD_NUMERIC_UNARY_OP_FP(NegOp, neg)
BUILD_NUMERIC_UNARY_OP_FP(SqrtOp, sqrt)
BUILD_NUMERIC_UNARY_OP_FP(TruncOp, trunc)
BUILD_NUMERIC_UNARY_OP_INT(ClzOp, clz)
BUILD_NUMERIC_UNARY_OP_INT(CtzOp, ctz)
BUILD_NUMERIC_UNARY_OP_INT(EqzOp, eqz)
BUILD_NUMERIC_UNARY_OP_INT(PopCntOp, popcnt)

// Don't need these anymore so let's undef them.
#undef BUILD_NUMERIC_BINOP_FP
#undef BUILD_NUMERIC_BINOP_INT
#undef BUILD_NUMERIC_BINOP_INTFP
#undef BUILD_NUMERIC_UNARY_OP_FP
#undef BUILD_NUMERIC_UNARY_OP_INT
#undef BUILD_NUMERIC_OP
#undef BUILD_NUMERIC_CAST_OP

template <typename opType, typename inputType, typename outputType,
          typename... extraArgsT>
inline parsed_inst_t ExpressionParser::buildConvertOp(OpBuilder &builder,
                                                      extraArgsT... extraArgs) {
  static_assert(std::is_arithmetic_v<inputType>,
                "InputType should be an arithmetic type");
  static_assert(std::is_arithmetic_v<outputType>,
                "OutputType should be an arithmetic type");
  auto intype = buildLiteralType<inputType>(builder);
  auto outType = buildLiteralType<outputType>(builder);
  auto operand = popOperands(intype);
  if (failed(operand))
    return failure();
  auto op = builder.create<opType>(*currentOpLoc, outType, operand->front(),
                                   extraArgs...);
  LDBG() << "Built operation: " << op;
  return {{op.getResult()}};
}

template <>
inline parsed_inst_t ExpressionParser::parseSpecificInstruction<
    WasmBinaryEncoding::OpCode::demoteF64ToF32>(OpBuilder &builder) {
  return buildConvertOp<DemoteOp, double, float>(builder);
}

template <>
inline parsed_inst_t
ExpressionParser::parseSpecificInstruction<WasmBinaryEncoding::OpCode::wrap>(
    OpBuilder &builder) {
  return buildConvertOp<WrapOp, int64_t, int32_t>(builder);
}

#define BUILD_CONVERSION_OP(IN_T, OUT_T, SOURCE_OP, TARGET_OP)                 \
  template <>                                                                  \
  inline parsed_inst_t ExpressionParser::parseSpecificInstruction<             \
      WasmBinaryEncoding::OpCode::SOURCE_OP>(OpBuilder & builder) {            \
    return buildConvertOp<TARGET_OP, IN_T, OUT_T>(builder);                    \
  }

#define BUILD_CONVERT_OP_FOR(DEST_T, WIDTH)                                    \
  BUILD_CONVERSION_OP(uint32_t, DEST_T, convertUI32F##WIDTH, ConvertUOp)       \
  BUILD_CONVERSION_OP(int32_t, DEST_T, convertSI32F##WIDTH, ConvertSOp)        \
  BUILD_CONVERSION_OP(uint64_t, DEST_T, convertUI64F##WIDTH, ConvertUOp)       \
  BUILD_CONVERSION_OP(int64_t, DEST_T, convertSI64F##WIDTH, ConvertSOp)

BUILD_CONVERT_OP_FOR(float, 32)
BUILD_CONVERT_OP_FOR(double, 64)

#undef BUILD_CONVERT_OP_FOR

BUILD_CONVERSION_OP(int32_t, int64_t, extendS, ExtendSI32Op)
BUILD_CONVERSION_OP(int32_t, int64_t, extendU, ExtendUI32Op)

#undef BUILD_CONVERSION_OP

#define BUILD_SLICE_EXTEND_PARSER(IT_WIDTH, EXTRACT_WIDTH)                     \
  template <>                                                                  \
  parsed_inst_t ExpressionParser::parseSpecificInstruction<                    \
      WasmBinaryEncoding::OpCode::extendI##IT_WIDTH##EXTRACT_WIDTH##S>(        \
      OpBuilder & builder) {                                                   \
    using inout_t = int##IT_WIDTH##_t;                                         \
    auto attr = builder.getUI32IntegerAttr(EXTRACT_WIDTH);                     \
    return buildConvertOp<ExtendLowBitsSOp, inout_t, inout_t>(builder, attr);  \
  }

BUILD_SLICE_EXTEND_PARSER(32, 8)
BUILD_SLICE_EXTEND_PARSER(32, 16)
BUILD_SLICE_EXTEND_PARSER(64, 8)
BUILD_SLICE_EXTEND_PARSER(64, 16)
BUILD_SLICE_EXTEND_PARSER(64, 32)

#undef BUILD_SLICE_EXTEND_PARSER

template <>
inline parsed_inst_t ExpressionParser::parseSpecificInstruction<
    WasmBinaryEncoding::OpCode::promoteF32ToF64>(OpBuilder &builder) {
  return buildConvertOp<PromoteOp, float, double>(builder);
}

#define BUILD_REINTERPRET_PARSER(WIDTH, FP_TYPE)                               \
  template <>                                                                  \
  inline parsed_inst_t ExpressionParser::parseSpecificInstruction<             \
      WasmBinaryEncoding::OpCode::reinterpretF##WIDTH##AsI##WIDTH>(OpBuilder & \
                                                                   builder) {  \
    return buildConvertOp<ReinterpretOp, FP_TYPE, int##WIDTH##_t>(builder);    \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  inline parsed_inst_t ExpressionParser::parseSpecificInstruction<             \
      WasmBinaryEncoding::OpCode::reinterpretI##WIDTH##AsF##WIDTH>(OpBuilder & \
                                                                   builder) {  \
    return buildConvertOp<ReinterpretOp, int##WIDTH##_t, FP_TYPE>(builder);    \
  }

BUILD_REINTERPRET_PARSER(32, float)
BUILD_REINTERPRET_PARSER(64, double)

#undef BUILD_REINTERPRET_PARSER

class WasmBinaryParser {
private:
  struct SectionRegistry {
    using section_location_t = StringRef;

    std::array<SmallVector<section_location_t>, highestWasmSectionID + 1>
        registry;

    template <WasmSectionType SecType>
    std::conditional_t<sectionShouldBeUnique(SecType),
                       std::optional<section_location_t>,
                       ArrayRef<section_location_t>>
    getContentForSection() const {
      constexpr auto idx = static_cast<size_t>(SecType);
      if constexpr (sectionShouldBeUnique(SecType)) {
        return registry[idx].empty() ? std::nullopt
                                     : std::make_optional(registry[idx][0]);
      } else {
        return registry[idx];
      }
    }

    bool hasSection(WasmSectionType secType) const {
      return !registry[static_cast<size_t>(secType)].empty();
    }

    ///
    /// @returns success if registration valid, failure in case registration
    /// can't be done (if another section of same type already exist and this
    /// section type should only be present once)
    ///
    LogicalResult registerSection(WasmSectionType secType,
                                  section_location_t location, Location loc) {
      if (sectionShouldBeUnique(secType) && hasSection(secType))
        return emitError(loc,
                         "trying to add a second instance of unique section");

      registry[static_cast<size_t>(secType)].push_back(location);
      emitRemark(loc, "Adding section with section ID ")
          << static_cast<uint8_t>(secType);
      return success();
    }

    LogicalResult populateFromBody(ParserHead ph) {
      while (!ph.end()) {
        FileLineColLoc sectionLoc = ph.getLocation();
        FailureOr<WasmSectionType> secType = ph.parseWasmSectionType();
        if (failed(secType))
          return failure();

        FailureOr<uint32_t> secSizeParsed = ph.parseLiteral<uint32_t>();
        if (failed(secSizeParsed))
          return failure();

        uint32_t secSize = *secSizeParsed;
        FailureOr<StringRef> sectionContent = ph.consumeNBytes(secSize);
        if (failed(sectionContent))
          return failure();

        LogicalResult registration =
            registerSection(*secType, *sectionContent, sectionLoc);

        if (failed(registration))
          return failure();
      }
      return success();
    }
  };

  auto getLocation(int offset = 0) const {
    return FileLineColLoc::get(srcName, 0, offset);
  }

  template <WasmSectionType>
  LogicalResult parseSectionItem(ParserHead &, size_t);

  template <WasmSectionType section>
  LogicalResult parseSection() {
    auto secName = std::string{wasmSectionName<section>};
    auto sectionNameAttr =
        StringAttr::get(ctx, srcName.strref() + ":" + secName + "-SECTION");
    unsigned offset = 0;
    auto getLocation = [sectionNameAttr, &offset]() {
      return FileLineColLoc::get(sectionNameAttr, 0, offset);
    };
    auto secContent = registry.getContentForSection<section>();
    if (!secContent) {
      LDBG() << secName << " section is not present in file.";
      return success();
    }

    auto secSrc = secContent.value();
    ParserHead ph{secSrc, sectionNameAttr};
    FailureOr<uint32_t> nElemsParsed = ph.parseVectorSize();
    if (failed(nElemsParsed))
      return failure();
    uint32_t nElems = *nElemsParsed;
    LDBG() << "starting to parse " << nElems << " items for section "
           << secName;
    for (size_t i = 0; i < nElems; ++i) {
      if (failed(parseSectionItem<section>(ph, i)))
        return failure();
    }

    if (!ph.end())
      return emitError(getLocation(), "unparsed garbage at end of section ")
             << secName;
    return success();
  }

  /// Handles the registration of a function import
  LogicalResult visitImport(Location loc, StringRef moduleName,
                            StringRef importName, TypeIdxRecord tid) {
    using llvm::Twine;
    if (tid.id >= symbols.moduleFuncTypes.size())
      return emitError(loc, "invalid type id: ")
             << tid.id << ". Only " << symbols.moduleFuncTypes.size()
             << " type registrations";
    FunctionType type = symbols.moduleFuncTypes[tid.id];
    std::string symbol = symbols.getNewFuncSymbolName();
    auto funcOp = FuncImportOp::create(builder, loc, symbol, moduleName,
                                       importName, type);
    symbols.funcSymbols.push_back({{FlatSymbolRefAttr::get(funcOp)}, type});
    return funcOp.verify();
  }

  /// Handles the registration of a memory import
  LogicalResult visitImport(Location loc, StringRef moduleName,
                            StringRef importName, LimitType limitType) {
    std::string symbol = symbols.getNewMemorySymbolName();
    auto memOp = MemImportOp::create(builder, loc, symbol, moduleName,
                                     importName, limitType);
    symbols.memSymbols.push_back({FlatSymbolRefAttr::get(memOp)});
    return memOp.verify();
  }

  /// Handles the registration of a table import
  LogicalResult visitImport(Location loc, StringRef moduleName,
                            StringRef importName, TableType tableType) {
    std::string symbol = symbols.getNewTableSymbolName();
    auto tableOp = TableImportOp::create(builder, loc, symbol, moduleName,
                                         importName, tableType);
    symbols.tableSymbols.push_back({FlatSymbolRefAttr::get(tableOp)});
    return tableOp.verify();
  }

  /// Handles the registration of a global variable import
  LogicalResult visitImport(Location loc, StringRef moduleName,
                            StringRef importName, GlobalTypeRecord globalType) {
    std::string symbol = symbols.getNewGlobalSymbolName();
    auto giOp =
        GlobalImportOp::create(builder, loc, symbol, moduleName, importName,
                               globalType.type, globalType.isMutable);
    symbols.globalSymbols.push_back(
        {{FlatSymbolRefAttr::get(giOp)}, giOp.getType()});
    return giOp.verify();
  }

  // Detect occurence of errors
  LogicalResult peekDiag(Diagnostic &diag) {
    if (diag.getSeverity() == DiagnosticSeverity::Error)
      isValid = false;
    return failure();
  }

public:
  WasmBinaryParser(llvm::SourceMgr &sourceMgr, MLIRContext *ctx)
      : builder{ctx}, ctx{ctx} {
    ctx->getDiagEngine().registerHandler(
        [this](Diagnostic &diag) { return peekDiag(diag); });
    ctx->loadAllAvailableDialects();
    if (sourceMgr.getNumBuffers() != 1) {
      emitError(UnknownLoc::get(ctx), "one source file should be provided");
      return;
    }
    uint32_t sourceBufId = sourceMgr.getMainFileID();
    StringRef source = sourceMgr.getMemoryBuffer(sourceBufId)->getBuffer();
    srcName = StringAttr::get(
        ctx, sourceMgr.getMemoryBuffer(sourceBufId)->getBufferIdentifier());

    auto parser = ParserHead{source, srcName};
    auto const wasmHeader = StringRef{"\0asm", 4};
    FileLineColLoc magicLoc = parser.getLocation();
    FailureOr<StringRef> magic = parser.consumeNBytes(wasmHeader.size());
    if (failed(magic) || magic->compare(wasmHeader)) {
      emitError(magicLoc, "source file does not contain valid Wasm header");
      return;
    }
    auto const expectedVersionString = StringRef{"\1\0\0\0", 4};
    FileLineColLoc versionLoc = parser.getLocation();
    FailureOr<StringRef> version =
        parser.consumeNBytes(expectedVersionString.size());
    if (failed(version))
      return;
    if (version->compare(expectedVersionString)) {
      emitError(versionLoc,
                "unsupported Wasm version. only version 1 is supported");
      return;
    }
    LogicalResult fillRegistry = registry.populateFromBody(parser.copy());
    if (failed(fillRegistry))
      return;

    mOp = ModuleOp::create(builder, getLocation());
    builder.setInsertionPointToStart(&mOp.getBodyRegion().front());
    LogicalResult parsingTypes = parseSection<WasmSectionType::TYPE>();
    if (failed(parsingTypes))
      return;

    LogicalResult parsingImports = parseSection<WasmSectionType::IMPORT>();
    if (failed(parsingImports))
      return;

    firstInternalFuncID = symbols.funcSymbols.size();

    LogicalResult parsingFunctions = parseSection<WasmSectionType::FUNCTION>();
    if (failed(parsingFunctions))
      return;

    LogicalResult parsingTables = parseSection<WasmSectionType::TABLE>();
    if (failed(parsingTables))
      return;

    LogicalResult parsingMems = parseSection<WasmSectionType::MEMORY>();
    if (failed(parsingMems))
      return;

    LogicalResult parsingGlobals = parseSection<WasmSectionType::GLOBAL>();
    if (failed(parsingGlobals))
      return;

    LogicalResult parsingCode = parseSection<WasmSectionType::CODE>();
    if (failed(parsingCode))
      return;

    LogicalResult parsingExports = parseSection<WasmSectionType::EXPORT>();
    if (failed(parsingExports))
      return;

    // Copy over sizes of containers into statistics.
    LDBG() << "WASM Imports:"
           << "\n"
           << " - Num functions: " << symbols.funcSymbols.size() << "\n"
           << " - Num globals: " << symbols.globalSymbols.size() << "\n"
           << " - Num memories: " << symbols.memSymbols.size() << "\n"
           << " - Num tables: " << symbols.tableSymbols.size();
  }

  ModuleOp getModule() {
    if (isValid)
      return mOp;
    if (mOp)
      mOp.erase();
    return ModuleOp{};
  }

private:
  mlir::StringAttr srcName;
  OpBuilder builder;
  WasmModuleSymbolTables symbols;
  MLIRContext *ctx;
  ModuleOp mOp;
  SectionRegistry registry;
  size_t firstInternalFuncID{0};
  bool isValid{true};
};

template <>
LogicalResult
WasmBinaryParser::parseSectionItem<WasmSectionType::IMPORT>(ParserHead &ph,
                                                            size_t) {
  FileLineColLoc importLoc = ph.getLocation();
  auto moduleName = ph.parseName();
  if (failed(moduleName))
    return failure();

  auto importName = ph.parseName();
  if (failed(importName))
    return failure();

  FailureOr<ImportDesc> import = ph.parseImportDesc(ctx);
  if (failed(import))
    return failure();

  return std::visit(
      [this, importLoc, &moduleName, &importName](auto import) {
        return visitImport(importLoc, *moduleName, *importName, import);
      },
      *import);
}

template <>
LogicalResult
WasmBinaryParser::parseSectionItem<WasmSectionType::EXPORT>(ParserHead &ph,
                                                            size_t) {
  FileLineColLoc exportLoc = ph.getLocation();

  auto exportName = ph.parseName();
  if (failed(exportName))
    return failure();

  FailureOr<std::byte> opcode = ph.consumeByte();
  if (failed(opcode))
    return failure();

  FailureOr<uint32_t> idx = ph.parseLiteral<uint32_t>();
  if (failed(idx))
    return failure();

  using SymbolRefDesc = std::variant<SmallVector<SymbolRefContainer>,
                                     SmallVector<GlobalSymbolRefContainer>,
                                     SmallVector<FunctionSymbolRefContainer>>;

  SymbolRefDesc currentSymbolList;
  std::string symbolType = "";
  switch (*opcode) {
  case WasmBinaryEncoding::Export::function:
    symbolType = "function";
    currentSymbolList = symbols.funcSymbols;
    break;
  case WasmBinaryEncoding::Export::table:
    symbolType = "table";
    currentSymbolList = symbols.tableSymbols;
    break;
  case WasmBinaryEncoding::Export::memory:
    symbolType = "memory";
    currentSymbolList = symbols.memSymbols;
    break;
  case WasmBinaryEncoding::Export::global:
    symbolType = "global";
    currentSymbolList = symbols.globalSymbols;
    break;
  default:
    return emitError(exportLoc, "invalid value for export type: ")
           << std::to_integer<unsigned>(*opcode);
  }

  auto currentSymbol = std::visit(
      [&](const auto &list) -> FailureOr<FlatSymbolRefAttr> {
        if (*idx > list.size()) {
          emitError(
              exportLoc,
              llvm::formatv(
                  "trying to export {0} {1} which is undefined in this scope",
                  symbolType, *idx));
          return failure();
        }
        return list[*idx].symbol;
      },
      currentSymbolList);

  if (failed(currentSymbol))
    return failure();

  Operation *op = SymbolTable::lookupSymbolIn(mOp, *currentSymbol);
  op->setAttr("exported", UnitAttr::get(op->getContext()));
  StringAttr symName = SymbolTable::getSymbolName(op);
  return SymbolTable{mOp}.rename(symName, *exportName);
}

template <>
LogicalResult
WasmBinaryParser::parseSectionItem<WasmSectionType::TABLE>(ParserHead &ph,
                                                           size_t) {
  FileLineColLoc opLocation = ph.getLocation();
  FailureOr<TableType> tableType = ph.parseTableType(ctx);
  if (failed(tableType))
    return failure();
  LDBG() << "  Parsed table description: " << *tableType;
  StringAttr symbol = builder.getStringAttr(symbols.getNewTableSymbolName());
  auto tableOp =
      TableOp::create(builder, opLocation, symbol.strref(), *tableType);
  symbols.tableSymbols.push_back({SymbolRefAttr::get(tableOp)});
  return success();
}

template <>
LogicalResult
WasmBinaryParser::parseSectionItem<WasmSectionType::FUNCTION>(ParserHead &ph,
                                                              size_t) {
  FileLineColLoc opLoc = ph.getLocation();
  auto typeIdxParsed = ph.parseLiteral<uint32_t>();
  if (failed(typeIdxParsed))
    return failure();
  uint32_t typeIdx = *typeIdxParsed;
  if (typeIdx >= symbols.moduleFuncTypes.size())
    return emitError(getLocation(), "invalid type index: ") << typeIdx;
  std::string symbol = symbols.getNewFuncSymbolName();
  auto funcOp =
      FuncOp::create(builder, opLoc, symbol, symbols.moduleFuncTypes[typeIdx]);
  Block *block = funcOp.addEntryBlock();
  OpBuilder::InsertionGuard guard{builder};
  builder.setInsertionPointToEnd(block);
  ReturnOp::create(builder, opLoc);
  symbols.funcSymbols.push_back(
      {{FlatSymbolRefAttr::get(funcOp.getSymNameAttr())},
       symbols.moduleFuncTypes[typeIdx]});
  return funcOp.verify();
}

template <>
LogicalResult
WasmBinaryParser::parseSectionItem<WasmSectionType::TYPE>(ParserHead &ph,
                                                          size_t) {
  FailureOr<FunctionType> funcType = ph.parseFunctionType(ctx);
  if (failed(funcType))
    return failure();
  LDBG() << "Parsed function type " << *funcType;
  symbols.moduleFuncTypes.push_back(*funcType);
  return success();
}

template <>
LogicalResult
WasmBinaryParser::parseSectionItem<WasmSectionType::MEMORY>(ParserHead &ph,
                                                            size_t) {
  FileLineColLoc opLocation = ph.getLocation();
  FailureOr<LimitType> memory = ph.parseLimit(ctx);
  if (failed(memory))
    return failure();

  LDBG() << "  Registering memory " << *memory;
  std::string symbol = symbols.getNewMemorySymbolName();
  auto memOp = MemOp::create(builder, opLocation, symbol, *memory);
  symbols.memSymbols.push_back({SymbolRefAttr::get(memOp)});
  return success();
}

template <>
LogicalResult
WasmBinaryParser::parseSectionItem<WasmSectionType::GLOBAL>(ParserHead &ph,
                                                            size_t) {
  FileLineColLoc globalLocation = ph.getLocation();
  auto globalTypeParsed = ph.parseGlobalType(ctx);
  if (failed(globalTypeParsed))
    return failure();

  GlobalTypeRecord globalType = *globalTypeParsed;
  auto symbol = builder.getStringAttr(symbols.getNewGlobalSymbolName());
  auto globalOp = wasmssa::GlobalOp::create(
      builder, globalLocation, symbol, globalType.type, globalType.isMutable);
  symbols.globalSymbols.push_back(
      {{FlatSymbolRefAttr::get(globalOp)}, globalOp.getType()});
  OpBuilder::InsertionGuard guard{builder};
  Block *block = builder.createBlock(&globalOp.getInitializer());
  builder.setInsertionPointToStart(block);
  parsed_inst_t expr = ph.parseExpression(builder, symbols);
  if (failed(expr))
    return failure();
  if (block->empty())
    return emitError(globalLocation, "global with empty initializer");
  if (expr->size() != 1 && (*expr)[0].getType() != globalType.type)
    return emitError(
        globalLocation,
        "initializer result type does not match global declaration type");
  ReturnOp::create(builder, globalLocation, *expr);
  return success();
}

template <>
LogicalResult WasmBinaryParser::parseSectionItem<WasmSectionType::CODE>(
    ParserHead &ph, size_t innerFunctionId) {
  unsigned long funcId = innerFunctionId + firstInternalFuncID;
  FunctionSymbolRefContainer symRef = symbols.funcSymbols[funcId];
  auto funcOp =
      dyn_cast<FuncOp>(SymbolTable::lookupSymbolIn(mOp, symRef.symbol));
  assert(funcOp);
  if (failed(ph.parseCodeFor(funcOp, symbols)))
    return failure();
  return success();
}
} // namespace

namespace mlir::wasm {
OwningOpRef<ModuleOp> importWebAssemblyToModule(llvm::SourceMgr &source,
                                                MLIRContext *context) {
  WasmBinaryParser wBN{source, context};
  ModuleOp mOp = wBN.getModule();
  if (mOp)
    return {mOp};

  return {nullptr};
}
} // namespace mlir::wasm
