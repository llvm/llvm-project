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

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  /// A simple dump function for debugging.
  /// Writes output to llvm::dbgs().
  LLVM_DUMP_METHOD void dump() const;
#endif

private:
  SmallVector<Value> values;
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

  struct ParseResultWithInfo {
    SmallVector<Value> opResults;
    std::byte endingByte;
  };

public:
  template <std::byte ParseEndByte = WasmBinaryEncoding::endByte>
  parsed_inst_t parse(OpBuilder &builder, UniqueByte<ParseEndByte> = {});

  template <std::byte... ExpressionParseEnd>
  FailureOr<ParseResultWithInfo>
  parse(OpBuilder &builder,
        ByteSequence<ExpressionParseEnd...> parsingEndFilters);

  FailureOr<SmallVector<Value>> popOperands(TypeRange operandTypes) {
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
        auto local = builder.create<LocalOp>(varLoc, *varT);
        locals.push_back(local.getResult());
      }
    }
    parsed_inst_t res = cParser.parseExpression(builder, symbols, locals);
    if (failed(res))
      return failure();
    if (!cParser.end())
      return emitError(cParser.getLocation(),
                       "unparsed garbage remaining at end of code block");
    builder.create<ReturnOp>(func->getLoc(), *res);
    returnOp->erase();
    return success();
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
  llvm::dbgs() << "<Top>"
               << "\n";
  // Stack is pushed to via push_back. Therefore the top of the stack is the
  // end of the vector. Iterate in reverse so that the first thing we print
  // is the top of the stack.
  size_t stackSize = size();
  for (size_t idx = 0; idx < stackSize; idx++) {
    size_t actualIdx = stackSize - 1 - idx;
    llvm::dbgs() << "  ";
    values[actualIdx].dump();
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
           << values.size() << " values.";
  size_t stackIdxOffset = values.size() - operandTypes.size();
  SmallVector<Value> res{};
  res.reserve(operandTypes.size());
  for (size_t i{0}; i < operandTypes.size(); ++i) {
    Value operand = values[i + stackIdxOffset];
    Type stackType = operand.getType();
    if (stackType != operandTypes[i])
      return emitError(*opLoc, "invalid operand type on stack. expecting ")
             << operandTypes[i] << ", value on stack is of type " << stackType
             << ".";
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
  return {{builder.create<LocalGetOp>(instLoc, locals[*id]).getResult()}};
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
  auto globalOp = builder.create<GlobalGetOp>(instLoc, globalVar.globalType,
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
        "invalid stack access, trying to access a value on an empty stack.");

  parsed_inst_t poppedOp = popOperands(locals[*id].getType().getElementType());
  if (failed(poppedOp))
    return failure();
  return {
      builder.create<OpToCreate>(*currentOpLoc, locals[*id], poppedOp->front())
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
  std::fill(tysToPop.begin(), tysToPop.end(), ty);
  auto operands = popOperands(tysToPop);
  if (failed(operands))
    return failure();
  auto op = builder.create<opcode>(*currentOpLoc, *operands).getResult();
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
BUILD_NUMERIC_BINOP_FP(MaxOp, max)
BUILD_NUMERIC_BINOP_FP(MinOp, min)
BUILD_NUMERIC_BINOP_INT(AndOp, and)
BUILD_NUMERIC_BINOP_INT(DivSIOp, divS)
BUILD_NUMERIC_BINOP_INT(DivUIOp, divU)
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
BUILD_NUMERIC_BINOP_INTFP(MulOp, mul)
BUILD_NUMERIC_BINOP_INTFP(SubOp, sub)
BUILD_NUMERIC_UNARY_OP_FP(AbsOp, abs)
BUILD_NUMERIC_UNARY_OP_FP(CeilOp, ceil)
BUILD_NUMERIC_UNARY_OP_FP(FloorOp, floor)
BUILD_NUMERIC_UNARY_OP_FP(NegOp, neg)
BUILD_NUMERIC_UNARY_OP_FP(SqrtOp, sqrt)
BUILD_NUMERIC_UNARY_OP_FP(TruncOp, trunc)
BUILD_NUMERIC_UNARY_OP_INT(ClzOp, clz)
BUILD_NUMERIC_UNARY_OP_INT(CtzOp, ctz)
BUILD_NUMERIC_UNARY_OP_INT(PopCntOp, popcnt)

// Don't need these anymore so let's undef them.
#undef BUILD_NUMERIC_BINOP_FP
#undef BUILD_NUMERIC_BINOP_INT
#undef BUILD_NUMERIC_BINOP_INTFP
#undef BUILD_NUMERIC_UNARY_OP_FP
#undef BUILD_NUMERIC_UNARY_OP_INT
#undef BUILD_NUMERIC_OP
#undef BUILD_NUMERIC_CAST_OP

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
             << " type registration.";
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
      emitError(magicLoc, "source file does not contain valid Wasm header.");
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
  SymbolTable::setSymbolVisibility(op, SymbolTable::Visibility::Public);
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
  auto globalOp = builder.create<wasmssa::GlobalOp>(
      globalLocation, symbol, globalType.type, globalType.isMutable);
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
  builder.create<ReturnOp>(globalLocation, *expr);
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
