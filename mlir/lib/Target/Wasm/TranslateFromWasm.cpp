//===- TranslateFromWasm.cpp - Translating to C++ calls -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/WasmSSA/IR/WasmSSA.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/Target/Wasm/WasmBinaryEncoding.h"
#include "mlir/Target/Wasm/WasmImporter.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LEB128.h"

#include <variant>

#define DEBUG_TYPE "wasm-translate"

// Statistics.
STATISTIC(numFunctionSectionItems, "Parsed functions");
STATISTIC(numGlobalSectionItems, "Parsed globals");
STATISTIC(numMemorySectionItems, "Parsed memories");
STATISTIC(numTableSectionItems, "Parsed tables");

static_assert(CHAR_BIT == 8, "This code expects std::byte to be exactly 8 bits");

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
  constexpr const char *wasmSectionName<WasmSectionType::section> = #section;
APPLY_WASM_SEC_TRANSFORM
#undef WASM_SEC_TRANSFORM

constexpr bool sectionShouldBeUnique(WasmSectionType secType) {
  return secType != WasmSectionType::CUSTOM;
}

template <std::byte... Bytes>
struct ByteSequence{};

template <std::byte... Bytes1, std::byte... Bytes2>
constexpr ByteSequence<Bytes1..., Bytes2...>
operator+(ByteSequence<Bytes1...>, ByteSequence<Bytes2...>) {
  return {};
}

/// Template class for representing a byte sequence of only one byte
template<std::byte Byte>
struct UniqueByte : ByteSequence<Byte> {};

template <typename T, T... Values>
constexpr ByteSequence<std::byte{Values}...>
byteSeqFromIntSeq(std::integer_sequence<T, Values...>) {
  return {};
}

constexpr auto allOpCodes =
    byteSeqFromIntSeq(std::make_integer_sequence<int, 256>());

constexpr ByteSequence<
    WasmBinaryEncoding::Type::i32, WasmBinaryEncoding::Type::i64,
    WasmBinaryEncoding::Type::f32, WasmBinaryEncoding::Type::f64,
    WasmBinaryEncoding::Type::v128>
    valueTypesEncodings{};

template<std::byte... allowedFlags>
constexpr bool isValueOneOf(std::byte value, ByteSequence<allowedFlags...> = {}) {
  return  ((value == allowedFlags) | ... | false);
}

template<std::byte... flags>
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

using ImportDesc = std::variant<TypeIdxRecord, TableType, LimitType, GlobalTypeRecord>;

struct WasmModuleSymbolTables {
  llvm::SmallVector<FunctionSymbolRefContainer> funcSymbols;
  llvm::SmallVector<GlobalSymbolRefContainer> globalSymbols;
  llvm::SmallVector<SymbolRefContainer> memSymbols;
  llvm::SmallVector<SymbolRefContainer> tableSymbols;
  llvm::SmallVector<FunctionType> moduleFuncTypes;

  std::string getNewSymbolName(llvm::StringRef prefix, size_t id) const {
    return (prefix + llvm::Twine{id}).str();
  }

  std::string getNewFuncSymbolName() const {
    auto id = funcSymbols.size();
    return getNewSymbolName("func_", id);
  }

  std::string getNewGlobalSymbolName() const {
    auto id = globalSymbols.size();
    return getNewSymbolName("global_", id);
  }

  std::string getNewMemorySymbolName() const {
    auto id = memSymbols.size();
    return getNewSymbolName("mem_", id);
  }

  std::string getNewTableSymbolName() const {
    auto id = tableSymbols.size();
    return getNewSymbolName("table_", id);
  }
};
class ParserHead {
public:
  ParserHead(llvm::StringRef src, StringAttr name) : head{src}, locName{name} {}
  ParserHead(ParserHead &&) = default;
private:
  ParserHead(ParserHead const &other) = default;

public:
  auto getLocation() const {
    return FileLineColLoc::get(locName, 0, anchorOffset + offset);
  }

  llvm::FailureOr<llvm::StringRef> consumeNBytes(size_t nBytes) {
    LLVM_DEBUG(llvm::dbgs() << "Consume " << nBytes << " bytes\n");
    LLVM_DEBUG(llvm::dbgs() << "  Bytes remaining: " << size() << "\n");
    LLVM_DEBUG(llvm::dbgs() << "  Current offset: " << offset << "\n");
    if (nBytes > size())
      return emitError(getLocation(), "trying to extract ")
             << nBytes << "bytes when only " << size() << "are avilables";

    auto res = head.slice(offset, offset + nBytes);
    offset += nBytes;
    LLVM_DEBUG(llvm::dbgs()
               << "  Updated offset (+" << nBytes << "): " << offset << "\n");
    return res;
  }

  llvm::FailureOr<std::byte> consumeByte() {
    auto res = consumeNBytes(1);
    if (failed(res))
      return failure();
    return std::byte{*res->bytes_begin()};
  }

  template <typename T>
  llvm::FailureOr<T> parseLiteral();

  llvm::FailureOr<uint32_t> parseVectorSize();

private:
  // TODO: This is equivalent to parseLiteral<uint32_t> and could be removed
  // if parseLiteral specialization were moved here, but default GCC on Ubuntu
  // 22.04 has bug with template specialization in class declaration
  inline llvm::FailureOr<uint32_t> parseUI32();
  inline llvm::FailureOr<int64_t> parseI64();

public:
  llvm::FailureOr<llvm::StringRef> parseName() {
    auto size = parseVectorSize();
    if (failed(size))
      return failure();

    return consumeNBytes(*size);
  }

  llvm::FailureOr<WasmSectionType> parseWasmSectionType() {
    auto id = consumeByte();
    if (failed(id))
      return failure();
    if (std::to_integer<unsigned>(*id) > highestWasmSectionID)
      return emitError(getLocation(), "Invalid section ID: ")
             << static_cast<int>(*id);
    return static_cast<WasmSectionType>(*id);
  }

  llvm::FailureOr<LimitType> parseLimit(MLIRContext *ctx) {
    using WasmLimits = WasmBinaryEncoding::LimitHeader;
    auto limitLocation = getLocation();
    auto limitHeader = consumeByte();
    if (failed(limitHeader))
      return failure();

    if (isNotIn<WasmLimits::bothLimits, WasmLimits::lowLimitOnly>(*limitHeader))
      return emitError(limitLocation, "Invalid limit header: ")
             << static_cast<int>(*limitHeader);
    auto minParse = parseUI32();
    if (failed(minParse))
      return failure();
    std::optional<uint32_t> max{std::nullopt};
    if (*limitHeader == WasmLimits::bothLimits) {
      auto maxParse = parseUI32();
      if (failed(maxParse))
        return failure();
      max = *maxParse;
    }
    return LimitType::get(ctx, *minParse, max);
  }

  llvm::FailureOr<Type> parseValueType(MLIRContext *ctx) {
    auto typeLoc = getLocation();
    auto typeEncoding = consumeByte();
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
      return emitError(typeLoc, "Invalid value type encoding: ")
             << static_cast<int>(*typeEncoding);
    }
  }

  llvm::FailureOr<GlobalTypeRecord> parseGlobalType(MLIRContext *ctx) {
    using WasmGlobalMut = WasmBinaryEncoding::GlobalMutability;
    auto typeParsed = parseValueType(ctx);
    if (failed(typeParsed))
      return failure();
    auto mutLoc = getLocation();
    auto mutSpec = consumeByte();
    if (failed(mutSpec))
      return failure();
    if (isNotIn<WasmGlobalMut::isConst, WasmGlobalMut::isMutable>(*mutSpec))
      return emitError(mutLoc, "Invalid global mutability specifier: ")
             << static_cast<int>(*mutSpec);
    return GlobalTypeRecord{*typeParsed, *mutSpec == WasmGlobalMut::isMutable};
  }

  llvm::FailureOr<TupleType> parseResultType(MLIRContext *ctx) {
    auto nParamsParsed = parseVectorSize();
    if (failed(nParamsParsed))
      return failure();
    auto nParams = *nParamsParsed;
    llvm::SmallVector<Type> res{};
    res.reserve(nParams);
    for (size_t i = 0; i < nParams; ++i) {
      auto parsedType = parseValueType(ctx);
      if (failed(parsedType))
        return failure();
      res.push_back(*parsedType);
    }
    return TupleType::get(ctx, res);
  }

  llvm::FailureOr<FunctionType> parseFunctionType(MLIRContext *ctx) {
    auto typeLoc = getLocation();
    auto funcTypeHeader = consumeByte();
    if (failed(funcTypeHeader))
      return failure();
    if (*funcTypeHeader != WasmBinaryEncoding::Type::funcType)
      return emitError(typeLoc, "Invalid function type header byte. Expecting ")
             << std::to_integer<unsigned>(
                    WasmBinaryEncoding::Type::funcType)
             << " got " << std::to_integer<unsigned>(*funcTypeHeader);
    auto inputTypes = parseResultType(ctx);
    if (failed(inputTypes))
      return failure();

    auto resTypes = parseResultType(ctx);
    if (failed(resTypes))
      return failure();

    return FunctionType::get(ctx, inputTypes->getTypes(), resTypes->getTypes());
  }

  llvm::FailureOr<TypeIdxRecord> parseTypeIndex() {
    auto res = parseUI32();
    if (failed(res))
      return failure();
    return TypeIdxRecord{*res};
  }

  llvm::FailureOr<TableType> parseTableType(MLIRContext *ctx) {
    auto elmTypeParse = parseValueType(ctx);
    if (failed(elmTypeParse))
      return failure();
    if (!isWasmRefType(*elmTypeParse))
      return emitError(getLocation(), "Invalid element type for table");
    auto limitParse = parseLimit(ctx);
    if (failed(limitParse))
      return failure();
    return TableType::get(ctx, *elmTypeParse, *limitParse);
  }

  llvm::FailureOr<ImportDesc> parseImportDesc(MLIRContext *ctx) {
    auto importLoc = getLocation();
    auto importType = consumeByte();
    auto packager = [](auto parseResult) -> llvm::FailureOr<ImportDesc> {
      if (llvm::failed(parseResult))
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
      return emitError(importLoc, "Invalid import type descriptor: ")
             << static_cast<int>(*importType);
    }
  }
  bool end() const { return curHead().empty(); }

  ParserHead copy() const {
    return *this;
  }

private:
  llvm::StringRef curHead() const { return head.drop_front(offset); }

  llvm::FailureOr<std::byte> peek() const {
    if (end())
      return emitError(
          getLocation(),
          "trying to peek at next byte, but input stream is empty");
    return static_cast<std::byte>(curHead().front());
  }

  size_t size() const { return head.size() - offset; }

  llvm::StringRef head;
  StringAttr locName;
  unsigned anchorOffset{0};
  unsigned offset{0};
};

template <>
llvm::FailureOr<float> ParserHead::parseLiteral<float>() {
  auto bytes = consumeNBytes(4);
  if (failed(bytes))
    return failure();
  float result;
  std::memcpy(&result, bytes->bytes_begin(), 4);
  return result;
}

template <>
llvm::FailureOr<double> ParserHead::parseLiteral<double>() {
  auto bytes = consumeNBytes(8);
  if (failed(bytes))
    return failure();
  double result;
  std::memcpy(&result, bytes->bytes_begin(), 8);
  return result;
}

template <>
llvm::FailureOr<uint32_t> ParserHead::parseLiteral<uint32_t>() {
  char const *error = nullptr;
  uint32_t res{0};
  unsigned encodingSize{0};
  auto src = curHead();
  auto decoded = llvm::decodeULEB128(src.bytes_begin(), &encodingSize,
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
llvm::FailureOr<int32_t> ParserHead::parseLiteral<int32_t>() {
  char const *error = nullptr;
  int32_t res{0};
  unsigned encodingSize{0};
  auto src = curHead();
  auto decoded = llvm::decodeSLEB128(src.bytes_begin(), &encodingSize,
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
llvm::FailureOr<int64_t> ParserHead::parseLiteral<int64_t>() {
  char const *error = nullptr;
  unsigned encodingSize{0};
  auto src = curHead();
  auto res = llvm::decodeSLEB128(src.bytes_begin(), &encodingSize,
                                 src.bytes_end(), &error);
  if (error)
    return emitError(getLocation(), error);

  offset += encodingSize;
  return res;
}

llvm::FailureOr<uint32_t> ParserHead::parseVectorSize() {
  return parseLiteral<uint32_t>();
}

inline llvm::FailureOr<uint32_t> ParserHead::parseUI32() {
  return parseLiteral<uint32_t>();
}

inline llvm::FailureOr<int64_t> ParserHead::parseI64() {
  return parseLiteral<int64_t>();
}

class WasmBinaryParser {
private:
  struct SectionRegistry {
    using section_location_t = llvm::StringRef;

    std::array<llvm::SmallVector<section_location_t>, highestWasmSectionID+1> registry;

    template <WasmSectionType SecType>
    std::conditional_t<sectionShouldBeUnique(SecType),
                       std::optional<section_location_t>,
                       llvm::ArrayRef<section_location_t>>
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
                         "Trying to add a second instance of unique section");

      registry[static_cast<size_t>(secType)].push_back(location);
      emitRemark(loc, "Adding section with section ID ")
          << static_cast<uint8_t>(secType);
      return success();
    }

    LogicalResult populateFromBody(ParserHead ph) {
      while (!ph.end()) {
        auto sectionLoc = ph.getLocation();
        auto secType = ph.parseWasmSectionType();
        if (failed(secType))
          return failure();

        auto secSizeParsed = ph.parseLiteral<uint32_t>();
        if (failed(secSizeParsed))
          return failure();

        auto secSize = *secSizeParsed;
        auto sectionContent = ph.consumeNBytes(secSize);
        if (failed(sectionContent))
          return failure();

        auto registration =
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
      LLVM_DEBUG(llvm::dbgs() << secName << " section is not present in file.");
      return success();
    }

    auto secSrc = secContent.value();
    ParserHead ph{secSrc, sectionNameAttr};
    auto nElemsParsed = ph.parseVectorSize();
    if (failed(nElemsParsed))
      return failure();
    auto nElems = *nElemsParsed;
    LLVM_DEBUG(llvm::dbgs() << "Starting to parse " << nElems
                            << " items for section " << secName << ".\n");
    for (size_t i = 0; i < nElems; ++i) {
      if (failed(parseSectionItem<section>(ph, i)))
        return failure();
    }

    if (!ph.end())
      return emitError(getLocation(), "Unparsed garbage at end of section ")
             << secName;
    return success();
  }

  /// Handles the registration of a function import
  LogicalResult visitImport(Location loc, llvm::StringRef moduleName,
                            llvm::StringRef importName, TypeIdxRecord tid) {
    using llvm::Twine;
    if (tid.id >= symbols.moduleFuncTypes.size())
      return emitError(loc, "Invalid type id: ")
             << tid.id << ". Only " << symbols.moduleFuncTypes.size()
             << " type registration.";
    auto type = symbols.moduleFuncTypes[tid.id];
    auto symbol = symbols.getNewFuncSymbolName();
    auto funcOp = builder.create<FuncImportOp>(
        loc, symbol, moduleName, importName, type);
    symbols.funcSymbols.push_back({{FlatSymbolRefAttr::get(funcOp)}, type});
    return funcOp.verify();
  }

  /// Handles the registration of a memory import
  LogicalResult visitImport(Location loc, llvm::StringRef moduleName,
                            llvm::StringRef importName, LimitType limitType) {
    auto symbol = symbols.getNewMemorySymbolName();
    auto memOp = builder.create<MemImportOp>(loc, symbol, moduleName,
                                             importName, limitType);
    symbols.memSymbols.push_back({FlatSymbolRefAttr::get(memOp)});
    return memOp.verify();
  }

  /// Handles the registration of a table import
  LogicalResult visitImport(Location loc, llvm::StringRef moduleName,
                            llvm::StringRef importName, TableType tableType) {
    auto symbol = symbols.getNewTableSymbolName();
    auto tableOp = builder.create<TableImportOp>(loc, symbol, moduleName,
                                                 importName, tableType);
    symbols.tableSymbols.push_back({FlatSymbolRefAttr::get(tableOp)});
    return tableOp.verify();
  }

  /// Handles the registration of a global variable import
  LogicalResult visitImport(Location loc, llvm::StringRef moduleName,
                            llvm::StringRef importName,
                            GlobalTypeRecord globalType) {
    auto symbol = symbols.getNewGlobalSymbolName();
    auto giOp =
        builder.create<GlobalImportOp>(loc, symbol, moduleName, importName,
                                       globalType.type, globalType.isMutable);
    symbols.globalSymbols.push_back({{FlatSymbolRefAttr::get(giOp)}, giOp.getType()});
    return giOp.verify();
  }

public:
  WasmBinaryParser(llvm::SourceMgr &sourceMgr, MLIRContext *ctx)
      : builder{ctx}, ctx{ctx} {
    ctx->loadAllAvailableDialects();
    if (sourceMgr.getNumBuffers() != 1) {
      emitError(UnknownLoc::get(ctx), "One source file should be provided");
      return;
    }
    auto sourceBufId = sourceMgr.getMainFileID();
    auto source = sourceMgr.getMemoryBuffer(sourceBufId)->getBuffer();
    srcName = StringAttr::get(
      ctx, sourceMgr.getMemoryBuffer(sourceBufId)->getBufferIdentifier());

    auto parser = ParserHead{source, srcName};
    auto const wasmHeader = StringRef{"\0asm", 4};
    auto magicLoc = parser.getLocation();
    auto magic = parser.consumeNBytes(wasmHeader.size());
    if (failed(magic) || magic->compare(wasmHeader)) {
      emitError(magicLoc,
                "Source file does not contain valid Wasm header.");
      return;
    }
    auto const expectedVersionString = StringRef{"\1\0\0\0", 4};
    auto versionLoc = parser.getLocation();
    auto version = parser.consumeNBytes(expectedVersionString.size());
    if (failed(version))
      return;
    if (version->compare(expectedVersionString)) {
      emitError(versionLoc,
                "Unsupported Wasm version. Only version 1 is supported.");
      return;
    }
    auto fillRegistry = registry.populateFromBody(parser.copy());
    if (failed(fillRegistry))
      return;

    mOp = builder.create<ModuleOp>(getLocation());
    builder.setInsertionPointToStart(
        &mOp.getBodyRegion().front());
    auto parsingTypes = parseSection<WasmSectionType::TYPE>();
    if (failed(parsingTypes))
      return;

    auto parsingImports = parseSection<WasmSectionType::IMPORT>();
    if (failed(parsingImports))
      return;

    firstInternalFuncID = symbols.funcSymbols.size();

    auto parsingFunctions = parseSection<WasmSectionType::FUNCTION>();
    if (failed(parsingFunctions))
      return;


    // Copy over sizes of containers into statistics.
    numFunctionSectionItems = symbols.funcSymbols.size();
    numGlobalSectionItems = symbols.globalSymbols.size();
    numMemorySectionItems = symbols.memSymbols.size();
    numTableSectionItems = symbols.tableSymbols.size();
  }

  ModuleOp getModule() { return mOp; }

private:
  mlir::StringAttr srcName;
  OpBuilder builder;
  WasmModuleSymbolTables symbols;
  MLIRContext *ctx;
  ModuleOp mOp;
  SectionRegistry registry;
  size_t firstInternalFuncID{0};
};

template <>
LogicalResult
WasmBinaryParser::parseSectionItem<WasmSectionType::IMPORT>(ParserHead &ph, size_t) {
  auto importLoc = ph.getLocation();
  auto moduleName = ph.parseName();
  if (failed(moduleName))
    return failure();

  auto importName = ph.parseName();
  if (failed(importName))
    return failure();

  auto import = ph.parseImportDesc(ctx);
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
WasmBinaryParser::parseSectionItem<WasmSectionType::FUNCTION>(ParserHead &ph,
                                                              size_t) {
  auto opLoc = ph.getLocation();
  auto typeIdxParsed = ph.parseLiteral<uint32_t>();
  if (failed(typeIdxParsed))
    return failure();
  auto typeIdx = *typeIdxParsed;
  if (typeIdx >= symbols.moduleFuncTypes.size())
    return emitError(getLocation(), "Invalid type index: ") << typeIdx;
  auto symbol = symbols.getNewFuncSymbolName();
  auto funcOp =
      builder.create<FuncOp>(opLoc, symbol, symbols.moduleFuncTypes[typeIdx]);
  auto *block = funcOp.addEntryBlock();
  auto ip = builder.saveInsertionPoint();
  builder.setInsertionPointToEnd(block);
  builder.create<ReturnOp>(opLoc);
  builder.restoreInsertionPoint(ip);
  symbols.funcSymbols.push_back(
      {{FlatSymbolRefAttr::get(funcOp.getSymNameAttr())},
       symbols.moduleFuncTypes[typeIdx]});
  return funcOp.verify();
}

template <>
LogicalResult
WasmBinaryParser::parseSectionItem<WasmSectionType::TYPE>(ParserHead &ph,
                                                          size_t) {
  auto funcType = ph.parseFunctionType(ctx);
  if (failed(funcType))
    return failure();
  LLVM_DEBUG(llvm::dbgs() << "Parsed function type " << *funcType << '\n');
  symbols.moduleFuncTypes.push_back(*funcType);
  return success();
}
} // namespace

namespace mlir {
namespace wasm {
OwningOpRef<ModuleOp> importWebAssemblyToModule(llvm::SourceMgr &source,
                                                MLIRContext *context) {
  WasmBinaryParser wBN{source, context};
  auto mOp = wBN.getModule();
  if (mOp)
    return {mOp};

  return {nullptr};
}
} // namespace wasm
} // namespace mlir
