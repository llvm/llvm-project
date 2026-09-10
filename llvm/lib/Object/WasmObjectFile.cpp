//===- WasmObjectFile.cpp - Wasm object file implementation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/BinaryFormat/Wasm.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/Error.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Object/Wasm.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "llvm/TargetParser/Triple.h"
#include <cassert>
#include <cstdint>
#include <cstring>

#define DEBUG_TYPE "wasm-object"

using namespace llvm;
using namespace object;

void WasmSymbol::print(raw_ostream &Out) const {
  Out << "Name=" << Info.Name
      << ", Kind=" << toString(wasm::WasmSymbolType(Info.Kind)) << ", Flags=0x"
      << Twine::utohexstr(Info.Flags) << " [";
  switch (getBinding()) {
    case wasm::WASM_SYMBOL_BINDING_GLOBAL: Out << "global"; break;
    case wasm::WASM_SYMBOL_BINDING_LOCAL: Out << "local"; break;
    case wasm::WASM_SYMBOL_BINDING_WEAK: Out << "weak"; break;
  }
  if (isHidden())
    Out << ", hidden";
  else
    Out << ", default";
  if (Info.Flags & wasm::WASM_SYMBOL_NO_STRIP)
    Out << ", no_strip";
  if (Info.Flags & wasm::WASM_SYMBOL_TLS)
    Out << ", tls";
  if (Info.Flags & wasm::WASM_SYMBOL_ABSOLUTE)
    Out << ", absolute";
  if (Info.Flags & wasm::WASM_SYMBOL_EXPORTED)
    Out << ", exported";
  if (isUndefined())
    Out << ", undefined";
  Out << "]";
  if (!isTypeData()) {
    Out << ", ElemIndex=" << Info.ElementIndex;
  } else if (isDefined()) {
    Out << ", Segment=" << Info.DataRef.Segment;
    Out << ", Offset=" << Info.DataRef.Offset;
    Out << ", Size=" << Info.DataRef.Size;
  }
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void WasmSymbol::dump() const { print(dbgs()); }
#endif

Expected<std::unique_ptr<WasmObjectFile>>
ObjectFile::createWasmObjectFile(MemoryBufferRef Buffer) {
  Error Err = Error::success();
  auto ObjectFile = std::make_unique<WasmObjectFile>(Buffer, Err);
  if (Err)
    return std::move(Err);

  return std::move(ObjectFile);
}

#define VARINT7_MAX ((1 << 7) - 1)
#define VARINT7_MIN (-(1 << 7))
#define VARUINT7_MAX (1 << 7)
#define VARUINT1_MAX (1)

#define RETURN_IF_ERROR(Expr)                                                  \
  do {                                                                         \
    if (Error E = (Expr))                                                      \
      return E;                                                                \
  } while (false)

#define UNWRAP_OR_RETURN(Name, Expr)                                           \
  auto Name##OrErr = (Expr);                                                   \
  if (!Name##OrErr)                                                            \
    return Name##OrErr.takeError();                                            \
  auto Name = *Name##OrErr

static Error parseFailed(const Twine &Msg) {
  return make_error<GenericBinaryError>(Msg, object_error::parse_failed);
}

static Expected<uint8_t> readUint8(WasmObjectFile::ReadContext &Ctx) {
  if (Ctx.Ptr == Ctx.End)
    return parseFailed("EOF while reading uint8");
  return *Ctx.Ptr++;
}

static Expected<uint32_t> readUint32(WasmObjectFile::ReadContext &Ctx) {
  if (Ctx.Ptr + 4 > Ctx.End)
    return parseFailed("EOF while reading uint32");
  uint32_t Result = support::endian::read32le(Ctx.Ptr);
  Ctx.Ptr += 4;
  return Result;
}

static Expected<int32_t> readFloat32(WasmObjectFile::ReadContext &Ctx) {
  if (Ctx.Ptr + 4 > Ctx.End)
    return parseFailed("EOF while reading float32");
  int32_t Result = 0;
  memcpy(&Result, Ctx.Ptr, sizeof(Result));
  Ctx.Ptr += sizeof(Result);
  return Result;
}

static Expected<int64_t> readFloat64(WasmObjectFile::ReadContext &Ctx) {
  if (Ctx.Ptr + 8 > Ctx.End)
    return parseFailed("EOF while reading float64");
  int64_t Result = 0;
  memcpy(&Result, Ctx.Ptr, sizeof(Result));
  Ctx.Ptr += sizeof(Result);
  return Result;
}

static Expected<uint64_t> readULEB128(WasmObjectFile::ReadContext &Ctx) {
  unsigned Count;
  const char *ErrorStr = nullptr;
  uint64_t Result = decodeULEB128(Ctx.Ptr, &Count, Ctx.End, &ErrorStr);
  if (ErrorStr)
    return parseFailed(ErrorStr);
  Ctx.Ptr += Count;
  return Result;
}

static Expected<int64_t> readLEB128(WasmObjectFile::ReadContext &Ctx) {
  unsigned Count;
  const char *ErrorStr = nullptr;
  uint64_t Decoded = decodeSLEB128(Ctx.Ptr, &Count, Ctx.End, &ErrorStr);
  if (ErrorStr)
    return parseFailed(ErrorStr);
  Ctx.Ptr += Count;
  return static_cast<int64_t>(Decoded);
}

static Expected<uint8_t> readVaruint1(WasmObjectFile::ReadContext &Ctx) {
  UNWRAP_OR_RETURN(Value, readLEB128(Ctx));
  if (Value > VARUINT1_MAX || Value < 0)
    return parseFailed("LEB is outside Varuint1 range");
  return Value;
}

static Expected<int32_t> readVarint32(WasmObjectFile::ReadContext &Ctx) {
  UNWRAP_OR_RETURN(Value, readLEB128(Ctx));
  if (Value > INT32_MAX || Value < INT32_MIN)
    return parseFailed("LEB is outside Varint32 range");
  return Value;
}

static Expected<uint32_t> readVaruint32(WasmObjectFile::ReadContext &Ctx) {
  UNWRAP_OR_RETURN(Value, readULEB128(Ctx));
  if (Value > UINT32_MAX)
    return parseFailed("LEB is outside Varuint32 range");
  return Value;
}

static Expected<int64_t> readVarint64(WasmObjectFile::ReadContext &Ctx) {
  return readLEB128(Ctx);
}

static Expected<uint64_t> readVaruint64(WasmObjectFile::ReadContext &Ctx) {
  return readULEB128(Ctx);
}

static Expected<StringRef> readString(WasmObjectFile::ReadContext &Ctx) {
  UNWRAP_OR_RETURN(StringLen, readVaruint32(Ctx));
  if (StringLen > static_cast<uint64_t>(Ctx.End - Ctx.Ptr))
    return parseFailed("EOF while reading string");

  StringRef Result(reinterpret_cast<const char *>(Ctx.Ptr), StringLen);
  Ctx.Ptr += StringLen;
  return Result;
}

static Expected<uint8_t> readOpcode(WasmObjectFile::ReadContext &Ctx) {
  return readUint8(Ctx);
}

static Expected<wasm::ValType> parseValType(WasmObjectFile::ReadContext &Ctx,
                                            uint32_t Code) {
  // only directly encoded FUNCREF/EXTERNREF/EXNREF are supported
  // (not ref null func, ref null extern, or ref null exn)
  switch (Code) {
  case wasm::WASM_TYPE_I32:
  case wasm::WASM_TYPE_I64:
  case wasm::WASM_TYPE_F32:
  case wasm::WASM_TYPE_F64:
  case wasm::WASM_TYPE_V128:
  case wasm::WASM_TYPE_FUNCREF:
  case wasm::WASM_TYPE_EXTERNREF:
  case wasm::WASM_TYPE_EXNREF:
    return wasm::ValType(Code);
  }
  if (Code == wasm::WASM_TYPE_NULLABLE || Code == wasm::WASM_TYPE_NONNULLABLE) {
    RETURN_IF_ERROR(readVarint64(Ctx).takeError());
  }
  return wasm::ValType(wasm::ValType::OTHERREF);
}

static Error readInitExpr(wasm::WasmInitExpr &Expr,
                          WasmObjectFile::ReadContext &Ctx) {
  auto Start = Ctx.Ptr;

  Expr.Extended = false;
  RETURN_IF_ERROR(readOpcode(Ctx).moveInto(Expr.Inst.Opcode));
  switch (Expr.Inst.Opcode) {
  case wasm::WASM_OPCODE_I32_CONST:
    RETURN_IF_ERROR(readVarint32(Ctx).moveInto(Expr.Inst.Value.Int32));
    break;
  case wasm::WASM_OPCODE_I64_CONST:
    RETURN_IF_ERROR(readVarint64(Ctx).moveInto(Expr.Inst.Value.Int64));
    break;
  case wasm::WASM_OPCODE_F32_CONST: {
    int32_t Float32;
    RETURN_IF_ERROR(readFloat32(Ctx).moveInto(Float32));
    Expr.Inst.Value.Float32 = Float32;
    break;
  }
  case wasm::WASM_OPCODE_F64_CONST: {
    int64_t Float64;
    RETURN_IF_ERROR(readFloat64(Ctx).moveInto(Float64));
    Expr.Inst.Value.Float64 = Float64;
    break;
  }
  case wasm::WASM_OPCODE_GLOBAL_GET: {
    uint64_t Global;
    RETURN_IF_ERROR(readULEB128(Ctx).moveInto(Global));
    Expr.Inst.Value.Global = Global;
    break;
  }
  case wasm::WASM_OPCODE_REF_NULL: {
    uint32_t TypeCode;
    RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(TypeCode));
    RETURN_IF_ERROR(parseValType(Ctx, TypeCode).takeError());
    break;
  }
  default:
    Expr.Extended = true;
  }

  if (!Expr.Extended) {
    uint8_t EndOpcode;
    RETURN_IF_ERROR(readOpcode(Ctx).moveInto(EndOpcode));
    if (EndOpcode != wasm::WASM_OPCODE_END)
      Expr.Extended = true;
  }

  if (Expr.Extended) {
    Ctx.Ptr = Start;
    while (true) {
      uint8_t Opcode;
      RETURN_IF_ERROR(readOpcode(Ctx).moveInto(Opcode));
      switch (Opcode) {
      case wasm::WASM_OPCODE_I32_CONST:
      case wasm::WASM_OPCODE_GLOBAL_GET:
      case wasm::WASM_OPCODE_REF_NULL:
      case wasm::WASM_OPCODE_REF_FUNC:
      case wasm::WASM_OPCODE_I64_CONST: {
        RETURN_IF_ERROR(readULEB128(Ctx).takeError());
        break;
      }
      case wasm::WASM_OPCODE_F32_CONST: {
        RETURN_IF_ERROR(readFloat32(Ctx).takeError());
        break;
      }
      case wasm::WASM_OPCODE_F64_CONST: {
        RETURN_IF_ERROR(readFloat64(Ctx).takeError());
        break;
      }
      case wasm::WASM_OPCODE_I32_ADD:
      case wasm::WASM_OPCODE_I32_SUB:
      case wasm::WASM_OPCODE_I32_MUL:
      case wasm::WASM_OPCODE_I64_ADD:
      case wasm::WASM_OPCODE_I64_SUB:
      case wasm::WASM_OPCODE_I64_MUL:
        break;
      case wasm::WASM_OPCODE_GC_PREFIX:
        break;
      // The GC opcodes are in a separate (prefixed space). This flat switch
      // structure works as long as there is no overlap between the GC and
      // general opcodes used in init exprs.
      case wasm::WASM_OPCODE_STRUCT_NEW:
      case wasm::WASM_OPCODE_STRUCT_NEW_DEFAULT:
      case wasm::WASM_OPCODE_ARRAY_NEW:
      case wasm::WASM_OPCODE_ARRAY_NEW_DEFAULT: {
        RETURN_IF_ERROR(readULEB128(Ctx).takeError());
        break;
      }
      case wasm::WASM_OPCODE_ARRAY_NEW_FIXED: {
        RETURN_IF_ERROR(readULEB128(Ctx).takeError());
        RETURN_IF_ERROR(readULEB128(Ctx).takeError());
        break;
      }
      case wasm::WASM_OPCODE_REF_I31:
        break;
      case wasm::WASM_OPCODE_END:
        Expr.Body = ArrayRef<uint8_t>(Start, Ctx.Ptr - Start);
        return Error::success();
      default:
        return make_error<GenericBinaryError>("invalid opcode in init_expr: " +
                                                  Twine(unsigned(Opcode)),
                                              object_error::parse_failed);
      }
    }
  }

  return Error::success();
}

static Expected<wasm::WasmLimits> readLimits(WasmObjectFile::ReadContext &Ctx) {
  wasm::WasmLimits Result;
  UNWRAP_OR_RETURN(Flags, readVaruint32(Ctx));
  Result.Flags = Flags;

  UNWRAP_OR_RETURN(Minimum, readVaruint64(Ctx));
  Result.Minimum = Minimum;

  if (Result.Flags & wasm::WASM_LIMITS_FLAG_HAS_MAX) {
    UNWRAP_OR_RETURN(Maximum, readVaruint64(Ctx));
    Result.Maximum = Maximum;
  }
  if (Result.Flags & wasm::WASM_LIMITS_FLAG_HAS_PAGE_SIZE) {
    UNWRAP_OR_RETURN(PageSizeLog2, readVaruint32(Ctx));
    if (PageSizeLog2 >= 32)
      return parseFailed("log2(wasm page size) too large");
    Result.PageSize = 1U << PageSizeLog2;
  }
  return Result;
}

static Expected<wasm::WasmTableType>
readTableType(WasmObjectFile::ReadContext &Ctx) {
  UNWRAP_OR_RETURN(ElemTypeCode, readVaruint32(Ctx));
  UNWRAP_OR_RETURN(ElemType, parseValType(Ctx, ElemTypeCode));
  UNWRAP_OR_RETURN(Limits, readLimits(Ctx));

  wasm::WasmTableType TableType;
  TableType.ElemType = ElemType;
  TableType.Limits = Limits;
  return TableType;
}

static Error readSection(WasmSection &Section, WasmObjectFile::ReadContext &Ctx,
                         WasmSectionOrderChecker &Checker) {
  uint8_t SectionType;
  RETURN_IF_ERROR(readUint8(Ctx).moveInto(SectionType));
  Section.Type = SectionType;
  LLVM_DEBUG(dbgs() << "readSection type=" << Section.Type << "\n");
  // When reading the section's size, store the size of the LEB used to encode
  // it. This allows objcopy/strip to reproduce the binary identically.
  const uint8_t *PreSizePtr = Ctx.Ptr;
  uint32_t Size;
  RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Size));
  Section.HeaderSecSizeEncodingLen = Ctx.Ptr - PreSizePtr;
  Section.Offset = Ctx.Ptr - Ctx.Start;
  if (Size == 0)
    return make_error<StringError>("zero length section",
                                   object_error::parse_failed);
  if (Ctx.Ptr + Size > Ctx.End)
    return make_error<StringError>("section too large",
                                   object_error::parse_failed);
  if (Section.Type == wasm::WASM_SEC_CUSTOM) {
    WasmObjectFile::ReadContext SectionCtx;
    SectionCtx.Start = Ctx.Ptr;
    SectionCtx.Ptr = Ctx.Ptr;
    SectionCtx.End = Ctx.Ptr + Size;

    RETURN_IF_ERROR(readString(SectionCtx).moveInto(Section.Name));

    uint32_t SectionNameSize = SectionCtx.Ptr - SectionCtx.Start;
    Ctx.Ptr += SectionNameSize;
    Size -= SectionNameSize;
  }

  if (!Checker.isValidSectionOrder(Section.Type, Section.Name)) {
    return make_error<StringError>("out of order section type: " +
                                       llvm::to_string(Section.Type),
                                   object_error::parse_failed);
  }

  Section.Content = ArrayRef<uint8_t>(Ctx.Ptr, Size);
  Ctx.Ptr += Size;
  return Error::success();
}

WasmObjectFile::WasmObjectFile(MemoryBufferRef Buffer, Error &Err)
    : ObjectFile(Binary::ID_Wasm, Buffer) {
  ErrorAsOutParameter ErrAsOutParam(Err);
  Header.Magic = getData().substr(0, 4);
  if (Header.Magic != StringRef("\0asm", 4)) {
    Err = make_error<StringError>("invalid magic number",
                                  object_error::parse_failed);
    return;
  }

  ReadContext Ctx;
  Ctx.Start = getData().bytes_begin();
  Ctx.Ptr = Ctx.Start + 4;
  Ctx.End = Ctx.Start + getData().size();

  if (Ctx.Ptr + 4 > Ctx.End) {
    Err = make_error<StringError>("missing version number",
                                  object_error::parse_failed);
    return;
  }

  Expected<uint32_t> Version = readUint32(Ctx);
  if (!Version) {
    Err = Version.takeError();
    return;
  }
  Header.Version = *Version;
  if (Header.Version != wasm::WasmVersion) {
    Err = make_error<StringError>("invalid version number: " +
                                      Twine(Header.Version),
                                  object_error::parse_failed);
    return;
  }

  WasmSectionOrderChecker Checker;
  while (Ctx.Ptr < Ctx.End) {
    WasmSection Sec;
    if ((Err = readSection(Sec, Ctx, Checker)))
      return;
    if ((Err = parseSection(Sec)))
      return;

    Sections.push_back(Sec);
  }
}

Error WasmObjectFile::parseSection(WasmSection &Sec) {
  ReadContext Ctx;
  Ctx.Start = Sec.Content.data();
  Ctx.End = Ctx.Start + Sec.Content.size();
  Ctx.Ptr = Ctx.Start;
  switch (Sec.Type) {
  case wasm::WASM_SEC_CUSTOM:
    return parseCustomSection(Sec, Ctx);
  case wasm::WASM_SEC_TYPE:
    return parseTypeSection(Ctx);
  case wasm::WASM_SEC_IMPORT:
    return parseImportSection(Ctx);
  case wasm::WASM_SEC_FUNCTION:
    return parseFunctionSection(Ctx);
  case wasm::WASM_SEC_TABLE:
    return parseTableSection(Ctx);
  case wasm::WASM_SEC_MEMORY:
    return parseMemorySection(Ctx);
  case wasm::WASM_SEC_TAG:
    return parseTagSection(Ctx);
  case wasm::WASM_SEC_GLOBAL:
    return parseGlobalSection(Ctx);
  case wasm::WASM_SEC_EXPORT:
    return parseExportSection(Ctx);
  case wasm::WASM_SEC_START:
    return parseStartSection(Ctx);
  case wasm::WASM_SEC_ELEM:
    return parseElemSection(Ctx);
  case wasm::WASM_SEC_CODE:
    return parseCodeSection(Ctx);
  case wasm::WASM_SEC_DATA:
    return parseDataSection(Ctx);
  case wasm::WASM_SEC_DATACOUNT:
    return parseDataCountSection(Ctx);
  default:
    return make_error<GenericBinaryError>(
        "invalid section type: " + Twine(Sec.Type), object_error::parse_failed);
  }
}

Error WasmObjectFile::parseDylinkSection(ReadContext &Ctx) {
  // Legacy "dylink" section support.
  // See parseDylink0Section for the current "dylink.0" section parsing.
  HasDylinkSection = true;
  RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(DylinkInfo.MemorySize));
  RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(DylinkInfo.MemoryAlignment));
  RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(DylinkInfo.TableSize));
  RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(DylinkInfo.TableAlignment));
  uint32_t Count;
  RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Count));
  while (Count--) {
    StringRef Name;
    RETURN_IF_ERROR(readString(Ctx).moveInto(Name));
    DylinkInfo.Needed.push_back(Name);
  }

  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("dylink section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseDylink0Section(ReadContext &Ctx) {
  // See
  // https://github.com/WebAssembly/tool-conventions/blob/main/DynamicLinking.md
  HasDylinkSection = true;

  const uint8_t *OrigEnd = Ctx.End;
  while (Ctx.Ptr < OrigEnd) {
    Ctx.End = OrigEnd;
    uint8_t Type;
    RETURN_IF_ERROR(readUint8(Ctx).moveInto(Type));
    uint32_t Size;
    RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Size));
    LLVM_DEBUG(dbgs() << "readSubsection type=" << int(Type) << " size=" << Size
                      << "\n");
    Ctx.End = Ctx.Ptr + Size;
    uint32_t Count;
    switch (Type) {
    case wasm::WASM_DYLINK_MEM_INFO:
      RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(DylinkInfo.MemorySize));
      RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(DylinkInfo.MemoryAlignment));
      RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(DylinkInfo.TableSize));
      RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(DylinkInfo.TableAlignment));
      break;
    case wasm::WASM_DYLINK_NEEDED:
      RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Count));
      while (Count--) {
        StringRef Name;
        RETURN_IF_ERROR(readString(Ctx).moveInto(Name));
        DylinkInfo.Needed.push_back(Name);
      }
      break;
    case wasm::WASM_DYLINK_EXPORT_INFO: {
      uint32_t Count;
      RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Count));
      while (Count--) {
        StringRef ExportName;
        RETURN_IF_ERROR(readString(Ctx).moveInto(ExportName));
        uint32_t ExportFlags;
        RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(ExportFlags));
        DylinkInfo.ExportInfo.push_back({ExportName, ExportFlags});
      }
      break;
    }
    case wasm::WASM_DYLINK_IMPORT_INFO: {
      uint32_t Count;
      RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Count));
      while (Count--) {
        StringRef ImportModule;
        RETURN_IF_ERROR(readString(Ctx).moveInto(ImportModule));
        StringRef ImportField;
        RETURN_IF_ERROR(readString(Ctx).moveInto(ImportField));
        uint32_t ImportFlags;
        RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(ImportFlags));
        DylinkInfo.ImportInfo.push_back(
            {ImportModule, ImportField, ImportFlags});
      }
      break;
    }
    case wasm::WASM_DYLINK_RUNTIME_PATH: {
      RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Count));
      while (Count--) {
        StringRef Name;
        RETURN_IF_ERROR(readString(Ctx).moveInto(Name));
        DylinkInfo.RuntimePath.push_back(Name);
      }
      break;
    }
    default:
      LLVM_DEBUG(dbgs() << "unknown dylink.0 sub-section: " << Type << "\n");
      Ctx.Ptr += Size;
      break;
    }
    if (Ctx.Ptr != Ctx.End) {
      return make_error<GenericBinaryError>(
          "dylink.0 sub-section ended prematurely", object_error::parse_failed);
    }
  }

  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("dylink.0 section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseNameSection(ReadContext &Ctx) {
  llvm::DenseSet<uint64_t> SeenFunctions;
  llvm::DenseSet<uint64_t> SeenGlobals;
  llvm::DenseSet<uint64_t> SeenSegments;

  // If we have linking section (symbol table) or if we are parsing a DSO
  // then we don't use the name section for symbol information.
  bool PopulateSymbolTable = !HasLinkingSection && !HasDylinkSection;

  // If we are using the name section for symbol information then it will
  // supersede any symbols created by the export section.
  if (PopulateSymbolTable)
    Symbols.clear();

  while (Ctx.Ptr < Ctx.End) {
    uint8_t Type;
    RETURN_IF_ERROR(readUint8(Ctx).moveInto(Type));
    uint32_t Size;
    RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Size));
    const uint8_t *SubSectionEnd = Ctx.Ptr + Size;

    switch (Type) {
    case wasm::WASM_NAMES_FUNCTION:
    case wasm::WASM_NAMES_GLOBAL:
    case wasm::WASM_NAMES_DATA_SEGMENT: {
      uint32_t Count;
      RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Count));
      while (Count--) {
        uint32_t Index;
        RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Index));
        StringRef Name;
        RETURN_IF_ERROR(readString(Ctx).moveInto(Name));
        wasm::NameType nameType = wasm::NameType::FUNCTION;
        wasm::WasmSymbolInfo Info{Name,
                                  /*Kind */ wasm::WASM_SYMBOL_TYPE_FUNCTION,
                                  /* Flags */ 0,
                                  /* ImportModule */ std::nullopt,
                                  /* ImportName */ std::nullopt,
                                  /* ExportName */ std::nullopt,
                                  {/* ElementIndex */ Index}};
        const wasm::WasmSignature *Signature = nullptr;
        const wasm::WasmGlobalType *GlobalType = nullptr;
        const wasm::WasmTableType *TableType = nullptr;
        if (Type == wasm::WASM_NAMES_FUNCTION) {
          if (!SeenFunctions.insert(Index).second)
            return make_error<GenericBinaryError>(
                "function named more than once", object_error::parse_failed);
          if (!isValidFunctionIndex(Index) || Name.empty())
            return make_error<GenericBinaryError>("invalid function name entry",
                                                  object_error::parse_failed);

          if (isDefinedFunctionIndex(Index)) {
            wasm::WasmFunction &F = getDefinedFunction(Index);
            F.DebugName = Name;
            Signature = &Signatures[F.SigIndex];
            if (F.ExportName) {
              Info.ExportName = F.ExportName;
              Info.Flags |= wasm::WASM_SYMBOL_BINDING_GLOBAL;
            } else {
              Info.Flags |= wasm::WASM_SYMBOL_BINDING_LOCAL;
            }
          } else {
            Info.Flags |= wasm::WASM_SYMBOL_UNDEFINED;
          }
        } else if (Type == wasm::WASM_NAMES_GLOBAL) {
          if (!SeenGlobals.insert(Index).second)
            return make_error<GenericBinaryError>("global named more than once",
                                                  object_error::parse_failed);
          if (!isValidGlobalIndex(Index) || Name.empty())
            return make_error<GenericBinaryError>("invalid global name entry",
                                                  object_error::parse_failed);
          nameType = wasm::NameType::GLOBAL;
          Info.Kind = wasm::WASM_SYMBOL_TYPE_GLOBAL;
          if (isDefinedGlobalIndex(Index)) {
            GlobalType = &getDefinedGlobal(Index).Type;
          } else {
            Info.Flags |= wasm::WASM_SYMBOL_UNDEFINED;
          }
        } else {
          if (!SeenSegments.insert(Index).second)
            return make_error<GenericBinaryError>(
                "segment named more than once", object_error::parse_failed);
          if (Index >= DataSegments.size())
            return make_error<GenericBinaryError>("invalid data segment name entry",
                                                  object_error::parse_failed);
          nameType = wasm::NameType::DATA_SEGMENT;
          Info.Kind = wasm::WASM_SYMBOL_TYPE_DATA;
          Info.Flags |= wasm::WASM_SYMBOL_BINDING_LOCAL;
          assert(Index < DataSegments.size());
          Info.DataRef = wasm::WasmDataReference{
              Index, 0, DataSegments[Index].Data.Content.size()};
        }
        DebugNames.push_back(wasm::WasmDebugName{nameType, Index, Name});
        if (PopulateSymbolTable)
          Symbols.emplace_back(Info, GlobalType, TableType, Signature);
      }
      break;
    }
    // Ignore local names for now
    case wasm::WASM_NAMES_LOCAL:
    default:
      Ctx.Ptr += Size;
      break;
    }
    if (Ctx.Ptr != SubSectionEnd)
      return make_error<GenericBinaryError>(
          "name sub-section ended prematurely", object_error::parse_failed);
  }

  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("name section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseLinkingSection(ReadContext &Ctx) {
  HasLinkingSection = true;

  RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(LinkingData.Version));
  if (LinkingData.Version != wasm::WasmMetadataVersion) {
    return make_error<GenericBinaryError>(
        "unexpected metadata version: " + Twine(LinkingData.Version) +
            " (Expected: " + Twine(wasm::WasmMetadataVersion) + ")",
        object_error::parse_failed);
  }

  const uint8_t *OrigEnd = Ctx.End;
  while (Ctx.Ptr < OrigEnd) {
    Ctx.End = OrigEnd;
    uint8_t Type;
    RETURN_IF_ERROR(readUint8(Ctx).moveInto(Type));
    uint32_t Size;
    RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Size));
    LLVM_DEBUG(dbgs() << "readSubsection type=" << int(Type) << " size=" << Size
                      << "\n");
    Ctx.End = Ctx.Ptr + Size;
    switch (Type) {
    case wasm::WASM_SYMBOL_TABLE:
      if (Error Err = parseLinkingSectionSymtab(Ctx))
        return Err;
      break;
    case wasm::WASM_SEGMENT_INFO: {
      uint32_t Count;
      RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Count));
      if (Count > DataSegments.size())
        return make_error<GenericBinaryError>("too many segment names",
                                              object_error::parse_failed);
      for (uint32_t I = 0; I < Count; I++) {
        RETURN_IF_ERROR(readString(Ctx).moveInto(DataSegments[I].Data.Name));
        RETURN_IF_ERROR(
            readVaruint32(Ctx).moveInto(DataSegments[I].Data.Alignment));
        if (DataSegments[I].Data.Alignment > 32)
          return make_error<GenericBinaryError>(
              "invalid data segment alignment: `" + DataSegments[I].Data.Name +
                  "` (alignment: " + Twine(DataSegments[I].Data.Alignment) +
                  ")",
              object_error::parse_failed);
        RETURN_IF_ERROR(
            readVaruint32(Ctx).moveInto(DataSegments[I].Data.LinkingFlags));
      }
      break;
    }
    case wasm::WASM_INIT_FUNCS: {
      uint32_t Count;
      RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Count));
      LinkingData.InitFunctions.reserve(Count);
      for (uint32_t I = 0; I < Count; I++) {
        wasm::WasmInitFunc Init;
        RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Init.Priority));
        RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Init.Symbol));
        if (!isValidFunctionSymbol(Init.Symbol))
          return make_error<GenericBinaryError>("invalid function symbol: " +
                                                    Twine(Init.Symbol),
                                                object_error::parse_failed);
        LinkingData.InitFunctions.emplace_back(Init);
      }
      break;
    }
    case wasm::WASM_COMDAT_INFO:
      if (Error Err = parseLinkingSectionComdat(Ctx))
        return Err;
      break;
    default:
      Ctx.Ptr += Size;
      break;
    }
    if (Ctx.Ptr != Ctx.End)
      return make_error<GenericBinaryError>(
          "linking sub-section ended prematurely", object_error::parse_failed);
  }
  if (Ctx.Ptr != OrigEnd)
    return make_error<GenericBinaryError>("linking section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseLinkingSectionSymtab(ReadContext &Ctx) {
  uint32_t Count;
  RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Count));
  // Clear out any symbol information that was derived from the exports
  // section.
  Symbols.clear();
  Symbols.reserve(Count);
  StringSet<> SymbolNames;

  std::vector<wasm::WasmImport *> ImportedGlobals;
  std::vector<wasm::WasmImport *> ImportedFunctions;
  std::vector<wasm::WasmImport *> ImportedTags;
  std::vector<wasm::WasmImport *> ImportedTables;
  ImportedGlobals.reserve(Imports.size());
  ImportedFunctions.reserve(Imports.size());
  ImportedTags.reserve(Imports.size());
  ImportedTables.reserve(Imports.size());
  for (auto &I : Imports) {
    if (I.Kind == wasm::WASM_EXTERNAL_FUNCTION)
      ImportedFunctions.emplace_back(&I);
    else if (I.Kind == wasm::WASM_EXTERNAL_GLOBAL)
      ImportedGlobals.emplace_back(&I);
    else if (I.Kind == wasm::WASM_EXTERNAL_TAG)
      ImportedTags.emplace_back(&I);
    else if (I.Kind == wasm::WASM_EXTERNAL_TABLE)
      ImportedTables.emplace_back(&I);
  }

  while (Count--) {
    wasm::WasmSymbolInfo Info;
    const wasm::WasmSignature *Signature = nullptr;
    const wasm::WasmGlobalType *GlobalType = nullptr;
    const wasm::WasmTableType *TableType = nullptr;

    RETURN_IF_ERROR(readUint8(Ctx).moveInto(Info.Kind));
    RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Info.Flags));
    bool IsDefined = (Info.Flags & wasm::WASM_SYMBOL_UNDEFINED) == 0;

    switch (Info.Kind) {
    case wasm::WASM_SYMBOL_TYPE_FUNCTION:
      RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Info.ElementIndex));
      if (!isValidFunctionIndex(Info.ElementIndex) ||
          IsDefined != isDefinedFunctionIndex(Info.ElementIndex))
        return make_error<GenericBinaryError>("invalid function symbol index",
                                              object_error::parse_failed);
      if (IsDefined) {
        RETURN_IF_ERROR(readString(Ctx).moveInto(Info.Name));
        unsigned FuncIndex = Info.ElementIndex - NumImportedFunctions;
        wasm::WasmFunction &Function = Functions[FuncIndex];
        Signature = &Signatures[Function.SigIndex];
        if (Function.SymbolName.empty())
          Function.SymbolName = Info.Name;
      } else {
        wasm::WasmImport &Import = *ImportedFunctions[Info.ElementIndex];
        if ((Info.Flags & wasm::WASM_SYMBOL_EXPLICIT_NAME) != 0) {
          RETURN_IF_ERROR(readString(Ctx).moveInto(Info.Name));
          Info.ImportName = Import.Field;
        } else {
          Info.Name = Import.Field;
        }
        Signature = &Signatures[Import.SigIndex];
        Info.ImportModule = Import.Module;
      }
      break;

    case wasm::WASM_SYMBOL_TYPE_GLOBAL:
      RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Info.ElementIndex));
      if (!isValidGlobalIndex(Info.ElementIndex) ||
          IsDefined != isDefinedGlobalIndex(Info.ElementIndex))
        return make_error<GenericBinaryError>("invalid global symbol index",
                                              object_error::parse_failed);
      if (!IsDefined && (Info.Flags & wasm::WASM_SYMBOL_BINDING_MASK) ==
                            wasm::WASM_SYMBOL_BINDING_WEAK)
        return make_error<GenericBinaryError>("undefined weak global symbol",
                                              object_error::parse_failed);
      if (IsDefined) {
        RETURN_IF_ERROR(readString(Ctx).moveInto(Info.Name));
        unsigned GlobalIndex = Info.ElementIndex - NumImportedGlobals;
        wasm::WasmGlobal &Global = Globals[GlobalIndex];
        GlobalType = &Global.Type;
        if (Global.SymbolName.empty())
          Global.SymbolName = Info.Name;
      } else {
        wasm::WasmImport &Import = *ImportedGlobals[Info.ElementIndex];
        if ((Info.Flags & wasm::WASM_SYMBOL_EXPLICIT_NAME) != 0) {
          RETURN_IF_ERROR(readString(Ctx).moveInto(Info.Name));
          Info.ImportName = Import.Field;
        } else {
          Info.Name = Import.Field;
        }
        GlobalType = &Import.Global;
        Info.ImportModule = Import.Module;
      }
      break;

    case wasm::WASM_SYMBOL_TYPE_TABLE:
      RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Info.ElementIndex));
      if (!isValidTableNumber(Info.ElementIndex) ||
          IsDefined != isDefinedTableNumber(Info.ElementIndex))
        return make_error<GenericBinaryError>("invalid table symbol index",
                                              object_error::parse_failed);
      if (!IsDefined && (Info.Flags & wasm::WASM_SYMBOL_BINDING_MASK) ==
                            wasm::WASM_SYMBOL_BINDING_WEAK)
        return make_error<GenericBinaryError>("undefined weak table symbol",
                                              object_error::parse_failed);
      if (IsDefined) {
        RETURN_IF_ERROR(readString(Ctx).moveInto(Info.Name));
        unsigned TableNumber = Info.ElementIndex - NumImportedTables;
        wasm::WasmTable &Table = Tables[TableNumber];
        TableType = &Table.Type;
        if (Table.SymbolName.empty())
          Table.SymbolName = Info.Name;
      } else {
        wasm::WasmImport &Import = *ImportedTables[Info.ElementIndex];
        if ((Info.Flags & wasm::WASM_SYMBOL_EXPLICIT_NAME) != 0) {
          RETURN_IF_ERROR(readString(Ctx).moveInto(Info.Name));
          Info.ImportName = Import.Field;
        } else {
          Info.Name = Import.Field;
        }
        TableType = &Import.Table;
        Info.ImportModule = Import.Module;
      }
      break;

    case wasm::WASM_SYMBOL_TYPE_DATA:
      RETURN_IF_ERROR(readString(Ctx).moveInto(Info.Name));
      if (IsDefined) {
        if ((Info.Flags & wasm::WASM_SYMBOL_BINDING_MASK) ==
            wasm::WASM_SYMBOL_BINDING_COMMON) {
          if (Info.Flags & wasm::WASM_SYMBOL_ABSOLUTE)
            return make_error<GenericBinaryError>(
                "common symbols cannot be absolute: " + Info.Name,
                object_error::parse_failed);
          uint64_t Size;
          RETURN_IF_ERROR(readVaruint64(Ctx).moveInto(Size));
          uint8_t Alignment;
          RETURN_IF_ERROR(readUint8(Ctx).moveInto(Alignment));
          if (Alignment > 32)
            return make_error<GenericBinaryError>(
                "invalid common symbol alignment: `" + Info.Name +
                    "` (alignment: " + Twine(unsigned(Alignment)) + ")",
                object_error::parse_failed);
          Info.CommonRef = wasm::WasmCommonReference{Size, Alignment};
        } else {
          uint32_t Index;
          RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Index));
          uint64_t Offset;
          RETURN_IF_ERROR(readVaruint64(Ctx).moveInto(Offset));
          uint64_t Size;
          RETURN_IF_ERROR(readVaruint64(Ctx).moveInto(Size));
          if (!(Info.Flags & wasm::WASM_SYMBOL_ABSOLUTE)) {
            if (Index >= DataSegments.size())
              return make_error<GenericBinaryError>(
                  "invalid data segment index: " + Twine(Index),
                  object_error::parse_failed);
            size_t SegmentSize = DataSegments[Index].Data.Content.size();
            if (Offset > SegmentSize)
              return make_error<GenericBinaryError>(
                  "invalid data symbol offset: `" + Info.Name +
                      "` (offset: " + Twine(Offset) +
                      " segment size: " + Twine(SegmentSize) + ")",
                  object_error::parse_failed);
          }
          Info.DataRef = wasm::WasmDataReference{Index, Offset, Size};
        }
      }
      break;

    case wasm::WASM_SYMBOL_TYPE_SECTION: {
      if ((Info.Flags & wasm::WASM_SYMBOL_BINDING_MASK) !=
          wasm::WASM_SYMBOL_BINDING_LOCAL)
        return make_error<GenericBinaryError>(
            "section symbols must have local binding",
            object_error::parse_failed);
      RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Info.ElementIndex));
      // Use somewhat unique section name as symbol name.
      StringRef SectionName = Sections[Info.ElementIndex].Name;
      Info.Name = SectionName;
      break;
    }

    case wasm::WASM_SYMBOL_TYPE_TAG: {
      RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Info.ElementIndex));
      if (!isValidTagIndex(Info.ElementIndex) ||
          IsDefined != isDefinedTagIndex(Info.ElementIndex))
        return make_error<GenericBinaryError>("invalid tag symbol index",
                                              object_error::parse_failed);
      if (!IsDefined && (Info.Flags & wasm::WASM_SYMBOL_BINDING_MASK) ==
                            wasm::WASM_SYMBOL_BINDING_WEAK)
        return make_error<GenericBinaryError>("undefined weak global symbol",
                                              object_error::parse_failed);
      if (IsDefined) {
        RETURN_IF_ERROR(readString(Ctx).moveInto(Info.Name));
        unsigned TagIndex = Info.ElementIndex - NumImportedTags;
        wasm::WasmTag &Tag = Tags[TagIndex];
        Signature = &Signatures[Tag.SigIndex];
        if (Tag.SymbolName.empty())
          Tag.SymbolName = Info.Name;

      } else {
        wasm::WasmImport &Import = *ImportedTags[Info.ElementIndex];
        if ((Info.Flags & wasm::WASM_SYMBOL_EXPLICIT_NAME) != 0) {
          RETURN_IF_ERROR(readString(Ctx).moveInto(Info.Name));
          Info.ImportName = Import.Field;
        } else {
          Info.Name = Import.Field;
        }
        Signature = &Signatures[Import.SigIndex];
        Info.ImportModule = Import.Module;
      }
      break;
    }

    default:
      return make_error<GenericBinaryError>("invalid symbol type: " +
                                                Twine(unsigned(Info.Kind)),
                                            object_error::parse_failed);
    }

    if ((Info.Flags & wasm::WASM_SYMBOL_BINDING_MASK) !=
            wasm::WASM_SYMBOL_BINDING_LOCAL &&
        !SymbolNames.insert(Info.Name).second)
      return make_error<GenericBinaryError>("duplicate symbol name " +
                                                Twine(Info.Name),
                                            object_error::parse_failed);
    Symbols.emplace_back(Info, GlobalType, TableType, Signature);
    LLVM_DEBUG(dbgs() << "Adding symbol: " << Symbols.back() << "\n");
  }

  return Error::success();
}

Error WasmObjectFile::parseLinkingSectionComdat(ReadContext &Ctx) {
  uint32_t ComdatCount;
  RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(ComdatCount));
  StringSet<> ComdatSet;
  for (unsigned ComdatIndex = 0; ComdatIndex < ComdatCount; ++ComdatIndex) {
    StringRef Name;
    RETURN_IF_ERROR(readString(Ctx).moveInto(Name));
    if (Name.empty() || !ComdatSet.insert(Name).second)
      return make_error<GenericBinaryError>("bad/duplicate COMDAT name " +
                                                Twine(Name),
                                            object_error::parse_failed);
    LinkingData.Comdats.emplace_back(Name);
    uint32_t Flags;
    RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Flags));
    if (Flags != 0)
      return make_error<GenericBinaryError>("unsupported COMDAT flags",
                                            object_error::parse_failed);

    uint32_t EntryCount;
    RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(EntryCount));
    while (EntryCount--) {
      unsigned Kind;
      RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Kind));
      unsigned Index;
      RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Index));
      switch (Kind) {
      default:
        return make_error<GenericBinaryError>("invalid COMDAT entry type",
                                              object_error::parse_failed);
      case wasm::WASM_COMDAT_DATA:
        if (Index >= DataSegments.size())
          return make_error<GenericBinaryError>(
              "COMDAT data index out of range", object_error::parse_failed);
        if (DataSegments[Index].Data.Comdat != UINT32_MAX)
          return make_error<GenericBinaryError>("data segment in two COMDATs",
                                                object_error::parse_failed);
        DataSegments[Index].Data.Comdat = ComdatIndex;
        break;
      case wasm::WASM_COMDAT_FUNCTION:
        if (!isDefinedFunctionIndex(Index))
          return make_error<GenericBinaryError>(
              "COMDAT function index out of range", object_error::parse_failed);
        if (getDefinedFunction(Index).Comdat != UINT32_MAX)
          return make_error<GenericBinaryError>("function in two COMDATs",
                                                object_error::parse_failed);
        getDefinedFunction(Index).Comdat = ComdatIndex;
        break;
      case wasm::WASM_COMDAT_SECTION:
        if (Index >= Sections.size())
          return make_error<GenericBinaryError>(
              "COMDAT section index out of range", object_error::parse_failed);
        if (Sections[Index].Type != wasm::WASM_SEC_CUSTOM)
          return make_error<GenericBinaryError>(
              "non-custom section in a COMDAT", object_error::parse_failed);
        Sections[Index].Comdat = ComdatIndex;
        break;
      }
    }
  }
  return Error::success();
}

Error WasmObjectFile::parseProducersSection(ReadContext &Ctx) {
  llvm::SmallSet<StringRef, 3> FieldsSeen;
  uint32_t Fields;
  RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Fields));
  for (size_t I = 0; I < Fields; ++I) {
    StringRef FieldName;
    RETURN_IF_ERROR(readString(Ctx).moveInto(FieldName));
    if (!FieldsSeen.insert(FieldName).second)
      return make_error<GenericBinaryError>(
          "producers section does not have unique fields",
          object_error::parse_failed);
    std::vector<std::pair<std::string, std::string>> *ProducerVec = nullptr;
    if (FieldName == "language") {
      ProducerVec = &ProducerInfo.Languages;
    } else if (FieldName == "processed-by") {
      ProducerVec = &ProducerInfo.Tools;
    } else if (FieldName == "sdk") {
      ProducerVec = &ProducerInfo.SDKs;
    } else {
      return make_error<GenericBinaryError>(
          "producers section field is not named one of language, processed-by, "
          "or sdk",
          object_error::parse_failed);
    }
    uint32_t ValueCount;
    RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(ValueCount));
    llvm::SmallSet<StringRef, 8> ProducersSeen;
    for (size_t J = 0; J < ValueCount; ++J) {
      StringRef Name;
      RETURN_IF_ERROR(readString(Ctx).moveInto(Name));
      StringRef Version;
      RETURN_IF_ERROR(readString(Ctx).moveInto(Version));
      if (!ProducersSeen.insert(Name).second) {
        return make_error<GenericBinaryError>(
            "producers section contains repeated producer",
            object_error::parse_failed);
      }
      ProducerVec->emplace_back(std::string(Name), std::string(Version));
    }
  }
  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("producers section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseTargetFeaturesSection(ReadContext &Ctx) {
  llvm::SmallSet<std::string, 8> FeaturesSeen;
  uint32_t FeatureCount;
  RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(FeatureCount));
  for (size_t I = 0; I < FeatureCount; ++I) {
    wasm::WasmFeatureEntry Feature;
    RETURN_IF_ERROR(readUint8(Ctx).moveInto(Feature.Prefix));
    switch (Feature.Prefix) {
    case wasm::WASM_FEATURE_PREFIX_USED:
    case wasm::WASM_FEATURE_PREFIX_DISALLOWED:
      break;
    default:
      return make_error<GenericBinaryError>("unknown feature policy prefix",
                                            object_error::parse_failed);
    }
    StringRef FeatureName;
    RETURN_IF_ERROR(readString(Ctx).moveInto(FeatureName));
    Feature.Name = std::string(FeatureName);
    if (!FeaturesSeen.insert(Feature.Name).second)
      return make_error<GenericBinaryError>(
          "target features section contains repeated feature \"" +
              Feature.Name + "\"",
          object_error::parse_failed);
    TargetFeatures.push_back(Feature);
  }
  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>(
        "target features section ended prematurely",
        object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseRelocSection(StringRef Name, ReadContext &Ctx) {
  uint32_t SectionIndex;
  RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(SectionIndex));
  if (SectionIndex >= Sections.size())
    return make_error<GenericBinaryError>("invalid section index",
                                          object_error::parse_failed);
  WasmSection &Section = Sections[SectionIndex];
  uint32_t RelocCount;
  RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(RelocCount));
  uint32_t EndOffset = Section.Content.size();
  uint32_t PreviousOffset = 0;
  while (RelocCount--) {
    wasm::WasmRelocation Reloc = {};
    uint32_t type;
    RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(type));
    Reloc.Type = type;
    uint32_t Offset;
    RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Offset));
    Reloc.Offset = Offset;
    if (Reloc.Offset < PreviousOffset)
      return make_error<GenericBinaryError>("relocations not in offset order",
                                            object_error::parse_failed);

    auto badReloc = [&](StringRef msg) {
      if (Reloc.Index >= Symbols.size())
        return make_error<GenericBinaryError>(
            msg + ": index " + Twine(Reloc.Index) + " out of range",
            object_error::parse_failed);
      return make_error<GenericBinaryError>(
          msg + ": " + Twine(Symbols[Reloc.Index].Info.Name),
          object_error::parse_failed);
    };

    PreviousOffset = Reloc.Offset;
    RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Reloc.Index));
    switch (type) {
    case wasm::R_WASM_FUNCTION_INDEX_LEB:
    case wasm::R_WASM_FUNCTION_INDEX_I32:
    case wasm::R_WASM_TABLE_INDEX_SLEB:
    case wasm::R_WASM_TABLE_INDEX_SLEB64:
    case wasm::R_WASM_TABLE_INDEX_I32:
    case wasm::R_WASM_TABLE_INDEX_I64:
    case wasm::R_WASM_TABLE_INDEX_REL_SLEB:
    case wasm::R_WASM_TABLE_INDEX_REL_SLEB64:
      if (!isValidFunctionSymbol(Reloc.Index))
        return badReloc("invalid function relocation");
      break;
    case wasm::R_WASM_TABLE_NUMBER_LEB:
      if (!isValidTableSymbol(Reloc.Index))
        return badReloc("invalid table relocation");
      break;
    case wasm::R_WASM_TYPE_INDEX_LEB:
      if (Reloc.Index >= Signatures.size())
        return badReloc("invalid relocation type index");
      break;
    case wasm::R_WASM_GLOBAL_INDEX_LEB:
      // R_WASM_GLOBAL_INDEX_LEB are can be used against function and data
      // symbols to refer to their GOT entries.
      if (!isValidGlobalSymbol(Reloc.Index) &&
          !isValidDataSymbol(Reloc.Index) &&
          !isValidFunctionSymbol(Reloc.Index))
        return badReloc("invalid global relocation");
      break;
    case wasm::R_WASM_GLOBAL_INDEX_I32:
      if (!isValidGlobalSymbol(Reloc.Index))
        return badReloc("invalid global relocation");
      break;
    case wasm::R_WASM_TAG_INDEX_LEB:
      if (!isValidTagSymbol(Reloc.Index))
        return badReloc("invalid tag relocation");
      break;
    case wasm::R_WASM_MEMORY_ADDR_LEB:
    case wasm::R_WASM_MEMORY_ADDR_SLEB:
    case wasm::R_WASM_MEMORY_ADDR_I32:
    case wasm::R_WASM_MEMORY_ADDR_REL_SLEB:
    case wasm::R_WASM_MEMORY_ADDR_TLS_SLEB:
    case wasm::R_WASM_MEMORY_ADDR_LOCREL_I32:
      if (!isValidDataSymbol(Reloc.Index))
        return badReloc("invalid data relocation");
      {
        int32_t Addend;
        RETURN_IF_ERROR(readVarint32(Ctx).moveInto(Addend));
        Reloc.Addend = Addend;
      }
      break;
    case wasm::R_WASM_MEMORY_ADDR_LEB64:
    case wasm::R_WASM_MEMORY_ADDR_SLEB64:
    case wasm::R_WASM_MEMORY_ADDR_I64:
    case wasm::R_WASM_MEMORY_ADDR_REL_SLEB64:
    case wasm::R_WASM_MEMORY_ADDR_TLS_SLEB64:
    case wasm::R_WASM_MEMORY_ADDR_LOCREL_I64:
      if (!isValidDataSymbol(Reloc.Index))
        return badReloc("invalid data relocation");
      RETURN_IF_ERROR(readVarint64(Ctx).moveInto(Reloc.Addend));
      break;
    case wasm::R_WASM_FUNCTION_OFFSET_I32:
      if (!isValidFunctionSymbol(Reloc.Index))
        return badReloc("invalid function relocation");
      {
        int32_t Addend;
        RETURN_IF_ERROR(readVarint32(Ctx).moveInto(Addend));
        Reloc.Addend = Addend;
      }
      break;
    case wasm::R_WASM_FUNCTION_OFFSET_I64:
      if (!isValidFunctionSymbol(Reloc.Index))
        return badReloc("invalid function relocation");
      RETURN_IF_ERROR(readVarint64(Ctx).moveInto(Reloc.Addend));
      break;
    case wasm::R_WASM_SECTION_OFFSET_I32:
      if (!isValidSectionSymbol(Reloc.Index))
        return badReloc("invalid section relocation");
      {
        int32_t Addend;
        RETURN_IF_ERROR(readVarint32(Ctx).moveInto(Addend));
        Reloc.Addend = Addend;
      }
      break;
    default:
      return make_error<GenericBinaryError>("invalid relocation type: " +
                                                Twine(type),
                                            object_error::parse_failed);
    }

    // Relocations must fit inside the section, and must appear in order.  They
    // also shouldn't overlap a function/element boundary, but we don't bother
    // to check that.
    uint64_t Size = 5;
    if (Reloc.Type == wasm::R_WASM_MEMORY_ADDR_LEB64 ||
        Reloc.Type == wasm::R_WASM_MEMORY_ADDR_SLEB64 ||
        Reloc.Type == wasm::R_WASM_MEMORY_ADDR_REL_SLEB64)
      Size = 10;
    if (Reloc.Type == wasm::R_WASM_TABLE_INDEX_I32 ||
        Reloc.Type == wasm::R_WASM_MEMORY_ADDR_I32 ||
        Reloc.Type == wasm::R_WASM_MEMORY_ADDR_LOCREL_I32 ||
        Reloc.Type == wasm::R_WASM_SECTION_OFFSET_I32 ||
        Reloc.Type == wasm::R_WASM_FUNCTION_OFFSET_I32 ||
        Reloc.Type == wasm::R_WASM_FUNCTION_INDEX_I32 ||
        Reloc.Type == wasm::R_WASM_GLOBAL_INDEX_I32)
      Size = 4;
    if (Reloc.Type == wasm::R_WASM_TABLE_INDEX_I64 ||
        Reloc.Type == wasm::R_WASM_MEMORY_ADDR_I64 ||
        Reloc.Type == wasm::R_WASM_FUNCTION_OFFSET_I64 ||
        Reloc.Type == wasm::R_WASM_MEMORY_ADDR_LOCREL_I64)
      Size = 8;
    if (Reloc.Offset + Size > EndOffset)
      return make_error<GenericBinaryError>("invalid relocation offset",
                                            object_error::parse_failed);

    Section.Relocations.push_back(Reloc);
  }
  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("reloc section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseCustomSection(WasmSection &Sec, ReadContext &Ctx) {
  if (Sec.Name == "dylink") {
    if (Error Err = parseDylinkSection(Ctx))
      return Err;
  } else if (Sec.Name == "dylink.0") {
    if (Error Err = parseDylink0Section(Ctx))
      return Err;
  } else if (Sec.Name == "name") {
    if (Error Err = parseNameSection(Ctx))
      return Err;
  } else if (Sec.Name == "linking") {
    if (Error Err = parseLinkingSection(Ctx))
      return Err;
  } else if (Sec.Name == "producers") {
    if (Error Err = parseProducersSection(Ctx))
      return Err;
  } else if (Sec.Name == "target_features") {
    if (Error Err = parseTargetFeaturesSection(Ctx))
      return Err;
  } else if (Sec.Name.starts_with("reloc.")) {
    if (Error Err = parseRelocSection(Sec.Name, Ctx))
      return Err;
  }
  return Error::success();
}

Error WasmObjectFile::parseTypeSection(ReadContext &Ctx) {
  auto parseFieldDef = [&]() -> Error {
    uint32_t TypeCode;
    RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(TypeCode));
    RETURN_IF_ERROR(parseValType(Ctx, TypeCode).takeError());
    RETURN_IF_ERROR(readVaruint32(Ctx).takeError());
    return Error::success();
  };

  uint32_t Count;
  RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Count));
  Signatures.reserve(Count);
  while (Count--) {
    wasm::WasmSignature Sig;
    uint8_t Form;
    RETURN_IF_ERROR(readUint8(Ctx).moveInto(Form));
    if (Form == wasm::WASM_TYPE_REC) {
      // Rec groups expand the type index space (beyond what was declared at
      // the top of the section, and also consume one element in that space.
      uint32_t RecSize;
      RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(RecSize));
      if (RecSize == 0)
        return make_error<GenericBinaryError>("Rec group size cannot be 0",
                                              object_error::parse_failed);
      Signatures.reserve(Signatures.size() + RecSize);
      Count += RecSize;
      Sig.Kind = wasm::WasmSignature::Placeholder;
      Signatures.push_back(std::move(Sig));
      HasUnmodeledTypes = true;
      continue;
    }
    if (Form != wasm::WASM_TYPE_FUNC) {
      // Currently LLVM only models function types, and not other composite
      // types. Here we parse the type declarations just enough to skip past
      // them in the binary.
      if (Form == wasm::WASM_TYPE_SUB || Form == wasm::WASM_TYPE_SUB_FINAL) {
        uint32_t Supers;
        RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Supers));
        if (Supers > 0) {
          if (Supers != 1)
            return make_error<GenericBinaryError>(
                "Invalid number of supertypes", object_error::parse_failed);
          RETURN_IF_ERROR(readVaruint32(Ctx).takeError());
        }
        uint32_t FormVal;
        RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(FormVal));
        Form = FormVal;
      }
      if (Form == wasm::WASM_TYPE_STRUCT) {
        uint32_t FieldCount;
        RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(FieldCount));
        while (FieldCount--) {
          if (Error E = parseFieldDef())
            return E;
        }
      } else if (Form == wasm::WASM_TYPE_ARRAY) {
        if (Error E = parseFieldDef())
          return E;
      } else {
        return make_error<GenericBinaryError>("bad form",
                                              object_error::parse_failed);
      }
      Sig.Kind = wasm::WasmSignature::Placeholder;
      Signatures.push_back(std::move(Sig));
      HasUnmodeledTypes = true;
      continue;
    }

    uint32_t ParamCount;
    RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(ParamCount));
    Sig.Params.reserve(ParamCount);
    while (ParamCount--) {
      uint8_t ParamType;
      RETURN_IF_ERROR(readUint8(Ctx).moveInto(ParamType));
      wasm::ValType ParamValType;
      RETURN_IF_ERROR(parseValType(Ctx, ParamType).moveInto(ParamValType));
      Sig.Params.push_back(ParamValType);
    }
    uint32_t ReturnCount;
    RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(ReturnCount));
    while (ReturnCount--) {
      uint8_t ReturnType;
      RETURN_IF_ERROR(readUint8(Ctx).moveInto(ReturnType));
      wasm::ValType ReturnValType;
      RETURN_IF_ERROR(parseValType(Ctx, ReturnType).moveInto(ReturnValType));
      Sig.Returns.push_back(ReturnValType);
    }

    Signatures.push_back(std::move(Sig));
  }
  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("type section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseImport(ReadContext &Ctx, wasm::WasmImport &Im) {
  switch (Im.Kind) {
  case wasm::WASM_EXTERNAL_FUNCTION:
    NumImportedFunctions++;
    RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Im.SigIndex));
    if (Im.SigIndex >= Signatures.size())
      return make_error<GenericBinaryError>("invalid function type",
                                            object_error::parse_failed);
    break;
  case wasm::WASM_EXTERNAL_GLOBAL:
    NumImportedGlobals++;
    RETURN_IF_ERROR(readUint8(Ctx).moveInto(Im.Global.Type));
    uint8_t Mutable;
    RETURN_IF_ERROR(readVaruint1(Ctx).moveInto(Mutable));
    Im.Global.Mutable = Mutable;
    break;
  case wasm::WASM_EXTERNAL_MEMORY:
    RETURN_IF_ERROR(readLimits(Ctx).moveInto(Im.Memory));
    if (Im.Memory.Flags & wasm::WASM_LIMITS_FLAG_IS_64)
      HasMemory64 = true;
    break;
  case wasm::WASM_EXTERNAL_TABLE: {
    RETURN_IF_ERROR(readTableType(Ctx).moveInto(Im.Table));
    NumImportedTables++;
    auto ElemType = Im.Table.ElemType;
    if (ElemType != wasm::ValType::FUNCREF &&
        ElemType != wasm::ValType::EXTERNREF &&
        ElemType != wasm::ValType::EXNREF &&
        ElemType != wasm::ValType::OTHERREF)
      return make_error<GenericBinaryError>("invalid table element type",
                                            object_error::parse_failed);
    break;
  }
  case wasm::WASM_EXTERNAL_TAG:
    NumImportedTags++;
    uint8_t Attr;
    RETURN_IF_ERROR(readUint8(Ctx).moveInto(Attr));
    if (Attr != 0) // Reserved 'attribute' field
      return make_error<GenericBinaryError>("invalid attribute",
                                            object_error::parse_failed);
    RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Im.SigIndex));
    if (Im.SigIndex >= Signatures.size())
      return make_error<GenericBinaryError>("invalid tag type",
                                            object_error::parse_failed);
    break;
  default:
    return make_error<GenericBinaryError>("unexpected import kind: " +
                                              Twine(unsigned(Im.Kind)),
                                          object_error::parse_failed);
  }
  Imports.push_back(Im);
  return Error::success();
}

Error WasmObjectFile::parseImportSection(ReadContext &Ctx) {
  uint32_t Count;
  RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Count));
  Imports.reserve(Count);
  uint32_t I = 0;
  while (I < Count) {
    wasm::WasmImport Im;
    RETURN_IF_ERROR(readString(Ctx).moveInto(Im.Module));
    RETURN_IF_ERROR(readString(Ctx).moveInto(Im.Field));
    RETURN_IF_ERROR(readUint8(Ctx).moveInto(Im.Kind));
    // 0x7E/0x7F along with an empty Field signals a block of compact imports.
    if (Im.Kind == 0x7E && Im.Field == "") {
      return make_error<GenericBinaryError>(
          "compact import format (0x7E) is not yet supported",
          object_error::parse_failed);
    } else if (Im.Kind == 0x7F && Im.Field == "") {
      uint32_t NumCompactImports;
      RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(NumCompactImports));
      while (NumCompactImports--) {
        RETURN_IF_ERROR(readString(Ctx).moveInto(Im.Field));
        RETURN_IF_ERROR(readUint8(Ctx).moveInto(Im.Kind));
        Error rtn = parseImport(Ctx, Im);
        if (rtn)
          return rtn;
        I++;
      }
    } else {
      Error rtn = parseImport(Ctx, Im);
      if (rtn)
        return rtn;
      I++;
    }
  }
  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("import section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseFunctionSection(ReadContext &Ctx) {
  uint32_t Count;
  RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Count));
  Functions.reserve(Count);
  uint32_t NumTypes = Signatures.size();
  while (Count--) {
    uint32_t Type;
    RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Type));
    if (Type >= NumTypes)
      return make_error<GenericBinaryError>("invalid function type",
                                            object_error::parse_failed);
    wasm::WasmFunction F;
    F.SigIndex = Type;
    Functions.push_back(F);
  }
  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("function section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseTableSection(ReadContext &Ctx) {
  TableSection = Sections.size();
  uint32_t Count;
  RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Count));
  Tables.reserve(Count);
  while (Count--) {
    wasm::WasmTable T;
    RETURN_IF_ERROR(readTableType(Ctx).moveInto(T.Type));
    T.Index = NumImportedTables + Tables.size();
    Tables.push_back(T);
    auto ElemType = Tables.back().Type.ElemType;
    if (ElemType != wasm::ValType::FUNCREF &&
        ElemType != wasm::ValType::EXTERNREF &&
        ElemType != wasm::ValType::EXNREF &&
        ElemType != wasm::ValType::OTHERREF) {
      return make_error<GenericBinaryError>("invalid table element type",
                                            object_error::parse_failed);
    }
  }
  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("table section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseMemorySection(ReadContext &Ctx) {
  uint32_t Count;
  RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Count));
  Memories.reserve(Count);
  while (Count--) {
    wasm::WasmLimits Limits;
    RETURN_IF_ERROR(readLimits(Ctx).moveInto(Limits));
    if (Limits.Flags & wasm::WASM_LIMITS_FLAG_IS_64)
      HasMemory64 = true;
    Memories.push_back(Limits);
  }
  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("memory section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseTagSection(ReadContext &Ctx) {
  TagSection = Sections.size();
  uint32_t Count;
  RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Count));
  Tags.reserve(Count);
  uint32_t NumTypes = Signatures.size();
  while (Count--) {
    uint8_t Attr;
    RETURN_IF_ERROR(readUint8(Ctx).moveInto(Attr));
    if (Attr != 0) // Reserved 'attribute' field
      return make_error<GenericBinaryError>("invalid attribute",
                                            object_error::parse_failed);
    uint32_t Type;
    RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Type));
    if (Type >= NumTypes)
      return make_error<GenericBinaryError>("invalid tag type",
                                            object_error::parse_failed);
    wasm::WasmTag Tag;
    Tag.Index = NumImportedTags + Tags.size();
    Tag.SigIndex = Type;
    Signatures[Type].Kind = wasm::WasmSignature::Tag;
    Tags.push_back(Tag);
  }

  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("tag section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseGlobalSection(ReadContext &Ctx) {
  GlobalSection = Sections.size();
  const uint8_t *SectionStart = Ctx.Ptr;
  uint32_t Count;
  RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Count));
  Globals.reserve(Count);
  while (Count--) {
    wasm::WasmGlobal Global;
    Global.Index = NumImportedGlobals + Globals.size();
    const uint8_t *GlobalStart = Ctx.Ptr;
    Global.Offset = static_cast<uint32_t>(GlobalStart - SectionStart);
    uint32_t GlobalOpcode;
    RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(GlobalOpcode));
    wasm::ValType GlobalValType;
    RETURN_IF_ERROR(parseValType(Ctx, GlobalOpcode).moveInto(GlobalValType));
    Global.Type.Type = (uint8_t)GlobalValType;
    uint8_t Mutable;
    RETURN_IF_ERROR(readVaruint1(Ctx).moveInto(Mutable));
    Global.Type.Mutable = Mutable;
    if (Error Err = readInitExpr(Global.InitExpr, Ctx))
      return Err;
    Global.Size = static_cast<uint32_t>(Ctx.Ptr - GlobalStart);
    Globals.push_back(Global);
  }
  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("global section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseExportSection(ReadContext &Ctx) {
  uint32_t Count;
  RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Count));
  Exports.reserve(Count);
  Symbols.reserve(Count);

  // Build hash map of export flags for faster cross-referencing
  llvm::DenseMap<StringRef, uint32_t> ExportFlags;
  if (HasDylinkSection) {
    for (const auto &ExportInfo : DylinkInfo.ExportInfo) {
      ExportFlags[ExportInfo.Name] = ExportInfo.Flags;
    }
  }

  for (uint32_t I = 0; I < Count; I++) {
    wasm::WasmExport Ex;
    RETURN_IF_ERROR(readString(Ctx).moveInto(Ex.Name));
    RETURN_IF_ERROR(readUint8(Ctx).moveInto(Ex.Kind));
    RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Ex.Index));
    const wasm::WasmSignature *Signature = nullptr;
    const wasm::WasmGlobalType *GlobalType = nullptr;
    const wasm::WasmTableType *TableType = nullptr;
    wasm::WasmSymbolInfo Info;
    Info.Name = Ex.Name;
    Info.Flags = 0;
    // For shared objects, symbol flags may be specified in the dylink section
    // instead of the export section
    if (HasDylinkSection) {
      auto It = ExportFlags.find(Ex.Name);
      if (It != ExportFlags.end()) {
        Info.Flags = It->second;
      }
    }
    switch (Ex.Kind) {
    case wasm::WASM_EXTERNAL_FUNCTION: {
      if (!isValidFunctionIndex(Ex.Index))
        return make_error<GenericBinaryError>("invalid function export",
                                              object_error::parse_failed);
      Info.Kind = wasm::WASM_SYMBOL_TYPE_FUNCTION;
      Info.ElementIndex = Ex.Index;
      if (isDefinedFunctionIndex(Ex.Index)) {
        getDefinedFunction(Ex.Index).ExportName = Ex.Name;
        unsigned FuncIndex = Info.ElementIndex - NumImportedFunctions;
        wasm::WasmFunction &Function = Functions[FuncIndex];
        Signature = &Signatures[Function.SigIndex];
      }
      // Else the function is imported. LLVM object files don't use this
      // pattern and we still treat this as an undefined symbol, but we want to
      // parse it without crashing.
      break;
    }
    case wasm::WASM_EXTERNAL_GLOBAL: {
      if (!isValidGlobalIndex(Ex.Index))
        return make_error<GenericBinaryError>("invalid global export",
                                              object_error::parse_failed);
      Info.Kind = wasm::WASM_SYMBOL_TYPE_DATA;
      uint64_t Offset = 0;
      if (isDefinedGlobalIndex(Ex.Index)) {
        auto Global = getDefinedGlobal(Ex.Index);
        if (!Global.InitExpr.Extended) {
          auto Inst = Global.InitExpr.Inst;
          if (Inst.Opcode == wasm::WASM_OPCODE_I32_CONST) {
            Offset = Inst.Value.Int32;
          } else if (Inst.Opcode == wasm::WASM_OPCODE_I64_CONST) {
            Offset = Inst.Value.Int64;
          }
        }
      }
      Info.DataRef = wasm::WasmDataReference{0, Offset, 0};
      break;
    }
    case wasm::WASM_EXTERNAL_TAG:
      if (!isValidTagIndex(Ex.Index))
        return make_error<GenericBinaryError>("invalid tag export",
                                              object_error::parse_failed);
      Info.Kind = wasm::WASM_SYMBOL_TYPE_TAG;
      Info.ElementIndex = Ex.Index;
      if (isDefinedTagIndex(Ex.Index)) {
        unsigned TagIndex = Ex.Index - NumImportedTags;
        Signature = &Signatures[Tags[TagIndex].SigIndex];
      }
      break;
    case wasm::WASM_EXTERNAL_MEMORY:
      break;
    case wasm::WASM_EXTERNAL_TABLE:
      Info.Kind = wasm::WASM_SYMBOL_TYPE_TABLE;
      Info.ElementIndex = Ex.Index;
      break;
    default:
      return make_error<GenericBinaryError>("unexpected export kind",
                                            object_error::parse_failed);
    }
    Exports.push_back(Ex);
    if (Ex.Kind != wasm::WASM_EXTERNAL_MEMORY) {
      Symbols.emplace_back(Info, GlobalType, TableType, Signature);
      LLVM_DEBUG(dbgs() << "Adding symbol: " << Symbols.back() << "\n");
    }
  }
  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("export section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

bool WasmObjectFile::isValidFunctionIndex(uint32_t Index) const {
  return Index < NumImportedFunctions + Functions.size();
}

bool WasmObjectFile::isDefinedFunctionIndex(uint32_t Index) const {
  return Index >= NumImportedFunctions && isValidFunctionIndex(Index);
}

bool WasmObjectFile::isValidGlobalIndex(uint32_t Index) const {
  return Index < NumImportedGlobals + Globals.size();
}

bool WasmObjectFile::isValidTableNumber(uint32_t Index) const {
  return Index < NumImportedTables + Tables.size();
}

bool WasmObjectFile::isDefinedGlobalIndex(uint32_t Index) const {
  return Index >= NumImportedGlobals && isValidGlobalIndex(Index);
}

bool WasmObjectFile::isDefinedTableNumber(uint32_t Index) const {
  return Index >= NumImportedTables && isValidTableNumber(Index);
}

bool WasmObjectFile::isValidTagIndex(uint32_t Index) const {
  return Index < NumImportedTags + Tags.size();
}

bool WasmObjectFile::isDefinedTagIndex(uint32_t Index) const {
  return Index >= NumImportedTags && isValidTagIndex(Index);
}

bool WasmObjectFile::isValidFunctionSymbol(uint32_t Index) const {
  return Index < Symbols.size() && Symbols[Index].isTypeFunction();
}

bool WasmObjectFile::isValidTableSymbol(uint32_t Index) const {
  return Index < Symbols.size() && Symbols[Index].isTypeTable();
}

bool WasmObjectFile::isValidGlobalSymbol(uint32_t Index) const {
  return Index < Symbols.size() && Symbols[Index].isTypeGlobal();
}

bool WasmObjectFile::isValidTagSymbol(uint32_t Index) const {
  return Index < Symbols.size() && Symbols[Index].isTypeTag();
}

bool WasmObjectFile::isValidDataSymbol(uint32_t Index) const {
  return Index < Symbols.size() && Symbols[Index].isTypeData();
}

bool WasmObjectFile::isValidSectionSymbol(uint32_t Index) const {
  return Index < Symbols.size() && Symbols[Index].isTypeSection();
}

wasm::WasmFunction &WasmObjectFile::getDefinedFunction(uint32_t Index) {
  assert(isDefinedFunctionIndex(Index));
  return Functions[Index - NumImportedFunctions];
}

const wasm::WasmFunction &
WasmObjectFile::getDefinedFunction(uint32_t Index) const {
  assert(isDefinedFunctionIndex(Index));
  return Functions[Index - NumImportedFunctions];
}

const wasm::WasmGlobal &WasmObjectFile::getDefinedGlobal(uint32_t Index) const {
  assert(isDefinedGlobalIndex(Index));
  return Globals[Index - NumImportedGlobals];
}

wasm::WasmTag &WasmObjectFile::getDefinedTag(uint32_t Index) {
  assert(isDefinedTagIndex(Index));
  return Tags[Index - NumImportedTags];
}

Error WasmObjectFile::parseStartSection(ReadContext &Ctx) {
  RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(StartFunction));
  if (!isValidFunctionIndex(StartFunction))
    return make_error<GenericBinaryError>("invalid start function",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseCodeSection(ReadContext &Ctx) {
  CodeSection = Sections.size();
  uint32_t FunctionCount;
  RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(FunctionCount));
  if (FunctionCount != Functions.size()) {
    return make_error<GenericBinaryError>("invalid function count",
                                          object_error::parse_failed);
  }

  for (uint32_t i = 0; i < FunctionCount; i++) {
    wasm::WasmFunction& Function = Functions[i];
    const uint8_t *FunctionStart = Ctx.Ptr;
    uint32_t Size;
    RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Size));
    const uint8_t *FunctionEnd = Ctx.Ptr + Size;

    Function.CodeOffset = Ctx.Ptr - FunctionStart;
    Function.Index = NumImportedFunctions + i;
    Function.CodeSectionOffset = FunctionStart - Ctx.Start;
    Function.Size = FunctionEnd - FunctionStart;

    uint32_t NumLocalDecls;
    RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(NumLocalDecls));
    Function.Locals.reserve(NumLocalDecls);
    while (NumLocalDecls--) {
      wasm::WasmLocalDecl Decl;
      RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Decl.Count));
      RETURN_IF_ERROR(readUint8(Ctx).moveInto(Decl.Type));
      Function.Locals.push_back(Decl);
    }

    uint32_t BodySize = FunctionEnd - Ctx.Ptr;
    // Ensure that Function is within Ctx's buffer.
    if (Ctx.Ptr + BodySize > Ctx.End) {
      return make_error<GenericBinaryError>("Function extends beyond buffer",
                                            object_error::parse_failed);
    }
    Function.Body = ArrayRef<uint8_t>(Ctx.Ptr, BodySize);
    // This will be set later when reading in the linking metadata section.
    Function.Comdat = UINT32_MAX;
    Ctx.Ptr += BodySize;
    assert(Ctx.Ptr == FunctionEnd);
  }
  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("code section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseElemSection(ReadContext &Ctx) {
  uint32_t Count;
  RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Count));
  ElemSegments.reserve(Count);
  while (Count--) {
    wasm::WasmElemSegment Segment;
    RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Segment.Flags));

    uint32_t SupportedFlags = wasm::WASM_ELEM_SEGMENT_HAS_TABLE_NUMBER |
                              wasm::WASM_ELEM_SEGMENT_IS_PASSIVE |
                              wasm::WASM_ELEM_SEGMENT_HAS_INIT_EXPRS;
    if (Segment.Flags & ~SupportedFlags)
      return make_error<GenericBinaryError>(
          "Unsupported flags for element segment", object_error::parse_failed);

    wasm::ElemSegmentMode Mode;
    if ((Segment.Flags & wasm::WASM_ELEM_SEGMENT_IS_PASSIVE) == 0) {
      Mode = wasm::ElemSegmentMode::Active;
    } else if (Segment.Flags & wasm::WASM_ELEM_SEGMENT_IS_DECLARATIVE) {
      Mode = wasm::ElemSegmentMode::Declarative;
    } else {
      Mode = wasm::ElemSegmentMode::Passive;
    }
    bool HasTableNumber =
        Mode == wasm::ElemSegmentMode::Active &&
        (Segment.Flags & wasm::WASM_ELEM_SEGMENT_HAS_TABLE_NUMBER);
    bool HasElemKind =
        (Segment.Flags & wasm::WASM_ELEM_SEGMENT_MASK_HAS_ELEM_DESC) &&
        !(Segment.Flags & wasm::WASM_ELEM_SEGMENT_HAS_INIT_EXPRS);
    bool HasElemType =
        (Segment.Flags & wasm::WASM_ELEM_SEGMENT_MASK_HAS_ELEM_DESC) &&
        (Segment.Flags & wasm::WASM_ELEM_SEGMENT_HAS_INIT_EXPRS);
    bool HasInitExprs =
        (Segment.Flags & wasm::WASM_ELEM_SEGMENT_HAS_INIT_EXPRS);

    if (HasTableNumber) {
      RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Segment.TableNumber));
    } else {
      Segment.TableNumber = 0;
    }

    if (!isValidTableNumber(Segment.TableNumber))
      return make_error<GenericBinaryError>("invalid TableNumber",
                                            object_error::parse_failed);

    if (Mode != wasm::ElemSegmentMode::Active) {
      Segment.Offset.Extended = false;
      Segment.Offset.Inst.Opcode = wasm::WASM_OPCODE_I32_CONST;
      Segment.Offset.Inst.Value.Int32 = 0;
    } else {
      if (Error Err = readInitExpr(Segment.Offset, Ctx))
        return Err;
    }

    if (HasElemKind) {
      uint32_t ElemKind;
      RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(ElemKind));
      if (Segment.Flags & wasm::WASM_ELEM_SEGMENT_HAS_INIT_EXPRS) {
        RETURN_IF_ERROR(parseValType(Ctx, ElemKind).moveInto(Segment.ElemKind));
        if (Segment.ElemKind != wasm::ValType::FUNCREF &&
            Segment.ElemKind != wasm::ValType::EXTERNREF &&
            Segment.ElemKind != wasm::ValType::EXNREF &&
            Segment.ElemKind != wasm::ValType::OTHERREF) {
          return make_error<GenericBinaryError>("invalid elem type",
                                                object_error::parse_failed);
        }
      } else {
        if (ElemKind != 0)
          return make_error<GenericBinaryError>("invalid elem type",
                                                object_error::parse_failed);
        Segment.ElemKind = wasm::ValType::FUNCREF;
      }
    } else if (HasElemType) {
      uint32_t ElemTypeCode;
      RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(ElemTypeCode));
      RETURN_IF_ERROR(
          parseValType(Ctx, ElemTypeCode).moveInto(Segment.ElemKind));
    } else {
      Segment.ElemKind = wasm::ValType::FUNCREF;
    }

    uint32_t NumElems;
    RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(NumElems));

    if (HasInitExprs) {
      while (NumElems--) {
        wasm::WasmInitExpr Expr;
        if (Error Err = readInitExpr(Expr, Ctx))
          return Err;
      }
    } else {
      while (NumElems--) {
        uint32_t Index;
        RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Index));
        Segment.Functions.push_back(Index);
      }
    }
    ElemSegments.push_back(Segment);
  }
  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("elem section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseDataSection(ReadContext &Ctx) {
  DataSection = Sections.size();
  uint32_t Count;
  RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Count));
  if (DataCount && Count != *DataCount)
    return make_error<GenericBinaryError>(
        "number of data segments does not match DataCount section");
  DataSegments.reserve(Count);
  while (Count--) {
    WasmSegment Segment;
    RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Segment.Data.InitFlags));
    if (Segment.Data.InitFlags & wasm::WASM_DATA_SEGMENT_HAS_MEMINDEX) {
      RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Segment.Data.MemoryIndex));
    } else {
      Segment.Data.MemoryIndex = 0;
    }
    if ((Segment.Data.InitFlags & wasm::WASM_DATA_SEGMENT_IS_PASSIVE) == 0) {
      if (Error Err = readInitExpr(Segment.Data.Offset, Ctx))
        return Err;
    } else {
      Segment.Data.Offset.Extended = false;
      Segment.Data.Offset.Inst.Opcode = wasm::WASM_OPCODE_I32_CONST;
      Segment.Data.Offset.Inst.Value.Int32 = 0;
    }
    uint32_t Size;
    RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Size));
    if (Size > (size_t)(Ctx.End - Ctx.Ptr))
      return make_error<GenericBinaryError>("invalid segment size",
                                            object_error::parse_failed);
    Segment.Data.Content = ArrayRef<uint8_t>(Ctx.Ptr, Size);
    // The rest of these Data fields are set later, when reading in the linking
    // metadata section.
    Segment.Data.Alignment = 0;
    Segment.Data.LinkingFlags = 0;
    Segment.Data.Comdat = UINT32_MAX;
    Segment.SectionOffset = Ctx.Ptr - Ctx.Start;
    Ctx.Ptr += Size;
    DataSegments.push_back(Segment);
  }
  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("data section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseDataCountSection(ReadContext &Ctx) {
  uint32_t Count;
  RETURN_IF_ERROR(readVaruint32(Ctx).moveInto(Count));
  DataCount = Count;
  return Error::success();
}

const wasm::WasmObjectHeader &WasmObjectFile::getHeader() const {
  return Header;
}

void WasmObjectFile::moveSymbolNext(DataRefImpl &Symb) const { Symb.d.b++; }

Expected<uint32_t> WasmObjectFile::getSymbolFlags(DataRefImpl Symb) const {
  uint32_t Result = SymbolRef::SF_None;
  const WasmSymbol &Sym = getWasmSymbol(Symb);

  LLVM_DEBUG(dbgs() << "getSymbolFlags: ptr=" << &Sym << " " << Sym << "\n");
  if (Sym.isBindingWeak())
    Result |= SymbolRef::SF_Weak;
  if (!Sym.isBindingLocal())
    Result |= SymbolRef::SF_Global;
  if (Sym.isHidden())
    Result |= SymbolRef::SF_Hidden;
  if (!Sym.isDefined())
    Result |= SymbolRef::SF_Undefined;
  if (Sym.isTypeFunction())
    Result |= SymbolRef::SF_Executable;
  return Result;
}

basic_symbol_iterator WasmObjectFile::symbol_begin() const {
  DataRefImpl Ref;
  Ref.d.a = 1; // Arbitrary non-zero value so that Ref.p is non-null
  Ref.d.b = 0; // Symbol index
  return BasicSymbolRef(Ref, this);
}

basic_symbol_iterator WasmObjectFile::symbol_end() const {
  DataRefImpl Ref;
  Ref.d.a = 1; // Arbitrary non-zero value so that Ref.p is non-null
  Ref.d.b = Symbols.size(); // Symbol index
  return BasicSymbolRef(Ref, this);
}

const WasmSymbol &WasmObjectFile::getWasmSymbol(const DataRefImpl &Symb) const {
  return Symbols[Symb.d.b];
}

const WasmSymbol &WasmObjectFile::getWasmSymbol(const SymbolRef &Symb) const {
  return getWasmSymbol(Symb.getRawDataRefImpl());
}

Expected<StringRef> WasmObjectFile::getSymbolName(DataRefImpl Symb) const {
  return getWasmSymbol(Symb).Info.Name;
}

Expected<uint64_t> WasmObjectFile::getSymbolAddress(DataRefImpl Symb) const {
  auto &Sym = getWasmSymbol(Symb);
  if (!Sym.isDefined())
    return 0;
  UNWRAP_OR_RETURN(Sec, getSymbolSection(Symb));
  uint32_t SectionAddress = getSectionAddress(Sec->getRawDataRefImpl());
  if (Sym.Info.Kind == wasm::WASM_SYMBOL_TYPE_FUNCTION &&
      isDefinedFunctionIndex(Sym.Info.ElementIndex)) {
    return getDefinedFunction(Sym.Info.ElementIndex).CodeSectionOffset +
           SectionAddress;
  }
  if (Sym.Info.Kind == wasm::WASM_SYMBOL_TYPE_GLOBAL &&
      isDefinedGlobalIndex(Sym.Info.ElementIndex)) {
    return getDefinedGlobal(Sym.Info.ElementIndex).Offset + SectionAddress;
  }

  return getSymbolValue(Symb);
}

uint64_t WasmObjectFile::getWasmSymbolValue(const WasmSymbol &Sym) const {
  switch (Sym.Info.Kind) {
  case wasm::WASM_SYMBOL_TYPE_FUNCTION:
  case wasm::WASM_SYMBOL_TYPE_GLOBAL:
  case wasm::WASM_SYMBOL_TYPE_TAG:
  case wasm::WASM_SYMBOL_TYPE_TABLE:
    return Sym.Info.ElementIndex;
  case wasm::WASM_SYMBOL_TYPE_DATA: {
    // The value of a data symbol is the segment offset, plus the symbol
    // offset within the segment.
    uint32_t SegmentIndex = Sym.Info.DataRef.Segment;
    const wasm::WasmDataSegment &Segment = DataSegments[SegmentIndex].Data;
    if (Segment.Offset.Extended) {
      llvm_unreachable("extended init exprs not supported");
    } else if (Segment.Offset.Inst.Opcode == wasm::WASM_OPCODE_I32_CONST) {
      return Segment.Offset.Inst.Value.Int32 + Sym.Info.DataRef.Offset;
    } else if (Segment.Offset.Inst.Opcode == wasm::WASM_OPCODE_I64_CONST) {
      return Segment.Offset.Inst.Value.Int64 + Sym.Info.DataRef.Offset;
    } else if (Segment.Offset.Inst.Opcode == wasm::WASM_OPCODE_GLOBAL_GET) {
      return Sym.Info.DataRef.Offset;
    } else {
      llvm_unreachable("unknown init expr opcode");
    }
  }
  case wasm::WASM_SYMBOL_TYPE_SECTION:
    return 0;
  }
  llvm_unreachable("invalid symbol type");
}

uint64_t WasmObjectFile::getSymbolValueImpl(DataRefImpl Symb) const {
  return getWasmSymbolValue(getWasmSymbol(Symb));
}

uint32_t WasmObjectFile::getSymbolAlignment(DataRefImpl Symb) const {
  llvm_unreachable("not yet implemented");
  return 0;
}

uint64_t WasmObjectFile::getCommonSymbolSizeImpl(DataRefImpl Symb) const {
  llvm_unreachable("not yet implemented");
  return 0;
}

Expected<SymbolRef::Type>
WasmObjectFile::getSymbolType(DataRefImpl Symb) const {
  const WasmSymbol &Sym = getWasmSymbol(Symb);

  switch (Sym.Info.Kind) {
  case wasm::WASM_SYMBOL_TYPE_FUNCTION:
    return SymbolRef::ST_Function;
  case wasm::WASM_SYMBOL_TYPE_GLOBAL:
    return SymbolRef::ST_Other;
  case wasm::WASM_SYMBOL_TYPE_DATA:
    return SymbolRef::ST_Data;
  case wasm::WASM_SYMBOL_TYPE_SECTION:
    return SymbolRef::ST_Debug;
  case wasm::WASM_SYMBOL_TYPE_TAG:
    return SymbolRef::ST_Other;
  case wasm::WASM_SYMBOL_TYPE_TABLE:
    return SymbolRef::ST_Other;
  }

  llvm_unreachable("unknown WasmSymbol::SymbolType");
  return SymbolRef::ST_Other;
}

Expected<section_iterator>
WasmObjectFile::getSymbolSection(DataRefImpl Symb) const {
  const WasmSymbol &Sym = getWasmSymbol(Symb);
  if (Sym.isUndefined())
    return section_end();

  DataRefImpl Ref;
  Ref.d.a = getSymbolSectionIdImpl(Sym);
  return section_iterator(SectionRef(Ref, this));
}

uint32_t WasmObjectFile::getSymbolSectionId(SymbolRef Symb) const {
  const WasmSymbol &Sym = getWasmSymbol(Symb);
  return getSymbolSectionIdImpl(Sym);
}

uint32_t WasmObjectFile::getSymbolSectionIdImpl(const WasmSymbol &Sym) const {
  switch (Sym.Info.Kind) {
  case wasm::WASM_SYMBOL_TYPE_FUNCTION:
    return CodeSection;
  case wasm::WASM_SYMBOL_TYPE_GLOBAL:
    return GlobalSection;
  case wasm::WASM_SYMBOL_TYPE_DATA:
    return DataSection;
  case wasm::WASM_SYMBOL_TYPE_SECTION:
    return Sym.Info.ElementIndex;
  case wasm::WASM_SYMBOL_TYPE_TAG:
    return TagSection;
  case wasm::WASM_SYMBOL_TYPE_TABLE:
    return TableSection;
  default:
    llvm_unreachable("unknown WasmSymbol::SymbolType");
  }
}

uint32_t WasmObjectFile::getSymbolSize(SymbolRef Symb) const {
  const WasmSymbol &Sym = getWasmSymbol(Symb);
  if (!Sym.isDefined())
    return 0;
  if (Sym.isTypeGlobal())
    return getDefinedGlobal(Sym.Info.ElementIndex).Size;
  if (Sym.isTypeData())
    return Sym.Info.DataRef.Size;
  if (Sym.isTypeFunction())
    return functions()[Sym.Info.ElementIndex - getNumImportedFunctions()].Size;
  // Currently symbol size is only tracked for data segments and functions. In
  // principle we could also track size (e.g. binary size) for tables, globals
  // and element segments etc too.
  return 0;
}

void WasmObjectFile::moveSectionNext(DataRefImpl &Sec) const { Sec.d.a++; }

Expected<StringRef> WasmObjectFile::getSectionName(DataRefImpl Sec) const {
  const WasmSection &S = Sections[Sec.d.a];
  if (S.Type == wasm::WASM_SEC_CUSTOM)
    return S.Name;
  if (S.Type > wasm::WASM_SEC_LAST_KNOWN)
    return createStringError(object_error::invalid_section_index, "");
  return wasm::sectionTypeToString(S.Type);
}

uint64_t WasmObjectFile::getSectionAddress(DataRefImpl Sec) const {
  // For object files, use 0 for section addresses, and section offsets for
  // symbol addresses. For linked files, use file offsets.
  // See also getSymbolAddress.
  return isRelocatableObject() || isSharedObject() ? 0
                                                   : Sections[Sec.d.a].Offset;
}

uint64_t WasmObjectFile::getSectionIndex(DataRefImpl Sec) const {
  return Sec.d.a;
}

uint64_t WasmObjectFile::getSectionSize(DataRefImpl Sec) const {
  const WasmSection &S = Sections[Sec.d.a];
  return S.Content.size();
}

Expected<ArrayRef<uint8_t>>
WasmObjectFile::getSectionContents(DataRefImpl Sec) const {
  const WasmSection &S = Sections[Sec.d.a];
  // This will never fail since wasm sections can never be empty (user-sections
  // must have a name and non-user sections each have a defined structure).
  return S.Content;
}

uint64_t WasmObjectFile::getSectionAlignment(DataRefImpl Sec) const {
  return 1;
}

bool WasmObjectFile::isSectionCompressed(DataRefImpl Sec) const {
  return false;
}

bool WasmObjectFile::isSectionText(DataRefImpl Sec) const {
  return getWasmSection(Sec).Type == wasm::WASM_SEC_CODE;
}

bool WasmObjectFile::isSectionData(DataRefImpl Sec) const {
  return getWasmSection(Sec).Type == wasm::WASM_SEC_DATA;
}

bool WasmObjectFile::isSectionBSS(DataRefImpl Sec) const { return false; }

bool WasmObjectFile::isSectionVirtual(DataRefImpl Sec) const { return false; }

relocation_iterator WasmObjectFile::section_rel_begin(DataRefImpl Ref) const {
  DataRefImpl RelocRef;
  RelocRef.d.a = Ref.d.a;
  RelocRef.d.b = 0;
  return relocation_iterator(RelocationRef(RelocRef, this));
}

relocation_iterator WasmObjectFile::section_rel_end(DataRefImpl Ref) const {
  const WasmSection &Sec = getWasmSection(Ref);
  DataRefImpl RelocRef;
  RelocRef.d.a = Ref.d.a;
  RelocRef.d.b = Sec.Relocations.size();
  return relocation_iterator(RelocationRef(RelocRef, this));
}

void WasmObjectFile::moveRelocationNext(DataRefImpl &Rel) const { Rel.d.b++; }

uint64_t WasmObjectFile::getRelocationOffset(DataRefImpl Ref) const {
  const wasm::WasmRelocation &Rel = getWasmRelocation(Ref);
  return Rel.Offset;
}

symbol_iterator WasmObjectFile::getRelocationSymbol(DataRefImpl Ref) const {
  const wasm::WasmRelocation &Rel = getWasmRelocation(Ref);
  if (Rel.Type == wasm::R_WASM_TYPE_INDEX_LEB)
    return symbol_end();
  DataRefImpl Sym;
  Sym.d.a = 1;
  Sym.d.b = Rel.Index;
  return symbol_iterator(SymbolRef(Sym, this));
}

uint64_t WasmObjectFile::getRelocationType(DataRefImpl Ref) const {
  const wasm::WasmRelocation &Rel = getWasmRelocation(Ref);
  return Rel.Type;
}

void WasmObjectFile::getRelocationTypeName(
    DataRefImpl Ref, SmallVectorImpl<char> &Result) const {
  const wasm::WasmRelocation &Rel = getWasmRelocation(Ref);
  StringRef Res = "Unknown";

#define WASM_RELOC(name, value)                                                \
  case wasm::name:                                                             \
    Res = #name;                                                               \
    break;

  switch (Rel.Type) {
#include "llvm/BinaryFormat/WasmRelocs.def"
  }

#undef WASM_RELOC

  Result.append(Res.begin(), Res.end());
}

section_iterator WasmObjectFile::section_begin() const {
  DataRefImpl Ref;
  Ref.d.a = 0;
  return section_iterator(SectionRef(Ref, this));
}

section_iterator WasmObjectFile::section_end() const {
  DataRefImpl Ref;
  Ref.d.a = Sections.size();
  return section_iterator(SectionRef(Ref, this));
}

uint8_t WasmObjectFile::getBytesInAddress() const {
  return HasMemory64 ? 8 : 4;
}

StringRef WasmObjectFile::getFileFormatName() const { return "WASM"; }

Triple::ArchType WasmObjectFile::getArch() const {
  return HasMemory64 ? Triple::wasm64 : Triple::wasm32;
}

Expected<SubtargetFeatures> WasmObjectFile::getFeatures() const {
  return SubtargetFeatures();
}

bool WasmObjectFile::isRelocatableObject() const { return HasLinkingSection; }

bool WasmObjectFile::isSharedObject() const { return HasDylinkSection; }

const WasmSection &WasmObjectFile::getWasmSection(DataRefImpl Ref) const {
  assert(Ref.d.a < Sections.size());
  return Sections[Ref.d.a];
}

const WasmSection &
WasmObjectFile::getWasmSection(const SectionRef &Section) const {
  return getWasmSection(Section.getRawDataRefImpl());
}

const wasm::WasmRelocation &
WasmObjectFile::getWasmRelocation(const RelocationRef &Ref) const {
  return getWasmRelocation(Ref.getRawDataRefImpl());
}

const wasm::WasmRelocation &
WasmObjectFile::getWasmRelocation(DataRefImpl Ref) const {
  assert(Ref.d.a < Sections.size());
  const WasmSection &Sec = Sections[Ref.d.a];
  assert(Ref.d.b < Sec.Relocations.size());
  return Sec.Relocations[Ref.d.b];
}

int WasmSectionOrderChecker::getSectionOrder(unsigned ID,
                                             StringRef CustomSectionName) {
  switch (ID) {
  case wasm::WASM_SEC_CUSTOM:
    return StringSwitch<unsigned>(CustomSectionName)
        .Case("dylink", WASM_SEC_ORDER_DYLINK)
        .Case("dylink.0", WASM_SEC_ORDER_DYLINK)
        .Case("linking", WASM_SEC_ORDER_LINKING)
        .StartsWith("reloc.", WASM_SEC_ORDER_RELOC)
        .Case("name", WASM_SEC_ORDER_NAME)
        .Case("producers", WASM_SEC_ORDER_PRODUCERS)
        .Case("target_features", WASM_SEC_ORDER_TARGET_FEATURES)
        .Default(WASM_SEC_ORDER_NONE);
  case wasm::WASM_SEC_TYPE:
    return WASM_SEC_ORDER_TYPE;
  case wasm::WASM_SEC_IMPORT:
    return WASM_SEC_ORDER_IMPORT;
  case wasm::WASM_SEC_FUNCTION:
    return WASM_SEC_ORDER_FUNCTION;
  case wasm::WASM_SEC_TABLE:
    return WASM_SEC_ORDER_TABLE;
  case wasm::WASM_SEC_MEMORY:
    return WASM_SEC_ORDER_MEMORY;
  case wasm::WASM_SEC_GLOBAL:
    return WASM_SEC_ORDER_GLOBAL;
  case wasm::WASM_SEC_EXPORT:
    return WASM_SEC_ORDER_EXPORT;
  case wasm::WASM_SEC_START:
    return WASM_SEC_ORDER_START;
  case wasm::WASM_SEC_ELEM:
    return WASM_SEC_ORDER_ELEM;
  case wasm::WASM_SEC_CODE:
    return WASM_SEC_ORDER_CODE;
  case wasm::WASM_SEC_DATA:
    return WASM_SEC_ORDER_DATA;
  case wasm::WASM_SEC_DATACOUNT:
    return WASM_SEC_ORDER_DATACOUNT;
  case wasm::WASM_SEC_TAG:
    return WASM_SEC_ORDER_TAG;
  default:
    return WASM_SEC_ORDER_NONE;
  }
}

// Represents the edges in a directed graph where any node B reachable from node
// A is not allowed to appear before A in the section ordering, but may appear
// afterward.
int WasmSectionOrderChecker::DisallowedPredecessors
    [WASM_NUM_SEC_ORDERS][WASM_NUM_SEC_ORDERS] = {
        // WASM_SEC_ORDER_NONE
        {},
        // WASM_SEC_ORDER_TYPE
        {WASM_SEC_ORDER_TYPE, WASM_SEC_ORDER_IMPORT},
        // WASM_SEC_ORDER_IMPORT
        {WASM_SEC_ORDER_IMPORT, WASM_SEC_ORDER_FUNCTION},
        // WASM_SEC_ORDER_FUNCTION
        {WASM_SEC_ORDER_FUNCTION, WASM_SEC_ORDER_TABLE},
        // WASM_SEC_ORDER_TABLE
        {WASM_SEC_ORDER_TABLE, WASM_SEC_ORDER_MEMORY},
        // WASM_SEC_ORDER_MEMORY
        {WASM_SEC_ORDER_MEMORY, WASM_SEC_ORDER_TAG},
        // WASM_SEC_ORDER_TAG
        {WASM_SEC_ORDER_TAG, WASM_SEC_ORDER_GLOBAL},
        // WASM_SEC_ORDER_GLOBAL
        {WASM_SEC_ORDER_GLOBAL, WASM_SEC_ORDER_EXPORT},
        // WASM_SEC_ORDER_EXPORT
        {WASM_SEC_ORDER_EXPORT, WASM_SEC_ORDER_START},
        // WASM_SEC_ORDER_START
        {WASM_SEC_ORDER_START, WASM_SEC_ORDER_ELEM},
        // WASM_SEC_ORDER_ELEM
        {WASM_SEC_ORDER_ELEM, WASM_SEC_ORDER_DATACOUNT},
        // WASM_SEC_ORDER_DATACOUNT
        {WASM_SEC_ORDER_DATACOUNT, WASM_SEC_ORDER_CODE},
        // WASM_SEC_ORDER_CODE
        {WASM_SEC_ORDER_CODE, WASM_SEC_ORDER_DATA},
        // WASM_SEC_ORDER_DATA
        {WASM_SEC_ORDER_DATA, WASM_SEC_ORDER_LINKING},

        // Custom Sections
        // WASM_SEC_ORDER_DYLINK
        {WASM_SEC_ORDER_DYLINK, WASM_SEC_ORDER_TYPE},
        // WASM_SEC_ORDER_LINKING
        {WASM_SEC_ORDER_LINKING, WASM_SEC_ORDER_RELOC, WASM_SEC_ORDER_NAME},
        // WASM_SEC_ORDER_RELOC (can be repeated)
        {},
        // WASM_SEC_ORDER_NAME
        {WASM_SEC_ORDER_NAME, WASM_SEC_ORDER_PRODUCERS},
        // WASM_SEC_ORDER_PRODUCERS
        {WASM_SEC_ORDER_PRODUCERS, WASM_SEC_ORDER_TARGET_FEATURES},
        // WASM_SEC_ORDER_TARGET_FEATURES
        {WASM_SEC_ORDER_TARGET_FEATURES}};

bool WasmSectionOrderChecker::isValidSectionOrder(unsigned ID,
                                                  StringRef CustomSectionName) {
  int Order = getSectionOrder(ID, CustomSectionName);
  if (Order == WASM_SEC_ORDER_NONE)
    return true;

  // Disallowed predecessors we need to check for
  SmallVector<int, WASM_NUM_SEC_ORDERS> WorkList;

  // Keep track of completed checks to avoid repeating work
  bool Checked[WASM_NUM_SEC_ORDERS] = {};

  int Curr = Order;
  while (true) {
    // Add new disallowed predecessors to work list
    for (size_t I = 0;; ++I) {
      int Next = DisallowedPredecessors[Curr][I];
      if (Next == WASM_SEC_ORDER_NONE)
        break;
      if (Checked[Next])
        continue;
      WorkList.push_back(Next);
      Checked[Next] = true;
    }

    if (WorkList.empty())
      break;

    // Consider next disallowed predecessor
    Curr = WorkList.pop_back_val();
    if (Seen[Curr])
      return false;
  }

  // Have not seen any disallowed predecessors
  Seen[Order] = true;
  return true;
}
