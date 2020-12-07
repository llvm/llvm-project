//===- WasmObjectFile.cpp - Wasm object file implementation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Triple.h"
#include "llvm/BinaryFormat/Wasm.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/Error.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Object/Wasm.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/ScopedPrinter.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <system_error>

#define DEBUG_TYPE "wasm-object"

using namespace llvm;
using namespace object;

void WasmSymbol::print(raw_ostream &Out) const {
  Out << "Name=" << Info.Name
      << ", Kind=" << toString(wasm::WasmSymbolType(Info.Kind))
      << ", Flags=" << Info.Flags;
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

static uint8_t readUint8(WasmObjectFile::ReadContext &Ctx) {
  if (Ctx.Ptr == Ctx.End)
    report_fatal_error("EOF while reading uint8");
  return *Ctx.Ptr++;
}

static uint32_t readUint32(WasmObjectFile::ReadContext &Ctx) {
  if (Ctx.Ptr + 4 > Ctx.End)
    report_fatal_error("EOF while reading uint32");
  uint32_t Result = support::endian::read32le(Ctx.Ptr);
  Ctx.Ptr += 4;
  return Result;
}

static int32_t readFloat32(WasmObjectFile::ReadContext &Ctx) {
  if (Ctx.Ptr + 4 > Ctx.End)
    report_fatal_error("EOF while reading float64");
  int32_t Result = 0;
  memcpy(&Result, Ctx.Ptr, sizeof(Result));
  Ctx.Ptr += sizeof(Result);
  return Result;
}

static int64_t readFloat64(WasmObjectFile::ReadContext &Ctx) {
  if (Ctx.Ptr + 8 > Ctx.End)
    report_fatal_error("EOF while reading float64");
  int64_t Result = 0;
  memcpy(&Result, Ctx.Ptr, sizeof(Result));
  Ctx.Ptr += sizeof(Result);
  return Result;
}

static uint64_t readULEB128(WasmObjectFile::ReadContext &Ctx) {
  unsigned Count;
  const char *Error = nullptr;
  uint64_t Result = decodeULEB128(Ctx.Ptr, &Count, Ctx.End, &Error);
  if (Error)
    report_fatal_error(Error);
  Ctx.Ptr += Count;
  return Result;
}

static StringRef readString(WasmObjectFile::ReadContext &Ctx) {
  uint32_t StringLen = readULEB128(Ctx);
  if (Ctx.Ptr + StringLen > Ctx.End)
    report_fatal_error("EOF while reading string");
  StringRef Return =
      StringRef(reinterpret_cast<const char *>(Ctx.Ptr), StringLen);
  Ctx.Ptr += StringLen;
  return Return;
}

static int64_t readLEB128(WasmObjectFile::ReadContext &Ctx) {
  unsigned Count;
  const char *Error = nullptr;
  uint64_t Result = decodeSLEB128(Ctx.Ptr, &Count, Ctx.End, &Error);
  if (Error)
    report_fatal_error(Error);
  Ctx.Ptr += Count;
  return Result;
}

static uint8_t readVaruint1(WasmObjectFile::ReadContext &Ctx) {
  int64_t Result = readLEB128(Ctx);
  if (Result > VARUINT1_MAX || Result < 0)
    report_fatal_error("LEB is outside Varuint1 range");
  return Result;
}

static int32_t readVarint32(WasmObjectFile::ReadContext &Ctx) {
  int64_t Result = readLEB128(Ctx);
  if (Result > INT32_MAX || Result < INT32_MIN)
    report_fatal_error("LEB is outside Varint32 range");
  return Result;
}

static uint32_t readVaruint32(WasmObjectFile::ReadContext &Ctx) {
  uint64_t Result = readULEB128(Ctx);
  if (Result > UINT32_MAX)
    report_fatal_error("LEB is outside Varuint32 range");
  return Result;
}

static int64_t readVarint64(WasmObjectFile::ReadContext &Ctx) {
  return readLEB128(Ctx);
}

static uint64_t readVaruint64(WasmObjectFile::ReadContext &Ctx) {
  return readULEB128(Ctx);
}

static uint8_t readOpcode(WasmObjectFile::ReadContext &Ctx) {
  return readUint8(Ctx);
}

static Error readInitExpr(wasm::WasmInitExpr &Expr,
                          WasmObjectFile::ReadContext &Ctx) {
  Expr.Opcode = readOpcode(Ctx);

  switch (Expr.Opcode) {
  case wasm::WASM_OPCODE_I32_CONST:
    Expr.Value.Int32 = readVarint32(Ctx);
    break;
  case wasm::WASM_OPCODE_I64_CONST:
    Expr.Value.Int64 = readVarint64(Ctx);
    break;
  case wasm::WASM_OPCODE_F32_CONST:
    Expr.Value.Float32 = readFloat32(Ctx);
    break;
  case wasm::WASM_OPCODE_F64_CONST:
    Expr.Value.Float64 = readFloat64(Ctx);
    break;
  case wasm::WASM_OPCODE_GLOBAL_GET:
    Expr.Value.Global = readULEB128(Ctx);
    break;
  case wasm::WASM_OPCODE_REF_NULL: {
    wasm::ValType Ty = static_cast<wasm::ValType>(readULEB128(Ctx));
    if (Ty != wasm::ValType::EXTERNREF) {
      return make_error<GenericBinaryError>("Invalid type for ref.null",
                                            object_error::parse_failed);
    }
    break;
  }
  default:
    return make_error<GenericBinaryError>("Invalid opcode in init_expr",
                                          object_error::parse_failed);
  }

  uint8_t EndOpcode = readOpcode(Ctx);
  if (EndOpcode != wasm::WASM_OPCODE_END) {
    return make_error<GenericBinaryError>("Invalid init_expr",
                                          object_error::parse_failed);
  }
  return Error::success();
}

static wasm::WasmLimits readLimits(WasmObjectFile::ReadContext &Ctx) {
  wasm::WasmLimits Result;
  Result.Flags = readVaruint32(Ctx);
  Result.Initial = readVaruint64(Ctx);
  if (Result.Flags & wasm::WASM_LIMITS_FLAG_HAS_MAX)
    Result.Maximum = readVaruint64(Ctx);
  return Result;
}

static wasm::WasmTableType readTableType(WasmObjectFile::ReadContext &Ctx) {
  wasm::WasmTableType TableType;
  TableType.ElemType = readUint8(Ctx);
  TableType.Limits = readLimits(Ctx);
  return TableType;
}

static Error readSection(WasmSection &Section, WasmObjectFile::ReadContext &Ctx,
                         WasmSectionOrderChecker &Checker) {
  Section.Offset = Ctx.Ptr - Ctx.Start;
  Section.Type = readUint8(Ctx);
  LLVM_DEBUG(dbgs() << "readSection type=" << Section.Type << "\n");
  uint32_t Size = readVaruint32(Ctx);
  if (Size == 0)
    return make_error<StringError>("Zero length section",
                                   object_error::parse_failed);
  if (Ctx.Ptr + Size > Ctx.End)
    return make_error<StringError>("Section too large",
                                   object_error::parse_failed);
  if (Section.Type == wasm::WASM_SEC_CUSTOM) {
    WasmObjectFile::ReadContext SectionCtx;
    SectionCtx.Start = Ctx.Ptr;
    SectionCtx.Ptr = Ctx.Ptr;
    SectionCtx.End = Ctx.Ptr + Size;

    Section.Name = readString(SectionCtx);

    uint32_t SectionNameSize = SectionCtx.Ptr - SectionCtx.Start;
    Ctx.Ptr += SectionNameSize;
    Size -= SectionNameSize;
  }

  if (!Checker.isValidSectionOrder(Section.Type, Section.Name)) {
    return make_error<StringError>("Out of order section type: " +
                                       llvm::to_string(Section.Type),
                                   object_error::parse_failed);
  }

  Section.Content = ArrayRef<uint8_t>(Ctx.Ptr, Size);
  Ctx.Ptr += Size;
  return Error::success();
}

WasmObjectFile::WasmObjectFile(MemoryBufferRef Buffer, Error &Err)
    : ObjectFile(Binary::ID_Wasm, Buffer) {
  ErrorAsOutParameter ErrAsOutParam(&Err);
  Header.Magic = getData().substr(0, 4);
  if (Header.Magic != StringRef("\0asm", 4)) {
    Err =
        make_error<StringError>("Bad magic number", object_error::parse_failed);
    return;
  }

  ReadContext Ctx;
  Ctx.Start = getData().bytes_begin();
  Ctx.Ptr = Ctx.Start + 4;
  Ctx.End = Ctx.Start + getData().size();

  if (Ctx.Ptr + 4 > Ctx.End) {
    Err = make_error<StringError>("Missing version number",
                                  object_error::parse_failed);
    return;
  }

  Header.Version = readUint32(Ctx);
  if (Header.Version != wasm::WasmVersion) {
    Err = make_error<StringError>("Bad version number",
                                  object_error::parse_failed);
    return;
  }

  WasmSection Sec;
  WasmSectionOrderChecker Checker;
  while (Ctx.Ptr < Ctx.End) {
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
  case wasm::WASM_SEC_EVENT:
    return parseEventSection(Ctx);
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
        "Invalid section type: " + Twine(Sec.Type), object_error::parse_failed);
  }
}

Error WasmObjectFile::parseDylinkSection(ReadContext &Ctx) {
  // See https://github.com/WebAssembly/tool-conventions/blob/master/DynamicLinking.md
  HasDylinkSection = true;
  DylinkInfo.MemorySize = readVaruint32(Ctx);
  DylinkInfo.MemoryAlignment = readVaruint32(Ctx);
  DylinkInfo.TableSize = readVaruint32(Ctx);
  DylinkInfo.TableAlignment = readVaruint32(Ctx);
  uint32_t Count = readVaruint32(Ctx);
  while (Count--) {
    DylinkInfo.Needed.push_back(readString(Ctx));
  }
  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("dylink section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseNameSection(ReadContext &Ctx) {
  llvm::DenseSet<uint64_t> SeenFunctions;
  llvm::DenseSet<uint64_t> SeenGlobals;
  if (FunctionTypes.size() && !SeenCodeSection) {
    return make_error<GenericBinaryError>("Names must come after code section",
                                          object_error::parse_failed);
  }

  while (Ctx.Ptr < Ctx.End) {
    uint8_t Type = readUint8(Ctx);
    uint32_t Size = readVaruint32(Ctx);
    const uint8_t *SubSectionEnd = Ctx.Ptr + Size;
    switch (Type) {
    case wasm::WASM_NAMES_FUNCTION:
    case wasm::WASM_NAMES_GLOBAL: {
      uint32_t Count = readVaruint32(Ctx);
      while (Count--) {
        uint32_t Index = readVaruint32(Ctx);
        StringRef Name = readString(Ctx);
        if (Type == wasm::WASM_NAMES_FUNCTION) {
          if (!SeenFunctions.insert(Index).second)
            return make_error<GenericBinaryError>(
                "Function named more than once", object_error::parse_failed);
          if (!isValidFunctionIndex(Index) || Name.empty())
            return make_error<GenericBinaryError>("Invalid name entry",
                                                  object_error::parse_failed);

          if (isDefinedFunctionIndex(Index))
            getDefinedFunction(Index).DebugName = Name;
        } else {
          if (!SeenGlobals.insert(Index).second)
            return make_error<GenericBinaryError>("Global named more than once",
                                                  object_error::parse_failed);
          if (!isValidGlobalIndex(Index) || Name.empty())
            return make_error<GenericBinaryError>("Invalid name entry",
                                                  object_error::parse_failed);
        }
        wasm::NameType T = Type == wasm::WASM_NAMES_FUNCTION
                               ? wasm::NameType::FUNCTION
                               : wasm::NameType::GLOBAL;
        DebugNames.push_back(wasm::WasmDebugName{T, Index, Name});
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
          "Name sub-section ended prematurely", object_error::parse_failed);
  }

  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("Name section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseLinkingSection(ReadContext &Ctx) {
  HasLinkingSection = true;
  if (FunctionTypes.size() && !SeenCodeSection) {
    return make_error<GenericBinaryError>(
        "Linking data must come after code section",
        object_error::parse_failed);
  }

  LinkingData.Version = readVaruint32(Ctx);
  if (LinkingData.Version != wasm::WasmMetadataVersion) {
    return make_error<GenericBinaryError>(
        "Unexpected metadata version: " + Twine(LinkingData.Version) +
            " (Expected: " + Twine(wasm::WasmMetadataVersion) + ")",
        object_error::parse_failed);
  }

  const uint8_t *OrigEnd = Ctx.End;
  while (Ctx.Ptr < OrigEnd) {
    Ctx.End = OrigEnd;
    uint8_t Type = readUint8(Ctx);
    uint32_t Size = readVaruint32(Ctx);
    LLVM_DEBUG(dbgs() << "readSubsection type=" << int(Type) << " size=" << Size
                      << "\n");
    Ctx.End = Ctx.Ptr + Size;
    switch (Type) {
    case wasm::WASM_SYMBOL_TABLE:
      if (Error Err = parseLinkingSectionSymtab(Ctx))
        return Err;
      break;
    case wasm::WASM_SEGMENT_INFO: {
      uint32_t Count = readVaruint32(Ctx);
      if (Count > DataSegments.size())
        return make_error<GenericBinaryError>("Too many segment names",
                                              object_error::parse_failed);
      for (uint32_t I = 0; I < Count; I++) {
        DataSegments[I].Data.Name = readString(Ctx);
        DataSegments[I].Data.Alignment = readVaruint32(Ctx);
        DataSegments[I].Data.LinkerFlags = readVaruint32(Ctx);
      }
      break;
    }
    case wasm::WASM_INIT_FUNCS: {
      uint32_t Count = readVaruint32(Ctx);
      LinkingData.InitFunctions.reserve(Count);
      for (uint32_t I = 0; I < Count; I++) {
        wasm::WasmInitFunc Init;
        Init.Priority = readVaruint32(Ctx);
        Init.Symbol = readVaruint32(Ctx);
        if (!isValidFunctionSymbol(Init.Symbol))
          return make_error<GenericBinaryError>("Invalid function symbol: " +
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
          "Linking sub-section ended prematurely", object_error::parse_failed);
  }
  if (Ctx.Ptr != OrigEnd)
    return make_error<GenericBinaryError>("Linking section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseLinkingSectionSymtab(ReadContext &Ctx) {
  uint32_t Count = readVaruint32(Ctx);
  LinkingData.SymbolTable.reserve(Count);
  Symbols.reserve(Count);
  StringSet<> SymbolNames;

  std::vector<wasm::WasmImport *> ImportedGlobals;
  std::vector<wasm::WasmImport *> ImportedFunctions;
  std::vector<wasm::WasmImport *> ImportedEvents;
  std::vector<wasm::WasmImport *> ImportedTables;
  ImportedGlobals.reserve(Imports.size());
  ImportedFunctions.reserve(Imports.size());
  ImportedEvents.reserve(Imports.size());
  ImportedTables.reserve(Imports.size());
  for (auto &I : Imports) {
    if (I.Kind == wasm::WASM_EXTERNAL_FUNCTION)
      ImportedFunctions.emplace_back(&I);
    else if (I.Kind == wasm::WASM_EXTERNAL_GLOBAL)
      ImportedGlobals.emplace_back(&I);
    else if (I.Kind == wasm::WASM_EXTERNAL_EVENT)
      ImportedEvents.emplace_back(&I);
    else if (I.Kind == wasm::WASM_EXTERNAL_TABLE)
      ImportedTables.emplace_back(&I);
  }

  while (Count--) {
    wasm::WasmSymbolInfo Info;
    const wasm::WasmSignature *Signature = nullptr;
    const wasm::WasmGlobalType *GlobalType = nullptr;
    uint8_t TableType = 0;
    const wasm::WasmEventType *EventType = nullptr;

    Info.Kind = readUint8(Ctx);
    Info.Flags = readVaruint32(Ctx);
    bool IsDefined = (Info.Flags & wasm::WASM_SYMBOL_UNDEFINED) == 0;

    switch (Info.Kind) {
    case wasm::WASM_SYMBOL_TYPE_FUNCTION:
      Info.ElementIndex = readVaruint32(Ctx);
      if (!isValidFunctionIndex(Info.ElementIndex) ||
          IsDefined != isDefinedFunctionIndex(Info.ElementIndex))
        return make_error<GenericBinaryError>("invalid function symbol index",
                                              object_error::parse_failed);
      if (IsDefined) {
        Info.Name = readString(Ctx);
        unsigned FuncIndex = Info.ElementIndex - NumImportedFunctions;
        Signature = &Signatures[FunctionTypes[FuncIndex]];
        wasm::WasmFunction &Function = Functions[FuncIndex];
        if (Function.SymbolName.empty())
          Function.SymbolName = Info.Name;
      } else {
        wasm::WasmImport &Import = *ImportedFunctions[Info.ElementIndex];
        if ((Info.Flags & wasm::WASM_SYMBOL_EXPLICIT_NAME) != 0) {
          Info.Name = readString(Ctx);
          Info.ImportName = Import.Field;
        } else {
          Info.Name = Import.Field;
        }
        Signature = &Signatures[Import.SigIndex];
        if (!Import.Module.empty()) {
          Info.ImportModule = Import.Module;
        }
      }
      break;

    case wasm::WASM_SYMBOL_TYPE_GLOBAL:
      Info.ElementIndex = readVaruint32(Ctx);
      if (!isValidGlobalIndex(Info.ElementIndex) ||
          IsDefined != isDefinedGlobalIndex(Info.ElementIndex))
        return make_error<GenericBinaryError>("invalid global symbol index",
                                              object_error::parse_failed);
      if (!IsDefined && (Info.Flags & wasm::WASM_SYMBOL_BINDING_MASK) ==
                            wasm::WASM_SYMBOL_BINDING_WEAK)
        return make_error<GenericBinaryError>("undefined weak global symbol",
                                              object_error::parse_failed);
      if (IsDefined) {
        Info.Name = readString(Ctx);
        unsigned GlobalIndex = Info.ElementIndex - NumImportedGlobals;
        wasm::WasmGlobal &Global = Globals[GlobalIndex];
        GlobalType = &Global.Type;
        if (Global.SymbolName.empty())
          Global.SymbolName = Info.Name;
      } else {
        wasm::WasmImport &Import = *ImportedGlobals[Info.ElementIndex];
        if ((Info.Flags & wasm::WASM_SYMBOL_EXPLICIT_NAME) != 0) {
          Info.Name = readString(Ctx);
          Info.ImportName = Import.Field;
        } else {
          Info.Name = Import.Field;
        }
        GlobalType = &Import.Global;
        if (!Import.Module.empty()) {
          Info.ImportModule = Import.Module;
        }
      }
      break;

    case wasm::WASM_SYMBOL_TYPE_TABLE:
      Info.ElementIndex = readVaruint32(Ctx);
      if (!isValidTableIndex(Info.ElementIndex) ||
          IsDefined != isDefinedTableIndex(Info.ElementIndex))
        return make_error<GenericBinaryError>("invalid table symbol index",
                                              object_error::parse_failed);
      if (!IsDefined && (Info.Flags & wasm::WASM_SYMBOL_BINDING_MASK) ==
                            wasm::WASM_SYMBOL_BINDING_WEAK)
        return make_error<GenericBinaryError>("undefined weak table symbol",
                                              object_error::parse_failed);
      if (IsDefined) {
        Info.Name = readString(Ctx);
        unsigned TableIndex = Info.ElementIndex - NumImportedTables;
        wasm::WasmTable &Table = Tables[TableIndex];
        TableType = Table.Type.ElemType;
        if (Table.SymbolName.empty())
          Table.SymbolName = Info.Name;
      } else {
        wasm::WasmImport &Import = *ImportedTables[Info.ElementIndex];
        if ((Info.Flags & wasm::WASM_SYMBOL_EXPLICIT_NAME) != 0) {
          Info.Name = readString(Ctx);
          Info.ImportName = Import.Field;
        } else {
          Info.Name = Import.Field;
        }
        TableType = Import.Table.ElemType;
        // FIXME: Parse limits here too.
        if (!Import.Module.empty()) {
          Info.ImportModule = Import.Module;
        }
      }
      break;

    case wasm::WASM_SYMBOL_TYPE_DATA:
      Info.Name = readString(Ctx);
      if (IsDefined) {
        auto Index = readVaruint32(Ctx);
        if (Index >= DataSegments.size())
          return make_error<GenericBinaryError>("invalid data symbol index",
                                                object_error::parse_failed);
        auto Offset = readVaruint64(Ctx);
        auto Size = readVaruint64(Ctx);
        if (Offset + Size > DataSegments[Index].Data.Content.size())
          return make_error<GenericBinaryError>("invalid data symbol offset",
                                                object_error::parse_failed);
        Info.DataRef = wasm::WasmDataReference{Index, Offset, Size};
      }
      break;

    case wasm::WASM_SYMBOL_TYPE_SECTION: {
      if ((Info.Flags & wasm::WASM_SYMBOL_BINDING_MASK) !=
          wasm::WASM_SYMBOL_BINDING_LOCAL)
        return make_error<GenericBinaryError>(
            "Section symbols must have local binding",
            object_error::parse_failed);
      Info.ElementIndex = readVaruint32(Ctx);
      // Use somewhat unique section name as symbol name.
      StringRef SectionName = Sections[Info.ElementIndex].Name;
      Info.Name = SectionName;
      break;
    }

    case wasm::WASM_SYMBOL_TYPE_EVENT: {
      Info.ElementIndex = readVaruint32(Ctx);
      if (!isValidEventIndex(Info.ElementIndex) ||
          IsDefined != isDefinedEventIndex(Info.ElementIndex))
        return make_error<GenericBinaryError>("invalid event symbol index",
                                              object_error::parse_failed);
      if (!IsDefined && (Info.Flags & wasm::WASM_SYMBOL_BINDING_MASK) ==
                            wasm::WASM_SYMBOL_BINDING_WEAK)
        return make_error<GenericBinaryError>("undefined weak global symbol",
                                              object_error::parse_failed);
      if (IsDefined) {
        Info.Name = readString(Ctx);
        unsigned EventIndex = Info.ElementIndex - NumImportedEvents;
        wasm::WasmEvent &Event = Events[EventIndex];
        Signature = &Signatures[Event.Type.SigIndex];
        EventType = &Event.Type;
        if (Event.SymbolName.empty())
          Event.SymbolName = Info.Name;

      } else {
        wasm::WasmImport &Import = *ImportedEvents[Info.ElementIndex];
        if ((Info.Flags & wasm::WASM_SYMBOL_EXPLICIT_NAME) != 0) {
          Info.Name = readString(Ctx);
          Info.ImportName = Import.Field;
        } else {
          Info.Name = Import.Field;
        }
        EventType = &Import.Event;
        Signature = &Signatures[EventType->SigIndex];
        if (!Import.Module.empty()) {
          Info.ImportModule = Import.Module;
        }
      }
      break;
    }

    default:
      return make_error<GenericBinaryError>("Invalid symbol type",
                                            object_error::parse_failed);
    }

    if ((Info.Flags & wasm::WASM_SYMBOL_BINDING_MASK) !=
            wasm::WASM_SYMBOL_BINDING_LOCAL &&
        !SymbolNames.insert(Info.Name).second)
      return make_error<GenericBinaryError>("Duplicate symbol name " +
                                                Twine(Info.Name),
                                            object_error::parse_failed);
    LinkingData.SymbolTable.emplace_back(Info);
    Symbols.emplace_back(LinkingData.SymbolTable.back(), GlobalType, TableType,
                         EventType, Signature);
    LLVM_DEBUG(dbgs() << "Adding symbol: " << Symbols.back() << "\n");
  }

  return Error::success();
}

Error WasmObjectFile::parseLinkingSectionComdat(ReadContext &Ctx) {
  uint32_t ComdatCount = readVaruint32(Ctx);
  StringSet<> ComdatSet;
  for (unsigned ComdatIndex = 0; ComdatIndex < ComdatCount; ++ComdatIndex) {
    StringRef Name = readString(Ctx);
    if (Name.empty() || !ComdatSet.insert(Name).second)
      return make_error<GenericBinaryError>("Bad/duplicate COMDAT name " +
                                                Twine(Name),
                                            object_error::parse_failed);
    LinkingData.Comdats.emplace_back(Name);
    uint32_t Flags = readVaruint32(Ctx);
    if (Flags != 0)
      return make_error<GenericBinaryError>("Unsupported COMDAT flags",
                                            object_error::parse_failed);

    uint32_t EntryCount = readVaruint32(Ctx);
    while (EntryCount--) {
      unsigned Kind = readVaruint32(Ctx);
      unsigned Index = readVaruint32(Ctx);
      switch (Kind) {
      default:
        return make_error<GenericBinaryError>("Invalid COMDAT entry type",
                                              object_error::parse_failed);
      case wasm::WASM_COMDAT_DATA:
        if (Index >= DataSegments.size())
          return make_error<GenericBinaryError>(
              "COMDAT data index out of range", object_error::parse_failed);
        if (DataSegments[Index].Data.Comdat != UINT32_MAX)
          return make_error<GenericBinaryError>("Data segment in two COMDATs",
                                                object_error::parse_failed);
        DataSegments[Index].Data.Comdat = ComdatIndex;
        break;
      case wasm::WASM_COMDAT_FUNCTION:
        if (!isDefinedFunctionIndex(Index))
          return make_error<GenericBinaryError>(
              "COMDAT function index out of range", object_error::parse_failed);
        if (getDefinedFunction(Index).Comdat != UINT32_MAX)
          return make_error<GenericBinaryError>("Function in two COMDATs",
                                                object_error::parse_failed);
        getDefinedFunction(Index).Comdat = ComdatIndex;
        break;
      case wasm::WASM_COMDAT_SECTION:
        if (Index >= Sections.size())
          return make_error<GenericBinaryError>(
              "COMDAT section index out of range", object_error::parse_failed);
        if (Sections[Index].Type != wasm::WASM_SEC_CUSTOM)
          return make_error<GenericBinaryError>(
              "Non-custom section in a COMDAT", object_error::parse_failed);
        Sections[Index].Comdat = ComdatIndex;
        break;
      }
    }
  }
  return Error::success();
}

Error WasmObjectFile::parseProducersSection(ReadContext &Ctx) {
  llvm::SmallSet<StringRef, 3> FieldsSeen;
  uint32_t Fields = readVaruint32(Ctx);
  for (size_t I = 0; I < Fields; ++I) {
    StringRef FieldName = readString(Ctx);
    if (!FieldsSeen.insert(FieldName).second)
      return make_error<GenericBinaryError>(
          "Producers section does not have unique fields",
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
          "Producers section field is not named one of language, processed-by, "
          "or sdk",
          object_error::parse_failed);
    }
    uint32_t ValueCount = readVaruint32(Ctx);
    llvm::SmallSet<StringRef, 8> ProducersSeen;
    for (size_t J = 0; J < ValueCount; ++J) {
      StringRef Name = readString(Ctx);
      StringRef Version = readString(Ctx);
      if (!ProducersSeen.insert(Name).second) {
        return make_error<GenericBinaryError>(
            "Producers section contains repeated producer",
            object_error::parse_failed);
      }
      ProducerVec->emplace_back(std::string(Name), std::string(Version));
    }
  }
  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("Producers section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseTargetFeaturesSection(ReadContext &Ctx) {
  llvm::SmallSet<std::string, 8> FeaturesSeen;
  uint32_t FeatureCount = readVaruint32(Ctx);
  for (size_t I = 0; I < FeatureCount; ++I) {
    wasm::WasmFeatureEntry Feature;
    Feature.Prefix = readUint8(Ctx);
    switch (Feature.Prefix) {
    case wasm::WASM_FEATURE_PREFIX_USED:
    case wasm::WASM_FEATURE_PREFIX_REQUIRED:
    case wasm::WASM_FEATURE_PREFIX_DISALLOWED:
      break;
    default:
      return make_error<GenericBinaryError>("Unknown feature policy prefix",
                                            object_error::parse_failed);
    }
    Feature.Name = std::string(readString(Ctx));
    if (!FeaturesSeen.insert(Feature.Name).second)
      return make_error<GenericBinaryError>(
          "Target features section contains repeated feature \"" +
              Feature.Name + "\"",
          object_error::parse_failed);
    TargetFeatures.push_back(Feature);
  }
  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>(
        "Target features section ended prematurely",
        object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseRelocSection(StringRef Name, ReadContext &Ctx) {
  uint32_t SectionIndex = readVaruint32(Ctx);
  if (SectionIndex >= Sections.size())
    return make_error<GenericBinaryError>("Invalid section index",
                                          object_error::parse_failed);
  WasmSection &Section = Sections[SectionIndex];
  uint32_t RelocCount = readVaruint32(Ctx);
  uint32_t EndOffset = Section.Content.size();
  uint32_t PreviousOffset = 0;
  while (RelocCount--) {
    wasm::WasmRelocation Reloc = {};
    Reloc.Type = readVaruint32(Ctx);
    Reloc.Offset = readVaruint32(Ctx);
    if (Reloc.Offset < PreviousOffset)
      return make_error<GenericBinaryError>("Relocations not in offset order",
                                            object_error::parse_failed);
    PreviousOffset = Reloc.Offset;
    Reloc.Index = readVaruint32(Ctx);
    switch (Reloc.Type) {
    case wasm::R_WASM_FUNCTION_INDEX_LEB:
    case wasm::R_WASM_TABLE_INDEX_SLEB:
    case wasm::R_WASM_TABLE_INDEX_SLEB64:
    case wasm::R_WASM_TABLE_INDEX_I32:
    case wasm::R_WASM_TABLE_INDEX_I64:
    case wasm::R_WASM_TABLE_INDEX_REL_SLEB:
      if (!isValidFunctionSymbol(Reloc.Index))
        return make_error<GenericBinaryError>("Bad relocation function index",
                                              object_error::parse_failed);
      break;
    case wasm::R_WASM_TABLE_NUMBER_LEB:
      if (!isValidTableSymbol(Reloc.Index))
        return make_error<GenericBinaryError>("Bad relocation table index",
                                              object_error::parse_failed);
      break;
    case wasm::R_WASM_TYPE_INDEX_LEB:
      if (Reloc.Index >= Signatures.size())
        return make_error<GenericBinaryError>("Bad relocation type index",
                                              object_error::parse_failed);
      break;
    case wasm::R_WASM_GLOBAL_INDEX_LEB:
      // R_WASM_GLOBAL_INDEX_LEB are can be used against function and data
      // symbols to refer to their GOT entries.
      if (!isValidGlobalSymbol(Reloc.Index) &&
          !isValidDataSymbol(Reloc.Index) &&
          !isValidFunctionSymbol(Reloc.Index))
        return make_error<GenericBinaryError>("Bad relocation global index",
                                              object_error::parse_failed);
      break;
    case wasm::R_WASM_GLOBAL_INDEX_I32:
      if (!isValidGlobalSymbol(Reloc.Index))
        return make_error<GenericBinaryError>("Bad relocation global index",
                                              object_error::parse_failed);
      break;
    case wasm::R_WASM_EVENT_INDEX_LEB:
      if (!isValidEventSymbol(Reloc.Index))
        return make_error<GenericBinaryError>("Bad relocation event index",
                                              object_error::parse_failed);
      break;
    case wasm::R_WASM_MEMORY_ADDR_LEB:
    case wasm::R_WASM_MEMORY_ADDR_SLEB:
    case wasm::R_WASM_MEMORY_ADDR_I32:
    case wasm::R_WASM_MEMORY_ADDR_REL_SLEB:
    case wasm::R_WASM_MEMORY_ADDR_TLS_SLEB:
      if (!isValidDataSymbol(Reloc.Index))
        return make_error<GenericBinaryError>("Bad relocation data index",
                                              object_error::parse_failed);
      Reloc.Addend = readVarint32(Ctx);
      break;
    case wasm::R_WASM_MEMORY_ADDR_LEB64:
    case wasm::R_WASM_MEMORY_ADDR_SLEB64:
    case wasm::R_WASM_MEMORY_ADDR_I64:
    case wasm::R_WASM_MEMORY_ADDR_REL_SLEB64:
      if (!isValidDataSymbol(Reloc.Index))
        return make_error<GenericBinaryError>("Bad relocation data index",
                                              object_error::parse_failed);
      Reloc.Addend = readVarint64(Ctx);
      break;
    case wasm::R_WASM_FUNCTION_OFFSET_I32:
      if (!isValidFunctionSymbol(Reloc.Index))
        return make_error<GenericBinaryError>("Bad relocation function index",
                                              object_error::parse_failed);
      Reloc.Addend = readVarint32(Ctx);
      break;
    case wasm::R_WASM_FUNCTION_OFFSET_I64:
      if (!isValidFunctionSymbol(Reloc.Index))
        return make_error<GenericBinaryError>("Bad relocation function index",
                                              object_error::parse_failed);
      Reloc.Addend = readVarint64(Ctx);
      break;
    case wasm::R_WASM_SECTION_OFFSET_I32:
      if (!isValidSectionSymbol(Reloc.Index))
        return make_error<GenericBinaryError>("Bad relocation section index",
                                              object_error::parse_failed);
      Reloc.Addend = readVarint32(Ctx);
      break;
    default:
      return make_error<GenericBinaryError>("Bad relocation type: " +
                                                Twine(Reloc.Type),
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
        Reloc.Type == wasm::R_WASM_SECTION_OFFSET_I32 ||
        Reloc.Type == wasm::R_WASM_FUNCTION_OFFSET_I32 ||
        Reloc.Type == wasm::R_WASM_GLOBAL_INDEX_I32)
      Size = 4;
    if (Reloc.Type == wasm::R_WASM_TABLE_INDEX_I64 ||
        Reloc.Type == wasm::R_WASM_MEMORY_ADDR_I64 ||
        Reloc.Type == wasm::R_WASM_FUNCTION_OFFSET_I64)
      Size = 8;
    if (Reloc.Offset + Size > EndOffset)
      return make_error<GenericBinaryError>("Bad relocation offset",
                                            object_error::parse_failed);

    Section.Relocations.push_back(Reloc);
  }
  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("Reloc section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseCustomSection(WasmSection &Sec, ReadContext &Ctx) {
  if (Sec.Name == "dylink") {
    if (Error Err = parseDylinkSection(Ctx))
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
  } else if (Sec.Name.startswith("reloc.")) {
    if (Error Err = parseRelocSection(Sec.Name, Ctx))
      return Err;
  }
  return Error::success();
}

Error WasmObjectFile::parseTypeSection(ReadContext &Ctx) {
  uint32_t Count = readVaruint32(Ctx);
  Signatures.reserve(Count);
  while (Count--) {
    wasm::WasmSignature Sig;
    uint8_t Form = readUint8(Ctx);
    if (Form != wasm::WASM_TYPE_FUNC) {
      return make_error<GenericBinaryError>("Invalid signature type",
                                            object_error::parse_failed);
    }
    uint32_t ParamCount = readVaruint32(Ctx);
    Sig.Params.reserve(ParamCount);
    while (ParamCount--) {
      uint32_t ParamType = readUint8(Ctx);
      Sig.Params.push_back(wasm::ValType(ParamType));
    }
    uint32_t ReturnCount = readVaruint32(Ctx);
    while (ReturnCount--) {
      uint32_t ReturnType = readUint8(Ctx);
      Sig.Returns.push_back(wasm::ValType(ReturnType));
    }
    Signatures.push_back(std::move(Sig));
  }
  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("Type section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseImportSection(ReadContext &Ctx) {
  uint32_t Count = readVaruint32(Ctx);
  Imports.reserve(Count);
  for (uint32_t I = 0; I < Count; I++) {
    wasm::WasmImport Im;
    Im.Module = readString(Ctx);
    Im.Field = readString(Ctx);
    Im.Kind = readUint8(Ctx);
    switch (Im.Kind) {
    case wasm::WASM_EXTERNAL_FUNCTION:
      NumImportedFunctions++;
      Im.SigIndex = readVaruint32(Ctx);
      break;
    case wasm::WASM_EXTERNAL_GLOBAL:
      NumImportedGlobals++;
      Im.Global.Type = readUint8(Ctx);
      Im.Global.Mutable = readVaruint1(Ctx);
      break;
    case wasm::WASM_EXTERNAL_MEMORY:
      Im.Memory = readLimits(Ctx);
      if (Im.Memory.Flags & wasm::WASM_LIMITS_FLAG_IS_64)
        HasMemory64 = true;
      break;
    case wasm::WASM_EXTERNAL_TABLE: {
      Im.Table = readTableType(Ctx);
      NumImportedTables++;
      auto ElemType = Im.Table.ElemType;
      if (ElemType != wasm::WASM_TYPE_FUNCREF &&
          ElemType != wasm::WASM_TYPE_EXTERNREF)
        return make_error<GenericBinaryError>("Invalid table element type",
                                              object_error::parse_failed);
      break;
    }
    case wasm::WASM_EXTERNAL_EVENT:
      NumImportedEvents++;
      Im.Event.Attribute = readVarint32(Ctx);
      Im.Event.SigIndex = readVarint32(Ctx);
      break;
    default:
      return make_error<GenericBinaryError>("Unexpected import kind",
                                            object_error::parse_failed);
    }
    Imports.push_back(Im);
  }
  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("Import section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseFunctionSection(ReadContext &Ctx) {
  uint32_t Count = readVaruint32(Ctx);
  FunctionTypes.reserve(Count);
  Functions.resize(Count);
  uint32_t NumTypes = Signatures.size();
  while (Count--) {
    uint32_t Type = readVaruint32(Ctx);
    if (Type >= NumTypes)
      return make_error<GenericBinaryError>("Invalid function type",
                                            object_error::parse_failed);
    FunctionTypes.push_back(Type);
  }
  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("Function section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseTableSection(ReadContext &Ctx) {
  TableSection = Sections.size();
  uint32_t Count = readVaruint32(Ctx);
  Tables.reserve(Count);
  while (Count--) {
    wasm::WasmTable T;
    T.Type = readTableType(Ctx);
    T.Index = NumImportedTables + Tables.size();
    Tables.push_back(T);
    auto ElemType = Tables.back().Type.ElemType;
    if (ElemType != wasm::WASM_TYPE_FUNCREF &&
        ElemType != wasm::WASM_TYPE_EXTERNREF) {
      return make_error<GenericBinaryError>("Invalid table element type",
                                            object_error::parse_failed);
    }
  }
  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("Table section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseMemorySection(ReadContext &Ctx) {
  uint32_t Count = readVaruint32(Ctx);
  Memories.reserve(Count);
  while (Count--) {
    auto Limits = readLimits(Ctx);
    if (Limits.Flags & wasm::WASM_LIMITS_FLAG_IS_64)
      HasMemory64 = true;
    Memories.push_back(Limits);
  }
  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("Memory section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseEventSection(ReadContext &Ctx) {
  EventSection = Sections.size();
  uint32_t Count = readVarint32(Ctx);
  Events.reserve(Count);
  while (Count--) {
    wasm::WasmEvent Event;
    Event.Index = NumImportedEvents + Events.size();
    Event.Type.Attribute = readVaruint32(Ctx);
    Event.Type.SigIndex = readVarint32(Ctx);
    Events.push_back(Event);
  }

  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("Event section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseGlobalSection(ReadContext &Ctx) {
  GlobalSection = Sections.size();
  uint32_t Count = readVaruint32(Ctx);
  Globals.reserve(Count);
  while (Count--) {
    wasm::WasmGlobal Global;
    Global.Index = NumImportedGlobals + Globals.size();
    Global.Type.Type = readUint8(Ctx);
    Global.Type.Mutable = readVaruint1(Ctx);
    if (Error Err = readInitExpr(Global.InitExpr, Ctx))
      return Err;
    Globals.push_back(Global);
  }
  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("Global section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseExportSection(ReadContext &Ctx) {
  uint32_t Count = readVaruint32(Ctx);
  Exports.reserve(Count);
  for (uint32_t I = 0; I < Count; I++) {
    wasm::WasmExport Ex;
    Ex.Name = readString(Ctx);
    Ex.Kind = readUint8(Ctx);
    Ex.Index = readVaruint32(Ctx);
    switch (Ex.Kind) {
    case wasm::WASM_EXTERNAL_FUNCTION:

      if (!isDefinedFunctionIndex(Ex.Index))
        return make_error<GenericBinaryError>("Invalid function export",
                                              object_error::parse_failed);
      getDefinedFunction(Ex.Index).ExportName = Ex.Name;
      break;
    case wasm::WASM_EXTERNAL_GLOBAL:
      if (!isValidGlobalIndex(Ex.Index))
        return make_error<GenericBinaryError>("Invalid global export",
                                              object_error::parse_failed);
      break;
    case wasm::WASM_EXTERNAL_EVENT:
      if (!isValidEventIndex(Ex.Index))
        return make_error<GenericBinaryError>("Invalid event export",
                                              object_error::parse_failed);
      break;
    case wasm::WASM_EXTERNAL_MEMORY:
    case wasm::WASM_EXTERNAL_TABLE:
      break;
    default:
      return make_error<GenericBinaryError>("Unexpected export kind",
                                            object_error::parse_failed);
    }
    Exports.push_back(Ex);
  }
  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("Export section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

bool WasmObjectFile::isValidFunctionIndex(uint32_t Index) const {
  return Index < NumImportedFunctions + FunctionTypes.size();
}

bool WasmObjectFile::isDefinedFunctionIndex(uint32_t Index) const {
  return Index >= NumImportedFunctions && isValidFunctionIndex(Index);
}

bool WasmObjectFile::isValidGlobalIndex(uint32_t Index) const {
  return Index < NumImportedGlobals + Globals.size();
}

bool WasmObjectFile::isValidTableIndex(uint32_t Index) const {
  return Index < NumImportedTables + Tables.size();
}

bool WasmObjectFile::isDefinedGlobalIndex(uint32_t Index) const {
  return Index >= NumImportedGlobals && isValidGlobalIndex(Index);
}

bool WasmObjectFile::isDefinedTableIndex(uint32_t Index) const {
  return Index >= NumImportedTables && isValidTableIndex(Index);
}

bool WasmObjectFile::isValidEventIndex(uint32_t Index) const {
  return Index < NumImportedEvents + Events.size();
}

bool WasmObjectFile::isDefinedEventIndex(uint32_t Index) const {
  return Index >= NumImportedEvents && isValidEventIndex(Index);
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

bool WasmObjectFile::isValidEventSymbol(uint32_t Index) const {
  return Index < Symbols.size() && Symbols[Index].isTypeEvent();
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

wasm::WasmGlobal &WasmObjectFile::getDefinedGlobal(uint32_t Index) {
  assert(isDefinedGlobalIndex(Index));
  return Globals[Index - NumImportedGlobals];
}

wasm::WasmEvent &WasmObjectFile::getDefinedEvent(uint32_t Index) {
  assert(isDefinedEventIndex(Index));
  return Events[Index - NumImportedEvents];
}

Error WasmObjectFile::parseStartSection(ReadContext &Ctx) {
  StartFunction = readVaruint32(Ctx);
  if (!isValidFunctionIndex(StartFunction))
    return make_error<GenericBinaryError>("Invalid start function",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseCodeSection(ReadContext &Ctx) {
  SeenCodeSection = true;
  CodeSection = Sections.size();
  uint32_t FunctionCount = readVaruint32(Ctx);
  if (FunctionCount != FunctionTypes.size()) {
    return make_error<GenericBinaryError>("Invalid function count",
                                          object_error::parse_failed);
  }

  for (uint32_t i = 0; i < FunctionCount; i++) {
    wasm::WasmFunction& Function = Functions[i];
    const uint8_t *FunctionStart = Ctx.Ptr;
    uint32_t Size = readVaruint32(Ctx);
    const uint8_t *FunctionEnd = Ctx.Ptr + Size;

    Function.CodeOffset = Ctx.Ptr - FunctionStart;
    Function.Index = NumImportedFunctions + i;
    Function.CodeSectionOffset = FunctionStart - Ctx.Start;
    Function.Size = FunctionEnd - FunctionStart;

    uint32_t NumLocalDecls = readVaruint32(Ctx);
    Function.Locals.reserve(NumLocalDecls);
    while (NumLocalDecls--) {
      wasm::WasmLocalDecl Decl;
      Decl.Count = readVaruint32(Ctx);
      Decl.Type = readUint8(Ctx);
      Function.Locals.push_back(Decl);
    }

    uint32_t BodySize = FunctionEnd - Ctx.Ptr;
    Function.Body = ArrayRef<uint8_t>(Ctx.Ptr, BodySize);
    // This will be set later when reading in the linking metadata section.
    Function.Comdat = UINT32_MAX;
    Ctx.Ptr += BodySize;
    assert(Ctx.Ptr == FunctionEnd);
  }
  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("Code section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseElemSection(ReadContext &Ctx) {
  uint32_t Count = readVaruint32(Ctx);
  ElemSegments.reserve(Count);
  while (Count--) {
    wasm::WasmElemSegment Segment;
    Segment.TableIndex = readVaruint32(Ctx);
    if (Segment.TableIndex != 0) {
      return make_error<GenericBinaryError>("Invalid TableIndex",
                                            object_error::parse_failed);
    }
    if (Error Err = readInitExpr(Segment.Offset, Ctx))
      return Err;
    uint32_t NumElems = readVaruint32(Ctx);
    while (NumElems--) {
      Segment.Functions.push_back(readVaruint32(Ctx));
    }
    ElemSegments.push_back(Segment);
  }
  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("Elem section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseDataSection(ReadContext &Ctx) {
  DataSection = Sections.size();
  uint32_t Count = readVaruint32(Ctx);
  if (DataCount && Count != DataCount.getValue())
    return make_error<GenericBinaryError>(
        "Number of data segments does not match DataCount section");
  DataSegments.reserve(Count);
  while (Count--) {
    WasmSegment Segment;
    Segment.Data.InitFlags = readVaruint32(Ctx);
    Segment.Data.MemoryIndex = (Segment.Data.InitFlags & wasm::WASM_SEGMENT_HAS_MEMINDEX)
                               ? readVaruint32(Ctx) : 0;
    if ((Segment.Data.InitFlags & wasm::WASM_SEGMENT_IS_PASSIVE) == 0) {
      if (Error Err = readInitExpr(Segment.Data.Offset, Ctx))
        return Err;
    } else {
      Segment.Data.Offset.Opcode = wasm::WASM_OPCODE_I32_CONST;
      Segment.Data.Offset.Value.Int32 = 0;
    }
    uint32_t Size = readVaruint32(Ctx);
    if (Size > (size_t)(Ctx.End - Ctx.Ptr))
      return make_error<GenericBinaryError>("Invalid segment size",
                                            object_error::parse_failed);
    Segment.Data.Content = ArrayRef<uint8_t>(Ctx.Ptr, Size);
    // The rest of these Data fields are set later, when reading in the linking
    // metadata section.
    Segment.Data.Alignment = 0;
    Segment.Data.LinkerFlags = 0;
    Segment.Data.Comdat = UINT32_MAX;
    Segment.SectionOffset = Ctx.Ptr - Ctx.Start;
    Ctx.Ptr += Size;
    DataSegments.push_back(Segment);
  }
  if (Ctx.Ptr != Ctx.End)
    return make_error<GenericBinaryError>("Data section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseDataCountSection(ReadContext &Ctx) {
  DataCount = readVaruint32(Ctx);
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
  if (Sym.Info.Kind == wasm::WASM_SYMBOL_TYPE_FUNCTION &&
      isDefinedFunctionIndex(Sym.Info.ElementIndex))
    return getDefinedFunction(Sym.Info.ElementIndex).CodeSectionOffset;
  else
    return getSymbolValue(Symb);
}

uint64_t WasmObjectFile::getWasmSymbolValue(const WasmSymbol &Sym) const {
  switch (Sym.Info.Kind) {
  case wasm::WASM_SYMBOL_TYPE_FUNCTION:
  case wasm::WASM_SYMBOL_TYPE_GLOBAL:
  case wasm::WASM_SYMBOL_TYPE_EVENT:
  case wasm::WASM_SYMBOL_TYPE_TABLE:
    return Sym.Info.ElementIndex;
  case wasm::WASM_SYMBOL_TYPE_DATA: {
    // The value of a data symbol is the segment offset, plus the symbol
    // offset within the segment.
    uint32_t SegmentIndex = Sym.Info.DataRef.Segment;
    const wasm::WasmDataSegment &Segment = DataSegments[SegmentIndex].Data;
    if (Segment.Offset.Opcode == wasm::WASM_OPCODE_I32_CONST) {
      return Segment.Offset.Value.Int32 + Sym.Info.DataRef.Offset;
    } else if (Segment.Offset.Opcode == wasm::WASM_OPCODE_I64_CONST) {
      return Segment.Offset.Value.Int64 + Sym.Info.DataRef.Offset;
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
  case wasm::WASM_SYMBOL_TYPE_EVENT:
    return SymbolRef::ST_Other;
  case wasm::WASM_SYMBOL_TYPE_TABLE:
    return SymbolRef::ST_Other;
  }

  llvm_unreachable("Unknown WasmSymbol::SymbolType");
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
  case wasm::WASM_SYMBOL_TYPE_EVENT:
    return EventSection;
  case wasm::WASM_SYMBOL_TYPE_TABLE:
    return TableSection;
  default:
    llvm_unreachable("Unknown WasmSymbol::SymbolType");
  }
}

void WasmObjectFile::moveSectionNext(DataRefImpl &Sec) const { Sec.d.a++; }

Expected<StringRef> WasmObjectFile::getSectionName(DataRefImpl Sec) const {
  const WasmSection &S = Sections[Sec.d.a];
#define ECase(X)                                                               \
  case wasm::WASM_SEC_##X:                                                     \
    return #X;
  switch (S.Type) {
    ECase(TYPE);
    ECase(IMPORT);
    ECase(FUNCTION);
    ECase(TABLE);
    ECase(MEMORY);
    ECase(GLOBAL);
    ECase(EVENT);
    ECase(EXPORT);
    ECase(START);
    ECase(ELEM);
    ECase(CODE);
    ECase(DATA);
    ECase(DATACOUNT);
  case wasm::WASM_SEC_CUSTOM:
    return S.Name;
  default:
    return createStringError(object_error::invalid_section_index, "");
  }
#undef ECase
}

uint64_t WasmObjectFile::getSectionAddress(DataRefImpl Sec) const { return 0; }

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

SubtargetFeatures WasmObjectFile::getFeatures() const {
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
  case wasm::WASM_SEC_EVENT:
    return WASM_SEC_ORDER_EVENT;
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
        {WASM_SEC_ORDER_MEMORY, WASM_SEC_ORDER_EVENT},
        // WASM_SEC_ORDER_EVENT
        {WASM_SEC_ORDER_EVENT, WASM_SEC_ORDER_GLOBAL},
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
