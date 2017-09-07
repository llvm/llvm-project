//===- WasmObjectFile.cpp - Wasm object file implementation ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
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
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <system_error>

#define DEBUG_TYPE "wasm-object"

using namespace llvm;
using namespace object;

Expected<std::unique_ptr<WasmObjectFile>>
ObjectFile::createWasmObjectFile(MemoryBufferRef Buffer) {
  Error Err = Error::success();
  auto ObjectFile = llvm::make_unique<WasmObjectFile>(Buffer, Err);
  if (Err)
    return std::move(Err);

  return std::move(ObjectFile);
}

#define VARINT7_MAX ((1<<7)-1)
#define VARINT7_MIN (-(1<<7))
#define VARUINT7_MAX (1<<7)
#define VARUINT1_MAX (1)

static uint8_t readUint8(const uint8_t *&Ptr) { return *Ptr++; }

static uint32_t readUint32(const uint8_t *&Ptr) {
  uint32_t Result = support::endian::read32le(Ptr);
  Ptr += sizeof(Result);
  return Result;
}

static int32_t readFloat32(const uint8_t *&Ptr) {
  int32_t Result = 0;
  memcpy(&Result, Ptr, sizeof(Result));
  Ptr += sizeof(Result);
  return Result;
}

static int64_t readFloat64(const uint8_t *&Ptr) {
  int64_t Result = 0;
  memcpy(&Result, Ptr, sizeof(Result));
  Ptr += sizeof(Result);
  return Result;
}

static uint64_t readULEB128(const uint8_t *&Ptr) {
  unsigned Count;
  uint64_t Result = decodeULEB128(Ptr, &Count);
  Ptr += Count;
  return Result;
}

static StringRef readString(const uint8_t *&Ptr) {
  uint32_t StringLen = readULEB128(Ptr);
  StringRef Return = StringRef(reinterpret_cast<const char *>(Ptr), StringLen);
  Ptr += StringLen;
  return Return;
}

static int64_t readLEB128(const uint8_t *&Ptr) {
  unsigned Count;
  uint64_t Result = decodeSLEB128(Ptr, &Count);
  Ptr += Count;
  return Result;
}

static uint8_t readVaruint1(const uint8_t *&Ptr) {
  int64_t result = readLEB128(Ptr);
  assert(result <= VARUINT1_MAX && result >= 0);
  return result;
}

static int8_t readVarint7(const uint8_t *&Ptr) {
  int64_t result = readLEB128(Ptr);
  assert(result <= VARINT7_MAX && result >= VARINT7_MIN);
  return result;
}

static uint8_t readVaruint7(const uint8_t *&Ptr) {
  uint64_t result = readULEB128(Ptr);
  assert(result <= VARUINT7_MAX);
  return result;
}

static int32_t readVarint32(const uint8_t *&Ptr) {
  int64_t result = readLEB128(Ptr);
  assert(result <= INT32_MAX && result >= INT32_MIN);
  return result;
}

static uint32_t readVaruint32(const uint8_t *&Ptr) {
  uint64_t result = readULEB128(Ptr);
  assert(result <= UINT32_MAX);
  return result;
}

static int64_t readVarint64(const uint8_t *&Ptr) {
  return readLEB128(Ptr);
}

static uint8_t readOpcode(const uint8_t *&Ptr) {
  return readUint8(Ptr);
}

static Error readInitExpr(wasm::WasmInitExpr &Expr, const uint8_t *&Ptr) {
  Expr.Opcode = readOpcode(Ptr);

  switch (Expr.Opcode) {
  case wasm::WASM_OPCODE_I32_CONST:
    Expr.Value.Int32 = readVarint32(Ptr);
    break;
  case wasm::WASM_OPCODE_I64_CONST:
    Expr.Value.Int64 = readVarint64(Ptr);
    break;
  case wasm::WASM_OPCODE_F32_CONST:
    Expr.Value.Float32 = readFloat32(Ptr);
    break;
  case wasm::WASM_OPCODE_F64_CONST:
    Expr.Value.Float64 = readFloat64(Ptr);
    break;
  case wasm::WASM_OPCODE_GET_GLOBAL:
    Expr.Value.Global = readULEB128(Ptr);
    break;
  default:
    return make_error<GenericBinaryError>("Invalid opcode in init_expr",
                                          object_error::parse_failed);
  }

  uint8_t EndOpcode = readOpcode(Ptr);
  if (EndOpcode != wasm::WASM_OPCODE_END) {
    return make_error<GenericBinaryError>("Invalid init_expr",
                                          object_error::parse_failed);
  }
  return Error::success();
}

static wasm::WasmLimits readLimits(const uint8_t *&Ptr) {
  wasm::WasmLimits Result;
  Result.Flags = readVaruint1(Ptr);
  Result.Initial = readVaruint32(Ptr);
  if (Result.Flags & wasm::WASM_LIMITS_FLAG_HAS_MAX)
    Result.Maximum = readVaruint32(Ptr);
  return Result;
}

static wasm::WasmTable readTable(const uint8_t *&Ptr) {
  wasm::WasmTable Table;
  Table.ElemType = readVarint7(Ptr);
  Table.Limits = readLimits(Ptr);
  return Table;
}

static Error readSection(WasmSection &Section, const uint8_t *&Ptr,
                         const uint8_t *Start) {
  // TODO(sbc): Avoid reading past EOF in the case of malformed files.
  Section.Offset = Ptr - Start;
  Section.Type = readVaruint7(Ptr);
  uint32_t Size = readVaruint32(Ptr);
  if (Size == 0)
    return make_error<StringError>("Zero length section",
                                   object_error::parse_failed);
  Section.Content = ArrayRef<uint8_t>(Ptr, Size);
  Ptr += Size;
  return Error::success();
}

WasmObjectFile::WasmObjectFile(MemoryBufferRef Buffer, Error &Err)
    : ObjectFile(Binary::ID_Wasm, Buffer) {
  LinkingData.DataAlignment = 0;
  LinkingData.DataSize = 0;

  ErrorAsOutParameter ErrAsOutParam(&Err);
  Header.Magic = getData().substr(0, 4);
  if (Header.Magic != StringRef("\0asm", 4)) {
    Err = make_error<StringError>("Bad magic number",
                                  object_error::parse_failed);
    return;
  }

  const uint8_t *Eof = getPtr(getData().size());
  const uint8_t *Ptr = getPtr(4);

  if (Ptr + 4 > Eof) {
    Err = make_error<StringError>("Missing version number",
                                  object_error::parse_failed);
    return;
  }

  Header.Version = readUint32(Ptr);
  if (Header.Version != wasm::WasmVersion) {
    Err = make_error<StringError>("Bad version number",
                                  object_error::parse_failed);
    return;
  }

  WasmSection Sec;
  while (Ptr < Eof) {
    if ((Err = readSection(Sec, Ptr, getPtr(0))))
      return;
    if ((Err = parseSection(Sec)))
      return;

    Sections.push_back(Sec);
  }
}

Error WasmObjectFile::parseSection(WasmSection &Sec) {
  const uint8_t* Start = Sec.Content.data();
  const uint8_t* End = Start + Sec.Content.size();
  switch (Sec.Type) {
  case wasm::WASM_SEC_CUSTOM:
    return parseCustomSection(Sec, Start, End);
  case wasm::WASM_SEC_TYPE:
    return parseTypeSection(Start, End);
  case wasm::WASM_SEC_IMPORT:
    return parseImportSection(Start, End);
  case wasm::WASM_SEC_FUNCTION:
    return parseFunctionSection(Start, End);
  case wasm::WASM_SEC_TABLE:
    return parseTableSection(Start, End);
  case wasm::WASM_SEC_MEMORY:
    return parseMemorySection(Start, End);
  case wasm::WASM_SEC_GLOBAL:
    return parseGlobalSection(Start, End);
  case wasm::WASM_SEC_EXPORT:
    return parseExportSection(Start, End);
  case wasm::WASM_SEC_START:
    return parseStartSection(Start, End);
  case wasm::WASM_SEC_ELEM:
    return parseElemSection(Start, End);
  case wasm::WASM_SEC_CODE:
    return parseCodeSection(Start, End);
  case wasm::WASM_SEC_DATA:
    return parseDataSection(Start, End);
  default:
    return make_error<GenericBinaryError>("Bad section type",
                                          object_error::parse_failed);
  }
}

Error WasmObjectFile::parseNameSection(const uint8_t *Ptr, const uint8_t *End) {
  while (Ptr < End) {
    uint8_t Type = readVarint7(Ptr);
    uint32_t Size = readVaruint32(Ptr);
    const uint8_t *SubSectionEnd = Ptr + Size;
    switch (Type) {
    case wasm::WASM_NAMES_FUNCTION: {
      uint32_t Count = readVaruint32(Ptr);
      while (Count--) {
        uint32_t Index = readVaruint32(Ptr);
        StringRef Name = readString(Ptr);
        if (!Name.empty())
          Symbols.emplace_back(Name,
                               WasmSymbol::SymbolType::DEBUG_FUNCTION_NAME,
                               Sections.size(), Index);
      }
      break;
    }
    // Ignore local names for now
    case wasm::WASM_NAMES_LOCAL:
    default:
      Ptr += Size;
      break;
    }
    if (Ptr != SubSectionEnd)
      return make_error<GenericBinaryError>("Name sub-section ended prematurely",
                                            object_error::parse_failed);
  }

  if (Ptr != End)
    return make_error<GenericBinaryError>("Name section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

void WasmObjectFile::populateSymbolTable() {
  // Add imports to symbol table
  size_t ImportIndex = 0;
  for (const wasm::WasmImport& Import : Imports) {
    switch (Import.Kind) {
    case wasm::WASM_EXTERNAL_GLOBAL:
      assert(Import.Global.Type == wasm::WASM_TYPE_I32);
      SymbolMap.try_emplace(Import.Field, Symbols.size());
      Symbols.emplace_back(Import.Field, WasmSymbol::SymbolType::GLOBAL_IMPORT,
                           ImportSection, ImportIndex);
      DEBUG(dbgs() << "Adding import: " << Symbols.back()
                   << " sym index:" << Symbols.size() << "\n");
      break;
    case wasm::WASM_EXTERNAL_FUNCTION:
      SymbolMap.try_emplace(Import.Field, Symbols.size());
      Symbols.emplace_back(Import.Field,
                           WasmSymbol::SymbolType::FUNCTION_IMPORT,
                           ImportSection, ImportIndex);
      DEBUG(dbgs() << "Adding import: " << Symbols.back()
                   << " sym index:" << Symbols.size() << "\n");
      break;
    default:
      break;
    }
    ImportIndex++;
  }

  // Add exports to symbol table
  size_t ExportIndex = 0;
  for (const wasm::WasmExport& Export : Exports) {
    if (Export.Kind == wasm::WASM_EXTERNAL_FUNCTION ||
        Export.Kind == wasm::WASM_EXTERNAL_GLOBAL) {
      WasmSymbol::SymbolType ExportType =
          Export.Kind == wasm::WASM_EXTERNAL_FUNCTION
              ? WasmSymbol::SymbolType::FUNCTION_EXPORT
              : WasmSymbol::SymbolType::GLOBAL_EXPORT;
      auto Pair = SymbolMap.try_emplace(Export.Name, Symbols.size());
      if (Pair.second) {
        Symbols.emplace_back(Export.Name, ExportType,
                             ExportSection, ExportIndex);
        DEBUG(dbgs() << "Adding export: " << Symbols.back()
                     << " sym index:" << Symbols.size() << "\n");
      } else {
        uint32_t SymIndex = Pair.first->second;
        Symbols[SymIndex] =
            WasmSymbol(Export.Name, ExportType, ExportSection, ExportIndex);
        DEBUG(dbgs() << "Replacing existing symbol:  " << Symbols[SymIndex]
                     << " sym index:" << SymIndex << "\n");
      }
    }
    ExportIndex++;
  }
}

Error WasmObjectFile::parseLinkingSection(const uint8_t *Ptr,
                                          const uint8_t *End) {
  HasLinkingSection = true;

  // Only populate the symbol table with imports and exports if the object
  // has a linking section (i.e. its a relocatable object file). Otherwise
  // the global might not represent symbols at all.
  populateSymbolTable();

  while (Ptr < End) {
    uint8_t Type = readVarint7(Ptr);
    uint32_t Size = readVaruint32(Ptr);
    const uint8_t *SubSectionEnd = Ptr + Size;
    switch (Type) {
    case wasm::WASM_SYMBOL_INFO: {
      uint32_t Count = readVaruint32(Ptr);
      while (Count--) {
        StringRef Symbol = readString(Ptr);
        DEBUG(dbgs() << "reading syminfo: " << Symbol << "\n");
        uint32_t Flags = readVaruint32(Ptr);
        auto iter = SymbolMap.find(Symbol);
        if (iter == SymbolMap.end()) {
          return make_error<GenericBinaryError>(
              "Invalid symbol name in linking section: " + Symbol,
              object_error::parse_failed);
        }
        uint32_t SymIndex = iter->second;
        assert(SymIndex < Symbols.size());
        Symbols[SymIndex].Flags = Flags;
        DEBUG(dbgs() << "Set symbol flags index:"
                     << SymIndex << " name:"
                     << Symbols[SymIndex].Name << " exptected:"
                     << Symbol << " flags: " << Flags << "\n");
      }
      break;
    }
    case wasm::WASM_DATA_SIZE:
      LinkingData.DataSize = readVaruint32(Ptr);
      break;
    case wasm::WASM_DATA_ALIGNMENT:
      LinkingData.DataAlignment = readVaruint32(Ptr);
      break;
    case wasm::WASM_STACK_POINTER:
    default:
      Ptr += Size;
      break;
    }
    if (Ptr != SubSectionEnd)
      return make_error<GenericBinaryError>(
          "Linking sub-section ended prematurely", object_error::parse_failed);
  }
  if (Ptr != End)
    return make_error<GenericBinaryError>("Linking section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

WasmSection* WasmObjectFile::findCustomSectionByName(StringRef Name) {
  for (WasmSection& Section : Sections) {
    if (Section.Type == wasm::WASM_SEC_CUSTOM && Section.Name == Name)
      return &Section;
  }
  return nullptr;
}

WasmSection* WasmObjectFile::findSectionByType(uint32_t Type) {
  assert(Type != wasm::WASM_SEC_CUSTOM);
  for (WasmSection& Section : Sections) {
    if (Section.Type == Type)
      return &Section;
  }
  return nullptr;
}

Error WasmObjectFile::parseRelocSection(StringRef Name, const uint8_t *Ptr,
                                        const uint8_t *End) {
  uint8_t SectionCode = readVarint7(Ptr);
  WasmSection* Section = nullptr;
  if (SectionCode == wasm::WASM_SEC_CUSTOM) {
    StringRef Name = readString(Ptr);
    Section = findCustomSectionByName(Name);
  } else {
    Section = findSectionByType(SectionCode);
  }
  if (!Section)
    return make_error<GenericBinaryError>("Invalid section code",
                                          object_error::parse_failed);
  uint32_t RelocCount = readVaruint32(Ptr);
  while (RelocCount--) {
    wasm::WasmRelocation Reloc;
    memset(&Reloc, 0, sizeof(Reloc));
    Reloc.Type = readVaruint32(Ptr);
    Reloc.Offset = readVaruint32(Ptr);
    Reloc.Index = readVaruint32(Ptr);
    switch (Reloc.Type) {
    case wasm::R_WEBASSEMBLY_FUNCTION_INDEX_LEB:
    case wasm::R_WEBASSEMBLY_TABLE_INDEX_SLEB:
    case wasm::R_WEBASSEMBLY_TABLE_INDEX_I32:
    case wasm::R_WEBASSEMBLY_TYPE_INDEX_LEB:
    case wasm::R_WEBASSEMBLY_GLOBAL_INDEX_LEB:
      break;
    case wasm::R_WEBASSEMBLY_MEMORY_ADDR_LEB:
    case wasm::R_WEBASSEMBLY_MEMORY_ADDR_SLEB:
    case wasm::R_WEBASSEMBLY_MEMORY_ADDR_I32:
      Reloc.Addend = readVarint32(Ptr);
      break;
    default:
      return make_error<GenericBinaryError>("Bad relocation type: " +
                                                Twine(Reloc.Type),
                                            object_error::parse_failed);
    }
    Section->Relocations.push_back(Reloc);
  }
  if (Ptr != End)
    return make_error<GenericBinaryError>("Reloc section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseCustomSection(WasmSection &Sec,
                                         const uint8_t *Ptr, const uint8_t *End) {
  Sec.Name = readString(Ptr);
  if (Sec.Name == "name") {
    if (Error Err = parseNameSection(Ptr, End))
      return Err;
  } else if (Sec.Name == "linking") {
    if (Error Err = parseLinkingSection(Ptr, End))
      return Err;
  } else if (Sec.Name.startswith("reloc.")) {
    if (Error Err = parseRelocSection(Sec.Name, Ptr, End))
      return Err;
  }
  return Error::success();
}

Error WasmObjectFile::parseTypeSection(const uint8_t *Ptr, const uint8_t *End) {
  uint32_t Count = readVaruint32(Ptr);
  Signatures.reserve(Count);
  while (Count--) {
    wasm::WasmSignature Sig;
    Sig.ReturnType = wasm::WASM_TYPE_NORESULT;
    int8_t Form = readVarint7(Ptr);
    if (Form != wasm::WASM_TYPE_FUNC) {
      return make_error<GenericBinaryError>("Invalid signature type",
                                            object_error::parse_failed);
    }
    uint32_t ParamCount = readVaruint32(Ptr);
    Sig.ParamTypes.reserve(ParamCount);
    while (ParamCount--) {
      uint32_t ParamType = readVarint7(Ptr);
      Sig.ParamTypes.push_back(ParamType);
    }
    uint32_t ReturnCount = readVaruint32(Ptr);
    if (ReturnCount) {
      if (ReturnCount != 1) {
        return make_error<GenericBinaryError>(
            "Multiple return types not supported", object_error::parse_failed);
      }
      Sig.ReturnType = readVarint7(Ptr);
    }
    Signatures.push_back(Sig);
  }
  if (Ptr != End)
    return make_error<GenericBinaryError>("Type section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseImportSection(const uint8_t *Ptr, const uint8_t *End) {
  ImportSection = Sections.size();
  uint32_t Count = readVaruint32(Ptr);
  Imports.reserve(Count);
  for (uint32_t i = 0; i < Count; i++) {
    wasm::WasmImport Im;
    Im.Module = readString(Ptr);
    Im.Field = readString(Ptr);
    Im.Kind = readUint8(Ptr);
    switch (Im.Kind) {
    case wasm::WASM_EXTERNAL_FUNCTION:
      NumImportedFunctions++;
      Im.SigIndex = readVaruint32(Ptr);
      break;
    case wasm::WASM_EXTERNAL_GLOBAL:
      NumImportedGlobals++;
      Im.Global.Type = readVarint7(Ptr);
      Im.Global.Mutable = readVaruint1(Ptr);
      break;
    case wasm::WASM_EXTERNAL_MEMORY:
      Im.Memory = readLimits(Ptr);
      break;
    case wasm::WASM_EXTERNAL_TABLE:
      Im.Table = readTable(Ptr);
      if (Im.Table.ElemType != wasm::WASM_TYPE_ANYFUNC)
        return make_error<GenericBinaryError>("Invalid table element type",
                                              object_error::parse_failed);
      break;
    default:
      return make_error<GenericBinaryError>(
          "Unexpected import kind", object_error::parse_failed);
    }
    Imports.push_back(Im);
  }
  if (Ptr != End)
    return make_error<GenericBinaryError>("Import section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseFunctionSection(const uint8_t *Ptr, const uint8_t *End) {
  uint32_t Count = readVaruint32(Ptr);
  FunctionTypes.reserve(Count);
  while (Count--) {
    FunctionTypes.push_back(readVaruint32(Ptr));
  }
  if (Ptr != End)
    return make_error<GenericBinaryError>("Function section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseTableSection(const uint8_t *Ptr, const uint8_t *End) {
  uint32_t Count = readVaruint32(Ptr);
  Tables.reserve(Count);
  while (Count--) {
    Tables.push_back(readTable(Ptr));
    if (Tables.back().ElemType != wasm::WASM_TYPE_ANYFUNC) {
      return make_error<GenericBinaryError>("Invalid table element type",
                                            object_error::parse_failed);
    }
  }
  if (Ptr != End)
    return make_error<GenericBinaryError>("Table section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseMemorySection(const uint8_t *Ptr, const uint8_t *End) {
  uint32_t Count = readVaruint32(Ptr);
  Memories.reserve(Count);
  while (Count--) {
    Memories.push_back(readLimits(Ptr));
  }
  if (Ptr != End)
    return make_error<GenericBinaryError>("Memory section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseGlobalSection(const uint8_t *Ptr, const uint8_t *End) {
  uint32_t Count = readVaruint32(Ptr);
  Globals.reserve(Count);
  while (Count--) {
    wasm::WasmGlobal Global;
    Global.Type = readVarint7(Ptr);
    Global.Mutable = readVaruint1(Ptr);
    if (Error Err = readInitExpr(Global.InitExpr, Ptr))
      return Err;
    Globals.push_back(Global);
  }
  if (Ptr != End)
    return make_error<GenericBinaryError>("Global section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseExportSection(const uint8_t *Ptr, const uint8_t *End) {
  ExportSection = Sections.size();
  uint32_t Count = readVaruint32(Ptr);
  Exports.reserve(Count);
  for (uint32_t i = 0; i < Count; i++) {
    wasm::WasmExport Ex;
    Ex.Name = readString(Ptr);
    Ex.Kind = readUint8(Ptr);
    Ex.Index = readVaruint32(Ptr);
    switch (Ex.Kind) {
    case wasm::WASM_EXTERNAL_FUNCTION:
      if (Ex.Index >= FunctionTypes.size() + NumImportedFunctions)
        return make_error<GenericBinaryError>("Invalid function export",
                                              object_error::parse_failed);
      break;
    case wasm::WASM_EXTERNAL_GLOBAL: {
      if (Ex.Index >= Globals.size() + NumImportedGlobals)
        return make_error<GenericBinaryError>("Invalid global export",
                                              object_error::parse_failed);
      break;
    }
    case wasm::WASM_EXTERNAL_MEMORY:
    case wasm::WASM_EXTERNAL_TABLE:
      break;
    default:
      return make_error<GenericBinaryError>(
          "Unexpected export kind", object_error::parse_failed);
    }
    Exports.push_back(Ex);
  }
  if (Ptr != End)
    return make_error<GenericBinaryError>("Export section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseStartSection(const uint8_t *Ptr, const uint8_t *End) {
  StartFunction = readVaruint32(Ptr);
  if (StartFunction >= FunctionTypes.size())
    return make_error<GenericBinaryError>("Invalid start function",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseCodeSection(const uint8_t *Ptr, const uint8_t *End) {
  uint32_t FunctionCount = readVaruint32(Ptr);
  if (FunctionCount != FunctionTypes.size()) {
    return make_error<GenericBinaryError>("Invalid function count",
                                          object_error::parse_failed);
  }

  CodeSection = ArrayRef<uint8_t>(Ptr, End - Ptr);

  while (FunctionCount--) {
    wasm::WasmFunction Function;
    uint32_t FunctionSize = readVaruint32(Ptr);
    const uint8_t *FunctionEnd = Ptr + FunctionSize;

    uint32_t NumLocalDecls = readVaruint32(Ptr);
    Function.Locals.reserve(NumLocalDecls);
    while (NumLocalDecls--) {
      wasm::WasmLocalDecl Decl;
      Decl.Count = readVaruint32(Ptr);
      Decl.Type = readVarint7(Ptr);
      Function.Locals.push_back(Decl);
    }

    uint32_t BodySize = FunctionEnd - Ptr;
    Function.Body = ArrayRef<uint8_t>(Ptr, BodySize);
    Ptr += BodySize;
    assert(Ptr == FunctionEnd);
    Functions.push_back(Function);
  }
  if (Ptr != End)
    return make_error<GenericBinaryError>("Code section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseElemSection(const uint8_t *Ptr, const uint8_t *End) {
  uint32_t Count = readVaruint32(Ptr);
  ElemSegments.reserve(Count);
  while (Count--) {
    wasm::WasmElemSegment Segment;
    Segment.TableIndex = readVaruint32(Ptr);
    if (Segment.TableIndex != 0) {
      return make_error<GenericBinaryError>("Invalid TableIndex",
                                            object_error::parse_failed);
    }
    if (Error Err = readInitExpr(Segment.Offset, Ptr))
      return Err;
    uint32_t NumElems = readVaruint32(Ptr);
    while (NumElems--) {
      Segment.Functions.push_back(readVaruint32(Ptr));
    }
    ElemSegments.push_back(Segment);
  }
  if (Ptr != End)
    return make_error<GenericBinaryError>("Elem section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseDataSection(const uint8_t *Ptr, const uint8_t *End) {
  const uint8_t *Start = Ptr;
  uint32_t Count = readVaruint32(Ptr);
  DataSegments.reserve(Count);
  while (Count--) {
    WasmSegment Segment;
    Segment.Data.MemoryIndex = readVaruint32(Ptr);
    if (Error Err = readInitExpr(Segment.Data.Offset, Ptr))
      return Err;
    uint32_t Size = readVaruint32(Ptr);
    Segment.Data.Content = ArrayRef<uint8_t>(Ptr, Size);
    Segment.SectionOffset = Ptr - Start;
    Ptr += Size;
    DataSegments.push_back(Segment);
  }
  if (Ptr != End)
    return make_error<GenericBinaryError>("Data section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

const uint8_t *WasmObjectFile::getPtr(size_t Offset) const {
  return reinterpret_cast<const uint8_t *>(getData().substr(Offset, 1).data());
}

const wasm::WasmObjectHeader &WasmObjectFile::getHeader() const {
  return Header;
}

void WasmObjectFile::moveSymbolNext(DataRefImpl &Symb) const { Symb.d.a++; }

uint32_t WasmObjectFile::getSymbolFlags(DataRefImpl Symb) const {
  uint32_t Result = SymbolRef::SF_None;
  const WasmSymbol &Sym = getWasmSymbol(Symb);

  DEBUG(dbgs() << "getSymbolFlags: ptr=" << &Sym << " " << Sym << "\n");
  if (Sym.Flags & wasm::WASM_SYMBOL_FLAG_WEAK)
    Result |= SymbolRef::SF_Weak;

  switch (Sym.Type) {
  case WasmSymbol::SymbolType::FUNCTION_IMPORT:
    Result |= SymbolRef::SF_Undefined | SymbolRef::SF_Executable;
    break;
  case WasmSymbol::SymbolType::FUNCTION_EXPORT:
    Result |= SymbolRef::SF_Global | SymbolRef::SF_Executable;
    break;
  case WasmSymbol::SymbolType::DEBUG_FUNCTION_NAME:
    Result |= SymbolRef::SF_Executable;
    Result |= SymbolRef::SF_FormatSpecific;
    break;
  case WasmSymbol::SymbolType::GLOBAL_IMPORT:
    Result |= SymbolRef::SF_Undefined;
    break;
  case WasmSymbol::SymbolType::GLOBAL_EXPORT:
    Result |= SymbolRef::SF_Global;
    break;
  }

  return Result;
}

basic_symbol_iterator WasmObjectFile::symbol_begin() const {
  DataRefImpl Ref;
  Ref.d.a = 0;
  return BasicSymbolRef(Ref, this);
}

basic_symbol_iterator WasmObjectFile::symbol_end() const {
  DataRefImpl Ref;
  Ref.d.a = Symbols.size();
  return BasicSymbolRef(Ref, this);
}

const WasmSymbol &WasmObjectFile::getWasmSymbol(const DataRefImpl &Symb) const {
  return Symbols[Symb.d.a];
}

const WasmSymbol &WasmObjectFile::getWasmSymbol(const SymbolRef &Symb) const {
  return getWasmSymbol(Symb.getRawDataRefImpl());
}

Expected<StringRef> WasmObjectFile::getSymbolName(DataRefImpl Symb) const {
  return getWasmSymbol(Symb).Name;
}

Expected<uint64_t> WasmObjectFile::getSymbolAddress(DataRefImpl Symb) const {
  return getSymbolValue(Symb);
}

uint64_t WasmObjectFile::getSymbolValueImpl(DataRefImpl Symb) const {
  const WasmSymbol& Sym = getWasmSymbol(Symb);
  switch (Sym.Type) {
  case WasmSymbol::SymbolType::FUNCTION_IMPORT:
  case WasmSymbol::SymbolType::GLOBAL_IMPORT:
    return 0;
  case WasmSymbol::SymbolType::FUNCTION_EXPORT:
    return Exports[Sym.ElementIndex].Index;
  case WasmSymbol::SymbolType::GLOBAL_EXPORT: {
    uint32_t GlobalIndex = Exports[Sym.ElementIndex].Index - NumImportedGlobals;
    assert(GlobalIndex < Globals.size());
    const wasm::WasmGlobal& Global = Globals[GlobalIndex];
    // WasmSymbols correspond only to I32_CONST globals
    assert(Global.InitExpr.Opcode == wasm::WASM_OPCODE_I32_CONST);
    return Global.InitExpr.Value.Int32;
  }
  case WasmSymbol::SymbolType::DEBUG_FUNCTION_NAME:
    return Sym.ElementIndex;
  }
  llvm_unreachable("invalid symbol type");
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

  switch (Sym.Type) {
  case WasmSymbol::SymbolType::FUNCTION_IMPORT:
  case WasmSymbol::SymbolType::FUNCTION_EXPORT:
  case WasmSymbol::SymbolType::DEBUG_FUNCTION_NAME:
    return SymbolRef::ST_Function;
  case WasmSymbol::SymbolType::GLOBAL_IMPORT:
  case WasmSymbol::SymbolType::GLOBAL_EXPORT:
    return SymbolRef::ST_Data;
  }

  llvm_unreachable("Unknown WasmSymbol::SymbolType");
  return SymbolRef::ST_Other;
}

Expected<section_iterator>
WasmObjectFile::getSymbolSection(DataRefImpl Symb) const {
  DataRefImpl Ref;
  Ref.d.a = getWasmSymbol(Symb).Section;
  return section_iterator(SectionRef(Ref, this));
}

void WasmObjectFile::moveSectionNext(DataRefImpl &Sec) const { Sec.d.a++; }

std::error_code WasmObjectFile::getSectionName(DataRefImpl Sec,
                                               StringRef &Res) const {
  const WasmSection &S = Sections[Sec.d.a];
#define ECase(X)                                                               \
  case wasm::WASM_SEC_##X:                                                     \
    Res = #X;                                                                  \
    break
  switch (S.Type) {
    ECase(TYPE);
    ECase(IMPORT);
    ECase(FUNCTION);
    ECase(TABLE);
    ECase(MEMORY);
    ECase(GLOBAL);
    ECase(EXPORT);
    ECase(START);
    ECase(ELEM);
    ECase(CODE);
    ECase(DATA);
  case wasm::WASM_SEC_CUSTOM:
    Res = S.Name;
    break;
  default:
    return object_error::invalid_section_index;
  }
#undef ECase
  return std::error_code();
}

uint64_t WasmObjectFile::getSectionAddress(DataRefImpl Sec) const { return 0; }

uint64_t WasmObjectFile::getSectionIndex(DataRefImpl Sec) const {
  return Sec.d.a;
}

uint64_t WasmObjectFile::getSectionSize(DataRefImpl Sec) const {
  const WasmSection &S = Sections[Sec.d.a];
  return S.Content.size();
}

std::error_code WasmObjectFile::getSectionContents(DataRefImpl Sec,
                                                   StringRef &Res) const {
  const WasmSection &S = Sections[Sec.d.a];
  // This will never fail since wasm sections can never be empty (user-sections
  // must have a name and non-user sections each have a defined structure).
  Res = StringRef(reinterpret_cast<const char *>(S.Content.data()),
                  S.Content.size());
  return std::error_code();
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

bool WasmObjectFile::isSectionBitcode(DataRefImpl Sec) const { return false; }

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

void WasmObjectFile::moveRelocationNext(DataRefImpl &Rel) const {
  Rel.d.b++;
}

uint64_t WasmObjectFile::getRelocationOffset(DataRefImpl Ref) const {
  const wasm::WasmRelocation &Rel = getWasmRelocation(Ref);
  return Rel.Offset;
}

symbol_iterator WasmObjectFile::getRelocationSymbol(DataRefImpl Rel) const {
  llvm_unreachable("not yet implemented");
  SymbolRef Ref;
  return symbol_iterator(Ref);
}

uint64_t WasmObjectFile::getRelocationType(DataRefImpl Ref) const {
  const wasm::WasmRelocation &Rel = getWasmRelocation(Ref);
  return Rel.Type;
}

void WasmObjectFile::getRelocationTypeName(
    DataRefImpl Ref, SmallVectorImpl<char> &Result) const {
  const wasm::WasmRelocation& Rel = getWasmRelocation(Ref);
  StringRef Res = "Unknown";

#define WASM_RELOC(name, value)  \
  case wasm::name:              \
    Res = #name;               \
    break;

  switch (Rel.Type) {
#include "llvm/BinaryFormat/WasmRelocs/WebAssembly.def"
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

uint8_t WasmObjectFile::getBytesInAddress() const { return 4; }

StringRef WasmObjectFile::getFileFormatName() const { return "WASM"; }

unsigned WasmObjectFile::getArch() const { return Triple::wasm32; }

SubtargetFeatures WasmObjectFile::getFeatures() const {
  return SubtargetFeatures();
}

bool WasmObjectFile::isRelocatableObject() const {
  return HasLinkingSection;
}

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
  const WasmSection& Sec = Sections[Ref.d.a];
  assert(Ref.d.b < Sec.Relocations.size());
  return Sec.Relocations[Ref.d.b];
}
