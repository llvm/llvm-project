//===- WasmObjectFile.cpp - Wasm object file implementation ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
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
  Table.ElemType = readUint8(Ptr);
  Table.Limits = readLimits(Ptr);
  return Table;
}

static Error readSection(WasmSection &Section, const uint8_t *&Ptr,
                         const uint8_t *Start, const uint8_t *Eof) {
  Section.Offset = Ptr - Start;
  Section.Type = readUint8(Ptr);
  uint32_t Size = readVaruint32(Ptr);
  if (Size == 0)
    return make_error<StringError>("Zero length section",
                                   object_error::parse_failed);
  if (Ptr + Size > Eof)
    return make_error<StringError>("Section too large",
                                   object_error::parse_failed);
  if (Section.Type == wasm::WASM_SEC_CUSTOM) {
    const uint8_t *NameStart = Ptr;
    Section.Name = readString(Ptr);
    Size -= Ptr - NameStart;
  }
  Section.Content = ArrayRef<uint8_t>(Ptr, Size);
  Ptr += Size;
  return Error::success();
}

WasmObjectFile::WasmObjectFile(MemoryBufferRef Buffer, Error &Err)
    : ObjectFile(Binary::ID_Wasm, Buffer) {
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
    if ((Err = readSection(Sec, Ptr, getPtr(0), Eof)))
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
  llvm::DenseSet<uint64_t> Seen;
  if (Functions.size() != FunctionTypes.size()) {
    return make_error<GenericBinaryError>("Names must come after code section",
                                          object_error::parse_failed);
  }

  while (Ptr < End) {
    uint8_t Type = readUint8(Ptr);
    uint32_t Size = readVaruint32(Ptr);
    const uint8_t *SubSectionEnd = Ptr + Size;
    switch (Type) {
    case wasm::WASM_NAMES_FUNCTION: {
      uint32_t Count = readVaruint32(Ptr);
      while (Count--) {
        uint32_t Index = readVaruint32(Ptr);
        if (!Seen.insert(Index).second)
          return make_error<GenericBinaryError>("Function named more than once",
                                                object_error::parse_failed);
        StringRef Name = readString(Ptr);
        if (!isValidFunctionIndex(Index) || Name.empty())
          return make_error<GenericBinaryError>("Invalid name entry",
                                                object_error::parse_failed);
        DebugNames.push_back(wasm::WasmFunctionName{Index, Name});
        if (isDefinedFunctionIndex(Index)) {
          // Override any existing name; the name specified by the "names"
          // section is the Function's canonical name.
          getDefinedFunction(Index).Name = Name;
        }
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

Error WasmObjectFile::parseLinkingSection(const uint8_t *Ptr,
                                          const uint8_t *End) {
  HasLinkingSection = true;
  if (Functions.size() != FunctionTypes.size()) {
    return make_error<GenericBinaryError>(
        "Linking data must come after code section", object_error::parse_failed);
  }

  while (Ptr < End) {
    uint8_t Type = readUint8(Ptr);
    uint32_t Size = readVaruint32(Ptr);
    const uint8_t *SubSectionEnd = Ptr + Size;
    switch (Type) {
    case wasm::WASM_SYMBOL_TABLE:
      if (Error Err = parseLinkingSectionSymtab(Ptr, SubSectionEnd))
        return Err;
      break;
    case wasm::WASM_SEGMENT_INFO: {
      uint32_t Count = readVaruint32(Ptr);
      if (Count > DataSegments.size())
        return make_error<GenericBinaryError>("Too many segment names",
                                              object_error::parse_failed);
      for (uint32_t i = 0; i < Count; i++) {
        DataSegments[i].Data.Name = readString(Ptr);
        DataSegments[i].Data.Alignment = readVaruint32(Ptr);
        DataSegments[i].Data.Flags = readVaruint32(Ptr);
      }
      break;
    }
    case wasm::WASM_INIT_FUNCS: {
      uint32_t Count = readVaruint32(Ptr);
      LinkingData.InitFunctions.reserve(Count);
      for (uint32_t i = 0; i < Count; i++) {
        wasm::WasmInitFunc Init;
        Init.Priority = readVaruint32(Ptr);
        Init.Symbol = readVaruint32(Ptr);
        if (!isValidFunctionSymbol(Init.Symbol))
          return make_error<GenericBinaryError>("Invalid function symbol: " +
                                                    Twine(Init.Symbol),
                                                object_error::parse_failed);
        LinkingData.InitFunctions.emplace_back(Init);
      }
      break;
    }
    case wasm::WASM_COMDAT_INFO:
      if (Error Err = parseLinkingSectionComdat(Ptr, SubSectionEnd))
        return Err;
      break;
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

Error WasmObjectFile::parseLinkingSectionSymtab(const uint8_t *&Ptr,
                                                const uint8_t *End) {
  uint32_t Count = readVaruint32(Ptr);
  LinkingData.SymbolTable.reserve(Count);
  Symbols.reserve(Count);
  StringSet<> SymbolNames;

  std::vector<wasm::WasmImport *> ImportedGlobals;
  std::vector<wasm::WasmImport *> ImportedFunctions;
  ImportedGlobals.reserve(Imports.size());
  ImportedFunctions.reserve(Imports.size());
  for (auto &I : Imports) {
    if (I.Kind == wasm::WASM_EXTERNAL_FUNCTION)
      ImportedFunctions.emplace_back(&I);
    else if (I.Kind == wasm::WASM_EXTERNAL_GLOBAL)
      ImportedGlobals.emplace_back(&I);
  }

  while (Count--) {
    wasm::WasmSymbolInfo Info;
    const wasm::WasmSignature *FunctionType = nullptr;
    const wasm::WasmGlobalType *GlobalType = nullptr;

    Info.Kind = readUint8(Ptr);
    Info.Flags = readVaruint32(Ptr);
    bool IsDefined = (Info.Flags & wasm::WASM_SYMBOL_UNDEFINED) == 0;

    switch (Info.Kind) {
    case wasm::WASM_SYMBOL_TYPE_FUNCTION:
      Info.ElementIndex = readVaruint32(Ptr);
      if (!isValidFunctionIndex(Info.ElementIndex) ||
          IsDefined != isDefinedFunctionIndex(Info.ElementIndex))
        return make_error<GenericBinaryError>("invalid function symbol index",
                                              object_error::parse_failed);
      if (IsDefined) {
        Info.Name = readString(Ptr);
        unsigned FuncIndex = Info.ElementIndex - NumImportedFunctions;
        FunctionType = &Signatures[FunctionTypes[FuncIndex]];
        wasm::WasmFunction &Function = Functions[FuncIndex];
        if (Function.Name.empty()) {
          // Use the symbol's name to set a name for the Function, but only if
          // one hasn't already been set.
          Function.Name = Info.Name;
        }
      } else {
        wasm::WasmImport &Import = *ImportedFunctions[Info.ElementIndex];
        FunctionType = &Signatures[Import.SigIndex];
        Info.Name = Import.Field;
      }
      break;

    case wasm::WASM_SYMBOL_TYPE_GLOBAL:
      Info.ElementIndex = readVaruint32(Ptr);
      if (!isValidGlobalIndex(Info.ElementIndex) ||
          IsDefined != isDefinedGlobalIndex(Info.ElementIndex))
        return make_error<GenericBinaryError>("invalid global symbol index",
                                              object_error::parse_failed);
      if (!IsDefined &&
          (Info.Flags & wasm::WASM_SYMBOL_BINDING_MASK) ==
              wasm::WASM_SYMBOL_BINDING_WEAK)
        return make_error<GenericBinaryError>("undefined weak global symbol",
                                              object_error::parse_failed);
      if (IsDefined) {
        Info.Name = readString(Ptr);
        unsigned GlobalIndex = Info.ElementIndex - NumImportedGlobals;
        wasm::WasmGlobal &Global = Globals[GlobalIndex];
        GlobalType = &Global.Type;
        if (Global.Name.empty()) {
          // Use the symbol's name to set a name for the Global, but only if
          // one hasn't already been set.
          Global.Name = Info.Name;
        }
      } else {
        wasm::WasmImport &Import = *ImportedGlobals[Info.ElementIndex];
        Info.Name = Import.Field;
        GlobalType = &Import.Global;
      }
      break;

    case wasm::WASM_SYMBOL_TYPE_DATA:
      Info.Name = readString(Ptr);
      if (IsDefined) {
        uint32_t Index = readVaruint32(Ptr);
        if (Index >= DataSegments.size())
          return make_error<GenericBinaryError>("invalid data symbol index",
                                                object_error::parse_failed);
        uint32_t Offset = readVaruint32(Ptr);
        uint32_t Size = readVaruint32(Ptr);
        if (Offset + Size > DataSegments[Index].Data.Content.size())
          return make_error<GenericBinaryError>("invalid data symbol offset",
                                                object_error::parse_failed);
        Info.DataRef = wasm::WasmDataReference{Index, Offset, Size};
      }
      break;

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
    Symbols.emplace_back(LinkingData.SymbolTable.back(), FunctionType,
                         GlobalType);
    DEBUG(dbgs() << "Adding symbol: " << Symbols.back() << "\n");
  }

  return Error::success();
}

Error WasmObjectFile::parseLinkingSectionComdat(const uint8_t *&Ptr,
                                                const uint8_t *End)
{
  uint32_t ComdatCount = readVaruint32(Ptr);
  StringSet<> ComdatSet;
  for (unsigned ComdatIndex = 0; ComdatIndex < ComdatCount; ++ComdatIndex) {
    StringRef Name = readString(Ptr);
    if (Name.empty() || !ComdatSet.insert(Name).second)
      return make_error<GenericBinaryError>("Bad/duplicate COMDAT name " + Twine(Name),
                                            object_error::parse_failed);
    LinkingData.Comdats.emplace_back(Name);
    uint32_t Flags = readVaruint32(Ptr);
    if (Flags != 0)
      return make_error<GenericBinaryError>("Unsupported COMDAT flags",
                                            object_error::parse_failed);

    uint32_t EntryCount = readVaruint32(Ptr);
    while (EntryCount--) {
      unsigned Kind = readVaruint32(Ptr);
      unsigned Index = readVaruint32(Ptr);
      switch (Kind) {
      default:
        return make_error<GenericBinaryError>("Invalid COMDAT entry type",
                                              object_error::parse_failed);
      case wasm::WASM_COMDAT_DATA:
        if (Index >= DataSegments.size())
          return make_error<GenericBinaryError>("COMDAT data index out of range",
                                                object_error::parse_failed);
        if (DataSegments[Index].Data.Comdat != UINT32_MAX)
          return make_error<GenericBinaryError>("Data segment in two COMDATs",
                                                object_error::parse_failed);
        DataSegments[Index].Data.Comdat = ComdatIndex;
        break;
      case wasm::WASM_COMDAT_FUNCTION:
        if (!isDefinedFunctionIndex(Index))
          return make_error<GenericBinaryError>("COMDAT function index out of range",
                                                object_error::parse_failed);
        if (getDefinedFunction(Index).Comdat != UINT32_MAX)
          return make_error<GenericBinaryError>("Function in two COMDATs",
                                                object_error::parse_failed);
        getDefinedFunction(Index).Comdat = ComdatIndex;
        break;
      }
    }
  }
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
  uint8_t SectionCode = readUint8(Ptr);
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
  uint32_t EndOffset = Section->Content.size();
  while (RelocCount--) {
    wasm::WasmRelocation Reloc = {};
    Reloc.Type = readVaruint32(Ptr);
    Reloc.Offset = readVaruint32(Ptr);
    Reloc.Index = readVaruint32(Ptr);
    switch (Reloc.Type) {
    case wasm::R_WEBASSEMBLY_FUNCTION_INDEX_LEB:
    case wasm::R_WEBASSEMBLY_TABLE_INDEX_SLEB:
    case wasm::R_WEBASSEMBLY_TABLE_INDEX_I32:
      if (!isValidFunctionSymbol(Reloc.Index))
        return make_error<GenericBinaryError>("Bad relocation function index",
                                              object_error::parse_failed);
      break;
    case wasm::R_WEBASSEMBLY_TYPE_INDEX_LEB:
      if (Reloc.Index >= Signatures.size())
        return make_error<GenericBinaryError>("Bad relocation type index",
                                              object_error::parse_failed);
      break;
    case wasm::R_WEBASSEMBLY_GLOBAL_INDEX_LEB:
      if (!isValidGlobalSymbol(Reloc.Index))
        return make_error<GenericBinaryError>("Bad relocation global index",
                                              object_error::parse_failed);
      break;
    case wasm::R_WEBASSEMBLY_MEMORY_ADDR_LEB:
    case wasm::R_WEBASSEMBLY_MEMORY_ADDR_SLEB:
    case wasm::R_WEBASSEMBLY_MEMORY_ADDR_I32:
      if (!isValidDataSymbol(Reloc.Index))
        return make_error<GenericBinaryError>("Bad relocation data index",
                                              object_error::parse_failed);
      Reloc.Addend = readVarint32(Ptr);
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
    if (Reloc.Type == wasm::R_WEBASSEMBLY_TABLE_INDEX_I32 ||
        Reloc.Type == wasm::R_WEBASSEMBLY_MEMORY_ADDR_I32)
      Size = 4;
    if (Reloc.Offset + Size > EndOffset)
      return make_error<GenericBinaryError>("Bad relocation offset",
                                            object_error::parse_failed);

    Section->Relocations.push_back(Reloc);
  }
  if (Ptr != End)
    return make_error<GenericBinaryError>("Reloc section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseCustomSection(WasmSection &Sec,
                                         const uint8_t *Ptr, const uint8_t *End) {
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
    uint8_t Form = readUint8(Ptr);
    if (Form != wasm::WASM_TYPE_FUNC) {
      return make_error<GenericBinaryError>("Invalid signature type",
                                            object_error::parse_failed);
    }
    uint32_t ParamCount = readVaruint32(Ptr);
    Sig.ParamTypes.reserve(ParamCount);
    while (ParamCount--) {
      uint32_t ParamType = readUint8(Ptr);
      Sig.ParamTypes.push_back(ParamType);
    }
    uint32_t ReturnCount = readVaruint32(Ptr);
    if (ReturnCount) {
      if (ReturnCount != 1) {
        return make_error<GenericBinaryError>(
            "Multiple return types not supported", object_error::parse_failed);
      }
      Sig.ReturnType = readUint8(Ptr);
    }
    Signatures.push_back(Sig);
  }
  if (Ptr != End)
    return make_error<GenericBinaryError>("Type section ended prematurely",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseImportSection(const uint8_t *Ptr, const uint8_t *End) {
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
      Im.Global.Type = readUint8(Ptr);
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
  uint32_t NumTypes = Signatures.size();
  while (Count--) {
    uint32_t Type = readVaruint32(Ptr);
    if (Type >= NumTypes)
      return make_error<GenericBinaryError>("Invalid function type",
                                            object_error::parse_failed);
    FunctionTypes.push_back(Type);
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
  GlobalSection = Sections.size();
  uint32_t Count = readVaruint32(Ptr);
  Globals.reserve(Count);
  while (Count--) {
    wasm::WasmGlobal Global;
    Global.Index = NumImportedGlobals + Globals.size();
    Global.Type.Type = readUint8(Ptr);
    Global.Type.Mutable = readVaruint1(Ptr);
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
  uint32_t Count = readVaruint32(Ptr);
  Exports.reserve(Count);
  for (uint32_t i = 0; i < Count; i++) {
    wasm::WasmExport Ex;
    Ex.Name = readString(Ptr);
    Ex.Kind = readUint8(Ptr);
    Ex.Index = readVaruint32(Ptr);
    switch (Ex.Kind) {
    case wasm::WASM_EXTERNAL_FUNCTION:
      if (!isValidFunctionIndex(Ex.Index))
        return make_error<GenericBinaryError>("Invalid function export",
                                              object_error::parse_failed);
      break;
    case wasm::WASM_EXTERNAL_GLOBAL:
      if (!isValidGlobalIndex(Ex.Index))
        return make_error<GenericBinaryError>("Invalid global export",
                                              object_error::parse_failed);
      break;
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

bool WasmObjectFile::isValidFunctionIndex(uint32_t Index) const {
  return Index < NumImportedFunctions + FunctionTypes.size();
}

bool WasmObjectFile::isDefinedFunctionIndex(uint32_t Index) const {
  return Index >= NumImportedFunctions && isValidFunctionIndex(Index);
}

bool WasmObjectFile::isValidGlobalIndex(uint32_t Index) const {
  return Index < NumImportedGlobals + Globals.size();
}

bool WasmObjectFile::isDefinedGlobalIndex(uint32_t Index) const {
  return Index >= NumImportedGlobals && isValidGlobalIndex(Index);
}

bool WasmObjectFile::isValidFunctionSymbol(uint32_t Index) const {
  return Index < Symbols.size() && Symbols[Index].isTypeFunction();
}

bool WasmObjectFile::isValidGlobalSymbol(uint32_t Index) const {
  return Index < Symbols.size() && Symbols[Index].isTypeGlobal();
}

bool WasmObjectFile::isValidDataSymbol(uint32_t Index) const {
  return Index < Symbols.size() && Symbols[Index].isTypeData();
}

wasm::WasmFunction &WasmObjectFile::getDefinedFunction(uint32_t Index) {
  assert(isDefinedFunctionIndex(Index));
  return Functions[Index - NumImportedFunctions];
}

wasm::WasmGlobal &WasmObjectFile::getDefinedGlobal(uint32_t Index) {
  assert(isDefinedGlobalIndex(Index));
  return Globals[Index - NumImportedGlobals];
}

Error WasmObjectFile::parseStartSection(const uint8_t *Ptr, const uint8_t *End) {
  StartFunction = readVaruint32(Ptr);
  if (!isValidFunctionIndex(StartFunction))
    return make_error<GenericBinaryError>("Invalid start function",
                                          object_error::parse_failed);
  return Error::success();
}

Error WasmObjectFile::parseCodeSection(const uint8_t *Ptr, const uint8_t *End) {
  CodeSection = Sections.size();
  const uint8_t *CodeSectionStart = Ptr;
  uint32_t FunctionCount = readVaruint32(Ptr);
  if (FunctionCount != FunctionTypes.size()) {
    return make_error<GenericBinaryError>("Invalid function count",
                                          object_error::parse_failed);
  }

  while (FunctionCount--) {
    wasm::WasmFunction Function;
    const uint8_t *FunctionStart = Ptr;
    uint32_t Size = readVaruint32(Ptr);
    const uint8_t *FunctionEnd = Ptr + Size;

    Function.Index = NumImportedFunctions + Functions.size();
    Function.CodeSectionOffset = FunctionStart - CodeSectionStart;
    Function.Size = FunctionEnd - FunctionStart;

    uint32_t NumLocalDecls = readVaruint32(Ptr);
    Function.Locals.reserve(NumLocalDecls);
    while (NumLocalDecls--) {
      wasm::WasmLocalDecl Decl;
      Decl.Count = readVaruint32(Ptr);
      Decl.Type = readUint8(Ptr);
      Function.Locals.push_back(Decl);
    }

    uint32_t BodySize = FunctionEnd - Ptr;
    Function.Body = ArrayRef<uint8_t>(Ptr, BodySize);
    // This will be set later when reading in the linking metadata section.
    Function.Comdat = UINT32_MAX;
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
  DataSection = Sections.size();
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
    // The rest of these Data fields are set later, when reading in the linking
    // metadata section.
    Segment.Data.Alignment = 0;
    Segment.Data.Flags = 0;
    Segment.Data.Comdat = UINT32_MAX;
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
  return getWasmSymbol(Symb).Info.Name;
}

Expected<uint64_t> WasmObjectFile::getSymbolAddress(DataRefImpl Symb) const {
  return getSymbolValue(Symb);
}

uint64_t WasmObjectFile::getWasmSymbolValue(const WasmSymbol& Sym) const {
  switch (Sym.Info.Kind) {
  case wasm::WASM_SYMBOL_TYPE_FUNCTION:
  case wasm::WASM_SYMBOL_TYPE_GLOBAL:
    return Sym.Info.ElementIndex;
  case wasm::WASM_SYMBOL_TYPE_DATA: {
    // The value of a data symbol is the segment offset, plus the symbol
    // offset within the segment.
    uint32_t SegmentIndex = Sym.Info.DataRef.Segment;
    const wasm::WasmDataSegment &Segment = DataSegments[SegmentIndex].Data;
    assert(Segment.Offset.Opcode == wasm::WASM_OPCODE_I32_CONST);
    return Segment.Offset.Value.Int32 + Sym.Info.DataRef.Offset;
  }
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
  }

  llvm_unreachable("Unknown WasmSymbol::SymbolType");
  return SymbolRef::ST_Other;
}

Expected<section_iterator>
WasmObjectFile::getSymbolSection(DataRefImpl Symb) const {
  const WasmSymbol& Sym = getWasmSymbol(Symb);
  if (Sym.isUndefined())
    return section_end();

  DataRefImpl Ref;
  switch (Sym.Info.Kind) {
  case wasm::WASM_SYMBOL_TYPE_FUNCTION:
    Ref.d.a = CodeSection;
    break;
  case wasm::WASM_SYMBOL_TYPE_GLOBAL:
    Ref.d.a = GlobalSection;
    break;
  case wasm::WASM_SYMBOL_TYPE_DATA:
    Ref.d.a = DataSection;
    break;
  default:
    llvm_unreachable("Unknown WasmSymbol::SymbolType");
  }
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

uint8_t WasmObjectFile::getBytesInAddress() const { return 4; }

StringRef WasmObjectFile::getFileFormatName() const { return "WASM"; }

Triple::ArchType WasmObjectFile::getArch() const { return Triple::wasm32; }

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
