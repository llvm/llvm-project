//===- lib/MC/WasmObjectWriter.cpp - Wasm File Writer ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Wasm object file writer information.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/BinaryFormat/Wasm.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSectionWasm.h"
#include "llvm/MC/MCSymbolWasm.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/MCWasmObjectWriter.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/StringSaver.h"
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "mc"

namespace {

// For patching purposes, we need to remember where each section starts, both
// for patching up the section size field, and for patching up references to
// locations within the section.
struct SectionBookkeeping {
  // Where the size of the section is written.
  uint64_t SizeOffset;
  // Where the contents of the section starts (after the header).
  uint64_t ContentsOffset;
};

// The signature of a wasm function, in a struct capable of being used as a
// DenseMap key.
struct WasmFunctionType {
  // Support empty and tombstone instances, needed by DenseMap.
  enum { Plain, Empty, Tombstone } State;

  // The return types of the function.
  SmallVector<wasm::ValType, 1> Returns;

  // The parameter types of the function.
  SmallVector<wasm::ValType, 4> Params;

  WasmFunctionType() : State(Plain) {}

  bool operator==(const WasmFunctionType &Other) const {
    return State == Other.State && Returns == Other.Returns &&
           Params == Other.Params;
  }
};

// Traits for using WasmFunctionType in a DenseMap.
struct WasmFunctionTypeDenseMapInfo {
  static WasmFunctionType getEmptyKey() {
    WasmFunctionType FuncTy;
    FuncTy.State = WasmFunctionType::Empty;
    return FuncTy;
  }
  static WasmFunctionType getTombstoneKey() {
    WasmFunctionType FuncTy;
    FuncTy.State = WasmFunctionType::Tombstone;
    return FuncTy;
  }
  static unsigned getHashValue(const WasmFunctionType &FuncTy) {
    uintptr_t Value = FuncTy.State;
    for (wasm::ValType Ret : FuncTy.Returns)
      Value += DenseMapInfo<int32_t>::getHashValue(int32_t(Ret));
    for (wasm::ValType Param : FuncTy.Params)
      Value += DenseMapInfo<int32_t>::getHashValue(int32_t(Param));
    return Value;
  }
  static bool isEqual(const WasmFunctionType &LHS,
                      const WasmFunctionType &RHS) {
    return LHS == RHS;
  }
};

// A wasm data segment.  A wasm binary contains only a single data section
// but that can contain many segments, each with their own virtual location
// in memory.  Each MCSection data created by llvm is modeled as its own
// wasm data segment.
struct WasmDataSegment {
  MCSectionWasm *Section;
  StringRef Name;
  uint32_t Offset;
  uint32_t Alignment;
  uint32_t Flags;
  SmallVector<char, 4> Data;
};

// A wasm import to be written into the import section.
struct WasmImport {
  StringRef ModuleName;
  StringRef FieldName;
  unsigned Kind;
  int32_t Type;
};

// A wasm function to be written into the function section.
struct WasmFunction {
  int32_t Type;
  const MCSymbolWasm *Sym;
};

// A wasm export to be written into the export section.
struct WasmExport {
  StringRef FieldName;
  unsigned Kind;
  uint32_t Index;
};

// A wasm global to be written into the global section.
struct WasmGlobal {
  wasm::ValType Type;
  bool IsMutable;
  bool HasImport;
  uint64_t InitialValue;
  uint32_t ImportIndex;
};

// Information about a single relocation.
struct WasmRelocationEntry {
  uint64_t Offset;                  // Where is the relocation.
  const MCSymbolWasm *Symbol;       // The symbol to relocate with.
  int64_t Addend;                   // A value to add to the symbol.
  unsigned Type;                    // The type of the relocation.
  const MCSectionWasm *FixupSection;// The section the relocation is targeting.

  WasmRelocationEntry(uint64_t Offset, const MCSymbolWasm *Symbol,
                      int64_t Addend, unsigned Type,
                      const MCSectionWasm *FixupSection)
      : Offset(Offset), Symbol(Symbol), Addend(Addend), Type(Type),
        FixupSection(FixupSection) {}

  bool hasAddend() const {
    switch (Type) {
    case wasm::R_WEBASSEMBLY_MEMORY_ADDR_LEB:
    case wasm::R_WEBASSEMBLY_MEMORY_ADDR_SLEB:
    case wasm::R_WEBASSEMBLY_MEMORY_ADDR_I32:
      return true;
    default:
      return false;
    }
  }

  void print(raw_ostream &Out) const {
    Out << "Off=" << Offset << ", Sym=" << *Symbol << ", Addend=" << Addend
        << ", Type=" << Type
        << ", FixupSection=" << FixupSection->getSectionName();
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const { print(dbgs()); }
#endif
};

#if !defined(NDEBUG)
raw_ostream &operator<<(raw_ostream &OS, const WasmRelocationEntry &Rel) {
  Rel.print(OS);
  return OS;
}
#endif

class WasmObjectWriter : public MCObjectWriter {
  /// Helper struct for containing some precomputed information on symbols.
  struct WasmSymbolData {
    const MCSymbolWasm *Symbol;
    StringRef Name;

    // Support lexicographic sorting.
    bool operator<(const WasmSymbolData &RHS) const { return Name < RHS.Name; }
  };

  /// The target specific Wasm writer instance.
  std::unique_ptr<MCWasmObjectTargetWriter> TargetObjectWriter;

  // Relocations for fixing up references in the code section.
  std::vector<WasmRelocationEntry> CodeRelocations;

  // Relocations for fixing up references in the data section.
  std::vector<WasmRelocationEntry> DataRelocations;

  // Index values to use for fixing up call_indirect type indices.
  // Maps function symbols to the index of the type of the function
  DenseMap<const MCSymbolWasm *, uint32_t> TypeIndices;
  // Maps function symbols to the table element index space. Used
  // for TABLE_INDEX relocation types (i.e. address taken functions).
  DenseMap<const MCSymbolWasm *, uint32_t> IndirectSymbolIndices;
  // Maps function/global symbols to the function/global index space.
  DenseMap<const MCSymbolWasm *, uint32_t> SymbolIndices;

  DenseMap<WasmFunctionType, int32_t, WasmFunctionTypeDenseMapInfo>
      FunctionTypeIndices;
  SmallVector<WasmFunctionType, 4> FunctionTypes;
  SmallVector<WasmGlobal, 4> Globals;
  unsigned NumGlobalImports = 0;

  // TargetObjectWriter wrappers.
  bool is64Bit() const { return TargetObjectWriter->is64Bit(); }
  unsigned getRelocType(const MCValue &Target, const MCFixup &Fixup) const {
    return TargetObjectWriter->getRelocType(Target, Fixup);
  }

  void startSection(SectionBookkeeping &Section, unsigned SectionId,
                    const char *Name = nullptr);
  void endSection(SectionBookkeeping &Section);

public:
  WasmObjectWriter(std::unique_ptr<MCWasmObjectTargetWriter> MOTW,
                   raw_pwrite_stream &OS)
      : MCObjectWriter(OS, /*IsLittleEndian=*/true),
        TargetObjectWriter(std::move(MOTW)) {}

private:
  ~WasmObjectWriter() override;

  void reset() override {
    CodeRelocations.clear();
    DataRelocations.clear();
    TypeIndices.clear();
    SymbolIndices.clear();
    IndirectSymbolIndices.clear();
    FunctionTypeIndices.clear();
    FunctionTypes.clear();
    Globals.clear();
    MCObjectWriter::reset();
    NumGlobalImports = 0;
  }

  void writeHeader(const MCAssembler &Asm);

  void recordRelocation(MCAssembler &Asm, const MCAsmLayout &Layout,
                        const MCFragment *Fragment, const MCFixup &Fixup,
                        MCValue Target, uint64_t &FixedValue) override;

  void executePostLayoutBinding(MCAssembler &Asm,
                                const MCAsmLayout &Layout) override;

  void writeObject(MCAssembler &Asm, const MCAsmLayout &Layout) override;

  void writeString(const StringRef Str) {
    encodeULEB128(Str.size(), getStream());
    writeBytes(Str);
  }

  void writeValueType(wasm::ValType Ty) {
    encodeSLEB128(int32_t(Ty), getStream());
  }

  void writeTypeSection(ArrayRef<WasmFunctionType> FunctionTypes);
  void writeImportSection(ArrayRef<WasmImport> Imports);
  void writeFunctionSection(ArrayRef<WasmFunction> Functions);
  void writeTableSection(uint32_t NumElements);
  void writeMemorySection(uint32_t DataSize);
  void writeGlobalSection();
  void writeExportSection(ArrayRef<WasmExport> Exports);
  void writeElemSection(ArrayRef<uint32_t> TableElems);
  void writeCodeSection(const MCAssembler &Asm, const MCAsmLayout &Layout,
                        ArrayRef<WasmFunction> Functions);
  void writeDataSection(ArrayRef<WasmDataSegment> Segments);
  void writeNameSection(ArrayRef<WasmFunction> Functions,
                        ArrayRef<WasmImport> Imports,
                        uint32_t NumFuncImports);
  void writeCodeRelocSection();
  void writeDataRelocSection();
  void writeLinkingMetaDataSection(
      ArrayRef<WasmDataSegment> Segments, uint32_t DataSize,
      SmallVector<std::pair<StringRef, uint32_t>, 4> SymbolFlags,
      bool HasStackPointer, uint32_t StackPointerGlobal);

  uint32_t getProvisionalValue(const WasmRelocationEntry &RelEntry);
  void applyRelocations(ArrayRef<WasmRelocationEntry> Relocations,
                        uint64_t ContentsOffset);

  void writeRelocations(ArrayRef<WasmRelocationEntry> Relocations);
  uint32_t getRelocationIndexValue(const WasmRelocationEntry &RelEntry);
  uint32_t getFunctionType(const MCSymbolWasm& Symbol);
  uint32_t registerFunctionType(const MCSymbolWasm& Symbol);
};

} // end anonymous namespace

WasmObjectWriter::~WasmObjectWriter() {}

// Write out a section header and a patchable section size field.
void WasmObjectWriter::startSection(SectionBookkeeping &Section,
                                    unsigned SectionId,
                                    const char *Name) {
  assert((Name != nullptr) == (SectionId == wasm::WASM_SEC_CUSTOM) &&
         "Only custom sections can have names");

  DEBUG(dbgs() << "startSection " << SectionId << ": " << Name << "\n");
  encodeULEB128(SectionId, getStream());

  Section.SizeOffset = getStream().tell();

  // The section size. We don't know the size yet, so reserve enough space
  // for any 32-bit value; we'll patch it later.
  encodeULEB128(UINT32_MAX, getStream());

  // The position where the section starts, for measuring its size.
  Section.ContentsOffset = getStream().tell();

  // Custom sections in wasm also have a string identifier.
  if (SectionId == wasm::WASM_SEC_CUSTOM) {
    assert(Name);
    writeString(StringRef(Name));
  }
}

// Now that the section is complete and we know how big it is, patch up the
// section size field at the start of the section.
void WasmObjectWriter::endSection(SectionBookkeeping &Section) {
  uint64_t Size = getStream().tell() - Section.ContentsOffset;
  if (uint32_t(Size) != Size)
    report_fatal_error("section size does not fit in a uint32_t");

  DEBUG(dbgs() << "endSection size=" << Size << "\n");

  // Write the final section size to the payload_len field, which follows
  // the section id byte.
  uint8_t Buffer[16];
  unsigned SizeLen = encodeULEB128(Size, Buffer, 5);
  assert(SizeLen == 5);
  getStream().pwrite((char *)Buffer, SizeLen, Section.SizeOffset);
}

// Emit the Wasm header.
void WasmObjectWriter::writeHeader(const MCAssembler &Asm) {
  writeBytes(StringRef(wasm::WasmMagic, sizeof(wasm::WasmMagic)));
  writeLE32(wasm::WasmVersion);
}

void WasmObjectWriter::executePostLayoutBinding(MCAssembler &Asm,
                                                const MCAsmLayout &Layout) {
}

void WasmObjectWriter::recordRelocation(MCAssembler &Asm,
                                        const MCAsmLayout &Layout,
                                        const MCFragment *Fragment,
                                        const MCFixup &Fixup, MCValue Target,
                                        uint64_t &FixedValue) {
  MCAsmBackend &Backend = Asm.getBackend();
  bool IsPCRel = Backend.getFixupKindInfo(Fixup.getKind()).Flags &
                 MCFixupKindInfo::FKF_IsPCRel;
  const auto &FixupSection = cast<MCSectionWasm>(*Fragment->getParent());
  uint64_t C = Target.getConstant();
  uint64_t FixupOffset = Layout.getFragmentOffset(Fragment) + Fixup.getOffset();
  MCContext &Ctx = Asm.getContext();

  if (const MCSymbolRefExpr *RefB = Target.getSymB()) {
    assert(RefB->getKind() == MCSymbolRefExpr::VK_None &&
           "Should not have constructed this");

    // Let A, B and C being the components of Target and R be the location of
    // the fixup. If the fixup is not pcrel, we want to compute (A - B + C).
    // If it is pcrel, we want to compute (A - B + C - R).

    // In general, Wasm has no relocations for -B. It can only represent (A + C)
    // or (A + C - R). If B = R + K and the relocation is not pcrel, we can
    // replace B to implement it: (A - R - K + C)
    if (IsPCRel) {
      Ctx.reportError(
          Fixup.getLoc(),
          "No relocation available to represent this relative expression");
      return;
    }

    const auto &SymB = cast<MCSymbolWasm>(RefB->getSymbol());

    if (SymB.isUndefined()) {
      Ctx.reportError(Fixup.getLoc(),
                      Twine("symbol '") + SymB.getName() +
                          "' can not be undefined in a subtraction expression");
      return;
    }

    assert(!SymB.isAbsolute() && "Should have been folded");
    const MCSection &SecB = SymB.getSection();
    if (&SecB != &FixupSection) {
      Ctx.reportError(Fixup.getLoc(),
                      "Cannot represent a difference across sections");
      return;
    }

    uint64_t SymBOffset = Layout.getSymbolOffset(SymB);
    uint64_t K = SymBOffset - FixupOffset;
    IsPCRel = true;
    C -= K;
  }

  // We either rejected the fixup or folded B into C at this point.
  const MCSymbolRefExpr *RefA = Target.getSymA();
  const auto *SymA = RefA ? cast<MCSymbolWasm>(&RefA->getSymbol()) : nullptr;

  if (SymA && SymA->isVariable()) {
    const MCExpr *Expr = SymA->getVariableValue();
    const auto *Inner = cast<MCSymbolRefExpr>(Expr);
    if (Inner->getKind() == MCSymbolRefExpr::VK_WEAKREF)
      llvm_unreachable("weakref used in reloc not yet implemented");
  }

  // Put any constant offset in an addend. Offsets can be negative, and
  // LLVM expects wrapping, in contrast to wasm's immediates which can't
  // be negative and don't wrap.
  FixedValue = 0;

  if (SymA)
    SymA->setUsedInReloc();

  assert(!IsPCRel);
  assert(SymA);

  unsigned Type = getRelocType(Target, Fixup);

  WasmRelocationEntry Rec(FixupOffset, SymA, C, Type, &FixupSection);
  DEBUG(dbgs() << "WasmReloc: " << Rec << "\n");

  if (FixupSection.hasInstructions())
    CodeRelocations.push_back(Rec);
  else
    DataRelocations.push_back(Rec);
}

// Write X as an (unsigned) LEB value at offset Offset in Stream, padded
// to allow patching.
static void
WritePatchableLEB(raw_pwrite_stream &Stream, uint32_t X, uint64_t Offset) {
  uint8_t Buffer[5];
  unsigned SizeLen = encodeULEB128(X, Buffer, 5);
  assert(SizeLen == 5);
  Stream.pwrite((char *)Buffer, SizeLen, Offset);
}

// Write X as an signed LEB value at offset Offset in Stream, padded
// to allow patching.
static void
WritePatchableSLEB(raw_pwrite_stream &Stream, int32_t X, uint64_t Offset) {
  uint8_t Buffer[5];
  unsigned SizeLen = encodeSLEB128(X, Buffer, 5);
  assert(SizeLen == 5);
  Stream.pwrite((char *)Buffer, SizeLen, Offset);
}

// Write X as a plain integer value at offset Offset in Stream.
static void WriteI32(raw_pwrite_stream &Stream, uint32_t X, uint64_t Offset) {
  uint8_t Buffer[4];
  support::endian::write32le(Buffer, X);
  Stream.pwrite((char *)Buffer, sizeof(Buffer), Offset);
}

static const MCSymbolWasm* ResolveSymbol(const MCSymbolWasm& Symbol) {
  if (Symbol.isVariable()) {
    const MCExpr *Expr = Symbol.getVariableValue();
    auto *Inner = cast<MCSymbolRefExpr>(Expr);
    return cast<MCSymbolWasm>(&Inner->getSymbol());
  }
  return &Symbol;
}

// Compute a value to write into the code at the location covered
// by RelEntry. This value isn't used by the static linker, since
// we have addends; it just serves to make the code more readable
// and to make standalone wasm modules directly usable.
uint32_t
WasmObjectWriter::getProvisionalValue(const WasmRelocationEntry &RelEntry) {
  const MCSymbolWasm *Sym = ResolveSymbol(*RelEntry.Symbol);

  // For undefined symbols, use a hopefully invalid value.
  if (!Sym->isDefined(/*SetUsed=*/false))
    return UINT32_MAX;

  uint32_t GlobalIndex = SymbolIndices[Sym];
  const WasmGlobal& Global = Globals[GlobalIndex - NumGlobalImports];
  uint64_t Address = Global.InitialValue + RelEntry.Addend;

  // Ignore overflow. LLVM allows address arithmetic to silently wrap.
  uint32_t Value = Address;

  return Value;
}

static void addData(SmallVectorImpl<char> &DataBytes,
                    MCSectionWasm &DataSection) {
  DEBUG(errs() << "addData: " << DataSection.getSectionName() << "\n");

  DataBytes.resize(alignTo(DataBytes.size(), DataSection.getAlignment()));

  for (const MCFragment &Frag : DataSection) {
    if (Frag.hasInstructions())
      report_fatal_error("only data supported in data sections");

    if (auto *Align = dyn_cast<MCAlignFragment>(&Frag)) {
      if (Align->getValueSize() != 1)
        report_fatal_error("only byte values supported for alignment");
      // If nops are requested, use zeros, as this is the data section.
      uint8_t Value = Align->hasEmitNops() ? 0 : Align->getValue();
      uint64_t Size = std::min<uint64_t>(alignTo(DataBytes.size(),
                                                 Align->getAlignment()),
                                         DataBytes.size() +
                                             Align->getMaxBytesToEmit());
      DataBytes.resize(Size, Value);
    } else if (auto *Fill = dyn_cast<MCFillFragment>(&Frag)) {
      DataBytes.insert(DataBytes.end(), Fill->getSize(), Fill->getValue());
    } else {
      const auto &DataFrag = cast<MCDataFragment>(Frag);
      const SmallVectorImpl<char> &Contents = DataFrag.getContents();

      DataBytes.insert(DataBytes.end(), Contents.begin(), Contents.end());
    }
  }

  DEBUG(dbgs() << "addData -> " << DataBytes.size() << "\n");
}

uint32_t WasmObjectWriter::getRelocationIndexValue(
    const WasmRelocationEntry &RelEntry) {
  switch (RelEntry.Type) {
  case wasm::R_WEBASSEMBLY_TABLE_INDEX_SLEB:
  case wasm::R_WEBASSEMBLY_TABLE_INDEX_I32:
    if (!IndirectSymbolIndices.count(RelEntry.Symbol))
      report_fatal_error("symbol not found table index space: " +
                         RelEntry.Symbol->getName());
    return IndirectSymbolIndices[RelEntry.Symbol];
  case wasm::R_WEBASSEMBLY_FUNCTION_INDEX_LEB:
  case wasm::R_WEBASSEMBLY_GLOBAL_INDEX_LEB:
  case wasm::R_WEBASSEMBLY_MEMORY_ADDR_LEB:
  case wasm::R_WEBASSEMBLY_MEMORY_ADDR_SLEB:
  case wasm::R_WEBASSEMBLY_MEMORY_ADDR_I32:
    if (!SymbolIndices.count(RelEntry.Symbol))
      report_fatal_error("symbol not found function/global index space: " +
                         RelEntry.Symbol->getName());
    return SymbolIndices[RelEntry.Symbol];
  case wasm::R_WEBASSEMBLY_TYPE_INDEX_LEB:
    if (!TypeIndices.count(RelEntry.Symbol))
      report_fatal_error("symbol not found in type index space: " +
                         RelEntry.Symbol->getName());
    return TypeIndices[RelEntry.Symbol];
  default:
    llvm_unreachable("invalid relocation type");
  }
}

// Apply the portions of the relocation records that we can handle ourselves
// directly.
void WasmObjectWriter::applyRelocations(
    ArrayRef<WasmRelocationEntry> Relocations, uint64_t ContentsOffset) {
  raw_pwrite_stream &Stream = getStream();
  for (const WasmRelocationEntry &RelEntry : Relocations) {
    uint64_t Offset = ContentsOffset +
                      RelEntry.FixupSection->getSectionOffset() +
                      RelEntry.Offset;

    DEBUG(dbgs() << "applyRelocation: " << RelEntry << "\n");
    switch (RelEntry.Type) {
    case wasm::R_WEBASSEMBLY_TABLE_INDEX_SLEB:
    case wasm::R_WEBASSEMBLY_FUNCTION_INDEX_LEB:
    case wasm::R_WEBASSEMBLY_TYPE_INDEX_LEB:
    case wasm::R_WEBASSEMBLY_GLOBAL_INDEX_LEB: {
      uint32_t Index = getRelocationIndexValue(RelEntry);
      WritePatchableSLEB(Stream, Index, Offset);
      break;
    }
    case wasm::R_WEBASSEMBLY_TABLE_INDEX_I32: {
      uint32_t Index = getRelocationIndexValue(RelEntry);
      WriteI32(Stream, Index, Offset);
      break;
    }
    case wasm::R_WEBASSEMBLY_MEMORY_ADDR_SLEB: {
      uint32_t Value = getProvisionalValue(RelEntry);
      WritePatchableSLEB(Stream, Value, Offset);
      break;
    }
    case wasm::R_WEBASSEMBLY_MEMORY_ADDR_LEB: {
      uint32_t Value = getProvisionalValue(RelEntry);
      WritePatchableLEB(Stream, Value, Offset);
      break;
    }
    case wasm::R_WEBASSEMBLY_MEMORY_ADDR_I32: {
      uint32_t Value = getProvisionalValue(RelEntry);
      WriteI32(Stream, Value, Offset);
      break;
    }
    default:
      llvm_unreachable("invalid relocation type");
    }
  }
}

// Write out the portions of the relocation records that the linker will
// need to handle.
void WasmObjectWriter::writeRelocations(
    ArrayRef<WasmRelocationEntry> Relocations) {
  raw_pwrite_stream &Stream = getStream();
  for (const WasmRelocationEntry& RelEntry : Relocations) {

    uint64_t Offset = RelEntry.Offset +
                      RelEntry.FixupSection->getSectionOffset();
    uint32_t Index = getRelocationIndexValue(RelEntry);

    encodeULEB128(RelEntry.Type, Stream);
    encodeULEB128(Offset, Stream);
    encodeULEB128(Index, Stream);
    if (RelEntry.hasAddend())
      encodeSLEB128(RelEntry.Addend, Stream);
  }
}

void WasmObjectWriter::writeTypeSection(
    ArrayRef<WasmFunctionType> FunctionTypes) {
  if (FunctionTypes.empty())
    return;

  SectionBookkeeping Section;
  startSection(Section, wasm::WASM_SEC_TYPE);

  encodeULEB128(FunctionTypes.size(), getStream());

  for (const WasmFunctionType &FuncTy : FunctionTypes) {
    encodeSLEB128(wasm::WASM_TYPE_FUNC, getStream());
    encodeULEB128(FuncTy.Params.size(), getStream());
    for (wasm::ValType Ty : FuncTy.Params)
      writeValueType(Ty);
    encodeULEB128(FuncTy.Returns.size(), getStream());
    for (wasm::ValType Ty : FuncTy.Returns)
      writeValueType(Ty);
  }

  endSection(Section);
}

void WasmObjectWriter::writeImportSection(ArrayRef<WasmImport> Imports) {
  if (Imports.empty())
    return;

  SectionBookkeeping Section;
  startSection(Section, wasm::WASM_SEC_IMPORT);

  encodeULEB128(Imports.size(), getStream());
  for (const WasmImport &Import : Imports) {
    writeString(Import.ModuleName);
    writeString(Import.FieldName);

    encodeULEB128(Import.Kind, getStream());

    switch (Import.Kind) {
    case wasm::WASM_EXTERNAL_FUNCTION:
      encodeULEB128(Import.Type, getStream());
      break;
    case wasm::WASM_EXTERNAL_GLOBAL:
      encodeSLEB128(int32_t(Import.Type), getStream());
      encodeULEB128(0, getStream()); // mutability
      break;
    default:
      llvm_unreachable("unsupported import kind");
    }
  }

  endSection(Section);
}

void WasmObjectWriter::writeFunctionSection(ArrayRef<WasmFunction> Functions) {
  if (Functions.empty())
    return;

  SectionBookkeeping Section;
  startSection(Section, wasm::WASM_SEC_FUNCTION);

  encodeULEB128(Functions.size(), getStream());
  for (const WasmFunction &Func : Functions)
    encodeULEB128(Func.Type, getStream());

  endSection(Section);
}

void WasmObjectWriter::writeTableSection(uint32_t NumElements) {
  // For now, always emit the table section, since indirect calls are not
  // valid without it. In the future, we could perhaps be more clever and omit
  // it if there are no indirect calls.

  SectionBookkeeping Section;
  startSection(Section, wasm::WASM_SEC_TABLE);

  encodeULEB128(1, getStream());                       // The number of tables.
                                                       // Fixed to 1 for now.
  encodeSLEB128(wasm::WASM_TYPE_ANYFUNC, getStream()); // Type of table
  encodeULEB128(0, getStream());                       // flags
  encodeULEB128(NumElements, getStream());             // initial

  endSection(Section);
}

void WasmObjectWriter::writeMemorySection(uint32_t DataSize) {
  // For now, always emit the memory section, since loads and stores are not
  // valid without it. In the future, we could perhaps be more clever and omit
  // it if there are no loads or stores.
  SectionBookkeeping Section;
  uint32_t NumPages = (DataSize + wasm::WasmPageSize - 1) / wasm::WasmPageSize;

  startSection(Section, wasm::WASM_SEC_MEMORY);
  encodeULEB128(1, getStream()); // number of memory spaces

  encodeULEB128(0, getStream()); // flags
  encodeULEB128(NumPages, getStream()); // initial

  endSection(Section);
}

void WasmObjectWriter::writeGlobalSection() {
  if (Globals.empty())
    return;

  SectionBookkeeping Section;
  startSection(Section, wasm::WASM_SEC_GLOBAL);

  encodeULEB128(Globals.size(), getStream());
  for (const WasmGlobal &Global : Globals) {
    writeValueType(Global.Type);
    write8(Global.IsMutable);

    if (Global.HasImport) {
      assert(Global.InitialValue == 0);
      write8(wasm::WASM_OPCODE_GET_GLOBAL);
      encodeULEB128(Global.ImportIndex, getStream());
    } else {
      assert(Global.ImportIndex == 0);
      write8(wasm::WASM_OPCODE_I32_CONST);
      encodeSLEB128(Global.InitialValue, getStream()); // offset
    }
    write8(wasm::WASM_OPCODE_END);
  }

  endSection(Section);
}

void WasmObjectWriter::writeExportSection(ArrayRef<WasmExport> Exports) {
  if (Exports.empty())
    return;

  SectionBookkeeping Section;
  startSection(Section, wasm::WASM_SEC_EXPORT);

  encodeULEB128(Exports.size(), getStream());
  for (const WasmExport &Export : Exports) {
    writeString(Export.FieldName);
    encodeSLEB128(Export.Kind, getStream());
    encodeULEB128(Export.Index, getStream());
  }

  endSection(Section);
}

void WasmObjectWriter::writeElemSection(ArrayRef<uint32_t> TableElems) {
  if (TableElems.empty())
    return;

  SectionBookkeeping Section;
  startSection(Section, wasm::WASM_SEC_ELEM);

  encodeULEB128(1, getStream()); // number of "segments"
  encodeULEB128(0, getStream()); // the table index

  // init expr for starting offset
  write8(wasm::WASM_OPCODE_I32_CONST);
  encodeSLEB128(0, getStream());
  write8(wasm::WASM_OPCODE_END);

  encodeULEB128(TableElems.size(), getStream());
  for (uint32_t Elem : TableElems)
    encodeULEB128(Elem, getStream());

  endSection(Section);
}

void WasmObjectWriter::writeCodeSection(const MCAssembler &Asm,
                                        const MCAsmLayout &Layout,
                                        ArrayRef<WasmFunction> Functions) {
  if (Functions.empty())
    return;

  SectionBookkeeping Section;
  startSection(Section, wasm::WASM_SEC_CODE);

  encodeULEB128(Functions.size(), getStream());

  for (const WasmFunction &Func : Functions) {
    auto &FuncSection = static_cast<MCSectionWasm &>(Func.Sym->getSection());

    int64_t Size = 0;
    if (!Func.Sym->getSize()->evaluateAsAbsolute(Size, Layout))
      report_fatal_error(".size expression must be evaluatable");

    encodeULEB128(Size, getStream());
    FuncSection.setSectionOffset(getStream().tell() - Section.ContentsOffset);
    Asm.writeSectionData(&FuncSection, Layout);
  }

  // Apply fixups.
  applyRelocations(CodeRelocations, Section.ContentsOffset);

  endSection(Section);
}

void WasmObjectWriter::writeDataSection(ArrayRef<WasmDataSegment> Segments) {
  if (Segments.empty())
    return;

  SectionBookkeeping Section;
  startSection(Section, wasm::WASM_SEC_DATA);

  encodeULEB128(Segments.size(), getStream()); // count

  for (const WasmDataSegment & Segment : Segments) {
    encodeULEB128(0, getStream()); // memory index
    write8(wasm::WASM_OPCODE_I32_CONST);
    encodeSLEB128(Segment.Offset, getStream()); // offset
    write8(wasm::WASM_OPCODE_END);
    encodeULEB128(Segment.Data.size(), getStream()); // size
    Segment.Section->setSectionOffset(getStream().tell() - Section.ContentsOffset);
    writeBytes(Segment.Data); // data
  }

  // Apply fixups.
  applyRelocations(DataRelocations, Section.ContentsOffset);

  endSection(Section);
}

void WasmObjectWriter::writeNameSection(
    ArrayRef<WasmFunction> Functions,
    ArrayRef<WasmImport> Imports,
    unsigned NumFuncImports) {
  uint32_t TotalFunctions = NumFuncImports + Functions.size();
  if (TotalFunctions == 0)
    return;

  SectionBookkeeping Section;
  startSection(Section, wasm::WASM_SEC_CUSTOM, "name");
  SectionBookkeeping SubSection;
  startSection(SubSection, wasm::WASM_NAMES_FUNCTION);

  encodeULEB128(TotalFunctions, getStream());
  uint32_t Index = 0;
  for (const WasmImport &Import : Imports) {
    if (Import.Kind == wasm::WASM_EXTERNAL_FUNCTION) {
      encodeULEB128(Index, getStream());
      writeString(Import.FieldName);
      ++Index;
    }
  }
  for (const WasmFunction &Func : Functions) {
    encodeULEB128(Index, getStream());
    writeString(Func.Sym->getName());
    ++Index;
  }

  endSection(SubSection);
  endSection(Section);
}

void WasmObjectWriter::writeCodeRelocSection() {
  // See: https://github.com/WebAssembly/tool-conventions/blob/master/Linking.md
  // for descriptions of the reloc sections.

  if (CodeRelocations.empty())
    return;

  SectionBookkeeping Section;
  startSection(Section, wasm::WASM_SEC_CUSTOM, "reloc.CODE");

  encodeULEB128(wasm::WASM_SEC_CODE, getStream());
  encodeULEB128(CodeRelocations.size(), getStream());

  writeRelocations(CodeRelocations);

  endSection(Section);
}

void WasmObjectWriter::writeDataRelocSection() {
  // See: https://github.com/WebAssembly/tool-conventions/blob/master/Linking.md
  // for descriptions of the reloc sections.

  if (DataRelocations.empty())
    return;

  SectionBookkeeping Section;
  startSection(Section, wasm::WASM_SEC_CUSTOM, "reloc.DATA");

  encodeULEB128(wasm::WASM_SEC_DATA, getStream());
  encodeULEB128(DataRelocations.size(), getStream());

  writeRelocations(DataRelocations);

  endSection(Section);
}

void WasmObjectWriter::writeLinkingMetaDataSection(
    ArrayRef<WasmDataSegment> Segments, uint32_t DataSize,
    SmallVector<std::pair<StringRef, uint32_t>, 4> SymbolFlags,
    bool HasStackPointer, uint32_t StackPointerGlobal) {
  SectionBookkeeping Section;
  startSection(Section, wasm::WASM_SEC_CUSTOM, "linking");
  SectionBookkeeping SubSection;

  if (HasStackPointer) {
    startSection(SubSection, wasm::WASM_STACK_POINTER);
    encodeULEB128(StackPointerGlobal, getStream()); // id
    endSection(SubSection);
  }

  if (SymbolFlags.size() != 0) {
    startSection(SubSection, wasm::WASM_SYMBOL_INFO);
    encodeULEB128(SymbolFlags.size(), getStream());
    for (auto Pair: SymbolFlags) {
      writeString(Pair.first);
      encodeULEB128(Pair.second, getStream());
    }
    endSection(SubSection);
  }

  if (DataSize > 0) {
    startSection(SubSection, wasm::WASM_DATA_SIZE);
    encodeULEB128(DataSize, getStream());
    endSection(SubSection);
  }

  if (Segments.size()) {
    startSection(SubSection, wasm::WASM_SEGMENT_INFO);
    encodeULEB128(Segments.size(), getStream());
    for (const WasmDataSegment &Segment : Segments) {
      writeString(Segment.Name);
      encodeULEB128(Segment.Alignment, getStream());
      encodeULEB128(Segment.Flags, getStream());
    }
    endSection(SubSection);
  }

  endSection(Section);
}

uint32_t WasmObjectWriter::getFunctionType(const MCSymbolWasm& Symbol) {
  assert(Symbol.isFunction());
  assert(TypeIndices.count(&Symbol));
  return TypeIndices[&Symbol];
}

uint32_t WasmObjectWriter::registerFunctionType(const MCSymbolWasm& Symbol) {
  assert(Symbol.isFunction());

  WasmFunctionType F;
  const MCSymbolWasm* ResolvedSym = ResolveSymbol(Symbol);
  F.Returns = ResolvedSym->getReturns();
  F.Params = ResolvedSym->getParams();

  auto Pair =
      FunctionTypeIndices.insert(std::make_pair(F, FunctionTypes.size()));
  if (Pair.second)
    FunctionTypes.push_back(F);
  TypeIndices[&Symbol] = Pair.first->second;

  DEBUG(dbgs() << "registerFunctionType: " << Symbol << " new:" << Pair.second << "\n");
  DEBUG(dbgs() << "  -> type index: " << Pair.first->second << "\n");
  return Pair.first->second;
}

void WasmObjectWriter::writeObject(MCAssembler &Asm,
                                   const MCAsmLayout &Layout) {
  DEBUG(dbgs() << "WasmObjectWriter::writeObject\n");
  MCContext &Ctx = Asm.getContext();
  wasm::ValType PtrType = is64Bit() ? wasm::ValType::I64 : wasm::ValType::I32;

  // Collect information from the available symbols.
  SmallVector<WasmFunction, 4> Functions;
  SmallVector<uint32_t, 4> TableElems;
  SmallVector<WasmImport, 4> Imports;
  SmallVector<WasmExport, 4> Exports;
  SmallVector<std::pair<StringRef, uint32_t>, 4> SymbolFlags;
  SmallPtrSet<const MCSymbolWasm *, 4> IsAddressTaken;
  unsigned NumFuncImports = 0;
  SmallVector<WasmDataSegment, 4> DataSegments;
  uint32_t StackPointerGlobal = 0;
  uint32_t DataSize = 0;
  bool HasStackPointer = false;

  // Populate the IsAddressTaken set.
  for (const WasmRelocationEntry &RelEntry : CodeRelocations) {
    switch (RelEntry.Type) {
    case wasm::R_WEBASSEMBLY_TABLE_INDEX_SLEB:
    case wasm::R_WEBASSEMBLY_MEMORY_ADDR_SLEB:
      IsAddressTaken.insert(RelEntry.Symbol);
      break;
    default:
      break;
    }
  }
  for (const WasmRelocationEntry &RelEntry : DataRelocations) {
    switch (RelEntry.Type) {
    case wasm::R_WEBASSEMBLY_TABLE_INDEX_I32:
    case wasm::R_WEBASSEMBLY_MEMORY_ADDR_I32:
      IsAddressTaken.insert(RelEntry.Symbol);
      break;
    default:
      break;
    }
  }

  // Populate FunctionTypeIndices and Imports.
  for (const MCSymbol &S : Asm.symbols()) {
    const auto &WS = static_cast<const MCSymbolWasm &>(S);

    if (WS.isTemporary())
      continue;

    if (WS.isFunction())
      registerFunctionType(WS);

    // If the symbol is not defined in this translation unit, import it.
    if (!WS.isDefined(/*SetUsed=*/false)) {
      WasmImport Import;
      Import.ModuleName = WS.getModuleName();
      Import.FieldName = WS.getName();

      if (WS.isFunction()) {
        Import.Kind = wasm::WASM_EXTERNAL_FUNCTION;
        Import.Type = getFunctionType(WS);
        SymbolIndices[&WS] = NumFuncImports;
        ++NumFuncImports;
      } else {
        Import.Kind = wasm::WASM_EXTERNAL_GLOBAL;
        Import.Type = int32_t(PtrType);
        SymbolIndices[&WS] = NumGlobalImports;
        ++NumGlobalImports;
      }

      Imports.push_back(Import);
    }
  }

  // In the special .global_variables section, we've encoded global
  // variables used by the function. Translate them into the Globals
  // list.
  MCSectionWasm *GlobalVars = Ctx.getWasmSection(".global_variables", wasm::WASM_SEC_DATA);
  if (!GlobalVars->getFragmentList().empty()) {
    if (GlobalVars->getFragmentList().size() != 1)
      report_fatal_error("only one .global_variables fragment supported");
    const MCFragment &Frag = *GlobalVars->begin();
    if (Frag.hasInstructions() || Frag.getKind() != MCFragment::FT_Data)
      report_fatal_error("only data supported in .global_variables");
    const auto &DataFrag = cast<MCDataFragment>(Frag);
    if (!DataFrag.getFixups().empty())
      report_fatal_error("fixups not supported in .global_variables");
    const SmallVectorImpl<char> &Contents = DataFrag.getContents();
    for (const uint8_t *p = (const uint8_t *)Contents.data(),
                     *end = (const uint8_t *)Contents.data() + Contents.size();
         p != end; ) {
      WasmGlobal G;
      if (end - p < 3)
        report_fatal_error("truncated global variable encoding");
      G.Type = wasm::ValType(int8_t(*p++));
      G.IsMutable = bool(*p++);
      G.HasImport = bool(*p++);
      if (G.HasImport) {
        G.InitialValue = 0;

        WasmImport Import;
        Import.ModuleName = (const char *)p;
        const uint8_t *nul = (const uint8_t *)memchr(p, '\0', end - p);
        if (!nul)
          report_fatal_error("global module name must be nul-terminated");
        p = nul + 1;
        nul = (const uint8_t *)memchr(p, '\0', end - p);
        if (!nul)
          report_fatal_error("global base name must be nul-terminated");
        Import.FieldName = (const char *)p;
        p = nul + 1;

        Import.Kind = wasm::WASM_EXTERNAL_GLOBAL;
        Import.Type = int32_t(G.Type);

        G.ImportIndex = NumGlobalImports;
        ++NumGlobalImports;

        Imports.push_back(Import);
      } else {
        unsigned n;
        G.InitialValue = decodeSLEB128(p, &n);
        G.ImportIndex = 0;
        if ((ptrdiff_t)n > end - p)
          report_fatal_error("global initial value must be valid SLEB128");
        p += n;
      }
      Globals.push_back(G);
    }
  }

  // In the special .stack_pointer section, we've encoded the stack pointer
  // index.
  MCSectionWasm *StackPtr = Ctx.getWasmSection(".stack_pointer", wasm::WASM_SEC_DATA);
  if (!StackPtr->getFragmentList().empty()) {
    if (StackPtr->getFragmentList().size() != 1)
      report_fatal_error("only one .stack_pointer fragment supported");
    const MCFragment &Frag = *StackPtr->begin();
    if (Frag.hasInstructions() || Frag.getKind() != MCFragment::FT_Data)
      report_fatal_error("only data supported in .stack_pointer");
    const auto &DataFrag = cast<MCDataFragment>(Frag);
    if (!DataFrag.getFixups().empty())
      report_fatal_error("fixups not supported in .stack_pointer");
    const SmallVectorImpl<char> &Contents = DataFrag.getContents();
    if (Contents.size() != 4)
      report_fatal_error("only one entry supported in .stack_pointer");
    HasStackPointer = true;
    StackPointerGlobal = NumGlobalImports + *(const int32_t *)Contents.data();
  }

  for (MCSection &Sec : Asm) {
    auto &Section = static_cast<MCSectionWasm &>(Sec);
    if (Section.getType() != wasm::WASM_SEC_DATA)
      continue;

    DataSize = alignTo(DataSize, Section.getAlignment());
    DataSegments.emplace_back();
    WasmDataSegment &Segment = DataSegments.back();
    Segment.Name = Section.getSectionName();
    Segment.Offset = DataSize;
    Segment.Section = &Section;
    addData(Segment.Data, Section);
    Segment.Alignment = Section.getAlignment();
    Segment.Flags = 0;
    DataSize += Segment.Data.size();
    Section.setMemoryOffset(Segment.Offset);
  }

  // Handle regular defined and undefined symbols.
  for (const MCSymbol &S : Asm.symbols()) {
    // Ignore unnamed temporary symbols, which aren't ever exported, imported,
    // or used in relocations.
    if (S.isTemporary() && S.getName().empty())
      continue;

    const auto &WS = static_cast<const MCSymbolWasm &>(S);
    DEBUG(dbgs() << "MCSymbol: '" << S << "'"
                 << " isDefined=" << S.isDefined() << " isExternal="
                 << S.isExternal() << " isTemporary=" << S.isTemporary()
                 << " isFunction=" << WS.isFunction()
                 << " isWeak=" << WS.isWeak()
                 << " isVariable=" << WS.isVariable() << "\n");

    if (WS.isWeak())
      SymbolFlags.emplace_back(WS.getName(), wasm::WASM_SYMBOL_BINDING_WEAK);

    if (WS.isVariable())
      continue;

    unsigned Index;

    if (WS.isFunction()) {
      if (WS.isDefined(/*SetUsed=*/false)) {
        if (WS.getOffset() != 0)
          report_fatal_error(
              "function sections must contain one function each");

        if (WS.getSize() == 0)
          report_fatal_error(
              "function symbols must have a size set with .size");

        // A definition. Take the next available index.
        Index = NumFuncImports + Functions.size();

        // Prepare the function.
        WasmFunction Func;
        Func.Type = getFunctionType(WS);
        Func.Sym = &WS;
        SymbolIndices[&WS] = Index;
        Functions.push_back(Func);
      } else {
        // An import; the index was assigned above.
        Index = SymbolIndices.find(&WS)->second;
      }

      DEBUG(dbgs() << "  -> function index: " << Index << "\n");

      // If needed, prepare the function to be called indirectly.
      if (IsAddressTaken.count(&WS) != 0) {
        IndirectSymbolIndices[&WS] = TableElems.size();
        DEBUG(dbgs() << "  -> adding to table: " << TableElems.size() << "\n");
        TableElems.push_back(Index);
      }
    } else {
      if (WS.isTemporary() && !WS.getSize())
        continue;

      if (!WS.isDefined(/*SetUsed=*/false))
        continue;

      if (!WS.getSize())
        report_fatal_error("data symbols must have a size set with .size: " +
                           WS.getName());

      int64_t Size = 0;
      if (!WS.getSize()->evaluateAsAbsolute(Size, Layout))
        report_fatal_error(".size expression must be evaluatable");

      // For each global, prepare a corresponding wasm global holding its
      // address.  For externals these will also be named exports.
      Index = NumGlobalImports + Globals.size();
      auto &DataSection = static_cast<MCSectionWasm &>(WS.getSection());

      WasmGlobal Global;
      Global.Type = PtrType;
      Global.IsMutable = false;
      Global.HasImport = false;
      Global.InitialValue = DataSection.getMemoryOffset() + Layout.getSymbolOffset(WS);
      Global.ImportIndex = 0;
      SymbolIndices[&WS] = Index;
      DEBUG(dbgs() << "  -> global index: " << Index << "\n");
      Globals.push_back(Global);
    }

    // If the symbol is visible outside this translation unit, export it.
    if (WS.isDefined(/*SetUsed=*/false)) {
      WasmExport Export;
      Export.FieldName = WS.getName();
      Export.Index = Index;
      if (WS.isFunction())
        Export.Kind = wasm::WASM_EXTERNAL_FUNCTION;
      else
        Export.Kind = wasm::WASM_EXTERNAL_GLOBAL;
      DEBUG(dbgs() << "  -> export " << Exports.size() << "\n");
      Exports.push_back(Export);
      if (!WS.isExternal())
        SymbolFlags.emplace_back(WS.getName(), wasm::WASM_SYMBOL_BINDING_LOCAL);
    }
  }

  // Handle weak aliases. We need to process these in a separate pass because
  // we need to have processed the target of the alias before the alias itself
  // and the symbols are not necessarily ordered in this way.
  for (const MCSymbol &S : Asm.symbols()) {
    if (!S.isVariable())
      continue;

    assert(S.isDefined(/*SetUsed=*/false));

    // Find the target symbol of this weak alias and export that index
    const auto &WS = static_cast<const MCSymbolWasm &>(S);
    const MCSymbolWasm *ResolvedSym = ResolveSymbol(WS);
    DEBUG(dbgs() << WS.getName() << ": weak alias of '" << *ResolvedSym << "'\n");
    assert(SymbolIndices.count(ResolvedSym) > 0);
    uint32_t Index = SymbolIndices.find(ResolvedSym)->second;
    DEBUG(dbgs() << "  -> index:" << Index << "\n");

    SymbolIndices[&WS] = Index;
    WasmExport Export;
    Export.FieldName = WS.getName();
    Export.Index = Index;
    if (WS.isFunction())
      Export.Kind = wasm::WASM_EXTERNAL_FUNCTION;
    else
      Export.Kind = wasm::WASM_EXTERNAL_GLOBAL;
    DEBUG(dbgs() << "  -> export " << Exports.size() << "\n");
    Exports.push_back(Export);

    if (!WS.isExternal())
      SymbolFlags.emplace_back(WS.getName(), wasm::WASM_SYMBOL_BINDING_LOCAL);
  }

  // Add types for indirect function calls.
  for (const WasmRelocationEntry &Fixup : CodeRelocations) {
    if (Fixup.Type != wasm::R_WEBASSEMBLY_TYPE_INDEX_LEB)
      continue;

    registerFunctionType(*Fixup.Symbol);
  }

  // Write out the Wasm header.
  writeHeader(Asm);

  writeTypeSection(FunctionTypes);
  writeImportSection(Imports);
  writeFunctionSection(Functions);
  writeTableSection(TableElems.size());
  writeMemorySection(DataSize);
  writeGlobalSection();
  writeExportSection(Exports);
  // TODO: Start Section
  writeElemSection(TableElems);
  writeCodeSection(Asm, Layout, Functions);
  writeDataSection(DataSegments);
  writeNameSection(Functions, Imports, NumFuncImports);
  writeCodeRelocSection();
  writeDataRelocSection();
  writeLinkingMetaDataSection(DataSegments, DataSize, SymbolFlags,
                              HasStackPointer, StackPointerGlobal);

  // TODO: Translate the .comment section to the output.
  // TODO: Translate debug sections to the output.
}

std::unique_ptr<MCObjectWriter>
llvm::createWasmObjectWriter(std::unique_ptr<MCWasmObjectTargetWriter> MOTW,
                             raw_pwrite_stream &OS) {
  // FIXME: Can't use make_unique<WasmObjectWriter>(...) as WasmObjectWriter's
  //        destructor is private. Is that necessary?
  return std::unique_ptr<MCObjectWriter>(
      new WasmObjectWriter(std::move(MOTW), OS));
}
