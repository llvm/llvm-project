//===- lib/MC/WasmObjectWriter.cpp - Wasm File Writer ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Wasm object file writer information.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/BinaryFormat/Wasm.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSectionWasm.h"
#include "llvm/MC/MCSymbolWasm.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/MCWasmObjectWriter.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/StringSaver.h"
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "mc"

namespace {

// Went we ceate the indirect function table we start at 1, so that there is
// and emtpy slot at 0 and therefore calling a null function pointer will trap.
static const uint32_t InitialTableOffset = 1;

// For patching purposes, we need to remember where each section starts, both
// for patching up the section size field, and for patching up references to
// locations within the section.
struct SectionBookkeeping {
  // Where the size of the section is written.
  uint64_t SizeOffset;
  // Where the section header ends (without custom section name).
  uint64_t PayloadOffset;
  // Where the contents of the section starts.
  uint64_t ContentsOffset;
  uint32_t Index;
};

// The signature of a wasm function or event, in a struct capable of being used
// as a DenseMap key.
// TODO: Consider using wasm::WasmSignature directly instead.
struct WasmSignature {
  // Support empty and tombstone instances, needed by DenseMap.
  enum { Plain, Empty, Tombstone } State = Plain;

  // The return types of the function.
  SmallVector<wasm::ValType, 1> Returns;

  // The parameter types of the function.
  SmallVector<wasm::ValType, 4> Params;

  bool operator==(const WasmSignature &Other) const {
    return State == Other.State && Returns == Other.Returns &&
           Params == Other.Params;
  }
};

// Traits for using WasmSignature in a DenseMap.
struct WasmSignatureDenseMapInfo {
  static WasmSignature getEmptyKey() {
    WasmSignature Sig;
    Sig.State = WasmSignature::Empty;
    return Sig;
  }
  static WasmSignature getTombstoneKey() {
    WasmSignature Sig;
    Sig.State = WasmSignature::Tombstone;
    return Sig;
  }
  static unsigned getHashValue(const WasmSignature &Sig) {
    uintptr_t Value = Sig.State;
    for (wasm::ValType Ret : Sig.Returns)
      Value += DenseMapInfo<uint32_t>::getHashValue(uint32_t(Ret));
    for (wasm::ValType Param : Sig.Params)
      Value += DenseMapInfo<uint32_t>::getHashValue(uint32_t(Param));
    return Value;
  }
  static bool isEqual(const WasmSignature &LHS, const WasmSignature &RHS) {
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
  uint32_t InitFlags;
  uint64_t Offset;
  uint32_t Alignment;
  uint32_t LinkerFlags;
  SmallVector<char, 4> Data;
};

// A wasm function to be written into the function section.
struct WasmFunction {
  uint32_t SigIndex;
  const MCSymbolWasm *Sym;
};

// A wasm global to be written into the global section.
struct WasmGlobal {
  wasm::WasmGlobalType Type;
  uint64_t InitialValue;
};

// Information about a single item which is part of a COMDAT.  For each data
// segment or function which is in the COMDAT, there is a corresponding
// WasmComdatEntry.
struct WasmComdatEntry {
  unsigned Kind;
  uint32_t Index;
};

// Information about a single relocation.
struct WasmRelocationEntry {
  uint64_t Offset;                   // Where is the relocation.
  const MCSymbolWasm *Symbol;        // The symbol to relocate with.
  int64_t Addend;                    // A value to add to the symbol.
  unsigned Type;                     // The type of the relocation.
  const MCSectionWasm *FixupSection; // The section the relocation is targeting.

  WasmRelocationEntry(uint64_t Offset, const MCSymbolWasm *Symbol,
                      int64_t Addend, unsigned Type,
                      const MCSectionWasm *FixupSection)
      : Offset(Offset), Symbol(Symbol), Addend(Addend), Type(Type),
        FixupSection(FixupSection) {}

  bool hasAddend() const { return wasm::relocTypeHasAddend(Type); }

  void print(raw_ostream &Out) const {
    Out << wasm::relocTypetoString(Type) << " Off=" << Offset
        << ", Sym=" << *Symbol << ", Addend=" << Addend
        << ", FixupSection=" << FixupSection->getName();
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const { print(dbgs()); }
#endif
};

static const uint32_t InvalidIndex = -1;

struct WasmCustomSection {

  StringRef Name;
  MCSectionWasm *Section;

  uint32_t OutputContentsOffset;
  uint32_t OutputIndex;

  WasmCustomSection(StringRef Name, MCSectionWasm *Section)
      : Name(Name), Section(Section), OutputContentsOffset(0),
        OutputIndex(InvalidIndex) {}
};

#if !defined(NDEBUG)
raw_ostream &operator<<(raw_ostream &OS, const WasmRelocationEntry &Rel) {
  Rel.print(OS);
  return OS;
}
#endif

// Write X as an (unsigned) LEB value at offset Offset in Stream, padded
// to allow patching.
template <int W>
void writePatchableLEB(raw_pwrite_stream &Stream, uint64_t X, uint64_t Offset) {
  uint8_t Buffer[W];
  unsigned SizeLen = encodeULEB128(X, Buffer, W);
  assert(SizeLen == W);
  Stream.pwrite((char *)Buffer, SizeLen, Offset);
}

// Write X as an signed LEB value at offset Offset in Stream, padded
// to allow patching.
template <int W>
void writePatchableSLEB(raw_pwrite_stream &Stream, int64_t X, uint64_t Offset) {
  uint8_t Buffer[W];
  unsigned SizeLen = encodeSLEB128(X, Buffer, W);
  assert(SizeLen == W);
  Stream.pwrite((char *)Buffer, SizeLen, Offset);
}

// Write X as a plain integer value at offset Offset in Stream.
static void patchI32(raw_pwrite_stream &Stream, uint32_t X, uint64_t Offset) {
  uint8_t Buffer[4];
  support::endian::write32le(Buffer, X);
  Stream.pwrite((char *)Buffer, sizeof(Buffer), Offset);
}

static void patchI64(raw_pwrite_stream &Stream, uint64_t X, uint64_t Offset) {
  uint8_t Buffer[8];
  support::endian::write64le(Buffer, X);
  Stream.pwrite((char *)Buffer, sizeof(Buffer), Offset);
}

class WasmObjectWriter : public MCObjectWriter {
  support::endian::Writer W;

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
  DenseMap<const MCSymbolWasm *, uint32_t> TableIndices;
  // Maps function/global symbols to the function/global/event/section index
  // space.
  DenseMap<const MCSymbolWasm *, uint32_t> WasmIndices;
  DenseMap<const MCSymbolWasm *, uint32_t> GOTIndices;
  // Maps data symbols to the Wasm segment and offset/size with the segment.
  DenseMap<const MCSymbolWasm *, wasm::WasmDataReference> DataLocations;

  // Stores output data (index, relocations, content offset) for custom
  // section.
  std::vector<WasmCustomSection> CustomSections;
  std::unique_ptr<WasmCustomSection> ProducersSection;
  std::unique_ptr<WasmCustomSection> TargetFeaturesSection;
  // Relocations for fixing up references in the custom sections.
  DenseMap<const MCSectionWasm *, std::vector<WasmRelocationEntry>>
      CustomSectionsRelocations;

  // Map from section to defining function symbol.
  DenseMap<const MCSection *, const MCSymbol *> SectionFunctions;

  DenseMap<WasmSignature, uint32_t, WasmSignatureDenseMapInfo> SignatureIndices;
  SmallVector<WasmSignature, 4> Signatures;
  SmallVector<WasmDataSegment, 4> DataSegments;
  unsigned NumFunctionImports = 0;
  unsigned NumGlobalImports = 0;
  unsigned NumEventImports = 0;
  uint32_t SectionCount = 0;

  // TargetObjectWriter wrappers.
  bool is64Bit() const { return TargetObjectWriter->is64Bit(); }
  bool isEmscripten() const { return TargetObjectWriter->isEmscripten(); }

  void startSection(SectionBookkeeping &Section, unsigned SectionId);
  void startCustomSection(SectionBookkeeping &Section, StringRef Name);
  void endSection(SectionBookkeeping &Section);

public:
  WasmObjectWriter(std::unique_ptr<MCWasmObjectTargetWriter> MOTW,
                   raw_pwrite_stream &OS)
      : W(OS, support::little), TargetObjectWriter(std::move(MOTW)) {}

private:
  void reset() override {
    CodeRelocations.clear();
    DataRelocations.clear();
    TypeIndices.clear();
    WasmIndices.clear();
    GOTIndices.clear();
    TableIndices.clear();
    DataLocations.clear();
    CustomSections.clear();
    ProducersSection.reset();
    TargetFeaturesSection.reset();
    CustomSectionsRelocations.clear();
    SignatureIndices.clear();
    Signatures.clear();
    DataSegments.clear();
    SectionFunctions.clear();
    NumFunctionImports = 0;
    NumGlobalImports = 0;
    MCObjectWriter::reset();
  }

  void writeHeader(const MCAssembler &Asm);

  void recordRelocation(MCAssembler &Asm, const MCAsmLayout &Layout,
                        const MCFragment *Fragment, const MCFixup &Fixup,
                        MCValue Target, uint64_t &FixedValue) override;

  void executePostLayoutBinding(MCAssembler &Asm,
                                const MCAsmLayout &Layout) override;

  uint64_t writeObject(MCAssembler &Asm, const MCAsmLayout &Layout) override;

  void writeString(const StringRef Str) {
    encodeULEB128(Str.size(), W.OS);
    W.OS << Str;
  }

  void writeI32(int32_t val) {
    char Buffer[4];
    support::endian::write32le(Buffer, val);
    W.OS.write(Buffer, sizeof(Buffer));
  }

  void writeI64(int64_t val) {
    char Buffer[8];
    support::endian::write64le(Buffer, val);
    W.OS.write(Buffer, sizeof(Buffer));
  }

  void writeValueType(wasm::ValType Ty) { W.OS << static_cast<char>(Ty); }

  void writeTypeSection(ArrayRef<WasmSignature> Signatures);
  void writeImportSection(ArrayRef<wasm::WasmImport> Imports, uint64_t DataSize,
                          uint32_t NumElements);
  void writeFunctionSection(ArrayRef<WasmFunction> Functions);
  void writeExportSection(ArrayRef<wasm::WasmExport> Exports);
  void writeElemSection(ArrayRef<uint32_t> TableElems);
  void writeDataCountSection();
  uint32_t writeCodeSection(const MCAssembler &Asm, const MCAsmLayout &Layout,
                            ArrayRef<WasmFunction> Functions);
  uint32_t writeDataSection(const MCAsmLayout &Layout);
  void writeEventSection(ArrayRef<wasm::WasmEventType> Events);
  void writeGlobalSection(ArrayRef<wasm::WasmGlobal> Globals);
  void writeRelocSection(uint32_t SectionIndex, StringRef Name,
                         std::vector<WasmRelocationEntry> &Relocations);
  void writeLinkingMetaDataSection(
      ArrayRef<wasm::WasmSymbolInfo> SymbolInfos,
      ArrayRef<std::pair<uint16_t, uint32_t>> InitFuncs,
      const std::map<StringRef, std::vector<WasmComdatEntry>> &Comdats);
  void writeCustomSection(WasmCustomSection &CustomSection,
                          const MCAssembler &Asm, const MCAsmLayout &Layout);
  void writeCustomRelocSections();
  void
  updateCustomSectionRelocations(const SmallVector<WasmFunction, 4> &Functions,
                                 const MCAsmLayout &Layout);

  uint64_t getProvisionalValue(const WasmRelocationEntry &RelEntry,
                               const MCAsmLayout &Layout);
  void applyRelocations(ArrayRef<WasmRelocationEntry> Relocations,
                        uint64_t ContentsOffset, const MCAsmLayout &Layout);

  uint32_t getRelocationIndexValue(const WasmRelocationEntry &RelEntry);
  uint32_t getFunctionType(const MCSymbolWasm &Symbol);
  uint32_t getEventType(const MCSymbolWasm &Symbol);
  void registerFunctionType(const MCSymbolWasm &Symbol);
  void registerEventType(const MCSymbolWasm &Symbol);
};

} // end anonymous namespace

// Write out a section header and a patchable section size field.
void WasmObjectWriter::startSection(SectionBookkeeping &Section,
                                    unsigned SectionId) {
  LLVM_DEBUG(dbgs() << "startSection " << SectionId << "\n");
  W.OS << char(SectionId);

  Section.SizeOffset = W.OS.tell();

  // The section size. We don't know the size yet, so reserve enough space
  // for any 32-bit value; we'll patch it later.
  encodeULEB128(0, W.OS, 5);

  // The position where the section starts, for measuring its size.
  Section.ContentsOffset = W.OS.tell();
  Section.PayloadOffset = W.OS.tell();
  Section.Index = SectionCount++;
}

void WasmObjectWriter::startCustomSection(SectionBookkeeping &Section,
                                          StringRef Name) {
  LLVM_DEBUG(dbgs() << "startCustomSection " << Name << "\n");
  startSection(Section, wasm::WASM_SEC_CUSTOM);

  // The position where the section header ends, for measuring its size.
  Section.PayloadOffset = W.OS.tell();

  // Custom sections in wasm also have a string identifier.
  writeString(Name);

  // The position where the custom section starts.
  Section.ContentsOffset = W.OS.tell();
}

// Now that the section is complete and we know how big it is, patch up the
// section size field at the start of the section.
void WasmObjectWriter::endSection(SectionBookkeeping &Section) {
  uint64_t Size = W.OS.tell();
  // /dev/null doesn't support seek/tell and can report offset of 0.
  // Simply skip this patching in that case.
  if (!Size)
    return;

  Size -= Section.PayloadOffset;
  if (uint32_t(Size) != Size)
    report_fatal_error("section size does not fit in a uint32_t");

  LLVM_DEBUG(dbgs() << "endSection size=" << Size << "\n");

  // Write the final section size to the payload_len field, which follows
  // the section id byte.
  writePatchableLEB<5>(static_cast<raw_pwrite_stream &>(W.OS), Size,
                       Section.SizeOffset);
}

// Emit the Wasm header.
void WasmObjectWriter::writeHeader(const MCAssembler &Asm) {
  W.OS.write(wasm::WasmMagic, sizeof(wasm::WasmMagic));
  W.write<uint32_t>(wasm::WasmVersion);
}

void WasmObjectWriter::executePostLayoutBinding(MCAssembler &Asm,
                                                const MCAsmLayout &Layout) {
  // Build a map of sections to the function that defines them, for use
  // in recordRelocation.
  for (const MCSymbol &S : Asm.symbols()) {
    const auto &WS = static_cast<const MCSymbolWasm &>(S);
    if (WS.isDefined() && WS.isFunction() && !WS.isVariable()) {
      const auto &Sec = static_cast<const MCSectionWasm &>(S.getSection());
      auto Pair = SectionFunctions.insert(std::make_pair(&Sec, &S));
      if (!Pair.second)
        report_fatal_error("section already has a defining function: " +
                           Sec.getName());
    }
  }
}

void WasmObjectWriter::recordRelocation(MCAssembler &Asm,
                                        const MCAsmLayout &Layout,
                                        const MCFragment *Fragment,
                                        const MCFixup &Fixup, MCValue Target,
                                        uint64_t &FixedValue) {
  // The WebAssembly backend should never generate FKF_IsPCRel fixups
  assert(!(Asm.getBackend().getFixupKindInfo(Fixup.getKind()).Flags &
           MCFixupKindInfo::FKF_IsPCRel));

  const auto &FixupSection = cast<MCSectionWasm>(*Fragment->getParent());
  uint64_t C = Target.getConstant();
  uint64_t FixupOffset = Layout.getFragmentOffset(Fragment) + Fixup.getOffset();
  MCContext &Ctx = Asm.getContext();

  if (const MCSymbolRefExpr *RefB = Target.getSymB()) {
    // To get here the A - B expression must have failed evaluateAsRelocatable.
    // This means either A or B must be undefined and in WebAssembly we can't
    // support either of those cases.
    const auto &SymB = cast<MCSymbolWasm>(RefB->getSymbol());
    Ctx.reportError(
        Fixup.getLoc(),
        Twine("symbol '") + SymB.getName() +
            "': unsupported subtraction expression used in relocation.");
    return;
  }

  // We either rejected the fixup or folded B into C at this point.
  const MCSymbolRefExpr *RefA = Target.getSymA();
  const auto *SymA = cast<MCSymbolWasm>(&RefA->getSymbol());

  // The .init_array isn't translated as data, so don't do relocations in it.
  if (FixupSection.getName().startswith(".init_array")) {
    SymA->setUsedInInitArray();
    return;
  }

  if (SymA->isVariable()) {
    const MCExpr *Expr = SymA->getVariableValue();
    if (const auto *Inner = dyn_cast<MCSymbolRefExpr>(Expr))
      if (Inner->getKind() == MCSymbolRefExpr::VK_WEAKREF)
        llvm_unreachable("weakref used in reloc not yet implemented");
  }

  // Put any constant offset in an addend. Offsets can be negative, and
  // LLVM expects wrapping, in contrast to wasm's immediates which can't
  // be negative and don't wrap.
  FixedValue = 0;

  unsigned Type = TargetObjectWriter->getRelocType(Target, Fixup);

  // Absolute offset within a section or a function.
  // Currently only supported for for metadata sections.
  // See: test/MC/WebAssembly/blockaddress.ll
  if (Type == wasm::R_WASM_FUNCTION_OFFSET_I32 ||
      Type == wasm::R_WASM_SECTION_OFFSET_I32) {
    if (!FixupSection.getKind().isMetadata())
      report_fatal_error("relocations for function or section offsets are "
                         "only supported in metadata sections");

    const MCSymbol *SectionSymbol = nullptr;
    const MCSection &SecA = SymA->getSection();
    if (SecA.getKind().isText())
      SectionSymbol = SectionFunctions.find(&SecA)->second;
    else
      SectionSymbol = SecA.getBeginSymbol();
    if (!SectionSymbol)
      report_fatal_error("section symbol is required for relocation");

    C += Layout.getSymbolOffset(*SymA);
    SymA = cast<MCSymbolWasm>(SectionSymbol);
  }

  // Relocation other than R_WASM_TYPE_INDEX_LEB are required to be
  // against a named symbol.
  if (Type != wasm::R_WASM_TYPE_INDEX_LEB) {
    if (SymA->getName().empty())
      report_fatal_error("relocations against un-named temporaries are not yet "
                         "supported by wasm");

    SymA->setUsedInReloc();
  }

  if (RefA->getKind() == MCSymbolRefExpr::VK_GOT)
    SymA->setUsedInGOT();

  WasmRelocationEntry Rec(FixupOffset, SymA, C, Type, &FixupSection);
  LLVM_DEBUG(dbgs() << "WasmReloc: " << Rec << "\n");

  if (FixupSection.isWasmData()) {
    DataRelocations.push_back(Rec);
  } else if (FixupSection.getKind().isText()) {
    CodeRelocations.push_back(Rec);
  } else if (FixupSection.getKind().isMetadata()) {
    CustomSectionsRelocations[&FixupSection].push_back(Rec);
  } else {
    llvm_unreachable("unexpected section type");
  }
}

// Compute a value to write into the code at the location covered
// by RelEntry. This value isn't used by the static linker; it just serves
// to make the object format more readable and more likely to be directly
// useable.
uint64_t
WasmObjectWriter::getProvisionalValue(const WasmRelocationEntry &RelEntry,
                                      const MCAsmLayout &Layout) {
  if ((RelEntry.Type == wasm::R_WASM_GLOBAL_INDEX_LEB ||
       RelEntry.Type == wasm::R_WASM_GLOBAL_INDEX_I32) &&
      !RelEntry.Symbol->isGlobal()) {
    assert(GOTIndices.count(RelEntry.Symbol) > 0 && "symbol not found in GOT index space");
    return GOTIndices[RelEntry.Symbol];
  }

  switch (RelEntry.Type) {
  case wasm::R_WASM_TABLE_INDEX_REL_SLEB:
  case wasm::R_WASM_TABLE_INDEX_SLEB:
  case wasm::R_WASM_TABLE_INDEX_SLEB64:
  case wasm::R_WASM_TABLE_INDEX_I32:
  case wasm::R_WASM_TABLE_INDEX_I64: {
    // Provisional value is table address of the resolved symbol itself
    const MCSymbolWasm *Base =
        cast<MCSymbolWasm>(Layout.getBaseSymbol(*RelEntry.Symbol));
    assert(Base->isFunction());
    if (RelEntry.Type == wasm::R_WASM_TABLE_INDEX_REL_SLEB)
      return TableIndices[Base] - InitialTableOffset;
    else
      return TableIndices[Base];
  }
  case wasm::R_WASM_TYPE_INDEX_LEB:
    // Provisional value is same as the index
    return getRelocationIndexValue(RelEntry);
  case wasm::R_WASM_FUNCTION_INDEX_LEB:
  case wasm::R_WASM_GLOBAL_INDEX_LEB:
  case wasm::R_WASM_GLOBAL_INDEX_I32:
  case wasm::R_WASM_EVENT_INDEX_LEB:
    // Provisional value is function/global/event Wasm index
    assert(WasmIndices.count(RelEntry.Symbol) > 0 && "symbol not found in wasm index space");
    return WasmIndices[RelEntry.Symbol];
  case wasm::R_WASM_FUNCTION_OFFSET_I32:
  case wasm::R_WASM_SECTION_OFFSET_I32: {
    const auto &Section =
        static_cast<const MCSectionWasm &>(RelEntry.Symbol->getSection());
    return Section.getSectionOffset() + RelEntry.Addend;
  }
  case wasm::R_WASM_MEMORY_ADDR_LEB:
  case wasm::R_WASM_MEMORY_ADDR_LEB64:
  case wasm::R_WASM_MEMORY_ADDR_SLEB:
  case wasm::R_WASM_MEMORY_ADDR_SLEB64:
  case wasm::R_WASM_MEMORY_ADDR_REL_SLEB:
  case wasm::R_WASM_MEMORY_ADDR_REL_SLEB64:
  case wasm::R_WASM_MEMORY_ADDR_I32:
  case wasm::R_WASM_MEMORY_ADDR_I64: {
    // Provisional value is address of the global
    const MCSymbolWasm *Base =
        cast<MCSymbolWasm>(Layout.getBaseSymbol(*RelEntry.Symbol));
    // For undefined symbols, use zero
    if (!Base->isDefined())
      return 0;
    const wasm::WasmDataReference &Ref = DataLocations[Base];
    const WasmDataSegment &Segment = DataSegments[Ref.Segment];
    // Ignore overflow. LLVM allows address arithmetic to silently wrap.
    return Segment.Offset + Ref.Offset + RelEntry.Addend;
  }
  default:
    llvm_unreachable("invalid relocation type");
  }
}

static void addData(SmallVectorImpl<char> &DataBytes,
                    MCSectionWasm &DataSection) {
  LLVM_DEBUG(errs() << "addData: " << DataSection.getName() << "\n");

  DataBytes.resize(alignTo(DataBytes.size(), DataSection.getAlignment()));

  for (const MCFragment &Frag : DataSection) {
    if (Frag.hasInstructions())
      report_fatal_error("only data supported in data sections");

    if (auto *Align = dyn_cast<MCAlignFragment>(&Frag)) {
      if (Align->getValueSize() != 1)
        report_fatal_error("only byte values supported for alignment");
      // If nops are requested, use zeros, as this is the data section.
      uint8_t Value = Align->hasEmitNops() ? 0 : Align->getValue();
      uint64_t Size =
          std::min<uint64_t>(alignTo(DataBytes.size(), Align->getAlignment()),
                             DataBytes.size() + Align->getMaxBytesToEmit());
      DataBytes.resize(Size, Value);
    } else if (auto *Fill = dyn_cast<MCFillFragment>(&Frag)) {
      int64_t NumValues;
      if (!Fill->getNumValues().evaluateAsAbsolute(NumValues))
        llvm_unreachable("The fill should be an assembler constant");
      DataBytes.insert(DataBytes.end(), Fill->getValueSize() * NumValues,
                       Fill->getValue());
    } else if (auto *LEB = dyn_cast<MCLEBFragment>(&Frag)) {
      const SmallVectorImpl<char> &Contents = LEB->getContents();
      DataBytes.insert(DataBytes.end(), Contents.begin(), Contents.end());
    } else {
      const auto &DataFrag = cast<MCDataFragment>(Frag);
      const SmallVectorImpl<char> &Contents = DataFrag.getContents();
      DataBytes.insert(DataBytes.end(), Contents.begin(), Contents.end());
    }
  }

  LLVM_DEBUG(dbgs() << "addData -> " << DataBytes.size() << "\n");
}

uint32_t
WasmObjectWriter::getRelocationIndexValue(const WasmRelocationEntry &RelEntry) {
  if (RelEntry.Type == wasm::R_WASM_TYPE_INDEX_LEB) {
    if (!TypeIndices.count(RelEntry.Symbol))
      report_fatal_error("symbol not found in type index space: " +
                         RelEntry.Symbol->getName());
    return TypeIndices[RelEntry.Symbol];
  }

  return RelEntry.Symbol->getIndex();
}

// Apply the portions of the relocation records that we can handle ourselves
// directly.
void WasmObjectWriter::applyRelocations(
    ArrayRef<WasmRelocationEntry> Relocations, uint64_t ContentsOffset,
    const MCAsmLayout &Layout) {
  auto &Stream = static_cast<raw_pwrite_stream &>(W.OS);
  for (const WasmRelocationEntry &RelEntry : Relocations) {
    uint64_t Offset = ContentsOffset +
                      RelEntry.FixupSection->getSectionOffset() +
                      RelEntry.Offset;

    LLVM_DEBUG(dbgs() << "applyRelocation: " << RelEntry << "\n");
    auto Value = getProvisionalValue(RelEntry, Layout);

    switch (RelEntry.Type) {
    case wasm::R_WASM_FUNCTION_INDEX_LEB:
    case wasm::R_WASM_TYPE_INDEX_LEB:
    case wasm::R_WASM_GLOBAL_INDEX_LEB:
    case wasm::R_WASM_MEMORY_ADDR_LEB:
    case wasm::R_WASM_EVENT_INDEX_LEB:
      writePatchableLEB<5>(Stream, Value, Offset);
      break;
    case wasm::R_WASM_MEMORY_ADDR_LEB64:
      writePatchableLEB<10>(Stream, Value, Offset);
      break;
    case wasm::R_WASM_TABLE_INDEX_I32:
    case wasm::R_WASM_MEMORY_ADDR_I32:
    case wasm::R_WASM_FUNCTION_OFFSET_I32:
    case wasm::R_WASM_SECTION_OFFSET_I32:
    case wasm::R_WASM_GLOBAL_INDEX_I32:
      patchI32(Stream, Value, Offset);
      break;
    case wasm::R_WASM_TABLE_INDEX_I64:
    case wasm::R_WASM_MEMORY_ADDR_I64:
      patchI64(Stream, Value, Offset);
      break;
    case wasm::R_WASM_TABLE_INDEX_SLEB:
    case wasm::R_WASM_TABLE_INDEX_REL_SLEB:
    case wasm::R_WASM_MEMORY_ADDR_SLEB:
    case wasm::R_WASM_MEMORY_ADDR_REL_SLEB:
      writePatchableSLEB<5>(Stream, Value, Offset);
      break;
    case wasm::R_WASM_TABLE_INDEX_SLEB64:
    case wasm::R_WASM_MEMORY_ADDR_SLEB64:
    case wasm::R_WASM_MEMORY_ADDR_REL_SLEB64:
      writePatchableSLEB<10>(Stream, Value, Offset);
      break;
    default:
      llvm_unreachable("invalid relocation type");
    }
  }
}

void WasmObjectWriter::writeTypeSection(ArrayRef<WasmSignature> Signatures) {
  if (Signatures.empty())
    return;

  SectionBookkeeping Section;
  startSection(Section, wasm::WASM_SEC_TYPE);

  encodeULEB128(Signatures.size(), W.OS);

  for (const WasmSignature &Sig : Signatures) {
    W.OS << char(wasm::WASM_TYPE_FUNC);
    encodeULEB128(Sig.Params.size(), W.OS);
    for (wasm::ValType Ty : Sig.Params)
      writeValueType(Ty);
    encodeULEB128(Sig.Returns.size(), W.OS);
    for (wasm::ValType Ty : Sig.Returns)
      writeValueType(Ty);
  }

  endSection(Section);
}

void WasmObjectWriter::writeImportSection(ArrayRef<wasm::WasmImport> Imports,
                                          uint64_t DataSize,
                                          uint32_t NumElements) {
  if (Imports.empty())
    return;

  uint64_t NumPages = (DataSize + wasm::WasmPageSize - 1) / wasm::WasmPageSize;

  SectionBookkeeping Section;
  startSection(Section, wasm::WASM_SEC_IMPORT);

  encodeULEB128(Imports.size(), W.OS);
  for (const wasm::WasmImport &Import : Imports) {
    writeString(Import.Module);
    writeString(Import.Field);
    W.OS << char(Import.Kind);

    switch (Import.Kind) {
    case wasm::WASM_EXTERNAL_FUNCTION:
      encodeULEB128(Import.SigIndex, W.OS);
      break;
    case wasm::WASM_EXTERNAL_GLOBAL:
      W.OS << char(Import.Global.Type);
      W.OS << char(Import.Global.Mutable ? 1 : 0);
      break;
    case wasm::WASM_EXTERNAL_MEMORY:
      encodeULEB128(Import.Memory.Flags, W.OS);
      encodeULEB128(NumPages, W.OS);  // initial
      break;
    case wasm::WASM_EXTERNAL_TABLE:
      W.OS << char(Import.Table.ElemType);
      encodeULEB128(0, W.OS);           // flags
      encodeULEB128(NumElements, W.OS); // initial
      break;
    case wasm::WASM_EXTERNAL_EVENT:
      encodeULEB128(Import.Event.Attribute, W.OS);
      encodeULEB128(Import.Event.SigIndex, W.OS);
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

  encodeULEB128(Functions.size(), W.OS);
  for (const WasmFunction &Func : Functions)
    encodeULEB128(Func.SigIndex, W.OS);

  endSection(Section);
}

void WasmObjectWriter::writeEventSection(ArrayRef<wasm::WasmEventType> Events) {
  if (Events.empty())
    return;

  SectionBookkeeping Section;
  startSection(Section, wasm::WASM_SEC_EVENT);

  encodeULEB128(Events.size(), W.OS);
  for (const wasm::WasmEventType &Event : Events) {
    encodeULEB128(Event.Attribute, W.OS);
    encodeULEB128(Event.SigIndex, W.OS);
  }

  endSection(Section);
}

void WasmObjectWriter::writeGlobalSection(ArrayRef<wasm::WasmGlobal> Globals) {
  if (Globals.empty())
    return;

  SectionBookkeeping Section;
  startSection(Section, wasm::WASM_SEC_GLOBAL);

  encodeULEB128(Globals.size(), W.OS);
  for (const wasm::WasmGlobal &Global : Globals) {
    encodeULEB128(Global.Type.Type, W.OS);
    W.OS << char(Global.Type.Mutable);
    W.OS << char(Global.InitExpr.Opcode);
    switch (Global.Type.Type) {
    case wasm::WASM_TYPE_I32:
      encodeSLEB128(0, W.OS);
      break;
    case wasm::WASM_TYPE_I64:
      encodeSLEB128(0, W.OS);
      break;
    case wasm::WASM_TYPE_F32:
      writeI32(0);
      break;
    case wasm::WASM_TYPE_F64:
      writeI64(0);
      break;
    case wasm::WASM_TYPE_EXTERNREF:
      writeValueType(wasm::ValType::EXTERNREF);
      break;
    default:
      llvm_unreachable("unexpected type");
    }
    W.OS << char(wasm::WASM_OPCODE_END);
  }

  endSection(Section);
}

void WasmObjectWriter::writeExportSection(ArrayRef<wasm::WasmExport> Exports) {
  if (Exports.empty())
    return;

  SectionBookkeeping Section;
  startSection(Section, wasm::WASM_SEC_EXPORT);

  encodeULEB128(Exports.size(), W.OS);
  for (const wasm::WasmExport &Export : Exports) {
    writeString(Export.Name);
    W.OS << char(Export.Kind);
    encodeULEB128(Export.Index, W.OS);
  }

  endSection(Section);
}

void WasmObjectWriter::writeElemSection(ArrayRef<uint32_t> TableElems) {
  if (TableElems.empty())
    return;

  SectionBookkeeping Section;
  startSection(Section, wasm::WASM_SEC_ELEM);

  encodeULEB128(1, W.OS); // number of "segments"
  encodeULEB128(0, W.OS); // the table index

  // init expr for starting offset
  W.OS << char(wasm::WASM_OPCODE_I32_CONST);
  encodeSLEB128(InitialTableOffset, W.OS);
  W.OS << char(wasm::WASM_OPCODE_END);

  encodeULEB128(TableElems.size(), W.OS);
  for (uint32_t Elem : TableElems)
    encodeULEB128(Elem, W.OS);

  endSection(Section);
}

void WasmObjectWriter::writeDataCountSection() {
  if (DataSegments.empty())
    return;

  SectionBookkeeping Section;
  startSection(Section, wasm::WASM_SEC_DATACOUNT);
  encodeULEB128(DataSegments.size(), W.OS);
  endSection(Section);
}

uint32_t WasmObjectWriter::writeCodeSection(const MCAssembler &Asm,
                                            const MCAsmLayout &Layout,
                                            ArrayRef<WasmFunction> Functions) {
  if (Functions.empty())
    return 0;

  SectionBookkeeping Section;
  startSection(Section, wasm::WASM_SEC_CODE);

  encodeULEB128(Functions.size(), W.OS);

  for (const WasmFunction &Func : Functions) {
    auto &FuncSection = static_cast<MCSectionWasm &>(Func.Sym->getSection());

    int64_t Size = 0;
    if (!Func.Sym->getSize()->evaluateAsAbsolute(Size, Layout))
      report_fatal_error(".size expression must be evaluatable");

    encodeULEB128(Size, W.OS);
    FuncSection.setSectionOffset(W.OS.tell() - Section.ContentsOffset);
    Asm.writeSectionData(W.OS, &FuncSection, Layout);
  }

  // Apply fixups.
  applyRelocations(CodeRelocations, Section.ContentsOffset, Layout);

  endSection(Section);
  return Section.Index;
}

uint32_t WasmObjectWriter::writeDataSection(const MCAsmLayout &Layout) {
  if (DataSegments.empty())
    return 0;

  SectionBookkeeping Section;
  startSection(Section, wasm::WASM_SEC_DATA);

  encodeULEB128(DataSegments.size(), W.OS); // count

  for (const WasmDataSegment &Segment : DataSegments) {
    encodeULEB128(Segment.InitFlags, W.OS); // flags
    if (Segment.InitFlags & wasm::WASM_SEGMENT_HAS_MEMINDEX)
      encodeULEB128(0, W.OS); // memory index
    if ((Segment.InitFlags & wasm::WASM_SEGMENT_IS_PASSIVE) == 0) {
      W.OS << char(Segment.Offset > INT32_MAX ? wasm::WASM_OPCODE_I64_CONST
                                              : wasm::WASM_OPCODE_I32_CONST);
      encodeSLEB128(Segment.Offset, W.OS); // offset
      W.OS << char(wasm::WASM_OPCODE_END);
    }
    encodeULEB128(Segment.Data.size(), W.OS); // size
    Segment.Section->setSectionOffset(W.OS.tell() - Section.ContentsOffset);
    W.OS << Segment.Data; // data
  }

  // Apply fixups.
  applyRelocations(DataRelocations, Section.ContentsOffset, Layout);

  endSection(Section);
  return Section.Index;
}

void WasmObjectWriter::writeRelocSection(
    uint32_t SectionIndex, StringRef Name,
    std::vector<WasmRelocationEntry> &Relocs) {
  // See: https://github.com/WebAssembly/tool-conventions/blob/master/Linking.md
  // for descriptions of the reloc sections.

  if (Relocs.empty())
    return;

  // First, ensure the relocations are sorted in offset order.  In general they
  // should already be sorted since `recordRelocation` is called in offset
  // order, but for the code section we combine many MC sections into single
  // wasm section, and this order is determined by the order of Asm.Symbols()
  // not the sections order.
  llvm::stable_sort(
      Relocs, [](const WasmRelocationEntry &A, const WasmRelocationEntry &B) {
        return (A.Offset + A.FixupSection->getSectionOffset()) <
               (B.Offset + B.FixupSection->getSectionOffset());
      });

  SectionBookkeeping Section;
  startCustomSection(Section, std::string("reloc.") + Name.str());

  encodeULEB128(SectionIndex, W.OS);
  encodeULEB128(Relocs.size(), W.OS);
  for (const WasmRelocationEntry &RelEntry : Relocs) {
    uint64_t Offset =
        RelEntry.Offset + RelEntry.FixupSection->getSectionOffset();
    uint32_t Index = getRelocationIndexValue(RelEntry);

    W.OS << char(RelEntry.Type);
    encodeULEB128(Offset, W.OS);
    encodeULEB128(Index, W.OS);
    if (RelEntry.hasAddend())
      encodeSLEB128(RelEntry.Addend, W.OS);
  }

  endSection(Section);
}

void WasmObjectWriter::writeCustomRelocSections() {
  for (const auto &Sec : CustomSections) {
    auto &Relocations = CustomSectionsRelocations[Sec.Section];
    writeRelocSection(Sec.OutputIndex, Sec.Name, Relocations);
  }
}

void WasmObjectWriter::writeLinkingMetaDataSection(
    ArrayRef<wasm::WasmSymbolInfo> SymbolInfos,
    ArrayRef<std::pair<uint16_t, uint32_t>> InitFuncs,
    const std::map<StringRef, std::vector<WasmComdatEntry>> &Comdats) {
  SectionBookkeeping Section;
  startCustomSection(Section, "linking");
  encodeULEB128(wasm::WasmMetadataVersion, W.OS);

  SectionBookkeeping SubSection;
  if (SymbolInfos.size() != 0) {
    startSection(SubSection, wasm::WASM_SYMBOL_TABLE);
    encodeULEB128(SymbolInfos.size(), W.OS);
    for (const wasm::WasmSymbolInfo &Sym : SymbolInfos) {
      encodeULEB128(Sym.Kind, W.OS);
      encodeULEB128(Sym.Flags, W.OS);
      switch (Sym.Kind) {
      case wasm::WASM_SYMBOL_TYPE_FUNCTION:
      case wasm::WASM_SYMBOL_TYPE_GLOBAL:
      case wasm::WASM_SYMBOL_TYPE_EVENT:
        encodeULEB128(Sym.ElementIndex, W.OS);
        if ((Sym.Flags & wasm::WASM_SYMBOL_UNDEFINED) == 0 ||
            (Sym.Flags & wasm::WASM_SYMBOL_EXPLICIT_NAME) != 0)
          writeString(Sym.Name);
        break;
      case wasm::WASM_SYMBOL_TYPE_DATA:
        writeString(Sym.Name);
        if ((Sym.Flags & wasm::WASM_SYMBOL_UNDEFINED) == 0) {
          encodeULEB128(Sym.DataRef.Segment, W.OS);
          encodeULEB128(Sym.DataRef.Offset, W.OS);
          encodeULEB128(Sym.DataRef.Size, W.OS);
        }
        break;
      case wasm::WASM_SYMBOL_TYPE_SECTION: {
        const uint32_t SectionIndex =
            CustomSections[Sym.ElementIndex].OutputIndex;
        encodeULEB128(SectionIndex, W.OS);
        break;
      }
      default:
        llvm_unreachable("unexpected kind");
      }
    }
    endSection(SubSection);
  }

  if (DataSegments.size()) {
    startSection(SubSection, wasm::WASM_SEGMENT_INFO);
    encodeULEB128(DataSegments.size(), W.OS);
    for (const WasmDataSegment &Segment : DataSegments) {
      writeString(Segment.Name);
      encodeULEB128(Segment.Alignment, W.OS);
      encodeULEB128(Segment.LinkerFlags, W.OS);
    }
    endSection(SubSection);
  }

  if (!InitFuncs.empty()) {
    startSection(SubSection, wasm::WASM_INIT_FUNCS);
    encodeULEB128(InitFuncs.size(), W.OS);
    for (auto &StartFunc : InitFuncs) {
      encodeULEB128(StartFunc.first, W.OS);  // priority
      encodeULEB128(StartFunc.second, W.OS); // function index
    }
    endSection(SubSection);
  }

  if (Comdats.size()) {
    startSection(SubSection, wasm::WASM_COMDAT_INFO);
    encodeULEB128(Comdats.size(), W.OS);
    for (const auto &C : Comdats) {
      writeString(C.first);
      encodeULEB128(0, W.OS); // flags for future use
      encodeULEB128(C.second.size(), W.OS);
      for (const WasmComdatEntry &Entry : C.second) {
        encodeULEB128(Entry.Kind, W.OS);
        encodeULEB128(Entry.Index, W.OS);
      }
    }
    endSection(SubSection);
  }

  endSection(Section);
}

void WasmObjectWriter::writeCustomSection(WasmCustomSection &CustomSection,
                                          const MCAssembler &Asm,
                                          const MCAsmLayout &Layout) {
  SectionBookkeeping Section;
  auto *Sec = CustomSection.Section;
  startCustomSection(Section, CustomSection.Name);

  Sec->setSectionOffset(W.OS.tell() - Section.ContentsOffset);
  Asm.writeSectionData(W.OS, Sec, Layout);

  CustomSection.OutputContentsOffset = Section.ContentsOffset;
  CustomSection.OutputIndex = Section.Index;

  endSection(Section);

  // Apply fixups.
  auto &Relocations = CustomSectionsRelocations[CustomSection.Section];
  applyRelocations(Relocations, CustomSection.OutputContentsOffset, Layout);
}

uint32_t WasmObjectWriter::getFunctionType(const MCSymbolWasm &Symbol) {
  assert(Symbol.isFunction());
  assert(TypeIndices.count(&Symbol));
  return TypeIndices[&Symbol];
}

uint32_t WasmObjectWriter::getEventType(const MCSymbolWasm &Symbol) {
  assert(Symbol.isEvent());
  assert(TypeIndices.count(&Symbol));
  return TypeIndices[&Symbol];
}

void WasmObjectWriter::registerFunctionType(const MCSymbolWasm &Symbol) {
  assert(Symbol.isFunction());

  WasmSignature S;

  if (auto *Sig = Symbol.getSignature()) {
    S.Returns = Sig->Returns;
    S.Params = Sig->Params;
  }

  auto Pair = SignatureIndices.insert(std::make_pair(S, Signatures.size()));
  if (Pair.second)
    Signatures.push_back(S);
  TypeIndices[&Symbol] = Pair.first->second;

  LLVM_DEBUG(dbgs() << "registerFunctionType: " << Symbol
                    << " new:" << Pair.second << "\n");
  LLVM_DEBUG(dbgs() << "  -> type index: " << Pair.first->second << "\n");
}

void WasmObjectWriter::registerEventType(const MCSymbolWasm &Symbol) {
  assert(Symbol.isEvent());

  // TODO Currently we don't generate imported exceptions, but if we do, we
  // should have a way of infering types of imported exceptions.
  WasmSignature S;
  if (auto *Sig = Symbol.getSignature()) {
    S.Returns = Sig->Returns;
    S.Params = Sig->Params;
  }

  auto Pair = SignatureIndices.insert(std::make_pair(S, Signatures.size()));
  if (Pair.second)
    Signatures.push_back(S);
  TypeIndices[&Symbol] = Pair.first->second;

  LLVM_DEBUG(dbgs() << "registerEventType: " << Symbol << " new:" << Pair.second
                    << "\n");
  LLVM_DEBUG(dbgs() << "  -> type index: " << Pair.first->second << "\n");
}

static bool isInSymtab(const MCSymbolWasm &Sym) {
  if (Sym.isUsedInReloc() || Sym.isUsedInInitArray())
    return true;

  if (Sym.isComdat() && !Sym.isDefined())
    return false;

  if (Sym.isTemporary())
    return false;

  if (Sym.isSection())
    return false;

  return true;
}

uint64_t WasmObjectWriter::writeObject(MCAssembler &Asm,
                                       const MCAsmLayout &Layout) {
  uint64_t StartOffset = W.OS.tell();

  LLVM_DEBUG(dbgs() << "WasmObjectWriter::writeObject\n");

  // Collect information from the available symbols.
  SmallVector<WasmFunction, 4> Functions;
  SmallVector<uint32_t, 4> TableElems;
  SmallVector<wasm::WasmImport, 4> Imports;
  SmallVector<wasm::WasmExport, 4> Exports;
  SmallVector<wasm::WasmEventType, 1> Events;
  SmallVector<wasm::WasmGlobal, 1> Globals;
  SmallVector<wasm::WasmSymbolInfo, 4> SymbolInfos;
  SmallVector<std::pair<uint16_t, uint32_t>, 2> InitFuncs;
  std::map<StringRef, std::vector<WasmComdatEntry>> Comdats;
  uint64_t DataSize = 0;

  // For now, always emit the memory import, since loads and stores are not
  // valid without it. In the future, we could perhaps be more clever and omit
  // it if there are no loads or stores.
  wasm::WasmImport MemImport;
  MemImport.Module = "env";
  MemImport.Field = "__linear_memory";
  MemImport.Kind = wasm::WASM_EXTERNAL_MEMORY;
  MemImport.Memory.Flags = is64Bit() ? wasm::WASM_LIMITS_FLAG_IS_64
                                     : wasm::WASM_LIMITS_FLAG_NONE;
  Imports.push_back(MemImport);

  // For now, always emit the table section, since indirect calls are not
  // valid without it. In the future, we could perhaps be more clever and omit
  // it if there are no indirect calls.
  wasm::WasmImport TableImport;
  TableImport.Module = "env";
  TableImport.Field = "__indirect_function_table";
  TableImport.Kind = wasm::WASM_EXTERNAL_TABLE;
  TableImport.Table.ElemType = wasm::WASM_TYPE_FUNCREF;
  Imports.push_back(TableImport);

  // Populate SignatureIndices, and Imports and WasmIndices for undefined
  // symbols.  This must be done before populating WasmIndices for defined
  // symbols.
  for (const MCSymbol &S : Asm.symbols()) {
    const auto &WS = static_cast<const MCSymbolWasm &>(S);

    // Register types for all functions, including those with private linkage
    // (because wasm always needs a type signature).
    if (WS.isFunction()) {
      const MCSymbolWasm *Base = cast<MCSymbolWasm>(Layout.getBaseSymbol(S));
      registerFunctionType(*Base);
    }

    if (WS.isEvent())
      registerEventType(WS);

    if (WS.isTemporary())
      continue;

    // If the symbol is not defined in this translation unit, import it.
    if (!WS.isDefined() && !WS.isComdat()) {
      if (WS.isFunction()) {
        wasm::WasmImport Import;
        Import.Module = WS.getImportModule();
        Import.Field = WS.getImportName();
        Import.Kind = wasm::WASM_EXTERNAL_FUNCTION;
        Import.SigIndex = getFunctionType(WS);
        Imports.push_back(Import);
        assert(WasmIndices.count(&WS) == 0);
        WasmIndices[&WS] = NumFunctionImports++;
      } else if (WS.isGlobal()) {
        if (WS.isWeak())
          report_fatal_error("undefined global symbol cannot be weak");

        wasm::WasmImport Import;
        Import.Field = WS.getImportName();
        Import.Kind = wasm::WASM_EXTERNAL_GLOBAL;
        Import.Module = WS.getImportModule();
        Import.Global = WS.getGlobalType();
        Imports.push_back(Import);
        assert(WasmIndices.count(&WS) == 0);
        WasmIndices[&WS] = NumGlobalImports++;
      } else if (WS.isEvent()) {
        if (WS.isWeak())
          report_fatal_error("undefined event symbol cannot be weak");

        wasm::WasmImport Import;
        Import.Module = WS.getImportModule();
        Import.Field = WS.getImportName();
        Import.Kind = wasm::WASM_EXTERNAL_EVENT;
        Import.Event.Attribute = wasm::WASM_EVENT_ATTRIBUTE_EXCEPTION;
        Import.Event.SigIndex = getEventType(WS);
        Imports.push_back(Import);
        assert(WasmIndices.count(&WS) == 0);
        WasmIndices[&WS] = NumEventImports++;
      }
    }
  }

  // Add imports for GOT globals
  for (const MCSymbol &S : Asm.symbols()) {
    const auto &WS = static_cast<const MCSymbolWasm &>(S);
    if (WS.isUsedInGOT()) {
      wasm::WasmImport Import;
      if (WS.isFunction())
        Import.Module = "GOT.func";
      else
        Import.Module = "GOT.mem";
      Import.Field = WS.getName();
      Import.Kind = wasm::WASM_EXTERNAL_GLOBAL;
      Import.Global = {wasm::WASM_TYPE_I32, true};
      Imports.push_back(Import);
      assert(GOTIndices.count(&WS) == 0);
      GOTIndices[&WS] = NumGlobalImports++;
    }
  }

  // Populate DataSegments and CustomSections, which must be done before
  // populating DataLocations.
  for (MCSection &Sec : Asm) {
    auto &Section = static_cast<MCSectionWasm &>(Sec);
    StringRef SectionName = Section.getName();

    // .init_array sections are handled specially elsewhere.
    if (SectionName.startswith(".init_array"))
      continue;

    // Code is handled separately
    if (Section.getKind().isText())
      continue;

    if (Section.isWasmData()) {
      uint32_t SegmentIndex = DataSegments.size();
      DataSize = alignTo(DataSize, Section.getAlignment());
      DataSegments.emplace_back();
      WasmDataSegment &Segment = DataSegments.back();
      Segment.Name = SectionName;
      Segment.InitFlags =
          Section.getPassive() ? (uint32_t)wasm::WASM_SEGMENT_IS_PASSIVE : 0;
      Segment.Offset = DataSize;
      Segment.Section = &Section;
      addData(Segment.Data, Section);
      Segment.Alignment = Log2_32(Section.getAlignment());
      Segment.LinkerFlags = 0;
      DataSize += Segment.Data.size();
      Section.setSegmentIndex(SegmentIndex);

      if (const MCSymbolWasm *C = Section.getGroup()) {
        Comdats[C->getName()].emplace_back(
            WasmComdatEntry{wasm::WASM_COMDAT_DATA, SegmentIndex});
      }
    } else {
      // Create custom sections
      assert(Sec.getKind().isMetadata());

      StringRef Name = SectionName;

      // For user-defined custom sections, strip the prefix
      if (Name.startswith(".custom_section."))
        Name = Name.substr(strlen(".custom_section."));

      MCSymbol *Begin = Sec.getBeginSymbol();
      if (Begin) {
        WasmIndices[cast<MCSymbolWasm>(Begin)] = CustomSections.size();
        if (SectionName != Begin->getName())
          report_fatal_error("section name and begin symbol should match: " +
                             Twine(SectionName));
      }

      // Separate out the producers and target features sections
      if (Name == "producers") {
        ProducersSection = std::make_unique<WasmCustomSection>(Name, &Section);
        continue;
      }
      if (Name == "target_features") {
        TargetFeaturesSection =
            std::make_unique<WasmCustomSection>(Name, &Section);
        continue;
      }

      CustomSections.emplace_back(Name, &Section);
    }
  }

  // Populate WasmIndices and DataLocations for defined symbols.
  for (const MCSymbol &S : Asm.symbols()) {
    // Ignore unnamed temporary symbols, which aren't ever exported, imported,
    // or used in relocations.
    if (S.isTemporary() && S.getName().empty())
      continue;

    const auto &WS = static_cast<const MCSymbolWasm &>(S);
    LLVM_DEBUG(
        dbgs() << "MCSymbol: " << toString(WS.getType()) << " '" << S << "'"
               << " isDefined=" << S.isDefined() << " isExternal="
               << S.isExternal() << " isTemporary=" << S.isTemporary()
               << " isWeak=" << WS.isWeak() << " isHidden=" << WS.isHidden()
               << " isVariable=" << WS.isVariable() << "\n");

    if (WS.isVariable())
      continue;
    if (WS.isComdat() && !WS.isDefined())
      continue;

    if (WS.isFunction()) {
      unsigned Index;
      if (WS.isDefined()) {
        if (WS.getOffset() != 0)
          report_fatal_error(
              "function sections must contain one function each");

        if (WS.getSize() == nullptr)
          report_fatal_error(
              "function symbols must have a size set with .size");

        // A definition. Write out the function body.
        Index = NumFunctionImports + Functions.size();
        WasmFunction Func;
        Func.SigIndex = getFunctionType(WS);
        Func.Sym = &WS;
        WasmIndices[&WS] = Index;
        Functions.push_back(Func);

        auto &Section = static_cast<MCSectionWasm &>(WS.getSection());
        if (const MCSymbolWasm *C = Section.getGroup()) {
          Comdats[C->getName()].emplace_back(
              WasmComdatEntry{wasm::WASM_COMDAT_FUNCTION, Index});
        }

        if (WS.hasExportName()) {
          wasm::WasmExport Export;
          Export.Name = WS.getExportName();
          Export.Kind = wasm::WASM_EXTERNAL_FUNCTION;
          Export.Index = Index;
          Exports.push_back(Export);
        }
      } else {
        // An import; the index was assigned above.
        Index = WasmIndices.find(&WS)->second;
      }

      LLVM_DEBUG(dbgs() << "  -> function index: " << Index << "\n");

    } else if (WS.isData()) {
      if (!isInSymtab(WS))
        continue;

      if (!WS.isDefined()) {
        LLVM_DEBUG(dbgs() << "  -> segment index: -1"
                          << "\n");
        continue;
      }

      if (!WS.getSize())
        report_fatal_error("data symbols must have a size set with .size: " +
                           WS.getName());

      int64_t Size = 0;
      if (!WS.getSize()->evaluateAsAbsolute(Size, Layout))
        report_fatal_error(".size expression must be evaluatable");

      auto &DataSection = static_cast<MCSectionWasm &>(WS.getSection());
      if (!DataSection.isWasmData())
        report_fatal_error("data symbols must live in a data section: " +
                           WS.getName());

      // For each data symbol, export it in the symtab as a reference to the
      // corresponding Wasm data segment.
      wasm::WasmDataReference Ref = wasm::WasmDataReference{
          DataSection.getSegmentIndex(), Layout.getSymbolOffset(WS),
          static_cast<uint64_t>(Size)};
      DataLocations[&WS] = Ref;
      LLVM_DEBUG(dbgs() << "  -> segment index: " << Ref.Segment << "\n");

    } else if (WS.isGlobal()) {
      // A "true" Wasm global (currently just __stack_pointer)
      if (WS.isDefined()) {
        assert(WasmIndices.count(&WS) == 0);
        wasm::WasmGlobal Global;
        Global.Type = WS.getGlobalType();
        Global.Index = NumGlobalImports + Globals.size();
        switch (Global.Type.Type) {
        case wasm::WASM_TYPE_I32:
          Global.InitExpr.Opcode = wasm::WASM_OPCODE_I32_CONST;
          break;
        case wasm::WASM_TYPE_I64:
          Global.InitExpr.Opcode = wasm::WASM_OPCODE_I64_CONST;
          break;
        case wasm::WASM_TYPE_F32:
          Global.InitExpr.Opcode = wasm::WASM_OPCODE_F32_CONST;
          break;
        case wasm::WASM_TYPE_F64:
          Global.InitExpr.Opcode = wasm::WASM_OPCODE_F64_CONST;
          break;
        case wasm::WASM_TYPE_EXTERNREF:
          Global.InitExpr.Opcode = wasm::WASM_OPCODE_REF_NULL;
          break;
        default:
          llvm_unreachable("unexpected type");
        }
        WasmIndices[&WS] = Global.Index;
        Globals.push_back(Global);
      } else {
        // An import; the index was assigned above
        LLVM_DEBUG(dbgs() << "  -> global index: "
                          << WasmIndices.find(&WS)->second << "\n");
      }
    } else if (WS.isEvent()) {
      // C++ exception symbol (__cpp_exception)
      unsigned Index;
      if (WS.isDefined()) {
        assert(WasmIndices.count(&WS) == 0);
        Index = NumEventImports + Events.size();
        wasm::WasmEventType Event;
        Event.SigIndex = getEventType(WS);
        Event.Attribute = wasm::WASM_EVENT_ATTRIBUTE_EXCEPTION;
        WasmIndices[&WS] = Index;
        Events.push_back(Event);
      } else {
        // An import; the index was assigned above.
        assert(WasmIndices.count(&WS) > 0);
      }
      LLVM_DEBUG(dbgs() << "  -> event index: " << WasmIndices.find(&WS)->second
                        << "\n");

    } else {
      assert(WS.isSection());
    }
  }

  // Populate WasmIndices and DataLocations for aliased symbols.  We need to
  // process these in a separate pass because we need to have processed the
  // target of the alias before the alias itself and the symbols are not
  // necessarily ordered in this way.
  for (const MCSymbol &S : Asm.symbols()) {
    if (!S.isVariable())
      continue;

    assert(S.isDefined());

    const MCSymbolWasm *Base = cast<MCSymbolWasm>(Layout.getBaseSymbol(S));

    // Find the target symbol of this weak alias and export that index
    const auto &WS = static_cast<const MCSymbolWasm &>(S);
    LLVM_DEBUG(dbgs() << WS.getName() << ": weak alias of '" << *Base << "'\n");

    if (Base->isFunction()) {
      assert(WasmIndices.count(Base) > 0);
      uint32_t WasmIndex = WasmIndices.find(Base)->second;
      assert(WasmIndices.count(&WS) == 0);
      WasmIndices[&WS] = WasmIndex;
      LLVM_DEBUG(dbgs() << "  -> index:" << WasmIndex << "\n");
    } else if (Base->isData()) {
      auto &DataSection = static_cast<MCSectionWasm &>(WS.getSection());
      uint64_t Offset = Layout.getSymbolOffset(S);
      int64_t Size = 0;
      // For data symbol alias we use the size of the base symbol as the
      // size of the alias.  When an offset from the base is involved this
      // can result in a offset + size goes past the end of the data section
      // which out object format doesn't support.  So we must clamp it.
      if (!Base->getSize()->evaluateAsAbsolute(Size, Layout))
        report_fatal_error(".size expression must be evaluatable");
      const WasmDataSegment &Segment =
          DataSegments[DataSection.getSegmentIndex()];
      Size =
          std::min(static_cast<uint64_t>(Size), Segment.Data.size() - Offset);
      wasm::WasmDataReference Ref = wasm::WasmDataReference{
          DataSection.getSegmentIndex(),
          static_cast<uint32_t>(Layout.getSymbolOffset(S)),
          static_cast<uint32_t>(Size)};
      DataLocations[&WS] = Ref;
      LLVM_DEBUG(dbgs() << "  -> index:" << Ref.Segment << "\n");
    } else {
      report_fatal_error("don't yet support global/event aliases");
    }
  }

  // Finally, populate the symbol table itself, in its "natural" order.
  for (const MCSymbol &S : Asm.symbols()) {
    const auto &WS = static_cast<const MCSymbolWasm &>(S);
    if (!isInSymtab(WS)) {
      WS.setIndex(InvalidIndex);
      continue;
    }
    LLVM_DEBUG(dbgs() << "adding to symtab: " << WS << "\n");

    uint32_t Flags = 0;
    if (WS.isWeak())
      Flags |= wasm::WASM_SYMBOL_BINDING_WEAK;
    if (WS.isHidden())
      Flags |= wasm::WASM_SYMBOL_VISIBILITY_HIDDEN;
    if (!WS.isExternal() && WS.isDefined())
      Flags |= wasm::WASM_SYMBOL_BINDING_LOCAL;
    if (WS.isUndefined())
      Flags |= wasm::WASM_SYMBOL_UNDEFINED;
    if (WS.isNoStrip()) {
      Flags |= wasm::WASM_SYMBOL_NO_STRIP;
      if (isEmscripten()) {
        Flags |= wasm::WASM_SYMBOL_EXPORTED;
      }
    }
    if (WS.hasImportName())
      Flags |= wasm::WASM_SYMBOL_EXPLICIT_NAME;
    if (WS.hasExportName())
      Flags |= wasm::WASM_SYMBOL_EXPORTED;

    wasm::WasmSymbolInfo Info;
    Info.Name = WS.getName();
    Info.Kind = WS.getType();
    Info.Flags = Flags;
    if (!WS.isData()) {
      assert(WasmIndices.count(&WS) > 0);
      Info.ElementIndex = WasmIndices.find(&WS)->second;
    } else if (WS.isDefined()) {
      assert(DataLocations.count(&WS) > 0);
      Info.DataRef = DataLocations.find(&WS)->second;
    }
    WS.setIndex(SymbolInfos.size());
    SymbolInfos.emplace_back(Info);
  }

  {
    auto HandleReloc = [&](const WasmRelocationEntry &Rel) {
      // Functions referenced by a relocation need to put in the table.  This is
      // purely to make the object file's provisional values readable, and is
      // ignored by the linker, which re-calculates the relocations itself.
      if (Rel.Type != wasm::R_WASM_TABLE_INDEX_I32 &&
          Rel.Type != wasm::R_WASM_TABLE_INDEX_I64 &&
          Rel.Type != wasm::R_WASM_TABLE_INDEX_SLEB &&
          Rel.Type != wasm::R_WASM_TABLE_INDEX_SLEB64 &&
          Rel.Type != wasm::R_WASM_TABLE_INDEX_REL_SLEB)
        return;
      assert(Rel.Symbol->isFunction());
      const MCSymbolWasm *Base =
          cast<MCSymbolWasm>(Layout.getBaseSymbol(*Rel.Symbol));
      uint32_t FunctionIndex = WasmIndices.find(Base)->second;
      uint32_t TableIndex = TableElems.size() + InitialTableOffset;
      if (TableIndices.try_emplace(Base, TableIndex).second) {
        LLVM_DEBUG(dbgs() << "  -> adding " << Base->getName()
                          << " to table: " << TableIndex << "\n");
        TableElems.push_back(FunctionIndex);
        registerFunctionType(*Base);
      }
    };

    for (const WasmRelocationEntry &RelEntry : CodeRelocations)
      HandleReloc(RelEntry);
    for (const WasmRelocationEntry &RelEntry : DataRelocations)
      HandleReloc(RelEntry);
  }

  // Translate .init_array section contents into start functions.
  for (const MCSection &S : Asm) {
    const auto &WS = static_cast<const MCSectionWasm &>(S);
    if (WS.getName().startswith(".fini_array"))
      report_fatal_error(".fini_array sections are unsupported");
    if (!WS.getName().startswith(".init_array"))
      continue;
    if (WS.getFragmentList().empty())
      continue;

    // init_array is expected to contain a single non-empty data fragment
    if (WS.getFragmentList().size() != 3)
      report_fatal_error("only one .init_array section fragment supported");

    auto IT = WS.begin();
    const MCFragment &EmptyFrag = *IT;
    if (EmptyFrag.getKind() != MCFragment::FT_Data)
      report_fatal_error(".init_array section should be aligned");

    IT = std::next(IT);
    const MCFragment &AlignFrag = *IT;
    if (AlignFrag.getKind() != MCFragment::FT_Align)
      report_fatal_error(".init_array section should be aligned");
    if (cast<MCAlignFragment>(AlignFrag).getAlignment() != (is64Bit() ? 8 : 4))
      report_fatal_error(".init_array section should be aligned for pointers");

    const MCFragment &Frag = *std::next(IT);
    if (Frag.hasInstructions() || Frag.getKind() != MCFragment::FT_Data)
      report_fatal_error("only data supported in .init_array section");

    uint16_t Priority = UINT16_MAX;
    unsigned PrefixLength = strlen(".init_array");
    if (WS.getName().size() > PrefixLength) {
      if (WS.getName()[PrefixLength] != '.')
        report_fatal_error(
            ".init_array section priority should start with '.'");
      if (WS.getName().substr(PrefixLength + 1).getAsInteger(10, Priority))
        report_fatal_error("invalid .init_array section priority");
    }
    const auto &DataFrag = cast<MCDataFragment>(Frag);
    const SmallVectorImpl<char> &Contents = DataFrag.getContents();
    for (const uint8_t *
             P = (const uint8_t *)Contents.data(),
            *End = (const uint8_t *)Contents.data() + Contents.size();
         P != End; ++P) {
      if (*P != 0)
        report_fatal_error("non-symbolic data in .init_array section");
    }
    for (const MCFixup &Fixup : DataFrag.getFixups()) {
      assert(Fixup.getKind() ==
             MCFixup::getKindForSize(is64Bit() ? 8 : 4, false));
      const MCExpr *Expr = Fixup.getValue();
      auto *SymRef = dyn_cast<MCSymbolRefExpr>(Expr);
      if (!SymRef)
        report_fatal_error("fixups in .init_array should be symbol references");
      const auto &TargetSym = cast<const MCSymbolWasm>(SymRef->getSymbol());
      if (TargetSym.getIndex() == InvalidIndex)
        report_fatal_error("symbols in .init_array should exist in symtab");
      if (!TargetSym.isFunction())
        report_fatal_error("symbols in .init_array should be for functions");
      InitFuncs.push_back(
          std::make_pair(Priority, TargetSym.getIndex()));
    }
  }

  // Write out the Wasm header.
  writeHeader(Asm);

  writeTypeSection(Signatures);
  writeImportSection(Imports, DataSize, TableElems.size());
  writeFunctionSection(Functions);
  // Skip the "table" section; we import the table instead.
  // Skip the "memory" section; we import the memory instead.
  writeEventSection(Events);
  writeGlobalSection(Globals);
  writeExportSection(Exports);
  writeElemSection(TableElems);
  writeDataCountSection();
  uint32_t CodeSectionIndex = writeCodeSection(Asm, Layout, Functions);
  uint32_t DataSectionIndex = writeDataSection(Layout);
  for (auto &CustomSection : CustomSections)
    writeCustomSection(CustomSection, Asm, Layout);
  writeLinkingMetaDataSection(SymbolInfos, InitFuncs, Comdats);
  writeRelocSection(CodeSectionIndex, "CODE", CodeRelocations);
  writeRelocSection(DataSectionIndex, "DATA", DataRelocations);
  writeCustomRelocSections();
  if (ProducersSection)
    writeCustomSection(*ProducersSection, Asm, Layout);
  if (TargetFeaturesSection)
    writeCustomSection(*TargetFeaturesSection, Asm, Layout);

  // TODO: Translate the .comment section to the output.
  return W.OS.tell() - StartOffset;
}

std::unique_ptr<MCObjectWriter>
llvm::createWasmObjectWriter(std::unique_ptr<MCWasmObjectTargetWriter> MOTW,
                             raw_pwrite_stream &OS) {
  return std::make_unique<WasmObjectWriter>(std::move(MOTW), OS);
}
