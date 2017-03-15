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
#include "llvm/Support/Wasm.h"
#include <vector>

using namespace llvm;

#undef DEBUG_TYPE
#define DEBUG_TYPE "reloc-info"

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

// This record records information about a call_indirect which needs its
// type index fixed up once we've computed type indices.
struct TypeIndexFixup {
  uint64_t Offset;
  const MCSymbolWasm *Symbol;
  const MCSectionWasm *FixupSection;
  TypeIndexFixup(uint64_t O, const MCSymbolWasm *S, MCSectionWasm *F)
    : Offset(O), Symbol(S), FixupSection(F) {}
};

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

  // Fixups for call_indirect type indices.
  std::vector<TypeIndexFixup> TypeIndexFixups;

  // Index values to use for fixing up call_indirect type indices.
  std::vector<uint32_t> TypeIndexFixupTypes;

  // TargetObjectWriter wrappers.
  bool is64Bit() const { return TargetObjectWriter->is64Bit(); }
  unsigned getRelocType(MCContext &Ctx, const MCValue &Target,
                        const MCFixup &Fixup, bool IsPCRel) const {
    return TargetObjectWriter->getRelocType(Ctx, Target, Fixup, IsPCRel);
  }

  void startSection(SectionBookkeeping &Section, unsigned SectionId,
                    const char *Name = nullptr);
  void endSection(SectionBookkeeping &Section);

public:
  WasmObjectWriter(MCWasmObjectTargetWriter *MOTW, raw_pwrite_stream &OS)
      : MCObjectWriter(OS, /*IsLittleEndian=*/true), TargetObjectWriter(MOTW) {}

private:
  void reset() override {
    MCObjectWriter::reset();
  }

  ~WasmObjectWriter() override;

  void writeHeader(const MCAssembler &Asm);

  void writeValueType(wasm::ValType Ty) {
    encodeSLEB128(int32_t(Ty), getStream());
  }

  void recordRelocation(MCAssembler &Asm, const MCAsmLayout &Layout,
                        const MCFragment *Fragment, const MCFixup &Fixup,
                        MCValue Target, bool &IsPCRel,
                        uint64_t &FixedValue) override;

  void executePostLayoutBinding(MCAssembler &Asm,
                                const MCAsmLayout &Layout) override;

  void writeObject(MCAssembler &Asm, const MCAsmLayout &Layout) override;
};
} // end anonymous namespace

WasmObjectWriter::~WasmObjectWriter() {}

// Return the padding size to write a 32-bit value into a 5-byte ULEB128.
static unsigned PaddingFor5ByteULEB128(uint32_t X) {
  return X == 0 ? 4 : (4u - (31u - countLeadingZeros(X)) / 7u);
}

// Return the padding size to write a 32-bit value into a 5-byte SLEB128.
static unsigned PaddingFor5ByteSLEB128(int32_t X) {
  return 5 - getSLEB128Size(X);
}

// Write out a section header and a patchable section size field.
void WasmObjectWriter::startSection(SectionBookkeeping &Section,
                                    unsigned SectionId,
                                    const char *Name) {
  assert((Name != nullptr) == (SectionId == wasm::WASM_SEC_CUSTOM) &&
         "Only custom sections can have names");

  encodeULEB128(SectionId, getStream());

  Section.SizeOffset = getStream().tell();

  // The section size. We don't know the size yet, so reserve enough space
  // for any 32-bit value; we'll patch it later.
  encodeULEB128(UINT32_MAX, getStream());

  // The position where the section starts, for measuring its size.
  Section.ContentsOffset = getStream().tell();

  // Custom sections in wasm also have a string identifier.
  if (SectionId == wasm::WASM_SEC_CUSTOM) {
    encodeULEB128(strlen(Name), getStream());
    writeBytes(Name);
  }
}

// Now that the section is complete and we know how big it is, patch up the
// section size field at the start of the section.
void WasmObjectWriter::endSection(SectionBookkeeping &Section) {
  uint64_t Size = getStream().tell() - Section.ContentsOffset;
  if (uint32_t(Size) != Size)
    report_fatal_error("section size does not fit in a uint32_t");

  unsigned Padding = PaddingFor5ByteULEB128(Size);

  // Write the final section size to the payload_len field, which follows
  // the section id byte.
  uint8_t Buffer[16];
  unsigned SizeLen = encodeULEB128(Size, Buffer, Padding);
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
                                        bool &IsPCRel, uint64_t &FixedValue) {
  MCSectionWasm &FixupSection = cast<MCSectionWasm>(*Fragment->getParent());
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

  bool ViaWeakRef = false;
  if (SymA && SymA->isVariable()) {
    const MCExpr *Expr = SymA->getVariableValue();
    if (const auto *Inner = dyn_cast<MCSymbolRefExpr>(Expr)) {
      if (Inner->getKind() == MCSymbolRefExpr::VK_WEAKREF) {
        SymA = cast<MCSymbolWasm>(&Inner->getSymbol());
        ViaWeakRef = true;
      }
    }
  }

  // Put any constant offset in an addend. Offsets can be negative, and
  // LLVM expects wrapping, in contrast to wasm's immediates which can't
  // be negative and don't wrap.
  FixedValue = 0;

  if (SymA) {
    if (ViaWeakRef)
      llvm_unreachable("weakref used in reloc not yet implemented");
    else
      SymA->setUsedInReloc();
  }

  if (RefA) {
    if (RefA->getKind() == MCSymbolRefExpr::VK_WebAssembly_TYPEINDEX) {
      TypeIndexFixups.push_back(TypeIndexFixup(FixupOffset, SymA,
                                               &FixupSection));
      return;
    }
  }

  unsigned Type = getRelocType(Ctx, Target, Fixup, IsPCRel);

  WasmRelocationEntry Rec(FixupOffset, SymA, C, Type, &FixupSection);

  if (FixupSection.hasInstructions())
    CodeRelocations.push_back(Rec);
  else
    DataRelocations.push_back(Rec);
}

namespace {

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

// A wasm import to be written into the import section.
struct WasmImport {
  StringRef ModuleName;
  StringRef FieldName;
  unsigned Kind;
  uint32_t Type;
};

// A wasm function to be written into the function section.
struct WasmFunction {
  unsigned Type;
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
  unsigned Type;
  bool IsMutable;
  uint32_t InitialValue;
};

} // end anonymous namespace

// Write X as an (unsigned) LEB value at offset Offset in Stream, padded
// to allow patching.
static void
WritePatchableLEB(raw_pwrite_stream &Stream, uint32_t X, uint64_t Offset) {
  uint8_t Buffer[5];
  unsigned Padding = PaddingFor5ByteULEB128(X);
  unsigned SizeLen = encodeULEB128(X, Buffer, Padding);
  assert(SizeLen == 5);
  Stream.pwrite((char *)Buffer, SizeLen, Offset);
}

// Write X as an signed LEB value at offset Offset in Stream, padded
// to allow patching.
static void
WritePatchableSLEB(raw_pwrite_stream &Stream, int32_t X, uint64_t Offset) {
  uint8_t Buffer[5];
  unsigned Padding = PaddingFor5ByteSLEB128(X);
  unsigned SizeLen = encodeSLEB128(X, Buffer, Padding);
  assert(SizeLen == 5);
  Stream.pwrite((char *)Buffer, SizeLen, Offset);
}

// Write X as a plain integer value at offset Offset in Stream.
static void WriteI32(raw_pwrite_stream &Stream, uint32_t X, uint64_t Offset) {
  uint8_t Buffer[4];
  support::endian::write32le(Buffer, X);
  Stream.pwrite((char *)Buffer, sizeof(Buffer), Offset);
}

// Compute a value to write into the code at the location covered
// by RelEntry. This value isn't used by the static linker, since
// we have addends; it just serves to make the code more readable
// and to make standalone wasm modules directly usable.
static uint32_t ProvisionalValue(const WasmRelocationEntry &RelEntry) {
  const MCSymbolWasm *Sym = RelEntry.Symbol;

  // For undefined symbols, use a hopefully invalid value.
  if (!Sym->isDefined(false))
    return UINT32_MAX;

  MCSectionWasm &Section =
    cast<MCSectionWasm>(RelEntry.Symbol->getSection(false));
  uint64_t Address = Section.getSectionOffset() + RelEntry.Addend;

  // Ignore overflow. LLVM allows address arithmetic to silently wrap.
  uint32_t Value = Address;

  return Value;
}

// Apply the portions of the relocation records that we can handle ourselves
// directly.
static void ApplyRelocations(
    ArrayRef<WasmRelocationEntry> Relocations,
    raw_pwrite_stream &Stream,
    DenseMap<const MCSymbolWasm *, uint32_t> &SymbolIndices,
    uint64_t ContentsOffset)
{
  for (const WasmRelocationEntry &RelEntry : Relocations) {
    uint64_t Offset = ContentsOffset +
                      RelEntry.FixupSection->getSectionOffset() +
                      RelEntry.Offset;
    switch (RelEntry.Type) {
    case wasm::R_WEBASSEMBLY_FUNCTION_INDEX_LEB: {
      uint32_t Index = SymbolIndices[RelEntry.Symbol];
      assert(RelEntry.Addend == 0);

      WritePatchableLEB(Stream, Index, Offset);
      break;
    }
    case wasm::R_WEBASSEMBLY_TABLE_INDEX_SLEB: {
      uint32_t Index = SymbolIndices[RelEntry.Symbol];
      assert(RelEntry.Addend == 0);

      WritePatchableSLEB(Stream, Index, Offset);
      break;
    }
    case wasm::R_WEBASSEMBLY_GLOBAL_ADDR_SLEB: {
      uint32_t Value = ProvisionalValue(RelEntry);

      WritePatchableSLEB(Stream, Value, Offset);
      break;
    }
    case wasm::R_WEBASSEMBLY_GLOBAL_ADDR_LEB: {
      uint32_t Value = ProvisionalValue(RelEntry);

      WritePatchableLEB(Stream, Value, Offset);
      break;
    }
    case wasm::R_WEBASSEMBLY_TABLE_INDEX_I32: {
      uint32_t Index = SymbolIndices[RelEntry.Symbol];
      assert(RelEntry.Addend == 0);

      WriteI32(Stream, Index, Offset);
      break;
    }
    case wasm::R_WEBASSEMBLY_GLOBAL_ADDR_I32: {
      uint32_t Value = ProvisionalValue(RelEntry);

      WriteI32(Stream, Value, Offset);
      break;
    }
    default:
      break;
    }
  }
}

// Write out the portions of the relocation records that the linker will
// need to handle.
static void WriteRelocations(
    ArrayRef<WasmRelocationEntry> Relocations,
    raw_pwrite_stream &Stream,
    DenseMap<const MCSymbolWasm *, uint32_t> &SymbolIndices)
{
  for (const WasmRelocationEntry RelEntry : Relocations) {
    encodeULEB128(RelEntry.Type, Stream);

    uint64_t Offset = RelEntry.Offset +
                      RelEntry.FixupSection->getSectionOffset();
    uint32_t Index = SymbolIndices[RelEntry.Symbol];
    int64_t Addend = RelEntry.Addend;

    switch (RelEntry.Type) {
    case wasm::R_WEBASSEMBLY_FUNCTION_INDEX_LEB:
    case wasm::R_WEBASSEMBLY_TABLE_INDEX_SLEB:
    case wasm::R_WEBASSEMBLY_TABLE_INDEX_I32:
      encodeULEB128(Offset, Stream);
      encodeULEB128(Index, Stream);
      assert(Addend == 0 && "addends not supported for functions");
      break;
    case wasm::R_WEBASSEMBLY_GLOBAL_ADDR_LEB:
    case wasm::R_WEBASSEMBLY_GLOBAL_ADDR_SLEB:
    case wasm::R_WEBASSEMBLY_GLOBAL_ADDR_I32:
      encodeULEB128(Offset, Stream);
      encodeULEB128(Index, Stream);
      encodeSLEB128(Addend, Stream);
      break;
    default:
      llvm_unreachable("unsupported relocation type");
    }
  }
}

void WasmObjectWriter::writeObject(MCAssembler &Asm,
                                   const MCAsmLayout &Layout) {
  MCContext &Ctx = Asm.getContext();
  unsigned PtrType = is64Bit() ? wasm::WASM_TYPE_I64 : wasm::WASM_TYPE_I32;

  // Collect information from the available symbols.
  DenseMap<WasmFunctionType, unsigned, WasmFunctionTypeDenseMapInfo>
      FunctionTypeIndices;
  SmallVector<WasmFunctionType, 4> FunctionTypes;
  SmallVector<WasmFunction, 4> Functions;
  SmallVector<uint32_t, 4> TableElems;
  SmallVector<WasmGlobal, 4> Globals;
  SmallVector<WasmImport, 4> Imports;
  SmallVector<WasmExport, 4> Exports;
  DenseMap<const MCSymbolWasm *, uint32_t> SymbolIndices;
  SmallPtrSet<const MCSymbolWasm *, 4> IsAddressTaken;
  unsigned NumFuncImports = 0;
  unsigned NumGlobalImports = 0;
  SmallVector<char, 0> DataBytes;

  // Populate the IsAddressTaken set.
  for (WasmRelocationEntry RelEntry : CodeRelocations) {
    switch (RelEntry.Type) {
    case wasm::R_WEBASSEMBLY_TABLE_INDEX_SLEB:
    case wasm::R_WEBASSEMBLY_GLOBAL_ADDR_SLEB:
      IsAddressTaken.insert(RelEntry.Symbol);
      break;
    default:
      break;
    }
  }
  for (WasmRelocationEntry RelEntry : DataRelocations) {
    switch (RelEntry.Type) {
    case wasm::R_WEBASSEMBLY_TABLE_INDEX_I32:
    case wasm::R_WEBASSEMBLY_GLOBAL_ADDR_I32:
      IsAddressTaken.insert(RelEntry.Symbol);
      break;
    default:
      break;
    }
  }

  // Populate the Imports set.
  for (const MCSymbol &S : Asm.symbols()) {
    const auto &WS = static_cast<const MCSymbolWasm &>(S);
    unsigned Type;

    if (WS.isFunction()) {
      // Prepare the function's type, if we haven't seen it yet.
      WasmFunctionType F;
      F.Returns = WS.getReturns();
      F.Params = WS.getParams();
      auto Pair =
          FunctionTypeIndices.insert(std::make_pair(F, FunctionTypes.size()));
      if (Pair.second)
        FunctionTypes.push_back(F);

      Type = Pair.first->second;
    } else {
      Type = PtrType;
    }

    // If the symbol is not defined in this translation unit, import it.
    if (!WS.isTemporary() && !WS.isDefined(/*SetUsed=*/false)) {
      WasmImport Import;
      Import.ModuleName = WS.getModuleName();
      Import.FieldName = WS.getName();

      if (WS.isFunction()) {
        Import.Kind = wasm::WASM_EXTERNAL_FUNCTION;
        Import.Type = Type;
        SymbolIndices[&WS] = NumFuncImports;
        ++NumFuncImports;
      } else {
        Import.Kind = wasm::WASM_EXTERNAL_GLOBAL;
        Import.Type = Type;
        SymbolIndices[&WS] = NumGlobalImports;
        ++NumGlobalImports;
      }

      Imports.push_back(Import);
    }
  }

  // In the special .global_variables section, we've encoded global
  // variables used by the function. Translate them into the Globals
  // list.
  MCSectionWasm *GlobalVars = Ctx.getWasmSection(".global_variables", 0, 0);
  if (!GlobalVars->getFragmentList().empty()) {
    if (GlobalVars->getFragmentList().size() != 1)
      report_fatal_error("only one .global_variables fragment supported");
    const MCFragment &Frag = *GlobalVars->begin();
    if (Frag.hasInstructions() || Frag.getKind() != MCFragment::FT_Data)
      report_fatal_error("only data supported in .global_variables");
    const MCDataFragment &DataFrag = cast<MCDataFragment>(Frag);
    if (!DataFrag.getFixups().empty())
      report_fatal_error("fixups not supported in .global_variables");
    const SmallVectorImpl<char> &Contents = DataFrag.getContents();
    for (char p : Contents) {
      WasmGlobal G;
      G.Type = uint8_t(p);
      G.IsMutable = true;
      G.InitialValue = 0;
      Globals.push_back(G);
    }
  }

  // Handle defined symbols.
  for (const MCSymbol &S : Asm.symbols()) {
    // Ignore unnamed temporary symbols, which aren't ever exported, imported,
    // or used in relocations.
    if (S.isTemporary() && S.getName().empty())
      continue;
    const auto &WS = static_cast<const MCSymbolWasm &>(S);
    unsigned Index;
    if (WS.isFunction()) {
      // Prepare the function's type, if we haven't seen it yet.
      WasmFunctionType F;
      F.Returns = WS.getReturns();
      F.Params = WS.getParams();
      auto Pair =
          FunctionTypeIndices.insert(std::make_pair(F, FunctionTypes.size()));
      if (Pair.second)
        FunctionTypes.push_back(F);

      unsigned Type = Pair.first->second;

      if (WS.isDefined(/*SetUsed=*/false)) {
        // A definition. Take the next available index.
        Index = NumFuncImports + Functions.size();

        // Prepare the function.
        WasmFunction Func;
        Func.Type = Type;
        Func.Sym = &WS;
        SymbolIndices[&WS] = Index;
        Functions.push_back(Func);
      } else {
        // An import; the index was assigned above.
        Index = SymbolIndices.find(&WS)->second;
      }

      // If needed, prepare the function to be called indirectly.
      if (IsAddressTaken.count(&WS))
        TableElems.push_back(Index);
    } else {
      // For now, ignore temporary non-function symbols.
      if (S.isTemporary())
        continue;

      if (WS.getOffset() != 0)
        report_fatal_error("data sections must contain one variable each");
      if (!WS.getSize())
        report_fatal_error("data symbols must have a size set with .size");

      int64_t Size = 0;
      if (!WS.getSize()->evaluateAsAbsolute(Size, Layout))
        report_fatal_error(".size expression must be evaluatable");

      if (WS.isDefined(false)) {
        MCSectionWasm &DataSection =
            static_cast<MCSectionWasm &>(WS.getSection());

        if (uint64_t(Size) != Layout.getSectionFileSize(&DataSection))
          report_fatal_error("data sections must contain at most one variable");

        DataBytes.resize(alignTo(DataBytes.size(), DataSection.getAlignment()));

        DataSection.setSectionOffset(DataBytes.size());

        for (MCSection::iterator I = DataSection.begin(), E = DataSection.end();
             I != E; ++I) {
          const MCFragment &Frag = *I;
          if (Frag.hasInstructions())
            report_fatal_error("only data supported in data sections");

          if (const MCAlignFragment *Align = dyn_cast<MCAlignFragment>(&Frag)) {
            if (Align->getValueSize() != 1)
              report_fatal_error("only byte values supported for alignment");
            // If nops are requested, use zeros, as this is the data section.
            uint8_t Value = Align->hasEmitNops() ? 0 : Align->getValue();
            uint64_t Size = std::min<uint64_t>(alignTo(DataBytes.size(),
                                                       Align->getAlignment()),
                                               DataBytes.size() +
                                                   Align->getMaxBytesToEmit());
            DataBytes.resize(Size, Value);
          } else if (const MCFillFragment *Fill =
                                              dyn_cast<MCFillFragment>(&Frag)) {
            DataBytes.insert(DataBytes.end(), Size, Fill->getValue());
          } else {
            const MCDataFragment &DataFrag = cast<MCDataFragment>(Frag);
            const SmallVectorImpl<char> &Contents = DataFrag.getContents();

            DataBytes.insert(DataBytes.end(), Contents.begin(), Contents.end());
          }
        }

        // For each external global, prepare a corresponding wasm global
        // holding its address.
        if (WS.isExternal()) {
          Index = NumGlobalImports + Globals.size();

          WasmGlobal Global;
          Global.Type = PtrType;
          Global.IsMutable = false;
          Global.InitialValue = DataSection.getSectionOffset();
          SymbolIndices[&WS] = Index;
          Globals.push_back(Global);
        }
      }
    }

    // If the symbol is visible outside this translation unit, export it.
    if (WS.isExternal()) {
      assert(WS.isDefined(false));
      WasmExport Export;
      Export.FieldName = WS.getName();
      Export.Index = Index;

      if (WS.isFunction())
        Export.Kind = wasm::WASM_EXTERNAL_FUNCTION;
      else
        Export.Kind = wasm::WASM_EXTERNAL_GLOBAL;

      Exports.push_back(Export);
    }
  }

  // Add types for indirect function calls.
  for (const TypeIndexFixup &Fixup : TypeIndexFixups) {
    WasmFunctionType F;
    F.Returns = Fixup.Symbol->getReturns();
    F.Params = Fixup.Symbol->getParams();
    auto Pair =
        FunctionTypeIndices.insert(std::make_pair(F, FunctionTypes.size()));
    if (Pair.second)
      FunctionTypes.push_back(F);

    TypeIndexFixupTypes.push_back(Pair.first->second);
  }

  // Write out the Wasm header.
  writeHeader(Asm);

  SectionBookkeeping Section;

  // === Type Section =========================================================
  if (!FunctionTypes.empty()) {
    startSection(Section, wasm::WASM_SEC_TYPE);

    encodeULEB128(FunctionTypes.size(), getStream());

    for (WasmFunctionType &FuncTy : FunctionTypes) {
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

  // === Import Section ========================================================
  if (!Imports.empty()) {
    startSection(Section, wasm::WASM_SEC_IMPORT);

    encodeULEB128(Imports.size(), getStream());
    for (const WasmImport &Import : Imports) {
      StringRef ModuleName = Import.ModuleName;
      encodeULEB128(ModuleName.size(), getStream());
      writeBytes(ModuleName);

      StringRef FieldName = Import.FieldName;
      encodeULEB128(FieldName.size(), getStream());
      writeBytes(FieldName);

      encodeULEB128(Import.Kind, getStream());

      switch (Import.Kind) {
      case wasm::WASM_EXTERNAL_FUNCTION:
        encodeULEB128(Import.Type, getStream());
        break;
      case wasm::WASM_EXTERNAL_GLOBAL:
        encodeSLEB128(Import.Type, getStream());
        encodeULEB128(0, getStream()); // mutability
        break;
      default:
        llvm_unreachable("unsupported import kind");
      }
    }

    endSection(Section);
  }

  // === Function Section ======================================================
  if (!Functions.empty()) {
    startSection(Section, wasm::WASM_SEC_FUNCTION);

    encodeULEB128(Functions.size(), getStream());
    for (const WasmFunction &Func : Functions)
      encodeULEB128(Func.Type, getStream());

    endSection(Section);
  }

  // === Table Section =========================================================
  // For now, always emit the table section, since indirect calls are not
  // valid without it. In the future, we could perhaps be more clever and omit
  // it if there are no indirect calls.
  startSection(Section, wasm::WASM_SEC_TABLE);

  // The number of tables, fixed to 1 for now.
  encodeULEB128(1, getStream());

  encodeSLEB128(wasm::WASM_TYPE_ANYFUNC, getStream());

  encodeULEB128(0, getStream());                 // flags
  encodeULEB128(TableElems.size(), getStream()); // initial

  endSection(Section);

  // === Memory Section ========================================================
  // For now, always emit the memory section, since loads and stores are not
  // valid without it. In the future, we could perhaps be more clever and omit
  // it if there are no loads or stores.
  startSection(Section, wasm::WASM_SEC_MEMORY);

  encodeULEB128(1, getStream()); // number of memory spaces

  encodeULEB128(0, getStream()); // flags
  encodeULEB128(DataBytes.size(), getStream()); // initial

  endSection(Section);

  // === Global Section ========================================================
  if (!Globals.empty()) {
    startSection(Section, wasm::WASM_SEC_GLOBAL);

    encodeULEB128(Globals.size(), getStream());
    for (const WasmGlobal &Global : Globals) {
      encodeSLEB128(Global.Type, getStream());
      write8(Global.IsMutable);

      write8(wasm::WASM_OPCODE_I32_CONST);
      encodeSLEB128(Global.InitialValue, getStream()); // offset
      write8(wasm::WASM_OPCODE_END);
    }

    endSection(Section);
  }

  // === Export Section ========================================================
  if (!Exports.empty()) {
    startSection(Section, wasm::WASM_SEC_EXPORT);

    encodeULEB128(Exports.size(), getStream());
    for (const WasmExport &Export : Exports) {
      encodeULEB128(Export.FieldName.size(), getStream());
      writeBytes(Export.FieldName);

      encodeSLEB128(Export.Kind, getStream());

      encodeULEB128(Export.Index, getStream());
    }

    endSection(Section);
  }

#if 0 // TODO: Start Section
  if (HaveStartFunction) {
    // === Start Section =========================================================
    startSection(Section, wasm::WASM_SEC_START);

    encodeSLEB128(StartFunction, getStream());

    endSection(Section);
  }
#endif

  // === Elem Section ==========================================================
  if (!TableElems.empty()) {
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

  // === Code Section ==========================================================
  if (!Functions.empty()) {
    startSection(Section, wasm::WASM_SEC_CODE);

    encodeULEB128(Functions.size(), getStream());

    for (const WasmFunction &Func : Functions) {
      MCSectionWasm &FuncSection =
          static_cast<MCSectionWasm &>(Func.Sym->getSection());

      if (Func.Sym->isVariable())
        report_fatal_error("weak symbols not supported yet");

      if (Func.Sym->getOffset() != 0)
        report_fatal_error("function sections must contain one function each");

      if (!Func.Sym->getSize())
        report_fatal_error("function symbols must have a size set with .size");

      int64_t Size = 0;
      if (!Func.Sym->getSize()->evaluateAsAbsolute(Size, Layout))
        report_fatal_error(".size expression must be evaluatable");

      encodeULEB128(Size, getStream());

      FuncSection.setSectionOffset(getStream().tell() -
                                   Section.ContentsOffset);

      Asm.writeSectionData(&FuncSection, Layout);
    }

    // Apply the type index fixups for call_indirect etc. instructions.
    for (size_t i = 0, e = TypeIndexFixups.size(); i < e; ++i) {
      uint32_t Type = TypeIndexFixupTypes[i];
      unsigned Padding = PaddingFor5ByteULEB128(Type);

      const TypeIndexFixup &Fixup = TypeIndexFixups[i];
      uint64_t Offset = Fixup.Offset +
                        Fixup.FixupSection->getSectionOffset();

      uint8_t Buffer[16];
      unsigned SizeLen = encodeULEB128(Type, Buffer, Padding);
      assert(SizeLen == 5);
      getStream().pwrite((char *)Buffer, SizeLen,
                         Section.ContentsOffset + Offset);
    }

    // Apply fixups.
    ApplyRelocations(CodeRelocations, getStream(), SymbolIndices,
                     Section.ContentsOffset);

    endSection(Section);
  }

  // === Data Section ==========================================================
  if (!DataBytes.empty()) {
    startSection(Section, wasm::WASM_SEC_DATA);

    encodeULEB128(1, getStream()); // count
    encodeULEB128(0, getStream()); // memory index
    write8(wasm::WASM_OPCODE_I32_CONST);
    encodeSLEB128(0, getStream()); // offset
    write8(wasm::WASM_OPCODE_END);
    encodeULEB128(DataBytes.size(), getStream()); // size
    writeBytes(DataBytes); // data

    // Apply fixups.
    ApplyRelocations(DataRelocations, getStream(), SymbolIndices,
                     Section.ContentsOffset);

    endSection(Section);
  }

  // === Name Section ==========================================================
  if (NumFuncImports != 0 || !Functions.empty()) {
    startSection(Section, wasm::WASM_SEC_CUSTOM, "name");

    encodeULEB128(NumFuncImports + Functions.size(), getStream());
    for (const WasmImport &Import : Imports) {
      if (Import.Kind == wasm::WASM_EXTERNAL_FUNCTION) {
        encodeULEB128(Import.FieldName.size(), getStream());
        writeBytes(Import.FieldName);
        encodeULEB128(0, getStream()); // local count, meaningless for imports
      }
    }
    for (const WasmFunction &Func : Functions) {
      encodeULEB128(Func.Sym->getName().size(), getStream());
      writeBytes(Func.Sym->getName());

      // TODO: Local names.
      encodeULEB128(0, getStream()); // local count
    }

    endSection(Section);
  }

  // See: https://github.com/WebAssembly/tool-conventions/blob/master/Linking.md
  // for descriptions of the reloc sections.

  // === Code Reloc Section ====================================================
  if (!CodeRelocations.empty()) {
    startSection(Section, wasm::WASM_SEC_CUSTOM, "reloc.CODE");

    encodeULEB128(wasm::WASM_SEC_CODE, getStream());

    encodeULEB128(CodeRelocations.size(), getStream());

    WriteRelocations(CodeRelocations, getStream(), SymbolIndices);

    endSection(Section);
  }

  // === Data Reloc Section ====================================================
  if (!DataRelocations.empty()) {
    startSection(Section, wasm::WASM_SEC_CUSTOM, "reloc.DATA");

    encodeULEB128(wasm::WASM_SEC_DATA, getStream());

    encodeULEB128(DataRelocations.size(), getStream());

    WriteRelocations(DataRelocations, getStream(), SymbolIndices);

    endSection(Section);
  }

  // TODO: Translate the .comment section to the output.

  // TODO: Translate debug sections to the output.
}

MCObjectWriter *llvm::createWasmObjectWriter(MCWasmObjectTargetWriter *MOTW,
                                             raw_pwrite_stream &OS) {
  return new WasmObjectWriter(MOTW, OS);
}
