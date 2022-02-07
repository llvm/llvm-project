//===-- lib/MC/XCOFFObjectWriter.cpp - XCOFF file writer ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements XCOFF object file writer information.
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/XCOFF.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSectionXCOFF.h"
#include "llvm/MC/MCSymbolXCOFF.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/MCXCOFFObjectWriter.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"

#include <deque>

using namespace llvm;

// An XCOFF object file has a limited set of predefined sections. The most
// important ones for us (right now) are:
// .text --> contains program code and read-only data.
// .data --> contains initialized data, function descriptors, and the TOC.
// .bss  --> contains uninitialized data.
// Each of these sections is composed of 'Control Sections'. A Control Section
// is more commonly referred to as a csect. A csect is an indivisible unit of
// code or data, and acts as a container for symbols. A csect is mapped
// into a section based on its storage-mapping class, with the exception of
// XMC_RW which gets mapped to either .data or .bss based on whether it's
// explicitly initialized or not.
//
// We don't represent the sections in the MC layer as there is nothing
// interesting about them at at that level: they carry information that is
// only relevant to the ObjectWriter, so we materialize them in this class.
namespace {

constexpr unsigned DefaultSectionAlign = 4;
constexpr int16_t MaxSectionIndex = INT16_MAX;

// Packs the csect's alignment and type into a byte.
uint8_t getEncodedType(const MCSectionXCOFF *);

struct XCOFFRelocation {
  uint32_t SymbolTableIndex;
  uint32_t FixupOffsetInCsect;
  uint8_t SignAndSize;
  uint8_t Type;
};

// Wrapper around an MCSymbolXCOFF.
struct Symbol {
  const MCSymbolXCOFF *const MCSym;
  uint32_t SymbolTableIndex;

  XCOFF::StorageClass getStorageClass() const {
    return MCSym->getStorageClass();
  }
  StringRef getSymbolTableName() const { return MCSym->getSymbolTableName(); }
  Symbol(const MCSymbolXCOFF *MCSym) : MCSym(MCSym), SymbolTableIndex(-1) {}
};

// Wrapper for an MCSectionXCOFF.
// It can be a Csect or debug section or DWARF section and so on.
struct XCOFFSection {
  const MCSectionXCOFF *const MCSec;
  uint32_t SymbolTableIndex;
  uint32_t Address;
  uint32_t Size;

  SmallVector<Symbol, 1> Syms;
  SmallVector<XCOFFRelocation, 1> Relocations;
  StringRef getSymbolTableName() const { return MCSec->getSymbolTableName(); }
  XCOFFSection(const MCSectionXCOFF *MCSec)
      : MCSec(MCSec), SymbolTableIndex(-1), Address(-1), Size(0) {}
};

// Type to be used for a container representing a set of csects with
// (approximately) the same storage mapping class. For example all the csects
// with a storage mapping class of `xmc_pr` will get placed into the same
// container.
using CsectGroup = std::deque<XCOFFSection>;
using CsectGroups = std::deque<CsectGroup *>;

// The basic section entry defination. This Section represents a section entry
// in XCOFF section header table.
struct SectionEntry {
  char Name[XCOFF::NameSize];
  // The physical/virtual address of the section. For an object file
  // these values are equivalent.
  uint32_t Address;
  uint32_t Size;
  uint32_t FileOffsetToData;
  uint32_t FileOffsetToRelocations;
  uint32_t RelocationCount;
  int32_t Flags;

  int16_t Index;

  // XCOFF has special section numbers for symbols:
  // -2 Specifies N_DEBUG, a special symbolic debugging symbol.
  // -1 Specifies N_ABS, an absolute symbol. The symbol has a value but is not
  // relocatable.
  //  0 Specifies N_UNDEF, an undefined external symbol.
  // Therefore, we choose -3 (N_DEBUG - 1) to represent a section index that
  // hasn't been initialized.
  static constexpr int16_t UninitializedIndex =
      XCOFF::ReservedSectionNum::N_DEBUG - 1;

  SectionEntry(StringRef N, int32_t Flags)
      : Name(), Address(0), Size(0), FileOffsetToData(0),
        FileOffsetToRelocations(0), RelocationCount(0), Flags(Flags),
        Index(UninitializedIndex) {
    assert(N.size() <= XCOFF::NameSize && "section name too long");
    memcpy(Name, N.data(), N.size());
  }

  virtual void reset() {
    Address = 0;
    Size = 0;
    FileOffsetToData = 0;
    FileOffsetToRelocations = 0;
    RelocationCount = 0;
    Index = UninitializedIndex;
  }

  virtual ~SectionEntry() = default;
};

// Represents the data related to a section excluding the csects that make up
// the raw data of the section. The csects are stored separately as not all
// sections contain csects, and some sections contain csects which are better
// stored separately, e.g. the .data section containing read-write, descriptor,
// TOCBase and TOC-entry csects.
struct CsectSectionEntry : public SectionEntry {
  // Virtual sections do not need storage allocated in the object file.
  const bool IsVirtual;

  // This is a section containing csect groups.
  CsectGroups Groups;

  CsectSectionEntry(StringRef N, XCOFF::SectionTypeFlags Flags, bool IsVirtual,
                    CsectGroups Groups)
      : SectionEntry(N, Flags), IsVirtual(IsVirtual), Groups(Groups) {
    assert(N.size() <= XCOFF::NameSize && "section name too long");
    memcpy(Name, N.data(), N.size());
  }

  void reset() override {
    SectionEntry::reset();
    // Clear any csects we have stored.
    for (auto *Group : Groups)
      Group->clear();
  }

  virtual ~CsectSectionEntry() = default;
};

struct DwarfSectionEntry : public SectionEntry {
  // For DWARF section entry.
  std::unique_ptr<XCOFFSection> DwarfSect;

  DwarfSectionEntry(StringRef N, int32_t Flags,
                    std::unique_ptr<XCOFFSection> Sect)
      : SectionEntry(N, Flags | XCOFF::STYP_DWARF), DwarfSect(std::move(Sect)) {
    assert(DwarfSect->MCSec->isDwarfSect() &&
           "This should be a DWARF section!");
    assert(N.size() <= XCOFF::NameSize && "section name too long");
    memcpy(Name, N.data(), N.size());
  }

  DwarfSectionEntry(DwarfSectionEntry &&s) = default;

  virtual ~DwarfSectionEntry() = default;
};

class XCOFFObjectWriter : public MCObjectWriter {

  uint32_t SymbolTableEntryCount = 0;
  uint32_t SymbolTableOffset = 0;
  uint16_t SectionCount = 0;
  uint32_t RelocationEntryOffset = 0;

  support::endian::Writer W;
  std::unique_ptr<MCXCOFFObjectTargetWriter> TargetObjectWriter;
  StringTableBuilder Strings;

  // Maps the MCSection representation to its corresponding XCOFFSection
  // wrapper. Needed for finding the XCOFFSection to insert an MCSymbol into
  // from its containing MCSectionXCOFF.
  DenseMap<const MCSectionXCOFF *, XCOFFSection *> SectionMap;

  // Maps the MCSymbol representation to its corrresponding symbol table index.
  // Needed for relocation.
  DenseMap<const MCSymbol *, uint32_t> SymbolIndexMap;

  // CsectGroups. These store the csects which make up different parts of
  // the sections. Should have one for each set of csects that get mapped into
  // the same section and get handled in a 'similar' way.
  CsectGroup UndefinedCsects;
  CsectGroup ProgramCodeCsects;
  CsectGroup ReadOnlyCsects;
  CsectGroup DataCsects;
  CsectGroup FuncDSCsects;
  CsectGroup TOCCsects;
  CsectGroup BSSCsects;
  CsectGroup TDataCsects;
  CsectGroup TBSSCsects;

  // The Predefined sections.
  CsectSectionEntry Text;
  CsectSectionEntry Data;
  CsectSectionEntry BSS;
  CsectSectionEntry TData;
  CsectSectionEntry TBSS;

  // All the XCOFF sections, in the order they will appear in the section header
  // table.
  std::array<CsectSectionEntry *const, 5> Sections{
      {&Text, &Data, &BSS, &TData, &TBSS}};

  std::vector<DwarfSectionEntry> DwarfSections;

  CsectGroup &getCsectGroup(const MCSectionXCOFF *MCSec);

  virtual void reset() override;

  void executePostLayoutBinding(MCAssembler &, const MCAsmLayout &) override;

  void recordRelocation(MCAssembler &, const MCAsmLayout &, const MCFragment *,
                        const MCFixup &, MCValue, uint64_t &) override;

  uint64_t writeObject(MCAssembler &, const MCAsmLayout &) override;

  static bool nameShouldBeInStringTable(const StringRef &);
  void writeSymbolName(const StringRef &);
  void writeSymbolTableEntryForCsectMemberLabel(const Symbol &,
                                                const XCOFFSection &, int16_t,
                                                uint64_t);
  void writeSymbolTableEntryForControlSection(const XCOFFSection &, int16_t,
                                              XCOFF::StorageClass);
  void writeSymbolTableEntryForDwarfSection(const XCOFFSection &, int16_t);
  void writeFileHeader();
  void writeSectionHeaderTable();
  void writeSections(const MCAssembler &Asm, const MCAsmLayout &Layout);
  void writeSectionForControlSectionEntry(const MCAssembler &Asm,
                                          const MCAsmLayout &Layout,
                                          const CsectSectionEntry &CsectEntry,
                                          uint32_t &CurrentAddressLocation);
  void writeSectionForDwarfSectionEntry(const MCAssembler &Asm,
                                        const MCAsmLayout &Layout,
                                        const DwarfSectionEntry &DwarfEntry,
                                        uint32_t &CurrentAddressLocation);
  void writeSymbolTable(const MCAsmLayout &Layout);
  void writeRelocations();
  void writeRelocation(XCOFFRelocation Reloc, const XCOFFSection &Section);

  // Called after all the csects and symbols have been processed by
  // `executePostLayoutBinding`, this function handles building up the majority
  // of the structures in the object file representation. Namely:
  // *) Calculates physical/virtual addresses, raw-pointer offsets, and section
  //    sizes.
  // *) Assigns symbol table indices.
  // *) Builds up the section header table by adding any non-empty sections to
  //    `Sections`.
  void assignAddressesAndIndices(const MCAsmLayout &);
  void finalizeSectionInfo();

  bool
  needsAuxiliaryHeader() const { /* TODO aux header support not implemented. */
    return false;
  }

  // Returns the size of the auxiliary header to be written to the object file.
  size_t auxiliaryHeaderSize() const {
    assert(!needsAuxiliaryHeader() &&
           "Auxiliary header support not implemented.");
    return 0;
  }

public:
  XCOFFObjectWriter(std::unique_ptr<MCXCOFFObjectTargetWriter> MOTW,
                    raw_pwrite_stream &OS);
};

XCOFFObjectWriter::XCOFFObjectWriter(
    std::unique_ptr<MCXCOFFObjectTargetWriter> MOTW, raw_pwrite_stream &OS)
    : W(OS, support::big), TargetObjectWriter(std::move(MOTW)),
      Strings(StringTableBuilder::XCOFF),
      Text(".text", XCOFF::STYP_TEXT, /* IsVirtual */ false,
           CsectGroups{&ProgramCodeCsects, &ReadOnlyCsects}),
      Data(".data", XCOFF::STYP_DATA, /* IsVirtual */ false,
           CsectGroups{&DataCsects, &FuncDSCsects, &TOCCsects}),
      BSS(".bss", XCOFF::STYP_BSS, /* IsVirtual */ true,
          CsectGroups{&BSSCsects}),
      TData(".tdata", XCOFF::STYP_TDATA, /* IsVirtual */ false,
            CsectGroups{&TDataCsects}),
      TBSS(".tbss", XCOFF::STYP_TBSS, /* IsVirtual */ true,
           CsectGroups{&TBSSCsects}) {}

void XCOFFObjectWriter::reset() {
  // Clear the mappings we created.
  SymbolIndexMap.clear();
  SectionMap.clear();

  UndefinedCsects.clear();
  // Reset any sections we have written to, and empty the section header table.
  for (auto *Sec : Sections)
    Sec->reset();
  for (auto &DwarfSec : DwarfSections)
    DwarfSec.reset();

  // Reset states in XCOFFObjectWriter.
  SymbolTableEntryCount = 0;
  SymbolTableOffset = 0;
  SectionCount = 0;
  RelocationEntryOffset = 0;
  Strings.clear();

  MCObjectWriter::reset();
}

CsectGroup &XCOFFObjectWriter::getCsectGroup(const MCSectionXCOFF *MCSec) {
  switch (MCSec->getMappingClass()) {
  case XCOFF::XMC_PR:
    assert(XCOFF::XTY_SD == MCSec->getCSectType() &&
           "Only an initialized csect can contain program code.");
    return ProgramCodeCsects;
  case XCOFF::XMC_RO:
    assert(XCOFF::XTY_SD == MCSec->getCSectType() &&
           "Only an initialized csect can contain read only data.");
    return ReadOnlyCsects;
  case XCOFF::XMC_RW:
    if (XCOFF::XTY_CM == MCSec->getCSectType())
      return BSSCsects;

    if (XCOFF::XTY_SD == MCSec->getCSectType())
      return DataCsects;

    report_fatal_error("Unhandled mapping of read-write csect to section.");
  case XCOFF::XMC_DS:
    return FuncDSCsects;
  case XCOFF::XMC_BS:
    assert(XCOFF::XTY_CM == MCSec->getCSectType() &&
           "Mapping invalid csect. CSECT with bss storage class must be "
           "common type.");
    return BSSCsects;
  case XCOFF::XMC_TL:
    assert(XCOFF::XTY_SD == MCSec->getCSectType() &&
           "Mapping invalid csect. CSECT with tdata storage class must be "
           "an initialized csect.");
    return TDataCsects;
  case XCOFF::XMC_UL:
    assert(XCOFF::XTY_CM == MCSec->getCSectType() &&
           "Mapping invalid csect. CSECT with tbss storage class must be "
           "an uninitialized csect.");
    return TBSSCsects;
  case XCOFF::XMC_TC0:
    assert(XCOFF::XTY_SD == MCSec->getCSectType() &&
           "Only an initialized csect can contain TOC-base.");
    assert(TOCCsects.empty() &&
           "We should have only one TOC-base, and it should be the first csect "
           "in this CsectGroup.");
    return TOCCsects;
  case XCOFF::XMC_TC:
  case XCOFF::XMC_TE:
    assert(XCOFF::XTY_SD == MCSec->getCSectType() &&
           "Only an initialized csect can contain TC entry.");
    assert(!TOCCsects.empty() &&
           "We should at least have a TOC-base in this CsectGroup.");
    return TOCCsects;
  case XCOFF::XMC_TD:
    report_fatal_error("toc-data not yet supported when writing object files.");
  default:
    report_fatal_error("Unhandled mapping of csect to section.");
  }
}

static MCSectionXCOFF *getContainingCsect(const MCSymbolXCOFF *XSym) {
  if (XSym->isDefined())
    return cast<MCSectionXCOFF>(XSym->getFragment()->getParent());
  return XSym->getRepresentedCsect();
}

void XCOFFObjectWriter::executePostLayoutBinding(MCAssembler &Asm,
                                                 const MCAsmLayout &Layout) {
  if (TargetObjectWriter->is64Bit())
    report_fatal_error("64-bit XCOFF object files are not supported yet.");

  for (const auto &S : Asm) {
    const auto *MCSec = cast<const MCSectionXCOFF>(&S);
    assert(SectionMap.find(MCSec) == SectionMap.end() &&
           "Cannot add a section twice.");

    // If the name does not fit in the storage provided in the symbol table
    // entry, add it to the string table.
    if (nameShouldBeInStringTable(MCSec->getSymbolTableName()))
      Strings.add(MCSec->getSymbolTableName());
    if (MCSec->isCsect()) {
      // A new control section. Its CsectSectionEntry should already be staticly
      // generated as Text/Data/BSS/TDATA/TBSS. Add this section to the group of
      // the CsectSectionEntry.
      assert(XCOFF::XTY_ER != MCSec->getCSectType() &&
             "An undefined csect should not get registered.");
      CsectGroup &Group = getCsectGroup(MCSec);
      Group.emplace_back(MCSec);
      SectionMap[MCSec] = &Group.back();
    } else if (MCSec->isDwarfSect()) {
      // A new DwarfSectionEntry.
      std::unique_ptr<XCOFFSection> DwarfSec =
          std::make_unique<XCOFFSection>(MCSec);
      SectionMap[MCSec] = DwarfSec.get();

      DwarfSectionEntry SecEntry(MCSec->getName(),
                                 MCSec->getDwarfSubtypeFlags().getValue(),
                                 std::move(DwarfSec));
      DwarfSections.push_back(std::move(SecEntry));
    } else
      llvm_unreachable("unsupport section type!");
  }

  for (const MCSymbol &S : Asm.symbols()) {
    // Nothing to do for temporary symbols.
    if (S.isTemporary())
      continue;

    const MCSymbolXCOFF *XSym = cast<MCSymbolXCOFF>(&S);
    const MCSectionXCOFF *ContainingCsect = getContainingCsect(XSym);

    if (ContainingCsect->getCSectType() == XCOFF::XTY_ER) {
      // Handle undefined symbol.
      UndefinedCsects.emplace_back(ContainingCsect);
      SectionMap[ContainingCsect] = &UndefinedCsects.back();
      if (nameShouldBeInStringTable(ContainingCsect->getSymbolTableName()))
        Strings.add(ContainingCsect->getSymbolTableName());
      continue;
    }

    // If the symbol is the csect itself, we don't need to put the symbol
    // into csect's Syms.
    if (XSym == ContainingCsect->getQualNameSymbol())
      continue;

    // Only put a label into the symbol table when it is an external label.
    if (!XSym->isExternal())
      continue;

    assert(SectionMap.find(ContainingCsect) != SectionMap.end() &&
           "Expected containing csect to exist in map");
    XCOFFSection *Csect = SectionMap[ContainingCsect];
    // Lookup the containing csect and add the symbol to it.
    assert(Csect->MCSec->isCsect() && "only csect is supported now!");
    Csect->Syms.emplace_back(XSym);

    // If the name does not fit in the storage provided in the symbol table
    // entry, add it to the string table.
    if (nameShouldBeInStringTable(XSym->getSymbolTableName()))
      Strings.add(XSym->getSymbolTableName());
  }

  Strings.finalize();
  assignAddressesAndIndices(Layout);
}

void XCOFFObjectWriter::recordRelocation(MCAssembler &Asm,
                                         const MCAsmLayout &Layout,
                                         const MCFragment *Fragment,
                                         const MCFixup &Fixup, MCValue Target,
                                         uint64_t &FixedValue) {
  auto getIndex = [this](const MCSymbol *Sym,
                         const MCSectionXCOFF *ContainingCsect) {
    // If we could not find the symbol directly in SymbolIndexMap, this symbol
    // could either be a temporary symbol or an undefined symbol. In this case,
    // we would need to have the relocation reference its csect instead.
    return SymbolIndexMap.find(Sym) != SymbolIndexMap.end()
               ? SymbolIndexMap[Sym]
               : SymbolIndexMap[ContainingCsect->getQualNameSymbol()];
  };

  auto getVirtualAddress =
      [this, &Layout](const MCSymbol *Sym,
                      const MCSectionXCOFF *ContainingSect) -> uint64_t {
    // A DWARF section.
    if (ContainingSect->isDwarfSect())
      return Layout.getSymbolOffset(*Sym);

    // A csect.
    if (!Sym->isDefined())
      return SectionMap[ContainingSect]->Address;

    // A label.
    assert(Sym->isDefined() && "not a valid object that has address!");
    return SectionMap[ContainingSect]->Address + Layout.getSymbolOffset(*Sym);
  };

  const MCSymbol *const SymA = &Target.getSymA()->getSymbol();

  MCAsmBackend &Backend = Asm.getBackend();
  bool IsPCRel = Backend.getFixupKindInfo(Fixup.getKind()).Flags &
                 MCFixupKindInfo::FKF_IsPCRel;

  uint8_t Type;
  uint8_t SignAndSize;
  std::tie(Type, SignAndSize) =
      TargetObjectWriter->getRelocTypeAndSignSize(Target, Fixup, IsPCRel);

  const MCSectionXCOFF *SymASec = getContainingCsect(cast<MCSymbolXCOFF>(SymA));

  if (SymASec->isCsect() && SymASec->getMappingClass() == XCOFF::XMC_TD)
    report_fatal_error("toc-data not yet supported when writing object files.");

  assert(SectionMap.find(SymASec) != SectionMap.end() &&
         "Expected containing csect to exist in map.");

  const uint32_t Index = getIndex(SymA, SymASec);
  if (Type == XCOFF::RelocationType::R_POS ||
      Type == XCOFF::RelocationType::R_TLS)
    // The FixedValue should be symbol's virtual address in this object file
    // plus any constant value that we might get.
    FixedValue = getVirtualAddress(SymA, SymASec) + Target.getConstant();
  else if (Type == XCOFF::RelocationType::R_TLSM)
    // The FixedValue should always be zero since the region handle is only
    // known at load time.
    FixedValue = 0;
  else if (Type == XCOFF::RelocationType::R_TOC ||
           Type == XCOFF::RelocationType::R_TOCL) {
    // The FixedValue should be the TOC entry offset from the TOC-base plus any
    // constant offset value.
    const int64_t TOCEntryOffset = SectionMap[SymASec]->Address -
                                   TOCCsects.front().Address +
                                   Target.getConstant();
    if (Type == XCOFF::RelocationType::R_TOC && !isInt<16>(TOCEntryOffset))
      report_fatal_error("TOCEntryOffset overflows in small code model mode");

    FixedValue = TOCEntryOffset;
  }

  assert(
      (TargetObjectWriter->is64Bit() ||
       Fixup.getOffset() <= UINT32_MAX - Layout.getFragmentOffset(Fragment)) &&
      "Fragment offset + fixup offset is overflowed in 32-bit mode.");
  uint32_t FixupOffsetInCsect =
      Layout.getFragmentOffset(Fragment) + Fixup.getOffset();

  XCOFFRelocation Reloc = {Index, FixupOffsetInCsect, SignAndSize, Type};
  MCSectionXCOFF *RelocationSec = cast<MCSectionXCOFF>(Fragment->getParent());
  assert(SectionMap.find(RelocationSec) != SectionMap.end() &&
         "Expected containing csect to exist in map.");
  SectionMap[RelocationSec]->Relocations.push_back(Reloc);

  if (!Target.getSymB())
    return;

  const MCSymbol *const SymB = &Target.getSymB()->getSymbol();
  if (SymA == SymB)
    report_fatal_error("relocation for opposite term is not yet supported");

  const MCSectionXCOFF *SymBSec = getContainingCsect(cast<MCSymbolXCOFF>(SymB));
  assert(SectionMap.find(SymBSec) != SectionMap.end() &&
         "Expected containing csect to exist in map.");
  if (SymASec == SymBSec)
    report_fatal_error(
        "relocation for paired relocatable term is not yet supported");

  assert(Type == XCOFF::RelocationType::R_POS &&
         "SymA must be R_POS here if it's not opposite term or paired "
         "relocatable term.");
  const uint32_t IndexB = getIndex(SymB, SymBSec);
  // SymB must be R_NEG here, given the general form of Target(MCValue) is
  // "SymbolA - SymbolB + imm64".
  const uint8_t TypeB = XCOFF::RelocationType::R_NEG;
  XCOFFRelocation RelocB = {IndexB, FixupOffsetInCsect, SignAndSize, TypeB};
  SectionMap[RelocationSec]->Relocations.push_back(RelocB);
  // We already folded "SymbolA + imm64" above when Type is R_POS for SymbolA,
  // now we just need to fold "- SymbolB" here.
  FixedValue -= getVirtualAddress(SymB, SymBSec);
}

void XCOFFObjectWriter::writeSections(const MCAssembler &Asm,
                                      const MCAsmLayout &Layout) {
  uint32_t CurrentAddressLocation = 0;
  for (const auto *Section : Sections)
    writeSectionForControlSectionEntry(Asm, Layout, *Section,
                                       CurrentAddressLocation);
  for (const auto &DwarfSection : DwarfSections)
    writeSectionForDwarfSectionEntry(Asm, Layout, DwarfSection,
                                     CurrentAddressLocation);
}

uint64_t XCOFFObjectWriter::writeObject(MCAssembler &Asm,
                                        const MCAsmLayout &Layout) {
  // We always emit a timestamp of 0 for reproducibility, so ensure incremental
  // linking is not enabled, in case, like with Windows COFF, such a timestamp
  // is incompatible with incremental linking of XCOFF.
  if (Asm.isIncrementalLinkerCompatible())
    report_fatal_error("Incremental linking not supported for XCOFF.");

  if (TargetObjectWriter->is64Bit())
    report_fatal_error("64-bit XCOFF object files are not supported yet.");

  finalizeSectionInfo();
  uint64_t StartOffset = W.OS.tell();

  writeFileHeader();
  writeSectionHeaderTable();
  writeSections(Asm, Layout);
  writeRelocations();

  writeSymbolTable(Layout);
  // Write the string table.
  Strings.write(W.OS);

  return W.OS.tell() - StartOffset;
}

bool XCOFFObjectWriter::nameShouldBeInStringTable(const StringRef &SymbolName) {
  return SymbolName.size() > XCOFF::NameSize;
}

void XCOFFObjectWriter::writeSymbolName(const StringRef &SymbolName) {
  if (nameShouldBeInStringTable(SymbolName)) {
    W.write<int32_t>(0);
    W.write<uint32_t>(Strings.getOffset(SymbolName));
  } else {
    char Name[XCOFF::NameSize+1];
    std::strncpy(Name, SymbolName.data(), XCOFF::NameSize);
    ArrayRef<char> NameRef(Name, XCOFF::NameSize);
    W.write(NameRef);
  }
}

void XCOFFObjectWriter::writeSymbolTableEntryForCsectMemberLabel(
    const Symbol &SymbolRef, const XCOFFSection &CSectionRef,
    int16_t SectionIndex, uint64_t SymbolOffset) {
  // Name or Zeros and string table offset
  writeSymbolName(SymbolRef.getSymbolTableName());
  assert(SymbolOffset <= UINT32_MAX - CSectionRef.Address &&
         "Symbol address overflows.");
  W.write<uint32_t>(CSectionRef.Address + SymbolOffset);
  W.write<int16_t>(SectionIndex);
  // Basic/Derived type. See the description of the n_type field for symbol
  // table entries for a detailed description. Since we don't yet support
  // visibility, and all other bits are either optionally set or reserved, this
  // is always zero.
  // TODO FIXME How to assert a symbol's visibilty is default?
  // TODO Set the function indicator (bit 10, 0x0020) for functions
  // when debugging is enabled.
  W.write<uint16_t>(0);
  W.write<uint8_t>(SymbolRef.getStorageClass());
  // Always 1 aux entry for now.
  W.write<uint8_t>(1);

  // Now output the auxiliary entry.
  W.write<uint32_t>(CSectionRef.SymbolTableIndex);
  // Parameter typecheck hash. Not supported.
  W.write<uint32_t>(0);
  // Typecheck section number. Not supported.
  W.write<uint16_t>(0);
  // Symbol type: Label
  W.write<uint8_t>(XCOFF::XTY_LD);
  // Storage mapping class.
  W.write<uint8_t>(CSectionRef.MCSec->getMappingClass());
  // Reserved (x_stab).
  W.write<uint32_t>(0);
  // Reserved (x_snstab).
  W.write<uint16_t>(0);
}

void XCOFFObjectWriter::writeSymbolTableEntryForDwarfSection(
    const XCOFFSection &DwarfSectionRef, int16_t SectionIndex) {
  assert(DwarfSectionRef.MCSec->isDwarfSect() && "Not a DWARF section!");

  // n_name, n_zeros, n_offset
  writeSymbolName(DwarfSectionRef.getSymbolTableName());
  // n_value
  W.write<uint32_t>(0);
  // n_scnum
  W.write<int16_t>(SectionIndex);
  // n_type
  W.write<uint16_t>(0);
  // n_sclass
  W.write<uint8_t>(XCOFF::C_DWARF);
  // Always 1 aux entry for now.
  W.write<uint8_t>(1);

  // Now output the auxiliary entry.
  // x_scnlen
  W.write<uint32_t>(DwarfSectionRef.Size);
  // Reserved
  W.write<uint32_t>(0);
  // x_nreloc. Set to 0 for now.
  W.write<uint32_t>(0);
  // Reserved
  W.write<uint32_t>(0);
  // Reserved
  W.write<uint16_t>(0);
}

void XCOFFObjectWriter::writeSymbolTableEntryForControlSection(
    const XCOFFSection &CSectionRef, int16_t SectionIndex,
    XCOFF::StorageClass StorageClass) {
  // n_name, n_zeros, n_offset
  writeSymbolName(CSectionRef.getSymbolTableName());
  // n_value
  W.write<uint32_t>(CSectionRef.Address);
  // n_scnum
  W.write<int16_t>(SectionIndex);
  // Basic/Derived type. See the description of the n_type field for symbol
  // table entries for a detailed description. Since we don't yet support
  // visibility, and all other bits are either optionally set or reserved, this
  // is always zero.
  // TODO FIXME How to assert a symbol's visibilty is default?
  // TODO Set the function indicator (bit 10, 0x0020) for functions
  // when debugging is enabled.
  W.write<uint16_t>(0);
  // n_sclass
  W.write<uint8_t>(StorageClass);
  // Always 1 aux entry for now.
  W.write<uint8_t>(1);

  // Now output the auxiliary entry.
  W.write<uint32_t>(CSectionRef.Size);
  // Parameter typecheck hash. Not supported.
  W.write<uint32_t>(0);
  // Typecheck section number. Not supported.
  W.write<uint16_t>(0);
  // Symbol type.
  W.write<uint8_t>(getEncodedType(CSectionRef.MCSec));
  // Storage mapping class.
  W.write<uint8_t>(CSectionRef.MCSec->getMappingClass());
  // Reserved (x_stab).
  W.write<uint32_t>(0);
  // Reserved (x_snstab).
  W.write<uint16_t>(0);
}

void XCOFFObjectWriter::writeFileHeader() {
  // Magic.
  W.write<uint16_t>(0x01df);
  // Number of sections.
  W.write<uint16_t>(SectionCount);
  // Timestamp field. For reproducible output we write a 0, which represents no
  // timestamp.
  W.write<int32_t>(0);
  // Byte Offset to the start of the symbol table.
  W.write<uint32_t>(SymbolTableOffset);
  // Number of entries in the symbol table.
  W.write<int32_t>(SymbolTableEntryCount);
  // Size of the optional header.
  W.write<uint16_t>(0);
  // Flags.
  W.write<uint16_t>(0);
}

void XCOFFObjectWriter::writeSectionHeaderTable() {
  auto writeSectionHeader = [&](const SectionEntry *Sec, bool IsDwarf) {
    // Nothing to write for this Section.
    if (Sec->Index == SectionEntry::UninitializedIndex)
      return false;

    // Write Name.
    ArrayRef<char> NameRef(Sec->Name, XCOFF::NameSize);
    W.write(NameRef);

    // Write the Physical Address and Virtual Address. In an object file these
    // are the same.
    // We use 0 for DWARF sections' Physical and Virtual Addresses.
    if (!IsDwarf) {
      W.write<uint32_t>(Sec->Address);
      W.write<uint32_t>(Sec->Address);
    } else {
      W.write<uint32_t>(0);
      W.write<uint32_t>(0);
    }

    W.write<uint32_t>(Sec->Size);
    W.write<uint32_t>(Sec->FileOffsetToData);
    W.write<uint32_t>(Sec->FileOffsetToRelocations);

    // Line number pointer. Not supported yet.
    W.write<uint32_t>(0);

    W.write<uint16_t>(Sec->RelocationCount);

    // Line number counts. Not supported yet.
    W.write<uint16_t>(0);

    W.write<int32_t>(Sec->Flags);

    return true;
  };

  for (const auto *CsectSec : Sections)
    writeSectionHeader(CsectSec, /* IsDwarf */ false);
  for (const auto &DwarfSec : DwarfSections)
    writeSectionHeader(&DwarfSec, /* IsDwarf */ true);
}

void XCOFFObjectWriter::writeRelocation(XCOFFRelocation Reloc,
                                        const XCOFFSection &Section) {
  if (Section.MCSec->isCsect())
    W.write<uint32_t>(Section.Address + Reloc.FixupOffsetInCsect);
  else {
    // DWARF sections' address is set to 0.
    assert(Section.MCSec->isDwarfSect() && "unsupport section type!");
    W.write<uint32_t>(Reloc.FixupOffsetInCsect);
  }
  W.write<uint32_t>(Reloc.SymbolTableIndex);
  W.write<uint8_t>(Reloc.SignAndSize);
  W.write<uint8_t>(Reloc.Type);
}

void XCOFFObjectWriter::writeRelocations() {
  for (const auto *Section : Sections) {
    if (Section->Index == SectionEntry::UninitializedIndex)
      // Nothing to write for this Section.
      continue;

    for (const auto *Group : Section->Groups) {
      if (Group->empty())
        continue;

      for (const auto &Csect : *Group) {
        for (const auto Reloc : Csect.Relocations)
          writeRelocation(Reloc, Csect);
      }
    }
  }

  for (const auto &DwarfSection : DwarfSections)
    for (const auto &Reloc : DwarfSection.DwarfSect->Relocations)
      writeRelocation(Reloc, *DwarfSection.DwarfSect);
}

void XCOFFObjectWriter::writeSymbolTable(const MCAsmLayout &Layout) {
  // Write symbol 0 as C_FILE.
  // FIXME: support 64-bit C_FILE symbol.
  //
  // n_name. The n_name of a C_FILE symbol is the source filename when no
  // auxiliary entries are present. The source filename is alternatively
  // provided by an auxiliary entry, in which case the n_name of the C_FILE
  // symbol is `.file`.
  // FIXME: add the real source filename.
  writeSymbolName(".file");
  // n_value. The n_value of a C_FILE symbol is its symbol table index.
  W.write<uint32_t>(0);
  // n_scnum. N_DEBUG is a reserved section number for indicating a special
  // symbolic debugging symbol.
  W.write<int16_t>(XCOFF::ReservedSectionNum::N_DEBUG);
  // n_type. The n_type field of a C_FILE symbol encodes the source language and
  // CPU version info; zero indicates no info.
  W.write<uint16_t>(0);
  // n_sclass. The C_FILE symbol provides source file-name information,
  // source-language ID and CPU-version ID information and some other optional
  // infos.
  W.write<uint8_t>(XCOFF::C_FILE);
  // n_numaux. No aux entry for now.
  W.write<uint8_t>(0);

  for (const auto &Csect : UndefinedCsects) {
    writeSymbolTableEntryForControlSection(Csect,
                                           XCOFF::ReservedSectionNum::N_UNDEF,
                                           Csect.MCSec->getStorageClass());
  }

  for (const auto *Section : Sections) {
    if (Section->Index == SectionEntry::UninitializedIndex)
      // Nothing to write for this Section.
      continue;

    for (const auto *Group : Section->Groups) {
      if (Group->empty())
        continue;

      const int16_t SectionIndex = Section->Index;
      for (const auto &Csect : *Group) {
        // Write out the control section first and then each symbol in it.
        writeSymbolTableEntryForControlSection(Csect, SectionIndex,
                                               Csect.MCSec->getStorageClass());

        for (const auto &Sym : Csect.Syms)
          writeSymbolTableEntryForCsectMemberLabel(
              Sym, Csect, SectionIndex, Layout.getSymbolOffset(*(Sym.MCSym)));
      }
    }
  }

  for (const auto &DwarfSection : DwarfSections)
    writeSymbolTableEntryForDwarfSection(*DwarfSection.DwarfSect,
                                         DwarfSection.Index);
}

void XCOFFObjectWriter::finalizeSectionInfo() {
  for (auto *Section : Sections) {
    if (Section->Index == SectionEntry::UninitializedIndex)
      // Nothing to record for this Section.
      continue;

    for (const auto *Group : Section->Groups) {
      if (Group->empty())
        continue;

      for (auto &Csect : *Group) {
        const size_t CsectRelocCount = Csect.Relocations.size();
        if (CsectRelocCount >= XCOFF::RelocOverflow ||
            Section->RelocationCount >= XCOFF::RelocOverflow - CsectRelocCount)
          report_fatal_error(
              "relocation entries overflowed; overflow section is "
              "not implemented yet");

        Section->RelocationCount += CsectRelocCount;
      }
    }
  }

  for (auto &DwarfSection : DwarfSections)
    DwarfSection.RelocationCount = DwarfSection.DwarfSect->Relocations.size();

  // Calculate the file offset to the relocation entries.
  uint64_t RawPointer = RelocationEntryOffset;
  auto calcOffsetToRelocations = [&](SectionEntry *Sec, bool IsDwarf) {
    if (!IsDwarf && Sec->Index == SectionEntry::UninitializedIndex)
      return false;

    if (!Sec->RelocationCount)
      return false;

    Sec->FileOffsetToRelocations = RawPointer;
    const uint32_t RelocationSizeInSec =
        Sec->RelocationCount * XCOFF::RelocationSerializationSize32;
    RawPointer += RelocationSizeInSec;
    if (RawPointer > UINT32_MAX)
      report_fatal_error("Relocation data overflowed this object file.");

    return true;
  };

  for (auto *Sec : Sections)
    calcOffsetToRelocations(Sec, /* IsDwarf */ false);

  for (auto &DwarfSec : DwarfSections)
    calcOffsetToRelocations(&DwarfSec, /* IsDwarf */ true);

  // TODO Error check that the number of symbol table entries fits in 32-bits
  // signed ...
  if (SymbolTableEntryCount)
    SymbolTableOffset = RawPointer;
}

void XCOFFObjectWriter::assignAddressesAndIndices(const MCAsmLayout &Layout) {
  // The first symbol table entry (at index 0) is for the file name.
  uint32_t SymbolTableIndex = 1;

  // Calculate indices for undefined symbols.
  for (auto &Csect : UndefinedCsects) {
    Csect.Size = 0;
    Csect.Address = 0;
    Csect.SymbolTableIndex = SymbolTableIndex;
    SymbolIndexMap[Csect.MCSec->getQualNameSymbol()] = Csect.SymbolTableIndex;
    // 1 main and 1 auxiliary symbol table entry for each contained symbol.
    SymbolTableIndex += 2;
  }

  // The address corrresponds to the address of sections and symbols in the
  // object file. We place the shared address 0 immediately after the
  // section header table.
  uint32_t Address = 0;
  // Section indices are 1-based in XCOFF.
  int32_t SectionIndex = 1;
  bool HasTDataSection = false;

  for (auto *Section : Sections) {
    const bool IsEmpty =
        llvm::all_of(Section->Groups,
                     [](const CsectGroup *Group) { return Group->empty(); });
    if (IsEmpty)
      continue;

    if (SectionIndex > MaxSectionIndex)
      report_fatal_error("Section index overflow!");
    Section->Index = SectionIndex++;
    SectionCount++;

    bool SectionAddressSet = false;
    // Reset the starting address to 0 for TData section.
    if (Section->Flags == XCOFF::STYP_TDATA) {
      Address = 0;
      HasTDataSection = true;
    }
    // Reset the starting address to 0 for TBSS section if the object file does
    // not contain TData Section.
    if ((Section->Flags == XCOFF::STYP_TBSS) && !HasTDataSection)
      Address = 0;

    for (auto *Group : Section->Groups) {
      if (Group->empty())
        continue;

      for (auto &Csect : *Group) {
        const MCSectionXCOFF *MCSec = Csect.MCSec;
        Csect.Address = alignTo(Address, MCSec->getAlignment());
        Csect.Size = Layout.getSectionAddressSize(MCSec);
        Address = Csect.Address + Csect.Size;
        Csect.SymbolTableIndex = SymbolTableIndex;
        SymbolIndexMap[MCSec->getQualNameSymbol()] = Csect.SymbolTableIndex;
        // 1 main and 1 auxiliary symbol table entry for the csect.
        SymbolTableIndex += 2;

        for (auto &Sym : Csect.Syms) {
          Sym.SymbolTableIndex = SymbolTableIndex;
          SymbolIndexMap[Sym.MCSym] = Sym.SymbolTableIndex;
          // 1 main and 1 auxiliary symbol table entry for each contained
          // symbol.
          SymbolTableIndex += 2;
        }
      }

      if (!SectionAddressSet) {
        Section->Address = Group->front().Address;
        SectionAddressSet = true;
      }
    }

    // Make sure the address of the next section aligned to
    // DefaultSectionAlign.
    Address = alignTo(Address, DefaultSectionAlign);
    Section->Size = Address - Section->Address;
  }

  for (auto &DwarfSection : DwarfSections) {
    assert((SectionIndex <= MaxSectionIndex) && "Section index overflow!");

    XCOFFSection &DwarfSect = *DwarfSection.DwarfSect;
    const MCSectionXCOFF *MCSec = DwarfSect.MCSec;

    // Section index.
    DwarfSection.Index = SectionIndex++;
    SectionCount++;

    // Symbol index.
    DwarfSect.SymbolTableIndex = SymbolTableIndex;
    SymbolIndexMap[MCSec->getQualNameSymbol()] = DwarfSect.SymbolTableIndex;
    // 1 main and 1 auxiliary symbol table entry for the csect.
    SymbolTableIndex += 2;

    // Section address. Make it align to section alignment.
    // We use address 0 for DWARF sections' Physical and Virtual Addresses.
    // This address is used to tell where is the section in the final object.
    // See writeSectionForDwarfSectionEntry().
    DwarfSection.Address = DwarfSect.Address =
        alignTo(Address, MCSec->getAlignment());

    // Section size.
    // For DWARF section, we must use the real size which may be not aligned.
    DwarfSection.Size = DwarfSect.Size = Layout.getSectionAddressSize(MCSec);

    // Make the Address align to default alignment for follow section.
    Address = alignTo(DwarfSect.Address + DwarfSect.Size, DefaultSectionAlign);
  }

  SymbolTableEntryCount = SymbolTableIndex;

  // Calculate the RawPointer value for each section.
  uint64_t RawPointer = XCOFF::FileHeaderSize32 + auxiliaryHeaderSize() +
                        SectionCount * XCOFF::SectionHeaderSize32;
  for (auto *Sec : Sections) {
    if (Sec->Index == SectionEntry::UninitializedIndex || Sec->IsVirtual)
      continue;

    Sec->FileOffsetToData = RawPointer;
    RawPointer += Sec->Size;
    if (RawPointer > UINT32_MAX)
      report_fatal_error("Section raw data overflowed this object file.");
  }

  for (auto &DwarfSection : DwarfSections) {
    // Address of csect sections are always aligned to DefaultSectionAlign, but
    // address of DWARF section are aligned to Section alignment which may be
    // bigger than DefaultSectionAlign, need to execlude the padding bits.
    RawPointer =
          alignTo(RawPointer, DwarfSection.DwarfSect->MCSec->getAlignment());

    DwarfSection.FileOffsetToData = RawPointer;
    // Some section entries, like DWARF section size is not aligned, so
    // RawPointer may be not aligned.
    RawPointer += DwarfSection.Size;
    // Make sure RawPointer is aligned.
    RawPointer = alignTo(RawPointer, DefaultSectionAlign);

    assert(RawPointer <= UINT32_MAX &&
           "Section raw data overflowed this object file.");
  }

  RelocationEntryOffset = RawPointer;
}

void XCOFFObjectWriter::writeSectionForControlSectionEntry(
    const MCAssembler &Asm, const MCAsmLayout &Layout,
    const CsectSectionEntry &CsectEntry, uint32_t &CurrentAddressLocation) {
  // Nothing to write for this Section.
  if (CsectEntry.Index == SectionEntry::UninitializedIndex)
    return;

  // There could be a gap (without corresponding zero padding) between
  // sections.
  // There could be a gap (without corresponding zero padding) between
  // sections.
  assert(((CurrentAddressLocation <= CsectEntry.Address) ||
          (CsectEntry.Flags == XCOFF::STYP_TDATA) ||
          (CsectEntry.Flags == XCOFF::STYP_TBSS)) &&
         "CurrentAddressLocation should be less than or equal to section "
         "address if the section is not TData or TBSS.");

  CurrentAddressLocation = CsectEntry.Address;

  // For virtual sections, nothing to write. But need to increase
  // CurrentAddressLocation for later sections like DWARF section has a correct
  // writing location.
  if (CsectEntry.IsVirtual) {
    CurrentAddressLocation += CsectEntry.Size;
    return;
  }

  for (const auto &Group : CsectEntry.Groups) {
    for (const auto &Csect : *Group) {
      if (uint32_t PaddingSize = Csect.Address - CurrentAddressLocation)
        W.OS.write_zeros(PaddingSize);
      if (Csect.Size)
        Asm.writeSectionData(W.OS, Csect.MCSec, Layout);
      CurrentAddressLocation = Csect.Address + Csect.Size;
    }
  }

  // The size of the tail padding in a section is the end virtual address of
  // the current section minus the the end virtual address of the last csect
  // in that section.
  if (uint32_t PaddingSize =
          CsectEntry.Address + CsectEntry.Size - CurrentAddressLocation) {
    W.OS.write_zeros(PaddingSize);
    CurrentAddressLocation += PaddingSize;
  }
}

void XCOFFObjectWriter::writeSectionForDwarfSectionEntry(
    const MCAssembler &Asm, const MCAsmLayout &Layout,
    const DwarfSectionEntry &DwarfEntry, uint32_t &CurrentAddressLocation) {
  // There could be a gap (without corresponding zero padding) between
  // sections. For example DWARF section alignment is bigger than
  // DefaultSectionAlign.
  assert(CurrentAddressLocation <= DwarfEntry.Address &&
         "CurrentAddressLocation should be less than or equal to section "
         "address.");

  if (uint32_t PaddingSize = DwarfEntry.Address - CurrentAddressLocation)
    W.OS.write_zeros(PaddingSize);

  if (DwarfEntry.Size)
    Asm.writeSectionData(W.OS, DwarfEntry.DwarfSect->MCSec, Layout);

  CurrentAddressLocation = DwarfEntry.Address + DwarfEntry.Size;

  // DWARF section size is not aligned to DefaultSectionAlign.
  // Make sure CurrentAddressLocation is aligned to DefaultSectionAlign.
  uint32_t Mod = CurrentAddressLocation % DefaultSectionAlign;
  uint32_t TailPaddingSize = Mod ? DefaultSectionAlign - Mod : 0;
  if (TailPaddingSize)
    W.OS.write_zeros(TailPaddingSize);

  CurrentAddressLocation += TailPaddingSize;
}

// Takes the log base 2 of the alignment and shifts the result into the 5 most
// significant bits of a byte, then or's in the csect type into the least
// significant 3 bits.
uint8_t getEncodedType(const MCSectionXCOFF *Sec) {
  unsigned Align = Sec->getAlignment();
  assert(isPowerOf2_32(Align) && "Alignment must be a power of 2.");
  unsigned Log2Align = Log2_32(Align);
  // Result is a number in the range [0, 31] which fits in the 5 least
  // significant bits. Shift this value into the 5 most significant bits, and
  // bitwise-or in the csect type.
  uint8_t EncodedAlign = Log2Align << 3;
  return EncodedAlign | Sec->getCSectType();
}

} // end anonymous namespace

std::unique_ptr<MCObjectWriter>
llvm::createXCOFFObjectWriter(std::unique_ptr<MCXCOFFObjectTargetWriter> MOTW,
                              raw_pwrite_stream &OS) {
  return std::make_unique<XCOFFObjectWriter>(std::move(MOTW), OS);
}
