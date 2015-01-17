//===- lib/MC/MachObjectWriter.cpp - Mach-O File Writer -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCMachObjectWriter.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCMachOSymbolFlags.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MachO.h"
#include <vector>
using namespace llvm;

#define DEBUG_TYPE "mc"

void MachObjectWriter::reset() {
  Relocations.clear();
  IndirectSymBase.clear();
  StringTable.clear();
  LocalSymbolData.clear();
  ExternalSymbolData.clear();
  UndefinedSymbolData.clear();
  MCObjectWriter::reset();
}

bool MachObjectWriter::
doesSymbolRequireExternRelocation(const MCSymbolData *SD) {
  // Undefined symbols are always extern.
  if (SD->getSymbol().isUndefined())
    return true;

  // References to weak definitions require external relocation entries; the
  // definition may not always be the one in the same object file.
  if (SD->getFlags() & SF_WeakDefinition)
    return true;

  // Otherwise, we can use an internal relocation.
  return false;
}

bool MachObjectWriter::
MachSymbolData::operator<(const MachSymbolData &RHS) const {
  return SymbolData->getSymbol().getName() <
    RHS.SymbolData->getSymbol().getName();
}

bool MachObjectWriter::isFixupKindPCRel(const MCAssembler &Asm, unsigned Kind) {
  const MCFixupKindInfo &FKI = Asm.getBackend().getFixupKindInfo(
    (MCFixupKind) Kind);

  return FKI.Flags & MCFixupKindInfo::FKF_IsPCRel;
}

uint64_t MachObjectWriter::getFragmentAddress(const MCFragment *Fragment,
                                              const MCAsmLayout &Layout) const {
  return getSectionAddress(Fragment->getParent()) +
    Layout.getFragmentOffset(Fragment);
}

uint64_t MachObjectWriter::getSymbolAddress(const MCSymbolData* SD,
                                            const MCAsmLayout &Layout) const {
  const MCSymbol &S = SD->getSymbol();

  // If this is a variable, then recursively evaluate now.
  if (S.isVariable()) {
    if (const MCConstantExpr *C =
          dyn_cast<const MCConstantExpr>(S.getVariableValue()))
      return C->getValue();


    MCValue Target;
    if (!S.getVariableValue()->EvaluateAsRelocatable(Target, &Layout, nullptr))
      report_fatal_error("unable to evaluate offset for variable '" +
                         S.getName() + "'");

    // Verify that any used symbols are defined.
    if (Target.getSymA() && Target.getSymA()->getSymbol().isUndefined())
      report_fatal_error("unable to evaluate offset to undefined symbol '" +
                         Target.getSymA()->getSymbol().getName() + "'");
    if (Target.getSymB() && Target.getSymB()->getSymbol().isUndefined())
      report_fatal_error("unable to evaluate offset to undefined symbol '" +
                         Target.getSymB()->getSymbol().getName() + "'");

    uint64_t Address = Target.getConstant();
    if (Target.getSymA())
      Address += getSymbolAddress(&Layout.getAssembler().getSymbolData(
                                    Target.getSymA()->getSymbol()), Layout);
    if (Target.getSymB())
      Address += getSymbolAddress(&Layout.getAssembler().getSymbolData(
                                    Target.getSymB()->getSymbol()), Layout);
    return Address;
  }

  return getSectionAddress(SD->getFragment()->getParent()) +
    Layout.getSymbolOffset(SD);
}

uint64_t MachObjectWriter::getPaddingSize(const MCSectionData *SD,
                                          const MCAsmLayout &Layout) const {
  uint64_t EndAddr = getSectionAddress(SD) + Layout.getSectionAddressSize(SD);
  unsigned Next = SD->getLayoutOrder() + 1;
  if (Next >= Layout.getSectionOrder().size())
    return 0;

  const MCSectionData &NextSD = *Layout.getSectionOrder()[Next];
  if (NextSD.getSection().isVirtualSection())
    return 0;
  return OffsetToAlignment(EndAddr, NextSD.getAlignment());
}

void MachObjectWriter::WriteHeader(unsigned NumLoadCommands,
                                   unsigned LoadCommandsSize,
                                   bool SubsectionsViaSymbols) {
  uint32_t Flags = 0;

  if (SubsectionsViaSymbols)
    Flags |= MachO::MH_SUBSECTIONS_VIA_SYMBOLS;

  // struct mach_header (28 bytes) or
  // struct mach_header_64 (32 bytes)

  uint64_t Start = OS.tell();
  (void) Start;

  Write32(is64Bit() ? MachO::MH_MAGIC_64 : MachO::MH_MAGIC);

  Write32(TargetObjectWriter->getCPUType());
  Write32(TargetObjectWriter->getCPUSubtype());

  Write32(MachO::MH_OBJECT);
  Write32(NumLoadCommands);
  Write32(LoadCommandsSize);
  Write32(Flags);
  if (is64Bit())
    Write32(0); // reserved

  assert(OS.tell() - Start ==
         (is64Bit()?sizeof(MachO::mach_header_64): sizeof(MachO::mach_header)));
}

/// WriteSegmentLoadCommand - Write a segment load command.
///
/// \param NumSections The number of sections in this segment.
/// \param SectionDataSize The total size of the sections.
void MachObjectWriter::WriteSegmentLoadCommand(unsigned NumSections,
                                               uint64_t VMSize,
                                               uint64_t SectionDataStartOffset,
                                               uint64_t SectionDataSize) {
  // struct segment_command (56 bytes) or
  // struct segment_command_64 (72 bytes)

  uint64_t Start = OS.tell();
  (void) Start;

  unsigned SegmentLoadCommandSize =
    is64Bit() ? sizeof(MachO::segment_command_64):
    sizeof(MachO::segment_command);
  Write32(is64Bit() ? MachO::LC_SEGMENT_64 : MachO::LC_SEGMENT);
  Write32(SegmentLoadCommandSize +
          NumSections * (is64Bit() ? sizeof(MachO::section_64) :
                         sizeof(MachO::section)));

  WriteBytes("", 16);
  if (is64Bit()) {
    Write64(0); // vmaddr
    Write64(VMSize); // vmsize
    Write64(SectionDataStartOffset); // file offset
    Write64(SectionDataSize); // file size
  } else {
    Write32(0); // vmaddr
    Write32(VMSize); // vmsize
    Write32(SectionDataStartOffset); // file offset
    Write32(SectionDataSize); // file size
  }
  // maxprot
  Write32(MachO::VM_PROT_READ | MachO::VM_PROT_WRITE | MachO::VM_PROT_EXECUTE); 
  // initprot
  Write32(MachO::VM_PROT_READ | MachO::VM_PROT_WRITE | MachO::VM_PROT_EXECUTE); 
  Write32(NumSections);
  Write32(0); // flags

  assert(OS.tell() - Start == SegmentLoadCommandSize);
}

void MachObjectWriter::WriteSection(const MCAssembler &Asm,
                                    const MCAsmLayout &Layout,
                                    const MCSectionData &SD,
                                    uint64_t FileOffset,
                                    uint64_t RelocationsStart,
                                    unsigned NumRelocations) {
  uint64_t SectionSize = Layout.getSectionAddressSize(&SD);

  // The offset is unused for virtual sections.
  if (SD.getSection().isVirtualSection()) {
    assert(Layout.getSectionFileSize(&SD) == 0 && "Invalid file size!");
    FileOffset = 0;
  }

  // struct section (68 bytes) or
  // struct section_64 (80 bytes)

  uint64_t Start = OS.tell();
  (void) Start;

  const MCSectionMachO &Section = cast<MCSectionMachO>(SD.getSection());
  WriteBytes(Section.getSectionName(), 16);
  WriteBytes(Section.getSegmentName(), 16);
  if (is64Bit()) {
    Write64(getSectionAddress(&SD)); // address
    Write64(SectionSize); // size
  } else {
    Write32(getSectionAddress(&SD)); // address
    Write32(SectionSize); // size
  }
  Write32(FileOffset);

  unsigned Flags = Section.getTypeAndAttributes();
  if (SD.hasInstructions())
    Flags |= MachO::S_ATTR_SOME_INSTRUCTIONS;

  assert(isPowerOf2_32(SD.getAlignment()) && "Invalid alignment!");
  Write32(Log2_32(SD.getAlignment()));
  Write32(NumRelocations ? RelocationsStart : 0);
  Write32(NumRelocations);
  Write32(Flags);
  Write32(IndirectSymBase.lookup(&SD)); // reserved1
  Write32(Section.getStubSize()); // reserved2
  if (is64Bit())
    Write32(0); // reserved3

  assert(OS.tell() - Start == (is64Bit() ? sizeof(MachO::section_64) :
                               sizeof(MachO::section)));
}

void MachObjectWriter::WriteSymtabLoadCommand(uint32_t SymbolOffset,
                                              uint32_t NumSymbols,
                                              uint32_t StringTableOffset,
                                              uint32_t StringTableSize) {
  // struct symtab_command (24 bytes)

  uint64_t Start = OS.tell();
  (void) Start;

  Write32(MachO::LC_SYMTAB);
  Write32(sizeof(MachO::symtab_command));
  Write32(SymbolOffset);
  Write32(NumSymbols);
  Write32(StringTableOffset);
  Write32(StringTableSize);

  assert(OS.tell() - Start == sizeof(MachO::symtab_command));
}

void MachObjectWriter::WriteDysymtabLoadCommand(uint32_t FirstLocalSymbol,
                                                uint32_t NumLocalSymbols,
                                                uint32_t FirstExternalSymbol,
                                                uint32_t NumExternalSymbols,
                                                uint32_t FirstUndefinedSymbol,
                                                uint32_t NumUndefinedSymbols,
                                                uint32_t IndirectSymbolOffset,
                                                uint32_t NumIndirectSymbols) {
  // struct dysymtab_command (80 bytes)

  uint64_t Start = OS.tell();
  (void) Start;

  Write32(MachO::LC_DYSYMTAB);
  Write32(sizeof(MachO::dysymtab_command));
  Write32(FirstLocalSymbol);
  Write32(NumLocalSymbols);
  Write32(FirstExternalSymbol);
  Write32(NumExternalSymbols);
  Write32(FirstUndefinedSymbol);
  Write32(NumUndefinedSymbols);
  Write32(0); // tocoff
  Write32(0); // ntoc
  Write32(0); // modtaboff
  Write32(0); // nmodtab
  Write32(0); // extrefsymoff
  Write32(0); // nextrefsyms
  Write32(IndirectSymbolOffset);
  Write32(NumIndirectSymbols);
  Write32(0); // extreloff
  Write32(0); // nextrel
  Write32(0); // locreloff
  Write32(0); // nlocrel

  assert(OS.tell() - Start == sizeof(MachO::dysymtab_command));
}

MachObjectWriter::MachSymbolData *
MachObjectWriter::findSymbolData(const MCSymbol &Sym) {
  for (auto &Entry : LocalSymbolData)
    if (&Entry.SymbolData->getSymbol() == &Sym)
      return &Entry;

  for (auto &Entry : ExternalSymbolData)
    if (&Entry.SymbolData->getSymbol() == &Sym)
      return &Entry;

  for (auto &Entry : UndefinedSymbolData)
    if (&Entry.SymbolData->getSymbol() == &Sym)
      return &Entry;

  return nullptr;
}

void MachObjectWriter::WriteNlist(MachSymbolData &MSD,
                                  const MCAsmLayout &Layout) {
  MCSymbolData &Data = *MSD.SymbolData;
  const MCSymbol *Symbol = &Data.getSymbol();
  const MCSymbol *AliasedSymbol = &Symbol->AliasedSymbol();
  uint8_t SectionIndex = MSD.SectionIndex;
  uint8_t Type = 0;
  uint16_t Flags = Data.getFlags();
  uint64_t Address = 0;
  bool IsAlias = Symbol != AliasedSymbol;

  MachSymbolData *AliaseeInfo;
  if (IsAlias) {
    AliaseeInfo = findSymbolData(*AliasedSymbol);
    if (AliaseeInfo)
      SectionIndex = AliaseeInfo->SectionIndex;
    Symbol = AliasedSymbol;
  }

  // Set the N_TYPE bits. See <mach-o/nlist.h>.
  //
  // FIXME: Are the prebound or indirect fields possible here?
  if (IsAlias && Symbol->isUndefined())
    Type = MachO::N_INDR;
  else if (Symbol->isUndefined())
    Type = MachO::N_UNDF;
  else if (Symbol->isAbsolute())
    Type = MachO::N_ABS;
  else
    Type = MachO::N_SECT;

  // FIXME: Set STAB bits.

  if (Data.isPrivateExtern())
    Type |= MachO::N_PEXT;

  // Set external bit.
  if (Data.isExternal() || (!IsAlias && Symbol->isUndefined()))
    Type |= MachO::N_EXT;

  // Compute the symbol address.
  if (IsAlias && Symbol->isUndefined())
    Address = AliaseeInfo->StringIndex;
  else if (Symbol->isDefined())
    Address = getSymbolAddress(&Data, Layout);
  else if (Data.isCommon()) {
    // Common symbols are encoded with the size in the address
    // field, and their alignment in the flags.
    Address = Data.getCommonSize();

    // Common alignment is packed into the 'desc' bits.
    if (unsigned Align = Data.getCommonAlignment()) {
      unsigned Log2Size = Log2_32(Align);
      assert((1U << Log2Size) == Align && "Invalid 'common' alignment!");
      if (Log2Size > 15)
        report_fatal_error("invalid 'common' alignment '" +
                           Twine(Align) + "' for '" + Symbol->getName() + "'",
                           false);
      // FIXME: Keep this mask with the SymbolFlags enumeration.
      Flags = (Flags & 0xF0FF) | (Log2Size << 8);
    }
  }

  if (Layout.getAssembler().isThumbFunc(Symbol))
    Flags |= SF_ThumbFunc;

  // struct nlist (12 bytes)

  Write32(MSD.StringIndex);
  Write8(Type);
  Write8(SectionIndex);

  // The Mach-O streamer uses the lowest 16-bits of the flags for the 'desc'
  // value.
  Write16(Flags);
  if (is64Bit())
    Write64(Address);
  else
    Write32(Address);
}

void MachObjectWriter::WriteLinkeditLoadCommand(uint32_t Type,
                                                uint32_t DataOffset,
                                                uint32_t DataSize) {
  uint64_t Start = OS.tell();
  (void) Start;

  Write32(Type);
  Write32(sizeof(MachO::linkedit_data_command));
  Write32(DataOffset);
  Write32(DataSize);

  assert(OS.tell() - Start == sizeof(MachO::linkedit_data_command));
}

static unsigned ComputeLinkerOptionsLoadCommandSize(
  const std::vector<std::string> &Options, bool is64Bit)
{
  unsigned Size = sizeof(MachO::linker_option_command);
  for (unsigned i = 0, e = Options.size(); i != e; ++i)
    Size += Options[i].size() + 1;
  return RoundUpToAlignment(Size, is64Bit ? 8 : 4);
}

void MachObjectWriter::WriteLinkerOptionsLoadCommand(
  const std::vector<std::string> &Options)
{
  unsigned Size = ComputeLinkerOptionsLoadCommandSize(Options, is64Bit());
  uint64_t Start = OS.tell();
  (void) Start;

  Write32(MachO::LC_LINKER_OPTION);
  Write32(Size);
  Write32(Options.size());
  uint64_t BytesWritten = sizeof(MachO::linker_option_command);
  for (unsigned i = 0, e = Options.size(); i != e; ++i) {
    // Write each string, including the null byte.
    const std::string &Option = Options[i];
    WriteBytes(Option.c_str(), Option.size() + 1);
    BytesWritten += Option.size() + 1;
  }

  // Pad to a multiple of the pointer size.
  WriteBytes("", OffsetToAlignment(BytesWritten, is64Bit() ? 8 : 4));

  assert(OS.tell() - Start == Size);
}


void MachObjectWriter::RecordRelocation(const MCAssembler &Asm,
                                        const MCAsmLayout &Layout,
                                        const MCFragment *Fragment,
                                        const MCFixup &Fixup,
                                        MCValue Target,
                                        bool &IsPCRel,
                                        uint64_t &FixedValue) {
  TargetObjectWriter->RecordRelocation(this, Asm, Layout, Fragment, Fixup,
                                       Target, FixedValue);
}

void MachObjectWriter::BindIndirectSymbols(MCAssembler &Asm) {
  // This is the point where 'as' creates actual symbols for indirect symbols
  // (in the following two passes). It would be easier for us to do this sooner
  // when we see the attribute, but that makes getting the order in the symbol
  // table much more complicated than it is worth.
  //
  // FIXME: Revisit this when the dust settles.

  // Report errors for use of .indirect_symbol not in a symbol pointer section
  // or stub section.
  for (MCAssembler::indirect_symbol_iterator it = Asm.indirect_symbol_begin(),
         ie = Asm.indirect_symbol_end(); it != ie; ++it) {
    const MCSectionMachO &Section =
      cast<MCSectionMachO>(it->SectionData->getSection());

    if (Section.getType() != MachO::S_NON_LAZY_SYMBOL_POINTERS &&
        Section.getType() != MachO::S_LAZY_SYMBOL_POINTERS &&
        Section.getType() != MachO::S_SYMBOL_STUBS) {
	MCSymbol &Symbol = *it->Symbol;
	report_fatal_error("indirect symbol '" + Symbol.getName() +
                           "' not in a symbol pointer or stub section");
    }
  }

  // Bind non-lazy symbol pointers first.
  unsigned IndirectIndex = 0;
  for (MCAssembler::indirect_symbol_iterator it = Asm.indirect_symbol_begin(),
         ie = Asm.indirect_symbol_end(); it != ie; ++it, ++IndirectIndex) {
    const MCSectionMachO &Section =
      cast<MCSectionMachO>(it->SectionData->getSection());

    if (Section.getType() != MachO::S_NON_LAZY_SYMBOL_POINTERS)
      continue;

    // Initialize the section indirect symbol base, if necessary.
    IndirectSymBase.insert(std::make_pair(it->SectionData, IndirectIndex));

    Asm.getOrCreateSymbolData(*it->Symbol);
  }

  // Then lazy symbol pointers and symbol stubs.
  IndirectIndex = 0;
  for (MCAssembler::indirect_symbol_iterator it = Asm.indirect_symbol_begin(),
         ie = Asm.indirect_symbol_end(); it != ie; ++it, ++IndirectIndex) {
    const MCSectionMachO &Section =
      cast<MCSectionMachO>(it->SectionData->getSection());

    if (Section.getType() != MachO::S_LAZY_SYMBOL_POINTERS &&
        Section.getType() != MachO::S_SYMBOL_STUBS)
      continue;

    // Initialize the section indirect symbol base, if necessary.
    IndirectSymBase.insert(std::make_pair(it->SectionData, IndirectIndex));

    // Set the symbol type to undefined lazy, but only on construction.
    //
    // FIXME: Do not hardcode.
    bool Created;
    MCSymbolData &Entry = Asm.getOrCreateSymbolData(*it->Symbol, &Created);
    if (Created)
      Entry.setFlags(Entry.getFlags() | 0x0001);
  }
}

/// ComputeSymbolTable - Compute the symbol table data
void MachObjectWriter::ComputeSymbolTable(
    MCAssembler &Asm, std::vector<MachSymbolData> &LocalSymbolData,
    std::vector<MachSymbolData> &ExternalSymbolData,
    std::vector<MachSymbolData> &UndefinedSymbolData) {
  // Build section lookup table.
  DenseMap<const MCSection*, uint8_t> SectionIndexMap;
  unsigned Index = 1;
  for (MCAssembler::iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it, ++Index)
    SectionIndexMap[&it->getSection()] = Index;
  assert(Index <= 256 && "Too many sections!");

  // Build the string table.
  for (MCSymbolData &SD : Asm.symbols()) {
    const MCSymbol &Symbol = SD.getSymbol();
    if (!Asm.isSymbolLinkerVisible(Symbol))
      continue;

    StringTable.add(Symbol.getName());
  }
  StringTable.finalize(StringTableBuilder::MachO);

  // Build the symbol arrays but only for non-local symbols.
  //
  // The particular order that we collect and then sort the symbols is chosen to
  // match 'as'. Even though it doesn't matter for correctness, this is
  // important for letting us diff .o files.
  for (MCSymbolData &SD : Asm.symbols()) {
    const MCSymbol &Symbol = SD.getSymbol();

    // Ignore non-linker visible symbols.
    if (!Asm.isSymbolLinkerVisible(Symbol))
      continue;

    if (!SD.isExternal() && !Symbol.isUndefined())
      continue;

    MachSymbolData MSD;
    MSD.SymbolData = &SD;
    MSD.StringIndex = StringTable.getOffset(Symbol.getName());

    if (Symbol.isUndefined()) {
      MSD.SectionIndex = 0;
      UndefinedSymbolData.push_back(MSD);
    } else if (Symbol.isAbsolute()) {
      MSD.SectionIndex = 0;
      ExternalSymbolData.push_back(MSD);
    } else {
      MSD.SectionIndex = SectionIndexMap.lookup(&Symbol.getSection());
      assert(MSD.SectionIndex && "Invalid section index!");
      ExternalSymbolData.push_back(MSD);
    }
  }

  // Now add the data for local symbols.
  for (MCSymbolData &SD : Asm.symbols()) {
    const MCSymbol &Symbol = SD.getSymbol();

    // Ignore non-linker visible symbols.
    if (!Asm.isSymbolLinkerVisible(Symbol))
      continue;

    if (SD.isExternal() || Symbol.isUndefined())
      continue;

    MachSymbolData MSD;
    MSD.SymbolData = &SD;
    MSD.StringIndex = StringTable.getOffset(Symbol.getName());

    if (Symbol.isAbsolute()) {
      MSD.SectionIndex = 0;
      LocalSymbolData.push_back(MSD);
    } else {
      MSD.SectionIndex = SectionIndexMap.lookup(&Symbol.getSection());
      assert(MSD.SectionIndex && "Invalid section index!");
      LocalSymbolData.push_back(MSD);
    }
  }

  // External and undefined symbols are required to be in lexicographic order.
  std::sort(ExternalSymbolData.begin(), ExternalSymbolData.end());
  std::sort(UndefinedSymbolData.begin(), UndefinedSymbolData.end());

  // Set the symbol indices.
  Index = 0;
  for (unsigned i = 0, e = LocalSymbolData.size(); i != e; ++i)
    LocalSymbolData[i].SymbolData->setIndex(Index++);
  for (unsigned i = 0, e = ExternalSymbolData.size(); i != e; ++i)
    ExternalSymbolData[i].SymbolData->setIndex(Index++);
  for (unsigned i = 0, e = UndefinedSymbolData.size(); i != e; ++i)
    UndefinedSymbolData[i].SymbolData->setIndex(Index++);
}

void MachObjectWriter::computeSectionAddresses(const MCAssembler &Asm,
                                               const MCAsmLayout &Layout) {
  uint64_t StartAddress = 0;
  const SmallVectorImpl<MCSectionData*> &Order = Layout.getSectionOrder();
  for (int i = 0, n = Order.size(); i != n ; ++i) {
    const MCSectionData *SD = Order[i];
    StartAddress = RoundUpToAlignment(StartAddress, SD->getAlignment());
    SectionAddress[SD] = StartAddress;
    StartAddress += Layout.getSectionAddressSize(SD);

    // Explicitly pad the section to match the alignment requirements of the
    // following one. This is for 'gas' compatibility, it shouldn't
    /// strictly be necessary.
    StartAddress += getPaddingSize(SD, Layout);
  }
}

void MachObjectWriter::markAbsoluteVariableSymbols(MCAssembler &Asm,
                                                   const MCAsmLayout &Layout) {
  for (MCSymbolData &SD : Asm.symbols()) {
    if (!SD.getSymbol().isVariable())
      continue;

    // Is the variable is a symbol difference (SA - SB + C) expression,
    // and neither symbol is external, mark the variable as absolute.
    const MCExpr *Expr = SD.getSymbol().getVariableValue();
    MCValue Value;
    if (Expr->EvaluateAsRelocatable(Value, &Layout, nullptr)) {
      if (Value.getSymA() && Value.getSymB())
        const_cast<MCSymbol*>(&SD.getSymbol())->setAbsolute();
    }
  }
}

void MachObjectWriter::ExecutePostLayoutBinding(MCAssembler &Asm,
                                                const MCAsmLayout &Layout) {
  computeSectionAddresses(Asm, Layout);

  // Create symbol data for any indirect symbols.
  BindIndirectSymbols(Asm);

  // Mark symbol difference expressions in variables (from .set or = directives)
  // as absolute.
  markAbsoluteVariableSymbols(Asm, Layout);

  // Compute symbol table information and bind symbol indices.
  ComputeSymbolTable(Asm, LocalSymbolData, ExternalSymbolData,
                     UndefinedSymbolData);
}

bool MachObjectWriter::
IsSymbolRefDifferenceFullyResolvedImpl(const MCAssembler &Asm,
                                       const MCSymbolData &DataA,
                                       const MCFragment &FB,
                                       bool InSet,
                                       bool IsPCRel) const {
  if (InSet)
    return true;

  // The effective address is
  //     addr(atom(A)) + offset(A)
  //   - addr(atom(B)) - offset(B)
  // and the offsets are not relocatable, so the fixup is fully resolved when
  //  addr(atom(A)) - addr(atom(B)) == 0.
  const MCSymbolData *A_Base = nullptr, *B_Base = nullptr;

  const MCSymbol &SA = DataA.getSymbol().AliasedSymbol();
  const MCSection &SecA = SA.getSection();
  const MCSection &SecB = FB.getParent()->getSection();

  if (IsPCRel) {
    // The simple (Darwin, except on x86_64) way of dealing with this was to
    // assume that any reference to a temporary symbol *must* be a temporary
    // symbol in the same atom, unless the sections differ. Therefore, any PCrel
    // relocation to a temporary symbol (in the same section) is fully
    // resolved. This also works in conjunction with absolutized .set, which
    // requires the compiler to use .set to absolutize the differences between
    // symbols which the compiler knows to be assembly time constants, so we
    // don't need to worry about considering symbol differences fully resolved.
    //
    // If the file isn't using sub-sections-via-symbols, we can make the
    // same assumptions about any symbol that we normally make about
    // assembler locals.

    bool hasReliableSymbolDifference = isX86_64();
    if (!hasReliableSymbolDifference) {
      if (!SA.isInSection() || &SecA != &SecB ||
          (!SA.isTemporary() &&
           FB.getAtom() != Asm.getSymbolData(SA).getFragment()->getAtom() &&
           Asm.getSubsectionsViaSymbols()))
        return false;
      return true;
    }
    // For Darwin x86_64, there is one special case when the reference IsPCRel.
    // If the fragment with the reference does not have a base symbol but meets
    // the simple way of dealing with this, in that it is a temporary symbol in
    // the same atom then it is assumed to be fully resolved.  This is needed so
    // a relocation entry is not created and so the static linker does not
    // mess up the reference later.
    else if(!FB.getAtom() &&
            SA.isTemporary() && SA.isInSection() && &SecA == &SecB){
      return true;
    }
  } else {
    if (!TargetObjectWriter->useAggressiveSymbolFolding())
      return false;
  }

  // If they are not in the same section, we can't compute the diff.
  if (&SecA != &SecB)
    return false;

  const MCFragment *FA = Asm.getSymbolData(SA).getFragment();

  // Bail if the symbol has no fragment.
  if (!FA)
    return false;

  A_Base = FA->getAtom();
  B_Base = FB.getAtom();

  // If the atoms are the same, they are guaranteed to have the same address.
  if (A_Base == B_Base)
    return true;

  // Otherwise, we can't prove this is fully resolved.
  return false;
}

void MachObjectWriter::WriteObject(MCAssembler &Asm,
                                   const MCAsmLayout &Layout) {
  unsigned NumSections = Asm.size();
  const MCAssembler::VersionMinInfoType &VersionInfo =
    Layout.getAssembler().getVersionMinInfo();

  // The section data starts after the header, the segment load command (and
  // section headers) and the symbol table.
  unsigned NumLoadCommands = 1;
  uint64_t LoadCommandsSize = is64Bit() ?
    sizeof(MachO::segment_command_64) + NumSections * sizeof(MachO::section_64):
    sizeof(MachO::segment_command) + NumSections * sizeof(MachO::section);

  // Add the deployment target version info load command size, if used.
  if (VersionInfo.Major != 0) {
    ++NumLoadCommands;
    LoadCommandsSize += sizeof(MachO::version_min_command);
  }

  // Add the data-in-code load command size, if used.
  unsigned NumDataRegions = Asm.getDataRegions().size();
  if (NumDataRegions) {
    ++NumLoadCommands;
    LoadCommandsSize += sizeof(MachO::linkedit_data_command);
  }

  // Add the loh load command size, if used.
  uint64_t LOHRawSize = Asm.getLOHContainer().getEmitSize(*this, Layout);
  uint64_t LOHSize = RoundUpToAlignment(LOHRawSize, is64Bit() ? 8 : 4);
  if (LOHSize) {
    ++NumLoadCommands;
    LoadCommandsSize += sizeof(MachO::linkedit_data_command);
  }

  // Add the symbol table load command sizes, if used.
  unsigned NumSymbols = LocalSymbolData.size() + ExternalSymbolData.size() +
    UndefinedSymbolData.size();
  if (NumSymbols) {
    NumLoadCommands += 2;
    LoadCommandsSize += (sizeof(MachO::symtab_command) +
                         sizeof(MachO::dysymtab_command));
  }

  // Add the linker option load commands sizes.
  const std::vector<std::vector<std::string> > &LinkerOptions =
    Asm.getLinkerOptions();
  for (unsigned i = 0, e = LinkerOptions.size(); i != e; ++i) {
    ++NumLoadCommands;
    LoadCommandsSize += ComputeLinkerOptionsLoadCommandSize(LinkerOptions[i],
                                                            is64Bit());
  }
  
  // Compute the total size of the section data, as well as its file size and vm
  // size.
  uint64_t SectionDataStart = (is64Bit() ? sizeof(MachO::mach_header_64) :
                               sizeof(MachO::mach_header)) + LoadCommandsSize;
  uint64_t SectionDataSize = 0;
  uint64_t SectionDataFileSize = 0;
  uint64_t VMSize = 0;
  for (MCAssembler::const_iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it) {
    const MCSectionData &SD = *it;
    uint64_t Address = getSectionAddress(&SD);
    uint64_t Size = Layout.getSectionAddressSize(&SD);
    uint64_t FileSize = Layout.getSectionFileSize(&SD);
    FileSize += getPaddingSize(&SD, Layout);

    VMSize = std::max(VMSize, Address + Size);

    if (SD.getSection().isVirtualSection())
      continue;

    SectionDataSize = std::max(SectionDataSize, Address + Size);
    SectionDataFileSize = std::max(SectionDataFileSize, Address + FileSize);
  }

  // The section data is padded to 4 bytes.
  //
  // FIXME: Is this machine dependent?
  unsigned SectionDataPadding = OffsetToAlignment(SectionDataFileSize, 4);
  SectionDataFileSize += SectionDataPadding;

  // Write the prolog, starting with the header and load command...
  WriteHeader(NumLoadCommands, LoadCommandsSize,
              Asm.getSubsectionsViaSymbols());
  WriteSegmentLoadCommand(NumSections, VMSize,
                          SectionDataStart, SectionDataSize);

  // ... and then the section headers.
  uint64_t RelocTableEnd = SectionDataStart + SectionDataFileSize;
  for (MCAssembler::const_iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it) {
    std::vector<MachO::any_relocation_info> &Relocs = Relocations[it];
    unsigned NumRelocs = Relocs.size();
    uint64_t SectionStart = SectionDataStart + getSectionAddress(it);
    WriteSection(Asm, Layout, *it, SectionStart, RelocTableEnd, NumRelocs);
    RelocTableEnd += NumRelocs * sizeof(MachO::any_relocation_info);
  }

  // Write out the deployment target information, if it's available.
  if (VersionInfo.Major != 0) {
    assert(VersionInfo.Update < 256 && "unencodable update target version");
    assert(VersionInfo.Minor < 256 && "unencodable minor target version");
    assert(VersionInfo.Major < 65536 && "unencodable major target version");
    uint32_t EncodedVersion = VersionInfo.Update | (VersionInfo.Minor << 8) |
      (VersionInfo.Major << 16);
    Write32(VersionInfo.Kind == MCVM_OSXVersionMin ? MachO::LC_VERSION_MIN_MACOSX :
            MachO::LC_VERSION_MIN_IPHONEOS);
    Write32(sizeof(MachO::version_min_command));
    Write32(EncodedVersion);
    Write32(0);         // reserved.
  }

  // Write the data-in-code load command, if used.
  uint64_t DataInCodeTableEnd = RelocTableEnd + NumDataRegions * 8;
  if (NumDataRegions) {
    uint64_t DataRegionsOffset = RelocTableEnd;
    uint64_t DataRegionsSize = NumDataRegions * 8;
    WriteLinkeditLoadCommand(MachO::LC_DATA_IN_CODE, DataRegionsOffset,
                             DataRegionsSize);
  }

  // Write the loh load command, if used.
  uint64_t LOHTableEnd = DataInCodeTableEnd + LOHSize;
  if (LOHSize)
    WriteLinkeditLoadCommand(MachO::LC_LINKER_OPTIMIZATION_HINT,
                             DataInCodeTableEnd, LOHSize);

  // Write the symbol table load command, if used.
  if (NumSymbols) {
    unsigned FirstLocalSymbol = 0;
    unsigned NumLocalSymbols = LocalSymbolData.size();
    unsigned FirstExternalSymbol = FirstLocalSymbol + NumLocalSymbols;
    unsigned NumExternalSymbols = ExternalSymbolData.size();
    unsigned FirstUndefinedSymbol = FirstExternalSymbol + NumExternalSymbols;
    unsigned NumUndefinedSymbols = UndefinedSymbolData.size();
    unsigned NumIndirectSymbols = Asm.indirect_symbol_size();
    unsigned NumSymTabSymbols =
      NumLocalSymbols + NumExternalSymbols + NumUndefinedSymbols;
    uint64_t IndirectSymbolSize = NumIndirectSymbols * 4;
    uint64_t IndirectSymbolOffset = 0;

    // If used, the indirect symbols are written after the section data.
    if (NumIndirectSymbols)
      IndirectSymbolOffset = LOHTableEnd;

    // The symbol table is written after the indirect symbol data.
    uint64_t SymbolTableOffset = LOHTableEnd + IndirectSymbolSize;

    // The string table is written after symbol table.
    uint64_t StringTableOffset =
      SymbolTableOffset + NumSymTabSymbols * (is64Bit() ?
                                              sizeof(MachO::nlist_64) :
                                              sizeof(MachO::nlist));
    WriteSymtabLoadCommand(SymbolTableOffset, NumSymTabSymbols,
                           StringTableOffset, StringTable.data().size());

    WriteDysymtabLoadCommand(FirstLocalSymbol, NumLocalSymbols,
                             FirstExternalSymbol, NumExternalSymbols,
                             FirstUndefinedSymbol, NumUndefinedSymbols,
                             IndirectSymbolOffset, NumIndirectSymbols);
  }

  // Write the linker options load commands.
  for (unsigned i = 0, e = LinkerOptions.size(); i != e; ++i) {
    WriteLinkerOptionsLoadCommand(LinkerOptions[i]);
  }

  // Write the actual section data.
  for (MCAssembler::const_iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it) {
    Asm.writeSectionData(it, Layout);

    uint64_t Pad = getPaddingSize(it, Layout);
    for (unsigned int i = 0; i < Pad; ++i)
      Write8(0);
  }

  // Write the extra padding.
  WriteZeros(SectionDataPadding);

  // Write the relocation entries.
  for (MCAssembler::const_iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it) {
    // Write the section relocation entries, in reverse order to match 'as'
    // (approximately, the exact algorithm is more complicated than this).
    std::vector<MachO::any_relocation_info> &Relocs = Relocations[it];
    for (unsigned i = 0, e = Relocs.size(); i != e; ++i) {
      Write32(Relocs[e - i - 1].r_word0);
      Write32(Relocs[e - i - 1].r_word1);
    }
  }

  // Write out the data-in-code region payload, if there is one.
  for (MCAssembler::const_data_region_iterator
         it = Asm.data_region_begin(), ie = Asm.data_region_end();
         it != ie; ++it) {
    const DataRegionData *Data = &(*it);
    uint64_t Start =
      getSymbolAddress(&Layout.getAssembler().getSymbolData(*Data->Start),
                       Layout);
    uint64_t End =
      getSymbolAddress(&Layout.getAssembler().getSymbolData(*Data->End),
                       Layout);
    DEBUG(dbgs() << "data in code region-- kind: " << Data->Kind
                 << "  start: " << Start << "(" << Data->Start->getName() << ")"
                 << "  end: " << End << "(" << Data->End->getName() << ")"
                 << "  size: " << End - Start
                 << "\n");
    Write32(Start);
    Write16(End - Start);
    Write16(Data->Kind);
  }

  // Write out the loh commands, if there is one.
  if (LOHSize) {
#ifndef NDEBUG
    unsigned Start = OS.tell();
#endif
    Asm.getLOHContainer().Emit(*this, Layout);
    // Pad to a multiple of the pointer size.
    WriteBytes("", OffsetToAlignment(LOHRawSize, is64Bit() ? 8 : 4));
    assert(OS.tell() - Start == LOHSize);
  }

  // Write the symbol table data, if used.
  if (NumSymbols) {
    // Write the indirect symbol entries.
    for (MCAssembler::const_indirect_symbol_iterator
           it = Asm.indirect_symbol_begin(),
           ie = Asm.indirect_symbol_end(); it != ie; ++it) {
      // Indirect symbols in the non-lazy symbol pointer section have some
      // special handling.
      const MCSectionMachO &Section =
        static_cast<const MCSectionMachO&>(it->SectionData->getSection());
      if (Section.getType() == MachO::S_NON_LAZY_SYMBOL_POINTERS) {
        // If this symbol is defined and internal, mark it as such.
        if (it->Symbol->isDefined() &&
            !Asm.getSymbolData(*it->Symbol).isExternal()) {
          uint32_t Flags = MachO::INDIRECT_SYMBOL_LOCAL;
          if (it->Symbol->isAbsolute())
            Flags |= MachO::INDIRECT_SYMBOL_ABS;
          Write32(Flags);
          continue;
        }
      }

      Write32(Asm.getSymbolData(*it->Symbol).getIndex());
    }

    // FIXME: Check that offsets match computed ones.

    // Write the symbol table entries.
    for (unsigned i = 0, e = LocalSymbolData.size(); i != e; ++i)
      WriteNlist(LocalSymbolData[i], Layout);
    for (unsigned i = 0, e = ExternalSymbolData.size(); i != e; ++i)
      WriteNlist(ExternalSymbolData[i], Layout);
    for (unsigned i = 0, e = UndefinedSymbolData.size(); i != e; ++i)
      WriteNlist(UndefinedSymbolData[i], Layout);

    // Write the string table.
    OS << StringTable.data();
  }
}

MCObjectWriter *llvm::createMachObjectWriter(MCMachObjectTargetWriter *MOTW,
                                             raw_ostream &OS,
                                             bool IsLittleEndian) {
  return new MachObjectWriter(MOTW, OS, IsLittleEndian);
}
