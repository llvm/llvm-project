//===- lib/MC/ELFObjectWriter.cpp - ELF File Writer -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements ELF object file writer information.
//
//===----------------------------------------------------------------------===//

#include "MCELF.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCELFSymbolFlags.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ELF.h"

#include <vector>
using namespace llvm;

#undef  DEBUG_TYPE
#define DEBUG_TYPE "reloc-info"

namespace {
class ELFObjectWriter : public MCObjectWriter {
  protected:

    static bool isFixupKindPCRel(const MCAssembler &Asm, unsigned Kind);
    static bool RelocNeedsGOT(MCSymbolRefExpr::VariantKind Variant);
    static uint64_t SymbolValue(MCSymbolData &Data, const MCAsmLayout &Layout);
    static bool isInSymtab(const MCAssembler &Asm, const MCSymbolData &Data,
                           bool Used, bool Renamed);
    static bool isLocal(const MCSymbolData &Data, bool isSignature,
                        bool isUsedInReloc);
    static bool IsELFMetaDataSection(const MCSectionData &SD);
    static uint64_t DataSectionSize(const MCSectionData &SD);
    static uint64_t GetSectionFileSize(const MCAsmLayout &Layout,
                                       const MCSectionData &SD);
    static uint64_t GetSectionAddressSize(const MCAsmLayout &Layout,
                                          const MCSectionData &SD);

    void WriteDataSectionData(MCAssembler &Asm,
                              const MCAsmLayout &Layout,
                              const MCSectionELF &Section);

    /*static bool isFixupKindX86RIPRel(unsigned Kind) {
      return Kind == X86::reloc_riprel_4byte ||
        Kind == X86::reloc_riprel_4byte_movq_load;
    }*/

    /// ELFSymbolData - Helper struct for containing some precomputed
    /// information on symbols.
    struct ELFSymbolData {
      MCSymbolData *SymbolData;
      uint64_t StringIndex;
      uint32_t SectionIndex;

      // Support lexicographic sorting.
      bool operator<(const ELFSymbolData &RHS) const {
        if (MCELF::GetType(*SymbolData) == ELF::STT_FILE)
          return true;
        if (MCELF::GetType(*RHS.SymbolData) == ELF::STT_FILE)
          return false;
        return SymbolData->getSymbol().getName() <
               RHS.SymbolData->getSymbol().getName();
      }
    };

    /// The target specific ELF writer instance.
    llvm::OwningPtr<MCELFObjectTargetWriter> TargetObjectWriter;

    SmallPtrSet<const MCSymbol *, 16> UsedInReloc;
    SmallPtrSet<const MCSymbol *, 16> WeakrefUsedInReloc;
    DenseMap<const MCSymbol *, const MCSymbol *> Renames;

    llvm::DenseMap<const MCSectionData*,
                   std::vector<ELFRelocationEntry> > Relocations;
    DenseMap<const MCSection*, uint64_t> SectionStringTableIndex;

    /// @}
    /// @name Symbol Table Data
    /// @{

    SmallString<256> StringTable;
    std::vector<ELFSymbolData> LocalSymbolData;
    std::vector<ELFSymbolData> ExternalSymbolData;
    std::vector<ELFSymbolData> UndefinedSymbolData;

    /// @}

    bool NeedsGOT;

    bool NeedsSymtabShndx;

    // This holds the symbol table index of the last local symbol.
    unsigned LastLocalSymbolIndex;
    // This holds the .strtab section index.
    unsigned StringTableIndex;
    // This holds the .symtab section index.
    unsigned SymbolTableIndex;

    unsigned ShstrtabIndex;


    const MCSymbol *SymbolToReloc(const MCAssembler &Asm,
                                  const MCValue &Target,
                                  const MCFragment &F,
                                  const MCFixup &Fixup,
                                  bool IsPCRel) const;

    // TargetObjectWriter wrappers.
    const MCSymbol *ExplicitRelSym(const MCAssembler &Asm,
                                   const MCValue &Target,
                                   const MCFragment &F,
                                   const MCFixup &Fixup,
                                   bool IsPCRel) const {
      return TargetObjectWriter->ExplicitRelSym(Asm, Target, F, Fixup, IsPCRel);
    }
    const MCSymbol *undefinedExplicitRelSym(const MCValue &Target,
                                            const MCFixup &Fixup,
                                            bool IsPCRel) const {
      return TargetObjectWriter->undefinedExplicitRelSym(Target, Fixup, IsPCRel);
    }

    bool is64Bit() const { return TargetObjectWriter->is64Bit(); }
    bool hasRelocationAddend() const {
      return TargetObjectWriter->hasRelocationAddend();
    }
    unsigned getEFlags() const {
      return TargetObjectWriter->getEFlags();
    }
    unsigned GetRelocType(const MCValue &Target, const MCFixup &Fixup,
                          bool IsPCRel, bool IsRelocWithSymbol,
                          int64_t Addend) const {
      return TargetObjectWriter->GetRelocType(Target, Fixup, IsPCRel,
                                              IsRelocWithSymbol, Addend);
    }


  public:
    ELFObjectWriter(MCELFObjectTargetWriter *MOTW,
                    raw_ostream &_OS, bool IsLittleEndian)
      : MCObjectWriter(_OS, IsLittleEndian),
        TargetObjectWriter(MOTW),
        NeedsGOT(false), NeedsSymtabShndx(false){
    }

    virtual ~ELFObjectWriter();

    void WriteWord(uint64_t W) {
      if (is64Bit())
        Write64(W);
      else
        Write32(W);
    }

    void StringLE16(char *buf, uint16_t Value) {
      buf[0] = char(Value >> 0);
      buf[1] = char(Value >> 8);
    }

    void StringLE32(char *buf, uint32_t Value) {
      StringLE16(buf, uint16_t(Value >> 0));
      StringLE16(buf + 2, uint16_t(Value >> 16));
    }

    void StringLE64(char *buf, uint64_t Value) {
      StringLE32(buf, uint32_t(Value >> 0));
      StringLE32(buf + 4, uint32_t(Value >> 32));
    }

    void StringBE16(char *buf ,uint16_t Value) {
      buf[0] = char(Value >> 8);
      buf[1] = char(Value >> 0);
    }

    void StringBE32(char *buf, uint32_t Value) {
      StringBE16(buf, uint16_t(Value >> 16));
      StringBE16(buf + 2, uint16_t(Value >> 0));
    }

    void StringBE64(char *buf, uint64_t Value) {
      StringBE32(buf, uint32_t(Value >> 32));
      StringBE32(buf + 4, uint32_t(Value >> 0));
    }

    void String8(MCDataFragment &F, uint8_t Value) {
      char buf[1];
      buf[0] = Value;
      F.getContents() += StringRef(buf, 1);
    }

    void String16(MCDataFragment &F, uint16_t Value) {
      char buf[2];
      if (isLittleEndian())
        StringLE16(buf, Value);
      else
        StringBE16(buf, Value);
      F.getContents() += StringRef(buf, 2);
    }

    void String32(MCDataFragment &F, uint32_t Value) {
      char buf[4];
      if (isLittleEndian())
        StringLE32(buf, Value);
      else
        StringBE32(buf, Value);
      F.getContents() += StringRef(buf, 4);
    }

    void String64(MCDataFragment &F, uint64_t Value) {
      char buf[8];
      if (isLittleEndian())
        StringLE64(buf, Value);
      else
        StringBE64(buf, Value);
      F.getContents() += StringRef(buf, 8);
    }

    void WriteHeader(uint64_t SectionDataSize,
                     unsigned NumberOfSections);

    void WriteSymbolEntry(MCDataFragment *SymtabF,
                          MCDataFragment *ShndxF,
                          uint64_t name, uint8_t info,
                          uint64_t value, uint64_t size,
                          uint8_t other, uint32_t shndx,
                          bool Reserved);

    void WriteSymbol(MCDataFragment *SymtabF,  MCDataFragment *ShndxF,
                     ELFSymbolData &MSD,
                     const MCAsmLayout &Layout);

    typedef DenseMap<const MCSectionELF*, uint32_t> SectionIndexMapTy;
    void WriteSymbolTable(MCDataFragment *SymtabF,
                          MCDataFragment *ShndxF,
                          const MCAssembler &Asm,
                          const MCAsmLayout &Layout,
                          const SectionIndexMapTy &SectionIndexMap);

    virtual void RecordRelocation(const MCAssembler &Asm,
                                  const MCAsmLayout &Layout,
                                  const MCFragment *Fragment,
                                  const MCFixup &Fixup,
                                  MCValue Target, uint64_t &FixedValue);

    uint64_t getSymbolIndexInSymbolTable(const MCAssembler &Asm,
                                         const MCSymbol *S);

    // Map from a group section to the signature symbol
    typedef DenseMap<const MCSectionELF*, const MCSymbol*> GroupMapTy;
    // Map from a signature symbol to the group section
    typedef DenseMap<const MCSymbol*, const MCSectionELF*> RevGroupMapTy;
    // Map from a section to the section with the relocations
    typedef DenseMap<const MCSectionELF*, const MCSectionELF*> RelMapTy;
    // Map from a section to its offset
    typedef DenseMap<const MCSectionELF*, uint64_t> SectionOffsetMapTy;

    /// ComputeSymbolTable - Compute the symbol table data
    ///
    /// \param Asm - The assembler.
    /// \param SectionIndexMap - Maps a section to its index.
    /// \param RevGroupMap - Maps a signature symbol to the group section.
    /// \param NumRegularSections - Number of non-relocation sections.
    void ComputeSymbolTable(MCAssembler &Asm,
                            const SectionIndexMapTy &SectionIndexMap,
                            RevGroupMapTy RevGroupMap,
                            unsigned NumRegularSections);

    void ComputeIndexMap(MCAssembler &Asm,
                         SectionIndexMapTy &SectionIndexMap,
                         const RelMapTy &RelMap);

    void CreateRelocationSections(MCAssembler &Asm, MCAsmLayout &Layout,
                                  RelMapTy &RelMap);

    void WriteRelocations(MCAssembler &Asm, MCAsmLayout &Layout,
                          const RelMapTy &RelMap);

    void CreateMetadataSections(MCAssembler &Asm, MCAsmLayout &Layout,
                                SectionIndexMapTy &SectionIndexMap,
                                const RelMapTy &RelMap);

    // Create the sections that show up in the symbol table. Currently
    // those are the .note.GNU-stack section and the group sections.
    void CreateIndexedSections(MCAssembler &Asm, MCAsmLayout &Layout,
                               GroupMapTy &GroupMap,
                               RevGroupMapTy &RevGroupMap,
                               SectionIndexMapTy &SectionIndexMap,
                               const RelMapTy &RelMap);

    virtual void ExecutePostLayoutBinding(MCAssembler &Asm,
                                          const MCAsmLayout &Layout);

    void WriteSectionHeader(MCAssembler &Asm, const GroupMapTy &GroupMap,
                            const MCAsmLayout &Layout,
                            const SectionIndexMapTy &SectionIndexMap,
                            const SectionOffsetMapTy &SectionOffsetMap);

    void ComputeSectionOrder(MCAssembler &Asm,
                             std::vector<const MCSectionELF*> &Sections);

    void WriteSecHdrEntry(uint32_t Name, uint32_t Type, uint64_t Flags,
                          uint64_t Address, uint64_t Offset,
                          uint64_t Size, uint32_t Link, uint32_t Info,
                          uint64_t Alignment, uint64_t EntrySize);

    void WriteRelocationsFragment(const MCAssembler &Asm,
                                  MCDataFragment *F,
                                  const MCSectionData *SD);

    virtual bool
    IsSymbolRefDifferenceFullyResolvedImpl(const MCAssembler &Asm,
                                           const MCSymbolData &DataA,
                                           const MCFragment &FB,
                                           bool InSet,
                                           bool IsPCRel) const;

    virtual void WriteObject(MCAssembler &Asm, const MCAsmLayout &Layout);
    void WriteSection(MCAssembler &Asm,
                      const SectionIndexMapTy &SectionIndexMap,
                      uint32_t GroupSymbolIndex,
                      uint64_t Offset, uint64_t Size, uint64_t Alignment,
                      const MCSectionELF &Section);
  };
}

bool ELFObjectWriter::isFixupKindPCRel(const MCAssembler &Asm, unsigned Kind) {
  const MCFixupKindInfo &FKI =
    Asm.getBackend().getFixupKindInfo((MCFixupKind) Kind);

  return FKI.Flags & MCFixupKindInfo::FKF_IsPCRel;
}

bool ELFObjectWriter::RelocNeedsGOT(MCSymbolRefExpr::VariantKind Variant) {
  switch (Variant) {
  default:
    return false;
  case MCSymbolRefExpr::VK_GOT:
  case MCSymbolRefExpr::VK_PLT:
  case MCSymbolRefExpr::VK_GOTPCREL:
  case MCSymbolRefExpr::VK_GOTOFF:
  case MCSymbolRefExpr::VK_TPOFF:
  case MCSymbolRefExpr::VK_TLSGD:
  case MCSymbolRefExpr::VK_GOTTPOFF:
  case MCSymbolRefExpr::VK_INDNTPOFF:
  case MCSymbolRefExpr::VK_NTPOFF:
  case MCSymbolRefExpr::VK_GOTNTPOFF:
  case MCSymbolRefExpr::VK_TLSLDM:
  case MCSymbolRefExpr::VK_DTPOFF:
  case MCSymbolRefExpr::VK_TLSLD:
    return true;
  }
}

ELFObjectWriter::~ELFObjectWriter()
{}

// Emit the ELF header.
void ELFObjectWriter::WriteHeader(uint64_t SectionDataSize,
                                  unsigned NumberOfSections) {
  // ELF Header
  // ----------
  //
  // Note
  // ----
  // emitWord method behaves differently for ELF32 and ELF64, writing
  // 4 bytes in the former and 8 in the latter.

  Write8(0x7f); // e_ident[EI_MAG0]
  Write8('E');  // e_ident[EI_MAG1]
  Write8('L');  // e_ident[EI_MAG2]
  Write8('F');  // e_ident[EI_MAG3]

  Write8(is64Bit() ? ELF::ELFCLASS64 : ELF::ELFCLASS32); // e_ident[EI_CLASS]

  // e_ident[EI_DATA]
  Write8(isLittleEndian() ? ELF::ELFDATA2LSB : ELF::ELFDATA2MSB);

  Write8(ELF::EV_CURRENT);        // e_ident[EI_VERSION]
  // e_ident[EI_OSABI]
  Write8(TargetObjectWriter->getOSABI());
  Write8(0);                  // e_ident[EI_ABIVERSION]

  WriteZeros(ELF::EI_NIDENT - ELF::EI_PAD);

  Write16(ELF::ET_REL);             // e_type

  Write16(TargetObjectWriter->getEMachine()); // e_machine = target

  Write32(ELF::EV_CURRENT);         // e_version
  WriteWord(0);                    // e_entry, no entry point in .o file
  WriteWord(0);                    // e_phoff, no program header for .o
  WriteWord(SectionDataSize + (is64Bit() ? sizeof(ELF::Elf64_Ehdr) :
            sizeof(ELF::Elf32_Ehdr)));  // e_shoff = sec hdr table off in bytes

  // e_flags = whatever the target wants
  Write32(getEFlags());

  // e_ehsize = ELF header size
  Write16(is64Bit() ? sizeof(ELF::Elf64_Ehdr) : sizeof(ELF::Elf32_Ehdr));

  Write16(0);                  // e_phentsize = prog header entry size
  Write16(0);                  // e_phnum = # prog header entries = 0

  // e_shentsize = Section header entry size
  Write16(is64Bit() ? sizeof(ELF::Elf64_Shdr) : sizeof(ELF::Elf32_Shdr));

  // e_shnum     = # of section header ents
  if (NumberOfSections >= ELF::SHN_LORESERVE)
    Write16(ELF::SHN_UNDEF);
  else
    Write16(NumberOfSections);

  // e_shstrndx  = Section # of '.shstrtab'
  if (ShstrtabIndex >= ELF::SHN_LORESERVE)
    Write16(ELF::SHN_XINDEX);
  else
    Write16(ShstrtabIndex);
}

void ELFObjectWriter::WriteSymbolEntry(MCDataFragment *SymtabF,
                                       MCDataFragment *ShndxF,
                                       uint64_t name,
                                       uint8_t info, uint64_t value,
                                       uint64_t size, uint8_t other,
                                       uint32_t shndx,
                                       bool Reserved) {
  if (ShndxF) {
    if (shndx >= ELF::SHN_LORESERVE && !Reserved)
      String32(*ShndxF, shndx);
    else
      String32(*ShndxF, 0);
  }

  uint16_t Index = (shndx >= ELF::SHN_LORESERVE && !Reserved) ?
    uint16_t(ELF::SHN_XINDEX) : shndx;

  if (is64Bit()) {
    String32(*SymtabF, name);  // st_name
    String8(*SymtabF, info);   // st_info
    String8(*SymtabF, other);  // st_other
    String16(*SymtabF, Index); // st_shndx
    String64(*SymtabF, value); // st_value
    String64(*SymtabF, size);  // st_size
  } else {
    String32(*SymtabF, name);  // st_name
    String32(*SymtabF, value); // st_value
    String32(*SymtabF, size);  // st_size
    String8(*SymtabF, info);   // st_info
    String8(*SymtabF, other);  // st_other
    String16(*SymtabF, Index); // st_shndx
  }
}

uint64_t ELFObjectWriter::SymbolValue(MCSymbolData &Data,
                                      const MCAsmLayout &Layout) {
  if (Data.isCommon() && Data.isExternal())
    return Data.getCommonAlignment();

  const MCSymbol &Symbol = Data.getSymbol();

  if (Symbol.isAbsolute() && Symbol.isVariable()) {
    if (const MCExpr *Value = Symbol.getVariableValue()) {
      int64_t IntValue;
      if (Value->EvaluateAsAbsolute(IntValue, Layout))
        return (uint64_t)IntValue;
    }
  }

  if (!Symbol.isInSection())
    return 0;


  if (Data.getFragment()) {
    if (Data.getFlags() & ELF_Other_ThumbFunc)
      return Layout.getSymbolOffset(&Data)+1;
    else
      return Layout.getSymbolOffset(&Data);
  }

  return 0;
}

void ELFObjectWriter::ExecutePostLayoutBinding(MCAssembler &Asm,
                                               const MCAsmLayout &Layout) {
  // The presence of symbol versions causes undefined symbols and
  // versions declared with @@@ to be renamed.

  for (MCAssembler::symbol_iterator it = Asm.symbol_begin(),
         ie = Asm.symbol_end(); it != ie; ++it) {
    const MCSymbol &Alias = it->getSymbol();
    const MCSymbol &Symbol = Alias.AliasedSymbol();
    MCSymbolData &SD = Asm.getSymbolData(Symbol);

    // Not an alias.
    if (&Symbol == &Alias)
      continue;

    StringRef AliasName = Alias.getName();
    size_t Pos = AliasName.find('@');
    if (Pos == StringRef::npos)
      continue;

    // Aliases defined with .symvar copy the binding from the symbol they alias.
    // This is the first place we are able to copy this information.
    it->setExternal(SD.isExternal());
    MCELF::SetBinding(*it, MCELF::GetBinding(SD));

    StringRef Rest = AliasName.substr(Pos);
    if (!Symbol.isUndefined() && !Rest.startswith("@@@"))
      continue;

    // FIXME: produce a better error message.
    if (Symbol.isUndefined() && Rest.startswith("@@") &&
        !Rest.startswith("@@@"))
      report_fatal_error("A @@ version cannot be undefined");

    Renames.insert(std::make_pair(&Symbol, &Alias));
  }
}

void ELFObjectWriter::WriteSymbol(MCDataFragment *SymtabF,
                                  MCDataFragment *ShndxF,
                                  ELFSymbolData &MSD,
                                  const MCAsmLayout &Layout) {
  MCSymbolData &OrigData = *MSD.SymbolData;
  MCSymbolData &Data =
    Layout.getAssembler().getSymbolData(OrigData.getSymbol().AliasedSymbol());

  bool IsReserved = Data.isCommon() || Data.getSymbol().isAbsolute() ||
    Data.getSymbol().isVariable();

  uint8_t Binding = MCELF::GetBinding(OrigData);
  uint8_t Visibility = MCELF::GetVisibility(OrigData);
  uint8_t Type = MCELF::GetType(Data);

  uint8_t Info = (Binding << ELF_STB_Shift) | (Type << ELF_STT_Shift);
  uint8_t Other = Visibility;

  uint64_t Value = SymbolValue(Data, Layout);
  uint64_t Size = 0;

  assert(!(Data.isCommon() && !Data.isExternal()));

  const MCExpr *ESize = Data.getSize();
  if (ESize) {
    int64_t Res;
    if (!ESize->EvaluateAsAbsolute(Res, Layout))
      report_fatal_error("Size expression must be absolute.");
    Size = Res;
  }

  // Write out the symbol table entry
  WriteSymbolEntry(SymtabF, ShndxF, MSD.StringIndex, Info, Value,
                   Size, Other, MSD.SectionIndex, IsReserved);
}

void ELFObjectWriter::WriteSymbolTable(MCDataFragment *SymtabF,
                                       MCDataFragment *ShndxF,
                                       const MCAssembler &Asm,
                                       const MCAsmLayout &Layout,
                                    const SectionIndexMapTy &SectionIndexMap) {
  // The string table must be emitted first because we need the index
  // into the string table for all the symbol names.
  assert(StringTable.size() && "Missing string table");

  // FIXME: Make sure the start of the symbol table is aligned.

  // The first entry is the undefined symbol entry.
  WriteSymbolEntry(SymtabF, ShndxF, 0, 0, 0, 0, 0, 0, false);

  // Write the symbol table entries.
  LastLocalSymbolIndex = LocalSymbolData.size() + 1;
  for (unsigned i = 0, e = LocalSymbolData.size(); i != e; ++i) {
    ELFSymbolData &MSD = LocalSymbolData[i];
    WriteSymbol(SymtabF, ShndxF, MSD, Layout);
  }

  // Write out a symbol table entry for each regular section.
  for (MCAssembler::const_iterator i = Asm.begin(), e = Asm.end(); i != e;
       ++i) {
    const MCSectionELF &Section =
      static_cast<const MCSectionELF&>(i->getSection());
    if (Section.getType() == ELF::SHT_RELA ||
        Section.getType() == ELF::SHT_REL ||
        Section.getType() == ELF::SHT_STRTAB ||
        Section.getType() == ELF::SHT_SYMTAB ||
        Section.getType() == ELF::SHT_SYMTAB_SHNDX)
      continue;
    WriteSymbolEntry(SymtabF, ShndxF, 0, ELF::STT_SECTION, 0, 0,
                     ELF::STV_DEFAULT, SectionIndexMap.lookup(&Section),
                     false);
    LastLocalSymbolIndex++;
  }

  for (unsigned i = 0, e = ExternalSymbolData.size(); i != e; ++i) {
    ELFSymbolData &MSD = ExternalSymbolData[i];
    MCSymbolData &Data = *MSD.SymbolData;
    assert(((Data.getFlags() & ELF_STB_Global) ||
            (Data.getFlags() & ELF_STB_Weak)) &&
           "External symbol requires STB_GLOBAL or STB_WEAK flag");
    WriteSymbol(SymtabF, ShndxF, MSD, Layout);
    if (MCELF::GetBinding(Data) == ELF::STB_LOCAL)
      LastLocalSymbolIndex++;
  }

  for (unsigned i = 0, e = UndefinedSymbolData.size(); i != e; ++i) {
    ELFSymbolData &MSD = UndefinedSymbolData[i];
    MCSymbolData &Data = *MSD.SymbolData;
    WriteSymbol(SymtabF, ShndxF, MSD, Layout);
    if (MCELF::GetBinding(Data) == ELF::STB_LOCAL)
      LastLocalSymbolIndex++;
  }
}

const MCSymbol *ELFObjectWriter::SymbolToReloc(const MCAssembler &Asm,
                                               const MCValue &Target,
                                               const MCFragment &F,
                                               const MCFixup &Fixup,
                                               bool IsPCRel) const {
  const MCSymbol &Symbol = Target.getSymA()->getSymbol();
  const MCSymbol &ASymbol = Symbol.AliasedSymbol();
  const MCSymbol *Renamed = Renames.lookup(&Symbol);
  const MCSymbolData &SD = Asm.getSymbolData(Symbol);

  if (ASymbol.isUndefined()) {
    if (Renamed)
      return Renamed;
    return undefinedExplicitRelSym(Target, Fixup, IsPCRel);
  }

  if (SD.isExternal()) {
    if (Renamed)
      return Renamed;
    return &Symbol;
  }

  const MCSectionELF &Section =
    static_cast<const MCSectionELF&>(ASymbol.getSection());
  const SectionKind secKind = Section.getKind();

  if (secKind.isBSS())
    return ExplicitRelSym(Asm, Target, F, Fixup, IsPCRel);

  if (secKind.isThreadLocal()) {
    if (Renamed)
      return Renamed;
    return &Symbol;
  }

  MCSymbolRefExpr::VariantKind Kind = Target.getSymA()->getKind();
  const MCSectionELF &Sec2 =
    static_cast<const MCSectionELF&>(F.getParent()->getSection());

  if (&Sec2 != &Section &&
      (Kind == MCSymbolRefExpr::VK_PLT ||
       Kind == MCSymbolRefExpr::VK_GOTPCREL ||
       Kind == MCSymbolRefExpr::VK_GOTOFF)) {
    if (Renamed)
      return Renamed;
    return &Symbol;
  }

  if (Section.getFlags() & ELF::SHF_MERGE) {
    if (Target.getConstant() == 0)
      return ExplicitRelSym(Asm, Target, F, Fixup, IsPCRel);
    if (Renamed)
      return Renamed;
    return &Symbol;
  }

  return ExplicitRelSym(Asm, Target, F, Fixup, IsPCRel);

}


void ELFObjectWriter::RecordRelocation(const MCAssembler &Asm,
                                       const MCAsmLayout &Layout,
                                       const MCFragment *Fragment,
                                       const MCFixup &Fixup,
                                       MCValue Target,
                                       uint64_t &FixedValue) {
  int64_t Addend = 0;
  int Index = 0;
  int64_t Value = Target.getConstant();
  const MCSymbol *RelocSymbol = NULL;

  bool IsPCRel = isFixupKindPCRel(Asm, Fixup.getKind());
  if (!Target.isAbsolute()) {
    const MCSymbol &Symbol = Target.getSymA()->getSymbol();
    const MCSymbol &ASymbol = Symbol.AliasedSymbol();
    RelocSymbol = SymbolToReloc(Asm, Target, *Fragment, Fixup, IsPCRel);

    if (const MCSymbolRefExpr *RefB = Target.getSymB()) {
      const MCSymbol &SymbolB = RefB->getSymbol();
      MCSymbolData &SDB = Asm.getSymbolData(SymbolB);
      IsPCRel = true;

      // Offset of the symbol in the section
      int64_t a = Layout.getSymbolOffset(&SDB);

      // Offset of the relocation in the section
      int64_t b = Layout.getFragmentOffset(Fragment) + Fixup.getOffset();
      Value += b - a;
    }

    if (!RelocSymbol) {
      MCSymbolData &SD = Asm.getSymbolData(ASymbol);
      MCFragment *F = SD.getFragment();

      if (F) {
        Index = F->getParent()->getOrdinal() + 1;
        // Offset of the symbol in the section
        Value += Layout.getSymbolOffset(&SD);
      } else {
        Index = 0;
      }
    } else {
      if (Asm.getSymbolData(Symbol).getFlags() & ELF_Other_Weakref)
        WeakrefUsedInReloc.insert(RelocSymbol);
      else
        UsedInReloc.insert(RelocSymbol);
      Index = -1;
    }
    Addend = Value;
    if (hasRelocationAddend())
      Value = 0;
  }

  FixedValue = Value;
  unsigned Type = GetRelocType(Target, Fixup, IsPCRel,
                               (RelocSymbol != 0), Addend);
  MCSymbolRefExpr::VariantKind Modifier = Target.isAbsolute() ?
    MCSymbolRefExpr::VK_None : Target.getSymA()->getKind();
  if (RelocNeedsGOT(Modifier))
    NeedsGOT = true;

  uint64_t RelocOffset = Layout.getFragmentOffset(Fragment) +
    Fixup.getOffset();

  // FIXME: no tests cover this. Is adjustFixupOffset dead code?
  TargetObjectWriter->adjustFixupOffset(Fixup, RelocOffset);

  if (!hasRelocationAddend())
    Addend = 0;

  if (is64Bit())
    assert(isInt<64>(Addend));
  else
    assert(isInt<32>(Addend));

  ELFRelocationEntry ERE(RelocOffset, Index, Type, RelocSymbol, Addend, Fixup);
  Relocations[Fragment->getParent()].push_back(ERE);
}


uint64_t
ELFObjectWriter::getSymbolIndexInSymbolTable(const MCAssembler &Asm,
                                             const MCSymbol *S) {
  MCSymbolData &SD = Asm.getSymbolData(*S);
  return SD.getIndex();
}

bool ELFObjectWriter::isInSymtab(const MCAssembler &Asm,
                                 const MCSymbolData &Data,
                                 bool Used, bool Renamed) {
  if (Data.getFlags() & ELF_Other_Weakref)
    return false;

  if (Used)
    return true;

  if (Renamed)
    return false;

  const MCSymbol &Symbol = Data.getSymbol();

  if (Symbol.getName() == "_GLOBAL_OFFSET_TABLE_")
    return true;

  const MCSymbol &A = Symbol.AliasedSymbol();
  if (Symbol.isVariable() && !A.isVariable() && A.isUndefined())
    return false;

  bool IsGlobal = MCELF::GetBinding(Data) == ELF::STB_GLOBAL;
  if (!Symbol.isVariable() && Symbol.isUndefined() && !IsGlobal)
    return false;

  if (!Asm.isSymbolLinkerVisible(Symbol) && !Symbol.isUndefined())
    return false;

  if (Symbol.isTemporary())
    return false;

  return true;
}

bool ELFObjectWriter::isLocal(const MCSymbolData &Data, bool isSignature,
                              bool isUsedInReloc) {
  if (Data.isExternal())
    return false;

  const MCSymbol &Symbol = Data.getSymbol();
  const MCSymbol &RefSymbol = Symbol.AliasedSymbol();

  if (RefSymbol.isUndefined() && !RefSymbol.isVariable()) {
    if (isSignature && !isUsedInReloc)
      return true;

    return false;
  }

  return true;
}

void ELFObjectWriter::ComputeIndexMap(MCAssembler &Asm,
                                      SectionIndexMapTy &SectionIndexMap,
                                      const RelMapTy &RelMap) {
  unsigned Index = 1;
  for (MCAssembler::iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it) {
    const MCSectionELF &Section =
      static_cast<const MCSectionELF &>(it->getSection());
    if (Section.getType() != ELF::SHT_GROUP)
      continue;
    SectionIndexMap[&Section] = Index++;
  }

  for (MCAssembler::iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it) {
    const MCSectionELF &Section =
      static_cast<const MCSectionELF &>(it->getSection());
    if (Section.getType() == ELF::SHT_GROUP ||
        Section.getType() == ELF::SHT_REL ||
        Section.getType() == ELF::SHT_RELA)
      continue;
    SectionIndexMap[&Section] = Index++;
    const MCSectionELF *RelSection = RelMap.lookup(&Section);
    if (RelSection)
      SectionIndexMap[RelSection] = Index++;
  }
}

void ELFObjectWriter::ComputeSymbolTable(MCAssembler &Asm,
                                      const SectionIndexMapTy &SectionIndexMap,
                                         RevGroupMapTy RevGroupMap,
                                         unsigned NumRegularSections) {
  // FIXME: Is this the correct place to do this?
  // FIXME: Why is an undefined reference to _GLOBAL_OFFSET_TABLE_ needed?
  if (NeedsGOT) {
    llvm::StringRef Name = "_GLOBAL_OFFSET_TABLE_";
    MCSymbol *Sym = Asm.getContext().GetOrCreateSymbol(Name);
    MCSymbolData &Data = Asm.getOrCreateSymbolData(*Sym);
    Data.setExternal(true);
    MCELF::SetBinding(Data, ELF::STB_GLOBAL);
  }

  // Index 0 is always the empty string.
  StringMap<uint64_t> StringIndexMap;
  StringTable += '\x00';

  // FIXME: We could optimize suffixes in strtab in the same way we
  // optimize them in shstrtab.

  // Add the data for the symbols.
  for (MCAssembler::symbol_iterator it = Asm.symbol_begin(),
         ie = Asm.symbol_end(); it != ie; ++it) {
    const MCSymbol &Symbol = it->getSymbol();

    bool Used = UsedInReloc.count(&Symbol);
    bool WeakrefUsed = WeakrefUsedInReloc.count(&Symbol);
    bool isSignature = RevGroupMap.count(&Symbol);

    if (!isInSymtab(Asm, *it,
                    Used || WeakrefUsed || isSignature,
                    Renames.count(&Symbol)))
      continue;

    ELFSymbolData MSD;
    MSD.SymbolData = it;
    const MCSymbol &RefSymbol = Symbol.AliasedSymbol();

    // Undefined symbols are global, but this is the first place we
    // are able to set it.
    bool Local = isLocal(*it, isSignature, Used);
    if (!Local && MCELF::GetBinding(*it) == ELF::STB_LOCAL) {
      MCSymbolData &SD = Asm.getSymbolData(RefSymbol);
      MCELF::SetBinding(*it, ELF::STB_GLOBAL);
      MCELF::SetBinding(SD, ELF::STB_GLOBAL);
    }

    if (RefSymbol.isUndefined() && !Used && WeakrefUsed)
      MCELF::SetBinding(*it, ELF::STB_WEAK);

    if (it->isCommon()) {
      assert(!Local);
      MSD.SectionIndex = ELF::SHN_COMMON;
    } else if (Symbol.isAbsolute() || RefSymbol.isVariable()) {
      MSD.SectionIndex = ELF::SHN_ABS;
    } else if (RefSymbol.isUndefined()) {
      if (isSignature && !Used)
        MSD.SectionIndex = SectionIndexMap.lookup(RevGroupMap[&Symbol]);
      else
        MSD.SectionIndex = ELF::SHN_UNDEF;
    } else {
      const MCSectionELF &Section =
        static_cast<const MCSectionELF&>(RefSymbol.getSection());
      MSD.SectionIndex = SectionIndexMap.lookup(&Section);
      if (MSD.SectionIndex >= ELF::SHN_LORESERVE)
        NeedsSymtabShndx = true;
      assert(MSD.SectionIndex && "Invalid section index!");
    }

    // The @@@ in symbol version is replaced with @ in undefined symbols and
    // @@ in defined ones.
    StringRef Name = Symbol.getName();
    SmallString<32> Buf;

    size_t Pos = Name.find("@@@");
    if (Pos != StringRef::npos) {
      Buf += Name.substr(0, Pos);
      unsigned Skip = MSD.SectionIndex == ELF::SHN_UNDEF ? 2 : 1;
      Buf += Name.substr(Pos + Skip);
      Name = Buf;
    }

    uint64_t &Entry = StringIndexMap[Name];
    if (!Entry) {
      Entry = StringTable.size();
      StringTable += Name;
      StringTable += '\x00';
    }
    MSD.StringIndex = Entry;
    if (MSD.SectionIndex == ELF::SHN_UNDEF)
      UndefinedSymbolData.push_back(MSD);
    else if (Local)
      LocalSymbolData.push_back(MSD);
    else
      ExternalSymbolData.push_back(MSD);
  }

  // Symbols are required to be in lexicographic order.
  array_pod_sort(LocalSymbolData.begin(), LocalSymbolData.end());
  array_pod_sort(ExternalSymbolData.begin(), ExternalSymbolData.end());
  array_pod_sort(UndefinedSymbolData.begin(), UndefinedSymbolData.end());

  // Set the symbol indices. Local symbols must come before all other
  // symbols with non-local bindings.
  unsigned Index = 1;
  for (unsigned i = 0, e = LocalSymbolData.size(); i != e; ++i)
    LocalSymbolData[i].SymbolData->setIndex(Index++);

  Index += NumRegularSections;

  for (unsigned i = 0, e = ExternalSymbolData.size(); i != e; ++i)
    ExternalSymbolData[i].SymbolData->setIndex(Index++);
  for (unsigned i = 0, e = UndefinedSymbolData.size(); i != e; ++i)
    UndefinedSymbolData[i].SymbolData->setIndex(Index++);

  if (NumRegularSections > ELF::SHN_LORESERVE)
    NeedsSymtabShndx = true;
}

void ELFObjectWriter::CreateRelocationSections(MCAssembler &Asm,
                                               MCAsmLayout &Layout,
                                               RelMapTy &RelMap) {
  for (MCAssembler::const_iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it) {
    const MCSectionData &SD = *it;
    if (Relocations[&SD].empty())
      continue;

    MCContext &Ctx = Asm.getContext();
    const MCSectionELF &Section =
      static_cast<const MCSectionELF&>(SD.getSection());

    const StringRef SectionName = Section.getSectionName();
    std::string RelaSectionName = hasRelocationAddend() ? ".rela" : ".rel";
    RelaSectionName += SectionName;

    unsigned EntrySize;
    if (hasRelocationAddend())
      EntrySize = is64Bit() ? sizeof(ELF::Elf64_Rela) : sizeof(ELF::Elf32_Rela);
    else
      EntrySize = is64Bit() ? sizeof(ELF::Elf64_Rel) : sizeof(ELF::Elf32_Rel);

    const MCSectionELF *RelaSection =
      Ctx.getELFSection(RelaSectionName, hasRelocationAddend() ?
                        ELF::SHT_RELA : ELF::SHT_REL, 0,
                        SectionKind::getReadOnly(),
                        EntrySize, "");
    RelMap[&Section] = RelaSection;
    Asm.getOrCreateSectionData(*RelaSection);
  }
}

void ELFObjectWriter::WriteRelocations(MCAssembler &Asm, MCAsmLayout &Layout,
                                       const RelMapTy &RelMap) {
  for (MCAssembler::const_iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it) {
    const MCSectionData &SD = *it;
    const MCSectionELF &Section =
      static_cast<const MCSectionELF&>(SD.getSection());

    const MCSectionELF *RelaSection = RelMap.lookup(&Section);
    if (!RelaSection)
      continue;
    MCSectionData &RelaSD = Asm.getOrCreateSectionData(*RelaSection);
    RelaSD.setAlignment(is64Bit() ? 8 : 4);

    MCDataFragment *F = new MCDataFragment(&RelaSD);
    WriteRelocationsFragment(Asm, F, &*it);
  }
}

void ELFObjectWriter::WriteSecHdrEntry(uint32_t Name, uint32_t Type,
                                       uint64_t Flags, uint64_t Address,
                                       uint64_t Offset, uint64_t Size,
                                       uint32_t Link, uint32_t Info,
                                       uint64_t Alignment,
                                       uint64_t EntrySize) {
  Write32(Name);        // sh_name: index into string table
  Write32(Type);        // sh_type
  WriteWord(Flags);     // sh_flags
  WriteWord(Address);   // sh_addr
  WriteWord(Offset);    // sh_offset
  WriteWord(Size);      // sh_size
  Write32(Link);        // sh_link
  Write32(Info);        // sh_info
  WriteWord(Alignment); // sh_addralign
  WriteWord(EntrySize); // sh_entsize
}

void ELFObjectWriter::WriteRelocationsFragment(const MCAssembler &Asm,
                                               MCDataFragment *F,
                                               const MCSectionData *SD) {
  std::vector<ELFRelocationEntry> &Relocs = Relocations[SD];

  // Sort the relocation entries. Most targets just sort by r_offset, but some
  // (e.g., MIPS) have additional constraints.
  TargetObjectWriter->sortRelocs(Asm, Relocs);

  for (unsigned i = 0, e = Relocs.size(); i != e; ++i) {
    ELFRelocationEntry entry = Relocs[e - i - 1];

    if (!entry.Index)
      ;
    else if (entry.Index < 0)
      entry.Index = getSymbolIndexInSymbolTable(Asm, entry.Symbol);
    else
      entry.Index += LocalSymbolData.size();
    if (is64Bit()) {
      String64(*F, entry.r_offset);
      if (TargetObjectWriter->isN64()) {
        String32(*F, entry.Index);

        String8(*F, TargetObjectWriter->getRSsym(entry.Type));
        String8(*F, TargetObjectWriter->getRType3(entry.Type));
        String8(*F, TargetObjectWriter->getRType2(entry.Type));
        String8(*F, TargetObjectWriter->getRType(entry.Type));
      }
      else {
        struct ELF::Elf64_Rela ERE64;
        ERE64.setSymbolAndType(entry.Index, entry.Type);
        String64(*F, ERE64.r_info);
      }
      if (hasRelocationAddend())
        String64(*F, entry.r_addend);
    } else {
      String32(*F, entry.r_offset);

      struct ELF::Elf32_Rela ERE32;
      ERE32.setSymbolAndType(entry.Index, entry.Type);
      String32(*F, ERE32.r_info);

      if (hasRelocationAddend())
        String32(*F, entry.r_addend);
    }
  }
}

static int compareBySuffix(const void *a, const void *b) {
  const MCSectionELF *secA = *static_cast<const MCSectionELF* const *>(a);
  const MCSectionELF *secB = *static_cast<const MCSectionELF* const *>(b);
  const StringRef &NameA = secA->getSectionName();
  const StringRef &NameB = secB->getSectionName();
  const unsigned sizeA = NameA.size();
  const unsigned sizeB = NameB.size();
  const unsigned len = std::min(sizeA, sizeB);
  for (unsigned int i = 0; i < len; ++i) {
    char ca = NameA[sizeA - i - 1];
    char cb = NameB[sizeB - i - 1];
    if (ca != cb)
      return cb - ca;
  }

  return sizeB - sizeA;
}

void ELFObjectWriter::CreateMetadataSections(MCAssembler &Asm,
                                             MCAsmLayout &Layout,
                                             SectionIndexMapTy &SectionIndexMap,
                                             const RelMapTy &RelMap) {
  MCContext &Ctx = Asm.getContext();
  MCDataFragment *F;

  unsigned EntrySize = is64Bit() ? ELF::SYMENTRY_SIZE64 : ELF::SYMENTRY_SIZE32;

  // We construct .shstrtab, .symtab and .strtab in this order to match gnu as.
  const MCSectionELF *ShstrtabSection =
    Ctx.getELFSection(".shstrtab", ELF::SHT_STRTAB, 0,
                      SectionKind::getReadOnly());
  MCSectionData &ShstrtabSD = Asm.getOrCreateSectionData(*ShstrtabSection);
  ShstrtabSD.setAlignment(1);

  const MCSectionELF *SymtabSection =
    Ctx.getELFSection(".symtab", ELF::SHT_SYMTAB, 0,
                      SectionKind::getReadOnly(),
                      EntrySize, "");
  MCSectionData &SymtabSD = Asm.getOrCreateSectionData(*SymtabSection);
  SymtabSD.setAlignment(is64Bit() ? 8 : 4);

  MCSectionData *SymtabShndxSD = NULL;

  if (NeedsSymtabShndx) {
    const MCSectionELF *SymtabShndxSection =
      Ctx.getELFSection(".symtab_shndx", ELF::SHT_SYMTAB_SHNDX, 0,
                        SectionKind::getReadOnly(), 4, "");
    SymtabShndxSD = &Asm.getOrCreateSectionData(*SymtabShndxSection);
    SymtabShndxSD->setAlignment(4);
  }

  const MCSectionELF *StrtabSection;
  StrtabSection = Ctx.getELFSection(".strtab", ELF::SHT_STRTAB, 0,
                                    SectionKind::getReadOnly());
  MCSectionData &StrtabSD = Asm.getOrCreateSectionData(*StrtabSection);
  StrtabSD.setAlignment(1);

  ComputeIndexMap(Asm, SectionIndexMap, RelMap);

  ShstrtabIndex = SectionIndexMap.lookup(ShstrtabSection);
  SymbolTableIndex = SectionIndexMap.lookup(SymtabSection);
  StringTableIndex = SectionIndexMap.lookup(StrtabSection);

  // Symbol table
  F = new MCDataFragment(&SymtabSD);
  MCDataFragment *ShndxF = NULL;
  if (NeedsSymtabShndx) {
    ShndxF = new MCDataFragment(SymtabShndxSD);
  }
  WriteSymbolTable(F, ShndxF, Asm, Layout, SectionIndexMap);

  F = new MCDataFragment(&StrtabSD);
  F->getContents().append(StringTable.begin(), StringTable.end());

  F = new MCDataFragment(&ShstrtabSD);

  std::vector<const MCSectionELF*> Sections;
  for (MCAssembler::const_iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it) {
    const MCSectionELF &Section =
      static_cast<const MCSectionELF&>(it->getSection());
    Sections.push_back(&Section);
  }
  array_pod_sort(Sections.begin(), Sections.end(), compareBySuffix);

  // Section header string table.
  //
  // The first entry of a string table holds a null character so skip
  // section 0.
  uint64_t Index = 1;
  F->getContents() += '\x00';

  for (unsigned int I = 0, E = Sections.size(); I != E; ++I) {
    const MCSectionELF &Section = *Sections[I];

    StringRef Name = Section.getSectionName();
    if (I != 0) {
      StringRef PreviousName = Sections[I - 1]->getSectionName();
      if (PreviousName.endswith(Name)) {
        SectionStringTableIndex[&Section] = Index - Name.size() - 1;
        continue;
      }
    }
    // Remember the index into the string table so we can write it
    // into the sh_name field of the section header table.
    SectionStringTableIndex[&Section] = Index;

    Index += Name.size() + 1;
    F->getContents() += Name;
    F->getContents() += '\x00';
  }
}

void ELFObjectWriter::CreateIndexedSections(MCAssembler &Asm,
                                            MCAsmLayout &Layout,
                                            GroupMapTy &GroupMap,
                                            RevGroupMapTy &RevGroupMap,
                                            SectionIndexMapTy &SectionIndexMap,
                                            const RelMapTy &RelMap) {
  // Create the .note.GNU-stack section if needed.
  MCContext &Ctx = Asm.getContext();
  if (Asm.getNoExecStack()) {
    const MCSectionELF *GnuStackSection =
      Ctx.getELFSection(".note.GNU-stack", ELF::SHT_PROGBITS, 0,
                        SectionKind::getReadOnly());
    Asm.getOrCreateSectionData(*GnuStackSection);
  }

  // Build the groups
  for (MCAssembler::const_iterator it = Asm.begin(), ie = Asm.end();
       it != ie; ++it) {
    const MCSectionELF &Section =
      static_cast<const MCSectionELF&>(it->getSection());
    if (!(Section.getFlags() & ELF::SHF_GROUP))
      continue;

    const MCSymbol *SignatureSymbol = Section.getGroup();
    Asm.getOrCreateSymbolData(*SignatureSymbol);
    const MCSectionELF *&Group = RevGroupMap[SignatureSymbol];
    if (!Group) {
      Group = Ctx.CreateELFGroupSection();
      MCSectionData &Data = Asm.getOrCreateSectionData(*Group);
      Data.setAlignment(4);
      MCDataFragment *F = new MCDataFragment(&Data);
      String32(*F, ELF::GRP_COMDAT);
    }
    GroupMap[Group] = SignatureSymbol;
  }

  ComputeIndexMap(Asm, SectionIndexMap, RelMap);

  // Add sections to the groups
  for (MCAssembler::const_iterator it = Asm.begin(), ie = Asm.end();
       it != ie; ++it) {
    const MCSectionELF &Section =
      static_cast<const MCSectionELF&>(it->getSection());
    if (!(Section.getFlags() & ELF::SHF_GROUP))
      continue;
    const MCSectionELF *Group = RevGroupMap[Section.getGroup()];
    MCSectionData &Data = Asm.getOrCreateSectionData(*Group);
    // FIXME: we could use the previous fragment
    MCDataFragment *F = new MCDataFragment(&Data);
    unsigned Index = SectionIndexMap.lookup(&Section);
    String32(*F, Index);
  }
}

void ELFObjectWriter::WriteSection(MCAssembler &Asm,
                                   const SectionIndexMapTy &SectionIndexMap,
                                   uint32_t GroupSymbolIndex,
                                   uint64_t Offset, uint64_t Size,
                                   uint64_t Alignment,
                                   const MCSectionELF &Section) {
  uint64_t sh_link = 0;
  uint64_t sh_info = 0;

  switch(Section.getType()) {
  case ELF::SHT_DYNAMIC:
    sh_link = SectionStringTableIndex[&Section];
    sh_info = 0;
    break;

  case ELF::SHT_REL:
  case ELF::SHT_RELA: {
    const MCSectionELF *SymtabSection;
    const MCSectionELF *InfoSection;
    SymtabSection = Asm.getContext().getELFSection(".symtab", ELF::SHT_SYMTAB,
                                                   0,
                                                   SectionKind::getReadOnly());
    sh_link = SectionIndexMap.lookup(SymtabSection);
    assert(sh_link && ".symtab not found");

    // Remove ".rel" and ".rela" prefixes.
    unsigned SecNameLen = (Section.getType() == ELF::SHT_REL) ? 4 : 5;
    StringRef SectionName = Section.getSectionName().substr(SecNameLen);

    InfoSection = Asm.getContext().getELFSection(SectionName,
                                                 ELF::SHT_PROGBITS, 0,
                                                 SectionKind::getReadOnly());
    sh_info = SectionIndexMap.lookup(InfoSection);
    break;
  }

  case ELF::SHT_SYMTAB:
  case ELF::SHT_DYNSYM:
    sh_link = StringTableIndex;
    sh_info = LastLocalSymbolIndex;
    break;

  case ELF::SHT_SYMTAB_SHNDX:
    sh_link = SymbolTableIndex;
    break;

  case ELF::SHT_PROGBITS:
  case ELF::SHT_STRTAB:
  case ELF::SHT_NOBITS:
  case ELF::SHT_NOTE:
  case ELF::SHT_NULL:
  case ELF::SHT_ARM_ATTRIBUTES:
  case ELF::SHT_INIT_ARRAY:
  case ELF::SHT_FINI_ARRAY:
  case ELF::SHT_PREINIT_ARRAY:
  case ELF::SHT_X86_64_UNWIND:
    // Nothing to do.
    break;

  case ELF::SHT_GROUP:
    sh_link = SymbolTableIndex;
    sh_info = GroupSymbolIndex;
    break;

  default:
    assert(0 && "FIXME: sh_type value not supported!");
    break;
  }

  WriteSecHdrEntry(SectionStringTableIndex[&Section], Section.getType(),
                   Section.getFlags(), 0, Offset, Size, sh_link, sh_info,
                   Alignment, Section.getEntrySize());
}

bool ELFObjectWriter::IsELFMetaDataSection(const MCSectionData &SD) {
  return SD.getOrdinal() == ~UINT32_C(0) &&
    !SD.getSection().isVirtualSection();
}

uint64_t ELFObjectWriter::DataSectionSize(const MCSectionData &SD) {
  uint64_t Ret = 0;
  for (MCSectionData::const_iterator i = SD.begin(), e = SD.end(); i != e;
       ++i) {
    const MCFragment &F = *i;
    assert(F.getKind() == MCFragment::FT_Data);
    Ret += cast<MCDataFragment>(F).getContents().size();
  }
  return Ret;
}

uint64_t ELFObjectWriter::GetSectionFileSize(const MCAsmLayout &Layout,
                                             const MCSectionData &SD) {
  if (IsELFMetaDataSection(SD))
    return DataSectionSize(SD);
  return Layout.getSectionFileSize(&SD);
}

uint64_t ELFObjectWriter::GetSectionAddressSize(const MCAsmLayout &Layout,
                                                const MCSectionData &SD) {
  if (IsELFMetaDataSection(SD))
    return DataSectionSize(SD);
  return Layout.getSectionAddressSize(&SD);
}

void ELFObjectWriter::WriteDataSectionData(MCAssembler &Asm,
                                           const MCAsmLayout &Layout,
                                           const MCSectionELF &Section) {
  const MCSectionData &SD = Asm.getOrCreateSectionData(Section);

  uint64_t Padding = OffsetToAlignment(OS.tell(), SD.getAlignment());
  WriteZeros(Padding);

  if (IsELFMetaDataSection(SD)) {
    for (MCSectionData::const_iterator i = SD.begin(), e = SD.end(); i != e;
         ++i) {
      const MCFragment &F = *i;
      assert(F.getKind() == MCFragment::FT_Data);
      WriteBytes(cast<MCDataFragment>(F).getContents().str());
    }
  } else {
    Asm.writeSectionData(&SD, Layout);
  }
}

void ELFObjectWriter::WriteSectionHeader(MCAssembler &Asm,
                                         const GroupMapTy &GroupMap,
                                         const MCAsmLayout &Layout,
                                      const SectionIndexMapTy &SectionIndexMap,
                                   const SectionOffsetMapTy &SectionOffsetMap) {
  const unsigned NumSections = Asm.size() + 1;

  std::vector<const MCSectionELF*> Sections;
  Sections.resize(NumSections - 1);

  for (SectionIndexMapTy::const_iterator i=
         SectionIndexMap.begin(), e = SectionIndexMap.end(); i != e; ++i) {
    const std::pair<const MCSectionELF*, uint32_t> &p = *i;
    Sections[p.second - 1] = p.first;
  }

  // Null section first.
  uint64_t FirstSectionSize =
    NumSections >= ELF::SHN_LORESERVE ? NumSections : 0;
  uint32_t FirstSectionLink =
    ShstrtabIndex >= ELF::SHN_LORESERVE ? ShstrtabIndex : 0;
  WriteSecHdrEntry(0, 0, 0, 0, 0, FirstSectionSize, FirstSectionLink, 0, 0, 0);

  for (unsigned i = 0; i < NumSections - 1; ++i) {
    const MCSectionELF &Section = *Sections[i];
    const MCSectionData &SD = Asm.getOrCreateSectionData(Section);
    uint32_t GroupSymbolIndex;
    if (Section.getType() != ELF::SHT_GROUP)
      GroupSymbolIndex = 0;
    else
      GroupSymbolIndex = getSymbolIndexInSymbolTable(Asm,
                                                     GroupMap.lookup(&Section));

    uint64_t Size = GetSectionAddressSize(Layout, SD);

    WriteSection(Asm, SectionIndexMap, GroupSymbolIndex,
                 SectionOffsetMap.lookup(&Section), Size,
                 SD.getAlignment(), Section);
  }
}

void ELFObjectWriter::ComputeSectionOrder(MCAssembler &Asm,
                                  std::vector<const MCSectionELF*> &Sections) {
  for (MCAssembler::iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it) {
    const MCSectionELF &Section =
      static_cast<const MCSectionELF &>(it->getSection());
    if (Section.getType() == ELF::SHT_GROUP)
      Sections.push_back(&Section);
  }

  for (MCAssembler::iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it) {
    const MCSectionELF &Section =
      static_cast<const MCSectionELF &>(it->getSection());
    if (Section.getType() != ELF::SHT_GROUP &&
        Section.getType() != ELF::SHT_REL &&
        Section.getType() != ELF::SHT_RELA)
      Sections.push_back(&Section);
  }

  for (MCAssembler::iterator it = Asm.begin(),
         ie = Asm.end(); it != ie; ++it) {
    const MCSectionELF &Section =
      static_cast<const MCSectionELF &>(it->getSection());
    if (Section.getType() == ELF::SHT_REL ||
        Section.getType() == ELF::SHT_RELA)
      Sections.push_back(&Section);
  }
}

void ELFObjectWriter::WriteObject(MCAssembler &Asm,
                                  const MCAsmLayout &Layout) {
  GroupMapTy GroupMap;
  RevGroupMapTy RevGroupMap;
  SectionIndexMapTy SectionIndexMap;

  unsigned NumUserSections = Asm.size();

  DenseMap<const MCSectionELF*, const MCSectionELF*> RelMap;
  CreateRelocationSections(Asm, const_cast<MCAsmLayout&>(Layout), RelMap);

  const unsigned NumUserAndRelocSections = Asm.size();
  CreateIndexedSections(Asm, const_cast<MCAsmLayout&>(Layout), GroupMap,
                        RevGroupMap, SectionIndexMap, RelMap);
  const unsigned AllSections = Asm.size();
  const unsigned NumIndexedSections = AllSections - NumUserAndRelocSections;

  unsigned NumRegularSections = NumUserSections + NumIndexedSections;

  // Compute symbol table information.
  ComputeSymbolTable(Asm, SectionIndexMap, RevGroupMap, NumRegularSections);


  WriteRelocations(Asm, const_cast<MCAsmLayout&>(Layout), RelMap);

  CreateMetadataSections(const_cast<MCAssembler&>(Asm),
                         const_cast<MCAsmLayout&>(Layout),
                         SectionIndexMap,
                         RelMap);

  uint64_t NaturalAlignment = is64Bit() ? 8 : 4;
  uint64_t HeaderSize = is64Bit() ? sizeof(ELF::Elf64_Ehdr) :
                                    sizeof(ELF::Elf32_Ehdr);
  uint64_t FileOff = HeaderSize;

  std::vector<const MCSectionELF*> Sections;
  ComputeSectionOrder(Asm, Sections);
  unsigned NumSections = Sections.size();
  SectionOffsetMapTy SectionOffsetMap;
  for (unsigned i = 0; i < NumRegularSections + 1; ++i) {
    const MCSectionELF &Section = *Sections[i];
    const MCSectionData &SD = Asm.getOrCreateSectionData(Section);

    FileOff = RoundUpToAlignment(FileOff, SD.getAlignment());

    // Remember the offset into the file for this section.
    SectionOffsetMap[&Section] = FileOff;

    // Get the size of the section in the output file (including padding).
    FileOff += GetSectionFileSize(Layout, SD);
  }

  FileOff = RoundUpToAlignment(FileOff, NaturalAlignment);

  const unsigned SectionHeaderOffset = FileOff - HeaderSize;

  uint64_t SectionHeaderEntrySize = is64Bit() ?
    sizeof(ELF::Elf64_Shdr) : sizeof(ELF::Elf32_Shdr);
  FileOff += (NumSections + 1) * SectionHeaderEntrySize;

  for (unsigned i = NumRegularSections + 1; i < NumSections; ++i) {
    const MCSectionELF &Section = *Sections[i];
    const MCSectionData &SD = Asm.getOrCreateSectionData(Section);

    FileOff = RoundUpToAlignment(FileOff, SD.getAlignment());

    // Remember the offset into the file for this section.
    SectionOffsetMap[&Section] = FileOff;

    // Get the size of the section in the output file (including padding).
    FileOff += GetSectionFileSize(Layout, SD);
  }

  // Write out the ELF header ...
  WriteHeader(SectionHeaderOffset, NumSections + 1);

  // ... then the regular sections ...
  // + because of .shstrtab
  for (unsigned i = 0; i < NumRegularSections + 1; ++i)
    WriteDataSectionData(Asm, Layout, *Sections[i]);

  uint64_t Padding = OffsetToAlignment(OS.tell(), NaturalAlignment);
  WriteZeros(Padding);

  // ... then the section header table ...
  WriteSectionHeader(Asm, GroupMap, Layout, SectionIndexMap,
                     SectionOffsetMap);

  // ... and then the remaining sections ...
  for (unsigned i = NumRegularSections + 1; i < NumSections; ++i)
    WriteDataSectionData(Asm, Layout, *Sections[i]);
}

bool
ELFObjectWriter::IsSymbolRefDifferenceFullyResolvedImpl(const MCAssembler &Asm,
                                                      const MCSymbolData &DataA,
                                                      const MCFragment &FB,
                                                      bool InSet,
                                                      bool IsPCRel) const {
  if (DataA.getFlags() & ELF_STB_Weak)
    return false;
  return MCObjectWriter::IsSymbolRefDifferenceFullyResolvedImpl(
                                                 Asm, DataA, FB,InSet, IsPCRel);
}

MCObjectWriter *llvm::createELFObjectWriter(MCELFObjectTargetWriter *MOTW,
                                            raw_ostream &OS,
                                            bool IsLittleEndian) {
  return new ELFObjectWriter(MOTW, OS, IsLittleEndian);
}
