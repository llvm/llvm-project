//===- lib/MC/ELFObjectWriter.cpp - ELF File Writer -----------------------===//
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

#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELF.h"
#include "llvm/MC/MCELFSymbolFlags.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include <vector>
using namespace llvm;

#undef  DEBUG_TYPE
#define DEBUG_TYPE "reloc-info"

namespace {
class FragmentWriter {
  bool IsLittleEndian;

public:
  FragmentWriter(bool IsLittleEndian);
  template <typename T> void write(MCDataFragment &F, T Val);
};

typedef DenseMap<const MCSectionELF *, uint32_t> SectionIndexMapTy;

class SymbolTableWriter {
  MCAssembler &Asm;
  FragmentWriter &FWriter;
  bool Is64Bit;
  SectionIndexMapTy &SectionIndexMap;

  // The symbol .symtab fragment we are writting to.
  MCDataFragment *SymtabF;

  // .symtab_shndx fragment we are writting to.
  MCDataFragment *ShndxF;

  // The numbel of symbols written so far.
  unsigned NumWritten;

  void createSymtabShndx();

  template <typename T> void write(MCDataFragment &F, T Value);

public:
  SymbolTableWriter(MCAssembler &Asm, FragmentWriter &FWriter, bool Is64Bit,
                    SectionIndexMapTy &SectionIndexMap,
                    MCDataFragment *SymtabF);

  void writeSymbol(uint32_t name, uint8_t info, uint64_t value, uint64_t size,
                   uint8_t other, uint32_t shndx, bool Reserved);
};

struct ELFRelocationEntry {
  uint64_t Offset; // Where is the relocation.
  const MCSymbol *Symbol;       // The symbol to relocate with.
  unsigned Type;   // The type of the relocation.
  uint64_t Addend; // The addend to use.

  ELFRelocationEntry(uint64_t Offset, const MCSymbol *Symbol, unsigned Type,
                     uint64_t Addend)
      : Offset(Offset), Symbol(Symbol), Type(Type), Addend(Addend) {}
};

class ELFObjectWriter : public MCObjectWriter {
  FragmentWriter FWriter;

  protected:

    static bool isFixupKindPCRel(const MCAssembler &Asm, unsigned Kind);
    static bool RelocNeedsGOT(MCSymbolRefExpr::VariantKind Variant);
    static uint64_t SymbolValue(MCSymbolData &Data, const MCAsmLayout &Layout);
    static bool isInSymtab(const MCAsmLayout &Layout, const MCSymbolData &Data,
                           bool Used, bool Renamed);
    static bool isLocal(const MCSymbolData &Data, bool isUsedInReloc);
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
      StringRef Name;

      // Support lexicographic sorting.
      bool operator<(const ELFSymbolData &RHS) const {
        unsigned LHSType = MCELF::GetType(*SymbolData);
        unsigned RHSType = MCELF::GetType(*RHS.SymbolData);
        if (LHSType == ELF::STT_SECTION && RHSType != ELF::STT_SECTION)
          return false;
        if (LHSType != ELF::STT_SECTION && RHSType == ELF::STT_SECTION)
          return true;
        if (LHSType == ELF::STT_SECTION && RHSType == ELF::STT_SECTION)
          return SectionIndex < RHS.SectionIndex;
        return Name < RHS.Name;
      }
    };

    /// The target specific ELF writer instance.
    std::unique_ptr<MCELFObjectTargetWriter> TargetObjectWriter;

    SmallPtrSet<const MCSymbol *, 16> UsedInReloc;
    SmallPtrSet<const MCSymbol *, 16> WeakrefUsedInReloc;
    DenseMap<const MCSymbol *, const MCSymbol *> Renames;

    llvm::DenseMap<const MCSectionData *, std::vector<ELFRelocationEntry>>
    Relocations;
    StringTableBuilder ShStrTabBuilder;

    /// @}
    /// @name Symbol Table Data
    /// @{

    StringTableBuilder StrTabBuilder;
    std::vector<uint64_t> FileSymbolData;
    std::vector<ELFSymbolData> LocalSymbolData;
    std::vector<ELFSymbolData> ExternalSymbolData;
    std::vector<ELFSymbolData> UndefinedSymbolData;

    /// @}

    bool NeedsGOT;

    // This holds the symbol table index of the last local symbol.
    unsigned LastLocalSymbolIndex;
    // This holds the .strtab section index.
    unsigned StringTableIndex;
    // This holds the .symtab section index.
    unsigned SymbolTableIndex;

    unsigned ShstrtabIndex;


    // TargetObjectWriter wrappers.
    bool is64Bit() const { return TargetObjectWriter->is64Bit(); }
    bool hasRelocationAddend() const {
      return TargetObjectWriter->hasRelocationAddend();
    }
    unsigned GetRelocType(const MCValue &Target, const MCFixup &Fixup,
                          bool IsPCRel) const {
      return TargetObjectWriter->GetRelocType(Target, Fixup, IsPCRel);
    }

  public:
    ELFObjectWriter(MCELFObjectTargetWriter *MOTW, raw_ostream &_OS,
                    bool IsLittleEndian)
        : MCObjectWriter(_OS, IsLittleEndian), FWriter(IsLittleEndian),
          TargetObjectWriter(MOTW), NeedsGOT(false) {}

    virtual ~ELFObjectWriter();

    void WriteWord(uint64_t W) {
      if (is64Bit())
        Write64(W);
      else
        Write32(W);
    }

    template <typename T> void write(MCDataFragment &F, T Value) {
      FWriter.write(F, Value);
    }

    void WriteHeader(const MCAssembler &Asm,
                     uint64_t SectionDataSize,
                     unsigned NumberOfSections);

    void WriteSymbol(SymbolTableWriter &Writer, ELFSymbolData &MSD,
                     const MCAsmLayout &Layout);

    void WriteSymbolTable(MCDataFragment *SymtabF, MCAssembler &Asm,
                          const MCAsmLayout &Layout,
                          SectionIndexMapTy &SectionIndexMap);

    bool shouldRelocateWithSymbol(const MCAssembler &Asm,
                                  const MCSymbolRefExpr *RefA,
                                  const MCSymbolData *SD, uint64_t C,
                                  unsigned Type) const;

    void RecordRelocation(const MCAssembler &Asm, const MCAsmLayout &Layout,
                          const MCFragment *Fragment, const MCFixup &Fixup,
                          MCValue Target, bool &IsPCRel,
                          uint64_t &FixedValue) override;

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

    /// Compute the symbol table data
    ///
    /// \param Asm - The assembler.
    /// \param SectionIndexMap - Maps a section to its index.
    /// \param RevGroupMap - Maps a signature symbol to the group section.
    /// \param NumRegularSections - Number of non-relocation sections.
    void computeSymbolTable(MCAssembler &Asm, const MCAsmLayout &Layout,
                            const SectionIndexMapTy &SectionIndexMap,
                            const RevGroupMapTy &RevGroupMap,
                            unsigned NumRegularSections);

    void ComputeIndexMap(MCAssembler &Asm,
                         SectionIndexMapTy &SectionIndexMap,
                         const RelMapTy &RelMap);

    void CreateRelocationSections(MCAssembler &Asm, MCAsmLayout &Layout,
                                  RelMapTy &RelMap);

    void CompressDebugSections(MCAssembler &Asm, MCAsmLayout &Layout);

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

    void ExecutePostLayoutBinding(MCAssembler &Asm,
                                  const MCAsmLayout &Layout) override;

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

    bool
    IsSymbolRefDifferenceFullyResolvedImpl(const MCAssembler &Asm,
                                           const MCSymbolData &DataA,
                                           const MCFragment &FB,
                                           bool InSet,
                                           bool IsPCRel) const override;

    void WriteObject(MCAssembler &Asm, const MCAsmLayout &Layout) override;
    void WriteSection(MCAssembler &Asm,
                      const SectionIndexMapTy &SectionIndexMap,
                      uint32_t GroupSymbolIndex,
                      uint64_t Offset, uint64_t Size, uint64_t Alignment,
                      const MCSectionELF &Section);
  };
}

FragmentWriter::FragmentWriter(bool IsLittleEndian)
    : IsLittleEndian(IsLittleEndian) {}

template <typename T> void FragmentWriter::write(MCDataFragment &F, T Val) {
  if (IsLittleEndian)
    Val = support::endian::byte_swap<T, support::little>(Val);
  else
    Val = support::endian::byte_swap<T, support::big>(Val);
  const char *Start = (const char *)&Val;
  F.getContents().append(Start, Start + sizeof(T));
}

void SymbolTableWriter::createSymtabShndx() {
  if (ShndxF)
    return;

  MCContext &Ctx = Asm.getContext();
  const MCSectionELF *SymtabShndxSection =
      Ctx.getELFSection(".symtab_shndxr", ELF::SHT_SYMTAB_SHNDX, 0,
                        SectionKind::getReadOnly(), 4, "");
  MCSectionData *SymtabShndxSD =
      &Asm.getOrCreateSectionData(*SymtabShndxSection);
  SymtabShndxSD->setAlignment(4);
  ShndxF = new MCDataFragment(SymtabShndxSD);
  unsigned Index = SectionIndexMap.size() + 1;
  SectionIndexMap[SymtabShndxSection] = Index;

  for (unsigned I = 0; I < NumWritten; ++I)
    write(*ShndxF, uint32_t(0));
}

template <typename T>
void SymbolTableWriter::write(MCDataFragment &F, T Value) {
  FWriter.write(F, Value);
}

SymbolTableWriter::SymbolTableWriter(MCAssembler &Asm, FragmentWriter &FWriter,
                                     bool Is64Bit,
                                     SectionIndexMapTy &SectionIndexMap,
                                     MCDataFragment *SymtabF)
    : Asm(Asm), FWriter(FWriter), Is64Bit(Is64Bit),
      SectionIndexMap(SectionIndexMap), SymtabF(SymtabF), ShndxF(nullptr),
      NumWritten(0) {}

void SymbolTableWriter::writeSymbol(uint32_t name, uint8_t info, uint64_t value,
                                    uint64_t size, uint8_t other,
                                    uint32_t shndx, bool Reserved) {
  bool LargeIndex = shndx >= ELF::SHN_LORESERVE && !Reserved;

  if (LargeIndex)
    createSymtabShndx();

  if (ShndxF) {
    if (LargeIndex)
      write(*ShndxF, shndx);
    else
      write(*ShndxF, uint32_t(0));
  }

  uint16_t Index = LargeIndex ? uint16_t(ELF::SHN_XINDEX) : shndx;

  raw_svector_ostream OS(SymtabF->getContents());

  if (Is64Bit) {
    write(*SymtabF, name);  // st_name
    write(*SymtabF, info);  // st_info
    write(*SymtabF, other); // st_other
    write(*SymtabF, Index); // st_shndx
    write(*SymtabF, value); // st_value
    write(*SymtabF, size);  // st_size
  } else {
    write(*SymtabF, name);            // st_name
    write(*SymtabF, uint32_t(value)); // st_value
    write(*SymtabF, uint32_t(size));  // st_size
    write(*SymtabF, info);            // st_info
    write(*SymtabF, other);           // st_other
    write(*SymtabF, Index);           // st_shndx
  }

  ++NumWritten;
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
void ELFObjectWriter::WriteHeader(const MCAssembler &Asm,
                                  uint64_t SectionDataSize,
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
  Write32(Asm.getELFHeaderEFlags());

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

uint64_t ELFObjectWriter::SymbolValue(MCSymbolData &Data,
                                      const MCAsmLayout &Layout) {
  if (Data.isCommon() && Data.isExternal())
    return Data.getCommonAlignment();

  uint64_t Res;
  if (!Layout.getSymbolOffset(&Data, Res))
    return 0;

  if (Layout.getAssembler().isThumbFunc(&Data.getSymbol()))
    Res |= 1;

  return Res;
}

void ELFObjectWriter::ExecutePostLayoutBinding(MCAssembler &Asm,
                                               const MCAsmLayout &Layout) {
  // The presence of symbol versions causes undefined symbols and
  // versions declared with @@@ to be renamed.

  for (MCSymbolData &OriginalData : Asm.symbols()) {
    const MCSymbol &Alias = OriginalData.getSymbol();

    // Not an alias.
    if (!Alias.isVariable())
      continue;
    auto *Ref = dyn_cast<MCSymbolRefExpr>(Alias.getVariableValue());
    if (!Ref)
      continue;
    const MCSymbol &Symbol = Ref->getSymbol();
    MCSymbolData &SD = Asm.getSymbolData(Symbol);

    StringRef AliasName = Alias.getName();
    size_t Pos = AliasName.find('@');
    if (Pos == StringRef::npos)
      continue;

    // Aliases defined with .symvar copy the binding from the symbol they alias.
    // This is the first place we are able to copy this information.
    OriginalData.setExternal(SD.isExternal());
    MCELF::SetBinding(OriginalData, MCELF::GetBinding(SD));

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

static uint8_t mergeTypeForSet(uint8_t origType, uint8_t newType) {
  uint8_t Type = newType;

  // Propagation rules:
  // IFUNC > FUNC > OBJECT > NOTYPE
  // TLS_OBJECT > OBJECT > NOTYPE
  //
  // dont let the new type degrade the old type
  switch (origType) {
  default:
    break;
  case ELF::STT_GNU_IFUNC:
    if (Type == ELF::STT_FUNC || Type == ELF::STT_OBJECT ||
        Type == ELF::STT_NOTYPE || Type == ELF::STT_TLS)
      Type = ELF::STT_GNU_IFUNC;
    break;
  case ELF::STT_FUNC:
    if (Type == ELF::STT_OBJECT || Type == ELF::STT_NOTYPE ||
        Type == ELF::STT_TLS)
      Type = ELF::STT_FUNC;
    break;
  case ELF::STT_OBJECT:
    if (Type == ELF::STT_NOTYPE)
      Type = ELF::STT_OBJECT;
    break;
  case ELF::STT_TLS:
    if (Type == ELF::STT_OBJECT || Type == ELF::STT_NOTYPE ||
        Type == ELF::STT_GNU_IFUNC || Type == ELF::STT_FUNC)
      Type = ELF::STT_TLS;
    break;
  }

  return Type;
}

void ELFObjectWriter::WriteSymbol(SymbolTableWriter &Writer, ELFSymbolData &MSD,
                                  const MCAsmLayout &Layout) {
  MCSymbolData &OrigData = *MSD.SymbolData;
  assert((!OrigData.getFragment() ||
          (&OrigData.getFragment()->getParent()->getSection() ==
           &OrigData.getSymbol().getSection())) &&
         "The symbol's section doesn't match the fragment's symbol");
  const MCSymbol *Base = Layout.getBaseSymbol(OrigData.getSymbol());

  // This has to be in sync with when computeSymbolTable uses SHN_ABS or
  // SHN_COMMON.
  bool IsReserved = !Base || OrigData.isCommon();

  // Binding and Type share the same byte as upper and lower nibbles
  uint8_t Binding = MCELF::GetBinding(OrigData);
  uint8_t Type = MCELF::GetType(OrigData);
  MCSymbolData *BaseSD = nullptr;
  if (Base) {
    BaseSD = &Layout.getAssembler().getSymbolData(*Base);
    Type = mergeTypeForSet(Type, MCELF::GetType(*BaseSD));
  }
  uint8_t Info = (Binding << ELF_STB_Shift) | (Type << ELF_STT_Shift);

  // Other and Visibility share the same byte with Visibility using the lower
  // 2 bits
  uint8_t Visibility = MCELF::GetVisibility(OrigData);
  uint8_t Other = MCELF::getOther(OrigData) << (ELF_STO_Shift - ELF_STV_Shift);
  Other |= Visibility;

  uint64_t Value = SymbolValue(OrigData, Layout);
  uint64_t Size = 0;

  const MCExpr *ESize = OrigData.getSize();
  if (!ESize && Base)
    ESize = BaseSD->getSize();

  if (ESize) {
    int64_t Res;
    if (!ESize->EvaluateAsAbsolute(Res, Layout))
      report_fatal_error("Size expression must be absolute.");
    Size = Res;
  }

  // Write out the symbol table entry
  Writer.writeSymbol(MSD.StringIndex, Info, Value, Size, Other,
                     MSD.SectionIndex, IsReserved);
}

void ELFObjectWriter::WriteSymbolTable(MCDataFragment *SymtabF,
                                       MCAssembler &Asm,
                                       const MCAsmLayout &Layout,
                                       SectionIndexMapTy &SectionIndexMap) {
  // The string table must be emitted first because we need the index
  // into the string table for all the symbol names.

  // FIXME: Make sure the start of the symbol table is aligned.

  SymbolTableWriter Writer(Asm, FWriter, is64Bit(), SectionIndexMap, SymtabF);

  // The first entry is the undefined symbol entry.
  Writer.writeSymbol(0, 0, 0, 0, 0, 0, false);

  for (unsigned i = 0, e = FileSymbolData.size(); i != e; ++i) {
    Writer.writeSymbol(FileSymbolData[i], ELF::STT_FILE | ELF::STB_LOCAL, 0, 0,
                       ELF::STV_DEFAULT, ELF::SHN_ABS, true);
  }

  // Write the symbol table entries.
  LastLocalSymbolIndex = FileSymbolData.size() + LocalSymbolData.size() + 1;

  for (unsigned i = 0, e = LocalSymbolData.size(); i != e; ++i) {
    ELFSymbolData &MSD = LocalSymbolData[i];
    WriteSymbol(Writer, MSD, Layout);
  }

  for (unsigned i = 0, e = ExternalSymbolData.size(); i != e; ++i) {
    ELFSymbolData &MSD = ExternalSymbolData[i];
    MCSymbolData &Data = *MSD.SymbolData;
    assert(((Data.getFlags() & ELF_STB_Global) ||
            (Data.getFlags() & ELF_STB_Weak)) &&
           "External symbol requires STB_GLOBAL or STB_WEAK flag");
    WriteSymbol(Writer, MSD, Layout);
    if (MCELF::GetBinding(Data) == ELF::STB_LOCAL)
      LastLocalSymbolIndex++;
  }

  for (unsigned i = 0, e = UndefinedSymbolData.size(); i != e; ++i) {
    ELFSymbolData &MSD = UndefinedSymbolData[i];
    MCSymbolData &Data = *MSD.SymbolData;
    WriteSymbol(Writer, MSD, Layout);
    if (MCELF::GetBinding(Data) == ELF::STB_LOCAL)
      LastLocalSymbolIndex++;
  }
}

// It is always valid to create a relocation with a symbol. It is preferable
// to use a relocation with a section if that is possible. Using the section
// allows us to omit some local symbols from the symbol table.
bool ELFObjectWriter::shouldRelocateWithSymbol(const MCAssembler &Asm,
                                               const MCSymbolRefExpr *RefA,
                                               const MCSymbolData *SD,
                                               uint64_t C,
                                               unsigned Type) const {
  // A PCRel relocation to an absolute value has no symbol (or section). We
  // represent that with a relocation to a null section.
  if (!RefA)
    return false;

  MCSymbolRefExpr::VariantKind Kind = RefA->getKind();
  switch (Kind) {
  default:
    break;
  // The .odp creation emits a relocation against the symbol ".TOC." which
  // create a R_PPC64_TOC relocation. However the relocation symbol name
  // in final object creation should be NULL, since the symbol does not
  // really exist, it is just the reference to TOC base for the current
  // object file. Since the symbol is undefined, returning false results
  // in a relocation with a null section which is the desired result.
  case MCSymbolRefExpr::VK_PPC_TOCBASE:
    return false;

  // These VariantKind cause the relocation to refer to something other than
  // the symbol itself, like a linker generated table. Since the address of
  // symbol is not relevant, we cannot replace the symbol with the
  // section and patch the difference in the addend.
  case MCSymbolRefExpr::VK_GOT:
  case MCSymbolRefExpr::VK_PLT:
  case MCSymbolRefExpr::VK_GOTPCREL:
  case MCSymbolRefExpr::VK_Mips_GOT:
  case MCSymbolRefExpr::VK_PPC_GOT_LO:
  case MCSymbolRefExpr::VK_PPC_GOT_HI:
  case MCSymbolRefExpr::VK_PPC_GOT_HA:
    return true;
  }

  // An undefined symbol is not in any section, so the relocation has to point
  // to the symbol itself.
  const MCSymbol &Sym = SD->getSymbol();
  if (Sym.isUndefined())
    return true;

  unsigned Binding = MCELF::GetBinding(*SD);
  switch(Binding) {
  default:
    llvm_unreachable("Invalid Binding");
  case ELF::STB_LOCAL:
    break;
  case ELF::STB_WEAK:
    // If the symbol is weak, it might be overridden by a symbol in another
    // file. The relocation has to point to the symbol so that the linker
    // can update it.
    return true;
  case ELF::STB_GLOBAL:
    // Global ELF symbols can be preempted by the dynamic linker. The relocation
    // has to point to the symbol for a reason analogous to the STB_WEAK case.
    return true;
  }

  // If a relocation points to a mergeable section, we have to be careful.
  // If the offset is zero, a relocation with the section will encode the
  // same information. With a non-zero offset, the situation is different.
  // For example, a relocation can point 42 bytes past the end of a string.
  // If we change such a relocation to use the section, the linker would think
  // that it pointed to another string and subtracting 42 at runtime will
  // produce the wrong value.
  auto &Sec = cast<MCSectionELF>(Sym.getSection());
  unsigned Flags = Sec.getFlags();
  if (Flags & ELF::SHF_MERGE) {
    if (C != 0)
      return true;

    // It looks like gold has a bug (http://sourceware.org/PR16794) and can
    // only handle section relocations to mergeable sections if using RELA.
    if (!hasRelocationAddend())
      return true;
  }

  // Most TLS relocations use a got, so they need the symbol. Even those that
  // are just an offset (@tpoff), require a symbol in gold versions before
  // 5efeedf61e4fe720fd3e9a08e6c91c10abb66d42 (2014-09-26) which fixed
  // http://sourceware.org/PR16773.
  if (Flags & ELF::SHF_TLS)
    return true;

  // If the symbol is a thumb function the final relocation must set the lowest
  // bit. With a symbol that is done by just having the symbol have that bit
  // set, so we would lose the bit if we relocated with the section.
  // FIXME: We could use the section but add the bit to the relocation value.
  if (Asm.isThumbFunc(&Sym))
    return true;

  if (TargetObjectWriter->needsRelocateWithSymbol(*SD, Type))
    return true;
  return false;
}

static const MCSymbol *getWeakRef(const MCSymbolRefExpr &Ref) {
  const MCSymbol &Sym = Ref.getSymbol();

  if (Ref.getKind() == MCSymbolRefExpr::VK_WEAKREF)
    return &Sym;

  if (!Sym.isVariable())
    return nullptr;

  const MCExpr *Expr = Sym.getVariableValue();
  const auto *Inner = dyn_cast<MCSymbolRefExpr>(Expr);
  if (!Inner)
    return nullptr;

  if (Inner->getKind() == MCSymbolRefExpr::VK_WEAKREF)
    return &Inner->getSymbol();
  return nullptr;
}

void ELFObjectWriter::RecordRelocation(const MCAssembler &Asm,
                                       const MCAsmLayout &Layout,
                                       const MCFragment *Fragment,
                                       const MCFixup &Fixup,
                                       MCValue Target,
                                       bool &IsPCRel,
                                       uint64_t &FixedValue) {
  const MCSectionData *FixupSection = Fragment->getParent();
  uint64_t C = Target.getConstant();
  uint64_t FixupOffset = Layout.getFragmentOffset(Fragment) + Fixup.getOffset();

  if (const MCSymbolRefExpr *RefB = Target.getSymB()) {
    assert(RefB->getKind() == MCSymbolRefExpr::VK_None &&
           "Should not have constructed this");

    // Let A, B and C being the components of Target and R be the location of
    // the fixup. If the fixup is not pcrel, we want to compute (A - B + C).
    // If it is pcrel, we want to compute (A - B + C - R).

    // In general, ELF has no relocations for -B. It can only represent (A + C)
    // or (A + C - R). If B = R + K and the relocation is not pcrel, we can
    // replace B to implement it: (A - R - K + C)
    if (IsPCRel)
      Asm.getContext().FatalError(
          Fixup.getLoc(),
          "No relocation available to represent this relative expression");

    const MCSymbol &SymB = RefB->getSymbol();

    if (SymB.isUndefined())
      Asm.getContext().FatalError(
          Fixup.getLoc(),
          Twine("symbol '") + SymB.getName() +
              "' can not be undefined in a subtraction expression");

    assert(!SymB.isAbsolute() && "Should have been folded");
    const MCSection &SecB = SymB.getSection();
    if (&SecB != &FixupSection->getSection())
      Asm.getContext().FatalError(
          Fixup.getLoc(), "Cannot represent a difference across sections");

    const MCSymbolData &SymBD = Asm.getSymbolData(SymB);
    uint64_t SymBOffset = Layout.getSymbolOffset(&SymBD);
    uint64_t K = SymBOffset - FixupOffset;
    IsPCRel = true;
    C -= K;
  }

  // We either rejected the fixup or folded B into C at this point.
  const MCSymbolRefExpr *RefA = Target.getSymA();
  const MCSymbol *SymA = RefA ? &RefA->getSymbol() : nullptr;
  const MCSymbolData *SymAD = SymA ? &Asm.getSymbolData(*SymA) : nullptr;

  unsigned Type = GetRelocType(Target, Fixup, IsPCRel);
  bool RelocateWithSymbol = shouldRelocateWithSymbol(Asm, RefA, SymAD, C, Type);
  if (!RelocateWithSymbol && SymA && !SymA->isUndefined())
    C += Layout.getSymbolOffset(SymAD);

  uint64_t Addend = 0;
  if (hasRelocationAddend()) {
    Addend = C;
    C = 0;
  }

  FixedValue = C;

  // FIXME: What is this!?!?
  MCSymbolRefExpr::VariantKind Modifier =
      RefA ? RefA->getKind() : MCSymbolRefExpr::VK_None;
  if (RelocNeedsGOT(Modifier))
    NeedsGOT = true;

  if (!RelocateWithSymbol) {
    const MCSection *SecA =
        (SymA && !SymA->isUndefined()) ? &SymA->getSection() : nullptr;
    auto *ELFSec = cast_or_null<MCSectionELF>(SecA);
    MCSymbol *SectionSymbol =
        ELFSec ? Asm.getContext().getOrCreateSectionSymbol(*ELFSec)
               : nullptr;
    ELFRelocationEntry Rec(FixupOffset, SectionSymbol, Type, Addend);
    Relocations[FixupSection].push_back(Rec);
    return;
  }

  if (SymA) {
    if (const MCSymbol *R = Renames.lookup(SymA))
      SymA = R;

    if (const MCSymbol *WeakRef = getWeakRef(*RefA))
      WeakrefUsedInReloc.insert(WeakRef);
    else
      UsedInReloc.insert(SymA);
  }
  ELFRelocationEntry Rec(FixupOffset, SymA, Type, Addend);
  Relocations[FixupSection].push_back(Rec);
  return;
}


uint64_t
ELFObjectWriter::getSymbolIndexInSymbolTable(const MCAssembler &Asm,
                                             const MCSymbol *S) {
  const MCSymbolData &SD = Asm.getSymbolData(*S);
  return SD.getIndex();
}

bool ELFObjectWriter::isInSymtab(const MCAsmLayout &Layout,
                                 const MCSymbolData &Data, bool Used,
                                 bool Renamed) {
  const MCSymbol &Symbol = Data.getSymbol();
  if (Symbol.isVariable()) {
    const MCExpr *Expr = Symbol.getVariableValue();
    if (const MCSymbolRefExpr *Ref = dyn_cast<MCSymbolRefExpr>(Expr)) {
      if (Ref->getKind() == MCSymbolRefExpr::VK_WEAKREF)
        return false;
    }
  }

  if (Used)
    return true;

  if (Renamed)
    return false;

  if (Symbol.getName() == "_GLOBAL_OFFSET_TABLE_")
    return true;

  if (Symbol.isVariable()) {
    const MCSymbol *Base = Layout.getBaseSymbol(Symbol);
    if (Base && Base->isUndefined())
      return false;
  }

  bool IsGlobal = MCELF::GetBinding(Data) == ELF::STB_GLOBAL;
  if (!Symbol.isVariable() && Symbol.isUndefined() && !IsGlobal)
    return false;

  if (Symbol.isTemporary())
    return false;

  return true;
}

bool ELFObjectWriter::isLocal(const MCSymbolData &Data, bool isUsedInReloc) {
  if (Data.isExternal())
    return false;

  const MCSymbol &Symbol = Data.getSymbol();
  if (Symbol.isDefined())
    return true;

  if (isUsedInReloc)
    return false;

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

void
ELFObjectWriter::computeSymbolTable(MCAssembler &Asm, const MCAsmLayout &Layout,
                                    const SectionIndexMapTy &SectionIndexMap,
                                    const RevGroupMapTy &RevGroupMap,
                                    unsigned NumRegularSections) {
  // FIXME: Is this the correct place to do this?
  // FIXME: Why is an undefined reference to _GLOBAL_OFFSET_TABLE_ needed?
  if (NeedsGOT) {
    StringRef Name = "_GLOBAL_OFFSET_TABLE_";
    MCSymbol *Sym = Asm.getContext().GetOrCreateSymbol(Name);
    MCSymbolData &Data = Asm.getOrCreateSymbolData(*Sym);
    Data.setExternal(true);
    MCELF::SetBinding(Data, ELF::STB_GLOBAL);
  }

  // Add the data for the symbols.
  for (MCSymbolData &SD : Asm.symbols()) {
    const MCSymbol &Symbol = SD.getSymbol();

    bool Used = UsedInReloc.count(&Symbol);
    bool WeakrefUsed = WeakrefUsedInReloc.count(&Symbol);
    bool isSignature = RevGroupMap.count(&Symbol);

    if (!isInSymtab(Layout, SD,
                    Used || WeakrefUsed || isSignature,
                    Renames.count(&Symbol)))
      continue;

    ELFSymbolData MSD;
    MSD.SymbolData = &SD;
    const MCSymbol *BaseSymbol = Layout.getBaseSymbol(Symbol);

    // Undefined symbols are global, but this is the first place we
    // are able to set it.
    bool Local = isLocal(SD, Used);
    if (!Local && MCELF::GetBinding(SD) == ELF::STB_LOCAL) {
      assert(BaseSymbol);
      MCSymbolData &BaseData = Asm.getSymbolData(*BaseSymbol);
      MCELF::SetBinding(SD, ELF::STB_GLOBAL);
      MCELF::SetBinding(BaseData, ELF::STB_GLOBAL);
    }

    if (!BaseSymbol) {
      MSD.SectionIndex = ELF::SHN_ABS;
    } else if (SD.isCommon()) {
      assert(!Local);
      MSD.SectionIndex = ELF::SHN_COMMON;
    } else if (BaseSymbol->isUndefined()) {
      if (isSignature && !Used)
        MSD.SectionIndex = SectionIndexMap.lookup(RevGroupMap.lookup(&Symbol));
      else
        MSD.SectionIndex = ELF::SHN_UNDEF;
      if (!Used && WeakrefUsed)
        MCELF::SetBinding(SD, ELF::STB_WEAK);
    } else {
      const MCSectionELF &Section =
        static_cast<const MCSectionELF&>(BaseSymbol->getSection());
      MSD.SectionIndex = SectionIndexMap.lookup(&Section);
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

    // Sections have their own string table
    if (MCELF::GetType(SD) != ELF::STT_SECTION)
      MSD.Name = StrTabBuilder.add(Name);

    if (MSD.SectionIndex == ELF::SHN_UNDEF)
      UndefinedSymbolData.push_back(MSD);
    else if (Local)
      LocalSymbolData.push_back(MSD);
    else
      ExternalSymbolData.push_back(MSD);
  }

  for (auto i = Asm.file_names_begin(), e = Asm.file_names_end(); i != e; ++i)
    StrTabBuilder.add(*i);

  StrTabBuilder.finalize(StringTableBuilder::ELF);

  for (auto i = Asm.file_names_begin(), e = Asm.file_names_end(); i != e; ++i)
    FileSymbolData.push_back(StrTabBuilder.getOffset(*i));

  for (ELFSymbolData &MSD : LocalSymbolData)
    MSD.StringIndex = MCELF::GetType(*MSD.SymbolData) == ELF::STT_SECTION
                          ? 0
                          : StrTabBuilder.getOffset(MSD.Name);
  for (ELFSymbolData &MSD : ExternalSymbolData)
    MSD.StringIndex = StrTabBuilder.getOffset(MSD.Name);
  for (ELFSymbolData& MSD : UndefinedSymbolData)
    MSD.StringIndex = StrTabBuilder.getOffset(MSD.Name);

  // Symbols are required to be in lexicographic order.
  array_pod_sort(LocalSymbolData.begin(), LocalSymbolData.end());
  array_pod_sort(ExternalSymbolData.begin(), ExternalSymbolData.end());
  array_pod_sort(UndefinedSymbolData.begin(), UndefinedSymbolData.end());

  // Set the symbol indices. Local symbols must come before all other
  // symbols with non-local bindings.
  unsigned Index = FileSymbolData.size() + 1;
  for (unsigned i = 0, e = LocalSymbolData.size(); i != e; ++i)
    LocalSymbolData[i].SymbolData->setIndex(Index++);

  for (unsigned i = 0, e = ExternalSymbolData.size(); i != e; ++i)
    ExternalSymbolData[i].SymbolData->setIndex(Index++);
  for (unsigned i = 0, e = UndefinedSymbolData.size(); i != e; ++i)
    UndefinedSymbolData[i].SymbolData->setIndex(Index++);
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

    unsigned Flags = 0;
    StringRef Group = "";
    if (Section.getFlags() & ELF::SHF_GROUP) {
      Flags = ELF::SHF_GROUP;
      Group = Section.getGroup()->getName();
    }

    const MCSectionELF *RelaSection =
      Ctx.getELFSection(RelaSectionName, hasRelocationAddend() ?
                        ELF::SHT_RELA : ELF::SHT_REL, Flags,
                        SectionKind::getReadOnly(),
                        EntrySize, Group);
    RelMap[&Section] = RelaSection;
    Asm.getOrCreateSectionData(*RelaSection);
  }
}

static SmallVector<char, 128>
getUncompressedData(MCAsmLayout &Layout,
                    MCSectionData::FragmentListType &Fragments) {
  SmallVector<char, 128> UncompressedData;
  for (const MCFragment &F : Fragments) {
    const SmallVectorImpl<char> *Contents;
    switch (F.getKind()) {
    case MCFragment::FT_Data:
      Contents = &cast<MCDataFragment>(F).getContents();
      break;
    case MCFragment::FT_Dwarf:
      Contents = &cast<MCDwarfLineAddrFragment>(F).getContents();
      break;
    case MCFragment::FT_DwarfFrame:
      Contents = &cast<MCDwarfCallFrameFragment>(F).getContents();
      break;
    default:
      llvm_unreachable(
          "Not expecting any other fragment types in a debug_* section");
    }
    UncompressedData.append(Contents->begin(), Contents->end());
  }
  return UncompressedData;
}

// Include the debug info compression header:
// "ZLIB" followed by 8 bytes representing the uncompressed size of the section,
// useful for consumers to preallocate a buffer to decompress into.
static bool
prependCompressionHeader(uint64_t Size,
                         SmallVectorImpl<char> &CompressedContents) {
  static const StringRef Magic = "ZLIB";
  if (Size <= Magic.size() + sizeof(Size) + CompressedContents.size())
    return false;
  if (sys::IsLittleEndianHost)
    sys::swapByteOrder(Size);
  CompressedContents.insert(CompressedContents.begin(),
                            Magic.size() + sizeof(Size), 0);
  std::copy(Magic.begin(), Magic.end(), CompressedContents.begin());
  std::copy(reinterpret_cast<char *>(&Size),
            reinterpret_cast<char *>(&Size + 1),
            CompressedContents.begin() + Magic.size());
  return true;
}

// Return a single fragment containing the compressed contents of the whole
// section. Null if the section was not compressed for any reason.
static std::unique_ptr<MCDataFragment>
getCompressedFragment(MCAsmLayout &Layout,
                      MCSectionData::FragmentListType &Fragments) {
  std::unique_ptr<MCDataFragment> CompressedFragment(new MCDataFragment());

  // Gather the uncompressed data from all the fragments, recording the
  // alignment fragment, if seen, and any fixups.
  SmallVector<char, 128> UncompressedData =
      getUncompressedData(Layout, Fragments);

  SmallVectorImpl<char> &CompressedContents = CompressedFragment->getContents();

  zlib::Status Success = zlib::compress(
      StringRef(UncompressedData.data(), UncompressedData.size()),
      CompressedContents);
  if (Success != zlib::StatusOK)
    return nullptr;

  if (!prependCompressionHeader(UncompressedData.size(), CompressedContents))
    return nullptr;

  return CompressedFragment;
}

typedef DenseMap<const MCSectionData *, std::vector<MCSymbolData *>>
DefiningSymbolMap;

static void UpdateSymbols(const MCAsmLayout &Layout,
                          const std::vector<MCSymbolData *> &Symbols,
                          MCFragment &NewFragment) {
  for (MCSymbolData *Sym : Symbols) {
    Sym->setOffset(Sym->getOffset() +
                   Layout.getFragmentOffset(Sym->getFragment()));
    Sym->setFragment(&NewFragment);
  }
}

static void CompressDebugSection(MCAssembler &Asm, MCAsmLayout &Layout,
                                 const DefiningSymbolMap &DefiningSymbols,
                                 const MCSectionELF &Section,
                                 MCSectionData &SD) {
  StringRef SectionName = Section.getSectionName();
  MCSectionData::FragmentListType &Fragments = SD.getFragmentList();

  std::unique_ptr<MCDataFragment> CompressedFragment =
      getCompressedFragment(Layout, Fragments);

  // Leave the section as-is if the fragments could not be compressed.
  if (!CompressedFragment)
    return;

  // Update the fragment+offsets of any symbols referring to fragments in this
  // section to refer to the new fragment.
  auto I = DefiningSymbols.find(&SD);
  if (I != DefiningSymbols.end())
    UpdateSymbols(Layout, I->second, *CompressedFragment);

  // Invalidate the layout for the whole section since it will have new and
  // different fragments now.
  Layout.invalidateFragmentsFrom(&Fragments.front());
  Fragments.clear();

  // Complete the initialization of the new fragment
  CompressedFragment->setParent(&SD);
  CompressedFragment->setLayoutOrder(0);
  Fragments.push_back(CompressedFragment.release());

  // Rename from .debug_* to .zdebug_*
  Asm.getContext().renameELFSection(&Section,
                                    (".z" + SectionName.drop_front(1)).str());
}

void ELFObjectWriter::CompressDebugSections(MCAssembler &Asm,
                                            MCAsmLayout &Layout) {
  if (!Asm.getContext().getAsmInfo()->compressDebugSections())
    return;

  DefiningSymbolMap DefiningSymbols;

  for (MCSymbolData &SD : Asm.symbols())
    if (MCFragment *F = SD.getFragment())
      DefiningSymbols[F->getParent()].push_back(&SD);

  for (MCSectionData &SD : Asm) {
    const MCSectionELF &Section =
        static_cast<const MCSectionELF &>(SD.getSection());
    StringRef SectionName = Section.getSectionName();

    // Compressing debug_frame requires handling alignment fragments which is
    // more work (possibly generalizing MCAssembler.cpp:writeFragment to allow
    // for writing to arbitrary buffers) for little benefit.
    if (!SectionName.startswith(".debug_") || SectionName == ".debug_frame")
      continue;

    CompressDebugSection(Asm, Layout, DefiningSymbols, Section, SD);
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

// ELF doesn't require relocations to be in any order. We sort by the r_offset,
// just to match gnu as for easier comparison. The use type is an arbitrary way
// of making the sort deterministic.
static int cmpRel(const ELFRelocationEntry *AP, const ELFRelocationEntry *BP) {
  const ELFRelocationEntry &A = *AP;
  const ELFRelocationEntry &B = *BP;
  if (A.Offset != B.Offset)
    return B.Offset - A.Offset;
  if (B.Type != A.Type)
    return A.Type - B.Type;
  llvm_unreachable("ELFRelocs might be unstable!");
}

static void sortRelocs(const MCAssembler &Asm,
                       std::vector<ELFRelocationEntry> &Relocs) {
  array_pod_sort(Relocs.begin(), Relocs.end(), cmpRel);
}

void ELFObjectWriter::WriteRelocationsFragment(const MCAssembler &Asm,
                                               MCDataFragment *F,
                                               const MCSectionData *SD) {
  std::vector<ELFRelocationEntry> &Relocs = Relocations[SD];

  sortRelocs(Asm, Relocs);

  for (unsigned i = 0, e = Relocs.size(); i != e; ++i) {
    const ELFRelocationEntry &Entry = Relocs[e - i - 1];
    unsigned Index =
        Entry.Symbol ? getSymbolIndexInSymbolTable(Asm, Entry.Symbol) : 0;

    if (is64Bit()) {
      write(*F, Entry.Offset);
      if (TargetObjectWriter->isN64()) {
        write(*F, uint32_t(Index));

        write(*F, TargetObjectWriter->getRSsym(Entry.Type));
        write(*F, TargetObjectWriter->getRType3(Entry.Type));
        write(*F, TargetObjectWriter->getRType2(Entry.Type));
        write(*F, TargetObjectWriter->getRType(Entry.Type));
      } else {
        struct ELF::Elf64_Rela ERE64;
        ERE64.setSymbolAndType(Index, Entry.Type);
        write(*F, ERE64.r_info);
      }
      if (hasRelocationAddend())
        write(*F, Entry.Addend);
    } else {
      write(*F, uint32_t(Entry.Offset));

      struct ELF::Elf32_Rela ERE32;
      ERE32.setSymbolAndType(Index, Entry.Type);
      write(*F, ERE32.r_info);

      if (hasRelocationAddend())
        write(*F, uint32_t(Entry.Addend));
    }
  }
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
  WriteSymbolTable(F, Asm, Layout, SectionIndexMap);

  F = new MCDataFragment(&StrtabSD);
  F->getContents().append(StrTabBuilder.data().begin(),
                          StrTabBuilder.data().end());

  F = new MCDataFragment(&ShstrtabSD);

  // Section header string table.
  for (auto it = Asm.begin(), ie = Asm.end(); it != ie; ++it) {
    const MCSectionELF &Section =
      static_cast<const MCSectionELF&>(it->getSection());
    ShStrTabBuilder.add(Section.getSectionName());
  }
  ShStrTabBuilder.finalize(StringTableBuilder::ELF);
  F->getContents().append(ShStrTabBuilder.data().begin(),
                          ShStrTabBuilder.data().end());
}

void ELFObjectWriter::CreateIndexedSections(MCAssembler &Asm,
                                            MCAsmLayout &Layout,
                                            GroupMapTy &GroupMap,
                                            RevGroupMapTy &RevGroupMap,
                                            SectionIndexMapTy &SectionIndexMap,
                                            const RelMapTy &RelMap) {
  MCContext &Ctx = Asm.getContext();

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
      write(*F, uint32_t(ELF::GRP_COMDAT));
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
    uint32_t Index = SectionIndexMap.lookup(&Section);
    write(*F, Index);
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
    sh_link = ShStrTabBuilder.getOffset(Section.getSectionName());
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
    StringRef GroupName =
        Section.getGroup() ? Section.getGroup()->getName() : "";

    InfoSection = Asm.getContext().getELFSection(SectionName, ELF::SHT_PROGBITS,
                                                 0, SectionKind::getReadOnly(),
                                                 0, GroupName);
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
  case ELF::SHT_MIPS_REGINFO:
  case ELF::SHT_MIPS_OPTIONS:
  case ELF::SHT_MIPS_ABIFLAGS:
    // Nothing to do.
    break;

  case ELF::SHT_GROUP:
    sh_link = SymbolTableIndex;
    sh_info = GroupSymbolIndex;
    break;

  default:
    llvm_unreachable("FIXME: sh_type value not supported!");
  }

  if (TargetObjectWriter->getEMachine() == ELF::EM_ARM &&
      Section.getType() == ELF::SHT_ARM_EXIDX) {
    StringRef SecName(Section.getSectionName());
    if (SecName == ".ARM.exidx") {
      sh_link = SectionIndexMap.lookup(
        Asm.getContext().getELFSection(".text",
                                       ELF::SHT_PROGBITS,
                                       ELF::SHF_EXECINSTR | ELF::SHF_ALLOC,
                                       SectionKind::getText()));
    } else if (SecName.startswith(".ARM.exidx")) {
      StringRef GroupName =
          Section.getGroup() ? Section.getGroup()->getName() : "";
      sh_link = SectionIndexMap.lookup(Asm.getContext().getELFSection(
          SecName.substr(sizeof(".ARM.exidx") - 1), ELF::SHT_PROGBITS,
          ELF::SHF_EXECINSTR | ELF::SHF_ALLOC, SectionKind::getText(), 0,
          GroupName));
    }
  }

  WriteSecHdrEntry(ShStrTabBuilder.getOffset(Section.getSectionName()),
                   Section.getType(),
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
      WriteBytes(cast<MCDataFragment>(F).getContents());
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

  CompressDebugSections(Asm, const_cast<MCAsmLayout &>(Layout));

  DenseMap<const MCSectionELF*, const MCSectionELF*> RelMap;
  CreateRelocationSections(Asm, const_cast<MCAsmLayout&>(Layout), RelMap);

  const unsigned NumUserAndRelocSections = Asm.size();
  CreateIndexedSections(Asm, const_cast<MCAsmLayout&>(Layout), GroupMap,
                        RevGroupMap, SectionIndexMap, RelMap);
  const unsigned AllSections = Asm.size();
  const unsigned NumIndexedSections = AllSections - NumUserAndRelocSections;

  unsigned NumRegularSections = NumUserSections + NumIndexedSections;

  // Compute symbol table information.
  computeSymbolTable(Asm, Layout, SectionIndexMap, RevGroupMap,
                     NumRegularSections);

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
  WriteHeader(Asm, SectionHeaderOffset, NumSections + 1);

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
  if (DataA.getFlags() & ELF_STB_Weak || MCELF::GetType(DataA) == ELF::STT_GNU_IFUNC)
    return false;
  return MCObjectWriter::IsSymbolRefDifferenceFullyResolvedImpl(
                                                 Asm, DataA, FB,InSet, IsPCRel);
}

MCObjectWriter *llvm::createELFObjectWriter(MCELFObjectTargetWriter *MOTW,
                                            raw_ostream &OS,
                                            bool IsLittleEndian) {
  return new ELFObjectWriter(MOTW, OS, IsLittleEndian);
}
