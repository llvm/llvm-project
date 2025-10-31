//===- ELFObjHandler.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===/

#include "llvm/InterfaceStub/ELFObjHandler.h"
#include "llvm/InterfaceStub/IFSStub.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include <optional>

using llvm::object::ELFObjectFile;

using namespace llvm;
using namespace llvm::object;
using namespace llvm::ELF;

namespace llvm {
namespace ifs {

// Simple struct to hold relevant .dynamic entries.
struct DynamicEntries {
  uint64_t StrTabAddr = 0;
  uint64_t StrSize = 0;
  std::optional<uint64_t> SONameOffset;
  std::vector<uint64_t> NeededLibNames;
  // Symbol table:
  uint64_t DynSymAddr = 0;
  // Hash tables:
  std::optional<uint64_t> ElfHash;
  std::optional<uint64_t> GnuHash;
  // Version tables:
  std::optional<uint64_t> VerSym;
  std::optional<uint64_t> VerDef;
  std::optional<uint64_t> VerDefNum;
  std::optional<uint64_t> VerNeed;
  std::optional<uint64_t> VerNeedNum;
};

/// This initializes an ELF file header with information specific to a binary
/// dynamic shared object.
/// Offsets, indexes, links, etc. for section and program headers are just
/// zero-initialized as they will be updated elsewhere.
///
/// @param ElfHeader Target ELFT::Ehdr to populate.
/// @param Machine Target architecture (e_machine from ELF specifications).
template <class ELFT>
static void initELFHeader(typename ELFT::Ehdr &ElfHeader, uint16_t Machine) {
  memset(&ElfHeader, 0, sizeof(ElfHeader));
  // ELF identification.
  ElfHeader.e_ident[EI_MAG0] = ElfMagic[EI_MAG0];
  ElfHeader.e_ident[EI_MAG1] = ElfMagic[EI_MAG1];
  ElfHeader.e_ident[EI_MAG2] = ElfMagic[EI_MAG2];
  ElfHeader.e_ident[EI_MAG3] = ElfMagic[EI_MAG3];
  ElfHeader.e_ident[EI_CLASS] = ELFT::Is64Bits ? ELFCLASS64 : ELFCLASS32;
  bool IsLittleEndian = ELFT::Endianness == llvm::endianness::little;
  ElfHeader.e_ident[EI_DATA] = IsLittleEndian ? ELFDATA2LSB : ELFDATA2MSB;
  ElfHeader.e_ident[EI_VERSION] = EV_CURRENT;
  ElfHeader.e_ident[EI_OSABI] = ELFOSABI_NONE;

  // Remainder of ELF header.
  ElfHeader.e_type = ET_DYN;
  ElfHeader.e_machine = Machine;
  ElfHeader.e_version = EV_CURRENT;
  ElfHeader.e_ehsize = sizeof(typename ELFT::Ehdr);
  ElfHeader.e_phentsize = sizeof(typename ELFT::Phdr);
  ElfHeader.e_shentsize = sizeof(typename ELFT::Shdr);
}

namespace {
template <class ELFT> struct OutputSection {
  using Elf_Shdr = typename ELFT::Shdr;
  std::string Name;
  Elf_Shdr Shdr;
  uint64_t Addr;
  uint64_t Offset;
  uint64_t Size;
  uint64_t Align;
  uint32_t Index;
  bool NoBits = true;
};

template <class T, class ELFT>
struct ContentSection : public OutputSection<ELFT> {
  T Content;
  ContentSection() { this->NoBits = false; }
};

// This class just wraps StringTableBuilder for the purpose of adding a
// default constructor.
class ELFStringTableBuilder : public StringTableBuilder {
public:
  ELFStringTableBuilder() : StringTableBuilder(StringTableBuilder::ELF) {}
};

template <class ELFT> class ELFSymbolTableBuilder {
public:
  using Elf_Sym = typename ELFT::Sym;

  ELFSymbolTableBuilder() { Symbols.push_back({}); }

  void add(size_t StNameOffset, uint64_t StSize, uint8_t StBind, uint8_t StType,
           uint8_t StOther, uint16_t StShndx) {
    Elf_Sym S{};
    S.st_name = StNameOffset;
    S.st_size = StSize;
    S.st_info = (StBind << 4) | (StType & 0xf);
    S.st_other = StOther;
    S.st_shndx = StShndx;
    Symbols.push_back(S);
  }

  size_t getSize() const { return Symbols.size() * sizeof(Elf_Sym); }

  void write(uint8_t *Buf) const {
    memcpy(Buf, Symbols.data(), sizeof(Elf_Sym) * Symbols.size());
  }

private:
  llvm::SmallVector<Elf_Sym, 8> Symbols;
};

template <class ELFT> class ELFVersionSymbolBuilder {
public:
  using Elf_Versym = typename ELFT::Versym;

  ELFVersionSymbolBuilder() { Versions.push_back({}); }

  void add(uint16_t Index) {
    Elf_Versym VerSym;
    VerSym.vs_index = Index;
    Versions.push_back(VerSym);
  }

  size_t getSize() const { return Versions.size() * sizeof(Elf_Versym); }

  void write(uint8_t *Buf) const {
    memcpy(Buf, Versions.data(), Versions.size() * sizeof(Elf_Versym));
  }

private:
  llvm::SmallVector<Elf_Versym, 8> Versions;
};

template <class ELFT> class ELFVersionDefinitionBuilder {
public:
  using Elf_Verdef = typename ELFT::Verdef;
  using Elf_Verdaux = typename ELFT::Verdaux;

  ELFVersionDefinitionBuilder() { VerDAux.push_back({}); }

  void addDef(uint16_t Index, uint16_t Count, uint32_t Hash) {
    Elf_Verdef VerDef;
    VerDef.vd_version = VER_DEF_CURRENT;
    VerDef.vd_flags = Index == 1 ? VER_FLG_BASE : 0;
    VerDef.vd_ndx = Index;
    VerDef.vd_cnt = Count;
    VerDef.vd_hash = Hash;
    VerDef.vd_aux = sizeof(Elf_Verdef);
    VerDef.vd_next = sizeof(Elf_Verdef) + Count * sizeof(Elf_Verdaux);
    this->VerDef.push_back(VerDef);
    this->VerDAux.push_back({});
  }

  void addAux(uint16_t Vdndx, uint32_t Name) {
    Elf_Verdaux VerDAux;
    VerDAux.vda_name = Name;
    VerDAux.vda_next = sizeof(Elf_Verdaux);
    this->VerDAux[Vdndx].push_back(VerDAux);
  }

  void finalize() {
    if (!VerDef.empty())
      VerDef.back().vd_next = 0;
    for (llvm::SmallVector<Elf_Verdaux, 8> &VerDAux : VerDAux)
      if (!VerDAux.empty())
        VerDAux.back().vda_next = 0;
  }

  size_t getSize() const {
    size_t Count = 0;
    for (const llvm::SmallVector<Elf_Verdaux, 8> &I : VerDAux)
      Count += I.size();
    return Count * sizeof(Elf_Verdaux) + VerDef.size() * sizeof(Elf_Verdef);
  }

  void write(uint8_t *Buf) const {
    uint8_t *Ptr = Buf;
    size_t Count = 0;
    for (const Elf_Verdef &VerDef : VerDef) {
      Count = sizeof(Elf_Verdef);
      memcpy(Ptr, &VerDef, Count);
      Ptr += Count;
      Count = VerDef.vd_cnt * sizeof(Elf_Verdaux);
      memcpy(Ptr, VerDAux[VerDef.vd_ndx].data(), Count);
      Ptr += Count;
    }
  }

private:
  llvm::SmallVector<Elf_Verdef, 8> VerDef;
  std::vector<llvm::SmallVector<Elf_Verdaux, 8>> VerDAux;
};

template <class ELFT> class ELFVersionRequirementBuilder {
public:
  using Elf_Verneed = typename ELFT::Verneed;
  using Elf_Vernaux = typename ELFT::Vernaux;

  void addFile(uint16_t Count, uint32_t File) {
    Elf_Verneed VerNeed;
    VerNeed.vn_version = VER_NEED_CURRENT;
    VerNeed.vn_cnt = Count;
    VerNeed.vn_file = File;
    VerNeed.vn_aux = sizeof(Elf_Verneed);
    VerNeed.vn_next = sizeof(Elf_Verneed) + Count * sizeof(Elf_Vernaux);
    this->VerNeed.push_back(VerNeed);
    this->VerNAux.push_back({});
  }

  void addAux(uint16_t Vnndx, uint32_t Hash, uint16_t Vdndx, uint32_t Name) {
    Elf_Vernaux VerNAux;
    VerNAux.vna_hash = Hash;
    VerNAux.vna_flags = 0;
    VerNAux.vna_other = Vdndx;
    VerNAux.vna_name = Name;
    VerNAux.vna_next = sizeof(Elf_Vernaux);
    this->VerNAux[Vnndx].push_back(VerNAux);
  }

  void finalize() {
    if (!VerNeed.empty())
      VerNeed.back().vn_next = 0;
    for (llvm::SmallVector<Elf_Vernaux, 8> &VerNAux : VerNAux)
      if (!VerNAux.empty())
        VerNAux.back().vna_next = 0;
  }

  size_t getSize() const {
    size_t Count = 0;
    for (const llvm::SmallVector<Elf_Vernaux, 8> &I : VerNAux)
      Count += I.size();
    return Count * sizeof(Elf_Vernaux) + VerNeed.size() * sizeof(Elf_Verneed);
  }

  void write(uint8_t *Buf) const {
    uint8_t *Ptr = Buf;
    size_t Count = 0;
    size_t Vnndx = 0;
    for (const Elf_Verneed &VerNeed : VerNeed) {
      Count = sizeof(Elf_Verneed);
      memcpy(Ptr, &VerNeed, Count);
      Ptr += Count;
      Count = VerNeed.vn_cnt * sizeof(Elf_Vernaux);
      memcpy(Ptr, VerNAux[Vnndx].data(), Count);
      Ptr += Count;
    }
  }

private:
  llvm::SmallVector<Elf_Verneed, 8> VerNeed;
  std::vector<llvm::SmallVector<Elf_Vernaux, 8>> VerNAux;
};

template <class ELFT> class ELFDynamicTableBuilder {
public:
  using Elf_Dyn = typename ELFT::Dyn;

  size_t addAddr(uint64_t Tag, uint64_t Addr) {
    Elf_Dyn Entry;
    Entry.d_tag = Tag;
    Entry.d_un.d_ptr = Addr;
    Entries.push_back(Entry);
    return Entries.size() - 1;
  }

  void modifyAddr(size_t Index, uint64_t Addr) {
    Entries[Index].d_un.d_ptr = Addr;
  }

  size_t addValue(uint64_t Tag, uint64_t Value) {
    Elf_Dyn Entry;
    Entry.d_tag = Tag;
    Entry.d_un.d_val = Value;
    Entries.push_back(Entry);
    return Entries.size() - 1;
  }

  void modifyValue(size_t Index, uint64_t Value) {
    Entries[Index].d_un.d_val = Value;
  }

  size_t getSize() const {
    // Add DT_NULL entry at the end.
    return (Entries.size() + 1) * sizeof(Elf_Dyn);
  }

  void write(uint8_t *Buf) const {
    memcpy(Buf, Entries.data(), sizeof(Elf_Dyn) * Entries.size());
    // Add DT_NULL entry at the end.
    memset(Buf + sizeof(Elf_Dyn) * Entries.size(), 0, sizeof(Elf_Dyn));
  }

private:
  llvm::SmallVector<Elf_Dyn, 8> Entries;
};

template <class ELFT> class ELFStubBuilder {
public:
  using Elf_Ehdr = typename ELFT::Ehdr;
  using Elf_Shdr = typename ELFT::Shdr;
  using Elf_Phdr = typename ELFT::Phdr;
  using Elf_Sym = typename ELFT::Sym;
  using Elf_Addr = typename ELFT::Addr;
  using Elf_Dyn = typename ELFT::Dyn;

  ELFStubBuilder(const ELFStubBuilder &) = delete;
  ELFStubBuilder(ELFStubBuilder &&) = default;

  explicit ELFStubBuilder(const IFSStub &Stub) {
    DynSym.Name = ".dynsym";
    DynSym.Align = sizeof(Elf_Addr);
    DynStr.Name = ".dynstr";
    DynStr.Align = 1;
    DynTab.Name = ".dynamic";
    DynTab.Align = sizeof(Elf_Addr);
    ShStrTab.Name = ".shstrtab";
    ShStrTab.Align = 1;
    VerSym.Name = ".gnu.version";
    VerSym.Align = 2;
    VerDef.Name = ".gnu.version_d";
    VerDef.Align = sizeof(Elf_Addr);
    VerNeed.Name = ".gnu.version_r";
    VerNeed.Align = sizeof(Elf_Addr);

    // Populate string tables.
    for (const IFSSymbol &Sym : Stub.Symbols)
      DynStr.Content.add(Sym.Name);
    for (const IFSVerDef &VerDef : Stub.VersionDefinitions)
      DynStr.Content.add(VerDef.Name);
    for (const IFSVerNeed &VerNeed : Stub.VersionRequirements)
      for (const std::string &VerNeedName : VerNeed.Names)
        DynStr.Content.add(VerNeedName);
    for (const std::string &Lib : Stub.NeededLibs)
      DynStr.Content.add(Lib);
    if (Stub.SoName)
      DynStr.Content.add(*Stub.SoName);

    for (const IFSSymbol &Sym : Stub.Symbols)
      if (!Sym.Version.empty())
        WriteVerSym = true;
    WriteVerDef = !Stub.VersionDefinitions.empty();
    WriteVerNeed = !Stub.VersionRequirements.empty();

    std::vector<OutputSection<ELFT> *> Sections;
    Sections.push_back(&DynSym);
    Sections.push_back(&DynStr);
    if (WriteVerSym)
      Sections.push_back(&VerSym);
    if (WriteVerNeed)
      Sections.push_back(&VerDef);
    if (WriteVerNeed)
      Sections.push_back(&VerNeed);
    Sections.push_back(&DynTab);
    Sections.push_back(&ShStrTab);
    const OutputSection<ELFT> *LastSection = Sections.back();
    // Now set the Index and put sections names into ".shstrtab".
    uint64_t Index = 1;
    for (OutputSection<ELFT> *Sec : Sections) {
      Sec->Index = Index++;
      ShStrTab.Content.add(Sec->Name);
    }
    ShStrTab.Content.finalize();
    ShStrTab.Size = ShStrTab.Content.getSize();
    DynStr.Content.finalize();
    DynStr.Size = DynStr.Content.getSize();

    // Populate dynamic symbol table.
    for (const IFSSymbol &Sym : Stub.Symbols) {
      uint8_t Bind = Sym.Weak ? STB_WEAK : STB_GLOBAL;
      // For non-undefined symbols, value of the shndx is not relevant at link
      // time as long as it is not SHN_UNDEF. Set shndx to 1, which
      // points to ".dynsym".
      uint16_t Shndx = Sym.Undefined ? SHN_UNDEF : 1;
      uint64_t Size = Sym.Size.value_or(0);
      DynSym.Content.add(DynStr.Content.getOffset(Sym.Name), Size, Bind,
                         convertIFSSymbolTypeToELF(Sym.Type), 0, Shndx);
    }
    DynSym.Size = DynSym.Content.getSize();

    std::map<std::string, uint16_t> VerDefMap;
    size_t Vdndx = 0;
    size_t Vnndx = 0;

    // Populate symbol version definition table.
    for (const IFSVerDef &IFSVerDef : Stub.VersionDefinitions) {
      Vdndx++;
      VerDef.Content.addDef(Vdndx, IFSVerDef.Parents.size() + 1,
                            hashSysV(IFSVerDef.Name));
      VerDef.Content.addAux(Vdndx, DynStr.Content.getOffset(IFSVerDef.Name));
      for (const std::string &Parent : IFSVerDef.Parents) {
        VerDef.Content.addAux(Vdndx, DynStr.Content.getOffset(Parent));
      }
      VerDefMap.insert({IFSVerDef.Name, Vdndx});
    }
    VerDef.Content.finalize();
    VerDef.Size = VerDef.Content.getSize();

    // Populate symbol version requirement table.
    for (const IFSVerNeed &IFSVerNeed : Stub.VersionRequirements) {
      VerNeed.Content.addFile(IFSVerNeed.Names.size(),
                              DynStr.Content.getOffset(IFSVerNeed.File));
      for (const std::string &Name : IFSVerNeed.Names) {
        Vdndx++;
        VerNeed.Content.addAux(Vnndx, hashSysV(Name), Vdndx,
                               DynStr.Content.getOffset(Name));
        VerDefMap.insert({Name, Vdndx});
      }
      Vnndx++;
    }
    VerNeed.Content.finalize();
    VerNeed.Size = VerNeed.Content.getSize();

    // Populate dynamic symbol version table.
    for (const IFSSymbol &Sym : Stub.Symbols)
      VerSym.Content.add(Sym.Version.empty() ? 1 : VerDefMap[Sym.Version]);
    VerSym.Size = VerSym.Content.getSize();

    // Poplulate dynamic table.
    size_t DynSymIndex = DynTab.Content.addAddr(DT_SYMTAB, 0);
    size_t DynStrIndex = DynTab.Content.addAddr(DT_STRTAB, 0);
    size_t VerSymIndex;
    size_t VerDefIndex;
    size_t VerNeedIndex;
    DynTab.Content.addValue(DT_STRSZ, DynSym.Size);
    for (const std::string &Lib : Stub.NeededLibs)
      DynTab.Content.addValue(DT_NEEDED, DynStr.Content.getOffset(Lib));
    if (Stub.SoName)
      DynTab.Content.addValue(DT_SONAME,
                              DynStr.Content.getOffset(*Stub.SoName));
    if (WriteVerSym)
      VerSymIndex = DynTab.Content.addAddr(DT_VERSYM, 0);
    if (WriteVerDef)
      VerDefIndex = DynTab.Content.addAddr(DT_VERDEF, 0);
    if (WriteVerNeed)
      VerNeedIndex = DynTab.Content.addAddr(DT_VERNEED, 0);
    DynTab.Size = DynTab.Content.getSize();
    // Calculate sections' addresses and offsets.
    uint64_t CurrentOffset = sizeof(Elf_Ehdr) + ElfPhnum * sizeof(Elf_Phdr);
    for (OutputSection<ELFT> *Sec : Sections) {
      Sec->Offset = alignTo(CurrentOffset, Sec->Align);
      Sec->Addr = Sec->Offset;
      CurrentOffset = Sec->Offset + Sec->Size;
    }
    // Fill Addr back to dynamic table.
    DynTab.Content.modifyAddr(DynSymIndex, DynSym.Addr);
    DynTab.Content.modifyAddr(DynStrIndex, DynStr.Addr);
    if (WriteVerSym)
      DynTab.Content.modifyAddr(VerSymIndex, VerSym.Addr);
    if (WriteVerDef)
      DynTab.Content.modifyAddr(VerDefIndex, VerDef.Addr);
    if (WriteVerNeed)
      DynTab.Content.modifyAddr(VerNeedIndex, VerNeed.Addr);
    // Write section headers of string tables.
    fillSymTabShdr(DynSym, SHT_DYNSYM);
    fillStrTabShdr(DynStr, SHF_ALLOC);
    fillDynTabShdr(DynTab);
    fillStrTabShdr(ShStrTab);
    if (WriteVerSym)
      fillVerSymShdr(VerSym);
    if (WriteVerDef)
      fillVerDefShdr(VerDef, Stub.VersionDefinitions.size());
    if (WriteVerNeed)
      fillVerNeedShdr(VerNeed, Stub.VersionRequirements.size());

    // Finish initializing the ELF header.
    initELFHeader<ELFT>(ElfHeader, static_cast<uint16_t>(*Stub.Target.Arch));
    ElfHeader.e_shstrndx = ShStrTab.Index;
    ElfHeader.e_shnum = LastSection->Index + 1;
    ElfHeader.e_phnum = ElfPhnum;
    ElfHeader.e_phoff = sizeof(Elf_Ehdr);
    ElfHeader.e_shoff =
        alignTo(LastSection->Offset + LastSection->Size, sizeof(Elf_Addr));

    ElfProgramHeaders[0].p_type = PT_LOAD;
    ElfProgramHeaders[0].p_offset = 0;
    ElfProgramHeaders[0].p_vaddr = 0;
    ElfProgramHeaders[0].p_paddr = 0;
    ElfProgramHeaders[0].p_filesz = DynTab.Shdr.sh_offset;
    ElfProgramHeaders[0].p_memsz = DynTab.Shdr.sh_offset;
    ElfProgramHeaders[0].p_flags = PF_R;
    ElfProgramHeaders[0].p_align = 0x1000;

    ElfProgramHeaders[1].p_type = PT_DYNAMIC;
    ElfProgramHeaders[1].p_offset = DynTab.Shdr.sh_offset;
    ElfProgramHeaders[1].p_vaddr = DynTab.Shdr.sh_offset;
    ElfProgramHeaders[1].p_paddr = DynTab.Shdr.sh_offset;
    ElfProgramHeaders[1].p_filesz = DynTab.Shdr.sh_size;
    ElfProgramHeaders[1].p_memsz = DynTab.Shdr.sh_size;
    ElfProgramHeaders[1].p_flags = PF_R | PF_W;
    ElfProgramHeaders[1].p_align = DynTab.Shdr.sh_addralign;
  }

  size_t getSize() const {
    return ElfHeader.e_shoff + ElfHeader.e_shnum * sizeof(Elf_Shdr);
  }

  void write(uint8_t *Data) const {
    write(Data, ElfHeader);
    for (size_t I = 0; I < ElfPhnum; I++)
      write(Data + ElfHeader.e_phoff + I * sizeof(Elf_Phdr),
            ElfProgramHeaders[I]);
    DynSym.Content.write(Data + DynSym.Shdr.sh_offset);
    DynStr.Content.write(Data + DynStr.Shdr.sh_offset);
    DynTab.Content.write(Data + DynTab.Shdr.sh_offset);
    ShStrTab.Content.write(Data + ShStrTab.Shdr.sh_offset);
    if (WriteVerSym)
      VerSym.Content.write(Data + VerSym.Shdr.sh_offset);
    if (WriteVerDef)
      VerDef.Content.write(Data + VerDef.Shdr.sh_offset);
    if (WriteVerNeed)
      VerNeed.Content.write(Data + VerNeed.Shdr.sh_offset);
    writeShdr(Data, DynSym);
    writeShdr(Data, DynStr);
    writeShdr(Data, DynTab);
    writeShdr(Data, ShStrTab);
    if (WriteVerSym)
      writeShdr(Data, VerSym);
    if (WriteVerDef)
      writeShdr(Data, VerDef);
    if (WriteVerNeed)
      writeShdr(Data, VerNeed);
  }

private:
  Elf_Ehdr ElfHeader;
  static constexpr auto ElfPhnum = 2;
  Elf_Phdr ElfProgramHeaders[ElfPhnum];
  ContentSection<ELFStringTableBuilder, ELFT> DynStr;
  ContentSection<ELFStringTableBuilder, ELFT> ShStrTab;
  ContentSection<ELFSymbolTableBuilder<ELFT>, ELFT> DynSym;
  ContentSection<ELFDynamicTableBuilder<ELFT>, ELFT> DynTab;
  ContentSection<ELFVersionSymbolBuilder<ELFT>, ELFT> VerSym;
  ContentSection<ELFVersionDefinitionBuilder<ELFT>, ELFT> VerDef;
  ContentSection<ELFVersionRequirementBuilder<ELFT>, ELFT> VerNeed;
  bool WriteVerSym = false;
  bool WriteVerDef = false;
  bool WriteVerNeed = false;

  template <class T> static void write(uint8_t *Data, const T &Value) {
    *reinterpret_cast<T *>(Data) = Value;
  }

  void fillStrTabShdr(ContentSection<ELFStringTableBuilder, ELFT> &StrTab,
                      uint32_t ShFlags = 0) const {
    StrTab.Shdr.sh_type = SHT_STRTAB;
    StrTab.Shdr.sh_flags = ShFlags;
    StrTab.Shdr.sh_addr = StrTab.Addr;
    StrTab.Shdr.sh_offset = StrTab.Offset;
    StrTab.Shdr.sh_info = 0;
    StrTab.Shdr.sh_size = StrTab.Size;
    StrTab.Shdr.sh_name = ShStrTab.Content.getOffset(StrTab.Name);
    StrTab.Shdr.sh_addralign = StrTab.Align;
    StrTab.Shdr.sh_entsize = 0;
    StrTab.Shdr.sh_link = 0;
  }

  void fillSymTabShdr(ContentSection<ELFSymbolTableBuilder<ELFT>, ELFT> &SymTab,
                      uint32_t ShType) const {
    SymTab.Shdr.sh_type = ShType;
    SymTab.Shdr.sh_flags = SHF_ALLOC;
    SymTab.Shdr.sh_addr = SymTab.Addr;
    SymTab.Shdr.sh_offset = SymTab.Offset;
    // Only non-local symbols are included in the tbe file, so .dynsym only
    // contains 1 local symbol (the undefined symbol at index 0). The sh_info
    // should always be 1.
    SymTab.Shdr.sh_info = 1;
    SymTab.Shdr.sh_size = SymTab.Size;
    SymTab.Shdr.sh_name = this->ShStrTab.Content.getOffset(SymTab.Name);
    SymTab.Shdr.sh_addralign = SymTab.Align;
    SymTab.Shdr.sh_entsize = sizeof(Elf_Sym);
    SymTab.Shdr.sh_link = this->DynStr.Index;
  }

  void fillDynTabShdr(
      ContentSection<ELFDynamicTableBuilder<ELFT>, ELFT> &DynTab) const {
    DynTab.Shdr.sh_type = SHT_DYNAMIC;
    DynTab.Shdr.sh_flags = SHF_ALLOC;
    DynTab.Shdr.sh_addr = DynTab.Addr;
    DynTab.Shdr.sh_offset = DynTab.Offset;
    DynTab.Shdr.sh_info = 0;
    DynTab.Shdr.sh_size = DynTab.Size;
    DynTab.Shdr.sh_name = this->ShStrTab.Content.getOffset(DynTab.Name);
    DynTab.Shdr.sh_addralign = DynTab.Align;
    DynTab.Shdr.sh_entsize = sizeof(Elf_Dyn);
    DynTab.Shdr.sh_link = this->DynStr.Index;
  }

  void fillVerSymShdr(
      ContentSection<ELFVersionSymbolBuilder<ELFT>, ELFT> &VerSym) const {
    VerSym.Shdr.sh_type = SHT_GNU_versym;
    VerSym.Shdr.sh_flags = SHF_ALLOC;
    VerSym.Shdr.sh_addr = VerSym.Addr;
    VerSym.Shdr.sh_offset = VerSym.Offset;
    VerSym.Shdr.sh_info = 0;
    VerSym.Shdr.sh_size = VerSym.Size;
    VerSym.Shdr.sh_name = this->ShStrTab.Content.getOffset(VerSym.Name);
    VerSym.Shdr.sh_addralign = VerSym.Align;
    VerSym.Shdr.sh_entsize = sizeof(uint16_t);
    VerSym.Shdr.sh_link = this->DynSym.Index;
  }

  void fillVerDefShdr(
      ContentSection<ELFVersionDefinitionBuilder<ELFT>, ELFT> &VerDef,
      size_t Size) const {
    VerDef.Shdr.sh_type = SHT_GNU_verdef;
    VerDef.Shdr.sh_flags = SHF_ALLOC;
    VerDef.Shdr.sh_addr = VerDef.Addr;
    VerDef.Shdr.sh_offset = VerDef.Offset;
    VerDef.Shdr.sh_info = Size;
    VerDef.Shdr.sh_size = VerDef.Size;
    VerDef.Shdr.sh_name = this->ShStrTab.Content.getOffset(VerDef.Name);
    VerDef.Shdr.sh_addralign = VerDef.Align;
    VerDef.Shdr.sh_entsize = sizeof(Elf_Dyn);
    VerDef.Shdr.sh_link = this->DynStr.Index;
  }

  void fillVerNeedShdr(
      ContentSection<ELFVersionRequirementBuilder<ELFT>, ELFT> &VerNeed,
      size_t Size) const {
    VerNeed.Shdr.sh_type = SHT_GNU_verneed;
    VerNeed.Shdr.sh_flags = SHF_ALLOC;
    VerNeed.Shdr.sh_addr = VerNeed.Addr;
    VerNeed.Shdr.sh_offset = VerNeed.Offset;
    VerNeed.Shdr.sh_info = Size;
    VerNeed.Shdr.sh_size = VerNeed.Size;
    VerNeed.Shdr.sh_name = this->ShStrTab.Content.getOffset(VerNeed.Name);
    VerNeed.Shdr.sh_addralign = VerNeed.Align;
    VerNeed.Shdr.sh_entsize = sizeof(Elf_Dyn);
    VerNeed.Shdr.sh_link = this->DynStr.Index;
  }

  uint64_t shdrOffset(const OutputSection<ELFT> &Sec) const {
    return ElfHeader.e_shoff + Sec.Index * sizeof(Elf_Shdr);
  }

  void writeShdr(uint8_t *Data, const OutputSection<ELFT> &Sec) const {
    write(Data + shdrOffset(Sec), Sec.Shdr);
  }
};

/// This function takes an error, and appends a string of text to the end of
/// that error. Since "appending" to an Error isn't supported behavior of an
/// Error, this function technically creates a new error with the combined
/// message and consumes the old error.
///
/// @param Err Source error.
/// @param After Text to append at the end of Err's error message.
Error appendToError(Error Err, StringRef After) {
  std::string Message;
  raw_string_ostream Stream(Message);
  Stream << Err;
  Stream << " " << After;
  consumeError(std::move(Err));
  return createError(Stream.str());
}

template <class ELFT> class DynSym {
  using Elf_Shdr_Range = typename ELFT::ShdrRange;
  using Elf_Shdr = typename ELFT::Shdr;

public:
  static Expected<DynSym> create(const ELFFile<ELFT> &ElfFile,
                                 const DynamicEntries &DynEnt) {
    Expected<Elf_Shdr_Range> Shdrs = ElfFile.sections();
    if (!Shdrs)
      return Shdrs.takeError();
    return DynSym(ElfFile, DynEnt, *Shdrs);
  }

  Expected<const uint8_t *> getDynSym() {
    if (DynSymHdr)
      return ElfFile.base() + DynSymHdr->sh_offset;
    return getDynamicData(DynEnt.DynSymAddr, "dynamic symbol table");
  }

  const Elf_Shdr *getVerSym() { return findHdr(SHT_GNU_versym); }

  const Elf_Shdr *getVerDef() { return findHdr(SHT_GNU_verdef); }

  const Elf_Shdr *getVerNeed() { return findHdr(SHT_GNU_verneed); }

  Expected<StringRef> getDynStr() {
    if (DynSymHdr)
      return ElfFile.getStringTableForSymtab(*DynSymHdr, Shdrs);
    Expected<const uint8_t *> DataOrErr = getDynamicData(
        DynEnt.StrTabAddr, "dynamic string table", DynEnt.StrSize);
    if (!DataOrErr)
      return DataOrErr.takeError();
    return StringRef(reinterpret_cast<const char *>(*DataOrErr),
                     DynEnt.StrSize);
  }

private:
  DynSym(const ELFFile<ELFT> &ElfFile, const DynamicEntries &DynEnt,
         Elf_Shdr_Range Shdrs)
      : ElfFile(ElfFile), DynEnt(DynEnt), Shdrs(Shdrs),
        DynSymHdr(findHdr(SHT_DYNSYM)) {}

  const Elf_Shdr *findHdr(uint32_t Type) {
    for (const Elf_Shdr &Sec : Shdrs)
      if (Sec.sh_type == Type) {
        // If multiple .dynsym are present, use the first one.
        // This behavior aligns with llvm::object::ELFFile::getDynSymtabSize()
        return &Sec;
      }
    return nullptr;
  }

  Expected<const uint8_t *> getDynamicData(uint64_t EntAddr, StringRef Name,
                                           uint64_t Size = 0) {
    Expected<const uint8_t *> SecPtr = ElfFile.toMappedAddr(EntAddr);
    if (!SecPtr)
      return appendToError(
          SecPtr.takeError(),
          ("when locating " + Name + " section contents").str());
    Expected<const uint8_t *> SecEndPtr = ElfFile.toMappedAddr(EntAddr + Size);
    if (!SecEndPtr)
      return appendToError(
          SecEndPtr.takeError(),
          ("when locating " + Name + " section contents").str());
    return *SecPtr;
  }

  const ELFFile<ELFT> &ElfFile;
  const DynamicEntries &DynEnt;
  Elf_Shdr_Range Shdrs;
  const Elf_Shdr *DynSymHdr;
};
} // end anonymous namespace

/// This function behaves similarly to StringRef::substr(), but attempts to
/// terminate the returned StringRef at the first null terminator. If no null
/// terminator is found, an error is returned.
///
/// @param Str Source string to create a substring from.
/// @param Offset The start index of the desired substring.
static Expected<StringRef> terminatedSubstr(StringRef Str, size_t Offset) {
  size_t StrEnd = Str.find('\0', Offset);
  if (StrEnd == StringLiteral::npos) {
    return createError(
        "String overran bounds of string table (no null terminator)");
  }

  size_t StrLen = StrEnd - Offset;
  return Str.substr(Offset, StrLen);
}

/// This function populates a DynamicEntries struct using an ELFT::DynRange.
/// After populating the struct, the members are validated with
/// some basic correctness checks.
///
/// @param Dyn Target DynamicEntries struct to populate.
/// @param DynTable Source dynamic table.
template <class ELFT>
static Error populateDynamic(DynamicEntries &Dyn,
                             typename ELFT::DynRange DynTable) {
  if (DynTable.empty())
    return createError("No .dynamic section found");

  // Search .dynamic for relevant entries.
  bool FoundDynStr = false;
  bool FoundDynStrSz = false;
  bool FoundDynSym = false;
  for (auto &Entry : DynTable) {
    switch (Entry.d_tag) {
    case DT_SONAME:
      Dyn.SONameOffset = Entry.d_un.d_val;
      break;
    case DT_STRTAB:
      Dyn.StrTabAddr = Entry.d_un.d_ptr;
      FoundDynStr = true;
      break;
    case DT_STRSZ:
      Dyn.StrSize = Entry.d_un.d_val;
      FoundDynStrSz = true;
      break;
    case DT_NEEDED:
      Dyn.NeededLibNames.push_back(Entry.d_un.d_val);
      break;
    case DT_SYMTAB:
      Dyn.DynSymAddr = Entry.d_un.d_ptr;
      FoundDynSym = true;
      break;
    case DT_HASH:
      Dyn.ElfHash = Entry.d_un.d_ptr;
      break;
    case DT_GNU_HASH:
      Dyn.GnuHash = Entry.d_un.d_ptr;
      break;
    case DT_VERSYM:
      Dyn.VerSym = Entry.d_un.d_ptr;
      break;
    case DT_VERDEF:
      Dyn.VerDef = Entry.d_un.d_ptr;
      break;
    case DT_VERDEFNUM:
      Dyn.VerDefNum = Entry.d_un.d_val;
      break;
    case DT_VERNEED:
      Dyn.VerNeed = Entry.d_un.d_ptr;
      break;
    case DT_VERNEEDNUM:
      Dyn.VerNeedNum = Entry.d_un.d_val;
      break;
    }
  }

  if (!FoundDynStr) {
    return createError(
        "Couldn't locate dynamic string table (no DT_STRTAB entry)");
  }
  if (!FoundDynStrSz) {
    return createError(
        "Couldn't determine dynamic string table size (no DT_STRSZ entry)");
  }
  if (!FoundDynSym) {
    return createError(
        "Couldn't locate dynamic symbol table (no DT_SYMTAB entry)");
  }
  if (Dyn.SONameOffset && *Dyn.SONameOffset >= Dyn.StrSize) {
    return createStringError(object_error::parse_failed,
                             "DT_SONAME string offset (0x%016" PRIx64
                             ") outside of dynamic string table",
                             *Dyn.SONameOffset);
  }
  for (uint64_t Offset : Dyn.NeededLibNames) {
    if (Offset >= Dyn.StrSize) {
      return createStringError(object_error::parse_failed,
                               "DT_NEEDED string offset (0x%016" PRIx64
                               ") outside of dynamic string table",
                               Offset);
    }
  }

  return Error::success();
}

/// This function creates an IFSSymbol and populates all members using
/// information from a binary ELFT::Sym.
///
/// @param SymName The desired name of the IFSSymbol.
/// @param SymVer The desired version of the IFSSymbol.
/// @param RawSym ELFT::Sym to extract symbol information from.
template <class ELFT>
static IFSSymbol createELFSym(StringRef SymName, StringRef SymVer,
                              const typename ELFT::Sym &RawSym) {
  IFSSymbol TargetSym{std::string(SymName), std::string(SymVer)};
  uint8_t Binding = RawSym.getBinding();
  if (Binding == STB_WEAK)
    TargetSym.Weak = true;
  else
    TargetSym.Weak = false;

  TargetSym.Undefined = RawSym.isUndefined();
  TargetSym.Type = convertELFSymbolTypeToIFS(RawSym.st_info);

  if (TargetSym.Type == IFSSymbolType::Func) {
    TargetSym.Size = 0;
  } else {
    TargetSym.Size = RawSym.st_size;
  }
  return TargetSym;
}

/// This function populates an IFSStub with symbols using information read
/// from an ELF binary.
///
/// @param TargetStub IFSStub to add symbols to.
/// @param ElfFile Elf File object.
/// @param DynSym Range of dynamic symbols to add to TargetStub.
/// @param VerSymPtr Pointer to symbol version table in Elf File.
/// @param VersionMap Versions of dynamic symbols.
/// @param DynStr StringRef to the dynamic string table.
template <class ELFT>
static Error
populateSymbols(IFSStub &TargetStub, const ELFFile<ELFT> &ElfFile,
                const typename ELFT::SymRange DynSym,
                const typename ELFT::Shdr *VerSymPtr,
                const SmallVector<std::optional<VersionEntry>> &VersionMap,
                StringRef DynStr) {
  // Skips the first symbol since it's the NULL symbol.
  size_t I = 0;
  for (auto RawSym : DynSym.drop_front(1)) {
    I++;
    // If a symbol does not have global or weak binding, ignore it.
    uint8_t Binding = RawSym.getBinding();
    if (!(Binding == STB_GLOBAL || Binding == STB_WEAK))
      continue;
    // If a symbol doesn't have default or protected visibility, ignore it.
    uint8_t Visibility = RawSym.getVisibility();
    if (!(Visibility == STV_DEFAULT || Visibility == STV_PROTECTED))
      continue;
    // Create an IFSSymbol and populate it with information from the symbol
    // table entry.
    Expected<StringRef> SymName = terminatedSubstr(DynStr, RawSym.st_name);
    if (!SymName)
      return SymName.takeError();
    std::string VersionName;
    if (VerSymPtr) {
      Expected<const typename ELFT::Versym *> VerEntryOrErr =
          ElfFile.template getEntry<typename ELFT::Versym>(*VerSymPtr, I);
      if (!VerEntryOrErr)
        return appendToError(VerEntryOrErr.takeError(),
                             "when reading symbol versions");
      size_t VersionIndex = (*VerEntryOrErr)->vs_index & VERSYM_VERSION;
      if (VersionIndex > VersionMap.size() || !VersionMap[VersionIndex]) {
        return createError("SHT_GNU_versym section refers to a version index " +
                           Twine(VersionIndex) + " which is missing");
      }
      if (VersionIndex != VER_NDX_LOCAL && VersionIndex != VER_NDX_GLOBAL)
        VersionName = VersionMap[VersionIndex]->Name.c_str();
    }
    IFSSymbol Sym = createELFSym<ELFT>(*SymName, VersionName, RawSym);
    TargetStub.Symbols.push_back(std::move(Sym));
    // TODO: Populate symbol warning.
  }
  return Error::success();
}

/// Returns a new IFSStub with all members populated from an ELFObjectFile.
/// @param ElfObj Source ELFObjectFile.
template <class ELFT>
static Expected<std::unique_ptr<IFSStub>>
buildStub(const ELFObjectFile<ELFT> &ElfObj) {
  using Elf_Dyn_Range = typename ELFT::DynRange;
  using Elf_Sym_Range = typename ELFT::SymRange;
  using Elf_Sym = typename ELFT::Sym;
  using Elf_Shdr = typename ELFT::Shdr;
  std::unique_ptr<IFSStub> DestStub = std::make_unique<IFSStub>();
  const ELFFile<ELFT> &ElfFile = ElfObj.getELFFile();
  // Fetch .dynamic table.
  Expected<Elf_Dyn_Range> DynTable = ElfFile.dynamicEntries();
  if (!DynTable) {
    return DynTable.takeError();
  }

  // Collect relevant .dynamic entries.
  DynamicEntries DynEnt;
  if (Error Err = populateDynamic<ELFT>(DynEnt, *DynTable))
    return std::move(Err);
  Expected<DynSym<ELFT>> EDynSym = DynSym<ELFT>::create(ElfFile, DynEnt);
  if (!EDynSym)
    return EDynSym.takeError();

  Expected<StringRef> EDynStr = EDynSym->getDynStr();
  if (!EDynStr)
    return EDynStr.takeError();

  StringRef DynStr = *EDynStr;

  // Populate Arch from ELF header.
  DestStub->Target.Arch = static_cast<IFSArch>(ElfFile.getHeader().e_machine);
  DestStub->Target.BitWidth =
      convertELFBitWidthToIFS(ElfFile.getHeader().e_ident[EI_CLASS]);
  DestStub->Target.Endianness =
      convertELFEndiannessToIFS(ElfFile.getHeader().e_ident[EI_DATA]);
  DestStub->Target.ObjectFormat = "ELF";

  // Populate SoName from .dynamic entries and dynamic string table.
  if (DynEnt.SONameOffset) {
    Expected<StringRef> NameOrErr =
        terminatedSubstr(DynStr, *DynEnt.SONameOffset);
    if (!NameOrErr) {
      return appendToError(NameOrErr.takeError(), "when reading DT_SONAME");
    }
    DestStub->SoName = std::string(*NameOrErr);
  }

  // Populate NeededLibs from .dynamic entries and dynamic string table.
  for (uint64_t NeededStrOffset : DynEnt.NeededLibNames) {
    Expected<StringRef> LibNameOrErr =
        terminatedSubstr(DynStr, NeededStrOffset);
    if (!LibNameOrErr) {
      return appendToError(LibNameOrErr.takeError(), "when reading DT_NEEDED");
    }
    DestStub->NeededLibs.push_back(std::string(*LibNameOrErr));
  }

  const Elf_Shdr *VerDefPtr = EDynSym->getVerDef();
  if (VerDefPtr) {
    Expected<std::vector<VerDef>> VerDefOrError =
        ElfFile.getVersionDefinitions(*VerDefPtr);
    if (!VerDefOrError)
      return appendToError(VerDefOrError.takeError(),
                           "when reading symbol version definitions");
    for (const VerDef &VerDef : *VerDefOrError) {
      std::vector<std::string> Parents;
      for (const VerdAux &VerDAux : VerDef.AuxV) {
        Parents.push_back(VerDAux.Name);
      }
      DestStub->VersionDefinitions.push_back({VerDef.Name, std::move(Parents)});
    }
  }

  const Elf_Shdr *VerNeedPtr = EDynSym->getVerNeed();
  if (VerNeedPtr) {
    Expected<std::vector<VerNeed>> VerNeedOrError =
        ElfFile.getVersionDependencies(*VerNeedPtr);
    if (!VerNeedOrError)
      return appendToError(VerNeedOrError.takeError(),
                           "when reading symbol version needs");
    for (const VerNeed &VerNeed : *VerNeedOrError) {
      std::string File = VerNeed.File;
      IFSVerNeed IFSVerNeed;
      IFSVerNeed.File = VerNeed.File;
      for (const VernAux &VerNAux : VerNeed.AuxV) {
        IFSVerNeed.Names.push_back(VerNAux.Name.c_str());
      }
      DestStub->VersionRequirements.push_back(IFSVerNeed);
    }
  }

  // Populate Symbols from .dynsym table and dynamic string table.
  Expected<uint64_t> SymCount = ElfFile.getDynSymtabSize();
  if (!SymCount)
    return SymCount.takeError();
  if (*SymCount > 0) {
    Expected<SmallVector<std::optional<VersionEntry>>> VersionsOrError =
        ElfFile.loadVersionMap(VerNeedPtr, VerDefPtr);
    if (!VersionsOrError)
      return appendToError(VersionsOrError.takeError(),
                           "when reading dynamic symbol versions");
    // Get pointer to in-memory location of .dynsym section.
    Expected<const uint8_t *> DynSymPtr = EDynSym->getDynSym();
    if (!DynSymPtr)
      return appendToError(DynSymPtr.takeError(),
                           "when locating .dynsym section contents");
    Elf_Sym_Range DynSyms = ArrayRef<Elf_Sym>(
        reinterpret_cast<const Elf_Sym *>(*DynSymPtr), *SymCount);
    Error SymReadError =
        populateSymbols<ELFT>(*DestStub, ElfFile, DynSyms, EDynSym->getVerSym(),
                              *VersionsOrError, DynStr);
    if (SymReadError)
      return appendToError(std::move(SymReadError),
                           "when reading dynamic symbols");
  }

  return std::move(DestStub);
}

/// This function opens a file for writing and then writes a binary ELF stub to
/// the file.
///
/// @param FilePath File path for writing the ELF binary.
/// @param Stub Source InterFace Stub to generate a binary ELF stub from.
template <class ELFT>
static Error writeELFBinaryToFile(StringRef FilePath, const IFSStub &Stub,
                                  bool WriteIfChanged) {
  ELFStubBuilder<ELFT> Builder{Stub};
  // Write Stub to memory first.
  std::vector<uint8_t> Buf(Builder.getSize());
  Builder.write(Buf.data());

  if (WriteIfChanged) {
    if (ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrError =
            MemoryBuffer::getFile(FilePath)) {
      // Compare Stub output with existing Stub file.
      // If Stub file unchanged, abort updating.
      if ((*BufOrError)->getBufferSize() == Builder.getSize() &&
          !memcmp((*BufOrError)->getBufferStart(), Buf.data(),
                  Builder.getSize()))
        return Error::success();
    }
  }

  Expected<std::unique_ptr<FileOutputBuffer>> BufOrError =
      FileOutputBuffer::create(FilePath, Builder.getSize());
  if (!BufOrError)
    return createStringError(errc::invalid_argument,
                             toString(BufOrError.takeError()) +
                                 " when trying to open `" + FilePath +
                                 "` for writing");

  // Write binary to file.
  std::unique_ptr<FileOutputBuffer> FileBuf = std::move(*BufOrError);
  memcpy(FileBuf->getBufferStart(), Buf.data(), Buf.size());

  return FileBuf->commit();
}

Expected<std::unique_ptr<IFSStub>> readELFFile(MemoryBufferRef Buf) {
  Expected<std::unique_ptr<Binary>> BinOrErr = createBinary(Buf);
  if (!BinOrErr) {
    return BinOrErr.takeError();
  }

  Binary *Bin = BinOrErr->get();
  if (auto Obj = dyn_cast<ELFObjectFile<ELF32LE>>(Bin)) {
    return buildStub(*Obj);
  } else if (auto Obj = dyn_cast<ELFObjectFile<ELF64LE>>(Bin)) {
    return buildStub(*Obj);
  } else if (auto Obj = dyn_cast<ELFObjectFile<ELF32BE>>(Bin)) {
    return buildStub(*Obj);
  } else if (auto Obj = dyn_cast<ELFObjectFile<ELF64BE>>(Bin)) {
    return buildStub(*Obj);
  }
  return createStringError(errc::not_supported, "unsupported binary format");
}

// This function wraps the ELFT writeELFBinaryToFile() so writeBinaryStub()
// can be called without having to use ELFType templates directly.
Error writeBinaryStub(StringRef FilePath, const IFSStub &Stub,
                      bool WriteIfChanged) {
  assert(Stub.Target.Arch);
  assert(Stub.Target.BitWidth);
  assert(Stub.Target.Endianness);
  if (Stub.Target.BitWidth == IFSBitWidthType::IFS32) {
    if (Stub.Target.Endianness == IFSEndiannessType::Little) {
      return writeELFBinaryToFile<ELF32LE>(FilePath, Stub, WriteIfChanged);
    } else {
      return writeELFBinaryToFile<ELF32BE>(FilePath, Stub, WriteIfChanged);
    }
  } else {
    if (Stub.Target.Endianness == IFSEndiannessType::Little) {
      return writeELFBinaryToFile<ELF64LE>(FilePath, Stub, WriteIfChanged);
    } else {
      return writeELFBinaryToFile<ELF64BE>(FilePath, Stub, WriteIfChanged);
    }
  }
  llvm_unreachable("invalid binary output target");
}

} // end namespace ifs
} // end namespace llvm
