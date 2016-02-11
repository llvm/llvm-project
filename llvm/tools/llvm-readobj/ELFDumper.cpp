//===-- ELFDumper.cpp - ELF-specific dumper ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements the ELF-specific dumper for llvm-readobj.
///
//===----------------------------------------------------------------------===//

#include "llvm-readobj.h"
#include "ARMAttributeParser.h"
#include "ARMEHABIPrinter.h"
#include "Error.h"
#include "ObjDumper.h"
#include "StackMapPrinter.h"
#include "StreamWriter.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/ARMBuildAttributes.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MipsABIFlags.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;
using namespace llvm::object;
using namespace ELF;

#define LLVM_READOBJ_ENUM_CASE(ns, enum) \
  case ns::enum: return #enum;

#define ENUM_ENT(enum, altName) \
  { #enum, altName, ELF::enum }

#define ENUM_ENT_1(enum) \
  { #enum, #enum, ELF::enum }

namespace {

template <class ELFT> class DumpStyle;

/// \brief Represents a region described by entries in the .dynamic table.
struct DynRegionInfo {
  DynRegionInfo() : Addr(nullptr), Size(0), EntSize(0) {}
  DynRegionInfo(const void *A, uint64_t S, uint64_t ES)
      : Addr(A), Size(S), EntSize(ES) {}
  /// \brief Address in current address space.
  const void *Addr;
  /// \brief Size in bytes of the region.
  uint64_t Size;
  /// \brief Size of each entity in the region.
  uint64_t EntSize;

  template <typename Type> iterator_range<const Type *> getAsRange() const {
    const Type *Start = reinterpret_cast<const Type *>(Addr);
    if (Size == 0)
      return {Start, Start};
    if (EntSize != sizeof(Type) || Size % EntSize)
      reportError("Invalid entity size");
    return {Start, Start + (Size / EntSize)};
  }
};

template<typename ELFT>
class ELFDumper : public ObjDumper {
public:
  ELFDumper(const ELFFile<ELFT> *Obj, StreamWriter &Writer);

  void printFileHeaders() override;
  void printSections() override;
  void printRelocations() override;
  void printDynamicRelocations() override;
  void printSymbols() override;
  void printDynamicSymbols() override;
  void printUnwindInfo() override;

  void printDynamicTable() override;
  void printNeededLibraries() override;
  void printProgramHeaders() override;
  void printHashTable() override;
  void printGnuHashTable() override;
  void printLoadName() override;
  void printVersionInfo() override;
  void printGroupSections() override;

  void printAttributes() override;
  void printMipsPLTGOT() override;
  void printMipsABIFlags() override;
  void printMipsReginfo() override;

  void printStackMap() const override;

private:
  std::unique_ptr<DumpStyle<ELFT>> ELFDumperStyle;
  typedef ELFFile<ELFT> ELFO;
  typedef typename ELFO::Elf_Shdr Elf_Shdr;
  typedef typename ELFO::Elf_Sym Elf_Sym;
  typedef typename ELFO::Elf_Sym_Range Elf_Sym_Range;
  typedef typename ELFO::Elf_Dyn Elf_Dyn;
  typedef typename ELFO::Elf_Dyn_Range Elf_Dyn_Range;
  typedef typename ELFO::Elf_Rel Elf_Rel;
  typedef typename ELFO::Elf_Rela Elf_Rela;
  typedef typename ELFO::Elf_Rel_Range Elf_Rel_Range;
  typedef typename ELFO::Elf_Rela_Range Elf_Rela_Range;
  typedef typename ELFO::Elf_Phdr Elf_Phdr;
  typedef typename ELFO::Elf_Half Elf_Half;
  typedef typename ELFO::Elf_Hash Elf_Hash;
  typedef typename ELFO::Elf_GnuHash Elf_GnuHash;
  typedef typename ELFO::Elf_Ehdr Elf_Ehdr;
  typedef typename ELFO::Elf_Word Elf_Word;
  typedef typename ELFO::uintX_t uintX_t;
  typedef typename ELFO::Elf_Versym Elf_Versym;
  typedef typename ELFO::Elf_Verneed Elf_Verneed;
  typedef typename ELFO::Elf_Vernaux Elf_Vernaux;
  typedef typename ELFO::Elf_Verdef Elf_Verdef;
  typedef typename ELFO::Elf_Verdaux Elf_Verdaux;

  DynRegionInfo checkDRI(DynRegionInfo DRI) {
    if (DRI.Addr < Obj->base() ||
        (const uint8_t *)DRI.Addr + DRI.Size > Obj->base() + Obj->getBufSize())
      error(llvm::object::object_error::parse_failed);
    return DRI;
  }

  DynRegionInfo createDRIFrom(const Elf_Phdr *P, uintX_t EntSize) {
    return checkDRI({Obj->base() + P->p_offset, P->p_filesz, EntSize});
  }
  
  DynRegionInfo createDRIFrom(const Elf_Shdr *S) {
    return checkDRI({Obj->base() + S->sh_offset, S->sh_size, S->sh_entsize});
  }

  void parseDynamicTable(ArrayRef<const Elf_Phdr *> LoadSegments);

  void printSymbolsHelper(bool IsDynamic);
  void printSymbol(const Elf_Sym *Symbol, const Elf_Sym *FirstSym,
                   StringRef StrTable, bool IsDynamic);

  void printDynamicRelocation(Elf_Rela Rel);
  void printRelocations(const Elf_Shdr *Sec);
  void printRelocation(Elf_Rela Rel, const Elf_Shdr *SymTab);
  void printValue(uint64_t Type, uint64_t Value);

  Elf_Rel_Range dyn_rels() const;
  Elf_Rela_Range dyn_relas() const;
  StringRef getDynamicString(uint64_t Offset) const;
  StringRef getSymbolVersion(StringRef StrTab, const Elf_Sym *symb,
                             bool &IsDefault);
  void LoadVersionMap();
  void LoadVersionNeeds(const Elf_Shdr *ec) const;
  void LoadVersionDefs(const Elf_Shdr *sec) const;

  const ELFO *Obj;
  DynRegionInfo DynamicTable;

  // Dynamic relocation info.
  DynRegionInfo DynRelRegion;
  DynRegionInfo DynRelaRegion;
  DynRegionInfo DynPLTRelRegion;

  DynRegionInfo DynSymRegion;
  StringRef DynamicStringTable;
  StringRef SOName;
  const Elf_Hash *HashTable = nullptr;
  const Elf_GnuHash *GnuHashTable = nullptr;
  const Elf_Shdr *DotSymtabSec = nullptr;
  ArrayRef<Elf_Word> ShndxTable;

  const Elf_Shdr *dot_gnu_version_sec = nullptr;   // .gnu.version
  const Elf_Shdr *dot_gnu_version_r_sec = nullptr; // .gnu.version_r
  const Elf_Shdr *dot_gnu_version_d_sec = nullptr; // .gnu.version_d

  // Records for each version index the corresponding Verdef or Vernaux entry.
  // This is filled the first time LoadVersionMap() is called.
  class VersionMapEntry : public PointerIntPair<const void *, 1> {
  public:
    // If the integer is 0, this is an Elf_Verdef*.
    // If the integer is 1, this is an Elf_Vernaux*.
    VersionMapEntry() : PointerIntPair<const void *, 1>(nullptr, 0) {}
    VersionMapEntry(const Elf_Verdef *verdef)
        : PointerIntPair<const void *, 1>(verdef, 0) {}
    VersionMapEntry(const Elf_Vernaux *vernaux)
        : PointerIntPair<const void *, 1>(vernaux, 1) {}
    bool isNull() const { return getPointer() == nullptr; }
    bool isVerdef() const { return !isNull() && getInt() == 0; }
    bool isVernaux() const { return !isNull() && getInt() == 1; }
    const Elf_Verdef *getVerdef() const {
      return isVerdef() ? (const Elf_Verdef *)getPointer() : nullptr;
    }
    const Elf_Vernaux *getVernaux() const {
      return isVernaux() ? (const Elf_Vernaux *)getPointer() : nullptr;
    }
  };
  mutable SmallVector<VersionMapEntry, 16> VersionMap;

public:
  Elf_Dyn_Range dynamic_table() const {
    return DynamicTable.getAsRange<Elf_Dyn>();
  }

  Elf_Sym_Range dynamic_symbols() const {
    return DynSymRegion.getAsRange<Elf_Sym>();
  }

  std::string getFullSymbolName(const Elf_Sym *Symbol, StringRef StrTable,
                                bool IsDynamic);
  const Elf_Shdr *getDotSymtabSec() const { return DotSymtabSec; }
  ArrayRef<Elf_Word> getShndxTable() { return ShndxTable; }
  StringRef getDynamicStringTable() const { return DynamicStringTable; }
};

template <typename ELFT> class DumpStyle {
public:
  virtual void printFileHeaders(const ELFFile<ELFT> *Obj) = 0;
  virtual ~DumpStyle() { }
};

template <typename ELFT> class GNUStyle : public DumpStyle<ELFT> {
  formatted_raw_ostream OS;

public:
  typedef typename ELFFile<ELFT>::Elf_Ehdr Elf_Ehdr;
  GNUStyle(StreamWriter &W) : OS(W.getOStream()) {}
  void printFileHeaders(const ELFFile<ELFT> *Obj) override;

private:
  template <typename T, typename TEnum>
  std::string printEnum(T Value, ArrayRef<EnumEntry<TEnum>> EnumValues) {
    for (const auto &EnumItem : EnumValues)
      if (EnumItem.Value == Value)
        return EnumItem.AltName;
    return to_hexString(Value);
  }
};

template <typename ELFT> class LLVMStyle : public DumpStyle<ELFT> {
public:
  typedef typename ELFFile<ELFT>::Elf_Ehdr Elf_Ehdr;
  LLVMStyle(StreamWriter &W) : W(W) {}

  void printFileHeaders(const ELFFile<ELFT> *Obj) override;

private:
  StreamWriter &W;
};

template <class T> T errorOrDefault(ErrorOr<T> Val, T Default = T()) {
  if (!Val) {
    error(Val.getError());
    return Default;
  }

  return *Val;
}
} // namespace

namespace llvm {

template <class ELFT>
static std::error_code createELFDumper(const ELFFile<ELFT> *Obj,
                                       StreamWriter &Writer,
                                       std::unique_ptr<ObjDumper> &Result) {
  Result.reset(new ELFDumper<ELFT>(Obj, Writer));
  return readobj_error::success;
}

std::error_code createELFDumper(const object::ObjectFile *Obj,
                                StreamWriter &Writer,
                                std::unique_ptr<ObjDumper> &Result) {
  // Little-endian 32-bit
  if (const ELF32LEObjectFile *ELFObj = dyn_cast<ELF32LEObjectFile>(Obj))
    return createELFDumper(ELFObj->getELFFile(), Writer, Result);

  // Big-endian 32-bit
  if (const ELF32BEObjectFile *ELFObj = dyn_cast<ELF32BEObjectFile>(Obj))
    return createELFDumper(ELFObj->getELFFile(), Writer, Result);

  // Little-endian 64-bit
  if (const ELF64LEObjectFile *ELFObj = dyn_cast<ELF64LEObjectFile>(Obj))
    return createELFDumper(ELFObj->getELFFile(), Writer, Result);

  // Big-endian 64-bit
  if (const ELF64BEObjectFile *ELFObj = dyn_cast<ELF64BEObjectFile>(Obj))
    return createELFDumper(ELFObj->getELFFile(), Writer, Result);

  return readobj_error::unsupported_obj_file_format;
}

} // namespace llvm

// Iterate through the versions needed section, and place each Elf_Vernaux
// in the VersionMap according to its index.
template <class ELFT>
void ELFDumper<ELFT>::LoadVersionNeeds(const Elf_Shdr *sec) const {
  unsigned vn_size = sec->sh_size;  // Size of section in bytes
  unsigned vn_count = sec->sh_info; // Number of Verneed entries
  const char *sec_start = (const char *)Obj->base() + sec->sh_offset;
  const char *sec_end = sec_start + vn_size;
  // The first Verneed entry is at the start of the section.
  const char *p = sec_start;
  for (unsigned i = 0; i < vn_count; i++) {
    if (p + sizeof(Elf_Verneed) > sec_end)
      report_fatal_error("Section ended unexpectedly while scanning "
                         "version needed records.");
    const Elf_Verneed *vn = reinterpret_cast<const Elf_Verneed *>(p);
    if (vn->vn_version != ELF::VER_NEED_CURRENT)
      report_fatal_error("Unexpected verneed version");
    // Iterate through the Vernaux entries
    const char *paux = p + vn->vn_aux;
    for (unsigned j = 0; j < vn->vn_cnt; j++) {
      if (paux + sizeof(Elf_Vernaux) > sec_end)
        report_fatal_error("Section ended unexpected while scanning auxiliary "
                           "version needed records.");
      const Elf_Vernaux *vna = reinterpret_cast<const Elf_Vernaux *>(paux);
      size_t index = vna->vna_other & ELF::VERSYM_VERSION;
      if (index >= VersionMap.size())
        VersionMap.resize(index + 1);
      VersionMap[index] = VersionMapEntry(vna);
      paux += vna->vna_next;
    }
    p += vn->vn_next;
  }
}

// Iterate through the version definitions, and place each Elf_Verdef
// in the VersionMap according to its index.
template <class ELFT>
void ELFDumper<ELFT>::LoadVersionDefs(const Elf_Shdr *sec) const {
  unsigned vd_size = sec->sh_size;  // Size of section in bytes
  unsigned vd_count = sec->sh_info; // Number of Verdef entries
  const char *sec_start = (const char *)Obj->base() + sec->sh_offset;
  const char *sec_end = sec_start + vd_size;
  // The first Verdef entry is at the start of the section.
  const char *p = sec_start;
  for (unsigned i = 0; i < vd_count; i++) {
    if (p + sizeof(Elf_Verdef) > sec_end)
      report_fatal_error("Section ended unexpectedly while scanning "
                         "version definitions.");
    const Elf_Verdef *vd = reinterpret_cast<const Elf_Verdef *>(p);
    if (vd->vd_version != ELF::VER_DEF_CURRENT)
      report_fatal_error("Unexpected verdef version");
    size_t index = vd->vd_ndx & ELF::VERSYM_VERSION;
    if (index >= VersionMap.size())
      VersionMap.resize(index + 1);
    VersionMap[index] = VersionMapEntry(vd);
    p += vd->vd_next;
  }
}

template <class ELFT> void ELFDumper<ELFT>::LoadVersionMap() {
  // If there is no dynamic symtab or version table, there is nothing to do.
  if (!DynSymRegion.Addr || !dot_gnu_version_sec)
    return;

  // Has the VersionMap already been loaded?
  if (VersionMap.size() > 0)
    return;

  // The first two version indexes are reserved.
  // Index 0 is LOCAL, index 1 is GLOBAL.
  VersionMap.push_back(VersionMapEntry());
  VersionMap.push_back(VersionMapEntry());

  if (dot_gnu_version_d_sec)
    LoadVersionDefs(dot_gnu_version_d_sec);

  if (dot_gnu_version_r_sec)
    LoadVersionNeeds(dot_gnu_version_r_sec);
}


template <typename ELFO, class ELFT>
static void printVersionSymbolSection(ELFDumper<ELFT> *Dumper,
                                      const ELFO *Obj,
                                      const typename ELFO::Elf_Shdr *Sec,
                                      StreamWriter &W) {
  DictScope SS(W, "Version symbols");
  if (!Sec)
    return;
  StringRef Name = errorOrDefault(Obj->getSectionName(Sec));
  W.printNumber("Section Name", Name, Sec->sh_name);
  W.printHex("Address", Sec->sh_addr);
  W.printHex("Offset", Sec->sh_offset);
  W.printNumber("Link", Sec->sh_link);

  const uint8_t *P = (const uint8_t *)Obj->base() + Sec->sh_offset;
  StringRef StrTable = Dumper->getDynamicStringTable();

  // Same number of entries in the dynamic symbol table (DT_SYMTAB).
  ListScope Syms(W, "Symbols");
  for (const typename ELFO::Elf_Sym &Sym : Dumper->dynamic_symbols()) {
    DictScope S(W, "Symbol");
    std::string FullSymbolName =
        Dumper->getFullSymbolName(&Sym, StrTable, true /* IsDynamic */);
    W.printNumber("Version", *P);
    W.printString("Name", FullSymbolName);
    P += sizeof(typename ELFO::Elf_Half);
  }
}

template <typename ELFO, class ELFT>
static void printVersionDefinitionSection(ELFDumper<ELFT> *Dumper,
                                          const ELFO *Obj,
                                          const typename ELFO::Elf_Shdr *Sec,
                                          StreamWriter &W) {
  DictScope SD(W, "Version definition");
  if (!Sec)
    return;
  StringRef Name = errorOrDefault(Obj->getSectionName(Sec));
  W.printNumber("Section Name", Name, Sec->sh_name);
  W.printHex("Address", Sec->sh_addr);
  W.printHex("Offset", Sec->sh_offset);
  W.printNumber("Link", Sec->sh_link);

  unsigned verdef_entries = 0;
  // The number of entries in the section SHT_GNU_verdef
  // is determined by DT_VERDEFNUM tag.
  for (const typename ELFO::Elf_Dyn &Dyn : Dumper->dynamic_table()) {
    if (Dyn.d_tag == DT_VERDEFNUM)
      verdef_entries = Dyn.d_un.d_val;
  }
  const uint8_t *SecStartAddress =
      (const uint8_t *)Obj->base() + Sec->sh_offset;
  const uint8_t *SecEndAddress = SecStartAddress + Sec->sh_size;
  const uint8_t *P = SecStartAddress;
  ErrorOr<const typename ELFO::Elf_Shdr *> StrTabOrErr =
      Obj->getSection(Sec->sh_link);
  error(StrTabOrErr.getError());

  ListScope Entries(W, "Entries");
  for (unsigned i = 0; i < verdef_entries; ++i) {
    if (P + sizeof(typename ELFO::Elf_Verdef) > SecEndAddress)
      report_fatal_error("invalid offset in the section");
    auto *VD = reinterpret_cast<const typename ELFO::Elf_Verdef *>(P);
    DictScope Entry(W, "Entry");
    W.printHex("Offset", (uintptr_t)P - (uintptr_t)SecStartAddress);
    W.printNumber("Rev", VD->vd_version);
    // FIXME: print something more readable.
    W.printNumber("Flags", VD->vd_flags);
    W.printNumber("Index", VD->vd_ndx);
    W.printNumber("Cnt", VD->vd_cnt);
    W.printString("Name", StringRef((const char *)(Obj->base() +
                                                   (*StrTabOrErr)->sh_offset +
                                                   VD->getAux()->vda_name)));
    P += VD->vd_next;
  }
}

template <typename ELFT> void ELFDumper<ELFT>::printVersionInfo() {
  // Dump version symbol section.
  printVersionSymbolSection(this, Obj, dot_gnu_version_sec, W);

  // Dump version definition section.
  printVersionDefinitionSection(this, Obj, dot_gnu_version_d_sec, W);
}

template <typename ELFT>
StringRef ELFDumper<ELFT>::getSymbolVersion(StringRef StrTab,
                                            const Elf_Sym *symb,
                                            bool &IsDefault) {
  // This is a dynamic symbol. Look in the GNU symbol version table.
  if (!dot_gnu_version_sec) {
    // No version table.
    IsDefault = false;
    return StringRef("");
  }

  // Determine the position in the symbol table of this entry.
  size_t entry_index = (reinterpret_cast<uintptr_t>(symb) -
                        reinterpret_cast<uintptr_t>(DynSymRegion.Addr)) /
                       sizeof(Elf_Sym);

  // Get the corresponding version index entry
  const Elf_Versym *vs =
      Obj->template getEntry<Elf_Versym>(dot_gnu_version_sec, entry_index);
  size_t version_index = vs->vs_index & ELF::VERSYM_VERSION;

  // Special markers for unversioned symbols.
  if (version_index == ELF::VER_NDX_LOCAL ||
      version_index == ELF::VER_NDX_GLOBAL) {
    IsDefault = false;
    return StringRef("");
  }

  // Lookup this symbol in the version table
  LoadVersionMap();
  if (version_index >= VersionMap.size() || VersionMap[version_index].isNull())
    reportError("Invalid version entry");
  const VersionMapEntry &entry = VersionMap[version_index];

  // Get the version name string
  size_t name_offset;
  if (entry.isVerdef()) {
    // The first Verdaux entry holds the name.
    name_offset = entry.getVerdef()->getAux()->vda_name;
    IsDefault = !(vs->vs_index & ELF::VERSYM_HIDDEN);
  } else {
    name_offset = entry.getVernaux()->vna_name;
    IsDefault = false;
  }
  if (name_offset >= StrTab.size())
    reportError("Invalid string offset");
  return StringRef(StrTab.data() + name_offset);
}

template <typename ELFT>
std::string ELFDumper<ELFT>::getFullSymbolName(const Elf_Sym *Symbol,
                                               StringRef StrTable,
                                               bool IsDynamic) {
  StringRef SymbolName = errorOrDefault(Symbol->getName(StrTable));
  if (!IsDynamic)
    return SymbolName;

  std::string FullSymbolName(SymbolName);

  bool IsDefault;
  StringRef Version = getSymbolVersion(StrTable, &*Symbol, IsDefault);
  FullSymbolName += (IsDefault ? "@@" : "@");
  FullSymbolName += Version;
  return FullSymbolName;
}

template <typename ELFO>
static void
getSectionNameIndex(const ELFO &Obj, const typename ELFO::Elf_Sym *Symbol,
                    const typename ELFO::Elf_Sym *FirstSym,
                    ArrayRef<typename ELFO::Elf_Word> ShndxTable,
                    StringRef &SectionName, unsigned &SectionIndex) {
  SectionIndex = Symbol->st_shndx;
  if (Symbol->isUndefined())
    SectionName = "Undefined";
  else if (Symbol->isProcessorSpecific())
    SectionName = "Processor Specific";
  else if (Symbol->isOSSpecific())
    SectionName = "Operating System Specific";
  else if (Symbol->isAbsolute())
    SectionName = "Absolute";
  else if (Symbol->isCommon())
    SectionName = "Common";
  else if (Symbol->isReserved() && SectionIndex != SHN_XINDEX)
    SectionName = "Reserved";
  else {
    if (SectionIndex == SHN_XINDEX)
      SectionIndex =
          Obj.getExtendedSymbolTableIndex(Symbol, FirstSym, ShndxTable);
    ErrorOr<const typename ELFO::Elf_Shdr *> Sec = Obj.getSection(SectionIndex);
    error(Sec.getError());
    SectionName = errorOrDefault(Obj.getSectionName(*Sec));
  }
}

template <class ELFO>
static const typename ELFO::Elf_Shdr *
findNotEmptySectionByAddress(const ELFO *Obj, uint64_t Addr) {
  for (const auto &Shdr : Obj->sections())
    if (Shdr.sh_addr == Addr && Shdr.sh_size > 0)
      return &Shdr;
  return nullptr;
}

template <class ELFO>
static const typename ELFO::Elf_Shdr *findSectionByName(const ELFO &Obj,
                                                        StringRef Name) {
  for (const auto &Shdr : Obj.sections()) {
    if (Name == errorOrDefault(Obj.getSectionName(&Shdr)))
      return &Shdr;
  }
  return nullptr;
}

static const EnumEntry<unsigned> ElfClass[] = {
  {"None",   "none",   ELF::ELFCLASSNONE},
  {"32-bit", "ELF32",  ELF::ELFCLASS32},
  {"64-bit", "ELF64",  ELF::ELFCLASS64},
};

static const EnumEntry<unsigned> ElfDataEncoding[] = {
  {"None",         "none",                          ELF::ELFDATANONE},
  {"LittleEndian", "2's complement, little endian", ELF::ELFDATA2LSB},
  {"BigEndian",    "2's complement, big endian",    ELF::ELFDATA2MSB},
};

static const EnumEntry<unsigned> ElfObjectFileType[] = {
  {"None",         "NONE (none)",              ELF::ET_NONE},
  {"Relocatable",  "REL (Relocatable file)",   ELF::ET_REL},
  {"Executable",   "EXEC (Executable file)",   ELF::ET_EXEC},
  {"SharedObject", "DYN (Shared object file)", ELF::ET_DYN},
  {"Core",         "CORE (Core file)",         ELF::ET_CORE},
};

static const EnumEntry<unsigned> ElfOSABI[] = {
  {"SystemV",      "UNIX - System V",      ELF::ELFOSABI_NONE},
  {"HPUX",         "UNIX - HP-UX",         ELF::ELFOSABI_HPUX},
  {"NetBSD",       "UNIX - NetBSD",        ELF::ELFOSABI_NETBSD},
  {"GNU/Linux",    "UNIX - GNU",           ELF::ELFOSABI_LINUX},
  {"GNU/Hurd",     "GNU/Hurd",             ELF::ELFOSABI_HURD},
  {"Solaris",      "UNIX - Solaris",       ELF::ELFOSABI_SOLARIS},
  {"AIX",          "UNIX - AIX",           ELF::ELFOSABI_AIX},
  {"IRIX",         "UNIX - IRIX",          ELF::ELFOSABI_IRIX},
  {"FreeBSD",      "UNIX - FreeBSD",       ELF::ELFOSABI_FREEBSD},
  {"TRU64",        "UNIX - TRU64",         ELF::ELFOSABI_TRU64},
  {"Modesto",      "Novell - Modesto",     ELF::ELFOSABI_MODESTO},
  {"OpenBSD",      "UNIX - OpenBSD",       ELF::ELFOSABI_OPENBSD},
  {"OpenVMS",      "VMS - OpenVMS",        ELF::ELFOSABI_OPENVMS},
  {"NSK",          "HP - Non-Stop Kernel", ELF::ELFOSABI_NSK},
  {"AROS",         "AROS",                 ELF::ELFOSABI_AROS},
  {"FenixOS",      "FenixOS",              ELF::ELFOSABI_FENIXOS},
  {"CloudABI",     "CloudABI",             ELF::ELFOSABI_CLOUDABI},
  {"C6000_ELFABI", "Bare-metal C6000",     ELF::ELFOSABI_C6000_ELFABI},
  {"C6000_LINUX",  "Linux C6000",          ELF::ELFOSABI_C6000_LINUX},
  {"ARM",          "ARM",                  ELF::ELFOSABI_ARM},
  {"Standalone",   "Standalone App",       ELF::ELFOSABI_STANDALONE}
};

static const EnumEntry<unsigned> ElfMachineType[] = {
  ENUM_ENT(EM_NONE,          "None"),
  ENUM_ENT(EM_M32,           "WE32100"),
  ENUM_ENT(EM_SPARC,         "Sparc"),
  ENUM_ENT(EM_386,           "Intel 80386"),
  ENUM_ENT(EM_68K,           "MC68000"),
  ENUM_ENT(EM_88K,           "MC88000"),
  ENUM_ENT(EM_IAMCU,         "EM_IAMCU"),
  ENUM_ENT(EM_860,           "Intel 80860"),
  ENUM_ENT(EM_MIPS,          "MIPS R3000"),
  ENUM_ENT(EM_S370,          "IBM System/370"),
  ENUM_ENT(EM_MIPS_RS3_LE,   "MIPS R3000 little-endian"),
  ENUM_ENT(EM_PARISC,        "HPPA"),
  ENUM_ENT(EM_VPP500,        "Fujitsu VPP500"),
  ENUM_ENT(EM_SPARC32PLUS,   "Sparc v8+"),
  ENUM_ENT(EM_960,           "Intel 80960"),
  ENUM_ENT(EM_PPC,           "PowerPC"),
  ENUM_ENT(EM_PPC64,         "PowerPC64"),
  ENUM_ENT(EM_S390,          "IBM S/390"),
  ENUM_ENT(EM_SPU,           "SPU"),
  ENUM_ENT(EM_V800,          "NEC V800 series"),
  ENUM_ENT(EM_FR20,          "Fujistsu FR20"),
  ENUM_ENT(EM_RH32,          "TRW RH-32"),
  ENUM_ENT(EM_RCE,           "Motorola RCE"),
  ENUM_ENT(EM_ARM,           "ARM"),
  ENUM_ENT(EM_ALPHA,         "EM_ALPHA"),
  ENUM_ENT(EM_SH,            "Hitachi SH"),
  ENUM_ENT(EM_SPARCV9,       "Sparc v9"),
  ENUM_ENT(EM_TRICORE,       "Siemens Tricore"),
  ENUM_ENT(EM_ARC,           "ARC"),
  ENUM_ENT(EM_H8_300,        "Hitachi H8/300"),
  ENUM_ENT(EM_H8_300H,       "Hitachi H8/300H"),
  ENUM_ENT(EM_H8S,           "Hitachi H8S"),
  ENUM_ENT(EM_H8_500,        "Hitachi H8/500"),
  ENUM_ENT(EM_IA_64,         "Intel IA-64"),
  ENUM_ENT(EM_MIPS_X,        "Stanford MIPS-X"),
  ENUM_ENT(EM_COLDFIRE,      "Motorola Coldfire"),
  ENUM_ENT(EM_68HC12,        "Motorola MC68HC12 Microcontroller"),
  ENUM_ENT(EM_MMA,           "Fujitsu Multimedia Accelerator"),
  ENUM_ENT(EM_PCP,           "Siemens PCP"),
  ENUM_ENT(EM_NCPU,          "Sony nCPU embedded RISC processor"),
  ENUM_ENT(EM_NDR1,          "Denso NDR1 microprocesspr"),
  ENUM_ENT(EM_STARCORE,      "Motorola Star*Core processor"),
  ENUM_ENT(EM_ME16,          "Toyota ME16 processor"),
  ENUM_ENT(EM_ST100,         "STMicroelectronics ST100 processor"),
  ENUM_ENT(EM_TINYJ,         "Advanced Logic Corp. TinyJ embedded processor"),
  ENUM_ENT(EM_X86_64,        "Advanced Micro Devices X86-64"),
  ENUM_ENT(EM_PDSP,          "Sony DSP processor"),
  ENUM_ENT(EM_PDP10,         "Digital Equipment Corp. PDP-10"),
  ENUM_ENT(EM_PDP11,         "Digital Equipment Corp. PDP-11"),
  ENUM_ENT(EM_FX66,          "Siemens FX66 microcontroller"),
  ENUM_ENT(EM_ST9PLUS,       "STMicroelectronics ST9+ 8/16 bit microcontroller"),
  ENUM_ENT(EM_ST7,           "STMicroelectronics ST7 8-bit microcontroller"),
  ENUM_ENT(EM_68HC16,        "Motorola MC68HC16 Microcontroller"),
  ENUM_ENT(EM_68HC11,        "Motorola MC68HC11 Microcontroller"),
  ENUM_ENT(EM_68HC08,        "Motorola MC68HC08 Microcontroller"),
  ENUM_ENT(EM_68HC05,        "Motorola MC68HC05 Microcontroller"),
  ENUM_ENT(EM_SVX,           "Silicon Graphics SVx"),
  ENUM_ENT(EM_ST19,          "STMicroelectronics ST19 8-bit microcontroller"),
  ENUM_ENT(EM_VAX,           "Digital VAX"),
  ENUM_ENT(EM_CRIS,          "Axis Communications 32-bit embedded processor"),
  ENUM_ENT(EM_JAVELIN,       "Infineon Technologies 32-bit embedded cpu"),
  ENUM_ENT(EM_FIREPATH,      "Element 14 64-bit DSP processor"),
  ENUM_ENT(EM_ZSP,           "LSI Logic's 16-bit DSP processor"),
  ENUM_ENT(EM_MMIX,          "Donald Knuth's educational 64-bit processor"),
  ENUM_ENT(EM_HUANY,         "Harvard Universitys's machine-independent object format"),
  ENUM_ENT(EM_PRISM,         "Vitesse Prism"),
  ENUM_ENT(EM_AVR,           "Atmel AVR 8-bit microcontroller"),
  ENUM_ENT(EM_FR30,          "Fujitsu FR30"),
  ENUM_ENT(EM_D10V,          "Mitsubishi D10V"),
  ENUM_ENT(EM_D30V,          "Mitsubishi D30V"),
  ENUM_ENT(EM_V850,          "NEC v850"),
  ENUM_ENT(EM_M32R,          "Renesas M32R (formerly Mitsubishi M32r)"),
  ENUM_ENT(EM_MN10300,       "Matsushita MN10300"),
  ENUM_ENT(EM_MN10200,       "Matsushita MN10200"),
  ENUM_ENT(EM_PJ,            "picoJava"),
  ENUM_ENT(EM_OPENRISC,      "OpenRISC 32-bit embedded processor"),
  ENUM_ENT(EM_ARC_COMPACT,   "EM_ARC_COMPACT"),
  ENUM_ENT(EM_XTENSA,        "Tensilica Xtensa Processor"),
  ENUM_ENT(EM_VIDEOCORE,     "Alphamosaic VideoCore processor"),
  ENUM_ENT(EM_TMM_GPP,       "Thompson Multimedia General Purpose Processor"),
  ENUM_ENT(EM_NS32K,         "National Semiconductor 32000 series"),
  ENUM_ENT(EM_TPC,           "Tenor Network TPC processor"),
  ENUM_ENT(EM_SNP1K,         "EM_SNP1K"),
  ENUM_ENT(EM_ST200,         "STMicroelectronics ST200 microcontroller"),
  ENUM_ENT(EM_IP2K,          "Ubicom IP2xxx 8-bit microcontrollers"),
  ENUM_ENT(EM_MAX,           "MAX Processor"),
  ENUM_ENT(EM_CR,            "National Semiconductor CompactRISC"),
  ENUM_ENT(EM_F2MC16,        "Fujitsu F2MC16"),
  ENUM_ENT(EM_MSP430,        "Texas Instruments msp430 microcontroller"),
  ENUM_ENT(EM_BLACKFIN,      "Analog Devices Blackfin"),
  ENUM_ENT(EM_SE_C33,        "S1C33 Family of Seiko Epson processors"),
  ENUM_ENT(EM_SEP,           "Sharp embedded microprocessor"),
  ENUM_ENT(EM_ARCA,          "Arca RISC microprocessor"),
  ENUM_ENT(EM_UNICORE,       "Unicore"),
  ENUM_ENT(EM_EXCESS,        "eXcess 16/32/64-bit configurable embedded CPU"),
  ENUM_ENT(EM_DXP,           "Icera Semiconductor Inc. Deep Execution Processor"),
  ENUM_ENT(EM_ALTERA_NIOS2,  "Altera Nios"),
  ENUM_ENT(EM_CRX,           "National Semiconductor CRX microprocessor"),
  ENUM_ENT(EM_XGATE,         "Motorola XGATE embedded processor"),
  ENUM_ENT(EM_C166,          "Infineon Technologies xc16x"),
  ENUM_ENT(EM_M16C,          "Renesas M16C"),
  ENUM_ENT(EM_DSPIC30F,      "Microchip Technology dsPIC30F Digital Signal Controller"),
  ENUM_ENT(EM_CE,            "Freescale Communication Engine RISC core"),
  ENUM_ENT(EM_M32C,          "Renesas M32C"),
  ENUM_ENT(EM_TSK3000,       "Altium TSK3000 core"),
  ENUM_ENT(EM_RS08,          "Freescale RS08 embedded processor"),
  ENUM_ENT(EM_SHARC,         "EM_SHARC"),
  ENUM_ENT(EM_ECOG2,         "Cyan Technology eCOG2 microprocessor"),
  ENUM_ENT(EM_SCORE7,        "SUNPLUS S+Core"),
  ENUM_ENT(EM_DSP24,         "New Japan Radio (NJR) 24-bit DSP Processor"),
  ENUM_ENT(EM_VIDEOCORE3,    "Broadcom VideoCore III processor"),
  ENUM_ENT(EM_LATTICEMICO32, "Lattice Mico32"),
  ENUM_ENT(EM_SE_C17,        "Seiko Epson C17 family"),
  ENUM_ENT(EM_TI_C6000,      "Texas Instruments TMS320C6000 DSP family"),
  ENUM_ENT(EM_TI_C2000,      "Texas Instruments TMS320C2000 DSP family"),
  ENUM_ENT(EM_TI_C5500,      "Texas Instruments TMS320C55x DSP family"),
  ENUM_ENT(EM_MMDSP_PLUS,    "STMicroelectronics 64bit VLIW Data Signal Processor"),
  ENUM_ENT(EM_CYPRESS_M8C,   "Cypress M8C microprocessor"),
  ENUM_ENT(EM_R32C,          "Renesas R32C series microprocessors"),
  ENUM_ENT(EM_TRIMEDIA,      "NXP Semiconductors TriMedia architecture family"),
  ENUM_ENT(EM_HEXAGON,       "Qualcomm Hexagon"),
  ENUM_ENT(EM_8051,          "Intel 8051 and variants"),
  ENUM_ENT(EM_STXP7X,        "STMicroelectronics STxP7x family"),
  ENUM_ENT(EM_NDS32,         "Andes Technology compact code size embedded RISC processor family"),
  ENUM_ENT(EM_ECOG1,         "Cyan Technology eCOG1 microprocessor"),
  ENUM_ENT(EM_ECOG1X,        "Cyan Technology eCOG1X family"),
  ENUM_ENT(EM_MAXQ30,        "Dallas Semiconductor MAXQ30 Core microcontrollers"),
  ENUM_ENT(EM_XIMO16,        "New Japan Radio (NJR) 16-bit DSP Processor"),
  ENUM_ENT(EM_MANIK,         "M2000 Reconfigurable RISC Microprocessor"),
  ENUM_ENT(EM_CRAYNV2,       "Cray Inc. NV2 vector architecture"),
  ENUM_ENT(EM_RX,            "Renesas RX"),
  ENUM_ENT(EM_METAG,         "Imagination Technologies Meta processor architecture"),
  ENUM_ENT(EM_MCST_ELBRUS,   "MCST Elbrus general purpose hardware architecture"),
  ENUM_ENT(EM_ECOG16,        "Cyan Technology eCOG16 family"),
  ENUM_ENT(EM_CR16,          "Xilinx MicroBlaze"),
  ENUM_ENT(EM_ETPU,          "Freescale Extended Time Processing Unit"),
  ENUM_ENT(EM_SLE9X,         "Infineon Technologies SLE9X core"),
  ENUM_ENT(EM_L10M,          "EM_L10M"),
  ENUM_ENT(EM_K10M,          "EM_K10M"),
  ENUM_ENT(EM_AARCH64,       "AArch64"),
  ENUM_ENT(EM_AVR32,         "Atmel AVR 8-bit microcontroller"),
  ENUM_ENT(EM_STM8,          "STMicroeletronics STM8 8-bit microcontroller"),
  ENUM_ENT(EM_TILE64,        "Tilera TILE64 multicore architecture family"),
  ENUM_ENT(EM_TILEPRO,       "Tilera TILEPro multicore architecture family"),
  ENUM_ENT(EM_CUDA,          "NVIDIA CUDA architecture"),
  ENUM_ENT(EM_TILEGX,        "Tilera TILE-Gx multicore architecture family"),
  ENUM_ENT(EM_CLOUDSHIELD,   "EM_CLOUDSHIELD"),
  ENUM_ENT(EM_COREA_1ST,     "EM_COREA_1ST"),
  ENUM_ENT(EM_COREA_2ND,     "EM_COREA_2ND"),
  ENUM_ENT(EM_ARC_COMPACT2,  "EM_ARC_COMPACT2"),
  ENUM_ENT(EM_OPEN8,         "EM_OPEN8"),
  ENUM_ENT(EM_RL78,          "Renesas RL78"),
  ENUM_ENT(EM_VIDEOCORE5,    "Broadcom VideoCore V processor"),
  ENUM_ENT(EM_78KOR,         "EM_78KOR"),
  ENUM_ENT(EM_56800EX,       "EM_56800EX"),
  ENUM_ENT(EM_AMDGPU,        "EM_AMDGPU"),
  ENUM_ENT(EM_WEBASSEMBLY,   "EM_WEBASSEMBLY")
};

static const EnumEntry<unsigned> ElfSymbolBindings[] = {
    {"Local",  "LOCAL",  ELF::STB_LOCAL},
    {"Global", "GLOBAL", ELF::STB_GLOBAL},
    {"Weak",   "WEAK",   ELF::STB_WEAK},
    {"Unique", "UNIQUE", ELF::STB_GNU_UNIQUE}};

static const EnumEntry<unsigned> ElfSymbolTypes[] = {
    {"None",      "NOTYPE",   ELF::STT_NOTYPE},
    {"Object",    "OBJECT",   ELF::STT_OBJECT},
    {"Function",  "FUNCTION", ELF::STT_FUNC},
    {"Section",   "SECTION",  ELF::STT_SECTION},
    {"File",      "FILE",     ELF::STT_FILE},
    {"Common",    "COMMON",   ELF::STT_COMMON},
    {"TLS",       "TLS",      ELF::STT_TLS},
    {"GNU_IFunc", "IFUNC",    ELF::STT_GNU_IFUNC}};

static const EnumEntry<unsigned> AMDGPUSymbolTypes[] = {
  { "AMDGPU_HSA_KERNEL",            ELF::STT_AMDGPU_HSA_KERNEL },
  { "AMDGPU_HSA_INDIRECT_FUNCTION", ELF::STT_AMDGPU_HSA_INDIRECT_FUNCTION },
  { "AMDGPU_HSA_METADATA",          ELF::STT_AMDGPU_HSA_METADATA }
};

static const char *getElfSectionType(unsigned Arch, unsigned Type) {
  switch (Arch) {
  case ELF::EM_ARM:
    switch (Type) {
    LLVM_READOBJ_ENUM_CASE(ELF, SHT_ARM_EXIDX);
    LLVM_READOBJ_ENUM_CASE(ELF, SHT_ARM_PREEMPTMAP);
    LLVM_READOBJ_ENUM_CASE(ELF, SHT_ARM_ATTRIBUTES);
    LLVM_READOBJ_ENUM_CASE(ELF, SHT_ARM_DEBUGOVERLAY);
    LLVM_READOBJ_ENUM_CASE(ELF, SHT_ARM_OVERLAYSECTION);
    }
  case ELF::EM_HEXAGON:
    switch (Type) { LLVM_READOBJ_ENUM_CASE(ELF, SHT_HEX_ORDERED); }
  case ELF::EM_X86_64:
    switch (Type) { LLVM_READOBJ_ENUM_CASE(ELF, SHT_X86_64_UNWIND); }
  case ELF::EM_MIPS:
  case ELF::EM_MIPS_RS3_LE:
    switch (Type) {
    LLVM_READOBJ_ENUM_CASE(ELF, SHT_MIPS_REGINFO);
    LLVM_READOBJ_ENUM_CASE(ELF, SHT_MIPS_OPTIONS);
    LLVM_READOBJ_ENUM_CASE(ELF, SHT_MIPS_ABIFLAGS);
    }
  }

  switch (Type) {
  LLVM_READOBJ_ENUM_CASE(ELF, SHT_NULL              );
  LLVM_READOBJ_ENUM_CASE(ELF, SHT_PROGBITS          );
  LLVM_READOBJ_ENUM_CASE(ELF, SHT_SYMTAB            );
  LLVM_READOBJ_ENUM_CASE(ELF, SHT_STRTAB            );
  LLVM_READOBJ_ENUM_CASE(ELF, SHT_RELA              );
  LLVM_READOBJ_ENUM_CASE(ELF, SHT_HASH              );
  LLVM_READOBJ_ENUM_CASE(ELF, SHT_DYNAMIC           );
  LLVM_READOBJ_ENUM_CASE(ELF, SHT_NOTE              );
  LLVM_READOBJ_ENUM_CASE(ELF, SHT_NOBITS            );
  LLVM_READOBJ_ENUM_CASE(ELF, SHT_REL               );
  LLVM_READOBJ_ENUM_CASE(ELF, SHT_SHLIB             );
  LLVM_READOBJ_ENUM_CASE(ELF, SHT_DYNSYM            );
  LLVM_READOBJ_ENUM_CASE(ELF, SHT_INIT_ARRAY        );
  LLVM_READOBJ_ENUM_CASE(ELF, SHT_FINI_ARRAY        );
  LLVM_READOBJ_ENUM_CASE(ELF, SHT_PREINIT_ARRAY     );
  LLVM_READOBJ_ENUM_CASE(ELF, SHT_GROUP             );
  LLVM_READOBJ_ENUM_CASE(ELF, SHT_SYMTAB_SHNDX      );
  LLVM_READOBJ_ENUM_CASE(ELF, SHT_GNU_ATTRIBUTES    );
  LLVM_READOBJ_ENUM_CASE(ELF, SHT_GNU_HASH          );
  LLVM_READOBJ_ENUM_CASE(ELF, SHT_GNU_verdef        );
  LLVM_READOBJ_ENUM_CASE(ELF, SHT_GNU_verneed       );
  LLVM_READOBJ_ENUM_CASE(ELF, SHT_GNU_versym        );
  default: return "";
  }
}

static const char *getGroupType(uint32_t Flag) {
  if (Flag & ELF::GRP_COMDAT)
    return "COMDAT";
  else
    return "(unknown)";
}

static const EnumEntry<unsigned> ElfSectionFlags[] = {
  ENUM_ENT(SHF_WRITE,            "W"),
  ENUM_ENT(SHF_ALLOC,            "A"),
  ENUM_ENT(SHF_EXCLUDE,          "E"),
  ENUM_ENT(SHF_EXECINSTR,        "X"),
  ENUM_ENT(SHF_MERGE,            "M"),
  ENUM_ENT(SHF_STRINGS,          "S"),
  ENUM_ENT(SHF_INFO_LINK,        "I"),
  ENUM_ENT(SHF_LINK_ORDER,       "L"),
  ENUM_ENT(SHF_OS_NONCONFORMING, "o"),
  ENUM_ENT(SHF_GROUP,            "G"),
  ENUM_ENT(SHF_TLS,              "T"),
  ENUM_ENT_1(XCORE_SHF_CP_SECTION),
  ENUM_ENT_1(XCORE_SHF_DP_SECTION),
};

static const EnumEntry<unsigned> ElfAMDGPUSectionFlags[] = {
  LLVM_READOBJ_ENUM_ENT(ELF, SHF_AMDGPU_HSA_GLOBAL),
  LLVM_READOBJ_ENUM_ENT(ELF, SHF_AMDGPU_HSA_READONLY),
  LLVM_READOBJ_ENUM_ENT(ELF, SHF_AMDGPU_HSA_CODE),
  LLVM_READOBJ_ENUM_ENT(ELF, SHF_AMDGPU_HSA_AGENT)
};

static const EnumEntry<unsigned> ElfHexagonSectionFlags[] = {
  LLVM_READOBJ_ENUM_ENT(ELF, SHF_HEX_GPREL)
};

static const EnumEntry<unsigned> ElfMipsSectionFlags[] = {
  LLVM_READOBJ_ENUM_ENT(ELF, SHF_MIPS_NODUPES),
  LLVM_READOBJ_ENUM_ENT(ELF, SHF_MIPS_NAMES  ),
  LLVM_READOBJ_ENUM_ENT(ELF, SHF_MIPS_LOCAL  ),
  LLVM_READOBJ_ENUM_ENT(ELF, SHF_MIPS_NOSTRIP),
  LLVM_READOBJ_ENUM_ENT(ELF, SHF_MIPS_GPREL  ),
  LLVM_READOBJ_ENUM_ENT(ELF, SHF_MIPS_MERGE  ),
  LLVM_READOBJ_ENUM_ENT(ELF, SHF_MIPS_ADDR   ),
  LLVM_READOBJ_ENUM_ENT(ELF, SHF_MIPS_STRING )
};

static const EnumEntry<unsigned> ElfX86_64SectionFlags[] = {
  LLVM_READOBJ_ENUM_ENT(ELF, SHF_X86_64_LARGE)
};

static const char *getElfSegmentType(unsigned Arch, unsigned Type) {
  // Check potentially overlapped processor-specific
  // program header type.
  switch (Arch) {
  case ELF::EM_AMDGPU:
    switch (Type) {
    LLVM_READOBJ_ENUM_CASE(ELF, PT_AMDGPU_HSA_LOAD_GLOBAL_PROGRAM);
    LLVM_READOBJ_ENUM_CASE(ELF, PT_AMDGPU_HSA_LOAD_GLOBAL_AGENT);
    LLVM_READOBJ_ENUM_CASE(ELF, PT_AMDGPU_HSA_LOAD_READONLY_AGENT);
    LLVM_READOBJ_ENUM_CASE(ELF, PT_AMDGPU_HSA_LOAD_CODE_AGENT);
    }
  case ELF::EM_ARM:
    switch (Type) {
    LLVM_READOBJ_ENUM_CASE(ELF, PT_ARM_EXIDX);
    }
  case ELF::EM_MIPS:
  case ELF::EM_MIPS_RS3_LE:
    switch (Type) {
    LLVM_READOBJ_ENUM_CASE(ELF, PT_MIPS_REGINFO);
    LLVM_READOBJ_ENUM_CASE(ELF, PT_MIPS_RTPROC);
    LLVM_READOBJ_ENUM_CASE(ELF, PT_MIPS_OPTIONS);
    LLVM_READOBJ_ENUM_CASE(ELF, PT_MIPS_ABIFLAGS);
    }
  }

  switch (Type) {
  LLVM_READOBJ_ENUM_CASE(ELF, PT_NULL   );
  LLVM_READOBJ_ENUM_CASE(ELF, PT_LOAD   );
  LLVM_READOBJ_ENUM_CASE(ELF, PT_DYNAMIC);
  LLVM_READOBJ_ENUM_CASE(ELF, PT_INTERP );
  LLVM_READOBJ_ENUM_CASE(ELF, PT_NOTE   );
  LLVM_READOBJ_ENUM_CASE(ELF, PT_SHLIB  );
  LLVM_READOBJ_ENUM_CASE(ELF, PT_PHDR   );
  LLVM_READOBJ_ENUM_CASE(ELF, PT_TLS    );

  LLVM_READOBJ_ENUM_CASE(ELF, PT_GNU_EH_FRAME);
  LLVM_READOBJ_ENUM_CASE(ELF, PT_SUNW_UNWIND);

  LLVM_READOBJ_ENUM_CASE(ELF, PT_GNU_STACK);
  LLVM_READOBJ_ENUM_CASE(ELF, PT_GNU_RELRO);
  default: return "";
  }
}

static const EnumEntry<unsigned> ElfSegmentFlags[] = {
  LLVM_READOBJ_ENUM_ENT(ELF, PF_X),
  LLVM_READOBJ_ENUM_ENT(ELF, PF_W),
  LLVM_READOBJ_ENUM_ENT(ELF, PF_R)
};

static const EnumEntry<unsigned> ElfHeaderMipsFlags[] = {
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_NOREORDER),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_PIC),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_CPIC),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_ABI2),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_32BITMODE),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_FP64),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_NAN2008),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_ABI_O32),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_ABI_O64),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_ABI_EABI32),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_ABI_EABI64),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_MACH_3900),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_MACH_4010),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_MACH_4100),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_MACH_4650),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_MACH_4120),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_MACH_4111),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_MACH_SB1),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_MACH_OCTEON),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_MACH_XLR),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_MACH_OCTEON2),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_MACH_OCTEON3),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_MACH_5400),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_MACH_5900),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_MACH_5500),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_MACH_9000),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_MACH_LS2E),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_MACH_LS2F),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_MACH_LS3A),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_MICROMIPS),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_ARCH_ASE_M16),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_ARCH_ASE_MDMX),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_ARCH_1),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_ARCH_2),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_ARCH_3),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_ARCH_4),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_ARCH_5),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_ARCH_32),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_ARCH_64),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_ARCH_32R2),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_ARCH_64R2),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_ARCH_32R6),
  LLVM_READOBJ_ENUM_ENT(ELF, EF_MIPS_ARCH_64R6)
};

template <typename ELFT>
ELFDumper<ELFT>::ELFDumper(const ELFFile<ELFT> *Obj, StreamWriter &Writer)
    : ObjDumper(Writer), Obj(Obj) {

  SmallVector<const Elf_Phdr *, 4> LoadSegments;
  for (const Elf_Phdr &Phdr : Obj->program_headers()) {
    if (Phdr.p_type == ELF::PT_DYNAMIC) {
      DynamicTable = createDRIFrom(&Phdr, sizeof(Elf_Dyn));
      continue;
    }
    if (Phdr.p_type != ELF::PT_LOAD || Phdr.p_filesz == 0)
      continue;
    LoadSegments.push_back(&Phdr);
  }

  for (const Elf_Shdr &Sec : Obj->sections()) {
    switch (Sec.sh_type) {
    case ELF::SHT_SYMTAB:
      if (DotSymtabSec != nullptr)
        reportError("Multilpe SHT_SYMTAB");
      DotSymtabSec = &Sec;
      break;
    case ELF::SHT_DYNAMIC: {
      if (DynamicTable.Addr == nullptr)
        DynamicTable = createDRIFrom(&Sec);
      const Elf_Shdr *DynStrSec = unwrapOrError(Obj->getSection(Sec.sh_link));
      DynamicStringTable = unwrapOrError(Obj->getStringTable(DynStrSec));
      break;
    }
    case ELF::SHT_DYNSYM:
      // The dynamic table does not contain the size of the dynamic symbol
      // table, so get that from the section table if present.
      DynSymRegion = createDRIFrom(&Sec);
      break;
    case ELF::SHT_SYMTAB_SHNDX: {
      ErrorOr<ArrayRef<Elf_Word>> TableOrErr = Obj->getSHNDXTable(Sec);
      error(TableOrErr.getError());
      ShndxTable = *TableOrErr;
      break;
    }
    case ELF::SHT_GNU_versym:
      if (dot_gnu_version_sec != nullptr)
        reportError("Multiple SHT_GNU_versym");
      dot_gnu_version_sec = &Sec;
      break;
    case ELF::SHT_GNU_verdef:
      if (dot_gnu_version_d_sec != nullptr)
        reportError("Multiple SHT_GNU_verdef");
      dot_gnu_version_d_sec = &Sec;
      break;
    case ELF::SHT_GNU_verneed:
      if (dot_gnu_version_r_sec != nullptr)
        reportError("Multilpe SHT_GNU_verneed");
      dot_gnu_version_r_sec = &Sec;
      break;
    }
  }

  parseDynamicTable(LoadSegments);

  if (opts::Output == opts::GNU)
    ELFDumperStyle.reset(new GNUStyle<ELFT>(Writer));
  else
    ELFDumperStyle.reset(new LLVMStyle<ELFT>(Writer));
}

template <typename ELFT>
void ELFDumper<ELFT>::parseDynamicTable(
    ArrayRef<const Elf_Phdr *> LoadSegments) {
  auto toMappedAddr = [&](uint64_t VAddr) -> const uint8_t * {
    const Elf_Phdr *const *I = std::upper_bound(
        LoadSegments.begin(), LoadSegments.end(), VAddr, compareAddr<ELFT>);
    if (I == LoadSegments.begin())
      return nullptr;
    --I;
    const Elf_Phdr &Phdr = **I;
    uint64_t Delta = VAddr - Phdr.p_vaddr;
    if (Delta >= Phdr.p_filesz)
      return nullptr;
    return Obj->base() + Phdr.p_offset + Delta;
  };

  uint64_t SONameOffset = 0;
  const char *StringTableBegin = nullptr;
  uint64_t StringTableSize = 0;
  for (const Elf_Dyn &Dyn : dynamic_table()) {
    switch (Dyn.d_tag) {
    case ELF::DT_HASH:
      HashTable =
          reinterpret_cast<const Elf_Hash *>(toMappedAddr(Dyn.getPtr()));
      break;
    case ELF::DT_GNU_HASH:
      GnuHashTable =
          reinterpret_cast<const Elf_GnuHash *>(toMappedAddr(Dyn.getPtr()));
      break;
    case ELF::DT_STRTAB:
      StringTableBegin = (const char *)toMappedAddr(Dyn.getPtr());
      break;
    case ELF::DT_STRSZ:
      StringTableSize = Dyn.getVal();
      break;
    case ELF::DT_SYMTAB:
      if (DynSymRegion.Addr)
        break;
      DynSymRegion.Addr = toMappedAddr(Dyn.getPtr());
      DynSymRegion.EntSize = sizeof(Elf_Sym);
      // Figure out the size once we have scanned the entire dynamic table.
      break;
    case ELF::DT_RELA:
      DynRelaRegion.Addr = toMappedAddr(Dyn.getPtr());
      break;
    case ELF::DT_RELASZ:
      DynRelaRegion.Size = Dyn.getVal();
      break;
    case ELF::DT_RELAENT:
      DynRelaRegion.EntSize = Dyn.getVal();
      break;
    case ELF::DT_SONAME:
      SONameOffset = Dyn.getVal();
      break;
    case ELF::DT_REL:
      DynRelRegion.Addr = toMappedAddr(Dyn.getPtr());
      break;
    case ELF::DT_RELSZ:
      DynRelRegion.Size = Dyn.getVal();
      break;
    case ELF::DT_RELENT:
      DynRelRegion.EntSize = Dyn.getVal();
      break;
    case ELF::DT_PLTREL:
      if (Dyn.getVal() == DT_REL)
        DynPLTRelRegion.EntSize =  sizeof(Elf_Rel);
      else if (Dyn.getVal() == DT_RELA)
        DynPLTRelRegion.EntSize = sizeof(Elf_Rela);
      else
        reportError(Twine("unknown DT_PLTREL value of ") +
                    Twine((uint64_t)Dyn.getVal()));
      break;
    case ELF::DT_JMPREL:
      DynPLTRelRegion.Addr = toMappedAddr(Dyn.getPtr());
      break;
    case ELF::DT_PLTRELSZ:
      DynPLTRelRegion.Size = Dyn.getVal();
      break;
    }
  }
  if (StringTableBegin)
    DynamicStringTable = StringRef(StringTableBegin, StringTableSize);
  if (SONameOffset)
    SOName = getDynamicString(SONameOffset);
  if (DynSymRegion.Addr && !DynSymRegion.Size) {
    // There was no section table entry for the dynamic section, and there is
    // no DT entry describing its size, so attempt to guess at its size.
    // Initally guess that it ends at the end of the file.
    const void *Start = DynSymRegion.Addr;
    const void *End = Obj->base() + Obj->getBufSize();

    // Check all the sections we know about.
    for (const Elf_Shdr &Sec : Obj->sections()) {
      const void *Addr = Obj->base() + Sec.sh_offset;
      if (Addr >= Start && Addr < End)
        End = Addr;
    }
    
    // Check all the dynamic regions we know about.
    auto CheckDRI = [&](DynRegionInfo DRI) {
      if (DRI.Addr >= Start && DRI.Addr < End)
        End = DRI.Addr;
    };

    CheckDRI(DynamicTable);
    CheckDRI(DynRelRegion);
    CheckDRI(DynRelaRegion);
    CheckDRI(DynPLTRelRegion);
    
    if (DynamicStringTable.data() >= Start && DynamicStringTable.data() < End)
      End = DynamicStringTable.data();

    // Scan to the first invalid symbol.
    auto SymI = reinterpret_cast<const Elf_Sym *>(Start);
    for (; ((const char *)SymI + sizeof(Elf_Sym)) <= End; ++SymI) {
      uint32_t NameOffset = SymI->st_name;
      if (SymI > Start && !NameOffset)
        break;
      if (NameOffset >= DynamicStringTable.size())
        break;
      uint16_t SectionIndex = SymI->st_shndx;
      if ((Obj->getNumSections() && SectionIndex >= Obj->getNumSections()) &&
          SectionIndex < SHN_LORESERVE)
        break;
    }
    End = SymI;
    DynSymRegion.Size = (const char *)End - (const char *)Start;
  }
}

template <typename ELFT>
typename ELFDumper<ELFT>::Elf_Rel_Range ELFDumper<ELFT>::dyn_rels() const {
  return DynRelRegion.getAsRange<Elf_Rel>();
}

template <typename ELFT>
typename ELFDumper<ELFT>::Elf_Rela_Range ELFDumper<ELFT>::dyn_relas() const {
  return DynRelaRegion.getAsRange<Elf_Rela>();
}

template<class ELFT>
void ELFDumper<ELFT>::printFileHeaders() {
  ELFDumperStyle->printFileHeaders(Obj);
}

template<class ELFT>
void ELFDumper<ELFT>::printSections() {
  ListScope SectionsD(W, "Sections");

  int SectionIndex = -1;
  for (const Elf_Shdr &Sec : Obj->sections()) {
    ++SectionIndex;

    StringRef Name = errorOrDefault(Obj->getSectionName(&Sec));

    DictScope SectionD(W, "Section");
    W.printNumber("Index", SectionIndex);
    W.printNumber("Name", Name, Sec.sh_name);
    W.printHex("Type",
               getElfSectionType(Obj->getHeader()->e_machine, Sec.sh_type),
               Sec.sh_type);
    std::vector<EnumEntry<unsigned>> SectionFlags(std::begin(ElfSectionFlags),
                                                  std::end(ElfSectionFlags));
    switch (Obj->getHeader()->e_machine) {
    case EM_AMDGPU:
      SectionFlags.insert(SectionFlags.end(), std::begin(ElfAMDGPUSectionFlags),
                          std::end(ElfAMDGPUSectionFlags));
      break;
    case EM_HEXAGON:
      SectionFlags.insert(SectionFlags.end(),
                          std::begin(ElfHexagonSectionFlags),
                          std::end(ElfHexagonSectionFlags));
      break;
    case EM_MIPS:
      SectionFlags.insert(SectionFlags.end(), std::begin(ElfMipsSectionFlags),
                          std::end(ElfMipsSectionFlags));
      break;
    case EM_X86_64:
      SectionFlags.insert(SectionFlags.end(), std::begin(ElfX86_64SectionFlags),
                          std::end(ElfX86_64SectionFlags));
      break;
    default:
      // Nothing to do.
      break;
    }
    W.printFlags("Flags", Sec.sh_flags, makeArrayRef(SectionFlags));
    W.printHex("Address", Sec.sh_addr);
    W.printHex("Offset", Sec.sh_offset);
    W.printNumber("Size", Sec.sh_size);
    W.printNumber("Link", Sec.sh_link);
    W.printNumber("Info", Sec.sh_info);
    W.printNumber("AddressAlignment", Sec.sh_addralign);
    W.printNumber("EntrySize", Sec.sh_entsize);

    if (opts::SectionRelocations) {
      ListScope D(W, "Relocations");
      printRelocations(&Sec);
    }

    if (opts::SectionSymbols) {
      ListScope D(W, "Symbols");
      const Elf_Shdr *Symtab = DotSymtabSec;
      ErrorOr<StringRef> StrTableOrErr = Obj->getStringTableForSymtab(*Symtab);
      error(StrTableOrErr.getError());
      StringRef StrTable = *StrTableOrErr;

      for (const Elf_Sym &Sym : Obj->symbols(Symtab)) {
        ErrorOr<const Elf_Shdr *> SymSec =
            Obj->getSection(&Sym, Symtab, ShndxTable);
        if (!SymSec)
          continue;
        if (*SymSec == &Sec)
          printSymbol(&Sym, Obj->symbol_begin(Symtab), StrTable, false);
      }
    }

    if (opts::SectionData && Sec.sh_type != ELF::SHT_NOBITS) {
      ArrayRef<uint8_t> Data = errorOrDefault(Obj->getSectionContents(&Sec));
      W.printBinaryBlock("SectionData",
                         StringRef((const char *)Data.data(), Data.size()));
    }
  }
}

template<class ELFT>
void ELFDumper<ELFT>::printRelocations() {
  ListScope D(W, "Relocations");

  int SectionNumber = -1;
  for (const Elf_Shdr &Sec : Obj->sections()) {
    ++SectionNumber;

    if (Sec.sh_type != ELF::SHT_REL && Sec.sh_type != ELF::SHT_RELA)
      continue;

    StringRef Name = errorOrDefault(Obj->getSectionName(&Sec));

    W.startLine() << "Section (" << SectionNumber << ") " << Name << " {\n";
    W.indent();

    printRelocations(&Sec);

    W.unindent();
    W.startLine() << "}\n";
  }
}

template <class ELFT> void ELFDumper<ELFT>::printDynamicRelocations() {
  if (DynRelRegion.Size && DynRelaRegion.Size)
    report_fatal_error("There are both REL and RELA dynamic relocations");
  W.startLine() << "Dynamic Relocations {\n";
  W.indent();
  if (DynRelaRegion.Size > 0)
    for (const Elf_Rela &Rela : dyn_relas())
      printDynamicRelocation(Rela);
  else
    for (const Elf_Rel &Rel : dyn_rels()) {
      Elf_Rela Rela;
      Rela.r_offset = Rel.r_offset;
      Rela.r_info = Rel.r_info;
      Rela.r_addend = 0;
      printDynamicRelocation(Rela);
    }
  if (DynPLTRelRegion.EntSize == sizeof(Elf_Rela))
    for (const Elf_Rela &Rela : DynPLTRelRegion.getAsRange<Elf_Rela>())
      printDynamicRelocation(Rela);
  else
    for (const Elf_Rel &Rel : DynPLTRelRegion.getAsRange<Elf_Rel>()) {
      Elf_Rela Rela;
      Rela.r_offset = Rel.r_offset;
      Rela.r_info = Rel.r_info;
      Rela.r_addend = 0;
      printDynamicRelocation(Rela);
    }
  W.unindent();
  W.startLine() << "}\n";
}

template <class ELFT>
void ELFDumper<ELFT>::printRelocations(const Elf_Shdr *Sec) {
  ErrorOr<const Elf_Shdr *> SymTabOrErr = Obj->getSection(Sec->sh_link);
  error(SymTabOrErr.getError());
  const Elf_Shdr *SymTab = *SymTabOrErr;

  switch (Sec->sh_type) {
  case ELF::SHT_REL:
    for (const Elf_Rel &R : Obj->rels(Sec)) {
      Elf_Rela Rela;
      Rela.r_offset = R.r_offset;
      Rela.r_info = R.r_info;
      Rela.r_addend = 0;
      printRelocation(Rela, SymTab);
    }
    break;
  case ELF::SHT_RELA:
    for (const Elf_Rela &R : Obj->relas(Sec))
      printRelocation(R, SymTab);
    break;
  }
}

template <class ELFT>
void ELFDumper<ELFT>::printRelocation(Elf_Rela Rel, const Elf_Shdr *SymTab) {
  SmallString<32> RelocName;
  Obj->getRelocationTypeName(Rel.getType(Obj->isMips64EL()), RelocName);
  StringRef TargetName;
  const Elf_Sym *Sym = Obj->getRelocationSymbol(&Rel, SymTab);
  if (Sym && Sym->getType() == ELF::STT_SECTION) {
    ErrorOr<const Elf_Shdr *> Sec = Obj->getSection(Sym, SymTab, ShndxTable);
    error(Sec.getError());
    ErrorOr<StringRef> SecName = Obj->getSectionName(*Sec);
    if (SecName)
      TargetName = SecName.get();
  } else if (Sym) {
    ErrorOr<StringRef> StrTableOrErr = Obj->getStringTableForSymtab(*SymTab);
    error(StrTableOrErr.getError());
    TargetName = errorOrDefault(Sym->getName(*StrTableOrErr));
  }

  if (opts::ExpandRelocs) {
    DictScope Group(W, "Relocation");
    W.printHex("Offset", Rel.r_offset);
    W.printNumber("Type", RelocName, (int)Rel.getType(Obj->isMips64EL()));
    W.printNumber("Symbol", TargetName.size() > 0 ? TargetName : "-",
                  Rel.getSymbol(Obj->isMips64EL()));
    W.printHex("Addend", Rel.r_addend);
  } else {
    raw_ostream& OS = W.startLine();
    OS << W.hex(Rel.r_offset) << " " << RelocName << " "
       << (TargetName.size() > 0 ? TargetName : "-") << " "
       << W.hex(Rel.r_addend) << "\n";
  }
}

template <class ELFT>
void ELFDumper<ELFT>::printDynamicRelocation(Elf_Rela Rel) {
  SmallString<32> RelocName;
  Obj->getRelocationTypeName(Rel.getType(Obj->isMips64EL()), RelocName);
  StringRef SymbolName;
  uint32_t SymIndex = Rel.getSymbol(Obj->isMips64EL());
  const Elf_Sym *Sym =
      DynSymRegion.getAsRange<Elf_Sym>().begin() + SymIndex;
  SymbolName = errorOrDefault(Sym->getName(DynamicStringTable));
  if (opts::ExpandRelocs) {
    DictScope Group(W, "Relocation");
    W.printHex("Offset", Rel.r_offset);
    W.printNumber("Type", RelocName, (int)Rel.getType(Obj->isMips64EL()));
    W.printString("Symbol", SymbolName.size() > 0 ? SymbolName : "-");
    W.printHex("Addend", Rel.r_addend);
  } else {
    raw_ostream &OS = W.startLine();
    OS << W.hex(Rel.r_offset) << " " << RelocName << " "
       << (SymbolName.size() > 0 ? SymbolName : "-") << " "
       << W.hex(Rel.r_addend) << "\n";
  }
}

template<class ELFT>
void ELFDumper<ELFT>::printSymbolsHelper(bool IsDynamic) {
  StringRef StrTable = DynamicStringTable;
  Elf_Sym_Range Syms(nullptr, nullptr);
  if (IsDynamic)
    Syms = DynSymRegion.getAsRange<Elf_Sym>();
  else {
    if (!DotSymtabSec)
      return;
    ErrorOr<StringRef> StrTableOrErr =
        Obj->getStringTableForSymtab(*DotSymtabSec);
    error(StrTableOrErr.getError());
    StrTable = *StrTableOrErr;
    Syms = Obj->symbols(DotSymtabSec);
  }
  for (const Elf_Sym &Sym : Syms)
    printSymbol(&Sym, Syms.begin(), StrTable, IsDynamic);
}

template<class ELFT>
void ELFDumper<ELFT>::printSymbols() {
  ListScope Group(W, "Symbols");
  printSymbolsHelper(false);
}

template<class ELFT>
void ELFDumper<ELFT>::printDynamicSymbols() {
  ListScope Group(W, "DynamicSymbols");
  printSymbolsHelper(true);
}

template <class ELFT>
void ELFDumper<ELFT>::printSymbol(const Elf_Sym *Symbol,
                                  const Elf_Sym *FirstSym, StringRef StrTable,
                                  bool IsDynamic) {
  unsigned SectionIndex = 0;
  StringRef SectionName;
  getSectionNameIndex(*Obj, Symbol, FirstSym, ShndxTable, SectionName,
                      SectionIndex);
  std::string FullSymbolName = getFullSymbolName(Symbol, StrTable, IsDynamic);
  unsigned char SymbolType = Symbol->getType();

  DictScope D(W, "Symbol");
  W.printNumber("Name", FullSymbolName, Symbol->st_name);
  W.printHex   ("Value", Symbol->st_value);
  W.printNumber("Size", Symbol->st_size);
  W.printEnum  ("Binding", Symbol->getBinding(),
                  makeArrayRef(ElfSymbolBindings));
  if (Obj->getHeader()->e_machine == ELF::EM_AMDGPU &&
      SymbolType >= ELF::STT_LOOS && SymbolType < ELF::STT_HIOS)
    W.printEnum  ("Type", SymbolType, makeArrayRef(AMDGPUSymbolTypes));
  else
    W.printEnum  ("Type", SymbolType, makeArrayRef(ElfSymbolTypes));
  W.printNumber("Other", Symbol->st_other);
  W.printHex("Section", SectionName, SectionIndex);
}

#define LLVM_READOBJ_TYPE_CASE(name) \
  case DT_##name: return #name

static const char *getTypeString(uint64_t Type) {
  switch (Type) {
  LLVM_READOBJ_TYPE_CASE(BIND_NOW);
  LLVM_READOBJ_TYPE_CASE(DEBUG);
  LLVM_READOBJ_TYPE_CASE(FINI);
  LLVM_READOBJ_TYPE_CASE(FINI_ARRAY);
  LLVM_READOBJ_TYPE_CASE(FINI_ARRAYSZ);
  LLVM_READOBJ_TYPE_CASE(FLAGS);
  LLVM_READOBJ_TYPE_CASE(FLAGS_1);
  LLVM_READOBJ_TYPE_CASE(HASH);
  LLVM_READOBJ_TYPE_CASE(INIT);
  LLVM_READOBJ_TYPE_CASE(INIT_ARRAY);
  LLVM_READOBJ_TYPE_CASE(INIT_ARRAYSZ);
  LLVM_READOBJ_TYPE_CASE(PREINIT_ARRAY);
  LLVM_READOBJ_TYPE_CASE(PREINIT_ARRAYSZ);
  LLVM_READOBJ_TYPE_CASE(JMPREL);
  LLVM_READOBJ_TYPE_CASE(NEEDED);
  LLVM_READOBJ_TYPE_CASE(NULL);
  LLVM_READOBJ_TYPE_CASE(PLTGOT);
  LLVM_READOBJ_TYPE_CASE(PLTREL);
  LLVM_READOBJ_TYPE_CASE(PLTRELSZ);
  LLVM_READOBJ_TYPE_CASE(REL);
  LLVM_READOBJ_TYPE_CASE(RELA);
  LLVM_READOBJ_TYPE_CASE(RELENT);
  LLVM_READOBJ_TYPE_CASE(RELSZ);
  LLVM_READOBJ_TYPE_CASE(RELAENT);
  LLVM_READOBJ_TYPE_CASE(RELASZ);
  LLVM_READOBJ_TYPE_CASE(RPATH);
  LLVM_READOBJ_TYPE_CASE(RUNPATH);
  LLVM_READOBJ_TYPE_CASE(SONAME);
  LLVM_READOBJ_TYPE_CASE(STRSZ);
  LLVM_READOBJ_TYPE_CASE(STRTAB);
  LLVM_READOBJ_TYPE_CASE(SYMBOLIC);
  LLVM_READOBJ_TYPE_CASE(SYMENT);
  LLVM_READOBJ_TYPE_CASE(SYMTAB);
  LLVM_READOBJ_TYPE_CASE(TEXTREL);
  LLVM_READOBJ_TYPE_CASE(VERDEF);
  LLVM_READOBJ_TYPE_CASE(VERDEFNUM);
  LLVM_READOBJ_TYPE_CASE(VERNEED);
  LLVM_READOBJ_TYPE_CASE(VERNEEDNUM);
  LLVM_READOBJ_TYPE_CASE(VERSYM);
  LLVM_READOBJ_TYPE_CASE(RELACOUNT);
  LLVM_READOBJ_TYPE_CASE(RELCOUNT);
  LLVM_READOBJ_TYPE_CASE(GNU_HASH);
  LLVM_READOBJ_TYPE_CASE(TLSDESC_PLT);
  LLVM_READOBJ_TYPE_CASE(TLSDESC_GOT);
  LLVM_READOBJ_TYPE_CASE(MIPS_RLD_VERSION);
  LLVM_READOBJ_TYPE_CASE(MIPS_RLD_MAP_REL);
  LLVM_READOBJ_TYPE_CASE(MIPS_FLAGS);
  LLVM_READOBJ_TYPE_CASE(MIPS_BASE_ADDRESS);
  LLVM_READOBJ_TYPE_CASE(MIPS_LOCAL_GOTNO);
  LLVM_READOBJ_TYPE_CASE(MIPS_SYMTABNO);
  LLVM_READOBJ_TYPE_CASE(MIPS_UNREFEXTNO);
  LLVM_READOBJ_TYPE_CASE(MIPS_GOTSYM);
  LLVM_READOBJ_TYPE_CASE(MIPS_RLD_MAP);
  LLVM_READOBJ_TYPE_CASE(MIPS_PLTGOT);
  LLVM_READOBJ_TYPE_CASE(MIPS_OPTIONS);
  default: return "unknown";
  }
}

#undef LLVM_READOBJ_TYPE_CASE

#define LLVM_READOBJ_DT_FLAG_ENT(prefix, enum) \
  { #enum, prefix##_##enum }

static const EnumEntry<unsigned> ElfDynamicDTFlags[] = {
  LLVM_READOBJ_DT_FLAG_ENT(DF, ORIGIN),
  LLVM_READOBJ_DT_FLAG_ENT(DF, SYMBOLIC),
  LLVM_READOBJ_DT_FLAG_ENT(DF, TEXTREL),
  LLVM_READOBJ_DT_FLAG_ENT(DF, BIND_NOW),
  LLVM_READOBJ_DT_FLAG_ENT(DF, STATIC_TLS)
};

static const EnumEntry<unsigned> ElfDynamicDTFlags1[] = {
  LLVM_READOBJ_DT_FLAG_ENT(DF_1, NOW),
  LLVM_READOBJ_DT_FLAG_ENT(DF_1, GLOBAL),
  LLVM_READOBJ_DT_FLAG_ENT(DF_1, GROUP),
  LLVM_READOBJ_DT_FLAG_ENT(DF_1, NODELETE),
  LLVM_READOBJ_DT_FLAG_ENT(DF_1, LOADFLTR),
  LLVM_READOBJ_DT_FLAG_ENT(DF_1, INITFIRST),
  LLVM_READOBJ_DT_FLAG_ENT(DF_1, NOOPEN),
  LLVM_READOBJ_DT_FLAG_ENT(DF_1, ORIGIN),
  LLVM_READOBJ_DT_FLAG_ENT(DF_1, DIRECT),
  LLVM_READOBJ_DT_FLAG_ENT(DF_1, TRANS),
  LLVM_READOBJ_DT_FLAG_ENT(DF_1, INTERPOSE),
  LLVM_READOBJ_DT_FLAG_ENT(DF_1, NODEFLIB),
  LLVM_READOBJ_DT_FLAG_ENT(DF_1, NODUMP),
  LLVM_READOBJ_DT_FLAG_ENT(DF_1, CONFALT),
  LLVM_READOBJ_DT_FLAG_ENT(DF_1, ENDFILTEE),
  LLVM_READOBJ_DT_FLAG_ENT(DF_1, DISPRELDNE),
  LLVM_READOBJ_DT_FLAG_ENT(DF_1, NODIRECT),
  LLVM_READOBJ_DT_FLAG_ENT(DF_1, IGNMULDEF),
  LLVM_READOBJ_DT_FLAG_ENT(DF_1, NOKSYMS),
  LLVM_READOBJ_DT_FLAG_ENT(DF_1, NOHDR),
  LLVM_READOBJ_DT_FLAG_ENT(DF_1, EDITED),
  LLVM_READOBJ_DT_FLAG_ENT(DF_1, NORELOC),
  LLVM_READOBJ_DT_FLAG_ENT(DF_1, SYMINTPOSE),
  LLVM_READOBJ_DT_FLAG_ENT(DF_1, GLOBAUDIT),
  LLVM_READOBJ_DT_FLAG_ENT(DF_1, SINGLETON)
};

static const EnumEntry<unsigned> ElfDynamicDTMipsFlags[] = {
  LLVM_READOBJ_DT_FLAG_ENT(RHF, NONE),
  LLVM_READOBJ_DT_FLAG_ENT(RHF, QUICKSTART),
  LLVM_READOBJ_DT_FLAG_ENT(RHF, NOTPOT),
  LLVM_READOBJ_DT_FLAG_ENT(RHS, NO_LIBRARY_REPLACEMENT),
  LLVM_READOBJ_DT_FLAG_ENT(RHF, NO_MOVE),
  LLVM_READOBJ_DT_FLAG_ENT(RHF, SGI_ONLY),
  LLVM_READOBJ_DT_FLAG_ENT(RHF, GUARANTEE_INIT),
  LLVM_READOBJ_DT_FLAG_ENT(RHF, DELTA_C_PLUS_PLUS),
  LLVM_READOBJ_DT_FLAG_ENT(RHF, GUARANTEE_START_INIT),
  LLVM_READOBJ_DT_FLAG_ENT(RHF, PIXIE),
  LLVM_READOBJ_DT_FLAG_ENT(RHF, DEFAULT_DELAY_LOAD),
  LLVM_READOBJ_DT_FLAG_ENT(RHF, REQUICKSTART),
  LLVM_READOBJ_DT_FLAG_ENT(RHF, REQUICKSTARTED),
  LLVM_READOBJ_DT_FLAG_ENT(RHF, CORD),
  LLVM_READOBJ_DT_FLAG_ENT(RHF, NO_UNRES_UNDEF),
  LLVM_READOBJ_DT_FLAG_ENT(RHF, RLD_ORDER_SAFE)
};

#undef LLVM_READOBJ_DT_FLAG_ENT

template <typename T, typename TFlag>
void printFlags(T Value, ArrayRef<EnumEntry<TFlag>> Flags, raw_ostream &OS) {
  typedef EnumEntry<TFlag> FlagEntry;
  typedef SmallVector<FlagEntry, 10> FlagVector;
  FlagVector SetFlags;

  for (const auto &Flag : Flags) {
    if (Flag.Value == 0)
      continue;

    if ((Value & Flag.Value) == Flag.Value)
      SetFlags.push_back(Flag);
  }

  for (const auto &Flag : SetFlags) {
    OS << Flag.Name << " ";
  }
}

template <class ELFT>
StringRef ELFDumper<ELFT>::getDynamicString(uint64_t Value) const {
  if (Value >= DynamicStringTable.size())
    reportError("Invalid dynamic string table reference");
  return StringRef(DynamicStringTable.data() + Value);
}

template <class ELFT>
void ELFDumper<ELFT>::printValue(uint64_t Type, uint64_t Value) {
  raw_ostream &OS = W.getOStream();
  switch (Type) {
  case DT_PLTREL:
    if (Value == DT_REL) {
      OS << "REL";
      break;
    } else if (Value == DT_RELA) {
      OS << "RELA";
      break;
    }
  // Fallthrough.
  case DT_PLTGOT:
  case DT_HASH:
  case DT_STRTAB:
  case DT_SYMTAB:
  case DT_RELA:
  case DT_INIT:
  case DT_FINI:
  case DT_REL:
  case DT_JMPREL:
  case DT_INIT_ARRAY:
  case DT_FINI_ARRAY:
  case DT_PREINIT_ARRAY:
  case DT_DEBUG:
  case DT_VERDEF:
  case DT_VERNEED:
  case DT_VERSYM:
  case DT_GNU_HASH:
  case DT_NULL:
  case DT_MIPS_BASE_ADDRESS:
  case DT_MIPS_GOTSYM:
  case DT_MIPS_RLD_MAP:
  case DT_MIPS_RLD_MAP_REL:
  case DT_MIPS_PLTGOT:
  case DT_MIPS_OPTIONS:
    OS << format("0x%" PRIX64, Value);
    break;
  case DT_RELACOUNT:
  case DT_RELCOUNT:
  case DT_VERDEFNUM:
  case DT_VERNEEDNUM:
  case DT_MIPS_RLD_VERSION:
  case DT_MIPS_LOCAL_GOTNO:
  case DT_MIPS_SYMTABNO:
  case DT_MIPS_UNREFEXTNO:
    OS << Value;
    break;
  case DT_PLTRELSZ:
  case DT_RELASZ:
  case DT_RELAENT:
  case DT_STRSZ:
  case DT_SYMENT:
  case DT_RELSZ:
  case DT_RELENT:
  case DT_INIT_ARRAYSZ:
  case DT_FINI_ARRAYSZ:
  case DT_PREINIT_ARRAYSZ:
    OS << Value << " (bytes)";
    break;
  case DT_NEEDED:
    OS << "SharedLibrary (" << getDynamicString(Value) << ")";
    break;
  case DT_SONAME:
    OS << "LibrarySoname (" << getDynamicString(Value) << ")";
    break;
  case DT_RPATH:
  case DT_RUNPATH:
    OS << getDynamicString(Value);
    break;
  case DT_MIPS_FLAGS:
    printFlags(Value, makeArrayRef(ElfDynamicDTMipsFlags), OS);
    break;
  case DT_FLAGS:
    printFlags(Value, makeArrayRef(ElfDynamicDTFlags), OS);
    break;
  case DT_FLAGS_1:
    printFlags(Value, makeArrayRef(ElfDynamicDTFlags1), OS);
    break;
  default:
    OS << format("0x%" PRIX64, Value);
    break;
  }
}

template<class ELFT>
void ELFDumper<ELFT>::printUnwindInfo() {
  W.startLine() << "UnwindInfo not implemented.\n";
}

namespace {
template <> void ELFDumper<ELFType<support::little, false>>::printUnwindInfo() {
  const unsigned Machine = Obj->getHeader()->e_machine;
  if (Machine == EM_ARM) {
    ARM::EHABI::PrinterContext<ELFType<support::little, false>> Ctx(
        W, Obj, DotSymtabSec);
    return Ctx.PrintUnwindInformation();
  }
  W.startLine() << "UnwindInfo not implemented.\n";
}
}

template<class ELFT>
void ELFDumper<ELFT>::printDynamicTable() {
  auto I = dynamic_table().begin();
  auto E = dynamic_table().end();

  if (I == E)
    return;

  --E;
  while (I != E && E->getTag() == ELF::DT_NULL)
    --E;
  if (E->getTag() != ELF::DT_NULL)
    ++E;
  ++E;

  ptrdiff_t Total = std::distance(I, E);
  if (Total == 0)
    return;

  raw_ostream &OS = W.getOStream();
  W.startLine() << "DynamicSection [ (" << Total << " entries)\n";

  bool Is64 = ELFT::Is64Bits;

  W.startLine()
     << "  Tag" << (Is64 ? "                " : "        ") << "Type"
     << "                 " << "Name/Value\n";
  while (I != E) {
    const Elf_Dyn &Entry = *I;
    uintX_t Tag = Entry.getTag();
    ++I;
    W.startLine() << "  " << format_hex(Tag, Is64 ? 18 : 10, true) << " "
                  << format("%-21s", getTypeString(Tag));
    printValue(Tag, Entry.getVal());
    OS << "\n";
  }

  W.startLine() << "]\n";
}

template<class ELFT>
void ELFDumper<ELFT>::printNeededLibraries() {
  ListScope D(W, "NeededLibraries");

  typedef std::vector<StringRef> LibsTy;
  LibsTy Libs;

  for (const auto &Entry : dynamic_table())
    if (Entry.d_tag == ELF::DT_NEEDED)
      Libs.push_back(getDynamicString(Entry.d_un.d_val));

  std::stable_sort(Libs.begin(), Libs.end());

  for (const auto &L : Libs) {
    outs() << "  " << L << "\n";
  }
}

template<class ELFT>
void ELFDumper<ELFT>::printProgramHeaders() {
  ListScope L(W, "ProgramHeaders");

  for (const Elf_Phdr &Phdr : Obj->program_headers()) {
    DictScope P(W, "ProgramHeader");
    W.printHex("Type",
               getElfSegmentType(Obj->getHeader()->e_machine, Phdr.p_type),
               Phdr.p_type);
    W.printHex("Offset", Phdr.p_offset);
    W.printHex("VirtualAddress", Phdr.p_vaddr);
    W.printHex("PhysicalAddress", Phdr.p_paddr);
    W.printNumber("FileSize", Phdr.p_filesz);
    W.printNumber("MemSize", Phdr.p_memsz);
    W.printFlags("Flags", Phdr.p_flags, makeArrayRef(ElfSegmentFlags));
    W.printNumber("Alignment", Phdr.p_align);
  }
}

template <typename ELFT>
void ELFDumper<ELFT>::printHashTable() {
  DictScope D(W, "HashTable");
  if (!HashTable)
    return;
  W.printNumber("Num Buckets", HashTable->nbucket);
  W.printNumber("Num Chains", HashTable->nchain);
  W.printList("Buckets", HashTable->buckets());
  W.printList("Chains", HashTable->chains());
}

template <typename ELFT>
void ELFDumper<ELFT>::printGnuHashTable() {
  DictScope D(W, "GnuHashTable");
  if (!GnuHashTable)
    return;
  W.printNumber("Num Buckets", GnuHashTable->nbuckets);
  W.printNumber("First Hashed Symbol Index", GnuHashTable->symndx);
  W.printNumber("Num Mask Words", GnuHashTable->maskwords);
  W.printNumber("Shift Count", GnuHashTable->shift2);
  W.printHexList("Bloom Filter", GnuHashTable->filter());
  W.printList("Buckets", GnuHashTable->buckets());
  if (!DynSymRegion.Size || !DynSymRegion.EntSize)
    reportError("No dynamic symbol section");
  W.printHexList(
      "Values", GnuHashTable->values(DynSymRegion.Size / DynSymRegion.EntSize));
}

template <typename ELFT> void ELFDumper<ELFT>::printLoadName() {
  outs() << "LoadName: " << SOName << '\n';
}

template <class ELFT>
void ELFDumper<ELFT>::printAttributes() {
  W.startLine() << "Attributes not implemented.\n";
}

namespace {
template <> void ELFDumper<ELFType<support::little, false>>::printAttributes() {
  if (Obj->getHeader()->e_machine != EM_ARM) {
    W.startLine() << "Attributes not implemented.\n";
    return;
  }

  DictScope BA(W, "BuildAttributes");
  for (const ELFO::Elf_Shdr &Sec : Obj->sections()) {
    if (Sec.sh_type != ELF::SHT_ARM_ATTRIBUTES)
      continue;

    ErrorOr<ArrayRef<uint8_t>> Contents = Obj->getSectionContents(&Sec);
    if (!Contents)
      continue;

    if ((*Contents)[0] != ARMBuildAttrs::Format_Version) {
      errs() << "unrecognised FormatVersion: 0x" << utohexstr((*Contents)[0])
             << '\n';
      continue;
    }

    W.printHex("FormatVersion", (*Contents)[0]);
    if (Contents->size() == 1)
      continue;

    ARMAttributeParser(W).Parse(*Contents);
  }
}
}

namespace {
template <class ELFT> class MipsGOTParser {
public:
  typedef object::ELFFile<ELFT> ELFO;
  typedef typename ELFO::Elf_Shdr Elf_Shdr;
  typedef typename ELFO::Elf_Sym Elf_Sym;
  typedef typename ELFO::Elf_Dyn_Range Elf_Dyn_Range;
  typedef typename ELFO::Elf_Addr GOTEntry;
  typedef typename ELFO::Elf_Rel Elf_Rel;
  typedef typename ELFO::Elf_Rela Elf_Rela;

  MipsGOTParser(ELFDumper<ELFT> *Dumper, const ELFO *Obj,
                Elf_Dyn_Range DynTable, StreamWriter &W);

  void parseGOT();
  void parsePLT();

private:
  ELFDumper<ELFT> *Dumper;
  const ELFO *Obj;
  StreamWriter &W;
  llvm::Optional<uint64_t> DtPltGot;
  llvm::Optional<uint64_t> DtLocalGotNum;
  llvm::Optional<uint64_t> DtGotSym;
  llvm::Optional<uint64_t> DtMipsPltGot;
  llvm::Optional<uint64_t> DtJmpRel;

  std::size_t getGOTTotal(ArrayRef<uint8_t> GOT) const;
  const GOTEntry *makeGOTIter(ArrayRef<uint8_t> GOT, std::size_t EntryNum);

  void printGotEntry(uint64_t GotAddr, const GOTEntry *BeginIt,
                     const GOTEntry *It);
  void printGlobalGotEntry(uint64_t GotAddr, const GOTEntry *BeginIt,
                           const GOTEntry *It, const Elf_Sym *Sym,
                           StringRef StrTable, bool IsDynamic);
  void printPLTEntry(uint64_t PLTAddr, const GOTEntry *BeginIt,
                     const GOTEntry *It, StringRef Purpose);
  void printPLTEntry(uint64_t PLTAddr, const GOTEntry *BeginIt,
                     const GOTEntry *It, StringRef StrTable,
                     const Elf_Sym *Sym);
};
}

template <class ELFT>
MipsGOTParser<ELFT>::MipsGOTParser(ELFDumper<ELFT> *Dumper, const ELFO *Obj,
                                   Elf_Dyn_Range DynTable, StreamWriter &W)
    : Dumper(Dumper), Obj(Obj), W(W) {
  for (const auto &Entry : DynTable) {
    switch (Entry.getTag()) {
    case ELF::DT_PLTGOT:
      DtPltGot = Entry.getVal();
      break;
    case ELF::DT_MIPS_LOCAL_GOTNO:
      DtLocalGotNum = Entry.getVal();
      break;
    case ELF::DT_MIPS_GOTSYM:
      DtGotSym = Entry.getVal();
      break;
    case ELF::DT_MIPS_PLTGOT:
      DtMipsPltGot = Entry.getVal();
      break;
    case ELF::DT_JMPREL:
      DtJmpRel = Entry.getVal();
      break;
    }
  }
}

template <class ELFT> void MipsGOTParser<ELFT>::parseGOT() {
  // See "Global Offset Table" in Chapter 5 in the following document
  // for detailed GOT description.
  // ftp://www.linux-mips.org/pub/linux/mips/doc/ABI/mipsabi.pdf
  if (!DtPltGot) {
    W.startLine() << "Cannot find PLTGOT dynamic table tag.\n";
    return;
  }
  if (!DtLocalGotNum) {
    W.startLine() << "Cannot find MIPS_LOCAL_GOTNO dynamic table tag.\n";
    return;
  }
  if (!DtGotSym) {
    W.startLine() << "Cannot find MIPS_GOTSYM dynamic table tag.\n";
    return;
  }

  StringRef StrTable = Dumper->getDynamicStringTable();
  const Elf_Sym *DynSymBegin = Dumper->dynamic_symbols().begin();
  const Elf_Sym *DynSymEnd = Dumper->dynamic_symbols().end();
  std::size_t DynSymTotal = std::size_t(std::distance(DynSymBegin, DynSymEnd));

  if (*DtGotSym > DynSymTotal)
    report_fatal_error("MIPS_GOTSYM exceeds a number of dynamic symbols");

  std::size_t GlobalGotNum = DynSymTotal - *DtGotSym;

  if (*DtLocalGotNum + GlobalGotNum == 0) {
    W.startLine() << "GOT is empty.\n";
    return;
  }

  const Elf_Shdr *GOTShdr = findNotEmptySectionByAddress(Obj, *DtPltGot);
  if (!GOTShdr)
    report_fatal_error("There is no not empty GOT section at 0x" +
                       Twine::utohexstr(*DtPltGot));

  ErrorOr<ArrayRef<uint8_t>> GOT = Obj->getSectionContents(GOTShdr);

  if (*DtLocalGotNum + GlobalGotNum > getGOTTotal(*GOT))
    report_fatal_error("Number of GOT entries exceeds the size of GOT section");

  const GOTEntry *GotBegin = makeGOTIter(*GOT, 0);
  const GOTEntry *GotLocalEnd = makeGOTIter(*GOT, *DtLocalGotNum);
  const GOTEntry *It = GotBegin;

  DictScope GS(W, "Primary GOT");

  W.printHex("Canonical gp value", GOTShdr->sh_addr + 0x7ff0);
  {
    ListScope RS(W, "Reserved entries");

    {
      DictScope D(W, "Entry");
      printGotEntry(GOTShdr->sh_addr, GotBegin, It++);
      W.printString("Purpose", StringRef("Lazy resolver"));
    }

    if (It != GotLocalEnd && (*It >> (sizeof(GOTEntry) * 8 - 1)) != 0) {
      DictScope D(W, "Entry");
      printGotEntry(GOTShdr->sh_addr, GotBegin, It++);
      W.printString("Purpose", StringRef("Module pointer (GNU extension)"));
    }
  }
  {
    ListScope LS(W, "Local entries");
    for (; It != GotLocalEnd; ++It) {
      DictScope D(W, "Entry");
      printGotEntry(GOTShdr->sh_addr, GotBegin, It);
    }
  }
  {
    ListScope GS(W, "Global entries");

    const GOTEntry *GotGlobalEnd =
        makeGOTIter(*GOT, *DtLocalGotNum + GlobalGotNum);
    const Elf_Sym *GotDynSym = DynSymBegin + *DtGotSym;
    for (; It != GotGlobalEnd; ++It) {
      DictScope D(W, "Entry");
      printGlobalGotEntry(GOTShdr->sh_addr, GotBegin, It, GotDynSym++, StrTable,
                          true);
    }
  }

  std::size_t SpecGotNum = getGOTTotal(*GOT) - *DtLocalGotNum - GlobalGotNum;
  W.printNumber("Number of TLS and multi-GOT entries", uint64_t(SpecGotNum));
}

template <class ELFT> void MipsGOTParser<ELFT>::parsePLT() {
  if (!DtMipsPltGot) {
    W.startLine() << "Cannot find MIPS_PLTGOT dynamic table tag.\n";
    return;
  }
  if (!DtJmpRel) {
    W.startLine() << "Cannot find JMPREL dynamic table tag.\n";
    return;
  }

  const Elf_Shdr *PLTShdr = findNotEmptySectionByAddress(Obj, *DtMipsPltGot);
  if (!PLTShdr)
    report_fatal_error("There is no not empty PLTGOT section at 0x " +
                       Twine::utohexstr(*DtMipsPltGot));
  ErrorOr<ArrayRef<uint8_t>> PLT = Obj->getSectionContents(PLTShdr);

  const Elf_Shdr *PLTRelShdr = findNotEmptySectionByAddress(Obj, *DtJmpRel);
  if (!PLTRelShdr)
    report_fatal_error("There is no not empty RELPLT section at 0x" +
                       Twine::utohexstr(*DtJmpRel));
  ErrorOr<const Elf_Shdr *> SymTableOrErr =
      Obj->getSection(PLTRelShdr->sh_link);
  error(SymTableOrErr.getError());
  const Elf_Shdr *SymTable = *SymTableOrErr;
  ErrorOr<StringRef> StrTable = Obj->getStringTableForSymtab(*SymTable);
  error(StrTable.getError());

  const GOTEntry *PLTBegin = makeGOTIter(*PLT, 0);
  const GOTEntry *PLTEnd = makeGOTIter(*PLT, getGOTTotal(*PLT));
  const GOTEntry *It = PLTBegin;

  DictScope GS(W, "PLT GOT");
  {
    ListScope RS(W, "Reserved entries");
    printPLTEntry(PLTShdr->sh_addr, PLTBegin, It++, "PLT lazy resolver");
    if (It != PLTEnd)
      printPLTEntry(PLTShdr->sh_addr, PLTBegin, It++, "Module pointer");
  }
  {
    ListScope GS(W, "Entries");

    switch (PLTRelShdr->sh_type) {
    case ELF::SHT_REL:
      for (const Elf_Rel *RI = Obj->rel_begin(PLTRelShdr),
                         *RE = Obj->rel_end(PLTRelShdr);
           RI != RE && It != PLTEnd; ++RI, ++It) {
        const Elf_Sym *Sym = Obj->getRelocationSymbol(&*RI, SymTable);
        printPLTEntry(PLTShdr->sh_addr, PLTBegin, It, *StrTable, Sym);
      }
      break;
    case ELF::SHT_RELA:
      for (const Elf_Rela *RI = Obj->rela_begin(PLTRelShdr),
                          *RE = Obj->rela_end(PLTRelShdr);
           RI != RE && It != PLTEnd; ++RI, ++It) {
        const Elf_Sym *Sym = Obj->getRelocationSymbol(&*RI, SymTable);
        printPLTEntry(PLTShdr->sh_addr, PLTBegin, It, *StrTable, Sym);
      }
      break;
    }
  }
}

template <class ELFT>
std::size_t MipsGOTParser<ELFT>::getGOTTotal(ArrayRef<uint8_t> GOT) const {
  return GOT.size() / sizeof(GOTEntry);
}

template <class ELFT>
const typename MipsGOTParser<ELFT>::GOTEntry *
MipsGOTParser<ELFT>::makeGOTIter(ArrayRef<uint8_t> GOT, std::size_t EntryNum) {
  const char *Data = reinterpret_cast<const char *>(GOT.data());
  return reinterpret_cast<const GOTEntry *>(Data + EntryNum * sizeof(GOTEntry));
}

template <class ELFT>
void MipsGOTParser<ELFT>::printGotEntry(uint64_t GotAddr,
                                        const GOTEntry *BeginIt,
                                        const GOTEntry *It) {
  int64_t Offset = std::distance(BeginIt, It) * sizeof(GOTEntry);
  W.printHex("Address", GotAddr + Offset);
  W.printNumber("Access", Offset - 0x7ff0);
  W.printHex("Initial", *It);
}

template <class ELFT>
void MipsGOTParser<ELFT>::printGlobalGotEntry(
    uint64_t GotAddr, const GOTEntry *BeginIt, const GOTEntry *It,
    const Elf_Sym *Sym, StringRef StrTable, bool IsDynamic) {
  printGotEntry(GotAddr, BeginIt, It);

  W.printHex("Value", Sym->st_value);
  W.printEnum("Type", Sym->getType(), makeArrayRef(ElfSymbolTypes));

  unsigned SectionIndex = 0;
  StringRef SectionName;
  getSectionNameIndex(*Obj, Sym, Dumper->dynamic_symbols().begin(),
                      Dumper->getShndxTable(), SectionName, SectionIndex);
  W.printHex("Section", SectionName, SectionIndex);

  std::string FullSymbolName =
      Dumper->getFullSymbolName(Sym, StrTable, IsDynamic);
  W.printNumber("Name", FullSymbolName, Sym->st_name);
}

template <class ELFT>
void MipsGOTParser<ELFT>::printPLTEntry(uint64_t PLTAddr,
                                        const GOTEntry *BeginIt,
                                        const GOTEntry *It, StringRef Purpose) {
  DictScope D(W, "Entry");
  int64_t Offset = std::distance(BeginIt, It) * sizeof(GOTEntry);
  W.printHex("Address", PLTAddr + Offset);
  W.printHex("Initial", *It);
  W.printString("Purpose", Purpose);
}

template <class ELFT>
void MipsGOTParser<ELFT>::printPLTEntry(uint64_t PLTAddr,
                                        const GOTEntry *BeginIt,
                                        const GOTEntry *It, StringRef StrTable,
                                        const Elf_Sym *Sym) {
  DictScope D(W, "Entry");
  int64_t Offset = std::distance(BeginIt, It) * sizeof(GOTEntry);
  W.printHex("Address", PLTAddr + Offset);
  W.printHex("Initial", *It);
  W.printHex("Value", Sym->st_value);
  W.printEnum("Type", Sym->getType(), makeArrayRef(ElfSymbolTypes));

  unsigned SectionIndex = 0;
  StringRef SectionName;
  getSectionNameIndex(*Obj, Sym, Dumper->dynamic_symbols().begin(),
                      Dumper->getShndxTable(), SectionName, SectionIndex);
  W.printHex("Section", SectionName, SectionIndex);

  std::string FullSymbolName = Dumper->getFullSymbolName(Sym, StrTable, true);
  W.printNumber("Name", FullSymbolName, Sym->st_name);
}

template <class ELFT> void ELFDumper<ELFT>::printMipsPLTGOT() {
  if (Obj->getHeader()->e_machine != EM_MIPS) {
    W.startLine() << "MIPS PLT GOT is available for MIPS targets only.\n";
    return;
  }

  MipsGOTParser<ELFT> GOTParser(this, Obj, dynamic_table(), W);
  GOTParser.parseGOT();
  GOTParser.parsePLT();
}

static const EnumEntry<unsigned> ElfMipsISAExtType[] = {
  {"None",                    Mips::AFL_EXT_NONE},
  {"Broadcom SB-1",           Mips::AFL_EXT_SB1},
  {"Cavium Networks Octeon",  Mips::AFL_EXT_OCTEON},
  {"Cavium Networks Octeon2", Mips::AFL_EXT_OCTEON2},
  {"Cavium Networks OcteonP", Mips::AFL_EXT_OCTEONP},
  {"Cavium Networks Octeon3", Mips::AFL_EXT_OCTEON3},
  {"LSI R4010",               Mips::AFL_EXT_4010},
  {"Loongson 2E",             Mips::AFL_EXT_LOONGSON_2E},
  {"Loongson 2F",             Mips::AFL_EXT_LOONGSON_2F},
  {"Loongson 3A",             Mips::AFL_EXT_LOONGSON_3A},
  {"MIPS R4650",              Mips::AFL_EXT_4650},
  {"MIPS R5900",              Mips::AFL_EXT_5900},
  {"MIPS R10000",             Mips::AFL_EXT_10000},
  {"NEC VR4100",              Mips::AFL_EXT_4100},
  {"NEC VR4111/VR4181",       Mips::AFL_EXT_4111},
  {"NEC VR4120",              Mips::AFL_EXT_4120},
  {"NEC VR5400",              Mips::AFL_EXT_5400},
  {"NEC VR5500",              Mips::AFL_EXT_5500},
  {"RMI Xlr",                 Mips::AFL_EXT_XLR},
  {"Toshiba R3900",           Mips::AFL_EXT_3900}
};

static const EnumEntry<unsigned> ElfMipsASEFlags[] = {
  {"DSP",                Mips::AFL_ASE_DSP},
  {"DSPR2",              Mips::AFL_ASE_DSPR2},
  {"Enhanced VA Scheme", Mips::AFL_ASE_EVA},
  {"MCU",                Mips::AFL_ASE_MCU},
  {"MDMX",               Mips::AFL_ASE_MDMX},
  {"MIPS-3D",            Mips::AFL_ASE_MIPS3D},
  {"MT",                 Mips::AFL_ASE_MT},
  {"SmartMIPS",          Mips::AFL_ASE_SMARTMIPS},
  {"VZ",                 Mips::AFL_ASE_VIRT},
  {"MSA",                Mips::AFL_ASE_MSA},
  {"MIPS16",             Mips::AFL_ASE_MIPS16},
  {"microMIPS",          Mips::AFL_ASE_MICROMIPS},
  {"XPA",                Mips::AFL_ASE_XPA}
};

static const EnumEntry<unsigned> ElfMipsFpABIType[] = {
  {"Hard or soft float",                  Mips::Val_GNU_MIPS_ABI_FP_ANY},
  {"Hard float (double precision)",       Mips::Val_GNU_MIPS_ABI_FP_DOUBLE},
  {"Hard float (single precision)",       Mips::Val_GNU_MIPS_ABI_FP_SINGLE},
  {"Soft float",                          Mips::Val_GNU_MIPS_ABI_FP_SOFT},
  {"Hard float (MIPS32r2 64-bit FPU 12 callee-saved)",
   Mips::Val_GNU_MIPS_ABI_FP_OLD_64},
  {"Hard float (32-bit CPU, Any FPU)",    Mips::Val_GNU_MIPS_ABI_FP_XX},
  {"Hard float (32-bit CPU, 64-bit FPU)", Mips::Val_GNU_MIPS_ABI_FP_64},
  {"Hard float compat (32-bit CPU, 64-bit FPU)",
   Mips::Val_GNU_MIPS_ABI_FP_64A}
};

static const EnumEntry<unsigned> ElfMipsFlags1[] {
  {"ODDSPREG", Mips::AFL_FLAGS1_ODDSPREG},
};

static int getMipsRegisterSize(uint8_t Flag) {
  switch (Flag) {
  case Mips::AFL_REG_NONE:
    return 0;
  case Mips::AFL_REG_32:
    return 32;
  case Mips::AFL_REG_64:
    return 64;
  case Mips::AFL_REG_128:
    return 128;
  default:
    return -1;
  }
}

template <class ELFT> void ELFDumper<ELFT>::printMipsABIFlags() {
  const Elf_Shdr *Shdr = findSectionByName(*Obj, ".MIPS.abiflags");
  if (!Shdr) {
    W.startLine() << "There is no .MIPS.abiflags section in the file.\n";
    return;
  }
  ErrorOr<ArrayRef<uint8_t>> Sec = Obj->getSectionContents(Shdr);
  if (!Sec) {
    W.startLine() << "The .MIPS.abiflags section is empty.\n";
    return;
  }
  if (Sec->size() != sizeof(Elf_Mips_ABIFlags<ELFT>)) {
    W.startLine() << "The .MIPS.abiflags section has a wrong size.\n";
    return;
  }

  auto *Flags = reinterpret_cast<const Elf_Mips_ABIFlags<ELFT> *>(Sec->data());

  raw_ostream &OS = W.getOStream();
  DictScope GS(W, "MIPS ABI Flags");

  W.printNumber("Version", Flags->version);
  W.startLine() << "ISA: ";
  if (Flags->isa_rev <= 1)
    OS << format("MIPS%u", Flags->isa_level);
  else
    OS << format("MIPS%ur%u", Flags->isa_level, Flags->isa_rev);
  OS << "\n";
  W.printEnum("ISA Extension", Flags->isa_ext, makeArrayRef(ElfMipsISAExtType));
  W.printFlags("ASEs", Flags->ases, makeArrayRef(ElfMipsASEFlags));
  W.printEnum("FP ABI", Flags->fp_abi, makeArrayRef(ElfMipsFpABIType));
  W.printNumber("GPR size", getMipsRegisterSize(Flags->gpr_size));
  W.printNumber("CPR1 size", getMipsRegisterSize(Flags->cpr1_size));
  W.printNumber("CPR2 size", getMipsRegisterSize(Flags->cpr2_size));
  W.printFlags("Flags 1", Flags->flags1, makeArrayRef(ElfMipsFlags1));
  W.printHex("Flags 2", Flags->flags2);
}

template <class ELFT> void ELFDumper<ELFT>::printMipsReginfo() {
  const Elf_Shdr *Shdr = findSectionByName(*Obj, ".reginfo");
  if (!Shdr) {
    W.startLine() << "There is no .reginfo section in the file.\n";
    return;
  }
  ErrorOr<ArrayRef<uint8_t>> Sec = Obj->getSectionContents(Shdr);
  if (!Sec) {
    W.startLine() << "The .reginfo section is empty.\n";
    return;
  }
  if (Sec->size() != sizeof(Elf_Mips_RegInfo<ELFT>)) {
    W.startLine() << "The .reginfo section has a wrong size.\n";
    return;
  }

  auto *Reginfo = reinterpret_cast<const Elf_Mips_RegInfo<ELFT> *>(Sec->data());

  DictScope GS(W, "MIPS RegInfo");
  W.printHex("GP", Reginfo->ri_gp_value);
  W.printHex("General Mask", Reginfo->ri_gprmask);
  W.printHex("Co-Proc Mask0", Reginfo->ri_cprmask[0]);
  W.printHex("Co-Proc Mask1", Reginfo->ri_cprmask[1]);
  W.printHex("Co-Proc Mask2", Reginfo->ri_cprmask[2]);
  W.printHex("Co-Proc Mask3", Reginfo->ri_cprmask[3]);
}

template <class ELFT> void ELFDumper<ELFT>::printStackMap() const {
  const Elf_Shdr *StackMapSection = nullptr;
  for (const auto &Sec : Obj->sections()) {
    ErrorOr<StringRef> Name = Obj->getSectionName(&Sec);
    if (*Name == ".llvm_stackmaps") {
      StackMapSection = &Sec;
      break;
    }
  }

  if (!StackMapSection)
    return;

  StringRef StackMapContents;
  ErrorOr<ArrayRef<uint8_t>> StackMapContentsArray =
    Obj->getSectionContents(StackMapSection);

  prettyPrintStackMap(
              llvm::outs(),
              StackMapV1Parser<ELFT::TargetEndianness>(*StackMapContentsArray));
}

template <class ELFT> void ELFDumper<ELFT>::printGroupSections() {
  DictScope Lists(W, "Groups");
  uint32_t SectionIndex = 0;
  bool HasGroups = false;
  for (const Elf_Shdr &Sec : Obj->sections()) {
    if (Sec.sh_type == ELF::SHT_GROUP) {
      HasGroups = true;
      ErrorOr<const Elf_Shdr *> Symtab =
          errorOrDefault(Obj->getSection(Sec.sh_link));
      ErrorOr<StringRef> StrTableOrErr = Obj->getStringTableForSymtab(**Symtab);
      error(StrTableOrErr.getError());
      StringRef StrTable = *StrTableOrErr;
      const Elf_Sym *Sym =
          Obj->template getEntry<Elf_Sym>(*Symtab, Sec.sh_info);
      auto Data = errorOrDefault(
          Obj->template getSectionContentsAsArray<Elf_Word>(&Sec));
      DictScope D(W, "Group");
      StringRef Name = errorOrDefault(Obj->getSectionName(&Sec));
      W.printNumber("Name", Name, Sec.sh_name);
      W.printNumber("Index", SectionIndex);
      W.printHex("Type", getGroupType(Data[0]), Data[0]);
      W.startLine() << "Signature: " << StrTable.data() + Sym->st_name << "\n";
      {
        ListScope L(W, "Section(s) in group");
        size_t Member = 1;
        while (Member < Data.size()) {
          auto Sec = errorOrDefault(Obj->getSection(Data[Member]));
          const StringRef Name = errorOrDefault(Obj->getSectionName(Sec));
          W.startLine() << Name << " (" << Data[Member++] << ")\n";
        }
      }
    }
    ++SectionIndex;
  }
  if (!HasGroups)
    W.startLine() << "There are no group sections in the file.\n";
}

static inline void printFields(formatted_raw_ostream &OS, StringRef Str1,
                               StringRef Str2) {
  OS.PadToColumn(2u);
  OS << Str1;
  OS.PadToColumn(37u);
  OS << Str2 << "\n";
  OS.flush();
}

template <class ELFT>
void GNUStyle<ELFT>::printFileHeaders(const ELFFile<ELFT> *Obj) {
  const Elf_Ehdr *e = Obj->getHeader();
  OS << "ELF Header:\n";
  OS << "  Magic:  ";
  std::string Str;
  for (int i = 0; i < ELF::EI_NIDENT; i++)
    OS << format(" %02x", static_cast<int>(e->e_ident[i]));
  OS << "\n";
  Str = printEnum(e->e_ident[ELF::EI_CLASS], makeArrayRef(ElfClass));
  printFields(OS, "Class:", Str);
  Str = printEnum(e->e_ident[ELF::EI_DATA], makeArrayRef(ElfDataEncoding));
  printFields(OS, "Data:", Str);
  OS.PadToColumn(2u);
  OS << "Version:";
  OS.PadToColumn(37u);
  OS << to_hexString(e->e_ident[ELF::EI_VERSION]);
  if (e->e_version == ELF::EV_CURRENT)
    OS << " (current)";
  OS << "\n";
  Str = printEnum(e->e_ident[ELF::EI_OSABI], makeArrayRef(ElfOSABI));
  printFields(OS, "OS/ABI:", Str);
  Str = "0x" + to_hexString(e->e_version);
  Str = to_hexString(e->e_ident[ELF::EI_ABIVERSION]);
  printFields(OS, "ABI Version:", Str);
  Str = printEnum(e->e_type, makeArrayRef(ElfObjectFileType));
  printFields(OS, "Type:", Str);
  Str = printEnum(e->e_machine, makeArrayRef(ElfMachineType));
  printFields(OS, "Machine:", Str);
  Str = "0x" + to_hexString(e->e_version);
  printFields(OS, "Version:", Str);
  Str = "0x" + to_hexString(e->e_entry);
  printFields(OS, "Entry point address:", Str);
  Str = to_string(e->e_phoff) + " (bytes into file)";
  printFields(OS, "Start of program headers:", Str);
  Str = to_string(e->e_shoff) + " (bytes into file)";
  printFields(OS, "Start of section headers:", Str);
  Str = "0x" + to_hexString(e->e_flags);
  printFields(OS, "Flags:", Str);
  Str = to_string(e->e_ehsize) + " (bytes)";
  printFields(OS, "Size of this header:", Str);
  Str = to_string(e->e_phentsize) + " (bytes)";
  printFields(OS, "Size of program headers:", Str);
  Str = to_string(e->e_phnum);
  printFields(OS, "Number of program headers:", Str);
  Str = to_string(e->e_shentsize) + " (bytes)";
  printFields(OS, "Size of section headers:", Str);
  Str = to_string(e->e_shnum);
  printFields(OS, "Number of section headers:", Str);
  Str = to_string(e->e_shstrndx);
  printFields(OS, "Section header string table index:", Str);
}

template <class ELFT>
void LLVMStyle<ELFT>::printFileHeaders(const ELFFile<ELFT> *Obj) {
  const Elf_Ehdr *e = Obj->getHeader();
  {
    DictScope D(W, "ElfHeader");
    {
      DictScope D(W, "Ident");
      W.printBinary("Magic", makeArrayRef(e->e_ident).slice(ELF::EI_MAG0, 4));
      W.printEnum("Class", e->e_ident[ELF::EI_CLASS], makeArrayRef(ElfClass));
      W.printEnum("DataEncoding", e->e_ident[ELF::EI_DATA],
                  makeArrayRef(ElfDataEncoding));
      W.printNumber("FileVersion", e->e_ident[ELF::EI_VERSION]);

      // Handle architecture specific OS/ABI values.
      if (e->e_machine == ELF::EM_AMDGPU &&
          e->e_ident[ELF::EI_OSABI] == ELF::ELFOSABI_AMDGPU_HSA)
        W.printHex("OS/ABI", "AMDGPU_HSA", ELF::ELFOSABI_AMDGPU_HSA);
      else
        W.printEnum("OS/ABI", e->e_ident[ELF::EI_OSABI],
                    makeArrayRef(ElfOSABI));
      W.printNumber("ABIVersion", e->e_ident[ELF::EI_ABIVERSION]);
      W.printBinary("Unused", makeArrayRef(e->e_ident).slice(ELF::EI_PAD));
    }

    W.printEnum("Type", e->e_type, makeArrayRef(ElfObjectFileType));
    W.printEnum("Machine", e->e_machine, makeArrayRef(ElfMachineType));
    W.printNumber("Version", e->e_version);
    W.printHex("Entry", e->e_entry);
    W.printHex("ProgramHeaderOffset", e->e_phoff);
    W.printHex("SectionHeaderOffset", e->e_shoff);
    if (e->e_machine == EM_MIPS)
      W.printFlags("Flags", e->e_flags, makeArrayRef(ElfHeaderMipsFlags),
                   unsigned(ELF::EF_MIPS_ARCH), unsigned(ELF::EF_MIPS_ABI),
                   unsigned(ELF::EF_MIPS_MACH));
    else
      W.printFlags("Flags", e->e_flags);
    W.printNumber("HeaderSize", e->e_ehsize);
    W.printNumber("ProgramHeaderEntrySize", e->e_phentsize);
    W.printNumber("ProgramHeaderCount", e->e_phnum);
    W.printNumber("SectionHeaderEntrySize", e->e_shentsize);
    W.printNumber("SectionHeaderCount", e->e_shnum);
    W.printNumber("StringTableSectionIndex", e->e_shstrndx);
  }
}
