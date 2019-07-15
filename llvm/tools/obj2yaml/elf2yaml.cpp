//===------ utils/elf2yaml.cpp - obj2yaml conversion tool -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Error.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/ObjectYAML/ELFYAML.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/YAMLTraits.h"

using namespace llvm;

namespace {

template <class ELFT>
class ELFDumper {
  typedef object::Elf_Sym_Impl<ELFT> Elf_Sym;
  typedef typename ELFT::Dyn Elf_Dyn;
  typedef typename ELFT::Shdr Elf_Shdr;
  typedef typename ELFT::Word Elf_Word;
  typedef typename ELFT::Rel Elf_Rel;
  typedef typename ELFT::Rela Elf_Rela;

  ArrayRef<Elf_Shdr> Sections;
  ArrayRef<Elf_Sym> SymTable;

  DenseMap<StringRef, uint32_t> UsedSectionNames;
  std::vector<std::string> SectionNames;

  DenseMap<StringRef, uint32_t> UsedSymbolNames;
  std::vector<std::string> SymbolNames;

  Expected<StringRef> getUniquedSectionName(const Elf_Shdr *Sec);
  Expected<StringRef> getUniquedSymbolName(const Elf_Sym *Sym,
                                           StringRef StrTable,
                                           const Elf_Shdr *SymTab);

  const object::ELFFile<ELFT> &Obj;
  ArrayRef<Elf_Word> ShndxTable;

  Error dumpSymbols(const Elf_Shdr *Symtab,
                    std::vector<ELFYAML::Symbol> &Symbols);
  Error dumpSymbol(const Elf_Sym *Sym, const Elf_Shdr *SymTab,
                   StringRef StrTable, ELFYAML::Symbol &S);
  Error dumpCommonSection(const Elf_Shdr *Shdr, ELFYAML::Section &S);
  Error dumpCommonRelocationSection(const Elf_Shdr *Shdr,
                                    ELFYAML::RelocationSection &S);
  template <class RelT>
  Error dumpRelocation(const RelT *Rel, const Elf_Shdr *SymTab,
                       ELFYAML::Relocation &R);

  Expected<ELFYAML::DynamicSection *> dumpDynamicSection(const Elf_Shdr *Shdr);
  Expected<ELFYAML::RelocationSection *> dumpRelocSection(const Elf_Shdr *Shdr);
  Expected<ELFYAML::RawContentSection *>
  dumpContentSection(const Elf_Shdr *Shdr);
  Expected<ELFYAML::NoBitsSection *> dumpNoBitsSection(const Elf_Shdr *Shdr);
  Expected<ELFYAML::VerdefSection *> dumpVerdefSection(const Elf_Shdr *Shdr);
  Expected<ELFYAML::SymverSection *> dumpSymverSection(const Elf_Shdr *Shdr);
  Expected<ELFYAML::VerneedSection *> dumpVerneedSection(const Elf_Shdr *Shdr);
  Expected<ELFYAML::Group *> dumpGroup(const Elf_Shdr *Shdr);
  Expected<ELFYAML::MipsABIFlags *> dumpMipsABIFlags(const Elf_Shdr *Shdr);

public:
  ELFDumper(const object::ELFFile<ELFT> &O);
  Expected<ELFYAML::Object *> dump();
};

}

template <class ELFT>
ELFDumper<ELFT>::ELFDumper(const object::ELFFile<ELFT> &O)
    : Obj(O) {}

template <class ELFT>
Expected<StringRef>
ELFDumper<ELFT>::getUniquedSectionName(const Elf_Shdr *Sec) {
  unsigned SecIndex = Sec - &Sections[0];
  assert(&Sections[SecIndex] == Sec);
  if (!SectionNames[SecIndex].empty())
    return SectionNames[SecIndex];

  auto NameOrErr = Obj.getSectionName(Sec);
  if (!NameOrErr)
    return NameOrErr;
  StringRef Name = *NameOrErr;
  std::string &Ret = SectionNames[SecIndex];

  auto It = UsedSectionNames.insert({Name, 0});
  if (!It.second)
    Ret = (Name + " [" + Twine(++It.first->second) + "]").str();
  else
    Ret = Name;
  return Ret;
}

template <class ELFT>
Expected<StringRef>
ELFDumper<ELFT>::getUniquedSymbolName(const Elf_Sym *Sym, StringRef StrTable,
                                      const Elf_Shdr *SymTab) {
  Expected<StringRef> SymbolNameOrErr = Sym->getName(StrTable);
  if (!SymbolNameOrErr)
    return SymbolNameOrErr;
  StringRef Name = *SymbolNameOrErr;
  if (Name.empty() && Sym->getType() == ELF::STT_SECTION) {
    auto ShdrOrErr = Obj.getSection(Sym, SymTab, ShndxTable);
    if (!ShdrOrErr)
      return ShdrOrErr.takeError();
    return getUniquedSectionName(*ShdrOrErr);
  }

  // Symbols in .symtab can have duplicate names. For example, it is a common
  // situation for local symbols in a relocatable object. Here we assign unique
  // suffixes for such symbols so that we can differentiate them.
  if (SymTab->sh_type == ELF::SHT_SYMTAB) {
    unsigned Index = Sym - SymTable.data();
    if (!SymbolNames[Index].empty())
      return SymbolNames[Index];

    auto It = UsedSymbolNames.insert({Name, 0});
    if (!It.second)
      SymbolNames[Index] =
          (Name + " [" + Twine(++It.first->second) + "]").str();
    else
      SymbolNames[Index] = Name;
    return SymbolNames[Index];
  }

  return Name;
}

template <class ELFT> Expected<ELFYAML::Object *> ELFDumper<ELFT>::dump() {
  auto Y = make_unique<ELFYAML::Object>();

  // Dump header. We do not dump SHEntSize, SHOffset, SHNum and SHStrNdx field.
  // When not explicitly set, the values are set by yaml2obj automatically
  // and there is no need to dump them here.
  Y->Header.Class = ELFYAML::ELF_ELFCLASS(Obj.getHeader()->getFileClass());
  Y->Header.Data = ELFYAML::ELF_ELFDATA(Obj.getHeader()->getDataEncoding());
  Y->Header.OSABI = Obj.getHeader()->e_ident[ELF::EI_OSABI];
  Y->Header.ABIVersion = Obj.getHeader()->e_ident[ELF::EI_ABIVERSION];
  Y->Header.Type = Obj.getHeader()->e_type;
  Y->Header.Machine = Obj.getHeader()->e_machine;
  Y->Header.Flags = Obj.getHeader()->e_flags;
  Y->Header.Entry = Obj.getHeader()->e_entry;

  // Dump sections
  auto SectionsOrErr = Obj.sections();
  if (!SectionsOrErr)
    return SectionsOrErr.takeError();
  Sections = *SectionsOrErr;
  SectionNames.resize(Sections.size());

  // Dump symbols. We need to do this early because other sections might want
  // to access the deduplicated symbol names that we also create here.
  for (const Elf_Shdr &Sec : Sections) {
    if (Sec.sh_type == ELF::SHT_SYMTAB)
      if (Error E = dumpSymbols(&Sec, Y->Symbols))
        return std::move(E);
    if (Sec.sh_type == ELF::SHT_DYNSYM)
      if (Error E = dumpSymbols(&Sec, Y->DynamicSymbols))
        return std::move(E);
  }

  for (const Elf_Shdr &Sec : Sections) {
    switch (Sec.sh_type) {
    case ELF::SHT_DYNAMIC: {
      Expected<ELFYAML::DynamicSection *> SecOrErr = dumpDynamicSection(&Sec);
      if (!SecOrErr)
        return SecOrErr.takeError();
      Y->Sections.emplace_back(*SecOrErr);
      break;
    }
    case ELF::SHT_NULL:
    case ELF::SHT_STRTAB:
    case ELF::SHT_SYMTAB:
    case ELF::SHT_DYNSYM:
      // Do not dump these sections.
      break;
    case ELF::SHT_SYMTAB_SHNDX: {
      auto TableOrErr = Obj.getSHNDXTable(Sec);
      if (!TableOrErr)
        return TableOrErr.takeError();
      ShndxTable = *TableOrErr;
      break;
    }
    case ELF::SHT_REL:
    case ELF::SHT_RELA: {
      Expected<ELFYAML::RelocationSection *> SecOrErr = dumpRelocSection(&Sec);
      if (!SecOrErr)
        return SecOrErr.takeError();
      Y->Sections.emplace_back(*SecOrErr);
      break;
    }
    case ELF::SHT_GROUP: {
      Expected<ELFYAML::Group *> GroupOrErr = dumpGroup(&Sec);
      if (!GroupOrErr)
        return GroupOrErr.takeError();
      Y->Sections.emplace_back(*GroupOrErr);
      break;
    }
    case ELF::SHT_MIPS_ABIFLAGS: {
      Expected<ELFYAML::MipsABIFlags *> SecOrErr = dumpMipsABIFlags(&Sec);
      if (!SecOrErr)
        return SecOrErr.takeError();
      Y->Sections.emplace_back(*SecOrErr);
      break;
    }
    case ELF::SHT_NOBITS: {
      Expected<ELFYAML::NoBitsSection *> SecOrErr = dumpNoBitsSection(&Sec);
      if (!SecOrErr)
        return SecOrErr.takeError();
      Y->Sections.emplace_back(*SecOrErr);
      break;
    }
    case ELF::SHT_GNU_verdef: {
      Expected<ELFYAML::VerdefSection *> SecOrErr = dumpVerdefSection(&Sec);
      if (!SecOrErr)
        return SecOrErr.takeError();
      Y->Sections.emplace_back(*SecOrErr);
      break;
    }
    case ELF::SHT_GNU_versym: {
      Expected<ELFYAML::SymverSection *> SecOrErr = dumpSymverSection(&Sec);
      if (!SecOrErr)
        return SecOrErr.takeError();
      Y->Sections.emplace_back(*SecOrErr);
      break;
    }
    case ELF::SHT_GNU_verneed: {
      Expected<ELFYAML::VerneedSection *> SecOrErr = dumpVerneedSection(&Sec);
      if (!SecOrErr)
        return SecOrErr.takeError();
      Y->Sections.emplace_back(*SecOrErr);
      break;
    }
    default: {
      Expected<ELFYAML::RawContentSection *> SecOrErr =
          dumpContentSection(&Sec);
      if (!SecOrErr)
        return SecOrErr.takeError();
      Y->Sections.emplace_back(*SecOrErr);
    }
    }
  }

  return Y.release();
}

template <class ELFT>
Error ELFDumper<ELFT>::dumpSymbols(const Elf_Shdr *Symtab,
                             std::vector<ELFYAML::Symbol> &Symbols) {
  if (!Symtab)
    return Error::success();

  auto StrTableOrErr = Obj.getStringTableForSymtab(*Symtab);
  if (!StrTableOrErr)
    return StrTableOrErr.takeError();
  StringRef StrTable = *StrTableOrErr;

  auto SymtabOrErr = Obj.symbols(Symtab);
  if (!SymtabOrErr)
    return SymtabOrErr.takeError();

  if (Symtab->sh_type == ELF::SHT_SYMTAB) {
    SymTable = *SymtabOrErr;
    SymbolNames.resize(SymTable.size());
  }

  for (const auto &Sym : (*SymtabOrErr).drop_front()) {
    ELFYAML::Symbol S;
    if (auto EC = dumpSymbol(&Sym, Symtab, StrTable, S))
      return EC;
    Symbols.push_back(S);
  }

  return Error::success();
}

template <class ELFT>
Error ELFDumper<ELFT>::dumpSymbol(const Elf_Sym *Sym, const Elf_Shdr *SymTab,
                                  StringRef StrTable, ELFYAML::Symbol &S) {
  S.Type = Sym->getType();
  S.Value = Sym->st_value;
  S.Size = Sym->st_size;
  S.Other = Sym->st_other;
  S.Binding = Sym->getBinding();

  Expected<StringRef> SymbolNameOrErr =
      getUniquedSymbolName(Sym, StrTable, SymTab);
  if (!SymbolNameOrErr)
    return SymbolNameOrErr.takeError();
  S.Name = SymbolNameOrErr.get();

  if (Sym->st_shndx >= ELF::SHN_LORESERVE) {
    if (Sym->st_shndx == ELF::SHN_XINDEX)
      return createStringError(obj2yaml_error::not_implemented,
                               "SHN_XINDEX symbols are not supported");
    S.Index = (ELFYAML::ELF_SHN)Sym->st_shndx;
    return Error::success();
  }

  auto ShdrOrErr = Obj.getSection(Sym, SymTab, ShndxTable);
  if (!ShdrOrErr)
    return ShdrOrErr.takeError();
  const Elf_Shdr *Shdr = *ShdrOrErr;
  if (!Shdr)
    return Error::success();

  auto NameOrErr = getUniquedSectionName(Shdr);
  if (!NameOrErr)
    return NameOrErr.takeError();
  S.Section = NameOrErr.get();

  return Error::success();
}

template <class ELFT>
template <class RelT>
Error ELFDumper<ELFT>::dumpRelocation(const RelT *Rel, const Elf_Shdr *SymTab,
                                      ELFYAML::Relocation &R) {
  R.Type = Rel->getType(Obj.isMips64EL());
  R.Offset = Rel->r_offset;
  R.Addend = 0;

  auto SymOrErr = Obj.getRelocationSymbol(Rel, SymTab);
  if (!SymOrErr)
    return SymOrErr.takeError();
  const Elf_Sym *Sym = *SymOrErr;
  auto StrTabSec = Obj.getSection(SymTab->sh_link);
  if (!StrTabSec)
    return StrTabSec.takeError();
  auto StrTabOrErr = Obj.getStringTable(*StrTabSec);
  if (!StrTabOrErr)
    return StrTabOrErr.takeError();
  StringRef StrTab = *StrTabOrErr;

  if (Sym) {
    Expected<StringRef> NameOrErr = getUniquedSymbolName(Sym, StrTab, SymTab);
    if (!NameOrErr)
      return NameOrErr.takeError();
    R.Symbol = NameOrErr.get();
  } else {
    // We have some edge cases of relocations without a symbol associated,
    // e.g. an object containing the invalid (according to the System V
    // ABI) R_X86_64_NONE reloc. Create a symbol with an empty name instead
    // of crashing.
    R.Symbol = "";
  }

  return Error::success();
}

template <class ELFT>
Error ELFDumper<ELFT>::dumpCommonSection(const Elf_Shdr *Shdr,
                                         ELFYAML::Section &S) {
  // Dump fields. We do not dump the ShOffset field. When not explicitly
  // set, the value is set by yaml2obj automatically.
  S.Type = Shdr->sh_type;
  if (Shdr->sh_flags)
    S.Flags = static_cast<ELFYAML::ELF_SHF>(Shdr->sh_flags);
  S.Address = Shdr->sh_addr;
  S.AddressAlign = Shdr->sh_addralign;
  if (Shdr->sh_entsize)
    S.EntSize = static_cast<llvm::yaml::Hex64>(Shdr->sh_entsize);

  auto NameOrErr = getUniquedSectionName(Shdr);
  if (!NameOrErr)
    return NameOrErr.takeError();
  S.Name = NameOrErr.get();

  if (Shdr->sh_link != ELF::SHN_UNDEF) {
    auto LinkSection = Obj.getSection(Shdr->sh_link);
    if (LinkSection.takeError())
      return LinkSection.takeError();
    NameOrErr = getUniquedSectionName(*LinkSection);
    if (!NameOrErr)
      return NameOrErr.takeError();
    S.Link = NameOrErr.get();
  }

  return Error::success();
}

template <class ELFT>
Error ELFDumper<ELFT>::dumpCommonRelocationSection(
    const Elf_Shdr *Shdr, ELFYAML::RelocationSection &S) {
  if (Error E = dumpCommonSection(Shdr, S))
    return E;

  auto InfoSection = Obj.getSection(Shdr->sh_info);
  if (!InfoSection)
    return InfoSection.takeError();

  auto NameOrErr = getUniquedSectionName(*InfoSection);
  if (!NameOrErr)
    return NameOrErr.takeError();
  S.RelocatableSec = NameOrErr.get();

  return Error::success();
}

template <class ELFT>
Expected<ELFYAML::DynamicSection *>
ELFDumper<ELFT>::dumpDynamicSection(const Elf_Shdr *Shdr) {
  auto S = make_unique<ELFYAML::DynamicSection>();
  if (Error E = dumpCommonSection(Shdr, *S))
    return std::move(E);

  auto DynTagsOrErr = Obj.template getSectionContentsAsArray<Elf_Dyn>(Shdr);
  if (!DynTagsOrErr)
    return DynTagsOrErr.takeError();

  for (const Elf_Dyn &Dyn : *DynTagsOrErr)
    S->Entries.push_back({(ELFYAML::ELF_DYNTAG)Dyn.getTag(), Dyn.getVal()});

  return S.release();
}

template <class ELFT>
Expected<ELFYAML::RelocationSection *>
ELFDumper<ELFT>::dumpRelocSection(const Elf_Shdr *Shdr) {
  auto S = make_unique<ELFYAML::RelocationSection>();
  if (auto E = dumpCommonRelocationSection(Shdr, *S))
    return std::move(E);

  auto SymTabOrErr = Obj.getSection(Shdr->sh_link);
  if (!SymTabOrErr)
    return SymTabOrErr.takeError();
  const Elf_Shdr *SymTab = *SymTabOrErr;

  if (Shdr->sh_type == ELF::SHT_REL) {
    auto Rels = Obj.rels(Shdr);
    if (!Rels)
      return Rels.takeError();
    for (const Elf_Rel &Rel : *Rels) {
      ELFYAML::Relocation R;
      if (Error E = dumpRelocation(&Rel, SymTab, R))
        return std::move(E);
      S->Relocations.push_back(R);
    }
  } else {
    auto Rels = Obj.relas(Shdr);
    if (!Rels)
      return Rels.takeError();
    for (const Elf_Rela &Rel : *Rels) {
      ELFYAML::Relocation R;
      if (Error E = dumpRelocation(&Rel, SymTab, R))
        return std::move(E);
      R.Addend = Rel.r_addend;
      S->Relocations.push_back(R);
    }
  }

  return S.release();
}

template <class ELFT>
Expected<ELFYAML::RawContentSection *>
ELFDumper<ELFT>::dumpContentSection(const Elf_Shdr *Shdr) {
  auto S = make_unique<ELFYAML::RawContentSection>();
  if (Error E = dumpCommonSection(Shdr, *S))
    return std::move(E);

  auto ContentOrErr = Obj.getSectionContents(Shdr);
  if (!ContentOrErr)
    return ContentOrErr.takeError();
  ArrayRef<uint8_t> Content = *ContentOrErr;
  if (!Content.empty())
    S->Content = yaml::BinaryRef(Content);
  if (Shdr->sh_info)
    S->Info = static_cast<llvm::yaml::Hex64>(Shdr->sh_info);
  return S.release();
}

template <class ELFT>
Expected<ELFYAML::NoBitsSection *>
ELFDumper<ELFT>::dumpNoBitsSection(const Elf_Shdr *Shdr) {
  auto S = make_unique<ELFYAML::NoBitsSection>();
  if (Error E = dumpCommonSection(Shdr, *S))
    return std::move(E);
  S->Size = Shdr->sh_size;

  return S.release();
}

template <class ELFT>
Expected<ELFYAML::VerdefSection *>
ELFDumper<ELFT>::dumpVerdefSection(const Elf_Shdr *Shdr) {
  typedef typename ELFT::Verdef Elf_Verdef;
  typedef typename ELFT::Verdaux Elf_Verdaux;

  auto S = make_unique<ELFYAML::VerdefSection>();
  if (Error E = dumpCommonSection(Shdr, *S))
    return std::move(E);

  S->Info = Shdr->sh_info;

  auto StringTableShdrOrErr = Obj.getSection(Shdr->sh_link);
  if (!StringTableShdrOrErr)
    return StringTableShdrOrErr.takeError();

  auto StringTableOrErr = Obj.getStringTable(*StringTableShdrOrErr);
  if (!StringTableOrErr)
    return StringTableOrErr.takeError();

  auto Contents = Obj.getSectionContents(Shdr);
  if (!Contents)
    return Contents.takeError();

  llvm::ArrayRef<uint8_t> Data = *Contents;
  const uint8_t *Buf = Data.data();
  while (Buf) {
    const Elf_Verdef *Verdef = reinterpret_cast<const Elf_Verdef *>(Buf);
    ELFYAML::VerdefEntry Entry;
    Entry.Version = Verdef->vd_version;
    Entry.Flags = Verdef->vd_flags;
    Entry.VersionNdx = Verdef->vd_ndx;
    Entry.Hash = Verdef->vd_hash;

    const uint8_t *BufAux = Buf + Verdef->vd_aux;
    while (BufAux) {
      const Elf_Verdaux *Verdaux =
          reinterpret_cast<const Elf_Verdaux *>(BufAux);
      Entry.VerNames.push_back(
          StringTableOrErr->drop_front(Verdaux->vda_name).data());
      BufAux = Verdaux->vda_next ? BufAux + Verdaux->vda_next : nullptr;
    }

    S->Entries.push_back(Entry);
    Buf = Verdef->vd_next ? Buf + Verdef->vd_next : nullptr;
  }

  return S.release();
}

template <class ELFT>
Expected<ELFYAML::SymverSection *>
ELFDumper<ELFT>::dumpSymverSection(const Elf_Shdr *Shdr) {
  typedef typename ELFT::Half Elf_Half;

  auto S = make_unique<ELFYAML::SymverSection>();
  if (Error E = dumpCommonSection(Shdr, *S))
    return std::move(E);

  auto VersionsOrErr = Obj.template getSectionContentsAsArray<Elf_Half>(Shdr);
  if (!VersionsOrErr)
    return VersionsOrErr.takeError();
  for (const Elf_Half &E : *VersionsOrErr)
    S->Entries.push_back(E);

  return S.release();
}

template <class ELFT>
Expected<ELFYAML::VerneedSection *>
ELFDumper<ELFT>::dumpVerneedSection(const Elf_Shdr *Shdr) {
  typedef typename ELFT::Verneed Elf_Verneed;
  typedef typename ELFT::Vernaux Elf_Vernaux;

  auto S = make_unique<ELFYAML::VerneedSection>();
  if (Error E = dumpCommonSection(Shdr, *S))
    return std::move(E);

  S->Info = Shdr->sh_info;

  auto Contents = Obj.getSectionContents(Shdr);
  if (!Contents)
    return Contents.takeError();

  auto StringTableShdrOrErr = Obj.getSection(Shdr->sh_link);
  if (!StringTableShdrOrErr)
    return StringTableShdrOrErr.takeError();

  auto StringTableOrErr = Obj.getStringTable(*StringTableShdrOrErr);
  if (!StringTableOrErr)
    return StringTableOrErr.takeError();

  llvm::ArrayRef<uint8_t> Data = *Contents;
  const uint8_t *Buf = Data.data();
  while (Buf) {
    const Elf_Verneed *Verneed = reinterpret_cast<const Elf_Verneed *>(Buf);

    ELFYAML::VerneedEntry Entry;
    Entry.Version = Verneed->vn_version;
    Entry.File =
        StringRef(StringTableOrErr->drop_front(Verneed->vn_file).data());

    const uint8_t *BufAux = Buf + Verneed->vn_aux;
    while (BufAux) {
      const Elf_Vernaux *Vernaux =
          reinterpret_cast<const Elf_Vernaux *>(BufAux);

      ELFYAML::VernauxEntry Aux;
      Aux.Hash = Vernaux->vna_hash;
      Aux.Flags = Vernaux->vna_flags;
      Aux.Other = Vernaux->vna_other;
      Aux.Name =
          StringRef(StringTableOrErr->drop_front(Vernaux->vna_name).data());

      Entry.AuxV.push_back(Aux);
      BufAux = Vernaux->vna_next ? BufAux + Vernaux->vna_next : nullptr;
    }

    S->VerneedV.push_back(Entry);
    Buf = Verneed->vn_next ? Buf + Verneed->vn_next : nullptr;
  }

  return S.release();
}

template <class ELFT>
Expected<ELFYAML::Group *> ELFDumper<ELFT>::dumpGroup(const Elf_Shdr *Shdr) {
  auto S = make_unique<ELFYAML::Group>();
  if (Error E = dumpCommonSection(Shdr, *S))
    return std::move(E);

  auto SymtabOrErr = Obj.getSection(Shdr->sh_link);
  if (!SymtabOrErr)
    return SymtabOrErr.takeError();
  // Get symbol with index sh_info which name is the signature of the group.
  const Elf_Shdr *Symtab = *SymtabOrErr;
  auto SymOrErr = Obj.getSymbol(Symtab, Shdr->sh_info);
  if (!SymOrErr)
    return SymOrErr.takeError();
  auto StrTabOrErr = Obj.getStringTableForSymtab(*Symtab);
  if (!StrTabOrErr)
    return StrTabOrErr.takeError();

  Expected<StringRef> SymbolName =
      getUniquedSymbolName(*SymOrErr, *StrTabOrErr, Symtab);
  if (!SymbolName)
    return SymbolName.takeError();
  S->Signature = *SymbolName;

  auto MembersOrErr = Obj.template getSectionContentsAsArray<Elf_Word>(Shdr);
  if (!MembersOrErr)
    return MembersOrErr.takeError();

  for (Elf_Word Member : *MembersOrErr) {
    if (Member == llvm::ELF::GRP_COMDAT) {
      S->Members.push_back({"GRP_COMDAT"});
      continue;
    }

    auto SHdrOrErr = Obj.getSection(Member);
    if (!SHdrOrErr)
      return SHdrOrErr.takeError();
    auto NameOrErr = getUniquedSectionName(*SHdrOrErr);
    if (!NameOrErr)
      return NameOrErr.takeError();
    S->Members.push_back({*NameOrErr});
  }
  return S.release();
}

template <class ELFT>
Expected<ELFYAML::MipsABIFlags *>
ELFDumper<ELFT>::dumpMipsABIFlags(const Elf_Shdr *Shdr) {
  assert(Shdr->sh_type == ELF::SHT_MIPS_ABIFLAGS &&
         "Section type is not SHT_MIPS_ABIFLAGS");
  auto S = make_unique<ELFYAML::MipsABIFlags>();
  if (Error E = dumpCommonSection(Shdr, *S))
    return std::move(E);

  auto ContentOrErr = Obj.getSectionContents(Shdr);
  if (!ContentOrErr)
    return ContentOrErr.takeError();

  auto *Flags = reinterpret_cast<const object::Elf_Mips_ABIFlags<ELFT> *>(
      ContentOrErr.get().data());
  S->Version = Flags->version;
  S->ISALevel = Flags->isa_level;
  S->ISARevision = Flags->isa_rev;
  S->GPRSize = Flags->gpr_size;
  S->CPR1Size = Flags->cpr1_size;
  S->CPR2Size = Flags->cpr2_size;
  S->FpABI = Flags->fp_abi;
  S->ISAExtension = Flags->isa_ext;
  S->ASEs = Flags->ases;
  S->Flags1 = Flags->flags1;
  S->Flags2 = Flags->flags2;
  return S.release();
}

template <class ELFT>
static Error elf2yaml(raw_ostream &Out, const object::ELFFile<ELFT> &Obj) {
  ELFDumper<ELFT> Dumper(Obj);
  Expected<ELFYAML::Object *> YAMLOrErr = Dumper.dump();
  if (!YAMLOrErr)
    return YAMLOrErr.takeError();

  std::unique_ptr<ELFYAML::Object> YAML(YAMLOrErr.get());
  yaml::Output Yout(Out);
  Yout << *YAML;

  return Error::success();
}

Error elf2yaml(raw_ostream &Out, const object::ObjectFile &Obj) {
  if (const auto *ELFObj = dyn_cast<object::ELF32LEObjectFile>(&Obj))
    return elf2yaml(Out, *ELFObj->getELFFile());

  if (const auto *ELFObj = dyn_cast<object::ELF32BEObjectFile>(&Obj))
    return elf2yaml(Out, *ELFObj->getELFFile());

  if (const auto *ELFObj = dyn_cast<object::ELF64LEObjectFile>(&Obj))
    return elf2yaml(Out, *ELFObj->getELFFile());

  if (const auto *ELFObj = dyn_cast<object::ELF64BEObjectFile>(&Obj))
    return elf2yaml(Out, *ELFObj->getELFFile());

  llvm_unreachable("unknown ELF file format");
}
