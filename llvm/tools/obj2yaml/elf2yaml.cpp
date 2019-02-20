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

  // If the file has multiple sections with the same name, we add a
  // suffix to make them unique.
  unsigned Suffix = 0;
  DenseSet<StringRef> UsedSectionNames;
  std::vector<std::string> SectionNames;
  Expected<StringRef> getUniquedSectionName(const Elf_Shdr *Sec);
  Expected<StringRef> getSymbolName(const Elf_Sym *Sym, StringRef StrTable,
                                    const Elf_Shdr *SymTab);

  const object::ELFFile<ELFT> &Obj;
  ArrayRef<Elf_Word> ShndxTable;

  std::error_code dumpSymbols(const Elf_Shdr *Symtab,
                              ELFYAML::LocalGlobalWeakSymbols &Symbols);
  std::error_code dumpSymbol(const Elf_Sym *Sym, const Elf_Shdr *SymTab,
                             StringRef StrTable, ELFYAML::Symbol &S);
  std::error_code dumpCommonSection(const Elf_Shdr *Shdr, ELFYAML::Section &S);
  std::error_code dumpCommonRelocationSection(const Elf_Shdr *Shdr,
                                              ELFYAML::RelocationSection &S);
  template <class RelT>
  std::error_code dumpRelocation(const RelT *Rel, const Elf_Shdr *SymTab,
                                 ELFYAML::Relocation &R);
  
  ErrorOr<ELFYAML::DynamicSection *> dumpDynamicSection(const Elf_Shdr *Shdr);
  ErrorOr<ELFYAML::RelocationSection *> dumpRelocSection(const Elf_Shdr *Shdr);
  ErrorOr<ELFYAML::RawContentSection *>
  dumpContentSection(const Elf_Shdr *Shdr);
  ErrorOr<ELFYAML::NoBitsSection *> dumpNoBitsSection(const Elf_Shdr *Shdr);
  ErrorOr<ELFYAML::SymverSection *> dumpSymverSection(const Elf_Shdr *Shdr);
  ErrorOr<ELFYAML::VerneedSection *> dumpVerneedSection(const Elf_Shdr *Shdr);
  ErrorOr<ELFYAML::Group *> dumpGroup(const Elf_Shdr *Shdr);
  ErrorOr<ELFYAML::MipsABIFlags *> dumpMipsABIFlags(const Elf_Shdr *Shdr);

public:
  ELFDumper(const object::ELFFile<ELFT> &O);
  ErrorOr<ELFYAML::Object *> dump();
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
  Ret = Name;
  while (!UsedSectionNames.insert(Ret).second)
    Ret = (Name + to_string(++Suffix)).str();
  return Ret;
}

template <class ELFT>
Expected<StringRef> ELFDumper<ELFT>::getSymbolName(const Elf_Sym *Sym,
                                                   StringRef StrTable,
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
  return Name;
}

template <class ELFT> ErrorOr<ELFYAML::Object *> ELFDumper<ELFT>::dump() {
  auto Y = make_unique<ELFYAML::Object>();

  // Dump header
  Y->Header.Class = ELFYAML::ELF_ELFCLASS(Obj.getHeader()->getFileClass());
  Y->Header.Data = ELFYAML::ELF_ELFDATA(Obj.getHeader()->getDataEncoding());
  Y->Header.OSABI = Obj.getHeader()->e_ident[ELF::EI_OSABI];
  Y->Header.ABIVersion = Obj.getHeader()->e_ident[ELF::EI_ABIVERSION];
  Y->Header.Type = Obj.getHeader()->e_type;
  Y->Header.Machine = Obj.getHeader()->e_machine;
  Y->Header.Flags = Obj.getHeader()->e_flags;
  Y->Header.Entry = Obj.getHeader()->e_entry;

  const Elf_Shdr *Symtab = nullptr;
  const Elf_Shdr *DynSymtab = nullptr;

  // Dump sections
  auto SectionsOrErr = Obj.sections();
  if (!SectionsOrErr)
    return errorToErrorCode(SectionsOrErr.takeError());
  Sections = *SectionsOrErr;
  SectionNames.resize(Sections.size());
  for (const Elf_Shdr &Sec : Sections) {
    switch (Sec.sh_type) {
    case ELF::SHT_DYNAMIC: {
      ErrorOr<ELFYAML::DynamicSection *> S = dumpDynamicSection(&Sec);
      if (std::error_code EC = S.getError())
        return EC;
      Y->Sections.push_back(std::unique_ptr<ELFYAML::Section>(S.get()));
      break;
    }
    case ELF::SHT_NULL:
    case ELF::SHT_STRTAB:
      // Do not dump these sections.
      break;
    case ELF::SHT_SYMTAB:
      Symtab = &Sec;
      break;
    case ELF::SHT_DYNSYM:
      DynSymtab = &Sec;
      break;
    case ELF::SHT_SYMTAB_SHNDX: {
      auto TableOrErr = Obj.getSHNDXTable(Sec);
      if (!TableOrErr)
        return errorToErrorCode(TableOrErr.takeError());
      ShndxTable = *TableOrErr;
      break;
    }
    case ELF::SHT_REL:
    case ELF::SHT_RELA: {
      ErrorOr<ELFYAML::RelocationSection *> S = dumpRelocSection(&Sec);
      if (std::error_code EC = S.getError())
        return EC;
      Y->Sections.push_back(std::unique_ptr<ELFYAML::Section>(S.get()));
      break;
    }
    case ELF::SHT_GROUP: {
      ErrorOr<ELFYAML::Group *> G = dumpGroup(&Sec);
      if (std::error_code EC = G.getError())
        return EC;
      Y->Sections.push_back(std::unique_ptr<ELFYAML::Section>(G.get()));
      break;
    }
    case ELF::SHT_MIPS_ABIFLAGS: {
      ErrorOr<ELFYAML::MipsABIFlags *> G = dumpMipsABIFlags(&Sec);
      if (std::error_code EC = G.getError())
        return EC;
      Y->Sections.push_back(std::unique_ptr<ELFYAML::Section>(G.get()));
      break;
    }
    case ELF::SHT_NOBITS: {
      ErrorOr<ELFYAML::NoBitsSection *> S = dumpNoBitsSection(&Sec);
      if (std::error_code EC = S.getError())
        return EC;
      Y->Sections.push_back(std::unique_ptr<ELFYAML::Section>(S.get()));
      break;
    }
    case ELF::SHT_GNU_versym: {
      ErrorOr<ELFYAML::SymverSection *> S = dumpSymverSection(&Sec);
      if (std::error_code EC = S.getError())
        return EC;
      Y->Sections.push_back(std::unique_ptr<ELFYAML::Section>(S.get()));
      break;
    }
    case ELF::SHT_GNU_verneed: {
      ErrorOr<ELFYAML::VerneedSection *> S = dumpVerneedSection(&Sec);
      if (std::error_code EC = S.getError())
        return EC;
      Y->Sections.push_back(std::unique_ptr<ELFYAML::Section>(S.get()));
      break;
    }
    default: {
      ErrorOr<ELFYAML::RawContentSection *> S = dumpContentSection(&Sec);
      if (std::error_code EC = S.getError())
        return EC;
      Y->Sections.push_back(std::unique_ptr<ELFYAML::Section>(S.get()));
    }
    }
  }

  if (auto EC = dumpSymbols(Symtab, Y->Symbols))
    return EC;
  if (auto EC = dumpSymbols(DynSymtab, Y->DynamicSymbols))
    return EC;

  return Y.release();
}

template <class ELFT>
std::error_code
ELFDumper<ELFT>::dumpSymbols(const Elf_Shdr *Symtab,
                             ELFYAML::LocalGlobalWeakSymbols &Symbols) {
  if (!Symtab)
    return std::error_code();

  auto StrTableOrErr = Obj.getStringTableForSymtab(*Symtab);
  if (!StrTableOrErr)
    return errorToErrorCode(StrTableOrErr.takeError());
  StringRef StrTable = *StrTableOrErr;

  auto SymtabOrErr = Obj.symbols(Symtab);
  if (!SymtabOrErr)
    return errorToErrorCode(SymtabOrErr.takeError());

  bool IsFirstSym = true;
  for (const auto &Sym : *SymtabOrErr) {
    if (IsFirstSym) {
      IsFirstSym = false;
      continue;
    }

    ELFYAML::Symbol S;
    if (auto EC = dumpSymbol(&Sym, Symtab, StrTable, S))
      return EC;

    switch (Sym.getBinding()) {
    case ELF::STB_LOCAL:
      Symbols.Local.push_back(S);
      break;
    case ELF::STB_GLOBAL:
      Symbols.Global.push_back(S);
      break;
    case ELF::STB_WEAK:
      Symbols.Weak.push_back(S);
      break;
    default:
      llvm_unreachable("Unknown ELF symbol binding");
    }
  }

  return std::error_code();
}

template <class ELFT>
std::error_code
ELFDumper<ELFT>::dumpSymbol(const Elf_Sym *Sym, const Elf_Shdr *SymTab,
                            StringRef StrTable, ELFYAML::Symbol &S) {
  S.Type = Sym->getType();
  S.Value = Sym->st_value;
  S.Size = Sym->st_size;
  S.Other = Sym->st_other;

  Expected<StringRef> SymbolNameOrErr = getSymbolName(Sym, StrTable, SymTab);
  if (!SymbolNameOrErr)
    return errorToErrorCode(SymbolNameOrErr.takeError());
  S.Name = SymbolNameOrErr.get();

  auto ShdrOrErr = Obj.getSection(Sym, SymTab, ShndxTable);
  if (!ShdrOrErr)
    return errorToErrorCode(ShdrOrErr.takeError());
  const Elf_Shdr *Shdr = *ShdrOrErr;
  if (!Shdr)
    return obj2yaml_error::success;

  auto NameOrErr = getUniquedSectionName(Shdr);
  if (!NameOrErr)
    return errorToErrorCode(NameOrErr.takeError());
  S.Section = NameOrErr.get();

  return obj2yaml_error::success;
}

template <class ELFT>
template <class RelT>
std::error_code ELFDumper<ELFT>::dumpRelocation(const RelT *Rel,
                                                const Elf_Shdr *SymTab,
                                                ELFYAML::Relocation &R) {
  R.Type = Rel->getType(Obj.isMips64EL());
  R.Offset = Rel->r_offset;
  R.Addend = 0;

  auto SymOrErr = Obj.getRelocationSymbol(Rel, SymTab);
  if (!SymOrErr)
    return errorToErrorCode(SymOrErr.takeError());
  const Elf_Sym *Sym = *SymOrErr;
  auto StrTabSec = Obj.getSection(SymTab->sh_link);
  if (!StrTabSec)
    return errorToErrorCode(StrTabSec.takeError());
  auto StrTabOrErr = Obj.getStringTable(*StrTabSec);
  if (!StrTabOrErr)
    return errorToErrorCode(StrTabOrErr.takeError());
  StringRef StrTab = *StrTabOrErr;

  if (Sym) {
    Expected<StringRef> NameOrErr = getSymbolName(Sym, StrTab, SymTab);
    if (!NameOrErr)
      return errorToErrorCode(NameOrErr.takeError());
    R.Symbol = NameOrErr.get();
  } else {
    // We have some edge cases of relocations without a symbol associated,
    // e.g. an object containing the invalid (according to the System V
    // ABI) R_X86_64_NONE reloc. Create a symbol with an empty name instead
    // of crashing.
    R.Symbol = "";
  }

  return obj2yaml_error::success;
}

template <class ELFT>
std::error_code ELFDumper<ELFT>::dumpCommonSection(const Elf_Shdr *Shdr,
                                                   ELFYAML::Section &S) {
  S.Type = Shdr->sh_type;
  S.Flags = Shdr->sh_flags;
  S.Address = Shdr->sh_addr;
  S.AddressAlign = Shdr->sh_addralign;
  if (Shdr->sh_entsize)
    S.EntSize = static_cast<llvm::yaml::Hex64>(Shdr->sh_entsize);

  auto NameOrErr = getUniquedSectionName(Shdr);
  if (!NameOrErr)
    return errorToErrorCode(NameOrErr.takeError());
  S.Name = NameOrErr.get();

  if (Shdr->sh_link != ELF::SHN_UNDEF) {
    auto LinkSection = Obj.getSection(Shdr->sh_link);
    if (LinkSection.takeError())
      return errorToErrorCode(LinkSection.takeError());
    NameOrErr = getUniquedSectionName(*LinkSection);
    if (!NameOrErr)
      return errorToErrorCode(NameOrErr.takeError());
    S.Link = NameOrErr.get();
  }

  return obj2yaml_error::success;
}

template <class ELFT>
std::error_code
ELFDumper<ELFT>::dumpCommonRelocationSection(const Elf_Shdr *Shdr,
                                             ELFYAML::RelocationSection &S) {
  if (std::error_code EC = dumpCommonSection(Shdr, S))
    return EC;

  auto InfoSection = Obj.getSection(Shdr->sh_info);
  if (!InfoSection)
    return errorToErrorCode(InfoSection.takeError());

  auto NameOrErr = getUniquedSectionName(*InfoSection);
  if (!NameOrErr)
    return errorToErrorCode(NameOrErr.takeError());
  S.RelocatableSec = NameOrErr.get();

  return obj2yaml_error::success;
}

template <class ELFT>
ErrorOr<ELFYAML::DynamicSection *>
ELFDumper<ELFT>::dumpDynamicSection(const Elf_Shdr *Shdr) {
  auto S = make_unique<ELFYAML::DynamicSection>();
  if (std::error_code EC = dumpCommonSection(Shdr, *S))
    return EC;

  auto DynTagsOrErr = Obj.template getSectionContentsAsArray<Elf_Dyn>(Shdr);
  if (!DynTagsOrErr)
    return errorToErrorCode(DynTagsOrErr.takeError());

  for (const Elf_Dyn &Dyn : *DynTagsOrErr)
    S->Entries.push_back({(ELFYAML::ELF_DYNTAG)Dyn.getTag(), Dyn.getVal()});

  return S.release();
}

template <class ELFT>
ErrorOr<ELFYAML::RelocationSection *>
ELFDumper<ELFT>::dumpRelocSection(const Elf_Shdr *Shdr) {
  auto S = make_unique<ELFYAML::RelocationSection>();
  if (std::error_code EC = dumpCommonRelocationSection(Shdr, *S))
    return EC;

  auto SymTabOrErr = Obj.getSection(Shdr->sh_link);
  if (!SymTabOrErr)
    return errorToErrorCode(SymTabOrErr.takeError());
  const Elf_Shdr *SymTab = *SymTabOrErr;

  if (Shdr->sh_type == ELF::SHT_REL) {
    auto Rels = Obj.rels(Shdr);
    if (!Rels)
      return errorToErrorCode(Rels.takeError());
    for (const Elf_Rel &Rel : *Rels) {
      ELFYAML::Relocation R;
      if (std::error_code EC = dumpRelocation(&Rel, SymTab, R))
        return EC;
      S->Relocations.push_back(R);
    }
  } else {
    auto Rels = Obj.relas(Shdr);
    if (!Rels)
      return errorToErrorCode(Rels.takeError());
    for (const Elf_Rela &Rel : *Rels) {
      ELFYAML::Relocation R;
      if (std::error_code EC = dumpRelocation(&Rel, SymTab, R))
        return EC;
      R.Addend = Rel.r_addend;
      S->Relocations.push_back(R);
    }
  }

  return S.release();
}

template <class ELFT>
ErrorOr<ELFYAML::RawContentSection *>
ELFDumper<ELFT>::dumpContentSection(const Elf_Shdr *Shdr) {
  auto S = make_unique<ELFYAML::RawContentSection>();

  if (std::error_code EC = dumpCommonSection(Shdr, *S))
    return EC;

  auto ContentOrErr = Obj.getSectionContents(Shdr);
  if (!ContentOrErr)
    return errorToErrorCode(ContentOrErr.takeError());
  S->Content = yaml::BinaryRef(ContentOrErr.get());
  S->Size = S->Content.binary_size();

  return S.release();
}

template <class ELFT>
ErrorOr<ELFYAML::NoBitsSection *>
ELFDumper<ELFT>::dumpNoBitsSection(const Elf_Shdr *Shdr) {
  auto S = make_unique<ELFYAML::NoBitsSection>();

  if (std::error_code EC = dumpCommonSection(Shdr, *S))
    return EC;
  S->Size = Shdr->sh_size;

  return S.release();
}

template <class ELFT>
ErrorOr<ELFYAML::SymverSection *>
ELFDumper<ELFT>::dumpSymverSection(const Elf_Shdr *Shdr) {
  typedef typename ELFT::Half Elf_Half;

  auto S = make_unique<ELFYAML::SymverSection>();
  if (std::error_code EC = dumpCommonSection(Shdr, *S))
    return EC;

  auto VersionsOrErr = Obj.template getSectionContentsAsArray<Elf_Half>(Shdr);
  if (!VersionsOrErr)
    return errorToErrorCode(VersionsOrErr.takeError());
  for (const Elf_Half &E : *VersionsOrErr)
    S->Entries.push_back(E);

  return S.release();
}

template <class ELFT>
ErrorOr<ELFYAML::VerneedSection *>
ELFDumper<ELFT>::dumpVerneedSection(const Elf_Shdr *Shdr) {
  typedef typename ELFT::Verneed Elf_Verneed;
  typedef typename ELFT::Vernaux Elf_Vernaux;

  auto S = make_unique<ELFYAML::VerneedSection>();
  if (std::error_code EC = dumpCommonSection(Shdr, *S))
    return EC;

  S->Info = Shdr->sh_info;

  auto Contents = Obj.getSectionContents(Shdr);
  if (!Contents)
    return errorToErrorCode(Contents.takeError());

  auto StringTableShdrOrErr = Obj.getSection(Shdr->sh_link);
  if (!StringTableShdrOrErr)
    return errorToErrorCode(StringTableShdrOrErr.takeError());

  auto StringTableOrErr = Obj.getStringTable(*StringTableShdrOrErr);
  if (!StringTableOrErr)
    return errorToErrorCode(StringTableOrErr.takeError());

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
ErrorOr<ELFYAML::Group *> ELFDumper<ELFT>::dumpGroup(const Elf_Shdr *Shdr) {
  auto S = make_unique<ELFYAML::Group>();

  if (std::error_code EC = dumpCommonSection(Shdr, *S))
    return EC;
  // Get sh_info which is the signature.
  auto SymtabOrErr = Obj.getSection(Shdr->sh_link);
  if (!SymtabOrErr)
    return errorToErrorCode(SymtabOrErr.takeError());
  const Elf_Shdr *Symtab = *SymtabOrErr;
  auto SymOrErr = Obj.getSymbol(Symtab, Shdr->sh_info);
  if (!SymOrErr)
    return errorToErrorCode(SymOrErr.takeError());
  const Elf_Sym *symbol = *SymOrErr;
  auto StrTabOrErr = Obj.getStringTableForSymtab(*Symtab);
  if (!StrTabOrErr)
    return errorToErrorCode(StrTabOrErr.takeError());
  StringRef StrTab = *StrTabOrErr;
  auto sectionContents = Obj.getSectionContents(Shdr);
  if (!sectionContents)
    return errorToErrorCode(sectionContents.takeError());
  Expected<StringRef> symbolName = getSymbolName(symbol, StrTab, Symtab);
  if (!symbolName)
    return errorToErrorCode(symbolName.takeError());
  S->Signature = *symbolName;
  const Elf_Word *groupMembers =
      reinterpret_cast<const Elf_Word *>(sectionContents->data());
  const long count = (Shdr->sh_size) / sizeof(Elf_Word);
  ELFYAML::SectionOrType s;
  for (int i = 0; i < count; i++) {
    if (groupMembers[i] == llvm::ELF::GRP_COMDAT) {
      s.sectionNameOrType = "GRP_COMDAT";
    } else {
      auto sHdr = Obj.getSection(groupMembers[i]);
      if (!sHdr)
        return errorToErrorCode(sHdr.takeError());
      auto sectionName = getUniquedSectionName(*sHdr);
      if (!sectionName)
        return errorToErrorCode(sectionName.takeError());
      s.sectionNameOrType = *sectionName;
    }
    S->Members.push_back(s);
  }
  return S.release();
}

template <class ELFT>
ErrorOr<ELFYAML::MipsABIFlags *>
ELFDumper<ELFT>::dumpMipsABIFlags(const Elf_Shdr *Shdr) {
  assert(Shdr->sh_type == ELF::SHT_MIPS_ABIFLAGS &&
         "Section type is not SHT_MIPS_ABIFLAGS");
  auto S = make_unique<ELFYAML::MipsABIFlags>();
  if (std::error_code EC = dumpCommonSection(Shdr, *S))
    return EC;

  auto ContentOrErr = Obj.getSectionContents(Shdr);
  if (!ContentOrErr)
    return errorToErrorCode(ContentOrErr.takeError());

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
static std::error_code elf2yaml(raw_ostream &Out,
                                const object::ELFFile<ELFT> &Obj) {
  ELFDumper<ELFT> Dumper(Obj);
  ErrorOr<ELFYAML::Object *> YAMLOrErr = Dumper.dump();
  if (std::error_code EC = YAMLOrErr.getError())
    return EC;

  std::unique_ptr<ELFYAML::Object> YAML(YAMLOrErr.get());
  yaml::Output Yout(Out);
  Yout << *YAML;

  return std::error_code();
}

std::error_code elf2yaml(raw_ostream &Out, const object::ObjectFile &Obj) {
  if (const auto *ELFObj = dyn_cast<object::ELF32LEObjectFile>(&Obj))
    return elf2yaml(Out, *ELFObj->getELFFile());

  if (const auto *ELFObj = dyn_cast<object::ELF32BEObjectFile>(&Obj))
    return elf2yaml(Out, *ELFObj->getELFFile());

  if (const auto *ELFObj = dyn_cast<object::ELF64LEObjectFile>(&Obj))
    return elf2yaml(Out, *ELFObj->getELFFile());

  if (const auto *ELFObj = dyn_cast<object::ELF64BEObjectFile>(&Obj))
    return elf2yaml(Out, *ELFObj->getELFFile());

  return obj2yaml_error::unsupported_obj_file_format;
}
