//===------ utils/elf2yaml.cpp - obj2yaml conversion tool -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Error.h"
#include "obj2yaml.h"
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
  typedef typename object::ELFFile<ELFT>::Elf_Shdr Elf_Shdr;
  typedef typename object::ELFFile<ELFT>::Elf_Word Elf_Word;
  typedef typename object::ELFFile<ELFT>::Elf_Rel Elf_Rel;
  typedef typename object::ELFFile<ELFT>::Elf_Rela Elf_Rela;

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

  std::error_code dumpSymbol(const Elf_Sym *Sym, const Elf_Shdr *SymTab,
                             StringRef StrTable, ELFYAML::Symbol &S);
  std::error_code dumpCommonSection(const Elf_Shdr *Shdr, ELFYAML::Section &S);
  std::error_code dumpCommonRelocationSection(const Elf_Shdr *Shdr,
                                              ELFYAML::RelocationSection &S);
  template <class RelT>
  std::error_code dumpRelocation(const RelT *Rel, const Elf_Shdr *SymTab,
                                 ELFYAML::Relocation &R);

  ErrorOr<ELFYAML::RelocationSection *> dumpRelSection(const Elf_Shdr *Shdr);
  ErrorOr<ELFYAML::RelocationSection *> dumpRelaSection(const Elf_Shdr *Shdr);
  ErrorOr<ELFYAML::RawContentSection *>
  dumpContentSection(const Elf_Shdr *Shdr);
  ErrorOr<ELFYAML::NoBitsSection *> dumpNoBitsSection(const Elf_Shdr *Shdr);
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
  Y->Header.Type = Obj.getHeader()->e_type;
  Y->Header.Machine = Obj.getHeader()->e_machine;
  Y->Header.Flags = Obj.getHeader()->e_flags;
  Y->Header.Entry = Obj.getHeader()->e_entry;

  const Elf_Shdr *Symtab = nullptr;

  // Dump sections
  auto SectionsOrErr = Obj.sections();
  if (!SectionsOrErr)
    return errorToErrorCode(SectionsOrErr.takeError());
  Sections = *SectionsOrErr;
  SectionNames.resize(Sections.size());
  for (const Elf_Shdr &Sec : Sections) {
    switch (Sec.sh_type) {
    case ELF::SHT_NULL:
    case ELF::SHT_DYNSYM:
    case ELF::SHT_STRTAB:
      // Do not dump these sections.
      break;
    case ELF::SHT_SYMTAB:
      Symtab = &Sec;
      break;
    case ELF::SHT_SYMTAB_SHNDX: {
      auto TableOrErr = Obj.getSHNDXTable(Sec);
      if (!TableOrErr)
        return errorToErrorCode(TableOrErr.takeError());
      ShndxTable = *TableOrErr;
      break;
    }
    case ELF::SHT_RELA: {
      ErrorOr<ELFYAML::RelocationSection *> S = dumpRelaSection(&Sec);
      if (std::error_code EC = S.getError())
        return EC;
      Y->Sections.push_back(std::unique_ptr<ELFYAML::Section>(S.get()));
      break;
    }
    case ELF::SHT_REL: {
      ErrorOr<ELFYAML::RelocationSection *> S = dumpRelSection(&Sec);
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
    default: {
      ErrorOr<ELFYAML::RawContentSection *> S = dumpContentSection(&Sec);
      if (std::error_code EC = S.getError())
        return EC;
      Y->Sections.push_back(std::unique_ptr<ELFYAML::Section>(S.get()));
    }
    }
  }

  // Dump symbols
  auto StrTableOrErr = Obj.getStringTableForSymtab(*Symtab);
  if (!StrTableOrErr)
    return errorToErrorCode(StrTableOrErr.takeError());
  StringRef StrTable = *StrTableOrErr;

  bool IsFirstSym = true;
  auto SymtabOrErr = Obj.symbols(Symtab);
  if (!SymtabOrErr)
    return errorToErrorCode(SymtabOrErr.takeError());
  for (const Elf_Sym &Sym : *SymtabOrErr) {
    if (IsFirstSym) {
      IsFirstSym = false;
      continue;
    }

    ELFYAML::Symbol S;
    if (std::error_code EC =
            ELFDumper<ELFT>::dumpSymbol(&Sym, Symtab, StrTable, S))
      return EC;

    switch (Sym.getBinding())
    {
    case ELF::STB_LOCAL:
      Y->Symbols.Local.push_back(S);
      break;
    case ELF::STB_GLOBAL:
      Y->Symbols.Global.push_back(S);
      break;
    case ELF::STB_WEAK:
      Y->Symbols.Weak.push_back(S);
      break;
    default:
      llvm_unreachable("Unknown ELF symbol binding");
    }
  }

  return Y.release();
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
  S.Info = NameOrErr.get();

  return obj2yaml_error::success;
}

template <class ELFT>
ErrorOr<ELFYAML::RelocationSection *>
ELFDumper<ELFT>::dumpRelSection(const Elf_Shdr *Shdr) {
  assert(Shdr->sh_type == ELF::SHT_REL && "Section type is not SHT_REL");
  auto S = make_unique<ELFYAML::RelocationSection>();

  if (std::error_code EC = dumpCommonRelocationSection(Shdr, *S))
    return EC;

  auto SymTabOrErr = Obj.getSection(Shdr->sh_link);
  if (!SymTabOrErr)
    return errorToErrorCode(SymTabOrErr.takeError());
  const Elf_Shdr *SymTab = *SymTabOrErr;

  auto Rels = Obj.rels(Shdr);
  if (!Rels)
    return errorToErrorCode(Rels.takeError());
  for (const Elf_Rel &Rel : *Rels) {
    ELFYAML::Relocation R;
    if (std::error_code EC = dumpRelocation(&Rel, SymTab, R))
      return EC;
    S->Relocations.push_back(R);
  }

  return S.release();
}

template <class ELFT>
ErrorOr<ELFYAML::RelocationSection *>
ELFDumper<ELFT>::dumpRelaSection(const Elf_Shdr *Shdr) {
  assert(Shdr->sh_type == ELF::SHT_RELA && "Section type is not SHT_RELA");
  auto S = make_unique<ELFYAML::RelocationSection>();

  if (std::error_code EC = dumpCommonRelocationSection(Shdr, *S))
    return EC;

  auto SymTabOrErr = Obj.getSection(Shdr->sh_link);
  if (!SymTabOrErr)
    return errorToErrorCode(SymTabOrErr.takeError());
  const Elf_Shdr *SymTab = *SymTabOrErr;

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
  S->Info = *symbolName;
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
