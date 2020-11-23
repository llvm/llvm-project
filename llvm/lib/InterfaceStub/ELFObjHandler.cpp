//===- ELFObjHandler.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===/

#include "llvm/InterfaceStub/ELFObjHandler.h"
#include "llvm/InterfaceStub/ELFStub.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

using llvm::MemoryBufferRef;
using llvm::object::ELFObjectFile;

using namespace llvm;
using namespace llvm::object;
using namespace llvm::ELF;

namespace llvm {
namespace elfabi {

// Simple struct to hold relevant .dynamic entries.
struct DynamicEntries {
  uint64_t StrTabAddr = 0;
  uint64_t StrSize = 0;
  Optional<uint64_t> SONameOffset;
  std::vector<uint64_t> NeededLibNames;
  // Symbol table:
  uint64_t DynSymAddr = 0;
  // Hash tables:
  Optional<uint64_t> ElfHash;
  Optional<uint64_t> GnuHash;
};

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
  return createError(Stream.str().c_str());
}

/// This function populates a DynamicEntries struct using an ELFT::DynRange.
/// After populating the struct, the members are validated with
/// some basic sanity checks.
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
  if (Dyn.SONameOffset.hasValue() && *Dyn.SONameOffset >= Dyn.StrSize) {
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

/// This function finds the number of dynamic symbols using a GNU hash table.
///
/// @param Table The GNU hash table for .dynsym.
template <class ELFT>
static uint64_t getDynSymtabSize(const typename ELFT::GnuHash &Table) {
  using Elf_Word = typename ELFT::Word;
  if (Table.nbuckets == 0)
    return Table.symndx + 1;
  uint64_t LastSymIdx = 0;
  uint64_t BucketVal = 0;
  // Find the index of the first symbol in the last chain.
  for (Elf_Word Val : Table.buckets()) {
    BucketVal = std::max(BucketVal, (uint64_t)Val);
  }
  LastSymIdx += BucketVal;
  const Elf_Word *It =
      reinterpret_cast<const Elf_Word *>(Table.values(BucketVal).end());
  // Locate the end of the chain to find the last symbol index.
  while ((*It & 1) == 0) {
    LastSymIdx++;
    It++;
  }
  return LastSymIdx + 1;
}

/// This function determines the number of dynamic symbols.
/// Without access to section headers, the number of symbols must be determined
/// by parsing dynamic hash tables.
///
/// @param Dyn Entries with the locations of hash tables.
/// @param ElfFile The ElfFile that the section contents reside in.
template <class ELFT>
static Expected<uint64_t> getNumSyms(DynamicEntries &Dyn,
                                     const ELFFile<ELFT> &ElfFile) {
  using Elf_Hash = typename ELFT::Hash;
  using Elf_GnuHash = typename ELFT::GnuHash;
  // Search GNU hash table to try to find the upper bound of dynsym.
  if (Dyn.GnuHash.hasValue()) {
    Expected<const uint8_t *> TablePtr = ElfFile.toMappedAddr(*Dyn.GnuHash);
    if (!TablePtr)
      return TablePtr.takeError();
    const Elf_GnuHash *Table =
        reinterpret_cast<const Elf_GnuHash *>(TablePtr.get());
    return getDynSymtabSize<ELFT>(*Table);
  }
  // Search SYSV hash table to try to find the upper bound of dynsym.
  if (Dyn.ElfHash.hasValue()) {
    Expected<const uint8_t *> TablePtr = ElfFile.toMappedAddr(*Dyn.ElfHash);
    if (!TablePtr)
      return TablePtr.takeError();
    const Elf_Hash *Table = reinterpret_cast<const Elf_Hash *>(TablePtr.get());
    return Table->nchain;
  }
  return 0;
}

/// This function extracts symbol type from a symbol's st_info member and
/// maps it to an ELFSymbolType enum.
/// Currently, STT_NOTYPE, STT_OBJECT, STT_FUNC, and STT_TLS are supported.
/// Other symbol types are mapped to ELFSymbolType::Unknown.
///
/// @param Info Binary symbol st_info to extract symbol type from.
static ELFSymbolType convertInfoToType(uint8_t Info) {
  Info = Info & 0xf;
  switch (Info) {
  case ELF::STT_NOTYPE:
    return ELFSymbolType::NoType;
  case ELF::STT_OBJECT:
    return ELFSymbolType::Object;
  case ELF::STT_FUNC:
    return ELFSymbolType::Func;
  case ELF::STT_TLS:
    return ELFSymbolType::TLS;
  default:
    return ELFSymbolType::Unknown;
  }
}

/// This function creates an ELFSymbol and populates all members using
/// information from a binary ELFT::Sym.
///
/// @param SymName The desired name of the ELFSymbol.
/// @param RawSym ELFT::Sym to extract symbol information from.
template <class ELFT>
static ELFSymbol createELFSym(StringRef SymName,
                              const typename ELFT::Sym &RawSym) {
  ELFSymbol TargetSym{std::string(SymName)};
  uint8_t Binding = RawSym.getBinding();
  if (Binding == STB_WEAK)
    TargetSym.Weak = true;
  else
    TargetSym.Weak = false;

  TargetSym.Undefined = RawSym.isUndefined();
  TargetSym.Type = convertInfoToType(RawSym.st_info);

  if (TargetSym.Type == ELFSymbolType::Func) {
    TargetSym.Size = 0;
  } else {
    TargetSym.Size = RawSym.st_size;
  }
  return TargetSym;
}

/// This function populates an ELFStub with symbols using information read
/// from an ELF binary.
///
/// @param TargetStub ELFStub to add symbols to.
/// @param DynSym Range of dynamic symbols to add to TargetStub.
/// @param DynStr StringRef to the dynamic string table.
template <class ELFT>
static Error populateSymbols(ELFStub &TargetStub,
                             const typename ELFT::SymRange DynSym,
                             StringRef DynStr) {
  // Skips the first symbol since it's the NULL symbol.
  for (auto RawSym : DynSym.drop_front(1)) {
    // If a symbol does not have global or weak binding, ignore it.
    uint8_t Binding = RawSym.getBinding();
    if (!(Binding == STB_GLOBAL || Binding == STB_WEAK))
      continue;
    // If a symbol doesn't have default or protected visibility, ignore it.
    uint8_t Visibility = RawSym.getVisibility();
    if (!(Visibility == STV_DEFAULT || Visibility == STV_PROTECTED))
      continue;
    // Create an ELFSymbol and populate it with information from the symbol
    // table entry.
    Expected<StringRef> SymName = terminatedSubstr(DynStr, RawSym.st_name);
    if (!SymName)
      return SymName.takeError();
    ELFSymbol Sym = createELFSym<ELFT>(*SymName, RawSym);
    TargetStub.Symbols.insert(std::move(Sym));
    // TODO: Populate symbol warning.
  }
  return Error::success();
}

/// Returns a new ELFStub with all members populated from an ELFObjectFile.
/// @param ElfObj Source ELFObjectFile.
template <class ELFT>
static Expected<std::unique_ptr<ELFStub>>
buildStub(const ELFObjectFile<ELFT> &ElfObj) {
  using Elf_Dyn_Range = typename ELFT::DynRange;
  using Elf_Phdr_Range = typename ELFT::PhdrRange;
  using Elf_Sym_Range = typename ELFT::SymRange;
  using Elf_Sym = typename ELFT::Sym;
  std::unique_ptr<ELFStub> DestStub = std::make_unique<ELFStub>();
  const ELFFile<ELFT> *ElfFile = ElfObj.getELFFile();
  // Fetch .dynamic table.
  Expected<Elf_Dyn_Range> DynTable = ElfFile->dynamicEntries();
  if (!DynTable) {
    return DynTable.takeError();
  }

  // Fetch program headers.
  Expected<Elf_Phdr_Range> PHdrs = ElfFile->program_headers();
  if (!PHdrs) {
    return PHdrs.takeError();
  }

  // Collect relevant .dynamic entries.
  DynamicEntries DynEnt;
  if (Error Err = populateDynamic<ELFT>(DynEnt, *DynTable))
    return std::move(Err);

  // Get pointer to in-memory location of .dynstr section.
  Expected<const uint8_t *> DynStrPtr =
      ElfFile->toMappedAddr(DynEnt.StrTabAddr);
  if (!DynStrPtr)
    return appendToError(DynStrPtr.takeError(),
                         "when locating .dynstr section contents");

  StringRef DynStr(reinterpret_cast<const char *>(DynStrPtr.get()),
                   DynEnt.StrSize);

  // Populate Arch from ELF header.
  DestStub->Arch = ElfFile->getHeader().e_machine;

  // Populate SoName from .dynamic entries and dynamic string table.
  if (DynEnt.SONameOffset.hasValue()) {
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

  // Populate Symbols from .dynsym table and dynamic string table.
  Expected<uint64_t> SymCount = getNumSyms(DynEnt, *ElfFile);
  if (!SymCount)
    return SymCount.takeError();
  if (*SymCount > 0) {
    // Get pointer to in-memory location of .dynsym section.
    Expected<const uint8_t *> DynSymPtr =
        ElfFile->toMappedAddr(DynEnt.DynSymAddr);
    if (!DynSymPtr)
      return appendToError(DynSymPtr.takeError(),
                           "when locating .dynsym section contents");
    Elf_Sym_Range DynSyms = ArrayRef<Elf_Sym>(
        reinterpret_cast<const Elf_Sym *>(*DynSymPtr), *SymCount);
    Error SymReadError = populateSymbols<ELFT>(*DestStub, DynSyms, DynStr);
    if (SymReadError)
      return appendToError(std::move(SymReadError),
                           "when reading dynamic symbols");
  }

  return std::move(DestStub);
}

Expected<std::unique_ptr<ELFStub>> readELFFile(MemoryBufferRef Buf) {
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

  return createStringError(errc::not_supported, "Unsupported binary format");
}

} // end namespace elfabi
} // end namespace llvm
