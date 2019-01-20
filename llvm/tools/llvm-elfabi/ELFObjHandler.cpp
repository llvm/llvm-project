//===- ELFObjHandler.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===/

#include "ELFObjHandler.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TextAPI/ELF/ELFStub.h"

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
  if (Dyn.SONameOffset.hasValue() && *Dyn.SONameOffset >= Dyn.StrSize) {
    return createStringError(
        object_error::parse_failed,
        "DT_SONAME string offset (0x%016x) outside of dynamic string table",
        *Dyn.SONameOffset);
  }
  for (uint64_t Offset : Dyn.NeededLibNames) {
    if (Offset >= Dyn.StrSize) {
      return createStringError(
          object_error::parse_failed,
          "DT_NEEDED string offset (0x%016x) outside of dynamic string table",
          Offset);
    }
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
  std::unique_ptr<ELFStub> DestStub = make_unique<ELFStub>();
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

  // Convert .dynstr address to an offset.
  Expected<const uint8_t *> DynStrPtr =
      ElfFile->toMappedAddr(DynEnt.StrTabAddr);
  if (!DynStrPtr)
    return appendToError(DynStrPtr.takeError(),
                         "when locating .dynstr section contents");

  StringRef DynStr(reinterpret_cast<const char *>(DynStrPtr.get()),
                   DynEnt.StrSize);

  // Populate Arch from ELF header.
  DestStub->Arch = ElfFile->getHeader()->e_machine;

  // Populate SoName from .dynamic entries and dynamic string table.
  if (DynEnt.SONameOffset.hasValue()) {
    Expected<StringRef> NameOrErr =
        terminatedSubstr(DynStr, *DynEnt.SONameOffset);
    if (!NameOrErr) {
      return appendToError(NameOrErr.takeError(), "when reading DT_SONAME");
    }
    DestStub->SoName = *NameOrErr;
  }

  // Populate NeededLibs from .dynamic entries and dynamic string table.
  for (uint64_t NeededStrOffset : DynEnt.NeededLibNames) {
    Expected<StringRef> LibNameOrErr =
        terminatedSubstr(DynStr, NeededStrOffset);
    if (!LibNameOrErr) {
      return appendToError(LibNameOrErr.takeError(), "when reading DT_NEEDED");
    }
    DestStub->NeededLibs.push_back(*LibNameOrErr);
  }

  // TODO: Populate Symbols from .dynsym table and linked string table.

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
