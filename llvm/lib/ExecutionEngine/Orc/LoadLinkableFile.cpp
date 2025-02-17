//===------- LoadLinkableFile.cpp -- Load relocatables and archives -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/LoadLinkableFile.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/ExecutionEngine/Orc/MachO.h"
#include "llvm/Support/FileSystem.h"

#define DEBUG_TYPE "orc"

namespace llvm {
namespace orc {

static Expected<std::unique_ptr<MemoryBuffer>>
checkCOFFRelocatableObject(std::unique_ptr<MemoryBuffer> Obj,
                           const Triple &TT) {
  // TODO: Actually check the architecture of the file.
  return std::move(Obj);
}

static Expected<std::unique_ptr<MemoryBuffer>>
checkELFRelocatableObject(std::unique_ptr<MemoryBuffer> Obj, const Triple &TT) {
  // TODO: Actually check the architecture of the file.
  return std::move(Obj);
}

Expected<std::pair<std::unique_ptr<MemoryBuffer>, LinkableFileKind>>
loadLinkableFile(StringRef Path, const Triple &TT, LoadArchives LA,
                 std::optional<StringRef> IdentifierOverride) {
  if (!IdentifierOverride)
    IdentifierOverride = Path;

  Expected<sys::fs::file_t> FDOrErr =
      sys::fs::openNativeFileForRead(Path, sys::fs::OF_None);
  if (!FDOrErr)
    return createFileError(Path, FDOrErr.takeError());
  sys::fs::file_t FD = *FDOrErr;
  auto CloseFile = make_scope_exit([&]() { sys::fs::closeFile(FD); });

  auto Buf =
      MemoryBuffer::getOpenFile(FD, *IdentifierOverride, /*FileSize=*/-1);
  if (!Buf)
    return make_error<StringError>(
        StringRef("Could not load object at path ") + Path, Buf.getError());

  std::optional<Triple::ObjectFormatType> RequireFormat;
  if (TT.getObjectFormat() != Triple::UnknownObjectFormat)
    RequireFormat = TT.getObjectFormat();

  switch (identify_magic((*Buf)->getBuffer())) {
  case file_magic::archive:
    if (LA != LoadArchives::Never)
      return std::make_pair(std::move(*Buf), LinkableFileKind::Archive);
    return make_error<StringError>(
        Path + " does not contain a relocatable object file",
        inconvertibleErrorCode());
  case file_magic::coff_object:
    if (LA == LoadArchives::Required)
      return make_error<StringError>(Path + " does not contain an archive",
                                     inconvertibleErrorCode());

    if (!RequireFormat || *RequireFormat == Triple::COFF) {
      auto CheckedBuf = checkCOFFRelocatableObject(std::move(*Buf), TT);
      if (!CheckedBuf)
        return CheckedBuf.takeError();
      return std::make_pair(std::move(*CheckedBuf),
                            LinkableFileKind::RelocatableObject);
    }
    break;
  case file_magic::elf_relocatable:
    if (LA == LoadArchives::Required)
      return make_error<StringError>(Path + " does not contain an archive",
                                     inconvertibleErrorCode());

    if (!RequireFormat || *RequireFormat == Triple::ELF) {
      auto CheckedBuf = checkELFRelocatableObject(std::move(*Buf), TT);
      if (!CheckedBuf)
        return CheckedBuf.takeError();
      return std::make_pair(std::move(*CheckedBuf),
                            LinkableFileKind::RelocatableObject);
    }
    break;
  case file_magic::macho_object:
    if (LA == LoadArchives::Required)
      return make_error<StringError>(Path + " does not contain an archive",
                                     inconvertibleErrorCode());

    if (!RequireFormat || *RequireFormat == Triple::MachO) {
      auto CheckedBuf = checkMachORelocatableObject(std::move(*Buf), TT, false);
      if (!CheckedBuf)
        return CheckedBuf.takeError();
      return std::make_pair(std::move(*CheckedBuf),
                            LinkableFileKind::RelocatableObject);
    }
    break;
  case file_magic::macho_universal_binary:
    if (!RequireFormat || *RequireFormat == Triple::MachO)
      return loadLinkableSliceFromMachOUniversalBinary(
          FD, std::move(*Buf), TT, LA, Path, *IdentifierOverride);
    break;
  default:
    break;
  }

  return make_error<StringError>(
      Path +
          " does not contain a relocatable object file or archive compatible "
          "with " +
          TT.str(),
      inconvertibleErrorCode());
}

} // End namespace orc.
} // End namespace llvm.
