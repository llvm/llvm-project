//===----- LoadRelocatableObject.cpp -- Load relocatable object files -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/LoadRelocatableObject.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/ExecutionEngine/Orc/MachO.h"

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

Expected<std::unique_ptr<MemoryBuffer>>
loadRelocatableObject(StringRef Path, const Triple &TT) {
  auto Buf = MemoryBuffer::getFile(Path);
  if (!Buf)
    return createFileError(Path, Buf.getError());

  std::optional<Triple::ObjectFormatType> RequireFormat;
  if (TT.getObjectFormat() != Triple::UnknownObjectFormat)
    RequireFormat = TT.getObjectFormat();

  switch (identify_magic((*Buf)->getBuffer())) {
  case file_magic::coff_object:
    if (!RequireFormat || *RequireFormat == Triple::COFF)
      return checkCOFFRelocatableObject(std::move(*Buf), TT);
    break;
  case file_magic::elf_relocatable:
    if (!RequireFormat || *RequireFormat == Triple::ELF)
      return checkELFRelocatableObject(std::move(*Buf), TT);
    break;
  case file_magic::macho_object:
    if (!RequireFormat || *RequireFormat == Triple::MachO)
      return checkMachORelocatableObject(std::move(*Buf), TT, false);
    break;
  case file_magic::macho_universal_binary:
    if (!RequireFormat || *RequireFormat == Triple::MachO)
      return loadMachORelocatableObjectFromUniversalBinary(Path,
                                                           std::move(*Buf), TT);
    break;
  default:
    break;
  }
  return make_error<StringError>(
      Path + " does not contain a relocatable object file compatible with " +
          TT.str(),
      inconvertibleErrorCode());
}

} // End namespace orc.
} // End namespace llvm.
