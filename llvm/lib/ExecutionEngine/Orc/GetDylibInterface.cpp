//===-------- GetDylibInterface.cpp - Get interface for real dylib --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/GetDylibInterface.h"

#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/TapiUniversal.h"

#define DEBUG_TYPE "orc"

namespace llvm::orc {

Expected<SymbolNameSet> getDylibInterfaceFromDylib(ExecutionSession &ES,
                                                   Twine Path) {
  auto CPUType = MachO::getCPUType(ES.getTargetTriple());
  if (!CPUType)
    return CPUType.takeError();

  auto CPUSubType = MachO::getCPUSubType(ES.getTargetTriple());
  if (!CPUSubType)
    return CPUSubType.takeError();

  auto Buf = MemoryBuffer::getFile(Path);
  if (!Buf)
    return createFileError(Path, Buf.getError());

  auto BinFile = object::createBinary((*Buf)->getMemBufferRef());
  if (!BinFile)
    return BinFile.takeError();

  std::unique_ptr<object::MachOObjectFile> MachOFile;
  if (isa<object::MachOObjectFile>(**BinFile))
    MachOFile.reset(dyn_cast<object::MachOObjectFile>(BinFile->release()));
  else if (auto *MachOUni =
               dyn_cast<object::MachOUniversalBinary>(BinFile->get())) {
    for (auto &O : MachOUni->objects()) {
      if (O.getCPUType() == *CPUType &&
          (O.getCPUSubType() & ~MachO::CPU_SUBTYPE_MASK) == *CPUSubType) {
        if (auto Obj = O.getAsObjectFile())
          MachOFile = std::move(*Obj);
        else
          return Obj.takeError();
        break;
      }
    }
    if (!MachOFile)
      return make_error<StringError>("MachO universal binary at " + Path +
                                         " does not contain a slice for " +
                                         ES.getTargetTriple().str(),
                                     inconvertibleErrorCode());
  } else
    return make_error<StringError>("File at " + Path + " is not a MachO",
                                   inconvertibleErrorCode());

  if (MachOFile->getHeader().filetype != MachO::MH_DYLIB)
    return make_error<StringError>("MachO at " + Path + " is not a dylib",
                                   inconvertibleErrorCode());

  SymbolNameSet Symbols;
  for (auto &Sym : MachOFile->symbols()) {
    if (auto Name = Sym.getName())
      Symbols.insert(ES.intern(*Name));
    else
      return Name.takeError();
  }

  return std::move(Symbols);
}

Expected<SymbolNameSet> getDylibInterfaceFromTapiFile(ExecutionSession &ES,
                                                      Twine Path) {
  SymbolNameSet Symbols;

  auto TapiFileBuffer = MemoryBuffer::getFile(Path);
  if (!TapiFileBuffer)
    return createFileError(Path, TapiFileBuffer.getError());

  auto Tapi =
      object::TapiUniversal::create((*TapiFileBuffer)->getMemBufferRef());
  if (!Tapi)
    return Tapi.takeError();

  auto CPUType = MachO::getCPUType(ES.getTargetTriple());
  if (!CPUType)
    return CPUType.takeError();

  auto CPUSubType = MachO::getCPUSubType(ES.getTargetTriple());
  if (!CPUSubType)
    return CPUSubType.takeError();

  auto &IF = (*Tapi)->getInterfaceFile();
  auto Interface =
      IF.extract(MachO::getArchitectureFromCpuType(*CPUType, *CPUSubType));
  if (!Interface)
    return Interface.takeError();

  for (auto *Sym : (*Interface)->exports())
    Symbols.insert(ES.intern(Sym->getName()));

  return Symbols;
}

Expected<SymbolNameSet> getDylibInterface(ExecutionSession &ES, Twine Path) {
  file_magic Magic;
  if (auto EC = identify_magic(Path, Magic))
    return createFileError(Path, EC);

  switch (Magic) {
  case file_magic::macho_universal_binary:
  case file_magic::macho_dynamically_linked_shared_lib:
    return getDylibInterfaceFromDylib(ES, Path);
  case file_magic::tapi_file:
    return getDylibInterfaceFromTapiFile(ES, Path);
  default:
    return make_error<StringError>("Cannot get interface for " + Path +
                                       " unrecognized file type",
                                   inconvertibleErrorCode());
  }
}

} // namespace llvm::orc
