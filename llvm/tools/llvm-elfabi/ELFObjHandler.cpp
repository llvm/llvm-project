//===- ELFObjHandler.cpp --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
using namespace llvm::elfabi;
using namespace llvm::ELF;

namespace llvm {
namespace elfabi {

/// Returns a new ELFStub with all members populated from an ELFObjectFile.
/// @param ElfObj Source ELFObjectFile.
template <class ELFT>
Expected<std::unique_ptr<ELFStub>>
buildStub(const ELFObjectFile<ELFT> &ElfObj) {
  std::unique_ptr<ELFStub> DestStub = make_unique<ELFStub>();
  const ELFFile<ELFT> *ElfFile = ElfObj.getELFFile();

  DestStub->Arch = ElfFile->getHeader()->e_machine;

  // TODO: Populate SoName from .dynamic entries and linked string table.
  // TODO: Populate NeededLibs from .dynamic entries and linked string table.
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
