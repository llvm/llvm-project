//===----------------- MachO.cpp - MachO format utilities -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/MachO.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Support/FileSystem.h"

#define DEBUG_TYPE "orc"

namespace llvm {
namespace orc {

static std::string objDesc(const MemoryBufferRef &Obj, const Triple &TT,
                           bool ObjIsSlice) {
  std::string Desc;
  if (ObjIsSlice)
    Desc += (TT.getArchName() + " slice of universal binary").str();
  Desc += Obj.getBufferIdentifier();
  return Desc;
}

template <typename HeaderType>
static Error checkMachORelocatableObject(MemoryBufferRef Obj,
                                         bool SwapEndianness, const Triple &TT,
                                         bool ObjIsSlice) {
  StringRef Data = Obj.getBuffer();

  HeaderType Hdr;
  memcpy(&Hdr, Data.data(), sizeof(HeaderType));

  if (SwapEndianness)
    swapStruct(Hdr);

  if (Hdr.filetype != MachO::MH_OBJECT)
    return make_error<StringError>(objDesc(Obj, TT, ObjIsSlice) +
                                       " is not a MachO relocatable object",
                                   inconvertibleErrorCode());

  auto ObjArch = object::MachOObjectFile::getArch(Hdr.cputype, Hdr.cpusubtype);
  if (ObjArch != TT.getArch())
    return make_error<StringError>(
        objDesc(Obj, TT, ObjIsSlice) + Triple::getArchTypeName(ObjArch) +
            ", cannot be loaded into " + TT.str() + " process",
        inconvertibleErrorCode());

  return Error::success();
}

Error checkMachORelocatableObject(MemoryBufferRef Obj, const Triple &TT,
                                  bool ObjIsSlice) {
  StringRef Data = Obj.getBuffer();

  if (Data.size() < 4)
    return make_error<StringError>(
        objDesc(Obj, TT, ObjIsSlice) +
            " is not a valid MachO relocatable object file (truncated header)",
        inconvertibleErrorCode());

  uint32_t Magic;
  memcpy(&Magic, Data.data(), sizeof(uint32_t));

  switch (Magic) {
  case MachO::MH_MAGIC:
  case MachO::MH_CIGAM:
    return checkMachORelocatableObject<MachO::mach_header>(
        std::move(Obj), Magic == MachO::MH_CIGAM, TT, ObjIsSlice);
  case MachO::MH_MAGIC_64:
  case MachO::MH_CIGAM_64:
    return checkMachORelocatableObject<MachO::mach_header_64>(
        std::move(Obj), Magic == MachO::MH_CIGAM_64, TT, ObjIsSlice);
  default:
    return make_error<StringError>(
        objDesc(Obj, TT, ObjIsSlice) +
            " is not a valid MachO relocatable object (bad magic value)",
        inconvertibleErrorCode());
  }
}

Expected<std::unique_ptr<MemoryBuffer>>
checkMachORelocatableObject(std::unique_ptr<MemoryBuffer> Obj, const Triple &TT,
                            bool ObjIsSlice) {
  if (auto Err =
          checkMachORelocatableObject(Obj->getMemBufferRef(), TT, ObjIsSlice))
    return std::move(Err);
  return std::move(Obj);
}

Expected<std::pair<std::unique_ptr<MemoryBuffer>, LinkableFileKind>>
loadMachORelocatableObject(StringRef Path, const Triple &TT, LoadArchives LA,
                           std::optional<StringRef> IdentifierOverride) {
  assert((TT.getObjectFormat() == Triple::UnknownObjectFormat ||
          TT.getObjectFormat() == Triple::MachO) &&
         "TT must specify MachO or Unknown object format");

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
        StringRef("Could not load MachO object at path ") + Path,
        Buf.getError());

  switch (identify_magic((*Buf)->getBuffer())) {
  case file_magic::macho_object: {
    auto CheckedObj = checkMachORelocatableObject(std::move(*Buf), TT, false);
    if (!CheckedObj)
      return CheckedObj.takeError();
    return std::make_pair(std::move(*CheckedObj),
                          LinkableFileKind::RelocatableObject);
  }
  case file_magic::macho_universal_binary:
    return loadLinkableSliceFromMachOUniversalBinary(FD, std::move(*Buf), TT,
                                                     LoadArchives::Never, Path,
                                                     *IdentifierOverride);
  default:
    return make_error<StringError>(
        Path + " does not contain a relocatable object file compatible with " +
            TT.str(),
        inconvertibleErrorCode());
  }
}

Expected<std::pair<std::unique_ptr<MemoryBuffer>, LinkableFileKind>>
loadLinkableSliceFromMachOUniversalBinary(sys::fs::file_t FD,
                                          std::unique_ptr<MemoryBuffer> UBBuf,
                                          const Triple &TT, LoadArchives LA,
                                          StringRef UBPath,
                                          StringRef Identifier) {

  auto UniversalBin =
      object::MachOUniversalBinary::create(UBBuf->getMemBufferRef());
  if (!UniversalBin)
    return UniversalBin.takeError();

  auto SliceRange = getMachOSliceRangeForTriple(**UniversalBin, TT);
  if (!SliceRange)
    return SliceRange.takeError();

  auto Buf = MemoryBuffer::getOpenFileSlice(FD, Identifier, SliceRange->second,
                                            SliceRange->first);
  if (!Buf)
    return make_error<StringError>(
        "Could not load " + TT.getArchName() +
            " slice of MachO universal binary at path " + UBPath,
        Buf.getError());

  switch (identify_magic((*Buf)->getBuffer())) {
  case file_magic::archive:
    if (LA != LoadArchives::Never)
      return std::make_pair(std::move(*Buf), LinkableFileKind::Archive);
    break;
  case file_magic::macho_object: {
    if (LA != LoadArchives::Required) {
      auto CheckedObj = checkMachORelocatableObject(std::move(*Buf), TT, true);
      if (!CheckedObj)
        return CheckedObj.takeError();
      return std::make_pair(std::move(*CheckedObj),
                            LinkableFileKind::RelocatableObject);
    }
    break;
  }
  default:
    break;
  }

  auto FT = [&] {
    switch (LA) {
    case LoadArchives::Never:
      return "a mach-o relocatable object file";
    case LoadArchives::Allowed:
      return "a mach-o relocatable object file or archive";
    case LoadArchives::Required:
      return "an archive";
    }
    llvm_unreachable("Unknown LoadArchives enum");
  };

  return make_error<StringError>(TT.getArchName() + " slice of " + UBPath +
                                     " does not contain " + FT(),
                                 inconvertibleErrorCode());
}

Expected<std::pair<size_t, size_t>>
getMachOSliceRangeForTriple(object::MachOUniversalBinary &UB,
                            const Triple &TT) {

  for (const auto &Obj : UB.objects()) {
    auto ObjTT = Obj.getTriple();
    if (ObjTT.getArch() == TT.getArch() &&
        ObjTT.getSubArch() == TT.getSubArch() &&
        (TT.getVendor() == Triple::UnknownVendor ||
         ObjTT.getVendor() == TT.getVendor())) {
      // We found a match. Return the range for the slice.
      return std::make_pair(Obj.getOffset(), Obj.getSize());
    }
  }

  return make_error<StringError>(Twine("Universal binary ") + UB.getFileName() +
                                     " does not contain a slice for " +
                                     TT.str(),
                                 inconvertibleErrorCode());
}

Expected<std::pair<size_t, size_t>>
getMachOSliceRangeForTriple(MemoryBufferRef UBBuf, const Triple &TT) {

  auto UB = object::MachOUniversalBinary::create(UBBuf);
  if (!UB)
    return UB.takeError();

  return getMachOSliceRangeForTriple(**UB, TT);
}

} // End namespace orc.
} // End namespace llvm.
