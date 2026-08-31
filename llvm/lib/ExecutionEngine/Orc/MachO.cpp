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
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/ExecutionEngine/Orc/ExecutionUtils.h"
#include "llvm/ExecutionEngine/Orc/Layer.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/TapiUniversal.h"
#include "llvm/Support/FileSystem.h"

#define DEBUG_TYPE "orc"

namespace llvm {
namespace orc {

static std::string objDesc(const MemoryBufferRef &Obj, const Triple &TT,
                           bool ObjIsSlice) {
  std::string Desc;
  if (ObjIsSlice)
    Desc += (TT.getArchName() + " slice of universal binary ").str();
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

  auto ObjArchTT =
      object::MachOObjectFile::getArchTriple(Hdr.cputype, Hdr.cpusubtype);
  if (ObjArchTT.getArch() != TT.getArch())
    return make_error<StringError>(
        objDesc(Obj, TT, ObjIsSlice) + " (" + ObjArchTT.getArchName() +
            "), cannot be loaded into " + TT.str() + " process",
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
  llvm::scope_exit CloseFile([&]() { sys::fs::closeFile(FD); });

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

Expected<bool> ForceLoadMachOArchiveMembers::operator()(
    object::Archive &A, MemoryBufferRef MemberBuf, size_t Index) {

  auto LoadMember = [&]() {
    return StaticLibraryDefinitionGenerator::createMemberBuffer(A, MemberBuf,
                                                                Index);
  };

  if (!ObjCOnly) {
    // If we're loading all files then just load the buffer immediately. Return
    // false to indicate that there's no further loading to do here.
    if (auto Err = L.add(JD, LoadMember()))
      return Err;
    return false;
  }

  // We need to check whether this archive member contains any Objective-C
  // or Swift metadata.
  auto Obj = object::ObjectFile::createObjectFile(MemberBuf);
  if (!Obj) {
    // Invalid files are not loadable, but don't invalidate the archive.
    consumeError(Obj.takeError());
    return false;
  }

  if (auto *MachOObj = dyn_cast<object::MachOObjectFile>(&**Obj)) {
    // Load the object if any recognized special section is present.
    for (auto Sec : MachOObj->sections()) {
      auto SegName =
          MachOObj->getSectionFinalSegmentName(Sec.getRawDataRefImpl());
      if (auto SecName = Sec.getName()) {
        if (*SecName == "__objc_classlist" || *SecName == "__objc_protolist" ||
            *SecName == "__objc_clsrolist" || *SecName == "__objc_catlist" ||
            *SecName == "__objc_catlist2" || *SecName == "__objc_nlcatlist" ||
            (SegName == "__TEXT" && (*SecName).starts_with("__swift") &&
             *SecName != "__swift_modhash")) {
          if (auto Err = L.add(JD, LoadMember()))
            return Err;
          return false;
        }
      } else
        return SecName.takeError();
    }
  }

  // This is an object file but we didn't load it, so return true to indicate
  // that it's still loadable.
  return true;
}

SmallVector<std::pair<uint32_t, uint32_t>>
noFallbackArchs(uint32_t CPUType, uint32_t CPUSubType) {
  SmallVector<std::pair<uint32_t, uint32_t>> Result;
  Result.push_back({CPUType, CPUSubType});
  return Result;
}

SmallVector<std::pair<uint32_t, uint32_t>>
standardMachOFallbackArchs(uint32_t CPUType, uint32_t CPUSubType) {
  SmallVector<std::pair<uint32_t, uint32_t>> Archs;

  // Match given CPU type/subtype first.
  Archs.push_back({CPUType, CPUSubType});

  switch (CPUType) {
  case MachO::CPU_TYPE_ARM64:
    // Handle arm64 variants.
    switch (CPUSubType) {
    case MachO::CPU_SUBTYPE_ARM64_ALL:
      Archs.push_back({CPUType, MachO::CPU_SUBTYPE_ARM64E});
      break;
    default:
      break;
    }
    break;
  default:
    break;
  }

  return Archs;
}

Expected<SymbolNameSet>
getDylibInterfaceFromDylib(ExecutionSession &ES, Twine Path,
                           GetFallbackArchsFn GetFallbackArchs) {
  auto InitCPUType = MachO::getCPUType(ES.getTargetTriple());
  if (!InitCPUType)
    return InitCPUType.takeError();

  auto InitCPUSubType = MachO::getCPUSubType(ES.getTargetTriple());
  if (!InitCPUSubType)
    return InitCPUSubType.takeError();

  auto Buf = MemoryBuffer::getFile(Path);
  if (!Buf)
    return createFileError(Path, Buf.getError());

  auto BinFile = object::createBinary((*Buf)->getMemBufferRef());
  if (!BinFile)
    return BinFile.takeError();

  std::unique_ptr<object::MachOObjectFile> MachOFile;
  if (isa<object::MachOObjectFile>(**BinFile)) {
    MachOFile.reset(dyn_cast<object::MachOObjectFile>(BinFile->release()));

    // TODO: Check that dylib arch is compatible.
  } else if (auto *MachOUni =
                 dyn_cast<object::MachOUniversalBinary>(BinFile->get())) {
    SmallVector<std::pair<uint32_t, uint32_t>> ArchsToTry;
    if (GetFallbackArchs)
      ArchsToTry = GetFallbackArchs(*InitCPUType, *InitCPUSubType);
    else
      ArchsToTry.push_back({*InitCPUType, *InitCPUSubType});

    for (auto &[CPUType, CPUSubType] : ArchsToTry) {
      for (auto &O : MachOUni->objects()) {
        if (O.getCPUType() == CPUType &&
            (O.getCPUSubType() & ~MachO::CPU_SUBTYPE_MASK) == CPUSubType) {
          if (auto Obj = O.getAsObjectFile())
            MachOFile = std::move(*Obj);
          else
            return Obj.takeError();
          break;
        }
      }
      if (MachOFile) // If found, break out.
        break;
    }
    if (!MachOFile)
      return make_error<StringError>(
          "MachO universal binary at " + Path +
              " does not contain a compatible slice for " +
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

Expected<SymbolNameSet>
getDylibInterfaceFromTapiFile(ExecutionSession &ES, Twine Path,
                              GetFallbackArchsFn GetFallbackArchs) {
  SymbolNameSet Symbols;

  auto TapiFileBuffer = MemoryBuffer::getFile(Path);
  if (!TapiFileBuffer)
    return createFileError(Path, TapiFileBuffer.getError());

  auto Tapi =
      object::TapiUniversal::create((*TapiFileBuffer)->getMemBufferRef());
  if (!Tapi)
    return Tapi.takeError();

  auto InitCPUType = MachO::getCPUType(ES.getTargetTriple());
  if (!InitCPUType)
    return InitCPUType.takeError();

  auto InitCPUSubType = MachO::getCPUSubType(ES.getTargetTriple());
  if (!InitCPUSubType)
    return InitCPUSubType.takeError();

  SmallVector<std::pair<uint32_t, uint32_t>> ArchsToTry;
  if (GetFallbackArchs)
    ArchsToTry = GetFallbackArchs(*InitCPUType, *InitCPUSubType);
  else
    ArchsToTry.push_back({*InitCPUType, *InitCPUSubType});

  auto &IF = (*Tapi)->getInterfaceFile();

  auto ArchSet = IF.getArchitectures();
  for (auto [CPUType, CPUSubType] : ArchsToTry) {
    auto A = MachO::getArchitectureFromCpuType(CPUType, CPUSubType);
    if (ArchSet.has(A)) {
      if (auto Interface = IF.extract(A)) {
        for (auto *Sym : (*Interface)->exports())
          Symbols.insert(ES.intern(Sym->getName()));
        return Symbols;
      } else
        return Interface.takeError();
    }
  }

  return make_error<StringError>(
      "MachO interface file at " + Path +
          " does not contain a compatible slice for " +
          ES.getTargetTriple().str(),
      inconvertibleErrorCode());
}

Expected<SymbolNameSet> getDylibInterface(ExecutionSession &ES, Twine Path,
                                          GetFallbackArchsFn GetFallbackArchs) {
  file_magic Magic;
  if (auto EC = identify_magic(Path, Magic))
    return createFileError(Path, EC);

  switch (Magic) {
  case file_magic::macho_universal_binary:
  case file_magic::macho_dynamically_linked_shared_lib:
    return getDylibInterfaceFromDylib(ES, Path, std::move(GetFallbackArchs));
  case file_magic::tapi_file:
    return getDylibInterfaceFromTapiFile(ES, Path, std::move(GetFallbackArchs));
  default:
    return make_error<StringError>("Cannot get interface for " + Path +
                                       " unrecognized file type",
                                   inconvertibleErrorCode());
  }
}

} // End namespace orc.
} // End namespace llvm.
