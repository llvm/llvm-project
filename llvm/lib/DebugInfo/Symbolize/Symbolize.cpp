//===-- LLVMSymbolize.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation for LLVM symbolization library.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/Symbolize/Symbolize.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/Config/config.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/PDB/PDB.h"
#include "llvm/DebugInfo/PDB/PDBContext.h"
#include "llvm/DebugInfo/Symbolize/DIFetcher.h"
#include "llvm/DebugInfo/Symbolize/SymbolizableObjectFile.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Support/CRC.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <algorithm>
#include <cassert>
#include <cstring>

namespace llvm {
namespace symbolize {

template <typename T>
Expected<DILineInfo>
LLVMSymbolizer::symbolizeCodeCommon(const T &ModuleSpecifier,
                                    object::SectionedAddress ModuleOffset) {

  auto InfoOrErr = getOrCreateModuleInfo(ModuleSpecifier);
  if (!InfoOrErr)
    return InfoOrErr.takeError();

  SymbolizableModule *Info = *InfoOrErr;

  // A null module means an error has already been reported. Return an empty
  // result.
  if (!Info)
    return DILineInfo();

  // If the user is giving us relative addresses, add the preferred base of the
  // object to the offset before we do the query. It's what DIContext expects.
  if (Opts.RelativeAddresses)
    ModuleOffset.Address += Info->getModulePreferredBase();

  DILineInfo LineInfo = Info->symbolizeCode(
      ModuleOffset, DILineInfoSpecifier(Opts.PathStyle, Opts.PrintFunctions),
      Opts.UseSymbolTable);
  if (Opts.Demangle)
    LineInfo.FunctionName = DemangleName(LineInfo.FunctionName, Info);
  return LineInfo;
}

Expected<DILineInfo>
LLVMSymbolizer::symbolizeCode(const ObjectFile &Obj,
                              object::SectionedAddress ModuleOffset) {
  return symbolizeCodeCommon(Obj, ModuleOffset);
}

Expected<DILineInfo>
LLVMSymbolizer::symbolizeCode(const std::string &ModuleName,
                              object::SectionedAddress ModuleOffset) {
  return symbolizeCodeCommon(ModuleName, ModuleOffset);
}

template <typename T>
Expected<DIInliningInfo> LLVMSymbolizer::symbolizeInlinedCodeCommon(
    const T &ModuleSpecifier, object::SectionedAddress ModuleOffset) {
  auto InfoOrErr = getOrCreateModuleInfo(ModuleSpecifier);
  if (!InfoOrErr)
    return InfoOrErr.takeError();

  SymbolizableModule *Info = *InfoOrErr;

  // A null module means an error has already been reported. Return an empty
  // result.
  if (!Info)
    return DIInliningInfo();

  // If the user is giving us relative addresses, add the preferred base of the
  // object to the offset before we do the query. It's what DIContext expects.
  if (Opts.RelativeAddresses)
    ModuleOffset.Address += Info->getModulePreferredBase();

  DIInliningInfo InlinedContext = Info->symbolizeInlinedCode(
      ModuleOffset, DILineInfoSpecifier(Opts.PathStyle, Opts.PrintFunctions),
      Opts.UseSymbolTable);
  if (Opts.Demangle) {
    for (int i = 0, n = InlinedContext.getNumberOfFrames(); i < n; i++) {
      auto *Frame = InlinedContext.getMutableFrame(i);
      Frame->FunctionName = DemangleName(Frame->FunctionName, Info);
    }
  }
  return InlinedContext;
}

Expected<DIInliningInfo>
LLVMSymbolizer::symbolizeInlinedCode(const ObjectFile &Obj,
                                     object::SectionedAddress ModuleOffset) {
  return symbolizeInlinedCodeCommon(Obj, ModuleOffset);
}

Expected<DIInliningInfo>
LLVMSymbolizer::symbolizeInlinedCode(const std::string &ModuleName,
                                     object::SectionedAddress ModuleOffset) {
  return symbolizeInlinedCodeCommon(ModuleName, ModuleOffset);
}

template <typename T>
Expected<DIGlobal>
LLVMSymbolizer::symbolizeDataCommon(const T &ModuleSpecifier,
                                    object::SectionedAddress ModuleOffset) {

  auto InfoOrErr = getOrCreateModuleInfo(ModuleSpecifier);
  if (!InfoOrErr)
    return InfoOrErr.takeError();

  SymbolizableModule *Info = *InfoOrErr;
  // A null module means an error has already been reported. Return an empty
  // result.
  if (!Info)
    return DIGlobal();

  // If the user is giving us relative addresses, add the preferred base of
  // the object to the offset before we do the query. It's what DIContext
  // expects.
  if (Opts.RelativeAddresses)
    ModuleOffset.Address += Info->getModulePreferredBase();

  DIGlobal Global = Info->symbolizeData(ModuleOffset);
  if (Opts.Demangle)
    Global.Name = DemangleName(Global.Name, Info);
  return Global;
}

Expected<DIGlobal>
LLVMSymbolizer::symbolizeData(const ObjectFile &Obj,
                              object::SectionedAddress ModuleOffset) {
  return symbolizeDataCommon(Obj, ModuleOffset);
}

Expected<DIGlobal>
LLVMSymbolizer::symbolizeData(const std::string &ModuleName,
                              object::SectionedAddress ModuleOffset) {
  return symbolizeDataCommon(ModuleName, ModuleOffset);
}

template <typename T>
Expected<std::vector<DILocal>>
LLVMSymbolizer::symbolizeFrameCommon(const T &ModuleSpecifier,
                                     object::SectionedAddress ModuleOffset) {
  auto InfoOrErr = getOrCreateModuleInfo(ModuleSpecifier);
  if (!InfoOrErr)
    return InfoOrErr.takeError();

  SymbolizableModule *Info = *InfoOrErr;
  // A null module means an error has already been reported. Return an empty
  // result.
  if (!Info)
    return std::vector<DILocal>();

  // If the user is giving us relative addresses, add the preferred base of
  // the object to the offset before we do the query. It's what DIContext
  // expects.
  if (Opts.RelativeAddresses)
    ModuleOffset.Address += Info->getModulePreferredBase();

  return Info->symbolizeFrame(ModuleOffset);
}

Expected<std::vector<DILocal>>
LLVMSymbolizer::symbolizeFrame(const ObjectFile &Obj,
                               object::SectionedAddress ModuleOffset) {
  return symbolizeFrameCommon(Obj, ModuleOffset);
}

Expected<std::vector<DILocal>>
LLVMSymbolizer::symbolizeFrame(const std::string &ModuleName,
                               object::SectionedAddress ModuleOffset) {
  return symbolizeFrameCommon(ModuleName, ModuleOffset);
}

void LLVMSymbolizer::flush() {
  ObjectForUBPathAndArch.clear();
  BinaryForPath.clear();
  ObjectPairForPathArch.clear();
  Modules.clear();
}

namespace {

// For Path="/path/to/foo" and Basename="foo" assume that debug info is in
// /path/to/foo.dSYM/Contents/Resources/DWARF/foo.
// For Path="/path/to/bar.dSYM" and Basename="foo" assume that debug info is in
// /path/to/bar.dSYM/Contents/Resources/DWARF/foo.
std::string getDarwinDWARFResourceForPath(const std::string &Path,
                                          const std::string &Basename) {
  SmallString<16> ResourceName = StringRef(Path);
  if (sys::path::extension(Path) != ".dSYM") {
    ResourceName += ".dSYM";
  }
  sys::path::append(ResourceName, "Contents", "Resources", "DWARF");
  sys::path::append(ResourceName, Basename);
  return std::string(ResourceName.str());
}

bool checkFileCRC(StringRef Path, uint32_t CRCHash) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> MB =
      MemoryBuffer::getFileOrSTDIN(Path);
  if (!MB)
    return false;
  return CRCHash == llvm::crc32(arrayRefFromStringRef(MB.get()->getBuffer()));
}

bool getGNUDebuglinkContents(const ObjectFile *Obj, std::string &DebugName,
                             uint32_t &CRCHash) {
  if (!Obj)
    return false;
  for (const SectionRef &Section : Obj->sections()) {
    StringRef Name;
    consumeError(Section.getName().moveInto(Name));

    Name = Name.substr(Name.find_first_not_of("._"));
    if (Name == "gnu_debuglink") {
      Expected<StringRef> ContentsOrErr = Section.getContents();
      if (!ContentsOrErr) {
        consumeError(ContentsOrErr.takeError());
        return false;
      }
      DataExtractor DE(*ContentsOrErr, Obj->isLittleEndian(), 0);
      uint64_t Offset = 0;
      if (const char *DebugNameStr = DE.getCStr(&Offset)) {
        // 4-byte align the offset.
        Offset = (Offset + 3) & ~0x3;
        if (DE.isValidOffsetForDataOfSize(Offset, 4)) {
          DebugName = DebugNameStr;
          CRCHash = DE.getU32(&Offset);
          return true;
        }
      }
      break;
    }
  }
  return false;
}

bool darwinDsymMatchesBinary(const MachOObjectFile *DbgObj,
                             const MachOObjectFile *Obj) {
  ArrayRef<uint8_t> dbg_uuid = DbgObj->getUuid();
  ArrayRef<uint8_t> bin_uuid = Obj->getUuid();
  if (dbg_uuid.empty() || bin_uuid.empty())
    return false;
  return !memcmp(dbg_uuid.data(), bin_uuid.data(), dbg_uuid.size());
}

template <typename ELFT>
Optional<ArrayRef<uint8_t>> getBuildID(const ELFFile<ELFT> &Obj) {
  auto PhdrsOrErr = Obj.program_headers();
  if (!PhdrsOrErr) {
    consumeError(PhdrsOrErr.takeError());
    return {};
  }
  for (const auto &P : *PhdrsOrErr) {
    if (P.p_type != ELF::PT_NOTE)
      continue;
    Error Err = Error::success();
    for (auto N : Obj.notes(P, Err))
      if (N.getType() == ELF::NT_GNU_BUILD_ID &&
          N.getName() == ELF::ELF_NOTE_GNU)
        return N.getDesc();
    consumeError(std::move(Err));
  }
  return {};
}

Optional<ArrayRef<uint8_t>> getBuildID(const ELFObjectFileBase *Obj) {
  Optional<ArrayRef<uint8_t>> BuildID;
  if (auto *O = dyn_cast<ELFObjectFile<ELF32LE>>(Obj))
    BuildID = getBuildID(O->getELFFile());
  else if (auto *O = dyn_cast<ELFObjectFile<ELF32BE>>(Obj))
    BuildID = getBuildID(O->getELFFile());
  else if (auto *O = dyn_cast<ELFObjectFile<ELF64LE>>(Obj))
    BuildID = getBuildID(O->getELFFile());
  else if (auto *O = dyn_cast<ELFObjectFile<ELF64BE>>(Obj))
    BuildID = getBuildID(O->getELFFile());
  else
    llvm_unreachable("unsupported file format");
  return BuildID;
}

} // end anonymous namespace

ObjectFile *LLVMSymbolizer::lookUpDsymFile(const std::string &ExePath,
                                           const MachOObjectFile *MachExeObj,
                                           const std::string &ArchName) {
  // On Darwin we may find DWARF in separate object file in
  // resource directory.
  std::vector<std::string> DsymPaths;
  StringRef Filename = sys::path::filename(ExePath);
  DsymPaths.push_back(
      getDarwinDWARFResourceForPath(ExePath, std::string(Filename)));
  for (const auto &Path : Opts.DsymHints) {
    DsymPaths.push_back(
        getDarwinDWARFResourceForPath(Path, std::string(Filename)));
  }
  for (const auto &Path : DsymPaths) {
    auto DbgObjOrErr = getOrCreateObject(Path, ArchName);
    if (!DbgObjOrErr) {
      // Ignore errors, the file might not exist.
      consumeError(DbgObjOrErr.takeError());
      continue;
    }
    ObjectFile *DbgObj = DbgObjOrErr.get();
    if (!DbgObj)
      continue;
    const MachOObjectFile *MachDbgObj = dyn_cast<const MachOObjectFile>(DbgObj);
    if (!MachDbgObj)
      continue;
    if (darwinDsymMatchesBinary(MachDbgObj, MachExeObj))
      return DbgObj;
  }
  return nullptr;
}

ObjectFile *LLVMSymbolizer::lookUpDebuglinkObject(const std::string &Path,
                                                  const ObjectFile *Obj,
                                                  const std::string &ArchName) {
  std::string DebuglinkName;
  uint32_t CRCHash;
  std::string DebugBinaryPath;
  if (!getGNUDebuglinkContents(Obj, DebuglinkName, CRCHash))
    return nullptr;
  if (!findDebugBinary(Path, DebuglinkName, CRCHash, DebugBinaryPath))
    return nullptr;
  auto DbgObjOrErr = getOrCreateObject(DebugBinaryPath, ArchName);
  if (!DbgObjOrErr) {
    // Ignore errors, the file might not exist.
    consumeError(DbgObjOrErr.takeError());
    return nullptr;
  }
  return DbgObjOrErr.get();
}

ObjectFile *LLVMSymbolizer::lookUpBuildIDObject(const std::string &Path,
                                                const ELFObjectFileBase *Obj,
                                                const std::string &ArchName) {
  auto BuildID = getBuildID(Obj);
  if (!BuildID)
    return nullptr;
  if (BuildID->size() < 2)
    return nullptr;
  std::string DebugBinaryPath;
  if (!findDebugBinary(*BuildID, DebugBinaryPath))
    return nullptr;
  auto DbgObjOrErr = getOrCreateObject(DebugBinaryPath, ArchName);
  if (!DbgObjOrErr) {
    consumeError(DbgObjOrErr.takeError());
    return nullptr;
  }
  return DbgObjOrErr.get();
}

bool LLVMSymbolizer::findDebugBinary(const std::string &OrigPath,
                                     const std::string &DebuglinkName,
                                     uint32_t CRCHash, std::string &Result) {
  SmallString<16> OrigDir(OrigPath);
  llvm::sys::path::remove_filename(OrigDir);
  SmallString<16> DebugPath = OrigDir;
  // Try relative/path/to/original_binary/debuglink_name
  llvm::sys::path::append(DebugPath, DebuglinkName);
  if (checkFileCRC(DebugPath, CRCHash)) {
    Result = std::string(DebugPath.str());
    return true;
  }
  // Try relative/path/to/original_binary/.debug/debuglink_name
  DebugPath = OrigDir;
  llvm::sys::path::append(DebugPath, ".debug", DebuglinkName);
  if (checkFileCRC(DebugPath, CRCHash)) {
    Result = std::string(DebugPath.str());
    return true;
  }
  // Make the path absolute so that lookups will go to
  // "/usr/lib/debug/full/path/to/debug", not
  // "/usr/lib/debug/to/debug"
  llvm::sys::fs::make_absolute(OrigDir);
  if (!Opts.FallbackDebugPath.empty()) {
    // Try <FallbackDebugPath>/absolute/path/to/original_binary/debuglink_name
    DebugPath = Opts.FallbackDebugPath;
  } else {
#if defined(__NetBSD__)
    // Try /usr/libdata/debug/absolute/path/to/original_binary/debuglink_name
    DebugPath = "/usr/libdata/debug";
#else
    // Try /usr/lib/debug/absolute/path/to/original_binary/debuglink_name
    DebugPath = "/usr/lib/debug";
#endif
  }
  llvm::sys::path::append(DebugPath, llvm::sys::path::relative_path(OrigDir),
                          DebuglinkName);
  if (checkFileCRC(DebugPath, CRCHash)) {
    Result = std::string(DebugPath.str());
    return true;
  }
  return false;
}

bool LLVMSymbolizer::findDebugBinary(const ArrayRef<uint8_t> BuildID,
                                     std::string &Result) {
  Optional<std::string> Path;
  Path = LocalDIFetcher(Opts.DebugFileDirectory).fetchBuildID(BuildID);
  if (Path) {
    Result = std::move(*Path);
    return true;
  }

  // Try caller-provided debug info fetchers.
  for (const std::unique_ptr<DIFetcher> &Fetcher : DIFetchers) {
    Path = Fetcher->fetchBuildID(BuildID);
    if (Path) {
      Result = std::move(*Path);
      return true;
    }
  }

  return false;
}

Expected<LLVMSymbolizer::ObjectPair>
LLVMSymbolizer::getOrCreateObjectPair(const std::string &Path,
                                      const std::string &ArchName) {
  auto I = ObjectPairForPathArch.find(std::make_pair(Path, ArchName));
  if (I != ObjectPairForPathArch.end())
    return I->second;

  auto ObjOrErr = getOrCreateObject(Path, ArchName);
  if (!ObjOrErr) {
    ObjectPairForPathArch.emplace(std::make_pair(Path, ArchName),
                                  ObjectPair(nullptr, nullptr));
    return ObjOrErr.takeError();
  }

  ObjectFile *Obj = ObjOrErr.get();
  assert(Obj != nullptr);
  ObjectFile *DbgObj = nullptr;

  if (auto MachObj = dyn_cast<const MachOObjectFile>(Obj))
    DbgObj = lookUpDsymFile(Path, MachObj, ArchName);
  else if (auto ELFObj = dyn_cast<const ELFObjectFileBase>(Obj))
    DbgObj = lookUpBuildIDObject(Path, ELFObj, ArchName);
  if (!DbgObj)
    DbgObj = lookUpDebuglinkObject(Path, Obj, ArchName);
  if (!DbgObj)
    DbgObj = Obj;
  ObjectPair Res = std::make_pair(Obj, DbgObj);
  ObjectPairForPathArch.emplace(std::make_pair(Path, ArchName), Res);
  return Res;
}

Expected<ObjectFile *>
LLVMSymbolizer::getOrCreateObject(const std::string &Path,
                                  const std::string &ArchName) {
  Binary *Bin;
  auto Pair = BinaryForPath.emplace(Path, OwningBinary<Binary>());
  if (!Pair.second) {
    Bin = Pair.first->second.getBinary();
  } else {
    Expected<OwningBinary<Binary>> BinOrErr = createBinary(Path);
    if (!BinOrErr)
      return BinOrErr.takeError();
    Pair.first->second = std::move(BinOrErr.get());
    Bin = Pair.first->second.getBinary();
  }

  if (!Bin)
    return static_cast<ObjectFile *>(nullptr);

  if (MachOUniversalBinary *UB = dyn_cast_or_null<MachOUniversalBinary>(Bin)) {
    auto I = ObjectForUBPathAndArch.find(std::make_pair(Path, ArchName));
    if (I != ObjectForUBPathAndArch.end())
      return I->second.get();

    Expected<std::unique_ptr<ObjectFile>> ObjOrErr =
        UB->getMachOObjectForArch(ArchName);
    if (!ObjOrErr) {
      ObjectForUBPathAndArch.emplace(std::make_pair(Path, ArchName),
                                     std::unique_ptr<ObjectFile>());
      return ObjOrErr.takeError();
    }
    ObjectFile *Res = ObjOrErr->get();
    ObjectForUBPathAndArch.emplace(std::make_pair(Path, ArchName),
                                   std::move(ObjOrErr.get()));
    return Res;
  }
  if (Bin->isObject()) {
    return cast<ObjectFile>(Bin);
  }
  return errorCodeToError(object_error::arch_not_found);
}

Expected<SymbolizableModule *>
LLVMSymbolizer::createModuleInfo(const ObjectFile *Obj,
                                 std::unique_ptr<DIContext> Context,
                                 StringRef ModuleName) {
  auto InfoOrErr = SymbolizableObjectFile::create(Obj, std::move(Context),
                                                  Opts.UntagAddresses);
  std::unique_ptr<SymbolizableModule> SymMod;
  if (InfoOrErr)
    SymMod = std::move(*InfoOrErr);
  auto InsertResult = Modules.insert(
      std::make_pair(std::string(ModuleName), std::move(SymMod)));
  assert(InsertResult.second);
  if (!InfoOrErr)
    return InfoOrErr.takeError();
  return InsertResult.first->second.get();
}

Expected<SymbolizableModule *>
LLVMSymbolizer::getOrCreateModuleInfo(const std::string &ModuleName) {
  auto I = Modules.find(ModuleName);
  if (I != Modules.end())
    return I->second.get();

  std::string BinaryName = ModuleName;
  std::string ArchName = Opts.DefaultArch;
  size_t ColonPos = ModuleName.find_last_of(':');
  // Verify that substring after colon form a valid arch name.
  if (ColonPos != std::string::npos) {
    std::string ArchStr = ModuleName.substr(ColonPos + 1);
    if (Triple(ArchStr).getArch() != Triple::UnknownArch) {
      BinaryName = ModuleName.substr(0, ColonPos);
      ArchName = ArchStr;
    }
  }
  auto ObjectsOrErr = getOrCreateObjectPair(BinaryName, ArchName);
  if (!ObjectsOrErr) {
    // Failed to find valid object file.
    Modules.emplace(ModuleName, std::unique_ptr<SymbolizableModule>());
    return ObjectsOrErr.takeError();
  }
  ObjectPair Objects = ObjectsOrErr.get();

  std::unique_ptr<DIContext> Context;
  // If this is a COFF object containing PDB info, use a PDBContext to
  // symbolize. Otherwise, use DWARF.
  if (auto CoffObject = dyn_cast<COFFObjectFile>(Objects.first)) {
    const codeview::DebugInfo *DebugInfo;
    StringRef PDBFileName;
    auto EC = CoffObject->getDebugPDBInfo(DebugInfo, PDBFileName);
    if (!EC && DebugInfo != nullptr && !PDBFileName.empty()) {
      using namespace pdb;
      std::unique_ptr<IPDBSession> Session;

      PDB_ReaderType ReaderType =
          Opts.UseDIA ? PDB_ReaderType::DIA : PDB_ReaderType::Native;
      if (auto Err = loadDataForEXE(ReaderType, Objects.first->getFileName(),
                                    Session)) {
        Modules.emplace(ModuleName, std::unique_ptr<SymbolizableModule>());
        // Return along the PDB filename to provide more context
        return createFileError(PDBFileName, std::move(Err));
      }
      Context.reset(new PDBContext(*CoffObject, std::move(Session)));
    }
  }
  if (!Context)
    Context = DWARFContext::create(
        *Objects.second, DWARFContext::ProcessDebugRelocations::Process,
        nullptr, Opts.DWPName);
  return createModuleInfo(Objects.first, std::move(Context), ModuleName);
}

Expected<SymbolizableModule *>
LLVMSymbolizer::getOrCreateModuleInfo(const ObjectFile &Obj) {
  StringRef ObjName = Obj.getFileName();
  auto I = Modules.find(ObjName);
  if (I != Modules.end())
    return I->second.get();

  std::unique_ptr<DIContext> Context = DWARFContext::create(Obj);
  // FIXME: handle COFF object with PDB info to use PDBContext
  return createModuleInfo(&Obj, std::move(Context), ObjName);
}

namespace {

// Undo these various manglings for Win32 extern "C" functions:
// cdecl       - _foo
// stdcall     - _foo@12
// fastcall    - @foo@12
// vectorcall  - foo@@12
// These are all different linkage names for 'foo'.
StringRef demanglePE32ExternCFunc(StringRef SymbolName) {
  // Remove any '_' or '@' prefix.
  char Front = SymbolName.empty() ? '\0' : SymbolName[0];
  if (Front == '_' || Front == '@')
    SymbolName = SymbolName.drop_front();

  // Remove any '@[0-9]+' suffix.
  if (Front != '?') {
    size_t AtPos = SymbolName.rfind('@');
    if (AtPos != StringRef::npos &&
        all_of(drop_begin(SymbolName, AtPos + 1), isDigit))
      SymbolName = SymbolName.substr(0, AtPos);
  }

  // Remove any ending '@' for vectorcall.
  if (SymbolName.endswith("@"))
    SymbolName = SymbolName.drop_back();

  return SymbolName;
}

} // end anonymous namespace

std::string
LLVMSymbolizer::DemangleName(const std::string &Name,
                             const SymbolizableModule *DbiModuleDescriptor) {
  std::string Result;
  if (nonMicrosoftDemangle(Name.c_str(), Result))
    return Result;

  if (!Name.empty() && Name.front() == '?') {
    // Only do MSVC C++ demangling on symbols starting with '?'.
    int status = 0;
    char *DemangledName = microsoftDemangle(
        Name.c_str(), nullptr, nullptr, nullptr, &status,
        MSDemangleFlags(MSDF_NoAccessSpecifier | MSDF_NoCallingConvention |
                        MSDF_NoMemberType | MSDF_NoReturnType));
    if (status != 0)
      return Name;
    Result = DemangledName;
    free(DemangledName);
    return Result;
  }

  if (DbiModuleDescriptor && DbiModuleDescriptor->isWin32Module())
    return std::string(demanglePE32ExternCFunc(Name));
  return Name;
}

} // namespace symbolize
} // namespace llvm
