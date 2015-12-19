//===-- LLVMSymbolize.cpp -------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation for LLVM symbolization library.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/Symbolize/Symbolize.h"

#include "SymbolizableObjectFile.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Config/config.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/PDB/PDB.h"
#include "llvm/DebugInfo/PDB/PDBContext.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <stdlib.h>

#if defined(_MSC_VER)
#include <Windows.h>
#include <DbgHelp.h>
#pragma comment(lib, "dbghelp.lib")

// Windows.h conflicts with our COFF header definitions.
#ifdef IMAGE_FILE_MACHINE_I386
#undef IMAGE_FILE_MACHINE_I386
#endif
#endif

namespace llvm {
namespace symbolize {

ErrorOr<DILineInfo> LLVMSymbolizer::symbolizeCode(const std::string &ModuleName,
                                                  uint64_t ModuleOffset) {
  auto InfoOrErr = getOrCreateModuleInfo(ModuleName);
  if (auto EC = InfoOrErr.getError())
    return EC;
  SymbolizableModule *Info = InfoOrErr.get();

  // If the user is giving us relative addresses, add the preferred base of the
  // object to the offset before we do the query. It's what DIContext expects.
  if (Opts.RelativeAddresses)
    ModuleOffset += Info->getModulePreferredBase();

  DILineInfo LineInfo = Info->symbolizeCode(ModuleOffset, Opts.PrintFunctions,
                                            Opts.UseSymbolTable);
  if (Opts.Demangle)
    LineInfo.FunctionName = DemangleName(LineInfo.FunctionName, Info);
  return LineInfo;
}

ErrorOr<DIInliningInfo>
LLVMSymbolizer::symbolizeInlinedCode(const std::string &ModuleName,
                                     uint64_t ModuleOffset) {
  auto InfoOrErr = getOrCreateModuleInfo(ModuleName);
  if (auto EC = InfoOrErr.getError())
    return EC;
  SymbolizableModule *Info = InfoOrErr.get();

  // If the user is giving us relative addresses, add the preferred base of the
  // object to the offset before we do the query. It's what DIContext expects.
  if (Opts.RelativeAddresses)
    ModuleOffset += Info->getModulePreferredBase();

  DIInliningInfo InlinedContext = Info->symbolizeInlinedCode(
      ModuleOffset, Opts.PrintFunctions, Opts.UseSymbolTable);
  if (Opts.Demangle) {
    for (int i = 0, n = InlinedContext.getNumberOfFrames(); i < n; i++) {
      auto *Frame = InlinedContext.getMutableFrame(i);
      Frame->FunctionName = DemangleName(Frame->FunctionName, Info);
    }
  }
  return InlinedContext;
}

ErrorOr<DIGlobal> LLVMSymbolizer::symbolizeData(const std::string &ModuleName,
                                                uint64_t ModuleOffset) {
  auto InfoOrErr = getOrCreateModuleInfo(ModuleName);
  if (auto EC = InfoOrErr.getError())
    return EC;
  SymbolizableModule *Info = InfoOrErr.get();

  // If the user is giving us relative addresses, add the preferred base of
  // the object to the offset before we do the query. It's what DIContext
  // expects.
  if (Opts.RelativeAddresses)
    ModuleOffset += Info->getModulePreferredBase();

  DIGlobal Global = Info->symbolizeData(ModuleOffset);
  if (Opts.Demangle)
    Global.Name = DemangleName(Global.Name, Info);
  return Global;
}

void LLVMSymbolizer::flush() {
  ObjectForUBPathAndArch.clear();
  BinaryForPath.clear();
  ObjectPairForPathArch.clear();
  Modules.clear();
}

// For Path="/path/to/foo" and Basename="foo" assume that debug info is in
// /path/to/foo.dSYM/Contents/Resources/DWARF/foo.
// For Path="/path/to/bar.dSYM" and Basename="foo" assume that debug info is in
// /path/to/bar.dSYM/Contents/Resources/DWARF/foo.
static
std::string getDarwinDWARFResourceForPath(
    const std::string &Path, const std::string &Basename) {
  SmallString<16> ResourceName = StringRef(Path);
  if (sys::path::extension(Path) != ".dSYM") {
    ResourceName += ".dSYM";
  }
  sys::path::append(ResourceName, "Contents", "Resources", "DWARF");
  sys::path::append(ResourceName, Basename);
  return ResourceName.str();
}

static bool checkFileCRC(StringRef Path, uint32_t CRCHash) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> MB =
      MemoryBuffer::getFileOrSTDIN(Path);
  if (!MB)
    return false;
  return !zlib::isAvailable() || CRCHash == zlib::crc32(MB.get()->getBuffer());
}

static bool findDebugBinary(const std::string &OrigPath,
                            const std::string &DebuglinkName, uint32_t CRCHash,
                            std::string &Result) {
  std::string OrigRealPath = OrigPath;
#if defined(HAVE_REALPATH)
  if (char *RP = realpath(OrigPath.c_str(), nullptr)) {
    OrigRealPath = RP;
    free(RP);
  }
#endif
  SmallString<16> OrigDir(OrigRealPath);
  llvm::sys::path::remove_filename(OrigDir);
  SmallString<16> DebugPath = OrigDir;
  // Try /path/to/original_binary/debuglink_name
  llvm::sys::path::append(DebugPath, DebuglinkName);
  if (checkFileCRC(DebugPath, CRCHash)) {
    Result = DebugPath.str();
    return true;
  }
  // Try /path/to/original_binary/.debug/debuglink_name
  DebugPath = OrigRealPath;
  llvm::sys::path::append(DebugPath, ".debug", DebuglinkName);
  if (checkFileCRC(DebugPath, CRCHash)) {
    Result = DebugPath.str();
    return true;
  }
  // Try /usr/lib/debug/path/to/original_binary/debuglink_name
  DebugPath = "/usr/lib/debug";
  llvm::sys::path::append(DebugPath, llvm::sys::path::relative_path(OrigDir),
                          DebuglinkName);
  if (checkFileCRC(DebugPath, CRCHash)) {
    Result = DebugPath.str();
    return true;
  }
  return false;
}

static bool getGNUDebuglinkContents(const ObjectFile *Obj, std::string &DebugName,
                                    uint32_t &CRCHash) {
  if (!Obj)
    return false;
  for (const SectionRef &Section : Obj->sections()) {
    StringRef Name;
    Section.getName(Name);
    Name = Name.substr(Name.find_first_not_of("._"));
    if (Name == "gnu_debuglink") {
      StringRef Data;
      Section.getContents(Data);
      DataExtractor DE(Data, Obj->isLittleEndian(), 0);
      uint32_t Offset = 0;
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

static
bool darwinDsymMatchesBinary(const MachOObjectFile *DbgObj,
                             const MachOObjectFile *Obj) {
  ArrayRef<uint8_t> dbg_uuid = DbgObj->getUuid();
  ArrayRef<uint8_t> bin_uuid = Obj->getUuid();
  if (dbg_uuid.empty() || bin_uuid.empty())
    return false;
  return !memcmp(dbg_uuid.data(), bin_uuid.data(), dbg_uuid.size());
}

ObjectFile *LLVMSymbolizer::lookUpDsymFile(const std::string &ExePath,
    const MachOObjectFile *MachExeObj, const std::string &ArchName) {
  // On Darwin we may find DWARF in separate object file in
  // resource directory.
  std::vector<std::string> DsymPaths;
  StringRef Filename = sys::path::filename(ExePath);
  DsymPaths.push_back(getDarwinDWARFResourceForPath(ExePath, Filename));
  for (const auto &Path : Opts.DsymHints) {
    DsymPaths.push_back(getDarwinDWARFResourceForPath(Path, Filename));
  }
  for (const auto &Path : DsymPaths) {
    auto DbgObjOrErr = getOrCreateObject(Path, ArchName);
    if (!DbgObjOrErr)
      continue;
    ObjectFile *DbgObj = DbgObjOrErr.get();
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
  if (!DbgObjOrErr)
    return nullptr;
  return DbgObjOrErr.get();
}

ErrorOr<LLVMSymbolizer::ObjectPair>
LLVMSymbolizer::getOrCreateObjectPair(const std::string &Path,
                                      const std::string &ArchName) {
  const auto &I = ObjectPairForPathArch.find(std::make_pair(Path, ArchName));
  if (I != ObjectPairForPathArch.end())
    return I->second;

  auto ObjOrErr = getOrCreateObject(Path, ArchName);
  if (auto EC = ObjOrErr.getError()) {
    ObjectPairForPathArch.insert(
        std::make_pair(std::make_pair(Path, ArchName), EC));
    return EC;
  }

  ObjectFile *Obj = ObjOrErr.get();
  assert(Obj != nullptr);
  ObjectFile *DbgObj = nullptr;

  if (auto MachObj = dyn_cast<const MachOObjectFile>(Obj))
    DbgObj = lookUpDsymFile(Path, MachObj, ArchName);
  if (!DbgObj)
    DbgObj = lookUpDebuglinkObject(Path, Obj, ArchName);
  if (!DbgObj)
    DbgObj = Obj;
  ObjectPair Res = std::make_pair(Obj, DbgObj);
  ObjectPairForPathArch.insert(
      std::make_pair(std::make_pair(Path, ArchName), Res));
  return Res;
}

ErrorOr<ObjectFile *>
LLVMSymbolizer::getOrCreateObject(const std::string &Path,
                                  const std::string &ArchName) {
  const auto &I = BinaryForPath.find(Path);
  Binary *Bin = nullptr;
  if (I == BinaryForPath.end()) {
    ErrorOr<OwningBinary<Binary>> BinOrErr = createBinary(Path);
    if (auto EC = BinOrErr.getError()) {
      BinaryForPath.insert(std::make_pair(Path, EC));
      return EC;
    }
    Bin = BinOrErr->getBinary();
    BinaryForPath.insert(std::make_pair(Path, std::move(BinOrErr.get())));
  } else if (auto EC = I->second.getError()) {
    return EC;
  } else {
    Bin = I->second->getBinary();
  }

  assert(Bin != nullptr);

  if (MachOUniversalBinary *UB = dyn_cast<MachOUniversalBinary>(Bin)) {
    const auto &I = ObjectForUBPathAndArch.find(std::make_pair(Path, ArchName));
    if (I != ObjectForUBPathAndArch.end()) {
      if (auto EC = I->second.getError())
        return EC;
      return I->second->get();
    }
    ErrorOr<std::unique_ptr<ObjectFile>> ObjOrErr =
        UB->getObjectForArch(ArchName);
    if (auto EC = ObjOrErr.getError()) {
      ObjectForUBPathAndArch.insert(
          std::make_pair(std::make_pair(Path, ArchName), EC));
      return EC;
    }
    ObjectFile *Res = ObjOrErr->get();
    ObjectForUBPathAndArch.insert(std::make_pair(std::make_pair(Path, ArchName),
                                                 std::move(ObjOrErr.get())));
    return Res;
  }
  if (Bin->isObject()) {
    return cast<ObjectFile>(Bin);
  }
  return object_error::arch_not_found;
}

ErrorOr<SymbolizableModule *>
LLVMSymbolizer::getOrCreateModuleInfo(const std::string &ModuleName) {
  const auto &I = Modules.find(ModuleName);
  if (I != Modules.end()) {
    auto &InfoOrErr = I->second;
    if (auto EC = InfoOrErr.getError())
      return EC;
    return InfoOrErr->get();
  }
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
  if (auto EC = ObjectsOrErr.getError()) {
    // Failed to find valid object file.
    Modules.insert(std::make_pair(ModuleName, EC));
    return EC;
  }
  ObjectPair Objects = ObjectsOrErr.get();

  std::unique_ptr<DIContext> Context;
  if (auto CoffObject = dyn_cast<COFFObjectFile>(Objects.first)) {
    // If this is a COFF object, assume it contains PDB debug information.  If
    // we don't find any we will fall back to the DWARF case.
    std::unique_ptr<IPDBSession> Session;
    PDB_ErrorCode Error = loadDataForEXE(PDB_ReaderType::DIA,
                                         Objects.first->getFileName(), Session);
    if (Error == PDB_ErrorCode::Success) {
      Context.reset(new PDBContext(*CoffObject, std::move(Session)));
    }
  }
  if (!Context)
    Context.reset(new DWARFContextInMemory(*Objects.second));
  assert(Context);
  auto InfoOrErr =
      SymbolizableObjectFile::create(Objects.first, std::move(Context));
  auto InsertResult =
      Modules.insert(std::make_pair(ModuleName, std::move(InfoOrErr)));
  assert(InsertResult.second);
  if (auto EC = InsertResult.first->second.getError())
    return EC;
  return InsertResult.first->second->get();
}

// Undo these various manglings for Win32 extern "C" functions:
// cdecl       - _foo
// stdcall     - _foo@12
// fastcall    - @foo@12
// vectorcall  - foo@@12
// These are all different linkage names for 'foo'.
static StringRef demanglePE32ExternCFunc(StringRef SymbolName) {
  // Remove any '_' or '@' prefix.
  char Front = SymbolName.empty() ? '\0' : SymbolName[0];
  if (Front == '_' || Front == '@')
    SymbolName = SymbolName.drop_front();

  // Remove any '@[0-9]+' suffix.
  if (Front != '?') {
    size_t AtPos = SymbolName.rfind('@');
    if (AtPos != StringRef::npos &&
        std::all_of(SymbolName.begin() + AtPos + 1, SymbolName.end(),
                    [](char C) { return C >= '0' && C <= '9'; })) {
      SymbolName = SymbolName.substr(0, AtPos);
    }
  }

  // Remove any ending '@' for vectorcall.
  if (SymbolName.endswith("@"))
    SymbolName = SymbolName.drop_back();

  return SymbolName;
}

#if !defined(_MSC_VER)
// Assume that __cxa_demangle is provided by libcxxabi (except for Windows).
extern "C" char *__cxa_demangle(const char *mangled_name, char *output_buffer,
                                size_t *length, int *status);
#endif

std::string LLVMSymbolizer::DemangleName(const std::string &Name,
                                         const SymbolizableModule *ModInfo) {
#if !defined(_MSC_VER)
  // We can spoil names of symbols with C linkage, so use an heuristic
  // approach to check if the name should be demangled.
  if (Name.substr(0, 2) == "_Z") {
    int status = 0;
    char *DemangledName = __cxa_demangle(Name.c_str(), nullptr, nullptr, &status);
    if (status != 0)
      return Name;
    std::string Result = DemangledName;
    free(DemangledName);
    return Result;
  }
#else
  if (!Name.empty() && Name.front() == '?') {
    // Only do MSVC C++ demangling on symbols starting with '?'.
    char DemangledName[1024] = {0};
    DWORD result = ::UnDecorateSymbolName(
        Name.c_str(), DemangledName, 1023,
        UNDNAME_NO_ACCESS_SPECIFIERS |       // Strip public, private, protected
            UNDNAME_NO_ALLOCATION_LANGUAGE | // Strip __thiscall, __stdcall, etc
            UNDNAME_NO_THROW_SIGNATURES |    // Strip throw() specifications
            UNDNAME_NO_MEMBER_TYPE | // Strip virtual, static, etc specifiers
            UNDNAME_NO_MS_KEYWORDS | // Strip all MS extension keywords
            UNDNAME_NO_FUNCTION_RETURNS); // Strip function return types
    return (result == 0) ? Name : std::string(DemangledName);
  }
#endif
  if (ModInfo && ModInfo->isWin32Module())
    return std::string(demanglePE32ExternCFunc(Name));
  return Name;
}

} // namespace symbolize
} // namespace llvm
