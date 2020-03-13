//===- llvm-objcopy.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm-objcopy.h"
#include "Buffer.h"
#include "COFF/COFFObjcopy.h"
#include "CopyConfig.h"
#include "ELF/ELFObjcopy.h"
#include "MachO/MachOObjcopy.h"
#include "wasm/WasmObjcopy.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Object/Error.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/Wasm.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <memory>
#include <string>
#include <system_error>
#include <utility>

namespace llvm {
namespace objcopy {

// The name this program was invoked as.
StringRef ToolName;

LLVM_ATTRIBUTE_NORETURN void error(Twine Message) {
  WithColor::error(errs(), ToolName) << Message << "\n";
  exit(1);
}

LLVM_ATTRIBUTE_NORETURN void error(Error E) {
  assert(E);
  std::string Buf;
  raw_string_ostream OS(Buf);
  logAllUnhandledErrors(std::move(E), OS);
  OS.flush();
  WithColor::error(errs(), ToolName) << Buf;
  exit(1);
}

LLVM_ATTRIBUTE_NORETURN void reportError(StringRef File, std::error_code EC) {
  assert(EC);
  error(createFileError(File, EC));
}

LLVM_ATTRIBUTE_NORETURN void reportError(StringRef File, Error E) {
  assert(E);
  std::string Buf;
  raw_string_ostream OS(Buf);
  logAllUnhandledErrors(std::move(E), OS);
  OS.flush();
  WithColor::error(errs(), ToolName) << "'" << File << "': " << Buf;
  exit(1);
}

ErrorSuccess reportWarning(Error E) {
  assert(E);
  WithColor::warning(errs(), ToolName) << toString(std::move(E)) << '\n';
  return Error::success();
}

} // end namespace objcopy
} // end namespace llvm

using namespace llvm;
using namespace llvm::object;
using namespace llvm::objcopy;

// For regular archives this function simply calls llvm::writeArchive,
// For thin archives it writes the archive file itself as well as its members.
static Error deepWriteArchive(StringRef ArcName,
                              ArrayRef<NewArchiveMember> NewMembers,
                              bool WriteSymtab, object::Archive::Kind Kind,
                              bool Deterministic, bool Thin) {
  if (Error E = writeArchive(ArcName, NewMembers, WriteSymtab, Kind,
                             Deterministic, Thin))
    return createFileError(ArcName, std::move(E));

  if (!Thin)
    return Error::success();

  for (const NewArchiveMember &Member : NewMembers) {
    // Internally, FileBuffer will use the buffer created by
    // FileOutputBuffer::create, for regular files (that is the case for
    // deepWriteArchive) FileOutputBuffer::create will return OnDiskBuffer.
    // OnDiskBuffer uses a temporary file and then renames it. So in reality
    // there is no inefficiency / duplicated in-memory buffers in this case. For
    // now in-memory buffers can not be completely avoided since
    // NewArchiveMember still requires them even though writeArchive does not
    // write them on disk.
    FileBuffer FB(Member.MemberName);
    if (Error E = FB.allocate(Member.Buf->getBufferSize()))
      return E;
    std::copy(Member.Buf->getBufferStart(), Member.Buf->getBufferEnd(),
              FB.getBufferStart());
    if (Error E = FB.commit())
      return E;
  }
  return Error::success();
}

/// The function executeObjcopyOnIHex does the dispatch based on the format
/// of the output specified by the command line options.
static Error executeObjcopyOnIHex(CopyConfig &Config, MemoryBuffer &In,
                                  Buffer &Out) {
  // TODO: support output formats other than ELF.
  if (Error E = Config.parseELFConfig())
    return E;
  return elf::executeObjcopyOnIHex(Config, In, Out);
}

/// The function executeObjcopyOnRawBinary does the dispatch based on the format
/// of the output specified by the command line options.
static Error executeObjcopyOnRawBinary(CopyConfig &Config, MemoryBuffer &In,
                                       Buffer &Out) {
  switch (Config.OutputFormat) {
  case FileFormat::ELF:
  // FIXME: Currently, we call elf::executeObjcopyOnRawBinary even if the
  // output format is binary/ihex or it's not given. This behavior differs from
  // GNU objcopy. See https://bugs.llvm.org/show_bug.cgi?id=42171 for details.
  case FileFormat::Binary:
  case FileFormat::IHex:
  case FileFormat::Unspecified:
    if (Error E = Config.parseELFConfig())
      return E;
    return elf::executeObjcopyOnRawBinary(Config, In, Out);
  }

  llvm_unreachable("unsupported output format");
}

/// The function executeObjcopyOnBinary does the dispatch based on the format
/// of the input binary (ELF, MachO or COFF).
static Error executeObjcopyOnBinary(CopyConfig &Config, object::Binary &In,
                                    Buffer &Out) {
  if (auto *ELFBinary = dyn_cast<object::ELFObjectFileBase>(&In)) {
    if (Error E = Config.parseELFConfig())
      return E;
    return elf::executeObjcopyOnBinary(Config, *ELFBinary, Out);
  } else if (auto *COFFBinary = dyn_cast<object::COFFObjectFile>(&In))
    return coff::executeObjcopyOnBinary(Config, *COFFBinary, Out);
  else if (auto *MachOBinary = dyn_cast<object::MachOObjectFile>(&In))
    return macho::executeObjcopyOnBinary(Config, *MachOBinary, Out);
  else if (auto *WasmBinary = dyn_cast<object::WasmObjectFile>(&In))
    return objcopy::wasm::executeObjcopyOnBinary(Config, *WasmBinary, Out);
  else
    return createStringError(object_error::invalid_file_type,
                             "unsupported object file format");
}

static Error executeObjcopyOnArchive(CopyConfig &Config, const Archive &Ar) {
  std::vector<NewArchiveMember> NewArchiveMembers;
  Error Err = Error::success();
  for (const Archive::Child &Child : Ar.children(Err)) {
    Expected<StringRef> ChildNameOrErr = Child.getName();
    if (!ChildNameOrErr)
      return createFileError(Ar.getFileName(), ChildNameOrErr.takeError());

    Expected<std::unique_ptr<Binary>> ChildOrErr = Child.getAsBinary();
    if (!ChildOrErr)
      return createFileError(Ar.getFileName() + "(" + *ChildNameOrErr + ")",
                             ChildOrErr.takeError());

    MemBuffer MB(ChildNameOrErr.get());
    if (Error E = executeObjcopyOnBinary(Config, *ChildOrErr->get(), MB))
      return E;

    Expected<NewArchiveMember> Member =
        NewArchiveMember::getOldMember(Child, Config.DeterministicArchives);
    if (!Member)
      return createFileError(Ar.getFileName(), Member.takeError());
    Member->Buf = MB.releaseMemoryBuffer();
    Member->MemberName = Member->Buf->getBufferIdentifier();
    NewArchiveMembers.push_back(std::move(*Member));
  }
  if (Err)
    return createFileError(Config.InputFilename, std::move(Err));

  return deepWriteArchive(Config.OutputFilename, NewArchiveMembers,
                          Ar.hasSymbolTable(), Ar.kind(),
                          Config.DeterministicArchives, Ar.isThin());
}

static Error restoreStatOnFile(StringRef Filename,
                               const sys::fs::file_status &Stat,
                               bool PreserveDates) {
  int FD;

  // Writing to stdout should not be treated as an error here, just
  // do not set access/modification times or permissions.
  if (Filename == "-")
    return Error::success();

  if (auto EC =
          sys::fs::openFileForWrite(Filename, FD, sys::fs::CD_OpenExisting))
    return createFileError(Filename, EC);

  if (PreserveDates)
    if (auto EC = sys::fs::setLastAccessAndModificationTime(
            FD, Stat.getLastAccessedTime(), Stat.getLastModificationTime()))
      return createFileError(Filename, EC);

  sys::fs::file_status OStat;
  if (std::error_code EC = sys::fs::status(FD, OStat))
    return createFileError(Filename, EC);
  if (OStat.type() == sys::fs::file_type::regular_file)
#ifdef _WIN32
    if (auto EC = sys::fs::setPermissions(
            Filename, static_cast<sys::fs::perms>(Stat.permissions() &
                                                  ~sys::fs::getUmask())))
#else
    if (auto EC = sys::fs::setPermissions(
            FD, static_cast<sys::fs::perms>(Stat.permissions() &
                                            ~sys::fs::getUmask())))
#endif
      return createFileError(Filename, EC);

  if (auto EC = sys::Process::SafelyCloseFileDescriptor(FD))
    return createFileError(Filename, EC);

  return Error::success();
}

/// The function executeObjcopy does the higher level dispatch based on the type
/// of input (raw binary, archive or single object file) and takes care of the
/// format-agnostic modifications, i.e. preserving dates.
static Error executeObjcopy(CopyConfig &Config) {
  sys::fs::file_status Stat;
  if (Config.InputFilename != "-") {
    if (auto EC = sys::fs::status(Config.InputFilename, Stat))
      return createFileError(Config.InputFilename, EC);
  } else {
    Stat.permissions(static_cast<sys::fs::perms>(0777));
  }

  using ProcessRawFn = Error (*)(CopyConfig &, MemoryBuffer &, Buffer &);
  ProcessRawFn ProcessRaw;
  switch (Config.InputFormat) {
  case FileFormat::Binary:
    ProcessRaw = executeObjcopyOnRawBinary;
    break;
  case FileFormat::IHex:
    ProcessRaw = executeObjcopyOnIHex;
    break;
  default:
    ProcessRaw = nullptr;
  }

  if (ProcessRaw) {
    auto BufOrErr = MemoryBuffer::getFileOrSTDIN(Config.InputFilename);
    if (!BufOrErr)
      return createFileError(Config.InputFilename, BufOrErr.getError());
    FileBuffer FB(Config.OutputFilename);
    if (Error E = ProcessRaw(Config, *BufOrErr->get(), FB))
      return E;
  } else {
    Expected<OwningBinary<llvm::object::Binary>> BinaryOrErr =
        createBinary(Config.InputFilename);
    if (!BinaryOrErr)
      return createFileError(Config.InputFilename, BinaryOrErr.takeError());

    if (Archive *Ar = dyn_cast<Archive>(BinaryOrErr.get().getBinary())) {
      if (Error E = executeObjcopyOnArchive(Config, *Ar))
        return E;
    } else {
      FileBuffer FB(Config.OutputFilename);
      if (Error E = executeObjcopyOnBinary(Config,
                                           *BinaryOrErr.get().getBinary(), FB))
        return E;
    }
  }

  if (Error E =
          restoreStatOnFile(Config.OutputFilename, Stat, Config.PreserveDates))
    return E;

  if (!Config.SplitDWO.empty()) {
    Stat.permissions(static_cast<sys::fs::perms>(0666));
    if (Error E =
            restoreStatOnFile(Config.SplitDWO, Stat, Config.PreserveDates))
      return E;
  }

  return Error::success();
}

namespace {

enum class ToolType { Objcopy, Strip, InstallNameTool };

} // anonymous namespace

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  ToolName = argv[0];
  ToolType Tool = StringSwitch<ToolType>(sys::path::stem(ToolName))
                      .EndsWith("strip", ToolType::Strip)
                      .EndsWith("install-name-tool", ToolType::InstallNameTool)
                      .EndsWith("install_name_tool", ToolType::InstallNameTool)
                      .Default(ToolType::Objcopy);
  // Expand response files.
  // TODO: Move these lines, which are copied from lib/Support/CommandLine.cpp,
  // into a separate function in the CommandLine library and call that function
  // here. This is duplicated code.
  SmallVector<const char *, 20> NewArgv(argv, argv + argc);
  BumpPtrAllocator A;
  StringSaver Saver(A);
  cl::ExpandResponseFiles(Saver,
                          Triple(sys::getProcessTriple()).isOSWindows()
                              ? cl::TokenizeWindowsCommandLine
                              : cl::TokenizeGNUCommandLine,
                          NewArgv);

  auto Args = makeArrayRef(NewArgv).drop_front();
  Expected<DriverConfig> DriverConfig =
      (Tool == ToolType::Strip) ? parseStripOptions(Args, reportWarning)
                                : ((Tool == ToolType::InstallNameTool)
                                       ? parseInstallNameToolOptions(Args)
                                       : parseObjcopyOptions(Args, reportWarning));
  if (!DriverConfig) {
    logAllUnhandledErrors(DriverConfig.takeError(),
                          WithColor::error(errs(), ToolName));
    return 1;
  }
  for (CopyConfig &CopyConfig : DriverConfig->CopyConfigs) {
    if (Error E = executeObjcopy(CopyConfig)) {
      logAllUnhandledErrors(std::move(E), WithColor::error(errs(), ToolName));
      return 1;
    }
  }

  return 0;
}
