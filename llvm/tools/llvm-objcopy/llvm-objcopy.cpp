//===- llvm-objcopy.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm-objcopy.h"
#include "COFF/COFFObjcopy.h"
#include "CopyConfig.h"
#include "ELF/ELFObjcopy.h"
#include "MachO/MachOObjcopy.h"
#include "wasm/WasmObjcopy.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ELFTypes.h"
#include "llvm/Object/Error.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/Wasm.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
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

Error writeToFile(StringRef OutputFileName,
                  std::function<Error(raw_ostream &)> Write) {
  if (OutputFileName == "-")
    return Write(outs());

  if (OutputFileName == "/dev/null") {
    raw_null_ostream Out;
    return Write(Out);
  }

  unsigned Mode = sys::fs::all_read | sys::fs::all_write | sys::fs::all_exe;
  Expected<sys::fs::TempFile> Temp =
      sys::fs::TempFile::create(OutputFileName + ".temp-objcopy-%%%%%%", Mode);
  if (!Temp)
    return createFileError(OutputFileName, Temp.takeError());

  raw_fd_ostream Out(Temp->FD, false);

  if (Error E = Write(Out)) {
    if (Error DiscardError = Temp->discard())
      return joinErrors(std::move(E), std::move(DiscardError));
    return E;
  }
  Out.flush();

  return Temp->keep(OutputFileName);
}

// The name this program was invoked as.
StringRef ToolName;

ErrorSuccess reportWarning(Error E) {
  assert(E);
  WithColor::warning(errs(), ToolName) << toString(std::move(E)) << '\n';
  return Error::success();
}

static Expected<DriverConfig> getDriverConfig(ArrayRef<const char *> Args) {
  StringRef Stem = sys::path::stem(ToolName);
  auto Is = [=](StringRef Tool) {
    // We need to recognize the following filenames:
    //
    // llvm-objcopy -> objcopy
    // strip-10.exe -> strip
    // powerpc64-unknown-freebsd13-objcopy -> objcopy
    // llvm-install-name-tool -> install-name-tool
    auto I = Stem.rfind_lower(Tool);
    return I != StringRef::npos &&
           (I + Tool.size() == Stem.size() || !isAlnum(Stem[I + Tool.size()]));
  };

  if (Is("bitcode-strip") || Is("bitcode_strip"))
    return parseBitcodeStripOptions(Args);
  else if (Is("strip"))
    return parseStripOptions(Args, reportWarning);
  else if (Is("install-name-tool") || Is("install_name_tool"))
    return parseInstallNameToolOptions(Args);
  else
    return parseObjcopyOptions(Args, reportWarning);
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
    // For regular files (as is the case for deepWriteArchive),
    // FileOutputBuffer::create will return OnDiskBuffer.
    // OnDiskBuffer uses a temporary file and then renames it. So in reality
    // there is no inefficiency / duplicated in-memory buffers in this case. For
    // now in-memory buffers can not be completely avoided since
    // NewArchiveMember still requires them even though writeArchive does not
    // write them on disk.
    Expected<std::unique_ptr<FileOutputBuffer>> FB =
        FileOutputBuffer::create(Member.MemberName, Member.Buf->getBufferSize(),
                                 FileOutputBuffer::F_executable);
    if (!FB)
      return FB.takeError();
    std::copy(Member.Buf->getBufferStart(), Member.Buf->getBufferEnd(),
              (*FB)->getBufferStart());
    if (Error E = (*FB)->commit())
      return E;
  }
  return Error::success();
}

/// The function executeObjcopyOnIHex does the dispatch based on the format
/// of the output specified by the command line options.
static Error executeObjcopyOnIHex(CopyConfig &Config, MemoryBuffer &In,
                                  raw_ostream &Out) {
  // TODO: support output formats other than ELF.
  if (Error E = Config.parseELFConfig())
    return E;
  return elf::executeObjcopyOnIHex(Config, In, Out);
}

/// The function executeObjcopyOnRawBinary does the dispatch based on the format
/// of the output specified by the command line options.
static Error executeObjcopyOnRawBinary(CopyConfig &Config, MemoryBuffer &In,
                                       raw_ostream &Out) {
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
                                    raw_ostream &Out) {
  if (auto *ELFBinary = dyn_cast<object::ELFObjectFileBase>(&In)) {
    if (Error E = Config.parseELFConfig())
      return E;
    return elf::executeObjcopyOnBinary(Config, *ELFBinary, Out);
  } else if (auto *COFFBinary = dyn_cast<object::COFFObjectFile>(&In))
    return coff::executeObjcopyOnBinary(Config, *COFFBinary, Out);
  else if (auto *MachOBinary = dyn_cast<object::MachOObjectFile>(&In))
    return macho::executeObjcopyOnBinary(Config, *MachOBinary, Out);
  else if (auto *MachOUniversalBinary =
               dyn_cast<object::MachOUniversalBinary>(&In))
    return macho::executeObjcopyOnMachOUniversalBinary(
        Config, *MachOUniversalBinary, Out);
  else if (auto *WasmBinary = dyn_cast<object::WasmObjectFile>(&In))
    return objcopy::wasm::executeObjcopyOnBinary(Config, *WasmBinary, Out);
  else
    return createStringError(object_error::invalid_file_type,
                             "unsupported object file format");
}

namespace llvm {
namespace objcopy {

Expected<std::vector<NewArchiveMember>>
createNewArchiveMembers(CopyConfig &Config, const Archive &Ar) {
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

    SmallVector<char, 0> Buffer;
    raw_svector_ostream MemStream(Buffer);

    if (Error E = executeObjcopyOnBinary(Config, *ChildOrErr->get(), MemStream))
      return std::move(E);

    Expected<NewArchiveMember> Member =
        NewArchiveMember::getOldMember(Child, Config.DeterministicArchives);
    if (!Member)
      return createFileError(Ar.getFileName(), Member.takeError());

    Member->Buf = std::make_unique<SmallVectorMemoryBuffer>(
        std::move(Buffer), ChildNameOrErr.get());
    Member->MemberName = Member->Buf->getBufferIdentifier();
    NewArchiveMembers.push_back(std::move(*Member));
  }
  if (Err)
    return createFileError(Config.InputFilename, std::move(Err));
  return std::move(NewArchiveMembers);
}

} // end namespace objcopy
} // end namespace llvm

static Error executeObjcopyOnArchive(CopyConfig &Config,
                                     const object::Archive &Ar) {
  Expected<std::vector<NewArchiveMember>> NewArchiveMembersOrErr =
      createNewArchiveMembers(Config, Ar);
  if (!NewArchiveMembersOrErr)
    return NewArchiveMembersOrErr.takeError();
  return deepWriteArchive(Config.OutputFilename, *NewArchiveMembersOrErr,
                          Ar.hasSymbolTable(), Ar.kind(),
                          Config.DeterministicArchives, Ar.isThin());
}

static Error restoreStatOnFile(StringRef Filename,
                               const sys::fs::file_status &Stat,
                               const CopyConfig &Config) {
  int FD;

  // Writing to stdout should not be treated as an error here, just
  // do not set access/modification times or permissions.
  if (Filename == "-")
    return Error::success();

  if (auto EC =
          sys::fs::openFileForWrite(Filename, FD, sys::fs::CD_OpenExisting))
    return createFileError(Filename, EC);

  if (Config.PreserveDates)
    if (auto EC = sys::fs::setLastAccessAndModificationTime(
            FD, Stat.getLastAccessedTime(), Stat.getLastModificationTime()))
      return createFileError(Filename, EC);

  sys::fs::file_status OStat;
  if (std::error_code EC = sys::fs::status(FD, OStat))
    return createFileError(Filename, EC);
  if (OStat.type() == sys::fs::file_type::regular_file) {
    sys::fs::perms Perm = Stat.permissions();
    if (Config.InputFilename != Config.OutputFilename)
      Perm = static_cast<sys::fs::perms>(Perm & ~sys::fs::getUmask() & ~06000);
#ifdef _WIN32
    if (auto EC = sys::fs::setPermissions(Filename, Perm))
#else
    if (auto EC = sys::fs::setPermissions(FD, Perm))
#endif
      return createFileError(Filename, EC);

#ifndef _WIN32
    // Keep ownership if llvm-objcopy is called under root.
    if (Config.InputFilename == Config.OutputFilename && OStat.getUser() == 0)
      sys::fs::changeFileOwnership(FD, Stat.getUser(), Stat.getGroup());
#endif
  }

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

  std::function<Error(raw_ostream & OutFile)> ObjcopyFunc;

  OwningBinary<llvm::object::Binary> BinaryHolder;
  std::unique_ptr<MemoryBuffer> MemoryBufferHolder;

  if (Config.InputFormat == FileFormat::Binary ||
      Config.InputFormat == FileFormat::IHex) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
        MemoryBuffer::getFileOrSTDIN(Config.InputFilename);
    if (!BufOrErr)
      return createFileError(Config.InputFilename, BufOrErr.getError());
    MemoryBufferHolder = std::move(*BufOrErr);

    if (Config.InputFormat == FileFormat::Binary)
      ObjcopyFunc = [&](raw_ostream &OutFile) -> Error {
        // Handle FileFormat::Binary.
        return executeObjcopyOnRawBinary(Config, *MemoryBufferHolder, OutFile);
      };
    else
      ObjcopyFunc = [&](raw_ostream &OutFile) -> Error {
        // Handle FileFormat::IHex.
        return executeObjcopyOnIHex(Config, *MemoryBufferHolder, OutFile);
      };
  } else {
    Expected<OwningBinary<llvm::object::Binary>> BinaryOrErr =
        createBinary(Config.InputFilename);
    if (!BinaryOrErr)
      return createFileError(Config.InputFilename, BinaryOrErr.takeError());
    BinaryHolder = std::move(*BinaryOrErr);

    if (Archive *Ar = dyn_cast<Archive>(BinaryHolder.getBinary())) {
      // Handle Archive.
      if (Error E = executeObjcopyOnArchive(Config, *Ar))
        return E;
    } else {
      // Handle llvm::object::Binary.
      ObjcopyFunc = [&](raw_ostream &OutFile) -> Error {
        return executeObjcopyOnBinary(Config, *BinaryHolder.getBinary(),
                                      OutFile);
      };
    }
  }

  if (ObjcopyFunc) {
    if (Config.SplitDWO.empty()) {
      // Apply transformations described by Config and store result into
      // Config.OutputFilename using specified ObjcopyFunc function.
      if (Error E = writeToFile(Config.OutputFilename, ObjcopyFunc))
        return E;
    } else {
      Config.ExtractDWO = true;
      Config.StripDWO = false;
      // Copy .dwo tables from the Config.InputFilename into Config.SplitDWO
      // file using specified ObjcopyFunc function.
      if (Error E = writeToFile(Config.SplitDWO, ObjcopyFunc))
        return E;
      Config.ExtractDWO = false;
      Config.StripDWO = true;
      // Apply transformations described by Config, remove .dwo tables and
      // store result into Config.OutputFilename using specified ObjcopyFunc
      // function.
      if (Error E = writeToFile(Config.OutputFilename, ObjcopyFunc))
        return E;
    }
  }

  if (Error E = restoreStatOnFile(Config.OutputFilename, Stat, Config))
    return E;

  if (!Config.SplitDWO.empty()) {
    Stat.permissions(static_cast<sys::fs::perms>(0666));
    if (Error E = restoreStatOnFile(Config.SplitDWO, Stat, Config))
      return E;
  }

  return Error::success();
}

namespace {

} // anonymous namespace

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  ToolName = argv[0];

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
  Expected<DriverConfig> DriverConfig = getDriverConfig(Args);

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
