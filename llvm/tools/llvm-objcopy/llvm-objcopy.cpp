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
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Memory.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
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
  WithColor::error(errs(), ToolName) << Message << ".\n";
  errs().flush();
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

/// The function executeObjcopyOnRawBinary does the dispatch based on the format
/// of the output specified by the command line options.
static Error executeObjcopyOnRawBinary(const CopyConfig &Config,
                                       MemoryBuffer &In, Buffer &Out) {
  // TODO: llvm-objcopy should parse CopyConfig.OutputFormat to recognize
  // formats other than ELF / "binary" and invoke
  // elf::executeObjcopyOnRawBinary, macho::executeObjcopyOnRawBinary or
  // coff::executeObjcopyOnRawBinary accordingly.
  return elf::executeObjcopyOnRawBinary(Config, In, Out);
}

/// The function executeObjcopyOnBinary does the dispatch based on the format
/// of the input binary (ELF, MachO or COFF).
static Error executeObjcopyOnBinary(const CopyConfig &Config,
                                    object::Binary &In, Buffer &Out) {
  if (auto *ELFBinary = dyn_cast<object::ELFObjectFileBase>(&In))
    return elf::executeObjcopyOnBinary(Config, *ELFBinary, Out);
  else if (auto *COFFBinary = dyn_cast<object::COFFObjectFile>(&In))
    return coff::executeObjcopyOnBinary(Config, *COFFBinary, Out);
  else if (auto *MachOBinary = dyn_cast<object::MachOObjectFile>(&In))
    return macho::executeObjcopyOnBinary(Config, *MachOBinary, Out);
  else
    return createStringError(object_error::invalid_file_type,
                             "Unsupported object file format");
}

static Error executeObjcopyOnArchive(const CopyConfig &Config,
                                     const Archive &Ar) {
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

static Error restoreDateOnFile(StringRef Filename,
                               const sys::fs::file_status &Stat) {
  int FD;

  if (auto EC =
          sys::fs::openFileForWrite(Filename, FD, sys::fs::CD_OpenExisting))
    return createFileError(Filename, EC);

  if (auto EC = sys::fs::setLastAccessAndModificationTime(
          FD, Stat.getLastAccessedTime(), Stat.getLastModificationTime()))
    return createFileError(Filename, EC);

  if (auto EC = sys::Process::SafelyCloseFileDescriptor(FD))
    return createFileError(Filename, EC);

  return Error::success();
}

/// The function executeObjcopy does the higher level dispatch based on the type
/// of input (raw binary, archive or single object file) and takes care of the
/// format-agnostic modifications, i.e. preserving dates.
static Error executeObjcopy(const CopyConfig &Config) {
  sys::fs::file_status Stat;
  if (Config.PreserveDates)
    if (auto EC = sys::fs::status(Config.InputFilename, Stat))
      return createFileError(Config.InputFilename, EC);

  if (Config.InputFormat == "binary") {
    auto BufOrErr = MemoryBuffer::getFile(Config.InputFilename);
    if (!BufOrErr)
      return createFileError(Config.InputFilename, BufOrErr.getError());
    FileBuffer FB(Config.OutputFilename);
    if (Error E = executeObjcopyOnRawBinary(Config, *BufOrErr->get(), FB))
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

  if (Config.PreserveDates) {
    if (Error E = restoreDateOnFile(Config.OutputFilename, Stat))
      return E;
    if (!Config.SplitDWO.empty())
      if (Error E = restoreDateOnFile(Config.SplitDWO, Stat))
        return E;
  }

  return Error::success();
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  ToolName = argv[0];
  bool IsStrip = sys::path::stem(ToolName).contains("strip");
  Expected<DriverConfig> DriverConfig =
      IsStrip ? parseStripOptions(makeArrayRef(argv + 1, argc))
              : parseObjcopyOptions(makeArrayRef(argv + 1, argc));
  if (!DriverConfig) {
    logAllUnhandledErrors(DriverConfig.takeError(),
                          WithColor::error(errs(), ToolName));
    return 1;
  }
  for (const CopyConfig &CopyConfig : DriverConfig->CopyConfigs) {
    if (Error E = executeObjcopy(CopyConfig)) {
      logAllUnhandledErrors(std::move(E), WithColor::error(errs(), ToolName));
      return 1;
    }
  }

  return 0;
}
