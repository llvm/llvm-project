//===-- llvm-libtool-darwin.cpp - a tool for creating libraries -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A utility for creating static and dynamic libraries for Darwin.
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/WithColor.h"
#include "llvm/TextAPI/MachO/Architecture.h"

using namespace llvm;
using namespace llvm::object;

cl::OptionCategory LibtoolCategory("llvm-libtool-darwin Options");

static cl::opt<std::string> OutputFile("o", cl::desc("Specify output filename"),
                                       cl::value_desc("filename"), cl::Required,
                                       cl::cat(LibtoolCategory));

static cl::list<std::string> InputFiles(cl::Positional,
                                        cl::desc("<input files>"),
                                        cl::ZeroOrMore,
                                        cl::cat(LibtoolCategory));

static cl::opt<std::string> ArchType(
    "arch_only", cl::desc("Specify architecture type for output library"),
    cl::value_desc("arch_type"), cl::ZeroOrMore, cl::cat(LibtoolCategory));

enum class Operation { Static };

static cl::opt<Operation> LibraryOperation(
    cl::desc("Library Type: "),
    cl::values(
        clEnumValN(Operation::Static, "static",
                   "Produce a statically linked library from the input files")),
    cl::Required, cl::cat(LibtoolCategory));

static cl::opt<bool> DeterministicOption(
    "D", cl::desc("Use zero for timestamps and UIDs/GIDs (Default)"),
    cl::init(false), cl::cat(LibtoolCategory));

static cl::opt<bool>
    NonDeterministicOption("U", cl::desc("Use actual timestamps and UIDs/GIDs"),
                           cl::init(false), cl::cat(LibtoolCategory));

static cl::opt<std::string>
    FileList("filelist",
             cl::desc("Pass in file containing a list of filenames"),
             cl::value_desc("listfile[,dirname]"), cl::cat(LibtoolCategory));

struct Config {
  bool Deterministic = true; // Updated by 'D' and 'U' modifiers.
};

static Error processFileList() {
  StringRef FileName, DirName;
  std::tie(FileName, DirName) = StringRef(FileList).rsplit(",");

  ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrErr =
      MemoryBuffer::getFileOrSTDIN(FileName, /*FileSize=*/-1,
                                   /*RequiresNullTerminator=*/false);
  if (std::error_code EC = FileOrErr.getError())
    return createFileError(FileName, errorCodeToError(EC));
  const MemoryBuffer &Ref = *FileOrErr.get();

  line_iterator I(Ref, /*SkipBlanks=*/false);
  if (I.is_at_eof())
    return createStringError(std::errc::invalid_argument,
                             "file list file: '%s' is empty",
                             FileName.str().c_str());
  for (; !I.is_at_eof(); ++I) {
    StringRef Line = *I;
    if (Line.empty())
      return createStringError(std::errc::invalid_argument,
                               "file list file: '%s': filename cannot be empty",
                               FileName.str().c_str());

    SmallString<128> Path;
    if (!DirName.empty())
      sys::path::append(Path, DirName, Line);
    else
      sys::path::append(Path, Line);
    InputFiles.push_back(static_cast<std::string>(Path));
  }
  return Error::success();
}

static Error validateArchitectureName(StringRef ArchitectureName) {
  if (!MachOObjectFile::isValidArch(ArchitectureName)) {
    std::string Buf;
    raw_string_ostream OS(Buf);
    for (StringRef Arch : MachOObjectFile::getValidArchs())
      OS << Arch << " ";

    return createStringError(
        std::errc::invalid_argument,
        "invalid architecture '%s': valid architecture names are %s",
        ArchitectureName.str().c_str(), OS.str().c_str());
  }
  return Error::success();
}

// Check that a file's architecture [FileCPUType, FileCPUSubtype]
// matches the architecture specified under -arch_only flag.
static bool acceptFileArch(uint32_t FileCPUType, uint32_t FileCPUSubtype) {
  uint32_t ArchCPUType, ArchCPUSubtype;
  std::tie(ArchCPUType, ArchCPUSubtype) = MachO::getCPUTypeFromArchitecture(
      MachO::getArchitectureFromName(ArchType));

  if (ArchCPUType != FileCPUType)
    return false;

  switch (ArchCPUType) {
  case MachO::CPU_TYPE_ARM:
  case MachO::CPU_TYPE_ARM64_32:
  case MachO::CPU_TYPE_X86_64:
    return ArchCPUSubtype == FileCPUSubtype;

  case MachO::CPU_TYPE_ARM64:
    if (ArchCPUSubtype == MachO::CPU_SUBTYPE_ARM64_ALL)
      return FileCPUSubtype == MachO::CPU_SUBTYPE_ARM64_ALL ||
             FileCPUSubtype == MachO::CPU_SUBTYPE_ARM64_V8;
    else
      return ArchCPUSubtype == FileCPUSubtype;

  default:
    return true;
  }
}

static Error verifyAndAddMachOObject(std::vector<NewArchiveMember> &Members,
                                     NewArchiveMember Member) {
  auto MBRef = Member.Buf->getMemBufferRef();
  Expected<std::unique_ptr<object::ObjectFile>> ObjOrErr =
      object::ObjectFile::createObjectFile(MBRef);

  // Throw error if not a valid object file.
  if (!ObjOrErr)
    return createFileError(Member.MemberName, ObjOrErr.takeError());

  // Throw error if not in Mach-O format.
  if (!isa<object::MachOObjectFile>(**ObjOrErr))
    return createStringError(std::errc::invalid_argument,
                             "'%s': format not supported",
                             Member.MemberName.data());

  auto *O = dyn_cast<MachOObjectFile>(ObjOrErr->get());
  uint32_t FileCPUType, FileCPUSubtype;
  std::tie(FileCPUType, FileCPUSubtype) = MachO::getCPUTypeFromArchitecture(
      MachO::getArchitectureFromName(O->getArchTriple().getArchName()));

  // If -arch_only is specified then skip this file if it doesn't match
  // the architecture specified.
  if (!ArchType.empty() && !acceptFileArch(FileCPUType, FileCPUSubtype)) {
    return Error::success();
  }

  Members.push_back(std::move(Member));
  return Error::success();
}

static Error addChildMember(std::vector<NewArchiveMember> &Members,
                            const object::Archive::Child &M, const Config &C) {
  Expected<NewArchiveMember> NMOrErr =
      NewArchiveMember::getOldMember(M, C.Deterministic);
  if (!NMOrErr)
    return NMOrErr.takeError();

  if (Error E = verifyAndAddMachOObject(Members, std::move(*NMOrErr)))
    return E;

  return Error::success();
}

static Error processArchive(std::vector<NewArchiveMember> &Members,
                            object::Archive &Lib, StringRef FileName,
                            const Config &C) {
  Error Err = Error::success();
  for (const object::Archive::Child &Child : Lib.children(Err))
    if (Error E = addChildMember(Members, Child, C))
      return createFileError(FileName, std::move(E));
  if (Err)
    return createFileError(FileName, std::move(Err));

  return Error::success();
}

static Error
addArchiveMembers(std::vector<NewArchiveMember> &Members,
                  std::vector<std::unique_ptr<MemoryBuffer>> &ArchiveBuffers,
                  NewArchiveMember NM, StringRef FileName, const Config &C) {
  Expected<std::unique_ptr<Archive>> LibOrErr =
      object::Archive::create(NM.Buf->getMemBufferRef());
  if (!LibOrErr)
    return createFileError(FileName, LibOrErr.takeError());

  if (Error E = processArchive(Members, **LibOrErr, FileName, C))
    return E;

  // Update vector ArchiveBuffers with the MemoryBuffers to transfer
  // ownership.
  ArchiveBuffers.push_back(std::move(NM.Buf));
  return Error::success();
}

static Error addUniversalMembers(
    std::vector<NewArchiveMember> &Members,
    std::vector<std::unique_ptr<MemoryBuffer>> &UniversalBuffers,
    NewArchiveMember NM, StringRef FileName, const Config &C) {
  Expected<std::unique_ptr<MachOUniversalBinary>> BinaryOrErr =
      MachOUniversalBinary::create(NM.Buf->getMemBufferRef());
  if (!BinaryOrErr)
    return createFileError(FileName, BinaryOrErr.takeError());

  auto *UO = BinaryOrErr->get();
  for (const MachOUniversalBinary::ObjectForArch &O : UO->objects()) {

    Expected<std::unique_ptr<MachOObjectFile>> MachOObjOrErr =
        O.getAsObjectFile();
    if (MachOObjOrErr) {
      NewArchiveMember NewMember =
          NewArchiveMember(MachOObjOrErr->get()->getMemoryBufferRef());
      NewMember.MemberName = sys::path::filename(NewMember.MemberName);

      if (Error E = verifyAndAddMachOObject(Members, std::move(NewMember)))
        return E;
      continue;
    }

    Expected<std::unique_ptr<Archive>> ArchiveOrError = O.getAsArchive();
    if (ArchiveOrError) {
      // A universal file member can either be a MachOObjectFile or an Archive.
      // In case we can successfully cast the member as an Archive, it is safe
      // to throw away the error generated due to casting the object as a
      // MachOObjectFile.
      consumeError(MachOObjOrErr.takeError());

      if (Error E = processArchive(Members, **ArchiveOrError, FileName, C))
        return E;
      continue;
    }

    Error CombinedError =
        joinErrors(ArchiveOrError.takeError(), MachOObjOrErr.takeError());
    return createFileError(FileName, std::move(CombinedError));
  }

  // Update vector UniversalBuffers with the MemoryBuffers to transfer
  // ownership.
  UniversalBuffers.push_back(std::move(NM.Buf));
  return Error::success();
}

static Error addMember(std::vector<NewArchiveMember> &Members,
                       std::vector<std::unique_ptr<MemoryBuffer>> &FileBuffers,
                       StringRef FileName, const Config &C) {
  Expected<NewArchiveMember> NMOrErr =
      NewArchiveMember::getFile(FileName, C.Deterministic);
  if (!NMOrErr)
    return createFileError(FileName, NMOrErr.takeError());

  // For regular archives, use the basename of the object path for the member
  // name.
  NMOrErr->MemberName = sys::path::filename(NMOrErr->MemberName);
  file_magic Magic = identify_magic(NMOrErr->Buf->getBuffer());

  // Flatten archives.
  if (Magic == file_magic::archive)
    return addArchiveMembers(Members, FileBuffers, std::move(*NMOrErr),
                             FileName, C);

  // Flatten universal files.
  if (Magic == file_magic::macho_universal_binary)
    return addUniversalMembers(Members, FileBuffers, std::move(*NMOrErr),
                               FileName, C);

  if (Error E = verifyAndAddMachOObject(Members, std::move(*NMOrErr)))
    return E;
  return Error::success();
}

static Error createStaticLibrary(const Config &C) {
  std::vector<NewArchiveMember> NewMembers;
  std::vector<std::unique_ptr<MemoryBuffer>> FileBuffers;
  for (StringRef FileName : InputFiles)
    if (Error E = addMember(NewMembers, FileBuffers, FileName, C))
      return E;

  if (NewMembers.empty() && !ArchType.empty())
    return createStringError(std::errc::invalid_argument,
                             "no library created (no object files in input "
                             "files matching -arch_only %s)",
                             ArchType.c_str());

  if (Error E =
          writeArchive(OutputFile, NewMembers,
                       /*WriteSymtab=*/true,
                       /*Kind=*/object::Archive::K_DARWIN, C.Deterministic,
                       /*Thin=*/false))
    return E;
  return Error::success();
}

static Expected<Config> parseCommandLine(int Argc, char **Argv) {
  Config C;
  cl::ParseCommandLineOptions(Argc, Argv, "llvm-libtool-darwin\n");

  if (DeterministicOption && NonDeterministicOption)
    return createStringError(std::errc::invalid_argument,
                             "cannot specify both -D and -U flags");
  else if (NonDeterministicOption)
    C.Deterministic = false;

  if (!FileList.empty())
    if (Error E = processFileList())
      return std::move(E);

  if (InputFiles.empty())
    return createStringError(std::errc::invalid_argument,
                             "no input files specified");

  if (ArchType.getNumOccurrences())
    if (Error E = validateArchitectureName(ArchType))
      return std::move(E);

  return C;
}

int main(int Argc, char **Argv) {
  InitLLVM X(Argc, Argv);
  cl::HideUnrelatedOptions({&LibtoolCategory, &ColorCategory});
  Expected<Config> ConfigOrErr = parseCommandLine(Argc, Argv);
  if (!ConfigOrErr) {
    WithColor::defaultErrorHandler(ConfigOrErr.takeError());
    return EXIT_FAILURE;
  }

  Config C = *ConfigOrErr;
  switch (LibraryOperation) {
  case Operation::Static:
    if (Error E = createStaticLibrary(C)) {
      WithColor::defaultErrorHandler(std::move(E));
      return EXIT_FAILURE;
    }
    break;
  }
}
