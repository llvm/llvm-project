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
#include "llvm/IR/LLVMContext.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/MachOUniversalWriter.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/WithColor.h"
#include "llvm/TextAPI/MachO/Architecture.h"
#include <map>

using namespace llvm;
using namespace llvm::object;

static LLVMContext LLVMCtx;

typedef std::map<uint64_t, std::vector<NewArchiveMember>>
    MembersPerArchitectureMap;

cl::OptionCategory LibtoolCategory("llvm-libtool-darwin Options");

static cl::opt<std::string> OutputFile("o", cl::desc("Specify output filename"),
                                       cl::value_desc("filename"),
                                       cl::cat(LibtoolCategory));

static cl::list<std::string> InputFiles(cl::Positional,
                                        cl::desc("<input files>"),
                                        cl::ZeroOrMore,
                                        cl::cat(LibtoolCategory));

static cl::opt<std::string> ArchType(
    "arch_only", cl::desc("Specify architecture type for output library"),
    cl::value_desc("arch_type"), cl::ZeroOrMore, cl::cat(LibtoolCategory));

enum class Operation { None, Static };

static cl::opt<Operation> LibraryOperation(
    cl::desc("Library Type: "),
    cl::values(
        clEnumValN(Operation::Static, "static",
                   "Produce a statically linked library from the input files")),
    cl::init(Operation::None), cl::cat(LibtoolCategory));

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

static cl::list<std::string> Libraries(
    "l",
    cl::desc(
        "l<x> searches for the library libx.a in the library search path. If"
        " the string 'x' ends with '.o', then the library 'x' is searched for"
        " without prepending 'lib' or appending '.a'"),
    cl::ZeroOrMore, cl::Prefix, cl::cat(LibtoolCategory));

static cl::list<std::string> LibrarySearchDirs(
    "L",
    cl::desc(
        "L<dir> adds <dir> to the list of directories in which to search for"
        " libraries"),
    cl::ZeroOrMore, cl::Prefix, cl::cat(LibtoolCategory));

static cl::opt<bool>
    VersionOption("V", cl::desc("Print the version number and exit"),
                  cl::cat(LibtoolCategory));

static cl::opt<bool> NoWarningForNoSymbols(
    "no_warning_for_no_symbols",
    cl::desc("Do not warn about files that have no symbols"),
    cl::cat(LibtoolCategory), cl::init(false));

static const std::array<std::string, 3> StandardSearchDirs{
    "/lib",
    "/usr/lib",
    "/usr/local/lib",
};

struct Config {
  bool Deterministic = true; // Updated by 'D' and 'U' modifiers.
  uint32_t ArchCPUType;
  uint32_t ArchCPUSubtype;
};

static Expected<std::string> searchForFile(const Twine &FileName) {

  auto FindLib =
      [FileName](ArrayRef<std::string> SearchDirs) -> Optional<std::string> {
    for (StringRef Dir : SearchDirs) {
      SmallString<128> Path;
      sys::path::append(Path, Dir, FileName);

      if (sys::fs::exists(Path))
        return std::string(Path);
    }
    return None;
  };

  Optional<std::string> Found = FindLib(LibrarySearchDirs);
  if (!Found)
    Found = FindLib(StandardSearchDirs);
  if (Found)
    return *Found;

  return createStringError(std::errc::invalid_argument,
                           "cannot locate file '%s'", FileName.str().c_str());
}

static Error processCommandLineLibraries() {
  for (StringRef BaseName : Libraries) {
    Expected<std::string> FullPath = searchForFile(
        BaseName.endswith(".o") ? BaseName.str() : "lib" + BaseName + ".a");
    if (!FullPath)
      return FullPath.takeError();
    InputFiles.push_back(FullPath.get());
  }

  return Error::success();
}

static Error processFileList() {
  StringRef FileName, DirName;
  std::tie(FileName, DirName) = StringRef(FileList).rsplit(",");

  ErrorOr<std::unique_ptr<MemoryBuffer>> FileOrErr =
      MemoryBuffer::getFileOrSTDIN(FileName, /*IsText=*/false,
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

static uint64_t getCPUID(uint32_t CPUType, uint32_t CPUSubtype) {
  switch (CPUType) {
  case MachO::CPU_TYPE_ARM:
  case MachO::CPU_TYPE_ARM64:
  case MachO::CPU_TYPE_ARM64_32:
  case MachO::CPU_TYPE_X86_64:
    // We consider CPUSubtype only for the above 4 CPUTypes to match cctools'
    // libtool behavior.
    return static_cast<uint64_t>(CPUType) << 32 | CPUSubtype;
  default:
    return CPUType;
  }
}

// Check that a file's architecture [FileCPUType, FileCPUSubtype]
// matches the architecture specified under -arch_only flag.
static bool acceptFileArch(uint32_t FileCPUType, uint32_t FileCPUSubtype,
                           const Config &C) {
  if (C.ArchCPUType != FileCPUType)
    return false;

  switch (C.ArchCPUType) {
  case MachO::CPU_TYPE_ARM:
  case MachO::CPU_TYPE_ARM64_32:
  case MachO::CPU_TYPE_X86_64:
    return C.ArchCPUSubtype == FileCPUSubtype;

  case MachO::CPU_TYPE_ARM64:
    if (C.ArchCPUSubtype == MachO::CPU_SUBTYPE_ARM64_ALL)
      return FileCPUSubtype == MachO::CPU_SUBTYPE_ARM64_ALL ||
             FileCPUSubtype == MachO::CPU_SUBTYPE_ARM64_V8;
    else
      return C.ArchCPUSubtype == FileCPUSubtype;

  default:
    return true;
  }
}

static Error verifyAndAddMachOObject(MembersPerArchitectureMap &Members,
                                     NewArchiveMember Member, const Config &C) {
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
  if (!ArchType.empty() && !acceptFileArch(FileCPUType, FileCPUSubtype, C)) {
    return Error::success();
  }

  if (!NoWarningForNoSymbols && O->symbols().empty())
    WithColor::warning() << Member.MemberName + " has no symbols\n";

  uint64_t FileCPUID = getCPUID(FileCPUType, FileCPUSubtype);
  Members[FileCPUID].push_back(std::move(Member));
  return Error::success();
}

static Error verifyAndAddIRObject(MembersPerArchitectureMap &Members,
                                  NewArchiveMember Member, const Config &C) {
  auto MBRef = Member.Buf->getMemBufferRef();
  Expected<std::unique_ptr<object::IRObjectFile>> IROrErr =
      object::IRObjectFile::create(MBRef, LLVMCtx);

  // Throw error if not a valid IR object file.
  if (!IROrErr)
    return createFileError(Member.MemberName, IROrErr.takeError());

  Triple TT = Triple(IROrErr->get()->getTargetTriple());

  Expected<uint32_t> FileCPUTypeOrErr = MachO::getCPUType(TT);
  if (!FileCPUTypeOrErr)
    return FileCPUTypeOrErr.takeError();

  Expected<uint32_t> FileCPUSubTypeOrErr = MachO::getCPUSubType(TT);
  if (!FileCPUSubTypeOrErr)
    return FileCPUSubTypeOrErr.takeError();

  // If -arch_only is specified then skip this file if it doesn't match
  // the architecture specified.
  if (!ArchType.empty() &&
      !acceptFileArch(*FileCPUTypeOrErr, *FileCPUSubTypeOrErr, C)) {
    return Error::success();
  }

  uint64_t FileCPUID = getCPUID(*FileCPUTypeOrErr, *FileCPUSubTypeOrErr);
  Members[FileCPUID].push_back(std::move(Member));
  return Error::success();
}

static Error addChildMember(MembersPerArchitectureMap &Members,
                            const object::Archive::Child &M, const Config &C) {
  Expected<NewArchiveMember> NMOrErr =
      NewArchiveMember::getOldMember(M, C.Deterministic);
  if (!NMOrErr)
    return NMOrErr.takeError();

  file_magic Magic = identify_magic(NMOrErr->Buf->getBuffer());

  if (Magic == file_magic::bitcode)
    return verifyAndAddIRObject(Members, std::move(*NMOrErr), C);

  if (Error E = verifyAndAddMachOObject(Members, std::move(*NMOrErr), C))
    return E;

  return Error::success();
}

static Error processArchive(MembersPerArchitectureMap &Members,
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
addArchiveMembers(MembersPerArchitectureMap &Members,
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
    MembersPerArchitectureMap &Members,
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

      if (Error E = verifyAndAddMachOObject(Members, std::move(NewMember), C))
        return E;
      continue;
    }

    Expected<std::unique_ptr<IRObjectFile>> IRObjectOrError =
        O.getAsIRObject(LLVMCtx);
    if (IRObjectOrError) {
      // A universal file member can be a MachOObjectFile, an IRObject or an
      // Archive. In case we can successfully cast the member as an IRObject, it
      // is safe to throw away the error generated due to casting the object as
      // a MachOObjectFile.
      consumeError(MachOObjOrErr.takeError());

      NewArchiveMember NewMember =
          NewArchiveMember(IRObjectOrError->get()->getMemoryBufferRef());
      NewMember.MemberName = sys::path::filename(NewMember.MemberName);

      if (Error E = verifyAndAddIRObject(Members, std::move(NewMember), C))
        return E;
      continue;
    }

    Expected<std::unique_ptr<Archive>> ArchiveOrError = O.getAsArchive();
    if (ArchiveOrError) {
      // A universal file member can be a MachOObjectFile, an IRObject or an
      // Archive. In case we can successfully cast the member as an Archive, it
      // is safe to throw away the error generated due to casting the object as
      // a MachOObjectFile.
      consumeError(MachOObjOrErr.takeError());
      consumeError(IRObjectOrError.takeError());

      if (Error E = processArchive(Members, **ArchiveOrError, FileName, C))
        return E;
      continue;
    }

    Error CombinedError = joinErrors(
        ArchiveOrError.takeError(),
        joinErrors(IRObjectOrError.takeError(), MachOObjOrErr.takeError()));
    return createFileError(FileName, std::move(CombinedError));
  }

  // Update vector UniversalBuffers with the MemoryBuffers to transfer
  // ownership.
  UniversalBuffers.push_back(std::move(NM.Buf));
  return Error::success();
}

static Error addMember(MembersPerArchitectureMap &Members,
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

  // Bitcode files.
  if (Magic == file_magic::bitcode)
    return verifyAndAddIRObject(Members, std::move(*NMOrErr), C);

  if (Error E = verifyAndAddMachOObject(Members, std::move(*NMOrErr), C))
    return E;
  return Error::success();
}

static Expected<SmallVector<Slice, 2>>
buildSlices(ArrayRef<OwningBinary<Archive>> OutputBinaries) {
  SmallVector<Slice, 2> Slices;

  for (const auto &OB : OutputBinaries) {
    const Archive &A = *OB.getBinary();
    Expected<Slice> ArchiveSlice = Slice::create(A, &LLVMCtx);
    if (!ArchiveSlice)
      return ArchiveSlice.takeError();
    Slices.push_back(*ArchiveSlice);
  }
  return Slices;
}

static Error createStaticLibrary(const Config &C) {
  MembersPerArchitectureMap NewMembers;
  std::vector<std::unique_ptr<MemoryBuffer>> FileBuffers;
  for (StringRef FileName : InputFiles)
    if (Error E = addMember(NewMembers, FileBuffers, FileName, C))
      return E;

  if (!ArchType.empty()) {
    uint64_t ArchCPUID = getCPUID(C.ArchCPUType, C.ArchCPUSubtype);
    if (NewMembers.find(ArchCPUID) == NewMembers.end())
      return createStringError(std::errc::invalid_argument,
                               "no library created (no object files in input "
                               "files matching -arch_only %s)",
                               ArchType.c_str());
  }

  if (NewMembers.size() == 1) {
    if (Error E =
            writeArchive(OutputFile, NewMembers.begin()->second,
                         /*WriteSymtab=*/true,
                         /*Kind=*/object::Archive::K_DARWIN, C.Deterministic,
                         /*Thin=*/false))
      return E;
  } else {
    SmallVector<OwningBinary<Archive>, 2> OutputBinaries;
    for (const std::pair<const uint64_t, std::vector<NewArchiveMember>> &M :
         NewMembers) {
      Expected<std::unique_ptr<MemoryBuffer>> OutputBufferOrErr =
          writeArchiveToBuffer(M.second,
                               /*WriteSymtab=*/true,
                               /*Kind=*/object::Archive::K_DARWIN,
                               C.Deterministic,
                               /*Thin=*/false);
      if (!OutputBufferOrErr)
        return OutputBufferOrErr.takeError();
      std::unique_ptr<MemoryBuffer> &OutputBuffer = OutputBufferOrErr.get();

      Expected<std::unique_ptr<Archive>> ArchiveOrError =
          Archive::create(OutputBuffer->getMemBufferRef());
      if (!ArchiveOrError)
        return ArchiveOrError.takeError();
      std::unique_ptr<Archive> &A = ArchiveOrError.get();

      OutputBinaries.push_back(
          OwningBinary<Archive>(std::move(A), std::move(OutputBuffer)));
    }

    Expected<SmallVector<Slice, 2>> Slices = buildSlices(OutputBinaries);
    if (!Slices)
      return Slices.takeError();

    llvm::stable_sort(*Slices);
    if (Error E = writeUniversalBinary(*Slices, OutputFile))
      return E;
  }
  return Error::success();
}

static Expected<Config> parseCommandLine(int Argc, char **Argv) {
  Config C;
  cl::ParseCommandLineOptions(Argc, Argv, "llvm-libtool-darwin\n");

  if (LibraryOperation == Operation::None) {
    if (!VersionOption) {
      std::string Error;
      raw_string_ostream Stream(Error);
      LibraryOperation.error("must be specified", "", Stream);
      return createStringError(std::errc::invalid_argument, Error.c_str());
    }
    return C;
  }

  if (OutputFile.empty()) {
    std::string Error;
    raw_string_ostream Stream(Error);
    OutputFile.error("must be specified", "o", Stream);
    return createStringError(std::errc::invalid_argument, Error.c_str());
  }

  if (DeterministicOption && NonDeterministicOption)
    return createStringError(std::errc::invalid_argument,
                             "cannot specify both -D and -U flags");
  else if (NonDeterministicOption)
    C.Deterministic = false;

  if (!Libraries.empty())
    if (Error E = processCommandLineLibraries())
      return std::move(E);

  if (!FileList.empty())
    if (Error E = processFileList())
      return std::move(E);

  if (InputFiles.empty())
    return createStringError(std::errc::invalid_argument,
                             "no input files specified");

  if (ArchType.getNumOccurrences()) {
    if (Error E = validateArchitectureName(ArchType))
      return std::move(E);

    std::tie(C.ArchCPUType, C.ArchCPUSubtype) =
        MachO::getCPUTypeFromArchitecture(
            MachO::getArchitectureFromName(ArchType));
  }

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

  if (VersionOption)
    cl::PrintVersionMessage();

  Config C = *ConfigOrErr;
  switch (LibraryOperation) {
  case Operation::None:
    break;
  case Operation::Static:
    if (Error E = createStaticLibrary(C)) {
      WithColor::defaultErrorHandler(std::move(E));
      return EXIT_FAILURE;
    }
    break;
  }
}
