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
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/WithColor.h"

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

static Error verifyMachOObject(const NewArchiveMember &Member) {
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

  return Error::success();
}

static Error addChildMember(std::vector<NewArchiveMember> &Members,
                            const object::Archive::Child &M, const Config &C) {
  Expected<NewArchiveMember> NMOrErr =
      NewArchiveMember::getOldMember(M, C.Deterministic);
  if (!NMOrErr)
    return NMOrErr.takeError();

  // Verify that Member is a Mach-O object file.
  if (Error E = verifyMachOObject(*NMOrErr))
    return E;

  Members.push_back(std::move(*NMOrErr));
  return Error::success();
}

static Error
addMember(std::vector<NewArchiveMember> &Members, StringRef FileName,
          std::vector<std::unique_ptr<MemoryBuffer>> &ArchiveBuffers,
          const Config &C) {
  Expected<NewArchiveMember> NMOrErr =
      NewArchiveMember::getFile(FileName, C.Deterministic);
  if (!NMOrErr)
    return createFileError(FileName, NMOrErr.takeError());

  // For regular archives, use the basename of the object path for the member
  // name.
  NMOrErr->MemberName = sys::path::filename(NMOrErr->MemberName);

  // Flatten archives.
  if (identify_magic(NMOrErr->Buf->getBuffer()) == file_magic::archive) {
    Expected<std::unique_ptr<Archive>> LibOrErr =
        object::Archive::create(NMOrErr->Buf->getMemBufferRef());
    if (!LibOrErr)
      return createFileError(FileName, LibOrErr.takeError());
    object::Archive &Lib = **LibOrErr;

    Error Err = Error::success();
    for (const object::Archive::Child &Child : Lib.children(Err))
      if (Error E = addChildMember(Members, Child, C))
        return createFileError(FileName, std::move(E));
    if (Err)
      return createFileError(FileName, std::move(Err));

    // Update vector ArchiveBuffers with the MemoryBuffers to transfer
    // ownership.
    ArchiveBuffers.push_back(std::move(NMOrErr->Buf));
    return Error::success();
  }

  // Verify that Member is a Mach-O object file.
  if (Error E = verifyMachOObject(*NMOrErr))
    return E;

  Members.push_back(std::move(*NMOrErr));
  return Error::success();
}

static Error createStaticLibrary(const Config &C) {
  std::vector<NewArchiveMember> NewMembers;
  std::vector<std::unique_ptr<MemoryBuffer>> ArchiveBuffers;
  for (StringRef Member : InputFiles)
    if (Error E = addMember(NewMembers, Member, ArchiveBuffers, C))
      return E;

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
