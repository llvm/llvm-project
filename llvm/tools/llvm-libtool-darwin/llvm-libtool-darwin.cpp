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

#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/WithColor.h"

using namespace llvm;
using namespace llvm::object;

cl::OptionCategory LibtoolCategory("llvm-libtool-darwin Options");

static cl::opt<std::string> OutputFile("o", cl::desc("Specify output filename"),
                                       cl::value_desc("filename"), cl::Required,
                                       cl::cat(LibtoolCategory));

static cl::list<std::string> InputFiles(cl::Positional,
                                        cl::desc("<input files>"),
                                        cl::OneOrMore,
                                        cl::cat(LibtoolCategory));

enum class Operation { Static };

static cl::opt<Operation> LibraryOperation(
    cl::desc("Library Type: "),
    cl::values(
        clEnumValN(Operation::Static, "static",
                   "Produce a statically linked library from the input files")),
    cl::Required, cl::cat(LibtoolCategory));

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

static Error addMember(std::vector<NewArchiveMember> &Members,
                       StringRef FileName) {
  Expected<NewArchiveMember> NMOrErr =
      NewArchiveMember::getFile(FileName, /*Deterministic=*/true);

  if (!NMOrErr)
    return createFileError(FileName, NMOrErr.takeError());

  // For regular archives, use the basename of the object path for the member
  // name.
  NMOrErr->MemberName = sys::path::filename(NMOrErr->MemberName);

  // Verify that Member is a Mach-O object file.
  if (Error E = verifyMachOObject(*NMOrErr))
    return E;

  Members.push_back(std::move(*NMOrErr));
  return Error::success();
}

static Error createStaticLibrary() {
  std::vector<NewArchiveMember> NewMembers;
  for (StringRef Member : InputFiles)
    if (Error E = addMember(NewMembers, Member))
      return E;

  if (Error E = writeArchive(OutputFile, NewMembers,
                             /*WriteSymtab=*/true,
                             /*Kind=*/object::Archive::K_DARWIN,
                             /*Deterministic=*/true,
                             /*Thin=*/false))
    return E;
  return Error::success();
}

int main(int Argc, char **Argv) {
  InitLLVM X(Argc, Argv);
  cl::HideUnrelatedOptions({&LibtoolCategory, &ColorCategory});
  cl::ParseCommandLineOptions(Argc, Argv, "llvm-libtool-darwin\n");

  switch (LibraryOperation) {
  case Operation::Static:
    if (Error E = createStaticLibrary()) {
      WithColor::defaultErrorHandler(std::move(E));
      exit(EXIT_FAILURE);
    }
    break;
  }
}
