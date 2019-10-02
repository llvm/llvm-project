//===- LibDriver.cpp - lib.exe-compatible driver --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines an interface to a lib.exe-compatible driver that also understands
// bitcode files. Used by llvm-lib and lld-link /lib.
//
//===----------------------------------------------------------------------===//

#include "llvm/ToolDrivers/llvm-lib/LibDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/WindowsMachineFlag.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

enum {
  OPT_INVALID = 0,
#define OPTION(_1, _2, ID, _4, _5, _6, _7, _8, _9, _10, _11, _12) OPT_##ID,
#include "Options.inc"
#undef OPTION
};

#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "Options.inc"
#undef PREFIX

static const opt::OptTable::Info InfoTable[] = {
#define OPTION(X1, X2, ID, KIND, GROUP, ALIAS, X7, X8, X9, X10, X11, X12)      \
  {X1, X2, X10,         X11,         OPT_##ID, opt::Option::KIND##Class,       \
   X9, X8, OPT_##GROUP, OPT_##ALIAS, X7,       X12},
#include "Options.inc"
#undef OPTION
};

class LibOptTable : public opt::OptTable {
public:
  LibOptTable() : OptTable(InfoTable, true) {}
};

}

static std::string getOutputPath(opt::InputArgList *Args,
                                 const NewArchiveMember &FirstMember) {
  if (auto *Arg = Args->getLastArg(OPT_out))
    return Arg->getValue();
  SmallString<128> Val = StringRef(FirstMember.Buf->getBufferIdentifier());
  sys::path::replace_extension(Val, ".lib");
  return Val.str();
}

static std::vector<StringRef> getSearchPaths(opt::InputArgList *Args,
                                             StringSaver &Saver) {
  std::vector<StringRef> Ret;
  // Add current directory as first item of the search path.
  Ret.push_back("");

  // Add /libpath flags.
  for (auto *Arg : Args->filtered(OPT_libpath))
    Ret.push_back(Arg->getValue());

  // Add $LIB.
  Optional<std::string> EnvOpt = sys::Process::GetEnv("LIB");
  if (!EnvOpt.hasValue())
    return Ret;
  StringRef Env = Saver.save(*EnvOpt);
  while (!Env.empty()) {
    StringRef Path;
    std::tie(Path, Env) = Env.split(';');
    Ret.push_back(Path);
  }
  return Ret;
}

static std::string findInputFile(StringRef File, ArrayRef<StringRef> Paths) {
  for (StringRef Dir : Paths) {
    SmallString<128> Path = Dir;
    sys::path::append(Path, File);
    if (sys::fs::exists(Path))
      return Path.str().str();
  }
  return "";
}

static void fatalOpenError(llvm::Error E, Twine File) {
  if (!E)
    return;
  handleAllErrors(std::move(E), [&](const llvm::ErrorInfoBase &EIB) {
    llvm::errs() << "error opening '" << File << "': " << EIB.message() << '\n';
    exit(1);
  });
}

static void doList(opt::InputArgList& Args) {
  // lib.exe prints the contents of the first archive file.
  std::unique_ptr<MemoryBuffer> B;
  for (auto *Arg : Args.filtered(OPT_INPUT)) {
    // Create or open the archive object.
    ErrorOr<std::unique_ptr<MemoryBuffer>> MaybeBuf =
        MemoryBuffer::getFile(Arg->getValue(), -1, false);
    fatalOpenError(errorCodeToError(MaybeBuf.getError()), Arg->getValue());

    if (identify_magic(MaybeBuf.get()->getBuffer()) == file_magic::archive) {
      B = std::move(MaybeBuf.get());
      break;
    }
  }

  // lib.exe doesn't print an error if no .lib files are passed.
  if (!B)
    return;

  Error Err = Error::success();
  object::Archive Archive(B.get()->getMemBufferRef(), Err);
  fatalOpenError(std::move(Err), B->getBufferIdentifier());

  for (auto &C : Archive.children(Err)) {
    Expected<StringRef> NameOrErr = C.getName();
    fatalOpenError(NameOrErr.takeError(), B->getBufferIdentifier());
    StringRef Name = NameOrErr.get();
    llvm::outs() << Name << '\n';
  }
  fatalOpenError(std::move(Err), B->getBufferIdentifier());
}

static COFF::MachineTypes getCOFFFileMachine(MemoryBufferRef MB) {
  std::error_code EC;
  object::COFFObjectFile Obj(MB, EC);
  if (EC) {
    llvm::errs() << MB.getBufferIdentifier()
                 << ": failed to open: " << EC.message() << '\n';
    exit(1);
  }

  uint16_t Machine = Obj.getMachine();
  if (Machine != COFF::IMAGE_FILE_MACHINE_I386 &&
      Machine != COFF::IMAGE_FILE_MACHINE_AMD64 &&
      Machine != COFF::IMAGE_FILE_MACHINE_ARMNT &&
      Machine != COFF::IMAGE_FILE_MACHINE_ARM64) {
    llvm::errs() << MB.getBufferIdentifier() << ": unknown machine: " << Machine
                 << '\n';
    exit(1);
  }

  return static_cast<COFF::MachineTypes>(Machine);
}

static COFF::MachineTypes getBitcodeFileMachine(MemoryBufferRef MB) {
  Expected<std::string> TripleStr = getBitcodeTargetTriple(MB);
  if (!TripleStr) {
    llvm::errs() << MB.getBufferIdentifier()
                 << ": failed to get target triple from bitcode\n";
    exit(1);
  }

  switch (Triple(*TripleStr).getArch()) {
  case Triple::x86:
    return COFF::IMAGE_FILE_MACHINE_I386;
  case Triple::x86_64:
    return COFF::IMAGE_FILE_MACHINE_AMD64;
  case Triple::arm:
    return COFF::IMAGE_FILE_MACHINE_ARMNT;
  case Triple::aarch64:
    return COFF::IMAGE_FILE_MACHINE_ARM64;
  default:
    llvm::errs() << MB.getBufferIdentifier()
                 << ": unknown arch in target triple " << *TripleStr << '\n';
    exit(1);
  }
}

static void appendFile(std::vector<NewArchiveMember> &Members,
                       COFF::MachineTypes &LibMachine,
                       std::string &LibMachineSource, MemoryBufferRef MB) {
  file_magic Magic = identify_magic(MB.getBuffer());

  if (Magic != file_magic::coff_object && Magic != file_magic::bitcode &&
      Magic != file_magic::archive && Magic != file_magic::windows_resource) {
    llvm::errs() << MB.getBufferIdentifier()
                 << ": not a COFF object, bitcode, archive or resource file\n";
    exit(1);
  }

  // If a user attempts to add an archive to another archive, llvm-lib doesn't
  // handle the first archive file as a single file. Instead, it extracts all
  // members from the archive and add them to the second archive. This beahvior
  // is for compatibility with Microsoft's lib command.
  if (Magic == file_magic::archive) {
    Error Err = Error::success();
    object::Archive Archive(MB, Err);
    fatalOpenError(std::move(Err), MB.getBufferIdentifier());

    for (auto &C : Archive.children(Err)) {
      Expected<MemoryBufferRef> ChildMB = C.getMemoryBufferRef();
      if (!ChildMB) {
        handleAllErrors(ChildMB.takeError(), [&](const ErrorInfoBase &EIB) {
          llvm::errs() << MB.getBufferIdentifier() << ": " << EIB.message()
                       << "\n";
        });
        exit(1);
      }

      appendFile(Members, LibMachine, LibMachineSource, *ChildMB);
    }

    fatalOpenError(std::move(Err), MB.getBufferIdentifier());
    return;
  }

  // Check that all input files have the same machine type.
  // Mixing normal objects and LTO bitcode files is fine as long as they
  // have the same machine type.
  // Doing this here duplicates the header parsing work that writeArchive()
  // below does, but it's not a lot of work and it's a bit awkward to do
  // in writeArchive() which needs to support many tools, can't assume the
  // input is COFF, and doesn't have a good way to report errors.
  if (Magic == file_magic::coff_object || Magic == file_magic::bitcode) {
    COFF::MachineTypes FileMachine = (Magic == file_magic::coff_object)
                                         ? getCOFFFileMachine(MB)
                                         : getBitcodeFileMachine(MB);

    // FIXME: Once lld-link rejects multiple resource .obj files:
    // Call convertResToCOFF() on .res files and add the resulting
    // COFF file to the .lib output instead of adding the .res file, and remove
    // this check. See PR42180.
    if (FileMachine != COFF::IMAGE_FILE_MACHINE_UNKNOWN) {
      if (LibMachine == COFF::IMAGE_FILE_MACHINE_UNKNOWN) {
        LibMachine = FileMachine;
        LibMachineSource =
            (" (inferred from earlier file '" + MB.getBufferIdentifier() + "')")
                .str();
      } else if (LibMachine != FileMachine) {
        llvm::errs() << MB.getBufferIdentifier() << ": file machine type "
                     << machineToStr(FileMachine)
                     << " conflicts with library machine type "
                     << machineToStr(LibMachine) << LibMachineSource << '\n';
        exit(1);
      }
    }
  }

  Members.emplace_back(MB);
}

int llvm::libDriverMain(ArrayRef<const char *> ArgsArr) {
  BumpPtrAllocator Alloc;
  StringSaver Saver(Alloc);

  // Parse command line arguments.
  SmallVector<const char *, 20> NewArgs(ArgsArr.begin(), ArgsArr.end());
  cl::ExpandResponseFiles(Saver, cl::TokenizeWindowsCommandLine, NewArgs);
  ArgsArr = NewArgs;

  LibOptTable Table;
  unsigned MissingIndex;
  unsigned MissingCount;
  opt::InputArgList Args =
      Table.ParseArgs(ArgsArr.slice(1), MissingIndex, MissingCount);
  if (MissingCount) {
    llvm::errs() << "missing arg value for \""
                 << Args.getArgString(MissingIndex) << "\", expected "
                 << MissingCount
                 << (MissingCount == 1 ? " argument.\n" : " arguments.\n");
    return 1;
  }
  for (auto *Arg : Args.filtered(OPT_UNKNOWN))
    llvm::errs() << "ignoring unknown argument: " << Arg->getAsString(Args)
                 << "\n";

  // Handle /help
  if (Args.hasArg(OPT_help)) {
    Table.PrintHelp(outs(), "llvm-lib [options] file...", "LLVM Lib");
    return 0;
  }

  // If no input files, silently do nothing to match lib.exe.
  if (!Args.hasArgNoClaim(OPT_INPUT))
    return 0;

  if (Args.hasArg(OPT_lst)) {
    doList(Args);
    return 0;
  }

  std::vector<StringRef> SearchPaths = getSearchPaths(&Args, Saver);

  COFF::MachineTypes LibMachine = COFF::IMAGE_FILE_MACHINE_UNKNOWN;
  std::string LibMachineSource;
  if (auto *Arg = Args.getLastArg(OPT_machine)) {
    LibMachine = getMachineType(Arg->getValue());
    if (LibMachine == COFF::IMAGE_FILE_MACHINE_UNKNOWN) {
      llvm::errs() << "unknown /machine: arg " << Arg->getValue() << '\n';
      return 1;
    }
    LibMachineSource =
        std::string(" (from '/machine:") + Arg->getValue() + "' flag)";
  }

  std::vector<std::unique_ptr<MemoryBuffer>> MBs;
  StringSet<> Seen;
  std::vector<NewArchiveMember> Members;

  // Create a NewArchiveMember for each input file.
  for (auto *Arg : Args.filtered(OPT_INPUT)) {
    // Find a file
    std::string Path = findInputFile(Arg->getValue(), SearchPaths);
    if (Path.empty()) {
      llvm::errs() << Arg->getValue() << ": no such file or directory\n";
      return 1;
    }

    // Input files are uniquified by pathname. If you specify the exact same
    // path more than once, all but the first one are ignored.
    //
    // Note that there's a loophole in the rule; you can prepend `.\` or
    // something like that to a path to make it look different, and they are
    // handled as if they were different files. This behavior is compatible with
    // Microsoft lib.exe.
    if (!Seen.insert(Path).second)
      continue;

    // Open a file.
    ErrorOr<std::unique_ptr<MemoryBuffer>> MOrErr =
        MemoryBuffer::getFile(Path, -1, false);
    fatalOpenError(errorCodeToError(MOrErr.getError()), Path);
    MemoryBufferRef MBRef = (*MOrErr)->getMemBufferRef();

    // Append a file.
    appendFile(Members, LibMachine, LibMachineSource, MBRef);

    // Take the ownership of the file buffer to keep the file open.
    MBs.push_back(std::move(*MOrErr));
  }

  // Create an archive file.
  std::string OutputPath = getOutputPath(&Args, Members[0]);
  // llvm-lib uses relative paths for both regular and thin archives, unlike
  // standard GNU ar, which only uses relative paths for thin archives and
  // basenames for regular archives.
  for (NewArchiveMember &Member : Members) {
    if (sys::path::is_relative(Member.MemberName)) {
      Expected<std::string> PathOrErr =
          computeArchiveRelativePath(OutputPath, Member.MemberName);
      if (PathOrErr)
        Member.MemberName = Saver.save(*PathOrErr);
    }
  }

  if (Error E =
          writeArchive(OutputPath, Members,
                       /*WriteSymtab=*/true, object::Archive::K_GNU,
                       /*Deterministic*/ true, Args.hasArg(OPT_llvmlibthin))) {
    handleAllErrors(std::move(E), [&](const ErrorInfoBase &EI) {
      llvm::errs() << OutputPath << ": " << EI.message() << "\n";
    });
    return 1;
  }

  return 0;
}
