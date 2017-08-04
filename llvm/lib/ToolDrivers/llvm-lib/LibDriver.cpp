//===- LibDriver.cpp - lib.exe-compatible driver --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Defines an interface to a lib.exe-compatible driver that also understands
// bitcode files. Used by llvm-lib and lld-link /lib.
//
//===----------------------------------------------------------------------===//

#include "llvm/ToolDrivers/llvm-lib/LibDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/ArchiveWriter.h"
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

static const llvm::opt::OptTable::Info infoTable[] = {
#define OPTION(X1, X2, ID, KIND, GROUP, ALIAS, X6, X7, X8, X9, X10, X11)       \
  {X1, X2, X9,          X10,         OPT_##ID, llvm::opt::Option::KIND##Class, \
   X8, X7, OPT_##GROUP, OPT_##ALIAS, X6,       X11},
#include "Options.inc"
#undef OPTION
};

class LibOptTable : public llvm::opt::OptTable {
public:
  LibOptTable() : OptTable(infoTable, true) {}
};

}

static std::string getOutputPath(llvm::opt::InputArgList *Args,
                                 const llvm::NewArchiveMember &FirstMember) {
  if (auto *Arg = Args->getLastArg(OPT_out))
    return Arg->getValue();
  SmallString<128> Val = StringRef(FirstMember.Buf->getBufferIdentifier());
  llvm::sys::path::replace_extension(Val, ".lib");
  return Val.str();
}

static std::vector<StringRef> getSearchPaths(llvm::opt::InputArgList *Args,
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

static Optional<std::string> findInputFile(StringRef File,
                                           ArrayRef<StringRef> Paths) {
  for (auto Dir : Paths) {
    SmallString<128> Path = Dir;
    sys::path::append(Path, File);
    if (sys::fs::exists(Path))
      return Path.str().str();
  }
  return Optional<std::string>();
}

int llvm::libDriverMain(llvm::ArrayRef<const char*> ArgsArr) {
  SmallVector<const char *, 20> NewArgs(ArgsArr.begin(), ArgsArr.end());
  BumpPtrAllocator Alloc;
  StringSaver Saver(Alloc);
  cl::ExpandResponseFiles(Saver, cl::TokenizeWindowsCommandLine, NewArgs);
  ArgsArr = NewArgs;

  LibOptTable Table;
  unsigned MissingIndex;
  unsigned MissingCount;
  llvm::opt::InputArgList Args =
      Table.ParseArgs(ArgsArr.slice(1), MissingIndex, MissingCount);
  if (MissingCount) {
    llvm::errs() << "missing arg value for \""
                 << Args.getArgString(MissingIndex) << "\", expected "
                 << MissingCount
                 << (MissingCount == 1 ? " argument.\n" : " arguments.\n");
    return 1;
  }
  for (auto *Arg : Args.filtered(OPT_UNKNOWN))
    llvm::errs() << "ignoring unknown argument: " << Arg->getSpelling() << "\n";

  if (!Args.hasArgNoClaim(OPT_INPUT)) {
    // No input files.  To match lib.exe, silently do nothing.
    return 0;
  }

  std::vector<StringRef> SearchPaths = getSearchPaths(&Args, Saver);

  std::vector<llvm::NewArchiveMember> Members;
  for (auto *Arg : Args.filtered(OPT_INPUT)) {
    Optional<std::string> Path = findInputFile(Arg->getValue(), SearchPaths);
    if (!Path.hasValue()) {
      llvm::errs() << Arg->getValue() << ": no such file or directory\n";
      return 1;
    }
    Expected<NewArchiveMember> MOrErr =
        NewArchiveMember::getFile(Saver.save(*Path), /*Deterministic=*/true);
    if (!MOrErr) {
      handleAllErrors(MOrErr.takeError(), [&](const llvm::ErrorInfoBase &EIB) {
        llvm::errs() << Arg->getValue() << ": " << EIB.message() << "\n";
      });
      return 1;
    }
    llvm::file_magic Magic = llvm::identify_magic(MOrErr->Buf->getBuffer());
    if (Magic != llvm::file_magic::coff_object &&
        Magic != llvm::file_magic::bitcode &&
        Magic != llvm::file_magic::windows_resource) {
      llvm::errs() << Arg->getValue()
                   << ": not a COFF object, bitcode or resource file\n";
      return 1;
    }
    Members.emplace_back(std::move(*MOrErr));
  }

  std::pair<StringRef, std::error_code> Result =
      llvm::writeArchive(getOutputPath(&Args, Members[0]), Members,
                         /*WriteSymtab=*/true, object::Archive::K_GNU,
                         /*Deterministic*/ true, Args.hasArg(OPT_llvmlibthin));

  if (Result.second) {
    if (Result.first.empty())
      Result.first = ArgsArr[0];
    llvm::errs() << Result.first << ": " << Result.second.message() << "\n";
    return 1;
  }

  return 0;
}
