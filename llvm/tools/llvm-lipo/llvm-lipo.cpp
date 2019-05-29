//===-- llvm-lipo.cpp - a tool for manipulating universal binaries --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A utility for creating / splitting / inspecting universal binaries.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/Triple.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/WithColor.h"

using namespace llvm;
using namespace llvm::object;

static const StringRef ToolName = "llvm-lipo";

LLVM_ATTRIBUTE_NORETURN static void reportError(Twine Message) {
  WithColor::error(errs(), ToolName) << Message << "\n";
  errs().flush();
  exit(EXIT_FAILURE);
}

LLVM_ATTRIBUTE_NORETURN static void reportError(StringRef File, Error E) {
  assert(E);
  std::string Buf;
  raw_string_ostream OS(Buf);
  logAllUnhandledErrors(std::move(E), OS);
  OS.flush();
  WithColor::error(errs(), ToolName) << "'" << File << "': " << Buf;
  exit(EXIT_FAILURE);
}

namespace {
enum LipoID {
  LIPO_INVALID = 0, // This is not an option ID.
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  LIPO_##ID,
#include "LipoOpts.inc"
#undef OPTION
};

#define PREFIX(NAME, VALUE) const char *const LIPO_##NAME[] = VALUE;
#include "LipoOpts.inc"
#undef PREFIX

static const opt::OptTable::Info LipoInfoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  {LIPO_##PREFIX, NAME,      HELPTEXT,                                         \
   METAVAR,       LIPO_##ID, opt::Option::KIND##Class,                         \
   PARAM,         FLAGS,     LIPO_##GROUP,                                     \
   LIPO_##ALIAS,  ALIASARGS, VALUES},
#include "LipoOpts.inc"
#undef OPTION
};

class LipoOptTable : public opt::OptTable {
public:
  LipoOptTable() : OptTable(LipoInfoTable) {}
};

struct Config {
  SmallVector<std::string, 1> InputFiles;
  SmallVector<std::string, 1> VerifyArchList;
};

} // end namespace

static Config parseLipoOptions(ArrayRef<const char *> ArgsArr) {
  Config C;
  LipoOptTable T;
  unsigned MissingArgumentIndex, MissingArgumentCount;
  llvm::opt::InputArgList InputArgs =
      T.ParseArgs(ArgsArr, MissingArgumentIndex, MissingArgumentCount);

  if (InputArgs.size() == 0) {
    // PrintHelp does not accept Twine.
    T.PrintHelp(errs(), "llvm-lipo input[s] option[s]", "llvm-lipo");
    exit(EXIT_FAILURE);
  }

  if (InputArgs.hasArg(LIPO_help)) {
    // PrintHelp does not accept Twine.
    T.PrintHelp(outs(), "llvm-lipo input[s] option[s]", "llvm-lipo");
    exit(EXIT_SUCCESS);
  }

  if (InputArgs.hasArg(LIPO_version)) {
    outs() << ToolName + "\n";
    cl::PrintVersionMessage();
    exit(EXIT_SUCCESS);
  }

  for (auto Arg : InputArgs.filtered(LIPO_UNKNOWN))
    reportError("unknown argument '" + Arg->getAsString(InputArgs) + "'");

  for (auto Arg : InputArgs.filtered(LIPO_INPUT))
    C.InputFiles.push_back(Arg->getValue());
  if (C.InputFiles.empty())
    reportError("at least one input file should be specified");

  if (InputArgs.hasArg(LIPO_verify_arch)) {
    for (auto A : InputArgs.getAllArgValues(LIPO_verify_arch))
      C.VerifyArchList.push_back(A);
    if (C.VerifyArchList.empty())
      reportError(
          "verify_arch requires at least one architecture to be specified");
    if (C.InputFiles.size() > 1)
      reportError("verify_arch expects a single input file");
  }
  return C;
}

static SmallVector<OwningBinary<Binary>, 1>
readInputBinaries(ArrayRef<std::string> InputFiles) {
  SmallVector<OwningBinary<Binary>, 1> InputBinaries;
  for (StringRef InputFile : InputFiles) {
    Expected<OwningBinary<llvm::object::Binary>> BinaryOrErr =
        createBinary(InputFile);
    if (!BinaryOrErr)
      reportError(InputFile, BinaryOrErr.takeError());
    if (!isa<MachOObjectFile>(BinaryOrErr->getBinary()) &&
        !isa<MachOUniversalBinary>(BinaryOrErr->getBinary()))
      reportError("File " + InputFile + " has unsupported binary format");
    InputBinaries.push_back(std::move(*BinaryOrErr));
  }
  return InputBinaries;
}

LLVM_ATTRIBUTE_NORETURN
static void verifyArch(ArrayRef<OwningBinary<Binary>> InputBinaries,
                       ArrayRef<std::string> VerifyArchList) {
  assert(!InputBinaries.empty() &&
         "The list of input binaries should be non-empty");
  assert(!VerifyArchList.empty() &&
         "The list of architectures should be non-empty");
  assert(InputBinaries.size() == 1 && "Incorrect number of input binaries");

  for (StringRef Arch : VerifyArchList)
    if (Triple(Arch).getArch() == Triple::ArchType::UnknownArch)
      reportError("Invalid architecture: " + Arch);

  if (auto UO =
          dyn_cast<MachOUniversalBinary>(InputBinaries.front().getBinary())) {
    for (StringRef Arch : VerifyArchList) {
      Expected<std::unique_ptr<MachOObjectFile>> Obj =
          UO->getObjectForArch(Arch);
      if (!Obj)
        exit(EXIT_FAILURE);
    }
  } else if (auto O =
                 dyn_cast<MachOObjectFile>(InputBinaries.front().getBinary())) {
    const Triple::ArchType ObjectArch = O->getArch();
    for (StringRef Arch : VerifyArchList)
      if (ObjectArch != Triple(Arch).getArch())
        exit(EXIT_FAILURE);
  } else {
    llvm_unreachable("Unexpected binary format");
  }
  exit(EXIT_SUCCESS);
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  Config C = parseLipoOptions(makeArrayRef(argv + 1, argc));
  SmallVector<OwningBinary<Binary>, 1> InputBinaries =
      readInputBinaries(C.InputFiles);
  if (!C.VerifyArchList.empty())
    verifyArch(InputBinaries, C.VerifyArchList);
  return EXIT_SUCCESS;
}
