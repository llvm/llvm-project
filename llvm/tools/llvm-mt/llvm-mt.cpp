//===- llvm-mt.cpp - Merge .manifest files ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
//
// Merge .manifest files.  This is intended to be a platform-independent port
// of Microsoft's mt.exe.
//
//===---------------------------------------------------------------------===//

#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/WindowsManifest/WindowsManifestMerger.h"

#include <system_error>

using namespace llvm;

namespace {

enum ID {
  OPT_INVALID = 0, // This is not an option ID.
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
  OPT_##ID,
#include "Opts.inc"
#undef OPTION
};

#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "Opts.inc"
#undef PREFIX

static const opt::OptTable::Info InfoTable[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, PARAM,  \
               HELPTEXT, METAVAR, VALUES)                                      \
{                                                                              \
      PREFIX,      NAME,      HELPTEXT,                                        \
      METAVAR,     OPT_##ID,  opt::Option::KIND##Class,                        \
      PARAM,       FLAGS,     OPT_##GROUP,                                     \
      OPT_##ALIAS, ALIASARGS, VALUES},
#include "Opts.inc"
#undef OPTION
};

class CvtResOptTable : public opt::OptTable {
public:
  CvtResOptTable() : OptTable(InfoTable, true) {}
};
} // namespace

LLVM_ATTRIBUTE_NORETURN void reportError(Twine Msg) {
  errs() << "llvm-mt error: " << Msg << "\n";
  exit(1);
}

static void reportError(StringRef Input, std::error_code EC) {
  reportError(Twine(Input) + ": " + EC.message());
}

void error(std::error_code EC) {
  if (EC)
    reportError(EC.message());
}

void error(Error EC) {
  if (EC)
    handleAllErrors(std::move(EC), [&](const ErrorInfoBase &EI) {
      reportError(EI.message());
    });
}

int main(int Argc, const char **Argv) {
  InitLLVM X(Argc, Argv);

  CvtResOptTable T;
  unsigned MAI, MAC;
  ArrayRef<const char *> ArgsArr = makeArrayRef(Argv + 1, Argc - 1);
  opt::InputArgList InputArgs = T.ParseArgs(ArgsArr, MAI, MAC);

  for (auto *Arg : InputArgs.filtered(OPT_INPUT)) {
    auto ArgString = Arg->getAsString(InputArgs);
    std::string Diag;
    raw_string_ostream OS(Diag);
    OS << "invalid option '" << ArgString << "'";

    std::string Nearest;
    if (T.findNearest(ArgString, Nearest) < 2)
      OS << ", did you mean '" << Nearest << "'?";

    reportError(OS.str());
  }

  for (auto &Arg : InputArgs) {
    if (Arg->getOption().matches(OPT_unsupported)) {
      outs() << "llvm-mt: ignoring unsupported '" << Arg->getOption().getName()
             << "' option\n";
    }
  }

  if (InputArgs.hasArg(OPT_help)) {
    T.PrintHelp(outs(), "mt", "Manifest Tool", false);
    return 0;
  }

  std::vector<std::string> InputFiles = InputArgs.getAllArgValues(OPT_manifest);

  if (InputFiles.size() == 0) {
    reportError("no input file specified");
  }

  StringRef OutputFile;
  if (InputArgs.hasArg(OPT_out)) {
    OutputFile = InputArgs.getLastArgValue(OPT_out);
  } else if (InputFiles.size() == 1) {
    OutputFile = InputFiles[0];
  } else {
    reportError("no output file specified");
  }

  windows_manifest::WindowsManifestMerger Merger;

  for (const auto &File : InputFiles) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> ManifestOrErr =
        MemoryBuffer::getFile(File);
    if (!ManifestOrErr)
      reportError(File, ManifestOrErr.getError());
    MemoryBuffer &Manifest = *ManifestOrErr.get();
    error(Merger.merge(Manifest));
  }

  std::unique_ptr<MemoryBuffer> OutputBuffer = Merger.getMergedManifest();
  if (!OutputBuffer)
    reportError("empty manifest not written");
  Expected<std::unique_ptr<FileOutputBuffer>> FileOrErr =
      FileOutputBuffer::create(OutputFile, OutputBuffer->getBufferSize());
  if (!FileOrErr)
    reportError(OutputFile, errorToErrorCode(FileOrErr.takeError()));
  std::unique_ptr<FileOutputBuffer> FileBuffer = std::move(*FileOrErr);
  std::copy(OutputBuffer->getBufferStart(), OutputBuffer->getBufferEnd(),
            FileBuffer->getBufferStart());
  error(FileBuffer->commit());
  return 0;
}
