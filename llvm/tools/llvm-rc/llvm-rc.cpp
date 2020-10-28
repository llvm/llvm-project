//===-- llvm-rc.cpp - Compile .rc scripts into .res -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compile .rc scripts into .res files. This is intended to be a
// platform-independent port of Microsoft's rc.exe tool.
//
//===----------------------------------------------------------------------===//

#include "ResourceFileWriter.h"
#include "ResourceScriptCppFilter.h"
#include "ResourceScriptParser.h"
#include "ResourceScriptStmt.h"
#include "ResourceScriptToken.h"

#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <system_error>

using namespace llvm;
using namespace llvm::rc;

namespace {

// Input options tables.

enum ID {
  OPT_INVALID = 0, // This is not a correct option ID.
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
  {                                                                            \
      PREFIX,      NAME,      HELPTEXT,                                        \
      METAVAR,     OPT_##ID,  opt::Option::KIND##Class,                        \
      PARAM,       FLAGS,     OPT_##GROUP,                                     \
      OPT_##ALIAS, ALIASARGS, VALUES},
#include "Opts.inc"
#undef OPTION
};

class RcOptTable : public opt::OptTable {
public:
  RcOptTable() : OptTable(InfoTable, /* IgnoreCase = */ true) {}
};

static ExitOnError ExitOnErr;

LLVM_ATTRIBUTE_NORETURN static void fatalError(const Twine &Message) {
  errs() << Message << "\n";
  exit(1);
}

} // anonymous namespace

int main(int Argc, const char **Argv) {
  InitLLVM X(Argc, Argv);
  ExitOnErr.setBanner("llvm-rc: ");

  RcOptTable T;
  unsigned MAI, MAC;
  const char **DashDash = std::find_if(
      Argv + 1, Argv + Argc, [](StringRef Str) { return Str == "--"; });
  ArrayRef<const char *> ArgsArr = makeArrayRef(Argv + 1, DashDash);

  opt::InputArgList InputArgs = T.ParseArgs(ArgsArr, MAI, MAC);

  // The tool prints nothing when invoked with no command-line arguments.
  if (InputArgs.hasArg(OPT_help)) {
    T.PrintHelp(outs(), "rc [options] file...", "Resource Converter", false);
    return 0;
  }

  const bool BeVerbose = InputArgs.hasArg(OPT_verbose);

  std::vector<std::string> InArgsInfo = InputArgs.getAllArgValues(OPT_INPUT);
  if (DashDash != Argv + Argc)
    InArgsInfo.insert(InArgsInfo.end(), DashDash + 1, Argv + Argc);
  if (InArgsInfo.size() != 1) {
    fatalError("Exactly one input file should be provided.");
  }

  // Read and tokenize the input file.
  ErrorOr<std::unique_ptr<MemoryBuffer>> File =
      MemoryBuffer::getFile(InArgsInfo[0]);
  if (!File) {
    fatalError("Error opening file '" + Twine(InArgsInfo[0]) +
               "': " + File.getError().message());
  }

  std::unique_ptr<MemoryBuffer> FileContents = std::move(*File);
  StringRef Contents = FileContents->getBuffer();

  std::string FilteredContents = filterCppOutput(Contents);
  std::vector<RCToken> Tokens = ExitOnErr(tokenizeRC(FilteredContents));

  if (BeVerbose) {
    const Twine TokenNames[] = {
#define TOKEN(Name) #Name,
#define SHORT_TOKEN(Name, Ch) #Name,
#include "ResourceScriptTokenList.def"
    };

    for (const RCToken &Token : Tokens) {
      outs() << TokenNames[static_cast<int>(Token.kind())] << ": "
             << Token.value();
      if (Token.kind() == RCToken::Kind::Int)
        outs() << "; int value = " << Token.intValue();

      outs() << "\n";
    }
  }

  WriterParams Params;
  SmallString<128> InputFile(InArgsInfo[0]);
  llvm::sys::fs::make_absolute(InputFile);
  Params.InputFilePath = InputFile;
  Params.Include = InputArgs.getAllArgValues(OPT_includepath);
  Params.NoInclude = InputArgs.getAllArgValues(OPT_noinclude);

  if (InputArgs.hasArg(OPT_codepage)) {
    if (InputArgs.getLastArgValue(OPT_codepage)
            .getAsInteger(10, Params.CodePage))
      fatalError("Invalid code page: " +
                 InputArgs.getLastArgValue(OPT_codepage));
    switch (Params.CodePage) {
    case CpAcp:
    case CpWin1252:
    case CpUtf8:
      break;
    default:
      fatalError(
          "Unsupported code page, only 0, 1252 and 65001 are supported!");
    }
  }

  std::unique_ptr<ResourceFileWriter> Visitor;
  bool IsDryRun = InputArgs.hasArg(OPT_dry_run);

  if (!IsDryRun) {
    auto OutArgsInfo = InputArgs.getAllArgValues(OPT_fileout);
    if (OutArgsInfo.empty()) {
      SmallString<128> OutputFile = InputFile;
      llvm::sys::path::replace_extension(OutputFile, "res");
      OutArgsInfo.push_back(std::string(OutputFile.str()));
    }

    if (OutArgsInfo.size() != 1)
      fatalError(
          "No more than one output file should be provided (using /FO flag).");

    std::error_code EC;
    auto FOut = std::make_unique<raw_fd_ostream>(
        OutArgsInfo[0], EC, sys::fs::FA_Read | sys::fs::FA_Write);
    if (EC)
      fatalError("Error opening output file '" + OutArgsInfo[0] +
                 "': " + EC.message());
    Visitor = std::make_unique<ResourceFileWriter>(Params, std::move(FOut));
    Visitor->AppendNull = InputArgs.hasArg(OPT_add_null);

    ExitOnErr(NullResource().visit(Visitor.get()));

    // Set the default language; choose en-US arbitrarily.
    unsigned PrimaryLangId = 0x09, SubLangId = 0x01;
    if (InputArgs.hasArg(OPT_lang_id)) {
      unsigned LangId;
      if (InputArgs.getLastArgValue(OPT_lang_id).getAsInteger(16, LangId))
        fatalError("Invalid language id: " +
                   InputArgs.getLastArgValue(OPT_lang_id));
      PrimaryLangId = LangId & 0x3ff;
      SubLangId = LangId >> 10;
    }
    ExitOnErr(LanguageResource(PrimaryLangId, SubLangId).visit(Visitor.get()));
  }

  rc::RCParser Parser{std::move(Tokens)};
  while (!Parser.isEof()) {
    auto Resource = ExitOnErr(Parser.parseSingleResource());
    if (BeVerbose)
      Resource->log(outs());
    if (!IsDryRun)
      ExitOnErr(Resource->visit(Visitor.get()));
  }

  // STRINGTABLE resources come at the very end.
  if (!IsDryRun)
    ExitOnErr(Visitor->dumpAllStringTables());

  return 0;
}
