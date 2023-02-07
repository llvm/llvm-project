//===-- llvm-c++filt.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringExtras.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include <cstdlib>
#include <iostream>

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

#define PREFIX(NAME, VALUE)                                                    \
  static constexpr llvm::StringLiteral NAME##_init[] = VALUE;                  \
  static constexpr llvm::ArrayRef<llvm::StringLiteral> NAME(                   \
      NAME##_init, std::size(NAME##_init) - 1);
#include "Opts.inc"
#undef PREFIX

static constexpr opt::OptTable::Info InfoTable[] = {
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

class CxxfiltOptTable : public opt::GenericOptTable {
public:
  CxxfiltOptTable() : opt::GenericOptTable(InfoTable) {
    setGroupedShortOptions(true);
  }
};
} // namespace

static bool StripUnderscore;
static bool Types;

static StringRef ToolName;

static void error(const Twine &Message) {
  WithColor::error(errs(), ToolName) << Message << '\n';
  exit(1);
}

static std::string demangle(const std::string &Mangled) {
  const char *DecoratedStr = Mangled.c_str();
  if (StripUnderscore)
    if (DecoratedStr[0] == '_')
      ++DecoratedStr;

  std::string Result;
  if (nonMicrosoftDemangle(DecoratedStr, Result))
    return Result;

  std::string Prefix;
  char *Undecorated = nullptr;

  if (Types)
    Undecorated = itaniumDemangle(DecoratedStr, nullptr, nullptr, nullptr);

  if (!Undecorated && strncmp(DecoratedStr, "__imp_", 6) == 0) {
    Prefix = "import thunk for ";
    Undecorated = itaniumDemangle(DecoratedStr + 6, nullptr, nullptr, nullptr);
  }

  Result = Undecorated ? Prefix + Undecorated : Mangled;
  free(Undecorated);
  return Result;
}

// Split 'Source' on any character that fails to pass 'IsLegalChar'.  The
// returned vector consists of pairs where 'first' is the delimited word, and
// 'second' are the delimiters following that word.
static void SplitStringDelims(
    StringRef Source,
    SmallVectorImpl<std::pair<StringRef, StringRef>> &OutFragments,
    function_ref<bool(char)> IsLegalChar) {
  // The beginning of the input string.
  const auto Head = Source.begin();

  // Obtain any leading delimiters.
  auto Start = std::find_if(Head, Source.end(), IsLegalChar);
  if (Start != Head)
    OutFragments.push_back({"", Source.slice(0, Start - Head)});

  // Capture each word and the delimiters following that word.
  while (Start != Source.end()) {
    Start = std::find_if(Start, Source.end(), IsLegalChar);
    auto End = std::find_if_not(Start, Source.end(), IsLegalChar);
    auto DEnd = std::find_if(End, Source.end(), IsLegalChar);
    OutFragments.push_back({Source.slice(Start - Head, End - Head),
                            Source.slice(End - Head, DEnd - Head)});
    Start = DEnd;
  }
}

// This returns true if 'C' is a character that can show up in an
// Itanium-mangled string.
static bool IsLegalItaniumChar(char C) {
  // Itanium CXX ABI [External Names]p5.1.1:
  // '$' and '.' in mangled names are reserved for private implementations.
  return isAlnum(C) || C == '.' || C == '$' || C == '_';
}

// If 'Split' is true, then 'Mangled' is broken into individual words and each
// word is demangled.  Otherwise, the entire string is treated as a single
// mangled item.  The result is output to 'OS'.
static void demangleLine(llvm::raw_ostream &OS, StringRef Mangled, bool Split) {
  std::string Result;
  if (Split) {
    SmallVector<std::pair<StringRef, StringRef>, 16> Words;
    SplitStringDelims(Mangled, Words, IsLegalItaniumChar);
    for (const auto &Word : Words)
      Result += ::demangle(std::string(Word.first)) + Word.second.str();
  } else
    Result = ::demangle(std::string(Mangled));
  OS << Result << '\n';
  OS.flush();
}

int llvm_cxxfilt_main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  BumpPtrAllocator A;
  StringSaver Saver(A);
  CxxfiltOptTable Tbl;
  ToolName = argv[0];
  opt::InputArgList Args = Tbl.parseArgs(argc, argv, OPT_UNKNOWN, Saver,
                                         [&](StringRef Msg) { error(Msg); });
  if (Args.hasArg(OPT_help)) {
    Tbl.printHelp(outs(),
                  (Twine(ToolName) + " [options] <mangled>").str().c_str(),
                  "LLVM symbol undecoration tool");
    // TODO Replace this with OptTable API once it adds extrahelp support.
    outs() << "\nPass @FILE as argument to read options from FILE.\n";
    return 0;
  }
  if (Args.hasArg(OPT_version)) {
    outs() << ToolName << '\n';
    cl::PrintVersionMessage();
    return 0;
  }

  // The default value depends on the default triple. Mach-O has symbols
  // prefixed with "_", so strip by default.
  if (opt::Arg *A =
          Args.getLastArg(OPT_strip_underscore, OPT_no_strip_underscore))
    StripUnderscore = A->getOption().matches(OPT_strip_underscore);
  else
    StripUnderscore = Triple(sys::getProcessTriple()).isOSBinFormatMachO();

  Types = Args.hasArg(OPT_types);

  std::vector<std::string> Decorated = Args.getAllArgValues(OPT_INPUT);
  if (Decorated.empty())
    for (std::string Mangled; std::getline(std::cin, Mangled);)
      demangleLine(llvm::outs(), Mangled, true);
  else
    for (const auto &Symbol : Decorated)
      demangleLine(llvm::outs(), Symbol, false);

  return EXIT_SUCCESS;
}
