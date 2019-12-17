//===-- llvm-c++filt.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <iostream>

using namespace llvm;

enum Style {
  Auto,  ///< auto-detect mangling
  GNU,   ///< GNU
  Lucid, ///< Lucid compiler (lcc)
  ARM,
  HP,    ///< HP compiler (xCC)
  EDG,   ///< EDG compiler
  GNUv3, ///< GNU C++ v3 ABI
  Java,  ///< Java (gcj)
  GNAT   ///< ADA compiler (gnat)
};
static cl::opt<Style>
    Format("format", cl::desc("decoration style"),
           cl::values(clEnumValN(Auto, "auto", "auto-detect style"),
                      clEnumValN(GNU, "gnu", "GNU (itanium) style")),
           cl::init(Auto));
static cl::alias FormatShort("s", cl::desc("alias for --format"),
                             cl::aliasopt(Format));

static cl::opt<bool> StripUnderscore("strip-underscore",
                                     cl::desc("strip the leading underscore"),
                                     cl::init(false));
static cl::alias StripUnderscoreShort("_",
                                      cl::desc("alias for --strip-underscore"),
                                      cl::aliasopt(StripUnderscore));
static cl::opt<bool>
    NoStripUnderscore("no-strip-underscore",
                      cl::desc("do not strip the leading underscore"),
                      cl::init(false));
static cl::alias
    NoStripUnderscoreShort("n", cl::desc("alias for --no-strip-underscore"),
                           cl::aliasopt(NoStripUnderscore));

static cl::opt<bool>
    Types("types",
          cl::desc("attempt to demangle types as well as function names"),
          cl::init(false));
static cl::alias TypesShort("t", cl::desc("alias for --types"),
                            cl::aliasopt(Types));

static cl::list<std::string>
Decorated(cl::Positional, cl::desc("<mangled>"), cl::ZeroOrMore);

static cl::extrahelp
    HelpResponse("\nPass @FILE as argument to read options from FILE.\n");

static bool shouldStripUnderscore() {
  if (StripUnderscore)
    return true;
  if (NoStripUnderscore)
    return false;
  // If none of them are set, use the default value for platform.
  // macho has symbols prefix with "_" so strip by default.
  return Triple(sys::getProcessTriple()).isOSBinFormatMachO();
}

static std::string demangle(const std::string &Mangled) {
  int Status;
  std::string Prefix;

  const char *DecoratedStr = Mangled.c_str();
  if (shouldStripUnderscore())
    if (DecoratedStr[0] == '_')
      ++DecoratedStr;
  size_t DecoratedLength = strlen(DecoratedStr);

  char *Undecorated = nullptr;

  if (Types ||
      ((DecoratedLength >= 2 && strncmp(DecoratedStr, "_Z", 2) == 0) ||
       (DecoratedLength >= 4 && strncmp(DecoratedStr, "___Z", 4) == 0)))
    Undecorated = itaniumDemangle(DecoratedStr, nullptr, nullptr, &Status);

  if (!Undecorated &&
      (DecoratedLength > 6 && strncmp(DecoratedStr, "__imp_", 6) == 0)) {
    Prefix = "import thunk for ";
    Undecorated = itaniumDemangle(DecoratedStr + 6, nullptr, nullptr, &Status);
  }

  std::string Result(Undecorated ? Prefix + Undecorated : Mangled);
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
  return isalnum(C) || C == '.' || C == '$' || C == '_';
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
      Result += ::demangle(Word.first) + Word.second.str();
  } else
    Result = ::demangle(Mangled);
  OS << Result << '\n';
  OS.flush();
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  cl::ParseCommandLineOptions(argc, argv, "llvm symbol undecoration tool\n");

  if (Decorated.empty())
    for (std::string Mangled; std::getline(std::cin, Mangled);)
      demangleLine(llvm::outs(), Mangled, true);
  else
    for (const auto &Symbol : Decorated)
      demangleLine(llvm::outs(), Symbol, false);

  return EXIT_SUCCESS;
}
