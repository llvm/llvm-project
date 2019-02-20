//===-- llvm-c++filt.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringExtras.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/CommandLine.h"
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
    Types("types",
          cl::desc("attempt to demangle types as well as function names"),
          cl::init(false));
static cl::alias TypesShort("t", cl::desc("alias for --types"),
                            cl::aliasopt(Types));

static cl::list<std::string>
Decorated(cl::Positional, cl::desc("<mangled>"), cl::ZeroOrMore);

static std::string demangle(llvm::raw_ostream &OS, const std::string &Mangled) {
  int Status;

  const char *Decorated = Mangled.c_str();
  if (StripUnderscore)
    if (Decorated[0] == '_')
      ++Decorated;
  size_t DecoratedLength = strlen(Decorated);

  char *Undecorated = nullptr;

  if (Types || ((DecoratedLength >= 2 && strncmp(Decorated, "_Z", 2) == 0) ||
                (DecoratedLength >= 4 && strncmp(Decorated, "___Z", 4) == 0)))
    Undecorated = itaniumDemangle(Decorated, nullptr, nullptr, &Status);

  if (!Undecorated &&
      (DecoratedLength > 6 && strncmp(Decorated, "__imp_", 6) == 0)) {
    OS << "import thunk for ";
    Undecorated = itaniumDemangle(Decorated + 6, nullptr, nullptr, &Status);
  }

  std::string Result(Undecorated ? Undecorated : Mangled);
  free(Undecorated);
  return Result;
}

// If 'Split' is true, then 'Mangled' is broken into individual words and each
// word is demangled.  Otherwise, the entire string is treated as a single
// mangled item.  The result is output to 'OS'.
static void demangleLine(llvm::raw_ostream &OS, StringRef Mangled, bool Split) {
  std::string Result;
  if (Split) {
    SmallVector<StringRef, 16> Words;
    SplitString(Mangled, Words);
    for (auto Word : Words)
      Result += demangle(OS, Word) + ' ';
    // Remove the trailing space character.
    if (Result.back() == ' ')
      Result.pop_back();
  } else
    Result = demangle(OS, Mangled);
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
