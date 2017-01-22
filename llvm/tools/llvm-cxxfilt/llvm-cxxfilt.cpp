//===-- llvm-c++filt.cpp --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
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
  GNAT   ///< ADA copiler (gnat)
};
static cl::opt<Style>
    Format("format", cl::desc("decoration style"),
           cl::values(clEnumValN(Auto, "auto", "auto-detect style"),
                      clEnumValN(GNU, "gnu", "GNU (itanium) style")),
           cl::init(Auto));
static cl::alias FormatShort("s", cl::desc("alias for --format"),
                             cl::aliasopt(Format));

static cl::opt<bool>
    Types("types",
          cl::desc("attempt to demangle types as well as function names"),
          cl::init(false));
static cl::alias TypesShort("t", cl::desc("alias for --types"),
                            cl::aliasopt(Types));

static cl::list<std::string>
Decorated(cl::Positional, cl::desc("<mangled>"), cl::ZeroOrMore);

static void demangle(llvm::raw_ostream &OS, const std::string &Mangled) {
  int Status;
  char *Demangled = nullptr;
  if (Types || ((Mangled.size() >= 2 && Mangled.compare(0, 2, "_Z")) ||
                (Mangled.size() >= 4 && Mangled.compare(0, 4, "___Z"))))
    Demangled = itaniumDemangle(Mangled.c_str(), nullptr, nullptr, &Status);
  OS << (Demangled ? Demangled : Mangled) << '\n';
  free(Demangled);
}

int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);

  cl::ParseCommandLineOptions(argc, argv, "llvm symbol undecoration tool\n");

  if (Decorated.empty())
    for (std::string Mangled; std::getline(std::cin, Mangled);)
      demangle(llvm::outs(), Mangled);
  else
    for (const auto &Symbol : Decorated)
      demangle(llvm::outs(), Symbol);

  return EXIT_SUCCESS;
}
