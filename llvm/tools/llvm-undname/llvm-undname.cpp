//===-- llvm-undname.cpp - Microsoft ABI name undecorator
//------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This utility works like the windows undname utility. It converts mangled
// Microsoft symbol names into pretty C/C++ human-readable names.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

using namespace llvm;

cl::opt<bool> DumpBackReferences("backrefs", cl::Optional,
                                 cl::desc("dump backreferences"), cl::Hidden,
                                 cl::init(false));
cl::list<std::string> Symbols(cl::Positional, cl::desc("<input symbols>"),
                              cl::ZeroOrMore);

static void demangle(const std::string &S) {
  int Status;
  MSDemangleFlags Flags = MSDF_None;
  if (DumpBackReferences)
    Flags = MSDemangleFlags(Flags | MSDF_DumpBackrefs);

  char *ResultBuf =
      microsoftDemangle(S.c_str(), nullptr, nullptr, &Status, Flags);
  if (Status == llvm::demangle_success) {
    outs() << ResultBuf << "\n";
    outs().flush();
  } else {
    WithColor::error() << "Invalid mangled name\n";
  }
  std::free(ResultBuf);
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  cl::ParseCommandLineOptions(argc, argv, "llvm-undname\n");

  if (Symbols.empty()) {
    while (true) {
      std::string LineStr;
      std::getline(std::cin, LineStr);
      if (std::cin.eof())
        break;

      StringRef Line(LineStr);
      Line = Line.trim();
      if (Line.empty() || Line.startswith("#") || Line.startswith(";"))
        continue;

      // If the user is manually typing in these decorated names, don't echo
      // them to the terminal a second time.  If they're coming from redirected
      // input, however, then we should display the input line so that the
      // mangled and demangled name can be easily correlated in the output.
      if (!sys::Process::StandardInIsUserInput()) {
        outs() << Line << "\n";
        outs().flush();
      }
      demangle(Line);
      outs() << "\n";
    }
  } else {
    for (StringRef S : Symbols) {
      outs() << S << "\n";
      outs().flush();
      demangle(S);
      outs() << "\n";
    }
  }

  return 0;
}
