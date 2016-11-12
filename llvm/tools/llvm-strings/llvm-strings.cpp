//===-- llvm-strings.cpp - Printable String dumping utility ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This program is a utility that works like binutils "strings", that is, it
// prints out printable strings in a binary, objdump, or archive file.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/Binary.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include <cctype>
#include <string>

using namespace llvm;
using namespace llvm::object;

static cl::list<std::string> InputFileNames(cl::Positional,
                                            cl::desc("<input object files>"),
                                            cl::ZeroOrMore);

static void strings(raw_ostream &OS, StringRef Contents) {
  const char *P = nullptr, *E = nullptr, *S = nullptr;
  for (P = Contents.begin(), E = Contents.end(); P < E; ++P) {
    if (std::isgraph(*P) || std::isblank(*P)) {
      if (S == nullptr)
        S = P;
    } else if (S) {
      if (P - S > 3)
        OS << StringRef(S, P - S) << '\n';
      S = nullptr;
    }
  }
  if (S && E - S > 3)
    OS << StringRef(S, E - S) << '\n';
}

int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);

  cl::ParseCommandLineOptions(argc, argv, "llvm string dumper\n");

  if (InputFileNames.empty())
    InputFileNames.push_back("-");

  for (const auto &File : InputFileNames) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer =
        MemoryBuffer::getFileOrSTDIN(File);
    if (std::error_code EC = Buffer.getError())
      errs() << File << ": " << EC.message() << '\n';
    else
      strings(llvm::outs(), Buffer.get()->getMemBufferRef().getBuffer());
  }

  return EXIT_SUCCESS;
}
