//===-- llvm-strings.cpp - Printable String dumping utility ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "llvm/Support/Format.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include <cctype>
#include <string>

using namespace llvm;
using namespace llvm::object;

static cl::list<std::string> InputFileNames(cl::Positional,
                                            cl::desc("<input object files>"),
                                            cl::ZeroOrMore);

static cl::opt<bool>
    PrintFileName("print-file-name",
                  cl::desc("Print the name of the file before each string"));
static cl::alias PrintFileNameShort("f", cl::desc(""),
                                    cl::aliasopt(PrintFileName));

static cl::opt<int>
    MinLength("bytes", cl::desc("Print sequences of the specified length"),
              cl::init(4));
static cl::alias MinLengthShort("n", cl::desc(""), cl::aliasopt(MinLength));

static cl::opt<bool>
    AllSections("all",
                  cl::desc("Check all sections, not just the data section"));
static cl::alias AllSectionsShort("a", cl::desc(""),
                                    cl::aliasopt(AllSections));

enum radix { none, octal, hexadecimal, decimal };
static cl::opt<radix>
    Radix("radix", cl::desc("print the offset within the file"),
          cl::values(clEnumValN(octal, "o", "octal"),
                     clEnumValN(hexadecimal, "x", "hexadecimal"),
                     clEnumValN(decimal, "d", "decimal")),
          cl::init(none));
static cl::alias RadixShort("t", cl::desc(""), cl::aliasopt(Radix));

static void strings(raw_ostream &OS, StringRef FileName, StringRef Contents) {
  auto print = [&OS, FileName](unsigned Offset, StringRef L) {
    if (L.size() < static_cast<size_t>(MinLength))
      return;
    if (PrintFileName)
      OS << FileName << ": ";
    switch (Radix) {
    case none:
      break;
    case octal:
      OS << format("%7o ", Offset);
      break;
    case hexadecimal:
      OS << format("%7x ", Offset);
      break;
    case decimal:
      OS << format("%7u ", Offset);
      break;
    }
    OS << L << '\n';
  };

  const char *B = Contents.begin();
  const char *P = nullptr, *E = nullptr, *S = nullptr;
  for (P = Contents.begin(), E = Contents.end(); P < E; ++P) {
    if (isPrint(*P) || *P == '\t') {
      if (S == nullptr)
        S = P;
    } else if (S) {
      print(S - B, StringRef(S, P - S));
      S = nullptr;
    }
  }
  if (S)
    print(S - B, StringRef(S, E - S));
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  cl::ParseCommandLineOptions(argc, argv, "llvm string dumper\n");
  if (MinLength == 0) {
    errs() << "invalid minimum string length 0\n";
    return EXIT_FAILURE;
  }

  if (InputFileNames.empty())
    InputFileNames.push_back("-");

  for (const auto &File : InputFileNames) {
    ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer =
        MemoryBuffer::getFileOrSTDIN(File);
    if (std::error_code EC = Buffer.getError())
      errs() << File << ": " << EC.message() << '\n';
    else
      strings(llvm::outs(), File == "-" ? "{standard input}" : File,
              Buffer.get()->getMemBufferRef().getBuffer());
  }

  return EXIT_SUCCESS;
}
