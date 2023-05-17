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

#include "Opts.inc"
#include "llvm/Object/Binary.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/WithColor.h"
#include <cctype>
#include <string>

using namespace llvm;
using namespace llvm::object;

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
  static constexpr StringLiteral NAME##_init[] = VALUE;                        \
  static constexpr ArrayRef<StringLiteral> NAME(NAME##_init,                   \
                                                std::size(NAME##_init) - 1);
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

class StringsOptTable : public opt::GenericOptTable {
public:
  StringsOptTable() : GenericOptTable(InfoTable) {
    setGroupedShortOptions(true);
  }
};
} // namespace

static StringRef ToolName;

static cl::list<std::string> InputFileNames(cl::Positional,
                                            cl::desc("<input object files>"));

static int MinLength = 4;
static bool PrintFileName;

enum radix { none, octal, hexadecimal, decimal };
static radix Radix;

[[noreturn]] static void reportCmdLineError(const Twine &Message) {
  WithColor::error(errs(), ToolName) << Message << "\n";
  exit(1);
}

template <typename T>
static void parseIntArg(const opt::InputArgList &Args, int ID, T &Value) {
  if (const opt::Arg *A = Args.getLastArg(ID)) {
    StringRef V(A->getValue());
    if (!llvm::to_integer(V, Value, 0) || Value <= 0)
      reportCmdLineError("expected a positive integer, but got '" + V + "'");
  }
}

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
  BumpPtrAllocator A;
  StringSaver Saver(A);
  StringsOptTable Tbl;
  ToolName = argv[0];
  opt::InputArgList Args =
      Tbl.parseArgs(argc, argv, OPT_UNKNOWN, Saver,
                    [&](StringRef Msg) { reportCmdLineError(Msg); });
  if (Args.hasArg(OPT_help)) {
    Tbl.printHelp(
        outs(),
        (Twine(ToolName) + " [options] <input object files>").str().c_str(),
        "llvm string dumper");
    // TODO Replace this with OptTable API once it adds extrahelp support.
    outs() << "\nPass @FILE as argument to read options from FILE.\n";
    return 0;
  }
  if (Args.hasArg(OPT_version)) {
    outs() << ToolName << '\n';
    cl::PrintVersionMessage();
    return 0;
  }

  parseIntArg(Args, OPT_bytes_EQ, MinLength);
  PrintFileName = Args.hasArg(OPT_print_file_name);
  StringRef R = Args.getLastArgValue(OPT_radix_EQ);
  if (R.empty())
    Radix = none;
  else if (R == "o")
    Radix = octal;
  else if (R == "d")
    Radix = decimal;
  else if (R == "x")
    Radix = hexadecimal;
  else
    reportCmdLineError("--radix value should be one of: '' (no offset), 'o' "
                       "(octal), 'd' (decimal), 'x' (hexadecimal)");

  if (MinLength == 0) {
    errs() << "invalid minimum string length 0\n";
    return EXIT_FAILURE;
  }

  std::vector<std::string> InputFileNames = Args.getAllArgValues(OPT_INPUT);
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
