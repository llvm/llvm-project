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
#include "llvm/ADT/StringExtras.h"
#include "llvm/Object/Binary.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Errno.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
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
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "Opts.inc"
#undef OPTION
};

#define OPTTABLE_STR_TABLE_CODE
#include "Opts.inc"
#undef OPTTABLE_STR_TABLE_CODE

#define OPTTABLE_PREFIXES_TABLE_CODE
#include "Opts.inc"
#undef OPTTABLE_PREFIXES_TABLE_CODE

using namespace llvm::opt;
static constexpr opt::OptTable::Info InfoTable[] = {
#define OPTION(...) LLVM_CONSTRUCT_OPT_INFO(__VA_ARGS__),
#include "Opts.inc"
#undef OPTION
};

class StringsOptTable : public opt::GenericOptTable {
public:
  StringsOptTable()
      : GenericOptTable(OptionStrTable, OptionPrefixesTable, InfoTable) {
    setGroupedShortOptions(true);
    setDashDashParsing(true);
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

static void strings(raw_ostream &OS, StringRef FileName,
                    sys::fs::file_t FileHandle) {
  SmallString<sys::fs::DefaultReadChunkSize> Buffer;
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

  unsigned Offset = 0, LocalOffset = 0, CurSize = 0;
  Buffer.resize(sys::fs::DefaultReadChunkSize);
  auto FillBuffer = [&Buffer, FileHandle, &FileName]() -> unsigned {
    Expected<size_t> ReadBytesOrErr = sys::fs::readNativeFile(
        FileHandle,
        MutableArrayRef(Buffer.begin(), sys::fs::DefaultReadChunkSize));
    if (!ReadBytesOrErr) {
      errs() << FileName << ": "
             << errorToErrorCode(ReadBytesOrErr.takeError()).message() << '\n';
      return 0;
    }
    return *ReadBytesOrErr;
  };
  std::string StringBuffer;
  while (true) {
    if (LocalOffset == CurSize) {
      CurSize = FillBuffer();
      if (CurSize == 0)
        break;
      LocalOffset = 0;
    }
    char C = Buffer[LocalOffset++];
    if (isPrint(C) || C == '\t') {
      StringBuffer.push_back(C);
    } else if (StringBuffer.size()) {
      print(Offset, StringRef(StringBuffer.c_str(), StringBuffer.size()));
      Offset += StringBuffer.size();
      StringBuffer.clear();
      ++Offset;
    } else {
      ++Offset;
    }
  }

  if (StringBuffer.size())
    print(Offset, StringRef(StringBuffer.c_str(), StringBuffer.size()));
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
    sys::fs::file_t FD;
    if (File == "-") {
      FD = sys::fs::getStdinHandle();
    } else {
      Expected<sys::fs::file_t> FDOrErr =
          sys::fs::openNativeFileForRead(File, sys::fs::OF_None);
      ;
      if (!FDOrErr) {
        errs() << File << ": "
               << errorToErrorCode(FDOrErr.takeError()).message() << '\n';
        continue;
      }
      FD = *FDOrErr;
    }

    strings(llvm::outs(), File == "-" ? "{standard input}" : File, FD);
  }

  return EXIT_SUCCESS;
}
