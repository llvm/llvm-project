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

static constexpr int DefaultMinLength = 4;
static int MinLength = DefaultMinLength;
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

static bool isStringChar(char C) { return isPrint(C) || C == '\t'; }

static void strings(raw_ostream &OS, StringRef FileName,
                    sys::fs::file_t Handle) {
  SmallString<sys::fs::DefaultReadChunkSize> Buffer;
  auto printHeader = [&OS, FileName](size_t StringStart) {
    if (PrintFileName)
      OS << FileName << ": ";
    switch (Radix) {
    case none:
      break;
    case octal:
      OS << format("%7o ", StringStart);
      break;
    case hexadecimal:
      OS << format("%7x ", StringStart);
      break;
    case decimal:
      OS << format("%7u ", StringStart);
      break;
    }
  };

  // To handle very large files without consuming excessive memory, we read the
  // file in a little at a time and process it then rather than reading the
  // entire file at once.
  //
  // A string is only buffered until it is known to be long enough to print;
  // from then on it is streamed out directly, so an arbitrarily long string
  // never needs an arbitrarily large buffer. Candidate therefore only ever
  // holds a run that is shorter than MinLength and that was cut off by the end
  // of a chunk.
  const size_t Min = MinLength;
  SmallString<DefaultMinLength> Candidate;
  bool InString = false;
  // Offset of the start of the current chunk within the file.
  size_t ChunkOffset = 0;

  Buffer.resize_for_overwrite(sys::fs::DefaultReadChunkSize);

  // To prevent performance regression under O0, access the raw pointer instead
  // of using methods provided by the standard library, which are not inlined
  // under O0.
  while (true) {
    Expected<size_t> ReadBytesOrErr = sys::fs::readNativeFile(
        Handle, MutableArrayRef(Buffer.data(), Buffer.size()));
    if (!ReadBytesOrErr) {
      errs() << FileName << ": "
             << errorToErrorCode(ReadBytesOrErr.takeError()).message() << '\n';
      return;
    }
    size_t ChunkSize = *ReadBytesOrErr;
    if (ChunkSize == 0)
      break;

    const char *const Begin = Buffer.data();
    const char *const End = Begin + ChunkSize;
    const char *Cur = Begin;

    // Handle the remaining part from the previous chunk.
    // The previous chunk can be either no longer than MinSize or part of the
    // string.
    // Keep the buffer size bounded. With a small Min, a long string spanning
    // multiple chunks will have at most DefaultReadChunkSize bytes, since the
    // buffer is printed immediately with the header (guarded by the second if).
    // With a large Min, the buffer must hold at least Min bytes, since we need
    // enough data to decide whether to print it.
    if (InString || !Candidate.empty()) {
      while (Cur != End && isStringChar(*Cur))
        ++Cur;
      size_t Len = Cur - Begin;
      if (InString) {
        OS << StringRef(Begin, Len);
      } else if (Candidate.size() + Len >= Min) {
        printHeader(ChunkOffset - Candidate.size());
        OS << Candidate << StringRef(Begin, Len);
        Candidate.clear();
        InString = true;
      } else if (Cur == End) {
        Candidate.append(Begin, End);
      } else {
        Candidate.clear();
      }

      if (Cur == End) {
        ChunkOffset += ChunkSize;
        continue;
      }
      if (InString) {
        OS << '\n';
        InString = false;
      }
    }

    const char *S = nullptr;
    for (; Cur != End; ++Cur) {
      if (isStringChar(*Cur)) {
        if (!S)
          S = Cur;
      } else if (S) {
        if (static_cast<size_t>(Cur - S) >= Min) {
          printHeader(ChunkOffset + (S - Begin));
          OS << StringRef(S, Cur - S) << '\n';
        }
        S = nullptr;
      }
    }

    // Concatenate the last part.
    // If the current buffer is no longer than Min, we can't print it, so just
    // add it to the candidate buffer. If it is used, mark it as InString to
    // prevent printing the header twice.
    if (S) {
      size_t Len = End - S;
      if (Len >= Min) {
        printHeader(ChunkOffset + (S - Begin));
        OS << StringRef(S, Len);
        InString = true;
      } else {
        Candidate.append(S, End);
      }
    }
    ChunkOffset += ChunkSize;
  }

  if (InString)
    OS << '\n';
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
    if (File == "-") {
      strings(llvm::outs(), "{standard input}", sys::fs::getStdinHandle());
    } else {
      Expected<sys::fs::file_t> FDOrErr =
          sys::fs::openNativeFileForRead(File, sys::fs::OF_TextWithCRLF);
      if (!FDOrErr) {
        errs() << File
               << ": cannot open file: " << toString(FDOrErr.takeError())
               << '\n';
        continue;
      }
      strings(llvm::outs(), File, *FDOrErr);
    }
  }

  return EXIT_SUCCESS;
}
