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
    // The previous chunk can be either shorter than MinSize or part of the
    // string.
    // Keep the buffer size bounded. With a small Min, a long string spanning
    // multiple chunks will have at most DefaultReadChunkSize bytes, since the
    // buffer is printed immediately with the header (guarded by the second if).
    // With a large Min, the buffer must hold at least Min bytes, since we need
    // enough data to decide whether to print it.
    if (InString || !Candidate.empty()) {
      // Find the end of the current buffer
      while (Cur != End && isStringChar(*Cur))
        ++Cur;
      size_t Len = Cur - Begin;
      if (InString) {
        // Print the remaining part if the previous chunk has already printed
        // the header. E.g. header: aaaaa | bbbbb, where | is the chunk
        // boundary.                        ^
        OS << StringRef(Begin, Len);
      } else if (Candidate.size() + Len >= Min) {
        // If the header hasn't been printed yet (e.g. the previous candidate
        // was smaller than Min), but we can print it now, print the header
        // first, followed by the candidate from the previous chunk and the
        // current string. E.g. '\0' | bbbbbb
        //                             ^
        printHeader(ChunkOffset - Candidate.size());
        OS << Candidate << StringRef(Begin, Len);
        Candidate.clear();
        InString = true;
      } else if (Cur == End) {
        // If the current chunk + previous candidate is still smaller than Min ,
        // append it to Candidate
        Candidate.append(Begin, End);
      } else {
        // If the string has terminated but is still smaller than Min, clear the
        // buffer since it is too short to print.
        Candidate.clear();
      }

      if (Cur == End) {
        // Finish handling the current chunk and update ChunkOffset.
        ChunkOffset += ChunkSize;
        continue;
      }
      if (InString) {
        // We haven't reached the end of the chunk, which means the string is
        // terminated. Add a '\n' to start printing a new string.
        OS << '\n';
        InString = false;
      }
    }

    // At this point, we are always at the start of a new string because the
    // remaining part of the previous string has already been handled.
    const char *StrHead = nullptr;
    for (; Cur != End; ++Cur) {
      if (isStringChar(*Cur)) {
        // Find the start of the next string
        if (!StrHead)
          StrHead = Cur;
      } else if (StrHead) {
        // If it is not a StringChar, we have reached the end of the current
        // string. Try to print it.
        if (static_cast<size_t>(Cur - StrHead) >= Min) {
          printHeader(ChunkOffset + (StrHead - Begin));
          OS << StringRef(StrHead, Cur - StrHead) << '\n';
        }
        StrHead = nullptr;
      }
    }

    // The last string spans multiple chunks. If it is larger than Min, print
    // the header immediately and set the InString flag to avoid printing it
    // again.
    // e.g. aaaaa | bbbbb
    //          ^
    if (StrHead) {
      size_t Len = End - StrHead;
      // Print it, or append it to Candidate if it is too short.
      if (Len >= Min) {
        printHeader(ChunkOffset + (StrHead - Begin));
        OS << StringRef(StrHead, Len);
        InString = true;
      } else {
        Candidate.append(StrHead, End);
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
