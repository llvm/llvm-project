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
#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SwapByteOrder.h"
#include "llvm/Support/Unicode.h"
#include "llvm/Support/WithColor.h"
#include <cctype>
#include <locale>
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

using namespace llvm::opt;
#define OPTTABLE_CODE
#include "Opts.inc"

class StringsOptTable : public opt::OptTable {
public:
  StringsOptTable() : OptTable(optionTables()) {
    setGroupedShortOptions(true);
    setDashDashParsing(true);
  }
};

static StringRef ToolName;

static cl::list<std::string> InputFileNames(cl::Positional,
                                            cl::desc("<input object files>"));

static constexpr int DefaultMinLength = 4;
static int MinLength = DefaultMinLength;
static bool PrintFileName;

enum class Encoding { Ascii, Locale, Utf8 };
static Encoding Encoding;

enum class Radix { None, Octal, Hexadecimal, Decimal };
static Radix Radix;
} // namespace

[[noreturn]] static void reportCmdLineError(const Twine &Message) {
  WithColor::error(errs(), ToolName) << Message << "\n";
  exit(1);
}

[[noreturn]] static void invalidArgValue(Arg *Arg) {
  reportCmdLineError("'" + StringRef(Arg->getValue()) +
                     "' is not a valid value for '" + Arg->getSpelling() + "'");
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
                    sys::fs::file_t Handle) {
  SmallString<sys::fs::DefaultReadChunkSize> Buffer;
  auto PrintHeader = [&OS, FileName](size_t StringStart) {
    if (PrintFileName)
      OS << FileName << ": ";
    switch (Radix) {
    case Radix::None:
      break;
    case Radix::Octal:
      OS << format("%7o ", StringStart);
      break;
    case Radix::Hexadecimal:
      OS << format("%7x ", StringStart);
      break;
    case Radix::Decimal:
      OS << format("%7u ", StringStart);
      break;
    }
  };

  std::locale Loc("");
  auto &Cvt = std::use_facet<std::codecvt<wchar_t, char, std::mbstate_t>>(Loc);
  auto &Ctype = std::use_facet<std::ctype<wchar_t>>(Loc);

  auto IsStringChar = [&Ctype](UTF32 Ch) {
    if (Ch == '\t')
      return true;

    switch (Encoding) {
    case Encoding::Ascii:
      return isPrint(Ch);

    case Encoding::Locale:
      return Ctype.is(std::ctype_base::print, Ch);

    case Encoding::Utf8:
      return sys::unicode::isPrintable(Ch);
    }

    llvm_unreachable("unhandled encoding");
  };

  // Tries to read one character from the bytes P...E and store it in Ch.
  // Returns true if a (possibly invalid) character was read, false otherwise.
  //
  // If P...E starts with a valid complete character, P and MBState are updated
  // and true is returned.
  // If P...E is empty, or holds an incomplete character and AtEOF is false, P
  // and MBState are unchanged, Ch is set to 0, and false is returned. This is
  // intended to allow more bytes to be read and Read to be called again.
  // If P...E holds an incomplete character and AtEOF is true, or if P...E
  // starts with an invalid character, the first byte is skipped and MBState is
  // reset to allow continuing from the next point, Ch is set to 0, and true is
  // returned.
  auto Read = [&Cvt](const char *&P, const char *E, std::mbstate_t &MBState,
                     bool AtEOF, UTF32 &Ch) -> bool {
    if (P == E)
      return false;

    switch (Encoding) {
    case Encoding::Ascii:
      Ch = *P++;
      return true;

    case Encoding::Locale: {
      const char *N;
      wchar_t WCh;
      wchar_t *WNext;
      std::mbstate_t SaveMBState = MBState;
      const auto Res = Cvt.in(MBState, P, E, N, &WCh, &WCh + 1, WNext);
      assert(Res != std::codecvt_base::noconv);

      if (WNext != &WCh) {
        // Only treat a non-null character as a successful conversion, as a
        // null character may be the result of an incomplete multibyte
        // character followed by a null byte.
        if (WCh) {
          // Note: this assumes wchar_t is UCS2 or UTF32.
          Ch = WCh;
          P = N;
          return true;
        }
        // Otherwise treat it as an error. A null byte is safe to treat as an
        // error, as a null byte is never printable in any locale.
      } else if ((Res == std::codecvt_base::ok ||
                  Res == std::codecvt_base::partial) &&
                 !AtEOF) {
        // If we got a partial result but no character was written, we have an
        // incomplete multibyte character.  Do not treat this as an error,
        // instead reset the conversion state so that we can try again if/when
        // we have more characters, unless we know there are no more characters.
        Ch = 0;
        MBState = SaveMBState;
        return false;
      }
      // If there was any error, reset the state to allow the next byte to
      // start a character.
      Ch = 0;
      MBState = {};
      ++P;
      return true;
    }

    case Encoding::Utf8: {
      const UTF8 *UP = reinterpret_cast<const UTF8 *>(P);
      const UTF8 *UE = reinterpret_cast<const UTF8 *>(E);
      UTF32 *Next = &Ch;
      const auto Res =
          ConvertUTF8toUTF32Partial(&UP, UE, &Next, &Ch + 1, strictConversion);
      if (Next != &Ch) {
        if (Ch) {
          P = reinterpret_cast<const char *>(UP);
          return true;
        }
      } else if (Res == sourceExhausted && !AtEOF) {
        Ch = 0;
        return false;
      }
      Ch = 0;
      ++P;
      return true;
    }
    }

    llvm_unreachable("unhandled encoding");
  };

  auto EndString = [&OS, &Cvt](std::mbstate_t &MBState) {
    if (Encoding == Encoding::Locale) {
      char Buf[MB_LEN_MAX];
      char *End;
      auto Res = Cvt.unshift(MBState, Buf, &Buf[MB_LEN_MAX], End);
      if (Res == std::codecvt_base::ok) {
        // Note: This is only required for stateful encodings such as the
        // ISO-2022 ones.
        OS << StringRef(Buf, End - Buf);
      }
    };
    OS << '\n';
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
  size_t CandidateLength = 0;
  bool InString = false;
  std::mbstate_t MBState{};
  // Offset of the start of the current chunk within the file.
  size_t ChunkOffset = 0;

  while (true) {
    // Size of any partial character left over from the previous chunk.
    size_t PartialCharSize = Buffer.size();
    Buffer.resize_for_overwrite(PartialCharSize +
                                sys::fs::DefaultReadChunkSize);

    Expected<size_t> ReadBytesOrErr = sys::fs::readNativeFile(
        Handle,
        MutableArrayRef(Buffer.begin() + PartialCharSize, Buffer.end()));
    if (!ReadBytesOrErr) {
      errs() << FileName << ": "
             << errorToErrorCode(ReadBytesOrErr.takeError()).message() << '\n';
      return;
    }
    const bool AtEOF = *ReadBytesOrErr == 0;
    const size_t ChunkSize = PartialCharSize + *ReadBytesOrErr;
    if (ChunkSize == 0)
      break;

    Buffer.resize_for_overwrite(ChunkSize);

    // To prevent performance regression under O0, access the raw pointer
    // instead of using methods provided by the standard library, which are not
    // inlined under O0.
    const char *Begin = Buffer.data();
    const char *End = Begin + ChunkSize;
    const char *Cur = Begin;

    UTF32 Ch;

    // Handle the remaining part from the previous chunk.
    // The previous chunk can be either shorter than MinSize or part of the
    // string.
    // Keep the buffer size bounded. With a small Min, a long string spanning
    // multiple chunks will have at most DefaultReadChunkSize bytes, since the
    // buffer is printed immediately with the header (guarded by the second if).
    // With a large Min, the buffer must hold at least Min bytes, since we need
    // enough data to decide whether to print it.
    if (InString || !Candidate.empty()) {
      const char *StringEnd;
      std::mbstate_t PrevMBState;
      size_t Len = 0;
      bool EndOfChunk;

      // Find the end of the current string.
      for (;;) {
        StringEnd = Cur;
        PrevMBState = MBState;
        EndOfChunk = !Read(Cur, End, MBState, AtEOF, Ch);
        if (EndOfChunk || !IsStringChar(Ch))
          break;
        ++Len;
      }

      size_t Size = StringEnd - Begin;
      if (InString) {
        // Print the remaining part if the previous chunk has already printed
        // the header. E.g. header: aaaaa | bbbbb, where | is the chunk
        // boundary.
        // Output: Header: aaaaabbbbb, where bbbbb is printed in here.
        OS << StringRef(Begin, Size);
      } else if (CandidateLength + Len >= Min) {
        // If the header hasn't been printed yet (i.e. the previous candidate
        // was smaller than Min), but we can print it now, print the header
        // first, followed by the candidate from the previous chunk and the
        // current string. E.g. aa | bbbbbb
        // Output Header: aabbbbbb, where aabbbbbb is printed in here.
        PrintHeader(ChunkOffset - Candidate.size());
        OS << Candidate << StringRef(Begin, Size);
        Candidate.clear();
        CandidateLength = 0;
        InString = true;
      } else if (EndOfChunk) {
        // If the current chunk + previous candidate is still smaller than Min,
        // append it to Candidate.
        Candidate.append(Begin, StringEnd);
        CandidateLength += Len;
      } else {
        // If the string has terminated but is still smaller than Min, clear the
        // buffer since it is too short to print.
        Candidate.clear();
        CandidateLength = 0;
      }

      if (EndOfChunk || Cur == End) {
        // Finish handling the current chunk and update ChunkOffset.
        ChunkOffset += Cur - Begin;
        Buffer.erase(Buffer.begin(), Cur);
        continue;
      }

      if (InString) {
        // We haven't reached the end of the chunk, which means the string is
        // terminated. Add a '\n' to start printing a new string.
        EndString(PrevMBState);
        InString = false;
      }
    }

    // At this point, we are always at the start of a new string because the
    // remaining part of the previous string has already been handled.
    const char *StrHead = nullptr;
    size_t Len = 0;
    for (;;) {
      const char *Prev = Cur;
      std::mbstate_t PrevMBState = MBState;
      if (!Read(Cur, End, MBState, AtEOF, Ch))
        break;
      if (IsStringChar(Ch)) {
        // Find the start of the next string.
        if (!StrHead)
          StrHead = Prev;
        ++Len;
      } else if (StrHead) {
        // If it is not a printable character, we have reached the end of the
        // current string. Print it if long enough.
        if (Len >= Min) {
          PrintHeader(ChunkOffset + (StrHead - Begin));
          OS << StringRef(StrHead, Prev - StrHead);
          EndString(PrevMBState);
        }
        StrHead = nullptr;
        Len = 0;
      }
    }

    // The last string could span multiple chunks. If it is larger than Min,
    // print the header immediately and set the InString flag to avoid printing
    // it again.
    if (StrHead) {
      // Print it, or append it to Candidate if it is too short.
      if (Len >= Min) {
        PrintHeader(ChunkOffset + (StrHead - Begin));
        OS << StringRef(StrHead, Cur - StrHead);
        InString = true;
      } else {
        Candidate.append(StrHead, Cur);
        CandidateLength += Len;
      }
    }

    ChunkOffset += Cur - Begin;
    Buffer.erase(Buffer.begin(), Cur);

    if (AtEOF)
      break;
  }

  if (InString)
    EndString(MBState);
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

  Arg *EncodingArg = Args.getLastArg(OPT_encoding_EQ);
  if (!EncodingArg) {
    Encoding = Encoding::Locale;
  } else {
    auto EncodingVal = llvm::StringSwitch<std::optional<enum Encoding>>(
                           EncodingArg->getValue())
                           .Case("s", Encoding::Ascii)
                           .Case("S", Encoding::Locale)
                           .Case("utf8", Encoding::Utf8)
                           .Default(std::nullopt);
    if (!EncodingVal)
      invalidArgValue(EncodingArg);
    Encoding = *EncodingVal;
  }
  parseIntArg(Args, OPT_bytes_EQ, MinLength);
  PrintFileName = Args.hasArg(OPT_print_file_name);
  Arg *RadixArg = Args.getLastArg(OPT_radix_EQ);
  if (!RadixArg) {
    Radix = Radix::None;
  } else {
    Radix = llvm::StringSwitch<enum Radix>(RadixArg->getValue())
                .Case("o", Radix::Octal)
                .Case("d", Radix::Decimal)
                .Case("x", Radix::Hexadecimal)
                .Default(Radix::None);
    if (Radix == Radix::None)
      invalidArgValue(RadixArg);
  }

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
