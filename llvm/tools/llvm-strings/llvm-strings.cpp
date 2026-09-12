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

static StringRef ToolName;

static cl::list<std::string> InputFileNames(cl::Positional,
                                            cl::desc("<input object files>"));

static int MinLength = 4;
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

static void strings(raw_ostream &OS, StringRef FileName, StringRef Contents) {
  std::locale Loc("");
  auto &Cvt = std::use_facet<std::codecvt<wchar_t, char, std::mbstate_t>>(Loc);
  auto &Ctype = std::use_facet<std::ctype<wchar_t>>(Loc);
  std::mbstate_t MBState{};

  auto Read = [&Cvt, &MBState](const char *&P, const char *E) -> UTF32 {
    UTF32 Ch;

    switch (Encoding) {
    case Encoding::Ascii:
      Ch = *P++;
      break;

    case Encoding::Locale: {
      const char *N;
      wchar_t WCh;
      wchar_t *WNext;
      [[maybe_unused]] const auto Res =
          Cvt.in(MBState, P, E, N, &WCh, &WCh + 1, WNext);
      assert(Res != std::codecvt_base::noconv);

      // Only treat a non-null character as a successful conversion, as a null
      // character may be the result of an incomplete multibyte character
      // followed by a null byte. A null byte is safe to treat as an error, as
      // a null byte is never printable in any locale.
      if (WNext != &WCh && WCh) {
        // Note: this assumes wchar_t is UCS2 or UTF32.
        Ch = WCh;
        P = N;
      } else {
        // If there was any error, skip the current byte and reset the state to
        // allow the next byte to start a character.
        Ch = 0;
        MBState = {};
        ++P;
      }
      break;
    }

    case Encoding::Utf8: {
      const UTF8 *UP = reinterpret_cast<const UTF8 *>(P);
      const UTF8 *UE = reinterpret_cast<const UTF8 *>(E);
      UTF32 *Next = &Ch;
      ConvertUTF8toUTF32(&UP, UE, &Next, &Ch + 1, strictConversion);
      if (Next == &Ch || !Ch) {
        // If there was any error, skip the current byte to allow the next to
        // start a character.
        Ch = 0;
        UP = std::next(reinterpret_cast<const UTF8 *>(P));
      }
      P = reinterpret_cast<const char *>(UP);
      break;
    }
    }

    return Ch;
  };

  auto Print = [&OS, FileName](unsigned Offset, StringRef L, size_t N) {
    if (N < static_cast<size_t>(MinLength))
      return;
    if (PrintFileName)
      OS << FileName << ": ";
    switch (Radix) {
    case Radix::None:
      break;
    case Radix::Octal:
      OS << format("%7o ", Offset);
      break;
    case Radix::Hexadecimal:
      OS << format("%7x ", Offset);
      break;
    case Radix::Decimal:
      OS << format("%7u ", Offset);
      break;
    }

    OS << L << '\n';
  };

  std::size_t NumPrintable = 0;

  const char *B = Contents.begin();
  const char *P = Contents.begin(), *E = Contents.end(), *S = nullptr;
  for (P = Contents.begin(), E = Contents.end(); P < E;) {
    const char *N = P;
    UTF32 Ch = Read(N, E);

    const bool Printable =
        Ch == '\t' ||
        (Encoding == Encoding::Ascii    ? Ch <= 0x7f && isPrint(Ch)
         : Encoding == Encoding::Locale ? Ctype.is(std::ctype_base::print, Ch)
                                        : sys::unicode::isPrintable(Ch));

    if (Printable) {
      if (S == nullptr)
        S = P;
      ++NumPrintable;
    } else if (S) {
      Print(S - B, StringRef(S, P - S), NumPrintable);
      S = nullptr;
      NumPrintable = 0;
    }

    P = N;
  }
  if (S)
    Print(S - B, StringRef(S, E - S), NumPrintable);
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
    ErrorOr<std::unique_ptr<MemoryBuffer>> Buffer =
        MemoryBuffer::getFileOrSTDIN(File, /*IsText=*/true);
    if (std::error_code EC = Buffer.getError())
      errs() << File << ": " << EC.message() << '\n';
    else
      strings(llvm::outs(), File == "-" ? "{standard input}" : File,
              Buffer.get()->getMemBufferRef().getBuffer());
  }

  return EXIT_SUCCESS;
}
