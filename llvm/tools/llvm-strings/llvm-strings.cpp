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
} // namespace

static StringRef ToolName;

static cl::list<std::string> InputFileNames(cl::Positional,
                                            cl::desc("<input object files>"));

static int MinLength = 4;
static bool PrintFileName;

enum class encoding { s, S, u, b, l, B, L };
static encoding Encoding;

enum class radix { none, octal, hexadecimal, decimal };
static radix Radix;

enum class unicode { default_, invalid, locale, escape, hex, highlight };
static unicode Unicode;

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
  std::locale loc("");
  auto &cvt = std::use_facet<std::codecvt<wchar_t, char, std::mbstate_t>>(loc);
  auto &ctype = std::use_facet<std::ctype<wchar_t>>(loc);
  std::mbstate_t mbs{};

  const bool PrintRawBytes =
      Encoding == encoding::s ||
      (Encoding == encoding::S &&
       (Unicode == unicode::invalid || Unicode == unicode::locale)) ||
      (Encoding == encoding::u &&
       (Unicode == unicode::default_ || Unicode == unicode::invalid));

  auto read = [&cvt, &mbs](const char *&P, const char *E) -> UTF32 {
    UTF32 Ch;

    switch (Encoding) {
    case encoding::s:
      Ch = *P++;
      break;

    case encoding::S: {
      const char *N;
      wchar_t WCh;
      wchar_t *WNext;
      [[maybe_unused]] const auto res =
          cvt.in(mbs, P, E, N, &WCh, &WCh + 1, WNext);
      assert(res != std::codecvt_base::noconv);

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
        mbs = {};
        ++P;
      }
      break;
    }

    case encoding::u: {
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

    case encoding::b:
    case encoding::l: {
      const UTF16 *UP = reinterpret_cast<const UTF16 *>(P);
      const UTF16 *UE = reinterpret_cast<const UTF16 *>(E);
      const bool DoByteSwap =
          (Encoding == encoding::b &&
           endianness::native == endianness::little) ||
          (Encoding == encoding::l && endianness::native == endianness::big);
      UTF32 *Next = &Ch;
      if (!DoByteSwap) {
        ConvertUTF16toUTF32(&UP, UE, &Next, &Ch + 1, strictConversion);
      } else {
        // We never need more than two UTF16 words to make up one character.
        const UTF16 Buf[2] = {byteswap(UP[0]),
                              UP + 1 == UE ? UTF16(0) : byteswap(UP[1])};
        const UTF16 *BufP = Buf;
        ConvertUTF16toUTF32(&BufP, &Buf[2], &Next, &Ch + 1, strictConversion);
        UP += BufP - Buf;
      }
      if (Next == &Ch || !Ch) {
        // If there was any error, skip the current word to allow the next to
        // start a character.
        Ch = 0;
        UP = std::next(reinterpret_cast<const UTF16 *>(P));
      }
      P = reinterpret_cast<const char *>(UP);
      break;
    }

    case encoding::B:
    case encoding::L: {
      const UTF32 *UP = reinterpret_cast<const UTF32 *>(P);
      const bool DoByteSwap =
          (Encoding == encoding::B &&
           endianness::native == endianness::little) ||
          (Encoding == encoding::L && endianness::native == endianness::big);
      Ch = DoByteSwap ? byteswap(*UP) : *UP;
      ++UP;
      P = reinterpret_cast<const char *>(UP);
      break;
    }
    }

    return Ch;
  };

  auto print = [&OS, FileName, PrintRawBytes, &read,
                &cvt](unsigned Offset, StringRef L, size_t N) {
    if (N < static_cast<size_t>(MinLength))
      return;
    if (PrintFileName)
      OS << FileName << ": ";
    switch (Radix) {
    case radix::none:
      break;
    case radix::octal:
      OS << format("%7o ", Offset);
      break;
    case radix::hexadecimal:
      OS << format("%7x ", Offset);
      break;
    case radix::decimal:
      OS << format("%7u ", Offset);
      break;
    }

    if (PrintRawBytes) {
      OS << L << '\n';
    } else {
      mbstate_t mbs = {};

      const char *P = L.begin();
      const char *E = L.end();
      while (P < E) {
        const UTF32 Ch = read(P, E);
        if (Unicode == unicode::invalid || Ch <= 0x7F) {
          OS << (char)Ch;
          continue;
        }

        if (Unicode == unicode::locale) {
          // If we have a 16-bit wchar_t and the character does not fit, replace
          // it with U+FFFD.
          wchar_t WCh = Ch;
          const wchar_t *WNext;
          if ((UTF32)WCh != Ch)
            WCh = 0xFFFD;
          char mbstring[MB_LEN_MAX];
          char *mbend;
          [[maybe_unused]] const auto res =
              cvt.out(mbs, &WCh, &WCh + 1, WNext, mbstring,
                      &mbstring[MB_LEN_MAX], mbend);
          assert(res == std::codecvt_base::ok);
          OS << StringRef(mbstring, mbend - mbstring);
          continue;
        }

        if (Unicode == unicode::escape || Unicode == unicode::highlight) {
          WithColor COS(OS, raw_ostream::RED, false, false,
                        Unicode == unicode::highlight ? ColorMode::Auto
                                                      : ColorMode::Disable);
          if (Ch <= 0xFFFF)
            COS << "\\u" << utohexstr(Ch, false, 4);
          else
            COS << "\\U" << utohexstr(Ch, false, 8);
          continue;
        }

        char UTF8[4];
        char *UTF8end = UTF8;
        ConvertCodePointToUTF8(Ch, UTF8end);
        if (Unicode == unicode::hex) {
          OS << "<0x";
          for (unsigned char Byte : StringRef(UTF8, UTF8end - UTF8))
            OS << utohexstr(Byte, true, 2);
          OS << '>';
          continue;
        }

        assert(Unicode == unicode::default_);
        OS << StringRef(UTF8, UTF8end - UTF8);
      }
      OS << '\n';
    }
  };

  std::size_t NumPrintable = 0;

  const char *B = Contents.begin();
  const char *P = Contents.begin(), *E = Contents.end(), *S = nullptr;
  for (P = Contents.begin(), E = Contents.end(); P < E;) {
    const char *N = P;
    UTF32 Ch = read(N, E);

    const bool Printable =
        Ch == '\t' ||
        (Unicode == unicode::invalid ? Ch <= 0x7f && isPrint(Ch)
         : Encoding == encoding::S   ? ctype.is(std::ctype_base::print, Ch)
                                     : sys::unicode::isPrintable(Ch));

    if (Printable) {
      if (S == nullptr)
        S = P;
      ++NumPrintable;
    } else if (S) {
      print(S - B, StringRef(S, P - S), NumPrintable);
      S = nullptr;
      NumPrintable = 0;
    }

    P = N;
  }
  if (S)
    print(S - B, StringRef(S, E - S), NumPrintable);
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

  const auto *EncodingArg = Args.getLastArg(OPT_encoding_EQ);
  const auto *UnicodeArg = Args.getLastArg(OPT_unicode_EQ);

  Encoding =
      EncodingArg
          ? llvm::StringSwitch<encoding>(EncodingArg->getValue())
                .Case("s", encoding::s)
                .Case("S", encoding::S)
                .Case("u", encoding::u)
                .Case("b", encoding::b)
                .Case("l", encoding::l)
                .Case("B", encoding::B)
                .Case("L", encoding::L)
                .Predicate(
                    [](StringRef) -> bool {
                      reportCmdLineError(
                          "--encoding value should be one of: "
                          "'s' (7-bit characters), "
                          "'S' (8-bit characters), "
                          "'u' (UTF-8 characters), "
                          "'b', 'l' (16-bit big/little endian characters), "
                          "'B', 'L' (32-bit big/little endian characters)");
                    },
                    encoding{})
                .DefaultUnreachable()
          : encoding::S;
  parseIntArg(Args, OPT_bytes_EQ, MinLength);
  PrintFileName = Args.hasArg(OPT_print_file_name);
  StringRef R = Args.getLastArgValue(OPT_radix_EQ);
  if (R.empty())
    Radix = radix::none;
  else if (R == "o")
    Radix = radix::octal;
  else if (R == "d")
    Radix = radix::decimal;
  else if (R == "x")
    Radix = radix::hexadecimal;
  else
    reportCmdLineError("--radix value should be one of: '' (no offset), 'o' "
                       "(octal), 'd' (decimal), 'x' (hexadecimal)");
  Unicode = UnicodeArg ? llvm::StringSwitch<unicode>(UnicodeArg->getValue())
                             .Case("default", unicode::default_)
                             .Case("invalid", unicode::invalid)
                             .Case("locale", unicode::locale)
                             .Case("escape", unicode::escape)
                             .Case("hex", unicode::hex)
                             .Case("highlight", unicode::highlight)
                             .Predicate(
                                 [](StringRef) -> bool {
                                   reportCmdLineError(
                                       "--unicode value should be one of: "
                                       "default, "
                                       "invalid, "
                                       "locale, "
                                       "escape, "
                                       "hex, "
                                       "highlight");
                                 },
                                 unicode::default_)
                             .DefaultUnreachable()
                       : unicode::locale;

  // The defaults are ugly to maintain compatibility with GNU strings in the
  // common cases. The general idea is that --encoding specifies the input
  // encoding, --unicode specifies the output encoding, but when one is
  // specified, the other defaults to the best match with the limitation that
  // the output encoding is never UTF-16 or UTF-32.
  if (!EncodingArg && Unicode != unicode::invalid && Unicode != unicode::locale)
    Encoding = encoding::u;

  if (!UnicodeArg && Encoding != encoding::S)
    Unicode = unicode::default_;

  // With --encoding=s, all output encodings would give the same results. Use
  // the simplest one.
  if (Encoding == encoding::s)
    Unicode = unicode::invalid;

  // With --encoding=u --unicode=invalid, we do not need to spend time decoding
  // UTF-8 only to reject whatever results we get, we can reject multibyte
  // characters right away.
  if (Encoding == encoding::u && Unicode == unicode::invalid)
    Encoding = encoding::s;

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
