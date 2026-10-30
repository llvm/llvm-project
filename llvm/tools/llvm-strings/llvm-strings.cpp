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
#include "llvm/Support/UnicodeCharRanges.h"
#include "llvm/Support/WithColor.h"
#include <bitset>
#include <clocale>
#include <cuchar>
#include <string>

#ifdef LLVM_ON_UNIX
#include <langinfo.h>
#endif

#ifdef _WIN32
#include "llvm/Support/Windows/WindowsSupport.h"
#endif

#ifdef __APPLE__
#include <iconv.h>
#endif

using namespace llvm;
using namespace llvm::object;

namespace llvm::sys::unicode {
extern const UnicodeCharSet::CharRanges GeneratedCharacterRanges;
extern const UnicodeCharSet::CharRanges GeneratedControlCharacterRanges;
} // namespace llvm::sys::unicode

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

// Ideally we would use mbrtoc32() and c32rtomb() unconditionally but there are
// problems on most platforms, thus we introduce wrappers. On Windows,
// mbrtoc32() ignores the locale and only converts UTF-8, but wchar_t is UTF-16
// and is close enough, as long as we make sure to avoid it for UTF-8 locales.
// On Apple, <uchar.h> is not available, and wchar_t is not always UTF-32. We
// implement mbrtoc() and crtomb() through iconv().
// On glibc, mbrtoc32() is a wrapper for mbrtowc() and never returns (size_t)-3,
// instead it handles this case by returning 0 but setting Ch to a non-null
// character.
#if defined(_WIN32)
using char_t = wchar_t;
static size_t mbrtoc(char_t *Ch, const char *S, size_t N,
                     std::mbstate_t *MBState) {
  size_t Result = std::mbrtowc(Ch, S, N, MBState);
  if (Result == 0 && *Ch)
    Result = (size_t)-3;
  return Result;
}
static size_t crtomb(char *S, char_t Ch, std::mbstate_t *MBState) {
  return std::wcrtomb(S, Ch, MBState);
}
#elif defined(__APPLE__)
using char_t = char32_t;
static size_t mbrtoc(char_t *Ch, const char *S, size_t N,
                     std::mbstate_t *MBState) {
  static struct IconvHandle {
    IconvHandle() : Value(iconv_open("UCS-4-INTERNAL", nl_langinfo(CODESET))) {
      if (Value == (iconv_t)-1)
        report_fatal_error("failed to open iconv");
    }
    ~IconvHandle() { iconv_close(Value); }

    iconv_t Value;
  } CD;

  wchar_t WCh;
  size_t BytesRead = mbrtowc(&WCh, S, N, MBState);
  if (BytesRead != (size_t)-2 && BytesRead != (size_t)-1) {
    char Cvt[MB_LEN_MAX];
    std::mbstate_t CvtMBState{};
    const size_t BytesWritten = wcrtomb(Cvt, WCh, &CvtMBState);
    char *PIn = Cvt;
    size_t NIn = BytesWritten;
    char *POut = reinterpret_cast<char *>(Ch);
    size_t NOut = sizeof(char_t);
    const size_t NumErrors = iconv(CD.Value, &PIn, &NIn, &POut, &NOut);
    // mbrtowc() may be lenient and accept invalid characters that iconv()
    // rejects.
    if (NIn == BytesWritten)
      return (size_t)-1;
    if (NumErrors || NIn || NOut)
      llvm_unreachable("internal error: iconv failed on valid character");
  }
  return BytesRead;
}
static size_t crtomb(char *S, char_t Ch, std::mbstate_t *MBState) {
  static struct IconvHandle {
    IconvHandle() : Value(iconv_open(nl_langinfo(CODESET), "UCS-4-INTERNAL")) {
      if (Value == (iconv_t)-1)
        report_fatal_error("failed to open iconv");
    }
    ~IconvHandle() { iconv_close(Value); }

    iconv_t Value;
  } CD;

  char Cvt[MB_LEN_MAX];
  std::mbstate_t CvtMBState{};
  char *PIn = reinterpret_cast<char *>(&Ch);
  size_t NIn = sizeof(char_t);
  char *POut = Cvt;
  size_t NOut = MB_LEN_MAX;
  const size_t NumErrors = iconv(CD.Value, &PIn, &NIn, &POut, &NOut);
  if (NumErrors == (size_t)-1 || NIn)
    llvm_unreachable("internal error: iconv failed on valid character");
  if (NumErrors)
    return (size_t)-1;
  wchar_t WCh;
  if (mbrtowc(&WCh, Cvt, PIn - Cvt, &CvtMBState) != size_t(PIn - Cvt))
    llvm_unreachable("internal error: mbrtowc failed on valid character");
  return wcrtomb(S, WCh, MBState);
}
#else
using char_t = char32_t;
static size_t mbrtoc(char_t *Ch, const char *S, size_t N,
                     std::mbstate_t *MBState) {
  size_t Result = std::mbrtoc32(Ch, S, N, MBState);
  if (Result == 0 && *Ch)
    Result = (size_t)-3;
  return Result;
}
static size_t crtomb(char *S, char_t Ch, std::mbstate_t *MBState) {
  return std::c32rtomb(S, Ch, MBState);
}
#endif

struct CharSet {
  template <typename T> struct Base {
    using CharT = T;

    // Tries to read one character from the bytes Cur...End and store it in Ch.
    // Returns true if a (possibly invalid) character was read, false otherwise.
    //
    // If Cur...End starts with a valid complete character, Cur and MBState are
    // updated and true is returned.
    // If Cur...End is empty, or holds an incomplete character and AtEOF is
    // false, Cur and MBState are unchanged, Ch is set to 0, and false is
    // returned. This is intended to allow more bytes to be read and readChar to
    // be called again. If Cur...End holds an incomplete character and AtEOF is
    // true, or if Cur...End starts with an invalid character, the first bytes
    // are skipped and MBState is reset to allow continuing from the next point,
    // Ch is set to 0, and true is returned. This case is indistinguishable to
    // the caller from a null byte being successfully converted to a null
    // character, but a null character is never printable under any settings and
    // that is how invalid bytes are treated as well.
    //
    // The number of skipped bytes for invalid characters follows the Unicode
    // definition of the maximal subpart of an ill-formed subsequence, applied
    // to arbitrary locales: it is the longest subsequence that could start a
    // valid multibyte character, or if the initial byte cannot start a valid
    // multibyte character, the initial byte. This allows consistent error
    // recovery.
    static bool readChar(const char *&Cur, const char *End, bool AtEOF,
                         CharT &Ch);

    static bool isStringChar(CharT Ch);

    static void startString() {}

    static void writeString(raw_ostream &OS, StringRef Str) { OS << Str; }

    static void endString(raw_ostream &OS) { OS << '\n'; }
  };

  struct MultiByteStateless : Base<UTF32> {
    std::mbstate_t InMBState{};

    bool readChar(const char *&Cur, const char *End, bool AtEOF, CharT &Ch) {
      if (Cur == End)
        return false;

      const char *Next = Cur;
      std::mbstate_t NextMBState = InMBState;
      char_t UCh;
      for (;;) {
        // Read one byte at a time. This is usually not the best way to use
        // mbrtoc32(), usually it would make more sense to pass the size of the
        // buffer, but the strings utility is unusual in that it is expected to
        // encounter many bytes that do not form valid characters and it is more
        // useful to optimise for this case.
        const size_t BytesRead = mbrtoc(&UCh, Next, 1, &NextMBState);
        switch (BytesRead) {
        case 1:
          ++Next;
          LLVM_FALLTHROUGH;

        case (size_t)-3:
          Cur = Next;
          InMBState = NextMBState;
          Ch = std::min<UTF32>(UCh, 0x10FFFF);
          return true;

        case (size_t)-2:
          // We encountered a byte that is a valid start or continuation of a
          // multibyte character. If we have more bytes, carry on. If we don't
          // have more bytes yet, but we are not at the end of file, return
          // false. If we don't have more bytes and we are at the end of file,
          // fall through to treat it as an error.
          ++Next;
          if (Next != End)
            continue;
          if (!AtEOF)
            return false;
          LLVM_FALLTHROUGH;

        case 0:
        case (size_t)-1:
          // We cannot form a valid non-null character.
          //
          // If we processed any bytes already that formed an incomplete
          // multibyte character, treat those bytes as a single null character,
          // otherwise treat the current byte as a single null character.
          if (Next == Cur)
            ++Next;
          Cur = Next;
          InMBState = {};
          Ch = 0;
          return true;
        }
      }
    }

    static bool isStringChar(CharT Ch) {
      using StringCharSet = std::bitset<0x110000>;
      static const StringCharSet &StringChars = []() -> const StringCharSet & {
        static StringCharSet StringChars;

        using llvm::sys::unicode::GeneratedCharacterRanges;
        using llvm::sys::unicode::GeneratedControlCharacterRanges;
        using llvm::sys::UnicodeCharRange;

        for (const UnicodeCharRange &Range : GeneratedCharacterRanges)
          for (UTF32 Ch = Range.Lower; Ch <= Range.Upper; ++Ch)
            StringChars[Ch] = true;
        for (const UnicodeCharRange &Range : GeneratedControlCharacterRanges)
          for (UTF32 Ch = Range.Lower; Ch <= Range.Upper; ++Ch)
            StringChars[Ch] = false;
        StringChars[U'\t'] = true;

        return StringChars;
      }();

      return StringChars[Ch];
    }
  };

  // Note: The complexity in writing is only required for stateful encodings
  // such as the ISO-2022 ones.
  struct MultiByteStateful : MultiByteStateless {
    std::mbstate_t StrMBState{}, OutMBState{};

    void startString() { StrMBState = InMBState; }

    void writeString(raw_ostream &OS, StringRef Str) {
      const char *Cur = Str.begin(), *End = Str.end();
      while (Cur != End) {
        char_t UCh;
        const size_t BytesRead = mbrtoc(&UCh, Cur, End - Cur, &StrMBState);
        assert(BytesRead != 0 && BytesRead != (size_t)-1 &&
               BytesRead != (size_t)-2 &&
               "If a string was initially valid, it cannot to become invalid "
               "when re-reading");
        if (BytesRead != (size_t)-3)
          Cur += BytesRead;

        char Buf[MB_LEN_MAX];
        const size_t BytesWritten = crtomb(Buf, UCh, &OutMBState);
        assert(BytesWritten != (size_t)-1 &&
               "If we successfully read a character, we must also be able to "
               "write it");
        OS << StringRef(Buf, BytesWritten);
      }
    }

    void endString(raw_ostream &OS) {
      char Buf[MB_LEN_MAX];
      const size_t BytesWritten = crtomb(Buf, 0, &OutMBState);
      assert(BytesWritten != (size_t)-1 &&
             "A null character must always be valid");
      Buf[BytesWritten - 1] = '\n';
      OS << StringRef(Buf, BytesWritten);
    }
  };

  struct Utf8 : MultiByteStateless {
    static bool readChar(const char *&Cur, const char *End, bool AtEOF,
                         CharT &Ch) {
      if (Cur == End)
        return false;

      const UTF8 *UTF8Cur = reinterpret_cast<const UTF8 *>(Cur);
      const UTF8 *UTF8Next = UTF8Cur;
      const UTF8 *UTF8End = reinterpret_cast<const UTF8 *>(End);
      UTF32 *UTF32Next = &Ch;
      const auto Res = ConvertUTF8toUTF32(&UTF8Next, UTF8End, &UTF32Next,
                                          &Ch + 1, strictConversion);
      if (UTF8Next != UTF8Cur) {
        assert(UTF32Next != &Ch);
        assert(Ch <= 0x10FFFF &&
               "ConvertUTF8toUTF32 produced an out-of-range character");
      } else if (Res == sourceExhausted && !AtEOF) {
        return false;
      } else {
        assert(UTF32Next == &Ch);
        UTF8Next +=
            findMaximalSubpartOfIllFormedUTF8Sequence(UTF8Next, UTF8End);
        Ch = 0;
      }
      Cur = reinterpret_cast<const char *>(UTF8Next);
      return true;
    }
  };

  struct SingleByte : Base<unsigned char> {
    static bool readChar(const char *&Cur, const char *End, bool AtEOF,
                         CharT &Ch) {
      if (Cur == End)
        return false;

      Ch = *reinterpret_cast<const unsigned char *>(Cur++);
      return true;
    }

    static bool isStringChar(CharT Ch) {
      using StringCharSet = std::bitset<0x100>;
      static const StringCharSet &StringChars = []() -> const StringCharSet & {
        static StringCharSet StringChars;

        MultiByteStateless MB;
        for (int I = 0; I < 0x100; ++I) {
          const unsigned char Ch = I;
          const char *Cur = reinterpret_cast<const char *>(&Ch), *End = Cur + 1;
          MultiByteStateless::CharT UCh;
          if (MB.readChar(Cur, End, /*AtEOF=*/true, UCh) &&
              MB.isStringChar(UCh))
            StringChars[I] = true;
        }

        return StringChars;
      }();

      return StringChars[Ch];
    }
  };

  struct Ascii : SingleByte {
    static bool isStringChar(CharT Ch) {
      return Ch == '\t' || (Ch >= ' ' && Ch <= '~');
    }
  };
};

template <typename CharSetT>
void strings(raw_ostream &OS, StringRef FileName, sys::fs::file_t Handle) {
  using CharT = typename CharSetT::CharT;
  CharSetT CharSet;

  SmallString<sys::fs::DefaultReadChunkSize> Buffer;
  auto PrintHeader = [&OS, FileName](size_t StringStart) {
    if (PrintFileName)
      OS << FileName << ": ";
    switch (Radix) {
    case Radix::None:
      break;
    case Radix::Octal:
      OS << format("%7zo ", StringStart);
      break;
    case Radix::Hexadecimal:
      OS << format("%7zx ", StringStart);
      break;
    case Radix::Decimal:
      OS << format("%7zu ", StringStart);
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
  const size_t MinLengthChars = MinLength;
  SmallString<DefaultMinLength> Candidate;
  size_t CandidateLengthChars = 0;
  bool InString = false;
  // Offset of the start of the current chunk within the file.
  size_t ChunkOffset = 0;

  while (true) {
    // Size of any partial character left over from the previous chunk.
    const size_t PendingCharSizeBytes = Buffer.size();
    Buffer.resize_for_overwrite(PendingCharSizeBytes +
                                sys::fs::DefaultReadChunkSize);

    Expected<size_t> ReadBytesOrErr = sys::fs::readNativeFile(
        Handle,
        MutableArrayRef(Buffer.begin() + PendingCharSizeBytes, Buffer.end()));
    if (!ReadBytesOrErr) {
      errs() << FileName << ": "
             << errorToErrorCode(ReadBytesOrErr.takeError()).message() << '\n';
      return;
    }
    const bool AtEOF = *ReadBytesOrErr == 0;
    const size_t ChunkSize = PendingCharSizeBytes + *ReadBytesOrErr;
    if (ChunkSize == 0)
      break;

    Buffer.resize_for_overwrite(ChunkSize);

    // To prevent performance regression under O0, access the raw pointer
    // instead of using methods provided by the standard library, which are not
    // inlined under O0.
    const char *Begin = Buffer.data();
    const char *End = Begin + ChunkSize;
    const char *Cur = Begin;

    CharT Ch;

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
      size_t LengthChars = 0;
      bool EndOfChunk;

      // Find the end of the current string.
      for (;;) {
        StringEnd = Cur;
        EndOfChunk = !CharSet.readChar(Cur, End, AtEOF, Ch);
        if (EndOfChunk || !CharSet.isStringChar(Ch))
          break;
        ++LengthChars;
      }

      size_t SizeBytes = StringEnd - Begin;
      if (InString) {
        // Print the remaining part if the previous chunk has already printed
        // the header. E.g. header: aaaaa | bbbbb, where | is the chunk
        // boundary.
        // Output: Header: aaaaabbbbb, where bbbbb is printed in here.
        CharSet.writeString(OS, StringRef(Begin, SizeBytes));
      } else if (CandidateLengthChars + LengthChars >= MinLengthChars) {
        // If the header hasn't been printed yet (i.e. the previous candidate
        // was smaller than Min), but we can print it now, print the header
        // first, followed by the candidate from the previous chunk and the
        // current string. E.g. aa | bbbbbb
        // Output Header: aabbbbbb, where aabbbbbb is printed in here.
        PrintHeader(ChunkOffset - Candidate.size());
        CharSet.writeString(OS, Candidate);
        CharSet.writeString(OS, StringRef(Begin, SizeBytes));
        Candidate.clear();
        CandidateLengthChars = 0;
        InString = true;
      } else if (EndOfChunk) {
        // If the current chunk + previous candidate is still smaller than Min,
        // append it to Candidate.
        Candidate.append(Begin, StringEnd);
        CandidateLengthChars += LengthChars;
      } else {
        // If the string has terminated but is still smaller than Min, clear the
        // buffer since it is too short to print.
        Candidate.clear();
        CandidateLengthChars = 0;
      }

      if (EndOfChunk) {
        if (AtEOF)
          break;

        // Finish handling the current chunk and update ChunkOffset.
        ChunkOffset += Cur - Begin;
        Buffer.erase(Buffer.begin(), Cur);
        continue;
      }

      if (InString) {
        // We haven't reached the end of the chunk, which means the string is
        // terminated.
        CharSet.endString(OS);
        InString = false;
      }
    }

    // At this point, we are always at the start of a new string because the
    // remaining part of the previous string has already been handled.
    const char *StrHead = nullptr;
    size_t LengthChars = 0;
    for (;;) {
      const char *Prev = Cur;
      if (!StrHead)
        CharSet.startString();
      if (!CharSet.readChar(Cur, End, AtEOF, Ch))
        break;
      if (CharSet.isStringChar(Ch)) {
        // Find the start of the next string.
        if (!StrHead)
          StrHead = Prev;
        ++LengthChars;
      } else if (StrHead) {
        // If it is not a printable character, we have reached the end of the
        // current string. Print it if long enough.
        if (LengthChars >= MinLengthChars) {
          PrintHeader(ChunkOffset + (StrHead - Begin));
          CharSet.writeString(OS, StringRef(StrHead, Prev - StrHead));
          CharSet.endString(OS);
        }
        StrHead = nullptr;
        LengthChars = 0;
      }
    }

    // The last string could span multiple chunks. If it is larger than Min,
    // print the header immediately and set the InString flag to avoid printing
    // it again.
    if (StrHead) {
      // Print it, or append it to Candidate if it is too short.
      if (LengthChars >= MinLengthChars) {
        PrintHeader(ChunkOffset + (StrHead - Begin));
        CharSet.writeString(OS, StringRef(StrHead, Cur - StrHead));
        InString = true;
      } else {
        Candidate.append(StrHead, Cur);
        CandidateLengthChars += LengthChars;
      }
    }

    ChunkOffset += Cur - Begin;
    Buffer.erase(Buffer.begin(), Cur);

    if (AtEOF)
      break;
  }

  if (InString)
    CharSet.endString(OS);
}
} // namespace

int main(int argc, char **argv) {
  std::setlocale(LC_ALL, "");

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

  void (*const StringsImpl)(raw_ostream &OS, StringRef FileName,
                            sys::fs::file_t Handle) = [] {
    switch (Encoding) {
    case Encoding::Ascii:
      return strings<CharSet::Ascii>;

    case Encoding::Locale:
      if (MB_CUR_MAX == 1)
        return strings<CharSet::SingleByte>;

#ifdef LLVM_ON_UNIX
      // Using LLVM's built-in decoder is faster than going through libc, but
      // this doesn't affect correctness (assuming 32-bit wchar_t), so it's fine
      // if this check has false negatives (e.g. if some OS returns "utf8"
      // instead).
      if (StringRef(nl_langinfo(CODESET)) == "UTF-8")
        return strings<CharSet::Utf8>;
#endif
#ifdef _WIN32
      // Using LLVM's built-in decoder is necessary for correctness as
      // Strings<CharSet::MultiByteStateless> on WIndows is implemented on top
      // of 16-bit wchar_t which cannot represent all code points.
      if (GetACP() == CP_UTF8)
        return strings<CharSet::Utf8>;
#endif

      if (!mblen(NULL, 0))
        return strings<CharSet::MultiByteStateless>;
      else
        return strings<CharSet::MultiByteStateful>;

    case Encoding::Utf8:
      return strings<CharSet::Utf8>;
    }

    llvm_unreachable("unhandled encoding");
  }();

  std::vector<std::string> InputFileNames = Args.getAllArgValues(OPT_INPUT);
  if (InputFileNames.empty())
    InputFileNames.push_back("-");

  for (const auto &File : InputFileNames) {
    if (File == "-") {
      StringsImpl(llvm::outs(), "{standard input}", sys::fs::getStdinHandle());
    } else {
      Expected<sys::fs::file_t> FDOrErr =
          sys::fs::openNativeFileForRead(File, sys::fs::OF_TextWithCRLF);
      if (!FDOrErr) {
        errs() << File
               << ": cannot open file: " << toString(FDOrErr.takeError())
               << '\n';
        continue;
      }
      StringsImpl(llvm::outs(), File, *FDOrErr);
    }
  }

  return EXIT_SUCCESS;
}
