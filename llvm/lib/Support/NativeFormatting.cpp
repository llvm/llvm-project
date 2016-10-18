//===- NativeFormatting.cpp - Low level formatting helpers -------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/NativeFormatting.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Format.h"

using namespace llvm;

static bool isHexStyle(IntegerStyle S) {
  switch (S) {
  case IntegerStyle::HexLowerNoPrefix:
  case IntegerStyle::HexLowerPrefix:
  case IntegerStyle::HexUpperNoPrefix:
  case IntegerStyle::HexUpperPrefix:
    return true;
  default:
    return false;
  }
  LLVM_BUILTIN_UNREACHABLE;
}

static HexPrintStyle intHexStyleToHexStyle(IntegerStyle S) {
  assert(isHexStyle(S));
  switch (S) {
  case IntegerStyle::HexLowerNoPrefix:
    return HexPrintStyle::Lower;
  case IntegerStyle::HexLowerPrefix:
    return HexPrintStyle::PrefixLower;
  case IntegerStyle::HexUpperNoPrefix:
    return HexPrintStyle::Upper;
  case IntegerStyle::HexUpperPrefix:
    return HexPrintStyle::PrefixUpper;
  default:
    break;
  }
  LLVM_BUILTIN_UNREACHABLE;
}

static void writePadding(raw_ostream &S, Optional<int> FieldWidth,
                         size_t Chars) {
  if (!FieldWidth.hasValue())
    return;

  int Pad = *FieldWidth - Chars;
  if (Pad > 0)
    S.indent(Pad);
}

template<typename T, std::size_t N>
static int format_to_buffer(T Value, char (&Buffer)[N]) {
  char *EndPtr = std::end(Buffer);
  char *CurPtr = EndPtr;

  do {
    *--CurPtr = '0' + char(Value % 10);
    Value /= 10;
  } while (Value);
  return EndPtr - CurPtr;
}

static void repeat_char(raw_ostream &S, char C, size_t Times) {
  for (size_t I = 0; I < Times; ++I)
    S << C;
}

static void writeWithCommas(raw_ostream &S, ArrayRef<char> Buffer) {
  assert(!Buffer.empty());

  ArrayRef<char> ThisGroup;
  int InitialDigits = ((Buffer.size() - 1) % 3) + 1;
  ThisGroup = Buffer.take_front(InitialDigits);
  S.write(ThisGroup.data(), ThisGroup.size());

  Buffer = Buffer.drop_front(InitialDigits);
  assert(Buffer.size() % 3 == 0);
  while (!Buffer.empty()) {
    S << ',';
    ThisGroup = Buffer.take_front(3);
    S.write(ThisGroup.data(), 3);
    Buffer = Buffer.drop_front(3);
  }
}

template <typename T>
static void write_unsigned_impl(raw_ostream &S, T N, IntegerStyle Style,
                                Optional<size_t> Precision, Optional<int> Width,
                                bool IsNegative) {
  static_assert(std::is_unsigned<T>::value, "Value is not unsigned!");

  if (Style == IntegerStyle::Exponent) {
    write_double(S, static_cast<double>(N), FloatStyle::Exponent, Precision,
                 Width);
    return;
  } else if (Style == IntegerStyle::ExponentUpper) {
    write_double(S, static_cast<double>(N), FloatStyle::ExponentUpper,
                 Precision, Width);
    return;
  } else if (isHexStyle(Style)) {
    write_hex(S, N, intHexStyleToHexStyle(Style), Precision, Width);
    return;
  }

  size_t Prec = Precision.getValueOr(getDefaultPrecision(Style));
  char NumberBuffer[128];
  std::memset(NumberBuffer, '0', sizeof(NumberBuffer));

  size_t Len = 0;
  Len = format_to_buffer(N, NumberBuffer);

  bool WriteDecimal =
      ((Style == IntegerStyle::Fixed || Style == IntegerStyle::Percent) &&
       Prec > 0);

  size_t LeadingZeros = 0;
  if ((Style == IntegerStyle::Integer || Style == IntegerStyle::Number) &&
      Prec > 0) {
    if (Prec > Len)
      LeadingZeros = Prec - Len;
  }

  Len += LeadingZeros;

  // One for the decimal sign, one for each point of precision.
  size_t DecimalChars = WriteDecimal ? 1 + Prec : 0;

  // One character for the negative sign.
  size_t Neg = (IsNegative) ? 1 : 0;

  // One comma for each group of 3 digits.
  size_t Commas = (Style != IntegerStyle::Number) ? 0 : (Len - 1) / 3;

  size_t PercentChars = 0;
  if (Style == IntegerStyle::Percent) {
    // For all numbers except 0, we append two additional 0s.
    PercentChars = (N == 0) ? 1 : 3;
  }

  writePadding(S, Width, Len + DecimalChars + Neg + Commas + PercentChars);

  if (IsNegative)
    S << '-';
  if (Style == IntegerStyle::Number) {
    writeWithCommas(S, ArrayRef<char>(std::end(NumberBuffer) - Len, Len));
  } else {
    S.write(std::end(NumberBuffer) - Len, Len);
    if (Style == IntegerStyle::Percent && N != 0) {
      // Rather than multiply by 100, write the characters manually, in case the
      // multiplication would overflow.
      S << "00";
    }
  }

  if (WriteDecimal) {
    S << '.';
    repeat_char(S, '0', Prec);
  }
  if (Style == IntegerStyle::Percent)
    S << '%';
}

template <typename T>
static void write_unsigned(raw_ostream &S, T N, IntegerStyle Style,
                           Optional<size_t> Precision, Optional<int> Width,
                           bool IsNegative = false) {
  // Output using 32-bit div/mod if possible.
  if (N == static_cast<uint32_t>(N))
    write_unsigned_impl(S, static_cast<uint32_t>(N), Style, Precision, Width,
                        IsNegative);
  else
    write_unsigned_impl(S, N, Style, Precision, Width, IsNegative);
}

template <typename T>
static void write_signed(raw_ostream &S, T N, IntegerStyle Style,
                         Optional<size_t> Precision, Optional<int> Width) {
  static_assert(std::is_signed<T>::value, "Value is not signed!");

  using UnsignedT = typename std::make_unsigned<T>::type;

  if (N >= 0) {
    write_unsigned(S, static_cast<UnsignedT>(N), Style, Precision, Width);
    return;
  }

  UnsignedT UN = -(UnsignedT)N;
  if (isHexStyle(Style)) {
    static_assert(sizeof(UnsignedT) == sizeof(T),
                  "Types do not have the same size!");
    std::memcpy(&UN, &N, sizeof(N));
    write_hex(S, UN, intHexStyleToHexStyle(Style), Precision, Width);
    return;
  }
  write_unsigned(S, UN, Style, Precision, Width, true);
}

void llvm::write_integer(raw_ostream &S, unsigned int N, IntegerStyle Style,
                         Optional<size_t> Precision, Optional<int> Width) {
  write_unsigned(S, N, Style, Precision, Width);
}

void llvm::write_integer(raw_ostream &S, int N, IntegerStyle Style,
                         Optional<size_t> Precision, Optional<int> Width) {
  write_signed(S, N, Style, Precision, Width);
}

void llvm::write_integer(raw_ostream &S, unsigned long N, IntegerStyle Style,
                         Optional<size_t> Precision, Optional<int> Width) {
  write_unsigned(S, N, Style, Precision, Width);
}

void llvm::write_integer(raw_ostream &S, long N, IntegerStyle Style,
                         Optional<size_t> Precision, Optional<int> Width) {
  write_signed(S, N, Style, Precision, Width);
}

void llvm::write_integer(raw_ostream &S, unsigned long long N,
                         IntegerStyle Style, Optional<size_t> Precision,
                         Optional<int> Width) {
  write_unsigned(S, N, Style, Precision, Width);
}

void llvm::write_integer(raw_ostream &S, long long N, IntegerStyle Style,
                         Optional<size_t> Precision, Optional<int> Width) {
  write_signed(S, N, Style, Precision, Width);
}

void llvm::write_hex(raw_ostream &S, uint64_t N, HexPrintStyle Style,
                     Optional<size_t> Precision, Optional<int> Width) {
  const size_t kMaxWidth = 128u;

  size_t Prec =
      std::min(kMaxWidth, Precision.getValueOr(getDefaultPrecision(Style)));

  unsigned Nibbles = (64 - countLeadingZeros(N) + 3) / 4;
  bool Prefix = (Style == HexPrintStyle::PrefixLower ||
                 Style == HexPrintStyle::PrefixUpper);
  bool Upper =
      (Style == HexPrintStyle::Upper || Style == HexPrintStyle::PrefixUpper);
  unsigned PrefixChars = Prefix ? 2 : 0;
  unsigned NumChars = std::max(static_cast<unsigned>(Prec),
                               std::max(1u, Nibbles) + PrefixChars);

  char NumberBuffer[kMaxWidth];
  ::memset(NumberBuffer, '0', llvm::array_lengthof(NumberBuffer));
  if (Prefix)
    NumberBuffer[1] = 'x';
  char *EndPtr = NumberBuffer + NumChars;
  char *CurPtr = EndPtr;
  while (N) {
    unsigned char x = static_cast<unsigned char>(N) % 16;
    *--CurPtr = hexdigit(x, !Upper);
    N /= 16;
  }

  writePadding(S, Width, NumChars);
  S.write(NumberBuffer, NumChars);
}

void llvm::write_double(raw_ostream &S, double N, FloatStyle Style,
                        Optional<size_t> Precision, Optional<int> Width) {
  size_t Prec = Precision.getValueOr(getDefaultPrecision(Style));

  if (std::isnan(N)) {
    writePadding(S, Width, 3);
    S << "nan";
    return;
  } else if (std::isinf(N)) {
    writePadding(S, Width, 3);
    S << "INF";
    return;
  }

  char Letter;
  if (Style == FloatStyle::Exponent)
    Letter = 'e';
  else if (Style == FloatStyle::ExponentUpper)
    Letter = 'E';
  else
    Letter = 'f';

  SmallString<8> Spec;
  llvm::raw_svector_ostream Out(Spec);
  Out << "%." << Prec << Letter;

  if (Style == FloatStyle::Exponent || Style == FloatStyle::ExponentUpper) {
#ifdef _WIN32
// On MSVCRT and compatible, output of %e is incompatible to Posix
// by default. Number of exponent digits should be at least 2. "%+03d"
// FIXME: Implement our formatter to here or Support/Format.h!
#if defined(__MINGW32__)
    // FIXME: It should be generic to C++11.
    if (N == 0.0 && std::signbit(N)) {
      const char *NegativeZero = "-0.000000e+00";
      writePadding(S, Width, strlen(NegativeZero));
      S << NegativeZero;
      return;
    }
#else
    int fpcl = _fpclass(N);

    // negative zero
    if (fpcl == _FPCLASS_NZ) {
      const char *NegativeZero = "-0.000000e+00";
      writePadding(S, Width, strlen(NegativeZero));
      S << NegativeZero;
      return;
    }
#endif

    char buf[32];
    unsigned len;
    len = format(Spec.c_str(), N).snprint(buf, sizeof(buf));
    if (len <= sizeof(buf) - 2) {
      if (len >= 5 && (buf[len - 5] == 'e' || buf[len - 5] == 'E') &&
          buf[len - 3] == '0') {
        int cs = buf[len - 4];
        if (cs == '+' || cs == '-') {
          int c1 = buf[len - 2];
          int c0 = buf[len - 1];
          if (isdigit(static_cast<unsigned char>(c1)) &&
              isdigit(static_cast<unsigned char>(c0))) {
            // Trim leading '0': "...e+012" -> "...e+12\0"
            buf[len - 3] = c1;
            buf[len - 2] = c0;
            buf[--len] = 0;
          }
        }
      }
      writePadding(S, Width, len);
      S << buf;
      return;
    }
#endif
  }

  if (Style == FloatStyle::Percent)
    N *= 100.0;

  char Buf[32];
  unsigned Len;
  Len = format(Spec.c_str(), N).snprint(Buf, sizeof(Buf));
  if (Style == FloatStyle::Percent)
    ++Len;
  writePadding(S, Width, Len);
  S << Buf;
  if (Style == FloatStyle::Percent)
    S << '%';
}

IntegerStyle llvm::hexStyleToIntHexStyle(HexPrintStyle S) {
  switch (S) {
  case HexPrintStyle::Upper:
    return IntegerStyle::HexUpperNoPrefix;
  case HexPrintStyle::Lower:
    return IntegerStyle::HexLowerNoPrefix;
  case HexPrintStyle::PrefixUpper:
    return IntegerStyle::HexUpperPrefix;
  case HexPrintStyle::PrefixLower:
    return IntegerStyle::HexLowerPrefix;
  }
  LLVM_BUILTIN_UNREACHABLE;
}

size_t llvm::getDefaultPrecision(FloatStyle Style) {
  switch (Style) {
  case FloatStyle::Exponent:
  case FloatStyle::ExponentUpper:
    return 6; // Number of decimal places.
  case FloatStyle::Fixed:
  case FloatStyle::Percent:
    return 2; // Number of decimal places.
  }
  LLVM_BUILTIN_UNREACHABLE;
}

size_t llvm::getDefaultPrecision(IntegerStyle Style) {
  switch (Style) {
  case IntegerStyle::Exponent:
  case IntegerStyle::ExponentUpper:
    return 6; // Number of decimal places.
  case IntegerStyle::Number:
  case IntegerStyle::Integer:
    return 0; // Minimum number of digits required.
  case IntegerStyle::Fixed:
    return 2; // Number of decimal places.
  case IntegerStyle::Percent:
    return 0; // Number of decimal places.
  case IntegerStyle::HexLowerNoPrefix:
  case IntegerStyle::HexLowerPrefix:
  case IntegerStyle::HexUpperNoPrefix:
  case IntegerStyle::HexUpperPrefix:
    return getDefaultPrecision(intHexStyleToHexStyle(Style));
  }
  LLVM_BUILTIN_UNREACHABLE;
}

size_t llvm::getDefaultPrecision(HexPrintStyle) {
  // Number of digits in the resulting string.
  return 0;
}
