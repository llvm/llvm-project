//===- NativeFormatting.cpp - Low level formatting helpers -------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/NativeFormatting.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Format.h"

using namespace llvm;

template<typename T, std::size_t N>
static int format_to_buffer(T Value, char (&Buffer)[N]) {
  char *EndPtr = std::end(Buffer);
  char *CurPtr = EndPtr;

  while (Value) {
    *--CurPtr = '0' + char(Value % 10);
    Value /= 10;
  }
  return EndPtr - CurPtr;
}

void llvm::write_ulong(raw_ostream &S, unsigned long N, std::size_t MinWidth) {
  // Zero is a special case.
  if (N == 0) {
    if (MinWidth > 0)
      S.indent(MinWidth - 1);
    S << '0';
    return;
  }

  char NumberBuffer[20];
  int Len = format_to_buffer(N, NumberBuffer);
  int Pad = (MinWidth == 0) ? 0 : MinWidth - Len;
  if (Pad > 0)
    S.indent(Pad);
  S.write(std::end(NumberBuffer) - Len, Len);
}

void llvm::write_long(raw_ostream &S, long N, std::size_t MinWidth) {
  if (N >= 0) {
    write_ulong(S, static_cast<unsigned long>(N), MinWidth);
    return;
  }

  unsigned long UN = -(unsigned long)N;
  if (MinWidth > 0)
    --MinWidth;

  char NumberBuffer[20];
  int Len = format_to_buffer(UN, NumberBuffer);
  int Pad = (MinWidth == 0) ? 0 : MinWidth - Len;
  if (Pad > 0)
    S.indent(Pad);
  S.write('-');
  S.write(std::end(NumberBuffer) - Len, Len);
}

void llvm::write_ulonglong(raw_ostream &S, unsigned long long N,
                           std::size_t MinWidth) {
  // Output using 32-bit div/mod when possible.
  if (N == static_cast<unsigned long>(N)) {
    write_ulong(S, static_cast<unsigned long>(N), MinWidth);
    return;
  }

  char NumberBuffer[32];
  int Len = format_to_buffer(N, NumberBuffer);
  int Pad = (MinWidth == 0) ? 0 : MinWidth - Len;
  if (Pad > 0)
    S.indent(Pad);
  S.write(std::end(NumberBuffer) - Len, Len);
}

void llvm::write_longlong(raw_ostream &S, long long N, std::size_t MinWidth) {
  if (N >= 0) {
    write_ulonglong(S, static_cast<unsigned long long>(N), MinWidth);
    return;
  }

  // Avoid undefined behavior on INT64_MIN with a cast.
  unsigned long long UN = -(unsigned long long)N;
  if (MinWidth > 0)
    --MinWidth;

  char NumberBuffer[32];
  int Len = format_to_buffer(UN, NumberBuffer);
  int Pad = (MinWidth == 0) ? 0 : MinWidth - Len;
  if (Pad > 0)
    S.indent(Pad);
  S.write('-');
  S.write(std::end(NumberBuffer) - Len, Len);
}

void llvm::write_hex(raw_ostream &S, unsigned long long N, std::size_t MinWidth,
                     bool Upper, bool Prefix) {
  unsigned Nibbles = (64 - countLeadingZeros(N) + 3) / 4;
  unsigned PrefixChars = Prefix ? 2 : 0;
  unsigned Width = std::max(static_cast<unsigned>(MinWidth),
                            std::max(1u, Nibbles) + PrefixChars);

  char NumberBuffer[20] = "0x0000000000000000";
  if (!Prefix)
    NumberBuffer[1] = '0';
  char *EndPtr = NumberBuffer + Width;
  char *CurPtr = EndPtr;
  while (N) {
    unsigned char x = static_cast<unsigned char>(N) % 16;
    *--CurPtr = hexdigit(x, !Upper);
    N /= 16;
  }

  S.write(NumberBuffer, Width);
}

void llvm::write_double(raw_ostream &S, double N, std::size_t MinWidth,
                        std::size_t MinDecimals, FloatStyle Style) {
  char Letter = (Style == FloatStyle::Exponent) ? 'e' : 'f';
  SmallString<8> Spec;
  llvm::raw_svector_ostream Out(Spec);
  Out << '%';
  if (MinWidth > 0)
    Out << MinWidth;
  if (MinDecimals > 0)
    Out << '.' << MinDecimals;
  Out << Letter;

  if (Style == FloatStyle::Exponent) {
#ifdef _WIN32
// On MSVCRT and compatible, output of %e is incompatible to Posix
// by default. Number of exponent digits should be at least 2. "%+03d"
// FIXME: Implement our formatter to here or Support/Format.h!
#if defined(__MINGW32__)
    // FIXME: It should be generic to C++11.
    if (N == 0.0 && std::signbit(N)) {
      S << "-0.000000e+00";
      return;
    }
#else
    int fpcl = _fpclass(N);

    // negative zero
    if (fpcl == _FPCLASS_NZ) {
      S << "-0.000000e+00";
      return;
    }
#endif

    char buf[16];
    unsigned len;
    len = format(Spec.c_str(), N).snprint(buf, sizeof(buf));
    if (len <= sizeof(buf) - 2) {
      if (len >= 5 && buf[len - 5] == 'e' && buf[len - 3] == '0') {
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
      S << buf;
      return;
    }
#endif
  }

  S << format(Spec.c_str(), N);
}
