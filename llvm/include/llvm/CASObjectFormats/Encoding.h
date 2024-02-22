//===- llvm/CASObjectFormats/Encoding.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CASOBJECTFORMATS_ENCODING_H
#define LLVM_CASOBJECTFORMATS_ENCODING_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include <numeric>
#include <type_traits>

namespace llvm {
namespace casobjectformats {
namespace encoding {

template <class T, std::enable_if_t<std::is_integral<T>::value &&
                                        std::numeric_limits<T>::is_signed,
                                    bool> = true>
static std::make_unsigned_t<T> rotateSign(T SV) {
  using UT = std::make_unsigned_t<T>;
  if (SV == std::numeric_limits<T>::min())
    return std::numeric_limits<UT>::max();
  UT UV = (UT(SV < 0 ? -SV - 1 : SV) << 1) | UT(SV < 0);
  return UV;
}

template <class T, std::enable_if_t<std::is_integral<T>::value &&
                                        !std::numeric_limits<T>::is_signed,
                                    bool> = true>
static std::make_signed_t<T> unrotateSign(T UV) {
  using ST = std::make_signed_t<T>;
  if (UV == std::numeric_limits<T>::max())
    return std::numeric_limits<ST>::min();
  ST SV = UV & 1 ? -ST(UV >> 1) - 1 : UV >> 1;
  return SV;
}

template <class T, std::enable_if_t<std::is_integral<T>::value &&
                                        !std::numeric_limits<T>::is_signed,
                                    bool> = true>
static void writeVBR8(T UV, SmallVectorImpl<char> &Data) {
  const unsigned TotalBits = sizeof(T) * 8;
  unsigned WrittenBits = 0;
  do {
    const unsigned RemainingBits = TotalBits - WrittenBits;
    const bool IsLastByte = RemainingBits <= 8;
    const unsigned BitsToWrite = IsLastByte ? RemainingBits : 7;
    unsigned char Byte = UV;
    UV >>= BitsToWrite;
    if (UV) {
      assert(!IsLastByte);
      Byte |= 0x80U;
    } else if (!IsLastByte) {
      Byte &= 0x7FU;
    }
    Data.push_back(Byte);
    WrittenBits += BitsToWrite;
  } while (UV);
}

template <class T, std::enable_if_t<std::is_integral<T>::value &&
                                        std::numeric_limits<T>::is_signed,
                                    bool> = true>
static void writeVBR8(T SV, SmallVectorImpl<char> &Data) {
  writeVBR8(rotateSign(SV), Data);
}

template <class T, std::enable_if_t<std::is_integral<T>::value &&
                                        !std::numeric_limits<T>::is_signed,
                                    bool> = true>
static Error consumeVBR8(StringRef &Data, T &UV) {
  const unsigned TotalBits = sizeof(T) * 8;
  unsigned ReadBits = 0;
  T ReadUV = 0U;
  while (TotalBits > ReadBits) {
    if (Data.empty())
      return createStringError(inconvertibleErrorCode(), "vbr8 missing bytes");
    const unsigned RemainingBits = TotalBits - ReadBits;
    const bool IsLastByte = RemainingBits <= 8;
    const unsigned BitsToRead = IsLastByte ? RemainingBits : 7;
    const T Byte = Data[0];
    Data = Data.drop_front();

    ReadUV |= (IsLastByte ? Byte : Byte & 0x7FU) << ReadBits;
    ReadBits += BitsToRead;

    if (IsLastByte || !(Byte & 0x80U)) {
      UV = ReadUV;
      return Error::success();
    }
  }
  // Should be unreachable.
  llvm::report_fatal_error(
      createStringError(inconvertibleErrorCode(), "vbr8 not finished"));
}

template <class T, std::enable_if_t<std::is_integral<T>::value &&
                                        std::numeric_limits<T>::is_signed,
                                    bool> = true>
static Error consumeVBR8(StringRef &Data, T &SV) {
  std::make_unsigned_t<T> UV;
  if (Error E = consumeVBR8(Data, UV))
    return E;
  SV = unrotateSign(UV);
  return Error::success();
}

template <class T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
static Expected<StringRef> readVBR8(StringRef Data, T &V) {
  if (Error E = consumeVBR8(Data, V))
    return std::move(E);
  return Data;
}

} // end namespace encoding
} // end namespace casobjectformats
} // end namespace llvm

#endif // LLVM_CASOBJECTFORMATS_ENCODING_H
