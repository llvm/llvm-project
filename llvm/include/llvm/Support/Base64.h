//===--- Base64.h - Base64 Encoder/Decoder ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides generic base64 encoder/decoder.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_BASE64_H
#define LLVM_SUPPORT_BASE64_H

#include "llvm/Support/Error.h"
#include <cstdint>
#include <string>

namespace llvm {

template <class InputBytes> std::string encodeBase64(InputBytes const &Bytes) {
  static const char Table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                              "abcdefghijklmnopqrstuvwxyz"
                              "0123456789+/";
  std::string Buffer;
  Buffer.resize(((Bytes.size() + 2) / 3) * 4);

  size_t i = 0, j = 0;
  for (size_t n = Bytes.size() / 3 * 3; i < n; i += 3, j += 4) {
    uint32_t x = ((unsigned char)Bytes[i] << 16) |
                 ((unsigned char)Bytes[i + 1] << 8) |
                 (unsigned char)Bytes[i + 2];
    Buffer[j + 0] = Table[(x >> 18) & 63];
    Buffer[j + 1] = Table[(x >> 12) & 63];
    Buffer[j + 2] = Table[(x >> 6) & 63];
    Buffer[j + 3] = Table[x & 63];
  }
  if (i + 1 == Bytes.size()) {
    uint32_t x = ((unsigned char)Bytes[i] << 16);
    Buffer[j + 0] = Table[(x >> 18) & 63];
    Buffer[j + 1] = Table[(x >> 12) & 63];
    Buffer[j + 2] = '=';
    Buffer[j + 3] = '=';
  } else if (i + 2 == Bytes.size()) {
    uint32_t x =
        ((unsigned char)Bytes[i] << 16) | ((unsigned char)Bytes[i + 1] << 8);
    Buffer[j + 0] = Table[(x >> 18) & 63];
    Buffer[j + 1] = Table[(x >> 12) & 63];
    Buffer[j + 2] = Table[(x >> 6) & 63];
    Buffer[j + 3] = '=';
  }
  return Buffer;
}

template <class OutputBytes>
llvm::Error decodeBase64(llvm::StringRef Input, OutputBytes &Output) {
  // Invalid table value with short name to fit in the table init below. The
  // invalid value is 64 since valid base64 values are 0 - 63.
  constexpr char Inv = 64;
  static char DecodeTable[] = {
      Inv, Inv, Inv, Inv, Inv, Inv, Inv, Inv, // ........
      Inv, Inv, Inv, Inv, Inv, Inv, Inv, Inv, // ........
      Inv, Inv, Inv, Inv, Inv, Inv, Inv, Inv, // ........
      Inv, Inv, Inv, Inv, Inv, Inv, Inv, Inv, // ........
      Inv, Inv, Inv, Inv, Inv, Inv, Inv, Inv, // ........
      Inv, Inv, Inv, 62,  Inv, Inv, Inv, 63,  // ...+.../
      52,  53,  54,  55,  56,  57,  58,  59,  // 01234567
      60,  61,  Inv, Inv, Inv, 0,   Inv, Inv, // 89...=..
      Inv, 0,   1,   2,   3,   4,   5,   6,   // .ABCDEFG
      7,   8,   9,   10,  11,  12,  13,  14,  // HIJKLMNO
      15,  16,  17,  18,  19,  20,  21,  22,  // PQRSTUVW
      23,  24,  25,  Inv, Inv, Inv, Inv, Inv, // XYZ.....
      Inv, 26,  27,  28,  29,  30,  31,  32,  // .abcdefg
      33,  34,  35,  36,  37,  38,  39,  40,  // hijklmno
      41,  42,  43,  44,  45,  46,  47,  48,  // pqrstuvw
      49,  50,  51                            // xyz.....
  };
  auto decodeBase64Byte = [](uint8_t Ch) -> char {
    if (Ch >= sizeof(DecodeTable))
      return Inv;
    return DecodeTable[Ch];
  };
  Output.clear();
  const uint64_t InputLength = Input.size();
  if (InputLength == 0)
    return Error::success();
  // Make sure we have a valid input string length which must be a multiple
  // of 4.
  if ((InputLength % 4) != 0)
    return createStringError(std::errc::illegal_byte_sequence,
                             "Base64 encoded strings must be a multiple of 4 "
                             "bytes in length");
  const uint64_t FirstValidEqualIdx = InputLength - 2;
  char Hex64Bytes[4];
  for (uint64_t Idx = 0; Idx < InputLength; Idx += 4) {
    for (uint64_t ByteOffset = 0; ByteOffset < 4; ++ByteOffset) {
      const uint64_t ByteIdx = Idx + ByteOffset;
      const char Byte = Input[ByteIdx];
      const char DecodedByte = decodeBase64Byte(Byte);
      bool Illegal = DecodedByte == Inv;
      if (!Illegal && Byte == '=') {
        if (ByteIdx < FirstValidEqualIdx) {
          // We have an '=' in the middle of the string which is invalid, only
          // the last two characters can be '=' characters.
          Illegal = true;
        } else if (ByteIdx == FirstValidEqualIdx && Input[ByteIdx + 1] != '=') {
          // We have an equal second to last from the end and the last character
          // is not also an equal, so the '=' character is invalid
          Illegal = true;
        }
      }
      if (Illegal)
        return createStringError(
            std::errc::illegal_byte_sequence,
            "Invalid Base64 character %#2.2x at index %" PRIu64, Byte, ByteIdx);
      Hex64Bytes[ByteOffset] = DecodedByte;
    }
    // Now we have 6 bits of 3 bytes in value in each of the Hex64Bytes bytes.
    // Extract the right bytes into the Output buffer.
    Output.push_back((Hex64Bytes[0] << 2) + ((Hex64Bytes[1] >> 4) & 0x03));
    Output.push_back((Hex64Bytes[1] << 4) + ((Hex64Bytes[2] >> 2) & 0x0f));
    Output.push_back((Hex64Bytes[2] << 6) + (Hex64Bytes[3] & 0x3f));
  }
  // If we had valid trailing '=' characters strip the right number of bytes
  // from the end of the output buffer. We already know that the Input length
  // it a multiple of 4 and is not zero, so direct character access is safe.
  if (Input.back() == '=') {
    Output.pop_back();
    if (Input[InputLength - 2] == '=')
      Output.pop_back();
  }
  return Error::success();
}

} // end namespace llvm

#endif
