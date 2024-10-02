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
#include <vector>

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

llvm::Error decodeBase64(llvm::StringRef Input, std::vector<char> &Output);

// General-purpose Base64 encoder/decoder class wrapper.
class Base64 {
public:
  using byte = std::byte;

  // Get the size of the encoded byte sequence of given size.
  static size_t getEncodedSize(size_t SrcSize);

  // Encode a byte sequence of given size into an output stream.
  // Returns the number of bytes in the encoded result.
  static size_t encode(const byte *Src, raw_ostream &Out, size_t SrcSize);

  // Get the size of the encoded byte sequence of given size.
  static size_t getDecodedSize(size_t SrcSize);

  // Decode a sequence of given size into a pre-allocated memory.
  // Returns the number of bytes in the decoded result or 0 in case of error.
  static Expected<size_t> decode(const char *Src, byte *Dst, size_t SrcSize);

  // Allocate minimum required amount of memory and decode a sequence of given
  // size into it.
  // Returns the decoded result. The size can be obtained via getDecodedSize.
  static Expected<std::unique_ptr<byte[]>> decode(const char *Src,
                                                  size_t SrcSize);
};

} // end namespace llvm

#endif
