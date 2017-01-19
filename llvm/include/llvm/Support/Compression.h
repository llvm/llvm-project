//===-- llvm/Support/Compression.h ---Compression----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains basic functions for compression/uncompression.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_COMPRESSION_H
#define LLVM_SUPPORT_COMPRESSION_H

#include "llvm/Support/DataTypes.h"

namespace llvm {
template <typename T> class SmallVectorImpl;
class Error;
class StringRef;

namespace zlib {

enum CompressionLevel {
  NoCompression,
  DefaultCompression,
  BestSpeedCompression,
  BestSizeCompression
};

bool isAvailable();

Error compress(StringRef InputBuffer, SmallVectorImpl<char> &CompressedBuffer,
               CompressionLevel Level = DefaultCompression);

Error uncompress(StringRef InputBuffer, char *UncompressedBuffer,
                 size_t &UncompressedSize);

Error uncompress(StringRef InputBuffer,
                 SmallVectorImpl<char> &UncompressedBuffer,
                 size_t UncompressedSize);

uint32_t crc32(StringRef Buffer);

}  // End of namespace zlib

} // End of namespace llvm

#endif

