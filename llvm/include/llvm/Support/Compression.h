//===-- llvm/Support/Compression.h ---Compression----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains basic functions for compression/uncompression.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_COMPRESSION_H
#define LLVM_SUPPORT_COMPRESSION_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {
template <typename T> class SmallVectorImpl;
class Error;
class StringRef;

namespace compression {
namespace zlib {

constexpr int NoCompression = 0;
constexpr int BestSpeedCompression = 1;
constexpr int DefaultCompression = 6;
constexpr int BestSizeCompression = 9;

bool isAvailable();

void compress(ArrayRef<uint8_t> Input,
              SmallVectorImpl<uint8_t> &CompressedBuffer,
              int Level = DefaultCompression);

Error uncompress(ArrayRef<uint8_t> Input, uint8_t *UncompressedBuffer,
                 size_t &UncompressedSize);

Error uncompress(ArrayRef<uint8_t> Input,
                 SmallVectorImpl<uint8_t> &UncompressedBuffer,
                 size_t UncompressedSize);

} // End of namespace zlib

} // End of namespace compression

} // End of namespace llvm

#endif
