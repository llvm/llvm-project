//===- llvm/unittest/Support/CompressionTest.cpp - Compression tests ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements unit tests for the Compression functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Compression.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/config.h"
#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::compression;

namespace {

#if LLVM_ENABLE_ZLIB
static void testZlibCompression(StringRef Input, int Level) {
  SmallString<32> Compressed;
  SmallString<32> Uncompressed;
  zlib::compress(Input, Compressed, Level);

  // Check that uncompressed buffer is the same as original.
  Error E = zlib::uncompress(Compressed, Uncompressed, Input.size());
  consumeError(std::move(E));

  EXPECT_EQ(Input, Uncompressed);
  if (Input.size() > 0) {
    // Uncompression fails if expected length is too short.
    E = zlib::uncompress(Compressed, Uncompressed, Input.size() - 1);
    EXPECT_EQ("zlib error: Z_BUF_ERROR", llvm::toString(std::move(E)));
  }
}

TEST(CompressionTest, Zlib) {
  testZlibCompression("", zlib::DefaultCompression);

  testZlibCompression("hello, world!", zlib::NoCompression);
  testZlibCompression("hello, world!", zlib::BestSizeCompression);
  testZlibCompression("hello, world!", zlib::BestSpeedCompression);
  testZlibCompression("hello, world!", zlib::DefaultCompression);

  const size_t kSize = 1024;
  char BinaryData[kSize];
  for (size_t i = 0; i < kSize; ++i)
    BinaryData[i] = i & 255;
  StringRef BinaryDataStr(BinaryData, kSize);

  testZlibCompression(BinaryDataStr, zlib::NoCompression);
  testZlibCompression(BinaryDataStr, zlib::BestSizeCompression);
  testZlibCompression(BinaryDataStr, zlib::BestSpeedCompression);
  testZlibCompression(BinaryDataStr, zlib::DefaultCompression);
}
#endif

}
