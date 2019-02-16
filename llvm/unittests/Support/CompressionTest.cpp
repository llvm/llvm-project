//===- llvm/unittest/Support/CompressionTest.cpp - Compression tests ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

namespace {

#if LLVM_ENABLE_ZLIB == 1 && HAVE_LIBZ

void TestZlibCompression(StringRef Input, int Level) {
  SmallString<32> Compressed;
  SmallString<32> Uncompressed;

  Error E = zlib::compress(Input, Compressed, Level);
  EXPECT_FALSE(E);
  consumeError(std::move(E));

  // Check that uncompressed buffer is the same as original.
  E = zlib::uncompress(Compressed, Uncompressed, Input.size());
  EXPECT_FALSE(E);
  consumeError(std::move(E));

  EXPECT_EQ(Input, Uncompressed);
  if (Input.size() > 0) {
    // Uncompression fails if expected length is too short.
    E = zlib::uncompress(Compressed, Uncompressed, Input.size() - 1);
    EXPECT_EQ("zlib error: Z_BUF_ERROR", llvm::toString(std::move(E)));
  }
}

TEST(CompressionTest, Zlib) {
  TestZlibCompression("", zlib::DefaultCompression);

  TestZlibCompression("hello, world!", zlib::NoCompression);
  TestZlibCompression("hello, world!", zlib::BestSizeCompression);
  TestZlibCompression("hello, world!", zlib::BestSpeedCompression);
  TestZlibCompression("hello, world!", zlib::DefaultCompression);

  const size_t kSize = 1024;
  char BinaryData[kSize];
  for (size_t i = 0; i < kSize; ++i) {
    BinaryData[i] = i & 255;
  }
  StringRef BinaryDataStr(BinaryData, kSize);

  TestZlibCompression(BinaryDataStr, zlib::NoCompression);
  TestZlibCompression(BinaryDataStr, zlib::BestSizeCompression);
  TestZlibCompression(BinaryDataStr, zlib::BestSpeedCompression);
  TestZlibCompression(BinaryDataStr, zlib::DefaultCompression);
}

TEST(CompressionTest, ZlibCRC32) {
  EXPECT_EQ(
      0x414FA339U,
      zlib::crc32(StringRef("The quick brown fox jumps over the lazy dog")));
}

#endif

}
