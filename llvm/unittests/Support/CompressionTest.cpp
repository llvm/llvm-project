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
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/config.h"
#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::compression;

namespace {

#if LLVM_ENABLE_ZLIB
static void testZlibCompression(StringRef Input, int Level) {
  SmallVector<uint8_t, 0> Compressed;
  SmallVector<uint8_t, 0> Uncompressed;
  zlib::compress(arrayRefFromStringRef(Input), Compressed, Level);

  // Check that uncompressed buffer is the same as original.
  Error E = zlib::decompress(Compressed, Uncompressed, Input.size());
  EXPECT_FALSE(std::move(E));
  EXPECT_EQ(Input, toStringRef(Uncompressed));

  // decompress with Z dispatches to zlib::decompress.
  E = compression::decompress(DebugCompressionType::Zlib, Compressed,
                              Uncompressed, Input.size());
  EXPECT_FALSE(std::move(E));
  EXPECT_EQ(Input, toStringRef(Uncompressed));

  // decompress infers zlib from the RFC 1950 header.
  E = compression::decompress(Compressed, Uncompressed, Input.size());
  EXPECT_FALSE(std::move(E));
  EXPECT_EQ(Input, toStringRef(Uncompressed));

  if (Input.size() > 0) {
    // Decompression fails if expected length is too short.
    E = zlib::decompress(Compressed, Uncompressed, Input.size() - 1);
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

#if LLVM_ENABLE_ZSTD
static void testZstdCompression(StringRef Input, int Level) {
  SmallVector<uint8_t, 0> Compressed;
  SmallVector<uint8_t, 0> Uncompressed;
  zstd::compress(arrayRefFromStringRef(Input), Compressed, Level);

  // Check that uncompressed buffer is the same as original.
  Error E = zstd::decompress(Compressed, Uncompressed, Input.size());
  EXPECT_FALSE(std::move(E));
  EXPECT_EQ(Input, toStringRef(Uncompressed));

  // decompress with Zstd dispatches to zstd::decompress.
  E = compression::decompress(DebugCompressionType::Zstd, Compressed,
                              Uncompressed, Input.size());
  EXPECT_FALSE(std::move(E));
  EXPECT_EQ(Input, toStringRef(Uncompressed));

  // decompress infers Zstd from the frame magic.
  E = compression::decompress(Compressed, Uncompressed, Input.size());
  EXPECT_FALSE(std::move(E));
  EXPECT_EQ(Input, toStringRef(Uncompressed));

  if (Input.size() > 0) {
    // Decompression fails if expected length is too short.
    E = zstd::decompress(Compressed, Uncompressed, Input.size() - 1);
    EXPECT_EQ("Destination buffer is too small", llvm::toString(std::move(E)));
  }
}

TEST(CompressionTest, Zstd) {
  testZstdCompression("", zstd::DefaultCompression);

  testZstdCompression("hello, world!", zstd::NoCompression);
  testZstdCompression("hello, world!", zstd::BestSizeCompression);
  testZstdCompression("hello, world!", zstd::BestSpeedCompression);
  testZstdCompression("hello, world!", zstd::DefaultCompression);

  const size_t kSize = 1024;
  char BinaryData[kSize];
  for (size_t i = 0; i < kSize; ++i)
    BinaryData[i] = i & 255;
  StringRef BinaryDataStr(BinaryData, kSize);

  testZstdCompression(BinaryDataStr, zstd::NoCompression);
  testZstdCompression(BinaryDataStr, zstd::BestSizeCompression);
  testZstdCompression(BinaryDataStr, zstd::BestSpeedCompression);
  testZstdCompression(BinaryDataStr, zstd::DefaultCompression);
}
#endif

TEST(CompressionTest, IdentifyHeaders) {
  EXPECT_STREQ("unknown compression format",
               getReasonIfUnsupported(ArrayRef<uint8_t>()));
  uint8_t Truncated[] = {0x78};
  EXPECT_STREQ("unknown compression format", getReasonIfUnsupported(Truncated));

  // RFC 1950 headers LLVM's compress2 does not emit.
  uint8_t SmallWindow[] = {0x28, 0x15}; // CINFO=2, FCHECK valid
  EXPECT_EQ(getReasonIfUnsupported(Format::Zlib),
            getReasonIfUnsupported(ArrayRef<uint8_t>(SmallWindow)));
  uint8_t WithDict[] = {0x78, 0x20}; // FDICT set, FCHECK valid
  EXPECT_EQ(getReasonIfUnsupported(Format::Zlib),
            getReasonIfUnsupported(ArrayRef<uint8_t>(WithDict)));

  uint8_t BadFCheck[] = {0x78, 0x00};
  EXPECT_STREQ("unknown compression format", getReasonIfUnsupported(BadFCheck));
  uint8_t BadCINFO[] = {0x88, 0x01};
  EXPECT_STREQ("unknown compression format", getReasonIfUnsupported(BadCINFO));
  uint8_t BadCM[] = {0x79, 0x9c};
  EXPECT_STREQ("unknown compression format", getReasonIfUnsupported(BadCM));

  uint8_t ZstdMagic[] = {0x28, 0xb5, 0x2f, 0xfd};
  EXPECT_EQ(getReasonIfUnsupported(Format::Zstd),
            getReasonIfUnsupported(ArrayRef<uint8_t>(ZstdMagic)));

  uint8_t Unknown[] = {0x00, 0x01, 0x02, 0x03};
  EXPECT_STREQ("unknown compression format", getReasonIfUnsupported(Unknown));
  SmallVector<uint8_t, 0> Out;
  Error E = compression::decompress(ArrayRef<uint8_t>(Unknown), Out, 0);
  EXPECT_EQ("unknown compression format", toString(std::move(E)));
}
} // namespace
