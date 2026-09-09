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
#include "llvm/Testing/Support/Error.h"
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

#if LLVM_ENABLE_LZMA

/// `xz --check=crc32 -9` of the empty string. LLVM implements xz decompression
/// but not compression, so the fixtures are literals rather than round trips.
static constexpr uint8_t XzEmptyData[] = {
    0xfd, 0x37, 0x7a, 0x58, 0x5a, 0x00, 0x00, 0x01, 0x69, 0x22, 0xde,
    0x36, 0x00, 0x00, 0x00, 0x00, 0x1c, 0xdf, 0x44, 0x21, 0x90, 0x42,
    0x99, 0x0d, 0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x59, 0x5a,
};

/// `xz --check=crc32 -9` of "hello, world!".
static constexpr uint8_t XzTextData[] = {
    0xfd, 0x37, 0x7a, 0x58, 0x5a, 0x00, 0x00, 0x01, 0x69, 0x22, 0xde, 0x36,
    0x02, 0x00, 0x21, 0x01, 0x1c, 0x00, 0x00, 0x00, 0x10, 0xcf, 0x58, 0xcc,
    0x01, 0x00, 0x0c, 0x68, 0x65, 0x6c, 0x6c, 0x6f, 0x2c, 0x20, 0x77, 0x6f,
    0x72, 0x6c, 0x64, 0x21, 0x00, 0x00, 0x00, 0x00, 0x13, 0x8d, 0x98, 0x58,
    0x00, 0x01, 0x21, 0x0d, 0x75, 0xdc, 0xa8, 0xd2, 0x90, 0x42, 0x99, 0x0d,
    0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x59, 0x5a,
};

// A stream is a header, the blocks, the index, and a footer of the same size.
static constexpr size_t XzStreamHeaderSize = 12;
static constexpr size_t XzStreamFooterSize = 12;

static void testXzDecompression(ArrayRef<uint8_t> Compressed,
                                StringRef Expected) {
  // The uncompressed size comes from the stream index, not from the caller.
  SmallVector<uint8_t, 0> Uncompressed;
  ASSERT_THAT_ERROR(xz::decompress(Compressed, Uncompressed), Succeeded());
  EXPECT_EQ(Expected, toStringRef(Uncompressed));
}

static std::string xzDecompressError(ArrayRef<uint8_t> Input) {
  // Prefilled, both to prove a failure empties it and so that no case can
  // observe what an earlier one left behind.
  SmallVector<uint8_t, 0> Output(8, 0xaa);
  std::string Message = llvm::toString(xz::decompress(Input, Output));
  EXPECT_TRUE(Output.empty());
  return Message;
}

TEST(CompressionTest, Xz) {
  EXPECT_TRUE(xz::isAvailable());

  testXzDecompression(XzEmptyData, "");
  testXzDecompression(XzTextData, "hello, world!");
}

TEST(CompressionTest, XzDecompressErrors) {
  ArrayRef<uint8_t> Compressed(XzTextData);
  auto FlipByte = [&](size_t Offset) {
    SmallVector<uint8_t, 0> Corrupt(Compressed);
    Corrupt[Offset] ^= 0xff;
    return Corrupt;
  };

  // Too small to hold a stream header.
  EXPECT_EQ("size of xz-compressed blob (4 bytes) is smaller than the "
            "LZMA_STREAM_HEADER_SIZE (12 bytes)",
            xzDecompressError(Compressed.take_front(4)));

  // The footer recording where the index lives is gone.
  EXPECT_EQ("lzma_stream_footer_decode()=lzma error: LZMA_FORMAT_ERROR",
            xzDecompressError(Compressed.drop_back(4)));

  // The footer is intact but the index it points back to has been cut off.
  EXPECT_EQ("xz-compressed buffer size (12 bytes) too small (required at "
            "least 32 bytes)",
            xzDecompressError(Compressed.take_back(XzStreamFooterSize)));

  // The index's CRC32, in the four bytes just before the footer, is corrupt.
  size_t IndexCrcOffset = Compressed.size() - XzStreamFooterSize - 1;
  EXPECT_EQ("lzma_index_buffer_decode()=lzma error: LZMA_DATA_ERROR",
            xzDecompressError(FlipByte(IndexCrcOffset)));

  // The payload is corrupt. It follows the stream header and the block header,
  // whose length in four-byte units is held in its first byte.
  size_t BlockHeaderSize = (Compressed[XzStreamHeaderSize] + 1) * 4;
  EXPECT_EQ("lzma_stream_buffer_decode()=lzma error: LZMA_DATA_ERROR",
            xzDecompressError(FlipByte(XzStreamHeaderSize + BlockHeaderSize)));
}
#endif
} // namespace
