//===-- IntelPTCollectorTests.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "IntelPTCollector.h"
#include "llvm/ADT/ArrayRef.h"


using namespace lldb_private;
using namespace process_linux;

size_t ReadCylicBufferWrapper(void *buf, size_t buf_size, void *cyc_buf,
                              size_t cyc_buf_size, size_t cyc_start,
                              size_t offset) {
  llvm::MutableArrayRef<uint8_t> dst(reinterpret_cast<uint8_t *>(buf),
                                     buf_size);
  llvm::ArrayRef<uint8_t> src(reinterpret_cast<uint8_t *>(cyc_buf),
                              cyc_buf_size);
  IntelPTThreadTrace::ReadCyclicBuffer(dst, src, cyc_start, offset);
  return dst.size();
}

TEST(CyclicBuffer, EdgeCases) {
  size_t bytes_read;
  uint8_t cyclic_buffer[6] = {'l', 'i', 'c', 'c', 'y', 'c'};

  // We will always leave the last bytes untouched
  // so that string comparisons work.
  char smaller_buffer[4] = {};

  // empty buffer to read into
  bytes_read = ReadCylicBufferWrapper(smaller_buffer, 0, cyclic_buffer,
                                      sizeof(cyclic_buffer), 3, 0);
  ASSERT_EQ(0u, bytes_read);

  // empty cyclic buffer
  bytes_read = ReadCylicBufferWrapper(smaller_buffer, sizeof(smaller_buffer),
                                      cyclic_buffer, 0, 3, 0);
  ASSERT_EQ(0u, bytes_read);

  // bigger offset
  bytes_read =
      ReadCylicBufferWrapper(smaller_buffer, sizeof(smaller_buffer),
                             cyclic_buffer, sizeof(cyclic_buffer), 3, 6);
  ASSERT_EQ(0u, bytes_read);

  // wrong offset
  bytes_read =
      ReadCylicBufferWrapper(smaller_buffer, sizeof(smaller_buffer),
                             cyclic_buffer, sizeof(cyclic_buffer), 3, 7);
  ASSERT_EQ(0u, bytes_read);

  // wrong start
  bytes_read =
      ReadCylicBufferWrapper(smaller_buffer, sizeof(smaller_buffer),
                             cyclic_buffer, sizeof(cyclic_buffer), 3, 7);
  ASSERT_EQ(0u, bytes_read);
}

TEST(CyclicBuffer, EqualSizeBuffer) {
  size_t bytes_read = 0;
  uint8_t cyclic_buffer[6] = {'l', 'i', 'c', 'c', 'y', 'c'};

  char cyclic[] = "cyclic";
  for (size_t i = 0; i < sizeof(cyclic); i++) {
    // We will always leave the last bytes untouched
    // so that string comparisons work.
    char equal_size_buffer[7] = {};
    bytes_read =
        ReadCylicBufferWrapper(equal_size_buffer, sizeof(cyclic_buffer),
                               cyclic_buffer, sizeof(cyclic_buffer), 3, i);
    ASSERT_EQ((sizeof(cyclic) - i - 1), bytes_read);
    ASSERT_STREQ(equal_size_buffer, (cyclic + i));
  }
}

TEST(CyclicBuffer, SmallerSizeBuffer) {
  size_t bytes_read;
  uint8_t cyclic_buffer[6] = {'l', 'i', 'c', 'c', 'y', 'c'};

  // We will always leave the last bytes untouched
  // so that string comparisons work.
  char smaller_buffer[4] = {};
  bytes_read =
      ReadCylicBufferWrapper(smaller_buffer, (sizeof(smaller_buffer) - 1),
                             cyclic_buffer, sizeof(cyclic_buffer), 3, 0);
  ASSERT_EQ(3u, bytes_read);
  ASSERT_STREQ(smaller_buffer, "cyc");

  bytes_read =
      ReadCylicBufferWrapper(smaller_buffer, (sizeof(smaller_buffer) - 1),
                             cyclic_buffer, sizeof(cyclic_buffer), 3, 1);
  ASSERT_EQ(3u, bytes_read);
  ASSERT_STREQ(smaller_buffer, "ycl");

  bytes_read =
      ReadCylicBufferWrapper(smaller_buffer, (sizeof(smaller_buffer) - 1),
                             cyclic_buffer, sizeof(cyclic_buffer), 3, 2);
  ASSERT_EQ(3u, bytes_read);
  ASSERT_STREQ(smaller_buffer, "cli");

  bytes_read =
      ReadCylicBufferWrapper(smaller_buffer, (sizeof(smaller_buffer) - 1),
                             cyclic_buffer, sizeof(cyclic_buffer), 3, 3);
  ASSERT_EQ(3u, bytes_read);
  ASSERT_STREQ(smaller_buffer, "lic");

  {
    char smaller_buffer[4] = {};
    bytes_read =
        ReadCylicBufferWrapper(smaller_buffer, (sizeof(smaller_buffer) - 1),
                               cyclic_buffer, sizeof(cyclic_buffer), 3, 4);
    ASSERT_EQ(2u, bytes_read);
    ASSERT_STREQ(smaller_buffer, "ic");
  }
  {
    char smaller_buffer[4] = {};
    bytes_read =
        ReadCylicBufferWrapper(smaller_buffer, (sizeof(smaller_buffer) - 1),
                               cyclic_buffer, sizeof(cyclic_buffer), 3, 5);
    ASSERT_EQ(1u, bytes_read);
    ASSERT_STREQ(smaller_buffer, "c");
  }
}

TEST(CyclicBuffer, BiggerSizeBuffer) {
  size_t bytes_read = 0;
  uint8_t cyclic_buffer[6] = {'l', 'i', 'c', 'c', 'y', 'c'};

  char cyclic[] = "cyclic";
  for (size_t i = 0; i < sizeof(cyclic); i++) {
    // We will always leave the last bytes untouched
    // so that string comparisons work.
    char bigger_buffer[10] = {};
    bytes_read =
        ReadCylicBufferWrapper(bigger_buffer, (sizeof(bigger_buffer) - 1),
                               cyclic_buffer, sizeof(cyclic_buffer), 3, i);
    ASSERT_EQ((sizeof(cyclic) - i - 1), bytes_read);
    ASSERT_STREQ(bigger_buffer, (cyclic + i));
  }
}
