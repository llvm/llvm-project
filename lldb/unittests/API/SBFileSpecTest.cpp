//===-- SBFileSpecTest.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===/

// Use the umbrella header for -Wdocumentation.
#include "lldb/API/LLDB.h"

#include "lldb/API/SBFileSpec.h"
#include "gtest/gtest.h"

#include <cstring>
#include <string>

TEST(SBFileSpecTest, GetPath) {
  const std::string path = "/tmp/lldb-sbfilespec-test/file.txt";
  const size_t needed_len = path.size() + 1; // including NULL byte.

  lldb::SBFileSpec fs(path.c_str(), /*resolve=*/false);
  ASSERT_TRUE(fs.IsValid());

  // Verify large buffer returns needed_len and fills the buffer.
  char buf[256];
  constexpr size_t buf_size = sizeof(buf);
  std::memset(buf, 'X', buf_size);

  ASSERT_GE(buf_size, needed_len);
  EXPECT_EQ(fs.GetPath(buf, buf_size), needed_len);
  EXPECT_STREQ(buf, path.c_str());

  // Verify querying path returns the size needed without writing.
  EXPECT_EQ(fs.GetPath(nullptr, 0), needed_len);

  // Verify smaller buffer returns the full needed size (including the null
  // byte) and the buffer is truncated and NUL-terminated.
  char small_buf[8];
  constexpr size_t small_buf_size = sizeof(small_buf);
  std::memset(small_buf, 'X', small_buf_size);
  EXPECT_EQ(fs.GetPath(small_buf, small_buf_size), needed_len);
  EXPECT_EQ(small_buf[small_buf_size - 1], '\0');
  EXPECT_EQ(std::strncmp(small_buf, path.c_str(), small_buf_size - 1), 0);

  // Verify empty filespec returns 0 and NUL-terminates the buffer.
  lldb::SBFileSpec empty_fs;
  char empty_buf[16];
  std::memset(empty_buf, 'X', sizeof(empty_buf));
  EXPECT_EQ(empty_fs.GetPath(empty_buf, sizeof(empty_buf)), 0U);
  EXPECT_EQ(empty_buf[0], '\0');
}
