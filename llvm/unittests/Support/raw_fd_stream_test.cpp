//===- llvm/unittest/Support/raw_fd_stream_test.cpp - raw_fd_stream tests -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallString.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(raw_fd_streamTest, ReadAfterWrite) {
  SmallString<64> Path;
  int FD;
  ASSERT_FALSE(sys::fs::createTemporaryFile("foo", "bar", FD, Path));
  FileRemover Cleanup(Path);
  std::error_code EC;
  raw_fd_stream OS(Path, EC);
  EXPECT_TRUE(!EC);

  char Bytes[8];

  OS.write("01234567", 8);

  OS.seek(3);
  EXPECT_EQ(OS.read(Bytes, 2), 2);
  EXPECT_EQ(Bytes[0], '3');
  EXPECT_EQ(Bytes[1], '4');

  OS.seek(4);
  OS.write("xyz", 3);

  OS.seek(0);
  EXPECT_EQ(OS.read(Bytes, 8), 8);
  EXPECT_EQ(Bytes[0], '0');
  EXPECT_EQ(Bytes[1], '1');
  EXPECT_EQ(Bytes[2], '2');
  EXPECT_EQ(Bytes[3], '3');
  EXPECT_EQ(Bytes[4], 'x');
  EXPECT_EQ(Bytes[5], 'y');
  EXPECT_EQ(Bytes[6], 'z');
  EXPECT_EQ(Bytes[7], '7');
}

TEST(raw_fd_streamTest, DynCast) {
  {
    std::error_code EC;
    raw_fd_stream OS("-", EC);
    EXPECT_TRUE(dyn_cast<raw_fd_stream>(&OS));
  }
  {
    std::error_code EC;
    raw_fd_ostream OS("-", EC);
    EXPECT_FALSE(dyn_cast<raw_fd_stream>(&OS));
  }
}

} // namespace
