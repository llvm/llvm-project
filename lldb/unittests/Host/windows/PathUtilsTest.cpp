//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/windows/PathUtils.h"
#include "gtest/gtest.h"

using namespace lldb_private;

TEST(PathUtilsTest, StripExtendedLengthPrefixDrive) {
  EXPECT_EQ("C:\\a\\b.exe", StripExtendedLengthPrefix("\\\\?\\C:\\a\\b.exe"));
}

TEST(PathUtilsTest, StripExtendedLengthPrefixUNC) {
  EXPECT_EQ("\\\\server\\share\\a.dll",
            StripExtendedLengthPrefix("\\\\?\\UNC\\server\\share\\a.dll"));
}

TEST(PathUtilsTest, StripExtendedLengthPrefixUnchanged) {
  EXPECT_EQ("C:\\a\\b.exe", StripExtendedLengthPrefix("C:\\a\\b.exe"));
  EXPECT_EQ("\\\\server\\share\\a.dll",
            StripExtendedLengthPrefix("\\\\server\\share\\a.dll"));
  EXPECT_EQ("", StripExtendedLengthPrefix(""));
}
