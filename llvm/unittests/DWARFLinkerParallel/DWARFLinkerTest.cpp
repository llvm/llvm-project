//===- llvm/unittest/DWARFLinkerParallel/DWARFLinkerTest.cpp --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DWARFLinker/Utils.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace dwarf_linker;

#define DEVELOPER_DIR "/Applications/Xcode.app/Contents/Developer"

namespace {

TEST(DWARFLinker, PathTest) {
  EXPECT_EQ(guessDeveloperDir("/Foo"), "");
  EXPECT_EQ(guessDeveloperDir("Foo.sdk"), "");
  EXPECT_EQ(guessDeveloperDir(
                DEVELOPER_DIR
                "/Platforms/MacOSX.platform/Developer/SDKs/MacOSX14.4.sdk"),
            DEVELOPER_DIR);
  EXPECT_EQ(guessDeveloperDir(DEVELOPER_DIR "/SDKs/MacOSX.sdk"), DEVELOPER_DIR);
}

} // anonymous namespace
