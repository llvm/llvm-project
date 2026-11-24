//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/WindowsMachineFlag.h"
#include "llvm/ADT/StringRef.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(WindowsMachineFlagTest, KnownMachineTypes) {
  EXPECT_EQ("arm", machineToStr(COFF::IMAGE_FILE_MACHINE_ARMNT));
  EXPECT_EQ("arm64", machineToStr(COFF::IMAGE_FILE_MACHINE_ARM64));
  EXPECT_EQ("arm64ec", machineToStr(COFF::IMAGE_FILE_MACHINE_ARM64EC));
  EXPECT_EQ("arm64x", machineToStr(COFF::IMAGE_FILE_MACHINE_ARM64X));
  EXPECT_EQ("x64", machineToStr(COFF::IMAGE_FILE_MACHINE_AMD64));
  EXPECT_EQ("x86", machineToStr(COFF::IMAGE_FILE_MACHINE_I386));
}

TEST(WindowsMachineFlagTest, UnknownMachineTypes) {
  EXPECT_EQ("unknown", machineToStr(COFF::IMAGE_FILE_MACHINE_POWERPC));
  EXPECT_EQ("unknown", machineToStr(COFF::IMAGE_FILE_MACHINE_UNKNOWN));
}
