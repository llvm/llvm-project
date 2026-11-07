//===- WindowsMachineFlagTest.cpp - Tests for WindowsMachineFlag.cpp ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/WindowsMachineFlag.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/COFF.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(WindowsMachineFlagTest, UnknownMachineTypes) {
  EXPECT_EQ("unknown", machineToStr(COFF::IMAGE_FILE_MACHINE_POWERPC));
  EXPECT_EQ("unknown", machineToStr(COFF::IMAGE_FILE_MACHINE_UNKNOWN));
}
