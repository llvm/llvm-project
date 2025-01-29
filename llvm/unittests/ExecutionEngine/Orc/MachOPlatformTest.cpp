//===---------- MachOPlatformTest.cpp - MachPlatform API Tests ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/MachOPlatform.h"
#include "llvm/BinaryFormat/MachO.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;

TEST(MachOPlatformTests, BuildVersionOptsFromTriple) {

  auto darwinOS = MachOPlatform::HeaderOptions::BuildVersionOpts::fromTriple(
      Triple("arm64-apple-darwin"), 0, 0);
  EXPECT_FALSE(darwinOS);

  auto macOS = MachOPlatform::HeaderOptions::BuildVersionOpts::fromTriple(
      Triple("arm64-apple-macosx"), 0, 0);
  EXPECT_TRUE(macOS);
  EXPECT_EQ(macOS->Platform, MachO::PLATFORM_MACOS);

  auto iOS = MachOPlatform::HeaderOptions::BuildVersionOpts::fromTriple(
      Triple("arm64-apple-ios"), 0, 0);
  EXPECT_TRUE(iOS);
  EXPECT_EQ(iOS->Platform, MachO::PLATFORM_IOS);

  auto iOSSim = MachOPlatform::HeaderOptions::BuildVersionOpts::fromTriple(
      Triple("arm64-apple-ios-simulator"), 0, 0);
  EXPECT_TRUE(iOSSim);
  EXPECT_EQ(iOSSim->Platform, MachO::PLATFORM_IOSSIMULATOR);

  auto tvOS = MachOPlatform::HeaderOptions::BuildVersionOpts::fromTriple(
      Triple("arm64-apple-tvos"), 0, 0);
  EXPECT_TRUE(tvOS);
  EXPECT_EQ(tvOS->Platform, MachO::PLATFORM_TVOS);

  auto tvOSSim = MachOPlatform::HeaderOptions::BuildVersionOpts::fromTriple(
      Triple("arm64-apple-tvos-simulator"), 0, 0);
  EXPECT_TRUE(tvOSSim);
  EXPECT_EQ(tvOSSim->Platform, MachO::PLATFORM_TVOSSIMULATOR);

  auto watchOS = MachOPlatform::HeaderOptions::BuildVersionOpts::fromTriple(
      Triple("arm64-apple-watchos"), 0, 0);
  EXPECT_TRUE(watchOS);
  EXPECT_EQ(watchOS->Platform, MachO::PLATFORM_WATCHOS);

  auto watchOSSim = MachOPlatform::HeaderOptions::BuildVersionOpts::fromTriple(
      Triple("arm64-apple-watchos-simulator"), 0, 0);
  EXPECT_TRUE(watchOSSim);
  EXPECT_EQ(watchOSSim->Platform, MachO::PLATFORM_WATCHOSSIMULATOR);
}
