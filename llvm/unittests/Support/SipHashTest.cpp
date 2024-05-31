//===- llvm/unittest/Support/SipHashTest.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/SipHash.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

TEST(SipHashTest, PointerAuthSipHash) {
  // Test some basic cases, for 16 bit and 64 bit results.
  EXPECT_EQ(0xE793U, getPointerAuthStableSipHash16(""));
  EXPECT_EQ(0xF468U, getPointerAuthStableSipHash16("strlen"));
  EXPECT_EQ(0x2D15U, getPointerAuthStableSipHash16("_ZN1 ind; f"));

  EXPECT_EQ(0xB2BB69BB0A2AC0F1U, getPointerAuthStableSipHash64(""));
  EXPECT_EQ(0x9304ABFF427B72E8U, getPointerAuthStableSipHash64("strlen"));
  EXPECT_EQ(0x55F45179A08AE51BU, getPointerAuthStableSipHash64("_ZN1 ind; f"));

  // Test some known strings that are already enshrined in the ABI.
  EXPECT_EQ(0x6AE1U, getPointerAuthStableSipHash16("isa"));
  EXPECT_EQ(0xB5ABU, getPointerAuthStableSipHash16("objc_class:superclass"));
  EXPECT_EQ(0xC0BBU, getPointerAuthStableSipHash16("block_descriptor"));
  EXPECT_EQ(0xC310U, getPointerAuthStableSipHash16("method_list_t"));

  // Test the limits that apply to 16 bit results but don't to 64 bit results.
  EXPECT_EQ(1U,                  getPointerAuthStableSipHash16("_Zptrkvttf"));
  EXPECT_EQ(0x314FD87E0611F020U, getPointerAuthStableSipHash64("_Zptrkvttf"));

  EXPECT_EQ(0xFFFFU,             getPointerAuthStableSipHash16("_Zaflhllod"));
  EXPECT_EQ(0x1292F635FB3DFBF8U, getPointerAuthStableSipHash64("_Zaflhllod"));
}

} // end anonymous namespace
