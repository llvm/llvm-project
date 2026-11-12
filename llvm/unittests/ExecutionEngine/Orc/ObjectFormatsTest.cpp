//===-------- ObjectLinkingLayerTest.cpp - ObjectLinkingLayer tests -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Shared/ObjectFormats.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::orc;

namespace {

TEST(ObjectFormatsTest, MachOInitializerSections) {
  EXPECT_TRUE(isMachOInitializerSection("__DATA,__objc_selrefs"));
  EXPECT_TRUE(isMachOInitializerSection("__DATA,__mod_init_func"));
  EXPECT_TRUE(isMachOInitializerSection("__DATA,__objc_classlist"));
  EXPECT_TRUE(isMachOInitializerSection("__TEXT,__swift5_proto"));
  EXPECT_TRUE(isMachOInitializerSection("__TEXT,__swift5_protos"));
  EXPECT_TRUE(isMachOInitializerSection("__TEXT,__swift5_types"));
  EXPECT_FALSE(isMachOInitializerSection("__DATA,__not_an_init_sec"));
}

TEST(ObjectFormatsTest, ELFInitializerSections) {
  EXPECT_TRUE(isELFInitializerSection(".init_array"));
  EXPECT_TRUE(isELFInitializerSection(".init_array.0"));
  EXPECT_FALSE(isELFInitializerSection(".text"));
  EXPECT_TRUE(isELFInitializerSection(".ctors.0"));
}

} // end anonymous namespace
