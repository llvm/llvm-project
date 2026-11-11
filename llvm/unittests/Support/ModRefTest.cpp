//===- llvm/unittest/Support/ModRefTest.cpp - ModRef tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ModRef.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <string>

using namespace llvm;

namespace {

// Verify that printing a MemoryEffects does not end with a ,.
TEST(ModRefTest, PrintMemoryEffects) {
  std::string S;
  raw_string_ostream OS(S);
  OS << MemoryEffects::none();
  EXPECT_EQ(S, "ArgMem: NoModRef, InaccessibleMem: NoModRef, ErrnoMem: "
               "NoModRef, Other: NoModRef, ThreadIdMem: NoModRef, TargetMem0: "
               "NoModRef, TargetMem1: NoModRef");
}

TEST(ModRefTest, ThreadIdMem) {
  MemoryEffects ME = MemoryEffects::threadIdMemOnly(ModRefInfo::Ref);
  EXPECT_EQ(ME.getModRef(IRMemLocation::ThreadIdMem), ModRefInfo::Ref);
  EXPECT_EQ(ME.getModRef(IRMemLocation::Other), ModRefInfo::NoModRef);
  EXPECT_EQ(ME.getModRef(IRMemLocation::TargetMem0), ModRefInfo::NoModRef);
  EXPECT_EQ(ME.getModRef(IRMemLocation::TargetMem1), ModRefInfo::NoModRef);
  EXPECT_TRUE(ME.onlyReadsMemory());
  EXPECT_FALSE(ME.onlyAccessesInaccessibleMem());

  EXPECT_EQ(MemoryEffects::readOnly().getModRef(IRMemLocation::ThreadIdMem),
            ModRefInfo::Ref);
  EXPECT_EQ(MemoryEffects::argMemOnly().getModRef(IRMemLocation::ThreadIdMem),
            ModRefInfo::NoModRef);
  EXPECT_EQ(MemoryEffects::inaccessibleMemOnly().getModRef(
                IRMemLocation::ThreadIdMem),
            ModRefInfo::NoModRef);
}

} // namespace
