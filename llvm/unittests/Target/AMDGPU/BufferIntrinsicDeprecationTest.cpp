//===- BufferIntrinsicDeprecationTest.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test to ensure we don't direct users to an intrinsic that doesn't exist when
// warning about deprecated <4 x i32> intrinsics in AMDGPULowerIntrinsics
//
//===----------------------------------------------------------------------===//

#include "AMDGPUInstrInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "gtest/gtest.h"

using namespace llvm;

static bool isPtrBufferIntrinsic(Intrinsic::ID ID, StringRef Name) {
  if (ID == Intrinsic::amdgcn_s_buffer_prefetch_data)
    return true;
  return Name.contains(".ptr.");
}

namespace {

TEST(AMDGPUBufferIntrinsicDeprecation,
     PtrReplacementNamesAreTotalAndInjective) {
  unsigned NumDeprecated = 0, NumPtr = 0;
  DenseSet<unsigned> Replacements;

  for (unsigned ID = 1; ID < Intrinsic::num_intrinsics; ++ID) {
    const AMDGPU::RsrcIntrinsic *RI = AMDGPU::lookupRsrcIntrinsic(ID);
    // No image fat pointers yet.
    if (!RI || RI->IsImage)
      continue;

    StringRef Name = Intrinsic::getBaseName(ID);
    if (isPtrBufferIntrinsic(ID, Name)) {
      ++NumPtr;
      continue;
    }
    ++NumDeprecated;

    std::string Repl = AMDGPU::getPtrBufferIntrinsicName(Name);
    ASSERT_FALSE(Repl.empty()) << "no replacement derived for " << Name;

    unsigned ReplID = Intrinsic::lookupIntrinsicID(Repl);
    ASSERT_NE(ReplID, Intrinsic::not_intrinsic)
        << Name << " -> " << Repl << " is not an intrinsic";

    const AMDGPU::RsrcIntrinsic *ReplRI = AMDGPU::lookupRsrcIntrinsic(ReplID);
    ASSERT_TRUE(ReplRI) << Repl << " is not a resource intrinsic";
    EXPECT_FALSE(ReplRI->IsImage) << Repl << " is an image intrinsic";
    EXPECT_EQ(ReplRI->RsrcArg, RI->RsrcArg)
        << Name << " -> " << Repl << " moves the resource operand";

    EXPECT_TRUE(Replacements.insert(ReplID).second)
        << Repl << " is the replacement for more than one intrinsic";
  }

  EXPECT_EQ(NumDeprecated, 55u);
  EXPECT_EQ(NumPtr, 56u);
}

TEST(AMDGPUBufferIntrinsicDeprecation, ReplacementNameSpellings) {
  EXPECT_EQ(AMDGPU::getPtrBufferIntrinsicName("llvm.amdgcn.s.buffer.load"),
            "llvm.amdgcn.ptr.s.buffer.load");
  EXPECT_EQ(AMDGPU::getPtrBufferIntrinsicName("llvm.amdgcn.raw.buffer.load"),
            "llvm.amdgcn.raw.ptr.buffer.load");
  EXPECT_EQ(
      AMDGPU::getPtrBufferIntrinsicName("llvm.amdgcn.struct.buffer.store"),
      "llvm.amdgcn.struct.ptr.buffer.store");

  EXPECT_EQ(
      AMDGPU::getPtrBufferIntrinsicName("llvm.amdgcn.raw.atomic.buffer.load"),
      "llvm.amdgcn.raw.ptr.atomic.buffer.load");

  EXPECT_EQ(AMDGPU::getPtrBufferIntrinsicName("llvm.amdgcn.image.load.1d"), "");
}

} // end anonymous namespace
