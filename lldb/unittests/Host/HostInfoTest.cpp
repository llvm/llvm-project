//===-- HostTest.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/HostInfo.h"
#include "gtest/gtest.h"

using namespace lldb_private;
using namespace llvm;

namespace {
class HostInfoTest: public ::testing::Test {
  public:
    void SetUp() override { HostInfo::Initialize(); }
    void TearDown() override { HostInfo::Terminate(); }
};
}

TEST_F(HostInfoTest, GetAugmentedArchSpec) {
  // Fully specified triple should not be changed.
  ArchSpec spec = HostInfo::GetAugmentedArchSpec("x86_64-pc-linux-gnu");
  EXPECT_EQ(spec.GetTriple().getTriple(), "x86_64-pc-linux-gnu");

  // Same goes if we specify at least one of (os, vendor, env).
  spec = HostInfo::GetAugmentedArchSpec("x86_64-pc");
  EXPECT_EQ(spec.GetTriple().getTriple(), "x86_64-pc");

  // But if we specify only an arch, we should fill in the rest from the host.
  spec = HostInfo::GetAugmentedArchSpec("x86_64");
  Triple triple(sys::getDefaultTargetTriple());
  EXPECT_EQ(spec.GetTriple().getArch(), Triple::x86_64);
  EXPECT_EQ(spec.GetTriple().getOS(), triple.getOS());
  EXPECT_EQ(spec.GetTriple().getVendor(), triple.getVendor());
  EXPECT_EQ(spec.GetTriple().getEnvironment(), triple.getEnvironment());
}
