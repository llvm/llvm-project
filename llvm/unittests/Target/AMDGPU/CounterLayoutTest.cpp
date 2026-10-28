//===- CounterLayoutTest.cpp - Backend vs TargetParser counter layout -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUUnitTests.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"
#include "gtest/gtest.h"

using namespace llvm;

// The backend reads the layout from subtarget feature bits, TargetParser from
// its generated GPU table. Both must agree on every GPU.
TEST_F(AMDGPUTestBase, CounterLayoutMatchesTargetParser) {
  Triple TT("amdgcn-amd-amdhsa");
  std::string Err;
  const Target *T = TargetRegistry::lookupTarget(TT, Err);
  ASSERT_TRUE(T) << Err;

  SmallVector<StringRef> CPUs;
  AMDGPU::fillValidArchListAMDGCN(CPUs);
  CPUs.append({"generic", "generic-hsa"});
  for (StringRef CPU : CPUs) {
    std::unique_ptr<MCSubtargetInfo> STI(T->createMCSubtargetInfo(TT, CPU, ""));
    ASSERT_TRUE(STI) << CPU;
    EXPECT_EQ(AMDGPU::getCounterLayout(*STI),
              AMDGPU::getCounterLayout(AMDGPU::parseArchAMDGCN(CPU)))
        << CPU;
  }
}
