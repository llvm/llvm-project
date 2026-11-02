//===- CounterLayoutTest.cpp - Wait counter layout per GPU ----------------===//
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
#include "gtest/gtest.h"

using namespace llvm;

static AMDGPU::CounterLayout getLayout(StringRef CPU) {
  Triple TT("amdgcn-amd-amdhsa");
  std::string Err;
  const Target *T = TargetRegistry::lookupTarget(TT, Err);
  std::unique_ptr<MCSubtargetInfo> STI(T->createMCSubtargetInfo(TT, CPU, ""));
  return AMDGPU::getCounterLayout(*STI);
}

TEST_F(AMDGPUTestBase, CounterLayoutLegacy) {
  // Masks: vmcnt, lgkmcnt, s_waitcnt, storecnt, holdcnt.
  auto Check = [](StringRef CPU, unsigned Vm, unsigned Lgkm, unsigned Waitcnt,
                  unsigned Store, unsigned Hold) {
    SCOPED_TRACE(CPU);
    AMDGPU::CounterLayout L = getLayout(CPU);
    EXPECT_EQ(AMDGPU::getVmcntBitMask(L), Vm);
    EXPECT_EQ(AMDGPU::getExpcntBitMask(L), 7u);
    EXPECT_EQ(AMDGPU::getLgkmcntBitMask(L), Lgkm);
    EXPECT_EQ(AMDGPU::getWaitcntBitMask(L), Waitcnt);
    EXPECT_EQ(AMDGPU::getStorecntBitMask(L), Store);
    EXPECT_EQ(AMDGPU::DepCtr::getHoldCntBitMask(L), Hold);
    EXPECT_EQ(AMDGPU::getLoadcntBitMask(L), 0u);
    EXPECT_EQ(AMDGPU::getBarrierMemberCountBitMask(L), 63u);
  };
  Check("gfx600", 15, 15, 0xF7F, 0, 0);
  Check("gfx900", 63, 15, 0xCF7F, 0, 0);
  Check("gfx1010", 63, 63, 0xFF7F, 63, 0);
  Check("gfx1030", 63, 63, 0xFF7F, 63, 1);
  Check("gfx1100", 63, 63, 0xFFF7, 63, 1);

  // Pseudo targets get the SI layout.
  Check("generic", 15, 15, 0xF7F, 0, 0);
  Check("generic-hsa", 15, 15, 0xF7F, 0, 0);
}

TEST_F(AMDGPUTestBase, CounterLayoutExtended) {
  // gfx12+: loadcnt, samplecnt, bvhcnt, dscnt, kmcnt, xcnt, asynccnt, barrier.
  auto Check = [](StringRef CPU, unsigned X, unsigned Async, unsigned Barrier) {
    SCOPED_TRACE(CPU);
    AMDGPU::CounterLayout L = getLayout(CPU);
    EXPECT_EQ(AMDGPU::getLoadcntBitMask(L), 63u);
    EXPECT_EQ(AMDGPU::getSamplecntBitMask(L), 63u);
    EXPECT_EQ(AMDGPU::getBvhcntBitMask(L), 7u);
    EXPECT_EQ(AMDGPU::getDscntBitMask(L), 63u);
    EXPECT_EQ(AMDGPU::getKmcntBitMask(L), 31u);
    EXPECT_EQ(AMDGPU::getStorecntBitMask(L), 63u);
    EXPECT_EQ(AMDGPU::getXcntBitMask(L), X);
    EXPECT_EQ(AMDGPU::getAsynccntBitMask(L), Async);
    EXPECT_EQ(AMDGPU::getBarrierMemberCountBitMask(L), Barrier);
    EXPECT_EQ(AMDGPU::DepCtr::getHoldCntBitMask(L), 1u);
  };
  Check("gfx1200", 0, 0, 63);
  Check("gfx1250", 63, 63, 63);
  Check("gfx1310", 0, 0, 255);
}

TEST_F(AMDGPUTestBase, CounterLayoutEncoding) {
  // gfx9: vmcnt[3:0] and [15:14], expcnt[6:4], lgkmcnt[11:8].
  AMDGPU::CounterLayout GFX9 = getLayout("gfx900");
  unsigned Enc = AMDGPU::encodeWaitcnt(GFX9, 42, 5, 9);
  EXPECT_EQ(Enc, 0x895Au);
  unsigned Vm, Exp, Lgkm;
  AMDGPU::decodeWaitcnt(GFX9, Enc, Vm, Exp, Lgkm);
  EXPECT_EQ(Vm, 42u);
  EXPECT_EQ(Exp, 5u);
  EXPECT_EQ(Lgkm, 9u);

  // gfx11: vmcnt[15:10], lgkmcnt[9:4], expcnt[2:0].
  AMDGPU::CounterLayout GFX11 = getLayout("gfx1100");
  EXPECT_EQ(AMDGPU::encodeWaitcnt(GFX11, 42, 5, 9), 0xA895u);

  // gfx12: loadcnt/storecnt[13:8], dscnt[5:0].
  AMDGPU::CounterLayout GFX12 = getLayout("gfx1200");
  EXPECT_EQ(AMDGPU::encodeLoadcntDscnt(GFX12, 42, 9), 0x2A09u);
  EXPECT_EQ(AMDGPU::encodeStorecntDscnt(GFX12, 42, 9), 0x2A09u);
  EXPECT_EQ(AMDGPU::decodeLoadcnt(GFX12, 0x2A09), 42u);
  EXPECT_EQ(AMDGPU::decodeDscnt(GFX12, 0x2A09), 9u);

  // HoldCnt at bit 7, only from gfx10.3.
  AMDGPU::CounterLayout GFX1030 = getLayout("gfx1030");
  EXPECT_EQ(AMDGPU::DepCtr::encodeFieldHoldCnt(0, 1, GFX1030), 0x80u);
  EXPECT_EQ(AMDGPU::DepCtr::decodeFieldHoldCnt(0x80, GFX1030), 1u);
  EXPECT_EQ(AMDGPU::DepCtr::encodeFieldHoldCnt(0, 1, GFX9), 0u);
}
