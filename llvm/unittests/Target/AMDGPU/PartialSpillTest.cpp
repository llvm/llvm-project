//===- PartialSpillTest.cpp - Partial SGPR spill tests
//---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUUnitTests.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
class PartialSpillTest : public AMDGPUCodeGenTestBase {
public:
  void SetUp() override { setUpImpl("amdgcn-amd-amdhsa", "gfx1151", ""); }
};

TEST_F(PartialSpillTest, ExactMasksAndSlotOffsets) {
  ASSERT_TRUE(parseMIR(R"MIR(
---
name: func
tracksRegLiveness: true
machineFunctionInfo:
  isEntryFunction: true
body: |
  bb.0:
    S_ENDPGM 0
...
)MIR"));
  MachineFunction &MF = getMF("func");
  const SIInstrInfo &TII = *MF.getSubtarget<GCNSubtarget>().getInstrInfo();
  const SIRegisterInfo &TRI = TII.getRegisterInfo();
  auto &MRI = MF.getRegInfo();
  auto &MFI = *MF.getInfo<SIMachineFunctionInfo>();

  for (const TargetRegisterClass *RC :
       {&AMDGPU::SGPR_64RegClass, &AMDGPU::SGPR_128RegClass,
        &AMDGPU::SGPR_512RegClass, &AMDGPU::SGPR_1024RegClass}) {
    auto Parts = TRI.getRegSplitParts(RC, 4);
    Register Reg = MRI.createVirtualRegister(RC);
    bool SGPRStack =
        RC == &AMDGPU::SGPR_128RegClass || RC == &AMDGPU::SGPR_1024RegClass;
    MFI.setHasSpilledSGPRs(false);
    int FI = MF.getFrameInfo().CreateSpillStackObject(
        TRI.getSpillSize(*RC), Align(4),
        SGPRStack ? TargetStackID::SGPRSpill : TargetStackID::Default);
    LaneBitmask All = MRI.getMaxLaneMaskForVReg(Reg);
    EXPECT_TRUE(TII.canStoreRegToStackSlotWithLanes(RC, All));
    EXPECT_TRUE(TII.canStoreRegToStackSlotWithLanes(RC, LaneBitmask()));
    EXPECT_FALSE(TII.canStoreRegToStackSlotWithLanes(RC, LaneBitmask(1)));

    // Include the last word to exercise the boundary of a 1024-bit tuple,
    // and a discontiguous mask to ensure lane numbers aren't packed together.
    LaneBitmask Mask = TRI.getSubRegIndexLaneMask(Parts.front()) |
                       TRI.getSubRegIndexLaneMask(Parts.back());
    MachineBasicBlock &MBB = MF.front();
    auto End = MBB.begin();
    for (bool IsKill : {false, true}) {
      TII.storeRegToStackSlot(MBB, End, Reg, IsKill, FI, RC, Register(),
                              MachineInstr::NoFlags, Mask);
      EXPECT_EQ(MFI.hasSpilledSGPRs(), SGPRStack);
      ASSERT_EQ(std::distance(MBB.begin(), End), 2);
      unsigned I = 0;
      for (MachineInstr &MI : make_range(MBB.begin(), End)) {
        unsigned Word = I++ == 0 ? 0 : Parts.size() - 1;
        EXPECT_EQ(MI.getOpcode(), AMDGPU::SI_SPILL_S32_SAVE_PARTIAL);
        EXPECT_EQ(MI.getOperand(0).getReg(), Reg);
        EXPECT_EQ(MI.getOperand(0).getSubReg(), unsigned(Parts[Word]));
        EXPECT_EQ(MI.getOperand(0).isKill(), IsKill && I == 2);
        EXPECT_EQ(MI.getOperand(1).getIndex(), FI);
        EXPECT_EQ(MI.getOperand(2).getImm(), Word);
        ASSERT_EQ(MI.memoperands().size(), 2u);
        const auto *Store = MI.memoperands()[0];
        const auto *Load = MI.memoperands()[1];
        EXPECT_TRUE(Store->isStore());
        EXPECT_FALSE(Store->isLoad());
        EXPECT_EQ(Store->getOffset(), Word * 4);
        EXPECT_EQ(Store->getSize(), LocationSize::precise(4));
        EXPECT_TRUE(Load->isLoad());
        EXPECT_FALSE(Load->isStore());
        EXPECT_EQ(Load->getOffset(), 0);
        EXPECT_EQ(Load->getSize(),
                  LocationSize::precise(TRI.getSpillSize(*RC)));
        int StackSlot;
        EXPECT_FALSE(TII.isStoreToStackSlot(MI, StackSlot));
        EXPECT_FALSE(TII.isLoadFromStackSlot(MI, StackSlot));
      }
      MBB.erase(MBB.begin(), End);
    }
    TII.storeRegToStackSlot(MBB, End, Reg, false, FI, RC, Register(),
                            MachineInstr::NoFlags, LaneBitmask());
    EXPECT_EQ(MBB.begin(), End);
  }
  for (const TargetRegisterClass *RC :
       {&AMDGPU::SGPR_32RegClass, &AMDGPU::VGPR_32RegClass,
        &AMDGPU::VReg_128RegClass})
    EXPECT_FALSE(TII.canStoreRegToStackSlotWithLanes(RC, LaneBitmask(3)));
  EXPECT_FALSE(TII.canStoreRegToStackSlotWithLanes(&AMDGPU::SGPR_128RegClass,
                                                   LaneBitmask(1u << 8)));
}
} // namespace
