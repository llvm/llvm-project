//===--------- llvm/unittests/Target/AMDGPU/LiveRegUnits.cpp --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUUnitTests.h"
#include "GCNSubtarget.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "gtest/gtest.h"

#include "AMDGPUGenSubtargetInfo.inc"

using namespace llvm;

class LiveRegUnitsTest : public AMDGPUCodeGenTestBase {
public:
  void SetUp() override { setUpImpl("amdgcn-amd-", "gfx1200", ""); }
};

TEST_F(LiveRegUnitsTest, TestVGPRBlockLoadStore) {
  // Add a very simple MIR snippet that saves and restores a block of VGPRs. The
  // body of the function, represented by a S_NOP, clobbers one CSR (v42) and
  // one caller-saved register (v49), and reads one CSR (v61) and one
  // callee-saved register (v53).
  StringRef MIRString = R"MIR(
name:            vgpr-block-insts
stack:
- { id: 0, name: '', type: spill-slot, offset: 0, size: 16, alignment: 4,
    stack-id: default, callee-saved-register: '$vgpr40_vgpr41_vgpr42_vgpr43_vgpr44_vgpr45_vgpr46_vgpr47_vgpr48_vgpr49_vgpr50_vgpr51_vgpr52_vgpr53_vgpr54_vgpr55_vgpr56_vgpr57_vgpr58_vgpr59_vgpr60_vgpr61_vgpr62_vgpr63_vgpr64_vgpr65_vgpr66_vgpr67_vgpr68_vgpr69_vgpr70_vgpr71',
    callee-saved-restored: true, debug-info-variable: '', debug-info-expression: '',
    debug-info-location: '' }
body:             |
  bb.0:
    liveins: $sgpr30_sgpr31, $vgpr42_vgpr43_vgpr44_vgpr45_vgpr46_vgpr47_vgpr48_vgpr49_vgpr50_vgpr51_vgpr52_vgpr53_vgpr54_vgpr55_vgpr56_vgpr57_vgpr58_vgpr59_vgpr60_vgpr61_vgpr62_vgpr63_vgpr64_vgpr65_vgpr66_vgpr67_vgpr68_vgpr69_vgpr70_vgpr71_vgpr72_vgpr73

    $m0 = S_MOV_B32 1
    SCRATCH_STORE_BLOCK_SADDR $vgpr42_vgpr43_vgpr44_vgpr45_vgpr46_vgpr47_vgpr48_vgpr49_vgpr50_vgpr51_vgpr52_vgpr53_vgpr54_vgpr55_vgpr56_vgpr57_vgpr58_vgpr59_vgpr60_vgpr61_vgpr62_vgpr63_vgpr64_vgpr65_vgpr66_vgpr67_vgpr68_vgpr69_vgpr70_vgpr71_vgpr72_vgpr73, $sgpr32, 0, 0, implicit $exec, implicit $flat_scr, implicit $m0 :: (store (s1024) into %stack.0, align 4, addrspace 5)
    S_NOP 0, implicit-def $vgpr42, implicit-def $vgpr49, implicit $vgpr53, implicit $vgpr61
    $m0 = S_MOV_B32 1
   $vgpr42_vgpr43_vgpr44_vgpr45_vgpr46_vgpr47_vgpr48_vgpr49_vgpr50_vgpr51_vgpr52_vgpr53_vgpr54_vgpr55_vgpr56_vgpr57_vgpr58_vgpr59_vgpr60_vgpr61_vgpr62_vgpr63_vgpr64_vgpr65_vgpr66_vgpr67_vgpr68_vgpr69_vgpr70_vgpr71_vgpr72_vgpr73 = SCRATCH_LOAD_BLOCK_SADDR $sgpr32, 0, 0, implicit $exec, implicit $flat_scr, implicit $m0, implicit $vgpr43, implicit $vgpr44, implicit $vgpr45, implicit $vgpr46, implicit $vgpr47, implicit $vgpr56, implicit $vgpr57, implicit $vgpr58, implicit $vgpr59, implicit $vgpr60, implicit $vgpr61, implicit $vgpr62, implicit $vgpr63, implicit $vgpr72, implicit $vgpr73 :: (load (s1024) from %stack.0, align 4, addrspace 5)
    S_SETPC_B64_return $sgpr30_sgpr31
...
)MIR";

  ASSERT_TRUE(parseMIR(MIRString));
  MachineFunction &MF = getMF("vgpr-block-insts");
  auto *MBB = MF.getBlockNumbered(0);
  auto MIt = --MBB->instr_end();

  LiveRegUnits LiveUnits;
  LiveUnits.init(*MF.getSubtarget<GCNSubtarget>().getRegisterInfo());

  LiveUnits.addLiveOuts(*MBB);
  LiveUnits.stepBackward(*MIt);

  // Right after the restore, we expect all the CSRs to be unavailable.
  // Check v40-v88 (callee and caller saved regs interleaved in blocks of 8).
  for (unsigned I = 0; I < 8; ++I) {
    EXPECT_FALSE(LiveUnits.available(AMDGPU::VGPR40 + I)) << "I = " << I;
    EXPECT_TRUE(LiveUnits.available(AMDGPU::VGPR48 + I)) << "I = " << I;
    EXPECT_FALSE(LiveUnits.available(AMDGPU::VGPR56 + I)) << "I = " << I;
    EXPECT_TRUE(LiveUnits.available(AMDGPU::VGPR64 + I)) << "I = " << I;
    EXPECT_FALSE(LiveUnits.available(AMDGPU::VGPR72 + I)) << "I = " << I;
    EXPECT_TRUE(LiveUnits.available(AMDGPU::VGPR80 + I)) << "I = " << I;
  }

  --MIt;
  LiveUnits.stepBackward(*MIt);

  // Right before the restore, we expect the CSRs that are actually transferred
  // (in this case v42) to be available. Everything else should be the same as
  // before.
  for (unsigned I = 0; I < 8; ++I) {
    if (I == 2)
      EXPECT_TRUE(LiveUnits.available(AMDGPU::VGPR40 + I)) << "I = " << I;
    else
      EXPECT_FALSE(LiveUnits.available(AMDGPU::VGPR40 + I)) << "I = " << I;
    EXPECT_TRUE(LiveUnits.available(AMDGPU::VGPR48 + I)) << "I = " << I;
    EXPECT_FALSE(LiveUnits.available(AMDGPU::VGPR56 + I)) << "I = " << I;
    EXPECT_TRUE(LiveUnits.available(AMDGPU::VGPR64 + I)) << "I = " << I;
    EXPECT_FALSE(LiveUnits.available(AMDGPU::VGPR72 + I)) << "I = " << I;
    EXPECT_TRUE(LiveUnits.available(AMDGPU::VGPR80 + I)) << "I = " << I;
  }

  --MIt; // Set m0 has no effect on VGPRs.
  LiveUnits.stepBackward(*MIt);
  --MIt; // S_NOP.
  LiveUnits.stepBackward(*MIt);

  // The S_NOP uses one of the caller-saved registers (v53), so that won't be
  // available anymore.
  for (unsigned I = 0; I < 8; ++I) {
    if (I == 2)
      EXPECT_TRUE(LiveUnits.available(AMDGPU::VGPR40 + I)) << "I = " << I;
    else
      EXPECT_FALSE(LiveUnits.available(AMDGPU::VGPR40 + I)) << "I = " << I;
    if (I == 5)
      EXPECT_FALSE(LiveUnits.available(AMDGPU::VGPR48 + I)) << "I = " << I;
    else
      EXPECT_TRUE(LiveUnits.available(AMDGPU::VGPR48 + I)) << "I = " << I;
    EXPECT_FALSE(LiveUnits.available(AMDGPU::VGPR56 + I)) << "I = " << I;
    EXPECT_TRUE(LiveUnits.available(AMDGPU::VGPR64 + I)) << "I = " << I;
    EXPECT_FALSE(LiveUnits.available(AMDGPU::VGPR72 + I)) << "I = " << I;
    EXPECT_TRUE(LiveUnits.available(AMDGPU::VGPR80 + I)) << "I = " << I;
  }

  --MIt;
  LiveUnits.stepBackward(*MIt);

  // Right before the save, all the VGPRs in the block that we're saving will be
  // unavailable, regardless of whether they're callee or caller saved. This is
  // unfortunate and should probably be fixed somehow.
  // VGPRs outside the block will only be unavailable if they're callee saved.
  for (unsigned I = 0; I < 8; ++I) {
    EXPECT_FALSE(LiveUnits.available(AMDGPU::VGPR40 + I)) << "I = " << I;
    EXPECT_FALSE(LiveUnits.available(AMDGPU::VGPR48 + I)) << "I = " << I;
    EXPECT_FALSE(LiveUnits.available(AMDGPU::VGPR56 + I)) << "I = " << I;
    EXPECT_FALSE(LiveUnits.available(AMDGPU::VGPR64 + I)) << "I = " << I;
    EXPECT_FALSE(LiveUnits.available(AMDGPU::VGPR72 + I)) << "I = " << I;
    EXPECT_TRUE(LiveUnits.available(AMDGPU::VGPR80 + I)) << "I = " << I;
  }
}
