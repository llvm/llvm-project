//===- llvm/unittests/Target/AMDGPU/InstSizes.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUUnitTests.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class InstSizesTest : public AMDGPUCodeGenTestBase {
public:
  void SetUp() override { setUpImpl("amdgpu9.42-amd-amdhsa", "", ""); }
};

// getInstSizeInBytes appends a 4-byte literal word for a VALU/SALU instruction
// whose source operand is a non-inline immediate. The offset/cpol/swz/IsAsync
// fields of an LDS-DMA buffer load are packed into the instruction word and are
// not source operands, so they must not be counted as a literal: the load is
// 8 bytes, not 12.
TEST_F(InstSizesTest, BufferLoadLdsIsNotOverSized) {
  StringRef MIR = R"MIR(
name: buffer_load_lds
body: |
  bb.0:
    BUFFER_LOAD_DWORD_LDS_OFFEN $vgpr1, $sgpr8_sgpr9_sgpr10_sgpr11, 0, 0, 0, 0, 0, implicit $exec, implicit $m0
    BUFFER_LOAD_DWORD_LDS_OFFSET $sgpr8_sgpr9_sgpr10_sgpr11, 0, 0, 0, 0, 0, implicit $exec, implicit $m0
    $vgpr0 = V_MOV_B32_e32 12345, implicit $exec
    $vgpr0 = V_MOV_B32_e32 1, implicit $exec
    S_ENDPGM 0
...
)MIR";
  ASSERT_TRUE(parseMIR(MIR));
  MachineFunction &MF = getMF("buffer_load_lds");
  const SIInstrInfo *TII = MF.getSubtarget<GCNSubtarget>().getInstrInfo();

  auto I = MF.getBlockNumbered(0)->begin();

  // The LDS-DMA buffer loads carry no trailing literal: 8 bytes, not 12.
  EXPECT_EQ(AMDGPU::BUFFER_LOAD_DWORD_LDS_OFFEN, I->getOpcode());
  EXPECT_EQ(8u, TII->getInstSizeInBytes(*I));

  ++I;
  EXPECT_EQ(AMDGPU::BUFFER_LOAD_DWORD_LDS_OFFSET, I->getOpcode());
  EXPECT_EQ(8u, TII->getInstSizeInBytes(*I));

  // Sanity check: non-inline constant becomes a trailing 32-bit literal.
  ++I;
  EXPECT_EQ(AMDGPU::V_MOV_B32_e32, I->getOpcode());
  EXPECT_EQ(8u, TII->getInstSizeInBytes(*I));

  // Sanity check: inline constant is encoded in the instruction word.
  ++I;
  EXPECT_EQ(AMDGPU::V_MOV_B32_e32, I->getOpcode());
  EXPECT_EQ(4u, TII->getInstSizeInBytes(*I));
}

} // end anonymous namespace
