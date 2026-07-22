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
  void SetUp() override { setUpImpl("amdgcn-amd-amdhsa", "gfx942", ""); }
};

// getInstSizeInBytes may append a trailing literal word for VALU/SALU
// instructions. Only source operands can be encoded as a literal, so the
// immediate modifier fields of an LDS-DMA buffer load (offset, cpol, swz, ...)
// must not be counted. BUFFER_LOAD_DWORD_LDS_OFFEN is therefore 8 bytes.
TEST_F(InstSizesTest, BufferLoadDwordLdsIsNotOverSized) {
  StringRef MIR = R"MIR(
name: buffer_load_dword_lds_offen
body: |
  bb.0:
    BUFFER_LOAD_DWORD_LDS_OFFEN $vgpr1, $sgpr8_sgpr9_sgpr10_sgpr11, 0, 0, 0, 0, 0, implicit $exec, implicit $m0
    $vgpr0 = V_MOV_B32_e32 12345, implicit $exec
    $vgpr0 = V_MOV_B32_e32 1, implicit $exec
    S_ENDPGM 0
...
)MIR";
  ASSERT_TRUE(parseMIR(MIR));
  MachineFunction &MF = getMF("buffer_load_dword_lds_offen");
  const SIInstrInfo *TII = MF.getSubtarget<GCNSubtarget>().getInstrInfo();

  auto I = MF.getBlockNumbered(0)->begin();

  // The LDS-DMA buffer load has no trailing literal: 8 bytes, not 12.
  EXPECT_EQ(AMDGPU::BUFFER_LOAD_DWORD_LDS_OFFEN, I->getOpcode());
  EXPECT_EQ(8u, TII->getInstSizeInBytes(*I));

  // Positive control: a genuine non-inline literal in a source operand still
  // adds a 4-byte literal word (4-byte opcode + 4-byte literal).
  ++I;
  EXPECT_EQ(8u, TII->getInstSizeInBytes(*I));

  // Positive control: an inline constant is encoded for free (4 bytes).
  ++I;
  EXPECT_EQ(4u, TII->getInstSizeInBytes(*I));
}

} // end anonymous namespace
