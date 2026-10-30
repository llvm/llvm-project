//===- llvm/unittests/Target/AMDGPU/PkF32OpSelCommute.cpp -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUUnitTests.h"
#include "GCNSubtarget.h"
#include "SIDefines.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

// GFX950 v_pk_{add,mul,fma}_f32 must not get opsel[1:0] == 2'b10 (add/mul) or
// OP_SEL_0 on src1/src2 (fma). Swapping src0 and src1 swaps their modifiers, so
// a swap that would create such an encoding has to be refused. In the MIR
// below, src modifier 8 is OP_SEL_1 only (the identity swizzle) and 4 is
// OP_SEL_0 only (the swapped swizzle).
const char *PkF32MIR = R"MIR(
name: pk_f32
body: |
  bb.0:
    $vgpr0_vgpr1 = V_PK_ADD_F32 8, $vgpr2_vgpr3, 8, $vgpr4_vgpr5, 0, 0, 0, 0, 0, implicit $mode, implicit $exec
    $vgpr0_vgpr1 = V_PK_ADD_F32 4, $vgpr2_vgpr3, 8, $vgpr4_vgpr5, 0, 0, 0, 0, 0, implicit $mode, implicit $exec
    $vgpr0_vgpr1 = V_PK_ADD_F32 4, $vgpr2_vgpr3, 4, $vgpr4_vgpr5, 0, 0, 0, 0, 0, implicit $mode, implicit $exec
    $vgpr0_vgpr1 = V_PK_MUL_F32 4, $vgpr2_vgpr3, 8, $vgpr4_vgpr5, 0, 0, 0, 0, 0, implicit $mode, implicit $exec
    $vgpr0_vgpr1 = V_PK_MUL_F32 8, $vgpr2_vgpr3, 8, $vgpr4_vgpr5, 0, 0, 0, 0, 0, implicit $mode, implicit $exec
    $vgpr0_vgpr1 = V_PK_FMA_F32 8, $vgpr2_vgpr3, 8, $vgpr4_vgpr5, 8, $vgpr6_vgpr7, 0, 0, 0, 0, 0, implicit $mode, implicit $exec
    $vgpr0_vgpr1 = V_PK_FMA_F32 4, $vgpr2_vgpr3, 8, $vgpr4_vgpr5, 8, $vgpr6_vgpr7, 0, 0, 0, 0, 0, implicit $mode, implicit $exec
    S_ENDPGM 0
...
)MIR";

struct CommuteResult {
  bool Commuted;
  unsigned Src0Mods, Src1Mods;
  Register Src0, Src1;
};

CommuteResult tryCommute(const SIInstrInfo &TII, MachineInstr &MI) {
  bool Commuted = TII.commuteInstruction(MI) != nullptr;
  return {Commuted,
          (unsigned)TII.getNamedOperand(MI, AMDGPU::OpName::src0_modifiers)
              ->getImm(),
          (unsigned)TII.getNamedOperand(MI, AMDGPU::OpName::src1_modifiers)
              ->getImm(),
          TII.getNamedOperand(MI, AMDGPU::OpName::src0)->getReg(),
          TII.getNamedOperand(MI, AMDGPU::OpName::src1)->getReg()};
}

class PkF32OpSelCommuteGFX950Test : public AMDGPUCodeGenTestBase {
public:
  void SetUp() override { setUpImpl("amdgpu9.50-amd-amdhsa", "", ""); }
};

class PkF32OpSelCommuteGFX90ATest : public AMDGPUCodeGenTestBase {
public:
  void SetUp() override { setUpImpl("amdgcn-amd-amdhsa", "gfx90a", ""); }
};

TEST_F(PkF32OpSelCommuteGFX950Test, RefusesSwapsThatCreateInvalidOpSel) {
  ASSERT_TRUE(parseMIR(PkF32MIR));
  MachineFunction &MF = getMF("pk_f32");
  const SIInstrInfo *TII = MF.getSubtarget<GCNSubtarget>().getInstrInfo();
  auto I = MF.getBlockNumbered(0)->begin();
  Register V23 = TII->getNamedOperand(*I, AMDGPU::OpName::src0)->getReg();
  Register V45 = TII->getNamedOperand(*I, AMDGPU::OpName::src1)->getReg();

  // add, identity/identity (opsel[1:0] == 2'b00): the swap is harmless.
  CommuteResult R = tryCommute(*TII, *I);
  EXPECT_TRUE(R.Commuted);
  EXPECT_EQ(8u, R.Src0Mods);
  EXPECT_EQ(8u, R.Src1Mods);
  EXPECT_EQ(V45, R.Src0);
  EXPECT_EQ(V23, R.Src1);

  // add, src0 swapped only (opsel[1:0] == 2'b01): a swap would give 2'b10.
  ++I;
  R = tryCommute(*TII, *I);
  EXPECT_FALSE(R.Commuted);
  EXPECT_EQ(4u, R.Src0Mods);
  EXPECT_EQ(8u, R.Src1Mods);
  EXPECT_EQ(V23, R.Src0);
  EXPECT_EQ(V45, R.Src1);

  // add, both swapped (opsel[1:0] == 2'b11): the swap is harmless.
  ++I;
  R = tryCommute(*TII, *I);
  EXPECT_TRUE(R.Commuted);
  EXPECT_EQ(4u, R.Src0Mods);
  EXPECT_EQ(4u, R.Src1Mods);

  // mul follows the same rule as add.
  ++I;
  EXPECT_EQ(AMDGPU::V_PK_MUL_F32, I->getOpcode());
  R = tryCommute(*TII, *I);
  EXPECT_FALSE(R.Commuted);
  EXPECT_EQ(4u, R.Src0Mods);
  EXPECT_EQ(8u, R.Src1Mods);

  ++I;
  R = tryCommute(*TII, *I);
  EXPECT_TRUE(R.Commuted);

  // fma, no OP_SEL_0 anywhere: the swap is harmless.
  ++I;
  EXPECT_EQ(AMDGPU::V_PK_FMA_F32, I->getOpcode());
  R = tryCommute(*TII, *I);
  EXPECT_TRUE(R.Commuted);

  // fma, OP_SEL_0 on src0: a swap would move it to src1, which is invalid.
  ++I;
  R = tryCommute(*TII, *I);
  EXPECT_FALSE(R.Commuted);
  EXPECT_EQ(4u, R.Src0Mods);
  EXPECT_EQ(8u, R.Src1Mods);
}

// gfx90a has v_pk_{add,mul,fma}_f32 without the op_sel restriction, so every
// swap stays allowed.
TEST_F(PkF32OpSelCommuteGFX90ATest, AllowsAllSwaps) {
  ASSERT_TRUE(parseMIR(PkF32MIR));
  MachineFunction &MF = getMF("pk_f32");
  const SIInstrInfo *TII = MF.getSubtarget<GCNSubtarget>().getInstrInfo();
  for (MachineInstr &MI : *MF.getBlockNumbered(0)) {
    if (MI.getOpcode() == AMDGPU::S_ENDPGM)
      continue;
    EXPECT_TRUE(tryCommute(*TII, MI).Commuted) << MI;
  }
}

} // end anonymous namespace
