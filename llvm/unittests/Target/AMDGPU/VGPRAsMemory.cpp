//===--------- llvm/unittests/Target/AMDGPU/VGPRAsMemory.cpp --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Two properties of the VGPR "as memory" (address space 13) indexed accesses
// are asserted here rather than in a lit test, because no pass can be made to
// observe them: these pseudos read M0, which is written in any function that
// uses them, and that alone stops MachineLICM and MachineSink from moving
// them. Their safety today is therefore incidental, and the properties below
// are what it would rest on if that incidental protection ever went away.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUUnitTests.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "gtest/gtest.h"

#include "AMDGPUGenSubtargetInfo.inc"

using namespace llvm;

class VGPRAsMemoryTest : public AMDGPUCodeGenTestBase {
public:
  void SetUp() override { setUpImpl("amdgpu12.00-amd-", "", ""); }
};

// An indexed access reads or writes the per-lane vector registers of the active
// lanes, so which lanes are active is part of what it does. Its implicit use of
// EXEC must not be reported ignorable: that is what would otherwise let it be
// hoisted or sunk across a write to EXEC, changing the set of lanes touched.
TEST_F(VGPRAsMemoryTest, ExecUseIsNotIgnorable) {
  StringRef MIRString = R"MIR(
name: exec_use
body:             |
  bb.0:
    liveins: $m0, $sgpr0, $vgpr0

    $vgpr1 = V_LOAD_IDX_B32 0, implicit $m0, implicit $exec :: (load (s32), addrspace 13)
    V_STORE_IDX_B32 $vgpr0, 0, implicit $m0, implicit $exec :: (store (s32), addrspace 13)
    $vgpr1 = V_LOAD_IDX_GPR_IDX_B32 $sgpr0, 0, implicit-def dead $m0, implicit $m0, implicit $exec :: (load (s32), addrspace 13)
    V_STORE_IDX_GPR_IDX_B32 $vgpr0, $sgpr0, 0, implicit-def dead $m0, implicit $m0, implicit $exec :: (store (s32), addrspace 13)
    $vgpr2 = V_MOV_B32_e32 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  ASSERT_TRUE(parseMIR(MIRString));
  MachineFunction &MF = getMF("exec_use");
  const SIInstrInfo *TII = MF.getSubtarget<GCNSubtarget>().getInstrInfo();
  MachineBasicBlock *MBB = MF.getBlockNumbered(0);

  auto ExecUseOf = [](const MachineInstr &MI) -> const MachineOperand * {
    for (const MachineOperand &MO : MI.implicit_operands())
      if (MO.getReg() == AMDGPU::EXEC)
        return &MO;
    return nullptr;
  };

  for (MachineInstr &MI : *MBB) {
    const MachineOperand *Exec = ExecUseOf(MI);
    switch (MI.getOpcode()) {
    case AMDGPU::V_LOAD_IDX_B32:
    case AMDGPU::V_STORE_IDX_B32:
    case AMDGPU::V_LOAD_IDX_GPR_IDX_B32:
    case AMDGPU::V_STORE_IDX_GPR_IDX_B32:
      ASSERT_NE(Exec, nullptr) << "indexed access lost its implicit EXEC";
      EXPECT_FALSE(TII->isIgnorableUse(MI, MI.getOperandNo(Exec)))
          << "an indexed access may not be moved across a write to EXEC";
      break;
    case AMDGPU::V_MOV_B32_e32:
      // The contrast: a plain lane-wise move produces the same value in every
      // lane it writes, so its EXEC use really is ignorable.
      ASSERT_NE(Exec, nullptr);
      EXPECT_TRUE(TII->isIgnorableUse(MI, MI.getOperandNo(Exec)));
      break;
    default:
      break;
    }
  }
}

// Two whole-dword accesses indexed by M0 - the form every access has on a
// movrel subtarget - with M0 redefined between them. The dwords they touch are
// (first M0)+1 and (second M0)+0, which are the same dword whenever the second
// index is one more than the first, so they may alias. Disjointness is decided
// from a base operand and the constant offset, and these have no base operand
// that could tell the two M0 values apart.
TEST_F(VGPRAsMemoryTest, M0IndexedAccessesAcrossAM0RedefMayAlias) {
  StringRef MIRString = R"MIR(
name: m0_redef
body:             |
  bb.0:
    liveins: $sgpr0, $sgpr1, $vgpr0

    $m0 = COPY $sgpr0
    $vgpr1 = V_LOAD_IDX_B32 1, implicit $m0, implicit $exec :: (load (s32), addrspace 13)
    $m0 = COPY $sgpr1
    V_STORE_IDX_B32 $vgpr0, 0, implicit $m0, implicit $exec :: (store (s32), addrspace 13)
    S_ENDPGM 0
...
)MIR";

  ASSERT_TRUE(parseMIR(MIRString));
  MachineFunction &MF = getMF("m0_redef");
  const SIInstrInfo *TII = MF.getSubtarget<GCNSubtarget>().getInstrInfo();
  MachineBasicBlock *MBB = MF.getBlockNumbered(0);

  const MachineInstr *Load = nullptr;
  const MachineInstr *Store = nullptr;
  for (MachineInstr &MI : *MBB) {
    if (MI.getOpcode() == AMDGPU::V_LOAD_IDX_B32)
      Load = &MI;
    else if (MI.getOpcode() == AMDGPU::V_STORE_IDX_B32)
      Store = &MI;
  }
  ASSERT_NE(Load, nullptr);
  ASSERT_NE(Store, nullptr);

  EXPECT_FALSE(TII->areMemAccessesTriviallyDisjoint(*Load, *Store))
      << "M0 is redefined between these accesses, so their dword indices are "
         "unrelated and they must not be reported disjoint";
}
