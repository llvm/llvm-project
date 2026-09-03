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
// observe them: these pseudos carry implicit-def $m0, so they define a physical
// register, and MachineLICM, MachineSink and MachineCSE all decline to touch
// them for that reason alone. Their safety today is therefore incidental, and
// the properties below are what it would rest on if that incidental protection
// ever went away.
//
// Each is paired with an ordinary VALU that must answer the other way, so a
// change that made the query answer uniformly fails here rather than passing
// vacuously.
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
    liveins: $m0, $vgpr0

    $vgpr1 = V_LOAD_IDX_B32 $m0, 0, implicit $exec :: (load (s32), addrspace 13)
    V_STORE_IDX_B32 $vgpr0, $m0, 0, implicit $exec :: (store (s32), addrspace 13)
    $vgpr2 = V_MOV_B32_e32 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  ASSERT_TRUE(parseMIR(MIRString));
  MachineFunction &MF = getMF("exec_use");
  const SIInstrInfo *TII = MF.getSubtarget<GCNSubtarget>().getInstrInfo();
  MachineBasicBlock *MBB = MF.getBlockNumbered(0);

  auto ExecUseOf = [](const MachineInstr &MI) -> const MachineOperand * {
    for (const MachineOperand &MO : MI.operands())
      if (MO.isReg() && MO.isImplicit() && MO.getReg() == AMDGPU::EXEC)
        return &MO;
    return nullptr;
  };

  for (MachineInstr &MI : *MBB) {
    const MachineOperand *Exec = ExecUseOf(MI);
    switch (MI.getOpcode()) {
    case AMDGPU::V_LOAD_IDX_B32:
    case AMDGPU::V_STORE_IDX_B32:
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

// An indexed load reads the wave's per-lane view of its vector registers, so
// the value is divergent however the index was computed. Reporting it as
// possibly-uniform would invite a readfirstlane, broadcasting one lane's value
// across the wave.
TEST_F(VGPRAsMemoryTest, IndexedLoadIsNeverUniform) {
  StringRef MIRString = R"MIR(
name: uniformity
body:             |
  bb.0:
    liveins: $m0

    $vgpr0 = V_LOAD_IDX_B32 $m0, 0, implicit $exec :: (load (s32), addrspace 13)
    $vgpr1 = V_MOV_B32_e32 0, implicit $exec
    S_ENDPGM 0
...
)MIR";

  ASSERT_TRUE(parseMIR(MIRString));
  MachineFunction &MF = getMF("uniformity");
  const SIInstrInfo *TII = MF.getSubtarget<GCNSubtarget>().getInstrInfo();
  MachineBasicBlock *MBB = MF.getBlockNumbered(0);

  for (MachineInstr &MI : *MBB) {
    switch (MI.getOpcode()) {
    case AMDGPU::V_LOAD_IDX_B32:
      EXPECT_EQ(TII->getValueUniformity(MI), ValueUniformity::NeverUniform);
      break;
    case AMDGPU::V_MOV_B32_e32:
      // The contrast: an ordinary move is only as divergent as its operands.
      EXPECT_EQ(TII->getValueUniformity(MI), ValueUniformity::Default);
      break;
    default:
      break;
    }
  }
}

// A sub-dword access is not described by a dword index and width alone - the
// bit position within the dword is part of the address - so these pseudos carry
// no entry in the width table, deliberately. getMemOperandsWithOffsetWidth must
// decline to describe one rather than reaching getBitWidth() and the
// llvm_unreachable behind it. With assertions off that unreachable is UB, and
// the caller it would reach through is an aliasing decision, so the failure
// would be a garbage width silently declaring two overlapping accesses
// disjoint.
//
// Note the class does match these pseudos, so the dyn_cast is not what protects
// the call; the width-table lookup inside it is.
TEST_F(VGPRAsMemoryTest, SubDwordAccessHasNoDescribableWidth) {
  StringRef MIRString = R"MIR(
name: width
body:             |
  bb.0:
    liveins: $m0, $sgpr0, $vgpr0

    $vgpr1 = V_LOAD_IDX_B32 $m0, 0, implicit $exec :: (load (s32), addrspace 13)
    $vgpr2 = V_LOAD_IDX_BITS $m0, 0, 8, $sgpr0, 0, implicit $exec :: (load (s8), addrspace 13)
    $vgpr3 = V_LOAD_IDX_BITS $m0, 0, 8, $sgpr0, 0, implicit $exec :: (load (s8), addrspace 13)
    S_ENDPGM 0
...
)MIR";

  ASSERT_TRUE(parseMIR(MIRString));
  MachineFunction &MF = getMF("width");
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  const TargetRegisterInfo *TRI = ST.getRegisterInfo();
  MachineBasicBlock *MBB = MF.getBlockNumbered(0);

  auto Describe = [&](const MachineInstr &MI) {
    SmallVector<const MachineOperand *, 4> BaseOps;
    int64_t Offset = 0;
    bool OffsetIsScalable = false;
    LocationSize Width = LocationSize::precise(0);
    return TII->getMemOperandsWithOffsetWidth(MI, BaseOps, Offset,
                                              OffsetIsScalable, Width, TRI);
  };

  const MachineInstr *FirstBits = nullptr;
  const MachineInstr *SecondBits = nullptr;
  for (MachineInstr &MI : *MBB) {
    switch (MI.getOpcode()) {
    case AMDGPU::V_LOAD_IDX_B32:
      // A whole-dword access is describable: dword index plus a known width.
      EXPECT_TRUE(Describe(MI));
      break;
    case AMDGPU::V_LOAD_IDX_BITS:
      EXPECT_FALSE(Describe(MI))
          << "a sub-dword access must be reported as opaque";
      (FirstBits ? SecondBits : FirstBits) = &MI;
      break;
    default:
      break;
    }
  }

  // And the caller that would otherwise reach the unreachable: with no width to
  // compare, two sub-dword accesses have to be assumed to overlap.
  ASSERT_NE(FirstBits, nullptr);
  ASSERT_NE(SecondBits, nullptr);
  EXPECT_FALSE(TII->areMemAccessesTriviallyDisjoint(*FirstBits, *SecondBits));
}
