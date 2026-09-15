//===- AMDGPUOptimizeVGPREncodingTest.cpp -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUOptimizeVGPREncoding.h"
#include "AMDGPUUnitTests.h"
#include "GCNRegPressure.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LiveRegMatrix.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/MathExtras.h"
#include "gtest/gtest.h"

using namespace llvm;

/// MSB group, identified by an unsigned ID in [0, NumMSBGroups).
using MSBGroup = unsigned;
static constexpr unsigned MSBGroupSize = 256;
static constexpr unsigned NumMSBGroups = 4;
static constexpr unsigned DefaultGroup = 0;

/// Operand type where the MSB group is relevant, identified by an unsigned ID
/// in [0, NumOprdTypes).
using OprdType = unsigned;
static constexpr unsigned NumOprdTypes = 4;

/// A VGPR definition with an initial physical mapping.
struct VGPRDef {
  /// The register's class, in MIR-spelling (e.g., "vgpr_32").
  StringRef RegClass;
  /// Index of the physical register, offset within the MSB group to the first
  /// non-reserved VGPR.
  unsigned PhysRegIdx;
  /// MSB group to which the physical register must belong.
  MSBGroup Group;
};

/// An "abstract" MODE-using instruction. We don't care about their exact nature
/// for the sake of unit tests, just that they have specific operand sets that
/// get their MSBs from MODE.
///
/// Instructions are created via static methods each mapping to different
/// opcodes and operand-usage. All arguments to these functions must be
/// registers (virtual or physical) in MIR-spelling (e.g.,
/// "%0"/"$vgpr2_vgpr3"). Virtual registers used in source positions must have
/// been defined through a \ref VGPRDef or a previous \p ModeUsingInstr in the
/// instruction order. Physical registers may be reserved ones.
struct ModeUsingInstr {
  /// Creates a V_FMA_F32_e64 instruction.
  static ModeUsingInstr vFMA(StringRef Dst, StringRef Src0, StringRef Src1,
                             StringRef Src2) {
    ModeUsingInstr Instr(InstrType::V_FMA);
    Instr.Regs[SRC0] = Src0;
    Instr.Regs[SRC1] = Src1;
    Instr.Regs[SRC2] = Src2;
    Instr.Regs[DST] = Dst;
    return Instr;
  }

  /// Serializes the instruction to string.
  std::string toString() const {
    SmallString<256> S;
    switch (Ty) {
    case InstrType::V_FMA:
      return (Twine(Regs[DST]) + ":vgpr_32 = V_FMA_F32_e64 0, " +
              Twine(Regs[SRC0]) + ":vgpr_32, 0, " + Twine(Regs[SRC1]) +
              ":vgpr_32, 0, " + Twine(Regs[SRC2]) +
              ":vgpr_32, 0, 0, implicit $mode, implicit $exec")
          .toNullTerminatedStringRef(S)
          .str();
    }
  }

private:
  enum { SRC0 = 0, SRC1 = 1, SRC2 = 2, DST = 3 };
  std::array<StringRef, NumOprdTypes> Regs;

  enum class InstrType { V_FMA };
  InstrType Ty;

  ModeUsingInstr(InstrType Ty) : Ty(Ty) { Regs.fill(""); }
};

class AMDGPUOptimizeVGPREncodingTest : public AMDGPUCodeGenTestBase {
public:
  MachineFunction *MF;
  VirtRegMap *VRM;
  LiveRegMatrix *LRM;

  MachineRegisterInfo *MRI;
  const LiveIntervals *LIS;
  const SIRegisterInfo *TRI;
  const SIInstrInfo *TII;
  RegisterClassInfo RegClassInfo;
  unsigned NumFreeVGPRsPerGroup;

  void SetUp() override { setUpImpl("amdgpu12.50--", "", ""); }

  /// Marks every VGPR as reserved except the top NumFreeVGPRsPerGroup registers
  /// of each MSB group, so that the pass only ever has that small window in
  /// each MSB group to (re-)assign registers into.
  void reserveVGPRs() {
    const TargetRegisterClass &RC = AMDGPU::VGPR_32RegClass;
    for (unsigned I = 0, E = RC.getNumRegs(); I != E; ++I) {
      if (I % MSBGroupSize < MSBGroupSize - NumFreeVGPRsPerGroup)
        MRI->reserveReg(RC.getRegister(I), TRI);
    }
  }

  /// Programatically creates an MIR function and returns whether it
  /// successfully parsed. The function's entry block contains an IMPLICIT_DEF
  /// for all VGPRs in \p RegDefs, which are assigned to starting physical
  /// registers. Then, as many blocks are created as elements in \p MIRBlocks,
  /// and each is populated with each array element's list of MODE-using
  /// instructions. Finally, an exit block adds an implicit use for all VGPRs in
  /// \p RegDefs, ensuring they are live over all middle blocks. To constrain
  /// register re-assignments and make tests more trackable, all VGPRs but the
  /// top \p NumFreeVGPRsPerGroup in each MSB group are marked reserved.
  bool createMIRAndAssign(
      ArrayRef<VGPRDef> RegDefs,
      ArrayRef<const SmallVectorImpl<ModeUsingInstr> *> MIRBlocks,
      unsigned NumFreeVGPRsPerGroup) {
    assert(NumFreeVGPRsPerGroup < MSBGroupSize);

    std::string MIRString = R"MIR(
--- |
  define amdgpu_kernel void @func() #0 {
    ret void
  }
  attributes #0 = { "amdgpu-flat-work-group-size"="1,32" }
...
---
name: func
tracksRegLiveness: true
machineFunctionInfo:
  isEntryFunction: true
body:             |
  bb.0:
)MIR";

    // All registrer definitions go in the entry block. All MODE-using
    // instructions go in subsequent blocks, and finally an exit block with
    // implicit uses of all registers to extend their live-range over the entire
    // interesting part of the function.
    std::string RegisterUses;
    for (const auto &[VirtRegIdx, Assignment] : enumerate(RegDefs)) {
      // Each virtual register gets an IMPLICIT_DEF.
      MIRString += "    %" + std::to_string(VirtRegIdx) + ':' +
                   Assignment.RegClass.str() + " = IMPLICIT_DEF\n";
      // Accumulate uses for later.
      RegisterUses += ", implicit %" + std::to_string(VirtRegIdx);
    }
    for (const auto &[Idx, Block] : enumerate(MIRBlocks)) {
      MIRString += "  bb." + std::to_string(Idx + 1) + ":\n";
      for (const ModeUsingInstr &Instr : *Block)
        MIRString += "    " + Instr.toString() + '\n';
    }
    MIRString += "  bb." + std::to_string(MIRBlocks.size() + 1) +
                 ":\n    S_NOP 0" + RegisterUses + "\n...\n";
    if (!parseMIR(MIRString))
      return false;

    MF = &getMF("func");
    VRM = &MFAM.getResult<VirtRegMapAnalysis>(*MF);
    LRM = &MFAM.getResult<LiveRegMatrixAnalysis>(*MF);
    MRI = &MF->getRegInfo();
    LIS = &MFAM.getResult<LiveIntervalsAnalysis>(*MF);
    TRI = static_cast<const SIRegisterInfo *>(&VRM->getTargetRegInfo());
    TII =
        static_cast<const GCNSubtarget *>(&MF->getSubtarget())->getInstrInfo();
    RegClassInfo.runOnMachineFunction(*MF);

    // Reserve VGPRs except the last NumFreeVGPRsPerGroup in each group, then
    // create an initial virtual-to-physical assignment using free physical
    // registers.
    this->NumFreeVGPRsPerGroup = NumFreeVGPRsPerGroup;
    reserveVGPRs();
    for (const auto &[VirtRegIdx, Assignment] : enumerate(RegDefs))
      assign(VirtRegIdx, Assignment.PhysRegIdx, Assignment.Group);
    return true;
  }

  /// Assigns virtual register with index \p VirtRegIdx to physical
  /// register \p PhysRegIdx (offset within the MSB group to the first
  /// non-reserved VGPR) in \p Group.
  void assign(unsigned VirtRegIdx, unsigned PhysRegIdx, MSBGroup Group) {
    assert(Group < NumMSBGroups && "invalid MSB group");

    Register VirtReg = Register::index2VirtReg(VirtRegIdx);
    assert(LIS->hasInterval(VirtReg) && "invalid virt index");
    const LiveInterval &LI = LIS->getInterval(VirtReg);
    const TargetRegisterClass &RC = *MRI->getRegClass(VirtReg);
    unsigned Width = divideCeil(TRI->getRegSizeInBits(RC), 32);
    assert(PhysRegIdx < NumFreeVGPRsPerGroup / Width && "invalid phys idx");

    // PhysRegIdx selects one of the Width-sized slots of the free window at
    // the top of the MSB group.
    MCRegister Lo = AMDGPU::VGPR_32RegClass.getRegister(
        Group * MSBGroupSize + MSBGroupSize - NumFreeVGPRsPerGroup +
        PhysRegIdx * Width);
    MCRegister PhysReg =
        Width == 1 ? Lo : TRI->getMatchingSuperReg(Lo, AMDGPU::sub0, &RC);
    assert(PhysReg && !MRI->isReserved(PhysReg) && "register must be free");
    LRM->assign(LI, PhysReg);
  }

  /// Returns \p Reg's MSB group.
  MSBGroup getVGPRGroup(Register Reg) const {
    MCRegister PhysReg = Reg.isVirtual() ? VRM->getPhys(Reg) : Reg.asMCReg();
    return TRI->getHWRegIndex(PhysReg) >> 8;
  }

  /// Counts the number of S_SET_VGPR_MSB instructions the \ref MF will require.
  unsigned countSetModeInstrs() const {
    unsigned NumSetModeInstrs = 0;
    for (const MachineBasicBlock &MBB : *MF)
      NumSetModeInstrs += countSetModeInstrs(MBB);
    return NumSetModeInstrs;
  }

  /// Counts the number of S_SET_VGPR_MSB instructions \p MBB will require.
  unsigned countSetModeInstrs(const MachineBasicBlock &MBB) const {
    unsigned NumSetModeInstrs = 0;

    // A std::nullopt for a particular operand type means that there exists a
    // previous S_SET_VGPR_MSB instruction whose group for that operand type is
    // not yet constrained i.e., onto which we can piggyback a later group
    // requirement.
    std::array<std::optional<MSBGroup>, NumOprdTypes> CurrentGroups, MIGroups;

    // Groups start with all MSBs set to the default group.
    CurrentGroups.fill(DefaultGroup);

    for (const MachineInstr &MI : MBB) {
      const MCInstrDesc &Desc = MI.getDesc();
      const auto [Table, VOPDTable] =
          AMDGPU::getVGPRLoweringOperandTables(Desc);
      if (!Table)
        continue;

      auto GetRelevantMO =
          [&](const AMDGPU::OpName &Name) -> const MachineOperand * {
        if (Name == AMDGPU::OpName::NUM_OPERAND_NAMES)
          return nullptr;

        const MachineOperand *MO = TII->getNamedOperand(MI, Name);
        if (!MO || !MO->isReg())
          return nullptr;

        Register Reg = MO->getReg();
        const TargetRegisterClass *RC = TRI->getRegClassForReg(*MRI, Reg);
        return (RC && SIRegisterInfo::isVGPRClass(RC)) ? MO : nullptr;
      };

      MIGroups.fill(std::nullopt);
      for (OprdType Oprd : seq(NumOprdTypes)) {
        const MachineOperand *MO = GetRelevantMO(Table[Oprd]);
        if (!MO)
          continue;

        // Tied src2 uses of VOP2 and 32-bit-encoded VOP3 only depend on the
        // vdst bit and are handled with the def, so they are not their own
        // operand for MSB group purposes.
        if (Table[Oprd] == AMDGPU::OpName::src2 && !MO->isDef() &&
            MO->isTied() &&
            (SIInstrInfo::isVOP2(MI) ||
             (SIInstrInfo::isVOP3(MI) &&
              TII->hasVALU32BitEncoding(MI.getOpcode()))))
          continue;

        MIGroups[Oprd] = getVGPRGroup(MO->getReg());
      }

      if (VOPDTable) {
        for (OprdType Oprd : seq(NumOprdTypes)) {
          const MachineOperand *MO = GetRelevantMO(VOPDTable[Oprd]);
          if (MO)
            MIGroups[Oprd] = getVGPRGroup(MO->getReg());
        }
      }

      // Merge group requirements from the MI with the current ones, and
      // determine whether we need a new S_SET_VGPR_MSB.
      for (auto [Current, Requirement] : zip(CurrentGroups, MIGroups)) {
        if (Current.has_value()) {
          if (Requirement.has_value() && *Current != *Requirement) {
            ++NumSetModeInstrs;
            CurrentGroups = MIGroups;
            break;
          }
        } else {
          // Piggyback into previous S_SET_VGPR_MSB instruction.
          Current = Requirement;
        }
      }
    }

    // Groups must end with all MSBs set to the default group so an extra
    // S_SET_VGPR_MSB instruction would be inserted if that is not the case.
    for (const std::optional<MSBGroup> &Group : CurrentGroups) {
      if (Group.has_value() && *Group != DefaultGroup) {
        ++NumSetModeInstrs;
        break;
      }
    }

    return NumSetModeInstrs;
  }

  /// Runs the optimization pass, expecting \p ExpectModeSetBefore
  /// S_SET_VGPR_MSB instructions to be required in the function before it, and
  /// \p ExpectModeSetAfter after it.
  void runWithExpectation(unsigned ExpectModeSetBefore,
                          unsigned ExpectModeSetAfter) {
    unsigned ActualModeSetBefore = countSetModeInstrs();
    EXPECT_EQ(ExpectModeSetBefore, ActualModeSetBefore);

    AMDGPUOptimizeVGPREncodingPass Pass;
    Pass.run(*MF, MFAM);

    unsigned ActualModeSetAfter = countSetModeInstrs();
    EXPECT_EQ(ExpectModeSetAfter, ActualModeSetAfter);
  }
};

/// All registers of the VFMA are in group 1, requiring S_SET_VGPR_MSB around
/// the instruction because all operand types start (and must end) in the
/// default group.
TEST_F(AMDGPUOptimizeVGPREncodingTest, ReassignToDefaultGroup) {
  SmallVector<VGPRDef> Registers{
      {"vgpr_32", 0, 1}, // %0
      {"vgpr_32", 1, 1}, // %1
      {"vgpr_32", 2, 1}, // %2
      {"vgpr_32", 3, 1}, // %3
  };
  SmallVector<ModeUsingInstr> Instructions{
      ModeUsingInstr::vFMA("%0", "%1", "%2", "%3")};

  ASSERT_TRUE(createMIRAndAssign(Registers, {&Instructions}, 8));
  runWithExpectation(2, 0);
}

/// Each VFMA has all its registers in the same non-default group, and the two
/// VFMAs have different groups. All registers should all be re-assigned to the
/// default group which has just enough free registers. The pass must avoid a
/// local-maxima where all registers end up in one of the VFMA's group.
TEST_F(AMDGPUOptimizeVGPREncodingTest, ReassignToDefaultGroupVFMAConflict) {
  SmallVector<VGPRDef> Registers{
      {"vgpr_32", 0, 1}, // %0
      {"vgpr_32", 1, 1}, // %1
      {"vgpr_32", 2, 1}, // %2
      {"vgpr_32", 3, 1}, // %3
      {"vgpr_32", 0, 2}, // %4
      {"vgpr_32", 1, 2}, // %5
      {"vgpr_32", 2, 2}, // %6
      {"vgpr_32", 3, 2}, // %7
  };
  SmallVector<ModeUsingInstr> Instructions{
      ModeUsingInstr::vFMA("%0", "%1", "%2", "%3"),
      ModeUsingInstr::vFMA("%4", "%5", "%6", "%7")};
  ASSERT_TRUE(createMIRAndAssign(Registers, {&Instructions}, 8));
  runWithExpectation(3, 0);
}

/// Each VFMA has each of its register in a different group, and consecutive
/// VFMAs have matching operands in different groups. There is a single free
/// register in each group, which makes it hard for the pass to find
/// re-assignments, even though there is a solution that only requires a single
/// S_SET_VGPR_MSB.
TEST_F(AMDGPUOptimizeVGPREncodingTest, SingleFreeRegPerGroup) {
  SmallVector<VGPRDef> Registers{
      {"vgpr_32", 0, 0}, // %0
      {"vgpr_32", 0, 1}, // %1
      {"vgpr_32", 0, 2}, // %2
      {"vgpr_32", 0, 3}, // %3
      {"vgpr_32", 1, 1}, // %4
      {"vgpr_32", 1, 2}, // %5
      {"vgpr_32", 1, 3}, // %6
      {"vgpr_32", 1, 0}, // %7
      {"vgpr_32", 2, 2}, // %8
      {"vgpr_32", 2, 3}, // %9
      {"vgpr_32", 2, 0}, // %10
      {"vgpr_32", 2, 1}, // %11
      {"vgpr_32", 3, 3}, // %12
      {"vgpr_32", 3, 0}, // %13
      {"vgpr_32", 3, 1}, // %14
      {"vgpr_32", 3, 2}, // %15
  };
  SmallVector<ModeUsingInstr> Instructions{
      ModeUsingInstr::vFMA("%0", "%1", "%2", "%3"),
      ModeUsingInstr::vFMA("%4", "%5", "%6", "%7"),
      ModeUsingInstr::vFMA("%8", "%9", "%10", "%11"),
      ModeUsingInstr::vFMA("%12", "%13", "%14", "%15")};
  ASSERT_TRUE(createMIRAndAssign(Registers, {&Instructions}, 5));
  runWithExpectation(5, 5);
}
