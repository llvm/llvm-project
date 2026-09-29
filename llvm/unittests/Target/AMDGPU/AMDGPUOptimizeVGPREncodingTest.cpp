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
#include "llvm/Support/raw_ostream.h"
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
struct VirtVGPRDef {
  /// The register's name, with the leading '%'.
  StringRef RegName;
  /// The register's class, in MIR-spelling (e.g., "vgpr_32").
  StringRef RegClass;
  /// MSB group to which the physical register must belong.
  MSBGroup Group;
  /// Index of the physical register, offset within the MSB group to the first
  /// non-reserved VGPR.
  unsigned PhysRegIdx;

  VirtVGPRDef(StringRef RegDef, MSBGroup Group, unsigned PhysRegIdx)
      : Group(Group), PhysRegIdx(PhysRegIdx) {
    std::tie(RegName, RegClass) = RegDef.split(':');
  }
};

/// A physical VGPR definition.
using PhysVGPRDef = StringRef;

/// An "abstract" MODE-using instruction. We don't care about their exact nature
/// for the sake of unit tests, just that they have specific operand sets that
/// get their MSBs from MODE.
///
/// Instructions are created via static methods each mapping to different
/// opcodes and operand-usage. All arguments to these functions must be
/// registers (virtual or physical) in MIR-spelling (e.g.,
/// "%0"/"$vgpr2_vgpr3"). Virtual registers used in source positions must have
/// been defined through a \ref VGPRDef or a previous \p ModeUsingInstr in the
/// instruction order. Virtual registers are assumed to be of class "vgpr_32"
/// unless their class is explicitly provided (e.g., "%0:vgpr_32_lo256").
/// Physical registers may be reserved ones.
struct ModeUsingInstr {
  /// Creates a VOP1 instruction.
  static ModeUsingInstr createVOP1(StringRef Dst, StringRef Src0) {
    ModeUsingInstr Instr(InstrType::VOP1);
    Instr.Regs[SRC0] = Src0;
    Instr.Regs[DST] = Dst;
    return Instr;
  }

  /// Creates a VOP2 instruction.
  static ModeUsingInstr createVOP2(StringRef Dst, StringRef Src0,
                                   StringRef Src1) {
    ModeUsingInstr Instr(InstrType::VOP2);
    Instr.Regs[SRC0] = Src0;
    Instr.Regs[SRC1] = Src1;
    Instr.Regs[DST] = Dst;
    return Instr;
  }

  /// Creates a VOP3 instruction.
  static ModeUsingInstr createVOP3(StringRef Dst, StringRef Src0,
                                   StringRef Src1, StringRef Src2) {
    ModeUsingInstr Instr(InstrType::VOP3);
    Instr.Regs[SRC0] = Src0;
    Instr.Regs[SRC1] = Src1;
    Instr.Regs[SRC2] = Src2;
    Instr.Regs[DST] = Dst;
    return Instr;
  }

  /// Creates a VOP2 instruction whose src2 is tied to its destination.
  static ModeUsingInstr createVOP2Tied(StringRef DstSrc2, StringRef Src0,
                                       StringRef Src1) {
    ModeUsingInstr Instr(InstrType::VOP2Tied);
    Instr.Regs[SRC0] = Src0;
    Instr.Regs[SRC1] = Src1;
    Instr.Regs[SRC2] = DstSrc2;
    Instr.Regs[DST] = DstSrc2;
    return Instr;
  }

  /// Creates a VOPC instruction.
  static ModeUsingInstr createVOPC(StringRef Src0, StringRef Src1) {
    ModeUsingInstr Instr(InstrType::VOPC);
    Instr.Regs[SRC0] = Src0;
    Instr.Regs[SRC1] = Src1;
    return Instr;
  }

  /// Creates a VOPD instruction whose X and Y components both have a
  /// destination, src0 and src1.
  static ModeUsingInstr createVOPD(StringRef DstX, StringRef Src0X,
                                   StringRef Src1X, StringRef DstY,
                                   StringRef Src0Y, StringRef Src1Y) {
    ModeUsingInstr Instr(InstrType::VOPD);
    Instr.Regs[SRC0] = Src0X;
    Instr.Regs[SRC1] = Src1X;
    Instr.Regs[DST] = DstX;
    Instr.RegsY[SRC0] = Src0Y;
    Instr.RegsY[SRC1] = Src1Y;
    Instr.RegsY[DST] = DstY;
    return Instr;
  }

  /// Creates a VOPD instruction whose X component has no src1 but whose Y
  /// component does.
  static ModeUsingInstr createVOPDNoSrc1X(StringRef DstX, StringRef Src0X,
                                          StringRef DstY, StringRef Src0Y,
                                          StringRef Src1Y) {
    ModeUsingInstr Instr(InstrType::VOPDNoSrc1X);
    Instr.Regs[SRC0] = Src0X;
    Instr.Regs[DST] = DstX;
    Instr.RegsY[SRC0] = Src0Y;
    Instr.RegsY[SRC1] = Src1Y;
    Instr.RegsY[DST] = DstY;
    return Instr;
  }

  /// Serializes the instruction to string.
  std::string toString() const {
    std::string Instr;
    raw_string_ostream OS(Instr);
    switch (Ty) {
    case InstrType::VOP1:
      OS << regToString(DST) << " = nofpexcept V_CEIL_F32_e32 "
         << regToString(SRC0) << ", implicit $mode, implicit $exec";
      break;
    case InstrType::VOP2:
      OS << regToString(DST) << " = nofpexcept V_ADD_F32_e64 0, "
         << regToString(SRC0) << ", 0, " << regToString(SRC1)
         << ", 0, 0, implicit $mode, implicit $exec";
      break;
    case InstrType::VOP3:
      OS << regToString(DST) << " = V_FMA_F32_e64 0, " << regToString(SRC0)
         << ", 0, " << regToString(SRC1) << ", 0, " << regToString(SRC2)
         << ", 0, 0, implicit $mode, implicit $exec";
      break;
    case InstrType::VOP2Tied:
      OS << regToString(DST) << " = nofpexcept V_FMAC_F32_e32 "
         << regToString(SRC0) << ", " << regToString(SRC1) << ", "
         << regToString(SRC2) << ", implicit $mode, implicit $exec";
      break;
    case InstrType::VOPC:
      OS << "V_CMPX_EQ_I32_e32 " << regToString(SRC0) << ", "
         << regToString(SRC1)
         << ", implicit-def $exec, implicit-def $vcc, implicit $exec";
      break;
    case InstrType::VOPD:
      OS << regToString(DST) << ", " << regToString(DST, true)
         << " = V_DUAL_SUB_F32_e32_X_MUL_F32_e32_gfx1250 " << regToString(SRC0)
         << ", " << regToString(SRC1) << ", " << regToString(SRC0, true) << ", "
         << regToString(SRC1, true) << ", implicit $mode, implicit $exec";
      break;
    case InstrType::VOPDNoSrc1X:
      OS << regToString(DST) << ", " << regToString(DST, true)
         << " = V_DUAL_MOV_B32_e32_X_ADD_F32_e32_gfx1250 " << regToString(SRC0)
         << ", " << regToString(SRC0, true) << ", " << regToString(SRC1, true)
         << ", implicit $mode, implicit $exec";
      break;
    }
    return Instr;
  }

  /// Serializes the register of operand type \p Oprd, from the Y component of
  /// VOPD instructions if \p IsY is true.
  std::string regToString(OprdType Oprd, bool IsY = false) const {
    StringRef Reg = IsY ? RegsY[Oprd] : Regs[Oprd];
    return Reg.str() +
           (Reg.starts_with("%") && !Reg.contains(':') ? ":vgpr_32" : "");
  }

private:
  enum { SRC0 = 0, SRC1 = 1, SRC2 = 2, DST = 3 };
  /// Registers of each operand type. For VOPD instructions, \ref Regs holds
  /// the X component's registers and \ref RegsY the Y component's.
  std::array<StringRef, NumOprdTypes> Regs, RegsY;

  enum class InstrType {
    VOP1,
    VOP2,
    VOP2Tied,
    VOP3,
    VOPC,
    VOPD,
    VOPDNoSrc1X,
  };
  InstrType Ty;

  ModeUsingInstr(InstrType Ty) : Ty(Ty) {
    Regs.fill("");
    RegsY.fill("");
  }
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

  void SetUp() override { setUpImpl("amdgpu12.50--", "", ""); }

  /// Marks every VGPR as reserved except the top \p NumFreeVGPRsPerGroup
  /// registers of each MSB group, so that the pass only ever has that small
  /// window in each MSB group to (re-)assign registers into.
  void reserveVGPRs(unsigned NumFreeVGPRsPerGroup) {
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
      ArrayRef<VirtVGPRDef> VirtDefs, ArrayRef<PhysVGPRDef> PhysDefs,
      ArrayRef<const SmallVectorImpl<ModeUsingInstr> *> MIRBlocks,
      unsigned NumFreeVGPRsPerGroup) {
    assert(NumFreeVGPRsPerGroup < MSBGroupSize);

    // Physical registers need to be marked as live-ins in all blocks.
    std::string LiveIns;
    if (!PhysDefs.empty()) {
      raw_string_ostream LiveInsOS(LiveIns);
      LiveInsOS << "    liveins: ";
      for (PhysVGPRDef PhysDef : drop_end(PhysDefs))
        LiveInsOS << PhysDef << ", ";
      LiveInsOS << PhysDefs.back() << '\n';
    }

    std::string MIRString;
    raw_string_ostream OS(MIRString);
    OS << R"MIR(
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
    OS << LiveIns;

    // All registrer definitions go in the entry block. All MODE-using
    // instructions go in subsequent blocks, and finally an exit block with
    // implicit uses of all registers to extend their live-range over the entire
    // interesting part of the function.
    std::string RegisterUses;
    raw_string_ostream RegUseOS(RegisterUses);

    for (const VirtVGPRDef VirtDef : VirtDefs) {
      // Each virtual register gets an IMPLICIT_DEF.
      OS << "    " << VirtDef.RegName << ':' << VirtDef.RegClass
         << " = IMPLICIT_DEF\n";
      // Accumulate uses for the last block.
      RegUseOS << ", implicit " << VirtDef.RegName;
    }
    for (const auto &[Idx, Block] : enumerate(MIRBlocks)) {
      OS << "  bb." << Idx + 1 << ":\n" << LiveIns;
      for (const ModeUsingInstr &Instr : *Block)
        OS << "    " << Instr.toString() << '\n';
    }
    OS << "  bb." << MIRBlocks.size() + 1 << ":\n    S_NOP 0" << RegisterUses
       << "\n    S_ENDPGM 0\n...\n";
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
    reserveVGPRs(NumFreeVGPRsPerGroup);
    for (const auto &[VirtRegIdx, Assignment] : enumerate(VirtDefs)) {
      OriginalAssignments.insert(
          {Assignment.RegName,
           {Register::index2VirtReg(VirtRegIdx), Assignment.Group}});
      assign(VirtRegIdx, Assignment.PhysRegIdx, Assignment.Group,
             NumFreeVGPRsPerGroup);
    }

    OriginalNumModeSet = getNumModeSets();
    return true;
  }

  /// Assigns virtual register with index \p VirtRegIdx to physical
  /// register \p PhysRegIdx (offset within the MSB group to the last \p
  /// NumFreeVGPRsPerGroup registers) in \p Group.
  void assign(unsigned VirtRegIdx, unsigned PhysRegIdx, MSBGroup Group,
              unsigned NumFreeVGPRsPerGroup) {
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

  /// Counts the number of MODE-setting instructions the \ref MF will require.
  unsigned getNumModeSets() const {
    unsigned NumSetModeInstrs = 0;
    for (const MachineBasicBlock &MBB : *MF)
      NumSetModeInstrs += countSetModeInstrs(MBB);
    return NumSetModeInstrs;
  }

  /// Counts the number of MODE-setting instructions \p MBB will require.
  unsigned countSetModeInstrs(const MachineBasicBlock &MBB) const {
    unsigned NumSetModeInstrs = 0;

    // A std::nullopt for a particular operand type means that there exists a
    // previous MODE-setting instruction whose group for that operand type is
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
      // determine whether we need a new MODE-setting.
      for (auto [Current, Requirement] : zip(CurrentGroups, MIGroups)) {
        if (Current.has_value()) {
          if (Requirement.has_value() && *Current != *Requirement) {
            ++NumSetModeInstrs;
            CurrentGroups = MIGroups;
            break;
          }
        } else {
          // Piggyback into previous MODE-setting instruction.
          Current = Requirement;
        }
      }
    }

    // Groups must end with all MSBs set to the default group so an extra
    // MODE-setting instruction would be inserted if that is not the case.
    for (const std::optional<MSBGroup> &Group : CurrentGroups) {
      if (Group.has_value() && *Group != DefaultGroup) {
        ++NumSetModeInstrs;
        break;
      }
    }

    return NumSetModeInstrs;
  }

  void runPass() { AMDGPUOptimizeVGPREncodingPass().run(*MF, MFAM); }

  /// Expects \p ExpectModeSetBefore MODE-setting instructions to be required in
  /// the function before the pass, and \p ExpectModeSetAfter after it.
  void expectNumModeSetChange(unsigned ExpectModeSetBefore,
                              unsigned ExpectModeSetAfter) {
    EXPECT_EQ(ExpectModeSetBefore, OriginalNumModeSet) << "before the pass";
    EXPECT_EQ(ExpectModeSetAfter, getNumModeSets()) << "after the pass";
  }

  /// Expects registers in \p Changes (identified by name, first element) to
  /// have moved to the group indicated by the second element. If \p Exhaustive
  /// is true, also expects that all other registers have not moved to a
  /// different group than the one they were assigned to at the beginning.
  void expectAssignmentChanges(ArrayRef<std::pair<StringRef, MSBGroup>> Changes,
                               bool Exhaustive) const {
    SmallDenseSet<StringRef, 4> ChangedRegs;
    for (const auto &[RegName, ExpectedGroup] : Changes) {
      auto Original = OriginalAssignments.find(RegName);
      ASSERT_NE(Original, OriginalAssignments.end())
          << "virtreg " << RegName << " does not exist";
      EXPECT_EQ(ExpectedGroup, getVGPRGroup(Original->second.VirtReg))
          << "for virtreg " << RegName;

      if (Exhaustive)
        ChangedRegs.insert(RegName);
    }

    if (!Exhaustive)
      return;
    for (const auto &[RegName, Assignment] : OriginalAssignments) {
      if (ChangedRegs.contains(RegName))
        continue;
      EXPECT_EQ(Assignment.Group, getVGPRGroup(Assignment.VirtReg))
          << "assignment of virtreg " << RegName << " unexpectedly changed";
    }
  }

private:
  struct VirtRegAndGroup {
    Register VirtReg;
    MSBGroup Group;
  };
  DenseMap<StringRef, VirtRegAndGroup> OriginalAssignments;
  unsigned OriginalNumModeSet;
};

/// All registers of the VOP3 are in group 1, requiring MODE-setting
/// instructions around it because all operand types start (and must end) in the
/// default group. All register should be re-assigned to the default group.
TEST_F(AMDGPUOptimizeVGPREncodingTest, ReassignToDefaultGroup) {
  SmallVector<VirtVGPRDef> VirtDefs{
      {"%0:vgpr_32", 1, 0},
      {"%1:vgpr_32", 1, 1},
      {"%2:vgpr_32", 1, 2},
      {"%3:vgpr_32", 1, 3},
  };
  SmallVector<ModeUsingInstr> Instructions{
      ModeUsingInstr::createVOP3("%0", "%1", "%2", "%3")};

  ASSERT_TRUE(createMIRAndAssign(VirtDefs, {}, {&Instructions}, 4));
  runPass();
  expectNumModeSetChange(2, 0);
}

/// Each VOP3 has all its registers in the same non-default group, and the two
/// VOP3s have different groups. All registers should be re-assigned to the
/// default group which has just enough free registers. The pass must avoid a
/// local-maximum where all registers end up in one of the VOP3's group.
TEST_F(AMDGPUOptimizeVGPREncodingTest, ReassignToDefaultGroupVOP3Conflict) {
  SmallVector<VirtVGPRDef> VirtDefs{
      {"%group1_0:vgpr_32", 1, 0}, {"%group1_1:vgpr_32", 1, 1},
      {"%group1_2:vgpr_32", 1, 2}, {"%group1_3:vgpr_32", 1, 3},
      {"%group2_0:vgpr_32", 2, 0}, {"%group2_1:vgpr_32", 2, 1},
      {"%group2_2:vgpr_32", 2, 2}, {"%group2_3:vgpr_32", 2, 3},
  };
  SmallVector<ModeUsingInstr> Instructions{
      ModeUsingInstr::createVOP3("%group1_0", "%group1_1", "%group1_2",
                                 "%group1_3"),
      ModeUsingInstr::createVOP3("%group2_0", "%group2_1", "%group2_2",
                                 "%group2_3")};
  ASSERT_TRUE(createMIRAndAssign(VirtDefs, {}, {&Instructions}, 8));
  runPass();
  expectNumModeSetChange(3, 0);
}

/// Each VOP3 has each of its register in a different group, and consecutive
/// VOP3 have matching operands in different groups. There is a single free
/// register in each group, which makes it hard for the pass to find
/// re-assignments, even though there is a solution that only requires a single
/// MODE-setting instruction.
TEST_F(AMDGPUOptimizeVGPREncodingTest, SingleFreeRegPerGroup) {
  SmallVector<VirtVGPRDef> VirtDefs{
      {"%0:vgpr_32", 0, 0},  {"%1:vgpr_32", 1, 0},  {"%2:vgpr_32", 2, 0},
      {"%3:vgpr_32", 3, 0},  {"%4:vgpr_32", 1, 1},  {"%5:vgpr_32", 2, 1},
      {"%6:vgpr_32", 3, 1},  {"%7:vgpr_32", 0, 1},  {"%8:vgpr_32", 2, 2},
      {"%9:vgpr_32", 3, 2},  {"%10:vgpr_32", 0, 2}, {"%11:vgpr_32", 1, 2},
      {"%12:vgpr_32", 3, 3}, {"%13:vgpr_32", 0, 3}, {"%14:vgpr_32", 1, 3},
      {"%15:vgpr_32", 2, 3},
  };
  SmallVector<ModeUsingInstr> Instructions{
      ModeUsingInstr::createVOP3("%0", "%1", "%2", "%3"),
      ModeUsingInstr::createVOP3("%4", "%5", "%6", "%7"),
      ModeUsingInstr::createVOP3("%8", "%9", "%10", "%11"),
      ModeUsingInstr::createVOP3("%12", "%13", "%14", "%15")};
  ASSERT_TRUE(createMIRAndAssign(VirtDefs, {}, {&Instructions}, 5));
  runPass();
  expectNumModeSetChange(5, 3);
}

/// Tests re-assignment priority w.r.t. the number of neighbors in each
/// group. There is a single free register in the default group. %group2 should
/// have re-assignment priority because it has twice as many neighboring
/// relationships with it.
TEST_F(AMDGPUOptimizeVGPREncodingTest, HigherNumberOfOccurencesWins) {
  SmallVector<VirtVGPRDef> VirtDefs{
      {"%group1:vgpr_32", 1, 0},
      {"%group2:vgpr_32", 2, 0},
  };
  SmallVector<ModeUsingInstr> Instructions{
      ModeUsingInstr::createVOP2("%group1", "%group2", "%group2"),
  };
  ASSERT_TRUE(createMIRAndAssign(VirtDefs, {}, {&Instructions}, 1));
  runPass();
  expectAssignmentChanges({{"%group2", 0}}, /*Exhaustive=*/true);
}

/// Tests re-assignment priority in the presence of neighbors in multiple
/// groups. There is a single free register in the default group. %clearMoveTo0
/// should have re-assignment priority because it has no neighboring
/// relationships with other groups, whereas %group1 and %group2 have.
TEST_F(AMDGPUOptimizeVGPREncodingTest, ClearerTargetWins) {
  SmallVector<VirtVGPRDef> VirtDefs{
      {"%group1:vgpr_32", 1, 0},
      {"%group2:vgpr_32", 2, 0},
      {"%clearMoveTo0:vgpr_32", 3, 0},
  };
  SmallVector<ModeUsingInstr> Instructions{
      ModeUsingInstr::createVOP1("%group1", "%clearMoveTo0"),
      ModeUsingInstr::createVOP1("%group2", "%clearMoveTo0"),
  };
  ASSERT_TRUE(createMIRAndAssign(VirtDefs, {}, {&Instructions}, 1));
  runPass();
  expectAssignmentChanges({{"%clearMoveTo0", 0}}, /*Exhaustive=*/true);
}

/// Tests re-assignment priority in the presence of pinned neighbors. There is a
/// single free register in group 0 (%vgpr255). All virtual registers not in
/// group 0 would benefit from a move to it. %goodPin's neighbors are all pinned
/// and in group 0 so it gets priority.
TEST_F(AMDGPUOptimizeVGPREncodingTest, GoodPinWins) {
  SmallVector<VirtVGPRDef> VirtDefs{
      {"%group0:vgpr_32", 0, 0},
      {"%goodPin:vgpr_32", 1, 0},
      {"%badPin:vgpr_32", 1, 1},
      {"%free:vgpr_32", 2, 0},
  };
  SmallVector<ModeUsingInstr> Instructions{
      ModeUsingInstr::createVOP3("$vgpr0", "%group0", "%group0", "$vgpr256"),
      ModeUsingInstr::createVOP3("%goodPin", "%free", "%badPin", "%badPin"),
      ModeUsingInstr::createVOP3("$vgpr0", "%group0", "%group0", "%group0"),
  };
  ASSERT_TRUE(
      createMIRAndAssign(VirtDefs, {"$vgpr0", "$vgpr256"}, {&Instructions}, 2));
  runPass();
  expectAssignmentChanges({{"%goodPin", 0}}, /*Exhaustive=*/true);
}

/// The tied src2 use of a VOP2 only depends on the MSB group of the
/// destination; it must not be considered a src2 occurrence. %moveTo0 should
/// move to the adjacent dst operands' group ($vgpr0, group 0) instead
/// of the adjacent src2 operands' group ($vgpr256, group 1).
TEST_F(AMDGPUOptimizeVGPREncodingTest, TiedSrc2FollowsDst) {
  SmallVector<VirtVGPRDef> VirtDefs{
      {"%moveTo0:vgpr_32", 2, 0},
  };
  SmallVector<ModeUsingInstr> Instructions{
      ModeUsingInstr::createVOP3("$vgpr0", "$vgpr0", "$vgpr256", "$vgpr256"),
      ModeUsingInstr::createVOP2Tied("%moveTo0", "$vgpr0", "%moveTo0"),
      ModeUsingInstr::createVOP3("$vgpr0", "$vgpr0", "$vgpr0", "$vgpr256"),
  };
  ASSERT_TRUE(
      createMIRAndAssign(VirtDefs, {"$vgpr0", "$vgpr256"}, {&Instructions}, 1));
  runPass();
  expectAssignmentChanges({{"%moveTo0", 0}}, /*Exhaustive=*/true);
}

/// All registers of the VOPD are in group 1, requiring MODE-setting
/// instructions around it. Re-assigning them all to the default group would be
/// profitable, but VOPD operands are pinned so nothing should change.
TEST_F(AMDGPUOptimizeVGPREncodingTest, VOPDRegsArePinned) {
  SmallVector<VirtVGPRDef> VirtDefs{
      {"%dstX:vgpr_32", 1, 0},  {"%src0X:vgpr_32", 1, 1},
      {"%src1X:vgpr_32", 1, 2}, {"%dstY:vgpr_32", 1, 3},
      {"%src0Y:vgpr_32", 1, 4}, {"%src1Y:vgpr_32", 1, 5},
  };
  SmallVector<ModeUsingInstr> Instructions{
      ModeUsingInstr::createVOPD("%dstX", "%src0X", "%src1X", "%dstY", "%src0Y",
                                 "%src1Y"),
  };
  ASSERT_TRUE(createMIRAndAssign(VirtDefs, {}, {&Instructions}, 6));
  runPass();
  expectAssignmentChanges({}, /*Exhaustive=*/true);
}

/// The X component of the VOPDs has no src1 but the Y component does, so the
/// latter provides the src1 MSB group of the VOPDs. %moveTo1's src1 neighbors
/// are thus $vgpr256 (group 1) on both sides, and it should move from group 2
/// to group 1.
TEST_F(AMDGPUOptimizeVGPREncodingTest, VOPDSecondComponentOprd) {
  SmallVector<VirtVGPRDef> VirtDefs{
      {"%moveTo1:vgpr_32", 2, 0},
  };
  SmallVector<ModeUsingInstr> Instructions{
      ModeUsingInstr::createVOPDNoSrc1X("$vgpr0", "$vgpr0", "$vgpr0", "$vgpr0",
                                        "$vgpr256"),
      ModeUsingInstr::createVOPC("$vgpr0", "%moveTo1"),
      ModeUsingInstr::createVOPDNoSrc1X("$vgpr0", "$vgpr0", "$vgpr0", "$vgpr0",
                                        "$vgpr256"),
  };
  ASSERT_TRUE(
      createMIRAndAssign(VirtDefs, {"$vgpr0", "$vgpr256"}, {&Instructions}, 1));
  runPass();
  expectAssignmentChanges({{"%moveTo1", 1}}, /*Exhaustive=*/true);
}

/// Registers whose class is confined to the first 256 VGPRs cannot leave the
/// first MSB group so they are pinned. %lo256's and %moveTo1's neighbors are
/// all in group 1, but only %moveTo1 should move there.
TEST_F(AMDGPUOptimizeVGPREncodingTest, Lo256RegsArePinned) {
  SmallVector<VirtVGPRDef> VirtDefs{
      {"%lo256:vgpr_32_lo256", 0, 0},
      {"%moveTo1:vgpr_32", 0, 1},
  };
  SmallVector<ModeUsingInstr> Instructions{
      ModeUsingInstr::createVOP1("$vgpr256", "$vgpr256"),
      ModeUsingInstr::createVOP1("%lo256:vgpr_32_lo256", "%moveTo1"),
      ModeUsingInstr::createVOP1("$vgpr256", "$vgpr256"),
  };
  ASSERT_TRUE(createMIRAndAssign(VirtDefs, {"$vgpr256"}, {&Instructions}, 2));
  runPass();
  expectAssignmentChanges({{"%moveTo1", 1}}, /*Exhaustive=*/true);
}
