//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_CFIINSTBUILDER_H
#define LLVM_CODEGEN_CFIINSTBUILDER_H

#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/MC/MCDwarf.h"

namespace llvm {

/// Helper class for creating CFI instructions and inserting them into MIR.
class CFIInstBuilder {
  MachineFunction &MF;
  MachineBasicBlock &MBB;
  MachineBasicBlock::iterator InsertPt;

  /// MIFlag to set on a MachineInstr. Typically, FrameSetup or FrameDestroy.
  MachineInstr::MIFlag MIFlag;

  /// Selects DWARF register numbering: debug or exception handling. Should be
  /// consistent with the choice of the ELF section (.debug_frame or .eh_frame)
  /// where CFI will be encoded.
  bool IsEH;

  // Cache frequently used variables.
  const TargetRegisterInfo &TRI;
  const MCInstrDesc &CFIID;
  const MIMetadata MIMD; // Default-initialized, no debug location desired.

public:
  CFIInstBuilder(MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertPt,
                 MachineInstr::MIFlag MIFlag, bool IsEH = true)
      : MF(*MBB.getParent()), MBB(MBB), MIFlag(MIFlag), IsEH(IsEH),
        TRI(*MF.getSubtarget().getRegisterInfo()),
        CFIID(MF.getSubtarget().getInstrInfo()->get(
            TargetOpcode::CFI_INSTRUCTION)) {
    setInsertPoint(InsertPt);
  }

  CFIInstBuilder(MachineBasicBlock *MBB, MachineInstr::MIFlag MIFlag,
                 bool IsEH = true)
      : CFIInstBuilder(*MBB, MBB->end(), MIFlag, IsEH) {}

  void setInsertPoint(MachineBasicBlock::iterator IP) { InsertPt = IP; }

  void insertCFIInst(const MCCFIInstruction &CFIInst) const {
    BuildMI(MBB, InsertPt, MIMD, CFIID)
        .addCFIIndex(MF.addFrameInst(CFIInst))
        .setMIFlag(MIFlag);
  }

  void buildDefCFA(MCRegister Reg, int64_t Offset) const {
    insertCFIInst(MCCFIInstruction::cfiDefCfa(
        nullptr, TRI.getDwarfRegNum(Reg, IsEH), Offset));
  }

  void buildDefCFARegister(MCRegister Reg) const {
    insertCFIInst(MCCFIInstruction::createDefCfaRegister(
        nullptr, TRI.getDwarfRegNum(Reg, IsEH)));
  }

  void buildDefCFAOffset(int64_t Offset) const {
    insertCFIInst(MCCFIInstruction::cfiDefCfaOffset(nullptr, Offset));
  }

  void buildOffset(MCRegister Reg, int64_t Offset) const {
    insertCFIInst(MCCFIInstruction::createOffset(
        nullptr, TRI.getDwarfRegNum(Reg, IsEH), Offset));
  }

  void buildRegister(MCRegister Reg1, MCRegister Reg2) const {
    insertCFIInst(MCCFIInstruction::createRegister(
        nullptr, TRI.getDwarfRegNum(Reg1, IsEH),
        TRI.getDwarfRegNum(Reg2, IsEH)));
  }

  void buildWindowSave() const {
    insertCFIInst(MCCFIInstruction::createWindowSave(nullptr));
  }

  void buildRestore(MCRegister Reg) const {
    insertCFIInst(MCCFIInstruction::createRestore(
        nullptr, TRI.getDwarfRegNum(Reg, IsEH)));
  }

  void buildUndefined(MCRegister Reg) const {
    insertCFIInst(MCCFIInstruction::createUndefined(
        nullptr, TRI.getDwarfRegNum(Reg, IsEH)));
  }

  void buildSameValue(MCRegister Reg) const {
    insertCFIInst(MCCFIInstruction::createSameValue(
        nullptr, TRI.getDwarfRegNum(Reg, IsEH)));
  }

  void buildEscape(StringRef Bytes, StringRef Comment = "") const {
    insertCFIInst(
        MCCFIInstruction::createEscape(nullptr, Bytes, SMLoc(), Comment));
  }
};

} // namespace llvm

#endif // LLVM_CODEGEN_CFIINSTBUILDER_H
