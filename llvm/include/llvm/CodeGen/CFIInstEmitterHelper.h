//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_CFIINSTREMITTERHELPER_H
#define LLVM_CODEGEN_CFIINSTREMITTERHELPER_H

#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/MC/MCDwarf.h"

namespace llvm {

/// Helper class for emitting CFI instructions into Machine IR.
class CFIInstEmitterHelper {
  MachineFunction &MF;
  MachineBasicBlock &MBB;
  MachineBasicBlock::iterator InsertPt;

  /// MIflag to set on a MachineInstr. Typically, FrameSetup or FrameDestroy.
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
  CFIInstEmitterHelper(MachineBasicBlock &MBB,
                       MachineBasicBlock::iterator InsertPt,
                       MachineInstr::MIFlag MIFlag, bool IsEH = true)
      : MF(*MBB.getParent()), MBB(MBB), MIFlag(MIFlag), IsEH(IsEH),
        TRI(*MF.getSubtarget().getRegisterInfo()),
        CFIID(MF.getSubtarget().getInstrInfo()->get(
            TargetOpcode::CFI_INSTRUCTION)) {
    setInsertPoint(InsertPt);
  }

  void setInsertPoint(MachineBasicBlock::iterator IP) { InsertPt = IP; }

  void emitCFIInst(const MCCFIInstruction &CFIInst) const {
    BuildMI(MBB, InsertPt, MIMD, CFIID)
        .addCFIIndex(MF.addFrameInst(CFIInst))
        .setMIFlag(MIFlag);
  }

  void emitDefCFA(MCRegister Reg, int64_t Offset) const {
    emitCFIInst(MCCFIInstruction::cfiDefCfa(
        nullptr, TRI.getDwarfRegNum(Reg, IsEH), Offset));
  }

  void emitDefCFARegister(MCRegister Reg) const {
    emitCFIInst(MCCFIInstruction::createDefCfaRegister(
        nullptr, TRI.getDwarfRegNum(Reg, IsEH)));
  }

  void emitDefCFAOffset(int64_t Offset) const {
    emitCFIInst(MCCFIInstruction::cfiDefCfaOffset(nullptr, Offset));
  }

  void emitOffset(MCRegister Reg, int64_t Offset) const {
    emitCFIInst(MCCFIInstruction::createOffset(
        nullptr, TRI.getDwarfRegNum(Reg, IsEH), Offset));
  }

  void emitRestore(MCRegister Reg) const {
    emitCFIInst(MCCFIInstruction::createRestore(nullptr,
                                                TRI.getDwarfRegNum(Reg, IsEH)));
  }

  void emitEscape(StringRef Bytes) const {
    emitCFIInst(MCCFIInstruction::createEscape(nullptr, Bytes));
  }
};

} // namespace llvm

#endif // LLVM_CODEGEN_CFIINSTREMITTERHELPER_H
