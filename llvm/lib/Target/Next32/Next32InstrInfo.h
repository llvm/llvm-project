//===-- Next32InstrInfo.h - Next32 Instruction Information ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Next32 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_Next32_Next32INSTRINFO_H
#define LLVM_LIB_TARGET_Next32_Next32INSTRINFO_H

#include "Next32RegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "Next32GenInstrInfo.inc"

namespace llvm {
class Next32Subtarget;

class Next32InstrInfo : public Next32GenInstrInfo {
  const Next32Subtarget &Subtarget;
  const Next32RegisterInfo RI;

public:
  explicit Next32InstrInfo(const Next32Subtarget &STI);

  const Next32RegisterInfo &getRegisterInfo() const { return RI; }

  void copyPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                   const DebugLoc &DL, MCRegister DestReg, MCRegister SrcReg,
                   bool KillSrc) const override;

  void storeRegToStackSlot(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MI, Register SrcReg,
                           bool isKill, int FrameIndex,
                           const TargetRegisterClass *RC,
                           const TargetRegisterInfo *TRI,
                           Register VReg) const override;

  void loadRegFromStackSlot(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MI, Register DestReg,
                            int FrameIndex, const TargetRegisterClass *RC,
                            const TargetRegisterInfo *TRI,
                            Register VReg) const override;

  bool expandPostRAPseudo(MachineInstr &MI) const override;

private:
  bool IsRegisterCopyable(Register Reg) const;

  void expandLoadStoreReg(MachineBasicBlock &MBB, MachineInstr &I) const;
  void expandPseudoLEA(MachineBasicBlock &MBB, MachineInstr &I) const;
  void expandGVMemWrite(MachineBasicBlock &MBB, MachineInstr &I) const;
  void expandGVMemRead(MachineBasicBlock &MBB, MachineInstr &I) const;
  void expandGMemWrite(MachineBasicBlock &MBB, MachineInstr &I) const;
  void expandGMemRead(MachineBasicBlock &MBB, MachineInstr &I) const;
  void expandFetchAndOp(MachineBasicBlock &MBB, MachineInstr &I) const;
  void expandCompareAndSwap(MachineBasicBlock &MBB, MachineInstr &I) const;
  void expandArithWithFlags(MachineBasicBlock &MBB, MachineInstr &I) const;
  void expandSetTID(MachineBasicBlock &MBB, MachineInstr &I) const;
  void expandFrameOffsetWrapper(MachineBasicBlock &MBB, MachineInstr &I) const;
};
} // namespace llvm

#endif
