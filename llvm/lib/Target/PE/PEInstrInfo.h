/* --- PEInstrInfo.h --- */

/* ------------------------------------------
Author: 高宇翔
Date: 4/1/2025
------------------------------------------ */

#ifndef PEINSTRINFO_H
#define PEINSTRINFO_H

#include "llvm/CodeGen/TargetInstrInfo.h"

// 使用CodeGen生成的源文件
#define GET_INSTRINFO_HEADER
#include "PEGenInstrInfo.inc"
namespace llvm {
class PEInstrInfo : public PEGenInstrInfo {
public:
  explicit PEInstrInfo();
  ~PEInstrInfo();

  void storeRegToStackSlot(
      MachineBasicBlock &MBB, MachineBasicBlock::iterator MI, Register SrcReg,
      bool isKill, int FrameIndex, const TargetRegisterClass *RC,
      const TargetRegisterInfo *TRI, Register VReg,
      MachineInstr::MIFlag Flags = MachineInstr::NoFlags) const override;

  void loadRegFromStackSlot(
      MachineBasicBlock &MBB, MachineBasicBlock::iterator MI, Register DestReg,
      int FrameIndex, const TargetRegisterClass *RC,
      const TargetRegisterInfo *TRI, Register VReg,
      MachineInstr::MIFlag Flags = MachineInstr::NoFlags) const override;
      
  void copyPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                   const DebugLoc &DL, Register DestReg, Register SrcReg,
                   bool KillSrc, bool RenamableDest = false,
                   bool RenamableSrc = false) const override;

private:
};
} // namespace llvm

#endif // PEINSTRINFO_H
