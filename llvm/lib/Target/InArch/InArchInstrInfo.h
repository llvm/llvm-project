#ifndef LLVM_LIB_TARGET_INARCH_INARCHINSTRINFO_H
#define LLVM_LIB_TARGET_INARCH_INARCHINSTRINFO_H

#include "InArchRegisterInfo.h"
#include "MCTargetDesc/InArchInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "InArchGenInstrInfo.inc"

namespace llvm {

class InArchSubtarget;

class InArchInstrInfo : public InArchGenInstrInfo {
public:
  InArchInstrInfo();

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

};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_SIM_SIMINSTRINFO_H