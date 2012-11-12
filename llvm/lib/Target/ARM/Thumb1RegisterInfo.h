//===- Thumb1RegisterInfo.h - Thumb-1 Register Information Impl -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Thumb-1 implementation of the TargetRegisterInfo
// class.
//
//===----------------------------------------------------------------------===//

#ifndef THUMB1REGISTERINFO_H
#define THUMB1REGISTERINFO_H

#include "ARM.h"
#include "ARMBaseRegisterInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"

namespace llvm {
  class ARMSubtarget;
  class ARMBaseInstrInfo;

struct Thumb1RegisterInfo : public ARMBaseRegisterInfo {
public:
  Thumb1RegisterInfo(const ARMBaseInstrInfo &tii, const ARMSubtarget &STI);

  const TargetRegisterClass*
  getLargestLegalSuperClass(const TargetRegisterClass *RC) const;

  const TargetRegisterClass*
  getPointerRegClass(const MachineFunction &MF, unsigned Kind = 0) const;

  /// emitLoadConstPool - Emits a load from constpool to materialize the
  /// specified immediate.
 void emitLoadConstPool(MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator &MBBI,
                        DebugLoc dl,
                        unsigned DestReg, unsigned SubIdx, int Val,
                        ARMCC::CondCodes Pred = ARMCC::AL,
                        unsigned PredReg = 0,
                        unsigned MIFlags = MachineInstr::NoFlags) const;

  /// Code Generation virtual methods...
  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const;

  // rewrite MI to access 'Offset' bytes from the FP. Update Offset to be
  // however much remains to be handled. Return 'true' if no further
  // work is required.
  bool rewriteFrameIndex(MachineBasicBlock::iterator II, unsigned FrameRegIdx,
                         unsigned FrameReg, int &Offset,
                         const ARMBaseInstrInfo &TII) const;
  void resolveFrameIndex(MachineBasicBlock::iterator I,
                         unsigned BaseReg, int64_t Offset) const;
  bool saveScavengerRegister(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator I,
                             MachineBasicBlock::iterator &UseMI,
                             const TargetRegisterClass *RC,
                             unsigned Reg) const;
  void eliminateFrameIndex(MachineBasicBlock::iterator II,
                           int SPAdj, RegScavenger *RS = NULL) const;
};
}

#endif // THUMB1REGISTERINFO_H
