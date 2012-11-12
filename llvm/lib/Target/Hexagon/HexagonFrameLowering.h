//=- HexagonFrameLowering.h - Define frame lowering for Hexagon --*- C++ -*--=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_FRAMEINFO_H
#define HEXAGON_FRAMEINFO_H

#include "Hexagon.h"
#include "HexagonSubtarget.h"
#include "llvm/Target/TargetFrameLowering.h"

namespace llvm {

class HexagonFrameLowering : public TargetFrameLowering {
private:
  const HexagonSubtarget &STI;
  void determineFrameLayout(MachineFunction &MF) const;

public:
  explicit HexagonFrameLowering(const HexagonSubtarget &sti)
    : TargetFrameLowering(StackGrowsDown, 8, 0), STI(sti) {
  }

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function.
  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;
  virtual bool
  spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MI,
                            const std::vector<CalleeSavedInfo> &CSI,
                            const TargetRegisterInfo *TRI) const;
  virtual bool
  restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MI,
                              const std::vector<CalleeSavedInfo> &CSI,
                              const TargetRegisterInfo *TRI) const;
  int getFrameIndexOffset(const MachineFunction &MF, int FI) const;
  bool hasFP(const MachineFunction &MF) const;
  bool hasTailCall(MachineBasicBlock &MBB) const;
};

} // End llvm namespace

#endif
