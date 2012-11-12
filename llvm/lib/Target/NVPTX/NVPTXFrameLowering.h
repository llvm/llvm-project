//===--- NVPTXFrameLowering.h - Define frame lowering for NVPTX -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//
//===----------------------------------------------------------------------===//

#ifndef NVPTX_FRAMELOWERING_H
#define NVPTX_FRAMELOWERING_H

#include "llvm/Target/TargetFrameLowering.h"


namespace llvm {
class NVPTXTargetMachine;

class NVPTXFrameLowering : public TargetFrameLowering {
  NVPTXTargetMachine &tm;
  bool is64bit;

public:
  explicit NVPTXFrameLowering(NVPTXTargetMachine &_tm, bool _is64bit)
  : TargetFrameLowering(TargetFrameLowering::StackGrowsUp, 8, 0),
    tm(_tm), is64bit(_is64bit) {}

  virtual bool hasFP(const MachineFunction &MF) const;
  virtual void emitPrologue(MachineFunction &MF) const;
  virtual void emitEpilogue(MachineFunction &MF,
                            MachineBasicBlock &MBB) const;
};

} // End llvm namespace

#endif
