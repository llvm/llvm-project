//===-- llvm/MC/MCInstrAnalysis.h - InstrDesc target hooks ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MCInstrAnalysis class which the MCTargetDescs can
// derive from to give additional information to MC.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"

namespace llvm {

class MCInstrAnalysis {
protected:
  friend class Target;
  const MCInstrInfo *Info;

public:
  MCInstrAnalysis(const MCInstrInfo *Info) : Info(Info) {}

  virtual ~MCInstrAnalysis() {}

  virtual bool isBranch(const MCInst &Inst) const {
    return Info->get(Inst.getOpcode()).isBranch();
  }

  virtual bool isConditionalBranch(const MCInst &Inst) const {
    return Info->get(Inst.getOpcode()).isConditionalBranch();
  }

  virtual bool isUnconditionalBranch(const MCInst &Inst) const {
    return Info->get(Inst.getOpcode()).isUnconditionalBranch();
  }

  virtual bool isIndirectBranch(const MCInst &Inst) const {
    return Info->get(Inst.getOpcode()).isIndirectBranch();
  }

  virtual bool isCall(const MCInst &Inst) const {
    return Info->get(Inst.getOpcode()).isCall();
  }

  virtual bool isReturn(const MCInst &Inst) const {
    return Info->get(Inst.getOpcode()).isReturn();
  }

  /// evaluateBranch - Given a branch instruction try to get the address the
  /// branch targets. Otherwise return -1.
  virtual uint64_t
  evaluateBranch(const MCInst &Inst, uint64_t Addr, uint64_t Size) const;
};

}
