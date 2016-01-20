//=- WebAssemblyInstrInfo.h - WebAssembly Instruction Information -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains the WebAssembly implementation of the
/// TargetInstrInfo class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYINSTRINFO_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYINSTRINFO_H

#include "WebAssemblyRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"

#define GET_INSTRINFO_HEADER
#include "WebAssemblyGenInstrInfo.inc"

namespace llvm {

class WebAssemblySubtarget;

class WebAssemblyInstrInfo final : public WebAssemblyGenInstrInfo {
  const WebAssemblyRegisterInfo RI;

public:
  explicit WebAssemblyInstrInfo(const WebAssemblySubtarget &STI);

  const WebAssemblyRegisterInfo &getRegisterInfo() const { return RI; }

  bool isReallyTriviallyReMaterializable(const MachineInstr *MI,
                                         AliasAnalysis *AA) const override;

  void copyPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                   DebugLoc DL, unsigned DestReg, unsigned SrcReg,
                   bool KillSrc) const override;

  bool AnalyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                     MachineBasicBlock *&FBB,
                     SmallVectorImpl<MachineOperand> &Cond,
                     bool AllowModify = false) const override;
  unsigned RemoveBranch(MachineBasicBlock &MBB) const override;
  unsigned InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                        MachineBasicBlock *FBB, ArrayRef<MachineOperand> Cond,
                        DebugLoc DL) const override;
  bool
  ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const override;
};

} // end namespace llvm

#endif
