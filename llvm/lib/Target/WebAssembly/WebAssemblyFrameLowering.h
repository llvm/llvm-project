// WebAssemblyFrameLowering.h - TargetFrameLowering for WebAssembly -*- C++ -*-/
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This class implements WebAssembly-specific bits of
/// TargetFrameLowering class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYFRAMELOWERING_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYFRAMELOWERING_H

#include "llvm/Target/TargetFrameLowering.h"

namespace llvm {

class WebAssemblyFrameLowering final : public TargetFrameLowering {
 public:
  WebAssemblyFrameLowering()
      : TargetFrameLowering(StackGrowsDown, /*StackAlignment=*/16,
                            /*LocalAreaOffset=*/0,
                            /*TransientStackAlignment=*/16,
                            /*StackRealignable=*/true) {}

  void eliminateCallFramePseudoInstr(
      MachineFunction &MF, MachineBasicBlock &MBB,
      MachineBasicBlock::iterator I) const override;

  /// These methods insert prolog and epilog code into the function.
  void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;

  bool hasFP(const MachineFunction &MF) const override;
  bool hasReservedCallFrame(const MachineFunction &MF) const override;
};

}  // end namespace llvm

#endif
