// WebAssemblyFrameLowering.h - TargetFrameLowering for WebAssembly -*- C++ -*-/
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This class implements WebAssembly-specific bits of
/// TargetFrameLowering class.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYFRAMELOWERING_H
#define LLVM_LIB_TARGET_WEBASSEMBLY_WEBASSEMBLYFRAMELOWERING_H

#include "llvm/CodeGen/TargetFrameLowering.h"

namespace llvm {
class MachineFrameInfo;

class WebAssemblyFrameLowering final : public TargetFrameLowering {
public:
  /// Size of the red zone for the user stack (leaf functions can use this much
  /// space below the stack pointer without writing it back to __stack_pointer
  /// global).
  // TODO: (ABI) Revisit and decide how large it should be.
  static const size_t RedZoneSize = 128;

  WebAssemblyFrameLowering()
      : TargetFrameLowering(StackGrowsDown, /*StackAlignment=*/16,
                            /*LocalAreaOffset=*/0,
                            /*TransientStackAlignment=*/16,
                            /*StackRealignable=*/true) {}

  MachineBasicBlock::iterator
  eliminateCallFramePseudoInstr(MachineFunction &MF, MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator I) const override;

  /// These methods insert prolog and epilog code into the function.
  void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;

  bool hasFP(const MachineFunction &MF) const override;
  bool hasReservedCallFrame(const MachineFunction &MF) const override;

  bool needsPrologForEH(const MachineFunction &MF) const;

  /// Write SP back to __stack_pointer global.
  void writeSPToGlobal(unsigned SrcReg, MachineFunction &MF,
                       MachineBasicBlock &MBB,
                       MachineBasicBlock::iterator &InsertStore,
                       const DebugLoc &DL) const;

private:
  bool hasBP(const MachineFunction &MF) const;
  bool needsSPForLocalFrame(const MachineFunction &MF) const;
  bool needsSP(const MachineFunction &MF) const;
  bool needsSPWriteback(const MachineFunction &MF) const;
};

} // end namespace llvm

#endif
