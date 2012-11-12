//===-- SPUFrameLowering.h - SPU Frame Lowering stuff ----------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains CellSPU frame information that doesn't fit anywhere else
// cleanly...
//
//===----------------------------------------------------------------------===//

#ifndef SPU_FRAMEINFO_H
#define SPU_FRAMEINFO_H

#include "SPURegisterInfo.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
  class SPUSubtarget;

  class SPUFrameLowering: public TargetFrameLowering {
    const SPUSubtarget &Subtarget;
    std::pair<unsigned, int> LR[1];

  public:
    SPUFrameLowering(const SPUSubtarget &sti);

    //! Determine the frame's layour
    void determineFrameLayout(MachineFunction &MF) const;

    /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
    /// the function.
    void emitPrologue(MachineFunction &MF) const;
    void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;

    //! Prediate: Target has dedicated frame pointer
    bool hasFP(const MachineFunction &MF) const;

    void processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                              RegScavenger *RS = NULL) const;

    //! Return a function's saved spill slots
    /*!
      For CellSPU, a function's saved spill slots is just the link register.
     */
    const std::pair<unsigned, int> *
    getCalleeSaveSpillSlots(unsigned &NumEntries) const;

    //! Stack slot size (16 bytes)
    static int stackSlotSize() {
      return 16;
    }
    //! Maximum frame offset representable by a signed 10-bit integer
    /*!
      This is the maximum frame offset that can be expressed as a 10-bit
      integer, used in D-form addresses.
     */
    static int maxFrameOffset() {
      return ((1 << 9) - 1) * stackSlotSize();
    }
    //! Minimum frame offset representable by a signed 10-bit integer
    static int minFrameOffset() {
      return -(1 << 9) * stackSlotSize();
    }
    //! Minimum frame size (enough to spill LR + SP)
    static int minStackSize() {
      return (2 * stackSlotSize());
    }
    //! Convert frame index to stack offset
    static int FItoStackOffset(int frame_index) {
      return frame_index * stackSlotSize();
    }
  };
}

#endif
