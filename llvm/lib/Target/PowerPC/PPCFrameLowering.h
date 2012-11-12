//===-- PPCFrameLowering.h - Define frame lowering for PowerPC --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef POWERPC_FRAMEINFO_H
#define POWERPC_FRAMEINFO_H

#include "PPC.h"
#include "PPCSubtarget.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/STLExtras.h"

namespace llvm {
  class PPCSubtarget;

class PPCFrameLowering: public TargetFrameLowering {
  const PPCSubtarget &Subtarget;

public:
  PPCFrameLowering(const PPCSubtarget &sti)
    : TargetFrameLowering(TargetFrameLowering::StackGrowsDown, 16, 0),
      Subtarget(sti) {
  }

  void determineFrameLayout(MachineFunction &MF) const;

  /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
  /// the function.
  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;

  bool hasFP(const MachineFunction &MF) const;
  bool needsFP(const MachineFunction &MF) const;

  void processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                            RegScavenger *RS = NULL) const;
  void processFunctionBeforeFrameFinalized(MachineFunction &MF) const;

  bool spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MI,
                                 const std::vector<CalleeSavedInfo> &CSI,
                                 const TargetRegisterInfo *TRI) const;

  bool restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                   const std::vector<CalleeSavedInfo> &CSI,
                                   const TargetRegisterInfo *TRI) const;

  /// targetHandlesStackFrameRounding - Returns true if the target is
  /// responsible for rounding up the stack frame (probably at emitPrologue
  /// time).
  bool targetHandlesStackFrameRounding() const { return true; }

  /// getReturnSaveOffset - Return the previous frame offset to save the
  /// return address.
  static unsigned getReturnSaveOffset(bool isPPC64, bool isDarwinABI) {
    if (isDarwinABI)
      return isPPC64 ? 16 : 8;
    // SVR4 ABI:
    return isPPC64 ? 16 : 4;
  }

  /// getFramePointerSaveOffset - Return the previous frame offset to save the
  /// frame pointer.
  static unsigned getFramePointerSaveOffset(bool isPPC64, bool isDarwinABI) {
    // For the Darwin ABI:
    // We cannot use the TOC save slot (offset +20) in the PowerPC linkage area
    // for saving the frame pointer (if needed.)  While the published ABI has
    // not used this slot since at least MacOSX 10.2, there is older code
    // around that does use it, and that needs to continue to work.
    if (isDarwinABI)
      return isPPC64 ? -8U : -4U;

    // SVR4 ABI: First slot in the general register save area.
    return isPPC64 ? -8U : -4U;
  }

  /// getLinkageSize - Return the size of the PowerPC ABI linkage area.
  ///
  static unsigned getLinkageSize(bool isPPC64, bool isDarwinABI) {
    if (isDarwinABI || isPPC64)
      return 6 * (isPPC64 ? 8 : 4);

    // SVR4 ABI:
    return 8;
  }

  /// getMinCallArgumentsSize - Return the size of the minium PowerPC ABI
  /// argument area.
  static unsigned getMinCallArgumentsSize(bool isPPC64, bool isDarwinABI) {
    // For the Darwin ABI / 64-bit SVR4 ABI:
    // The prolog code of the callee may store up to 8 GPR argument registers to
    // the stack, allowing va_start to index over them in memory if its varargs.
    // Because we cannot tell if this is needed on the caller side, we have to
    // conservatively assume that it is needed.  As such, make sure we have at
    // least enough stack space for the caller to store the 8 GPRs.
    if (isDarwinABI || isPPC64)
      return 8 * (isPPC64 ? 8 : 4);

    // 32-bit SVR4 ABI:
    // There is no default stack allocated for the 8 first GPR arguments.
    return 0;
  }

  /// getMinCallFrameSize - Return the minimum size a call frame can be using
  /// the PowerPC ABI.
  static unsigned getMinCallFrameSize(bool isPPC64, bool isDarwinABI) {
    // The call frame needs to be at least big enough for linkage and 8 args.
    return getLinkageSize(isPPC64, isDarwinABI) +
           getMinCallArgumentsSize(isPPC64, isDarwinABI);
  }

  // With the SVR4 ABI, callee-saved registers have fixed offsets on the stack.
  const SpillSlot *
  getCalleeSavedSpillSlots(unsigned &NumEntries) const {
    if (Subtarget.isDarwinABI()) {
      NumEntries = 1;
      if (Subtarget.isPPC64()) {
        static const SpillSlot darwin64Offsets = {PPC::X31, -8};
        return &darwin64Offsets;
      } else {
        static const SpillSlot darwinOffsets = {PPC::R31, -4};
        return &darwinOffsets;
      }
    }

    // Early exit if not using the SVR4 ABI.
    if (!Subtarget.isSVR4ABI()) {
      NumEntries = 0;
      return 0;
    }

    static const SpillSlot Offsets[] = {
      // Floating-point register save area offsets.
      {PPC::F31, -8},
      {PPC::F30, -16},
      {PPC::F29, -24},
      {PPC::F28, -32},
      {PPC::F27, -40},
      {PPC::F26, -48},
      {PPC::F25, -56},
      {PPC::F24, -64},
      {PPC::F23, -72},
      {PPC::F22, -80},
      {PPC::F21, -88},
      {PPC::F20, -96},
      {PPC::F19, -104},
      {PPC::F18, -112},
      {PPC::F17, -120},
      {PPC::F16, -128},
      {PPC::F15, -136},
      {PPC::F14, -144},

      // General register save area offsets.
      {PPC::R31, -4},
      {PPC::R30, -8},
      {PPC::R29, -12},
      {PPC::R28, -16},
      {PPC::R27, -20},
      {PPC::R26, -24},
      {PPC::R25, -28},
      {PPC::R24, -32},
      {PPC::R23, -36},
      {PPC::R22, -40},
      {PPC::R21, -44},
      {PPC::R20, -48},
      {PPC::R19, -52},
      {PPC::R18, -56},
      {PPC::R17, -60},
      {PPC::R16, -64},
      {PPC::R15, -68},
      {PPC::R14, -72},

      // CR save area offset.  We map each of the nonvolatile CR fields
      // to the slot for CR2, which is the first of the nonvolatile CR
      // fields to be assigned, so that we only allocate one save slot.
      // See PPCRegisterInfo::hasReservedSpillSlot() for more information.
      {PPC::CR2, -4},

      // VRSAVE save area offset.
      {PPC::VRSAVE, -4},

      // Vector register save area
      {PPC::V31, -16},
      {PPC::V30, -32},
      {PPC::V29, -48},
      {PPC::V28, -64},
      {PPC::V27, -80},
      {PPC::V26, -96},
      {PPC::V25, -112},
      {PPC::V24, -128},
      {PPC::V23, -144},
      {PPC::V22, -160},
      {PPC::V21, -176},
      {PPC::V20, -192}
    };

    static const SpillSlot Offsets64[] = {
      // Floating-point register save area offsets.
      {PPC::F31, -8},
      {PPC::F30, -16},
      {PPC::F29, -24},
      {PPC::F28, -32},
      {PPC::F27, -40},
      {PPC::F26, -48},
      {PPC::F25, -56},
      {PPC::F24, -64},
      {PPC::F23, -72},
      {PPC::F22, -80},
      {PPC::F21, -88},
      {PPC::F20, -96},
      {PPC::F19, -104},
      {PPC::F18, -112},
      {PPC::F17, -120},
      {PPC::F16, -128},
      {PPC::F15, -136},
      {PPC::F14, -144},

      // General register save area offsets.
      {PPC::X31, -8},
      {PPC::X30, -16},
      {PPC::X29, -24},
      {PPC::X28, -32},
      {PPC::X27, -40},
      {PPC::X26, -48},
      {PPC::X25, -56},
      {PPC::X24, -64},
      {PPC::X23, -72},
      {PPC::X22, -80},
      {PPC::X21, -88},
      {PPC::X20, -96},
      {PPC::X19, -104},
      {PPC::X18, -112},
      {PPC::X17, -120},
      {PPC::X16, -128},
      {PPC::X15, -136},
      {PPC::X14, -144},

      // VRSAVE save area offset.
      {PPC::VRSAVE, -4},

      // Vector register save area
      {PPC::V31, -16},
      {PPC::V30, -32},
      {PPC::V29, -48},
      {PPC::V28, -64},
      {PPC::V27, -80},
      {PPC::V26, -96},
      {PPC::V25, -112},
      {PPC::V24, -128},
      {PPC::V23, -144},
      {PPC::V22, -160},
      {PPC::V21, -176},
      {PPC::V20, -192}
    };

    if (Subtarget.isPPC64()) {
      NumEntries = array_lengthof(Offsets64);

      return Offsets64;
    } else {
      NumEntries = array_lengthof(Offsets);

      return Offsets;
    }
  }
};

} // End llvm namespace

#endif
