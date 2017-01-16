//===-- PPCMCTargetDesc.h - PowerPC Target Descriptions ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides PowerPC specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_POWERPC_MCTARGETDESC_PPCMCTARGETDESC_H
#define LLVM_LIB_TARGET_POWERPC_MCTARGETDESC_PPCMCTARGETDESC_H

// GCC #defines PPC on Linux but we use it as our namespace name
#undef PPC

#include "llvm/Support/MathExtras.h"
#include <cstdint>

namespace llvm {

class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCInstrInfo;
class MCObjectWriter;
class MCRegisterInfo;
class MCTargetOptions;
class Target;
class Triple;
class StringRef;
class raw_pwrite_stream;

Target &getThePPC32Target();
Target &getThePPC64Target();
Target &getThePPC64LETarget();

MCCodeEmitter *createPPCMCCodeEmitter(const MCInstrInfo &MCII,
                                      const MCRegisterInfo &MRI,
                                      MCContext &Ctx);

MCAsmBackend *createPPCAsmBackend(const Target &T, const MCRegisterInfo &MRI,
                                  const Triple &TT, StringRef CPU,
                                  const MCTargetOptions &Options);

/// Construct an PPC ELF object writer.
MCObjectWriter *createPPCELFObjectWriter(raw_pwrite_stream &OS, bool Is64Bit,
                                         bool IsLittleEndian, uint8_t OSABI);
/// Construct a PPC Mach-O object writer.
MCObjectWriter *createPPCMachObjectWriter(raw_pwrite_stream &OS, bool Is64Bit,
                                          uint32_t CPUType,
                                          uint32_t CPUSubtype);

/// Returns true iff Val consists of one contiguous run of 1s with any number of
/// 0s on either side.  The 1s are allowed to wrap from LSB to MSB, so
/// 0x000FFF0, 0x0000FFFF, and 0xFF0000FF are all runs.  0x0F0F0000 is not,
/// since all 1s are not contiguous.
static inline bool isRunOfOnes(unsigned Val, unsigned &MB, unsigned &ME) {
  if (!Val)
    return false;

  if (isShiftedMask_32(Val)) {
    // look for the first non-zero bit
    MB = countLeadingZeros(Val);
    // look for the first zero bit after the run of ones
    ME = countLeadingZeros((Val - 1) ^ Val);
    return true;
  } else {
    Val = ~Val; // invert mask
    if (isShiftedMask_32(Val)) {
      // effectively look for the first zero bit
      ME = countLeadingZeros(Val) - 1;
      // effectively look for the first one bit after the run of zeros
      MB = countLeadingZeros((Val - 1) ^ Val) + 1;
      return true;
    }
  }
  // no run present
  return false;
}

} // end namespace llvm

// Generated files will use "namespace PPC". To avoid symbol clash,
// undefine PPC here. PPC may be predefined on some hosts.
#undef PPC

// Defines symbolic names for PowerPC registers.  This defines a mapping from
// register name to register number.
//
#define GET_REGINFO_ENUM
#include "PPCGenRegisterInfo.inc"

// Defines symbolic names for the PowerPC instructions.
//
#define GET_INSTRINFO_ENUM
#include "PPCGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "PPCGenSubtargetInfo.inc"

#endif // LLVM_LIB_TARGET_POWERPC_MCTARGETDESC_PPCMCTARGETDESC_H
