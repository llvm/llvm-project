//===-- MipsMCTargetDesc.h - Mips Target Descriptions -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Mips specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MIPS_MCTARGETDESC_MIPSMCTARGETDESC_H
#define LLVM_LIB_TARGET_MIPS_MCTARGETDESC_MIPSMCTARGETDESC_H

#include "llvm/Support/DataTypes.h"

namespace llvm {
class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCInstrInfo;
class MCObjectWriter;
class MCRegisterInfo;
class MCSubtargetInfo;
class MCTargetOptions;
class StringRef;
class Target;
class Triple;
class raw_ostream;
class raw_pwrite_stream;

extern Target TheMipsTarget;
extern Target TheMipselTarget;
extern Target TheMips64Target;
extern Target TheMips64elTarget;

MCCodeEmitter *createMipsMCCodeEmitterEB(const MCInstrInfo &MCII,
                                         const MCRegisterInfo &MRI,
                                         MCContext &Ctx);
MCCodeEmitter *createMipsMCCodeEmitterEL(const MCInstrInfo &MCII,
                                         const MCRegisterInfo &MRI,
                                         MCContext &Ctx);

MCAsmBackend *createMipsAsmBackendEB32(const Target &T,
                                       const MCRegisterInfo &MRI,
                                       const Triple &TT, StringRef CPU,
                                       const MCTargetOptions &Options);
MCAsmBackend *createMipsAsmBackendEL32(const Target &T,
                                       const MCRegisterInfo &MRI,
                                       const Triple &TT, StringRef CPU,
                                       const MCTargetOptions &Options);
MCAsmBackend *createMipsAsmBackendEB64(const Target &T,
                                       const MCRegisterInfo &MRI,
                                       const Triple &TT, StringRef CPU,
                                       const MCTargetOptions &Options);
MCAsmBackend *createMipsAsmBackendEL64(const Target &T,
                                       const MCRegisterInfo &MRI,
                                       const Triple &TT, StringRef CPU,
                                       const MCTargetOptions &Options);

MCObjectWriter *createMipsELFObjectWriter(raw_pwrite_stream &OS, uint8_t OSABI,
                                          bool IsLittleEndian, bool Is64Bit);

namespace MIPS_MC {
StringRef selectMipsCPU(const Triple &TT, StringRef CPU);
}

} // End llvm namespace

// Defines symbolic names for Mips registers.  This defines a mapping from
// register name to register number.
#define GET_REGINFO_ENUM
#include "MipsGenRegisterInfo.inc"

// Defines symbolic names for the Mips instructions.
#define GET_INSTRINFO_ENUM
#include "MipsGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "MipsGenSubtargetInfo.inc"

#endif
