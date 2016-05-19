//===-- AVRMCTargetDesc.h - AVR Target Descriptions -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides AVR specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_AVR_MCTARGET_DESC_H
#define LLVM_AVR_MCTARGET_DESC_H

#include "llvm/Support/DataTypes.h"

namespace llvm {

class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCInstrInfo;
class MCObjectWriter;
class MCRegisterInfo;
class StringRef;
class Target;
class Triple;
class raw_pwrite_stream;

extern Target TheAVRTarget;

/// Creates a machine code emitter for AVR.
MCCodeEmitter *createAVRMCCodeEmitter(const MCInstrInfo &MCII,
                                      const MCRegisterInfo &MRI,
                                      MCContext &Ctx);

/// Creates an assembly backend for AVR.
MCAsmBackend *createAVRAsmBackend(const Target &T, const MCRegisterInfo &MRI,
                                  const Triple &TT, StringRef CPU);

/// Creates an ELF object writer for AVR.
MCObjectWriter *createAVRELFObjectWriter(raw_pwrite_stream &OS, uint8_t OSABI);

} // end namespace llvm

#define GET_REGINFO_ENUM
#include "AVRGenRegisterInfo.inc"

#define GET_INSTRINFO_ENUM
#include "AVRGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "AVRGenSubtargetInfo.inc"

#endif // LLVM_AVR_MCTARGET_DESC_H
