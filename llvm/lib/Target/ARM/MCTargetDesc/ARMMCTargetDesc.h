//===-- ARMMCTargetDesc.h - ARM Target Descriptions -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides ARM specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef ARMMCTARGETDESC_H
#define ARMMCTARGETDESC_H

#include "llvm/Support/DataTypes.h"
#include <string>

namespace llvm {
class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCInstrInfo;
class MCObjectWriter;
class MCRegisterInfo;
class MCSubtargetInfo;
class StringRef;
class Target;
class raw_ostream;

extern Target TheARMTarget, TheThumbTarget;

namespace ARM_MC {
  std::string ParseARMTriple(StringRef TT, StringRef CPU);

  /// createARMMCSubtargetInfo - Create a ARM MCSubtargetInfo instance.
  /// This is exposed so Asm parser, etc. do not need to go through
  /// TargetRegistry.
  MCSubtargetInfo *createARMMCSubtargetInfo(StringRef TT, StringRef CPU,
                                            StringRef FS);
}

MCCodeEmitter *createARMMCCodeEmitter(const MCInstrInfo &MCII,
                                      const MCRegisterInfo &MRI,
                                      const MCSubtargetInfo &STI,
                                      MCContext &Ctx);

MCAsmBackend *createARMAsmBackend(const Target &T, StringRef TT, StringRef CPU);

/// createARMELFObjectWriter - Construct an ELF Mach-O object writer.
MCObjectWriter *createARMELFObjectWriter(raw_ostream &OS,
                                         uint8_t OSABI);

/// createARMMachObjectWriter - Construct an ARM Mach-O object writer.
MCObjectWriter *createARMMachObjectWriter(raw_ostream &OS,
                                          bool Is64Bit,
                                          uint32_t CPUType,
                                          uint32_t CPUSubtype);

} // End llvm namespace

// Defines symbolic names for ARM registers.  This defines a mapping from
// register name to register number.
//
#define GET_REGINFO_ENUM
#include "ARMGenRegisterInfo.inc"

// Defines symbolic names for the ARM instructions.
//
#define GET_INSTRINFO_ENUM
#include "ARMGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "ARMGenSubtargetInfo.inc"

#endif
