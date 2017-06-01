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

#ifndef LLVM_LIB_TARGET_ARM_MCTARGETDESC_ARMMCTARGETDESC_H
#define LLVM_LIB_TARGET_ARM_MCTARGETDESC_ARMMCTARGETDESC_H

#include "llvm/Support/DataTypes.h"
#include <string>

namespace llvm {
class formatted_raw_ostream;
class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCInstrInfo;
class MCInstPrinter;
class MCObjectWriter;
class MCRegisterInfo;
class MCSubtargetInfo;
class MCStreamer;
class MCTargetOptions;
class MCRelocationInfo;
class MCTargetStreamer;
class StringRef;
class Target;
class Triple;
class raw_ostream;
class raw_pwrite_stream;

Target &getTheARMLETarget();
Target &getTheThumbLETarget();
Target &getTheARMBETarget();
Target &getTheThumbBETarget();

namespace ARM_MC {
std::string ParseARMTriple(const Triple &TT, StringRef CPU);

/// Create a ARM MCSubtargetInfo instance. This is exposed so Asm parser, etc.
/// do not need to go through TargetRegistry.
MCSubtargetInfo *createARMMCSubtargetInfo(const Triple &TT, StringRef CPU,
                                          StringRef FS);
}

MCTargetStreamer *createARMNullTargetStreamer(MCStreamer &S);
MCTargetStreamer *createARMTargetAsmStreamer(MCStreamer &S,
                                             formatted_raw_ostream &OS,
                                             MCInstPrinter *InstPrint,
                                             bool isVerboseAsm);
MCTargetStreamer *createARMObjectTargetStreamer(MCStreamer &S,
                                                const MCSubtargetInfo &STI);

MCCodeEmitter *createARMLEMCCodeEmitter(const MCInstrInfo &MCII,
                                        const MCRegisterInfo &MRI,
                                        MCContext &Ctx);

MCCodeEmitter *createARMBEMCCodeEmitter(const MCInstrInfo &MCII,
                                        const MCRegisterInfo &MRI,
                                        MCContext &Ctx);

MCAsmBackend *createARMAsmBackend(const Target &T, const MCRegisterInfo &MRI,
                                  const Triple &TT, StringRef CPU,
                                  const MCTargetOptions &Options,
                                  bool IsLittleEndian);

MCAsmBackend *createARMLEAsmBackend(const Target &T, const MCRegisterInfo &MRI,
                                    const Triple &TT, StringRef CPU,
                                    const MCTargetOptions &Options);

MCAsmBackend *createARMBEAsmBackend(const Target &T, const MCRegisterInfo &MRI,
                                    const Triple &TT, StringRef CPU,
                                    const MCTargetOptions &Options);

MCAsmBackend *createThumbLEAsmBackend(const Target &T,
                                      const MCRegisterInfo &MRI,
                                      const Triple &TT, StringRef CPU,
                                      const MCTargetOptions &Options);

MCAsmBackend *createThumbBEAsmBackend(const Target &T,
                                      const MCRegisterInfo &MRI,
                                      const Triple &TT, StringRef CPU,
                                      const MCTargetOptions &Options);

// Construct a PE/COFF machine code streamer which will generate a PE/COFF
// object file.
MCStreamer *createARMWinCOFFStreamer(MCContext &Context, MCAsmBackend &MAB,
                                     raw_pwrite_stream &OS,
                                     MCCodeEmitter *Emitter, bool RelaxAll,
                                     bool IncrementalLinkerCompatible);

/// Construct an ELF Mach-O object writer.
MCObjectWriter *createARMELFObjectWriter(raw_pwrite_stream &OS, uint8_t OSABI,
                                         bool IsLittleEndian);

/// Construct an ARM Mach-O object writer.
MCObjectWriter *createARMMachObjectWriter(raw_pwrite_stream &OS, bool Is64Bit,
                                          uint32_t CPUType,
                                          uint32_t CPUSubtype);

/// Construct an ARM PE/COFF object writer.
MCObjectWriter *createARMWinCOFFObjectWriter(raw_pwrite_stream &OS,
                                             bool Is64Bit);

/// Construct ARM Mach-O relocation info.
MCRelocationInfo *createARMMachORelocationInfo(MCContext &Ctx);
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
