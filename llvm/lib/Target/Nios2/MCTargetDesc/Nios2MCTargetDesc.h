//===-- Nios2MCTargetDesc.h - Nios2 Target Descriptions ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Nios2 specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NIOS2_MCTARGETDESC_NIOS2MCTARGETDESC_H
#define LLVM_LIB_TARGET_NIOS2_MCTARGETDESC_NIOS2MCTARGETDESC_H

#include <memory>

namespace llvm {
class MCAsmBackend;
class MCObjectWriter;
class MCRegisterInfo;
class MCSubtargetInfo;
class MCTargetOptions;
class Target;
class Triple;
class StringRef;
class raw_pwrite_stream;

Target &getTheNios2Target();

MCAsmBackend *createNios2AsmBackend(const Target &T, const MCSubtargetInfo &STI,
                                    const MCRegisterInfo &MRI,
                                    const MCTargetOptions &Options);

std::unique_ptr<MCObjectWriter>
createNios2ELFObjectWriter(raw_pwrite_stream &OS, uint8_t OSABI);

} // namespace llvm

// Defines symbolic names for Nios2 registers.  This defines a mapping from
// register name to register number.
#define GET_REGINFO_ENUM
#include "Nios2GenRegisterInfo.inc"

// Defines symbolic names for the Nios2 instructions.
#define GET_INSTRINFO_ENUM
#include "Nios2GenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "Nios2GenSubtargetInfo.inc"

#endif
