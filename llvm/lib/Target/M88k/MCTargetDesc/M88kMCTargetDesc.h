//===-- M88kMCTargetDesc.h - M88k target descriptions -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_M88K_MCTARGETDESC_M88KMCTARGETDESC_H
#define LLVM_LIB_TARGET_M88K_MCTARGETDESC_M88KMCTARGETDESC_H

#include "llvm/Support/DataTypes.h"

#include <memory>

namespace llvm {

class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCInstrInfo;
class MCObjectTargetWriter;
class MCRegisterInfo;
class MCSubtargetInfo;
class MCTargetOptions;
class StringRef;
class Target;
class Triple;
class raw_pwrite_stream;
class raw_ostream;

MCCodeEmitter *createM88kMCCodeEmitter(const MCInstrInfo &MCII,
                                       MCContext &Ctx);

MCAsmBackend *createM88kMCAsmBackend(const Target &T,
                                     const MCSubtargetInfo &STI,
                                     const MCRegisterInfo &MRI,
                                     const MCTargetOptions &Options);

std::unique_ptr<MCObjectTargetWriter> createM88kObjectWriter(uint8_t OSABI);
} // end namespace llvm

// Defines symbolic names for M88k registers.
// This defines a mapping from register name to register number.
#define GET_REGINFO_ENUM
#include "M88kGenRegisterInfo.inc"

// Defines symbolic names for the M88k instructions.
#define GET_INSTRINFO_ENUM
#include "M88kGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "M88kGenSubtargetInfo.inc"

#endif
