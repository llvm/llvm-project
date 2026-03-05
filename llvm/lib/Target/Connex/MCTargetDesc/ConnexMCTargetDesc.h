//===-- ConnexMCTargetDesc.h - Connex Target Descriptions -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides Connex specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_CONNEX_MCTARGETDESC_CONNEXMCTARGETDESC_H
#define LLVM_LIB_TARGET_CONNEX_MCTARGETDESC_CONNEXMCTARGETDESC_H

#include "llvm/Config/config.h"
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
class raw_ostream;
class raw_pwrite_stream;

extern Target TheConnexTarget;

MCCodeEmitter *createConnexMCCodeEmitter(const MCInstrInfo &MCII,
                                         MCContext &Ctx);

MCAsmBackend *createConnexAsmBackend(const Target &T,
                                     const MCSubtargetInfo &STI,
                                     const MCRegisterInfo &MRI,
                                     const MCTargetOptions &Options);

std::unique_ptr<MCObjectTargetWriter>
createConnexELFObjectWriter(uint8_t OSABI);
} // namespace llvm

// Defines symbolic names for Connex registers.  This defines a mapping from
// register name to register number.
//
#define GET_REGINFO_ENUM
#include "ConnexGenRegisterInfo.inc"

// Defines symbolic names for the Connex instructions.
//
#define GET_INSTRINFO_ENUM
#include "ConnexGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "ConnexGenSubtargetInfo.inc"

#endif
