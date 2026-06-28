//===-- SuperHMCTargetDesc.h - SuperH Target Descriptions -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file provides SuperH specific target descriptions.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SUPERH_MCTARGETDESC_SUPERHMCTARGETDESC_H
#define LLVM_LIB_TARGET_SUPERH_MCTARGETDESC_SUPERHMCTARGETDESC_H

#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {
class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCInstrInfo;
class MCRegisterInfo;
class MCSubtargetInfo;
class MCRelocationInfo;
class MCTargetOptions;
class Target;
class Triple;
class StringRef;
class raw_ostream;
class raw_pwrite_stream;

MCCodeEmitter *createSuperHMCCodeEmitter(const MCInstrInfo &MCII,
                                              MCContext &Ctx);

MCAsmBackend *createSuperHAsmBackend(const Target &T, const MCSubtargetInfo &STI,
                                    const MCRegisterInfo &MRI,
                                    const MCTargetOptions &Options);

std::unique_ptr<MCObjectTargetWriter> createSuperHELFObjectWriter(uint8_t OSABI);
}

#define GET_REGINFO_ENUM
#include "SuperHGenRegisterInfo.inc"

#define GET_INSTRINFO_ENUM
#define GET_INSTRINFO_MC_HELPER_DECLS
#include "SuperHGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "SuperHGenSubtargetInfo.inc"

#endif // LLVM_LIB_TARGET_SUPERH_MCTARGETDESC_SUPERHMCTARGETDESC_H