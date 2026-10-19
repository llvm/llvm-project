//===-- AGXMCTargetDesc.h - AGX target descriptions -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides AGX specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AGX_MCTARGETDESC_AGXMCTARGETDESC_H
#define LLVM_LIB_TARGET_AGX_MCTARGETDESC_AGXMCTARGETDESC_H

#include "Utils/AGXBaseInfo.h"
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
class Target;

MCCodeEmitter *createAGXMCCodeEmitter(const MCInstrInfo &MCII,
                                      MCContext &Ctx);

MCAsmBackend *createAGXAsmBackend(const Target &T, const MCSubtargetInfo &STI,
                                  const MCRegisterInfo &MRI,
                                  const MCTargetOptions &Options);

std::unique_ptr<MCObjectTargetWriter> createAGXELFObjectWriter(uint8_t OSABI);

} // namespace llvm

#define GET_REGINFO_ENUM
#include "AGXGenRegisterInfo.inc"

#define GET_INSTRINFO_ENUM
#define GET_INSTRINFO_OPERAND_ENUM
#define GET_INSTRINFO_MC_HELPER_DECLS
#include "AGXGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "AGXGenSubtargetInfo.inc"

#endif // LLVM_LIB_TARGET_AGX_MCTARGETDESC_AGXMCTARGETDESC_H
