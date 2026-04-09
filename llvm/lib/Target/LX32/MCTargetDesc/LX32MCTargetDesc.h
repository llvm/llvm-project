//===-- LX32MCTargetDesc.h - LX32 MC Target Description ------------------===//
//
// Part of the LX32 Project
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// This file declares LX32 MC target-description forward declarations.
// It is organized into the following sections:
//
//   Section 0 — Required LLVM includes
//   Section 1 — Forward declarations used by MC factories
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIB_TARGET_LX32_MCTARGETDESC_LX32MCTARGETDESC_H
#define LLVM_LIB_TARGET_LX32_MCTARGETDESC_LX32MCTARGETDESC_H

#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Support/DataTypes.h"
#include <memory>

namespace llvm {
class Target;
class MCInstrInfo;
class MCRegisterInfo;
class MCSubtargetInfo;
class MCContext;
class MCCodeEmitter;
class MCAsmBackend;
class MCTargetOptions;
class MCObjectTargetWriter;

MCCodeEmitter *createLX32MCCodeEmitter(const MCInstrInfo &MCII,
                                        MCContext &Ctx);

MCAsmBackend *createLX32AsmBackend(const Target &T,
                                    const MCSubtargetInfo &STI,
                                    const MCRegisterInfo &MRI,
                                    const MCTargetOptions &Options);

std::unique_ptr<MCObjectTargetWriter> createLX32ELFObjectWriter(uint8_t OSABI);

} // End llvm namespace

#endif // LLVM_LIB_TARGET_LX32_MCTARGETDESC_LX32MCTARGETDESC_H
