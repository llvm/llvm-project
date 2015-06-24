//===-- AMDGPUMCTargetDesc.h - AMDGPU Target Descriptions -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief Provides AMDGPU specific target descriptions.
//
//===----------------------------------------------------------------------===//
//

#ifndef LLVM_LIB_TARGET_R600_MCTARGETDESC_AMDGPUMCTARGETDESC_H
#define LLVM_LIB_TARGET_R600_MCTARGETDESC_AMDGPUMCTARGETDESC_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {
class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCInstrInfo;
class MCObjectWriter;
class MCRegisterInfo;
class MCSubtargetInfo;
class Target;
class raw_ostream;

extern Target TheAMDGPUTarget;
extern Target TheGCNTarget;

MCCodeEmitter *createR600MCCodeEmitter(const MCInstrInfo &MCII,
                                       const MCRegisterInfo &MRI,
                                       const MCSubtargetInfo &STI);

MCCodeEmitter *createSIMCCodeEmitter(const MCInstrInfo &MCII,
                                     const MCRegisterInfo &MRI,
                                     const MCSubtargetInfo &STI,
                                     MCContext &Ctx);

MCAsmBackend *createAMDGPUAsmBackend(const Target &T, const MCRegisterInfo &MRI,
                                     StringRef TT, StringRef CPU);

MCObjectWriter *createAMDGPUELFObjectWriter(raw_ostream &OS);
} // End llvm namespace

#define GET_REGINFO_ENUM
#include "AMDGPUGenRegisterInfo.inc"

#define GET_INSTRINFO_ENUM
#include "AMDGPUGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "AMDGPUGenSubtargetInfo.inc"

#endif
