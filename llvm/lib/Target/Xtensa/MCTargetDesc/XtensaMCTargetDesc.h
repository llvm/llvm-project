//===-- XtensaMCTargetDesc.h - Xtensa Target Descriptions -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides Xtensa specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_XTENSA_MCTARGETDESC_XTENSAMCTARGETDESC_H
#define LLVM_LIB_TARGET_XTENSA_MCTARGETDESC_XTENSAMCTARGETDESC_H
#include "llvm/Support/DataTypes.h"
#include <memory>

namespace llvm {

class FeatureBitset;
class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCInstrInfo;
class MCObjectTargetWriter;
class MCObjectWriter;
class MCRegister;
class MCRegisterInfo;
class MCSubtargetInfo;
class MCTargetOptions;
class MachineInstr;
class StringRef;
class Target;
class raw_ostream;

extern Target TheXtensaTarget;

MCCodeEmitter *createXtensaMCCodeEmitter(const MCInstrInfo &MCII,
                                         MCContext &Ctx);

MCAsmBackend *createXtensaAsmBackend(const Target &T,
                                     const MCSubtargetInfo &STI,
                                     const MCRegisterInfo &MRI,
                                     const MCTargetOptions &Options);
std::unique_ptr<MCObjectTargetWriter>
createXtensaObjectWriter(uint8_t OSABI, bool IsLittleEndian);

namespace Xtensa {
// Check address offset for load/store instructions.
// The offset should be multiple of scale.
bool isValidAddrOffset(int Scale, int64_t OffsetVal);

// Check address offset for load/store instructions.
bool isValidAddrOffsetForOpcode(unsigned Opcode, int64_t Offset);

enum RegisterAccessType {
  REGISTER_WRITE = 1,
  REGISTER_READ = 2,
  REGISTER_EXCHANGE = 3
};

// Verify if it's correct to use a special register.
bool checkRegister(MCRegister RegNo, const FeatureBitset &FeatureBits,
                   RegisterAccessType RA);

// Get Xtensa User Register by register encoding value.
MCRegister getUserRegister(unsigned Code, const MCRegisterInfo &MRI);
} // namespace Xtensa
} // end namespace llvm

// Defines symbolic names for Xtensa registers.
// This defines a mapping from register name to register number.
#define GET_REGINFO_ENUM
#include "XtensaGenRegisterInfo.inc"

// Defines symbolic names for the Xtensa instructions.
#define GET_INSTRINFO_ENUM
#include "XtensaGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "XtensaGenSubtargetInfo.inc"

#endif // LLVM_LIB_TARGET_XTENSA_MCTARGETDESC_XTENSAMCTARGETDESC_H
