//===-- P2MCTargetDesc.h - P2 Target Descriptions -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides P2 specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_P2_P2MCTARGETDESC_H
#define LLVM_LIB_TARGET_P2_P2MCTARGETDESC_H

#include "llvm/Support/DataTypes.h"
#include <memory>

namespace llvm {
    class MCAsmBackend;
    class MCCodeEmitter;
    class MCContext;
    class MCInstrInfo;
    class MCObjectWriter;
    class MCObjectTargetWriter;
    class MCRegisterInfo;
    class MCSubtargetInfo;
    class MCTargetOptions;
    class StringRef;
    class Target;
    class Triple;
    class raw_ostream;
    class raw_pwrite_stream;

    MCCodeEmitter *createP2MCCodeEmitter(const MCInstrInfo &MCII, const MCRegisterInfo &MRI, MCContext &Ctx);

    MCAsmBackend *createP2AsmBackend(const Target &T, const MCSubtargetInfo &STI, const MCRegisterInfo &MRI, const llvm::MCTargetOptions &TO);

    std::unique_ptr<MCObjectTargetWriter> createP2ELFObjectWriter(uint8_t OSABI);

    extern Target TheP2Target;

} // End llvm namespace

// Defines symbolic names for P2 registers.  This defines a mapping from
// register name to register number.
#define GET_REGINFO_ENUM
#include "P2GenRegisterInfo.inc"

// Defines symbolic names for the P2 instructions.
#define GET_INSTRINFO_ENUM
#include "P2GenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "P2GenSubtargetInfo.inc"

#endif