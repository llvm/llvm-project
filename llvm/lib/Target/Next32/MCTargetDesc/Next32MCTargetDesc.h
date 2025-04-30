//===-- Next32MCTargetDesc.h - Next32 Target Descriptions -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides Next32 specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_Next32_MCTARGETDESC_Next32MCTARGETDESC_H
#define LLVM_LIB_TARGET_Next32_MCTARGETDESC_Next32MCTARGETDESC_H

#include "llvm/Config/config.h"
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
class FeatureBitset;

Target &getTheNext32Target();

MCCodeEmitter *createNext32MCCodeEmitter(const MCInstrInfo &MCII,
                                         MCContext &Ctx);

MCAsmBackend *createNext32AsmBackend(const Target &T,
                                     const MCSubtargetInfo &STI,
                                     const MCRegisterInfo &MRI,
                                     const MCTargetOptions &Options);

std::unique_ptr<MCObjectTargetWriter>
createNext32ELFObjectWriter(uint8_t OSABI);
} // namespace llvm

// Defines symbolic names for Next32 registers.  This defines a mapping from
// register name to register number.
//
#define GET_REGINFO_ENUM
#include "Next32GenRegisterInfo.inc"

// Defines symbolic names for the Next32 instructions.
//
#define GET_INSTRINFO_ENUM
#define GET_INSTRINFO_MC_HELPER_DECLS
#include "Next32GenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "Next32GenSubtargetInfo.inc"

#endif
