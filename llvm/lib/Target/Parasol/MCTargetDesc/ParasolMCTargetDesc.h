//===-- ParasolMCTargetDesc.h - Parasol Target Descriptions ---------------===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
//
// This file provides Parasol specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PARASOL_MCTARGETDESC_PARASOLMCTARGETDESC_H
#define LLVM_LIB_TARGET_PARASOL_MCTARGETDESC_PARASOLMCTARGETDESC_H

// Defines symbolic names for Parasol registers. This defines a mapping from
// register name to register number.
#define GET_REGINFO_ENUM
#include "ParasolGenRegisterInfo.inc"

// Defines symbolic names for the Parasol instructions.
#define GET_INSTRINFO_ENUM
#include "ParasolGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "ParasolGenSubtargetInfo.inc"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"

// For Linux support to solve
// error: ‘unique_ptr’ in namespace ‘std’ does not name a template type
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

MCCodeEmitter *createParasolMCCodeEmitter(const MCInstrInfo &MCII,
                                          MCContext &Ctx);
MCAsmBackend *createParasolAsmBackend(const Target &T,
                                      const MCSubtargetInfo &STI,
                                      const MCRegisterInfo &MRI,
                                      const MCTargetOptions &Options);
std::unique_ptr<MCObjectTargetWriter>
createParasolELFObjectWriter(bool Is64Bit, uint8_t OSABI);

} // namespace llvm

#endif // end LLVM_LIB_TARGET_PARASOL_MCTARGETDESC_PARASOLMCTARGETDESC_H
