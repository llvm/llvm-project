#ifndef LLVM_LIB_TARGET_INARCH_MCTARGETDESC_INARCHMCTARGETDESC_H
#define LLVM_LIB_TARGET_INARCH_MCTARGETDESC_INARCHMCTARGETDESC_H

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

MCCodeEmitter *createInArchMCCodeEmitter(const MCInstrInfo &MCII, MCContext &Ctx);
MCAsmBackend *createInArchAsmBackend(const Target &T, const MCSubtargetInfo &STI,
                                  const MCRegisterInfo &MRI,
                                  const MCTargetOptions &Options);
std::unique_ptr<MCObjectTargetWriter> createInArchELFObjectWriter(bool Is64Bit,
                                                               uint8_t OSABI);
} // namespace llvm

// Defines symbolic names for InArch registers.  This defines a mapping from
// register name to register number.
//
#define GET_REGINFO_ENUM
#include "InArchGenRegisterInfo.inc"

#define GET_INSTRINFO_ENUM
#include "InArchGenInstrInfo.inc"

#endif // LLVM_LIB_TARGET_INARCH_MCTARGETDESC_INARCHMCTARGETDESC_H