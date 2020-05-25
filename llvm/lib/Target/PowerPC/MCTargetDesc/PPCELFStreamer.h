//===- PPCELFStreamer.h - ELF Object Output --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a custom MCELFStreamer for PowerPC.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PPC_MCELFSTREAMER_PPCELFSTREAMER_H
#define LLVM_LIB_TARGET_PPC_MCELFSTREAMER_PPCELFSTREAMER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCELFStreamer.h"
#include <memory>

namespace llvm {

class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCSubtargetInfo;

class PPCELFStreamer : public MCELFStreamer {
  // We need to keep track of the last label we emitted (only one) because
  // depending on whether the label is on the same line as an aligned
  // instruction or not, the label may refer to the instruction or the nop.
  MCSymbol *LastLabel;
  SMLoc LastLabelLoc;

public:
  PPCELFStreamer(MCContext &Context, std::unique_ptr<MCAsmBackend> MAB,
                 std::unique_ptr<MCObjectWriter> OW,
                 std::unique_ptr<MCCodeEmitter> Emitter);

  void emitInstruction(const MCInst &Inst, const MCSubtargetInfo &STI) override;

  // EmitLabel updates LastLabel and LastLabelLoc when a new label is emitted.
  void emitLabel(MCSymbol *Symbol, SMLoc Loc = SMLoc()) override;
private:
  void emitPrefixedInstruction(const MCInst &Inst, const MCSubtargetInfo &STI);
};

MCELFStreamer *createPPCELFStreamer(MCContext &Context,
                                    std::unique_ptr<MCAsmBackend> MAB,
                                    std::unique_ptr<MCObjectWriter> OW,
                                    std::unique_ptr<MCCodeEmitter> Emitter);
} // end namespace llvm

#endif // LLVM_LIB_TARGET_PPC_MCELFSTREAMER_PPCELFSTREAMER_H
