//===- PPCXCOFFStreamer.h - XCOFF Object Output -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a custom MCXCOFFStreamer for PowerPC.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PPC_MCXCOFFSTREAMER_PPCXCOFFSTREAMER_H
#define LLVM_LIB_TARGET_PPC_MCXCOFFSTREAMER_PPCXCOFFSTREAMER_H

#include "llvm/MC/MCXCOFFStreamer.h"

namespace llvm {

class PPCXCOFFStreamer : public MCXCOFFStreamer {
public:
  PPCXCOFFStreamer(MCContext &Context, std::unique_ptr<MCAsmBackend> MAB,
                   std::unique_ptr<MCObjectWriter> OW,
                   std::unique_ptr<MCCodeEmitter> Emitter);

  void emitInstruction(const MCInst &Inst, const MCSubtargetInfo &STI) override;

private:
  void emitPrefixedInstruction(const MCInst &Inst, const MCSubtargetInfo &STI);
};

MCXCOFFStreamer *createPPCXCOFFStreamer(MCContext &Context,
                                        std::unique_ptr<MCAsmBackend> MAB,
                                        std::unique_ptr<MCObjectWriter> OW,
                                        std::unique_ptr<MCCodeEmitter> Emitter);
} // end namespace llvm

#endif // LLVM_LIB_TARGET_PPC_MCXCOFFSTREAMER_PPCXCOFFSTREAMER_H
