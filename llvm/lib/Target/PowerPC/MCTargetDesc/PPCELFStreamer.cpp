//===-------- PPCELFStreamer.cpp - ELF Object Output ---------------------===//
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
// The purpose of the custom ELF streamer is to allow us to intercept
// instructions as they are being emitted and align all 8 byte instructions
// to a 64 byte boundary if required (by adding a 4 byte nop). This is important
// because 8 byte instructions are not allowed to cross 64 byte boundaries
// and by aliging anything that is within 4 bytes of the boundary we can
// guarantee that the 8 byte instructions do not cross that boundary.
//
//===----------------------------------------------------------------------===//


#include "PPCELFStreamer.h"
#include "PPCInstrInfo.h"
#include "PPCMCCodeEmitter.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SourceMgr.h"

using namespace llvm;

PPCELFStreamer::PPCELFStreamer(MCContext &Context,
                               std::unique_ptr<MCAsmBackend> MAB,
                               std::unique_ptr<MCObjectWriter> OW,
                               std::unique_ptr<MCCodeEmitter> Emitter)
    : MCELFStreamer(Context, std::move(MAB), std::move(OW),
                    std::move(Emitter)), LastLabel(NULL) {
}

void PPCELFStreamer::emitPrefixedInstruction(const MCInst &Inst,
                                             const MCSubtargetInfo &STI) {
  // Prefixed instructions must not cross a 64-byte boundary (i.e. prefix is
  // before the boundary and the remaining 4-bytes are after the boundary). In
  // order to achieve this, a nop is added prior to any such boundary-crossing
  // prefixed instruction. Align to 64 bytes if possible but add a maximum of 4
  // bytes when trying to do that. If alignment requires adding more than 4
  // bytes then the instruction won't be aligned. When emitting a code alignment
  // a new fragment is created for this alignment. This fragment will contain
  // all of the nops required as part of the alignment operation. In the cases
  // when no nops are added then The fragment is still created but it remains
  // empty.
  emitCodeAlignment(64, 4);

  // Emit the instruction.
  // Since the previous emit created a new fragment then adding this instruction
  // also forces the addition of a new fragment. Inst is now the first
  // instruction in that new fragment.
  MCELFStreamer::emitInstruction(Inst, STI);

  // The above instruction is forced to start a new fragment because it
  // comes after a code alignment fragment. Get that new fragment.
  MCFragment *InstructionFragment = getCurrentFragment();
  SMLoc InstLoc = Inst.getLoc();
  // Check if there was a last label emitted.
  if (LastLabel && !LastLabel->isUnset() && LastLabelLoc.isValid() &&
      InstLoc.isValid()) {
    const SourceMgr *SourceManager = getContext().getSourceManager();
    unsigned InstLine = SourceManager->FindLineNumber(InstLoc);
    unsigned LabelLine = SourceManager->FindLineNumber(LastLabelLoc);
    // If the Label and the Instruction are on the same line then move the
    // label to the top of the fragment containing the aligned instruction that
    // was just added.
    if (InstLine == LabelLine) {
      AssignFragment(LastLabel, InstructionFragment);
      LastLabel->setOffset(0);
    }
  }
}

void PPCELFStreamer::emitInstruction(const MCInst &Inst,
                                     const MCSubtargetInfo &STI) {
  PPCMCCodeEmitter *Emitter =
      static_cast<PPCMCCodeEmitter*>(getAssembler().getEmitterPtr());

  // Special handling is only for prefixed instructions.
  if (!Emitter->isPrefixedInstruction(Inst)) {
    MCELFStreamer::emitInstruction(Inst, STI);
    return;
  }
  emitPrefixedInstruction(Inst, STI);
}

void PPCELFStreamer::emitLabel(MCSymbol *Symbol, SMLoc Loc) {
  LastLabel = Symbol;
  LastLabelLoc = Loc;
  MCELFStreamer::emitLabel(Symbol);
}

MCELFStreamer *llvm::createPPCELFStreamer(
    MCContext &Context, std::unique_ptr<MCAsmBackend> MAB,
    std::unique_ptr<MCObjectWriter> OW,
    std::unique_ptr<MCCodeEmitter> Emitter) {
  return new PPCELFStreamer(Context, std::move(MAB), std::move(OW),
                            std::move(Emitter));
}
