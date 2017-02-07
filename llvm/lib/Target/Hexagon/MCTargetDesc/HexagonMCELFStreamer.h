//===- HexagonMCELFStreamer.h - Hexagon subclass of MCElfStreamer ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_MCTARGETDESC_HEXAGONMCELFSTREAMER_H
#define LLVM_LIB_TARGET_HEXAGON_MCTARGETDESC_HEXAGONMCELFSTREAMER_H

#include "MCTargetDesc/HexagonMCTargetDesc.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCInstrInfo.h"
#include <cstdint>
#include <memory>

namespace llvm {

class HexagonMCELFStreamer : public MCELFStreamer {
  std::unique_ptr<MCInstrInfo> MCII;

public:
  HexagonMCELFStreamer(MCContext &Context, MCAsmBackend &TAB,
                       raw_pwrite_stream &OS, MCCodeEmitter *Emitter)
      : MCELFStreamer(Context, TAB, OS, Emitter),
        MCII(createHexagonMCInstrInfo()) {}

  HexagonMCELFStreamer(MCContext &Context,
                       MCAsmBackend &TAB,
                       raw_pwrite_stream &OS, MCCodeEmitter *Emitter,
                       MCAssembler *Assembler) :
  MCELFStreamer(Context, TAB, OS, Emitter),
  MCII (createHexagonMCInstrInfo()) {}

  void EmitInstruction(const MCInst &Inst, const MCSubtargetInfo &STI) override;
  void EmitSymbol(const MCInst &Inst);
  void HexagonMCEmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                      unsigned ByteAlignment,
                                      unsigned AccessSize);
  void HexagonMCEmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                 unsigned ByteAlignment, unsigned AccessSize);
};

MCStreamer *createHexagonELFStreamer(Triple const &TT, MCContext &Context,
                                     MCAsmBackend &MAB, raw_pwrite_stream &OS,
                                     MCCodeEmitter *CE);

} // end namespace llvm

#endif // LLVM_LIB_TARGET_HEXAGON_MCTARGETDESC_HEXAGONMCELFSTREAMER_H
