//===- Next32ELFStreamer.h - ELF Object Output ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a custom MCELFStreamer which allows us to insert some hooks before
// emitting data into an actual object file.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NEXT32_MCTARGETDESC_NEXT32ELFSTREAMER_H
#define LLVM_LIB_TARGET_NEXT32_MCTARGETDESC_NEXT32ELFSTREAMER_H

#include "Next32FixupKinds.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCELFStreamer.h"
#include <memory>

namespace llvm {

class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCSubtargetInfo;

class Next32ELFStreamer : public MCELFStreamer {
public:
  Next32ELFStreamer(MCContext &Context, std::unique_ptr<MCAsmBackend> MAB,
                    std::unique_ptr<MCObjectWriter> OW,
                    std::unique_ptr<MCCodeEmitter> Emitter);

  void emitValueImpl(const MCExpr *Value, unsigned Size, SMLoc Loc) override;

private:
  void CreateFixup(const MCExpr *Value, SMLoc Loc, MCDataFragment *DF,
                   Next32::Fixups Kind);
};

MCELFStreamer *createNext32ELFStreamer(MCContext &Context,
                                       std::unique_ptr<MCAsmBackend> MAB,
                                       std::unique_ptr<MCObjectWriter> OW,
                                       std::unique_ptr<MCCodeEmitter> Emitter);
} // end namespace llvm

#endif // LLVM_LIB_TARGET_NEXT32_MCTARGETDESC_NEXT32ELFSTREAMER_H
