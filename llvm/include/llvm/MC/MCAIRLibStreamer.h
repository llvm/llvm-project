//===- MCAIRLibStreamer.h - MCAIRLibStreamer Interface ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Overrides MCObjectStreamer to disable unnecessary features with stubs. The
// AIR `.metallib` container is an opaque, monolithic blob produced by the
// AIR target's embedder pass; the streamer just routes its bytes through
// to the AIRLibObjectWriter.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCAIRLIBSTREAMER_H
#define LLVM_MC_MCAIRLIBSTREAMER_H

#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCObjectWriter.h"

namespace llvm {
class MCInst;
class raw_ostream;

/// `MCObjectStreamer` for the AIRLib container.
///
/// Stubs the symbol/common-symbol hooks because the metallib payload is a
/// single opaque byte blob assembled by the embedder pass; no MC-level
/// symbol bookkeeping is needed before the writer dumps it out.
class MCAIRLibStreamer : public MCObjectStreamer {
public:
  MCAIRLibStreamer(MCContext &Context, std::unique_ptr<MCAsmBackend> TAB,
                     std::unique_ptr<MCObjectWriter> OW,
                     std::unique_ptr<MCCodeEmitter> Emitter)
      : MCObjectStreamer(Context, std::move(TAB), std::move(OW),
                         std::move(Emitter)) {}

  bool emitSymbolAttribute(MCSymbol *, MCSymbolAttr) override { return false; }
  void emitCommonSymbol(MCSymbol *, uint64_t, Align) override {}
};

} // end namespace llvm

#endif // LLVM_MC_MCAIRLIBSTREAMER_H
