//===-- LoongArchWinCOFFStreamer.h -----------------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_LOONGARCH_MCTARGETDESC_LOONGARCHWINCOFFSTREAMER_H
#define LLVM_LIB_TARGET_LOONGARCH_MCTARGETDESC_LOONGARCHWINCOFFSTREAMER_H

#include "LoongArchTargetStreamer.h"
#include "llvm/MC/MCWinCOFFStreamer.h"

namespace llvm {

class LoongArchTargetWinCOFFStreamer : public LoongArchTargetStreamer {
public:
  LoongArchTargetWinCOFFStreamer(MCStreamer &S) : LoongArchTargetStreamer(S) {}
};

MCStreamer *
createLoongArchWinCOFFStreamer(MCContext &C,
                               std::unique_ptr<MCAsmBackend> &&MAB,
                               std::unique_ptr<MCObjectWriter> &&MOW,
                               std::unique_ptr<MCCodeEmitter> &&MCE);
} // end namespace llvm
#endif
