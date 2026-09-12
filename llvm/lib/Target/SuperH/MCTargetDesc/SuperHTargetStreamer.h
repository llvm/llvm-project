//===-- SuperHTargetStreamer.h - SuperH Target Streamer --------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SUPERH_MCTARGETDESC_SUPERHTARGETSTREAMER_H
#define LLVM_LIB_TARGET_SUPERH_MCTARGETDESC_SUPERHTARGETSTREAMER_H

#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCStreamer.h"

namespace llvm {
class formatted_raw_ostream;

class SuperHTargetStreamer : public MCTargetStreamer {
public:
  SuperHTargetStreamer(MCStreamer &S);
};

// This part is for ascii assembly output
class SuperHTargetAsmStreamer : public SuperHTargetStreamer {
  formatted_raw_ostream &OS;

public:
  SuperHTargetAsmStreamer(MCStreamer &S, formatted_raw_ostream &OS);
};

// This part is for ELF object output
class SuperHTargetELFStreamer : public SuperHTargetStreamer {
public:
  SuperHTargetELFStreamer(MCStreamer &S, const MCSubtargetInfo &STI);
  MCELFStreamer &getStreamer();
};
} // end namespace llvm

#endif