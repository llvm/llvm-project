//===-- RISCVTargetStreamer.h - RISCV Target Streamer ----------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVTARGETSTREAMER_H
#define LLVM_LIB_TARGET_RISCV_RISCVTARGETSTREAMER_H

#include "llvm/MC/MCStreamer.h"

namespace llvm {

class RISCVTargetStreamer : public MCTargetStreamer {
public:
  RISCVTargetStreamer(MCStreamer &S);
};
}
#endif
