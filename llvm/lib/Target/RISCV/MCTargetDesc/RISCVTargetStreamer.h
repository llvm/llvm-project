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

  virtual void emitDirectiveOptionRVC() = 0;
  virtual void emitDirectiveOptionNoRVC() = 0;
};

// This part is for ascii assembly output
class RISCVTargetAsmStreamer : public RISCVTargetStreamer {
  formatted_raw_ostream &OS;

public:
  RISCVTargetAsmStreamer(MCStreamer &S, formatted_raw_ostream &OS);

  void emitDirectiveOptionRVC() override;
  void emitDirectiveOptionNoRVC() override;
};

}
#endif
