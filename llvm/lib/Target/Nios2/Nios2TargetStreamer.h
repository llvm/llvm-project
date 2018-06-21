//===-- Nios2TargetStreamer.h - Nios2 Target Streamer ----------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_NIOS2_NIOS2TARGETSTREAMER_H
#define LLVM_LIB_TARGET_NIOS2_NIOS2TARGETSTREAMER_H

#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"

namespace llvm {

class Nios2TargetStreamer : public MCTargetStreamer {
public:
  Nios2TargetStreamer(MCStreamer &S);
};

// This part is for ascii assembly output
class Nios2TargetAsmStreamer : public Nios2TargetStreamer {
  formatted_raw_ostream &OS;

public:
  Nios2TargetAsmStreamer(MCStreamer &S, formatted_raw_ostream &OS);
};

} // namespace llvm
#endif
