//===-- M88kTargetStreamer.h - M88k Target Streamer ------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_M88K_MCTARGETDESC_M88KTARGETSTREAMER_H
#define LLVM_LIB_TARGET_M88K_MCTARGETDESC_M88KTARGETSTREAMER_H

#include "llvm/MC/MCStreamer.h"

namespace llvm {

class formatted_raw_ostream;
class MCELFStreamer;

class M88kTargetStreamer : public MCTargetStreamer {
public:
  M88kTargetStreamer(MCStreamer &S);
  ~M88kTargetStreamer() override;

  /// Callback used to implemnt the .requires_88110 directive.
  virtual void emitDirectiveRequires881100() = 0;
};

class M88kTargetAsmStreamer : public M88kTargetStreamer {
  formatted_raw_ostream &OS;

public:
  M88kTargetAsmStreamer(MCStreamer &S, formatted_raw_ostream &OS);
  void emitDirectiveRequires881100() override;
};

class M88kTargetELFStreamer : public M88kTargetStreamer {
  const MCSubtargetInfo &STI;
  MCStreamer &Streamer;
  bool Requires88110;

  MCELFStreamer &getStreamer();

public:
  M88kTargetELFStreamer(MCStreamer &S, const MCSubtargetInfo &STI);
  void emitDirectiveRequires881100() override;
  void finish() override;
};

} // end namespace llvm

#endif
