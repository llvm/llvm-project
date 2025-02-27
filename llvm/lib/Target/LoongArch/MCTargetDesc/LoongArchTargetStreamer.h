//===-- LoongArchTargetStreamer.h - LoongArch Target Streamer --*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_LOONGARCH_MCTARGETDESC_LOONGARCHTARGETSTREAMER_H
#define LLVM_LIB_TARGET_LOONGARCH_MCTARGETDESC_LOONGARCHTARGETSTREAMER_H

#include "LoongArch.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/FormattedStream.h"

namespace llvm {
class LoongArchTargetStreamer : public MCTargetStreamer {
  LoongArchABI::ABI TargetABI = LoongArchABI::ABI_Unknown;

public:
  LoongArchTargetStreamer(MCStreamer &S);
  void setTargetABI(LoongArchABI::ABI ABI);
  LoongArchABI::ABI getTargetABI() const { return TargetABI; }

  virtual void emitDirectiveOptionPush();
  virtual void emitDirectiveOptionPop();
  virtual void emitDirectiveOptionRelax();
  virtual void emitDirectiveOptionNoRelax();
};

// This part is for ascii assembly output.
class LoongArchTargetAsmStreamer : public LoongArchTargetStreamer {
  formatted_raw_ostream &OS;

public:
  LoongArchTargetAsmStreamer(MCStreamer &S, formatted_raw_ostream &OS);

  void emitDirectiveOptionPush() override;
  void emitDirectiveOptionPop() override;
  void emitDirectiveOptionRelax() override;
  void emitDirectiveOptionNoRelax() override;
};

} // end namespace llvm
#endif
