//===-- XtensaTargetStreamer.h - Xtensa Target Streamer --------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_XTENSA_XTENSATARGETSTREAMER_H
#define LLVM_LIB_TARGET_XTENSA_XTENSATARGETSTREAMER_H

#include "XtensaConstantPoolValue.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/SMLoc.h"

namespace llvm {
class XtensaTargetStreamer : public MCTargetStreamer {
public:
  XtensaTargetStreamer(MCStreamer &S);
  virtual void emitLiteral(MCSymbol *LblSym, const MCExpr *Value, SMLoc L) = 0;
  virtual void emitLiteralLabel(MCSymbol *LblSym, SMLoc L) = 0;
  virtual void emitLiteral(const MCExpr *Value, SMLoc L) = 0;
  virtual void emitLiteral(std::string str) = 0;
};

class XtensaTargetAsmStreamer : public XtensaTargetStreamer {
  formatted_raw_ostream &OS;

public:
  XtensaTargetAsmStreamer(MCStreamer &S, formatted_raw_ostream &OS);
  void emitLiteral(MCSymbol *LblSym, const MCExpr *Value, SMLoc L) override {}
  void emitLiteralLabel(MCSymbol *LblSym, SMLoc L) override {}
  void emitLiteral(const MCExpr *Value, SMLoc L) override {}
  void emitLiteral(std::string str) override;
};

class XtensaTargetELFStreamer : public XtensaTargetStreamer {
public:
  XtensaTargetELFStreamer(MCStreamer &S);
  MCELFStreamer &getStreamer();
  void emitLiteral(MCSymbol *LblSym, const MCExpr *Value, SMLoc L) override;
  void emitLiteralLabel(MCSymbol *LblSym, SMLoc L) override;
  void emitLiteral(const MCExpr *Value, SMLoc L) override;
  void emitLiteral(std::string str) override {}
};
} // end namespace llvm

#endif
