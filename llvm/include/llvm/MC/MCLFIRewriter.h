//===- llvm/MC/MCLFIRewriter.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file was written by the LFI and Native Client authors.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MCLFIRewriter class. This is an abstract
// class that encapsulates the rewriting logic for MCInsts.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCLFIREWRITER_H
#define LLVM_MC_MCLFIREWRITER_H

#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class MCInst;
class MCSubtargetInfo;
class MCStreamer;
class MCSymbol;

class MCLFIRewriter {
private:
  MCContext &Ctx;

protected:
  bool Enabled = true;
  std::unique_ptr<MCInstrInfo> InstInfo;
  std::unique_ptr<MCRegisterInfo> RegInfo;

public:
  MCLFIRewriter(MCContext &Ctx, std::unique_ptr<MCRegisterInfo> &&RI,
                std::unique_ptr<MCInstrInfo> &&II)
      : Ctx(Ctx), InstInfo(std::move(II)), RegInfo(std::move(RI)) {}

  LLVM_ABI void error(const MCInst &Inst, const char Msg[]);

  LLVM_ABI void disable();
  LLVM_ABI void enable();

  LLVM_ABI bool isCall(const MCInst &Inst) const;
  LLVM_ABI bool isBranch(const MCInst &Inst) const;
  LLVM_ABI bool isIndirectBranch(const MCInst &Inst) const;
  LLVM_ABI bool isReturn(const MCInst &Inst) const;

  LLVM_ABI bool mayLoad(const MCInst &Inst) const;
  LLVM_ABI bool mayStore(const MCInst &Inst) const;

  LLVM_ABI bool mayModifyRegister(const MCInst &Inst, MCRegister Reg) const;

  virtual ~MCLFIRewriter() = default;
  virtual bool rewriteInst(const MCInst &Inst, MCStreamer &Out,
                           const MCSubtargetInfo &STI) = 0;

  // Called when a label is emitted. Used for optimizations that require
  // information about jump targets, such as guard elimination.
  virtual void onLabel(const MCSymbol *Symbol) {}
};

} // namespace llvm
#endif
