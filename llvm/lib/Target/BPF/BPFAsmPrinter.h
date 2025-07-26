//===-- BPFFrameLowering.h - Define frame lowering for BPF -----*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_BPF_BPFASMPRINTER_H
#define LLVM_LIB_TARGET_BPF_BPFASMPRINTER_H

#include "BTFDebug.h"
#include "llvm/CodeGen/AsmPrinter.h"

namespace llvm {

class BPFAsmPrinter : public AsmPrinter {
public:
  explicit BPFAsmPrinter(TargetMachine &TM,
                         std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer), ID), BTF(nullptr) {}

  StringRef getPassName() const override { return "BPF Assembly Printer"; }
  bool doInitialization(Module &M) override;
  void printOperand(const MachineInstr *MI, int OpNum, raw_ostream &O);
  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       const char *ExtraCode, raw_ostream &O) override;
  bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNum,
                             const char *ExtraCode, raw_ostream &O) override;

  void emitInstruction(const MachineInstr *MI) override;
  MCSymbol *getJTPublicSymbol(unsigned JTI);
  MCSymbol *lowerGlobalValue(const GlobalValue *GVal);
  virtual void emitJumpTableInfo() override;

  static char ID;

private:
  BTFDebug *BTF;
};

} // namespace llvm

#endif /* LLVM_LIB_TARGET_BPF_BPFASMPRINTER_H */
