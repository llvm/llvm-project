//===-- BPFFrameLowering.h - Define frame lowering for BPF -----*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_BPF_BPFASMPRINTER_H
#define LLVM_LIB_TARGET_BPF_BPFASMPRINTER_H

#include "BPFTargetMachine.h"
#include "BTFDebug.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class BPFAsmPrinter : public AsmPrinter {
public:
  explicit BPFAsmPrinter(TargetMachine &TM,
                         std::unique_ptr<MCStreamer> Streamer);
  ~BPFAsmPrinter() override;

  StringRef getPassName() const override { return "BPF Assembly Printer"; }
  bool doInitialization(Module &M) override;
  bool doFinalization(Module &M) override;
  void printOperand(const MachineInstr *MI, int OpNum, raw_ostream &O);
  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       const char *ExtraCode, raw_ostream &O) override;
  bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNum,
                             const char *ExtraCode, raw_ostream &O) override;

  void emitInstruction(const MachineInstr *MI) override;
  void emitFunctionBodyEnd() override;
  MCSymbol *getJTPublicSymbol(unsigned JTI);
  void emitJumpTableInfo() override;

  static char ID;

private:
  BTFDebug *BTF;
  TargetMachine &TM;
  bool SawTrapCall = false;

  const BPFTargetMachine &getBTM() const;
};

class BPFAsmPrinterBeginPass
    : public RequiredPassInfoMixin<BPFAsmPrinterBeginPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

class BPFAsmPrinterPass : public RequiredPassInfoMixin<BPFAsmPrinterPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class BPFAsmPrinterEndPass
    : public RequiredPassInfoMixin<BPFAsmPrinterEndPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

} // namespace llvm

#endif /* LLVM_LIB_TARGET_BPF_BPFASMPRINTER_H */
