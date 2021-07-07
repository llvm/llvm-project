//===-- SystemZAsmPrinter.h - SystemZ LLVM assembly printer ----*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZASMPRINTER_H
#define LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZASMPRINTER_H

#include "SystemZMCInstLower.h"
#include "SystemZTargetMachine.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/StackMaps.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class MCStreamer;
class MachineBasicBlock;
class MachineInstr;
class Module;
class raw_ostream;

class LLVM_LIBRARY_VISIBILITY SystemZAsmPrinter : public AsmPrinter {
private:
  StackMaps SM;

  typedef std::pair<MCInst, const MCSubtargetInfo *> MCInstSTIPair;
  struct CmpMCInst {
    bool operator()(const MCInstSTIPair &MCI_STI_A,
                    const MCInstSTIPair &MCI_STI_B) const {
      if (MCI_STI_A.second != MCI_STI_B.second)
        return uintptr_t(MCI_STI_A.second) < uintptr_t(MCI_STI_B.second);
      const MCInst &A = MCI_STI_A.first;
      const MCInst &B = MCI_STI_B.first;
      assert(A.getNumOperands() == B.getNumOperands() &&
             A.getNumOperands() == 5 && A.getOperand(2).getImm() == 1 &&
             B.getOperand(2).getImm() == 1 && "Unexpected EXRL target MCInst");
      if (A.getOpcode() != B.getOpcode())
        return A.getOpcode() < B.getOpcode();
      if (A.getOperand(0).getReg() != B.getOperand(0).getReg())
        return A.getOperand(0).getReg() < B.getOperand(0).getReg();
      if (A.getOperand(1).getImm() != B.getOperand(1).getImm())
        return A.getOperand(1).getImm() < B.getOperand(1).getImm();
      if (A.getOperand(3).getReg() != B.getOperand(3).getReg())
        return A.getOperand(3).getReg() < B.getOperand(3).getReg();
      if (A.getOperand(4).getImm() != B.getOperand(4).getImm())
        return A.getOperand(4).getImm() < B.getOperand(4).getImm();
      return false;
    }
  };
  typedef std::map<MCInstSTIPair, MCSymbol *, CmpMCInst> EXRLT2SymMap;
  EXRLT2SymMap EXRLTargets2Sym;

public:
  SystemZAsmPrinter(TargetMachine &TM, std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer)), SM(*this) {}

  // Override AsmPrinter.
  StringRef getPassName() const override { return "SystemZ Assembly Printer"; }
  void emitInstruction(const MachineInstr *MI) override;
  void emitMachineConstantPoolValue(MachineConstantPoolValue *MCPV) override;
  void emitEndOfAsmFile(Module &M) override;
  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       const char *ExtraCode, raw_ostream &OS) override;
  bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                             const char *ExtraCode, raw_ostream &OS) override;

  bool doInitialization(Module &M) override {
    SM.reset();
    return AsmPrinter::doInitialization(M);
  }

private:
  void LowerFENTRY_CALL(const MachineInstr &MI, SystemZMCInstLower &MCIL);
  void LowerSTACKMAP(const MachineInstr &MI);
  void LowerPATCHPOINT(const MachineInstr &MI, SystemZMCInstLower &Lower);
  void emitEXRLTargetInstructions();
};
} // end namespace llvm

#endif
