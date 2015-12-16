//===-- HexagonAsmPrinter.h - Print machine code to an Hexagon .s file ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Hexagon Assembly printer class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_HEXAGON_HEXAGONASMPRINTER_H
#define LLVM_LIB_TARGET_HEXAGON_HEXAGONASMPRINTER_H

#include "Hexagon.h"
#include "HexagonTargetMachine.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
  class HexagonAsmPrinter : public AsmPrinter {
    const HexagonSubtarget *Subtarget;

  public:
    explicit HexagonAsmPrinter(TargetMachine &TM,
                               std::unique_ptr<MCStreamer> Streamer);

    bool runOnMachineFunction(MachineFunction &Fn) override {
      Subtarget = &Fn.getSubtarget<HexagonSubtarget>();
      return AsmPrinter::runOnMachineFunction(Fn);
    }

    const char *getPassName() const override {
      return "Hexagon Assembly Printer";
    }

    bool isBlockOnlyReachableByFallthrough(
                                   const MachineBasicBlock *MBB) const override;

    void EmitInstruction(const MachineInstr *MI) override;

    void HexagonProcessInstruction(MCInst &Inst,
                                   const MachineInstr &MBB);


    void printOperand(const MachineInstr *MI, unsigned OpNo, raw_ostream &O);
    bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                         unsigned AsmVariant, const char *ExtraCode,
                         raw_ostream &OS) override;
    bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNo,
                               unsigned AsmVariant, const char *ExtraCode,
                               raw_ostream &OS) override;

    static const char *getRegisterName(unsigned RegNo);
  };

} // end of llvm namespace

#endif
