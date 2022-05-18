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
#include "SystemZTargetStreamer.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/StackMaps.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class MCStreamer;
class MachineInstr;
class Module;
class raw_ostream;

class LLVM_LIBRARY_VISIBILITY SystemZAsmPrinter : public AsmPrinter {
private:
  StackMaps SM;
  MCSymbol *CurrentFnPPA1Sym;     // PPA1 Symbol.
  MCSymbol *CurrentFnEPMarkerSym; // Entry Point Marker.

  SystemZTargetStreamer *getTargetStreamer() {
    MCTargetStreamer *TS = OutStreamer->getTargetStreamer();
    assert(TS && "do not have a target streamer");
    return static_cast<SystemZTargetStreamer *>(TS);
  }

  /// Call type information for XPLINK.
  enum class CallType {
    BASR76 = 0,   // b'x000' == BASR  r7,r6
    BRAS7 = 1,    // b'x001' == BRAS  r7,ep
    RESVD_2 = 2,  // b'x010'
    BRASL7 = 3,   // b'x011' == BRASL r7,ep
    RESVD_4 = 4,  // b'x100'
    RESVD_5 = 5,  // b'x101'
    BALR1415 = 6, // b'x110' == BALR  r14,r15
    BASR33 = 7,   // b'x111' == BASR  r3,r3
  };

  void emitPPA1(MCSymbol *FnEndSym);

public:
  SystemZAsmPrinter(TargetMachine &TM, std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer)), SM(*this),
        CurrentFnPPA1Sym(nullptr), CurrentFnEPMarkerSym(nullptr) {}

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
  void emitFunctionEntryLabel() override;
  void emitFunctionBodyEnd() override;

private:
  void emitCallInformation(CallType CT);
  void LowerFENTRY_CALL(const MachineInstr &MI, SystemZMCInstLower &MCIL);
  void LowerSTACKMAP(const MachineInstr &MI);
  void LowerPATCHPOINT(const MachineInstr &MI, SystemZMCInstLower &Lower);
};
} // end namespace llvm

#endif
