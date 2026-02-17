#ifndef LLVM_LIB_TARGET_SC32_SC32ASMPRINTER_H
#define LLVM_LIB_TARGET_SC32_SC32ASMPRINTER_H

#include "llvm/CodeGen/AsmPrinter.h"

namespace llvm {

class SC32AsmPrinter : public AsmPrinter {
public:
  SC32AsmPrinter(TargetMachine &TM, std::unique_ptr<MCStreamer> Streamer);

  void emitInstruction(const MachineInstr *MI) override;

  void emitLinkage(const GlobalValue *GV, MCSymbol *GVSym) const override;
};

} // namespace llvm

#endif
