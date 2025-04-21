/* --- PEAsmPrinter.h --- */

/* ------------------------------------------
Author: undefined
Date: 4/9/2025
------------------------------------------ */

#ifndef PEASMPRINTER_H
#define PEASMPRINTER_H

#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/MC/MCInst.h"
#include "PETargetMachine.h"
#include "TargetInfo/PETargetInfo.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm{

class PESubtarget;
class PEAsmPrinter : public AsmPrinter{
    const PESubtarget *Subtarget;
public:
    explicit PEAsmPrinter(TargetMachine &TM,
        std::unique_ptr<MCStreamer> Streamer)
: AsmPrinter(TM, std::move(Streamer)) {}

    StringRef getPassName() const override { return "PE Assembly Printer"; }
    
    virtual bool runOnMachineFunction(MachineFunction &MF) override;

    void emitInstruction(const MachineInstr *MI) override;

private: 
    bool lowerPseudoInstExpansion(const MachineInstr *MI, MCInst &Inst);
    bool lowerToMCInst(const MachineInstr *MI, MCInst &OutMI);

private:

};
}

#endif // PEASMPRINTER_H
