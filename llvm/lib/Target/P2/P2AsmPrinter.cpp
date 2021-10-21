//===-- P2AsmPrinter.cpp - P2 LLVM Assembly Printer -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to GAS-format P2 assembly language.
//
//===----------------------------------------------------------------------===//

#include "P2AsmPrinter.h"

#include "MCTargetDesc/P2InstPrinter.h"
#include "P2.h"
#include "P2InstrInfo.h"
#include "TargetInfo/P2TargetInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

#define DEBUG_TYPE "p2-asm-printer"

bool P2AsmPrinter::runOnMachineFunction(MachineFunction &MF) {
    LLVM_DEBUG(errs() << "asm printer run on machine function\n");
    P2FI = MF.getInfo<P2FunctionInfo>();
    AsmPrinter::runOnMachineFunction(MF);
    return true;
}

void P2AsmPrinter::printOperand(const MachineInstr *MI, unsigned OpNo, raw_ostream &O) {
    const MachineOperand &MO = MI->getOperand(OpNo);

    switch (MO.getType()) {
        case MachineOperand::MO_Register:
            O << "$" << P2InstPrinter::getRegisterName(MO.getReg());
            break;
        case MachineOperand::MO_Immediate:
            O << "#" << MO.getImm();
            break;
        case MachineOperand::MO_GlobalAddress:
            O << "@" << getSymbol(MO.getGlobal());
            break;
        case MachineOperand::MO_ExternalSymbol:
            O << "$" << *GetExternalSymbolSymbol(MO.getSymbolName());
            break;
        case MachineOperand::MO_MachineBasicBlock:
            O << "%" << *MO.getMBB()->getSymbol();
            break;
        default:
            llvm_unreachable("asm printer operand not implemented yet!");
    }
}

bool P2AsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNum,
                                    const char *ExtraCode, raw_ostream &O) {
    // Default asm printer can only deal with some extra codes,
    // so try it first.
    bool Error = AsmPrinter::PrintAsmOperand(MI, OpNum, ExtraCode, O);

    const MachineOperand &MO = MI->getOperand(OpNum);
    if (Error && ExtraCode && ExtraCode[0]) {
        if (ExtraCode[1] != 0)
            return true; // Unknown modifier.

        switch(ExtraCode[0]) {
            case '#':
                // this is an immediate
            if ((MO.getType()) != MachineOperand::MO_Immediate)
                return true;
            O << "#" << MO.getImm();
                return false;
        }
    }

    if (Error)
        printOperand(MI, OpNum, O);

    return false;
}

bool P2AsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                          unsigned OpNum, const char *ExtraCode,
                                          raw_ostream &O) {
    // Currently we are expecting no ExtraCode
    if (ExtraCode) {
        return true; // Unknown modifier.
    }

    const MachineOperand &MO = MI->getOperand(OpNum);
    assert(MO.isReg() && "unexpected inline asm memory operand");
    O << P2InstPrinter::getRegisterName(MO.getReg());

    return false;
}

void P2AsmPrinter::emitInstruction(const MachineInstr *MI) {
    MCInst I;
    MCInstLowering.lowerInstruction(*MI, I);
    EmitToStreamer(*OutStreamer, I);
}

void P2AsmPrinter::emitFunctionEntryLabel() {
    OutStreamer->emitLabel(CurrentFnSym);
}
void P2AsmPrinter::emitFunctionBodyStart() {
    MCInstLowering.Initialize(&MF->getContext());
}
void P2AsmPrinter::emitFunctionBodyEnd() {

}
void P2AsmPrinter::emitStartOfAsmFile(Module &M) {
    //OutStreamer->emitRawText("\t\t\torg 0x200"); // all this code should start at the hub
}

// Force static initialization.
extern "C" void LLVMInitializeP2AsmPrinter() {
    llvm::RegisterAsmPrinter<llvm::P2AsmPrinter> X(getTheP2Target());
}