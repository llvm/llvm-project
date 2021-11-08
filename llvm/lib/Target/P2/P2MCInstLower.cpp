//===-- P2MCInstLower.cpp - Convert P2 MachineInstr to MCInst ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains code to lower P2 MachineInstrs to their corresponding
// MCInst records.
//
//===----------------------------------------------------------------------===//

#include "P2MCInstLower.h"

#include "P2InstrInfo.h"
#include "P2AsmPrinter.h"

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "p2-mc-ilower"

P2MCInstLower::P2MCInstLower(P2AsmPrinter &asmprinter) : AsmPrinter(asmprinter) {}

void P2MCInstLower::Initialize(MCContext* C) {
    Ctx = C;
}

MCOperand P2MCInstLower::LowerSymbolOperand(const MachineOperand &MO,
                                              MachineOperandType MOTy,
                                              unsigned Offset) const {
    const MCSymbol *Symbol;

    switch (MOTy) {
        case MachineOperand::MO_GlobalAddress:
            Symbol = AsmPrinter.getSymbol(MO.getGlobal());
            Symbol->setExternal(false);
            Offset += MO.getOffset();
        break;

        case MachineOperand::MO_MachineBasicBlock:
            Symbol = MO.getMBB()->getSymbol();
        break;

        case MachineOperand::MO_BlockAddress:
            Symbol = AsmPrinter.GetBlockAddressSymbol(MO.getBlockAddress());
            Offset += MO.getOffset();
        break;

        case MachineOperand::MO_JumpTableIndex:
            Symbol = AsmPrinter.GetJTISymbol(MO.getIndex());
        break;

        case MachineOperand::MO_ExternalSymbol:
            // for now, assume all external symbols are libcalls. Any actual functions get treated as global addresses.
            // I don't know if this will break something down the line. So when getting the call target,
            // check if the symbol is external and if it is, encode the call target with a different fixup
            LLVM_DEBUG(errs() << "external symbol: " << MO.getSymbolName() << "\n");
            Symbol = AsmPrinter.GetExternalSymbolSymbol(MO.getSymbolName());
            Symbol->setExternal(true);
        break;

        default:
            llvm_unreachable("<unknown operand type>");
    }

    const MCExpr *Expr = MCSymbolRefExpr::create(Symbol, *Ctx);

    if (Offset) {
        // Assume offset is never negative.
        assert(Offset > 0);
        Expr = MCBinaryExpr::createAdd(Expr, MCConstantExpr::create(Offset, *Ctx), *Ctx);
    }

    // if (TargetKind != Cpu0MCExpr::CEK_None)
    //     Expr = Cpu0MCExpr::create(TargetKind, Expr, *Ctx);

    MCOperand mco = MCOperand::createExpr(Expr);
    LLVM_DEBUG(mco.dump());
    LLVM_DEBUG(errs() << "... is external: " << Symbol->isExternal() << "\n");

    return mco;
}

MCOperand P2MCInstLower::lowerOperand(const MachineOperand& MO, unsigned offset) const {
    MachineOperandType MOTy = MO.getType();

    switch (MOTy) {

        default:
            LLVM_DEBUG(errs() << "Operand type: " << (int)MOTy << "\n");
            llvm_unreachable("MCInstrLower: unknown operand type");
        case MachineOperand::MO_GlobalAddress:
        case MachineOperand::MO_JumpTableIndex:
        case MachineOperand::MO_BlockAddress:
        case MachineOperand::MO_MachineBasicBlock:
        case MachineOperand::MO_ExternalSymbol:
            LLVM_DEBUG(errs() << "mcinst lower, MO: "; MO.dump());
            LLVM_DEBUG(errs() << "MO type: " << (int)MOTy << "\n");
            return LowerSymbolOperand(MO, MOTy, offset);

        case MachineOperand::MO_Register:
            // Ignore all implicit register operands.
            if (MO.isImplicit()) break;
            return MCOperand::createReg(MO.getReg());

        case MachineOperand::MO_Immediate:
            return MCOperand::createImm(MO.getImm() + offset);

        case MachineOperand::MO_RegisterMask:
            break;
    }

    return MCOperand();
}

void P2MCInstLower::lowerInstruction(const MachineInstr &MI, MCInst &OutMI) const {
    LLVM_DEBUG(errs() << "Lower instruction from MachineInstr to MCInst\n");
    LLVM_DEBUG(errs() << "\tOpcode: " << MI.getOpcode() << ", flags: " << MI.getDesc().TSFlags << "\n");
    OutMI.setOpcode(MI.getOpcode());
    
    auto flags = MI.getDesc().TSFlags;

    // mark that this MCInst will exist in a cogex function
    if (MI.getMF()->getFunction().hasFnAttribute(Attribute::Cogmain) | MI.getMF()->getFunction().hasFnAttribute(Attribute::Cogtext)) flags |= 1;

    OutMI.setFlags(flags);

    for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
        const MachineOperand &MO = MI.getOperand(i);
        MCOperand MCOp = lowerOperand(MO);

        if (MCOp.isValid())
            OutMI.addOperand(MCOp);
    }
}