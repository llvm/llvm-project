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
#include "P2BaseInfo.h"

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

MCOperand P2MCInstLower::lowerSymbolOperand(const MachineOperand &MO, MachineOperandType MOTy) const {
    const MCSymbol *Symbol;

    int Offset = 0;

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

    MCOperand mco = MCOperand::createExpr(Expr);
    LLVM_DEBUG(mco.dump());
    LLVM_DEBUG(errs() << "... is external: " << Symbol->isExternal() << "\n");

    return mco;
}

MCOperand P2MCInstLower::lowerOperand(const MachineOperand& MO) const {
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
            return lowerSymbolOperand(MO, MOTy);

        case MachineOperand::MO_Register:
            // Ignore all implicit register operands.
            if (MO.isImplicit()) break;
            return MCOperand::createReg(MO.getReg());

        case MachineOperand::MO_Immediate:
            return MCOperand::createImm(MO.getImm());

        case MachineOperand::MO_RegisterMask:
            break;
    }

    return MCOperand();
}

void P2MCInstLower::createAugInst(MCInst &aug, int type, int value, int condition) const {
    assert (type == 1 || type == 2 && "Unknown aug type");

    if (type == 1) {
        aug.setOpcode(P2::AUGS);
    } else {
        aug.setOpcode(P2::AUGD);
    }

    aug.addOperand(MCOperand::createImm(value));
    aug.addOperand(MCOperand::createImm(condition));
    aug.addOperand(MCOperand::createImm(P2::NOEFF));
}

void P2MCInstLower::createAugInst(const MachineInstr &MI, MCInst &aug, int op_num) const {
    // 1. Figure out if we need augd or augs
    assert(canAug(MI) && "Can't create aug for instruction!\n");
    int aug_type = 0; // 0 = none, 1 = augs, 2 = augd

    bool has_d = P2::hasDField(MI);
    bool has_s = P2::hasSField(MI);
    int s_num = P2::getSNum(MI);
    int d_num = P2::getDNum(MI);

    LLVM_DEBUG(errs() << "has_d = " << has_d << " has_s = " << has_s << " s_num = " << s_num << " d_num = " << d_num << "\n");

    if (has_d && op_num == d_num) {
        aug_type = 2;
    } else if (has_s && op_num == s_num) {
        aug_type = 1;
    }

    LLVM_DEBUG(errs() << "aug_type = " << aug_type << "\n");
    
    // 2. create the MCInst
    const MachineOperand &MO = MI.getOperand(op_num);
    int aug_i = 0;

    if (MO.isImm()) { // apply the actual immediate for immediate operands, for others (global address, etc), just insert 0 for later fixup
        aug_i = (MO.getImm() >> 9) & 0x7fffff;
    }

    createAugInst(aug, aug_type, aug_i, P2::getCondition(MI));
}

bool P2MCInstLower::canAug(const MachineInstr &MI) const {
    int type = P2::getInstructionForm(MI);

    if (type == P2::P2InstN || 
        type == P2::P2InstWRA ||
        type == P2::P2InstWRA ||
        type == P2::P2InstRA ||
        type == P2::P2InstD || 
        type == P2::P2InstCZ ||
        type == P2::P2InstCZD) return false;

    return true;
}

void P2MCInstLower::lowerInstruction(const MachineInstr &MI, MCInst &AugMI, MCInst &OutMI) const {
    LLVM_DEBUG(errs() << "Lower instruction from MachineInstr to MCInst\n");
    LLVM_DEBUG(MI.dump());
    LLVM_DEBUG(errs() << "\tOpcode: " << MI.getOpcode() << ", flags: " << MI.getDesc().TSFlags << "\n");

    OutMI.setOpcode(MI.getOpcode());
    
    auto flags = MI.getDesc().TSFlags;
    OutMI.setFlags(flags);

    // mark that this MCInst will exist in a cogex function
    if (MI.getMF()->getFunction().hasFnAttribute(Attribute::Cogmain) | MI.getMF()->getFunction().hasFnAttribute(Attribute::Cogtext)) flags |= (1<<13);

    for (unsigned i = 0, e = MI.getNumOperands(); i != e; ++i) {
        const MachineOperand &MO = MI.getOperand(i);
        MCOperand MCOp = lowerOperand(MO);

        if ((MO.isImm() || MO.isGlobal() || MO.isJTI()) && canAug(MI)) {
            if (MO.isImm()) {
                // basic immediates
                int imm = MCOp.getImm();

                if (imm >> 9) {
                    LLVM_DEBUG(errs() << "immediate " << imm << " requires an aug\n");
                    // we need an aug instruction
                    createAugInst(MI, AugMI, i);
                    MCOp.setImm(imm & 0x1ff);
                }
            } else {
                // global addresses that require an AUGS/D to be in inserted to be fixedup later
                createAugInst(MI, AugMI, i);
            }
            
        }

        if (MCOp.isValid())
            OutMI.addOperand(MCOp);
    }
}