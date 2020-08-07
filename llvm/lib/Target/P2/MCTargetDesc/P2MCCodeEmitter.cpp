//===-- P2MCCodeEmitter.cpp - Convert P2 Code to Machine Code ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the P2MCCodeEmitter class.
//
//===----------------------------------------------------------------------===//
//

#include "P2MCCodeEmitter.h"

#include "MCTargetDesc/P2BaseInfo.h"
#include "MCTargetDesc/P2FixupKinds.h"
//#include "MCTargetDesc/P2MCExpr.h"
#include "MCTargetDesc/P2MCTargetDesc.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "mccodeemitter"

#define GET_INSTRMAP_INFO
#include "P2GenInstrInfo.inc"
#undef GET_INSTRMAP_INFO

MCCodeEmitter *llvm::createP2MCCodeEmitter(const MCInstrInfo &MCII, const MCRegisterInfo &MRI, MCContext &Ctx) {
    return new P2MCCodeEmitter(MCII, Ctx);
}

void P2MCCodeEmitter::emitByte(unsigned char C, raw_ostream &OS) const {
    OS << (char)C;
}

void P2MCCodeEmitter::emitInstruction(uint64_t Val, unsigned Size, raw_ostream &OS) const {
    // Output the instruction encoding in little endian byte order.
    for (unsigned i = 0; i < Size; ++i) {
        emitByte((Val >> i*8) & 0xff, OS);
    }
}

void P2MCCodeEmitter::encodeInstruction(const MCInst &MI, raw_ostream &OS, SmallVectorImpl<MCFixup> &Fixups, const MCSubtargetInfo &STI) const {
    LLVM_DEBUG(errs() << "==== begin encode ====\n");
    LLVM_DEBUG(MI.dump());

    uint32_t bin = getBinaryCodeForInstr(MI, Fixups, STI);

    // Check for unimplemented opcodes.
    //unsigned op_code = MI.getOpcode();
    //const MCInstrDesc &Desc = MCII.get(op_code);

    LLVM_DEBUG(errs() << "emitting instruction binary: ");
    for (int i = 0; i < 32; i++) {
        LLVM_DEBUG(errs() << ((bin >> (31-i))&1));
    }

    LLVM_DEBUG(errs() << "\n");

    // Pseudo instructions don't get encoded and shouldn't be here
    // in the first place!
    // if ((TSFlags & P2::FormMask) == P2::Pseudo)
    //     llvm_unreachable("Pseudo opcode found in encodeInstruction()");

    emitInstruction(bin, 4, OS);

    LLVM_DEBUG(errs() << "==== end encode ====\n");
}

unsigned P2MCCodeEmitter::getJumpTargetOpValue(const MCInst &MI, unsigned OpNo, SmallVectorImpl<MCFixup> &Fixups, const MCSubtargetInfo &STI) const {
    const MCOperand &MO = MI.getOperand(OpNo);
    // If the destination is an immediate, we have nothing to do.
    if (MO.isImm()) return MO.getImm();

    assert(MO.isExpr() && "getJumpTargetOpValue expects only expressions");

    LLVM_DEBUG(errs() << "--- creating fixup for jump operand\n");

    const MCExpr *Expr = MO.getExpr();
    Fixups.push_back(MCFixup::create(0, Expr, MCFixupKind(P2::fixup_P2_PC20)));
    return 0;
}

unsigned P2MCCodeEmitter::encodeCallTarget(const MCInst &MI, unsigned OpNo, SmallVectorImpl<MCFixup> &Fixups,
                                            const MCSubtargetInfo &STI) const {
    const MCOperand &MO = MI.getOperand(OpNo);

    LLVM_DEBUG(errs() << "--- encode call target for operand: ");
    LLVM_DEBUG(MO.dump());

    if (MO.isExpr()) {
        LLVM_DEBUG(errs() << "call target for operand is an expression of kind: ");
        LLVM_DEBUG(errs() << (unsigned)MO.getExpr()->getKind() << "\n");
        MCFixupKind FixupKind;
        const MCSymbolRefExpr* expr = static_cast<const MCSymbolRefExpr*>(MO.getExpr());

        LLVM_DEBUG(expr->dump());

        if (expr->getSymbol().isExternal()) {
            LLVM_DEBUG(errs() << "creating COG fixup\n");
            FixupKind = static_cast<MCFixupKind>(P2::fixup_P2_COG9);
        } else {
            LLVM_DEBUG(errs() << "creating normal fixup\n");
            FixupKind = static_cast<MCFixupKind>(P2::fixup_P2_20);
        }

        Fixups.push_back(MCFixup::create(0, MO.getExpr(), FixupKind, MI.getLoc()));
        return 0;
    }

    assert(MO.isImm() && "non-immediate expression not handled by encodeCallTarget");

    auto Target = MO.getImm();
    return Target;
}

unsigned P2MCCodeEmitter::getExprOpValue(const MCInst &MI, const MCExpr *Expr, SmallVectorImpl<MCFixup> &Fixups,
                                            const MCSubtargetInfo &STI) const {

    MCExpr::ExprKind Kind = Expr->getKind();

    LLVM_DEBUG(errs() << " --- kind = " << (unsigned)Kind << "\n");

    if (Kind == MCExpr::Constant) {
        LLVM_DEBUG(errs() << " --- expression is a constant\n");
        return cast<MCConstantExpr>(Expr)->getValue();
    }

    if (Kind == MCExpr::Binary) {
        LLVM_DEBUG(errs() << " --- expression is binary\n");
        unsigned Res = getExprOpValue(MI, cast<MCBinaryExpr>(Expr)->getLHS(), Fixups, STI);
        Res += getExprOpValue(MI, cast<MCBinaryExpr>(Expr)->getRHS(), Fixups, STI);
        return Res;
    }

    if (Kind == MCExpr::SymbolRef) {
        LLVM_DEBUG(errs() << " --- expression is symbol ref\n");
        LLVM_DEBUG(MI.dump());
        LLVM_DEBUG(Expr->dump());
        MCFixupKind FixupKind = static_cast<MCFixupKind>(P2::fixup_P2_AUG20);
        Fixups.push_back(MCFixup::create(0, Expr, FixupKind));
        return 0;
    }

    if (Kind == MCExpr::Target) {
        llvm_unreachable("no implementation for target expressions!");
        return 0;
    }

    llvm_unreachable("unhandled expression operand!");

    return 0;
}

/// getMachineOpValue - Return binary encoding of operand. If the machine
/// operand requires relocation, record the relocation and return zero.
unsigned P2MCCodeEmitter::getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                                            SmallVectorImpl<MCFixup> &Fixups,
                                            const MCSubtargetInfo &STI) const {
    if (MO.isReg()) {
        unsigned Reg = MO.getReg();
        unsigned RegNo = Ctx.getRegisterInfo()->getEncodingValue(Reg);
        LLVM_DEBUG(errs() << "-- register number is " << RegNo << " for reg " << Reg << "\n");
        return RegNo;
    } else if (MO.isImm()) {
        LLVM_DEBUG(errs() << "-- immediate operand is " << MO.getImm() << "\n");
        return static_cast<unsigned>(MO.getImm());
    }

    LLVM_DEBUG(errs() << " -- operand is an expression\n");

    // MO must be an Expr.
    assert(MO.isExpr());
    return getExprOpValue(MI, MO.getExpr(), Fixups, STI);
}

/// getMemEncoding - Return binary encoding of memory related operand.
/// If the offset operand requires relocation, record the relocation.
unsigned P2MCCodeEmitter::getMemEncoding(const MCInst &MI, unsigned OpNo, SmallVectorImpl<MCFixup> &Fixups,
                                            const MCSubtargetInfo &STI) const {

    llvm_unreachable("getMemEncoding not implemented");
    // TODO
    assert(MI.getOperand(OpNo).isReg());
    unsigned RegBits = getMachineOpValue(MI, MI.getOperand(OpNo), Fixups, STI) << 16;
    unsigned OffBits = getMachineOpValue(MI, MI.getOperand(OpNo+1), Fixups, STI);

    return (OffBits & 0xFFFF) | RegBits;
}

#include "P2GenMCCodeEmitter.inc"