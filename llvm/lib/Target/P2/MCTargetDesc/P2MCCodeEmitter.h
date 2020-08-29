//===-- P2MCCodeEmitter.h - Convert P2 Code to Machine Code -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the P2MCCodeEmitter class.
//
//===----------------------------------------------------------------------===//
//

#ifndef LLVM_LIB_TARGET_P2_MCTARGETDESC_P2MCCODEEMITTER_H
#define LLVM_LIB_TARGET_P2_MCTARGETDESC_P2MCCODEEMITTER_H

#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/Support/DataTypes.h"

using namespace llvm;

namespace llvm {
    class MCContext;
    class MCExpr;
    class MCInst;
    class MCInstrInfo;
    class MCFixup;
    class MCOperand;
    class MCSubtargetInfo;
    class raw_ostream;

    class P2MCCodeEmitter : public MCCodeEmitter {
        P2MCCodeEmitter(const P2MCCodeEmitter &) = delete;
        void operator=(const P2MCCodeEmitter &) = delete;
        const MCInstrInfo &MCII;
        MCContext &Ctx;

    public:
        P2MCCodeEmitter(const MCInstrInfo &mcii, MCContext &Ctx_)
                : MCII(mcii), Ctx(Ctx_) {}

        ~P2MCCodeEmitter() override {}

        void emitByte(unsigned char C, raw_ostream &OS) const;

        void emitInstruction(uint64_t Val, unsigned Size, raw_ostream &OS) const;

        void encodeInstruction(const MCInst &MI, raw_ostream &OS, SmallVectorImpl<MCFixup> &Fixups, const MCSubtargetInfo &STI) const override;

        // TableGen'erated function for getting the
        // binary encoding for an instruction.
        uint64_t getBinaryCodeForInstr(const MCInst &MI, SmallVectorImpl<MCFixup> &Fixups, const MCSubtargetInfo &STI) const;

        // Return binary encoding of the jump target
        // target operand, such as jmp #function_addr.
        // If the machine operand requires relocation,
        // record the relocation and return zero.
        unsigned getJumpTargetOpValue(const MCInst &MI, unsigned OpNo, SmallVectorImpl<MCFixup> &Fixups, const MCSubtargetInfo &STI) const;

        // Return binary encoding of the call
        // target operand, such as call #function_addr.
        // If the machine operand requires relocation,
        // record the relocation and return zero.
        unsigned encodeCallTarget(const MCInst &MI, unsigned OpNo, SmallVectorImpl<MCFixup> &Fixups, const MCSubtargetInfo &STI) const;

        // Return binary encoding of the condition operand
        // unsigned encodeCondition(const MCInst &MI, unsigned OpNo, SmallVectorImpl<MCFixup> &Fixups, const MCSubtargetInfo &STI) const;
        // Return binary encoding of the effect operand (C/Z flags)
        // unsigned encodeEffect(const MCInst &MI, unsigned OpNo, SmallVectorImpl<MCFixup> &Fixups, const MCSubtargetInfo &STI) const;

        // getMachineOpValue - Return binary encoding of operand. If the machin
        // operand requires relocation, record the relocation and return zero.
        unsigned getMachineOpValue(const MCInst &MI, const MCOperand &MO, SmallVectorImpl<MCFixup> &Fixups, const MCSubtargetInfo &STI) const;

        unsigned getMemEncoding(const MCInst &MI, unsigned OpNo, SmallVectorImpl<MCFixup> &Fixups, const MCSubtargetInfo &STI) const;

        unsigned getExprOpValue(const MCInst &MI, const MCExpr *Expr, SmallVectorImpl<MCFixup> &Fixups, const MCSubtargetInfo &STI) const;
    };
}

#endif