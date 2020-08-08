//===- P2Disassembler.cpp - Disassembler for P2 ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is part of the P2 Disassembler.
//
//===----------------------------------------------------------------------===//

#include "P2.h"
#include "P2RegisterInfo.h"
#include "P2Subtarget.h"
#include "MCTargetDesc/P2MCTargetDesc.h"

#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCFixedLenDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "p2-disassembler"

typedef MCDisassembler::DecodeStatus DecodeStatus;

namespace {

/// A disassembler class for P2.
class P2Disassembler : public MCDisassembler {
    public:
    	P2Disassembler(const MCSubtargetInfo &STI, MCContext &Ctx) : MCDisassembler(STI, Ctx) {}
    	virtual ~P2Disassembler() {}

    	DecodeStatus getInstruction(MCInst &Instr, uint64_t &Size,
                                    ArrayRef<uint8_t> Bytes, uint64_t Address,
                                    raw_ostream &CStream) const override;
    };
}

static MCDisassembler *createP2Disassembler(const Target &T, const MCSubtargetInfo &STI, MCContext &Ctx) {
	return new P2Disassembler(STI, Ctx);
}

extern "C" void LLVMInitializeP2Disassembler() {
	// Register the disassembler.
	TargetRegistry::RegisterMCDisassembler(TheP2Target, createP2Disassembler);
}

static const uint16_t GPRDecoderTable[] = {
    P2::R0, P2::R1, P2::R2, P2::R3, P2::R4, P2::R5, P2::R6, P2::R7,
    P2::R8, P2::R9, P2::R10, P2::R11, P2::R12, P2::R13, P2::R14, P2::R15,
    P2::R16, P2::R17, P2::R18, P2::R19, P2::R20, P2::R21, P2::R22, P2::R23,
    P2::R24, P2::R25, P2::R26, P2::R27, P2::R28, P2::R29, P2::R30, P2::R31,
    P2::IJMP3, P2::IRET3, P2::IJMP2, P2::IRET2, P2::IJMP1, P2::IRET1,
    P2::PA, P2::PB, P2::PTRA, P2::PTRB, P2::DIRA, P2::DIRB, P2::OUTA, P2::OUTB,
    P2::INA, P2::INB
};

static uint16_t getRegForField(uint16_t r) {
    int reg_start = 0x1d0;
    if (r < reg_start || r > 0x1ff) {
        LLVM_DEBUG(errs() << "register address: " << r << "\n");
        llvm_unreachable("bad register address!");
    }

    return GPRDecoderTable[r-reg_start];
}

static DecodeStatus DecodeP2GPRRegisterClass(MCInst &Inst, unsigned RegNo, uint64_t Address, const void *Decoder);
static DecodeStatus DecodeIOInstruction(MCInst &Inst, unsigned Insn, uint64_t Address, const void *Decoder);
static DecodeStatus DecodeJumpInstruction(MCInst &Inst, unsigned Insn, uint64_t Address, const void *Decoder);
static DecodeStatus DecodeCmpInstruction(MCInst &Inst, unsigned Insn, uint64_t Address, const void *Decoder);
static DecodeStatus DecodeCordicInstruction(MCInst &Inst, unsigned Insn, uint64_t Address, const void *Decoder);

#include "P2GenDisassemblerTables.inc"

static DecodeStatus DecodeP2GPRRegisterClass(MCInst &Inst, unsigned RegNo, uint64_t Address, const void *Decoder) {
	unsigned Register = getRegForField(RegNo);
	Inst.addOperand(MCOperand::createReg(Register));
	return MCDisassembler::Success;
}

static DecodeStatus DecodeJumpInstruction(MCInst &Inst, unsigned Insn, uint64_t Address, const void *Decoder) {
    LLVM_DEBUG(errs() << "decode jump instruction\n");

    // for readability, print negatives out
    int32_t a_field = fieldFromInstruction(Insn, 0, 20);
    if (a_field > 0x7ffff) {
        a_field = -((~a_field & 0xfffff) + 1);
    }
    Inst.addOperand(MCOperand::createImm(a_field));

    return MCDisassembler::Success;
}

static DecodeStatus DecodeIOInstruction(MCInst &Inst, unsigned Insn, uint64_t Address, const void *Decoder) {

    LLVM_DEBUG(errs() << "decode IO instruction\n");

    unsigned d_field = fieldFromInstruction(Insn, 9, 9);
    unsigned is_imm = fieldFromInstruction(Insn, 18, 1);

    // first add the operand that is getting implicitly written, which is either DIRA/B or OUTA/B.
    // FIXME: we don't actually care what that is, so for now, always write to OUTA. this could eventually be a problem, but TBD how
    // (probably when optimization becomes involved.) Should actually look at the opcode and set the register to an implicit 64-bit
    // "register" that is the combination of A and B portions of OUT, DIR, and IN.
    Inst.addOperand(MCOperand::createReg(P2::OUTA));

    if (is_imm) {
        Inst.addOperand(MCOperand::createImm(d_field));
    } else {
        Inst.addOperand(MCOperand::createReg(getRegForField(d_field)));
    }

    return MCDisassembler::Success;
}

static DecodeStatus DecodeCmpInstruction(MCInst &Inst, unsigned Insn, uint64_t Address, const void *Decoder) {

    LLVM_DEBUG(errs() << "decode CMP instruction\n");

    unsigned d_field = fieldFromInstruction(Insn, 9, 9);
    unsigned s_field = fieldFromInstruction(Insn, 0, 9);
    unsigned is_imm = fieldFromInstruction(Insn, 18, 1);

    // first add the operand that is getting implicitly written, which in our case is the status flags.
    // similar to above with I/O.
    Inst.addOperand(MCOperand::createReg(P2::SW));

    Inst.addOperand(MCOperand::createReg(getRegForField(d_field)));

    if (is_imm) {
        Inst.addOperand(MCOperand::createImm(s_field));
    } else {
        Inst.addOperand(MCOperand::createReg(getRegForField(s_field)));
    }

    return MCDisassembler::Success;
}

static DecodeStatus DecodeCordicInstruction(MCInst &Inst, unsigned Insn, uint64_t Address, const void *Decoder) {
    LLVM_DEBUG(errs() << "Cordic decode\n");

    unsigned d_field = fieldFromInstruction(Insn, 9, 9);
    unsigned s_field = fieldFromInstruction(Insn, 0, 9);
    unsigned d_is_imm = fieldFromInstruction(Insn, 19, 1);
    unsigned s_is_imm = fieldFromInstruction(Insn, 18, 1);

    LLVM_DEBUG(errs() << "d = " << d_field << "; L = " << d_is_imm << "\n");
    LLVM_DEBUG(errs() << "s = " << s_field << "; I = " << s_is_imm << "\n");

    if (d_is_imm) {
        Inst.addOperand(MCOperand::createImm(d_field));
    } else {
        Inst.addOperand(MCOperand::createReg(getRegForField(d_field)));
    }

    if (s_is_imm) {
        Inst.addOperand(MCOperand::createImm(s_field));
    } else {
        Inst.addOperand(MCOperand::createReg(getRegForField(s_field)));
    }

    return MCDisassembler::Success;
}

static DecodeStatus readInstruction(ArrayRef<uint8_t> Bytes, uint64_t Address, uint64_t &Size, uint32_t &Insn) {

	if (Bytes.size() < 4) {
		Size = 0;
		return MCDisassembler::Fail;
	}

	Size = 4;
	Insn = (Bytes[0] << 0) | (Bytes[1] << 8) | (Bytes[2] << 16) | (Bytes[3] << 24); // instructions are little endian

	return MCDisassembler::Success;
}

DecodeStatus P2Disassembler::getInstruction(MCInst &Instr, uint64_t &Size, ArrayRef<uint8_t> Bytes,
                                            uint64_t Address, raw_ostream &CStream) const {

	uint32_t Insn;
	DecodeStatus Result;

	Result = readInstruction(Bytes, Address, Size, Insn);
	if (Result == MCDisassembler::Fail) return MCDisassembler::Fail;
    // set the TSFlags for the instruction printer
    // FIXME: add writing the C/Z flags to TSFlags for printing
    unsigned cond = fieldFromInstruction(Insn, 28, 4);
    Instr.setFlags((Instr.getFlags() & 0xf) | cond);

    LLVM_DEBUG(errs() << "get instruction: " << Insn << "\n");

	Result = decodeInstruction(DecoderTableP232, Instr, Insn, Address, this, STI);

	if (Result != MCDisassembler::Fail) {
		return Result;
	}

	return MCDisassembler::Fail;
}

typedef DecodeStatus (*DecodeFunc)(MCInst &MI, unsigned insn, uint64_t Address, const void *Decoder);

