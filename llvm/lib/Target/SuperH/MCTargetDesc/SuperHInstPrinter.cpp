//===-- SuperHMCTargetDesc.cpp - SuperH assembly syntax printer -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// This file provides the ability to write SuperH instructions to a .s file.
//
//===----------------------------------------------------------------------===//

#include "SuperHInstPrinter.h"
// #include "SuperHInstrInfo.h"
#include "llvm/ADT/StringExtras.h"
#include <llvm/MC/MCInst.h>

using namespace llvm;

#define DEBUG_TYPE "sh-isel"

#define PRINT_ALIAS_INSTR
// #include "SuperHGenAsmWriter.inc"

SuperHInstPrinter::SuperHInstPrinter(const MCAsmInfo &MAI, const MCInstrInfo &MII,
                    const MCRegisterInfo &MRI) : MCInstPrinter(MAI, MII, MRI) {}

void SuperHInstPrinter::printRegName(raw_ostream &OS, MCRegister Reg) {
	OS << StringRef(getRegisterName(Reg)).lower();
}

void SuperHInstPrinter::printInst(const MCInst *MI, uint64_t Address, StringRef Annot,
                 const MCSubtargetInfo &STI, raw_ostream &OS) {
	printInstruction(MI, Address, OS);
}

void SuperHInstPrinter::printOperand(const MCInst *MI, unsigned OpNo, raw_ostream &O) {
	const MCOperand &Op = MI->getOperand(OpNo);
	
    // Print Register
	if (Op.isReg()) {
		printRegName(O, Op.getReg());
		return;
	}

	// Print immediates
	if (Op.isImm()) {
		assert(Op.getImm() <= 255 && "Only 8-bit immediates are supported.");
		O << "#" << Op.getImm();
		return;
	}
}

// TODO: Delete this placeholder.
void SuperHInstPrinter::printInstruction(const MCInst *MI, uint64_t Address, raw_ostream &O) {

}

// TODO: Delete this placeholder.
const char *SuperHInstPrinter::getRegisterName(MCRegister Reg) {
	return "r";
}

std::pair<const char *, uint64_t> SuperHInstPrinter::getMnemonic(const MCInst &MI) const {
	std::pair<const char *, uint64_t> Ret;

	return Ret;
}