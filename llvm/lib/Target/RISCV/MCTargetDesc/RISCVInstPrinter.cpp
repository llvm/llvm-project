//===-- RISCVInstPrinter.cpp - Convert RISCV MCInst to asm syntax ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class prints an RISCV MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#include "RISCVInstPrinter.h"
#include "RISCVBaseInfo.h"
#include "RISCVMCExpr.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
using namespace llvm;

#define DEBUG_TYPE "asm-printer"

// Include the auto-generated portion of the assembly writer.
#define PRINT_ALIAS_INSTR
#include "RISCVGenAsmWriter.inc"

// Include the auto-generated portion of the compress emitter.
#define GEN_UNCOMPRESS_INSTR
#include "RISCVGenCompressInstEmitter.inc"

static cl::opt<bool>
    NoAliases("riscv-no-aliases",
              cl::desc("Disable the emission of assembler pseudo instructions"),
              cl::init(false), cl::Hidden);

// Print architectural register names rather than the ABI names (such as x2
// instead of sp).
// TODO: Make RISCVInstPrinter::getRegisterName non-static so that this can a
// member.
static bool ArchRegNames;

// The command-line flags above are used by llvm-mc and llc. They can be used by
// `llvm-objdump`, but we override their values here to handle options passed to
// `llvm-objdump` with `-M` (which matches GNU objdump). There did not seem to
// be an easier way to allow these options in all these tools, without doing it
// this way.
bool RISCVInstPrinter::applyTargetSpecificCLOption(StringRef Opt) {
  if (Opt == "no-aliases") {
    PrintAliases = false;
    return true;
  }
  if (Opt == "numeric") {
    ArchRegNames = true;
    return true;
  }

  return false;
}

void RISCVInstPrinter::printInst(const MCInst *MI, uint64_t Address,
                                 StringRef Annot, const MCSubtargetInfo &STI,
                                 raw_ostream &O) {
  bool Res = false;
  const MCInst *NewMI = MI;
  MCInst UncompressedMI;
  if (PrintAliases && !NoAliases)
    Res = uncompressInst(UncompressedMI, *MI, STI);
  if (Res)
    NewMI = const_cast<MCInst *>(&UncompressedMI);
  if (!PrintAliases || NoAliases || !printAliasInstr(NewMI, Address, STI, O))
    printInstruction(NewMI, Address, STI, O);
  printAnnotation(O, Annot);
}

void RISCVInstPrinter::printRegName(raw_ostream &O, MCRegister Reg) const {
  O << getRegisterName(Reg);
}

void RISCVInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                    const MCSubtargetInfo &STI, raw_ostream &O,
                                    const char *Modifier) {
  assert((Modifier == nullptr || Modifier[0] == 0) && "No modifiers supported");
  const MCOperand &MO = MI->getOperand(OpNo);

  if (MO.isReg()) {
    printRegName(O, MO.getReg());
    return;
  }

  if (MO.isImm()) {
    O << MO.getImm();
    return;
  }

  assert(MO.isExpr() && "Unknown operand kind in printOperand");
  MO.getExpr()->print(O, &MAI);
}

void RISCVInstPrinter::printBranchOperand(const MCInst *MI, uint64_t Address,
                                          unsigned OpNo,
                                          const MCSubtargetInfo &STI,
                                          raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNo);
  if (!MO.isImm())
    return printOperand(MI, OpNo, STI, O);

  if (PrintBranchImmAsAddress) {
    uint64_t Target = Address + MO.getImm();
    if (!STI.hasFeature(RISCV::Feature64Bit))
      Target &= 0xffffffff;
    O << formatHex(Target);
  } else {
    O << MO.getImm();
  }
}

void RISCVInstPrinter::printCSRSystemRegister(const MCInst *MI, unsigned OpNo,
                                              const MCSubtargetInfo &STI,
                                              raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNo).getImm();
  auto SysReg = RISCVSysReg::lookupSysRegByEncoding(Imm);
  if (SysReg && SysReg->haveRequiredFeatures(STI.getFeatureBits()))
    O << SysReg->Name;
  else
    O << Imm;
}

void RISCVInstPrinter::printFenceArg(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  unsigned FenceArg = MI->getOperand(OpNo).getImm();
  assert (((FenceArg >> 4) == 0) && "Invalid immediate in printFenceArg");

  if ((FenceArg & RISCVFenceField::I) != 0)
    O << 'i';
  if ((FenceArg & RISCVFenceField::O) != 0)
    O << 'o';
  if ((FenceArg & RISCVFenceField::R) != 0)
    O << 'r';
  if ((FenceArg & RISCVFenceField::W) != 0)
    O << 'w';
  if (FenceArg == 0)
    O << "0";
}

void RISCVInstPrinter::printFRMArg(const MCInst *MI, unsigned OpNo,
                                   const MCSubtargetInfo &STI, raw_ostream &O) {
  auto FRMArg =
      static_cast<RISCVFPRndMode::RoundingMode>(MI->getOperand(OpNo).getImm());
  O << RISCVFPRndMode::roundingModeToString(FRMArg);
}

void RISCVInstPrinter::printZeroOffsetMemOp(const MCInst *MI, unsigned OpNo,
                                            const MCSubtargetInfo &STI,
                                            raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNo);

  assert(MO.isReg() && "printZeroOffsetMemOp can only print register operands");
  O << "(";
  printRegName(O, MO.getReg());
  O << ")";
}

void RISCVInstPrinter::printVTypeI(const MCInst *MI, unsigned OpNo,
                                   const MCSubtargetInfo &STI, raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNo).getImm();
  // Print the raw immediate for reserved values: vlmul[2:0]=4, vsew[2:0]=0b1xx,
  // or non-zero in bits 8 and above.
  if (RISCVVType::getVLMUL(Imm) == RISCVII::VLMUL::LMUL_RESERVED ||
      RISCVVType::getSEW(Imm) > 64 || (Imm >> 8) != 0) {
    O << Imm;
    return;
  }
  // Print the text form.
  RISCVVType::printVType(Imm, O);
}

void RISCVInstPrinter::printVMaskReg(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &STI,
                                     raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNo);

  assert(MO.isReg() && "printVMaskReg can only print register operands");
  if (MO.getReg() == RISCV::NoRegister)
    return;
  O << ", ";
  printRegName(O, MO.getReg());
  O << ".t";
}

const char *RISCVInstPrinter::getRegisterName(MCRegister Reg) {
  return getRegisterName(Reg, ArchRegNames ? RISCV::NoRegAltName
                                           : RISCV::ABIRegAltName);
}
