//===-- RISCVInstPrinter.cpp - Convert RISC-V MCInst to asm syntax --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class prints an RISC-V MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#include "RISCVInstPrinter.h"
#include "RISCVBaseInfo.h"
#include "RISCVMCExpr.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
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
    Res = RISCVRVC::uncompress(UncompressedMI, *MI, STI);
  if (Res)
    NewMI = const_cast<MCInst *>(&UncompressedMI);
  if (!PrintAliases || NoAliases || !printAliasInstr(NewMI, Address, STI, O))
    printInstruction(NewMI, Address, STI, O);
  printAnnotation(O, Annot);
}

void RISCVInstPrinter::printRegName(raw_ostream &O, MCRegister Reg) const {
  markup(O, Markup::Register) << getRegisterName(Reg);
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
    markup(O, Markup::Immediate) << MO.getImm();
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
    markup(O, Markup::Target) << formatHex(Target);
  } else {
    markup(O, Markup::Target) << MO.getImm();
  }
}

void RISCVInstPrinter::printCSRSystemRegister(const MCInst *MI, unsigned OpNo,
                                              const MCSubtargetInfo &STI,
                                              raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNo).getImm();
  auto SiFiveReg = RISCVSysReg::lookupSiFiveRegByEncoding(Imm);
  auto SysReg = RISCVSysReg::lookupSysRegByEncoding(Imm);
  if (SiFiveReg && SiFiveReg->haveVendorRequiredFeatures(STI.getFeatureBits()))
    markup(O, Markup::Register) << SiFiveReg->Name;
  else if (SysReg && SysReg->haveRequiredFeatures(STI.getFeatureBits()))
    markup(O, Markup::Register) << SysReg->Name;
  else
    markup(O, Markup::Register) << Imm;
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
  if (PrintAliases && !NoAliases && FRMArg == RISCVFPRndMode::RoundingMode::DYN)
    return;
  O << ", " << RISCVFPRndMode::roundingModeToString(FRMArg);
}

void RISCVInstPrinter::printFPImmOperand(const MCInst *MI, unsigned OpNo,
                                         const MCSubtargetInfo &STI,
                                         raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNo).getImm();
  if (Imm == 1) {
    markup(O, Markup::Immediate) << "min";
  } else if (Imm == 30) {
    markup(O, Markup::Immediate) << "inf";
  } else if (Imm == 31) {
    markup(O, Markup::Immediate) << "nan";
  } else {
    float FPVal = RISCVLoadFPImm::getFPImm(Imm);
    // If the value is an integer, print a .0 fraction. Otherwise, use %g to
    // which will not print trailing zeros and will use scientific notation
    // if it is shorter than printing as a decimal. The smallest value requires
    // 12 digits of precision including the decimal.
    if (FPVal == (int)(FPVal))
      markup(O, Markup::Immediate) << format("%.1f", FPVal);
    else
      markup(O, Markup::Immediate) << format("%.12g", FPVal);
  }
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

void RISCVInstPrinter::printRlist(const MCInst *MI, unsigned OpNo,
                                  const MCSubtargetInfo &STI, raw_ostream &O) {
  unsigned Imm = MI->getOperand(OpNo).getImm();
  O << "{";
  switch (Imm) {
  case RISCVZC::RLISTENCODE::RA:
    markup(O, Markup::Register) << (ArchRegNames ? "x1" : "ra");
    break;
  case RISCVZC::RLISTENCODE::RA_S0:
    markup(O, Markup::Register) << (ArchRegNames ? "x1" : "ra");
    O << ", ";
    markup(O, Markup::Register) << (ArchRegNames ? "x8" : "s0");
    break;
  case RISCVZC::RLISTENCODE::RA_S0_S1:
    markup(O, Markup::Register) << (ArchRegNames ? "x1" : "ra");
    O << ", ";
    markup(O, Markup::Register) << (ArchRegNames ? "x8" : "s0");
    O << '-';
    markup(O, Markup::Register) << (ArchRegNames ? "x9" : "s1");
    break;
  case RISCVZC::RLISTENCODE::RA_S0_S2:
    markup(O, Markup::Register) << (ArchRegNames ? "x1" : "ra");
    O << ", ";
    markup(O, Markup::Register) << (ArchRegNames ? "x8" : "s0");
    O << '-';
    markup(O, Markup::Register) << (ArchRegNames ? "x9" : "s2");
    if (ArchRegNames) {
      O << ", ";
      markup(O, Markup::Register) << "x18";
    }
    break;
  case RISCVZC::RLISTENCODE::RA_S0_S3:
  case RISCVZC::RLISTENCODE::RA_S0_S4:
  case RISCVZC::RLISTENCODE::RA_S0_S5:
  case RISCVZC::RLISTENCODE::RA_S0_S6:
  case RISCVZC::RLISTENCODE::RA_S0_S7:
  case RISCVZC::RLISTENCODE::RA_S0_S8:
  case RISCVZC::RLISTENCODE::RA_S0_S9:
  case RISCVZC::RLISTENCODE::RA_S0_S11:
    markup(O, Markup::Register) << (ArchRegNames ? "x1" : "ra");
    O << ", ";
    markup(O, Markup::Register) << (ArchRegNames ? "x8" : "s0");
    O << '-';
    if (ArchRegNames) {
      markup(O, Markup::Register) << "x9";
      O << ", ";
      markup(O, Markup::Register) << "x18";
      O << '-';
    }
    markup(O, Markup::Register) << getRegisterName(
        RISCV::X19 + (Imm == RISCVZC::RLISTENCODE::RA_S0_S11
                          ? 8
                          : Imm - RISCVZC::RLISTENCODE::RA_S0_S3));
    break;
  default:
    llvm_unreachable("invalid register list");
  }
  O << "}";
}

void RISCVInstPrinter::printSpimm(const MCInst *MI, unsigned OpNo,
                                  const MCSubtargetInfo &STI, raw_ostream &O) {
  int64_t Imm = MI->getOperand(OpNo).getImm();
  unsigned Opcode = MI->getOpcode();
  bool IsRV64 = STI.hasFeature(RISCV::Feature64Bit);
  bool IsEABI = STI.hasFeature(RISCV::FeatureRVE);
  int64_t Spimm = 0;
  auto RlistVal = MI->getOperand(0).getImm();
  assert(RlistVal != 16 && "Incorrect rlist.");
  auto Base = RISCVZC::getStackAdjBase(RlistVal, IsRV64, IsEABI);
  Spimm = Imm + Base;
  assert((Spimm >= Base && Spimm <= Base + 48) && "Incorrect spimm");
  if (Opcode == RISCV::CM_PUSH)
    Spimm = -Spimm;

  // RAII guard for ANSI color escape sequences
  WithMarkup ScopedMarkup = markup(O, Markup::Immediate);
  RISCVZC::printSpimm(Spimm, O);
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
