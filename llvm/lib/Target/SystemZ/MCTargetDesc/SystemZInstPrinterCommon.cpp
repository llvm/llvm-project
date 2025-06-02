//=- SystemZInstPrinterCommon.cpp - Common SystemZ MCInst to assembly funcs -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SystemZInstPrinterCommon.h"
#include "MCTargetDesc/SystemZMCExpr.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

void SystemZInstPrinterCommon::printAddress(const MCAsmInfo *MAI,
                                            MCRegister Base,
                                            const MCOperand &DispMO,
                                            MCRegister Index, raw_ostream &O) {
  printOperand(DispMO, MAI, O);
  if (Base || Index) {
    O << '(';
    if (Index) {
      printRegName(O, Index);
      O << ',';
    }
    if (Base)
      printRegName(O, Base);
    else
      O << '0';
    O << ')';
  }
}

void SystemZInstPrinterCommon::printOperand(const MCOperand &MO,
                                            const MCAsmInfo *MAI,
                                            raw_ostream &O) {
  if (MO.isReg()) {
    if (!MO.getReg())
      O << '0';
    else
      printRegName(O, MO.getReg());
  } else if (MO.isImm())
    markup(O, Markup::Immediate) << MO.getImm();
  else if (MO.isExpr())
    MO.getExpr()->print(O, MAI);
  else
    llvm_unreachable("Invalid operand");
}

void SystemZInstPrinterCommon::printRegName(raw_ostream &O, MCRegister Reg) {
  printFormattedRegName(&MAI, Reg, O);
}

template <unsigned N>
void SystemZInstPrinterCommon::printUImmOperand(const MCInst *MI, int OpNum,
                                                raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);
  if (MO.isExpr()) {
    O << *MO.getExpr();
    return;
  }
  uint64_t Value = static_cast<uint64_t>(MO.getImm());
  assert(isUInt<N>(Value) && "Invalid uimm argument");
  markup(O, Markup::Immediate) << Value;
}

template <unsigned N>
void SystemZInstPrinterCommon::printSImmOperand(const MCInst *MI, int OpNum,
                                                raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);
  if (MO.isExpr()) {
    O << *MO.getExpr();
    return;
  }
  int64_t Value = MI->getOperand(OpNum).getImm();
  assert(isInt<N>(Value) && "Invalid simm argument");
  markup(O, Markup::Immediate) << Value;
}

void SystemZInstPrinterCommon::printU1ImmOperand(const MCInst *MI, int OpNum,
                                                 raw_ostream &O) {
  printUImmOperand<1>(MI, OpNum, O);
}

void SystemZInstPrinterCommon::printU2ImmOperand(const MCInst *MI, int OpNum,
                                                 raw_ostream &O) {
  printUImmOperand<2>(MI, OpNum, O);
}

void SystemZInstPrinterCommon::printU3ImmOperand(const MCInst *MI, int OpNum,
                                                 raw_ostream &O) {
  printUImmOperand<3>(MI, OpNum, O);
}

void SystemZInstPrinterCommon::printU4ImmOperand(const MCInst *MI, int OpNum,
                                                 raw_ostream &O) {
  printUImmOperand<4>(MI, OpNum, O);
}

void SystemZInstPrinterCommon::printS8ImmOperand(const MCInst *MI, int OpNum,
                                                 raw_ostream &O) {
  printSImmOperand<8>(MI, OpNum, O);
}

void SystemZInstPrinterCommon::printU8ImmOperand(const MCInst *MI, int OpNum,
                                                 raw_ostream &O) {
  printUImmOperand<8>(MI, OpNum, O);
}

void SystemZInstPrinterCommon::printU12ImmOperand(const MCInst *MI, int OpNum,
                                                  raw_ostream &O) {
  printUImmOperand<12>(MI, OpNum, O);
}

void SystemZInstPrinterCommon::printS16ImmOperand(const MCInst *MI, int OpNum,
                                                  raw_ostream &O) {
  printSImmOperand<16>(MI, OpNum, O);
}

void SystemZInstPrinterCommon::printU16ImmOperand(const MCInst *MI, int OpNum,
                                                  raw_ostream &O) {
  printUImmOperand<16>(MI, OpNum, O);
}

void SystemZInstPrinterCommon::printS32ImmOperand(const MCInst *MI, int OpNum,
                                                  raw_ostream &O) {
  printSImmOperand<32>(MI, OpNum, O);
}

void SystemZInstPrinterCommon::printU32ImmOperand(const MCInst *MI, int OpNum,
                                                  raw_ostream &O) {
  printUImmOperand<32>(MI, OpNum, O);
}

void SystemZInstPrinterCommon::printU48ImmOperand(const MCInst *MI, int OpNum,
                                                  raw_ostream &O) {
  printUImmOperand<48>(MI, OpNum, O);
}

void SystemZInstPrinterCommon::printPCRelOperand(const MCInst *MI,
                                                 uint64_t Address, int OpNum,
                                                 raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(OpNum);

  // If the label has already been resolved to an immediate offset (say, when
  // we're running the disassembler), just print the immediate.
  if (MO.isImm()) {
    int64_t Offset = MO.getImm();
    if (PrintBranchImmAsAddress)
      markup(O, Markup::Target) << formatHex(Address + Offset);
    else
      markup(O, Markup::Immediate) << formatImm(Offset);
    return;
  }

  // If the branch target is simply an address then print it in hex.
  const MCConstantExpr *BranchTarget = dyn_cast<MCConstantExpr>(MO.getExpr());
  int64_t TargetAddress;
  if (BranchTarget && BranchTarget->evaluateAsAbsolute(TargetAddress)) {
    markup(O, Markup::Target) << formatHex((uint64_t)TargetAddress);
  } else {
    // Otherwise, just print the expression.
    MO.getExpr()->print(O, &MAI);
  }
}

void SystemZInstPrinterCommon::printPCRelTLSOperand(const MCInst *MI,
                                                    uint64_t Address, int OpNum,
                                                    raw_ostream &O) {
  // Output the PC-relative operand.
  printPCRelOperand(MI, Address, OpNum, O);

  // Output the TLS marker if present.
  if ((unsigned)OpNum + 1 < MI->getNumOperands()) {
    const MCOperand &MO = MI->getOperand(OpNum + 1);
    const MCSymbolRefExpr &refExp = cast<MCSymbolRefExpr>(*MO.getExpr());
    switch (getSpecifier(&refExp)) {
    case SystemZMCExpr::VK_TLSGD:
      O << ":tls_gdcall:";
      break;
    case SystemZMCExpr::VK_TLSLDM:
      O << ":tls_ldcall:";
      break;
    default:
      llvm_unreachable("Unexpected symbol kind");
    }
    O << refExp.getSymbol().getName();
  }
}

void SystemZInstPrinterCommon::printOperand(const MCInst *MI, int OpNum,
                                            raw_ostream &O) {
  printOperand(MI->getOperand(OpNum), &MAI, O);
}

void SystemZInstPrinterCommon::printBDAddrOperand(const MCInst *MI, int OpNum,
                                                  raw_ostream &O) {
  printAddress(&MAI, MI->getOperand(OpNum).getReg(), MI->getOperand(OpNum + 1),
               0, O);
}

void SystemZInstPrinterCommon::printBDXAddrOperand(const MCInst *MI, int OpNum,
                                                   raw_ostream &O) {
  printAddress(&MAI, MI->getOperand(OpNum).getReg(), MI->getOperand(OpNum + 1),
               MI->getOperand(OpNum + 2).getReg(), O);
}

void SystemZInstPrinterCommon::printBDLAddrOperand(const MCInst *MI, int OpNum,
                                                   raw_ostream &O) {
  unsigned Base = MI->getOperand(OpNum).getReg();
  const MCOperand &DispMO = MI->getOperand(OpNum + 1);
  uint64_t Length = MI->getOperand(OpNum + 2).getImm();
  printOperand(DispMO, &MAI, O);
  O << '(' << Length;
  if (Base) {
    O << ",";
    printRegName(O, Base);
  }
  O << ')';
}

void SystemZInstPrinterCommon::printBDRAddrOperand(const MCInst *MI, int OpNum,
                                                   raw_ostream &O) {
  unsigned Base = MI->getOperand(OpNum).getReg();
  const MCOperand &DispMO = MI->getOperand(OpNum + 1);
  unsigned Length = MI->getOperand(OpNum + 2).getReg();
  printOperand(DispMO, &MAI, O);
  O << "(";
  printRegName(O, Length);
  if (Base) {
    O << ",";
    printRegName(O, Base);
  }
  O << ')';
}

void SystemZInstPrinterCommon::printBDVAddrOperand(const MCInst *MI, int OpNum,
                                                   raw_ostream &O) {
  printAddress(&MAI, MI->getOperand(OpNum).getReg(), MI->getOperand(OpNum + 1),
               MI->getOperand(OpNum + 2).getReg(), O);
}

void SystemZInstPrinterCommon::printLXAAddrOperand(const MCInst *MI, int OpNum,
                                             raw_ostream &O) {
  printAddress(&MAI, MI->getOperand(OpNum).getReg(), MI->getOperand(OpNum + 1),
               MI->getOperand(OpNum + 2).getReg(), O);
}

void SystemZInstPrinterCommon::printCond4Operand(const MCInst *MI, int OpNum,
                                                 raw_ostream &O) {
  static const char *const CondNames[] = {"o",  "h",  "nle", "l",   "nhe",
                                          "lh", "ne", "e",   "nlh", "he",
                                          "nl", "le", "nh",  "no"};
  uint64_t Imm = MI->getOperand(OpNum).getImm();
  assert(Imm > 0 && Imm < 15 && "Invalid condition");
  O << CondNames[Imm - 1];
}
