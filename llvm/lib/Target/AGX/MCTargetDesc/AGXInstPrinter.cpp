//===-- AGXInstPrinter.cpp - Convert AGX MCInst to assembly ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The syntax follows the applegpu disassembler (dougallj/applegpu), so that
// output can be compared with it directly: registers r0l/r0h/r0/r0_r1,
// uniforms u0l/u0/u0_u1, and source flags as suffixes (.cache, .discard,
// .abs, .neg, .sx).
//
//===----------------------------------------------------------------------===//

#include "AGXInstPrinter.h"
#include "MCTargetDesc/AGXMCTargetDesc.h"
#include "llvm/ADT/bit.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

#define PRINT_ALIAS_INSTR
#include "AGXGenAsmWriter.inc"

void AGXInstPrinter::printInst(const MCInst *MI, uint64_t Address,
                               StringRef Annot, const MCSubtargetInfo &STI,
                               raw_ostream &O) {
  if (!printAliasInstr(MI, Address, STI, O))
    printInstruction(MI, Address, STI, O);
  printAnnotation(O, Annot);
}

void AGXInstPrinter::printRegName(raw_ostream &O, MCRegister Reg) {
  printReg(Reg, O);
}

void AGXInstPrinter::printReg(MCRegister Reg, raw_ostream &O) {
  AGX::RegInfo Info = AGX::getRegInfo(Reg, MRI);
  unsigned Count = 0;
  AGX::RegKind EltKind = AGX::RegKind::GPR32;
  switch (Info.Kind) {
  case AGX::RegKind::GPR64:
    Count = 2;
    break;
  case AGX::RegKind::GPR96:
    Count = 3;
    break;
  case AGX::RegKind::GPR128:
    Count = 4;
    break;
  case AGX::RegKind::UGPR64:
    Count = 2;
    EltKind = AGX::RegKind::UGPR32;
    break;
  default:
    O << getRegisterName(Reg);
    return;
  }
  for (unsigned I = 0; I < Count; ++I) {
    if (I)
      O << '_';
    O << getRegisterName(
        AGX::getRegister(EltKind, Info.HalfIndex + 2 * I, MRI));
  }
}

static void printFlag(raw_ostream &O, unsigned Mods, unsigned Flag,
                      StringRef Name) {
  if (Mods & Flag)
    O << '.' << Name;
}

void AGXInstPrinter::printALUSrc(const MCInst *MI, unsigned OpNo, bool IsFloat,
                                 raw_ostream &O) {
  unsigned Mods = MI->getOperand(OpNo).getImm();
  const MCOperand &Val = MI->getOperand(OpNo + 1);
  if (Val.isImm()) {
    if (IsFloat)
      O << AGX::formatFPImm8(Val.getImm() & 0xff);
    else
      O << Val.getImm();
    return;
  }
  if (Val.isExpr()) {
    MAI.printExpr(O, *Val.getExpr());
    return;
  }
  printReg(Val.getReg(), O);
  printFlag(O, Mods, AGX::SRC_MOD_CACHE, "cache");
  printFlag(O, Mods, AGX::SRC_MOD_DISCARD, "discard");
  printFlag(O, Mods, AGX::SRC_MOD_ABS, "abs");
  printFlag(O, Mods, AGX::SRC_MOD_NEG, "neg");
  printFlag(O, Mods, AGX::SRC_MOD_SX, "sx");
}

#define AGX_INT_SRC_PRINTER(Kind)                                              \
  void AGXInstPrinter::print##Kind(const MCInst *MI, unsigned OpNo,            \
                                   const MCSubtargetInfo &, raw_ostream &O) {  \
    printALUSrc(MI, OpNo, /*IsFloat=*/false, O);                               \
  }
AGX_INT_SRC_PRINTER(Src)
AGX_INT_SRC_PRINTER(Src64)
AGX_INT_SRC_PRINTER(Src16)
AGX_INT_SRC_PRINTER(SrcSX)
AGX_INT_SRC_PRINTER(SrcSX64)
AGX_INT_SRC_PRINTER(CmpselSrc)
#undef AGX_INT_SRC_PRINTER

void AGXInstPrinter::printSrcF(const MCInst *MI, unsigned OpNo,
                               const MCSubtargetInfo &, raw_ostream &O) {
  printALUSrc(MI, OpNo, /*IsFloat=*/true, O);
}

void AGXInstPrinter::printSrcF16(const MCInst *MI, unsigned OpNo,
                                 const MCSubtargetInfo &, raw_ostream &O) {
  printALUSrc(MI, OpNo, /*IsFloat=*/true, O);
}

void AGXInstPrinter::printDst(const MCInst *MI, unsigned OpNo,
                              const MCSubtargetInfo &, raw_ostream &O) {
  printReg(MI->getOperand(OpNo).getReg(), O);
  if (MI->getOperand(OpNo + 1).getImm())
    O << ".cache";
}

void AGXInstPrinter::printImm(const MCInst *MI, unsigned OpNo,
                              const MCSubtargetInfo &, raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isImm())
    O << Op.getImm();
  else if (Op.isExpr())
    MAI.printExpr(O, *Op.getExpr());
  else
    O << getRegisterName(Op.getReg());
}

void AGXInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                  const MCSubtargetInfo &STI,
                                  raw_ostream &O) {
  printImm(MI, OpNo, STI, O);
}

void AGXInstPrinter::printShift(const MCInst *MI, unsigned OpNo,
                                const MCSubtargetInfo &, raw_ostream &O) {
  if (int64_t S = MI->getOperand(OpNo).getImm())
    O << ", lsl " << S;
}

void AGXInstPrinter::printStoreFlag(const MCInst *MI, unsigned OpNo,
                                    const MCSubtargetInfo &, raw_ostream &O) {
  if (int64_t V = MI->getOperand(OpNo).getImm())
    O << ", " << V;
}

void AGXInstPrinter::printBitfieldMask(const MCInst *MI, unsigned OpNo,
                                       const MCSubtargetInfo &,
                                       raw_ostream &O) {
  if (int64_t M = MI->getOperand(OpNo).getImm())
    O << ", mask 0x" << utohexstr((uint64_t(1) << M) - 1, /*LowerCase=*/false);
}

void AGXInstPrinter::printTruthTable(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &, raw_ostream &O) {
  int64_t TT = MI->getOperand(OpNo).getImm();
  for (unsigned I = 0; I < 4; ++I)
    O << ((TT >> I) & 1);
}

static void printEnum(raw_ostream &O, StringRef Name, int64_t Value) {
  if (Name.empty())
    O << Value;
  else
    O << Name;
}

void AGXInstPrinter::printICond(const MCInst *MI, unsigned OpNo,
                                const MCSubtargetInfo &, raw_ostream &O) {
  int64_t CC = MI->getOperand(OpNo).getImm();
  printEnum(O, AGX::getICondName(CC), CC);
}

void AGXInstPrinter::printFCond(const MCInst *MI, unsigned OpNo,
                                const MCSubtargetInfo &, raw_ostream &O) {
  int64_t CC = MI->getOperand(OpNo).getImm();
  printEnum(O, AGX::getFCondName(CC), CC);
}

void AGXInstPrinter::printICond3(const MCInst *MI, unsigned OpNo,
                                 const MCSubtargetInfo &STI, raw_ostream &O) {
  printICond(MI, OpNo, STI, O);
}

void AGXInstPrinter::printFCond3(const MCInst *MI, unsigned OpNo,
                                 const MCSubtargetInfo &STI, raw_ostream &O) {
  printFCond(MI, OpNo, STI, O);
}

void AGXInstPrinter::printConvertMode(const MCInst *MI, unsigned OpNo,
                                      const MCSubtargetInfo &, raw_ostream &O) {
  int64_t V = MI->getOperand(OpNo).getImm();
  printEnum(O, AGX::getConvertModeName(V), V);
}

void AGXInstPrinter::printRoundMode(const MCInst *MI, unsigned OpNo,
                                    const MCSubtargetInfo &, raw_ostream &O) {
  int64_t V = MI->getOperand(OpNo).getImm();
  printEnum(O, AGX::getRoundModeName(V), V);
}

void AGXInstPrinter::printMemFormat(const MCInst *MI, unsigned OpNo,
                                    const MCSubtargetInfo &, raw_ostream &O) {
  int64_t V = MI->getOperand(OpNo).getImm();
  printEnum(O, AGX::getMemFormatName(V), V);
}

void AGXInstPrinter::printMemMask(const MCInst *MI, unsigned OpNo,
                                  const MCSubtargetInfo &, raw_ostream &O) {
  int64_t V = MI->getOperand(OpNo).getImm();
  std::string Name = AGX::getMemMaskName(V);
  printEnum(O, Name, V);
}

void AGXInstPrinter::printSignedness(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &, raw_ostream &O) {
  O << (MI->getOperand(OpNo).getImm() ? "unsigned" : "signed");
}

void AGXInstPrinter::printAtomicOp(const MCInst *MI, unsigned OpNo,
                                   const MCSubtargetInfo &, raw_ostream &O) {
  int64_t V = MI->getOperand(OpNo).getImm();
  printEnum(O, AGX::getAtomicOpName(V), V);
}

void AGXInstPrinter::printSpecialReg(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &, raw_ostream &O) {
  int64_t SR = MI->getOperand(OpNo).getImm();
  O << "sr" << SR;
  StringRef Name = AGX::getSpecialRegName(SR);
  if (!Name.empty())
    O << " (" << Name << ')';
}

void AGXInstPrinter::printRegisterTuple(MCRegister First, unsigned Count,
                                        raw_ostream &O) {
  AGX::RegInfo Info = AGX::getRegInfo(First, MRI);
  bool Is16 = Info.Kind == AGX::RegKind::GPR16;
  unsigned Step = Is16 ? 1 : 2;
  AGX::RegKind Kind = Is16 ? AGX::RegKind::GPR16 : AGX::RegKind::GPR32;
  for (unsigned I = 0; I < std::max(Count, 1u); ++I) {
    if (I)
      O << '_';
    MCRegister R = AGX::getRegister(Kind, Info.HalfIndex + I * Step, MRI);
    if (R)
      O << getRegisterName(R);
    else
      O << "<out of range>";
  }
}

static unsigned getMemCount(const MCInst *MI) {
  int16_t MaskIdx = AGX::getNamedOperandIdx(MI->getOpcode(), AGX::OpName::mask);
  if (MaskIdx < 0)
    return 1;
  return llvm::popcount(uint64_t(MI->getOperand(MaskIdx).getImm()));
}

void AGXInstPrinter::printMemReg(const MCInst *MI, unsigned OpNo,
                                 const MCSubtargetInfo &, raw_ostream &O) {
  printRegisterTuple(MI->getOperand(OpNo).getReg(), getMemCount(MI), O);
}

void AGXInstPrinter::printAtomicDst(const MCInst *MI, unsigned OpNo,
                                    const MCSubtargetInfo &, raw_ostream &O) {
  MCRegister R = MI->getOperand(OpNo).getReg();
  if (!R)
    O << '_';
  else
    O << getRegisterName(R);
}

void AGXInstPrinter::printAtomicSrc(const MCInst *MI, unsigned OpNo,
                                    const MCSubtargetInfo &, raw_ostream &O) {
  printReg(MI->getOperand(OpNo).getReg(), O);
}

void AGXInstPrinter::printMemBase(const MCInst *MI, unsigned OpNo,
                                  const MCSubtargetInfo &, raw_ostream &O) {
  printReg(MI->getOperand(OpNo).getReg(), O);
}

void AGXInstPrinter::printMemIndex(const MCInst *MI, unsigned OpNo,
                                   const MCSubtargetInfo &, raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg())
    O << getRegisterName(Op.getReg());
  else
    O << int16_t(Op.getImm());
}

void AGXInstPrinter::printLocalBase(const MCInst *MI, unsigned OpNo,
                                    const MCSubtargetInfo &, raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg())
    O << getRegisterName(Op.getReg());
  else
    O << Op.getImm();
}

void AGXInstPrinter::printLocalIndex(const MCInst *MI, unsigned OpNo,
                                     const MCSubtargetInfo &, raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg())
    O << getRegisterName(Op.getReg());
  else
    O << int16_t(Op.getImm());
}

void AGXInstPrinter::printNestCounter(const MCInst *MI, unsigned OpNo,
                                      const MCSubtargetInfo &, raw_ostream &O) {
  O << "r0l";
  if (MI->getOperand(OpNo).getImm())
    O << ".cache";
}

void AGXInstPrinter::printBrTarget(const MCInst *MI, uint64_t Address,
                                   unsigned OpNo, const MCSubtargetInfo &,
                                   raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isExpr()) {
    MAI.printExpr(O, *Op.getExpr());
    return;
  }
  int64_t Offset = int32_t(Op.getImm());
  if (PrintBranchImmAsAddress) {
    O << formatHex(uint64_t(Address + Offset));
    return;
  }
  if (Offset >= 0)
    O << "pc+" << Offset;
  else
    O << "pc-" << -Offset;
}
