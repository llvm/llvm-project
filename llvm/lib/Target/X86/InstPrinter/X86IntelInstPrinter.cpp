//===-- X86IntelInstPrinter.cpp - Intel assembly instruction printing -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file includes code for rendering MCInst instances as Intel-style
// assembly.
//
//===----------------------------------------------------------------------===//

#include "X86IntelInstPrinter.h"
#include "MCTargetDesc/X86BaseInfo.h"
#include "X86InstComments.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <cstdint>

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

#include "X86GenAsmWriter1.inc"

void X86IntelInstPrinter::printRegName(raw_ostream &OS, unsigned RegNo) const {
  OS << getRegisterName(RegNo);
}

void X86IntelInstPrinter::printInst(const MCInst *MI, raw_ostream &OS,
                                    StringRef Annot,
                                    const MCSubtargetInfo &STI) {
  const MCInstrDesc &Desc = MII.get(MI->getOpcode());
  uint64_t TSFlags = Desc.TSFlags;
  unsigned Flags = MI->getFlags();

  if ((TSFlags & X86II::LOCK) || (Flags & X86::IP_HAS_LOCK))
    OS << "\tlock\t";

  if (Flags & X86::IP_HAS_REPEAT_NE)
    OS << "\trepne\t";
  else if (Flags & X86::IP_HAS_REPEAT)
    OS << "\trep\t";

  printInstruction(MI, OS);

  // Next always print the annotation.
  printAnnotation(OS, Annot);

  // If verbose assembly is enabled, we can print some informative comments.
  if (CommentStream)
    EmitAnyX86InstComments(MI, *CommentStream, getRegisterName);
}

void X86IntelInstPrinter::printSSEAVXCC(const MCInst *MI, unsigned Op,
                                        raw_ostream &O) {
  int64_t Imm = MI->getOperand(Op).getImm();
  switch (Imm) {
  default: llvm_unreachable("Invalid avxcc argument!");
  case    0: O << "eq"; break;
  case    1: O << "lt"; break;
  case    2: O << "le"; break;
  case    3: O << "unord"; break;
  case    4: O << "neq"; break;
  case    5: O << "nlt"; break;
  case    6: O << "nle"; break;
  case    7: O << "ord"; break;
  case    8: O << "eq_uq"; break;
  case    9: O << "nge"; break;
  case  0xa: O << "ngt"; break;
  case  0xb: O << "false"; break;
  case  0xc: O << "neq_oq"; break;
  case  0xd: O << "ge"; break;
  case  0xe: O << "gt"; break;
  case  0xf: O << "true"; break;
  case 0x10: O << "eq_os"; break;
  case 0x11: O << "lt_oq"; break;
  case 0x12: O << "le_oq"; break;
  case 0x13: O << "unord_s"; break;
  case 0x14: O << "neq_us"; break;
  case 0x15: O << "nlt_uq"; break;
  case 0x16: O << "nle_uq"; break;
  case 0x17: O << "ord_s"; break;
  case 0x18: O << "eq_us"; break;
  case 0x19: O << "nge_uq"; break;
  case 0x1a: O << "ngt_uq"; break;
  case 0x1b: O << "false_os"; break;
  case 0x1c: O << "neq_os"; break;
  case 0x1d: O << "ge_oq"; break;
  case 0x1e: O << "gt_oq"; break;
  case 0x1f: O << "true_us"; break;
  }
}

void X86IntelInstPrinter::printXOPCC(const MCInst *MI, unsigned Op,
                                     raw_ostream &O) {
  int64_t Imm = MI->getOperand(Op).getImm();
  switch (Imm) {
  default: llvm_unreachable("Invalid xopcc argument!");
  case 0: O << "lt"; break;
  case 1: O << "le"; break;
  case 2: O << "gt"; break;
  case 3: O << "ge"; break;
  case 4: O << "eq"; break;
  case 5: O << "neq"; break;
  case 6: O << "false"; break;
  case 7: O << "true"; break;
  }
}

void X86IntelInstPrinter::printRoundingControl(const MCInst *MI, unsigned Op,
                                               raw_ostream &O) {
  int64_t Imm = MI->getOperand(Op).getImm() & 0x3;
  switch (Imm) {
  case 0: O << "{rn-sae}"; break;
  case 1: O << "{rd-sae}"; break;
  case 2: O << "{ru-sae}"; break;
  case 3: O << "{rz-sae}"; break;
  }
}

/// printPCRelImm - This is used to print an immediate value that ends up
/// being encoded as a pc-relative value.
void X86IntelInstPrinter::printPCRelImm(const MCInst *MI, unsigned OpNo,
                                        raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isImm())
    O << formatImm(Op.getImm());
  else {
    assert(Op.isExpr() && "unknown pcrel immediate operand");
    // If a symbolic branch target was added as a constant expression then print
    // that address in hex.
    const MCConstantExpr *BranchTarget = dyn_cast<MCConstantExpr>(Op.getExpr());
    int64_t Address;
    if (BranchTarget && BranchTarget->evaluateAsAbsolute(Address)) {
      O << formatHex((uint64_t)Address);
    }
    else {
      // Otherwise, just print the expression.
      Op.getExpr()->print(O, &MAI);
    }
  }
}

void X86IntelInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                       raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    printRegName(O, Op.getReg());
  } else if (Op.isImm()) {
    O << formatImm((int64_t)Op.getImm());
  } else {
    assert(Op.isExpr() && "unknown operand kind in printOperand");
    O << "offset ";
    Op.getExpr()->print(O, &MAI);
  }
}

void X86IntelInstPrinter::printMemReference(const MCInst *MI, unsigned Op,
                                            raw_ostream &O) {
  const MCOperand &BaseReg  = MI->getOperand(Op+X86::AddrBaseReg);
  unsigned ScaleVal         = MI->getOperand(Op+X86::AddrScaleAmt).getImm();
  const MCOperand &IndexReg = MI->getOperand(Op+X86::AddrIndexReg);
  const MCOperand &DispSpec = MI->getOperand(Op+X86::AddrDisp);
  const MCOperand &SegReg   = MI->getOperand(Op+X86::AddrSegmentReg);

  // If this has a segment register, print it.
  if (SegReg.getReg()) {
    printOperand(MI, Op+X86::AddrSegmentReg, O);
    O << ':';
  }

  O << '[';

  bool NeedPlus = false;
  if (BaseReg.getReg()) {
    printOperand(MI, Op+X86::AddrBaseReg, O);
    NeedPlus = true;
  }

  if (IndexReg.getReg()) {
    if (NeedPlus) O << " + ";
    if (ScaleVal != 1)
      O << ScaleVal << '*';
    printOperand(MI, Op+X86::AddrIndexReg, O);
    NeedPlus = true;
  }

  if (!DispSpec.isImm()) {
    if (NeedPlus) O << " + ";
    assert(DispSpec.isExpr() && "non-immediate displacement for LEA?");
    DispSpec.getExpr()->print(O, &MAI);
  } else {
    int64_t DispVal = DispSpec.getImm();
    if (DispVal || (!IndexReg.getReg() && !BaseReg.getReg())) {
      if (NeedPlus) {
        if (DispVal > 0)
          O << " + ";
        else {
          O << " - ";
          DispVal = -DispVal;
        }
      }
      O << formatImm(DispVal);
    }
  }

  O << ']';
}

void X86IntelInstPrinter::printSrcIdx(const MCInst *MI, unsigned Op,
                                      raw_ostream &O) {
  const MCOperand &SegReg   = MI->getOperand(Op+1);

  // If this has a segment register, print it.
  if (SegReg.getReg()) {
    printOperand(MI, Op+1, O);
    O << ':';
  }
  O << '[';
  printOperand(MI, Op, O);
  O << ']';
}

void X86IntelInstPrinter::printDstIdx(const MCInst *MI, unsigned Op,
                                      raw_ostream &O) {
  // DI accesses are always ES-based.
  O << "es:[";
  printOperand(MI, Op, O);
  O << ']';
}

void X86IntelInstPrinter::printMemOffset(const MCInst *MI, unsigned Op,
                                         raw_ostream &O) {
  const MCOperand &DispSpec = MI->getOperand(Op);
  const MCOperand &SegReg   = MI->getOperand(Op+1);

  // If this has a segment register, print it.
  if (SegReg.getReg()) {
    printOperand(MI, Op+1, O);
    O << ':';
  }

  O << '[';

  if (DispSpec.isImm()) {
    O << formatImm(DispSpec.getImm());
  } else {
    assert(DispSpec.isExpr() && "non-immediate displacement?");
    DispSpec.getExpr()->print(O, &MAI);
  }

  O << ']';
}

void X86IntelInstPrinter::printU8Imm(const MCInst *MI, unsigned Op,
                                     raw_ostream &O) {
  if (MI->getOperand(Op).isExpr())
    return MI->getOperand(Op).getExpr()->print(O, &MAI);

  O << formatImm(MI->getOperand(Op).getImm() & 0xff);
}
