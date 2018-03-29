//===-- X86ATTInstPrinter.cpp - AT&T assembly instruction printing --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file includes code for rendering MCInst instances as AT&T-style
// assembly.
//
//===----------------------------------------------------------------------===//

#include "X86ATTInstPrinter.h"
#include "MCTargetDesc/X86BaseInfo.h"
#include "X86InstComments.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cinttypes>
#include <cstdint>

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

// Include the auto-generated portion of the assembly writer.
#define PRINT_ALIAS_INSTR
#include "X86GenAsmWriter.inc"

void X86ATTInstPrinter::printRegName(raw_ostream &OS, unsigned RegNo) const {
  OS << markup("<reg:") << '%' << getRegisterName(RegNo) << markup(">");
}

void X86ATTInstPrinter::printInst(const MCInst *MI, raw_ostream &OS,
                                  StringRef Annot, const MCSubtargetInfo &STI) {
  const MCInstrDesc &Desc = MII.get(MI->getOpcode());
  uint64_t TSFlags = Desc.TSFlags;

  // If verbose assembly is enabled, we can print some informative comments.
  if (CommentStream)
    HasCustomInstComment = EmitAnyX86InstComments(MI, *CommentStream, MII);

  unsigned Flags = MI->getFlags();
  if ((TSFlags & X86II::LOCK) || (Flags & X86::IP_HAS_LOCK))
    OS << "\tlock\t";

  if ((TSFlags & X86II::NOTRACK) || (Flags & X86::IP_HAS_NOTRACK))
    OS << "\tnotrack\t";

  if (Flags & X86::IP_HAS_REPEAT_NE)
    OS << "\trepne\t";
  else if (Flags & X86::IP_HAS_REPEAT)
    OS << "\trep\t";

  // Output CALLpcrel32 as "callq" in 64-bit mode.
  // In Intel annotation it's always emitted as "call".
  //
  // TODO: Probably this hack should be redesigned via InstAlias in
  // InstrInfo.td as soon as Requires clause is supported properly
  // for InstAlias.
  if (MI->getOpcode() == X86::CALLpcrel32 &&
      (STI.getFeatureBits()[X86::Mode64Bit])) {
    OS << "\tcallq\t";
    printPCRelImm(MI, 0, OS);
  }
  // data16 and data32 both have the same encoding of 0x66. While data32 is
  // valid only in 16 bit systems, data16 is valid in the rest.
  // There seems to be some lack of support of the Requires clause that causes
  // 0x66 to be interpreted as "data16" by the asm printer.
  // Thus we add an adjustment here in order to print the "right" instruction.
  else if (MI->getOpcode() == X86::DATA16_PREFIX &&
    (STI.getFeatureBits()[X86::Mode16Bit])) {
    MCInst Data32MI(*MI);
    Data32MI.setOpcode(X86::DATA32_PREFIX);
    printInstruction(&Data32MI, OS);
  }
  // Try to print any aliases first.
  else if (!printAliasInstr(MI, OS))
    printInstruction(MI, OS);

  // Next always print the annotation.
  printAnnotation(OS, Annot);
}

void X86ATTInstPrinter::printSSEAVXCC(const MCInst *MI, unsigned Op,
                                      raw_ostream &O) {
  int64_t Imm = MI->getOperand(Op).getImm();
  switch (Imm) {
  default: llvm_unreachable("Invalid ssecc/avxcc argument!");
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

void X86ATTInstPrinter::printXOPCC(const MCInst *MI, unsigned Op,
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

void X86ATTInstPrinter::printRoundingControl(const MCInst *MI, unsigned Op,
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
/// being encoded as a pc-relative value (e.g. for jumps and calls).  These
/// print slightly differently than normal immediates.  For example, a $ is not
/// emitted.
void X86ATTInstPrinter::printPCRelImm(const MCInst *MI, unsigned OpNo,
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
    } else {
      // Otherwise, just print the expression.
      Op.getExpr()->print(O, &MAI);
    }
  }
}

void X86ATTInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                     raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    printRegName(O, Op.getReg());
  } else if (Op.isImm()) {
    // Print immediates as signed values.
    int64_t Imm = Op.getImm();
    O << markup("<imm:") << '$' << formatImm(Imm) << markup(">");

    // TODO: This should be in a helper function in the base class, so it can
    // be used by other printers.

    // If there are no instruction-specific comments, add a comment clarifying
    // the hex value of the immediate operand when it isn't in the range
    // [-256,255].
    if (CommentStream && !HasCustomInstComment && (Imm > 255 || Imm < -256)) {
      // Don't print unnecessary hex sign bits. 
      if (Imm == (int16_t)(Imm))
        *CommentStream << format("imm = 0x%" PRIX16 "\n", (uint16_t)Imm);
      else if (Imm == (int32_t)(Imm))
        *CommentStream << format("imm = 0x%" PRIX32 "\n", (uint32_t)Imm);
      else
        *CommentStream << format("imm = 0x%" PRIX64 "\n", (uint64_t)Imm);
    }
  } else {
    assert(Op.isExpr() && "unknown operand kind in printOperand");
    O << markup("<imm:") << '$';
    Op.getExpr()->print(O, &MAI);
    O << markup(">");
  }
}

void X86ATTInstPrinter::printMemReference(const MCInst *MI, unsigned Op,
                                          raw_ostream &O) {
  const MCOperand &BaseReg = MI->getOperand(Op + X86::AddrBaseReg);
  const MCOperand &IndexReg = MI->getOperand(Op + X86::AddrIndexReg);
  const MCOperand &DispSpec = MI->getOperand(Op + X86::AddrDisp);
  const MCOperand &SegReg = MI->getOperand(Op + X86::AddrSegmentReg);

  O << markup("<mem:");

  // If this has a segment register, print it.
  if (SegReg.getReg()) {
    printOperand(MI, Op + X86::AddrSegmentReg, O);
    O << ':';
  }

  if (DispSpec.isImm()) {
    int64_t DispVal = DispSpec.getImm();
    if (DispVal || (!IndexReg.getReg() && !BaseReg.getReg()))
      O << formatImm(DispVal);
  } else {
    assert(DispSpec.isExpr() && "non-immediate displacement for LEA?");
    DispSpec.getExpr()->print(O, &MAI);
  }

  if (IndexReg.getReg() || BaseReg.getReg()) {
    O << '(';
    if (BaseReg.getReg())
      printOperand(MI, Op + X86::AddrBaseReg, O);

    if (IndexReg.getReg()) {
      O << ',';
      printOperand(MI, Op + X86::AddrIndexReg, O);
      unsigned ScaleVal = MI->getOperand(Op + X86::AddrScaleAmt).getImm();
      if (ScaleVal != 1) {
        O << ',' << markup("<imm:") << ScaleVal // never printed in hex.
          << markup(">");
      }
    }
    O << ')';
  }

  O << markup(">");
}

void X86ATTInstPrinter::printSrcIdx(const MCInst *MI, unsigned Op,
                                    raw_ostream &O) {
  const MCOperand &SegReg = MI->getOperand(Op + 1);

  O << markup("<mem:");

  // If this has a segment register, print it.
  if (SegReg.getReg()) {
    printOperand(MI, Op + 1, O);
    O << ':';
  }

  O << "(";
  printOperand(MI, Op, O);
  O << ")";

  O << markup(">");
}

void X86ATTInstPrinter::printDstIdx(const MCInst *MI, unsigned Op,
                                    raw_ostream &O) {
  O << markup("<mem:");

  O << "%es:(";
  printOperand(MI, Op, O);
  O << ")";

  O << markup(">");
}

void X86ATTInstPrinter::printMemOffset(const MCInst *MI, unsigned Op,
                                       raw_ostream &O) {
  const MCOperand &DispSpec = MI->getOperand(Op);
  const MCOperand &SegReg = MI->getOperand(Op + 1);

  O << markup("<mem:");

  // If this has a segment register, print it.
  if (SegReg.getReg()) {
    printOperand(MI, Op + 1, O);
    O << ':';
  }

  if (DispSpec.isImm()) {
    O << formatImm(DispSpec.getImm());
  } else {
    assert(DispSpec.isExpr() && "non-immediate displacement?");
    DispSpec.getExpr()->print(O, &MAI);
  }

  O << markup(">");
}

void X86ATTInstPrinter::printU8Imm(const MCInst *MI, unsigned Op,
                                   raw_ostream &O) {
  if (MI->getOperand(Op).isExpr())
    return printOperand(MI, Op, O);

  O << markup("<imm:") << '$' << formatImm(MI->getOperand(Op).getImm() & 0xff)
    << markup(">");
}
