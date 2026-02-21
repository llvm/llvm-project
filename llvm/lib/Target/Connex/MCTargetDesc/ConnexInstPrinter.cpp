//===-- ConnexInstPrinter.cpp - Convert Connex MCInst to asm syntax -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class prints a Connex MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#include "ConnexInstPrinter.h"
#include "Connex.h"
#include "ConnexConfig.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;

#define DEBUG_TYPE "asm-inst-printer"

// Include the auto-generated portion of the assembly writer.
#include "ConnexGenAsmWriter.inc"

/*
Note: As of Nov 2016, the LLVM APIs allow printing customized code only
here (and NOT in ConnexAsmPrinter.cpp, which around a year ago had some APIs).
*/

void ConnexInstPrinter::printInst(const MCInst *MI, uint64_t Address,
                                  StringRef Annot, const MCSubtargetInfo &STI,
                                  raw_ostream &O) {
  LLVM_DEBUG(dbgs() << "Entered ConnexInstPrinter::printInst()...\n");
  LLVM_DEBUG(dbgs() << "printInst(): *MI = " << *MI << "\n");
  LLVM_DEBUG(dbgs() << "printInst(): MI->getOpcode() = " << MI->getOpcode()
                    << "\n");
  LLVM_DEBUG(dbgs() << "printInst(): Address = " << Address << "\n");

  /* For some reason, ConnexGenAsmWriter.inc cannot print INLINEASM from the
     MachineInstr bundles I create in ConnexInstrInfo.cpp, expandPostRAPseudo(),
     and then unpack in [Target]AsmPrinter::EmitInstruction(),
     because of this definition they have:
    static const uint32_t OpInfo0[] =
      0U,>// PHI
      0U,>// INLINEASM
    ...
    etc.
    So I handle these INLINEASMs myself here.
     TODO: maybe explain better.
  */
  if (MI->getOpcode() == 1) {
    O << "        ";
    printOperand(MI, 0, O); // getOperand(0));
    O << " // custom code in ConnexInstPrinter::printInst() for INLINEASM";
  } else {
    printInstruction(MI, Address, O);
  }

  printAnnotation(O, Annot);
}

static void printExpr(const MCExpr *Expr, raw_ostream &O) {
#ifndef NDEBUG
  const MCSymbolRefExpr *SRE;

  if (const MCBinaryExpr *BE = dyn_cast<MCBinaryExpr>(Expr))
    SRE = dyn_cast<MCSymbolRefExpr>(BE->getLHS());
  else
    SRE = dyn_cast<MCSymbolRefExpr>(Expr);
  assert(SRE && "Unexpected MCExpr type.");

  MCSymbolRefExpr::VariantKind Kind = SRE->getKind();

  // assert(Kind == MCSymbolRefExpr::VK_None);
#endif

  // O << *Expr;
}

void ConnexInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                     raw_ostream &O, const char *Modifier) {
  LLVM_DEBUG(dbgs() << "Entered ConnexInstPrinter::printOperand(OpNo = " << OpNo
                    << ")...\n");
  LLVM_DEBUG(dbgs() << "ConnexInstPrinter::printOperand(): *MI = " << *MI
                    << "\n");
  LLVM_DEBUG(
      dbgs() << "ConnexInstPrinter::printOperand(): MI->getNumOperands() = "
             << MI->getNumOperands() << "\n");

  /* Simple failback, useful just for NOP -
   * TODO: I could take care of it in printInstruction(), which calls
   *   printOperand()
   */
  if (MI->getNumOperands() <= OpNo)
    return;

  LLVM_DEBUG(
      dbgs() << "ConnexInstPrinter::printOperand(): MI->getOperand(OpNo) = "
             << MI->getOperand(OpNo) << "\n");

  assert((Modifier == 0 || Modifier[0] == 0) && "No modifiers supported");

  const MCOperand &Op = MI->getOperand(OpNo);

  if (Op.isReg()) {
    // This handles registers, such as scalar r0 or vector R(0)
    O << getRegisterName(Op.getReg());
  } else if (Op.isImm()) {
    /* Normally we do NOT get here because this case is treated in
        printUnsignedImm(). */
    LLVM_DEBUG(dbgs() << "ConnexInstPrinter::printOperand(): Op.getImm() = "
                      << Op.getImm() << "\n");
    O << (int32_t)Op.getImm();
  } else {
    assert(Op.isExpr() && "Expected an expression");
    // printExpr(Op.getExpr(), O);
    // Inspired from MCTargetDesc/BPFInstPrinter.cpp
    MAI.printExpr(O, *Op.getExpr());
  }
}

template <unsigned Bits, unsigned Offset>
void ConnexInstPrinter::printUImm(const MCInst *MI, int opNum, raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(opNum);
  if (MO.isImm()) {
    uint64_t Imm = MO.getImm();
    Imm -= Offset;
    Imm &= (1 << Bits) - 1;
    Imm += Offset;
    O << formatImm(Imm);
    return;
  }

  printOperand(MI, opNum, O);
}

void ConnexInstPrinter::printMemOperand(const MCInst *MI, int OpNo,
                                        raw_ostream &O, const char *Modifier) {
  // We arrive here for instructions like: sth 0(r12), r14

  LLVM_DEBUG(dbgs() << "Entered ConnexInstPrinter::printMemOperand()\n");

  const MCOperand &RegOp = MI->getOperand(OpNo);
  const MCOperand &OffsetOp = MI->getOperand(OpNo + 1);

  // offset
  if (OffsetOp.isImm())
    O << formatDec(OffsetOp.getImm());
  else
    assert(0 && "Expected an immediate");

  // register
  assert(RegOp.isReg() && "Register operand not a register");
  O << '(' << getRegisterName(RegOp.getReg()) << ')';
}

// Inspired from MSP430InstPrinter.h
void ConnexInstPrinter::printSrcMemOperand(const MCInst *MI, unsigned OpNo,
                                           raw_ostream &O,
                                           const char *Modifier) {
  LLVM_DEBUG(dbgs() << "Entered ConnexInstPrinter::printSrcMemOperand()\n");

  const MCOperand &Base = MI->getOperand(0);
  const MCOperand &Disp = MI->getOperand(1);

  // Print displacement first

  // If the global address expression is a part of displacement field with a
  // register base, we should not emit any prefix symbol here, e.g.
  //   mov.w &foo, r1
  // vs
  //   mov.w glb(r1), r2
  // Otherwise (!) msp430-as will silently miscompile the output :(
  if (!Base.getReg())
    O << '&';

  if (Disp.isExpr()) {
    // Inspired from latest MSP430InstPrinter.cpp
    MAI.printExpr(O, *Disp.getExpr());
  }
  else {
    assert(Disp.isImm() && "Expected immediate in displacement field");
    O << Disp.getImm();
  }

  // Print register base field
  if (Base.getReg())
    O << '(' << getRegisterName(Base.getReg()) << ')';
}

void ConnexInstPrinter::printImm64Operand(const MCInst *MI, unsigned OpNo,
                                          raw_ostream &O) {
  LLVM_DEBUG(dbgs() << "Entered ConnexInstPrinter::printImm64Operand()\n");

  const MCOperand &Op = MI->getOperand(OpNo);

  if (Op.isImm()) {
    // This is for instructions like: ld_64 r3, 4294967296
    O << (uint64_t)Op.getImm();
  } else {
    // This is for instructions like: ld_64 r1, <MCOperand Expr:(CONNEX_VL)>
    O << Op;
  }
}

void ConnexInstPrinter::printScatterGatherMemOperand(const MCInst *MI,
                                                     unsigned OpNo,
                                                     raw_ostream &O) {
  LLVM_DEBUG(
      dbgs()
      << "Entered ConnexInstPrinter::printScatterGatherMemOperand() - "
         "NOTE that we discard the BasePtr of the TableGen MemOperand\n");
  /*
  IMPORTANT: Here, for the MCInst, the parameters do NOT follow the order from
             the .td file.
    Following include/llvm/Target/TargetSelectionDAG.td we have:

      // SDTypeProfile - This profile describes the type requirements of a
      // Selection DAG node.
      class SDTypeProfile<int numresults, int numoperands,
                          list<SDTypeConstraint> constraints> {
        int NumResults = numresults;
        int NumOperands = numoperands;
        list<SDTypeConstraint> Constraints = constraints;
      }

      // So: 3 input operands, 2 results.
      //   Params are: passthru, mask, index; results are: vector of i1,
      //               vector of ptr (actual result)
      //   Params are 0, 1, 2 and results are 3, 4.
      //   Operands 0 and 1 have vector type, with same number of elements.
      //   Operands 0 and 2 have identical types.
      //   Operands 1 and 3 have identical types.
      //       --> Opnd 3 (result 0?) is i1 vector
      //   Operand 4 (result 1?) has pointer type.
      //   Operand 1 is vector type with element type of i1.
      def SDTMaskedGather: SDTypeProfile<2, 3, [       // masked gather
        SDTCisVec<0>, SDTCisVec<1>, SDTCisSameAs<0, 2>, SDTCisSameAs<1, 3>,
        SDTCisPtrTy<4>, SDTCVecEltisVT<1, i1>, SDTCisSameNumEltsAs<0, 1>
      ]>;

      def masked_gather  : SDNode<"ISD::MGATHER",  SDTMaskedGather,
                             [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
  */

  if (MI->getNumOperands() > 4) {
    // We have an MGATHER operation
    const MCOperand &res = MI->getOperand(0);
    const MCOperand &index = MI->getOperand(4);
    const MCOperand &maskIn = MI->getOperand(1);
    const MCOperand &passthru = MI->getOperand(2);
    const MCOperand &maskOut = MI->getOperand(3);

    assert(index.isReg() && "index not a register");
    assert(passthru.isReg() && "passthru not a register");

    LLVM_DEBUG(dbgs() << "MI = " << *MI << "\n index = " << index
                      << "\n maskIn (bool vector register, which we actually "
                         "do NOT use) = "
                      << maskIn << "\n passthru = " << passthru
                      << "\n maskOut = " << maskOut << "\n res = " << res
                      << "\n");

    LLVM_DEBUG(dbgs() << "\n res = " << res << "\n");

    assert(res.isReg() && "res not a register");
    O << getRegisterName(index.getReg());
  } else {
    // We have an MSCATTER operation
    const MCOperand &value = MI->getOperand(1);
    const MCOperand &maskIn = MI->getOperand(0);
    const MCOperand &mask2 = MI->getOperand(2);
    const MCOperand &index = MI->getOperand(3);

    LLVM_DEBUG(dbgs() << "MI = " << *MI << "\n value (src) = " << value
                      << "\n maskIn (bool vector register, "
                         "which we actually do NOT use) = "
                      << maskIn << "\n index = " << index
                      << "\n mask2 = " << mask2 << "\n");
    O << getRegisterName(index.getReg());
  }

  LLVM_DEBUG(
      dbgs() << "Exiting ConnexInstPrinter::printScatterGatherMemOperand()\n");
}

// Taken from MipsInstPrinter.cpp
//  (required by ConnexGenAsmWriter.inc)
void ConnexInstPrinter::printUnsignedImm(const MCInst *MI, int opNum,
                                         raw_ostream &O) {
  char *res = NULL;

  LLVM_DEBUG(dbgs() << "Entered ConnexInstPrinter::printUnsignedImm()...\n");

  const MCOperand &MO = MI->getOperand(opNum);

  if (MO.isImm()) {
    unsigned int imm = MO.getImm();

    LLVM_DEBUG(dbgs() << "ConnexInstPrinter::printUnsignedImm(): imm = " << imm
                      << ", MI (ptr) = " << MI << ", *MI = " << *MI << "\n");

#ifdef GENERATE_ASSOCIATED_INLINEASM_FROM_LOOPVECTORIZE_PASS
    if (imm == VALUE_BOGUS_REPEAT_X_TIMES) {
      assert(0 && "This should NOT be executed since we don't "
                  "use symbolic LD_H, ST_H or REPEAT (using INLINEASMs "
                  "attached next to them) anymore");

      assert(MI->getOpcode() == Connex::REPEAT);
      /*
      res = getStringFromAssociatedInlineAsm(crtMI,
                                             const_cast<char *>("/*value*/"));
      */

      O << res;
    } else
#endif

        if (imm == CONNEX_MEM_NUM_ROWS + CONNEX_MEM_CONSTANT_OFFSET) {
      assert(0 && "This should NOT be executed since we don't "
                  "use symbolic LD_H, ST_H or REPEAT (using INLINEASMs "
                  "attached next to them) anymore");

      assert((MI->getOpcode() == Connex::LD_H) ||
             (MI->getOpcode() == Connex::ST_H));
#if 0
      res = getStringFromAssociatedInlineAsm(crtMI,
                                             const_cast<char *>("/*offset*/"));
#endif

      O << STR_LOOP_SYMBOLIC_INDEX << " + " << res;
    } else if (imm >= CONNEX_MEM_NUM_ROWS) {
      int spillRelativeOffset =
          (int)imm - CONNEX_MEM_NUM_ROWS - CONNEX_MEM_NUM_ROWS_EXTRA_FOR_SPILL;
      assert(spillRelativeOffset <= 1);
      // In few cases (Map.f16, SSD.f16) it is -1

      O << "CONNEX_MEM_SPILL_START_OFFSET";

      if (spillRelativeOffset >= 0)
        O << " + " << spillRelativeOffset;
      else
        O << " - " << -spillRelativeOffset;
    } else {
      O << imm; // (unsigned int)MO.getImm();
    }
  } else {
    printOperand(MI, opNum, O);
  }
}

// Inspired from [LLVM]/llvm/lib/Target/Mips/InstPrinter/MipsInstPrinter.h
void ConnexInstPrinter::printUnsignedImm8(const MCInst *MI, int opNum,
                                          raw_ostream &O) {
  const MCOperand &MO = MI->getOperand(opNum);

  if (MO.isImm())
    O << (unsigned short int)(unsigned char)MO.getImm();
  else
    printOperand(MI, opNum, O);
}
