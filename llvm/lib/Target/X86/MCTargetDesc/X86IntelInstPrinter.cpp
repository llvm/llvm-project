//===-- X86IntelInstPrinter.cpp - Intel assembly instruction printing -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file includes code for rendering MCInst instances as Intel-style
// assembly.
//
//===----------------------------------------------------------------------===//

#include "X86IntelInstPrinter.h"
#include "X86BaseInfo.h"
#include "X86InstComments.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <cstdint>

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

// Include the auto-generated portion of the assembly writer.
#define PRINT_ALIAS_INSTR
#include "X86GenAsmWriter1.inc"

void X86IntelInstPrinter::printRegName(raw_ostream &OS, MCRegister Reg) {
  markup(OS, Markup::Register) << getRegisterName(Reg);
}

void X86IntelInstPrinter::printInst(const MCInst *MI, uint64_t Address,
                                    StringRef Annot, const MCSubtargetInfo &STI,
                                    raw_ostream &OS) {
  printInstFlags(MI, OS, STI);

  // In 16-bit mode, print data16 as data32.
  if (MI->getOpcode() == X86::DATA16_PREFIX &&
      STI.hasFeature(X86::Is16Bit)) {
    OS << "\tdata32";
  } else if (!printAliasInstr(MI, Address, OS) && !printVecCompareInstr(MI, OS))
    printInstruction(MI, Address, OS);

  // Next always print the annotation.
  printAnnotation(OS, Annot);

  // If verbose assembly is enabled, we can print some informative comments.
  if (CommentStream)
    EmitAnyX86InstComments(MI, *CommentStream, MII);
}

bool X86IntelInstPrinter::printVecCompareInstr(const MCInst *MI, raw_ostream &OS) {
  if (MI->getNumOperands() == 0 ||
      !MI->getOperand(MI->getNumOperands() - 1).isImm())
    return false;

  int64_t Imm = MI->getOperand(MI->getNumOperands() - 1).getImm();

  const MCInstrDesc &Desc = MII.get(MI->getOpcode());

  // Custom print the vector compare instructions to get the immediate
  // translated into the mnemonic.
  switch (MI->getOpcode()) {
  case X86::CMPPDrmi:     case X86::CMPPDrri:
  case X86::CMPPSrmi:     case X86::CMPPSrri:
  case X86::CMPSDrmi:     case X86::CMPSDrri:
  case X86::CMPSDrmi_Int: case X86::CMPSDrri_Int:
  case X86::CMPSSrmi:     case X86::CMPSSrri:
  case X86::CMPSSrmi_Int: case X86::CMPSSrri_Int:
    if (Imm >= 0 && Imm <= 7) {
      OS << '\t';
      printCMPMnemonic(MI, /*IsVCMP*/false, OS);
      printOperand(MI, 0, OS);
      OS << ", ";
      // Skip operand 1 as its tied to the dest.

      if ((Desc.TSFlags & X86II::FormMask) == X86II::MRMSrcMem) {
        if ((Desc.TSFlags & X86II::OpPrefixMask) == X86II::XS)
          printdwordmem(MI, 2, OS);
        else if ((Desc.TSFlags & X86II::OpPrefixMask) == X86II::XD)
          printqwordmem(MI, 2, OS);
        else
          printxmmwordmem(MI, 2, OS);
      } else
        printOperand(MI, 2, OS);

      return true;
    }
    break;

  case X86::VCMPPDrmi:       case X86::VCMPPDrri:
  case X86::VCMPPDYrmi:      case X86::VCMPPDYrri:
  case X86::VCMPPDZ128rmi:   case X86::VCMPPDZ128rri:
  case X86::VCMPPDZ256rmi:   case X86::VCMPPDZ256rri:
  case X86::VCMPPDZrmi:      case X86::VCMPPDZrri:
  case X86::VCMPPSrmi:       case X86::VCMPPSrri:
  case X86::VCMPPSYrmi:      case X86::VCMPPSYrri:
  case X86::VCMPPSZ128rmi:   case X86::VCMPPSZ128rri:
  case X86::VCMPPSZ256rmi:   case X86::VCMPPSZ256rri:
  case X86::VCMPPSZrmi:      case X86::VCMPPSZrri:
  case X86::VCMPSDrmi:       case X86::VCMPSDrri:
  case X86::VCMPSDZrmi:      case X86::VCMPSDZrri:
  case X86::VCMPSDrmi_Int:   case X86::VCMPSDrri_Int:
  case X86::VCMPSDZrmi_Int:  case X86::VCMPSDZrri_Int:
  case X86::VCMPSSrmi:       case X86::VCMPSSrri:
  case X86::VCMPSSZrmi:      case X86::VCMPSSZrri:
  case X86::VCMPSSrmi_Int:   case X86::VCMPSSrri_Int:
  case X86::VCMPSSZrmi_Int:  case X86::VCMPSSZrri_Int:
  case X86::VCMPPDZ128rmik:  case X86::VCMPPDZ128rrik:
  case X86::VCMPPDZ256rmik:  case X86::VCMPPDZ256rrik:
  case X86::VCMPPDZrmik:     case X86::VCMPPDZrrik:
  case X86::VCMPPSZ128rmik:  case X86::VCMPPSZ128rrik:
  case X86::VCMPPSZ256rmik:  case X86::VCMPPSZ256rrik:
  case X86::VCMPPSZrmik:     case X86::VCMPPSZrrik:
  case X86::VCMPSDZrmik_Int: case X86::VCMPSDZrrik_Int:
  case X86::VCMPSSZrmik_Int: case X86::VCMPSSZrrik_Int:
  case X86::VCMPPDZ128rmbi:  case X86::VCMPPDZ128rmbik:
  case X86::VCMPPDZ256rmbi:  case X86::VCMPPDZ256rmbik:
  case X86::VCMPPDZrmbi:     case X86::VCMPPDZrmbik:
  case X86::VCMPPSZ128rmbi:  case X86::VCMPPSZ128rmbik:
  case X86::VCMPPSZ256rmbi:  case X86::VCMPPSZ256rmbik:
  case X86::VCMPPSZrmbi:     case X86::VCMPPSZrmbik:
  case X86::VCMPPDZrrib:     case X86::VCMPPDZrribk:
  case X86::VCMPPSZrrib:     case X86::VCMPPSZrribk:
  case X86::VCMPSDZrrib_Int: case X86::VCMPSDZrribk_Int:
  case X86::VCMPSSZrrib_Int: case X86::VCMPSSZrribk_Int:
  case X86::VCMPPHZ128rmi:   case X86::VCMPPHZ128rri:
  case X86::VCMPPHZ256rmi:   case X86::VCMPPHZ256rri:
  case X86::VCMPPHZrmi:      case X86::VCMPPHZrri:
  case X86::VCMPSHZrmi:      case X86::VCMPSHZrri:
  case X86::VCMPSHZrmi_Int:  case X86::VCMPSHZrri_Int:
  case X86::VCMPPHZ128rmik:  case X86::VCMPPHZ128rrik:
  case X86::VCMPPHZ256rmik:  case X86::VCMPPHZ256rrik:
  case X86::VCMPPHZrmik:     case X86::VCMPPHZrrik:
  case X86::VCMPSHZrmik_Int: case X86::VCMPSHZrrik_Int:
  case X86::VCMPPHZ128rmbi:  case X86::VCMPPHZ128rmbik:
  case X86::VCMPPHZ256rmbi:  case X86::VCMPPHZ256rmbik:
  case X86::VCMPPHZrmbi:     case X86::VCMPPHZrmbik:
  case X86::VCMPPHZrrib:     case X86::VCMPPHZrribk:
  case X86::VCMPSHZrrib_Int: case X86::VCMPSHZrribk_Int:
  case X86::VCMPBF16Z128rmi:  case X86::VCMPBF16Z128rri:
  case X86::VCMPBF16Z256rmi:  case X86::VCMPBF16Z256rri:
  case X86::VCMPBF16Zrmi:     case X86::VCMPBF16Zrri:
  case X86::VCMPBF16Z128rmik: case X86::VCMPBF16Z128rrik:
  case X86::VCMPBF16Z256rmik: case X86::VCMPBF16Z256rrik:
  case X86::VCMPBF16Zrmik:    case X86::VCMPBF16Zrrik:
  case X86::VCMPBF16Z128rmbi: case X86::VCMPBF16Z128rmbik:
  case X86::VCMPBF16Z256rmbi: case X86::VCMPBF16Z256rmbik:
  case X86::VCMPBF16Zrmbi:    case X86::VCMPBF16Zrmbik:
    if (Imm >= 0 && Imm <= 31) {
      OS << '\t';
      printCMPMnemonic(MI, /*IsVCMP*/true, OS);

      unsigned CurOp = 0;
      printOperand(MI, CurOp++, OS);

      if (Desc.TSFlags & X86II::EVEX_K) {
        // Print mask operand.
        OS << " {";
        printOperand(MI, CurOp++, OS);
        OS << "}";
      }
      OS << ", ";
      printOperand(MI, CurOp++, OS);
      OS << ", ";

      if ((Desc.TSFlags & X86II::FormMask) == X86II::MRMSrcMem) {
        if (Desc.TSFlags & X86II::EVEX_B) {
          // Broadcast form.
          // Load size is word for TA map. Otherwise it is based on W-bit.
          if ((Desc.TSFlags & X86II::OpMapMask) == X86II::TA) {
            assert(!(Desc.TSFlags & X86II::REX_W) && "Unknown W-bit value!");
            printwordmem(MI, CurOp++, OS);
          } else if (Desc.TSFlags & X86II::REX_W) {
            printqwordmem(MI, CurOp++, OS);
          } else {
            printdwordmem(MI, CurOp++, OS);
          }

          // Print the number of elements broadcasted.
          unsigned NumElts;
          if (Desc.TSFlags & X86II::EVEX_L2)
            NumElts = (Desc.TSFlags & X86II::REX_W) ? 8 : 16;
          else if (Desc.TSFlags & X86II::VEX_L)
            NumElts = (Desc.TSFlags & X86II::REX_W) ? 4 : 8;
          else
            NumElts = (Desc.TSFlags & X86II::REX_W) ? 2 : 4;
          if ((Desc.TSFlags & X86II::OpMapMask) == X86II::TA) {
            assert(!(Desc.TSFlags & X86II::REX_W) && "Unknown W-bit value!");
            NumElts *= 2;
          }
          OS << "{1to" << NumElts << "}";
        } else {
          if ((Desc.TSFlags & X86II::OpPrefixMask) == X86II::XS) {
            if ((Desc.TSFlags & X86II::OpMapMask) == X86II::TA)
              printwordmem(MI, CurOp++, OS);
            else
              printdwordmem(MI, CurOp++, OS);
          } else if ((Desc.TSFlags & X86II::OpPrefixMask) == X86II::XD &&
                     (Desc.TSFlags & X86II::OpMapMask) != X86II::TA) {
            printqwordmem(MI, CurOp++, OS);
          } else if (Desc.TSFlags & X86II::EVEX_L2) {
            printzmmwordmem(MI, CurOp++, OS);
          } else if (Desc.TSFlags & X86II::VEX_L) {
            printymmwordmem(MI, CurOp++, OS);
          } else {
            printxmmwordmem(MI, CurOp++, OS);
          }
        }
      } else {
        printOperand(MI, CurOp++, OS);
        if (Desc.TSFlags & X86II::EVEX_B)
          OS << ", {sae}";
      }

      return true;
    }
    break;

  case X86::VPCOMBmi:  case X86::VPCOMBri:
  case X86::VPCOMDmi:  case X86::VPCOMDri:
  case X86::VPCOMQmi:  case X86::VPCOMQri:
  case X86::VPCOMUBmi: case X86::VPCOMUBri:
  case X86::VPCOMUDmi: case X86::VPCOMUDri:
  case X86::VPCOMUQmi: case X86::VPCOMUQri:
  case X86::VPCOMUWmi: case X86::VPCOMUWri:
  case X86::VPCOMWmi:  case X86::VPCOMWri:
    if (Imm >= 0 && Imm <= 7) {
      OS << '\t';
      printVPCOMMnemonic(MI, OS);
      printOperand(MI, 0, OS);
      OS << ", ";
      printOperand(MI, 1, OS);
      OS << ", ";
      if ((Desc.TSFlags & X86II::FormMask) == X86II::MRMSrcMem)
        printxmmwordmem(MI, 2, OS);
      else
        printOperand(MI, 2, OS);
      return true;
    }
    break;

  case X86::VPCMPBZ128rmi:   case X86::VPCMPBZ128rri:
  case X86::VPCMPBZ256rmi:   case X86::VPCMPBZ256rri:
  case X86::VPCMPBZrmi:      case X86::VPCMPBZrri:
  case X86::VPCMPDZ128rmi:   case X86::VPCMPDZ128rri:
  case X86::VPCMPDZ256rmi:   case X86::VPCMPDZ256rri:
  case X86::VPCMPDZrmi:      case X86::VPCMPDZrri:
  case X86::VPCMPQZ128rmi:   case X86::VPCMPQZ128rri:
  case X86::VPCMPQZ256rmi:   case X86::VPCMPQZ256rri:
  case X86::VPCMPQZrmi:      case X86::VPCMPQZrri:
  case X86::VPCMPUBZ128rmi:  case X86::VPCMPUBZ128rri:
  case X86::VPCMPUBZ256rmi:  case X86::VPCMPUBZ256rri:
  case X86::VPCMPUBZrmi:     case X86::VPCMPUBZrri:
  case X86::VPCMPUDZ128rmi:  case X86::VPCMPUDZ128rri:
  case X86::VPCMPUDZ256rmi:  case X86::VPCMPUDZ256rri:
  case X86::VPCMPUDZrmi:     case X86::VPCMPUDZrri:
  case X86::VPCMPUQZ128rmi:  case X86::VPCMPUQZ128rri:
  case X86::VPCMPUQZ256rmi:  case X86::VPCMPUQZ256rri:
  case X86::VPCMPUQZrmi:     case X86::VPCMPUQZrri:
  case X86::VPCMPUWZ128rmi:  case X86::VPCMPUWZ128rri:
  case X86::VPCMPUWZ256rmi:  case X86::VPCMPUWZ256rri:
  case X86::VPCMPUWZrmi:     case X86::VPCMPUWZrri:
  case X86::VPCMPWZ128rmi:   case X86::VPCMPWZ128rri:
  case X86::VPCMPWZ256rmi:   case X86::VPCMPWZ256rri:
  case X86::VPCMPWZrmi:      case X86::VPCMPWZrri:
  case X86::VPCMPBZ128rmik:  case X86::VPCMPBZ128rrik:
  case X86::VPCMPBZ256rmik:  case X86::VPCMPBZ256rrik:
  case X86::VPCMPBZrmik:     case X86::VPCMPBZrrik:
  case X86::VPCMPDZ128rmik:  case X86::VPCMPDZ128rrik:
  case X86::VPCMPDZ256rmik:  case X86::VPCMPDZ256rrik:
  case X86::VPCMPDZrmik:     case X86::VPCMPDZrrik:
  case X86::VPCMPQZ128rmik:  case X86::VPCMPQZ128rrik:
  case X86::VPCMPQZ256rmik:  case X86::VPCMPQZ256rrik:
  case X86::VPCMPQZrmik:     case X86::VPCMPQZrrik:
  case X86::VPCMPUBZ128rmik: case X86::VPCMPUBZ128rrik:
  case X86::VPCMPUBZ256rmik: case X86::VPCMPUBZ256rrik:
  case X86::VPCMPUBZrmik:    case X86::VPCMPUBZrrik:
  case X86::VPCMPUDZ128rmik: case X86::VPCMPUDZ128rrik:
  case X86::VPCMPUDZ256rmik: case X86::VPCMPUDZ256rrik:
  case X86::VPCMPUDZrmik:    case X86::VPCMPUDZrrik:
  case X86::VPCMPUQZ128rmik: case X86::VPCMPUQZ128rrik:
  case X86::VPCMPUQZ256rmik: case X86::VPCMPUQZ256rrik:
  case X86::VPCMPUQZrmik:    case X86::VPCMPUQZrrik:
  case X86::VPCMPUWZ128rmik: case X86::VPCMPUWZ128rrik:
  case X86::VPCMPUWZ256rmik: case X86::VPCMPUWZ256rrik:
  case X86::VPCMPUWZrmik:    case X86::VPCMPUWZrrik:
  case X86::VPCMPWZ128rmik:  case X86::VPCMPWZ128rrik:
  case X86::VPCMPWZ256rmik:  case X86::VPCMPWZ256rrik:
  case X86::VPCMPWZrmik:     case X86::VPCMPWZrrik:
  case X86::VPCMPDZ128rmbi:  case X86::VPCMPDZ128rmbik:
  case X86::VPCMPDZ256rmbi:  case X86::VPCMPDZ256rmbik:
  case X86::VPCMPDZrmbi:     case X86::VPCMPDZrmbik:
  case X86::VPCMPQZ128rmbi:  case X86::VPCMPQZ128rmbik:
  case X86::VPCMPQZ256rmbi:  case X86::VPCMPQZ256rmbik:
  case X86::VPCMPQZrmbi:     case X86::VPCMPQZrmbik:
  case X86::VPCMPUDZ128rmbi: case X86::VPCMPUDZ128rmbik:
  case X86::VPCMPUDZ256rmbi: case X86::VPCMPUDZ256rmbik:
  case X86::VPCMPUDZrmbi:    case X86::VPCMPUDZrmbik:
  case X86::VPCMPUQZ128rmbi: case X86::VPCMPUQZ128rmbik:
  case X86::VPCMPUQZ256rmbi: case X86::VPCMPUQZ256rmbik:
  case X86::VPCMPUQZrmbi:    case X86::VPCMPUQZrmbik:
    if ((Imm >= 0 && Imm <= 2) || (Imm >= 4 && Imm <= 6)) {
      OS << '\t';
      printVPCMPMnemonic(MI, OS);

      unsigned CurOp = 0;
      printOperand(MI, CurOp++, OS);

      if (Desc.TSFlags & X86II::EVEX_K) {
        // Print mask operand.
        OS << " {";
        printOperand(MI, CurOp++, OS);
        OS << "}";
      }
      OS << ", ";
      printOperand(MI, CurOp++, OS);
      OS << ", ";

      if ((Desc.TSFlags & X86II::FormMask) == X86II::MRMSrcMem) {
        if (Desc.TSFlags & X86II::EVEX_B) {
          // Broadcast form.
          // Load size is based on W-bit as only D and Q are supported.
          if (Desc.TSFlags & X86II::REX_W)
            printqwordmem(MI, CurOp++, OS);
          else
            printdwordmem(MI, CurOp++, OS);

          // Print the number of elements broadcasted.
          unsigned NumElts;
          if (Desc.TSFlags & X86II::EVEX_L2)
            NumElts = (Desc.TSFlags & X86II::REX_W) ? 8 : 16;
          else if (Desc.TSFlags & X86II::VEX_L)
            NumElts = (Desc.TSFlags & X86II::REX_W) ? 4 : 8;
          else
            NumElts = (Desc.TSFlags & X86II::REX_W) ? 2 : 4;
          OS << "{1to" << NumElts << "}";
        } else {
          if (Desc.TSFlags & X86II::EVEX_L2)
            printzmmwordmem(MI, CurOp++, OS);
          else if (Desc.TSFlags & X86II::VEX_L)
            printymmwordmem(MI, CurOp++, OS);
          else
            printxmmwordmem(MI, CurOp++, OS);
        }
      } else {
        printOperand(MI, CurOp++, OS);
      }

      return true;
    }
    break;
  }

  return false;
}

void X86IntelInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                       raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    printRegName(O, Op.getReg());
  } else if (Op.isImm()) {
    markup(O, Markup::Immediate) << formatImm((int64_t)Op.getImm());
  } else {
    assert(Op.isExpr() && "unknown operand kind in printOperand");
    O << "offset ";
    MAI.printExpr(O, *Op.getExpr());
  }
}

void X86IntelInstPrinter::printMemReference(const MCInst *MI, unsigned Op,
                                            raw_ostream &O) {
  // Do not print the exact form of the memory operand if it references a known
  // binary object.
  if (SymbolizeOperands && MIA) {
    uint64_t Target;
    if (MIA->evaluateBranch(*MI, 0, 0, Target))
      return;
    if (MIA->evaluateMemoryOperandAddress(*MI, /*STI=*/nullptr, 0, 0))
      return;
  }
  const MCOperand &BaseReg  = MI->getOperand(Op+X86::AddrBaseReg);
  unsigned ScaleVal         = MI->getOperand(Op+X86::AddrScaleAmt).getImm();
  const MCOperand &IndexReg = MI->getOperand(Op+X86::AddrIndexReg);
  const MCOperand &DispSpec = MI->getOperand(Op+X86::AddrDisp);

  // If this has a segment register, print it.
  printOptionalSegReg(MI, Op + X86::AddrSegmentReg, O);

  WithMarkup M = markup(O, Markup::Memory);
  O << '[';

  bool NeedPlus = false;
  if (BaseReg.getReg()) {
    printOperand(MI, Op+X86::AddrBaseReg, O);
    NeedPlus = true;
  }

  if (IndexReg.getReg()) {
    if (NeedPlus) O << " + ";
    if (ScaleVal != 1 || !BaseReg.getReg())
      O << ScaleVal << '*';
    printOperand(MI, Op+X86::AddrIndexReg, O);
    NeedPlus = true;
  }

  if (!DispSpec.isImm()) {
    if (NeedPlus) O << " + ";
    assert(DispSpec.isExpr() && "non-immediate displacement for LEA?");
    MAI.printExpr(O, *DispSpec.getExpr());
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
      markup(O, Markup::Immediate) << formatImm(DispVal);
    }
  }

  O << ']';
}

void X86IntelInstPrinter::printSrcIdx(const MCInst *MI, unsigned Op,
                                      raw_ostream &O) {
  // If this has a segment register, print it.
  printOptionalSegReg(MI, Op + 1, O);

  WithMarkup M = markup(O, Markup::Memory);
  O << '[';
  printOperand(MI, Op, O);
  O << ']';
}

void X86IntelInstPrinter::printDstIdx(const MCInst *MI, unsigned Op,
                                      raw_ostream &O) {
  // DI accesses are always ES-based.
  O << "es:";

  WithMarkup M = markup(O, Markup::Memory);
  O << '[';
  printOperand(MI, Op, O);
  O << ']';
}

void X86IntelInstPrinter::printMemOffset(const MCInst *MI, unsigned Op,
                                         raw_ostream &O) {
  const MCOperand &DispSpec = MI->getOperand(Op);

  // If this has a segment register, print it.
  printOptionalSegReg(MI, Op + 1, O);

  WithMarkup M = markup(O, Markup::Memory);
  O << '[';

  if (DispSpec.isImm()) {
    markup(O, Markup::Immediate) << formatImm(DispSpec.getImm());
  } else {
    assert(DispSpec.isExpr() && "non-immediate displacement?");
    MAI.printExpr(O, *DispSpec.getExpr());
  }

  O << ']';
}

void X86IntelInstPrinter::printU8Imm(const MCInst *MI, unsigned Op,
                                     raw_ostream &O) {
  if (MI->getOperand(Op).isExpr())
    return MAI.printExpr(O, *MI->getOperand(Op).getExpr());

  markup(O, Markup::Immediate) << formatImm(MI->getOperand(Op).getImm() & 0xff);
}

void X86IntelInstPrinter::printSTiRegOperand(const MCInst *MI, unsigned OpNo,
                                            raw_ostream &OS) {
  MCRegister Reg = MI->getOperand(OpNo).getReg();
  // Override the default printing to print st(0) instead st.
  if (Reg == X86::ST0)
    OS << "st(0)";
  else
    printRegName(OS, Reg);
}
