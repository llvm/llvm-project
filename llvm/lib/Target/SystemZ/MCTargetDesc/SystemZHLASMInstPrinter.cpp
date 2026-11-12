//=- SystemZHLASMInstPrinter.cpp - Convert SystemZ MCInst to HLASM assembly -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SystemZHLASMInstPrinter.h"
#include "MCTargetDesc/SystemZMCTargetDesc.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

#include "SystemZGenHLASMAsmWriter.inc"

void SystemZHLASMInstPrinter::printFormattedRegName(const MCAsmInfo *MAI,
                                                    MCRegister Reg,
                                                    raw_ostream &O) {
  const char *RegName = getRegisterName(Reg);
  // Skip register prefix so that only register number is left
  assert(isalpha(RegName[0]) && isdigit(RegName[1]));
  markup(O, Markup::Register) << (RegName + 1);
}

void SystemZHLASMInstPrinter::printInst(const MCInst *MI, uint64_t Address,
                                        StringRef Annot,
                                        const MCSubtargetInfo &STI,
                                        raw_ostream &O) {
  MCInst TempMI;
  unsigned Opcode = MI->getOpcode();
  if (Opcode == SystemZ::BRC || Opcode == SystemZ::BRCL ||
      Opcode == SystemZ::BC || Opcode == SystemZ::BCR ||
      Opcode == SystemZ::BIC) {
    uint64_t CCValid = MI->getOperand(0).getImm();
    uint64_t CCMask = MI->getOperand(1).getImm();
    // Unsupported HLASM condition masks (which have no valid HLASM extended
    // mnemonics): "nle" (3), "nhe" (5), "lh" (6), "nlh" (9), "he" (10), "le"
    // (12).
    if (CCMask == 3 || CCMask == 5 || CCMask == 6 || CCMask == 9 ||
        CCMask == 10 || CCMask == 12) {
      if ((CCValid & 1) == 0) {
        // If CC 3 cannot happen, invert the low bit to yield a mask with a
        // valid HLASM extended mnemonic (e.g. 10 ("he") -> 11 ("nl")).
        TempMI = *MI;
        TempMI.getOperand(1).setImm(CCMask ^ 1);
        MI = &TempMI;
      } else {
        // Use variant instructions that print numerical masks.
        unsigned AsmOpcode = 0;
        switch (Opcode) {
        case SystemZ::BRC:
          AsmOpcode = SystemZ::BRCAsm;
          break;
        case SystemZ::BRCL:
          AsmOpcode = SystemZ::BRCLAsm;
          break;
        case SystemZ::BC:
          AsmOpcode = SystemZ::BCAsm;
          break;
        case SystemZ::BCR:
          AsmOpcode = SystemZ::BCRAsm;
          break;
        case SystemZ::BIC:
          AsmOpcode = SystemZ::BICAsm;
          break;
        }
        TempMI.setOpcode(AsmOpcode);
        TempMI.addOperand(MCOperand::createImm(CCMask));
        for (unsigned I = 2; I < MI->getNumOperands(); ++I)
          TempMI.addOperand(MI->getOperand(I));
        MI = &TempMI;
      }
    }
  }

  std::string Str;
  raw_string_ostream RSO(Str);
  printInstruction(MI, Address, RSO);
  // Eat the first tab character and replace it with a space since it is
  // hardcoded in AsmWriterEmitter::EmitPrintInstruction
  // TODO: introduce a line prefix member to AsmWriter to avoid this problem
  if (!Str.empty() && Str.front() == '\t')
    O << " " << Str.substr(1, Str.length());
  else
    O << Str;

  printAnnotation(O, Annot);
}
