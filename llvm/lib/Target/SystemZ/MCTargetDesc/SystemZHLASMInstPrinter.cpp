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
  unsigned Opcode = MI->getOpcode();
  if (Opcode == SystemZ::BRC || Opcode == SystemZ::BRCL ||
      Opcode == SystemZ::BC || Opcode == SystemZ::BCR ||
      Opcode == SystemZ::BIC) {
    uint64_t Imm = MI->getOperand(1).getImm();
    // Unsupported HLASM condition masks (which have no valid HLASM extended
    // mnemonics): "nle" (3), "nhe" (5), "lh" (6), "nlh" (9), "he" (10), "le"
    // (12).
    if (Imm == 3 || Imm == 5 || Imm == 6 || Imm == 9 || Imm == 10 ||
        Imm == 12) {
      std::string Str;
      raw_string_ostream RSO(Str);
      RSO << "\t";
      switch (Opcode) {
      case SystemZ::BRC:
        RSO << "brc";
        break;
      case SystemZ::BRCL:
        RSO << "brcl";
        break;
      case SystemZ::BC:
        RSO << "bc";
        break;
      case SystemZ::BCR:
        RSO << "bcr";
        break;
      case SystemZ::BIC:
        RSO << "bic";
        break;
      }
      RSO << "\t" << Imm << ",";
      if (Opcode == SystemZ::BC || Opcode == SystemZ::BIC)
        printBDXAddrOperand(MI, 2, RSO);
      else if (Opcode == SystemZ::BCR)
        printOperand(MI, 2, RSO);
      else // BRC or BRCL
        printPCRelOperand(MI, Address, 2, RSO);

      RSO.flush();
      if (!Str.empty() && Str.front() == '\t')
        O << " " << Str.substr(1, Str.length());
      else
        O << Str;

      printAnnotation(O, Annot);
      return;
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
