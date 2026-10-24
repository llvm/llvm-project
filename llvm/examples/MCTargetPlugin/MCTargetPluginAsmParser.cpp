//===-- MCTargetPluginAsmParser.cpp - Asm parser for the example target --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The "mcplugin" assembly parser. Real targets have TableGen write the matcher
// for them; with a single operand-less mnemonic to recognize, this one is
// written by hand.
//
//===----------------------------------------------------------------------===//

#include "MCTargetPlugin.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

namespace {

class MCPluginAsmParser : public MCTargetAsmParser {
public:
  MCPluginAsmParser(const MCSubtargetInfo &STI, MCAsmParser &Parser,
                    const MCInstrInfo &MII)
      : MCTargetAsmParser(STI, MII) {}

  bool parseRegister(MCRegister &Reg, SMLoc &StartLoc,
                     SMLoc &EndLoc) override {
    return true;
  }

  ParseStatus tryParseRegister(MCRegister &Reg, SMLoc &StartLoc,
                               SMLoc &EndLoc) override {
    return ParseStatus::NoMatch;
  }

  bool parseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) override {
    if (Name != "nop")
      return Error(NameLoc, "unknown instruction '" + Name + "'");

    return parseEOL();
  }

  bool matchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               OperandVector &Operands, MCStreamer &Out,
                               uint64_t &ErrorInfo,
                               bool MatchingInlineAsm) override {
    MCInst Inst;
    Inst.setOpcode(0);
    Inst.setLoc(IDLoc);
    Out.emitInstruction(Inst, getSTI());
    return false;
  }

  void convertToMapAndConstraints(unsigned Kind,
                                  const OperandVector &Operands) override {}
};

} // namespace

static MCTargetAsmParser *createMCPluginAsmParser(const MCSubtargetInfo &STI,
                                                  MCAsmParser &Parser,
                                                  const MCInstrInfo &MII) {
  return new MCPluginAsmParser(STI, Parser, MII);
}

static struct RegisterMCPluginAsmParser {
  RegisterMCPluginAsmParser() {
    TargetRegistry::RegisterMCAsmParser(getTheMCPluginTarget(),
                                        createMCPluginAsmParser);
  }
} Registration;
