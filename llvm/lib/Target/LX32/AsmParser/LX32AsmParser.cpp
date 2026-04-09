//===-- LX32AsmParser.cpp - Parse LX32 assembly to MCInst instructions --===//
//
// Part of the LX32 Project
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "../MCTargetDesc/LX32MCTargetDesc.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCAsmParserExtension.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/ADT/StringSwitch.h"
#include "../TargetInfo/LX32TargetInfo.h"

using namespace llvm;

// Put the enums here
#define GET_REGINFO_ENUM
#include "../TableGen/LX32GenRegisterInfo.inc"
#define GET_INSTRINFO_ENUM
#include "../TableGen/LX32GenInstrInfo.inc"

namespace {

class LX32Operand : public MCParsedAsmOperand {
  enum KindTy {
    Token,
    Register,
    Immediate,
  } Kind;

  StringRef Tok;
  unsigned RegNum;
  const MCExpr *ImmVal;
  SMLoc StartLoc, EndLoc;

public:
  LX32Operand(KindTy K) : MCParsedAsmOperand(), Kind(K) {}

  bool isToken() const override { return Kind == Token; }
  bool isReg() const override { return Kind == Register; }
  bool isImm() const override { return Kind == Immediate; }
  bool isMem() const override { return false; }

  static std::unique_ptr<LX32Operand> createToken(StringRef Str, SMLoc S) {
    auto Op = std::make_unique<LX32Operand>(Token);
    Op->Tok = Str;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static std::unique_ptr<LX32Operand> createReg(unsigned RegNo, SMLoc S, SMLoc E) {
    auto Op = std::make_unique<LX32Operand>(Register);
    Op->RegNum = RegNo;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<LX32Operand> createImm(const MCExpr *Val, SMLoc S, SMLoc E) {
    auto Op = std::make_unique<LX32Operand>(Immediate);
    Op->ImmVal = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  SMLoc getStartLoc() const override { return StartLoc; }
  SMLoc getEndLoc() const override { return EndLoc; }

  MCRegister getReg() const override {
    assert(Kind == Register && "Invalid access!");
    return RegNum;
  }

  const MCExpr *getImm() const {
    assert(Kind == Immediate && "Invalid access!");
    return ImmVal;
  }

  StringRef getToken() const {
    assert(Kind == Token && "Invalid access!");
    return Tok;
  }

  void print(raw_ostream &OS, const MCAsmInfo &MAI) const override {
    switch (Kind) {
    case Token:     OS << "Token: " << Tok; break;
    case Register:  OS << "Reg: " << RegNum; break;
    case Immediate: OS << "Imm"; break;
    }
  }

  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getReg()));
  }

  void addExpr(MCInst &Inst, const MCExpr *Expr) const {
    if (auto *CE = dyn_cast<MCConstantExpr>(Expr))
      Inst.addOperand(MCOperand::createImm(CE->getValue()));
    else
      Inst.addOperand(MCOperand::createExpr(Expr));
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getImm());
  }
};

} // end anonymous namespace

namespace llvm {
class LX32AsmParser : public MCTargetAsmParser {
  const MCRegisterInfo *MRI;

  bool matchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               OperandVector &Operands, MCStreamer &Out,
                               uint64_t &ErrorInfo,
                               bool MatchingInlineAsm) override;

  bool parseRegister(MCRegister &Reg, SMLoc &StartLoc, SMLoc &EndLoc) override;

  ParseStatus tryParseRegister(MCRegister &Reg, SMLoc &StartLoc,
                        SMLoc &EndLoc) override;

  bool parseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) override;

  ParseStatus parseDirective(AsmToken DirectiveID) override;

  bool parseOperand(OperandVector &Operands, StringRef Name);

public:
  LX32AsmParser(const MCSubtargetInfo &STI, MCAsmParser &Parser,
                const MCInstrInfo &MII, const MCTargetOptions &Options)
      : MCTargetAsmParser(Options, STI, MII), MRI(Parser.getContext().getRegisterInfo()) {
    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));
  }

#define GET_ASSEMBLER_HEADER
#include "../TableGen/LX32GenAsmMatcher.inc"
};
} // end namespace llvm

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#include "../TableGen/LX32GenAsmMatcher.inc"

using namespace llvm;

bool LX32AsmParser::matchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                                            OperandVector &Operands,
                                            MCStreamer &Out,
                                            uint64_t &ErrorInfo,
                                            bool MatchingInlineAsm) {
  MCInst Inst;
  unsigned MatchResult = MatchInstructionImpl(Operands, Inst, ErrorInfo,
                                              MatchingInlineAsm);
  switch (MatchResult) {
  case Match_Success:
    Inst.setLoc(IDLoc);
    Out.emitInstruction(Inst, getSTI());
    return false;
  case Match_MissingFeature:
    return Error(IDLoc, "instruction requires a CPU feature not currently enabled");
  case Match_InvalidOperand:
    return Error(IDLoc, "invalid operand for instruction");
  case Match_MnemonicFail:
    return Error(IDLoc, "invalid instruction mnemonic");
  default:
    return Error(IDLoc, "unknown error matching instruction");
  }
}

bool LX32AsmParser::parseRegister(MCRegister &Reg,
                                  SMLoc &StartLoc,
                                  SMLoc &EndLoc) {
  return tryParseRegister(Reg, StartLoc, EndLoc).isSuccess() ? false : true;
}

ParseStatus LX32AsmParser::tryParseRegister(MCRegister &Reg,
                                     SMLoc &StartLoc,
                                     SMLoc &EndLoc) {
  const AsmToken &Tok = getParser().getTok();
  StartLoc = Tok.getLoc();
  EndLoc = Tok.getEndLoc();
  if (Tok.isNot(AsmToken::Identifier))
    return ParseStatus::NoMatch;

  StringRef Name = Tok.getString();
  unsigned RegNum = MatchRegisterName(Name.lower());
  if (RegNum == 0)
    return ParseStatus::NoMatch;

  Reg = RegNum;
  getParser().Lex(); // consume the identifier
  return ParseStatus::Success;
}

bool LX32AsmParser::parseOperand(OperandVector &Operands,
                                 StringRef Mnemonic) {
  SMLoc S = getTok().getLoc();
  if (getLexer().is(AsmToken::LParen) || getLexer().is(AsmToken::RParen)) {
    Operands.push_back(LX32Operand::createToken(getTok().getString(), S));
    getLexer().Lex();
    return false;
  }

  MCRegister Reg;
  if (tryParseRegister(Reg, S, S).isSuccess()) {
    Operands.push_back(LX32Operand::createReg(Reg, S, getTok().getLoc()));
    return false;
  }

  // Handle immediate
  if (getLexer().is(AsmToken::Integer) || getLexer().is(AsmToken::Minus) ||
      getLexer().is(AsmToken::Identifier)) {
    const MCExpr *IdVal;
    if (getParser().parseExpression(IdVal))
      return true;

    // Special handling for memory %lo / %hi wrappers could be here if LX32 uses them.
    // For MVP, just accept basic expressions.
    Operands.push_back(LX32Operand::createImm(IdVal, S, getTok().getLoc()));
    return false;
  }

  // Not an operand we know how to parse.
  return true;
}

bool LX32AsmParser::parseInstruction(ParseInstructionInfo &Info, StringRef Name,
                                     SMLoc NameLoc, OperandVector &Operands) {
  // First operand is token for instruction
  Operands.push_back(LX32Operand::createToken(Name, NameLoc));

  if (getLexer().is(AsmToken::EndOfStatement))
    return false;

  // Parse operands
  while (true) {
    if (parseOperand(Operands, Name)) {
      return Error(getTok().getLoc(), "unexpected token in operand");
    }

    if (getLexer().is(AsmToken::EndOfStatement))
      break;

    if (getLexer().is(AsmToken::Comma)) {
      getLexer().Lex(); // Consume comma
    } else if (getLexer().is(AsmToken::LParen) || getLexer().is(AsmToken::RParen)) {
      // Handled in next loop iteration
    } else if (Operands.size() > 1 && static_cast<LX32Operand*>(Operands.back().get())->isToken() &&
              (static_cast<LX32Operand*>(Operands.back().get())->getToken() == "(" ||
               static_cast<LX32Operand*>(Operands.back().get())->getToken() == ")")) {
      // Previous token was a parenthesis, no comma required before next operand
    } else if (Operands.size() > 2 && static_cast<LX32Operand*>(Operands[Operands.size()-2].get())->isToken() &&
               static_cast<LX32Operand*>(Operands[Operands.size()-2].get())->getToken() == "(" &&
               static_cast<LX32Operand*>(Operands.back().get())->isReg()) {
      // Inside parenthesis, parsed a register, RParen comes next
    } else {
      return Error(getTok().getLoc(), "unexpected token in operand list");
    }
  }

  return false;
}

ParseStatus LX32AsmParser::parseDirective(AsmToken DirectiveID) {
  return ParseStatus::NoMatch; // Use default parser for directives
}

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeLX32AsmParser() {
  TargetRegistry::RegisterMCAsmParser(getTheLX32TargetInfo(),
                                      [](const MCSubtargetInfo &STI,
                                         MCAsmParser &Parser,
                                         const MCInstrInfo &MII,
                                         const MCTargetOptions &Options) -> MCTargetAsmParser * {
                                        return new LX32AsmParser(STI, Parser, MII, Options);
                                      });
}
