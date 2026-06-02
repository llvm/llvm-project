//===-- EZHAsmParser.cpp - Parse EZH assembly to MCInst instructions --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EZHInstrInfo.h"
#include "MCTargetDesc/EZHMCTargetDesc.h"
#include "TargetInfo/EZHTargetInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCParser/AsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/SMLoc.h"

using namespace llvm;

#define DEBUG_TYPE "ezh-asm-parser"

static MCRegister MatchRegisterName(StringRef Name);
namespace {

class EZHAsmParser : public MCTargetAsmParser {
  MCAsmParser &Parser;

  bool cvtShiftedImm(MCInst &Inst, const OperandVector &Operands);

#define GET_ASSEMBLER_HEADER
#include "EZHGenAsmMatcher.inc"

  bool matchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               OperandVector &Operands, MCStreamer &Out,
                               uint64_t &ErrorInfo,
                               bool MatchingInlineAsm) override;

  bool parseRegister(MCRegister &Reg, SMLoc &StartLoc, SMLoc &EndLoc) override;
  ParseStatus tryParseRegister(MCRegister &Reg, SMLoc &StartLoc,
                               SMLoc &EndLoc) override;

  bool parseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) override;

public:
  EZHAsmParser(const MCSubtargetInfo &STI, MCAsmParser &Parser,
               const MCInstrInfo &MII)
      : MCTargetAsmParser(STI, MII), Parser(Parser) {
    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));
  }
};

struct EZHOperand : public MCParsedAsmOperand {
  enum KindTy { Token, Register, Immediate } Kind;

  struct TokOp {
    const char *Data;
    unsigned Length;
  };

  struct RegOp {
    unsigned RegNum;
  };

  struct ImmOp {
    const MCExpr *Val;
  };

  union {
    struct TokOp Tok;
    struct RegOp Reg;
    struct ImmOp Imm;
  };

  SMLoc StartLoc, EndLoc;

public:
  EZHOperand(KindTy K) : MCParsedAsmOperand(), Kind(K) {}

  bool isToken() const override { return Kind == Token; }
  bool isReg() const override { return Kind == Register; }
  bool isImm() const override { return Kind == Immediate; }
  bool isMem() const override { return false; }

  SMLoc getStartLoc() const override { return StartLoc; }
  SMLoc getEndLoc() const override { return EndLoc; }

  MCRegister getReg() const override {
    assert(Kind == Register && "Invalid access!");
    return Reg.RegNum;
  }

  const MCExpr *getImm() const {
    assert(Kind == Immediate && "Invalid access!");
    return Imm.Val;
  }

  StringRef getToken() const {
    assert(Kind == Token && "Invalid access!");
    return StringRef(Tok.Data, Tok.Length);
  }

  void print(raw_ostream &OS) const {
    switch (Kind) {
    case Immediate:
      if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm()))
        OS << CE->getValue();
      else
        OS << "<expr>";
      break;
    case Register:
      OS << "<register " << getReg() << ">";
      break;
    case Token:
      OS << "'" << getToken() << "'";
      break;
    }
  }
  void print(raw_ostream &OS, const MCAsmInfo &MAI) const override {
    print(OS);
  }

  static std::unique_ptr<EZHOperand> createToken(StringRef Str, SMLoc S) {
    auto Op = std::make_unique<EZHOperand>(Token);
    Op->Tok.Data = Str.data();
    Op->Tok.Length = Str.size();
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static std::unique_ptr<EZHOperand> createReg(unsigned RegNum, SMLoc S,
                                               SMLoc E) {
    auto Op = std::make_unique<EZHOperand>(Register);
    Op->Reg.RegNum = RegNum;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<EZHOperand> createImm(const MCExpr *Val, SMLoc S,
                                               SMLoc E) {
    auto Op = std::make_unique<EZHOperand>(Immediate);
    Op->Imm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getReg()));
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCExpr *Expr = getImm();
    if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Expr))
      Inst.addOperand(MCOperand::createImm(CE->getValue()));
    else
      Inst.addOperand(MCOperand::createExpr(Expr));
  }

  // Custom operand matchers for EZH tablegen
  bool isbrtarget() const { return isImm(); }
  void addbrtargetOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }

  bool iscalltarget() const { return isImm(); }
  void addcalltargetOperands(MCInst &Inst, unsigned N) const {
    addImmOperands(Inst, N);
  }

  bool isshifted_imm11() const { return isImm(); }
  void addshifted_imm11Operands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands!");
    const MCExpr *Expr = getImm();
    if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Expr)) {
      int64_t Val = CE->getValue();
      uint64_t Shift = 0;
      if (Val != 0) {
        while ((Val & 1) == 0 && Shift < 31) {
          Val >>= 1;
          Shift++;
        }
      }
      Inst.addOperand(MCOperand::createImm(Val));
      Inst.addOperand(MCOperand::createImm(Shift));
    } else {
      Inst.addOperand(MCOperand::createExpr(Expr));
      Inst.addOperand(MCOperand::createImm(0));
    }
  }
};

} // end anonymous namespace

bool EZHAsmParser::cvtShiftedImm(MCInst &Inst, const OperandVector &Operands) {
  ((EZHOperand &)*Operands[1]).addRegOperands(Inst, 1);
  const MCExpr *Expr = ((EZHOperand &)*Operands[2]).getImm();
  if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Expr)) {
    int64_t Val = CE->getValue();
    uint64_t Shift = 0;
    if (Val != 0) {
      while ((Val & 1) == 0 && Shift < 31) {
        Val >>= 1;
        Shift++;
      }
    }
    Inst.addOperand(MCOperand::createImm(Val));
    Inst.addOperand(MCOperand::createImm(Shift));
  } else {
    Inst.addOperand(MCOperand::createExpr(Expr));
    Inst.addOperand(MCOperand::createImm(0));
  }
  return false;
}

bool EZHAsmParser::matchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                                           OperandVector &Operands,
                                           MCStreamer &Out, uint64_t &ErrorInfo,
                                           bool MatchingInlineAsm) {
  MCInst Inst;
  unsigned MatchResult =
      MatchInstructionImpl(Operands, Inst, ErrorInfo, MatchingInlineAsm);
  switch (MatchResult) {
  case Match_Success:
    Inst.setLoc(IDLoc);
    Out.emitInstruction(Inst, getSTI());
    return false;
  case Match_MissingFeature:
    return Error(IDLoc,
                 "instruction requires a CPU feature not currently enabled");
  case Match_InvalidOperand: {
    SMLoc ErrorLoc = IDLoc;
    if (ErrorInfo != ~0ULL) {
      if (ErrorInfo >= Operands.size())
        return Error(IDLoc, "too few operands for instruction");
      ErrorLoc = ((EZHOperand &)*Operands[ErrorInfo]).getStartLoc();
      if (ErrorLoc == SMLoc())
        ErrorLoc = IDLoc;
    }
    return Error(ErrorLoc, "invalid operand for instruction");
  }
  case Match_MnemonicFail:
    return Error(IDLoc, "invalid instruction");
  }
  llvm_unreachable("Implement any new match types added!");
}

bool EZHAsmParser::parseRegister(MCRegister &Reg, SMLoc &StartLoc,
                                 SMLoc &EndLoc) {
  auto Res = tryParseRegister(Reg, StartLoc, EndLoc);
  return !Res.isSuccess();
}

ParseStatus EZHAsmParser::tryParseRegister(MCRegister &Reg, SMLoc &StartLoc,
                                           SMLoc &EndLoc) {
  const AsmToken &Tok = Parser.getTok();
  if (Tok.isNot(AsmToken::Identifier))
    return ParseStatus::NoMatch;

  StringRef Name = Tok.getString();
  MCRegister RegNum = MatchRegisterName(Name);
  if (RegNum == 0)
    return ParseStatus::NoMatch;

  Reg = RegNum;
  StartLoc = Tok.getLoc();
  EndLoc = Tok.getEndLoc();
  Parser.Lex(); // Eat identifier token
  return ParseStatus::Success;
}

bool EZHAsmParser::parseInstruction(ParseInstructionInfo &Info, StringRef Name,
                                    SMLoc NameLoc, OperandVector &Operands) {
  Operands.push_back(EZHOperand::createToken(Name, NameLoc));

  if (getLexer().is(AsmToken::EndOfStatement))
    return false;

  // Read the first operand
  while (true) {
    SMLoc S = getLexer().getLoc();

    // Parse Register
    MCRegister RegNum;
    SMLoc StartLoc, EndLoc;
    if (tryParseRegister(RegNum, StartLoc, EndLoc).isSuccess()) {
      Operands.push_back(EZHOperand::createReg(RegNum, StartLoc, EndLoc));
    }
    // Parse Immediate
    else {
      const MCExpr *Expr;
      if (Parser.parseExpression(Expr))
        return Error(S, "unknown operand");
      Operands.push_back(EZHOperand::createImm(Expr, S, getLexer().getLoc()));
    }

    if (getLexer().is(AsmToken::EndOfStatement))
      break;

    if (getLexer().isNot(AsmToken::Comma))
      return Error(getLexer().getLoc(), "unexpected token in operand list");
    Parser.Lex(); // Eat the comma
  }

  return false;
}

// Force static initialization.
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeEZHAsmParser() {
  RegisterMCAsmParser<EZHAsmParser> X(getTheEZHTarget());
}

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#include "EZHGenAsmMatcher.inc"
