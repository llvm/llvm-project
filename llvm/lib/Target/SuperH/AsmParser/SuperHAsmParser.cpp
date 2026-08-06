//===-- SuperHAsmParser.cpp - Parse SH assembly to MCInst instructions ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/SuperHMCAsmInfo.h"
#include "MCTargetDesc/SuperHMCTargetDesc.h"
#include "SuperHRegisterInfo.h"
#include "TargetInfo/SuperHTargetInfo.h"
#include "llvm/Analysis/Utils/TrainingLogger.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegister.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/MC/MCAsmMacro.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCParser/AsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/DebugLog.h"
#include "iostream"
#include <cstddef>
#include <sstream>
#include <system_error>

using namespace llvm;

// The generated AsmMatcher SparcGenAsmMatcher uses "SuperH" as the target
// namespace. But SPARC backend uses "SH" as its namespace.
namespace llvm {
namespace SuperH {

    using namespace SH;

} // end namespace SuperH
} // end namespace llvm

namespace {
class SuperHOperand;

// Helper that gets a string from a SMLoc pair.
StringRef StrFromLoc(SMLoc StartLoc, SMLoc EndLoc) {
  ptrdiff_t Length = (ptrdiff_t)(EndLoc.getPointer()-StartLoc.getPointer());
  return StringRef(StartLoc.getPointer(), Length);
}

class SuperHAsmParser : public MCTargetAsmParser {
  MCAsmParser &Parser;
  const MCRegisterInfo &MRI;

#define GET_ASSEMBLER_HEADER
#include "SuperHGenAsmMatcher.inc"

  bool parseRegister(MCRegister &Reg, SMLoc &StartLoc, SMLoc &EndLoc) override;
  ParseStatus tryParseRegister(MCRegister &Reg, SMLoc &StartLoc, SMLoc &EndLoc) override;
  MCRegister matchRegisterName(const AsmToken &Tok, unsigned &RegKind);
  bool parseInstruction(ParseInstructionInfo &Info, StringRef Name, SMLoc NameLoc, OperandVector &Operands) override;
  ParseStatus parseDirective(AsmToken DirectiveID) override;
  bool parseGNUAttribute(SMLoc L);
  bool matchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                                       OperandVector &Operands, MCStreamer &Out,
                                       uint64_t &ErrorInfo,
                                       bool MatchingInlineAsm) override;

  ParseStatus parseOperand(OperandVector &Operands);
  ParseStatus parseRegister(MCRegister &Reg, unsigned &RegKind, SMLoc &StartLoc, SMLoc &EndLoc);
  ParseStatus parseImm(int64_t &Imm, SMLoc &StartLoc, SMLoc &EndLoc);

public:
  SuperHAsmParser(const MCSubtargetInfo &sti, MCAsmParser &parser, const MCInstrInfo &MII) 
    : MCTargetAsmParser(sti, MII), Parser(parser),
        MRI(*Parser.getContext().getRegisterInfo()) {

    setAvailableFeatures(ComputeAvailableFeatures(getSTI().getFeatureBits()));
  }
};

} // end anonymous namespace

namespace {

class SuperHOperand : public MCParsedAsmOperand {
public:
  enum RegisterKind {
    rk_None,
    rk_GPR,
    rk_FR32,
    rk_FR64,
    rk_VEC128,
    rk_XMTRX
  };

private:

  // These are set up as bitflags to allow general comparisons
  // to happen with some simple bit masking.
  enum KindTy {
    k_Token             = 0x001,
    k_Register          = 0x002,
    k_Immediate         = 0x004,
    k_Displacement      = 0x008,
    k_IndirectReg       = 0x010,
    k_IndirectRegInc    = 0x020,
    k_IndirectRegDec    = 0x040,
    k_IndirectIndex     = 0x080,
    k_Expression        = 0x100
  };

  unsigned Kind;
  SMLoc StartLoc, EndLoc;

  struct Token {
    const char *Data;
    unsigned Length;
  };

  struct RegOp {
    MCRegister Reg;
    RegisterKind Kind;
  };

  struct ImmOp {
    const MCExpr *Val;
  };

  struct DispOp {
    const MCExpr *Val;
  };

  struct ExprOp {
    const MCExpr *Val;
  };

  struct MemOp {
    MCRegister Base;
    MCRegister OffsetReg;
    const MCExpr *Offset;
  };

  union {
    struct Token Tok;
    struct RegOp Reg;
    struct ImmOp Imm;
    struct DispOp Disp;
    struct MemOp Mem;
    struct ExprOp Expr;
    unsigned ASI;
    unsigned Prefetch;
  };

public:
  SuperHOperand(KindTy K) : Kind(K) {}
  SuperHOperand(unsigned K) : Kind(K) {}

  bool isToken() const override { return Kind == k_Token; }
  bool isImm() const override { return Kind == k_Immediate; }
  bool isReg() const override { return Kind == k_Register; }
  bool isDisp() const { return Kind == k_Displacement; }
  bool isMem() const override { return (Kind & (k_IndirectReg | k_IndirectRegInc | k_IndirectRegDec | k_IndirectIndex)) != 0; }
  bool isIndirectReg() const { return (Kind & (k_IndirectReg | k_IndirectRegInc | k_IndirectRegDec)) != 0; }
  bool isAnyReg() const { return isReg() || isIndirectReg(); }

  SMLoc getStartLoc() const override { return StartLoc; }
  SMLoc getEndLoc() const override { return EndLoc; }

  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getReg()));
  }

  void addMemOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getReg()));
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCExpr *Expr = getImm();
    this->addExpr(Inst, Expr);
  }

  void addDispOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCExpr *Expr = getDisp();
    this->addExpr(Inst, Expr);
  }

  void addExpr(MCInst &Inst, const MCExpr *Expr) const{
    // Add as immediate when possible.  Null MCExpr = 0.
    if (!Expr)
      Inst.addOperand(MCOperand::createImm(0));
    else if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Expr))
      Inst.addOperand(MCOperand::createImm(CE->getValue()));
    else
      Inst.addOperand(MCOperand::createExpr(Expr));
  }

  StringRef getToken() const {
    assert(Kind == k_Token && "Invalid access!");
    return StringRef(Tok.Data, Tok.Length);
  }

  MCRegister getReg() const override {
    assert(this->isAnyReg() && "Invalid access!");
    return this->isReg() ? Reg.Reg : Mem.Base;
  }

  const MCExpr *getImm() const {
    assert((Kind & k_Immediate) && "Invalid access!");
    return Imm.Val;
  }

  const MCExpr *getDisp() const {
    assert((Kind & k_Displacement) && "Invalid access!");
    return Disp.Val;
  }

  const MCExpr *getOffset() const {
    assert((Kind == k_IndirectIndex) && "Invalid access!");
    return Mem.Offset;
  }

  static std::unique_ptr<SuperHOperand> CreateToken(StringRef Str, SMLoc S) {
    auto Op = std::make_unique<SuperHOperand>(k_Token);
    Op->Tok.Data = Str.data();
    Op->Tok.Length = Str.size();
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static std::unique_ptr<SuperHOperand> CreateReg(MCRegister Reg, unsigned Kind,
                                                  SMLoc S, SMLoc E) {
    auto Op = std::make_unique<SuperHOperand>(k_Register);
    Op->Reg.Reg = Reg;
    Op->Reg.Kind = (SuperHOperand::RegisterKind)Kind;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<SuperHOperand> CreateImm(const MCExpr *Val, SMLoc S, SMLoc E) {
    auto Op = std::make_unique<SuperHOperand>(k_Immediate);
    Op->Imm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<SuperHOperand> CreateDisp(const MCExpr *Val, SMLoc S, SMLoc E) {
    auto Op = std::make_unique<SuperHOperand>(k_Displacement);
    Op->Imm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<SuperHOperand> CreateSymRef(const MCExpr *Val, SMLoc S, SMLoc E) {
    auto Op = std::make_unique<SuperHOperand>(k_Immediate | k_Displacement);
    Op->Imm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<SuperHOperand> CreateIReg(MCRegister Reg, SMLoc S, SMLoc E) {
    auto Op = std::make_unique<SuperHOperand>(k_IndirectReg);
    Op->Mem.Base = Reg;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<SuperHOperand> CreateIRegInc(MCRegister Reg, SMLoc S, SMLoc E) {
    auto Op = std::make_unique<SuperHOperand>(k_IndirectRegInc);
    Op->Mem.Base = Reg;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<SuperHOperand> CreateIRegDec(MCRegister Reg, SMLoc S, SMLoc E) {
    auto Op = std::make_unique<SuperHOperand>(k_IndirectRegDec);
    Op->Mem.Base = Reg;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<SuperHOperand> CreateIIndex(MCRegister Base, MCRegister OffsetReg, 
                                                     const MCExpr *Offset, 
                                                     SMLoc S, SMLoc E) {
    auto Op = std::make_unique<SuperHOperand>(k_IndirectIndex);
    Op->Mem.Base = Base;
    Op->Mem.OffsetReg = OffsetReg;
    Op->Mem.Offset = Offset;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<SuperHOperand> CreateExpr(const MCExpr *Expr, 
                                                   SMLoc S, SMLoc E) {
    auto Op = std::make_unique<SuperHOperand>(k_Expression);
    Op->Expr.Val = Expr;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<SuperHOperand> CreateFromExpr(const MCExpr *Expr, 
                                                   SMLoc S, SMLoc E) {
    if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Expr))
      return CreateDisp(CE, S, E);

    if (const MCSymbolRefExpr *SRE = dyn_cast<MCSymbolRefExpr>(Expr)) {
      return CreateDisp(SRE, S, E);
    }

    return CreateExpr(Expr, S, E);
  }

  void print(raw_ostream &OS, const MCAsmInfo &I) const override {

    // Tokens don't have Start+End locations.
    if (Kind == k_Token) {
      OS << StringRef(Tok.Data, Tok.Length);
      return;
    }

    // Indirect register has @ prefix.
    if (isIndirectReg())
      OS << "@";
    
    // Pre-decrement.
    if (Kind == k_IndirectRegDec)
      OS << "-";

    // Register or other symbol.
    OS << StrFromLoc(this->getStartLoc(), this->getEndLoc());

    // Indirect Post-Increment.
    if (Kind == k_IndirectRegInc)
      OS << "+";
  }
};

} // end anonymous namespace

#define GET_MATCHER_IMPLEMENTATION
#define GET_REGISTER_MATCHER
#define GET_MNEMONIC_SPELL_CHECKER
#define GET_MNEMONIC_CHECKER
#include "SuperHGenAsmMatcher.inc"

MCRegister SuperHAsmParser::matchRegisterName(const AsmToken &Tok, unsigned &RegKind) {
  RegKind = SuperHOperand::rk_None;
  if(!Tok.is(AsmToken::Identifier))
    return SH::NoRegister;

  StringRef Name = Tok.getString();
  MCRegister Reg = MatchRegisterName(Name.lower());
  if (Reg) {

    // XMTRX register.
    if (Reg == SH::XMTRX) {
      RegKind = SuperHOperand::rk_XMTRX;
      return Reg;
    }

    // General purpose register class.
    if (MRI.getRegClass(SH::GPRRegClassID).contains(Reg)) {
      RegKind = SuperHOperand::rk_GPR;
      return Reg;
    }

    // 32-bit float registers.
    if (MRI.getRegClass(SH::FR32RegClassID).contains(Reg)) {
      RegKind = SuperHOperand::rk_FR32;
      return Reg;
    }

    // 64-bit float registers.
    if (MRI.getRegClass(SH::FR64RegClassID).contains(Reg)) {
      RegKind = SuperHOperand::rk_FR64;
      return Reg;
    }

    // 128-bit vector registers.
    if (MRI.getRegClass(SH::VEC128RegClassID).contains(Reg)) {
      RegKind = SuperHOperand::rk_VEC128;
      return Reg;
    }
  }
  return SH::NoRegister;
}

bool SuperHAsmParser::parseRegister(MCRegister &Reg, SMLoc &StartLoc, SMLoc &EndLoc) {
  if (!tryParseRegister(Reg, StartLoc, EndLoc).isSuccess())
      return Error(StartLoc, "invalid register name");
  return false;
}

ParseStatus SuperHAsmParser::tryParseRegister(MCRegister &Reg, SMLoc &StartLoc, SMLoc &EndLoc) {
  unsigned RegKind;
  const AsmToken &Tok = Parser.getTok();
  StartLoc = Tok.getLoc();
  EndLoc = Tok.getEndLoc();

  // SuperH Registers are in the form of an identifier.
  Reg = SH::NoRegister;
  if (getLexer().getKind() != AsmToken::Identifier)
    return ParseStatus::NoMatch;

  // Match.
  Reg = matchRegisterName(Tok, RegKind);
  if (RegKind == SuperHOperand::rk_None)
    return ParseStatus::NoMatch;

  // Consume the register.
  Parser.Lex();
  return ParseStatus::Success;
}

ParseStatus SuperHAsmParser::parseImm(int64_t &Imm, SMLoc &StartLoc, SMLoc &EndLoc) {
  const AsmToken &Tok = Parser.getTok();
  const MCExpr *Expr;
  StartLoc = Tok.getLoc();
  EndLoc = Tok.getEndLoc();

  // Eat % and $ which are used in SuperH asm.
  if (Tok.is(AsmToken::Percent) || Tok.is(AsmToken::Dollar))
    Parser.Lex();

  if (Parser.parseExpression(Expr, EndLoc))
    return ParseStatus::Failure;

  if (Expr->evaluateAsAbsolute(Imm))
    return ParseStatus::Success;

  return ParseStatus::Failure;
}

ParseStatus SuperHAsmParser::parseRegister(MCRegister &Reg, unsigned &RegKind, SMLoc &StartLoc, SMLoc &EndLoc) {
  const AsmToken &Tok = Parser.getTok();
  StartLoc = Tok.getLoc();
  EndLoc = Tok.getEndLoc();

  Reg = matchRegisterName(Tok, RegKind);
  if (Reg) {
    Parser.Lex();
    return ParseStatus::Success;
  }
  return ParseStatus::Failure;
}

ParseStatus SuperHAsmParser::parseOperand(OperandVector &Operands) {
  const AsmToken &Tok = Parser.getTok();
  SMLoc StartLoc = getLexer().getLoc();
  SMLoc EndLoc = getLexer().getLoc();


  switch(Tok.getKind()) {
  default: {

    // Try parsing as register first.
    unsigned RegKind;
    MCRegister Reg;
    if (parseRegister(Reg, RegKind, StartLoc, EndLoc).isSuccess()) {
      Operands.push_back(SuperHOperand::CreateReg(Reg, RegKind, StartLoc, EndLoc));
      return ParseStatus::Success;
    }

    // That failed, try parsing as expression.
    const MCExpr *EVal;
    if (Parser.parseExpression(EVal, EndLoc)) 
      return ParseStatus::Failure;

    Operands.push_back(SuperHOperand::CreateFromExpr(EVal, StartLoc, EndLoc));
    return ParseStatus::Success;
  }

  // Immediates.
  case llvm::AsmToken::Hash: {
    Parser.Lex();
    int64_t Imm;
    if (parseImm(Imm, StartLoc, EndLoc).isSuccess()) {
      const MCExpr *Val = MCConstantExpr::create(Imm, getContext());
      Operands.push_back(SuperHOperand::CreateImm(Val, StartLoc, EndLoc));
      return ParseStatus::Success;
    }

    // Un-lex on error
    getLexer().UnLex(Tok);
    return ParseStatus::Failure;
  }

  // Indirect Memory Access.
  case AsmToken::At: {
    Parser.Lex();

    // Handle the different kinds of indirect memory access.
    switch(Parser.getTok().getKind()) {
    default: {
      getLexer().UnLex(Tok);
      return ParseStatus::Failure;
    }
    
    // Handle Register Indirect with Pre-Decrement.
    case AsmToken::Minus: {
      Parser.Lex();

      unsigned RegKind;
      MCRegister Reg;
      if (parseRegister(Reg, RegKind, StartLoc, EndLoc).isSuccess()) {
        Operands.push_back(SuperHOperand::CreateIRegDec(Reg, StartLoc, EndLoc));
        return ParseStatus::Success;
      }

      getLexer().UnLex(Tok);
      return ParseStatus::Failure;
    }

    // Register Indirect
    case AsmToken::Identifier: {

      unsigned RegKind;
      MCRegister Reg;
      if (parseRegister(Reg, RegKind, StartLoc, EndLoc).isSuccess()) {

        // Handle Register Indirect with Increment.
        if (Parser.getTok().is(AsmToken::Plus)) {
          Parser.Lex();

          Operands.push_back(SuperHOperand::CreateIRegInc(Reg, StartLoc, EndLoc));
          return ParseStatus::Success;
        }

        Operands.push_back(SuperHOperand::CreateIReg(Reg, StartLoc, EndLoc));
        return ParseStatus::Success;
      }

      getLexer().UnLex(Tok);
      return ParseStatus::Failure;
    }
    }

    // Un-lex on error
    getLexer().UnLex(Tok);
    return ParseStatus::Failure;
  }
  }
}

bool SuperHAsmParser::parseInstruction(ParseInstructionInfo &Info, StringRef Name, SMLoc NameLoc, OperandVector &Operands) {

  // Match mnemonic.
  bool MS = SuperHCheckMnemonic(Name, this->getAvailableFeatures(), 0);
  if (!MS) {
    return Error(NameLoc, "invalid instruction mnemonic" + 
      SuperHMnemonicSpellCheck(Name, getAvailableFeatures(), 0));
  }

  // Chomp name and add it to the operands.
  Operands.push_back(SuperHOperand::CreateToken(Name, NameLoc));
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    
    // Initial operand
    if (!parseOperand(Operands).isSuccess()) {
      SMLoc Loc = getLexer().getLoc();
      return Error(Loc, "unexpected token");
    }

    // Followup operands.
    while (getLexer().is(AsmToken::Comma)) {
      Parser.Lex();

      // Parse and remember operand.
      if (!parseOperand(Operands).isSuccess()) {
        SMLoc Loc = getLexer().getLoc();
        return Error(Loc, "unexpected token");
      }
    }
  }

  // Consume EndOfStatement
  Parser.Lex();
  return false;
}

ParseStatus SuperHAsmParser::parseDirective(AsmToken DirectiveID) {
  StringRef IDVal = DirectiveID.getIdentifier();
  if (IDVal.starts_with(".gnu_attribute"))
    parseGNUAttribute(DirectiveID.getLoc());
  else
    return ParseStatus::NoMatch;
  return ParseStatus::Success;
}

bool SuperHAsmParser::parseGNUAttribute(SMLoc L) {
  int64_t Tag;
  int64_t IntegerValue;
  if (!getParser().parseGNUAttribute(L, Tag, IntegerValue))
    return false;

  getParser().getStreamer().emitGNUAttribute(Tag, IntegerValue);

  return true;
}

bool SuperHAsmParser::matchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                                     OperandVector &Operands, MCStreamer &Out,
                                     uint64_t &ErrorInfo,
                                     bool MatchingInlineAsm) {
  MCInst Inst;
  unsigned MatchResult = MatchInstructionImpl(Operands, Inst, ErrorInfo, MatchingInlineAsm);
  switch(MatchResult) {
  case Match_Success:
    Inst.setLoc(IDLoc);
    Out.emitInstruction(Inst, getSTI());
    return false;
  case Match_MissingFeature:
    return Error(IDLoc, "instruction requires a CPU feature not currently enabled.");
  case Match_InvalidOperand: {
    SMLoc ErrorLoc = IDLoc;
    if (ErrorInfo != ~0ULL) {
      if (ErrorInfo >= Operands.size())
        return Error(IDLoc, "too few operands for instruction");

      ErrorLoc = ((SuperHOperand &)*Operands[ErrorInfo]).getStartLoc();
      if (ErrorLoc == SMLoc())
        ErrorLoc = IDLoc;
    }

    return Error(ErrorLoc, "invalid operand for instruction");
  }
  case Match_MnemonicFail:
    return Error(IDLoc, "invalid instruction mnemonic");
  }
  return false;
}


extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeSuperHAsmParser() {
  RegisterMCAsmParser<SuperHAsmParser> A(getTheSuperHTarget());
  RegisterMCAsmParser<SuperHAsmParser> B(getTheSuperHLETarget());
}