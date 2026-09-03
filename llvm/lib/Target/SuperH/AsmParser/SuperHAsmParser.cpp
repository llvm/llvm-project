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
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/DebugLog.h"
#include "iostream"
#include <cstddef>
#include <sstream>
#include <system_error>

#define DEBUG_TYPE "sh-asmparser"

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

  bool parseInstruction(ParseInstructionInfo &Info, StringRef Name, SMLoc NameLoc, OperandVector &Operands) override;
  ParseStatus parseDirective(AsmToken DirectiveID) override;
  bool parseGNUAttribute(SMLoc L);
  bool matchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                                       OperandVector &Operands, MCStreamer &Out,
                                       uint64_t &ErrorInfo,
                                       bool MatchingInlineAsm) override;

  // Register Parsing
  ParseStatus tryParseRegister(MCRegister &Reg, SMLoc &StartLoc, SMLoc &EndLoc) override;
  bool parseRegister(MCRegister &Reg, SMLoc &StartLoc, SMLoc &EndLoc) override;

  // Parser helpers.
  MCRegister matchRegisterName(const AsmToken &Tok, unsigned &RegKind);
  ParseStatus tryParseImm(int64_t &Imm, SMLoc &StartLoc, SMLoc &EndLoc);
  ParseStatus tryParseRegister(MCRegister &Reg, unsigned &RegKind, SMLoc &StartLoc, SMLoc &EndLoc);
  ParseStatus tryParseRegRelative(MCRegister &BaseReg, MCRegister &OffReg, const MCExpr *&Offset, SMLoc &StartLoc, SMLoc &EndLoc);
  ParseStatus tryParseRegIndirect(MCRegister &Reg, bool &IsInc, bool &IsDec, SMLoc &StartLoc, SMLoc &EndLoc);

  // Parser entry points
  ParseStatus parseOperand(OperandVector &Operands);
  ParseStatus parseRegister(OperandVector &Operands);
  ParseStatus parseRegisterIndirect(OperandVector &Operands);
  ParseStatus parsePCRel(OperandVector &Operands);
  ParseStatus parseImm(OperandVector &Operands);
  ParseStatus parseDisp(OperandVector &Operands);

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
    rk_XMTRX,
    rk_SYS,
    rk_CTRL,
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
  bool isIReg() const { return Kind == k_IndirectReg; }
  bool isIRegInc() const { return Kind == k_IndirectRegInc; }
  bool isIRegDec() const { return Kind == k_IndirectRegDec; }
  bool isAnyReg() const { return isReg() || isIReg() || isIRegInc() || isIRegDec(); }
  bool isPCRel() const { return Kind == k_Displacement && Mem.Base == SH::PC; }

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

  void addPCRelOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    const MCExpr *Expr = getDisp();
    this->addExpr(Inst, Expr);
  }

  void addIRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getReg()));
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
    return Mem.Offset;
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

  static std::unique_ptr<SuperHOperand> CreateDisp(MCRegister Reg, const MCExpr *Val, SMLoc S, SMLoc E) {
    auto Op = std::make_unique<SuperHOperand>(k_Displacement);
    Op->Mem.Base = Reg;
    Op->Mem.Offset = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<SuperHOperand> CreateSymRef(const MCExpr *Val, SMLoc S, SMLoc E) {
    auto Op = std::make_unique<SuperHOperand>(k_Displacement);
    Op->Mem.Base = SH::PC;
    Op->Mem.Offset = Val;
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
      return CreateDisp(SH::PC, CE, S, E);

    if (const MCSymbolRefExpr *SRE = dyn_cast<MCSymbolRefExpr>(Expr)) {
      return CreateSymRef(SRE, S, E);
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
    if (isIReg())
      OS << "@";
    
    // Pre-decrement.
    if (isIRegDec())
      OS << "-";

    // Register or other symbol.
    OS << StrFromLoc(this->getStartLoc(), this->getEndLoc());

    // Indirect Post-Increment.
    if (isIRegInc())
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
  if(Tok.isNot(AsmToken::Identifier))
    return SH::NoRegister;

  std::string Name = Tok.getString().lower();
  MCRegister Reg = MatchRegisterName(Name);
  if (MRI.getRegClass(SH::GPRRegClassID).contains(Reg)) {
    
    // General purpose register class.
    RegKind = SuperHOperand::rk_GPR;
  } else if (Reg == SH::XMTRX) {
    
    // XMTRX register.
    RegKind = SuperHOperand::rk_XMTRX;
  } else if (MRI.getRegClass(SH::SYSRegClassID).contains(Reg)) {
    
    // System register class.
    RegKind = SuperHOperand::rk_SYS;
  } else if (MRI.getRegClass(SH::CTRLRegClassID).contains(Reg)) {
    
    // Control register class.
    RegKind = SuperHOperand::rk_CTRL;
  } else if (MRI.getRegClass(SH::FR32RegClassID).contains(Reg)) {
    
    // 32-bit float registers.
    RegKind = SuperHOperand::rk_FR32;
  } else if (MRI.getRegClass(SH::FR64RegClassID).contains(Reg)) {
    
    // 64-bit float registers.
    RegKind = SuperHOperand::rk_FR64;
  } else if (MRI.getRegClass(SH::VEC128RegClassID).contains(Reg)) {
    
    // 128-bit vector registers.
    RegKind = SuperHOperand::rk_VEC128;
  }

  return Reg;
}

bool SuperHAsmParser::parseRegister(MCRegister &Reg, SMLoc &StartLoc, SMLoc &EndLoc) {
  if (!tryParseRegister(Reg, StartLoc, EndLoc).isSuccess())
      return Error(StartLoc, "invalid register name");
  return false;
}

ParseStatus SuperHAsmParser::tryParseRegister(MCRegister &Reg, SMLoc &StartLoc, SMLoc &EndLoc) {
  unsigned RegKind;
  const AsmToken Tok = Parser.getTok();
  StartLoc = Tok.getLoc();
  EndLoc = Tok.getEndLoc();

  // Match.
  Reg = matchRegisterName(Tok, RegKind);
  if (Reg == SH::NoRegister)
    return ParseStatus::NoMatch;

  // Consume the register.
  Parser.Lex();
  return ParseStatus::Success;
}




//===----------------------------------------------------------------------===//
//                            Parser Helpers
//===----------------------------------------------------------------------===//

// Try parsing a register by its name.
ParseStatus SuperHAsmParser::tryParseRegister(MCRegister &Reg, unsigned &RegKind, SMLoc &StartLoc, SMLoc &EndLoc) {
  const AsmToken Tok = Parser.getTok();
  StartLoc = Tok.getLoc();
  EndLoc = Tok.getEndLoc();

  // Match.
  Reg = matchRegisterName(Tok, RegKind);
  if (Reg == SH::NoRegister)
    return ParseStatus::NoMatch;

  // Consume the register.
  Parser.Lex();
  return ParseStatus::Success;
}

// Parses register relative addressing modes.
ParseStatus SuperHAsmParser::tryParseRegRelative(MCRegister &BaseReg, MCRegister &OffReg, 
                                                 const MCExpr *&Offset, SMLoc &StartLoc, 
                                                 SMLoc &EndLoc) {
  const AsmToken Tok = getTok();
  unsigned RegKind;

  // NOTE:  SuperH Assemblers support subtituting the normal PC-relative form
  //        with a symbol reference.
  //
  //        The following instruction sequences are equivalent:        
  //          mova .ABC, r0  
  //          mova @(.ABC,pc), r0
  //          mova @(8,pc), r0      (Where 8 is an offset from PC that refers
  //                                 to the same address as the identifier would)
  if (getTok().is(AsmToken::Identifier)) {

    // Registers are *not* identifiers.
    if (matchRegisterName(getTok(), RegKind) != SH::NoRegister) {
      return ParseStatus::NoMatch;
    }

    auto *Sym = getContext().getOrCreateSymbol(getTok().getIdentifier());
    Offset = MCSymbolRefExpr::create(Sym, getContext());
    BaseReg = SH::PC;
    Parser.Lex();
    return ParseStatus::Success;
  }

  // Parse "@(" sequence
  if (getTok().isNot(AsmToken::At) || getLexer().peekTok().isNot(AsmToken::LParen)) {
    return ParseStatus::NoMatch;
  }
  Parser.Lex();
  Parser.Lex();

  // Parse identifier
  if (getTok().is(AsmToken::Identifier)) {

    // Try parsing base-register as well as identifiers.
    if (tryParseRegister(OffReg, RegKind, StartLoc, EndLoc).isNoMatch()) {

      auto *Sym = getContext().getOrCreateSymbol(getTok().getIdentifier());
      Offset = MCSymbolRefExpr::create(Sym, getContext());
      Parser.Lex();
    }
  } else if (getTok().is(AsmToken::Integer)) {

    // Try parsing numeric displacements.
    int64_t Disp;
    ParseStatus Result = tryParseImm(Disp, StartLoc, EndLoc);
    if (!Result.isSuccess())
      return Result;

    Offset = MCConstantExpr::create(Disp, getContext());
  } else {

    return Error(getTok().getLoc(), "expected identifier, register or displacement");
  }

  // Parse seperator
  if (parseToken(AsmToken::Comma, "expected ','"))
    return ParseStatus::Failure;

  // Parse base register
  ParseStatus Result = tryParseRegister(BaseReg, RegKind, StartLoc, EndLoc);
  if (!Result.isSuccess())
    return Error(StartLoc, "expected register");

  // Whoops! missing ending parenthesis.
  if (parseToken(AsmToken::RParen, "expected ')'"))
    return ParseStatus::Failure;

  return ParseStatus::Success;
}

// Parses register-indirect addressing in the following forms:
// 
//   @Rn  - Register Indirect
//   @-Rn - Register Indirect with Pre-decrement
//   @Rn+ - Register Indirect with Post-increment
ParseStatus SuperHAsmParser::tryParseRegIndirect(MCRegister &Reg, bool &IsInc, bool &IsDec, 
                                                 SMLoc &StartLoc, SMLoc &EndLoc) {
  const AsmToken Tok = Parser.getTok();
  SMLoc S = StartLoc;
  SMLoc E = EndLoc;
  unsigned RegKind;

  if (!Parser.parseOptionalToken(AsmToken::At))
    return ParseStatus::NoMatch;

  // Parse possible pre-decement
  if (Parser.parseOptionalToken(AsmToken::Minus))
    IsDec = true;

  // Parse Register
  if (!tryParseRegister(Reg, RegKind, S, E).isSuccess()) {
    return Error(E, "expected register");
  }

  // Parse possible post-increment
  if (Parser.parseOptionalToken(AsmToken::Plus))
    IsInc = true;

  // You can't do both increment and decrement.
  // this is just a straight up syntax error.
  if (IsInc && IsDec) {
    return Error(E, "either pre-decrement or post-increment expected");
  }

  EndLoc = Parser.getTok().getLoc();
  return ParseStatus::Success;
}

ParseStatus SuperHAsmParser::tryParseImm(int64_t &Imm, SMLoc &StartLoc, SMLoc &EndLoc) {
  const AsmToken Tok = Parser.getTok();

  // Eat % and $ which are used in SuperH asm.
  if (getTok().is(AsmToken::Percent) || getTok().is(AsmToken::Dollar))
    Parser.Lex();

  if (Parser.parseAbsoluteExpression(Imm)) {
    getLexer().UnLex(Tok);
    return ParseStatus::NoMatch;
  }

  EndLoc = Parser.getTok().getLoc();
  return ParseStatus::Success;
}



//===----------------------------------------------------------------------===//
//                            Parser Entrypoints
//===----------------------------------------------------------------------===//

ParseStatus SuperHAsmParser::parseImm(OperandVector &Operands) {
  LLVM_DEBUG(dbgs() << __FUNCTION__ << "\n");
  const AsmToken Tok = Parser.getTok();
  SMLoc StartLoc = getLexer().getLoc();
  SMLoc EndLoc = getLexer().getLoc();

  // Immediates start with a '#'
  if (!Parser.parseOptionalToken(AsmToken::Hash))
    return ParseStatus::NoMatch;
  
  int64_t Imm;
  if (tryParseImm(Imm, StartLoc, EndLoc).isSuccess()) {
    Operands.push_back(SuperHOperand::CreateImm(
      MCConstantExpr::create(Imm, getContext()), 
      StartLoc, 
      EndLoc
    ));
    return ParseStatus::Success;
  }

  getLexer().UnLex(Tok);
  return ParseStatus::NoMatch;
}

ParseStatus SuperHAsmParser::parseDisp(OperandVector &Operands) {
  LLVM_DEBUG(dbgs() << __FUNCTION__ << "\n");
  return ParseStatus::NoMatch;
}

ParseStatus SuperHAsmParser::parsePCRel(OperandVector &Operands) {
  LLVM_DEBUG(dbgs() << __FUNCTION__ << "\n");
  const AsmToken Tok = Parser.getTok();
  SMLoc StartLoc = getLexer().getLoc();
  SMLoc EndLoc = getLexer().getLoc();
  
  // Parse register relative.
  MCRegister BaseReg;
  MCRegister OffReg;
  const MCExpr *Offset;
  ParseStatus Result = tryParseRegRelative(BaseReg, OffReg, Offset, StartLoc, EndLoc);
  if (!Result.isSuccess())
    return Result;

  // Base register must be PC for this type.
  if (BaseReg != SH::PC) {
    return Error(getTok().getLoc(), "expected PC-relative displacement");
  }

  EndLoc = getLexer().getLoc();
  Operands.push_back(SuperHOperand::CreateSymRef(Offset, StartLoc, EndLoc));
  return ParseStatus::Success;
}

ParseStatus SuperHAsmParser::parseRegister(OperandVector &Operands) {
  LLVM_DEBUG(dbgs() << __FUNCTION__ << "\n");
  const AsmToken Tok = Parser.getTok();
  SMLoc StartLoc = Tok.getLoc();
  SMLoc EndLoc = Tok.getEndLoc();

  MCRegister Reg;
  unsigned RegKind;
  if (tryParseRegister(Reg, RegKind, StartLoc, EndLoc).isSuccess()) {
    Operands.push_back(SuperHOperand::CreateReg(Reg, RegKind, StartLoc, EndLoc));
    return ParseStatus::Success;
  }
  return ParseStatus::NoMatch;
}

ParseStatus SuperHAsmParser::parseRegisterIndirect(OperandVector &Operands) {
  LLVM_DEBUG(dbgs() << __FUNCTION__ << "\n");
  const AsmToken Tok = Parser.getTok();
  SMLoc StartLoc = getLexer().getLoc();
  SMLoc EndLoc = getLexer().getLoc();

  bool IsInc, IsDec;
  MCRegister Reg;

  ParseStatus Result = tryParseRegIndirect(Reg, IsInc, IsDec, StartLoc, EndLoc);
  if (!Result.isSuccess())
    return Result;

  // Indirect
  if (!IsInc && !IsDec) {

    EndLoc = getLexer().getLoc();
    Operands.push_back(SuperHOperand::CreateIReg(Reg, StartLoc, EndLoc));
    return ParseStatus::Success;
  }

  // Indirect Pre-Decrement
  if (!IsInc && IsDec) {

    EndLoc = getLexer().getLoc();
    Operands.push_back(SuperHOperand::CreateIRegDec(Reg, StartLoc, EndLoc));
    return ParseStatus::Success;
  }

  // Indirect Post-Increment
  if (IsInc && !IsDec) {

    EndLoc = getLexer().getLoc();
    Operands.push_back(SuperHOperand::CreateIRegInc(Reg, StartLoc, EndLoc));
    return ParseStatus::Success;
  }

  return ParseStatus::NoMatch;
}

ParseStatus SuperHAsmParser::parseDirective(AsmToken DirectiveID) {
  StringRef IDVal = DirectiveID.getIdentifier();

  if (IDVal.starts_with(".gnu_attribute")) {
    if (parseGNUAttribute(DirectiveID.getLoc()))
      return ParseStatus::Success;
  }
  if (IDVal.equals_insensitive(".little")) {
      return ParseStatus::Success;
  }
  if (IDVal.equals_insensitive(".big")) {
      return ParseStatus::Success;
  }
  return ParseStatus::NoMatch;
}

ParseStatus SuperHAsmParser::parseOperand(OperandVector &Operands) {
  switch (getLexer().getKind()) {
  default:
    return ParseStatus::NoMatch;

  case AsmToken::Identifier:
    return parseRegister(Operands);
  }
}

bool SuperHAsmParser::parseInstruction(ParseInstructionInfo &Info, StringRef Mnemonic, 
                                       SMLoc NameLoc, OperandVector &Operands) {

  // Match mnemonic.
  bool MS = SuperHCheckMnemonic(Mnemonic, this->getAvailableFeatures(), 0);
  if (!MS) {
    return Error(NameLoc, "invalid instruction mnemonic" + 
      SuperHMnemonicSpellCheck(Mnemonic, getAvailableFeatures(), 0));
  }

  // Chomp name and add it to the operands.
  Operands.push_back(SuperHOperand::CreateToken(Mnemonic, NameLoc));
  int OperandNum = -1;
  while (getLexer().isNot(AsmToken::EndOfStatement)) {
    OperandNum++;
    if (OperandNum > 0) {
      if (getLexer().is(AsmToken::Comma)) {
        Parser.Lex();
      }
    }

    ParseStatus Result = MatchOperandParserImpl(Operands, Mnemonic);
    if (Result.isSuccess()) {
      continue;
    }

    // NOTE:  All the Failure states are Errors in of themselves,
    //        as such we don't return an error in this branch.
    if (Result.isFailure()) {
      Parser.eatToEndOfStatement();
      return true;
    }
    
    // Initial operand
    if (!parseOperand(Operands).isSuccess()) {
      SMLoc Loc = getLexer().getLoc();
      Parser.eatToEndOfStatement();
      return Error(Loc, "unexpected token");
    }
  }

  // Consume EndOfStatement
  Parser.Lex();
  return false;
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
  
  LLVM_DEBUG(dbgs() << "matchAndEmitInstruction\n");
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