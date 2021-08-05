//===-- CSKYAsmParser.cpp - Parse CSKY assembly to MCInst instructions --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/CSKYMCExpr.h"
#include "MCTargetDesc/CSKYMCTargetDesc.h"
#include "TargetInfo/CSKYTargetInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

namespace {
struct CSKYOperand;

class CSKYAsmParser : public MCTargetAsmParser {

  bool generateImmOutOfRangeError(OperandVector &Operands, uint64_t ErrorInfo,
                                  int64_t Lower, int64_t Upper, Twine Msg);

  SMLoc getLoc() const { return getParser().getTok().getLoc(); }

  bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               OperandVector &Operands, MCStreamer &Out,
                               uint64_t &ErrorInfo,
                               bool MatchingInlineAsm) override;

  bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc) override;

  bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) override;

  bool ParseDirective(AsmToken DirectiveID) override;

  OperandMatchResultTy tryParseRegister(unsigned &RegNo, SMLoc &StartLoc,
                                        SMLoc &EndLoc) override;

// Auto-generated instruction matching functions
#define GET_ASSEMBLER_HEADER
#include "CSKYGenAsmMatcher.inc"

  OperandMatchResultTy parseImmediate(OperandVector &Operands);
  OperandMatchResultTy parseRegister(OperandVector &Operands);
  OperandMatchResultTy parseBaseRegImm(OperandVector &Operands);
  OperandMatchResultTy parseCSKYSymbol(OperandVector &Operands);
  OperandMatchResultTy parseConstpoolSymbol(OperandVector &Operands);

  bool parseOperand(OperandVector &Operands, StringRef Mnemonic);

public:
  enum CSKYMatchResultTy {
    Match_Dummy = FIRST_TARGET_MATCH_RESULT_TY,
#define GET_OPERAND_DIAGNOSTIC_TYPES
#include "CSKYGenAsmMatcher.inc"
#undef GET_OPERAND_DIAGNOSTIC_TYPES
  };

  CSKYAsmParser(const MCSubtargetInfo &STI, MCAsmParser &Parser,
                const MCInstrInfo &MII, const MCTargetOptions &Options)
      : MCTargetAsmParser(Options, STI, MII) {
    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));
  }
};

/// Instances of this class represent a parsed machine instruction.
struct CSKYOperand : public MCParsedAsmOperand {
  enum KindTy {
    Token,
    Register,
    Immediate,
  } Kind;

  struct RegOp {
    unsigned RegNum;
  };

  struct ImmOp {
    const MCExpr *Val;
  };

  SMLoc StartLoc, EndLoc;
  union {
    StringRef Tok;
    RegOp Reg;
    ImmOp Imm;
  };

  CSKYOperand(KindTy K) : MCParsedAsmOperand(), Kind(K) {}

public:
  CSKYOperand(const CSKYOperand &o) : MCParsedAsmOperand() {
    Kind = o.Kind;
    StartLoc = o.StartLoc;
    EndLoc = o.EndLoc;
    switch (Kind) {
    case Register:
      Reg = o.Reg;
      break;
    case Immediate:
      Imm = o.Imm;
      break;
    case Token:
      Tok = o.Tok;
      break;
    }
  }

  bool isToken() const override { return Kind == Token; }
  bool isReg() const override { return Kind == Register; }
  bool isImm() const override { return Kind == Immediate; }
  bool isMem() const override { return false; }

  static bool evaluateConstantImm(const MCExpr *Expr, int64_t &Imm) {
    if (auto CE = dyn_cast<MCConstantExpr>(Expr)) {
      Imm = CE->getValue();
      return true;
    }

    return false;
  }

  template <unsigned num, unsigned shift = 0> bool isUImm() const {
    if (!isImm())
      return false;

    int64_t Imm;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm);
    return IsConstantImm && isShiftedUInt<num, shift>(Imm);
  }

  template <unsigned num> bool isOImm() const {
    if (!isImm())
      return false;

    int64_t Imm;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm);
    return IsConstantImm && isUInt<num>(Imm - 1);
  }

  template <unsigned num, unsigned shift = 0> bool isSImm() const {
    if (!isImm())
      return false;

    int64_t Imm;
    bool IsConstantImm = evaluateConstantImm(getImm(), Imm);
    return IsConstantImm && isShiftedInt<num, shift>(Imm);
  }

  bool isUImm2() const { return isUImm<2>(); }
  bool isUImm5() const { return isUImm<5>(); }
  bool isUImm12() const { return isUImm<12>(); }
  bool isUImm16() const { return isUImm<16>(); }

  bool isOImm12() const { return isOImm<12>(); }
  bool isOImm16() const { return isOImm<16>(); }

  bool isUImm12Shift1() { return isUImm<12, 1>(); }
  bool isUImm12Shift2() { return isUImm<12, 2>(); }

  bool isSImm16Shift1() { return isSImm<16, 1>(); }

  bool isCSKYSymbol() const {
    int64_t Imm;
    // Must be of 'immediate' type but not a constant.
    return isImm() && !evaluateConstantImm(getImm(), Imm);
  }

  bool isConstpoolSymbol() const {
    int64_t Imm;
    // Must be of 'immediate' type but not a constant.
    return isImm() && !evaluateConstantImm(getImm(), Imm);
  }

  /// Gets location of the first token of this operand.
  SMLoc getStartLoc() const override { return StartLoc; }
  /// Gets location of the last token of this operand.
  SMLoc getEndLoc() const override { return EndLoc; }

  unsigned getReg() const override {
    assert(Kind == Register && "Invalid type access!");
    return Reg.RegNum;
  }

  const MCExpr *getImm() const {
    assert(Kind == Immediate && "Invalid type access!");
    return Imm.Val;
  }

  StringRef getToken() const {
    assert(Kind == Token && "Invalid type access!");
    return Tok;
  }

  void print(raw_ostream &OS) const override {
    switch (Kind) {
    case Immediate:
      OS << *getImm();
      break;
    case Register:
      OS << "<register x" << getReg() << ">";
      break;
    case Token:
      OS << "'" << getToken() << "'";
      break;
    }
  }

  static std::unique_ptr<CSKYOperand> createToken(StringRef Str, SMLoc S) {
    auto Op = std::make_unique<CSKYOperand>(Token);
    Op->Tok = Str;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static std::unique_ptr<CSKYOperand> createReg(unsigned RegNo, SMLoc S,
                                                SMLoc E) {
    auto Op = std::make_unique<CSKYOperand>(Register);
    Op->Reg.RegNum = RegNo;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<CSKYOperand> createImm(const MCExpr *Val, SMLoc S,
                                                SMLoc E) {
    auto Op = std::make_unique<CSKYOperand>(Immediate);
    Op->Imm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  void addExpr(MCInst &Inst, const MCExpr *Expr) const {
    assert(Expr && "Expr shouldn't be null!");
    if (auto *CE = dyn_cast<MCConstantExpr>(Expr))
      Inst.addOperand(MCOperand::createImm(CE->getValue()));
    else
      Inst.addOperand(MCOperand::createExpr(Expr));
  }

  // Used by the TableGen Code.
  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getReg()));
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getImm());
  }
};
} // end anonymous namespace.

#define GET_REGISTER_MATCHER
#define GET_SUBTARGET_FEATURE_NAME
#define GET_MATCHER_IMPLEMENTATION
#define GET_MNEMONIC_SPELL_CHECKER
#include "CSKYGenAsmMatcher.inc"

static std::string CSKYMnemonicSpellCheck(StringRef S, const FeatureBitset &FBS,
                                          unsigned VariantID = 0);

bool CSKYAsmParser::generateImmOutOfRangeError(
    OperandVector &Operands, uint64_t ErrorInfo, int64_t Lower, int64_t Upper,
    Twine Msg = "immediate must be an integer in the range") {
  SMLoc ErrorLoc = ((CSKYOperand &)*Operands[ErrorInfo]).getStartLoc();
  return Error(ErrorLoc, Msg + " [" + Twine(Lower) + ", " + Twine(Upper) + "]");
}

bool CSKYAsmParser::MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                                            OperandVector &Operands,
                                            MCStreamer &Out,
                                            uint64_t &ErrorInfo,
                                            bool MatchingInlineAsm) {
  MCInst Inst;
  FeatureBitset MissingFeatures;

  auto Result = MatchInstructionImpl(Operands, Inst, ErrorInfo, MissingFeatures,
                                     MatchingInlineAsm);
  switch (Result) {
  default:
    break;
  case Match_Success:
    Inst.setLoc(IDLoc);
    Out.emitInstruction(Inst, getSTI());
    return false;
  case Match_MissingFeature: {
    assert(MissingFeatures.any() && "Unknown missing features!");
    ListSeparator LS;
    std::string Msg = "instruction requires the following: ";
    for (unsigned i = 0, e = MissingFeatures.size(); i != e; ++i) {
      if (MissingFeatures[i]) {
        Msg += LS;
        Msg += getSubtargetFeatureName(i);
      }
    }
    return Error(IDLoc, Msg);
  }
  case Match_MnemonicFail: {
    FeatureBitset FBS = ComputeAvailableFeatures(getSTI().getFeatureBits());
    std::string Suggestion =
        CSKYMnemonicSpellCheck(((CSKYOperand &)*Operands[0]).getToken(), FBS);
    return Error(IDLoc, "unrecognized instruction mnemonic" + Suggestion);
  }
  case Match_InvalidTiedOperand:
  case Match_InvalidOperand: {
    SMLoc ErrorLoc = IDLoc;
    if (ErrorInfo != ~0U) {
      if (ErrorInfo >= Operands.size())
        return Error(ErrorLoc, "too few operands for instruction");

      ErrorLoc = ((CSKYOperand &)*Operands[ErrorInfo]).getStartLoc();
      if (ErrorLoc == SMLoc())
        ErrorLoc = IDLoc;
    }
    return Error(ErrorLoc, "invalid operand for instruction");
  }
  }

  // Handle the case when the error message is of specific type
  // other than the generic Match_InvalidOperand, and the
  // corresponding operand is missing.
  if (Result > FIRST_TARGET_MATCH_RESULT_TY) {
    SMLoc ErrorLoc = IDLoc;
    if (ErrorInfo != ~0U && ErrorInfo >= Operands.size())
      return Error(ErrorLoc, "too few operands for instruction");
  }

  switch (Result) {
  default:
    break;
  case Match_InvalidOImm12:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 1, (1 << 12));
  case Match_InvalidOImm16:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 1, (1 << 16));
  case Match_InvalidUImm2:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 2) - 1);
  case Match_InvalidUImm5:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 5) - 1);
  case Match_InvalidUImm12:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 12) - 1);
  case Match_InvalidUImm12Shift1:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, 0, (1 << 12) - 2,
        "immediate must be a multiple of 2 bytes in the range");
  case Match_InvalidUImm12Shift2:
    return generateImmOutOfRangeError(
        Operands, ErrorInfo, 0, (1 << 12) - 4,
        "immediate must be a multiple of 4 bytes in the range");
  case Match_InvalidUImm16:
    return generateImmOutOfRangeError(Operands, ErrorInfo, 0, (1 << 16) - 1);
  case Match_InvalidCSKYSymbol: {
    SMLoc ErrorLoc = ((CSKYOperand &)*Operands[ErrorInfo]).getStartLoc();
    return Error(ErrorLoc, "operand must be a symbol name");
  }
  case Match_InvalidConstpool: {
    SMLoc ErrorLoc = ((CSKYOperand &)*Operands[ErrorInfo]).getStartLoc();
    return Error(ErrorLoc, "operand must be a constpool symbol name");
  }
  }

  llvm_unreachable("Unknown match type detected!");
}

// Attempts to match Name as a register (either using the default name or
// alternative ABI names), setting RegNo to the matching register. Upon
// failure, returns true and sets RegNo to 0.
static bool matchRegisterNameHelper(MCRegister &RegNo, StringRef Name) {
  RegNo = MatchRegisterName(Name);

  if (RegNo == CSKY::NoRegister)
    RegNo = MatchRegisterAltName(Name);

  return RegNo == CSKY::NoRegister;
}

bool CSKYAsmParser::ParseRegister(unsigned &RegNo, SMLoc &StartLoc,
                                  SMLoc &EndLoc) {
  const AsmToken &Tok = getParser().getTok();
  StartLoc = Tok.getLoc();
  EndLoc = Tok.getEndLoc();
  StringRef Name = getLexer().getTok().getIdentifier();

  if (!matchRegisterNameHelper((MCRegister &)RegNo, Name)) {
    getParser().Lex(); // Eat identifier token.
    return false;
  }

  return Error(StartLoc, "invalid register name");
}

OperandMatchResultTy CSKYAsmParser::parseRegister(OperandVector &Operands) {
  SMLoc S = getLoc();
  SMLoc E = SMLoc::getFromPointer(S.getPointer() - 1);

  switch (getLexer().getKind()) {
  default:
    return MatchOperand_NoMatch;
  case AsmToken::Identifier: {
    StringRef Name = getLexer().getTok().getIdentifier();
    MCRegister RegNo;

    if (matchRegisterNameHelper((MCRegister &)RegNo, Name))
      return MatchOperand_NoMatch;

    getLexer().Lex();
    Operands.push_back(CSKYOperand::createReg(RegNo, S, E));

    return MatchOperand_Success;
  }
  }
}

OperandMatchResultTy CSKYAsmParser::parseBaseRegImm(OperandVector &Operands) {
  assert(getLexer().is(AsmToken::LParen));

  Operands.push_back(CSKYOperand::createToken("(", getLoc()));

  auto Tok = getParser().Lex(); // Eat '('

  if (parseRegister(Operands) != MatchOperand_Success) {
    getLexer().UnLex(Tok);
    Operands.pop_back();
    return MatchOperand_ParseFail;
  }

  if (getLexer().isNot(AsmToken::Comma)) {
    Error(getLoc(), "expected ','");
    return MatchOperand_ParseFail;
  }

  getParser().Lex(); // Eat ','

  if (parseRegister(Operands) == MatchOperand_Success) {
    if (getLexer().isNot(AsmToken::LessLess)) {
      Error(getLoc(), "expected '<<'");
      return MatchOperand_ParseFail;
    }

    Operands.push_back(CSKYOperand::createToken("<<", getLoc()));

    getParser().Lex(); // Eat '<<'

    if (parseImmediate(Operands) != MatchOperand_Success) {
      Error(getLoc(), "expected imm");
      return MatchOperand_ParseFail;
    }

  } else if (parseImmediate(Operands) != MatchOperand_Success) {
    Error(getLoc(), "expected imm");
    return MatchOperand_ParseFail;
  }

  if (getLexer().isNot(AsmToken::RParen)) {
    Error(getLoc(), "expected ')'");
    return MatchOperand_ParseFail;
  }

  Operands.push_back(CSKYOperand::createToken(")", getLoc()));

  getParser().Lex(); // Eat ')'

  return MatchOperand_Success;
}

OperandMatchResultTy CSKYAsmParser::parseImmediate(OperandVector &Operands) {
  switch (getLexer().getKind()) {
  default:
    return MatchOperand_NoMatch;
  case AsmToken::LParen:
  case AsmToken::Minus:
  case AsmToken::Plus:
  case AsmToken::Integer:
  case AsmToken::String:
    break;
  }

  const MCExpr *IdVal;
  SMLoc S = getLoc();
  if (getParser().parseExpression(IdVal))
    return MatchOperand_ParseFail;

  SMLoc E = SMLoc::getFromPointer(S.getPointer() - 1);
  Operands.push_back(CSKYOperand::createImm(IdVal, S, E));
  return MatchOperand_Success;
}

/// Looks at a token type and creates the relevant operand from this
/// information, adding to Operands. If operand was parsed, returns false, else
/// true.
bool CSKYAsmParser::parseOperand(OperandVector &Operands, StringRef Mnemonic) {
  // Check if the current operand has a custom associated parser, if so, try to
  // custom parse the operand, or fallback to the general approach.
  OperandMatchResultTy Result =
      MatchOperandParserImpl(Operands, Mnemonic, /*ParseForAllFeatures=*/true);
  if (Result == MatchOperand_Success)
    return false;
  if (Result == MatchOperand_ParseFail)
    return true;

  // Attempt to parse token as register
  if (parseRegister(Operands) == MatchOperand_Success)
    return false;

  // Attempt to parse token as (register, imm)
  if (getLexer().is(AsmToken::LParen))
    if (parseBaseRegImm(Operands) == MatchOperand_Success)
      return false;

  // Attempt to parse token as a imm.
  if (parseImmediate(Operands) == MatchOperand_Success)
    return false;

  // Finally we have exhausted all options and must declare defeat.
  Error(getLoc(), "unknown operand");
  return true;
}

OperandMatchResultTy CSKYAsmParser::parseCSKYSymbol(OperandVector &Operands) {
  SMLoc S = getLoc();
  SMLoc E = SMLoc::getFromPointer(S.getPointer() - 1);

  if (getLexer().getKind() != AsmToken::Identifier)
    return MatchOperand_NoMatch;

  StringRef Identifier;
  if (getParser().parseIdentifier(Identifier))
    return MatchOperand_ParseFail;

  CSKYMCExpr::VariantKind Kind = CSKYMCExpr::VK_CSKY_None;

  if (Identifier.consume_back("@GOT"))
    Kind = CSKYMCExpr::VK_CSKY_GOT;
  else if (Identifier.consume_back("@GOTOFF"))
    Kind = CSKYMCExpr::VK_CSKY_GOTOFF;
  else if (Identifier.consume_back("@PLT"))
    Kind = CSKYMCExpr::VK_CSKY_PLT;
  else if (Identifier.consume_back("@GOTPC"))
    Kind = CSKYMCExpr::VK_CSKY_GOTPC;

  MCSymbol *Sym = getContext().getOrCreateSymbol(Identifier);
  const MCExpr *Res =
      MCSymbolRefExpr::create(Sym, MCSymbolRefExpr::VK_None, getContext());

  if (Kind != CSKYMCExpr::VK_CSKY_None)
    Res = CSKYMCExpr::create(Res, Kind, getContext());

  Operands.push_back(CSKYOperand::createImm(Res, S, E));
  return MatchOperand_Success;
}

OperandMatchResultTy
CSKYAsmParser::parseConstpoolSymbol(OperandVector &Operands) {
  SMLoc S = getLoc();
  SMLoc E = SMLoc::getFromPointer(S.getPointer() - 1);

  if (getLexer().getKind() != AsmToken::LBrac)
    return MatchOperand_NoMatch;

  getLexer().Lex(); // Eat '['.

  if (getLexer().getKind() != AsmToken::Identifier)
    return MatchOperand_NoMatch;

  StringRef Identifier;
  if (getParser().parseIdentifier(Identifier))
    return MatchOperand_ParseFail;

  if (getLexer().getKind() != AsmToken::RBrac)
    return MatchOperand_NoMatch;

  getLexer().Lex(); // Eat ']'.

  MCSymbol *Sym = getContext().getOrCreateSymbol(Identifier);
  const MCExpr *Res =
      MCSymbolRefExpr::create(Sym, MCSymbolRefExpr::VK_None, getContext());
  Operands.push_back(CSKYOperand::createImm(Res, S, E));
  return MatchOperand_Success;
}

bool CSKYAsmParser::ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                                     SMLoc NameLoc, OperandVector &Operands) {
  // First operand is token for instruction.
  Operands.push_back(CSKYOperand::createToken(Name, NameLoc));

  // If there are no more operands, then finish.
  if (getLexer().is(AsmToken::EndOfStatement))
    return false;

  // Parse first operand.
  if (parseOperand(Operands, Name))
    return true;

  // Parse until end of statement, consuming commas between operands.
  while (getLexer().is(AsmToken::Comma)) {
    // Consume comma token.
    getLexer().Lex();

    // Parse next operand.
    if (parseOperand(Operands, Name))
      return true;
  }

  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    SMLoc Loc = getLexer().getLoc();
    getParser().eatToEndOfStatement();
    return Error(Loc, "unexpected token");
  }

  getParser().Lex(); // Consume the EndOfStatement.
  return false;
}

OperandMatchResultTy CSKYAsmParser::tryParseRegister(unsigned &RegNo,
                                                     SMLoc &StartLoc,
                                                     SMLoc &EndLoc) {
  const AsmToken &Tok = getParser().getTok();
  StartLoc = Tok.getLoc();
  EndLoc = Tok.getEndLoc();

  StringRef Name = getLexer().getTok().getIdentifier();

  if (matchRegisterNameHelper((MCRegister &)RegNo, Name))
    return MatchOperand_NoMatch;

  getParser().Lex(); // Eat identifier token.
  return MatchOperand_Success;
}

bool CSKYAsmParser::ParseDirective(AsmToken DirectiveID) { return true; }

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeCSKYAsmParser() {
  RegisterMCAsmParser<CSKYAsmParser> X(getTheCSKYTarget());
}
