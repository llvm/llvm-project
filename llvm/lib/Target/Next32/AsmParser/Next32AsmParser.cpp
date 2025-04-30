//===-- Next32AsmParser.cpp - Parse Next32 assembly to MCInst instructions ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/Next32MCTargetDesc.h"
#include "Next32InstrInfo.h"
#include "TargetInfo/Next32BaseInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Casting.h"

using namespace llvm;

namespace {
struct Next32Operand;

class Next32AsmParser : public MCTargetAsmParser {

  SMLoc getLoc() const { return getParser().getTok().getLoc(); }

  bool ExpandOperands(OperandVector &Operands);

  bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               OperandVector &Operands, MCStreamer &Out,
                               uint64_t &ErrorInfo,
                               bool MatchingInlineAsm) override;

  bool parseRegister(MCRegister &RegNo, SMLoc &StartLoc,
                     SMLoc &EndLoc) override;

  ParseStatus tryParseRegister(MCRegister &RegNo, SMLoc &StartLoc,
                               SMLoc &EndLoc) override;

  bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) override;

  bool ParseDirective(AsmToken DirectiveID) override;

  // "=" is used as assignment operator for assembly statment, so can't be used
  // for symbol assignment.
  bool equalIsAsmAssignment() override { return false; }
  // "*" is used for dereferencing memory that it will be the start of
  // statement.
  bool starIsStartOfStatement() override { return true; }

#define GET_ASSEMBLER_HEADER
#include "Next32GenAsmMatcher.inc"

  OperandMatchResultTy parseImmediate(OperandVector &Operands);
  OperandMatchResultTy parseRegister(OperandVector &Operands, bool cond);

public:
  enum Next32MatchResultTy {
    Match_Dummy = FIRST_TARGET_MATCH_RESULT_TY,
#define GET_OPERAND_DIAGNOSTIC_TYPES
#include "Next32GenAsmMatcher.inc"
#undef GET_OPERAND_DIAGNOSTIC_TYPES
  };

  Next32AsmParser(const MCSubtargetInfo &STI, MCAsmParser &Parser,
                  const MCInstrInfo &MII, const MCTargetOptions &Options)
      : MCTargetAsmParser(Options, STI, MII) {
    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));
  }
};

/// Next32Operand - Instances of this class represent a parsed machine
/// instruction
struct Next32Operand : public MCParsedAsmOperand {

  enum KindTy {
    Token,
    Register,
    Conditional,
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

  Next32Operand(KindTy K) : MCParsedAsmOperand(), Kind(K) {}

public:
  Next32Operand(const Next32Operand &o) : MCParsedAsmOperand() {
    Kind = o.Kind;
    StartLoc = o.StartLoc;
    EndLoc = o.EndLoc;

    switch (Kind) {
    case Register:
    case Conditional:
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
  bool isReg() const override {
    return Kind == Register || Kind == Conditional;
  }
  bool isImm() const override { return Kind == Immediate; }
  bool isMem() const override { return false; }

  bool isConditional() const { return Kind == Conditional; }

  /// getStartLoc - Gets location of the first token of this operand
  SMLoc getStartLoc() const override { return StartLoc; }
  /// getEndLoc - Gets location of the last token of this operand
  SMLoc getEndLoc() const override { return EndLoc; }

  MCRegister getReg() const override {
    assert((Kind == Register || Kind == Conditional) && "Invalid type access!");
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
    case Conditional:
      OS << "[r" << getReg() << "]";
      break;
    case Register:
      OS << "r" << getReg();
      break;
    case Token:
      OS << "'" << getToken() << "'";
      break;
    }
  }

  void addExpr(MCInst &Inst, const MCExpr *Expr) const {
    assert(Expr && "Expr shouldn't be null!");

    if (auto *CE = dyn_cast<MCConstantExpr>(Expr))
      Inst.addOperand(MCOperand::createImm(CE->getValue()));
    else
      Inst.addOperand(MCOperand::createExpr(Expr));
  }

  // Used by the TableGen Code
  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::createReg(getReg()));
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getImm());
  }

  std::unique_ptr<Next32Operand> createDuplicate() {
    auto Op = std::make_unique<Next32Operand>(*this);
    return Op;
  }

  void setMnemonicToken(StringRef mnemonic) {
    assert(Kind == Token && "Operand isn't a token");
    Tok = mnemonic;
  }

  static std::unique_ptr<Next32Operand> createToken(StringRef Str, SMLoc S) {
    auto Op = std::make_unique<Next32Operand>(Token);
    Op->Tok = Str;
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static std::unique_ptr<Next32Operand> createReg(unsigned RegNo, SMLoc S,
                                                  SMLoc E, bool cond) {
    auto Op = std::make_unique<Next32Operand>(cond ? Conditional : Register);
    Op->Reg.RegNum = RegNo;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static std::unique_ptr<Next32Operand> createImm(const MCExpr *Val, SMLoc S,
                                                  SMLoc E) {
    auto Op = std::make_unique<Next32Operand>(Immediate);
    Op->Imm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }
  static std::unique_ptr<Next32Operand> createConstantImm(int64_t Val,
                                                          MCContext &Ctx) {
    auto Op = std::make_unique<Next32Operand>(Immediate);
    Op->Imm.Val = MCConstantExpr::create(Val, Ctx);
    return Op;
  }
};
} // end anonymous namespace.

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#include "Next32GenAsmMatcher.inc"

bool Next32AsmParser::ExpandOperands(OperandVector &Operands) {
  // When the instruction is conditional, Next32AsmMatcher expect to match on
  // { [MCK_DOT_XXX], MCK_Imm, MCK_Imm, MCK_GPR32, [MCK_GPR32], MCK_Imm }
  // Operands[0] == instruction mnemonic
  // Operands[1] == conditional state
  // Operands[2] == parallel bit
  // Operands[3] == operand A
  // Operands[4] == operand B
  // Operands[5] == Conditional Register

  // When the instruction is unconditional, Next32AsmMatcher expect to match on
  // { [MCK_DOT_XXX], MCK_Imm, MCK_GPR32, MCK_GPR32 }
  // Operands[0] == instruction mnemonic
  // Operands[1] == parallel bit
  // Operands[2] == operand A
  // Operands[3] == operand B
  if (Operands.size() < 2)
    return false;
  Next32Operand &Op0 = (Next32Operand &)*Operands[0];
  Next32Operand &Op1 = (Next32Operand &)*Operands[1];
  Next32Operand CondOp = (Next32Operand &)*Operands.back();
  if (!Op0.isToken())
    return false;
  if (!Op1.isReg())
    return false;

  // Eat the conditional value
  if (CondOp.isConditional())
    Operands.pop_back();

  if (Operands.size() >= 4)
    return false;

  // Try to parse:
  //    mnemonic[.bmep].condition.p
  // Into:
  //    mnemonic.<dotxx>.condition.isParallel
  StringRef mnemonic = Op0.getToken();
  StringRef dotXXX;

  bool isParallel = false;
  bool isP = mnemonic.consume_back(Next32Helpers::GetParallelMnemonic());
  if (isP)
    Op0.setMnemonicToken(mnemonic);
  bool isWriter = mnemonic.consume_front(Next32Helpers::GetWriterMnemonic());
  int64_t conditional = Next32Constants::NoCondition;

  // Compare conditional for all other opcodes.
  size_t lastDot = mnemonic.rfind('.');
  if ((lastDot != StringRef::npos) &&
      (!isWriter || (mnemonic.count('.') == 2))) {
    // Check that we have conditional register
    if (!CondOp.isConditional())
      return false;
    // For writer we want to check that there is .bep.cond
    StringRef condString = mnemonic.drop_front(lastDot);
    conditional = Next32Helpers::GetCondCodeFromString(condString);
    if (conditional == Next32Constants::NoCondition)
      return false; // Unable to parse the condition
    mnemonic = mnemonic.drop_back(condString.size());
  }

  if (isWriter) {
    Op0.setMnemonicToken(Next32Helpers::GetWriterMnemonic());
    dotXXX = mnemonic;
  } else if (CondOp.isConditional()) {
    Op0.setMnemonicToken(mnemonic);
  }

  if (isP & mnemonic.consume_front(Next32Helpers::GetFeederMnemonic()))
    dotXXX = Next32Helpers::GetParallelMnemonic();
  else
    isParallel = isP;

  // Structure of Operands vector before changes
  // Operands[0] == mnemonic (without .p)
  // Operands[1] == operand A
  // Operands[2] == operand B (optional)

  // Adding Parallel bit before first operand
  Operands.insert(Operands.begin() + 1,
                  Next32Operand::createConstantImm(isParallel, getContext()));

  // Adding Conditionals and conditionalReg if any
  if (conditional != Next32Constants::NoCondition) {
    Operands.insert(Operands.begin() + 1, Next32Operand::createConstantImm(
                                              conditional, getContext()));
    Operands.push_back(Next32Operand::createToken("[", SMLoc()));
    Operands.push_back(CondOp.createDuplicate());
    Operands.push_back(Next32Operand::createToken("]", SMLoc()));
  }

  // Adding dotXXX as second argument, if any
  if (!dotXXX.empty()) {
    Operands.insert(Operands.begin() + 1, Op0.createDuplicate());
    ((Next32Operand &)*Operands[1]).setMnemonicToken(dotXXX);
  }

  // Unary operation, check: Feeder, inc, dec, sext, zext, neg, not ...
  if (Operands.size() == 2)
    return true;

  Next32Operand &Op2 = (Next32Operand &)*Operands[2];
  if (!Op2.isReg() && !Op2.isImm())
    return false;

  return true;
}

bool Next32AsmParser::MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                                              OperandVector &Operands,
                                              MCStreamer &Out,
                                              uint64_t &ErrorInfo,
                                              bool MatchingInlineAsm) {
  MCInst Inst;
  SMLoc ErrorLoc;

  if (!ExpandOperands(Operands))
    return Error(IDLoc, "additional inst constraint not met");

  switch (MatchInstructionImpl(Operands, Inst, ErrorInfo, MatchingInlineAsm)) {
  default:
    break;
  case Match_Success:
    Inst.setLoc(IDLoc);
    Out.emitInstruction(Inst, getSTI());
    return false;
  case Match_MissingFeature:
    return Error(IDLoc, "instruction use requires an option to be enabled");
  case Match_MnemonicFail:
    return Error(IDLoc, "unrecognized instruction mnemonic");
  case Match_InvalidOperand:
    ErrorLoc = IDLoc;

    if (ErrorInfo != ~0U) {
      if (ErrorInfo >= Operands.size())
        return Error(ErrorLoc, "too few operands for instruction");

      ErrorLoc = ((Next32Operand &)*Operands[ErrorInfo]).getStartLoc();

      if (ErrorLoc == SMLoc())
        ErrorLoc = IDLoc;
    }

    return Error(ErrorLoc, "invalid operand for instruction");
  }

  llvm_unreachable("Unknown match type detected!");
}

bool Next32AsmParser::parseRegister(MCRegister &RegNo, SMLoc &StartLoc,
                                    SMLoc &EndLoc) {
  if (!tryParseRegister(RegNo, StartLoc, EndLoc).isSuccess())
    return Error(StartLoc, "invalid register name");
  return false;
}

ParseStatus Next32AsmParser::tryParseRegister(MCRegister &RegNo,
                                              SMLoc &StartLoc, SMLoc &EndLoc) {
  const AsmToken &Tok = getParser().getTok();
  StartLoc = Tok.getLoc();
  EndLoc = Tok.getEndLoc();
  RegNo = 0;
  StringRef Name = getLexer().getTok().getIdentifier();

  if (!MatchRegisterName(Name)) {
    getParser().Lex(); // Eat identifier token.
    return ParseStatus::Success;
  }

  return ParseStatus::NoMatch;
}

OperandMatchResultTy Next32AsmParser::parseRegister(OperandVector &Operands,
                                                    bool cond) {
  SMLoc S = getLoc();
  SMLoc E = SMLoc::getFromPointer(S.getPointer() - 1);

  switch (getLexer().getKind()) {
  default:
    return MatchOperand_NoMatch;
  case AsmToken::Identifier:
    StringRef Name = getLexer().getTok().getIdentifier();
    unsigned RegNo = MatchRegisterName(Name);

    if (RegNo == 0)
      return MatchOperand_NoMatch;

    getLexer().Lex();
    Operands.push_back(Next32Operand::createReg(RegNo, S, E, cond));
  }
  return MatchOperand_Success;
}

OperandMatchResultTy Next32AsmParser::parseImmediate(OperandVector &Operands) {
  switch (getLexer().getKind()) {
  default:
    return MatchOperand_NoMatch;
  case AsmToken::LParen:
  case AsmToken::Minus:
  case AsmToken::Plus:
  case AsmToken::Integer:
  case AsmToken::String:
  case AsmToken::Identifier:
    break;
  }

  const MCExpr *IdVal;
  SMLoc S = getLoc();

  if (getParser().parseExpression(IdVal))
    return MatchOperand_ParseFail;

  SMLoc E = SMLoc::getFromPointer(S.getPointer() - 1);
  Operands.push_back(Next32Operand::createImm(IdVal, S, E));

  return MatchOperand_Success;
}

/// ParseInstruction - Parse an Next32 instruction which is in Next32 verifier
/// format.
bool Next32AsmParser::ParseInstruction(ParseInstructionInfo &Info,
                                       StringRef Name, SMLoc NameLoc,
                                       OperandVector &Operands) {
  // First operand in MCInst is instruction mnemonic.
  Operands.push_back(Next32Operand::createToken(Name, NameLoc));

  while (!getLexer().is(AsmToken::EndOfStatement)) {
    // Attempt to parse token as register
    if (parseRegister(Operands, false) != MatchOperand_Success) {
      // Attempt to parse token as an immediate
      if (parseImmediate(Operands) != MatchOperand_Success) {
        // Attempt to parse token as a symbol
        // Error, not EndOfStatement exception will be thrown.
        break;
      }
    }

    // Eat Comma if any, and check there are more tokens
    if (getLexer().is(AsmToken::Comma)) {
      getParser().Lex();
      if (getLexer().is(AsmToken::EndOfStatement))
        return Error(getLexer().getLoc(), "unexpected end of statement");
    } else if (getLexer().is(AsmToken::LBrac)) {
      getParser().Lex();
      if (parseRegister(Operands, true) != MatchOperand_Success)
        return Error(getLexer().getLoc(),
                     "unable to parse conditional operand");
      if (!getLexer().is(AsmToken::RBrac))
        return Error(getLexer().getLoc(),
                     "unable to parse conditional operand");
      getParser().Lex();
      break;
    }
  }

  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    SMLoc Loc = getLexer().getLoc();

    getParser().eatToEndOfStatement();

    return Error(Loc, "unexpected token");
  }

  // Consume the EndOfStatement.
  getParser().Lex();
  return false;
}

bool Next32AsmParser::ParseDirective(AsmToken DirectiveID) { return true; }

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeNext32AsmParser() {
  RegisterMCAsmParser<Next32AsmParser> X(getTheNext32Target());
}
