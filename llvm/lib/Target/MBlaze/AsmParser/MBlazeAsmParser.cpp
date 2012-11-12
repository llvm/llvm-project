//===-- MBlazeAsmParser.cpp - Parse MBlaze asm to MCInst instructions -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/MBlazeBaseInfo.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
using namespace llvm;

namespace {
struct MBlazeOperand;

class MBlazeAsmParser : public MCTargetAsmParser {
  MCAsmParser &Parser;

  MCAsmParser &getParser() const { return Parser; }
  MCAsmLexer &getLexer() const { return Parser.getLexer(); }

  void Warning(SMLoc L, const Twine &Msg) { Parser.Warning(L, Msg); }
  bool Error(SMLoc L, const Twine &Msg) { return Parser.Error(L, Msg); }

  MBlazeOperand *ParseMemory(SmallVectorImpl<MCParsedAsmOperand*> &Operands);
  MBlazeOperand *ParseRegister(unsigned &RegNo);
  MBlazeOperand *ParseImmediate();
  MBlazeOperand *ParseFsl();
  MBlazeOperand* ParseOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  virtual bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc);

  bool ParseDirectiveWord(unsigned Size, SMLoc L);

  bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                               MCStreamer &Out, unsigned &ErrorInfo,
                               bool MatchingInlineAsm);

  /// @name Auto-generated Match Functions
  /// {

#define GET_ASSEMBLER_HEADER
#include "MBlazeGenAsmMatcher.inc"

  /// }

public:
  MBlazeAsmParser(MCSubtargetInfo &_STI, MCAsmParser &_Parser)
    : MCTargetAsmParser(), Parser(_Parser) {}

  virtual bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                                SMLoc NameLoc,
                                SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  virtual bool ParseDirective(AsmToken DirectiveID);
};

/// MBlazeOperand - Instances of this class represent a parsed MBlaze machine
/// instruction.
struct MBlazeOperand : public MCParsedAsmOperand {
  enum KindTy {
    Token,
    Immediate,
    Register,
    Memory,
    Fsl
  } Kind;

  SMLoc StartLoc, EndLoc;

  union {
    struct {
      const char *Data;
      unsigned Length;
    } Tok;

    struct {
      unsigned RegNum;
    } Reg;

    struct {
      const MCExpr *Val;
    } Imm;

    struct {
      unsigned Base;
      unsigned OffReg;
      const MCExpr *Off;
    } Mem;

    struct {
      const MCExpr *Val;
    } FslImm;
  };

  MBlazeOperand(KindTy K) : MCParsedAsmOperand(), Kind(K) {}
public:
  MBlazeOperand(const MBlazeOperand &o) : MCParsedAsmOperand() {
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
    case Memory:
      Mem = o.Mem;
      break;
    case Fsl:
      FslImm = o.FslImm;
      break;
    }
  }

  /// getStartLoc - Get the location of the first token of this operand.
  SMLoc getStartLoc() const { return StartLoc; }

  /// getEndLoc - Get the location of the last token of this operand.
  SMLoc getEndLoc() const { return EndLoc; }

  unsigned getReg() const {
    assert(Kind == Register && "Invalid access!");
    return Reg.RegNum;
  }

  const MCExpr *getImm() const {
    assert(Kind == Immediate && "Invalid access!");
    return Imm.Val;
  }

  const MCExpr *getFslImm() const {
    assert(Kind == Fsl && "Invalid access!");
    return FslImm.Val;
  }

  unsigned getMemBase() const {
    assert(Kind == Memory && "Invalid access!");
    return Mem.Base;
  }

  const MCExpr* getMemOff() const {
    assert(Kind == Memory && "Invalid access!");
    return Mem.Off;
  }

  unsigned getMemOffReg() const {
    assert(Kind == Memory && "Invalid access!");
    return Mem.OffReg;
  }

  bool isToken() const { return Kind == Token; }
  bool isImm() const { return Kind == Immediate; }
  bool isMem() const { return Kind == Memory; }
  bool isFsl() const { return Kind == Fsl; }
  bool isReg() const { return Kind == Register; }

  void addExpr(MCInst &Inst, const MCExpr *Expr) const {
    // Add as immediates when possible.  Null MCExpr = 0.
    if (Expr == 0)
      Inst.addOperand(MCOperand::CreateImm(0));
    else if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Expr))
      Inst.addOperand(MCOperand::CreateImm(CE->getValue()));
    else
      Inst.addOperand(MCOperand::CreateExpr(Expr));
  }

  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getReg()));
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getImm());
  }

  void addFslOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getFslImm());
  }

  void addMemOperands(MCInst &Inst, unsigned N) const {
    assert(N == 2 && "Invalid number of operands!");

    Inst.addOperand(MCOperand::CreateReg(getMemBase()));

    unsigned RegOff = getMemOffReg();
    if (RegOff)
      Inst.addOperand(MCOperand::CreateReg(RegOff));
    else
      addExpr(Inst, getMemOff());
  }

  StringRef getToken() const {
    assert(Kind == Token && "Invalid access!");
    return StringRef(Tok.Data, Tok.Length);
  }

  virtual void print(raw_ostream &OS) const;

  static MBlazeOperand *CreateToken(StringRef Str, SMLoc S) {
    MBlazeOperand *Op = new MBlazeOperand(Token);
    Op->Tok.Data = Str.data();
    Op->Tok.Length = Str.size();
    Op->StartLoc = S;
    Op->EndLoc = S;
    return Op;
  }

  static MBlazeOperand *CreateReg(unsigned RegNum, SMLoc S, SMLoc E) {
    MBlazeOperand *Op = new MBlazeOperand(Register);
    Op->Reg.RegNum = RegNum;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static MBlazeOperand *CreateImm(const MCExpr *Val, SMLoc S, SMLoc E) {
    MBlazeOperand *Op = new MBlazeOperand(Immediate);
    Op->Imm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static MBlazeOperand *CreateFslImm(const MCExpr *Val, SMLoc S, SMLoc E) {
    MBlazeOperand *Op = new MBlazeOperand(Fsl);
    Op->Imm.Val = Val;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static MBlazeOperand *CreateMem(unsigned Base, const MCExpr *Off, SMLoc S,
                                  SMLoc E) {
    MBlazeOperand *Op = new MBlazeOperand(Memory);
    Op->Mem.Base = Base;
    Op->Mem.Off = Off;
    Op->Mem.OffReg = 0;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }

  static MBlazeOperand *CreateMem(unsigned Base, unsigned Off, SMLoc S,
                                  SMLoc E) {
    MBlazeOperand *Op = new MBlazeOperand(Memory);
    Op->Mem.Base = Base;
    Op->Mem.OffReg = Off;
    Op->Mem.Off = 0;
    Op->StartLoc = S;
    Op->EndLoc = E;
    return Op;
  }
};

} // end anonymous namespace.

void MBlazeOperand::print(raw_ostream &OS) const {
  switch (Kind) {
  case Immediate:
    getImm()->print(OS);
    break;
  case Register:
    OS << "<register R";
    OS << getMBlazeRegisterNumbering(getReg()) << ">";
    break;
  case Token:
    OS << "'" << getToken() << "'";
    break;
  case Memory: {
    OS << "<memory R";
    OS << getMBlazeRegisterNumbering(getMemBase());
    OS << ", ";

    unsigned RegOff = getMemOffReg();
    if (RegOff)
      OS << "R" << getMBlazeRegisterNumbering(RegOff);
    else
      OS << getMemOff();
    OS << ">";
    }
    break;
  case Fsl:
    getFslImm()->print(OS);
    break;
  }
}

/// @name Auto-generated Match Functions
/// {

static unsigned MatchRegisterName(StringRef Name);

/// }
//
bool MBlazeAsmParser::
MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                        SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                        MCStreamer &Out, unsigned &ErrorInfo,
                        bool MatchingInlineAsm) {
  MCInst Inst;
  switch (MatchInstructionImpl(Operands, Inst, ErrorInfo,
                               MatchingInlineAsm)) {
  default: break;
  case Match_Success:
    Out.EmitInstruction(Inst);
    return false;
  case Match_MissingFeature:
    return Error(IDLoc, "instruction use requires an option to be enabled");
  case Match_MnemonicFail:
      return Error(IDLoc, "unrecognized instruction mnemonic");
  case Match_InvalidOperand: {
    SMLoc ErrorLoc = IDLoc;
    if (ErrorInfo != ~0U) {
      if (ErrorInfo >= Operands.size())
        return Error(IDLoc, "too few operands for instruction");

      ErrorLoc = ((MBlazeOperand*)Operands[ErrorInfo])->getStartLoc();
      if (ErrorLoc == SMLoc()) ErrorLoc = IDLoc;
    }

    return Error(ErrorLoc, "invalid operand for instruction");
  }
  }

  llvm_unreachable("Implement any new match types added!");
}

MBlazeOperand *MBlazeAsmParser::
ParseMemory(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  if (Operands.size() != 4)
    return 0;

  MBlazeOperand &Base = *(MBlazeOperand*)Operands[2];
  MBlazeOperand &Offset = *(MBlazeOperand*)Operands[3];

  SMLoc S = Base.getStartLoc();
  SMLoc O = Offset.getStartLoc();
  SMLoc E = Offset.getEndLoc();

  if (!Base.isReg()) {
    Error(S, "base address must be a register");
    return 0;
  }

  if (!Offset.isReg() && !Offset.isImm()) {
    Error(O, "offset must be a register or immediate");
    return 0;
  }

  MBlazeOperand *Op;
  if (Offset.isReg())
    Op = MBlazeOperand::CreateMem(Base.getReg(), Offset.getReg(), S, E);
  else
    Op = MBlazeOperand::CreateMem(Base.getReg(), Offset.getImm(), S, E);

  delete Operands.pop_back_val();
  delete Operands.pop_back_val();
  Operands.push_back(Op);

  return Op;
}

bool MBlazeAsmParser::ParseRegister(unsigned &RegNo,
                                    SMLoc &StartLoc, SMLoc &EndLoc) {
  return (ParseRegister(RegNo) == 0);
}

MBlazeOperand *MBlazeAsmParser::ParseRegister(unsigned &RegNo) {
  SMLoc S = Parser.getTok().getLoc();
  SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);

  switch (getLexer().getKind()) {
  default: return 0;
  case AsmToken::Identifier:
    RegNo = MatchRegisterName(getLexer().getTok().getIdentifier());
    if (RegNo == 0)
      return 0;

    getLexer().Lex();
    return MBlazeOperand::CreateReg(RegNo, S, E);
  }
}

static unsigned MatchFslRegister(StringRef String) {
  if (!String.startswith("rfsl"))
    return -1;

  unsigned regNum;
  if (String.substr(4).getAsInteger(10,regNum))
    return -1;

  return regNum;
}

MBlazeOperand *MBlazeAsmParser::ParseFsl() {
  SMLoc S = Parser.getTok().getLoc();
  SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);

  switch (getLexer().getKind()) {
  default: return 0;
  case AsmToken::Identifier:
    unsigned reg = MatchFslRegister(getLexer().getTok().getIdentifier());
    if (reg >= 16)
      return 0;

    getLexer().Lex();
    const MCExpr *EVal = MCConstantExpr::Create(reg,getContext());
    return MBlazeOperand::CreateFslImm(EVal,S,E);
  }
}

MBlazeOperand *MBlazeAsmParser::ParseImmediate() {
  SMLoc S = Parser.getTok().getLoc();
  SMLoc E = SMLoc::getFromPointer(Parser.getTok().getLoc().getPointer() - 1);

  const MCExpr *EVal;
  switch (getLexer().getKind()) {
  default: return 0;
  case AsmToken::LParen:
  case AsmToken::Plus:
  case AsmToken::Minus:
  case AsmToken::Integer:
  case AsmToken::Identifier:
    if (getParser().ParseExpression(EVal))
      return 0;

    return MBlazeOperand::CreateImm(EVal, S, E);
  }
}

MBlazeOperand *MBlazeAsmParser::
ParseOperand(SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  MBlazeOperand *Op;

  // Attempt to parse the next token as a register name
  unsigned RegNo;
  Op = ParseRegister(RegNo);

  // Attempt to parse the next token as an FSL immediate
  if (!Op)
    Op = ParseFsl();

  // Attempt to parse the next token as an immediate
  if (!Op)
    Op = ParseImmediate();

  // If the token could not be parsed then fail
  if (!Op) {
    Error(Parser.getTok().getLoc(), "unknown operand");
    return 0;
  }

  // Push the parsed operand into the list of operands
  Operands.push_back(Op);
  return Op;
}

/// Parse an mblaze instruction mnemonic followed by its operands.
bool MBlazeAsmParser::
ParseInstruction(ParseInstructionInfo &Info, StringRef Name, SMLoc NameLoc,
                 SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  // The first operands is the token for the instruction name
  size_t dotLoc = Name.find('.');
  Operands.push_back(MBlazeOperand::CreateToken(Name.substr(0,dotLoc),NameLoc));
  if (dotLoc < Name.size())
    Operands.push_back(MBlazeOperand::CreateToken(Name.substr(dotLoc),NameLoc));

  // If there are no more operands then finish
  if (getLexer().is(AsmToken::EndOfStatement))
    return false;

  // Parse the first operand
  if (!ParseOperand(Operands))
    return true;

  while (getLexer().isNot(AsmToken::EndOfStatement) &&
         getLexer().is(AsmToken::Comma)) {
    // Consume the comma token
    getLexer().Lex();

    // Parse the next operand
    if (!ParseOperand(Operands))
      return true;
  }

  // If the instruction requires a memory operand then we need to
  // replace the last two operands (base+offset) with a single
  // memory operand.
  if (Name.startswith("lw") || Name.startswith("sw") ||
      Name.startswith("lh") || Name.startswith("sh") ||
      Name.startswith("lb") || Name.startswith("sb"))
    return (ParseMemory(Operands) == NULL);

  return false;
}

/// ParseDirective parses the MBlaze specific directives
bool MBlazeAsmParser::ParseDirective(AsmToken DirectiveID) {
  StringRef IDVal = DirectiveID.getIdentifier();
  if (IDVal == ".word")
    return ParseDirectiveWord(2, DirectiveID.getLoc());
  return true;
}

/// ParseDirectiveWord
///  ::= .word [ expression (, expression)* ]
bool MBlazeAsmParser::ParseDirectiveWord(unsigned Size, SMLoc L) {
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    for (;;) {
      const MCExpr *Value;
      if (getParser().ParseExpression(Value))
        return true;

      getParser().getStreamer().EmitValue(Value, Size, 0 /*addrspace*/);

      if (getLexer().is(AsmToken::EndOfStatement))
        break;

      // FIXME: Improve diagnostic.
      if (getLexer().isNot(AsmToken::Comma))
        return Error(L, "unexpected token in directive");
      Parser.Lex();
    }
  }

  Parser.Lex();
  return false;
}

extern "C" void LLVMInitializeMBlazeAsmLexer();

/// Force static initialization.
extern "C" void LLVMInitializeMBlazeAsmParser() {
  RegisterMCAsmParser<MBlazeAsmParser> X(TheMBlazeTarget);
  LLVMInitializeMBlazeAsmLexer();
}

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#include "MBlazeGenAsmMatcher.inc"
