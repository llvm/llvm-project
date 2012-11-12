//===-- X86AsmParser.cpp - Parse X86 assembly to MCInst instructions ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X86BaseInfo.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
struct X86Operand;

class X86AsmParser : public MCTargetAsmParser {
  MCSubtargetInfo &STI;
  MCAsmParser &Parser;
  ParseInstructionInfo *InstInfo;
private:
  MCAsmParser &getParser() const { return Parser; }

  MCAsmLexer &getLexer() const { return Parser.getLexer(); }

  bool Error(SMLoc L, const Twine &Msg,
             ArrayRef<SMRange> Ranges = ArrayRef<SMRange>(),
             bool MatchingInlineAsm = false) {
    if (MatchingInlineAsm) return true;
    return Parser.Error(L, Msg, Ranges);
  }

  X86Operand *ErrorOperand(SMLoc Loc, StringRef Msg) {
    Error(Loc, Msg);
    return 0;
  }

  X86Operand *ParseOperand();
  X86Operand *ParseATTOperand();
  X86Operand *ParseIntelOperand();
  X86Operand *ParseIntelOffsetOfOperator(SMLoc StartLoc);
  X86Operand *ParseIntelTypeOperator(SMLoc StartLoc);
  X86Operand *ParseIntelMemOperand(unsigned SegReg, SMLoc StartLoc);
  X86Operand *ParseIntelBracExpression(unsigned SegReg, unsigned Size);
  X86Operand *ParseMemOperand(unsigned SegReg, SMLoc StartLoc);

  bool ParseIntelDotOperator(const MCExpr *Disp, const MCExpr **NewDisp,
                             SmallString<64> &Err);

  bool ParseDirectiveWord(unsigned Size, SMLoc L);
  bool ParseDirectiveCode(StringRef IDVal, SMLoc L);

  bool processInstruction(MCInst &Inst,
                          const SmallVectorImpl<MCParsedAsmOperand*> &Ops);

  bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                               MCStreamer &Out, unsigned &ErrorInfo,
                               bool MatchingInlineAsm);

  /// isSrcOp - Returns true if operand is either (%rsi) or %ds:%(rsi)
  /// in 64bit mode or (%esi) or %es:(%esi) in 32bit mode.
  bool isSrcOp(X86Operand &Op);

  /// isDstOp - Returns true if operand is either (%rdi) or %es:(%rdi)
  /// in 64bit mode or (%edi) or %es:(%edi) in 32bit mode.
  bool isDstOp(X86Operand &Op);

  bool is64BitMode() const {
    // FIXME: Can tablegen auto-generate this?
    return (STI.getFeatureBits() & X86::Mode64Bit) != 0;
  }
  void SwitchMode() {
    unsigned FB = ComputeAvailableFeatures(STI.ToggleFeature(X86::Mode64Bit));
    setAvailableFeatures(FB);
  }

  /// @name Auto-generated Matcher Functions
  /// {

#define GET_ASSEMBLER_HEADER
#include "X86GenAsmMatcher.inc"

  /// }

public:
  X86AsmParser(MCSubtargetInfo &sti, MCAsmParser &parser)
    : MCTargetAsmParser(), STI(sti), Parser(parser), InstInfo(0) {

    // Initialize the set of available features.
    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));
  }
  virtual bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc);

  virtual bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                                SMLoc NameLoc,
                                SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  virtual bool ParseDirective(AsmToken DirectiveID);

  bool isParsingIntelSyntax() {
    return getParser().getAssemblerDialect();
  }
};
} // end anonymous namespace

/// @name Auto-generated Match Functions
/// {

static unsigned MatchRegisterName(StringRef Name);

/// }

static bool isImmSExti16i8Value(uint64_t Value) {
  return ((                                  Value <= 0x000000000000007FULL)||
          (0x000000000000FF80ULL <= Value && Value <= 0x000000000000FFFFULL)||
          (0xFFFFFFFFFFFFFF80ULL <= Value && Value <= 0xFFFFFFFFFFFFFFFFULL));
}

static bool isImmSExti32i8Value(uint64_t Value) {
  return ((                                  Value <= 0x000000000000007FULL)||
          (0x00000000FFFFFF80ULL <= Value && Value <= 0x00000000FFFFFFFFULL)||
          (0xFFFFFFFFFFFFFF80ULL <= Value && Value <= 0xFFFFFFFFFFFFFFFFULL));
}

static bool isImmZExtu32u8Value(uint64_t Value) {
    return (Value <= 0x00000000000000FFULL);
}

static bool isImmSExti64i8Value(uint64_t Value) {
  return ((                                  Value <= 0x000000000000007FULL)||
          (0xFFFFFFFFFFFFFF80ULL <= Value && Value <= 0xFFFFFFFFFFFFFFFFULL));
}

static bool isImmSExti64i32Value(uint64_t Value) {
  return ((                                  Value <= 0x000000007FFFFFFFULL)||
          (0xFFFFFFFF80000000ULL <= Value && Value <= 0xFFFFFFFFFFFFFFFFULL));
}
namespace {

/// X86Operand - Instances of this class represent a parsed X86 machine
/// instruction.
struct X86Operand : public MCParsedAsmOperand {
  enum KindTy {
    Token,
    Register,
    Immediate,
    Memory
  } Kind;

  SMLoc StartLoc, EndLoc;
  SMLoc OffsetOfLoc;

  union {
    struct {
      const char *Data;
      unsigned Length;
    } Tok;

    struct {
      unsigned RegNo;
    } Reg;

    struct {
      const MCExpr *Val;
      bool NeedAsmRewrite;
    } Imm;

    struct {
      unsigned SegReg;
      const MCExpr *Disp;
      unsigned BaseReg;
      unsigned IndexReg;
      unsigned Scale;
      unsigned Size;
      bool NeedSizeDir;
    } Mem;
  };

  X86Operand(KindTy K, SMLoc Start, SMLoc End)
    : Kind(K), StartLoc(Start), EndLoc(End) {}

  /// getStartLoc - Get the location of the first token of this operand.
  SMLoc getStartLoc() const { return StartLoc; }
  /// getEndLoc - Get the location of the last token of this operand.
  SMLoc getEndLoc() const { return EndLoc; }
  /// getLocRange - Get the range between the first and last token of this
  /// operand.
  SMRange getLocRange() const { return SMRange(StartLoc, EndLoc); }
  /// getOffsetOfLoc - Get the location of the offset operator.
  SMLoc getOffsetOfLoc() const { return OffsetOfLoc; }

  virtual void print(raw_ostream &OS) const {}

  StringRef getToken() const {
    assert(Kind == Token && "Invalid access!");
    return StringRef(Tok.Data, Tok.Length);
  }
  void setTokenValue(StringRef Value) {
    assert(Kind == Token && "Invalid access!");
    Tok.Data = Value.data();
    Tok.Length = Value.size();
  }

  unsigned getReg() const {
    assert(Kind == Register && "Invalid access!");
    return Reg.RegNo;
  }

  const MCExpr *getImm() const {
    assert(Kind == Immediate && "Invalid access!");
    return Imm.Val;
  }

  bool needAsmRewrite() const {
    assert(Kind == Immediate && "Invalid access!");
    return Imm.NeedAsmRewrite;
  }

  const MCExpr *getMemDisp() const {
    assert(Kind == Memory && "Invalid access!");
    return Mem.Disp;
  }
  unsigned getMemSegReg() const {
    assert(Kind == Memory && "Invalid access!");
    return Mem.SegReg;
  }
  unsigned getMemBaseReg() const {
    assert(Kind == Memory && "Invalid access!");
    return Mem.BaseReg;
  }
  unsigned getMemIndexReg() const {
    assert(Kind == Memory && "Invalid access!");
    return Mem.IndexReg;
  }
  unsigned getMemScale() const {
    assert(Kind == Memory && "Invalid access!");
    return Mem.Scale;
  }

  bool isToken() const {return Kind == Token; }

  bool isImm() const { return Kind == Immediate; }

  bool isImmSExti16i8() const {
    if (!isImm())
      return false;

    // If this isn't a constant expr, just assume it fits and let relaxation
    // handle it.
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE)
      return true;

    // Otherwise, check the value is in a range that makes sense for this
    // extension.
    return isImmSExti16i8Value(CE->getValue());
  }
  bool isImmSExti32i8() const {
    if (!isImm())
      return false;

    // If this isn't a constant expr, just assume it fits and let relaxation
    // handle it.
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE)
      return true;

    // Otherwise, check the value is in a range that makes sense for this
    // extension.
    return isImmSExti32i8Value(CE->getValue());
  }
  bool isImmZExtu32u8() const {
    if (!isImm())
      return false;

    // If this isn't a constant expr, just assume it fits and let relaxation
    // handle it.
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE)
      return true;

    // Otherwise, check the value is in a range that makes sense for this
    // extension.
    return isImmZExtu32u8Value(CE->getValue());
  }
  bool isImmSExti64i8() const {
    if (!isImm())
      return false;

    // If this isn't a constant expr, just assume it fits and let relaxation
    // handle it.
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE)
      return true;

    // Otherwise, check the value is in a range that makes sense for this
    // extension.
    return isImmSExti64i8Value(CE->getValue());
  }
  bool isImmSExti64i32() const {
    if (!isImm())
      return false;

    // If this isn't a constant expr, just assume it fits and let relaxation
    // handle it.
    const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getImm());
    if (!CE)
      return true;

    // Otherwise, check the value is in a range that makes sense for this
    // extension.
    return isImmSExti64i32Value(CE->getValue());
  }

  unsigned getMemSize() const {
    assert(Kind == Memory && "Invalid access!");
    return Mem.Size;
  }

  bool isOffsetOf() const {
    return OffsetOfLoc.getPointer();
  }

  bool needSizeDirective() const {
    assert(Kind == Memory && "Invalid access!");
    return Mem.NeedSizeDir;
  }

  bool isMem() const { return Kind == Memory; }
  bool isMem8() const {
    return Kind == Memory && (!Mem.Size || Mem.Size == 8);
  }
  bool isMem16() const {
    return Kind == Memory && (!Mem.Size || Mem.Size == 16);
  }
  bool isMem32() const {
    return Kind == Memory && (!Mem.Size || Mem.Size == 32);
  }
  bool isMem64() const {
    return Kind == Memory && (!Mem.Size || Mem.Size == 64);
  }
  bool isMem80() const {
    return Kind == Memory && (!Mem.Size || Mem.Size == 80);
  }
  bool isMem128() const {
    return Kind == Memory && (!Mem.Size || Mem.Size == 128);
  }
  bool isMem256() const {
    return Kind == Memory && (!Mem.Size || Mem.Size == 256);
  }

  bool isMemVX32() const {
    return Kind == Memory && (!Mem.Size || Mem.Size == 32) &&
      getMemIndexReg() >= X86::XMM0 && getMemIndexReg() <= X86::XMM15;
  }
  bool isMemVY32() const {
    return Kind == Memory && (!Mem.Size || Mem.Size == 32) &&
      getMemIndexReg() >= X86::YMM0 && getMemIndexReg() <= X86::YMM15;
  }
  bool isMemVX64() const {
    return Kind == Memory && (!Mem.Size || Mem.Size == 64) &&
      getMemIndexReg() >= X86::XMM0 && getMemIndexReg() <= X86::XMM15;
  }
  bool isMemVY64() const {
    return Kind == Memory && (!Mem.Size || Mem.Size == 64) &&
      getMemIndexReg() >= X86::YMM0 && getMemIndexReg() <= X86::YMM15;
  }

  bool isAbsMem() const {
    return Kind == Memory && !getMemSegReg() && !getMemBaseReg() &&
      !getMemIndexReg() && getMemScale() == 1;
  }

  bool isReg() const { return Kind == Register; }

  void addExpr(MCInst &Inst, const MCExpr *Expr) const {
    // Add as immediates when possible.
    if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(Expr))
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

  void addMem8Operands(MCInst &Inst, unsigned N) const {
    addMemOperands(Inst, N);
  }
  void addMem16Operands(MCInst &Inst, unsigned N) const {
    addMemOperands(Inst, N);
  }
  void addMem32Operands(MCInst &Inst, unsigned N) const {
    addMemOperands(Inst, N);
  }
  void addMem64Operands(MCInst &Inst, unsigned N) const {
    addMemOperands(Inst, N);
  }
  void addMem80Operands(MCInst &Inst, unsigned N) const {
    addMemOperands(Inst, N);
  }
  void addMem128Operands(MCInst &Inst, unsigned N) const {
    addMemOperands(Inst, N);
  }
  void addMem256Operands(MCInst &Inst, unsigned N) const {
    addMemOperands(Inst, N);
  }
  void addMemVX32Operands(MCInst &Inst, unsigned N) const {
    addMemOperands(Inst, N);
  }
  void addMemVY32Operands(MCInst &Inst, unsigned N) const {
    addMemOperands(Inst, N);
  }
  void addMemVX64Operands(MCInst &Inst, unsigned N) const {
    addMemOperands(Inst, N);
  }
  void addMemVY64Operands(MCInst &Inst, unsigned N) const {
    addMemOperands(Inst, N);
  }

  void addMemOperands(MCInst &Inst, unsigned N) const {
    assert((N == 5) && "Invalid number of operands!");
    Inst.addOperand(MCOperand::CreateReg(getMemBaseReg()));
    Inst.addOperand(MCOperand::CreateImm(getMemScale()));
    Inst.addOperand(MCOperand::CreateReg(getMemIndexReg()));
    addExpr(Inst, getMemDisp());
    Inst.addOperand(MCOperand::CreateReg(getMemSegReg()));
  }

  void addAbsMemOperands(MCInst &Inst, unsigned N) const {
    assert((N == 1) && "Invalid number of operands!");
    // Add as immediates when possible.
    if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getMemDisp()))
      Inst.addOperand(MCOperand::CreateImm(CE->getValue()));
    else
      Inst.addOperand(MCOperand::CreateExpr(getMemDisp()));
  }

  static X86Operand *CreateToken(StringRef Str, SMLoc Loc) {
    SMLoc EndLoc = SMLoc::getFromPointer(Loc.getPointer() + Str.size() - 1);
    X86Operand *Res = new X86Operand(Token, Loc, EndLoc);
    Res->Tok.Data = Str.data();
    Res->Tok.Length = Str.size();
    return Res;
  }

  static X86Operand *CreateReg(unsigned RegNo, SMLoc StartLoc, SMLoc EndLoc,
                               SMLoc OffsetOfLoc = SMLoc()) {
    X86Operand *Res = new X86Operand(Register, StartLoc, EndLoc);
    Res->Reg.RegNo = RegNo;
    Res->OffsetOfLoc = OffsetOfLoc;
    return Res;
  }

  static X86Operand *CreateImm(const MCExpr *Val, SMLoc StartLoc, SMLoc EndLoc,
                               bool NeedRewrite = true){
    X86Operand *Res = new X86Operand(Immediate, StartLoc, EndLoc);
    Res->Imm.Val = Val;
    Res->Imm.NeedAsmRewrite = NeedRewrite;
    return Res;
  }

  /// Create an absolute memory operand.
  static X86Operand *CreateMem(const MCExpr *Disp, SMLoc StartLoc, SMLoc EndLoc,
                               unsigned Size = 0, bool NeedSizeDir = false){
    X86Operand *Res = new X86Operand(Memory, StartLoc, EndLoc);
    Res->Mem.SegReg   = 0;
    Res->Mem.Disp     = Disp;
    Res->Mem.BaseReg  = 0;
    Res->Mem.IndexReg = 0;
    Res->Mem.Scale    = 1;
    Res->Mem.Size     = Size;
    Res->Mem.NeedSizeDir = NeedSizeDir;
    return Res;
  }

  /// Create a generalized memory operand.
  static X86Operand *CreateMem(unsigned SegReg, const MCExpr *Disp,
                               unsigned BaseReg, unsigned IndexReg,
                               unsigned Scale, SMLoc StartLoc, SMLoc EndLoc,
                               unsigned Size = 0, bool NeedSizeDir = false) {
    // We should never just have a displacement, that should be parsed as an
    // absolute memory operand.
    assert((SegReg || BaseReg || IndexReg) && "Invalid memory operand!");

    // The scale should always be one of {1,2,4,8}.
    assert(((Scale == 1 || Scale == 2 || Scale == 4 || Scale == 8)) &&
           "Invalid scale!");
    X86Operand *Res = new X86Operand(Memory, StartLoc, EndLoc);
    Res->Mem.SegReg   = SegReg;
    Res->Mem.Disp     = Disp;
    Res->Mem.BaseReg  = BaseReg;
    Res->Mem.IndexReg = IndexReg;
    Res->Mem.Scale    = Scale;
    Res->Mem.Size     = Size;
    Res->Mem.NeedSizeDir = NeedSizeDir;
    return Res;
  }
};

} // end anonymous namespace.

bool X86AsmParser::isSrcOp(X86Operand &Op) {
  unsigned basereg = is64BitMode() ? X86::RSI : X86::ESI;

  return (Op.isMem() &&
    (Op.Mem.SegReg == 0 || Op.Mem.SegReg == X86::DS) &&
    isa<MCConstantExpr>(Op.Mem.Disp) &&
    cast<MCConstantExpr>(Op.Mem.Disp)->getValue() == 0 &&
    Op.Mem.BaseReg == basereg && Op.Mem.IndexReg == 0);
}

bool X86AsmParser::isDstOp(X86Operand &Op) {
  unsigned basereg = is64BitMode() ? X86::RDI : X86::EDI;

  return Op.isMem() &&
    (Op.Mem.SegReg == 0 || Op.Mem.SegReg == X86::ES) &&
    isa<MCConstantExpr>(Op.Mem.Disp) &&
    cast<MCConstantExpr>(Op.Mem.Disp)->getValue() == 0 &&
    Op.Mem.BaseReg == basereg && Op.Mem.IndexReg == 0;
}

bool X86AsmParser::ParseRegister(unsigned &RegNo,
                                 SMLoc &StartLoc, SMLoc &EndLoc) {
  RegNo = 0;
  const AsmToken &PercentTok = Parser.getTok();
  StartLoc = PercentTok.getLoc();

  // If we encounter a %, ignore it. This code handles registers with and
  // without the prefix, unprefixed registers can occur in cfi directives.
  if (!isParsingIntelSyntax() && PercentTok.is(AsmToken::Percent))
    Parser.Lex(); // Eat percent token.

  const AsmToken &Tok = Parser.getTok();
  if (Tok.isNot(AsmToken::Identifier)) {
    if (isParsingIntelSyntax()) return true;
    return Error(StartLoc, "invalid register name",
                 SMRange(StartLoc, Tok.getEndLoc()));
  }

  RegNo = MatchRegisterName(Tok.getString());

  // If the match failed, try the register name as lowercase.
  if (RegNo == 0)
    RegNo = MatchRegisterName(Tok.getString().lower());

  if (!is64BitMode()) {
    // FIXME: This should be done using Requires<In32BitMode> and
    // Requires<In64BitMode> so "eiz" usage in 64-bit instructions can be also
    // checked.
    // FIXME: Check AH, CH, DH, BH cannot be used in an instruction requiring a
    // REX prefix.
    if (RegNo == X86::RIZ ||
        X86MCRegisterClasses[X86::GR64RegClassID].contains(RegNo) ||
        X86II::isX86_64NonExtLowByteReg(RegNo) ||
        X86II::isX86_64ExtendedReg(RegNo))
      return Error(StartLoc, "register %"
                   + Tok.getString() + " is only available in 64-bit mode",
                   SMRange(StartLoc, Tok.getEndLoc()));
  }

  // Parse "%st" as "%st(0)" and "%st(1)", which is multiple tokens.
  if (RegNo == 0 && (Tok.getString() == "st" || Tok.getString() == "ST")) {
    RegNo = X86::ST0;
    EndLoc = Tok.getLoc();
    Parser.Lex(); // Eat 'st'

    // Check to see if we have '(4)' after %st.
    if (getLexer().isNot(AsmToken::LParen))
      return false;
    // Lex the paren.
    getParser().Lex();

    const AsmToken &IntTok = Parser.getTok();
    if (IntTok.isNot(AsmToken::Integer))
      return Error(IntTok.getLoc(), "expected stack index");
    switch (IntTok.getIntVal()) {
    case 0: RegNo = X86::ST0; break;
    case 1: RegNo = X86::ST1; break;
    case 2: RegNo = X86::ST2; break;
    case 3: RegNo = X86::ST3; break;
    case 4: RegNo = X86::ST4; break;
    case 5: RegNo = X86::ST5; break;
    case 6: RegNo = X86::ST6; break;
    case 7: RegNo = X86::ST7; break;
    default: return Error(IntTok.getLoc(), "invalid stack index");
    }

    if (getParser().Lex().isNot(AsmToken::RParen))
      return Error(Parser.getTok().getLoc(), "expected ')'");

    EndLoc = Tok.getLoc();
    Parser.Lex(); // Eat ')'
    return false;
  }

  // If this is "db[0-7]", match it as an alias
  // for dr[0-7].
  if (RegNo == 0 && Tok.getString().size() == 3 &&
      Tok.getString().startswith("db")) {
    switch (Tok.getString()[2]) {
    case '0': RegNo = X86::DR0; break;
    case '1': RegNo = X86::DR1; break;
    case '2': RegNo = X86::DR2; break;
    case '3': RegNo = X86::DR3; break;
    case '4': RegNo = X86::DR4; break;
    case '5': RegNo = X86::DR5; break;
    case '6': RegNo = X86::DR6; break;
    case '7': RegNo = X86::DR7; break;
    }

    if (RegNo != 0) {
      EndLoc = Tok.getLoc();
      Parser.Lex(); // Eat it.
      return false;
    }
  }

  if (RegNo == 0) {
    if (isParsingIntelSyntax()) return true;
    return Error(StartLoc, "invalid register name",
                 SMRange(StartLoc, Tok.getEndLoc()));
  }

  EndLoc = Tok.getEndLoc();
  Parser.Lex(); // Eat identifier token.
  return false;
}

X86Operand *X86AsmParser::ParseOperand() {
  if (isParsingIntelSyntax())
    return ParseIntelOperand();
  return ParseATTOperand();
}

/// getIntelMemOperandSize - Return intel memory operand size.
static unsigned getIntelMemOperandSize(StringRef OpStr) {
  unsigned Size = StringSwitch<unsigned>(OpStr)
    .Cases("BYTE", "byte", 8)
    .Cases("WORD", "word", 16)
    .Cases("DWORD", "dword", 32)
    .Cases("QWORD", "qword", 64)
    .Cases("XWORD", "xword", 80)
    .Cases("XMMWORD", "xmmword", 128)
    .Cases("YMMWORD", "ymmword", 256)
    .Default(0);
  return Size;
}

X86Operand *X86AsmParser::ParseIntelBracExpression(unsigned SegReg, 
                                                   unsigned Size) {
  unsigned BaseReg = 0, IndexReg = 0, Scale = 1;
  const AsmToken &Tok = Parser.getTok();
  SMLoc Start = Tok.getLoc(), End;

  const MCExpr *Disp = MCConstantExpr::Create(0, getContext());
  // Parse [ BaseReg + Scale*IndexReg + Disp ] or [ symbol ]

  // Eat '['
  if (getLexer().isNot(AsmToken::LBrac))
    return ErrorOperand(Start, "Expected '[' token!");
  Parser.Lex();

  if (getLexer().is(AsmToken::Identifier)) {
    // Parse BaseReg
    if (ParseRegister(BaseReg, Start, End)) {
      // Handle '[' 'symbol' ']'
      if (getParser().ParseExpression(Disp, End)) return 0;
      if (getLexer().isNot(AsmToken::RBrac))
        return ErrorOperand(Start, "Expected ']' token!");
      Parser.Lex();
      End = Tok.getLoc();
      return X86Operand::CreateMem(Disp, Start, End, Size);
    }
  } else if (getLexer().is(AsmToken::Integer)) {
      int64_t Val = Tok.getIntVal();
      Parser.Lex();
      SMLoc Loc = Tok.getLoc();
      if (getLexer().is(AsmToken::RBrac)) {
        // Handle '[' number ']'
        Parser.Lex();
        End = Tok.getLoc();
        const MCExpr *Disp = MCConstantExpr::Create(Val, getContext());
        if (SegReg)
          return X86Operand::CreateMem(SegReg, Disp, 0, 0, Scale,
                                       Start, End, Size);
        return X86Operand::CreateMem(Disp, Start, End, Size);
      } else if (getLexer().is(AsmToken::Star)) {
        // Handle '[' Scale*IndexReg ']'
        Parser.Lex();
        SMLoc IdxRegLoc = Tok.getLoc();
        if (ParseRegister(IndexReg, IdxRegLoc, End))
          return ErrorOperand(IdxRegLoc, "Expected register");
        Scale = Val;
      } else
        return ErrorOperand(Loc, "Unexpected token");
  }

  // Parse ][ as a plus.
  bool ExpectRBrac = true;
  if (getLexer().is(AsmToken::RBrac)) {
    ExpectRBrac = false;
    Parser.Lex();
    End = Tok.getLoc();
  }

  if (getLexer().is(AsmToken::Plus) || getLexer().is(AsmToken::Minus) ||
      getLexer().is(AsmToken::LBrac)) {
    ExpectRBrac = true;
    bool isPlus = getLexer().is(AsmToken::Plus) ||
      getLexer().is(AsmToken::LBrac);
    Parser.Lex(); 
    SMLoc PlusLoc = Tok.getLoc();
    if (getLexer().is(AsmToken::Integer)) {
      int64_t Val = Tok.getIntVal();
      Parser.Lex();
      if (getLexer().is(AsmToken::Star)) {
        Parser.Lex();
        SMLoc IdxRegLoc = Tok.getLoc();
        if (ParseRegister(IndexReg, IdxRegLoc, End))
          return ErrorOperand(IdxRegLoc, "Expected register");
        Scale = Val;
      } else if (getLexer().is(AsmToken::RBrac)) {
        const MCExpr *ValExpr = MCConstantExpr::Create(Val, getContext());
        Disp = isPlus ? ValExpr : MCConstantExpr::Create(0-Val, getContext());
      } else
        return ErrorOperand(PlusLoc, "unexpected token after +");
    } else if (getLexer().is(AsmToken::Identifier)) {
      // This could be an index register or a displacement expression.
      End = Tok.getLoc();
      if (!IndexReg)
        ParseRegister(IndexReg, Start, End);
      else if (getParser().ParseExpression(Disp, End)) return 0;
    }
  }
  
  // Parse ][ as a plus.
  if (getLexer().is(AsmToken::RBrac)) {
    ExpectRBrac = false;
    Parser.Lex();
    End = Tok.getLoc();
    if (getLexer().is(AsmToken::LBrac)) {
      ExpectRBrac = true;
      Parser.Lex();
      if (getParser().ParseExpression(Disp, End))
        return 0;
    }
  } else if (ExpectRBrac) {
      if (getParser().ParseExpression(Disp, End))
        return 0;
  }

  if (ExpectRBrac) {
    if (getLexer().isNot(AsmToken::RBrac))
      return ErrorOperand(End, "expected ']' token!");
    Parser.Lex();
    End = Tok.getLoc();
  }

  // Parse the dot operator (e.g., [ebx].foo.bar).
  if (Tok.getString().startswith(".")) {
    SmallString<64> Err;
    const MCExpr *NewDisp;
    if (ParseIntelDotOperator(Disp, &NewDisp, Err))
      return ErrorOperand(Tok.getLoc(), Err);
    
    Parser.Lex();  // Eat the field.
    Disp = NewDisp;
  }

  End = Tok.getLoc();

  // handle [-42]
  if (!BaseReg && !IndexReg)
    return X86Operand::CreateMem(Disp, Start, End, Size);

  return X86Operand::CreateMem(SegReg, Disp, BaseReg, IndexReg, Scale,
                               Start, End, Size);
}

/// ParseIntelMemOperand - Parse intel style memory operand.
X86Operand *X86AsmParser::ParseIntelMemOperand(unsigned SegReg, SMLoc Start) {
  const AsmToken &Tok = Parser.getTok();
  SMLoc End;

  unsigned Size = getIntelMemOperandSize(Tok.getString());
  if (Size) {
    Parser.Lex();
    assert ((Tok.getString() == "PTR" || Tok.getString() == "ptr") &&
            "Unexpected token!");
    Parser.Lex();
  }

  if (getLexer().is(AsmToken::LBrac))
    return ParseIntelBracExpression(SegReg, Size);

  if (!ParseRegister(SegReg, Start, End)) {
    // Handel SegReg : [ ... ]
    if (getLexer().isNot(AsmToken::Colon))
      return ErrorOperand(Start, "Expected ':' token!");
    Parser.Lex(); // Eat :
    if (getLexer().isNot(AsmToken::LBrac))
      return ErrorOperand(Start, "Expected '[' token!");
    return ParseIntelBracExpression(SegReg, Size);
  }

  const MCExpr *Disp = MCConstantExpr::Create(0, getParser().getContext());
  if (getParser().ParseExpression(Disp, End)) return 0;
  End = Parser.getTok().getLoc();

  bool NeedSizeDir = false;
  if (!Size && isParsingInlineAsm()) {
    if (const MCSymbolRefExpr *SymRef = dyn_cast<MCSymbolRefExpr>(Disp)) {
      const MCSymbol &Sym = SymRef->getSymbol();
      // FIXME: The SemaLookup will fail if the name is anything other then an
      // identifier.
      // FIXME: Pass a valid SMLoc.
      SemaCallback->LookupInlineAsmIdentifier(Sym.getName(), NULL, Size);
      NeedSizeDir = Size > 0;
    }
  }
  if (!isParsingInlineAsm())
    return X86Operand::CreateMem(Disp, Start, End, Size);
  else
    // When parsing inline assembly we set the base register to a non-zero value
    // as we don't know the actual value at this time.  This is necessary to
    // get the matching correct in some cases.
    return X86Operand::CreateMem(/*SegReg*/0, Disp, /*BaseReg*/1, /*IndexReg*/0,
                                 /*Scale*/1, Start, End, Size, NeedSizeDir);
}

/// Parse the '.' operator.
bool X86AsmParser::ParseIntelDotOperator(const MCExpr *Disp,
                                         const MCExpr **NewDisp,
                                         SmallString<64> &Err) {
  AsmToken Tok = *&Parser.getTok();
  uint64_t OrigDispVal, DotDispVal;

  // FIXME: Handle non-constant expressions.
  if (const MCConstantExpr *OrigDisp = dyn_cast<MCConstantExpr>(Disp)) {
    OrigDispVal = OrigDisp->getValue();
  } else {
    Err = "Non-constant offsets are not supported!";
    return true;
  }

  // Drop the '.'.
  StringRef DotDispStr = Tok.getString().drop_front(1);

  // .Imm gets lexed as a real.
  if (Tok.is(AsmToken::Real)) {
    APInt DotDisp;
    DotDispStr.getAsInteger(10, DotDisp);
    DotDispVal = DotDisp.getZExtValue();
  } else if (Tok.is(AsmToken::Identifier)) {
    // We should only see an identifier when parsing the original inline asm.
    // The front-end should rewrite this in terms of immediates.
    assert (isParsingInlineAsm() && "Unexpected field name!");

    unsigned DotDisp;
    std::pair<StringRef, StringRef> BaseMember = DotDispStr.split('.');
    if (SemaCallback->LookupInlineAsmField(BaseMember.first, BaseMember.second,
                                           DotDisp)) {
      Err = "Unable to lookup field reference!";
      return true;
    }
    DotDispVal = DotDisp;
  } else {
    Err = "Unexpected token type!";
    return true;
  }

  if (isParsingInlineAsm() && Tok.is(AsmToken::Identifier)) {
    SMLoc Loc = SMLoc::getFromPointer(DotDispStr.data());
    unsigned Len = DotDispStr.size();
    unsigned Val = OrigDispVal + DotDispVal;
    InstInfo->AsmRewrites->push_back(AsmRewrite(AOK_DotOperator, Loc, Len,
                                                Val));
  }

  *NewDisp = MCConstantExpr::Create(OrigDispVal + DotDispVal, getContext());
  return false;
}

/// Parse the 'offset' operator.  This operator is used to specify the
/// location rather then the content of a variable.
X86Operand *X86AsmParser::ParseIntelOffsetOfOperator(SMLoc Start) {
  SMLoc OffsetOfLoc = Start;
  Parser.Lex(); // Eat offset.
  Start = Parser.getTok().getLoc();
  assert (Parser.getTok().is(AsmToken::Identifier) && "Expected an identifier");

  SMLoc End;
  const MCExpr *Val;
  if (getParser().ParseExpression(Val, End))
    return ErrorOperand(Start, "Unable to parse expression!");

  End = Parser.getTok().getLoc();

  // Don't emit the offset operator.
  InstInfo->AsmRewrites->push_back(AsmRewrite(AOK_Skip, OffsetOfLoc, 7));

  // The offset operator will have an 'r' constraint, thus we need to create
  // register operand to ensure proper matching.  Just pick a GPR based on
  // the size of a pointer.
  unsigned RegNo = is64BitMode() ? X86::RBX : X86::EBX;
  return X86Operand::CreateReg(RegNo, Start, End, OffsetOfLoc);
}

/// Parse the 'TYPE' operator.  The TYPE operator returns the size of a C or
/// C++ type or variable. If the variable is an array, TYPE returns the size of
/// a single element of the array.
X86Operand *X86AsmParser::ParseIntelTypeOperator(SMLoc Start) {
  SMLoc TypeLoc = Start;
  Parser.Lex(); // Eat offset.
  Start = Parser.getTok().getLoc();
  assert (Parser.getTok().is(AsmToken::Identifier) && "Expected an identifier");

  SMLoc End;
  const MCExpr *Val;
  if (getParser().ParseExpression(Val, End))
    return 0;

  End = Parser.getTok().getLoc();

  unsigned Size = 0;
  if (const MCSymbolRefExpr *SymRef = dyn_cast<MCSymbolRefExpr>(Val)) {
    const MCSymbol &Sym = SymRef->getSymbol();
    // FIXME: The SemaLookup will fail if the name is anything other then an
    // identifier.
    // FIXME: Pass a valid SMLoc.
    if (!SemaCallback->LookupInlineAsmIdentifier(Sym.getName(), NULL, Size))
      return ErrorOperand(Start, "Unable to lookup TYPE of expr!");

    Size /= 8; // Size is in terms of bits, but we want bytes in the context.
  }

  // Rewrite the type operator and the C or C++ type or variable in terms of an
  // immediate.  E.g. TYPE foo -> $$4
  unsigned Len = End.getPointer() - TypeLoc.getPointer();
  InstInfo->AsmRewrites->push_back(AsmRewrite(AOK_Imm, TypeLoc, Len, Size));

  const MCExpr *Imm = MCConstantExpr::Create(Size, getContext());
  return X86Operand::CreateImm(Imm, Start, End, /*NeedAsmRewrite*/false);
}

X86Operand *X86AsmParser::ParseIntelOperand() {
  SMLoc Start = Parser.getTok().getLoc(), End;

  // offset operator.
  StringRef AsmTokStr = Parser.getTok().getString();
  if ((AsmTokStr == "offset" || AsmTokStr == "OFFSET") &&
      isParsingInlineAsm())
    return ParseIntelOffsetOfOperator(Start);

  // Type directive.
  if ((AsmTokStr == "type" || AsmTokStr == "TYPE") &&
      isParsingInlineAsm())
    return ParseIntelTypeOperator(Start);

  // Unsupported directives.
  if (isParsingIntelSyntax() &&
      (AsmTokStr == "size" || AsmTokStr == "SIZE" ||
       AsmTokStr == "length" || AsmTokStr == "LENGTH"))
      return ErrorOperand(Start, "Unsupported directive!");

  // immediate.
  if (getLexer().is(AsmToken::Integer) || getLexer().is(AsmToken::Real) ||
      getLexer().is(AsmToken::Minus)) {
    const MCExpr *Val;
    if (!getParser().ParseExpression(Val, End)) {
      End = Parser.getTok().getLoc();
      return X86Operand::CreateImm(Val, Start, End);
    }
  }

  // register
  unsigned RegNo = 0;
  if (!ParseRegister(RegNo, Start, End)) {
    // If this is a segment register followed by a ':', then this is the start
    // of a memory reference, otherwise this is a normal register reference.
    if (getLexer().isNot(AsmToken::Colon))
      return X86Operand::CreateReg(RegNo, Start, Parser.getTok().getLoc());

    getParser().Lex(); // Eat the colon.
    return ParseIntelMemOperand(RegNo, Start);
  }

  // mem operand
  return ParseIntelMemOperand(0, Start);
}

X86Operand *X86AsmParser::ParseATTOperand() {
  switch (getLexer().getKind()) {
  default:
    // Parse a memory operand with no segment register.
    return ParseMemOperand(0, Parser.getTok().getLoc());
  case AsmToken::Percent: {
    // Read the register.
    unsigned RegNo;
    SMLoc Start, End;
    if (ParseRegister(RegNo, Start, End)) return 0;
    if (RegNo == X86::EIZ || RegNo == X86::RIZ) {
      Error(Start, "%eiz and %riz can only be used as index registers",
            SMRange(Start, End));
      return 0;
    }

    // If this is a segment register followed by a ':', then this is the start
    // of a memory reference, otherwise this is a normal register reference.
    if (getLexer().isNot(AsmToken::Colon))
      return X86Operand::CreateReg(RegNo, Start, End);


    getParser().Lex(); // Eat the colon.
    return ParseMemOperand(RegNo, Start);
  }
  case AsmToken::Dollar: {
    // $42 -> immediate.
    SMLoc Start = Parser.getTok().getLoc(), End;
    Parser.Lex();
    const MCExpr *Val;
    if (getParser().ParseExpression(Val, End))
      return 0;
    return X86Operand::CreateImm(Val, Start, End);
  }
  }
}

/// ParseMemOperand: segment: disp(basereg, indexreg, scale).  The '%ds:' prefix
/// has already been parsed if present.
X86Operand *X86AsmParser::ParseMemOperand(unsigned SegReg, SMLoc MemStart) {

  // We have to disambiguate a parenthesized expression "(4+5)" from the start
  // of a memory operand with a missing displacement "(%ebx)" or "(,%eax)".  The
  // only way to do this without lookahead is to eat the '(' and see what is
  // after it.
  const MCExpr *Disp = MCConstantExpr::Create(0, getParser().getContext());
  if (getLexer().isNot(AsmToken::LParen)) {
    SMLoc ExprEnd;
    if (getParser().ParseExpression(Disp, ExprEnd)) return 0;

    // After parsing the base expression we could either have a parenthesized
    // memory address or not.  If not, return now.  If so, eat the (.
    if (getLexer().isNot(AsmToken::LParen)) {
      // Unless we have a segment register, treat this as an immediate.
      if (SegReg == 0)
        return X86Operand::CreateMem(Disp, MemStart, ExprEnd);
      return X86Operand::CreateMem(SegReg, Disp, 0, 0, 1, MemStart, ExprEnd);
    }

    // Eat the '('.
    Parser.Lex();
  } else {
    // Okay, we have a '('.  We don't know if this is an expression or not, but
    // so we have to eat the ( to see beyond it.
    SMLoc LParenLoc = Parser.getTok().getLoc();
    Parser.Lex(); // Eat the '('.

    if (getLexer().is(AsmToken::Percent) || getLexer().is(AsmToken::Comma)) {
      // Nothing to do here, fall into the code below with the '(' part of the
      // memory operand consumed.
    } else {
      SMLoc ExprEnd;

      // It must be an parenthesized expression, parse it now.
      if (getParser().ParseParenExpression(Disp, ExprEnd))
        return 0;

      // After parsing the base expression we could either have a parenthesized
      // memory address or not.  If not, return now.  If so, eat the (.
      if (getLexer().isNot(AsmToken::LParen)) {
        // Unless we have a segment register, treat this as an immediate.
        if (SegReg == 0)
          return X86Operand::CreateMem(Disp, LParenLoc, ExprEnd);
        return X86Operand::CreateMem(SegReg, Disp, 0, 0, 1, MemStart, ExprEnd);
      }

      // Eat the '('.
      Parser.Lex();
    }
  }

  // If we reached here, then we just ate the ( of the memory operand.  Process
  // the rest of the memory operand.
  unsigned BaseReg = 0, IndexReg = 0, Scale = 1;
  SMLoc IndexLoc;

  if (getLexer().is(AsmToken::Percent)) {
    SMLoc StartLoc, EndLoc;
    if (ParseRegister(BaseReg, StartLoc, EndLoc)) return 0;
    if (BaseReg == X86::EIZ || BaseReg == X86::RIZ) {
      Error(StartLoc, "eiz and riz can only be used as index registers",
            SMRange(StartLoc, EndLoc));
      return 0;
    }
  }

  if (getLexer().is(AsmToken::Comma)) {
    Parser.Lex(); // Eat the comma.
    IndexLoc = Parser.getTok().getLoc();

    // Following the comma we should have either an index register, or a scale
    // value. We don't support the later form, but we want to parse it
    // correctly.
    //
    // Not that even though it would be completely consistent to support syntax
    // like "1(%eax,,1)", the assembler doesn't. Use "eiz" or "riz" for this.
    if (getLexer().is(AsmToken::Percent)) {
      SMLoc L;
      if (ParseRegister(IndexReg, L, L)) return 0;

      if (getLexer().isNot(AsmToken::RParen)) {
        // Parse the scale amount:
        //  ::= ',' [scale-expression]
        if (getLexer().isNot(AsmToken::Comma)) {
          Error(Parser.getTok().getLoc(),
                "expected comma in scale expression");
          return 0;
        }
        Parser.Lex(); // Eat the comma.

        if (getLexer().isNot(AsmToken::RParen)) {
          SMLoc Loc = Parser.getTok().getLoc();

          int64_t ScaleVal;
          if (getParser().ParseAbsoluteExpression(ScaleVal)){
            Error(Loc, "expected scale expression");
            return 0;
          }

          // Validate the scale amount.
          if (ScaleVal != 1 && ScaleVal != 2 && ScaleVal != 4 && ScaleVal != 8){
            Error(Loc, "scale factor in address must be 1, 2, 4 or 8");
            return 0;
          }
          Scale = (unsigned)ScaleVal;
        }
      }
    } else if (getLexer().isNot(AsmToken::RParen)) {
      // A scale amount without an index is ignored.
      // index.
      SMLoc Loc = Parser.getTok().getLoc();

      int64_t Value;
      if (getParser().ParseAbsoluteExpression(Value))
        return 0;

      if (Value != 1)
        Warning(Loc, "scale factor without index register is ignored");
      Scale = 1;
    }
  }

  // Ok, we've eaten the memory operand, verify we have a ')' and eat it too.
  if (getLexer().isNot(AsmToken::RParen)) {
    Error(Parser.getTok().getLoc(), "unexpected token in memory operand");
    return 0;
  }
  SMLoc MemEnd = Parser.getTok().getLoc();
  Parser.Lex(); // Eat the ')'.

  // If we have both a base register and an index register make sure they are
  // both 64-bit or 32-bit registers.
  // To support VSIB, IndexReg can be 128-bit or 256-bit registers.
  if (BaseReg != 0 && IndexReg != 0) {
    if (X86MCRegisterClasses[X86::GR64RegClassID].contains(BaseReg) &&
        (X86MCRegisterClasses[X86::GR16RegClassID].contains(IndexReg) ||
         X86MCRegisterClasses[X86::GR32RegClassID].contains(IndexReg)) &&
        IndexReg != X86::RIZ) {
      Error(IndexLoc, "index register is 32-bit, but base register is 64-bit");
      return 0;
    }
    if (X86MCRegisterClasses[X86::GR32RegClassID].contains(BaseReg) &&
        (X86MCRegisterClasses[X86::GR16RegClassID].contains(IndexReg) ||
         X86MCRegisterClasses[X86::GR64RegClassID].contains(IndexReg)) &&
        IndexReg != X86::EIZ){
      Error(IndexLoc, "index register is 64-bit, but base register is 32-bit");
      return 0;
    }
  }

  return X86Operand::CreateMem(SegReg, Disp, BaseReg, IndexReg, Scale,
                               MemStart, MemEnd);
}

bool X86AsmParser::
ParseInstruction(ParseInstructionInfo &Info, StringRef Name, SMLoc NameLoc,
                 SmallVectorImpl<MCParsedAsmOperand*> &Operands) {
  InstInfo = &Info;
  StringRef PatchedName = Name;

  // FIXME: Hack to recognize setneb as setne.
  if (PatchedName.startswith("set") && PatchedName.endswith("b") &&
      PatchedName != "setb" && PatchedName != "setnb")
    PatchedName = PatchedName.substr(0, Name.size()-1);

  // FIXME: Hack to recognize cmp<comparison code>{ss,sd,ps,pd}.
  const MCExpr *ExtraImmOp = 0;
  if ((PatchedName.startswith("cmp") || PatchedName.startswith("vcmp")) &&
      (PatchedName.endswith("ss") || PatchedName.endswith("sd") ||
       PatchedName.endswith("ps") || PatchedName.endswith("pd"))) {
    bool IsVCMP = PatchedName[0] == 'v';
    unsigned SSECCIdx = IsVCMP ? 4 : 3;
    unsigned SSEComparisonCode = StringSwitch<unsigned>(
      PatchedName.slice(SSECCIdx, PatchedName.size() - 2))
      .Case("eq",       0x00)
      .Case("lt",       0x01)
      .Case("le",       0x02)
      .Case("unord",    0x03)
      .Case("neq",      0x04)
      .Case("nlt",      0x05)
      .Case("nle",      0x06)
      .Case("ord",      0x07)
      /* AVX only from here */
      .Case("eq_uq",    0x08)
      .Case("nge",      0x09)
      .Case("ngt",      0x0A)
      .Case("false",    0x0B)
      .Case("neq_oq",   0x0C)
      .Case("ge",       0x0D)
      .Case("gt",       0x0E)
      .Case("true",     0x0F)
      .Case("eq_os",    0x10)
      .Case("lt_oq",    0x11)
      .Case("le_oq",    0x12)
      .Case("unord_s",  0x13)
      .Case("neq_us",   0x14)
      .Case("nlt_uq",   0x15)
      .Case("nle_uq",   0x16)
      .Case("ord_s",    0x17)
      .Case("eq_us",    0x18)
      .Case("nge_uq",   0x19)
      .Case("ngt_uq",   0x1A)
      .Case("false_os", 0x1B)
      .Case("neq_os",   0x1C)
      .Case("ge_oq",    0x1D)
      .Case("gt_oq",    0x1E)
      .Case("true_us",  0x1F)
      .Default(~0U);
    if (SSEComparisonCode != ~0U && (IsVCMP || SSEComparisonCode < 8)) {
      ExtraImmOp = MCConstantExpr::Create(SSEComparisonCode,
                                          getParser().getContext());
      if (PatchedName.endswith("ss")) {
        PatchedName = IsVCMP ? "vcmpss" : "cmpss";
      } else if (PatchedName.endswith("sd")) {
        PatchedName = IsVCMP ? "vcmpsd" : "cmpsd";
      } else if (PatchedName.endswith("ps")) {
        PatchedName = IsVCMP ? "vcmpps" : "cmpps";
      } else {
        assert(PatchedName.endswith("pd") && "Unexpected mnemonic!");
        PatchedName = IsVCMP ? "vcmppd" : "cmppd";
      }
    }
  }

  Operands.push_back(X86Operand::CreateToken(PatchedName, NameLoc));

  if (ExtraImmOp && !isParsingIntelSyntax())
    Operands.push_back(X86Operand::CreateImm(ExtraImmOp, NameLoc, NameLoc));

  // Determine whether this is an instruction prefix.
  bool isPrefix =
    Name == "lock" || Name == "rep" ||
    Name == "repe" || Name == "repz" ||
    Name == "repne" || Name == "repnz" ||
    Name == "rex64" || Name == "data16";


  // This does the actual operand parsing.  Don't parse any more if we have a
  // prefix juxtaposed with an operation like "lock incl 4(%rax)", because we
  // just want to parse the "lock" as the first instruction and the "incl" as
  // the next one.
  if (getLexer().isNot(AsmToken::EndOfStatement) && !isPrefix) {

    // Parse '*' modifier.
    if (getLexer().is(AsmToken::Star)) {
      SMLoc Loc = Parser.getTok().getLoc();
      Operands.push_back(X86Operand::CreateToken("*", Loc));
      Parser.Lex(); // Eat the star.
    }

    // Read the first operand.
    if (X86Operand *Op = ParseOperand())
      Operands.push_back(Op);
    else {
      Parser.EatToEndOfStatement();
      return true;
    }

    while (getLexer().is(AsmToken::Comma)) {
      Parser.Lex();  // Eat the comma.

      // Parse and remember the operand.
      if (X86Operand *Op = ParseOperand())
        Operands.push_back(Op);
      else {
        Parser.EatToEndOfStatement();
        return true;
      }
    }

    if (getLexer().isNot(AsmToken::EndOfStatement)) {
      SMLoc Loc = getLexer().getLoc();
      Parser.EatToEndOfStatement();
      return Error(Loc, "unexpected token in argument list");
    }
  }

  if (getLexer().is(AsmToken::EndOfStatement))
    Parser.Lex(); // Consume the EndOfStatement
  else if (isPrefix && getLexer().is(AsmToken::Slash))
    Parser.Lex(); // Consume the prefix separator Slash

  if (ExtraImmOp && isParsingIntelSyntax())
    Operands.push_back(X86Operand::CreateImm(ExtraImmOp, NameLoc, NameLoc));

  // This is a terrible hack to handle "out[bwl]? %al, (%dx)" ->
  // "outb %al, %dx".  Out doesn't take a memory form, but this is a widely
  // documented form in various unofficial manuals, so a lot of code uses it.
  if ((Name == "outb" || Name == "outw" || Name == "outl" || Name == "out") &&
      Operands.size() == 3) {
    X86Operand &Op = *(X86Operand*)Operands.back();
    if (Op.isMem() && Op.Mem.SegReg == 0 &&
        isa<MCConstantExpr>(Op.Mem.Disp) &&
        cast<MCConstantExpr>(Op.Mem.Disp)->getValue() == 0 &&
        Op.Mem.BaseReg == MatchRegisterName("dx") && Op.Mem.IndexReg == 0) {
      SMLoc Loc = Op.getEndLoc();
      Operands.back() = X86Operand::CreateReg(Op.Mem.BaseReg, Loc, Loc);
      delete &Op;
    }
  }
  // Same hack for "in[bwl]? (%dx), %al" -> "inb %dx, %al".
  if ((Name == "inb" || Name == "inw" || Name == "inl" || Name == "in") &&
      Operands.size() == 3) {
    X86Operand &Op = *(X86Operand*)Operands.begin()[1];
    if (Op.isMem() && Op.Mem.SegReg == 0 &&
        isa<MCConstantExpr>(Op.Mem.Disp) &&
        cast<MCConstantExpr>(Op.Mem.Disp)->getValue() == 0 &&
        Op.Mem.BaseReg == MatchRegisterName("dx") && Op.Mem.IndexReg == 0) {
      SMLoc Loc = Op.getEndLoc();
      Operands.begin()[1] = X86Operand::CreateReg(Op.Mem.BaseReg, Loc, Loc);
      delete &Op;
    }
  }
  // Transform "ins[bwl] %dx, %es:(%edi)" into "ins[bwl]"
  if (Name.startswith("ins") && Operands.size() == 3 &&
      (Name == "insb" || Name == "insw" || Name == "insl")) {
    X86Operand &Op = *(X86Operand*)Operands.begin()[1];
    X86Operand &Op2 = *(X86Operand*)Operands.begin()[2];
    if (Op.isReg() && Op.getReg() == X86::DX && isDstOp(Op2)) {
      Operands.pop_back();
      Operands.pop_back();
      delete &Op;
      delete &Op2;
    }
  }

  // Transform "outs[bwl] %ds:(%esi), %dx" into "out[bwl]"
  if (Name.startswith("outs") && Operands.size() == 3 &&
      (Name == "outsb" || Name == "outsw" || Name == "outsl")) {
    X86Operand &Op = *(X86Operand*)Operands.begin()[1];
    X86Operand &Op2 = *(X86Operand*)Operands.begin()[2];
    if (isSrcOp(Op) && Op2.isReg() && Op2.getReg() == X86::DX) {
      Operands.pop_back();
      Operands.pop_back();
      delete &Op;
      delete &Op2;
    }
  }

  // Transform "movs[bwl] %ds:(%esi), %es:(%edi)" into "movs[bwl]"
  if (Name.startswith("movs") && Operands.size() == 3 &&
      (Name == "movsb" || Name == "movsw" || Name == "movsl" ||
       (is64BitMode() && Name == "movsq"))) {
    X86Operand &Op = *(X86Operand*)Operands.begin()[1];
    X86Operand &Op2 = *(X86Operand*)Operands.begin()[2];
    if (isSrcOp(Op) && isDstOp(Op2)) {
      Operands.pop_back();
      Operands.pop_back();
      delete &Op;
      delete &Op2;
    }
  }
  // Transform "lods[bwl] %ds:(%esi),{%al,%ax,%eax,%rax}" into "lods[bwl]"
  if (Name.startswith("lods") && Operands.size() == 3 &&
      (Name == "lods" || Name == "lodsb" || Name == "lodsw" ||
       Name == "lodsl" || (is64BitMode() && Name == "lodsq"))) {
    X86Operand *Op1 = static_cast<X86Operand*>(Operands[1]);
    X86Operand *Op2 = static_cast<X86Operand*>(Operands[2]);
    if (isSrcOp(*Op1) && Op2->isReg()) {
      const char *ins;
      unsigned reg = Op2->getReg();
      bool isLods = Name == "lods";
      if (reg == X86::AL && (isLods || Name == "lodsb"))
        ins = "lodsb";
      else if (reg == X86::AX && (isLods || Name == "lodsw"))
        ins = "lodsw";
      else if (reg == X86::EAX && (isLods || Name == "lodsl"))
        ins = "lodsl";
      else if (reg == X86::RAX && (isLods || Name == "lodsq"))
        ins = "lodsq";
      else
        ins = NULL;
      if (ins != NULL) {
        Operands.pop_back();
        Operands.pop_back();
        delete Op1;
        delete Op2;
        if (Name != ins)
          static_cast<X86Operand*>(Operands[0])->setTokenValue(ins);
      }
    }
  }
  // Transform "stos[bwl] {%al,%ax,%eax,%rax},%es:(%edi)" into "stos[bwl]"
  if (Name.startswith("stos") && Operands.size() == 3 &&
      (Name == "stos" || Name == "stosb" || Name == "stosw" ||
       Name == "stosl" || (is64BitMode() && Name == "stosq"))) {
    X86Operand *Op1 = static_cast<X86Operand*>(Operands[1]);
    X86Operand *Op2 = static_cast<X86Operand*>(Operands[2]);
    if (isDstOp(*Op2) && Op1->isReg()) {
      const char *ins;
      unsigned reg = Op1->getReg();
      bool isStos = Name == "stos";
      if (reg == X86::AL && (isStos || Name == "stosb"))
        ins = "stosb";
      else if (reg == X86::AX && (isStos || Name == "stosw"))
        ins = "stosw";
      else if (reg == X86::EAX && (isStos || Name == "stosl"))
        ins = "stosl";
      else if (reg == X86::RAX && (isStos || Name == "stosq"))
        ins = "stosq";
      else
        ins = NULL;
      if (ins != NULL) {
        Operands.pop_back();
        Operands.pop_back();
        delete Op1;
        delete Op2;
        if (Name != ins)
          static_cast<X86Operand*>(Operands[0])->setTokenValue(ins);
      }
    }
  }

  // FIXME: Hack to handle recognize s{hr,ar,hl} $1, <op>.  Canonicalize to
  // "shift <op>".
  if ((Name.startswith("shr") || Name.startswith("sar") ||
       Name.startswith("shl") || Name.startswith("sal") ||
       Name.startswith("rcl") || Name.startswith("rcr") ||
       Name.startswith("rol") || Name.startswith("ror")) &&
      Operands.size() == 3) {
    if (isParsingIntelSyntax()) {
      // Intel syntax
      X86Operand *Op1 = static_cast<X86Operand*>(Operands[2]);
      if (Op1->isImm() && isa<MCConstantExpr>(Op1->getImm()) &&
          cast<MCConstantExpr>(Op1->getImm())->getValue() == 1) {
        delete Operands[2];
        Operands.pop_back();
      }
    } else {
      X86Operand *Op1 = static_cast<X86Operand*>(Operands[1]);
      if (Op1->isImm() && isa<MCConstantExpr>(Op1->getImm()) &&
          cast<MCConstantExpr>(Op1->getImm())->getValue() == 1) {
        delete Operands[1];
        Operands.erase(Operands.begin() + 1);
      }
    }
  }

  // Transforms "int $3" into "int3" as a size optimization.  We can't write an
  // instalias with an immediate operand yet.
  if (Name == "int" && Operands.size() == 2) {
    X86Operand *Op1 = static_cast<X86Operand*>(Operands[1]);
    if (Op1->isImm() && isa<MCConstantExpr>(Op1->getImm()) &&
        cast<MCConstantExpr>(Op1->getImm())->getValue() == 3) {
      delete Operands[1];
      Operands.erase(Operands.begin() + 1);
      static_cast<X86Operand*>(Operands[0])->setTokenValue("int3");
    }
  }

  return false;
}

bool X86AsmParser::
processInstruction(MCInst &Inst,
                   const SmallVectorImpl<MCParsedAsmOperand*> &Ops) {
  switch (Inst.getOpcode()) {
  default: return false;
  case X86::AND16i16: {
    if (!Inst.getOperand(0).isImm() ||
        !isImmSExti16i8Value(Inst.getOperand(0).getImm()))
      return false;

    MCInst TmpInst;
    TmpInst.setOpcode(X86::AND16ri8);
    TmpInst.addOperand(MCOperand::CreateReg(X86::AX));
    TmpInst.addOperand(MCOperand::CreateReg(X86::AX));
    TmpInst.addOperand(Inst.getOperand(0));
    Inst = TmpInst;
    return true;
  }
  case X86::AND32i32: {
    if (!Inst.getOperand(0).isImm() ||
        !isImmSExti32i8Value(Inst.getOperand(0).getImm()))
      return false;

    MCInst TmpInst;
    TmpInst.setOpcode(X86::AND32ri8);
    TmpInst.addOperand(MCOperand::CreateReg(X86::EAX));
    TmpInst.addOperand(MCOperand::CreateReg(X86::EAX));
    TmpInst.addOperand(Inst.getOperand(0));
    Inst = TmpInst;
    return true;
  }
  case X86::AND64i32: {
    if (!Inst.getOperand(0).isImm() ||
        !isImmSExti64i8Value(Inst.getOperand(0).getImm()))
      return false;

    MCInst TmpInst;
    TmpInst.setOpcode(X86::AND64ri8);
    TmpInst.addOperand(MCOperand::CreateReg(X86::RAX));
    TmpInst.addOperand(MCOperand::CreateReg(X86::RAX));
    TmpInst.addOperand(Inst.getOperand(0));
    Inst = TmpInst;
    return true;
  }
  case X86::XOR16i16: {
    if (!Inst.getOperand(0).isImm() ||
        !isImmSExti16i8Value(Inst.getOperand(0).getImm()))
      return false;

    MCInst TmpInst;
    TmpInst.setOpcode(X86::XOR16ri8);
    TmpInst.addOperand(MCOperand::CreateReg(X86::AX));
    TmpInst.addOperand(MCOperand::CreateReg(X86::AX));
    TmpInst.addOperand(Inst.getOperand(0));
    Inst = TmpInst;
    return true;
  }
  case X86::XOR32i32: {
    if (!Inst.getOperand(0).isImm() ||
        !isImmSExti32i8Value(Inst.getOperand(0).getImm()))
      return false;

    MCInst TmpInst;
    TmpInst.setOpcode(X86::XOR32ri8);
    TmpInst.addOperand(MCOperand::CreateReg(X86::EAX));
    TmpInst.addOperand(MCOperand::CreateReg(X86::EAX));
    TmpInst.addOperand(Inst.getOperand(0));
    Inst = TmpInst;
    return true;
  }
  case X86::XOR64i32: {
    if (!Inst.getOperand(0).isImm() ||
        !isImmSExti64i8Value(Inst.getOperand(0).getImm()))
      return false;

    MCInst TmpInst;
    TmpInst.setOpcode(X86::XOR64ri8);
    TmpInst.addOperand(MCOperand::CreateReg(X86::RAX));
    TmpInst.addOperand(MCOperand::CreateReg(X86::RAX));
    TmpInst.addOperand(Inst.getOperand(0));
    Inst = TmpInst;
    return true;
  }
  case X86::OR16i16: {
    if (!Inst.getOperand(0).isImm() ||
        !isImmSExti16i8Value(Inst.getOperand(0).getImm()))
      return false;

    MCInst TmpInst;
    TmpInst.setOpcode(X86::OR16ri8);
    TmpInst.addOperand(MCOperand::CreateReg(X86::AX));
    TmpInst.addOperand(MCOperand::CreateReg(X86::AX));
    TmpInst.addOperand(Inst.getOperand(0));
    Inst = TmpInst;
    return true;
  }
  case X86::OR32i32: {
    if (!Inst.getOperand(0).isImm() ||
        !isImmSExti32i8Value(Inst.getOperand(0).getImm()))
      return false;

    MCInst TmpInst;
    TmpInst.setOpcode(X86::OR32ri8);
    TmpInst.addOperand(MCOperand::CreateReg(X86::EAX));
    TmpInst.addOperand(MCOperand::CreateReg(X86::EAX));
    TmpInst.addOperand(Inst.getOperand(0));
    Inst = TmpInst;
    return true;
  }
  case X86::OR64i32: {
    if (!Inst.getOperand(0).isImm() ||
        !isImmSExti64i8Value(Inst.getOperand(0).getImm()))
      return false;

    MCInst TmpInst;
    TmpInst.setOpcode(X86::OR64ri8);
    TmpInst.addOperand(MCOperand::CreateReg(X86::RAX));
    TmpInst.addOperand(MCOperand::CreateReg(X86::RAX));
    TmpInst.addOperand(Inst.getOperand(0));
    Inst = TmpInst;
    return true;
  }
  case X86::CMP16i16: {
    if (!Inst.getOperand(0).isImm() ||
        !isImmSExti16i8Value(Inst.getOperand(0).getImm()))
      return false;

    MCInst TmpInst;
    TmpInst.setOpcode(X86::CMP16ri8);
    TmpInst.addOperand(MCOperand::CreateReg(X86::AX));
    TmpInst.addOperand(Inst.getOperand(0));
    Inst = TmpInst;
    return true;
  }
  case X86::CMP32i32: {
    if (!Inst.getOperand(0).isImm() ||
        !isImmSExti32i8Value(Inst.getOperand(0).getImm()))
      return false;

    MCInst TmpInst;
    TmpInst.setOpcode(X86::CMP32ri8);
    TmpInst.addOperand(MCOperand::CreateReg(X86::EAX));
    TmpInst.addOperand(Inst.getOperand(0));
    Inst = TmpInst;
    return true;
  }
  case X86::CMP64i32: {
    if (!Inst.getOperand(0).isImm() ||
        !isImmSExti64i8Value(Inst.getOperand(0).getImm()))
      return false;

    MCInst TmpInst;
    TmpInst.setOpcode(X86::CMP64ri8);
    TmpInst.addOperand(MCOperand::CreateReg(X86::RAX));
    TmpInst.addOperand(Inst.getOperand(0));
    Inst = TmpInst;
    return true;
  }
  case X86::ADD16i16: {
    if (!Inst.getOperand(0).isImm() ||
        !isImmSExti16i8Value(Inst.getOperand(0).getImm()))
      return false;

    MCInst TmpInst;
    TmpInst.setOpcode(X86::ADD16ri8);
    TmpInst.addOperand(MCOperand::CreateReg(X86::AX));
    TmpInst.addOperand(MCOperand::CreateReg(X86::AX));
    TmpInst.addOperand(Inst.getOperand(0));
    Inst = TmpInst;
    return true;
  }
  case X86::ADD32i32: {
    if (!Inst.getOperand(0).isImm() ||
        !isImmSExti32i8Value(Inst.getOperand(0).getImm()))
      return false;

    MCInst TmpInst;
    TmpInst.setOpcode(X86::ADD32ri8);
    TmpInst.addOperand(MCOperand::CreateReg(X86::EAX));
    TmpInst.addOperand(MCOperand::CreateReg(X86::EAX));
    TmpInst.addOperand(Inst.getOperand(0));
    Inst = TmpInst;
    return true;
  }
  case X86::ADD64i32: {
    if (!Inst.getOperand(0).isImm() ||
        !isImmSExti64i8Value(Inst.getOperand(0).getImm()))
      return false;

    MCInst TmpInst;
    TmpInst.setOpcode(X86::ADD64ri8);
    TmpInst.addOperand(MCOperand::CreateReg(X86::RAX));
    TmpInst.addOperand(MCOperand::CreateReg(X86::RAX));
    TmpInst.addOperand(Inst.getOperand(0));
    Inst = TmpInst;
    return true;
  }
  case X86::SUB16i16: {
    if (!Inst.getOperand(0).isImm() ||
        !isImmSExti16i8Value(Inst.getOperand(0).getImm()))
      return false;

    MCInst TmpInst;
    TmpInst.setOpcode(X86::SUB16ri8);
    TmpInst.addOperand(MCOperand::CreateReg(X86::AX));
    TmpInst.addOperand(MCOperand::CreateReg(X86::AX));
    TmpInst.addOperand(Inst.getOperand(0));
    Inst = TmpInst;
    return true;
  }
  case X86::SUB32i32: {
    if (!Inst.getOperand(0).isImm() ||
        !isImmSExti32i8Value(Inst.getOperand(0).getImm()))
      return false;

    MCInst TmpInst;
    TmpInst.setOpcode(X86::SUB32ri8);
    TmpInst.addOperand(MCOperand::CreateReg(X86::EAX));
    TmpInst.addOperand(MCOperand::CreateReg(X86::EAX));
    TmpInst.addOperand(Inst.getOperand(0));
    Inst = TmpInst;
    return true;
  }
  case X86::SUB64i32: {
    if (!Inst.getOperand(0).isImm() ||
        !isImmSExti64i8Value(Inst.getOperand(0).getImm()))
      return false;

    MCInst TmpInst;
    TmpInst.setOpcode(X86::SUB64ri8);
    TmpInst.addOperand(MCOperand::CreateReg(X86::RAX));
    TmpInst.addOperand(MCOperand::CreateReg(X86::RAX));
    TmpInst.addOperand(Inst.getOperand(0));
    Inst = TmpInst;
    return true;
  }
  }
}

bool X86AsmParser::
MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                        SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                        MCStreamer &Out, unsigned &ErrorInfo,
                        bool MatchingInlineAsm) {
  assert(!Operands.empty() && "Unexpect empty operand list!");
  X86Operand *Op = static_cast<X86Operand*>(Operands[0]);
  assert(Op->isToken() && "Leading operand should always be a mnemonic!");
  ArrayRef<SMRange> EmptyRanges = ArrayRef<SMRange>();

  // First, handle aliases that expand to multiple instructions.
  // FIXME: This should be replaced with a real .td file alias mechanism.
  // Also, MatchInstructionImpl should actually *do* the EmitInstruction
  // call.
  if (Op->getToken() == "fstsw" || Op->getToken() == "fstcw" ||
      Op->getToken() == "fstsww" || Op->getToken() == "fstcww" ||
      Op->getToken() == "finit" || Op->getToken() == "fsave" ||
      Op->getToken() == "fstenv" || Op->getToken() == "fclex") {
    MCInst Inst;
    Inst.setOpcode(X86::WAIT);
    Inst.setLoc(IDLoc);
    if (!MatchingInlineAsm)
      Out.EmitInstruction(Inst);

    const char *Repl =
      StringSwitch<const char*>(Op->getToken())
        .Case("finit",  "fninit")
        .Case("fsave",  "fnsave")
        .Case("fstcw",  "fnstcw")
        .Case("fstcww",  "fnstcw")
        .Case("fstenv", "fnstenv")
        .Case("fstsw",  "fnstsw")
        .Case("fstsww", "fnstsw")
        .Case("fclex",  "fnclex")
        .Default(0);
    assert(Repl && "Unknown wait-prefixed instruction");
    delete Operands[0];
    Operands[0] = X86Operand::CreateToken(Repl, IDLoc);
  }

  bool WasOriginallyInvalidOperand = false;
  MCInst Inst;

  // First, try a direct match.
  switch (MatchInstructionImpl(Operands, Inst,
                               ErrorInfo, MatchingInlineAsm,
                               isParsingIntelSyntax())) {
  default: break;
  case Match_Success:
    // Some instructions need post-processing to, for example, tweak which
    // encoding is selected. Loop on it while changes happen so the
    // individual transformations can chain off each other.
    if (!MatchingInlineAsm)
      while (processInstruction(Inst, Operands))
        ;

    Inst.setLoc(IDLoc);
    if (!MatchingInlineAsm)
      Out.EmitInstruction(Inst);
    Opcode = Inst.getOpcode();
    return false;
  case Match_MissingFeature:
    Error(IDLoc, "instruction requires a CPU feature not currently enabled",
          EmptyRanges, MatchingInlineAsm);
    return true;
  case Match_InvalidOperand:
    WasOriginallyInvalidOperand = true;
    break;
  case Match_MnemonicFail:
    break;
  }

  // FIXME: Ideally, we would only attempt suffix matches for things which are
  // valid prefixes, and we could just infer the right unambiguous
  // type. However, that requires substantially more matcher support than the
  // following hack.

  // Change the operand to point to a temporary token.
  StringRef Base = Op->getToken();
  SmallString<16> Tmp;
  Tmp += Base;
  Tmp += ' ';
  Op->setTokenValue(Tmp.str());

  // If this instruction starts with an 'f', then it is a floating point stack
  // instruction.  These come in up to three forms for 32-bit, 64-bit, and
  // 80-bit floating point, which use the suffixes s,l,t respectively.
  //
  // Otherwise, we assume that this may be an integer instruction, which comes
  // in 8/16/32/64-bit forms using the b,w,l,q suffixes respectively.
  const char *Suffixes = Base[0] != 'f' ? "bwlq" : "slt\0";

  // Check for the various suffix matches.
  Tmp[Base.size()] = Suffixes[0];
  unsigned ErrorInfoIgnore;
  unsigned Match1, Match2, Match3, Match4;

  Match1 = MatchInstructionImpl(Operands, Inst, ErrorInfoIgnore,
                                isParsingIntelSyntax());
  Tmp[Base.size()] = Suffixes[1];
  Match2 = MatchInstructionImpl(Operands, Inst, ErrorInfoIgnore,
                                isParsingIntelSyntax());
  Tmp[Base.size()] = Suffixes[2];
  Match3 = MatchInstructionImpl(Operands, Inst, ErrorInfoIgnore,
                                isParsingIntelSyntax());
  Tmp[Base.size()] = Suffixes[3];
  Match4 = MatchInstructionImpl(Operands, Inst, ErrorInfoIgnore,
                                isParsingIntelSyntax());

  // Restore the old token.
  Op->setTokenValue(Base);

  // If exactly one matched, then we treat that as a successful match (and the
  // instruction will already have been filled in correctly, since the failing
  // matches won't have modified it).
  unsigned NumSuccessfulMatches =
    (Match1 == Match_Success) + (Match2 == Match_Success) +
    (Match3 == Match_Success) + (Match4 == Match_Success);
  if (NumSuccessfulMatches == 1) {
    Inst.setLoc(IDLoc);
    if (!MatchingInlineAsm)
      Out.EmitInstruction(Inst);
    Opcode = Inst.getOpcode();
    return false;
  }

  // Otherwise, the match failed, try to produce a decent error message.

  // If we had multiple suffix matches, then identify this as an ambiguous
  // match.
  if (NumSuccessfulMatches > 1) {
    char MatchChars[4];
    unsigned NumMatches = 0;
    if (Match1 == Match_Success) MatchChars[NumMatches++] = Suffixes[0];
    if (Match2 == Match_Success) MatchChars[NumMatches++] = Suffixes[1];
    if (Match3 == Match_Success) MatchChars[NumMatches++] = Suffixes[2];
    if (Match4 == Match_Success) MatchChars[NumMatches++] = Suffixes[3];

    SmallString<126> Msg;
    raw_svector_ostream OS(Msg);
    OS << "ambiguous instructions require an explicit suffix (could be ";
    for (unsigned i = 0; i != NumMatches; ++i) {
      if (i != 0)
        OS << ", ";
      if (i + 1 == NumMatches)
        OS << "or ";
      OS << "'" << Base << MatchChars[i] << "'";
    }
    OS << ")";
    Error(IDLoc, OS.str(), EmptyRanges, MatchingInlineAsm);
    return true;
  }

  // Okay, we know that none of the variants matched successfully.

  // If all of the instructions reported an invalid mnemonic, then the original
  // mnemonic was invalid.
  if ((Match1 == Match_MnemonicFail) && (Match2 == Match_MnemonicFail) &&
      (Match3 == Match_MnemonicFail) && (Match4 == Match_MnemonicFail)) {
    if (!WasOriginallyInvalidOperand) {
      ArrayRef<SMRange> Ranges = MatchingInlineAsm ? EmptyRanges :
        Op->getLocRange();
      return Error(IDLoc, "invalid instruction mnemonic '" + Base + "'",
                   Ranges, MatchingInlineAsm);
    }

    // Recover location info for the operand if we know which was the problem.
    if (ErrorInfo != ~0U) {
      if (ErrorInfo >= Operands.size())
        return Error(IDLoc, "too few operands for instruction",
                     EmptyRanges, MatchingInlineAsm);

      X86Operand *Operand = (X86Operand*)Operands[ErrorInfo];
      if (Operand->getStartLoc().isValid()) {
        SMRange OperandRange = Operand->getLocRange();
        return Error(Operand->getStartLoc(), "invalid operand for instruction",
                     OperandRange, MatchingInlineAsm);
      }
    }

    return Error(IDLoc, "invalid operand for instruction", EmptyRanges,
                 MatchingInlineAsm);
  }

  // If one instruction matched with a missing feature, report this as a
  // missing feature.
  if ((Match1 == Match_MissingFeature) + (Match2 == Match_MissingFeature) +
      (Match3 == Match_MissingFeature) + (Match4 == Match_MissingFeature) == 1){
    Error(IDLoc, "instruction requires a CPU feature not currently enabled",
          EmptyRanges, MatchingInlineAsm);
    return true;
  }

  // If one instruction matched with an invalid operand, report this as an
  // operand failure.
  if ((Match1 == Match_InvalidOperand) + (Match2 == Match_InvalidOperand) +
      (Match3 == Match_InvalidOperand) + (Match4 == Match_InvalidOperand) == 1){
    Error(IDLoc, "invalid operand for instruction", EmptyRanges,
          MatchingInlineAsm);
    return true;
  }

  // If all of these were an outright failure, report it in a useless way.
  Error(IDLoc, "unknown use of instruction mnemonic without a size suffix",
        EmptyRanges, MatchingInlineAsm);
  return true;
}


bool X86AsmParser::ParseDirective(AsmToken DirectiveID) {
  StringRef IDVal = DirectiveID.getIdentifier();
  if (IDVal == ".word")
    return ParseDirectiveWord(2, DirectiveID.getLoc());
  else if (IDVal.startswith(".code"))
    return ParseDirectiveCode(IDVal, DirectiveID.getLoc());
  else if (IDVal.startswith(".att_syntax")) {
    getParser().setAssemblerDialect(0);
    return false;
  } else if (IDVal.startswith(".intel_syntax")) {
    getParser().setAssemblerDialect(1);
    if (getLexer().isNot(AsmToken::EndOfStatement)) {
      if(Parser.getTok().getString() == "noprefix") {
        // FIXME : Handle noprefix
        Parser.Lex();
      } else
        return true;
    }
    return false;
  }
  return true;
}

/// ParseDirectiveWord
///  ::= .word [ expression (, expression)* ]
bool X86AsmParser::ParseDirectiveWord(unsigned Size, SMLoc L) {
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

/// ParseDirectiveCode
///  ::= .code32 | .code64
bool X86AsmParser::ParseDirectiveCode(StringRef IDVal, SMLoc L) {
  if (IDVal == ".code32") {
    Parser.Lex();
    if (is64BitMode()) {
      SwitchMode();
      getParser().getStreamer().EmitAssemblerFlag(MCAF_Code32);
    }
  } else if (IDVal == ".code64") {
    Parser.Lex();
    if (!is64BitMode()) {
      SwitchMode();
      getParser().getStreamer().EmitAssemblerFlag(MCAF_Code64);
    }
  } else {
    return Error(L, "unexpected directive " + IDVal);
  }

  return false;
}


extern "C" void LLVMInitializeX86AsmLexer();

// Force static initialization.
extern "C" void LLVMInitializeX86AsmParser() {
  RegisterMCAsmParser<X86AsmParser> X(TheX86_32Target);
  RegisterMCAsmParser<X86AsmParser> Y(TheX86_64Target);
  LLVMInitializeX86AsmLexer();
}

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#include "X86GenAsmMatcher.inc"
