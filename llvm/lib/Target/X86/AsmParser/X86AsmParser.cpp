//===-- X86AsmParser.cpp - Parse X86 assembly to MCInst instructions ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X86BaseInfo.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
struct X86Operand;

static const char OpPrecedence[] = {
  0, // IC_PLUS
  0, // IC_MINUS
  1, // IC_MULTIPLY
  1, // IC_DIVIDE
  2, // IC_RPAREN
  3, // IC_LPAREN
  0, // IC_IMM
  0  // IC_REGISTER
};

class X86AsmParser : public MCTargetAsmParser {
  MCSubtargetInfo &STI;
  MCAsmParser &Parser;
  ParseInstructionInfo *InstInfo;
private:
  enum InfixCalculatorTok {
    IC_PLUS = 0,
    IC_MINUS,
    IC_MULTIPLY,
    IC_DIVIDE,
    IC_RPAREN,
    IC_LPAREN,
    IC_IMM,
    IC_REGISTER
  };

  class InfixCalculator {
    typedef std::pair< InfixCalculatorTok, int64_t > ICToken;
    SmallVector<InfixCalculatorTok, 4> InfixOperatorStack;
    SmallVector<ICToken, 4> PostfixStack;
    
  public:
    int64_t popOperand() {
      assert (!PostfixStack.empty() && "Poped an empty stack!");
      ICToken Op = PostfixStack.pop_back_val();
      assert ((Op.first == IC_IMM || Op.first == IC_REGISTER)
              && "Expected and immediate or register!");
      return Op.second;
    }
    void pushOperand(InfixCalculatorTok Op, int64_t Val = 0) {
      assert ((Op == IC_IMM || Op == IC_REGISTER) &&
              "Unexpected operand!");
      PostfixStack.push_back(std::make_pair(Op, Val));
    }
    
    void popOperator() { InfixOperatorStack.pop_back(); }
    void pushOperator(InfixCalculatorTok Op) {
      // Push the new operator if the stack is empty.
      if (InfixOperatorStack.empty()) {
        InfixOperatorStack.push_back(Op);
        return;
      }
      
      // Push the new operator if it has a higher precedence than the operator
      // on the top of the stack or the operator on the top of the stack is a
      // left parentheses.
      unsigned Idx = InfixOperatorStack.size() - 1;
      InfixCalculatorTok StackOp = InfixOperatorStack[Idx];
      if (OpPrecedence[Op] > OpPrecedence[StackOp] || StackOp == IC_LPAREN) {
        InfixOperatorStack.push_back(Op);
        return;
      }
      
      // The operator on the top of the stack has higher precedence than the
      // new operator.
      unsigned ParenCount = 0;
      while (1) {
        // Nothing to process.
        if (InfixOperatorStack.empty())
          break;
        
        Idx = InfixOperatorStack.size() - 1;
        StackOp = InfixOperatorStack[Idx];
        if (!(OpPrecedence[StackOp] >= OpPrecedence[Op] || ParenCount))
          break;
        
        // If we have an even parentheses count and we see a left parentheses,
        // then stop processing.
        if (!ParenCount && StackOp == IC_LPAREN)
          break;
        
        if (StackOp == IC_RPAREN) {
          ++ParenCount;
          InfixOperatorStack.pop_back();
        } else if (StackOp == IC_LPAREN) {
          --ParenCount;
          InfixOperatorStack.pop_back();
        } else {
          InfixOperatorStack.pop_back();
          PostfixStack.push_back(std::make_pair(StackOp, 0));
        }
      }
      // Push the new operator.
      InfixOperatorStack.push_back(Op);
    }
    int64_t execute() {
      // Push any remaining operators onto the postfix stack.
      while (!InfixOperatorStack.empty()) {
        InfixCalculatorTok StackOp = InfixOperatorStack.pop_back_val();
        if (StackOp != IC_LPAREN && StackOp != IC_RPAREN)
          PostfixStack.push_back(std::make_pair(StackOp, 0));
      }
      
      if (PostfixStack.empty())
        return 0;
      
      SmallVector<ICToken, 16> OperandStack;
      for (unsigned i = 0, e = PostfixStack.size(); i != e; ++i) {
        ICToken Op = PostfixStack[i];
        if (Op.first == IC_IMM || Op.first == IC_REGISTER) {
          OperandStack.push_back(Op);
        } else {
          assert (OperandStack.size() > 1 && "Too few operands.");
          int64_t Val;
          ICToken Op2 = OperandStack.pop_back_val();
          ICToken Op1 = OperandStack.pop_back_val();
          switch (Op.first) {
          default:
            report_fatal_error("Unexpected operator!");
            break;
          case IC_PLUS:
            Val = Op1.second + Op2.second;
            OperandStack.push_back(std::make_pair(IC_IMM, Val));
            break;
          case IC_MINUS:
            Val = Op1.second - Op2.second;
            OperandStack.push_back(std::make_pair(IC_IMM, Val));
            break;
          case IC_MULTIPLY:
            assert (Op1.first == IC_IMM && Op2.first == IC_IMM &&
                    "Multiply operation with an immediate and a register!");
            Val = Op1.second * Op2.second;
            OperandStack.push_back(std::make_pair(IC_IMM, Val));
            break;
          case IC_DIVIDE:
            assert (Op1.first == IC_IMM && Op2.first == IC_IMM &&
                    "Divide operation with an immediate and a register!");
            assert (Op2.second != 0 && "Division by zero!");
            Val = Op1.second / Op2.second;
            OperandStack.push_back(std::make_pair(IC_IMM, Val));
            break;
          }
        }
      }
      assert (OperandStack.size() == 1 && "Expected a single result.");
      return OperandStack.pop_back_val().second;
    }
  };

  enum IntelExprState {
    IES_PLUS,
    IES_MINUS,
    IES_MULTIPLY,
    IES_DIVIDE,
    IES_LBRAC,
    IES_RBRAC,
    IES_LPAREN,
    IES_RPAREN,
    IES_REGISTER,
    IES_INTEGER,
    IES_IDENTIFIER,
    IES_ERROR
  };

  class IntelExprStateMachine {
    IntelExprState State, PrevState;
    unsigned BaseReg, IndexReg, TmpReg, Scale;
    int64_t Imm;
    const MCExpr *Sym;
    StringRef SymName;
    bool StopOnLBrac, AddImmPrefix;
    InfixCalculator IC;
    InlineAsmIdentifierInfo Info;
  public:
    IntelExprStateMachine(int64_t imm, bool stoponlbrac, bool addimmprefix) :
      State(IES_PLUS), PrevState(IES_ERROR), BaseReg(0), IndexReg(0), TmpReg(0),
      Scale(1), Imm(imm), Sym(0), StopOnLBrac(stoponlbrac),
      AddImmPrefix(addimmprefix) { Info.clear(); }
    
    unsigned getBaseReg() { return BaseReg; }
    unsigned getIndexReg() { return IndexReg; }
    unsigned getScale() { return Scale; }
    const MCExpr *getSym() { return Sym; }
    StringRef getSymName() { return SymName; }
    int64_t getImm() { return Imm + IC.execute(); }
    bool isValidEndState() {
      return State == IES_RBRAC || State == IES_INTEGER;
    }
    bool getStopOnLBrac() { return StopOnLBrac; }
    bool getAddImmPrefix() { return AddImmPrefix; }
    bool hadError() { return State == IES_ERROR; }

    InlineAsmIdentifierInfo &getIdentifierInfo() {
      return Info;
    }

    void onPlus() {
      IntelExprState CurrState = State;
      switch (State) {
      default:
        State = IES_ERROR;
        break;
      case IES_INTEGER:
      case IES_RPAREN:
      case IES_REGISTER:
        State = IES_PLUS;
        IC.pushOperator(IC_PLUS);
        if (CurrState == IES_REGISTER && PrevState != IES_MULTIPLY) {
          // If we already have a BaseReg, then assume this is the IndexReg with
          // a scale of 1.
          if (!BaseReg) {
            BaseReg = TmpReg;
          } else {
            assert (!IndexReg && "BaseReg/IndexReg already set!");
            IndexReg = TmpReg;
            Scale = 1;
          }
        }
        break;
      }
      PrevState = CurrState;
    }
    void onMinus() {
      IntelExprState CurrState = State;
      switch (State) {
      default:
        State = IES_ERROR;
        break;
      case IES_PLUS:
      case IES_MULTIPLY:
      case IES_DIVIDE:
      case IES_LPAREN:
      case IES_RPAREN:
      case IES_LBRAC:
      case IES_RBRAC:
      case IES_INTEGER:
      case IES_REGISTER:
        State = IES_MINUS;
        // Only push the minus operator if it is not a unary operator.
        if (!(CurrState == IES_PLUS || CurrState == IES_MINUS ||
              CurrState == IES_MULTIPLY || CurrState == IES_DIVIDE ||
              CurrState == IES_LPAREN || CurrState == IES_LBRAC))
          IC.pushOperator(IC_MINUS);
        if (CurrState == IES_REGISTER && PrevState != IES_MULTIPLY) {
          // If we already have a BaseReg, then assume this is the IndexReg with
          // a scale of 1.
          if (!BaseReg) {
            BaseReg = TmpReg;
          } else {
            assert (!IndexReg && "BaseReg/IndexReg already set!");
            IndexReg = TmpReg;
            Scale = 1;
          }
        }
        break;
      }
      PrevState = CurrState;
    }
    void onRegister(unsigned Reg) {
      IntelExprState CurrState = State;
      switch (State) {
      default:
        State = IES_ERROR;
        break;
      case IES_PLUS:
      case IES_LPAREN:
        State = IES_REGISTER;
        TmpReg = Reg;
        IC.pushOperand(IC_REGISTER);
        break;
      case IES_MULTIPLY:
        // Index Register - Scale * Register
        if (PrevState == IES_INTEGER) {
          assert (!IndexReg && "IndexReg already set!");
          State = IES_REGISTER;
          IndexReg = Reg;
          // Get the scale and replace the 'Scale * Register' with '0'.
          Scale = IC.popOperand();
          IC.pushOperand(IC_IMM);
          IC.popOperator();
        } else {
          State = IES_ERROR;
        }
        break;
      }
      PrevState = CurrState;
    }
    void onIdentifierExpr(const MCExpr *SymRef, StringRef SymRefName) {
      PrevState = State;
      switch (State) {
      default:
        State = IES_ERROR;
        break;
      case IES_PLUS:
      case IES_MINUS:
        State = IES_INTEGER;
        Sym = SymRef;
        SymName = SymRefName;
        IC.pushOperand(IC_IMM);
        break;
      }
    }
    void onInteger(int64_t TmpInt) {
      IntelExprState CurrState = State;
      switch (State) {
      default:
        State = IES_ERROR;
        break;
      case IES_PLUS:
      case IES_MINUS:
      case IES_DIVIDE:
      case IES_MULTIPLY:
      case IES_LPAREN:
        State = IES_INTEGER;
        if (PrevState == IES_REGISTER && CurrState == IES_MULTIPLY) {
          // Index Register - Register * Scale
          assert (!IndexReg && "IndexReg already set!");
          IndexReg = TmpReg;
          Scale = TmpInt;
          // Get the scale and replace the 'Register * Scale' with '0'.
          IC.popOperator();
        } else if ((PrevState == IES_PLUS || PrevState == IES_MINUS ||
                    PrevState == IES_MULTIPLY || PrevState == IES_DIVIDE ||
                    PrevState == IES_LPAREN || PrevState == IES_LBRAC) &&
                   CurrState == IES_MINUS) {
          // Unary minus.  No need to pop the minus operand because it was never
          // pushed.
          IC.pushOperand(IC_IMM, -TmpInt); // Push -Imm.
        } else {
          IC.pushOperand(IC_IMM, TmpInt);
        }
        break;
      }
      PrevState = CurrState;
    }
    void onStar() {
      PrevState = State;
      switch (State) {
      default:
        State = IES_ERROR;
        break;
      case IES_INTEGER:
      case IES_REGISTER:
      case IES_RPAREN:
        State = IES_MULTIPLY;
        IC.pushOperator(IC_MULTIPLY);
        break;
      }
    }
    void onDivide() {
      PrevState = State;
      switch (State) {
      default:
        State = IES_ERROR;
        break;
      case IES_INTEGER:
      case IES_RPAREN:
        State = IES_DIVIDE;
        IC.pushOperator(IC_DIVIDE);
        break;
      }
    }
    void onLBrac() {
      PrevState = State;
      switch (State) {
      default:
        State = IES_ERROR;
        break;
      case IES_RBRAC:
        State = IES_PLUS;
        IC.pushOperator(IC_PLUS);
        break;
      }
    }
    void onRBrac() {
      IntelExprState CurrState = State;
      switch (State) {
      default:
        State = IES_ERROR;
        break;
      case IES_INTEGER:
      case IES_REGISTER:
      case IES_RPAREN:
        State = IES_RBRAC;
        if (CurrState == IES_REGISTER && PrevState != IES_MULTIPLY) {
          // If we already have a BaseReg, then assume this is the IndexReg with
          // a scale of 1.
          if (!BaseReg) {
            BaseReg = TmpReg;
          } else {
            assert (!IndexReg && "BaseReg/IndexReg already set!");
            IndexReg = TmpReg;
            Scale = 1;
          }
        }
        break;
      }
      PrevState = CurrState;
    }
    void onLParen() {
      IntelExprState CurrState = State;
      switch (State) {
      default:
        State = IES_ERROR;
        break;
      case IES_PLUS:
      case IES_MINUS:
      case IES_MULTIPLY:
      case IES_DIVIDE:
      case IES_LPAREN:
        // FIXME: We don't handle this type of unary minus, yet.
        if ((PrevState == IES_PLUS || PrevState == IES_MINUS ||
            PrevState == IES_MULTIPLY || PrevState == IES_DIVIDE ||
            PrevState == IES_LPAREN || PrevState == IES_LBRAC) &&
            CurrState == IES_MINUS) {
          State = IES_ERROR;
          break;
        }
        State = IES_LPAREN;
        IC.pushOperator(IC_LPAREN);
        break;
      }
      PrevState = CurrState;
    }
    void onRParen() {
      PrevState = State;
      switch (State) {
      default:
        State = IES_ERROR;
        break;
      case IES_INTEGER:
      case IES_REGISTER:
      case IES_RPAREN:
        State = IES_RPAREN;
        IC.pushOperator(IC_RPAREN);
        break;
      }
    }
  };

  MCAsmParser &getParser() const { return Parser; }

  MCAsmLexer &getLexer() const { return Parser.getLexer(); }

  bool Error(SMLoc L, const Twine &Msg,
             ArrayRef<SMRange> Ranges = None,
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
  X86Operand *ParseIntelOffsetOfOperator();
  bool ParseIntelDotOperator(const MCExpr *Disp, const MCExpr *&NewDisp);
  X86Operand *ParseIntelOperator(unsigned OpKind);
  X86Operand *ParseIntelSegmentOverride(unsigned SegReg, SMLoc Start, unsigned Size);
  X86Operand *ParseIntelMemOperand(int64_t ImmDisp, SMLoc StartLoc,
                                   unsigned Size);
  bool ParseIntelExpression(IntelExprStateMachine &SM, SMLoc &End);
  X86Operand *ParseIntelBracExpression(unsigned SegReg, SMLoc Start,
                                       int64_t ImmDisp, unsigned Size);
  bool ParseIntelIdentifier(const MCExpr *&Val, StringRef &Identifier,
                            InlineAsmIdentifierInfo &Info,
                            bool IsUnevaluatedOperand, SMLoc &End);

  X86Operand *ParseMemOperand(unsigned SegReg, SMLoc StartLoc);

  X86Operand *CreateMemForInlineAsm(unsigned SegReg, const MCExpr *Disp,
                                    unsigned BaseReg, unsigned IndexReg,
                                    unsigned Scale, SMLoc Start, SMLoc End,
                                    unsigned Size, StringRef Identifier,
                                    InlineAsmIdentifierInfo &Info);

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

  bool isParsingIntelSyntax() {
    return getParser().getAssemblerDialect();
  }

  /// @name Auto-generated Matcher Functions
  /// {

#define GET_ASSEMBLER_HEADER
#include "X86GenAsmMatcher.inc"

  /// }

public:
  X86AsmParser(MCSubtargetInfo &sti, MCAsmParser &parser,
               const MCInstrInfo &MII)
      : MCTargetAsmParser(), STI(sti), Parser(parser), InstInfo(0) {

    // Initialize the set of available features.
    setAvailableFeatures(ComputeAvailableFeatures(STI.getFeatureBits()));
  }
  virtual bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc);

  virtual bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                                SMLoc NameLoc,
                                SmallVectorImpl<MCParsedAsmOperand*> &Operands);

  virtual bool ParseDirective(AsmToken DirectiveID);
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
  StringRef SymName;
  void *OpDecl;
  bool AddressOf;

  struct TokOp {
    const char *Data;
    unsigned Length;
  };

  struct RegOp {
    unsigned RegNo;
  };

  struct ImmOp {
    const MCExpr *Val;
  };

  struct MemOp {
    unsigned SegReg;
    const MCExpr *Disp;
    unsigned BaseReg;
    unsigned IndexReg;
    unsigned Scale;
    unsigned Size;
  };

  union {
    struct TokOp Tok;
    struct RegOp Reg;
    struct ImmOp Imm;
    struct MemOp Mem;
  };

  X86Operand(KindTy K, SMLoc Start, SMLoc End)
    : Kind(K), StartLoc(Start), EndLoc(End) {}

  StringRef getSymName() { return SymName; }
  void *getOpDecl() { return OpDecl; }

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

  bool isOffsetOf() const {
    return OffsetOfLoc.getPointer();
  }

  bool needAddressOf() const {
    return AddressOf;
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
  bool isMem512() const {
    return Kind == Memory && (!Mem.Size || Mem.Size == 512);
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
  bool isMemVZ32() const {
    return Kind == Memory && (!Mem.Size || Mem.Size == 32) &&
      getMemIndexReg() >= X86::ZMM0 && getMemIndexReg() <= X86::ZMM31;
  }
  bool isMemVZ64() const {
    return Kind == Memory && (!Mem.Size || Mem.Size == 64) &&
      getMemIndexReg() >= X86::ZMM0 && getMemIndexReg() <= X86::ZMM31;
  }

  bool isAbsMem() const {
    return Kind == Memory && !getMemSegReg() && !getMemBaseReg() &&
      !getMemIndexReg() && getMemScale() == 1;
  }

  bool isMemOffs8() const {
    return Kind == Memory && !getMemSegReg() && !getMemBaseReg() &&
      !getMemIndexReg() && getMemScale() == 1 && (!Mem.Size || Mem.Size == 8);
  }
  bool isMemOffs16() const {
    return Kind == Memory && !getMemSegReg() && !getMemBaseReg() &&
      !getMemIndexReg() && getMemScale() == 1 && (!Mem.Size || Mem.Size == 16);
  }
  bool isMemOffs32() const {
    return Kind == Memory && !getMemSegReg() && !getMemBaseReg() &&
      !getMemIndexReg() && getMemScale() == 1 && (!Mem.Size || Mem.Size == 32);
  }
  bool isMemOffs64() const {
    return Kind == Memory && !getMemSegReg() && !getMemBaseReg() &&
      !getMemIndexReg() && getMemScale() == 1 && (!Mem.Size || Mem.Size == 64);
  }

  bool isReg() const { return Kind == Register; }

  bool isGR32orGR64() const {
    return Kind == Register &&
      (X86MCRegisterClasses[X86::GR32RegClassID].contains(getReg()) ||
      X86MCRegisterClasses[X86::GR64RegClassID].contains(getReg()));
  }

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

  static unsigned getGR32FromGR64(unsigned RegNo) {
    switch (RegNo) {
    default: llvm_unreachable("Unexpected register");
    case X86::RAX: return X86::EAX;
    case X86::RCX: return X86::ECX;
    case X86::RDX: return X86::EDX;
    case X86::RBX: return X86::EBX;
    case X86::RBP: return X86::EBP;
    case X86::RSP: return X86::ESP;
    case X86::RSI: return X86::ESI;
    case X86::RDI: return X86::EDI;
    case X86::R8: return X86::R8D;
    case X86::R9: return X86::R9D;
    case X86::R10: return X86::R10D;
    case X86::R11: return X86::R11D;
    case X86::R12: return X86::R12D;
    case X86::R13: return X86::R13D;
    case X86::R14: return X86::R14D;
    case X86::R15: return X86::R15D;
    case X86::RIP: return X86::EIP;
    }
  }

  void addGR32orGR64Operands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    unsigned RegNo = getReg();
    if (X86MCRegisterClasses[X86::GR64RegClassID].contains(RegNo))
      RegNo = getGR32FromGR64(RegNo);
    Inst.addOperand(MCOperand::CreateReg(RegNo));
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    addExpr(Inst, getImm());
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

  void addMemOffsOperands(MCInst &Inst, unsigned N) const {
    assert((N == 1) && "Invalid number of operands!");
    // Add as immediates when possible.
    if (const MCConstantExpr *CE = dyn_cast<MCConstantExpr>(getMemDisp()))
      Inst.addOperand(MCOperand::CreateImm(CE->getValue()));
    else
      Inst.addOperand(MCOperand::CreateExpr(getMemDisp()));
  }

  static X86Operand *CreateToken(StringRef Str, SMLoc Loc) {
    SMLoc EndLoc = SMLoc::getFromPointer(Loc.getPointer() + Str.size());
    X86Operand *Res = new X86Operand(Token, Loc, EndLoc);
    Res->Tok.Data = Str.data();
    Res->Tok.Length = Str.size();
    return Res;
  }

  static X86Operand *CreateReg(unsigned RegNo, SMLoc StartLoc, SMLoc EndLoc,
                               bool AddressOf = false,
                               SMLoc OffsetOfLoc = SMLoc(),
                               StringRef SymName = StringRef(),
                               void *OpDecl = 0) {
    X86Operand *Res = new X86Operand(Register, StartLoc, EndLoc);
    Res->Reg.RegNo = RegNo;
    Res->AddressOf = AddressOf;
    Res->OffsetOfLoc = OffsetOfLoc;
    Res->SymName = SymName;
    Res->OpDecl = OpDecl;
    return Res;
  }

  static X86Operand *CreateImm(const MCExpr *Val, SMLoc StartLoc, SMLoc EndLoc){
    X86Operand *Res = new X86Operand(Immediate, StartLoc, EndLoc);
    Res->Imm.Val = Val;
    return Res;
  }

  /// Create an absolute memory operand.
  static X86Operand *CreateMem(const MCExpr *Disp, SMLoc StartLoc, SMLoc EndLoc,
                               unsigned Size = 0, StringRef SymName = StringRef(),
                               void *OpDecl = 0) {
    X86Operand *Res = new X86Operand(Memory, StartLoc, EndLoc);
    Res->Mem.SegReg   = 0;
    Res->Mem.Disp     = Disp;
    Res->Mem.BaseReg  = 0;
    Res->Mem.IndexReg = 0;
    Res->Mem.Scale    = 1;
    Res->Mem.Size     = Size;
    Res->SymName      = SymName;
    Res->OpDecl       = OpDecl;
    Res->AddressOf    = false;
    return Res;
  }

  /// Create a generalized memory operand.
  static X86Operand *CreateMem(unsigned SegReg, const MCExpr *Disp,
                               unsigned BaseReg, unsigned IndexReg,
                               unsigned Scale, SMLoc StartLoc, SMLoc EndLoc,
                               unsigned Size = 0,
                               StringRef SymName = StringRef(),
                               void *OpDecl = 0) {
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
    Res->SymName      = SymName;
    Res->OpDecl       = OpDecl;
    Res->AddressOf    = false;
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
  EndLoc = Tok.getEndLoc();

  if (Tok.isNot(AsmToken::Identifier)) {
    if (isParsingIntelSyntax()) return true;
    return Error(StartLoc, "invalid register name",
                 SMRange(StartLoc, EndLoc));
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
                   SMRange(StartLoc, EndLoc));
  }

  // Parse "%st" as "%st(0)" and "%st(1)", which is multiple tokens.
  if (RegNo == 0 && (Tok.getString() == "st" || Tok.getString() == "ST")) {
    RegNo = X86::ST0;
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

    EndLoc = Parser.getTok().getEndLoc();
    Parser.Lex(); // Eat ')'
    return false;
  }

  EndLoc = Parser.getTok().getEndLoc();

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
      EndLoc = Parser.getTok().getEndLoc();
      Parser.Lex(); // Eat it.
      return false;
    }
  }

  if (RegNo == 0) {
    if (isParsingIntelSyntax()) return true;
    return Error(StartLoc, "invalid register name",
                 SMRange(StartLoc, EndLoc));
  }

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

X86Operand *
X86AsmParser::CreateMemForInlineAsm(unsigned SegReg, const MCExpr *Disp,
                                    unsigned BaseReg, unsigned IndexReg,
                                    unsigned Scale, SMLoc Start, SMLoc End,
                                    unsigned Size, StringRef Identifier,
                                    InlineAsmIdentifierInfo &Info){
  if (isa<MCSymbolRefExpr>(Disp)) {
    // If this is not a VarDecl then assume it is a FuncDecl or some other label
    // reference.  We need an 'r' constraint here, so we need to create register
    // operand to ensure proper matching.  Just pick a GPR based on the size of
    // a pointer.
    if (!Info.IsVarDecl) {
      unsigned RegNo = is64BitMode() ? X86::RBX : X86::EBX;
      return X86Operand::CreateReg(RegNo, Start, End, /*AddressOf=*/true,
                                   SMLoc(), Identifier, Info.OpDecl);
    }
    if (!Size) {
      Size = Info.Type * 8; // Size is in terms of bits in this context.
      if (Size)
        InstInfo->AsmRewrites->push_back(AsmRewrite(AOK_SizeDirective, Start,
                                                    /*Len=*/0, Size));
    }
  }

  // When parsing inline assembly we set the base register to a non-zero value
  // if we don't know the actual value at this time.  This is necessary to
  // get the matching correct in some cases.
  BaseReg = BaseReg ? BaseReg : 1;
  return X86Operand::CreateMem(SegReg, Disp, BaseReg, IndexReg, Scale, Start,
                               End, Size, Identifier, Info.OpDecl);
}

static void
RewriteIntelBracExpression(SmallVectorImpl<AsmRewrite> *AsmRewrites,
                           StringRef SymName, int64_t ImmDisp,
                           int64_t FinalImmDisp, SMLoc &BracLoc,
                           SMLoc &StartInBrac, SMLoc &End) {
  // Remove the '[' and ']' from the IR string.
  AsmRewrites->push_back(AsmRewrite(AOK_Skip, BracLoc, 1));
  AsmRewrites->push_back(AsmRewrite(AOK_Skip, End, 1));

  // If ImmDisp is non-zero, then we parsed a displacement before the
  // bracketed expression (i.e., ImmDisp [ BaseReg + Scale*IndexReg + Disp])
  // If ImmDisp doesn't match the displacement computed by the state machine
  // then we have an additional displacement in the bracketed expression.
  if (ImmDisp != FinalImmDisp) {
    if (ImmDisp) {
      // We have an immediate displacement before the bracketed expression.
      // Adjust this to match the final immediate displacement.
      bool Found = false;
      for (SmallVectorImpl<AsmRewrite>::iterator I = AsmRewrites->begin(),
             E = AsmRewrites->end(); I != E; ++I) {
        if ((*I).Loc.getPointer() > BracLoc.getPointer())
          continue;
        if ((*I).Kind == AOK_ImmPrefix || (*I).Kind == AOK_Imm) {
          assert (!Found && "ImmDisp already rewritten.");
          (*I).Kind = AOK_Imm;
          (*I).Len = BracLoc.getPointer() - (*I).Loc.getPointer();
          (*I).Val = FinalImmDisp;
          Found = true;
          break;
        }
      }
      assert (Found && "Unable to rewrite ImmDisp.");
      (void)Found;
    } else {
      // We have a symbolic and an immediate displacement, but no displacement
      // before the bracketed expression.  Put the immediate displacement
      // before the bracketed expression.
      AsmRewrites->push_back(AsmRewrite(AOK_Imm, BracLoc, 0, FinalImmDisp));
    }
  }
  // Remove all the ImmPrefix rewrites within the brackets.
  for (SmallVectorImpl<AsmRewrite>::iterator I = AsmRewrites->begin(),
         E = AsmRewrites->end(); I != E; ++I) {
    if ((*I).Loc.getPointer() < StartInBrac.getPointer())
      continue;
    if ((*I).Kind == AOK_ImmPrefix)
      (*I).Kind = AOK_Delete;
  }
  const char *SymLocPtr = SymName.data();
  // Skip everything before the symbol.        
  if (unsigned Len = SymLocPtr - StartInBrac.getPointer()) {
    assert(Len > 0 && "Expected a non-negative length.");
    AsmRewrites->push_back(AsmRewrite(AOK_Skip, StartInBrac, Len));
  }
  // Skip everything after the symbol.
  if (unsigned Len = End.getPointer() - (SymLocPtr + SymName.size())) {
    SMLoc Loc = SMLoc::getFromPointer(SymLocPtr + SymName.size());
    assert(Len > 0 && "Expected a non-negative length.");
    AsmRewrites->push_back(AsmRewrite(AOK_Skip, Loc, Len));
  }
}

bool X86AsmParser::ParseIntelExpression(IntelExprStateMachine &SM, SMLoc &End) {
  const AsmToken &Tok = Parser.getTok();

  bool Done = false;
  while (!Done) {
    bool UpdateLocLex = true;

    // The period in the dot operator (e.g., [ebx].foo.bar) is parsed as an
    // identifier.  Don't try an parse it as a register.
    if (Tok.getString().startswith("."))
      break;
    
    // If we're parsing an immediate expression, we don't expect a '['.
    if (SM.getStopOnLBrac() && getLexer().getKind() == AsmToken::LBrac)
      break;

    switch (getLexer().getKind()) {
    default: {
      if (SM.isValidEndState()) {
        Done = true;
        break;
      }
      return Error(Tok.getLoc(), "unknown token in expression");
    }
    case AsmToken::EndOfStatement: {
      Done = true;
      break;
    }
    case AsmToken::Identifier: {
      // This could be a register or a symbolic displacement.
      unsigned TmpReg;
      const MCExpr *Val;
      SMLoc IdentLoc = Tok.getLoc();
      StringRef Identifier = Tok.getString();
      if(!ParseRegister(TmpReg, IdentLoc, End)) {
        SM.onRegister(TmpReg);
        UpdateLocLex = false;
        break;
      } else {
        if (!isParsingInlineAsm()) {
          if (getParser().parsePrimaryExpr(Val, End))
            return Error(Tok.getLoc(), "Unexpected identifier!");
        } else {
          InlineAsmIdentifierInfo &Info = SM.getIdentifierInfo();
          if (ParseIntelIdentifier(Val, Identifier, Info,
                                   /*Unevaluated=*/false, End))
            return true;
        }
        SM.onIdentifierExpr(Val, Identifier);
        UpdateLocLex = false;
        break;
      }
      return Error(Tok.getLoc(), "Unexpected identifier!");
    }
    case AsmToken::Integer:
      if (isParsingInlineAsm() && SM.getAddImmPrefix())
        InstInfo->AsmRewrites->push_back(AsmRewrite(AOK_ImmPrefix,
                                                    Tok.getLoc()));
      SM.onInteger(Tok.getIntVal());
      break;
    case AsmToken::Plus:    SM.onPlus(); break;
    case AsmToken::Minus:   SM.onMinus(); break;
    case AsmToken::Star:    SM.onStar(); break;
    case AsmToken::Slash:   SM.onDivide(); break;
    case AsmToken::LBrac:   SM.onLBrac(); break;
    case AsmToken::RBrac:   SM.onRBrac(); break;
    case AsmToken::LParen:  SM.onLParen(); break;
    case AsmToken::RParen:  SM.onRParen(); break;
    }
    if (SM.hadError())
      return Error(Tok.getLoc(), "unknown token in expression");

    if (!Done && UpdateLocLex) {
      End = Tok.getLoc();
      Parser.Lex(); // Consume the token.
    }
  }
  return false;
}

X86Operand *X86AsmParser::ParseIntelBracExpression(unsigned SegReg, SMLoc Start,
                                                   int64_t ImmDisp,
                                                   unsigned Size) {
  const AsmToken &Tok = Parser.getTok();
  SMLoc BracLoc = Tok.getLoc(), End = Tok.getEndLoc();
  if (getLexer().isNot(AsmToken::LBrac))
    return ErrorOperand(BracLoc, "Expected '[' token!");
  Parser.Lex(); // Eat '['

  SMLoc StartInBrac = Tok.getLoc();
  // Parse [ Symbol + ImmDisp ] and [ BaseReg + Scale*IndexReg + ImmDisp ].  We
  // may have already parsed an immediate displacement before the bracketed
  // expression.
  IntelExprStateMachine SM(ImmDisp, /*StopOnLBrac=*/false, /*AddImmPrefix=*/true);
  if (ParseIntelExpression(SM, End))
    return 0;

  const MCExpr *Disp;
  if (const MCExpr *Sym = SM.getSym()) {
    // A symbolic displacement.
    Disp = Sym;
    if (isParsingInlineAsm())
      RewriteIntelBracExpression(InstInfo->AsmRewrites, SM.getSymName(),
                                 ImmDisp, SM.getImm(), BracLoc, StartInBrac,
                                 End);
  } else {
    // An immediate displacement only.   
    Disp = MCConstantExpr::Create(SM.getImm(), getContext());
  }

  // Parse the dot operator (e.g., [ebx].foo.bar).
  if (Tok.getString().startswith(".")) {
    const MCExpr *NewDisp;
    if (ParseIntelDotOperator(Disp, NewDisp))
      return 0;
    
    End = Tok.getEndLoc();
    Parser.Lex();  // Eat the field.
    Disp = NewDisp;
  }

  int BaseReg = SM.getBaseReg();
  int IndexReg = SM.getIndexReg();
  int Scale = SM.getScale();
  if (!isParsingInlineAsm()) {
    // handle [-42]
    if (!BaseReg && !IndexReg) {
      if (!SegReg)
        return X86Operand::CreateMem(Disp, Start, End, Size);
      else
        return X86Operand::CreateMem(SegReg, Disp, 0, 0, 1, Start, End, Size);
    }
    return X86Operand::CreateMem(SegReg, Disp, BaseReg, IndexReg, Scale, Start,
                                 End, Size);
  }

  InlineAsmIdentifierInfo &Info = SM.getIdentifierInfo();
  return CreateMemForInlineAsm(SegReg, Disp, BaseReg, IndexReg, Scale, Start,
                               End, Size, SM.getSymName(), Info);
}

// Inline assembly may use variable names with namespace alias qualifiers.
bool X86AsmParser::ParseIntelIdentifier(const MCExpr *&Val,
                                        StringRef &Identifier,
                                        InlineAsmIdentifierInfo &Info,
                                        bool IsUnevaluatedOperand, SMLoc &End) {
  assert (isParsingInlineAsm() && "Expected to be parsing inline assembly.");
  Val = 0;

  StringRef LineBuf(Identifier.data());
  SemaCallback->LookupInlineAsmIdentifier(LineBuf, Info, IsUnevaluatedOperand);

  const AsmToken &Tok = Parser.getTok();

  // Advance the token stream until the end of the current token is
  // after the end of what the frontend claimed.
  const char *EndPtr = Tok.getLoc().getPointer() + LineBuf.size();
  while (true) {
    End = Tok.getEndLoc();
    getLexer().Lex();

    assert(End.getPointer() <= EndPtr && "frontend claimed part of a token?");
    if (End.getPointer() == EndPtr) break;
  }

  // Create the symbol reference.
  Identifier = LineBuf;
  MCSymbol *Sym = getContext().GetOrCreateSymbol(Identifier);
  MCSymbolRefExpr::VariantKind Variant = MCSymbolRefExpr::VK_None;
  Val = MCSymbolRefExpr::Create(Sym, Variant, getParser().getContext());
  return false;
}

/// \brief Parse intel style segment override.
X86Operand *X86AsmParser::ParseIntelSegmentOverride(unsigned SegReg,
                                                    SMLoc Start,
                                                    unsigned Size) {
  assert(SegReg != 0 && "Tried to parse a segment override without a segment!");
  const AsmToken &Tok = Parser.getTok(); // Eat colon.
  if (Tok.isNot(AsmToken::Colon))
    return ErrorOperand(Tok.getLoc(), "Expected ':' token!");
  Parser.Lex(); // Eat ':'

  int64_t ImmDisp = 0;
  if (getLexer().is(AsmToken::Integer)) {
    ImmDisp = Tok.getIntVal();
    AsmToken ImmDispToken = Parser.Lex(); // Eat the integer.

    if (isParsingInlineAsm())
      InstInfo->AsmRewrites->push_back(
          AsmRewrite(AOK_ImmPrefix, ImmDispToken.getLoc()));

    if (getLexer().isNot(AsmToken::LBrac)) {
      // An immediate following a 'segment register', 'colon' token sequence can
      // be followed by a bracketed expression.  If it isn't we know we have our
      // final segment override.
      const MCExpr *Disp = MCConstantExpr::Create(ImmDisp, getContext());
      return X86Operand::CreateMem(SegReg, Disp, /*BaseReg=*/0, /*IndexReg=*/0,
                                   /*Scale=*/1, Start, ImmDispToken.getEndLoc(),
                                   Size);
    }
  }

  if (getLexer().is(AsmToken::LBrac))
    return ParseIntelBracExpression(SegReg, Start, ImmDisp, Size);

  const MCExpr *Val;
  SMLoc End;
  if (!isParsingInlineAsm()) {
    if (getParser().parsePrimaryExpr(Val, End))
      return ErrorOperand(Tok.getLoc(), "unknown token in expression");

    return X86Operand::CreateMem(Val, Start, End, Size);
  }

  InlineAsmIdentifierInfo Info;
  StringRef Identifier = Tok.getString();
  if (ParseIntelIdentifier(Val, Identifier, Info,
                           /*Unevaluated=*/false, End))
    return 0;
  return CreateMemForInlineAsm(/*SegReg=*/0, Val, /*BaseReg=*/0,/*IndexReg=*/0,
                               /*Scale=*/1, Start, End, Size, Identifier, Info);
}

/// ParseIntelMemOperand - Parse intel style memory operand.
X86Operand *X86AsmParser::ParseIntelMemOperand(int64_t ImmDisp, SMLoc Start,
                                               unsigned Size) {
  const AsmToken &Tok = Parser.getTok();
  SMLoc End;

  // Parse ImmDisp [ BaseReg + Scale*IndexReg + Disp ].
  if (getLexer().is(AsmToken::LBrac))
    return ParseIntelBracExpression(/*SegReg=*/0, Start, ImmDisp, Size);

  const MCExpr *Val;
  if (!isParsingInlineAsm()) {
    if (getParser().parsePrimaryExpr(Val, End))
      return ErrorOperand(Tok.getLoc(), "unknown token in expression");

    return X86Operand::CreateMem(Val, Start, End, Size);
  }

  InlineAsmIdentifierInfo Info;
  StringRef Identifier = Tok.getString();
  if (ParseIntelIdentifier(Val, Identifier, Info,
                           /*Unevaluated=*/false, End))
    return 0;
  return CreateMemForInlineAsm(/*SegReg=*/0, Val, /*BaseReg=*/0, /*IndexReg=*/0,
                               /*Scale=*/1, Start, End, Size, Identifier, Info);
}

/// Parse the '.' operator.
bool X86AsmParser::ParseIntelDotOperator(const MCExpr *Disp,
                                                const MCExpr *&NewDisp) {
  const AsmToken &Tok = Parser.getTok();
  int64_t OrigDispVal, DotDispVal;

  // FIXME: Handle non-constant expressions.
  if (const MCConstantExpr *OrigDisp = dyn_cast<MCConstantExpr>(Disp))
    OrigDispVal = OrigDisp->getValue();
  else
    return Error(Tok.getLoc(), "Non-constant offsets are not supported!");

  // Drop the '.'.
  StringRef DotDispStr = Tok.getString().drop_front(1);

  // .Imm gets lexed as a real.
  if (Tok.is(AsmToken::Real)) {
    APInt DotDisp;
    DotDispStr.getAsInteger(10, DotDisp);
    DotDispVal = DotDisp.getZExtValue();
  } else if (isParsingInlineAsm() && Tok.is(AsmToken::Identifier)) {
    unsigned DotDisp;
    std::pair<StringRef, StringRef> BaseMember = DotDispStr.split('.');
    if (SemaCallback->LookupInlineAsmField(BaseMember.first, BaseMember.second,
                                           DotDisp))
      return Error(Tok.getLoc(), "Unable to lookup field reference!");
    DotDispVal = DotDisp;
  } else
    return Error(Tok.getLoc(), "Unexpected token type!");

  if (isParsingInlineAsm() && Tok.is(AsmToken::Identifier)) {
    SMLoc Loc = SMLoc::getFromPointer(DotDispStr.data());
    unsigned Len = DotDispStr.size();
    unsigned Val = OrigDispVal + DotDispVal;
    InstInfo->AsmRewrites->push_back(AsmRewrite(AOK_DotOperator, Loc, Len,
                                                Val));
  }

  NewDisp = MCConstantExpr::Create(OrigDispVal + DotDispVal, getContext());
  return false;
}

/// Parse the 'offset' operator.  This operator is used to specify the
/// location rather then the content of a variable.
X86Operand *X86AsmParser::ParseIntelOffsetOfOperator() {
  const AsmToken &Tok = Parser.getTok();
  SMLoc OffsetOfLoc = Tok.getLoc();
  Parser.Lex(); // Eat offset.

  const MCExpr *Val;
  InlineAsmIdentifierInfo Info;
  SMLoc Start = Tok.getLoc(), End;
  StringRef Identifier = Tok.getString();
  if (ParseIntelIdentifier(Val, Identifier, Info,
                           /*Unevaluated=*/false, End))
    return 0;

  // Don't emit the offset operator.
  InstInfo->AsmRewrites->push_back(AsmRewrite(AOK_Skip, OffsetOfLoc, 7));

  // The offset operator will have an 'r' constraint, thus we need to create
  // register operand to ensure proper matching.  Just pick a GPR based on
  // the size of a pointer.
  unsigned RegNo = is64BitMode() ? X86::RBX : X86::EBX;
  return X86Operand::CreateReg(RegNo, Start, End, /*GetAddress=*/true,
                               OffsetOfLoc, Identifier, Info.OpDecl);
}

enum IntelOperatorKind {
  IOK_LENGTH,
  IOK_SIZE,
  IOK_TYPE
};

/// Parse the 'LENGTH', 'TYPE' and 'SIZE' operators.  The LENGTH operator
/// returns the number of elements in an array.  It returns the value 1 for
/// non-array variables.  The SIZE operator returns the size of a C or C++
/// variable.  A variable's size is the product of its LENGTH and TYPE.  The
/// TYPE operator returns the size of a C or C++ type or variable. If the
/// variable is an array, TYPE returns the size of a single element.
X86Operand *X86AsmParser::ParseIntelOperator(unsigned OpKind) {
  const AsmToken &Tok = Parser.getTok();
  SMLoc TypeLoc = Tok.getLoc();
  Parser.Lex(); // Eat operator.

  const MCExpr *Val = 0;
  InlineAsmIdentifierInfo Info;
  SMLoc Start = Tok.getLoc(), End;
  StringRef Identifier = Tok.getString();
  if (ParseIntelIdentifier(Val, Identifier, Info,
                           /*Unevaluated=*/true, End))
    return 0;

  if (!Info.OpDecl)
    return ErrorOperand(Start, "unable to lookup expression");

  unsigned CVal = 0;
  switch(OpKind) {
  default: llvm_unreachable("Unexpected operand kind!");
  case IOK_LENGTH: CVal = Info.Length; break;
  case IOK_SIZE: CVal = Info.Size; break;
  case IOK_TYPE: CVal = Info.Type; break;
  }

  // Rewrite the type operator and the C or C++ type or variable in terms of an
  // immediate.  E.g. TYPE foo -> $$4
  unsigned Len = End.getPointer() - TypeLoc.getPointer();
  InstInfo->AsmRewrites->push_back(AsmRewrite(AOK_Imm, TypeLoc, Len, CVal));

  const MCExpr *Imm = MCConstantExpr::Create(CVal, getContext());
  return X86Operand::CreateImm(Imm, Start, End);
}

X86Operand *X86AsmParser::ParseIntelOperand() {
  const AsmToken &Tok = Parser.getTok();
  SMLoc Start, End;

  // Offset, length, type and size operators.
  if (isParsingInlineAsm()) {
    StringRef AsmTokStr = Tok.getString();
    if (AsmTokStr == "offset" || AsmTokStr == "OFFSET")
      return ParseIntelOffsetOfOperator();
    if (AsmTokStr == "length" || AsmTokStr == "LENGTH")
      return ParseIntelOperator(IOK_LENGTH);
    if (AsmTokStr == "size" || AsmTokStr == "SIZE")
      return ParseIntelOperator(IOK_SIZE);
    if (AsmTokStr == "type" || AsmTokStr == "TYPE")
      return ParseIntelOperator(IOK_TYPE);
  }

  unsigned Size = getIntelMemOperandSize(Tok.getString());
  if (Size) {
    Parser.Lex(); // Eat operand size (e.g., byte, word).
    if (Tok.getString() != "PTR" && Tok.getString() != "ptr")
      return ErrorOperand(Start, "Expected 'PTR' or 'ptr' token!");
    Parser.Lex(); // Eat ptr.
  }
  Start = Tok.getLoc();

  // Immediate.
  if (getLexer().is(AsmToken::Integer) || getLexer().is(AsmToken::Minus) ||
      getLexer().is(AsmToken::LParen)) {    
    AsmToken StartTok = Tok;
    IntelExprStateMachine SM(/*Imm=*/0, /*StopOnLBrac=*/true,
                             /*AddImmPrefix=*/false);
    if (ParseIntelExpression(SM, End))
      return 0;

    int64_t Imm = SM.getImm();
    if (isParsingInlineAsm()) {
      unsigned Len = Tok.getLoc().getPointer() - Start.getPointer();
      if (StartTok.getString().size() == Len)
        // Just add a prefix if this wasn't a complex immediate expression.
        InstInfo->AsmRewrites->push_back(AsmRewrite(AOK_ImmPrefix, Start));
      else
        // Otherwise, rewrite the complex expression as a single immediate.
        InstInfo->AsmRewrites->push_back(AsmRewrite(AOK_Imm, Start, Len, Imm));
    }

    if (getLexer().isNot(AsmToken::LBrac)) {
      const MCExpr *ImmExpr = MCConstantExpr::Create(Imm, getContext());
      return X86Operand::CreateImm(ImmExpr, Start, End);
    }

    // Only positive immediates are valid.
    if (Imm < 0)
      return ErrorOperand(Start, "expected a positive immediate displacement "
                          "before bracketed expr.");

    // Parse ImmDisp [ BaseReg + Scale*IndexReg + Disp ].
    return ParseIntelMemOperand(Imm, Start, Size);
  }

  // Register.
  unsigned RegNo = 0;
  if (!ParseRegister(RegNo, Start, End)) {
    // If this is a segment register followed by a ':', then this is the start
    // of a segment override, otherwise this is a normal register reference.
    if (getLexer().isNot(AsmToken::Colon))
      return X86Operand::CreateReg(RegNo, Start, End);

    return ParseIntelSegmentOverride(/*SegReg=*/RegNo, Start, Size);
  }

  // Memory operand.
  return ParseIntelMemOperand(/*Disp=*/0, Start, Size);
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
    if (getParser().parseExpression(Val, End))
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
    if (getParser().parseExpression(Disp, ExprEnd)) return 0;

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
      if (getParser().parseParenExpression(Disp, ExprEnd))
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
          if (getParser().parseAbsoluteExpression(ScaleVal)){
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
      if (getParser().parseAbsoluteExpression(Value))
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
  SMLoc MemEnd = Parser.getTok().getEndLoc();
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
      Parser.eatToEndOfStatement();
      return true;
    }

    while (getLexer().is(AsmToken::Comma)) {
      Parser.Lex();  // Eat the comma.

      // Parse and remember the operand.
      if (X86Operand *Op = ParseOperand())
        Operands.push_back(Op);
      else {
        Parser.eatToEndOfStatement();
        return true;
      }
    }

    if (STI.getFeatureBits() & X86::FeatureAVX512) {
      // Parse mask register {%k1}
      if (getLexer().is(AsmToken::LCurly)) {
        SMLoc Loc = Parser.getTok().getLoc();
        Operands.push_back(X86Operand::CreateToken("{", Loc));
        Parser.Lex();  // Eat the {
        if (X86Operand *Op = ParseOperand()) {
          Operands.push_back(Op);
          if (!getLexer().is(AsmToken::RCurly)) {
            SMLoc Loc = getLexer().getLoc();
            Parser.eatToEndOfStatement();
            return Error(Loc, "Expected } at this point");
          }
          Loc = Parser.getTok().getLoc();
          Operands.push_back(X86Operand::CreateToken("}", Loc));
          Parser.Lex();  // Eat the }
        } else {
          Parser.eatToEndOfStatement();
          return true;
        }
      }
      // Parse "zeroing non-masked" semantic {z}
      if (getLexer().is(AsmToken::LCurly)) {
        SMLoc Loc = Parser.getTok().getLoc();
        Operands.push_back(X86Operand::CreateToken("{z}", Loc));
        Parser.Lex();  // Eat the {
        if (!getLexer().is(AsmToken::Identifier) || getLexer().getTok().getIdentifier() != "z") {
          SMLoc Loc = getLexer().getLoc();
          Parser.eatToEndOfStatement();
          return Error(Loc, "Expected z at this point");
        }
        Parser.Lex();  // Eat the z
        if (!getLexer().is(AsmToken::RCurly)) {
            SMLoc Loc = getLexer().getLoc();
            Parser.eatToEndOfStatement();
            return Error(Loc, "Expected } at this point");
        }
        Parser.Lex();  // Eat the }
      }
    }

    if (getLexer().isNot(AsmToken::EndOfStatement)) {
      SMLoc Loc = getLexer().getLoc();
      Parser.eatToEndOfStatement();
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

static bool convertToSExti8(MCInst &Inst, unsigned Opcode, unsigned Reg,
                            bool isCmp) {
  MCInst TmpInst;
  TmpInst.setOpcode(Opcode);
  if (!isCmp)
    TmpInst.addOperand(MCOperand::CreateReg(Reg));
  TmpInst.addOperand(MCOperand::CreateReg(Reg));
  TmpInst.addOperand(Inst.getOperand(0));
  Inst = TmpInst;
  return true;
}

static bool convert16i16to16ri8(MCInst &Inst, unsigned Opcode,
                                bool isCmp = false) {
  if (!Inst.getOperand(0).isImm() ||
      !isImmSExti16i8Value(Inst.getOperand(0).getImm()))
    return false;

  return convertToSExti8(Inst, Opcode, X86::AX, isCmp);
}

static bool convert32i32to32ri8(MCInst &Inst, unsigned Opcode,
                                bool isCmp = false) {
  if (!Inst.getOperand(0).isImm() ||
      !isImmSExti32i8Value(Inst.getOperand(0).getImm()))
    return false;

  return convertToSExti8(Inst, Opcode, X86::EAX, isCmp);
}

static bool convert64i32to64ri8(MCInst &Inst, unsigned Opcode,
                                bool isCmp = false) {
  if (!Inst.getOperand(0).isImm() ||
      !isImmSExti64i8Value(Inst.getOperand(0).getImm()))
    return false;

  return convertToSExti8(Inst, Opcode, X86::RAX, isCmp);
}

bool X86AsmParser::
processInstruction(MCInst &Inst,
                   const SmallVectorImpl<MCParsedAsmOperand*> &Ops) {
  switch (Inst.getOpcode()) {
  default: return false;
  case X86::AND16i16: return convert16i16to16ri8(Inst, X86::AND16ri8);
  case X86::AND32i32: return convert32i32to32ri8(Inst, X86::AND32ri8);
  case X86::AND64i32: return convert64i32to64ri8(Inst, X86::AND64ri8);
  case X86::XOR16i16: return convert16i16to16ri8(Inst, X86::XOR16ri8);
  case X86::XOR32i32: return convert32i32to32ri8(Inst, X86::XOR32ri8);
  case X86::XOR64i32: return convert64i32to64ri8(Inst, X86::XOR64ri8);
  case X86::OR16i16:  return convert16i16to16ri8(Inst, X86::OR16ri8);
  case X86::OR32i32:  return convert32i32to32ri8(Inst, X86::OR32ri8);
  case X86::OR64i32:  return convert64i32to64ri8(Inst, X86::OR64ri8);
  case X86::CMP16i16: return convert16i16to16ri8(Inst, X86::CMP16ri8, true);
  case X86::CMP32i32: return convert32i32to32ri8(Inst, X86::CMP32ri8, true);
  case X86::CMP64i32: return convert64i32to64ri8(Inst, X86::CMP64ri8, true);
  case X86::ADD16i16: return convert16i16to16ri8(Inst, X86::ADD16ri8);
  case X86::ADD32i32: return convert32i32to32ri8(Inst, X86::ADD32ri8);
  case X86::ADD64i32: return convert64i32to64ri8(Inst, X86::ADD64ri8);
  case X86::SUB16i16: return convert16i16to16ri8(Inst, X86::SUB16ri8);
  case X86::SUB32i32: return convert32i32to32ri8(Inst, X86::SUB32ri8);
  case X86::SUB64i32: return convert64i32to64ri8(Inst, X86::SUB64ri8);
  case X86::ADC16i16: return convert16i16to16ri8(Inst, X86::ADC16ri8);
  case X86::ADC32i32: return convert32i32to32ri8(Inst, X86::ADC32ri8);
  case X86::ADC64i32: return convert64i32to64ri8(Inst, X86::ADC64ri8);
  case X86::SBB16i16: return convert16i16to16ri8(Inst, X86::SBB16ri8);
  case X86::SBB32i32: return convert32i32to32ri8(Inst, X86::SBB32ri8);
  case X86::SBB64i32: return convert64i32to64ri8(Inst, X86::SBB64ri8);
  case X86::VMOVAPDrr:
  case X86::VMOVAPDYrr:
  case X86::VMOVAPSrr:
  case X86::VMOVAPSYrr:
  case X86::VMOVDQArr:
  case X86::VMOVDQAYrr:
  case X86::VMOVDQUrr:
  case X86::VMOVDQUYrr:
  case X86::VMOVUPDrr:
  case X86::VMOVUPDYrr:
  case X86::VMOVUPSrr:
  case X86::VMOVUPSYrr: {
    if (X86II::isX86_64ExtendedReg(Inst.getOperand(0).getReg()) ||
        !X86II::isX86_64ExtendedReg(Inst.getOperand(1).getReg()))
      return false;

    unsigned NewOpc;
    switch (Inst.getOpcode()) {
    default: llvm_unreachable("Invalid opcode");
    case X86::VMOVAPDrr:  NewOpc = X86::VMOVAPDrr_REV;  break;
    case X86::VMOVAPDYrr: NewOpc = X86::VMOVAPDYrr_REV; break;
    case X86::VMOVAPSrr:  NewOpc = X86::VMOVAPSrr_REV;  break;
    case X86::VMOVAPSYrr: NewOpc = X86::VMOVAPSYrr_REV; break;
    case X86::VMOVDQArr:  NewOpc = X86::VMOVDQArr_REV;  break;
    case X86::VMOVDQAYrr: NewOpc = X86::VMOVDQAYrr_REV; break;
    case X86::VMOVDQUrr:  NewOpc = X86::VMOVDQUrr_REV;  break;
    case X86::VMOVDQUYrr: NewOpc = X86::VMOVDQUYrr_REV; break;
    case X86::VMOVUPDrr:  NewOpc = X86::VMOVUPDrr_REV;  break;
    case X86::VMOVUPDYrr: NewOpc = X86::VMOVUPDYrr_REV; break;
    case X86::VMOVUPSrr:  NewOpc = X86::VMOVUPSrr_REV;  break;
    case X86::VMOVUPSYrr: NewOpc = X86::VMOVUPSYrr_REV; break;
    }
    Inst.setOpcode(NewOpc);
    return true;
  }
  case X86::VMOVSDrr:
  case X86::VMOVSSrr: {
    if (X86II::isX86_64ExtendedReg(Inst.getOperand(0).getReg()) ||
        !X86II::isX86_64ExtendedReg(Inst.getOperand(2).getReg()))
      return false;
    unsigned NewOpc;
    switch (Inst.getOpcode()) {
    default: llvm_unreachable("Invalid opcode");
    case X86::VMOVSDrr: NewOpc = X86::VMOVSDrr_REV;   break;
    case X86::VMOVSSrr: NewOpc = X86::VMOVSSrr_REV;   break;
    }
    Inst.setOpcode(NewOpc);
    return true;
  }
  }
}

static const char *getSubtargetFeatureName(unsigned Val);
bool X86AsmParser::
MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                        SmallVectorImpl<MCParsedAsmOperand*> &Operands,
                        MCStreamer &Out, unsigned &ErrorInfo,
                        bool MatchingInlineAsm) {
  assert(!Operands.empty() && "Unexpect empty operand list!");
  X86Operand *Op = static_cast<X86Operand*>(Operands[0]);
  assert(Op->isToken() && "Leading operand should always be a mnemonic!");
  ArrayRef<SMRange> EmptyRanges = None;

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
  case Match_MissingFeature: {
    assert(ErrorInfo && "Unknown missing feature!");
    // Special case the error message for the very common case where only
    // a single subtarget feature is missing.
    std::string Msg = "instruction requires:";
    unsigned Mask = 1;
    for (unsigned i = 0; i < (sizeof(ErrorInfo)*8-1); ++i) {
      if (ErrorInfo & Mask) {
        Msg += " ";
        Msg += getSubtargetFeatureName(ErrorInfo & Mask);
      }
      Mask <<= 1;
    }
    return Error(IDLoc, Msg, EmptyRanges, MatchingInlineAsm);
  }
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
  unsigned ErrorInfoMissingFeature = 0; // Init suppresses compiler warnings.
  unsigned Match1, Match2, Match3, Match4;

  Match1 = MatchInstructionImpl(Operands, Inst, ErrorInfoIgnore,
                                MatchingInlineAsm, isParsingIntelSyntax());
  // If this returned as a missing feature failure, remember that.
  if (Match1 == Match_MissingFeature)
    ErrorInfoMissingFeature = ErrorInfoIgnore;
  Tmp[Base.size()] = Suffixes[1];
  Match2 = MatchInstructionImpl(Operands, Inst, ErrorInfoIgnore,
                                MatchingInlineAsm, isParsingIntelSyntax());
  // If this returned as a missing feature failure, remember that.
  if (Match2 == Match_MissingFeature)
    ErrorInfoMissingFeature = ErrorInfoIgnore;
  Tmp[Base.size()] = Suffixes[2];
  Match3 = MatchInstructionImpl(Operands, Inst, ErrorInfoIgnore,
                                MatchingInlineAsm, isParsingIntelSyntax());
  // If this returned as a missing feature failure, remember that.
  if (Match3 == Match_MissingFeature)
    ErrorInfoMissingFeature = ErrorInfoIgnore;
  Tmp[Base.size()] = Suffixes[3];
  Match4 = MatchInstructionImpl(Operands, Inst, ErrorInfoIgnore,
                                MatchingInlineAsm, isParsingIntelSyntax());
  // If this returned as a missing feature failure, remember that.
  if (Match4 == Match_MissingFeature)
    ErrorInfoMissingFeature = ErrorInfoIgnore;

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
    std::string Msg = "instruction requires:";
    unsigned Mask = 1;
    for (unsigned i = 0; i < (sizeof(ErrorInfoMissingFeature)*8-1); ++i) {
      if (ErrorInfoMissingFeature & Mask) {
        Msg += " ";
        Msg += getSubtargetFeatureName(ErrorInfoMissingFeature & Mask);
      }
      Mask <<= 1;
    }
    return Error(IDLoc, Msg, EmptyRanges, MatchingInlineAsm);
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
      if (getParser().parseExpression(Value))
        return true;

      getParser().getStreamer().EmitValue(Value, Size);

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

// Force static initialization.
extern "C" void LLVMInitializeX86AsmParser() {
  RegisterMCAsmParser<X86AsmParser> X(TheX86_32Target);
  RegisterMCAsmParser<X86AsmParser> Y(TheX86_64Target);
}

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#define GET_SUBTARGET_FEATURE_NAME
#include "X86GenAsmMatcher.inc"
