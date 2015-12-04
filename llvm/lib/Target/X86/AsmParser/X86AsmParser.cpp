//===-- X86AsmParser.cpp - Parse X86 assembly to MCInst instructions ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X86BaseInfo.h"
#include "X86AsmInstrumentation.h"
#include "X86AsmParserCommon.h"
#include "X86Operand.h"
#include "X86ISelLowering.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
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
#include <algorithm>
#include <memory>

using namespace llvm;

namespace {

static const char OpPrecedence[] = {
  0, // IC_OR
  1, // IC_XOR
  2, // IC_AND
  3, // IC_LSHIFT
  3, // IC_RSHIFT
  4, // IC_PLUS
  4, // IC_MINUS
  5, // IC_MULTIPLY
  5, // IC_DIVIDE
  6, // IC_RPAREN
  7, // IC_LPAREN
  0, // IC_IMM
  0  // IC_REGISTER
};

class X86AsmParser : public MCTargetAsmParser {
  const MCInstrInfo &MII;
  ParseInstructionInfo *InstInfo;
  std::unique_ptr<X86AsmInstrumentation> Instrumentation;

private:
  SMLoc consumeToken() {
    MCAsmParser &Parser = getParser();
    SMLoc Result = Parser.getTok().getLoc();
    Parser.Lex();
    return Result;
  }

  enum InfixCalculatorTok {
    IC_OR = 0,
    IC_XOR,
    IC_AND,
    IC_LSHIFT,
    IC_RSHIFT,
    IC_PLUS,
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
          case IC_OR:
            assert (Op1.first == IC_IMM && Op2.first == IC_IMM &&
                    "Or operation with an immediate and a register!");
            Val = Op1.second | Op2.second;
            OperandStack.push_back(std::make_pair(IC_IMM, Val));
            break;
          case IC_XOR:
            assert(Op1.first == IC_IMM && Op2.first == IC_IMM &&
              "Xor operation with an immediate and a register!");
            Val = Op1.second ^ Op2.second;
            OperandStack.push_back(std::make_pair(IC_IMM, Val));
            break;
          case IC_AND:
            assert (Op1.first == IC_IMM && Op2.first == IC_IMM &&
                    "And operation with an immediate and a register!");
            Val = Op1.second & Op2.second;
            OperandStack.push_back(std::make_pair(IC_IMM, Val));
            break;
          case IC_LSHIFT:
            assert (Op1.first == IC_IMM && Op2.first == IC_IMM &&
                    "Left shift operation with an immediate and a register!");
            Val = Op1.second << Op2.second;
            OperandStack.push_back(std::make_pair(IC_IMM, Val));
            break;
          case IC_RSHIFT:
            assert (Op1.first == IC_IMM && Op2.first == IC_IMM &&
                    "Right shift operation with an immediate and a register!");
            Val = Op1.second >> Op2.second;
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
    IES_OR,
    IES_XOR,
    IES_AND,
    IES_LSHIFT,
    IES_RSHIFT,
    IES_PLUS,
    IES_MINUS,
    IES_NOT,
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
      Scale(1), Imm(imm), Sym(nullptr), StopOnLBrac(stoponlbrac),
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

    void onOr() {
      IntelExprState CurrState = State;
      switch (State) {
      default:
        State = IES_ERROR;
        break;
      case IES_INTEGER:
      case IES_RPAREN:
      case IES_REGISTER:
        State = IES_OR;
        IC.pushOperator(IC_OR);
        break;
      }
      PrevState = CurrState;
    }
    void onXor() {
      IntelExprState CurrState = State;
      switch (State) {
      default:
        State = IES_ERROR;
        break;
      case IES_INTEGER:
      case IES_RPAREN:
      case IES_REGISTER:
        State = IES_XOR;
        IC.pushOperator(IC_XOR);
        break;
      }
      PrevState = CurrState;
    }
    void onAnd() {
      IntelExprState CurrState = State;
      switch (State) {
      default:
        State = IES_ERROR;
        break;
      case IES_INTEGER:
      case IES_RPAREN:
      case IES_REGISTER:
        State = IES_AND;
        IC.pushOperator(IC_AND);
        break;
      }
      PrevState = CurrState;
    }
    void onLShift() {
      IntelExprState CurrState = State;
      switch (State) {
      default:
        State = IES_ERROR;
        break;
      case IES_INTEGER:
      case IES_RPAREN:
      case IES_REGISTER:
        State = IES_LSHIFT;
        IC.pushOperator(IC_LSHIFT);
        break;
      }
      PrevState = CurrState;
    }
    void onRShift() {
      IntelExprState CurrState = State;
      switch (State) {
      default:
        State = IES_ERROR;
        break;
      case IES_INTEGER:
      case IES_RPAREN:
      case IES_REGISTER:
        State = IES_RSHIFT;
        IC.pushOperator(IC_RSHIFT);
        break;
      }
      PrevState = CurrState;
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
      case IES_NOT:
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
    void onNot() {
      IntelExprState CurrState = State;
      switch (State) {
      default:
        State = IES_ERROR;
        break;
      case IES_PLUS:
      case IES_NOT:
        State = IES_NOT;
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
      case IES_NOT:
        State = IES_INTEGER;
        Sym = SymRef;
        SymName = SymRefName;
        IC.pushOperand(IC_IMM);
        break;
      }
    }
    bool onInteger(int64_t TmpInt, StringRef &ErrMsg) {
      IntelExprState CurrState = State;
      switch (State) {
      default:
        State = IES_ERROR;
        break;
      case IES_PLUS:
      case IES_MINUS:
      case IES_NOT:
      case IES_OR:
      case IES_XOR:
      case IES_AND:
      case IES_LSHIFT:
      case IES_RSHIFT:
      case IES_DIVIDE:
      case IES_MULTIPLY:
      case IES_LPAREN:
        State = IES_INTEGER;
        if (PrevState == IES_REGISTER && CurrState == IES_MULTIPLY) {
          // Index Register - Register * Scale
          assert (!IndexReg && "IndexReg already set!");
          IndexReg = TmpReg;
          Scale = TmpInt;
          if(Scale != 1 && Scale != 2 && Scale != 4 && Scale != 8) {
            ErrMsg = "scale factor in address must be 1, 2, 4 or 8";
            return true;
          }
          // Get the scale and replace the 'Register * Scale' with '0'.
          IC.popOperator();
        } else if ((PrevState == IES_PLUS || PrevState == IES_MINUS ||
                    PrevState == IES_OR || PrevState == IES_AND ||
                    PrevState == IES_LSHIFT || PrevState == IES_RSHIFT ||
                    PrevState == IES_MULTIPLY || PrevState == IES_DIVIDE ||
                    PrevState == IES_LPAREN || PrevState == IES_LBRAC ||
                    PrevState == IES_NOT || PrevState == IES_XOR) &&
                   CurrState == IES_MINUS) {
          // Unary minus.  No need to pop the minus operand because it was never
          // pushed.
          IC.pushOperand(IC_IMM, -TmpInt); // Push -Imm.
        } else if ((PrevState == IES_PLUS || PrevState == IES_MINUS ||
                    PrevState == IES_OR || PrevState == IES_AND ||
                    PrevState == IES_LSHIFT || PrevState == IES_RSHIFT ||
                    PrevState == IES_MULTIPLY || PrevState == IES_DIVIDE ||
                    PrevState == IES_LPAREN || PrevState == IES_LBRAC ||
                    PrevState == IES_NOT || PrevState == IES_XOR) &&
                   CurrState == IES_NOT) {
          // Unary not.  No need to pop the not operand because it was never
          // pushed.
          IC.pushOperand(IC_IMM, ~TmpInt); // Push ~Imm.
        } else {
          IC.pushOperand(IC_IMM, TmpInt);
        }
        break;
      }
      PrevState = CurrState;
      return false;
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
      case IES_NOT:
      case IES_OR:
      case IES_XOR:
      case IES_AND:
      case IES_LSHIFT:
      case IES_RSHIFT:
      case IES_MULTIPLY:
      case IES_DIVIDE:
      case IES_LPAREN:
        // FIXME: We don't handle this type of unary minus or not, yet.
        if ((PrevState == IES_PLUS || PrevState == IES_MINUS ||
            PrevState == IES_OR || PrevState == IES_AND ||
            PrevState == IES_LSHIFT || PrevState == IES_RSHIFT ||
            PrevState == IES_MULTIPLY || PrevState == IES_DIVIDE ||
            PrevState == IES_LPAREN || PrevState == IES_LBRAC ||
            PrevState == IES_NOT || PrevState == IES_XOR) &&
            (CurrState == IES_MINUS || CurrState == IES_NOT)) {
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

  bool Error(SMLoc L, const Twine &Msg,
             ArrayRef<SMRange> Ranges = None,
             bool MatchingInlineAsm = false) {
    MCAsmParser &Parser = getParser();
    if (MatchingInlineAsm) return true;
    return Parser.Error(L, Msg, Ranges);
  }

  bool ErrorAndEatStatement(SMLoc L, const Twine &Msg,
          ArrayRef<SMRange> Ranges = None,
          bool MatchingInlineAsm = false) {
    MCAsmParser &Parser = getParser();
    Parser.eatToEndOfStatement();
    return Error(L, Msg, Ranges, MatchingInlineAsm);
  }

  std::nullptr_t ErrorOperand(SMLoc Loc, StringRef Msg) {
    Error(Loc, Msg);
    return nullptr;
  }

  std::unique_ptr<X86Operand> DefaultMemSIOperand(SMLoc Loc);
  std::unique_ptr<X86Operand> DefaultMemDIOperand(SMLoc Loc);
  void AddDefaultSrcDestOperands(
      OperandVector& Operands, std::unique_ptr<llvm::MCParsedAsmOperand> &&Src,
      std::unique_ptr<llvm::MCParsedAsmOperand> &&Dst);
  std::unique_ptr<X86Operand> ParseOperand();
  std::unique_ptr<X86Operand> ParseATTOperand();
  std::unique_ptr<X86Operand> ParseIntelOperand();
  std::unique_ptr<X86Operand> ParseIntelOffsetOfOperator();
  bool ParseIntelDotOperator(const MCExpr *Disp, const MCExpr *&NewDisp);
  std::unique_ptr<X86Operand> ParseIntelOperator(unsigned OpKind);
  std::unique_ptr<X86Operand>
  ParseIntelSegmentOverride(unsigned SegReg, SMLoc Start, unsigned Size);
  std::unique_ptr<X86Operand>
  ParseIntelMemOperand(int64_t ImmDisp, SMLoc StartLoc, unsigned Size);
  std::unique_ptr<X86Operand> ParseRoundingModeOp(SMLoc Start, SMLoc End);
  bool ParseIntelExpression(IntelExprStateMachine &SM, SMLoc &End);
  std::unique_ptr<X86Operand> ParseIntelBracExpression(unsigned SegReg,
                                                       SMLoc Start,
                                                       int64_t ImmDisp,
                                                       unsigned Size);
  bool ParseIntelIdentifier(const MCExpr *&Val, StringRef &Identifier,
                            InlineAsmIdentifierInfo &Info,
                            bool IsUnevaluatedOperand, SMLoc &End);

  std::unique_ptr<X86Operand> ParseMemOperand(unsigned SegReg, SMLoc StartLoc);

  std::unique_ptr<X86Operand>
  CreateMemForInlineAsm(unsigned SegReg, const MCExpr *Disp, unsigned BaseReg,
                        unsigned IndexReg, unsigned Scale, SMLoc Start,
                        SMLoc End, unsigned Size, StringRef Identifier,
                        InlineAsmIdentifierInfo &Info);

  bool ParseDirectiveWord(unsigned Size, SMLoc L);
  bool ParseDirectiveCode(StringRef IDVal, SMLoc L);

  bool processInstruction(MCInst &Inst, const OperandVector &Ops);

  /// Wrapper around MCStreamer::EmitInstruction(). Possibly adds
  /// instrumentation around Inst.
  void EmitInstruction(MCInst &Inst, OperandVector &Operands, MCStreamer &Out);

  bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                               OperandVector &Operands, MCStreamer &Out,
                               uint64_t &ErrorInfo,
                               bool MatchingInlineAsm) override;

  void MatchFPUWaitAlias(SMLoc IDLoc, X86Operand &Op, OperandVector &Operands,
                         MCStreamer &Out, bool MatchingInlineAsm);

  bool ErrorMissingFeature(SMLoc IDLoc, uint64_t ErrorInfo,
                           bool MatchingInlineAsm);

  bool MatchAndEmitATTInstruction(SMLoc IDLoc, unsigned &Opcode,
                                  OperandVector &Operands, MCStreamer &Out,
                                  uint64_t &ErrorInfo,
                                  bool MatchingInlineAsm);

  bool MatchAndEmitIntelInstruction(SMLoc IDLoc, unsigned &Opcode,
                                    OperandVector &Operands, MCStreamer &Out,
                                    uint64_t &ErrorInfo,
                                    bool MatchingInlineAsm);

  bool OmitRegisterFromClobberLists(unsigned RegNo) override;

  /// doSrcDstMatch - Returns true if operands are matching in their
  /// word size (%si and %di, %esi and %edi, etc.). Order depends on
  /// the parsing mode (Intel vs. AT&T).
  bool doSrcDstMatch(X86Operand &Op1, X86Operand &Op2);

  /// Parses AVX512 specific operand primitives: masked registers ({%k<NUM>}, {z})
  /// and memory broadcasting ({1to<NUM>}) primitives, updating Operands vector if required.
  /// \return \c true if no parsing errors occurred, \c false otherwise.
  bool HandleAVX512Operand(OperandVector &Operands,
                           const MCParsedAsmOperand &Op);

  bool is64BitMode() const {
    // FIXME: Can tablegen auto-generate this?
    return getSTI().getFeatureBits()[X86::Mode64Bit];
  }
  bool is32BitMode() const {
    // FIXME: Can tablegen auto-generate this?
    return getSTI().getFeatureBits()[X86::Mode32Bit];
  }
  bool is16BitMode() const {
    // FIXME: Can tablegen auto-generate this?
    return getSTI().getFeatureBits()[X86::Mode16Bit];
  }
  void SwitchMode(unsigned mode) {
    MCSubtargetInfo &STI = copySTI();
    FeatureBitset AllModes({X86::Mode64Bit, X86::Mode32Bit, X86::Mode16Bit});
    FeatureBitset OldMode = STI.getFeatureBits() & AllModes;
    unsigned FB = ComputeAvailableFeatures(
      STI.ToggleFeature(OldMode.flip(mode)));
    setAvailableFeatures(FB);

    assert(FeatureBitset({mode}) == (STI.getFeatureBits() & AllModes));
  }

  unsigned getPointerWidth() {
    if (is16BitMode()) return 16;
    if (is32BitMode()) return 32;
    if (is64BitMode()) return 64;
    llvm_unreachable("invalid mode");
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
  X86AsmParser(const MCSubtargetInfo &sti, MCAsmParser &Parser,
               const MCInstrInfo &mii, const MCTargetOptions &Options)
    : MCTargetAsmParser(Options, sti), MII(mii), InstInfo(nullptr) {

    // Initialize the set of available features.
    setAvailableFeatures(ComputeAvailableFeatures(getSTI().getFeatureBits()));
    Instrumentation.reset(
        CreateX86AsmInstrumentation(Options, Parser.getContext(), STI));
  }

  bool ParseRegister(unsigned &RegNo, SMLoc &StartLoc, SMLoc &EndLoc) override;

  void SetFrameRegister(unsigned RegNo) override;

  bool ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) override;

  bool ParseDirective(AsmToken DirectiveID) override;
};
} // end anonymous namespace

/// @name Auto-generated Match Functions
/// {

static unsigned MatchRegisterName(StringRef Name);

/// }

static bool CheckBaseRegAndIndexReg(unsigned BaseReg, unsigned IndexReg,
                                    StringRef &ErrMsg) {
  // If we have both a base register and an index register make sure they are
  // both 64-bit or 32-bit registers.
  // To support VSIB, IndexReg can be 128-bit or 256-bit registers.
  if (BaseReg != 0 && IndexReg != 0) {
    if (X86MCRegisterClasses[X86::GR64RegClassID].contains(BaseReg) &&
        (X86MCRegisterClasses[X86::GR16RegClassID].contains(IndexReg) ||
         X86MCRegisterClasses[X86::GR32RegClassID].contains(IndexReg)) &&
        IndexReg != X86::RIZ) {
      ErrMsg = "base register is 64-bit, but index register is not";
      return true;
    }
    if (X86MCRegisterClasses[X86::GR32RegClassID].contains(BaseReg) &&
        (X86MCRegisterClasses[X86::GR16RegClassID].contains(IndexReg) ||
         X86MCRegisterClasses[X86::GR64RegClassID].contains(IndexReg)) &&
        IndexReg != X86::EIZ){
      ErrMsg = "base register is 32-bit, but index register is not";
      return true;
    }
    if (X86MCRegisterClasses[X86::GR16RegClassID].contains(BaseReg)) {
      if (X86MCRegisterClasses[X86::GR32RegClassID].contains(IndexReg) ||
          X86MCRegisterClasses[X86::GR64RegClassID].contains(IndexReg)) {
        ErrMsg = "base register is 16-bit, but index register is not";
        return true;
      }
      if (((BaseReg == X86::BX || BaseReg == X86::BP) &&
           IndexReg != X86::SI && IndexReg != X86::DI) ||
          ((BaseReg == X86::SI || BaseReg == X86::DI) &&
           IndexReg != X86::BX && IndexReg != X86::BP)) {
        ErrMsg = "invalid 16-bit base/index register combination";
        return true;
      }
    }
  }
  return false;
}

bool X86AsmParser::doSrcDstMatch(X86Operand &Op1, X86Operand &Op2)
{
  // Return true and let a normal complaint about bogus operands happen.
  if (!Op1.isMem() || !Op2.isMem())
    return true;

  // Actually these might be the other way round if Intel syntax is
  // being used. It doesn't matter.
  unsigned diReg = Op1.Mem.BaseReg;
  unsigned siReg = Op2.Mem.BaseReg;

  if (X86MCRegisterClasses[X86::GR16RegClassID].contains(siReg))
    return X86MCRegisterClasses[X86::GR16RegClassID].contains(diReg);
  if (X86MCRegisterClasses[X86::GR32RegClassID].contains(siReg))
    return X86MCRegisterClasses[X86::GR32RegClassID].contains(diReg);
  if (X86MCRegisterClasses[X86::GR64RegClassID].contains(siReg))
    return X86MCRegisterClasses[X86::GR64RegClassID].contains(diReg);
  // Again, return true and let another error happen.
  return true;
}

bool X86AsmParser::ParseRegister(unsigned &RegNo,
                                 SMLoc &StartLoc, SMLoc &EndLoc) {
  MCAsmParser &Parser = getParser();
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

  // The "flags" register cannot be referenced directly.
  // Treat it as an identifier instead.
  if (isParsingInlineAsm() && isParsingIntelSyntax() && RegNo == X86::EFLAGS)
    RegNo = 0;

  if (!is64BitMode()) {
    // FIXME: This should be done using Requires<Not64BitMode> and
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

void X86AsmParser::SetFrameRegister(unsigned RegNo) {
  Instrumentation->SetInitialFrameRegister(RegNo);
}

std::unique_ptr<X86Operand> X86AsmParser::DefaultMemSIOperand(SMLoc Loc) {
  unsigned basereg =
    is64BitMode() ? X86::RSI : (is32BitMode() ? X86::ESI : X86::SI);
  const MCExpr *Disp = MCConstantExpr::create(0, getContext());
  return X86Operand::CreateMem(getPointerWidth(), /*SegReg=*/0, Disp,
                               /*BaseReg=*/basereg, /*IndexReg=*/0, /*Scale=*/1,
                               Loc, Loc, 0);
}

std::unique_ptr<X86Operand> X86AsmParser::DefaultMemDIOperand(SMLoc Loc) {
  unsigned basereg =
    is64BitMode() ? X86::RDI : (is32BitMode() ? X86::EDI : X86::DI);
  const MCExpr *Disp = MCConstantExpr::create(0, getContext());
  return X86Operand::CreateMem(getPointerWidth(), /*SegReg=*/0, Disp,
                               /*BaseReg=*/basereg, /*IndexReg=*/0, /*Scale=*/1,
                               Loc, Loc, 0);
}

void X86AsmParser::AddDefaultSrcDestOperands(
    OperandVector& Operands, std::unique_ptr<llvm::MCParsedAsmOperand> &&Src,
    std::unique_ptr<llvm::MCParsedAsmOperand> &&Dst) {
  if (isParsingIntelSyntax()) {
    Operands.push_back(std::move(Dst));
    Operands.push_back(std::move(Src));
  }
  else {
    Operands.push_back(std::move(Src));
    Operands.push_back(std::move(Dst));
  }
}

std::unique_ptr<X86Operand> X86AsmParser::ParseOperand() {
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
    .Cases("MMWORD","mmword", 64)
    .Cases("XWORD", "xword", 80)
    .Cases("TBYTE", "tbyte", 80)
    .Cases("XMMWORD", "xmmword", 128)
    .Cases("YMMWORD", "ymmword", 256)
    .Cases("ZMMWORD", "zmmword", 512)
    .Cases("OPAQUE", "opaque", -1U) // needs to be non-zero, but doesn't matter
    .Default(0);
  return Size;
}

std::unique_ptr<X86Operand> X86AsmParser::CreateMemForInlineAsm(
    unsigned SegReg, const MCExpr *Disp, unsigned BaseReg, unsigned IndexReg,
    unsigned Scale, SMLoc Start, SMLoc End, unsigned Size, StringRef Identifier,
    InlineAsmIdentifierInfo &Info) {
  // If we found a decl other than a VarDecl, then assume it is a FuncDecl or
  // some other label reference.
  if (isa<MCSymbolRefExpr>(Disp) && Info.OpDecl && !Info.IsVarDecl) {
    // Insert an explicit size if the user didn't have one.
    if (!Size) {
      Size = getPointerWidth();
      InstInfo->AsmRewrites->emplace_back(AOK_SizeDirective, Start,
                                          /*Len=*/0, Size);
    }

    // Create an absolute memory reference in order to match against
    // instructions taking a PC relative operand.
    return X86Operand::CreateMem(getPointerWidth(), Disp, Start, End, Size,
                                 Identifier, Info.OpDecl);
  }

  // We either have a direct symbol reference, or an offset from a symbol.  The
  // parser always puts the symbol on the LHS, so look there for size
  // calculation purposes.
  const MCBinaryExpr *BinOp = dyn_cast<MCBinaryExpr>(Disp);
  bool IsSymRef =
      isa<MCSymbolRefExpr>(BinOp ? BinOp->getLHS() : Disp);
  if (IsSymRef) {
    if (!Size) {
      Size = Info.Type * 8; // Size is in terms of bits in this context.
      if (Size)
        InstInfo->AsmRewrites->emplace_back(AOK_SizeDirective, Start,
                                            /*Len=*/0, Size);
    }
  }

  // When parsing inline assembly we set the base register to a non-zero value
  // if we don't know the actual value at this time.  This is necessary to
  // get the matching correct in some cases.
  BaseReg = BaseReg ? BaseReg : 1;
  return X86Operand::CreateMem(getPointerWidth(), SegReg, Disp, BaseReg,
                               IndexReg, Scale, Start, End, Size, Identifier,
                               Info.OpDecl);
}

static void
RewriteIntelBracExpression(SmallVectorImpl<AsmRewrite> &AsmRewrites,
                           StringRef SymName, int64_t ImmDisp,
                           int64_t FinalImmDisp, SMLoc &BracLoc,
                           SMLoc &StartInBrac, SMLoc &End) {
  // Remove the '[' and ']' from the IR string.
  AsmRewrites.emplace_back(AOK_Skip, BracLoc, 1);
  AsmRewrites.emplace_back(AOK_Skip, End, 1);

  // If ImmDisp is non-zero, then we parsed a displacement before the
  // bracketed expression (i.e., ImmDisp [ BaseReg + Scale*IndexReg + Disp])
  // If ImmDisp doesn't match the displacement computed by the state machine
  // then we have an additional displacement in the bracketed expression.
  if (ImmDisp != FinalImmDisp) {
    if (ImmDisp) {
      // We have an immediate displacement before the bracketed expression.
      // Adjust this to match the final immediate displacement.
      bool Found = false;
      for (AsmRewrite &AR : AsmRewrites) {
        if (AR.Loc.getPointer() > BracLoc.getPointer())
          continue;
        if (AR.Kind == AOK_ImmPrefix || AR.Kind == AOK_Imm) {
          assert (!Found && "ImmDisp already rewritten.");
          AR.Kind = AOK_Imm;
          AR.Len = BracLoc.getPointer() - AR.Loc.getPointer();
          AR.Val = FinalImmDisp;
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
      AsmRewrites.emplace_back(AOK_Imm, BracLoc, 0, FinalImmDisp);
    }
  }
  // Remove all the ImmPrefix rewrites within the brackets.
  for (AsmRewrite &AR : AsmRewrites) {
    if (AR.Loc.getPointer() < StartInBrac.getPointer())
      continue;
    if (AR.Kind == AOK_ImmPrefix)
      AR.Kind = AOK_Delete;
  }
  const char *SymLocPtr = SymName.data();
  // Skip everything before the symbol.
  if (unsigned Len = SymLocPtr - StartInBrac.getPointer()) {
    assert(Len > 0 && "Expected a non-negative length.");
    AsmRewrites.emplace_back(AOK_Skip, StartInBrac, Len);
  }
  // Skip everything after the symbol.
  if (unsigned Len = End.getPointer() - (SymLocPtr + SymName.size())) {
    SMLoc Loc = SMLoc::getFromPointer(SymLocPtr + SymName.size());
    assert(Len > 0 && "Expected a non-negative length.");
    AsmRewrites.emplace_back(AOK_Skip, Loc, Len);
  }
}

bool X86AsmParser::ParseIntelExpression(IntelExprStateMachine &SM, SMLoc &End) {
  MCAsmParser &Parser = getParser();
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

    AsmToken::TokenKind TK = getLexer().getKind();
    switch (TK) {
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
    case AsmToken::String:
    case AsmToken::Identifier: {
      // This could be a register or a symbolic displacement.
      unsigned TmpReg;
      const MCExpr *Val;
      SMLoc IdentLoc = Tok.getLoc();
      StringRef Identifier = Tok.getString();
      if (TK != AsmToken::String && !ParseRegister(TmpReg, IdentLoc, End)) {
        SM.onRegister(TmpReg);
        UpdateLocLex = false;
        break;
      } else {
        if (!isParsingInlineAsm()) {
          if (getParser().parsePrimaryExpr(Val, End))
            return Error(Tok.getLoc(), "Unexpected identifier!");
        } else {
          // This is a dot operator, not an adjacent identifier.
          if (Identifier.find('.') != StringRef::npos) {
            return false;
          } else {
            InlineAsmIdentifierInfo &Info = SM.getIdentifierInfo();
            if (ParseIntelIdentifier(Val, Identifier, Info,
                                     /*Unevaluated=*/false, End))
              return true;
          }
        }
        SM.onIdentifierExpr(Val, Identifier);
        UpdateLocLex = false;
        break;
      }
      return Error(Tok.getLoc(), "Unexpected identifier!");
    }
    case AsmToken::Integer: {
      StringRef ErrMsg;
      if (isParsingInlineAsm() && SM.getAddImmPrefix())
        InstInfo->AsmRewrites->emplace_back(AOK_ImmPrefix, Tok.getLoc());
      // Look for 'b' or 'f' following an Integer as a directional label
      SMLoc Loc = getTok().getLoc();
      int64_t IntVal = getTok().getIntVal();
      End = consumeToken();
      UpdateLocLex = false;
      if (getLexer().getKind() == AsmToken::Identifier) {
        StringRef IDVal = getTok().getString();
        if (IDVal == "f" || IDVal == "b") {
          MCSymbol *Sym =
              getContext().getDirectionalLocalSymbol(IntVal, IDVal == "b");
          MCSymbolRefExpr::VariantKind Variant = MCSymbolRefExpr::VK_None;
          const MCExpr *Val =
              MCSymbolRefExpr::create(Sym, Variant, getContext());
          if (IDVal == "b" && Sym->isUndefined())
            return Error(Loc, "invalid reference to undefined symbol");
          StringRef Identifier = Sym->getName();
          SM.onIdentifierExpr(Val, Identifier);
          End = consumeToken();
        } else {
          if (SM.onInteger(IntVal, ErrMsg))
            return Error(Loc, ErrMsg);
        }
      } else {
        if (SM.onInteger(IntVal, ErrMsg))
          return Error(Loc, ErrMsg);
      }
      break;
    }
    case AsmToken::Plus:    SM.onPlus(); break;
    case AsmToken::Minus:   SM.onMinus(); break;
    case AsmToken::Tilde:   SM.onNot(); break;
    case AsmToken::Star:    SM.onStar(); break;
    case AsmToken::Slash:   SM.onDivide(); break;
    case AsmToken::Pipe:    SM.onOr(); break;
    case AsmToken::Caret:   SM.onXor(); break;
    case AsmToken::Amp:     SM.onAnd(); break;
    case AsmToken::LessLess:
                            SM.onLShift(); break;
    case AsmToken::GreaterGreater:
                            SM.onRShift(); break;
    case AsmToken::LBrac:   SM.onLBrac(); break;
    case AsmToken::RBrac:   SM.onRBrac(); break;
    case AsmToken::LParen:  SM.onLParen(); break;
    case AsmToken::RParen:  SM.onRParen(); break;
    }
    if (SM.hadError())
      return Error(Tok.getLoc(), "unknown token in expression");

    if (!Done && UpdateLocLex)
      End = consumeToken();
  }
  return false;
}

std::unique_ptr<X86Operand>
X86AsmParser::ParseIntelBracExpression(unsigned SegReg, SMLoc Start,
                                       int64_t ImmDisp, unsigned Size) {
  MCAsmParser &Parser = getParser();
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
    return nullptr;

  const MCExpr *Disp = nullptr;
  if (const MCExpr *Sym = SM.getSym()) {
    // A symbolic displacement.
    Disp = Sym;
    if (isParsingInlineAsm())
      RewriteIntelBracExpression(*InstInfo->AsmRewrites, SM.getSymName(),
                                 ImmDisp, SM.getImm(), BracLoc, StartInBrac,
                                 End);
  }

  if (SM.getImm() || !Disp) {
    const MCExpr *Imm = MCConstantExpr::create(SM.getImm(), getContext());
    if (Disp)
      Disp = MCBinaryExpr::createAdd(Disp, Imm, getContext());
    else
      Disp = Imm;  // An immediate displacement only.
  }

  // Parse struct field access.  Intel requires a dot, but MSVC doesn't.  MSVC
  // will in fact do global lookup the field name inside all global typedefs,
  // but we don't emulate that.
  if (Tok.getString().find('.') != StringRef::npos) {
    const MCExpr *NewDisp;
    if (ParseIntelDotOperator(Disp, NewDisp))
      return nullptr;

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
        return X86Operand::CreateMem(getPointerWidth(), Disp, Start, End, Size);
      return X86Operand::CreateMem(getPointerWidth(), SegReg, Disp, 0, 0, 1,
                                   Start, End, Size);
    }
    StringRef ErrMsg;
    if (CheckBaseRegAndIndexReg(BaseReg, IndexReg, ErrMsg)) {
      Error(StartInBrac, ErrMsg);
      return nullptr;
    }
    return X86Operand::CreateMem(getPointerWidth(), SegReg, Disp, BaseReg,
                                 IndexReg, Scale, Start, End, Size);
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
  MCAsmParser &Parser = getParser();
  assert(isParsingInlineAsm() && "Expected to be parsing inline assembly.");
  Val = nullptr;

  StringRef LineBuf(Identifier.data());
  void *Result =
    SemaCallback->LookupInlineAsmIdentifier(LineBuf, Info, IsUnevaluatedOperand);

  const AsmToken &Tok = Parser.getTok();
  SMLoc Loc = Tok.getLoc();

  // Advance the token stream until the end of the current token is
  // after the end of what the frontend claimed.
  const char *EndPtr = Tok.getLoc().getPointer() + LineBuf.size();
  do {
    End = Tok.getEndLoc();
    getLexer().Lex();
  } while (End.getPointer() < EndPtr);
  Identifier = LineBuf;

  // The frontend should end parsing on an assembler token boundary, unless it
  // failed parsing.
  assert((End.getPointer() == EndPtr || !Result) &&
         "frontend claimed part of a token?");

  // If the identifier lookup was unsuccessful, assume that we are dealing with
  // a label.
  if (!Result) {
    StringRef InternalName =
      SemaCallback->LookupInlineAsmLabel(Identifier, getSourceManager(),
                                         Loc, false);
    assert(InternalName.size() && "We should have an internal name here.");
    // Push a rewrite for replacing the identifier name with the internal name.
    InstInfo->AsmRewrites->emplace_back(AOK_Label, Loc, Identifier.size(),
                                        InternalName);
  }

  // Create the symbol reference.
  MCSymbol *Sym = getContext().getOrCreateSymbol(Identifier);
  MCSymbolRefExpr::VariantKind Variant = MCSymbolRefExpr::VK_None;
  Val = MCSymbolRefExpr::create(Sym, Variant, getParser().getContext());
  return false;
}

/// \brief Parse intel style segment override.
std::unique_ptr<X86Operand>
X86AsmParser::ParseIntelSegmentOverride(unsigned SegReg, SMLoc Start,
                                        unsigned Size) {
  MCAsmParser &Parser = getParser();
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
      InstInfo->AsmRewrites->emplace_back(AOK_ImmPrefix, ImmDispToken.getLoc());

    if (getLexer().isNot(AsmToken::LBrac)) {
      // An immediate following a 'segment register', 'colon' token sequence can
      // be followed by a bracketed expression.  If it isn't we know we have our
      // final segment override.
      const MCExpr *Disp = MCConstantExpr::create(ImmDisp, getContext());
      return X86Operand::CreateMem(getPointerWidth(), SegReg, Disp,
                                   /*BaseReg=*/0, /*IndexReg=*/0, /*Scale=*/1,
                                   Start, ImmDispToken.getEndLoc(), Size);
    }
  }

  if (getLexer().is(AsmToken::LBrac))
    return ParseIntelBracExpression(SegReg, Start, ImmDisp, Size);

  const MCExpr *Val;
  SMLoc End;
  if (!isParsingInlineAsm()) {
    if (getParser().parsePrimaryExpr(Val, End))
      return ErrorOperand(Tok.getLoc(), "unknown token in expression");

    return X86Operand::CreateMem(getPointerWidth(), Val, Start, End, Size);
  }

  InlineAsmIdentifierInfo Info;
  StringRef Identifier = Tok.getString();
  if (ParseIntelIdentifier(Val, Identifier, Info,
                           /*Unevaluated=*/false, End))
    return nullptr;
  return CreateMemForInlineAsm(/*SegReg=*/0, Val, /*BaseReg=*/0,/*IndexReg=*/0,
                               /*Scale=*/1, Start, End, Size, Identifier, Info);
}

//ParseRoundingModeOp - Parse AVX-512 rounding mode operand
std::unique_ptr<X86Operand>
X86AsmParser::ParseRoundingModeOp(SMLoc Start, SMLoc End) {
  MCAsmParser &Parser = getParser();
  const AsmToken &Tok = Parser.getTok();
  // Eat "{" and mark the current place.
  const SMLoc consumedToken = consumeToken();
  if (Tok.getIdentifier().startswith("r")){
    int rndMode = StringSwitch<int>(Tok.getIdentifier())
      .Case("rn", X86::STATIC_ROUNDING::TO_NEAREST_INT)
      .Case("rd", X86::STATIC_ROUNDING::TO_NEG_INF)
      .Case("ru", X86::STATIC_ROUNDING::TO_POS_INF)
      .Case("rz", X86::STATIC_ROUNDING::TO_ZERO)
      .Default(-1);
    if (-1 == rndMode)
      return ErrorOperand(Tok.getLoc(), "Invalid rounding mode.");
     Parser.Lex();  // Eat "r*" of r*-sae
    if (!getLexer().is(AsmToken::Minus))
      return ErrorOperand(Tok.getLoc(), "Expected - at this point");
    Parser.Lex();  // Eat "-"
    Parser.Lex();  // Eat the sae
    if (!getLexer().is(AsmToken::RCurly))
      return ErrorOperand(Tok.getLoc(), "Expected } at this point");
    Parser.Lex();  // Eat "}"
    const MCExpr *RndModeOp =
      MCConstantExpr::create(rndMode, Parser.getContext());
    return X86Operand::CreateImm(RndModeOp, Start, End);
  }
  if(Tok.getIdentifier().equals("sae")){
    Parser.Lex();  // Eat the sae
    if (!getLexer().is(AsmToken::RCurly))
      return ErrorOperand(Tok.getLoc(), "Expected } at this point");
    Parser.Lex();  // Eat "}"
    return X86Operand::CreateToken("{sae}", consumedToken);
  }
  return ErrorOperand(Tok.getLoc(), "unknown token in expression");
}
/// ParseIntelMemOperand - Parse intel style memory operand.
std::unique_ptr<X86Operand> X86AsmParser::ParseIntelMemOperand(int64_t ImmDisp,
                                                               SMLoc Start,
                                                               unsigned Size) {
  MCAsmParser &Parser = getParser();
  const AsmToken &Tok = Parser.getTok();
  SMLoc End;

  // Parse ImmDisp [ BaseReg + Scale*IndexReg + Disp ].
  if (getLexer().is(AsmToken::LBrac))
    return ParseIntelBracExpression(/*SegReg=*/0, Start, ImmDisp, Size);
  assert(ImmDisp == 0);

  const MCExpr *Val;
  if (!isParsingInlineAsm()) {
    if (getParser().parsePrimaryExpr(Val, End))
      return ErrorOperand(Tok.getLoc(), "unknown token in expression");

    return X86Operand::CreateMem(getPointerWidth(), Val, Start, End, Size);
  }

  InlineAsmIdentifierInfo Info;
  StringRef Identifier = Tok.getString();
  if (ParseIntelIdentifier(Val, Identifier, Info,
                           /*Unevaluated=*/false, End))
    return nullptr;

  if (!getLexer().is(AsmToken::LBrac))
    return CreateMemForInlineAsm(/*SegReg=*/0, Val, /*BaseReg=*/0, /*IndexReg=*/0,
                                 /*Scale=*/1, Start, End, Size, Identifier, Info);

  Parser.Lex(); // Eat '['

  // Parse Identifier [ ImmDisp ]
  IntelExprStateMachine SM(/*ImmDisp=*/0, /*StopOnLBrac=*/true,
                           /*AddImmPrefix=*/false);
  if (ParseIntelExpression(SM, End))
    return nullptr;

  if (SM.getSym()) {
    Error(Start, "cannot use more than one symbol in memory operand");
    return nullptr;
  }
  if (SM.getBaseReg()) {
    Error(Start, "cannot use base register with variable reference");
    return nullptr;
  }
  if (SM.getIndexReg()) {
    Error(Start, "cannot use index register with variable reference");
    return nullptr;
  }

  const MCExpr *Disp = MCConstantExpr::create(SM.getImm(), getContext());
  // BaseReg is non-zero to avoid assertions.  In the context of inline asm,
  // we're pointing to a local variable in memory, so the base register is
  // really the frame or stack pointer.
  return X86Operand::CreateMem(getPointerWidth(), /*SegReg=*/0, Disp,
                               /*BaseReg=*/1, /*IndexReg=*/0, /*Scale=*/1,
                               Start, End, Size, Identifier, Info.OpDecl);
}

/// Parse the '.' operator.
bool X86AsmParser::ParseIntelDotOperator(const MCExpr *Disp,
                                                const MCExpr *&NewDisp) {
  MCAsmParser &Parser = getParser();
  const AsmToken &Tok = Parser.getTok();
  int64_t OrigDispVal, DotDispVal;

  // FIXME: Handle non-constant expressions.
  if (const MCConstantExpr *OrigDisp = dyn_cast<MCConstantExpr>(Disp))
    OrigDispVal = OrigDisp->getValue();
  else
    return Error(Tok.getLoc(), "Non-constant offsets are not supported!");

  // Drop the optional '.'.
  StringRef DotDispStr = Tok.getString();
  if (DotDispStr.startswith("."))
    DotDispStr = DotDispStr.drop_front(1);

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
    InstInfo->AsmRewrites->emplace_back(AOK_DotOperator, Loc, Len, Val);
  }

  NewDisp = MCConstantExpr::create(OrigDispVal + DotDispVal, getContext());
  return false;
}

/// Parse the 'offset' operator.  This operator is used to specify the
/// location rather then the content of a variable.
std::unique_ptr<X86Operand> X86AsmParser::ParseIntelOffsetOfOperator() {
  MCAsmParser &Parser = getParser();
  const AsmToken &Tok = Parser.getTok();
  SMLoc OffsetOfLoc = Tok.getLoc();
  Parser.Lex(); // Eat offset.

  const MCExpr *Val;
  InlineAsmIdentifierInfo Info;
  SMLoc Start = Tok.getLoc(), End;
  StringRef Identifier = Tok.getString();
  if (ParseIntelIdentifier(Val, Identifier, Info,
                           /*Unevaluated=*/false, End))
    return nullptr;

  // Don't emit the offset operator.
  InstInfo->AsmRewrites->emplace_back(AOK_Skip, OffsetOfLoc, 7);

  // The offset operator will have an 'r' constraint, thus we need to create
  // register operand to ensure proper matching.  Just pick a GPR based on
  // the size of a pointer.
  unsigned RegNo =
      is64BitMode() ? X86::RBX : (is32BitMode() ? X86::EBX : X86::BX);
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
std::unique_ptr<X86Operand> X86AsmParser::ParseIntelOperator(unsigned OpKind) {
  MCAsmParser &Parser = getParser();
  const AsmToken &Tok = Parser.getTok();
  SMLoc TypeLoc = Tok.getLoc();
  Parser.Lex(); // Eat operator.

  const MCExpr *Val = nullptr;
  InlineAsmIdentifierInfo Info;
  SMLoc Start = Tok.getLoc(), End;
  StringRef Identifier = Tok.getString();
  if (ParseIntelIdentifier(Val, Identifier, Info,
                           /*Unevaluated=*/true, End))
    return nullptr;

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
  InstInfo->AsmRewrites->emplace_back(AOK_Imm, TypeLoc, Len, CVal);

  const MCExpr *Imm = MCConstantExpr::create(CVal, getContext());
  return X86Operand::CreateImm(Imm, Start, End);
}

std::unique_ptr<X86Operand> X86AsmParser::ParseIntelOperand() {
  MCAsmParser &Parser = getParser();
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

  bool PtrInOperand = false;
  unsigned Size = getIntelMemOperandSize(Tok.getString());
  if (Size) {
    Parser.Lex(); // Eat operand size (e.g., byte, word).
    if (Tok.getString() != "PTR" && Tok.getString() != "ptr")
      return ErrorOperand(Tok.getLoc(), "Expected 'PTR' or 'ptr' token!");
    Parser.Lex(); // Eat ptr.
    PtrInOperand = true;
  }
  Start = Tok.getLoc();

  // Immediate.
  if (getLexer().is(AsmToken::Integer) || getLexer().is(AsmToken::Minus) ||
      getLexer().is(AsmToken::Tilde) || getLexer().is(AsmToken::LParen)) {
    AsmToken StartTok = Tok;
    IntelExprStateMachine SM(/*Imm=*/0, /*StopOnLBrac=*/true,
                             /*AddImmPrefix=*/false);
    if (ParseIntelExpression(SM, End))
      return nullptr;

    int64_t Imm = SM.getImm();
    if (isParsingInlineAsm()) {
      unsigned Len = Tok.getLoc().getPointer() - Start.getPointer();
      if (StartTok.getString().size() == Len)
        // Just add a prefix if this wasn't a complex immediate expression.
        InstInfo->AsmRewrites->emplace_back(AOK_ImmPrefix, Start);
      else
        // Otherwise, rewrite the complex expression as a single immediate.
        InstInfo->AsmRewrites->emplace_back(AOK_Imm, Start, Len, Imm);
    }

    if (getLexer().isNot(AsmToken::LBrac)) {
      // If a directional label (ie. 1f or 2b) was parsed above from
      // ParseIntelExpression() then SM.getSym() was set to a pointer to
      // to the MCExpr with the directional local symbol and this is a
      // memory operand not an immediate operand.
      if (SM.getSym())
        return X86Operand::CreateMem(getPointerWidth(), SM.getSym(), Start, End,
                                     Size);

      const MCExpr *ImmExpr = MCConstantExpr::create(Imm, getContext());
      return X86Operand::CreateImm(ImmExpr, Start, End);
    }

    // Only positive immediates are valid.
    if (Imm < 0)
      return ErrorOperand(Start, "expected a positive immediate displacement "
                          "before bracketed expr.");

    // Parse ImmDisp [ BaseReg + Scale*IndexReg + Disp ].
    return ParseIntelMemOperand(Imm, Start, Size);
  }

  // rounding mode token
  if (getSTI().getFeatureBits()[X86::FeatureAVX512] &&
      getLexer().is(AsmToken::LCurly))
    return ParseRoundingModeOp(Start, End);

  // Register.
  unsigned RegNo = 0;
  if (!ParseRegister(RegNo, Start, End)) {
    // If this is a segment register followed by a ':', then this is the start
    // of a segment override, otherwise this is a normal register reference.
    // In case it is a normal register and there is ptr in the operand this 
    // is an error
    if (getLexer().isNot(AsmToken::Colon)){
      if (PtrInOperand){
        return ErrorOperand(Start, "expected memory operand after "
                                   "'ptr', found register operand instead");
      }
      return X86Operand::CreateReg(RegNo, Start, End);
    }
    
    return ParseIntelSegmentOverride(/*SegReg=*/RegNo, Start, Size);
  }

  // Memory operand.
  return ParseIntelMemOperand(/*Disp=*/0, Start, Size);
}

std::unique_ptr<X86Operand> X86AsmParser::ParseATTOperand() {
  MCAsmParser &Parser = getParser();
  switch (getLexer().getKind()) {
  default:
    // Parse a memory operand with no segment register.
    return ParseMemOperand(0, Parser.getTok().getLoc());
  case AsmToken::Percent: {
    // Read the register.
    unsigned RegNo;
    SMLoc Start, End;
    if (ParseRegister(RegNo, Start, End)) return nullptr;
    if (RegNo == X86::EIZ || RegNo == X86::RIZ) {
      Error(Start, "%eiz and %riz can only be used as index registers",
            SMRange(Start, End));
      return nullptr;
    }

    // If this is a segment register followed by a ':', then this is the start
    // of a memory reference, otherwise this is a normal register reference.
    if (getLexer().isNot(AsmToken::Colon))
      return X86Operand::CreateReg(RegNo, Start, End);

    if (!X86MCRegisterClasses[X86::SEGMENT_REGRegClassID].contains(RegNo))
      return ErrorOperand(Start, "invalid segment register");

    getParser().Lex(); // Eat the colon.
    return ParseMemOperand(RegNo, Start);
  }
  case AsmToken::Dollar: {
    // $42 -> immediate.
    SMLoc Start = Parser.getTok().getLoc(), End;
    Parser.Lex();
    const MCExpr *Val;
    if (getParser().parseExpression(Val, End))
      return nullptr;
    return X86Operand::CreateImm(Val, Start, End);
  }
  case AsmToken::LCurly:{
    SMLoc Start = Parser.getTok().getLoc(), End;
    if (getSTI().getFeatureBits()[X86::FeatureAVX512])
      return ParseRoundingModeOp(Start, End);
    return ErrorOperand(Start, "unknown token in expression");
  }
  }
}

bool X86AsmParser::HandleAVX512Operand(OperandVector &Operands,
                                       const MCParsedAsmOperand &Op) {
  MCAsmParser &Parser = getParser();
  if(getSTI().getFeatureBits()[X86::FeatureAVX512]) {
    if (getLexer().is(AsmToken::LCurly)) {
      // Eat "{" and mark the current place.
      const SMLoc consumedToken = consumeToken();
      // Distinguish {1to<NUM>} from {%k<NUM>}.
      if(getLexer().is(AsmToken::Integer)) {
        // Parse memory broadcasting ({1to<NUM>}).
        if (getLexer().getTok().getIntVal() != 1)
          return !ErrorAndEatStatement(getLexer().getLoc(),
                                       "Expected 1to<NUM> at this point");
        Parser.Lex();  // Eat "1" of 1to8
        if (!getLexer().is(AsmToken::Identifier) ||
            !getLexer().getTok().getIdentifier().startswith("to"))
          return !ErrorAndEatStatement(getLexer().getLoc(),
                                       "Expected 1to<NUM> at this point");
        // Recognize only reasonable suffixes.
        const char *BroadcastPrimitive =
          StringSwitch<const char*>(getLexer().getTok().getIdentifier())
            .Case("to2",  "{1to2}")
            .Case("to4",  "{1to4}")
            .Case("to8",  "{1to8}")
            .Case("to16", "{1to16}")
            .Default(nullptr);
        if (!BroadcastPrimitive)
          return !ErrorAndEatStatement(getLexer().getLoc(),
                                       "Invalid memory broadcast primitive.");
        Parser.Lex();  // Eat "toN" of 1toN
        if (!getLexer().is(AsmToken::RCurly))
          return !ErrorAndEatStatement(getLexer().getLoc(),
                                       "Expected } at this point");
        Parser.Lex();  // Eat "}"
        Operands.push_back(X86Operand::CreateToken(BroadcastPrimitive,
                                                   consumedToken));
        // No AVX512 specific primitives can pass
        // after memory broadcasting, so return.
        return true;
      } else {
        // Parse mask register {%k1}
        Operands.push_back(X86Operand::CreateToken("{", consumedToken));
        if (std::unique_ptr<X86Operand> Op = ParseOperand()) {
          Operands.push_back(std::move(Op));
          if (!getLexer().is(AsmToken::RCurly))
            return !ErrorAndEatStatement(getLexer().getLoc(),
                                         "Expected } at this point");
          Operands.push_back(X86Operand::CreateToken("}", consumeToken()));

          // Parse "zeroing non-masked" semantic {z}
          if (getLexer().is(AsmToken::LCurly)) {
            Operands.push_back(X86Operand::CreateToken("{z}", consumeToken()));
            if (!getLexer().is(AsmToken::Identifier) ||
                getLexer().getTok().getIdentifier() != "z")
              return !ErrorAndEatStatement(getLexer().getLoc(),
                                           "Expected z at this point");
            Parser.Lex();  // Eat the z
            if (!getLexer().is(AsmToken::RCurly))
              return !ErrorAndEatStatement(getLexer().getLoc(),
                                           "Expected } at this point");
            Parser.Lex();  // Eat the }
          }
        }
      }
    }
  }
  return true;
}

/// ParseMemOperand: segment: disp(basereg, indexreg, scale).  The '%ds:' prefix
/// has already been parsed if present.
std::unique_ptr<X86Operand> X86AsmParser::ParseMemOperand(unsigned SegReg,
                                                          SMLoc MemStart) {

  MCAsmParser &Parser = getParser();
  // We have to disambiguate a parenthesized expression "(4+5)" from the start
  // of a memory operand with a missing displacement "(%ebx)" or "(,%eax)".  The
  // only way to do this without lookahead is to eat the '(' and see what is
  // after it.
  const MCExpr *Disp = MCConstantExpr::create(0, getParser().getContext());
  if (getLexer().isNot(AsmToken::LParen)) {
    SMLoc ExprEnd;
    if (getParser().parseExpression(Disp, ExprEnd)) return nullptr;

    // After parsing the base expression we could either have a parenthesized
    // memory address or not.  If not, return now.  If so, eat the (.
    if (getLexer().isNot(AsmToken::LParen)) {
      // Unless we have a segment register, treat this as an immediate.
      if (SegReg == 0)
        return X86Operand::CreateMem(getPointerWidth(), Disp, MemStart, ExprEnd);
      return X86Operand::CreateMem(getPointerWidth(), SegReg, Disp, 0, 0, 1,
                                   MemStart, ExprEnd);
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
        return nullptr;

      // After parsing the base expression we could either have a parenthesized
      // memory address or not.  If not, return now.  If so, eat the (.
      if (getLexer().isNot(AsmToken::LParen)) {
        // Unless we have a segment register, treat this as an immediate.
        if (SegReg == 0)
          return X86Operand::CreateMem(getPointerWidth(), Disp, LParenLoc,
                                       ExprEnd);
        return X86Operand::CreateMem(getPointerWidth(), SegReg, Disp, 0, 0, 1,
                                     MemStart, ExprEnd);
      }

      // Eat the '('.
      Parser.Lex();
    }
  }

  // If we reached here, then we just ate the ( of the memory operand.  Process
  // the rest of the memory operand.
  unsigned BaseReg = 0, IndexReg = 0, Scale = 1;
  SMLoc IndexLoc, BaseLoc;

  if (getLexer().is(AsmToken::Percent)) {
    SMLoc StartLoc, EndLoc;
    BaseLoc = Parser.getTok().getLoc();
    if (ParseRegister(BaseReg, StartLoc, EndLoc)) return nullptr;
    if (BaseReg == X86::EIZ || BaseReg == X86::RIZ) {
      Error(StartLoc, "eiz and riz can only be used as index registers",
            SMRange(StartLoc, EndLoc));
      return nullptr;
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
      if (ParseRegister(IndexReg, L, L)) return nullptr;

      if (getLexer().isNot(AsmToken::RParen)) {
        // Parse the scale amount:
        //  ::= ',' [scale-expression]
        if (getLexer().isNot(AsmToken::Comma)) {
          Error(Parser.getTok().getLoc(),
                "expected comma in scale expression");
          return nullptr;
        }
        Parser.Lex(); // Eat the comma.

        if (getLexer().isNot(AsmToken::RParen)) {
          SMLoc Loc = Parser.getTok().getLoc();

          int64_t ScaleVal;
          if (getParser().parseAbsoluteExpression(ScaleVal)){
            Error(Loc, "expected scale expression");
            return nullptr;
          }

          // Validate the scale amount.
          if (X86MCRegisterClasses[X86::GR16RegClassID].contains(BaseReg) &&
              ScaleVal != 1) {
            Error(Loc, "scale factor in 16-bit address must be 1");
            return nullptr;
          }
          if (ScaleVal != 1 && ScaleVal != 2 && ScaleVal != 4 &&
              ScaleVal != 8) {
            Error(Loc, "scale factor in address must be 1, 2, 4 or 8");
            return nullptr;
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
        return nullptr;

      if (Value != 1)
        Warning(Loc, "scale factor without index register is ignored");
      Scale = 1;
    }
  }

  // Ok, we've eaten the memory operand, verify we have a ')' and eat it too.
  if (getLexer().isNot(AsmToken::RParen)) {
    Error(Parser.getTok().getLoc(), "unexpected token in memory operand");
    return nullptr;
  }
  SMLoc MemEnd = Parser.getTok().getEndLoc();
  Parser.Lex(); // Eat the ')'.

  // Check for use of invalid 16-bit registers. Only BX/BP/SI/DI are allowed,
  // and then only in non-64-bit modes. Except for DX, which is a special case
  // because an unofficial form of in/out instructions uses it.
  if (X86MCRegisterClasses[X86::GR16RegClassID].contains(BaseReg) &&
      (is64BitMode() || (BaseReg != X86::BX && BaseReg != X86::BP &&
                         BaseReg != X86::SI && BaseReg != X86::DI)) &&
      BaseReg != X86::DX) {
    Error(BaseLoc, "invalid 16-bit base register");
    return nullptr;
  }
  if (BaseReg == 0 &&
      X86MCRegisterClasses[X86::GR16RegClassID].contains(IndexReg)) {
    Error(IndexLoc, "16-bit memory operand may not include only index register");
    return nullptr;
  }

  StringRef ErrMsg;
  if (CheckBaseRegAndIndexReg(BaseReg, IndexReg, ErrMsg)) {
    Error(BaseLoc, ErrMsg);
    return nullptr;
  }

  if (SegReg || BaseReg || IndexReg)
    return X86Operand::CreateMem(getPointerWidth(), SegReg, Disp, BaseReg,
                                 IndexReg, Scale, MemStart, MemEnd);
  return X86Operand::CreateMem(getPointerWidth(), Disp, MemStart, MemEnd);
}

bool X86AsmParser::ParseInstruction(ParseInstructionInfo &Info, StringRef Name,
                                    SMLoc NameLoc, OperandVector &Operands) {
  MCAsmParser &Parser = getParser();
  InstInfo = &Info;
  StringRef PatchedName = Name;

  // FIXME: Hack to recognize setneb as setne.
  if (PatchedName.startswith("set") && PatchedName.endswith("b") &&
      PatchedName != "setb" && PatchedName != "setnb")
    PatchedName = PatchedName.substr(0, Name.size()-1);

  // FIXME: Hack to recognize cmp<comparison code>{ss,sd,ps,pd}.
  if ((PatchedName.startswith("cmp") || PatchedName.startswith("vcmp")) &&
      (PatchedName.endswith("ss") || PatchedName.endswith("sd") ||
       PatchedName.endswith("ps") || PatchedName.endswith("pd"))) {
    bool IsVCMP = PatchedName[0] == 'v';
    unsigned CCIdx = IsVCMP ? 4 : 3;
    unsigned ComparisonCode = StringSwitch<unsigned>(
      PatchedName.slice(CCIdx, PatchedName.size() - 2))
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
    if (ComparisonCode != ~0U && (IsVCMP || ComparisonCode < 8)) {

      Operands.push_back(X86Operand::CreateToken(PatchedName.slice(0, CCIdx),
                                                 NameLoc));

      const MCExpr *ImmOp = MCConstantExpr::create(ComparisonCode,
                                                   getParser().getContext());
      Operands.push_back(X86Operand::CreateImm(ImmOp, NameLoc, NameLoc));

      PatchedName = PatchedName.substr(PatchedName.size() - 2);
    }
  }

  // FIXME: Hack to recognize vpcmp<comparison code>{ub,uw,ud,uq,b,w,d,q}.
  if (PatchedName.startswith("vpcmp") &&
      (PatchedName.endswith("b") || PatchedName.endswith("w") ||
       PatchedName.endswith("d") || PatchedName.endswith("q"))) {
    unsigned CCIdx = PatchedName.drop_back().back() == 'u' ? 2 : 1;
    unsigned ComparisonCode = StringSwitch<unsigned>(
      PatchedName.slice(5, PatchedName.size() - CCIdx))
      .Case("eq",    0x0) // Only allowed on unsigned. Checked below.
      .Case("lt",    0x1)
      .Case("le",    0x2)
      //.Case("false", 0x3) // Not a documented alias.
      .Case("neq",   0x4)
      .Case("nlt",   0x5)
      .Case("nle",   0x6)
      //.Case("true",  0x7) // Not a documented alias.
      .Default(~0U);
    if (ComparisonCode != ~0U && (ComparisonCode != 0 || CCIdx == 2)) {
      Operands.push_back(X86Operand::CreateToken("vpcmp", NameLoc));

      const MCExpr *ImmOp = MCConstantExpr::create(ComparisonCode,
                                                   getParser().getContext());
      Operands.push_back(X86Operand::CreateImm(ImmOp, NameLoc, NameLoc));

      PatchedName = PatchedName.substr(PatchedName.size() - CCIdx);
    }
  }

  // FIXME: Hack to recognize vpcom<comparison code>{ub,uw,ud,uq,b,w,d,q}.
  if (PatchedName.startswith("vpcom") &&
      (PatchedName.endswith("b") || PatchedName.endswith("w") ||
       PatchedName.endswith("d") || PatchedName.endswith("q"))) {
    unsigned CCIdx = PatchedName.drop_back().back() == 'u' ? 2 : 1;
    unsigned ComparisonCode = StringSwitch<unsigned>(
      PatchedName.slice(5, PatchedName.size() - CCIdx))
      .Case("lt",    0x0)
      .Case("le",    0x1)
      .Case("gt",    0x2)
      .Case("ge",    0x3)
      .Case("eq",    0x4)
      .Case("neq",   0x5)
      .Case("false", 0x6)
      .Case("true",  0x7)
      .Default(~0U);
    if (ComparisonCode != ~0U) {
      Operands.push_back(X86Operand::CreateToken("vpcom", NameLoc));

      const MCExpr *ImmOp = MCConstantExpr::create(ComparisonCode,
                                                   getParser().getContext());
      Operands.push_back(X86Operand::CreateImm(ImmOp, NameLoc, NameLoc));

      PatchedName = PatchedName.substr(PatchedName.size() - CCIdx);
    }
  }

  Operands.push_back(X86Operand::CreateToken(PatchedName, NameLoc));

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
    if (getLexer().is(AsmToken::Star))
      Operands.push_back(X86Operand::CreateToken("*", consumeToken()));

    // Read the operands.
    while(1) {
      if (std::unique_ptr<X86Operand> Op = ParseOperand()) {
        Operands.push_back(std::move(Op));
        if (!HandleAVX512Operand(Operands, *Operands.back()))
          return true;
      } else {
         Parser.eatToEndOfStatement();
         return true;
      }
      // check for comma and eat it
      if (getLexer().is(AsmToken::Comma))
        Parser.Lex();
      else
        break;
     }

    if (getLexer().isNot(AsmToken::EndOfStatement))
      return ErrorAndEatStatement(getLexer().getLoc(),
                                  "unexpected token in argument list");
   }

  // Consume the EndOfStatement or the prefix separator Slash
  if (getLexer().is(AsmToken::EndOfStatement) ||
      (isPrefix && getLexer().is(AsmToken::Slash)))
    Parser.Lex();

  // This is for gas compatibility and cannot be done in td.
  // Adding "p" for some floating point with no argument.
  // For example: fsub --> fsubp
  bool IsFp =
    Name == "fsub" || Name == "fdiv" || Name == "fsubr" || Name == "fdivr";
  if (IsFp && Operands.size() == 1) {
    const char *Repl = StringSwitch<const char *>(Name)
      .Case("fsub", "fsubp")
      .Case("fdiv", "fdivp")
      .Case("fsubr", "fsubrp")
      .Case("fdivr", "fdivrp");
    static_cast<X86Operand &>(*Operands[0]).setTokenValue(Repl);
  }

  // This is a terrible hack to handle "out[bwl]? %al, (%dx)" ->
  // "outb %al, %dx".  Out doesn't take a memory form, but this is a widely
  // documented form in various unofficial manuals, so a lot of code uses it.
  if ((Name == "outb" || Name == "outw" || Name == "outl" || Name == "out") &&
      Operands.size() == 3) {
    X86Operand &Op = (X86Operand &)*Operands.back();
    if (Op.isMem() && Op.Mem.SegReg == 0 &&
        isa<MCConstantExpr>(Op.Mem.Disp) &&
        cast<MCConstantExpr>(Op.Mem.Disp)->getValue() == 0 &&
        Op.Mem.BaseReg == MatchRegisterName("dx") && Op.Mem.IndexReg == 0) {
      SMLoc Loc = Op.getEndLoc();
      Operands.back() = X86Operand::CreateReg(Op.Mem.BaseReg, Loc, Loc);
    }
  }
  // Same hack for "in[bwl]? (%dx), %al" -> "inb %dx, %al".
  if ((Name == "inb" || Name == "inw" || Name == "inl" || Name == "in") &&
      Operands.size() == 3) {
    X86Operand &Op = (X86Operand &)*Operands[1];
    if (Op.isMem() && Op.Mem.SegReg == 0 &&
        isa<MCConstantExpr>(Op.Mem.Disp) &&
        cast<MCConstantExpr>(Op.Mem.Disp)->getValue() == 0 &&
        Op.Mem.BaseReg == MatchRegisterName("dx") && Op.Mem.IndexReg == 0) {
      SMLoc Loc = Op.getEndLoc();
      Operands[1] = X86Operand::CreateReg(Op.Mem.BaseReg, Loc, Loc);
    }
  }

  // Append default arguments to "ins[bwld]"
  if (Name.startswith("ins") && Operands.size() == 1 &&
      (Name == "insb" || Name == "insw" || Name == "insl" || Name == "insd")) {
    AddDefaultSrcDestOperands(Operands,
                              X86Operand::CreateReg(X86::DX, NameLoc, NameLoc),
                              DefaultMemDIOperand(NameLoc));
  }

  // Append default arguments to "outs[bwld]"
  if (Name.startswith("outs") && Operands.size() == 1 &&
      (Name == "outsb" || Name == "outsw" || Name == "outsl" ||
       Name == "outsd" )) {
    AddDefaultSrcDestOperands(Operands,
                              DefaultMemSIOperand(NameLoc),
                              X86Operand::CreateReg(X86::DX, NameLoc, NameLoc));
  }

  // Transform "lods[bwlq]" into "lods[bwlq] ($SIREG)" for appropriate
  // values of $SIREG according to the mode. It would be nice if this
  // could be achieved with InstAlias in the tables.
  if (Name.startswith("lods") && Operands.size() == 1 &&
      (Name == "lods" || Name == "lodsb" || Name == "lodsw" ||
       Name == "lodsl" || Name == "lodsd" || Name == "lodsq"))
    Operands.push_back(DefaultMemSIOperand(NameLoc));

  // Transform "stos[bwlq]" into "stos[bwlq] ($DIREG)" for appropriate
  // values of $DIREG according to the mode. It would be nice if this
  // could be achieved with InstAlias in the tables.
  if (Name.startswith("stos") && Operands.size() == 1 &&
      (Name == "stos" || Name == "stosb" || Name == "stosw" ||
       Name == "stosl" || Name == "stosd" || Name == "stosq"))
    Operands.push_back(DefaultMemDIOperand(NameLoc));

  // Transform "scas[bwlq]" into "scas[bwlq] ($DIREG)" for appropriate
  // values of $DIREG according to the mode. It would be nice if this
  // could be achieved with InstAlias in the tables.
  if (Name.startswith("scas") && Operands.size() == 1 &&
      (Name == "scas" || Name == "scasb" || Name == "scasw" ||
       Name == "scasl" || Name == "scasd" || Name == "scasq"))
    Operands.push_back(DefaultMemDIOperand(NameLoc));

  // Add default SI and DI operands to "cmps[bwlq]".
  if (Name.startswith("cmps") &&
      (Name == "cmps" || Name == "cmpsb" || Name == "cmpsw" ||
       Name == "cmpsl" || Name == "cmpsd" || Name == "cmpsq")) {
    if (Operands.size() == 1) {
      AddDefaultSrcDestOperands(Operands,
                                DefaultMemDIOperand(NameLoc),
                                DefaultMemSIOperand(NameLoc));
    } else if (Operands.size() == 3) {
      X86Operand &Op = (X86Operand &)*Operands[1];
      X86Operand &Op2 = (X86Operand &)*Operands[2];
      if (!doSrcDstMatch(Op, Op2))
        return Error(Op.getStartLoc(),
                     "mismatching source and destination index registers");
    }
  }

  // Add default SI and DI operands to "movs[bwlq]".
  if ((Name.startswith("movs") &&
      (Name == "movs" || Name == "movsb" || Name == "movsw" ||
       Name == "movsl" || Name == "movsd" || Name == "movsq")) ||
      (Name.startswith("smov") &&
      (Name == "smov" || Name == "smovb" || Name == "smovw" ||
       Name == "smovl" || Name == "smovd" || Name == "smovq"))) {
    if (Operands.size() == 1) {
      if (Name == "movsd")
        Operands.back() = X86Operand::CreateToken("movsl", NameLoc);
      AddDefaultSrcDestOperands(Operands,
                                DefaultMemSIOperand(NameLoc),
                                DefaultMemDIOperand(NameLoc));
    } else if (Operands.size() == 3) {
      X86Operand &Op = (X86Operand &)*Operands[1];
      X86Operand &Op2 = (X86Operand &)*Operands[2];
      if (!doSrcDstMatch(Op, Op2))
        return Error(Op.getStartLoc(),
                     "mismatching source and destination index registers");
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
      X86Operand &Op1 = static_cast<X86Operand &>(*Operands[2]);
      if (Op1.isImm() && isa<MCConstantExpr>(Op1.getImm()) &&
          cast<MCConstantExpr>(Op1.getImm())->getValue() == 1)
        Operands.pop_back();
    } else {
      X86Operand &Op1 = static_cast<X86Operand &>(*Operands[1]);
      if (Op1.isImm() && isa<MCConstantExpr>(Op1.getImm()) &&
          cast<MCConstantExpr>(Op1.getImm())->getValue() == 1)
        Operands.erase(Operands.begin() + 1);
    }
  }

  // Transforms "int $3" into "int3" as a size optimization.  We can't write an
  // instalias with an immediate operand yet.
  if (Name == "int" && Operands.size() == 2) {
    X86Operand &Op1 = static_cast<X86Operand &>(*Operands[1]);
    if (Op1.isImm())
      if (auto *CE = dyn_cast<MCConstantExpr>(Op1.getImm()))
        if (CE->getValue() == 3) {
          Operands.erase(Operands.begin() + 1);
          static_cast<X86Operand &>(*Operands[0]).setTokenValue("int3");
        }
  }

  return false;
}

bool X86AsmParser::processInstruction(MCInst &Inst, const OperandVector &Ops) {
  switch (Inst.getOpcode()) {
  default: return false;
  case X86::VMOVZPQILo2PQIrr:
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
    case X86::VMOVZPQILo2PQIrr: NewOpc = X86::VMOVPQI2QIrr;   break;
    case X86::VMOVAPDrr:        NewOpc = X86::VMOVAPDrr_REV;  break;
    case X86::VMOVAPDYrr:       NewOpc = X86::VMOVAPDYrr_REV; break;
    case X86::VMOVAPSrr:        NewOpc = X86::VMOVAPSrr_REV;  break;
    case X86::VMOVAPSYrr:       NewOpc = X86::VMOVAPSYrr_REV; break;
    case X86::VMOVDQArr:        NewOpc = X86::VMOVDQArr_REV;  break;
    case X86::VMOVDQAYrr:       NewOpc = X86::VMOVDQAYrr_REV; break;
    case X86::VMOVDQUrr:        NewOpc = X86::VMOVDQUrr_REV;  break;
    case X86::VMOVDQUYrr:       NewOpc = X86::VMOVDQUYrr_REV; break;
    case X86::VMOVUPDrr:        NewOpc = X86::VMOVUPDrr_REV;  break;
    case X86::VMOVUPDYrr:       NewOpc = X86::VMOVUPDYrr_REV; break;
    case X86::VMOVUPSrr:        NewOpc = X86::VMOVUPSrr_REV;  break;
    case X86::VMOVUPSYrr:       NewOpc = X86::VMOVUPSYrr_REV; break;
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

static const char *getSubtargetFeatureName(uint64_t Val);

void X86AsmParser::EmitInstruction(MCInst &Inst, OperandVector &Operands,
                                   MCStreamer &Out) {
  Instrumentation->InstrumentAndEmitInstruction(Inst, Operands, getContext(),
                                                MII, Out);
}

bool X86AsmParser::MatchAndEmitInstruction(SMLoc IDLoc, unsigned &Opcode,
                                           OperandVector &Operands,
                                           MCStreamer &Out, uint64_t &ErrorInfo,
                                           bool MatchingInlineAsm) {
  if (isParsingIntelSyntax())
    return MatchAndEmitIntelInstruction(IDLoc, Opcode, Operands, Out, ErrorInfo,
                                        MatchingInlineAsm);
  return MatchAndEmitATTInstruction(IDLoc, Opcode, Operands, Out, ErrorInfo,
                                    MatchingInlineAsm);
}

void X86AsmParser::MatchFPUWaitAlias(SMLoc IDLoc, X86Operand &Op,
                                     OperandVector &Operands, MCStreamer &Out,
                                     bool MatchingInlineAsm) {
  // FIXME: This should be replaced with a real .td file alias mechanism.
  // Also, MatchInstructionImpl should actually *do* the EmitInstruction
  // call.
  const char *Repl = StringSwitch<const char *>(Op.getToken())
                         .Case("finit", "fninit")
                         .Case("fsave", "fnsave")
                         .Case("fstcw", "fnstcw")
                         .Case("fstcww", "fnstcw")
                         .Case("fstenv", "fnstenv")
                         .Case("fstsw", "fnstsw")
                         .Case("fstsww", "fnstsw")
                         .Case("fclex", "fnclex")
                         .Default(nullptr);
  if (Repl) {
    MCInst Inst;
    Inst.setOpcode(X86::WAIT);
    Inst.setLoc(IDLoc);
    if (!MatchingInlineAsm)
      EmitInstruction(Inst, Operands, Out);
    Operands[0] = X86Operand::CreateToken(Repl, IDLoc);
  }
}

bool X86AsmParser::ErrorMissingFeature(SMLoc IDLoc, uint64_t ErrorInfo,
                                       bool MatchingInlineAsm) {
  assert(ErrorInfo && "Unknown missing feature!");
  ArrayRef<SMRange> EmptyRanges = None;
  SmallString<126> Msg;
  raw_svector_ostream OS(Msg);
  OS << "instruction requires:";
  uint64_t Mask = 1;
  for (unsigned i = 0; i < (sizeof(ErrorInfo)*8-1); ++i) {
    if (ErrorInfo & Mask)
      OS << ' ' << getSubtargetFeatureName(ErrorInfo & Mask);
    Mask <<= 1;
  }
  return Error(IDLoc, OS.str(), EmptyRanges, MatchingInlineAsm);
}

bool X86AsmParser::MatchAndEmitATTInstruction(SMLoc IDLoc, unsigned &Opcode,
                                              OperandVector &Operands,
                                              MCStreamer &Out,
                                              uint64_t &ErrorInfo,
                                              bool MatchingInlineAsm) {
  assert(!Operands.empty() && "Unexpect empty operand list!");
  X86Operand &Op = static_cast<X86Operand &>(*Operands[0]);
  assert(Op.isToken() && "Leading operand should always be a mnemonic!");
  ArrayRef<SMRange> EmptyRanges = None;

  // First, handle aliases that expand to multiple instructions.
  MatchFPUWaitAlias(IDLoc, Op, Operands, Out, MatchingInlineAsm);

  bool WasOriginallyInvalidOperand = false;
  MCInst Inst;

  // First, try a direct match.
  switch (MatchInstructionImpl(Operands, Inst,
                               ErrorInfo, MatchingInlineAsm,
                               isParsingIntelSyntax())) {
  default: llvm_unreachable("Unexpected match result!");
  case Match_Success:
    // Some instructions need post-processing to, for example, tweak which
    // encoding is selected. Loop on it while changes happen so the
    // individual transformations can chain off each other.
    if (!MatchingInlineAsm)
      while (processInstruction(Inst, Operands))
        ;

    Inst.setLoc(IDLoc);
    if (!MatchingInlineAsm)
      EmitInstruction(Inst, Operands, Out);
    Opcode = Inst.getOpcode();
    return false;
  case Match_MissingFeature:
    return ErrorMissingFeature(IDLoc, ErrorInfo, MatchingInlineAsm);
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
  StringRef Base = Op.getToken();
  SmallString<16> Tmp;
  Tmp += Base;
  Tmp += ' ';
  Op.setTokenValue(Tmp);

  // If this instruction starts with an 'f', then it is a floating point stack
  // instruction.  These come in up to three forms for 32-bit, 64-bit, and
  // 80-bit floating point, which use the suffixes s,l,t respectively.
  //
  // Otherwise, we assume that this may be an integer instruction, which comes
  // in 8/16/32/64-bit forms using the b,w,l,q suffixes respectively.
  const char *Suffixes = Base[0] != 'f' ? "bwlq" : "slt\0";

  // Check for the various suffix matches.
  uint64_t ErrorInfoIgnore;
  uint64_t ErrorInfoMissingFeature = 0; // Init suppresses compiler warnings.
  unsigned Match[4];

  for (unsigned I = 0, E = array_lengthof(Match); I != E; ++I) {
    Tmp.back() = Suffixes[I];
    Match[I] = MatchInstructionImpl(Operands, Inst, ErrorInfoIgnore,
                                  MatchingInlineAsm, isParsingIntelSyntax());
    // If this returned as a missing feature failure, remember that.
    if (Match[I] == Match_MissingFeature)
      ErrorInfoMissingFeature = ErrorInfoIgnore;
  }

  // Restore the old token.
  Op.setTokenValue(Base);

  // If exactly one matched, then we treat that as a successful match (and the
  // instruction will already have been filled in correctly, since the failing
  // matches won't have modified it).
  unsigned NumSuccessfulMatches =
      std::count(std::begin(Match), std::end(Match), Match_Success);
  if (NumSuccessfulMatches == 1) {
    Inst.setLoc(IDLoc);
    if (!MatchingInlineAsm)
      EmitInstruction(Inst, Operands, Out);
    Opcode = Inst.getOpcode();
    return false;
  }

  // Otherwise, the match failed, try to produce a decent error message.

  // If we had multiple suffix matches, then identify this as an ambiguous
  // match.
  if (NumSuccessfulMatches > 1) {
    char MatchChars[4];
    unsigned NumMatches = 0;
    for (unsigned I = 0, E = array_lengthof(Match); I != E; ++I)
      if (Match[I] == Match_Success)
        MatchChars[NumMatches++] = Suffixes[I];

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
  if (std::count(std::begin(Match), std::end(Match), Match_MnemonicFail) == 4) {
    if (!WasOriginallyInvalidOperand) {
      ArrayRef<SMRange> Ranges =
          MatchingInlineAsm ? EmptyRanges : Op.getLocRange();
      return Error(IDLoc, "invalid instruction mnemonic '" + Base + "'",
                   Ranges, MatchingInlineAsm);
    }

    // Recover location info for the operand if we know which was the problem.
    if (ErrorInfo != ~0ULL) {
      if (ErrorInfo >= Operands.size())
        return Error(IDLoc, "too few operands for instruction",
                     EmptyRanges, MatchingInlineAsm);

      X86Operand &Operand = (X86Operand &)*Operands[ErrorInfo];
      if (Operand.getStartLoc().isValid()) {
        SMRange OperandRange = Operand.getLocRange();
        return Error(Operand.getStartLoc(), "invalid operand for instruction",
                     OperandRange, MatchingInlineAsm);
      }
    }

    return Error(IDLoc, "invalid operand for instruction", EmptyRanges,
                 MatchingInlineAsm);
  }

  // If one instruction matched with a missing feature, report this as a
  // missing feature.
  if (std::count(std::begin(Match), std::end(Match),
                 Match_MissingFeature) == 1) {
    ErrorInfo = ErrorInfoMissingFeature;
    return ErrorMissingFeature(IDLoc, ErrorInfoMissingFeature,
                               MatchingInlineAsm);
  }

  // If one instruction matched with an invalid operand, report this as an
  // operand failure.
  if (std::count(std::begin(Match), std::end(Match),
                 Match_InvalidOperand) == 1) {
    return Error(IDLoc, "invalid operand for instruction", EmptyRanges,
                 MatchingInlineAsm);
  }

  // If all of these were an outright failure, report it in a useless way.
  Error(IDLoc, "unknown use of instruction mnemonic without a size suffix",
        EmptyRanges, MatchingInlineAsm);
  return true;
}

bool X86AsmParser::MatchAndEmitIntelInstruction(SMLoc IDLoc, unsigned &Opcode,
                                                OperandVector &Operands,
                                                MCStreamer &Out,
                                                uint64_t &ErrorInfo,
                                                bool MatchingInlineAsm) {
  assert(!Operands.empty() && "Unexpect empty operand list!");
  X86Operand &Op = static_cast<X86Operand &>(*Operands[0]);
  assert(Op.isToken() && "Leading operand should always be a mnemonic!");
  StringRef Mnemonic = Op.getToken();
  ArrayRef<SMRange> EmptyRanges = None;

  // First, handle aliases that expand to multiple instructions.
  MatchFPUWaitAlias(IDLoc, Op, Operands, Out, MatchingInlineAsm);

  MCInst Inst;

  // Find one unsized memory operand, if present.
  X86Operand *UnsizedMemOp = nullptr;
  for (const auto &Op : Operands) {
    X86Operand *X86Op = static_cast<X86Operand *>(Op.get());
    if (X86Op->isMemUnsized())
      UnsizedMemOp = X86Op;
  }

  // Allow some instructions to have implicitly pointer-sized operands.  This is
  // compatible with gas.
  if (UnsizedMemOp) {
    static const char *const PtrSizedInstrs[] = {"call", "jmp", "push"};
    for (const char *Instr : PtrSizedInstrs) {
      if (Mnemonic == Instr) {
        UnsizedMemOp->Mem.Size = getPointerWidth();
        break;
      }
    }
  }

  // If an unsized memory operand is present, try to match with each memory
  // operand size.  In Intel assembly, the size is not part of the instruction
  // mnemonic.
  SmallVector<unsigned, 8> Match;
  uint64_t ErrorInfoMissingFeature = 0;
  if (UnsizedMemOp && UnsizedMemOp->isMemUnsized()) {
    static const unsigned MopSizes[] = {8, 16, 32, 64, 80, 128, 256, 512};
    for (unsigned Size : MopSizes) {
      UnsizedMemOp->Mem.Size = Size;
      uint64_t ErrorInfoIgnore;
      unsigned LastOpcode = Inst.getOpcode();
      unsigned M =
          MatchInstructionImpl(Operands, Inst, ErrorInfoIgnore,
                               MatchingInlineAsm, isParsingIntelSyntax());
      if (Match.empty() || LastOpcode != Inst.getOpcode())
        Match.push_back(M);

      // If this returned as a missing feature failure, remember that.
      if (Match.back() == Match_MissingFeature)
        ErrorInfoMissingFeature = ErrorInfoIgnore;
    }

    // Restore the size of the unsized memory operand if we modified it.
    if (UnsizedMemOp)
      UnsizedMemOp->Mem.Size = 0;
  }

  // If we haven't matched anything yet, this is not a basic integer or FPU
  // operation.  There shouldn't be any ambiguity in our mnemonic table, so try
  // matching with the unsized operand.
  if (Match.empty()) {
    Match.push_back(MatchInstructionImpl(Operands, Inst, ErrorInfo,
                                         MatchingInlineAsm,
                                         isParsingIntelSyntax()));
    // If this returned as a missing feature failure, remember that.
    if (Match.back() == Match_MissingFeature)
      ErrorInfoMissingFeature = ErrorInfo;
  }

  // Restore the size of the unsized memory operand if we modified it.
  if (UnsizedMemOp)
    UnsizedMemOp->Mem.Size = 0;

  // If it's a bad mnemonic, all results will be the same.
  if (Match.back() == Match_MnemonicFail) {
    ArrayRef<SMRange> Ranges =
        MatchingInlineAsm ? EmptyRanges : Op.getLocRange();
    return Error(IDLoc, "invalid instruction mnemonic '" + Mnemonic + "'",
                 Ranges, MatchingInlineAsm);
  }

  // If exactly one matched, then we treat that as a successful match (and the
  // instruction will already have been filled in correctly, since the failing
  // matches won't have modified it).
  unsigned NumSuccessfulMatches =
      std::count(std::begin(Match), std::end(Match), Match_Success);
  if (NumSuccessfulMatches == 1) {
    // Some instructions need post-processing to, for example, tweak which
    // encoding is selected. Loop on it while changes happen so the individual
    // transformations can chain off each other.
    if (!MatchingInlineAsm)
      while (processInstruction(Inst, Operands))
        ;
    Inst.setLoc(IDLoc);
    if (!MatchingInlineAsm)
      EmitInstruction(Inst, Operands, Out);
    Opcode = Inst.getOpcode();
    return false;
  } else if (NumSuccessfulMatches > 1) {
    assert(UnsizedMemOp &&
           "multiple matches only possible with unsized memory operands");
    ArrayRef<SMRange> Ranges =
        MatchingInlineAsm ? EmptyRanges : UnsizedMemOp->getLocRange();
    return Error(UnsizedMemOp->getStartLoc(),
                 "ambiguous operand size for instruction '" + Mnemonic + "\'",
                 Ranges, MatchingInlineAsm);
  }

  // If one instruction matched with a missing feature, report this as a
  // missing feature.
  if (std::count(std::begin(Match), std::end(Match),
                 Match_MissingFeature) == 1) {
    ErrorInfo = ErrorInfoMissingFeature;
    return ErrorMissingFeature(IDLoc, ErrorInfoMissingFeature,
                               MatchingInlineAsm);
  }

  // If one instruction matched with an invalid operand, report this as an
  // operand failure.
  if (std::count(std::begin(Match), std::end(Match),
                 Match_InvalidOperand) == 1) {
    return Error(IDLoc, "invalid operand for instruction", EmptyRanges,
                 MatchingInlineAsm);
  }

  // If all of these were an outright failure, report it in a useless way.
  return Error(IDLoc, "unknown instruction mnemonic", EmptyRanges,
               MatchingInlineAsm);
}

bool X86AsmParser::OmitRegisterFromClobberLists(unsigned RegNo) {
  return X86MCRegisterClasses[X86::SEGMENT_REGRegClassID].contains(RegNo);
}

bool X86AsmParser::ParseDirective(AsmToken DirectiveID) {
  MCAsmParser &Parser = getParser();
  StringRef IDVal = DirectiveID.getIdentifier();
  if (IDVal == ".word")
    return ParseDirectiveWord(2, DirectiveID.getLoc());
  else if (IDVal.startswith(".code"))
    return ParseDirectiveCode(IDVal, DirectiveID.getLoc());
  else if (IDVal.startswith(".att_syntax")) {
    if (getLexer().isNot(AsmToken::EndOfStatement)) {
      if (Parser.getTok().getString() == "prefix")
        Parser.Lex();
      else if (Parser.getTok().getString() == "noprefix")
        return Error(DirectiveID.getLoc(), "'.att_syntax noprefix' is not "
                                           "supported: registers must have a "
                                           "'%' prefix in .att_syntax");
    }
    getParser().setAssemblerDialect(0);
    return false;
  } else if (IDVal.startswith(".intel_syntax")) {
    getParser().setAssemblerDialect(1);
    if (getLexer().isNot(AsmToken::EndOfStatement)) {
      if (Parser.getTok().getString() == "noprefix")
        Parser.Lex();
      else if (Parser.getTok().getString() == "prefix")
        return Error(DirectiveID.getLoc(), "'.intel_syntax prefix' is not "
                                           "supported: registers must not have "
                                           "a '%' prefix in .intel_syntax");
    }
    return false;
  }
  return true;
}

/// ParseDirectiveWord
///  ::= .word [ expression (, expression)* ]
bool X86AsmParser::ParseDirectiveWord(unsigned Size, SMLoc L) {
  MCAsmParser &Parser = getParser();
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    for (;;) {
      const MCExpr *Value;
      SMLoc ExprLoc = getLexer().getLoc();
      if (getParser().parseExpression(Value))
        return false;

      if (const auto *MCE = dyn_cast<MCConstantExpr>(Value)) {
        assert(Size <= 8 && "Invalid size");
        uint64_t IntValue = MCE->getValue();
        if (!isUIntN(8 * Size, IntValue) && !isIntN(8 * Size, IntValue))
          return Error(ExprLoc, "literal value out of range for directive");
        getStreamer().EmitIntValue(IntValue, Size);
      } else {
        getStreamer().EmitValue(Value, Size, ExprLoc);
      }

      if (getLexer().is(AsmToken::EndOfStatement))
        break;

      // FIXME: Improve diagnostic.
      if (getLexer().isNot(AsmToken::Comma)) {
        Error(L, "unexpected token in directive");
        return false;
      }
      Parser.Lex();
    }
  }

  Parser.Lex();
  return false;
}

/// ParseDirectiveCode
///  ::= .code16 | .code32 | .code64
bool X86AsmParser::ParseDirectiveCode(StringRef IDVal, SMLoc L) {
  MCAsmParser &Parser = getParser();
  if (IDVal == ".code16") {
    Parser.Lex();
    if (!is16BitMode()) {
      SwitchMode(X86::Mode16Bit);
      getParser().getStreamer().EmitAssemblerFlag(MCAF_Code16);
    }
  } else if (IDVal == ".code32") {
    Parser.Lex();
    if (!is32BitMode()) {
      SwitchMode(X86::Mode32Bit);
      getParser().getStreamer().EmitAssemblerFlag(MCAF_Code32);
    }
  } else if (IDVal == ".code64") {
    Parser.Lex();
    if (!is64BitMode()) {
      SwitchMode(X86::Mode64Bit);
      getParser().getStreamer().EmitAssemblerFlag(MCAF_Code64);
    }
  } else {
    Error(L, "unknown directive " + IDVal);
    return false;
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
