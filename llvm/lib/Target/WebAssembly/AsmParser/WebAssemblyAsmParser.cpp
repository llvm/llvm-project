//==- WebAssemblyAsmParser.cpp - Assembler for WebAssembly -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file is part of the WebAssembly Assembler.
///
/// It contains code to translate a parsed .s file into MCInsts.
///
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "MCTargetDesc/WebAssemblyTargetStreamer.h"
#include "WebAssembly.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

#define DEBUG_TYPE "wasm-asm-parser"

namespace {

// We store register types as SimpleValueType to retain SIMD layout
// information, but must also be able to supply them as the (unnamed)
// register enum from WebAssemblyRegisterInfo.td/.inc.
static unsigned MVTToWasmReg(MVT::SimpleValueType Type) {
  switch(Type) {
    case MVT::i32: return WebAssembly::I32_0;
    case MVT::i64: return WebAssembly::I64_0;
    case MVT::f32: return WebAssembly::F32_0;
    case MVT::f64: return WebAssembly::F64_0;
    case MVT::v16i8: return WebAssembly::V128_0;
    case MVT::v8i16: return WebAssembly::V128_0;
    case MVT::v4i32: return WebAssembly::V128_0;
    case MVT::v4f32: return WebAssembly::V128_0;
    default: return MVT::INVALID_SIMPLE_VALUE_TYPE;
  }
}

/// WebAssemblyOperand - Instances of this class represent the operands in a
/// parsed WASM machine instruction.
struct WebAssemblyOperand : public MCParsedAsmOperand {
  enum KindTy { Token, Local, Stack, Integer, Float, Symbol } Kind;

  SMLoc StartLoc, EndLoc;

  struct TokOp {
    StringRef Tok;
  };

  struct RegOp {
    // This is a (virtual) local or stack register represented as 0..
    unsigned RegNo;
    // In most targets, the register number also encodes the type, but for
    // wasm we have to track that seperately since we have an unbounded
    // number of registers.
    // This has the unfortunate side effect that we supply a different value
    // to the table-gen matcher at different times in the process (when it
    // calls getReg() or addRegOperands().
    // TODO: While this works, it feels brittle. and would be nice to clean up.
    MVT::SimpleValueType Type;
  };

  struct IntOp {
    int64_t Val;
  };

  struct FltOp {
    double Val;
  };

  struct SymOp {
    const MCExpr *Exp;
  };

  union {
    struct TokOp Tok;
    struct RegOp Reg;
    struct IntOp Int;
    struct FltOp Flt;
    struct SymOp Sym;
  };

  WebAssemblyOperand(KindTy K, SMLoc Start, SMLoc End, TokOp T)
    : Kind(K), StartLoc(Start), EndLoc(End), Tok(T) {}
  WebAssemblyOperand(KindTy K, SMLoc Start, SMLoc End, RegOp R)
    : Kind(K), StartLoc(Start), EndLoc(End), Reg(R) {}
  WebAssemblyOperand(KindTy K, SMLoc Start, SMLoc End, IntOp I)
    : Kind(K), StartLoc(Start), EndLoc(End), Int(I) {}
  WebAssemblyOperand(KindTy K, SMLoc Start, SMLoc End, FltOp F)
    : Kind(K), StartLoc(Start), EndLoc(End), Flt(F) {}
  WebAssemblyOperand(KindTy K, SMLoc Start, SMLoc End, SymOp S)
    : Kind(K), StartLoc(Start), EndLoc(End), Sym(S) {}

  bool isToken() const override { return Kind == Token; }
  bool isImm() const override { return Kind == Integer ||
                                       Kind == Float ||
                                       Kind == Symbol; }
  bool isReg() const override { return Kind == Local || Kind == Stack; }
  bool isMem() const override { return false; }

  unsigned getReg() const override {
    assert(isReg());
    // This is called from the tablegen matcher (MatchInstructionImpl)
    // where it expects to match the type of register, see RegOp above.
    return MVTToWasmReg(Reg.Type);
  }

  StringRef getToken() const {
    assert(isToken());
    return Tok.Tok;
  }

  SMLoc getStartLoc() const override { return StartLoc; }
  SMLoc getEndLoc() const override { return EndLoc; }

  void addRegOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    assert(isReg() && "Not a register operand!");
    // This is called from the tablegen matcher (MatchInstructionImpl)
    // where it expects to output the actual register index, see RegOp above.
    unsigned R = Reg.RegNo;
    if (Kind == Stack) {
      // A stack register is represented as a large negative number.
      // See WebAssemblyRegNumbering::runOnMachineFunction and
      // getWARegStackId for why this | is needed.
      R |= INT32_MIN;
    }
    Inst.addOperand(MCOperand::createReg(R));
  }

  void addImmOperands(MCInst &Inst, unsigned N) const {
    assert(N == 1 && "Invalid number of operands!");
    if (Kind == Integer)
      Inst.addOperand(MCOperand::createImm(Int.Val));
    else if (Kind == Float)
      Inst.addOperand(MCOperand::createFPImm(Flt.Val));
    else if (Kind == Symbol)
      Inst.addOperand(MCOperand::createExpr(Sym.Exp));
    else
      llvm_unreachable("Should be immediate or symbol!");
  }

  void print(raw_ostream &OS) const override {
    switch (Kind) {
    case Token:
      OS << "Tok:" << Tok.Tok;
      break;
    case Local:
      OS << "Loc:" << Reg.RegNo << ":" << static_cast<int>(Reg.Type);
      break;
    case Stack:
      OS << "Stk:" << Reg.RegNo << ":" << static_cast<int>(Reg.Type);
      break;
    case Integer:
      OS << "Int:" << Int.Val;
      break;
    case Float:
      OS << "Flt:" << Flt.Val;
      break;
    case Symbol:
      OS << "Sym:" << Sym.Exp;
      break;
    }
  }
};

class WebAssemblyAsmParser final : public MCTargetAsmParser {
  MCAsmParser &Parser;
  MCAsmLexer &Lexer;
  // These are for the current function being parsed:
  // These are vectors since register assignments are so far non-sparse.
  // Replace by map if necessary.
  std::vector<MVT::SimpleValueType> LocalTypes;
  std::vector<MVT::SimpleValueType> StackTypes;
  MCSymbol *LastLabel;

public:
  WebAssemblyAsmParser(const MCSubtargetInfo &sti, MCAsmParser &Parser,
                       const MCInstrInfo &mii, const MCTargetOptions &Options)
      : MCTargetAsmParser(Options, sti, mii), Parser(Parser),
        Lexer(Parser.getLexer()), LastLabel(nullptr) {
  }

#define GET_ASSEMBLER_HEADER
#include "WebAssemblyGenAsmMatcher.inc"

  // TODO: This is required to be implemented, but appears unused.
  bool ParseRegister(unsigned &/*RegNo*/, SMLoc &/*StartLoc*/,
                     SMLoc &/*EndLoc*/) override {
    llvm_unreachable("ParseRegister is not implemented.");
  }

  bool Error(const StringRef &msg, const AsmToken &tok) {
    return Parser.Error(tok.getLoc(), msg + tok.getString());
  }

  bool IsNext(AsmToken::TokenKind Kind) {
    auto ok = Lexer.is(Kind);
    if (ok) Parser.Lex();
    return ok;
  }

  bool Expect(AsmToken::TokenKind Kind, const char *KindName) {
    if (!IsNext(Kind))
      return Error(std::string("Expected ") + KindName + ", instead got: ",
                   Lexer.getTok());
    return false;
  }

  MVT::SimpleValueType ParseRegType(const StringRef &RegType) {
    // Derive type from .param .local decls, or the instruction itself.
    return StringSwitch<MVT::SimpleValueType>(RegType)
        .Case("i32", MVT::i32)
        .Case("i64", MVT::i64)
        .Case("f32", MVT::f32)
        .Case("f64", MVT::f64)
        .Case("i8x16", MVT::v16i8)
        .Case("i16x8", MVT::v8i16)
        .Case("i32x4", MVT::v4i32)
        .Case("f32x4", MVT::v4f32)
        .Default(MVT::INVALID_SIMPLE_VALUE_TYPE);
  }

  MVT::SimpleValueType &GetType(
      std::vector<MVT::SimpleValueType> &Types, size_t i) {
    Types.resize(std::max(i + 1, Types.size()), MVT::INVALID_SIMPLE_VALUE_TYPE);
    return Types[i];
  }

  bool ParseReg(OperandVector &Operands, StringRef TypePrefix) {
    if (Lexer.is(AsmToken::Integer)) {
      auto &Local = Lexer.getTok();
      // This is a reference to a local, turn it into a virtual register.
      auto LocalNo = static_cast<unsigned>(Local.getIntVal());
      Operands.push_back(make_unique<WebAssemblyOperand>(
                           WebAssemblyOperand::Local, Local.getLoc(),
                           Local.getEndLoc(),
                           WebAssemblyOperand::RegOp{LocalNo,
                               GetType(LocalTypes, LocalNo)}));
      Parser.Lex();
    } else if (Lexer.is(AsmToken::Identifier)) {
      auto &StackRegTok = Lexer.getTok();
      // These are push/pop/drop pseudo stack registers, which we turn
      // into virtual registers also. The stackify pass will later turn them
      // back into implicit stack references if possible.
      auto StackReg = StackRegTok.getString();
      auto StackOp = StackReg.take_while([](char c) { return isalpha(c); });
      auto Reg = StackReg.drop_front(StackOp.size());
      unsigned long long ParsedRegNo = 0;
      if (!Reg.empty() && getAsUnsignedInteger(Reg, 10, ParsedRegNo))
        return Error("Cannot parse stack register index: ", StackRegTok);
      unsigned RegNo = static_cast<unsigned>(ParsedRegNo);
      if (StackOp == "push") {
        // This defines a result, record register type.
        auto RegType = ParseRegType(TypePrefix);
        GetType(StackTypes, RegNo) = RegType;
        Operands.push_back(make_unique<WebAssemblyOperand>(
                             WebAssemblyOperand::Stack,
                             StackRegTok.getLoc(),
                             StackRegTok.getEndLoc(),
                             WebAssemblyOperand::RegOp{RegNo, RegType}));
      } else if (StackOp == "pop") {
        // This uses a previously defined stack value.
        auto RegType = GetType(StackTypes, RegNo);
        Operands.push_back(make_unique<WebAssemblyOperand>(
                             WebAssemblyOperand::Stack,
                             StackRegTok.getLoc(),
                             StackRegTok.getEndLoc(),
                             WebAssemblyOperand::RegOp{RegNo, RegType}));
      } else if (StackOp == "drop") {
        // This operand will be dropped, since it is part of an instruction
        // whose result is void.
      } else {
        return Error("Unknown stack register prefix: ", StackRegTok);
      }
      Parser.Lex();
    } else {
      return Error(
            "Expected identifier/integer following $, instead got: ",
            Lexer.getTok());
    }
    IsNext(AsmToken::Equal);
    return false;
  }

  void ParseSingleInteger(bool IsNegative, OperandVector &Operands) {
    auto &Int = Lexer.getTok();
    int64_t Val = Int.getIntVal();
    if (IsNegative) Val = -Val;
    Operands.push_back(make_unique<WebAssemblyOperand>(
                         WebAssemblyOperand::Integer, Int.getLoc(),
                         Int.getEndLoc(), WebAssemblyOperand::IntOp{Val}));
    Parser.Lex();
  }

  bool ParseOperandStartingWithInteger(bool IsNegative,
                                       OperandVector &Operands,
                                       StringRef InstType) {
    ParseSingleInteger(IsNegative, Operands);
    if (Lexer.is(AsmToken::LParen)) {
      // Parse load/store operands of the form: offset($reg)align
      auto &LParen = Lexer.getTok();
      Operands.push_back(
            make_unique<WebAssemblyOperand>(WebAssemblyOperand::Token,
                                            LParen.getLoc(),
                                            LParen.getEndLoc(),
                                            WebAssemblyOperand::TokOp{
                                              LParen.getString()}));
      Parser.Lex();
      if (Expect(AsmToken::Dollar, "register")) return true;
      if (ParseReg(Operands, InstType)) return true;
      auto &RParen = Lexer.getTok();
      Operands.push_back(
            make_unique<WebAssemblyOperand>(WebAssemblyOperand::Token,
                                            RParen.getLoc(),
                                            RParen.getEndLoc(),
                                            WebAssemblyOperand::TokOp{
                                              RParen.getString()}));
      if (Expect(AsmToken::RParen, ")")) return true;
      if (Lexer.is(AsmToken::Integer)) {
        ParseSingleInteger(false, Operands);
      } else {
        // Alignment not specified.
        // FIXME: correctly derive a default from the instruction.
        Operands.push_back(make_unique<WebAssemblyOperand>(
                             WebAssemblyOperand::Integer, RParen.getLoc(),
                             RParen.getEndLoc(), WebAssemblyOperand::IntOp{0}));
      }
    }
    return false;
  }

  bool ParseInstruction(ParseInstructionInfo &/*Info*/, StringRef Name,
                        SMLoc NameLoc, OperandVector &Operands) override {
    Operands.push_back(
          make_unique<WebAssemblyOperand>(WebAssemblyOperand::Token, NameLoc,
                                          SMLoc::getFromPointer(
                                            NameLoc.getPointer() + Name.size()),
                                          WebAssemblyOperand::TokOp{
                                            StringRef(NameLoc.getPointer(),
                                                    Name.size())}));
    auto NamePair = Name.split('.');
    // If no '.', there is no type prefix.
    if (NamePair.second.empty()) std::swap(NamePair.first, NamePair.second);
    while (Lexer.isNot(AsmToken::EndOfStatement)) {
      auto &Tok = Lexer.getTok();
      switch (Tok.getKind()) {
      case AsmToken::Dollar: {
        Parser.Lex();
        if (ParseReg(Operands, NamePair.first)) return true;
        break;
      }
      case AsmToken::Identifier: {
        auto &Id = Lexer.getTok();
        const MCExpr *Val;
        SMLoc End;
        if (Parser.parsePrimaryExpr(Val, End))
          return Error("Cannot parse symbol: ", Lexer.getTok());
        Operands.push_back(make_unique<WebAssemblyOperand>(
                             WebAssemblyOperand::Symbol, Id.getLoc(),
                             Id.getEndLoc(), WebAssemblyOperand::SymOp{Val}));
        break;
      }
      case AsmToken::Minus:
        Parser.Lex();
        if (Lexer.isNot(AsmToken::Integer))
          return Error("Expected integer instead got: ", Lexer.getTok());
        if (ParseOperandStartingWithInteger(true, Operands, NamePair.first))
          return true;
        break;
      case AsmToken::Integer:
        if (ParseOperandStartingWithInteger(false, Operands, NamePair.first))
          return true;
        break;
      case AsmToken::Real: {
        double Val;
        if (Tok.getString().getAsDouble(Val, false))
          return Error("Cannot parse real: ", Tok);
        Operands.push_back(make_unique<WebAssemblyOperand>(
                             WebAssemblyOperand::Float, Tok.getLoc(),
                             Tok.getEndLoc(), WebAssemblyOperand::FltOp{Val}));
        Parser.Lex();
        break;
      }
      default:
        return Error("Unexpected token in operand: ", Tok);
      }
      if (Lexer.isNot(AsmToken::EndOfStatement)) {
        if (Expect(AsmToken::Comma, ",")) return true;
      }
    }
    Parser.Lex();
    // Call instructions are vararg, but the tablegen matcher doesn't seem to
    // support that, so for now we strip these extra operands.
    // This is problematic if these arguments are not simple $pop stack
    // registers, since e.g. a local register would get lost, so we check for
    // this. This can be the case when using -disable-wasm-explicit-locals
    // which currently s2wasm requires.
    // TODO: Instead, we can move this code to MatchAndEmitInstruction below and
    // actually generate get_local instructions on the fly.
    // Or even better, improve the matcher to support vararg?
    auto IsIndirect = NamePair.second == "call_indirect";
    if (IsIndirect || NamePair.second == "call") {
      // Figure out number of fixed operands from the instruction.
      size_t CallOperands = 1;  // The name token.
      if (!IsIndirect) CallOperands++;  // The function index.
      if (!NamePair.first.empty()) CallOperands++;  // The result register.
      if (Operands.size() > CallOperands) {
        // Ensure operands we drop are all $pop.
        for (size_t I = CallOperands; I < Operands.size(); I++) {
          auto Operand =
              reinterpret_cast<WebAssemblyOperand *>(Operands[I].get());
          if (Operand->Kind != WebAssemblyOperand::Stack)
            Parser.Error(NameLoc,
              "Call instruction has non-stack arguments, if this code was "
              "generated with -disable-wasm-explicit-locals please remove it");
        }
        // Drop unneeded operands.
        Operands.resize(CallOperands);
      }
    }
    // Block instructions require a signature index, but these are missing in
    // assembly, so we add a dummy one explicitly (since we have no control
    // over signature tables here, we assume these will be regenerated when
    // the wasm module is generated).
    if (NamePair.second == "block" || NamePair.second == "loop") {
      Operands.push_back(make_unique<WebAssemblyOperand>(
                           WebAssemblyOperand::Integer, NameLoc,
                           NameLoc, WebAssemblyOperand::IntOp{-1}));
    }
    // These don't specify the type, which has to derived from the local index.
    if (NamePair.second == "get_local" || NamePair.second == "tee_local") {
      if (Operands.size() >= 3 && Operands[1]->isReg() &&
          Operands[2]->isImm()) {
        auto Op1 = reinterpret_cast<WebAssemblyOperand *>(Operands[1].get());
        auto Op2 = reinterpret_cast<WebAssemblyOperand *>(Operands[2].get());
        auto Type = GetType(LocalTypes, static_cast<size_t>(Op2->Int.Val));
        Op1->Reg.Type = Type;
        GetType(StackTypes, Op1->Reg.RegNo) = Type;
      }
    }
    return false;
  }

  void onLabelParsed(MCSymbol *Symbol) override {
    LastLabel = Symbol;
  }

  bool ParseDirective(AsmToken DirectiveID) override {
    assert(DirectiveID.getKind() == AsmToken::Identifier);
    auto &Out = getStreamer();
    auto &TOut = reinterpret_cast<WebAssemblyTargetStreamer &>(
                   *Out.getTargetStreamer());
    // TODO: we're just parsing the subset of directives we're interested in,
    // and ignoring ones we don't recognise. We should ideally verify
    // all directives here.
    if (DirectiveID.getString() == ".type") {
      // This could be the start of a function, check if followed by
      // "label,@function"
      if (!(IsNext(AsmToken::Identifier) &&
            IsNext(AsmToken::Comma) &&
            IsNext(AsmToken::At) &&
            Lexer.is(AsmToken::Identifier)))
        return Error("Expected label,@type declaration, got: ", Lexer.getTok());
      if (Lexer.getTok().getString() == "function") {
        // Track locals from start of function.
        LocalTypes.clear();
        StackTypes.clear();
      }
      Parser.Lex();
      //Out.EmitSymbolAttribute(??, MCSA_ELF_TypeFunction);
    } else if (DirectiveID.getString() == ".param" ||
               DirectiveID.getString() == ".local") {
      // Track the number of locals, needed for correct virtual register
      // assignment elsewhere.
      // Also output a directive to the streamer.
      std::vector<MVT> Params;
      std::vector<MVT> Locals;
      while (Lexer.is(AsmToken::Identifier)) {
        auto RegType = ParseRegType(Lexer.getTok().getString());
        if (RegType == MVT::INVALID_SIMPLE_VALUE_TYPE) return true;
        LocalTypes.push_back(RegType);
        if (DirectiveID.getString() == ".param") {
          Params.push_back(RegType);
        } else {
          Locals.push_back(RegType);
        }
        Parser.Lex();
        if (!IsNext(AsmToken::Comma)) break;
      }
      assert(LastLabel);
      TOut.emitParam(LastLabel, Params);
      TOut.emitLocal(Locals);
    } else {
      // For now, ignore anydirective we don't recognize:
      while (Lexer.isNot(AsmToken::EndOfStatement)) Parser.Lex();
    }
    return Expect(AsmToken::EndOfStatement, "EOL");
  }

  bool MatchAndEmitInstruction(SMLoc IDLoc, unsigned &/*Opcode*/,
                               OperandVector &Operands,
                               MCStreamer &Out, uint64_t &ErrorInfo,
                               bool MatchingInlineAsm) override {
    MCInst Inst;
    unsigned MatchResult =
        MatchInstructionImpl(Operands, Inst, ErrorInfo, MatchingInlineAsm);
    switch (MatchResult) {
    case Match_Success: {
      Out.EmitInstruction(Inst, getSTI());
      return false;
    }
    case Match_MissingFeature:
      return Parser.Error(IDLoc,
          "instruction requires a WASM feature not currently enabled");
    case Match_MnemonicFail:
      return Parser.Error(IDLoc, "invalid instruction");
    case Match_NearMisses:
      return Parser.Error(IDLoc, "ambiguous instruction");
    case Match_InvalidTiedOperand:
    case Match_InvalidOperand: {
      SMLoc ErrorLoc = IDLoc;
      if (ErrorInfo != ~0ULL) {
        if (ErrorInfo >= Operands.size())
          return Parser.Error(IDLoc, "too few operands for instruction");
        ErrorLoc = Operands[ErrorInfo]->getStartLoc();
        if (ErrorLoc == SMLoc())
          ErrorLoc = IDLoc;
      }
      return Parser.Error(ErrorLoc, "invalid operand for instruction");
    }
    }
    llvm_unreachable("Implement any new match types added!");
  }
};
} // end anonymous namespace

// Force static initialization.
extern "C" void LLVMInitializeWebAssemblyAsmParser() {
  RegisterMCAsmParser<WebAssemblyAsmParser> X(getTheWebAssemblyTarget32());
  RegisterMCAsmParser<WebAssemblyAsmParser> Y(getTheWebAssemblyTarget64());
}

#define GET_REGISTER_MATCHER
#define GET_MATCHER_IMPLEMENTATION
#include "WebAssemblyGenAsmMatcher.inc"
