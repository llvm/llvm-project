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

/// WebAssemblyOperand - Instances of this class represent the operands in a
/// parsed WASM machine instruction.
struct WebAssemblyOperand : public MCParsedAsmOperand {
  enum KindTy { Token, Integer, Float, Symbol } Kind;

  SMLoc StartLoc, EndLoc;

  struct TokOp {
    StringRef Tok;
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
    struct IntOp Int;
    struct FltOp Flt;
    struct SymOp Sym;
  };

  WebAssemblyOperand(KindTy K, SMLoc Start, SMLoc End, TokOp T)
    : Kind(K), StartLoc(Start), EndLoc(End), Tok(T) {}
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
  bool isMem() const override { return false; }
  bool isReg() const override { return false; }

  unsigned getReg() const override {
    llvm_unreachable("Assembly inspects a register operand");
    return 0;
  }

  StringRef getToken() const {
    assert(isToken());
    return Tok.Tok;
  }

  SMLoc getStartLoc() const override { return StartLoc; }
  SMLoc getEndLoc() const override { return EndLoc; }

  void addRegOperands(MCInst &, unsigned) const {
    // Required by the assembly matcher.
    llvm_unreachable("Assembly matcher creates register operands");
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
  MCSymbol *LastLabel;

public:
  WebAssemblyAsmParser(const MCSubtargetInfo &sti, MCAsmParser &Parser,
                       const MCInstrInfo &mii, const MCTargetOptions &Options)
      : MCTargetAsmParser(Options, sti, mii), Parser(Parser),
        Lexer(Parser.getLexer()), LastLabel(nullptr) {
    setAvailableFeatures(ComputeAvailableFeatures(sti.getFeatureBits()));
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
        .Case("i64x2", MVT::v2i64)
        .Case("f32x4", MVT::v4f32)
        .Case("f64x2", MVT::v2f64)
        // arbitrarily chosen vector type to associate with "v128"
        // FIXME: should these be EVTs to avoid this arbitrary hack? Do we want
        // to accept more specific SIMD register types?
        .Case("v128", MVT::v16i8)
        .Default(MVT::INVALID_SIMPLE_VALUE_TYPE);
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
                                       StringRef InstName) {
    ParseSingleInteger(IsNegative, Operands);
    // FIXME: there is probably a cleaner way to do this.
    auto IsLoadStore = InstName.startswith("load") ||
                       InstName.startswith("store") ||
                       InstName.startswith("atomic_load") ||
                       InstName.startswith("atomic_store");
    if (IsLoadStore) {
      // Parse load/store operands of the form: offset align
      auto &Offset = Lexer.getTok();
      if (Offset.is(AsmToken::Integer)) {
        ParseSingleInteger(false, Operands);
      } else {
        // Alignment not specified.
        // FIXME: correctly derive a default from the instruction.
        // We can't just call WebAssembly::GetDefaultP2Align since we don't have
        // an opcode until after the assembly matcher.
        Operands.push_back(make_unique<WebAssemblyOperand>(
                             WebAssemblyOperand::Integer, Offset.getLoc(),
                             Offset.getEndLoc(), WebAssemblyOperand::IntOp{0}));
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
        if (ParseOperandStartingWithInteger(true, Operands, NamePair.second))
          return true;
        break;
      case AsmToken::Integer:
        if (ParseOperandStartingWithInteger(false, Operands, NamePair.second))
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
    // Block instructions require a signature index, but these are missing in
    // assembly, so we add a dummy one explicitly (since we have no control
    // over signature tables here, we assume these will be regenerated when
    // the wasm module is generated).
    if (NamePair.second == "block" || NamePair.second == "loop") {
      Operands.push_back(make_unique<WebAssemblyOperand>(
                           WebAssemblyOperand::Integer, NameLoc,
                           NameLoc, WebAssemblyOperand::IntOp{-1}));
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
