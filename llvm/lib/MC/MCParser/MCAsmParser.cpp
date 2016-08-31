//===-- MCAsmParser.cpp - Abstract Asm Parser Interface -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

MCAsmParser::MCAsmParser() : TargetParser(nullptr), ShowParsedOperands(0) {
}

MCAsmParser::~MCAsmParser() {
}

void MCAsmParser::setTargetParser(MCTargetAsmParser &P) {
  assert(!TargetParser && "Target parser is already initialized!");
  TargetParser = &P;
  TargetParser->Initialize(*this);
}

const AsmToken &MCAsmParser::getTok() const {
  return getLexer().getTok();
}

bool MCAsmParser::parseTokenLoc(SMLoc &Loc) {
  Loc = getTok().getLoc();
  return false;
}

bool MCAsmParser::parseEOL(const Twine &Msg) {
  if (getTok().getKind() == AsmToken::Hash) {
    StringRef CommentStr = parseStringToEndOfStatement();
    getLexer().Lex();
    getLexer().UnLex(AsmToken(AsmToken::EndOfStatement, CommentStr));
  }
  if (getTok().getKind() != AsmToken::EndOfStatement)
    return Error(getTok().getLoc(), Msg);
  Lex();
  return false;
}

bool MCAsmParser::parseToken(AsmToken::TokenKind T, const Twine &Msg) {
  if (T == AsmToken::EndOfStatement)
    return parseEOL(Msg);
  if (getTok().getKind() != T)
    return Error(getTok().getLoc(), Msg);
  Lex();
  return false;
}

bool MCAsmParser::parseIntToken(int64_t &V, const Twine &Msg) {
  if (getTok().getKind() != AsmToken::Integer)
    return TokError(Msg);
  V = getTok().getIntVal();
  Lex();
  return false;
}

bool MCAsmParser::parseOptionalToken(AsmToken::TokenKind T, bool &Present) {
  Present = (getTok().getKind() == T);
  if (Present)
    Lex();
  return false;
}

bool MCAsmParser::check(bool P, const Twine &Msg) {
  return check(P, getTok().getLoc(), Msg);
}

bool MCAsmParser::check(bool P, SMLoc Loc, const Twine &Msg) {
  if (P)
    return Error(Loc, Msg);
  return false;
}

bool MCAsmParser::TokError(const Twine &Msg, ArrayRef<SMRange> Ranges) {
  Error(getLexer().getLoc(), Msg, Ranges);
  return true;
}

bool MCAsmParser::parseExpression(const MCExpr *&Res) {
  SMLoc L;
  return parseExpression(Res, L);
}

LLVM_DUMP_METHOD void MCParsedAsmOperand::dump() const {
  dbgs() << "  " << *this;
}
