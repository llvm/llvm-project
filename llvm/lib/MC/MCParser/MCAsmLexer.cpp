//===- MCAsmLexer.cpp - Abstract Asm Lexer Interface ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

MCAsmLexer::MCAsmLexer() {
  CurTok.emplace_back(AsmToken::Space, StringRef());
}

MCAsmLexer::~MCAsmLexer() = default;

SMLoc MCAsmLexer::getLoc() const {
  return SMLoc::getFromPointer(TokStart);
}

SMLoc AsmToken::getLoc() const {
  return SMLoc::getFromPointer(Str.data());
}

SMLoc AsmToken::getEndLoc() const {
  return SMLoc::getFromPointer(Str.data() + Str.size());
}

SMRange AsmToken::getLocRange() const {
  return SMRange(getLoc(), getEndLoc());
}

void AsmToken::dump(raw_ostream &OS) const {
  switch (Kind) {
  case AsmToken::Error:
    OS << "error";
    break;
  case AsmToken::Identifier:
    OS << "identifier: " << getString();
    break;
  case AsmToken::Integer:
    OS << "int: " << getString();
    break;
  case AsmToken::Real:
    OS << "real: " << getString();
    break;
  case AsmToken::String:
    OS << "string: " << getString();
    break;

  case AsmToken::Amp:                OS << "Amp"; break;
  case AsmToken::AmpAmp:             OS << "AmpAmp"; break;
  case AsmToken::At:                 OS << "At"; break;
  case AsmToken::BackSlash:          OS << "BackSlash"; break;
  case AsmToken::BigNum:             OS << "BigNum"; break;
  case AsmToken::Caret:              OS << "Caret"; break;
  case AsmToken::Colon:              OS << "Colon"; break;
  case AsmToken::Comma:              OS << "Comma"; break;
  case AsmToken::Comment:            OS << "Comment"; break;
  case AsmToken::Dollar:             OS << "Dollar"; break;
  case AsmToken::Dot:                OS << "Dot"; break;
  case AsmToken::EndOfStatement:     OS << "EndOfStatement"; break;
  case AsmToken::Eof:                OS << "Eof"; break;
  case AsmToken::Equal:              OS << "Equal"; break;
  case AsmToken::EqualEqual:         OS << "EqualEqual"; break;
  case AsmToken::Exclaim:            OS << "Exclaim"; break;
  case AsmToken::ExclaimEqual:       OS << "ExclaimEqual"; break;
  case AsmToken::Greater:            OS << "Greater"; break;
  case AsmToken::GreaterEqual:       OS << "GreaterEqual"; break;
  case AsmToken::GreaterGreater:     OS << "GreaterGreater"; break;
  case AsmToken::Hash:               OS << "Hash"; break;
  case AsmToken::HashDirective:      OS << "HashDirective"; break;
  case AsmToken::LBrac:              OS << "LBrac"; break;
  case AsmToken::LCurly:             OS << "LCurly"; break;
  case AsmToken::LParen:             OS << "LParen"; break;
  case AsmToken::Less:               OS << "Less"; break;
  case AsmToken::LessEqual:          OS << "LessEqual"; break;
  case AsmToken::LessGreater:        OS << "LessGreater"; break;
  case AsmToken::LessLess:           OS << "LessLess"; break;
  case AsmToken::Minus:              OS << "Minus"; break;
  case AsmToken::MinusGreater:       OS << "MinusGreater"; break;
  case AsmToken::Percent:            OS << "Percent"; break;
  case AsmToken::Pipe:               OS << "Pipe"; break;
  case AsmToken::PipePipe:           OS << "PipePipe"; break;
  case AsmToken::Plus:               OS << "Plus"; break;
  case AsmToken::Question:           OS << "Question"; break;
  case AsmToken::RBrac:              OS << "RBrac"; break;
  case AsmToken::RCurly:             OS << "RCurly"; break;
  case AsmToken::RParen:             OS << "RParen"; break;
  case AsmToken::Slash:              OS << "Slash"; break;
  case AsmToken::Space:              OS << "Space"; break;
  case AsmToken::Star:               OS << "Star"; break;
  case AsmToken::Tilde:              OS << "Tilde"; break;
  }

  // Print the token string.
  OS << " (\"";
  OS.write_escaped(getString());
  OS << "\")";
}
