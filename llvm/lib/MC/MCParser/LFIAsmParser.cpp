//===- LFIAsmParser.cpp - LFI Assembly Parser -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file was written by the LFI and Native Client authors.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCLFIRewriter.h"
#include "llvm/MC/MCParser/MCAsmParserExtension.h"
#include "llvm/MC/MCStreamer.h"

using namespace llvm;

class LFIAsmParser : public MCAsmParserExtension {
  MCLFIRewriter *Rewriter;
  template <bool (LFIAsmParser::*HandlerMethod)(StringRef, SMLoc)>
  void addDirectiveHandler(StringRef Directive) {
    MCAsmParser::ExtensionDirectiveHandler Handler =
        std::make_pair(this, HandleDirective<LFIAsmParser, HandlerMethod>);

    getParser().addDirectiveHandler(Directive, Handler);
  }

public:
  LFIAsmParser(MCLFIRewriter *Exp) : Rewriter(Exp) {}
  void Initialize(MCAsmParser &Parser) override {
    // Call the base implementation.
    MCAsmParserExtension::Initialize(Parser);
    addDirectiveHandler<&LFIAsmParser::parseRewriteDisable>(
        ".lfi_rewrite_disable");
    addDirectiveHandler<&LFIAsmParser::parseRewriteEnable>(
        ".lfi_rewrite_enable");
  }

  /// ::= {.lfi_rewrite_disable}
  bool parseRewriteDisable(StringRef Directive, SMLoc Loc) {
    getParser().checkForValidSection();
    if (getLexer().isNot(AsmToken::EndOfStatement))
      return TokError("unexpected token");
    Lex();

    Rewriter->disable();

    return false;
  }

  /// ::= {.lfi_rewrite_enable}
  bool parseRewriteEnable(StringRef Directive, SMLoc Loc) {
    getParser().checkForValidSection();
    if (getLexer().isNot(AsmToken::EndOfStatement))
      return TokError("unexpected token");
    Lex();

    Rewriter->enable();

    return false;
  }
};

namespace llvm {
MCAsmParserExtension *createLFIAsmParser(MCLFIRewriter *Exp) {
  return new LFIAsmParser(Exp);
}
} // namespace llvm
