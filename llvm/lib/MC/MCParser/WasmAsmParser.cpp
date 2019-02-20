//===- WasmAsmParser.cpp - Wasm Assembly Parser -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// --
//
// Note, this is for wasm, the binary format (analogous to ELF), not wasm,
// the instruction set (analogous to x86), for which parsing code lives in
// WebAssemblyAsmParser.
//
// This file contains processing for generic directives implemented using
// MCTargetStreamer, the ones that depend on WebAssemblyTargetStreamer are in
// WebAssemblyAsmParser.
//
//===----------------------------------------------------------------------===//

#include "llvm/BinaryFormat/Wasm.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCAsmParserExtension.h"
#include "llvm/MC/MCSectionWasm.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCSymbolWasm.h"
#include "llvm/Support/MachineValueType.h"

using namespace llvm;

namespace {

class WasmAsmParser : public MCAsmParserExtension {
  MCAsmParser *Parser = nullptr;
  MCAsmLexer *Lexer = nullptr;

  template<bool (WasmAsmParser::*HandlerMethod)(StringRef, SMLoc)>
  void addDirectiveHandler(StringRef Directive) {
    MCAsmParser::ExtensionDirectiveHandler Handler = std::make_pair(
        this, HandleDirective<WasmAsmParser, HandlerMethod>);

    getParser().addDirectiveHandler(Directive, Handler);
  }

public:
  WasmAsmParser() { BracketExpressionsSupported = true; }

  void Initialize(MCAsmParser &P) override {
    Parser = &P;
    Lexer = &Parser->getLexer();
    // Call the base implementation.
    this->MCAsmParserExtension::Initialize(*Parser);

    addDirectiveHandler<&WasmAsmParser::parseSectionDirectiveText>(".text");
    addDirectiveHandler<&WasmAsmParser::parseSectionDirective>(".section");
    addDirectiveHandler<&WasmAsmParser::parseDirectiveSize>(".size");
    addDirectiveHandler<&WasmAsmParser::parseDirectiveType>(".type");
  }

  bool error(const StringRef &Msg, const AsmToken &Tok) {
    return Parser->Error(Tok.getLoc(), Msg + Tok.getString());
  }

  bool isNext(AsmToken::TokenKind Kind) {
    auto Ok = Lexer->is(Kind);
    if (Ok)
      Lex();
    return Ok;
  }

  bool expect(AsmToken::TokenKind Kind, const char *KindName) {
    if (!isNext(Kind))
      return error(std::string("Expected ") + KindName + ", instead got: ",
                   Lexer->getTok());
    return false;
  }

  bool parseSectionDirectiveText(StringRef, SMLoc) {
    // FIXME: .text currently no-op.
    return false;
  }

  bool parseSectionFlags(StringRef FlagStr, bool &Passive) {
    SmallVector<StringRef, 2> Flags;
    // If there are no flags, keep Flags empty
    FlagStr.split(Flags, ",", -1, false);
    for (auto &Flag : Flags) {
      if (Flag == "passive")
        Passive = true;
      else
        return error("Expected section flags, instead got: ", Lexer->getTok());
    }
    return false;
  }

  bool parseSectionDirective(StringRef, SMLoc) {
    StringRef Name;
    if (Parser->parseIdentifier(Name))
      return TokError("expected identifier in directive");

    if (expect(AsmToken::Comma, ","))
      return true;

    if (Lexer->isNot(AsmToken::String))
      return error("expected string in directive, instead got: ", Lexer->getTok());

    SectionKind Kind = StringSwitch<SectionKind>(Name)
                       .StartsWith(".data", SectionKind::getData())
                       .StartsWith(".rodata", SectionKind::getReadOnly())
                       .StartsWith(".text", SectionKind::getText())
                       .StartsWith(".custom_section", SectionKind::getMetadata());

    MCSectionWasm* Section = getContext().getWasmSection(Name, Kind);

    // Update section flags if present in this .section directive
    bool Passive = false;
    if (parseSectionFlags(getTok().getStringContents(), Passive))
      return true;

    if (Passive) {
      if (!Section->isWasmData())
        return Parser->Error(getTok().getLoc(),
                             "Only data sections can be passive");
      Section->setPassive();
    }

    Lex();

    if (expect(AsmToken::Comma, ",") || expect(AsmToken::At, "@") ||
        expect(AsmToken::EndOfStatement, "eol"))
      return true;
    // This is done automatically by the assembler for text sections currently,
    // so we don't need to emit that here. This is what it would do (and may
    // be needed later for other section types):
    // auto WS = getContext().getWasmSection(Name, SectionKind::getText());
    // getStreamer().SwitchSection(WS);
    return false;
  }

  // TODO: This function is almost the same as ELFAsmParser::ParseDirectiveSize
  // so maybe could be shared somehow.
  bool parseDirectiveSize(StringRef, SMLoc) {
    StringRef Name;
    if (Parser->parseIdentifier(Name))
      return TokError("expected identifier in directive");
    auto Sym = getContext().getOrCreateSymbol(Name);
    if (expect(AsmToken::Comma, ","))
      return true;
    const MCExpr *Expr;
    if (Parser->parseExpression(Expr))
      return true;
    if (expect(AsmToken::EndOfStatement, "eol"))
      return true;
    // This is done automatically by the assembler for functions currently,
    // so we don't need to emit that here. This is what it would do:
    (void)Sym;
    // getStreamer().emitELFSize(Sym, Expr);
    return false;
  }

  bool parseDirectiveType(StringRef, SMLoc) {
    // This could be the start of a function, check if followed by
    // "label,@function"
    if (!Lexer->is(AsmToken::Identifier))
      return error("Expected label after .type directive, got: ",
                   Lexer->getTok());
    auto WasmSym = cast<MCSymbolWasm>(
                     getStreamer().getContext().getOrCreateSymbol(
                       Lexer->getTok().getString()));
    Lex();
    if (!(isNext(AsmToken::Comma) && isNext(AsmToken::At) &&
          Lexer->is(AsmToken::Identifier)))
      return error("Expected label,@type declaration, got: ", Lexer->getTok());
    auto TypeName = Lexer->getTok().getString();
    if (TypeName == "function")
      WasmSym->setType(wasm::WASM_SYMBOL_TYPE_FUNCTION);
    else if (TypeName == "global")
      WasmSym->setType(wasm::WASM_SYMBOL_TYPE_GLOBAL);
    else
      return error("Unknown WASM symbol type: ", Lexer->getTok());
    Lex();
    return expect(AsmToken::EndOfStatement, "EOL");
  }
};

} // end anonymous namespace

namespace llvm {

MCAsmParserExtension *createWasmAsmParser() {
  return new WasmAsmParser;
}

} // end namespace llvm
