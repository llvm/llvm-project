//===- COFFMasmParser.cpp - COFF MASM Assembly Parser ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParserExtension.h"
#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbolCOFF.h"
#include "llvm/MC/SectionKind.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/SMLoc.h"
#include <cstdint>
#include <utility>

using namespace llvm;

namespace {

class COFFMasmParser : public MCAsmParserExtension {
  template <bool (COFFMasmParser::*HandlerMethod)(StringRef, SMLoc)>
  void addDirectiveHandler(StringRef Directive) {
    MCAsmParser::ExtensionDirectiveHandler Handler =
        std::make_pair(this, HandleDirective<COFFMasmParser, HandlerMethod>);
    getParser().addDirectiveHandler(Directive, Handler);
  }

  bool ParseSectionSwitch(StringRef Section, unsigned Characteristics,
                          SectionKind Kind);

  bool ParseSectionSwitch(StringRef Section, unsigned Characteristics,
                          SectionKind Kind, StringRef COMDATSymName,
                          COFF::COMDATType Type);

  bool ParseDirectiveProc(StringRef, SMLoc);
  bool ParseDirectiveEndProc(StringRef, SMLoc);
  bool ParseDirectiveSegment(StringRef, SMLoc);
  bool ParseDirectiveSegmentEnd(StringRef, SMLoc);
  bool ParseDirectiveIncludelib(StringRef, SMLoc);

  bool ParseDirectiveAlias(StringRef, SMLoc);

  bool ParseSEHDirectiveAllocStack(StringRef, SMLoc);
  bool ParseSEHDirectiveEndProlog(StringRef, SMLoc);

  bool IgnoreDirective(StringRef, SMLoc) {
    while (!getLexer().is(AsmToken::EndOfStatement)) {
      Lex();
    }
    return false;
  }

  void Initialize(MCAsmParser &Parser) override {
    // Call the base implementation.
    MCAsmParserExtension::Initialize(Parser);

    // x64 directives
    addDirectiveHandler<&COFFMasmParser::ParseSEHDirectiveAllocStack>(
        ".allocstack");
    addDirectiveHandler<&COFFMasmParser::ParseSEHDirectiveEndProlog>(
        ".endprolog");

    // Code label directives
    // label
    // org

    // Conditional control flow directives
    // .break
    // .continue
    // .else
    // .elseif
    // .endif
    // .endw
    // .if
    // .repeat
    // .until
    // .untilcxz
    // .while

    // Data allocation directives
    // align
    // even
    // mmword
    // tbyte
    // xmmword
    // ymmword

    // Listing control directives
    addDirectiveHandler<&COFFMasmParser::IgnoreDirective>(".cref");
    addDirectiveHandler<&COFFMasmParser::IgnoreDirective>(".list");
    addDirectiveHandler<&COFFMasmParser::IgnoreDirective>(".listall");
    addDirectiveHandler<&COFFMasmParser::IgnoreDirective>(".listif");
    addDirectiveHandler<&COFFMasmParser::IgnoreDirective>(".listmacro");
    addDirectiveHandler<&COFFMasmParser::IgnoreDirective>(".listmacroall");
    addDirectiveHandler<&COFFMasmParser::IgnoreDirective>(".nocref");
    addDirectiveHandler<&COFFMasmParser::IgnoreDirective>(".nolist");
    addDirectiveHandler<&COFFMasmParser::IgnoreDirective>(".nolistif");
    addDirectiveHandler<&COFFMasmParser::IgnoreDirective>(".nolistmacro");
    addDirectiveHandler<&COFFMasmParser::IgnoreDirective>("page");
    addDirectiveHandler<&COFFMasmParser::IgnoreDirective>("subtitle");
    addDirectiveHandler<&COFFMasmParser::IgnoreDirective>(".tfcond");
    addDirectiveHandler<&COFFMasmParser::IgnoreDirective>("title");

    // Macro directives
    // goto

    // Miscellaneous directives
    addDirectiveHandler<&COFFMasmParser::ParseDirectiveAlias>("alias");
    // assume
    // .fpo
    addDirectiveHandler<&COFFMasmParser::ParseDirectiveIncludelib>(
        "includelib");
    // option
    // popcontext
    // pushcontext
    // .safeseh

    // Procedure directives
    addDirectiveHandler<&COFFMasmParser::ParseDirectiveEndProc>("endp");
    // invoke (32-bit only)
    addDirectiveHandler<&COFFMasmParser::ParseDirectiveProc>("proc");
    // proto

    // Processor directives; all ignored
    addDirectiveHandler<&COFFMasmParser::IgnoreDirective>(".386");
    addDirectiveHandler<&COFFMasmParser::IgnoreDirective>(".386p");
    addDirectiveHandler<&COFFMasmParser::IgnoreDirective>(".387");
    addDirectiveHandler<&COFFMasmParser::IgnoreDirective>(".486");
    addDirectiveHandler<&COFFMasmParser::IgnoreDirective>(".486p");
    addDirectiveHandler<&COFFMasmParser::IgnoreDirective>(".586");
    addDirectiveHandler<&COFFMasmParser::IgnoreDirective>(".586p");
    addDirectiveHandler<&COFFMasmParser::IgnoreDirective>(".686");
    addDirectiveHandler<&COFFMasmParser::IgnoreDirective>(".686p");
    addDirectiveHandler<&COFFMasmParser::IgnoreDirective>(".k3d");
    addDirectiveHandler<&COFFMasmParser::IgnoreDirective>(".mmx");
    addDirectiveHandler<&COFFMasmParser::IgnoreDirective>(".xmm");

    // Scope directives
    // comm
    // externdef

    // Segment directives
    // .alpha (32-bit only, order segments alphabetically)
    // .dosseg (32-bit only, order segments in DOS convention)
    // .seq (32-bit only, order segments sequentially)
    addDirectiveHandler<&COFFMasmParser::ParseDirectiveSegmentEnd>("ends");
    // group (32-bit only)
    addDirectiveHandler<&COFFMasmParser::ParseDirectiveSegment>("segment");

    // Simplified segment directives
    addDirectiveHandler<&COFFMasmParser::ParseSectionDirectiveCode>(".code");
    // .const
    addDirectiveHandler<
        &COFFMasmParser::ParseSectionDirectiveInitializedData>(".data");
    addDirectiveHandler<
        &COFFMasmParser::ParseSectionDirectiveUninitializedData>(".data?");
    // .exit
    // .fardata
    // .fardata?
    addDirectiveHandler<&COFFMasmParser::IgnoreDirective>(".model");
    // .stack
    // .startup

    // String directives, written <name> <directive> <params>
    // catstr (equivalent to <name> TEXTEQU <params>)
    // instr (equivalent to <name> = @InStr(<params>))
    // sizestr (equivalent to <name> = @SizeStr(<params>))
    // substr (equivalent to <name> TEXTEQU @SubStr(<params>))

    // Structure and record directives
    // record
    // typedef
  }

  bool ParseSectionDirectiveCode(StringRef, SMLoc) {
    return ParseSectionSwitch(".text",
                              COFF::IMAGE_SCN_CNT_CODE
                            | COFF::IMAGE_SCN_MEM_EXECUTE
                            | COFF::IMAGE_SCN_MEM_READ,
                              SectionKind::getText());
  }

  bool ParseSectionDirectiveInitializedData(StringRef, SMLoc) {
    return ParseSectionSwitch(".data",
                              COFF::IMAGE_SCN_CNT_INITIALIZED_DATA
                            | COFF::IMAGE_SCN_MEM_READ
                            | COFF::IMAGE_SCN_MEM_WRITE,
                              SectionKind::getData());
  }

  bool ParseSectionDirectiveUninitializedData(StringRef, SMLoc) {
    return ParseSectionSwitch(".bss",
                              COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA
                            | COFF::IMAGE_SCN_MEM_READ
                            | COFF::IMAGE_SCN_MEM_WRITE,
                              SectionKind::getBSS());
  }

  StringRef CurrentProcedure;
  bool CurrentProcedureFramed;

public:
  COFFMasmParser() = default;
};

} // end anonymous namespace.

static SectionKind computeSectionKind(unsigned Flags) {
  if (Flags & COFF::IMAGE_SCN_MEM_EXECUTE)
    return SectionKind::getText();
  if (Flags & COFF::IMAGE_SCN_MEM_READ &&
      (Flags & COFF::IMAGE_SCN_MEM_WRITE) == 0)
    return SectionKind::getReadOnly();
  return SectionKind::getData();
}

bool COFFMasmParser::ParseSectionSwitch(StringRef Section,
                                        unsigned Characteristics,
                                        SectionKind Kind) {
  return ParseSectionSwitch(Section, Characteristics, Kind, "",
                            (COFF::COMDATType)0);
}

bool COFFMasmParser::ParseSectionSwitch(StringRef Section,
                                        unsigned Characteristics,
                                        SectionKind Kind,
                                        StringRef COMDATSymName,
                                        COFF::COMDATType Type) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in section switching directive");
  Lex();

  getStreamer().SwitchSection(getContext().getCOFFSection(
      Section, Characteristics, Kind, COMDATSymName, Type));

  return false;
}

bool COFFMasmParser::ParseDirectiveSegment(StringRef Directive, SMLoc Loc) {
  StringRef SegmentName;
  if (!getLexer().is(AsmToken::Identifier))
    return TokError("expected identifier in directive");
  SegmentName = getTok().getIdentifier();
  Lex();

  StringRef SectionName = SegmentName;
  SmallVector<char, 247> SectionNameVector;
  unsigned Flags = COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                   COFF::IMAGE_SCN_MEM_READ | COFF::IMAGE_SCN_MEM_WRITE;
  if (SegmentName == "_TEXT" || SegmentName.startswith("_TEXT$")) {
    if (SegmentName.size() == 5) {
      SectionName = ".text";
    } else {
      SectionName =
          (".text$" + SegmentName.substr(6)).toStringRef(SectionNameVector);
    }
    Flags = COFF::IMAGE_SCN_CNT_CODE | COFF::IMAGE_SCN_MEM_EXECUTE |
            COFF::IMAGE_SCN_MEM_READ;
  }
  SectionKind Kind = computeSectionKind(Flags);
  getStreamer().SwitchSection(getContext().getCOFFSection(
      SectionName, Flags, Kind, "", (COFF::COMDATType)(0)));
  return false;
}

/// ParseDirectiveSegmentEnd
///  ::= identifier "ends"
bool COFFMasmParser::ParseDirectiveSegmentEnd(StringRef Directive, SMLoc Loc) {
  StringRef SegmentName;
  if (!getLexer().is(AsmToken::Identifier))
    return TokError("expected identifier in directive");
  SegmentName = getTok().getIdentifier();

  // Ignore; no action necessary.
  Lex();
  return false;
}

/// ParseDirectiveIncludelib
///  ::= "includelib" identifier
bool COFFMasmParser::ParseDirectiveIncludelib(StringRef Directive, SMLoc Loc) {
  StringRef Lib;
  if (getParser().parseIdentifier(Lib))
    return TokError("expected identifier in includelib directive");

  unsigned Flags = COFF::IMAGE_SCN_MEM_PRELOAD | COFF::IMAGE_SCN_MEM_16BIT;
  SectionKind Kind = computeSectionKind(Flags);
  getStreamer().PushSection();
  getStreamer().SwitchSection(getContext().getCOFFSection(
      ".drectve", Flags, Kind, "", (COFF::COMDATType)(0)));
  getStreamer().emitBytes("/DEFAULTLIB:");
  getStreamer().emitBytes(Lib);
  getStreamer().emitBytes(" ");
  getStreamer().PopSection();
  return false;
}

/// ParseDirectiveProc
/// TODO(epastor): Implement parameters and other attributes.
///  ::= label "proc" [[distance]]
///          statements
///      label "endproc"
bool COFFMasmParser::ParseDirectiveProc(StringRef Directive, SMLoc Loc) {
  StringRef Label;
  if (getParser().parseIdentifier(Label))
    return Error(Loc, "expected identifier for procedure");
  if (getLexer().is(AsmToken::Identifier)) {
    StringRef nextVal = getTok().getString();
    SMLoc nextLoc = getTok().getLoc();
    if (nextVal.equals_insensitive("far")) {
      // TODO(epastor): Handle far procedure definitions.
      Lex();
      return Error(nextLoc, "far procedure definitions not yet supported");
    } else if (nextVal.equals_insensitive("near")) {
      Lex();
      nextVal = getTok().getString();
      nextLoc = getTok().getLoc();
    }
  }
  MCSymbolCOFF *Sym = cast<MCSymbolCOFF>(getContext().getOrCreateSymbol(Label));

  // Define symbol as simple external function
  Sym->setExternal(true);
  Sym->setType(COFF::IMAGE_SYM_DTYPE_FUNCTION << COFF::SCT_COMPLEX_TYPE_SHIFT);

  bool Framed = false;
  if (getLexer().is(AsmToken::Identifier) &&
      getTok().getString().equals_insensitive("frame")) {
    Lex();
    Framed = true;
    getStreamer().EmitWinCFIStartProc(Sym, Loc);
  }
  getStreamer().emitLabel(Sym, Loc);

  CurrentProcedure = Label;
  CurrentProcedureFramed = Framed;
  return false;
}
bool COFFMasmParser::ParseDirectiveEndProc(StringRef Directive, SMLoc Loc) {
  StringRef Label;
  SMLoc LabelLoc = getTok().getLoc();
  if (getParser().parseIdentifier(Label))
    return Error(LabelLoc, "expected identifier for procedure end");

  if (CurrentProcedure.empty())
    return Error(Loc, "endp outside of procedure block");
  else if (CurrentProcedure != Label)
    return Error(LabelLoc, "endp does not match current procedure '" +
                               CurrentProcedure + "'");

  if (CurrentProcedureFramed) {
    getStreamer().EmitWinCFIEndProc(Loc);
  }
  CurrentProcedure = "";
  CurrentProcedureFramed = false;
  return false;
}

bool COFFMasmParser::ParseDirectiveAlias(StringRef Directive, SMLoc Loc) {
  std::string AliasName, ActualName;
  if (getTok().isNot(AsmToken::Less) ||
      getParser().parseAngleBracketString(AliasName))
    return Error(getTok().getLoc(), "expected <aliasName>");
  if (getParser().parseToken(AsmToken::Equal))
    return addErrorSuffix(" in " + Directive + " directive");
  if (getTok().isNot(AsmToken::Less) ||
      getParser().parseAngleBracketString(ActualName))
    return Error(getTok().getLoc(), "expected <actualName>");

  MCSymbol *Alias = getContext().getOrCreateSymbol(AliasName);
  MCSymbol *Actual = getContext().getOrCreateSymbol(ActualName);

  getStreamer().emitWeakReference(Alias, Actual);

  return false;
}

bool COFFMasmParser::ParseSEHDirectiveAllocStack(StringRef Directive,
                                                 SMLoc Loc) {
  int64_t Size;
  SMLoc SizeLoc = getTok().getLoc();
  if (getParser().parseAbsoluteExpression(Size))
    return Error(SizeLoc, "expected integer size");
  if (Size % 8 != 0)
    return Error(SizeLoc, "stack size must be a multiple of 8");
  getStreamer().EmitWinCFIAllocStack(static_cast<unsigned>(Size), Loc);
  return false;
}

bool COFFMasmParser::ParseSEHDirectiveEndProlog(StringRef Directive,
                                                SMLoc Loc) {
  getStreamer().EmitWinCFIEndProlog(Loc);
  return false;
}

namespace llvm {

MCAsmParserExtension *createCOFFMasmParser() { return new COFFMasmParser; }

} // end namespace llvm
