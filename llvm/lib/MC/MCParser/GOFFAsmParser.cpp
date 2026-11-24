//===- GOFFAsmParser.cpp - GOFF Assembly Parser ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Parses HLASM-dialect GOFF directives for llvm-mc / s390x-ibm-zos:
//
//   <name> CSECT
//   <name> CATTR ALIGN(n)[,FILL(n)][,DEFLOAD|NOLOAD]
//                [,NOTEXECUTABLE|EXECUTABLE][,READONLY][,RMODE(n)]
//                [,PART(<partname>)
//                  [,PRIORITY(n)][,LINKAGE(OS|XPLINK)]
//                  [,SCOPE(SECTION|MODULE|LIBRARY|EXPORT)]
//                  [,WEAK|STRONG][,RENAME|NORENAME]]
//   <name> XATTR LINKAGE(OS|XPLINK)[,REFERENCE(CODE|DATA[,INDIRECT])]
//                [,PSECT(<partname>)][,SCOPE(SECTION|MODULE|LIBRARY|EXPORT)]
//   <name> ALIAS C'<alias_string>'|'<alias_string>'|<identifier>
//   <name> EQU <expression>[,<length>][,<type>]
//   [<name>] DS  0H|0F|0D|0B|0Q
//   [<name>] DC  VD(<sym>)|RD(<sym>)|AD(<sym>)|QD(<sym>)
//   [<name>] DC  [m]XL<n>'<hexdigits>'
//   [<name>] EXTRN <sym>[,<sym>...]
//   [<name>] WXTRN <sym>[,<sym>...]
//            ENTRY <sym>
//            END  [<entry>]
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/BinaryFormat/GOFF.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCGOFFAttributes.h"
#include "llvm/MC/MCParser/AsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCAsmParserExtension.h"
#include "llvm/MC/MCSectionGOFF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbolGOFF.h"
#include "llvm/MC/SectionKind.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"

using namespace llvm;

namespace {

class GOFFAsmParser : public MCAsmParserExtension {
  template <bool (GOFFAsmParser::*HandlerMethod)(StringRef, SMLoc)>
  void addDirectiveHandler(StringRef Directive) {
    MCAsmParser::ExtensionDirectiveHandler Handler =
        std::make_pair(this, HandleDirective<GOFFAsmParser, HandlerMethod>);
    getParser().addDirectiveHandler(Directive, Handler);
  }

  //===--------------------------------------------------------------------===//
  // Helpers
  //===--------------------------------------------------------------------===//

  // In HLASM fixed-format, LexHLASMStrings=false keeps SkipSpace=false so
  // spaces appear as AsmToken::Space tokens.  These helpers treat Space as
  // equivalent to EndOfStatement for operand-field termination purposes.

  // Skip any leading Space tokens.
  void skipSpaces() {
    while (getLexer().is(AsmToken::Space))
      Lex();
  }

  // Return true if the current token ends the operand field (EOL or trailing
  // spaces before EOL). In HLASM fixed-format mode spaces are explicit tokens.
  bool isEOL() {
    return getLexer().is(AsmToken::EndOfStatement) ||
           getLexer().is(AsmToken::Space);
  }

  // Parse a parenthesised absolute integer, e.g. ALIGN(3) -> 3.
  // A decimal number split across an HLASM continuation line is lexed as
  // adjacent Integer tokens, which are joined here.
  // Leaves the lexer after ')'.  Returns true on error.
  bool parseParenInt(int64_t &Val, const std::string &KeyU) {
    skipSpaces();
    if (getLexer().isNot(AsmToken::LParen))
      return TokError("expected '(' after " + KeyU);
    Lex();
    skipSpaces();
    if (getLexer().is(AsmToken::Integer)) {
      SMLoc NumLoc = getLexer().getLoc();
      std::string Digits;
      while (getLexer().is(AsmToken::Integer)) {
        Digits += getTok().getString().str();
        Lex();
        // Blanks after column 72 of a continued line precede the next piece.
        skipSpaces();
      }
      if (StringRef(Digits).getAsInteger(10, Val))
        return Error(NumLoc, "invalid integer");
    } else if (getParser().parseAbsoluteExpression(Val)) {
      return true;
    }
    skipSpaces();
    if (getLexer().isNot(AsmToken::RParen))
      return TokError("expected ')' after " + KeyU);
    Lex();
    return false;
  }

  // Parse a parenthesised identifier or integer token, e.g. RMODE(64) -> "64".
  // Leaves the lexer after ')'.  Returns true on error.
  bool parseParenIdent(StringRef &Ident, const std::string &KeyU) {
    skipSpaces();
    if (getLexer().isNot(AsmToken::LParen))
      return TokError("expected '(' after " + KeyU);
    Lex();
    skipSpaces();
    // Accept both identifier tokens (e.g. ANY) and integer tokens (e.g. 64).
    if (getLexer().is(AsmToken::Integer)) {
      Ident = getTok().getString();
      Lex();
    } else if (parseHLASMSymName(Ident)) {
      return TokError("expected identifier or integer");
    }
    skipSpaces();
    if (getLexer().isNot(AsmToken::RParen))
      return TokError("expected ')' after " + KeyU);
    Lex();
    return false;
  }

  // Parse an HLASM symbol name that may contain characters not accepted by
  // the standard parseIdentifier: dots ('.'), ampersands ('&'), hash ('#'),
  // at ('@'), dollar ('$'), and alphanumerics. Consecutive name tokens
  // (including those split across line continuations) are joined.
  //
  // Examples:  .&ppa2   L#PPA2   call_zos_i128#S
  //
  // On success the full name span is returned in Name and the lexer is left
  // positioned at the first non-name token.  Returns true on error.
  bool parseHLASMSymName(StringRef &Name) {
    // Accepted token kinds that may form part of a symbol name.
    auto isNamePart = [](AsmToken::TokenKind K) {
      return K == AsmToken::Identifier || K == AsmToken::Integer ||
             K == AsmToken::Dot || K == AsmToken::Amp || K == AsmToken::Hash ||
             K == AsmToken::At || K == AsmToken::Dollar ||
             K == AsmToken::Real || K == AsmToken::String;
    };

    skipSpaces();
    if (!isNamePart(getTok().getKind()))
      return TokError("expected symbol name");

    std::string Combined;
    while (isNamePart(getTok().getKind())) {
      std::string Part = getTok().getString().str();
      if (getTok().is(AsmToken::String)) {
        if (Part.size() >= 2 && Part.front() == '"' && Part.back() == '"')
          Part = Part.substr(1, Part.size() - 2);
      }
      Combined += Part;
      Lex();
      skipSpaces();
    }

    if (Combined.empty())
      return TokError("expected symbol name");

    Name = getContext().allocateString(Combined);
    return false;
  }

  // Remove HLASM continuation filler from a token that spans continued lines:
  // everything from column 72 to the end of a continued line, and the leading
  // 15 blanks of the line that follows it.  \p StartCol is the 0-based column
  // of Raw[0] on its line.
  static std::string stripHLASMContinuations(StringRef Raw, unsigned StartCol) {
    std::string Out;
    unsigned Col = StartCol;
    while (!Raw.empty()) {
      size_t EOL = Raw.find_first_of("\r\n");
      if (EOL == StringRef::npos) {
        Out += Raw;
        break;
      }
      size_t Keep = Col < 71 ? std::min<size_t>(EOL, 71 - Col) : 0;
      Out += Raw.substr(0, Keep);
      Raw = Raw.drop_front(EOL);
      Raw = Raw.consume_front("\r\n") ? Raw : Raw.drop_front(1);
      Raw = Raw.drop_front(std::min<size_t>(15, Raw.size()));
      Col = 15;
    }
    return Out;
  }

  // Return the 0-based column of Loc within its line.
  unsigned getColumn(SMLoc Loc) {
    SourceMgr &SM = getParser().getSourceManager();
    const MemoryBuffer *Buf =
        SM.getMemoryBuffer(SM.FindBufferContainingLoc(Loc));
    const char *P = Loc.getPointer();
    const char *Start = P;
    while (Start > Buf->getBufferStart() && Start[-1] != '\n' &&
           Start[-1] != '\r')
      --Start;
    return P - Start;
  }

  // Consume one or more commas (and any surrounding spaces); return true if
  // consumed.
  bool parseOptionalComma() {
    skipSpaces();
    bool Consumed = false;
    while (getLexer().is(AsmToken::Comma)) {
      Lex();
      skipSpaces();
      Consumed = true;
    }
    return Consumed;
  }

  //===--------------------------------------------------------------------===//
  // Directive handlers
  //===--------------------------------------------------------------------===//
  bool parseDirectiveCSECT(StringRef, SMLoc);
  bool parseDirectiveCATTR(StringRef, SMLoc);
  bool parseDirectiveXATTR(StringRef, SMLoc);
  bool parseDirectiveEQU(StringRef, SMLoc);
  bool parseDirectiveDS(StringRef, SMLoc);
  bool parseDirectiveDC(StringRef, SMLoc);
  bool parseDirectiveEXTRN(StringRef, SMLoc);
  bool parseDirectiveENTRY(StringRef, SMLoc);
  bool parseDirectiveALIAS(StringRef, SMLoc);
  bool parseDirectiveEND(StringRef, SMLoc);

  void Initialize(MCAsmParser &Parser) override {
    MCAsmParserExtension::Initialize(Parser);

    addDirectiveHandler<&GOFFAsmParser::parseDirectiveCSECT>("CSECT");
    addDirectiveHandler<&GOFFAsmParser::parseDirectiveCATTR>("CATTR");
    addDirectiveHandler<&GOFFAsmParser::parseDirectiveXATTR>("XATTR");
    addDirectiveHandler<&GOFFAsmParser::parseDirectiveALIAS>("ALIAS");
    addDirectiveHandler<&GOFFAsmParser::parseDirectiveEQU>("EQU");
    addDirectiveHandler<&GOFFAsmParser::parseDirectiveDS>("DS");
    addDirectiveHandler<&GOFFAsmParser::parseDirectiveDC>("DC");
    addDirectiveHandler<&GOFFAsmParser::parseDirectiveEXTRN>("EXTRN");
    addDirectiveHandler<&GOFFAsmParser::parseDirectiveEXTRN>("WXTRN");
    addDirectiveHandler<&GOFFAsmParser::parseDirectiveENTRY>("ENTRY");
    addDirectiveHandler<&GOFFAsmParser::parseDirectiveEND>("END");
  }

public:
  GOFFAsmParser() = default;
};

} // namespace

//===----------------------------------------------------------------------===//
// CSECT  –  introduce a section definition (SD)
//
//   <name> CSECT
//
// The current section has the label's name. it's used the SD name and the
// default C_CODE64 ED is created underneath it.
//===----------------------------------------------------------------------===//
bool GOFFAsmParser::parseDirectiveCSECT(StringRef, SMLoc Loc) {
  // In HLASM, "name CSECT" names a section definition (SD); "name" is NOT a
  // code label.  HLASMAsmParser::parseAsHLASMLabel has already called
  // emitLabel() on it and stored the name in LastHLASMLabelName.
  StringRef SDName = getParser().getLastHLASMLabelName();
  if (SDName.empty())
    return Error(Loc, "CSECT requires a label (section name) in column 1");

  if (!isEOL())
    return TokError("unexpected token in CSECT directive");

  MCContext &Ctx = getContext();

  // Mark the SD symbol as redefinable so that a repeated CSECT (resuming the
  // same section later in the file) does not produce a "symbol already
  // defined" error when parseAsHLASMLabel emits the label again.
  MCSymbol *NameSym = Ctx.lookupSymbol(SDName);
  if (NameSym)
    NameSym->setRedefinable(true);

  // Create (or resume) the SD.
  MCSectionGOFF *SD = Ctx.getGOFFSection(
      SectionKind::getText(), SDName,
      GOFF::SDAttr{GOFF::ESD_TA_Unspecified, GOFF::ESD_BSC_Unspecified});

  // Create the default C_CODE64 ED (readonly, RMODE(64)) under the SD.
  GOFF::EDAttr EDAttr;
  EDAttr.IsReadOnly = true;
  EDAttr.Rmode = GOFF::ESD_RMODE_64;
  EDAttr.NameSpace = GOFF::ESD_NS_NormalName;
  EDAttr.TextStyle = GOFF::ESD_TS_ByteOriented;
  EDAttr.BindAlgorithm = GOFF::ESD_BA_Concatenate;
  EDAttr.LoadBehavior = GOFF::ESD_LB_Initial;
  MCSectionGOFF *ED =
      Ctx.getGOFFSection(SectionKind::getText(), GOFF::CLASS_CODE, EDAttr, SD);
  ED->setAlignment(Align(8));

  getStreamer().switchSection(ED);
  return false;
}

//===----------------------------------------------------------------------===//
// CATTR  –  introduce an element definition (ED) / part reference (PR)
//
//   <name> CATTR ALIGN(n)[,FILL(n)][,DEFLOAD|NOLOAD]
//                [,NOTEXECUTABLE|EXECUTABLE][,READONLY][,RMODE(n)]
//                [,PART(<partname>)
//                  [,PRIORITY(n)][,LINKAGE(OS|XPLINK)]
//                  [,SCOPE(SECTION|MODULE|LIBRARY|EXPORT)]
//                  [,WEAK|STRONG][,RENAME|NORENAME]]
//
// The label in column 1 is the ED (class) name.
// The parent SD must have been introduced by a prior CSECT.
//===----------------------------------------------------------------------===//
bool GOFFAsmParser::parseDirectiveCATTR(StringRef, SMLoc Loc) {
  // The ED (class) name is the label in column 1 of this statement.
  StringRef EDName = getParser().getLastHLASMLabelName();
  if (EDName.empty())
    return Error(Loc, "CATTR requires a label (class name) in column 1");

  // Mark the ED-name symbol as redefinable for repeated CATTR occurrences.
  MCSymbol *EDSym = getContext().lookupSymbol(EDName);
  if (EDSym)
    EDSym->setRedefinable(true);

  // Find the current SD.  After a CSECT we are on its C_CODE64 ED (a child of
  // the SD), so walk up to the SD.
  MCSection *CurSec = getStreamer().getCurrentSectionOnly();
  if (!CurSec)
    return Error(Loc, "CATTR must follow a CSECT");

  MCSectionGOFF *ParentSD = static_cast<MCSectionGOFF *>(CurSec)->getParentSD();
  if (!ParentSD)
    return Error(Loc, "CATTR must follow a CSECT");

  // Attribute defaults.
  GOFF::EDAttr Attr;
  Attr.Rmode = GOFF::ESD_RMODE_64;
  uint8_t AlignLog = 3;
  StringRef PartName;
  bool IsExecutable = false; // track EXECUTABLE/NOTEXECUTABLE for PR

  // PR-specific attribute defaults (only used when PART(...) is present).
  GOFF::PRAttr PRAttrOverride;
  PRAttrOverride.Linkage = GOFF::ESD_LT_XPLink;
  PRAttrOverride.BindingScope = GOFF::ESD_BSC_Section;
  PRAttrOverride.BindingStrength = GOFF::ESD_BST_Strong;
  PRAttrOverride.IsRenameable = true;

  // Parse comma-separated key[([val])] options until EOL.
  while (!isEOL()) {
    parseOptionalComma();
    if (isEOL())
      break;
    StringRef Key;
    SMLoc KeyLoc = getLexer().getLoc();
    if (getParser().parseIdentifier(Key))
      return TokError("expected CATTR keyword");

    std::string KeyU = Key.upper();
    if (KeyU == "ALIGN") {
      int64_t Val;
      if (parseParenInt(Val, KeyU))
        return true;
      if (Val < 0 || Val > 12)
        return Error(KeyLoc, "ALIGN value out of range (0–12)");
      AlignLog = static_cast<uint8_t>(Val);
    } else if (KeyU == "FILL") {
      int64_t Val;
      if (parseParenInt(Val, KeyU))
        return true;
      Attr.FillByteValue = static_cast<uint8_t>(Val);
    } else if (KeyU == "DEFLOAD") {
      Attr.LoadBehavior = GOFF::ESD_LB_Deferred;
    } else if (KeyU == "NOLOAD") {
      Attr.LoadBehavior = GOFF::ESD_LB_NoLoad;
    } else if (KeyU == "NOTEXECUTABLE") {
      IsExecutable = false;
    } else if (KeyU == "EXECUTABLE") {
      IsExecutable = true;
    } else if (KeyU == "READONLY") {
      Attr.IsReadOnly = true;
    } else if (KeyU == "CHANGE") {
      // CHANGE → structured (byte-oriented records with change bits).
      Attr.TextStyle = GOFF::ESD_TS_Structured;
    } else if (KeyU == "MERGE") {
      // MERGE → binder merges identical classes rather than concatenating.
      Attr.BindAlgorithm = GOFF::ESD_BA_Merge;
    } else if (KeyU == "REMOVABLE") {
      Attr.IsRemovable = true;
    } else if (KeyU == "NAMESPACE") {
      StringRef NSStr;
      if (parseParenIdent(NSStr, KeyU))
        return true;
      std::string NSU = NSStr.upper();
      Attr.NameSpace = StringSwitch<GOFF::ESDNameSpaceId>(NSU)
                           .Case("NORMAL", GOFF::ESD_NS_NormalName)
                           .Case("PARTS", GOFF::ESD_NS_Parts)
                           .Case("PMB", GOFF::ESD_NS_ProgramManagementBinder)
                           .Case("PSEUDOREG", GOFF::ESD_NS_PseudoRegister)
                           .Default(static_cast<GOFF::ESDNameSpaceId>(~0u));
      if (Attr.NameSpace == static_cast<GOFF::ESDNameSpaceId>(~0u))
        return Error(KeyLoc, "unknown NAMESPACE value '" + NSStr + "'");
    } else if (KeyU == "RSECT") {
      int64_t Val;
      if (parseParenInt(Val, KeyU))
        return true;
      if (Val < 0 || Val > 3)
        return Error(KeyLoc, "RSECT value out of range (0–3)");
      Attr.ReservedQwords = static_cast<GOFF::ESDReserveQwords>(Val);
    } else if (KeyU == "PRIORITY") {
      // PRIORITY(n) sets the sort key on the PR symbol.
      int64_t Val;
      if (parseParenInt(Val, KeyU))
        return true;
      PRAttrOverride.SortKey = static_cast<uint32_t>(Val);
    } else if (KeyU == "LINKAGE") {
      // LINKAGE inside CATTR applies to the PR linkage type.
      StringRef Val;
      if (parseParenIdent(Val, KeyU))
        return true;
      std::string ValU = Val.upper();
      if (ValU == "OS")
        PRAttrOverride.Linkage = GOFF::ESD_LT_OS;
      else if (ValU == "XPLINK")
        PRAttrOverride.Linkage = GOFF::ESD_LT_XPLink;
      else
        return Error(KeyLoc, "unknown LINKAGE value '" + Val + "'");
    } else if (KeyU == "SCOPE") {
      // SCOPE inside CATTR applies to the PR binding scope.
      StringRef Val;
      if (parseParenIdent(Val, KeyU))
        return true;
      std::string ValU = Val.upper();
      PRAttrOverride.BindingScope =
          StringSwitch<GOFF::ESDBindingScope>(ValU)
              .Case("SECTION", GOFF::ESD_BSC_Section)
              .Case("MODULE", GOFF::ESD_BSC_Module)
              .Case("LIBRARY", GOFF::ESD_BSC_Library)
              .Case("EXPORT", GOFF::ESD_BSC_ImportExport)
              .Default(GOFF::ESD_BSC_Unspecified);
      if (PRAttrOverride.BindingScope == GOFF::ESD_BSC_Unspecified)
        return Error(KeyLoc, "unknown SCOPE value '" + Val + "'");
    } else if (KeyU == "WEAK") {
      PRAttrOverride.BindingStrength = GOFF::ESD_BST_Weak;
    } else if (KeyU == "STRONG") {
      PRAttrOverride.BindingStrength = GOFF::ESD_BST_Strong;
    } else if (KeyU == "RENAME") {
      PRAttrOverride.IsRenameable = true;
    } else if (KeyU == "NORENAME") {
      PRAttrOverride.IsRenameable = false;
    } else if (KeyU == "RMODE") {
      StringRef RmodeStr;
      if (parseParenIdent(RmodeStr, KeyU))
        return true;
      Attr.Rmode = StringSwitch<GOFF::ESDRmode>(RmodeStr)
                       .Case("24", GOFF::ESD_RMODE_24)
                       .Case("31", GOFF::ESD_RMODE_31)
                       .Case("64", GOFF::ESD_RMODE_64)
                       .Default(GOFF::ESD_RMODE_None);
      if (Attr.Rmode == GOFF::ESD_RMODE_None)
        return Error(KeyLoc, "unknown RMODE value '" + RmodeStr + "'");
    } else if (KeyU == "PART") {
      if (parseParenIdent(PartName, KeyU))
        return true;
    } else {
      // Unknown / future key — skip the optional (val) argument.
      if (getLexer().is(AsmToken::LParen)) {
        Lex();
        unsigned Depth = 1;
        while (Depth > 0 && getLexer().isNot(AsmToken::Eof)) {
          if (getLexer().is(AsmToken::LParen))
            ++Depth;
          else if (getLexer().is(AsmToken::RParen))
            --Depth;
          Lex();
        }
      }
    }
    parseOptionalComma();
  }

  MCContext &Ctx = getContext();
  SectionKind Kind = (!Attr.IsReadOnly || !IsExecutable)
                         ? SectionKind::getData()
                         : SectionKind::getText();

  MCSectionGOFF *ED = Ctx.getGOFFSection(Kind, EDName, Attr, ParentSD);
  ED->setAlignment(Align(1ULL << AlignLog));

  if (!PartName.empty()) {
    PartName = Ctx.getCanonicalHLASMName(PartName);
    PRAttrOverride.Executable =
        IsExecutable ? GOFF::ESD_EXE_CODE : GOFF::ESD_EXE_DATA;
    MCSectionGOFF *PR = Ctx.getGOFFSection(Kind, PartName, PRAttrOverride, ED);
    getStreamer().switchSection(PR);
  } else {
    getStreamer().switchSection(ED);
  }
  return false;
}

//===----------------------------------------------------------------------===//
// XATTR  –  set extended symbol attributes
//
//   <name> XATTR LINKAGE(OS|XPLINK)
//                [,REFERENCE(CODE|DATA[,INDIRECT])]
//                [,PSECT(<partname>)]
//                [,SCOPE(SECTION|MODULE|LIBRARY|EXPORT)]
//
// The label in column 1 names the symbol to annotate.  XATTR never emits
// bytes; it only sets properties on the MCSymbolGOFF.
//===----------------------------------------------------------------------===//
bool GOFFAsmParser::parseDirectiveXATTR(StringRef, SMLoc Loc) {
  StringRef SymName = getParser().getLastHLASMLabelName();
  if (SymName.empty())
    return Error(Loc, "XATTR requires a label (symbol name) in column 1");

  MCSymbol *Sym = getContext().getOrCreateSymbol(SymName);
  auto *GOFFSym = static_cast<MCSymbolGOFF *>(Sym);

  // Mark redefinable so repeated XATTR on same symbol is allowed.
  Sym->setRedefinable(true);

  // Defaults.
  GOFF::ESDLinkageType Linkage = GOFF::ESD_LT_XPLink;
  GOFF::ESDExecutable Executable = GOFF::ESD_EXE_Unspecified;
  bool IsIndirect = false;
  GOFF::ESDBindingScope BindingScope = GOFF::ESD_BSC_Unspecified;
  StringRef PSectName;

  bool SeenLinkage = false;

  // Parse comma-separated key(val) options until EOL.
  while (!isEOL()) {
    parseOptionalComma();
    if (isEOL())
      break;
    StringRef Key;
    SMLoc KeyLoc = getLexer().getLoc();
    if (getParser().parseIdentifier(Key))
      return TokError("expected XATTR keyword");

    std::string KeyU = Key.upper();
    if (KeyU == "LINKAGE") {
      StringRef Val;
      if (parseParenIdent(Val, KeyU))
        return true;
      std::string ValU = Val.upper();
      if (ValU == "OS")
        Linkage = GOFF::ESD_LT_OS;
      else if (ValU == "XPLINK")
        Linkage = GOFF::ESD_LT_XPLink;
      else
        return Error(KeyLoc, "unknown LINKAGE value '" + Val + "'");
      SeenLinkage = true;
    } else if (KeyU == "REFERENCE") {
      skipSpaces();
      if (getLexer().isNot(AsmToken::LParen))
        return TokError("expected '(' after REFERENCE");
      Lex();
      // Parse comma-separated reference flags: CODE|DATA, INDIRECT.
      while (getLexer().isNot(AsmToken::RParen) && !isEOL()) {
        skipSpaces();
        StringRef Flag;
        SMLoc FlagLoc = getLexer().getLoc();
        if (getParser().parseIdentifier(Flag))
          return TokError("expected REFERENCE flag");
        std::string FlagU = Flag.upper();
        if (FlagU == "CODE")
          Executable = GOFF::ESD_EXE_CODE;
        else if (FlagU == "DATA")
          Executable = GOFF::ESD_EXE_DATA;
        else if (FlagU == "INDIRECT")
          IsIndirect = true;
        else
          return Error(FlagLoc, "unknown REFERENCE flag '" + Flag + "'");
        parseOptionalComma();
      }
      skipSpaces();
      if (getLexer().isNot(AsmToken::RParen))
        return TokError("expected ')' after REFERENCE flags");
      Lex();
    } else if (KeyU == "PSECT") {
      if (parseParenIdent(PSectName, KeyU))
        return true;
    } else if (KeyU == "SCOPE") {
      StringRef Val;
      if (parseParenIdent(Val, KeyU))
        return true;
      std::string ValU = Val.upper();
      BindingScope = StringSwitch<GOFF::ESDBindingScope>(ValU)
                         .Case("SECTION", GOFF::ESD_BSC_Section)
                         .Case("MODULE", GOFF::ESD_BSC_Module)
                         .Case("LIBRARY", GOFF::ESD_BSC_Library)
                         .Case("EXPORT", GOFF::ESD_BSC_ImportExport)
                         .Default(GOFF::ESD_BSC_Unspecified);
      if (BindingScope == GOFF::ESD_BSC_Unspecified)
        return Error(KeyLoc, "unknown SCOPE value '" + Val + "'");
    } else {
      // Unknown / future key — skip the optional (val) argument.
      if (getLexer().is(AsmToken::LParen)) {
        Lex();
        unsigned Depth = 1;
        while (Depth > 0 && getLexer().isNot(AsmToken::Eof)) {
          if (getLexer().is(AsmToken::LParen))
            ++Depth;
          else if (getLexer().is(AsmToken::RParen))
            --Depth;
          Lex();
        }
      }
    }
    parseOptionalComma();
  }

  if (!SeenLinkage)
    return Error(Loc, "XATTR requires at least LINKAGE(OS|XPLINK)");

  // Apply linkage.
  GOFFSym->setLinkage(Linkage);

  // Apply code/data reference type.
  if (Executable != GOFF::ESD_EXE_Unspecified)
    GOFFSym->setCodeData(Executable);

  // Apply indirect reference flag.
  if (IsIndirect)
    GOFFSym->setIndirect(true);

  // Apply PSECT (associated data area section).
  // PSECT names a PR section created by a prior CATTR PART(...).
  // Use lookupGOFFSection to find it by name prefix in the uniquing map.
  if (!PSectName.empty()) {
    MCSectionGOFF *PSectSection = getContext().lookupGOFFSection(
        getContext().getCanonicalHLASMName(PSectName));
    if (PSectSection && PSectSection->isPR())
      GOFFSym->setADA(PSectSection);
  }

  // Apply binding scope.  SCOPE(EXPORT) implies the symbol is global.
  if (BindingScope != GOFF::ESD_BSC_Unspecified) {
    if (BindingScope == GOFF::ESD_BSC_ImportExport)
      getStreamer().emitSymbolAttribute(Sym, MCSA_Global);
    // For section/module/library scope on a defined symbol we need to set
    // IsExternal so getBindingScope() returns the right value.
    // Currently getBindingScope() returns Section for any defined symbol
    // unless isExternal()==true; so only EXPORT changes the GOFF output.
    // TODO: add an explicit BindingScope field to MCSymbolGOFF.
  }

  // If the XATTR label names the current PR section's own symbol, emit it
  // as a label so the symbol becomes defined in that section.  Without this,
  // the MCSymbol remains undefined and later triggers a spurious
  // defineExtern call in the GOFF writer.
  // After emitting, mark the symbol redefinable so that a following DS/DC
  // directive (which calls emitLabel with the same name after any alignment
  // padding) can move the symbol to the correct post-alignment offset.
  if (MCSection *CurSec = getStreamer().getCurrentSectionOnly()) {
    auto *PRSec = static_cast<MCSectionGOFF *>(CurSec);
    if (PRSec->isPR() && PRSec->getName().equals_insensitive(SymName)) {
      getStreamer().emitLabel(Sym, Loc);
      Sym->setRedefinable(true);
    }
  }

  return false;
}

//===----------------------------------------------------------------------===//
// ALIAS  –  define external alias name for a symbol
//
//   <name> ALIAS C'<alias_string>'
//   <name> ALIAS '<alias_string>'
//   <name> ALIAS <identifier>
//===----------------------------------------------------------------------===//
bool GOFFAsmParser::parseDirectiveALIAS(StringRef, SMLoc Loc) {
  StringRef SymName = getParser().getLastHLASMLabelName();
  if (SymName.empty())
    return Error(Loc, "ALIAS requires a label (symbol name) in column 1");

  skipSpaces();
  std::string AliasStr;

  if (getLexer().is(AsmToken::Identifier) &&
      getTok().getString().upper() == "C") {
    // Character string with C'...' prefix:
    AsmLexer &Lex = getLexer();
    Lex.setLexHLASMStrings(false);
    Lex.setLexMasmStrings(true);
    this->Lex(); // consume "C", lazily lexes '...'
    AsmToken QuoteTok = getTok();
    Lex.setLexMasmStrings(false);
    Lex.setLexHLASMStrings(true);
    this->Lex(); // consume the string token

    if (QuoteTok.isNot(AsmToken::String))
      return TokError("expected quoted string after ALIAS C");

    std::string RawStr = stripHLASMContinuations(QuoteTok.getString(),
                                                 getColumn(QuoteTok.getLoc()));
    StringRef Raw = RawStr;
    if (Raw.size() < 2 || Raw.front() != '\'' || Raw.back() != '\'')
      return TokError("malformed string in ALIAS directive");
    AliasStr = Raw.slice(1, Raw.size() - 1).str();
  } else if (getLexer().is(AsmToken::String)) {
    StringRef Raw = getTok().getString();
    if (Raw.size() < 2 || Raw.front() != '\'' || Raw.back() != '\'')
      return TokError("malformed string in ALIAS directive");
    AliasStr = Raw.slice(1, Raw.size() - 1).str();
    Lex();
  } else if (getLexer().is(AsmToken::Identifier)) {
    AliasStr = getTok().getString().str();
    Lex();
  } else {
    return TokError(
        "expected alias name (C'string' or identifier) in ALIAS directive");
  }

  if (!isEOL())
    return TokError("unexpected token after ALIAS");

  MCSymbol *Sym = getContext().getOrCreateSymbol(SymName);
  auto *GOFFSym = static_cast<MCSymbolGOFF *>(Sym);
  GOFFSym->setExternalName(getContext().allocateString(AliasStr));
  return false;
}

//===----------------------------------------------------------------------===//
// EQU  –  equate symbol to expression
//
//   <name> EQU <expression>[,<length>][,<type>]
//
//===----------------------------------------------------------------------===//
bool GOFFAsmParser::parseDirectiveEQU(StringRef, SMLoc Loc) {
  StringRef SymName = getParser().getLastHLASMLabelName();
  if (SymName.empty())
    return Error(Loc, "EQU requires a label (symbol name) in column 1");

  const MCExpr *Value = nullptr;
  if (getParser().parseExpression(Value))
    return true;

  // HLASM EQU may have optional length and type fields: EQU expr,length,type.
  // Consume any optional comma-separated trailing arguments until EOL.
  while (parseOptionalComma() && !isEOL()) {
    const MCExpr *DummyExpr = nullptr;
    if (getLexer().is(AsmToken::Identifier) ||
        getLexer().is(AsmToken::Integer)) {
      if (getParser().parseExpression(DummyExpr))
        Lex();
    } else {
      Lex();
    }
  }

  if (!isEOL())
    return TokError("unexpected token in EQU directive");

  MCSymbol *Sym = getContext().getOrCreateSymbol(SymName);
  Sym->setRedefinable(true);
  getStreamer().emitAssignment(Sym, Value);
  return false;
}

//===----------------------------------------------------------------------===//
// DS  –  define storage / alignment boundary
//
// Only the alignment form (duplication factor = 0) is supported for now.
//   DS 0B   byte      (1-byte align, no-op)
//   DS 0H   halfword  (2-byte align)
//   DS 0F   fullword  (4-byte align)
//   DS 0D   doubleword (8-byte align)
//   DS 0Q   quadword  (16-byte align)
//
//===----------------------------------------------------------------------===//
bool GOFFAsmParser::parseDirectiveDS(StringRef, SMLoc Loc) {
  int64_t DupFactor = 0;
  if (getLexer().is(AsmToken::Integer)) {
    DupFactor = getTok().getIntVal();
    Lex();
  }

  if (getLexer().isNot(AsmToken::Identifier))
    return TokError("expected DS type character (B, H, F, D, Q)");

  std::string TypeStr = getTok().getString().upper();
  Lex();

  if (!isEOL())
    return TokError("unexpected token in DS directive");

  if (DupFactor != 0)
    return Error(Loc, "DS with non-zero duplication factor not yet supported");

  unsigned AlignBytes = StringSwitch<unsigned>(TypeStr)
                            .Case("B", 1)
                            .Case("H", 2)
                            .Case("F", 4)
                            .Case("D", 8)
                            .Case("Q", 16)
                            .Default(0);

  if (AlignBytes == 0)
    return Error(Loc, "unsupported DS type '" + TypeStr + "'");

  Align Alignment(AlignBytes);

  getStreamer().emitValueToAlignment(Alignment);

  StringRef SymName = getParser().getLastHLASMLabelName();
  if (!SymName.empty()) {
    MCSymbol *Sym = getContext().getOrCreateSymbol(SymName);
    getStreamer().emitLabel(Sym, Loc);
  }

  return false;
}

//===----------------------------------------------------------------------===//
// DC  –  define constant
//
//   DC VD(<sym>)                8-byte absolute address (V-type reference)
//   DC QD(<sym>)                8-byte external address (Q-type reference)
//   DC RD(<sym>)                4-byte PSECT address    (R-type reference)
//   DC AD(<sym>|expr)           4-byte absolute address (A-type reference)
//   DC A(<sym>|expr)            4-byte absolute address (A-type reference)
//   DC AL<n>(<sym>|expr)        <n>-byte address constant
//   DC XL<n>'<hexdigits>'       hex byte literal of length <n>
//   DC <m>XL<n>'<hexdigits>'    hex byte literal of length <n> repeat <m> times
//   DC X'<hexdigits>'           hex byte literal
//
// Also supports comma-separated multi-operand lists, e.g.:
//   DC AD(L#tmp1/2),XL1'0'
//   DC XL1'03',XL1'E7',XL1'22'
//===----------------------------------------------------------------------===//
bool GOFFAsmParser::parseDirectiveDC(StringRef, SMLoc Loc) {
  StringRef SymName = getParser().getLastHLASMLabelName();
  if (!SymName.empty()) {
    MCSymbol *Sym = getContext().getOrCreateSymbol(SymName);
    getStreamer().emitLabel(Sym, Loc);
  }

  do {
    skipSpaces();

    // Optional duplication factor: DC <n>XL<m>'...' repeats the item n times.
    unsigned DupFactor = 1;
    if (getLexer().is(AsmToken::Integer)) {
      int64_t Val = getTok().getIntVal();
      if (Val < 1)
        return TokError("DC duplication factor must be a positive integer");
      DupFactor = static_cast<unsigned>(Val);
      Lex(); // consume the integer
      skipSpaces();
    }

    if (getLexer().isNot(AsmToken::Identifier))
      return TokError("expected DC type (VD, QD, RD, AD, A, AL<n>, XL<n>, X)");

    // Peek at the type string before consuming the token.
    std::string TypeStr = getTok().getString().upper();
    SMLoc ItemLoc = getLexer().getLoc();
    auto &MAI = getLexer().getMAI();

    // -----------------------------------------------------------------------
    // DC XL<n>'<hexdigits>' or DC X'<hexdigits>'  –  hex byte literal
    //
    // LexHLASMStrings=true rejects single-quote strings. Disable it before
    // consuming the type token, so the lazy-lex of the following '...'
    // token succeeds. Restore both flags afterwards.
    // -----------------------------------------------------------------------
    if ((TypeStr.size() >= 3 && TypeStr[0] == 'X' && TypeStr[1] == 'L') ||
        TypeStr == "X") {
      unsigned ByteCount = 0;
      bool HasExplicitLength = false;
      if (TypeStr != "X") {
        StringRef CountStr = StringRef(TypeStr).drop_front(2);
        if (CountStr.getAsInteger(10, ByteCount) || ByteCount == 0)
          return Error(ItemLoc, "invalid XL byte count in DC XL<n>");
        HasExplicitLength = true;
      }

      AsmLexer &Lex = getLexer();
      Lex.setLexHLASMStrings(false);
      Lex.setLexMasmStrings(true);
      this->Lex(); // consume "XL<n>" / "X"; lazily lexes following '...' token
      // Copy the token — getTok() returns a reference that becomes stale
      // once the next Lex() call modifies CurTok.
      AsmToken QuoteTok = getTok();
      Lex.setLexMasmStrings(false);
      Lex.setLexHLASMStrings(true);
      this->Lex(); // consume the string token

      if (QuoteTok.isNot(AsmToken::String))
        return TokError("expected quoted hex string after DC " + TypeStr);

      // Strip surrounding single quotes.
      std::string RawStr = stripHLASMContinuations(
          QuoteTok.getString(), getColumn(QuoteTok.getLoc()));
      StringRef Raw = RawStr;
      if (Raw.size() < 2 || Raw.front() != '\'' || Raw.back() != '\'')
        return TokError("malformed hex string in DC " + TypeStr);
      StringRef Hex = Raw.slice(1, Raw.size() - 1);

      if (!HasExplicitLength)
        ByteCount = (Hex.size() + 1) / 2;

      std::string PaddedHex;
      if (Hex.size() < ByteCount * 2) {
        // HLASM pads hex constants on the left with zeros.
        PaddedHex = std::string(ByteCount * 2 - Hex.size(), '0') + Hex.str();
        Hex = PaddedHex;
      } else if (Hex.size() > ByteCount * 2) {
        return Error(ItemLoc, "DC " + TypeStr + " requires at most " +
                                  std::to_string(ByteCount * 2) +
                                  " hex digits, got " +
                                  std::to_string(Hex.size()));
      }

      // Emit the bytes — decode two hex digits at a time.
      SmallVector<char, 32> Bytes;
      Bytes.resize(ByteCount);
      for (unsigned I = 0; I < ByteCount; ++I) {
        unsigned Hi = hexDigitValue(Hex[2 * I]);
        unsigned Lo = hexDigitValue(Hex[2 * I + 1]);
        if (Hi == ~0U || Lo == ~0U)
          return Error(ItemLoc, "invalid hex digit in DC " + TypeStr);
        Bytes[I] = static_cast<char>((Hi << 4) | Lo);
      }
      for (unsigned D = 0; D < DupFactor; ++D)
        getStreamer().emitBytes(StringRef(Bytes.data(), Bytes.size()));
      continue;
    }

    // -----------------------------------------------------------------------
    // DC VD/RD/QD/AD/JD/A/AL/Q/QL<n>(<expr>)  –  address constants
    // -----------------------------------------------------------------------
    unsigned Size = 4;
    if (TypeStr == "VD" || TypeStr == "RD" || TypeStr == "QD" ||
        TypeStr == "AD" || TypeStr == "JD") {
      Size = 8;
    } else if (TypeStr == "V" || TypeStr == "R" || TypeStr == "Q" ||
               TypeStr == "A" || TypeStr == "J") {
      Size = 4;
    } else if (TypeStr.size() >= 3 &&
               (TypeStr[0] == 'V' || TypeStr[0] == 'Q' || TypeStr[0] == 'A') &&
               TypeStr[1] == 'L') {
      StringRef CountStr = StringRef(TypeStr).drop_front(2);
      if (CountStr.getAsInteger(10, Size) || Size == 0 || Size > 8)
        return Error(ItemLoc, "invalid AL byte count in DC AL<n>");
    } else {
      return Error(ItemLoc, "unsupported DC type '" + TypeStr +
                                "' (supported: VD, RD, QD, AD, JD, *L<n>)");
    }
    uint16_t Spec = *MAI.getSpecifierForName(TypeStr.substr(0, 1) + "D");

    // Consume the type identifier.
    Lex();

    skipSpaces();
    if (getLexer().isNot(AsmToken::LParen))
      return TokError("expected '(' after DC " + TypeStr);
    Lex();

    const MCExpr *Value = nullptr;
    SMLoc SymLoc = getLexer().getLoc();

    // Check for nested address specifier, e.g. QD(sym)+XL8'0' inside AD(...)
    std::string InnerType;
    if (getLexer().is(AsmToken::Identifier) &&
        getLexer().peekTok().is(AsmToken::LParen)) {
      std::string PeekStr = getTok().getString().upper();
      if (PeekStr == "QD" || PeekStr == "RD" || PeekStr == "VD" ||
          PeekStr == "AD" || PeekStr == "JD" || PeekStr == "Q" ||
          PeekStr == "R" || PeekStr == "V" || PeekStr == "A" ||
          PeekStr == "J") {
        InnerType = PeekStr;
      }
    }

    if (!InnerType.empty()) {
      // Consume inner type (e.g. "QD") and '('
      Lex();
      Lex();
      StringRef InnerSymName;
      if (parseHLASMSymName(InnerSymName))
        return TokError("expected symbol name in " + InnerType);
      skipSpaces();
      if (getLexer().isNot(AsmToken::RParen))
        return TokError("expected ')' after " + InnerType + " symbol");
      Lex(); // consume inner ')'

      uint16_t InnerSpec =
          *MAI.getSpecifierForName(InnerType.substr(0, 1) + "D");

      MCSymbol *InnerSym = getContext().getOrCreateSymbol(
          getContext().getCanonicalHLASMName(InnerSymName));
      // Use MCSpecifierExpr so that the constant type is printed as e.g.
      // VD(sym) in assembly output.
      Value =
          MCSpecifierExpr::create(InnerSym, InnerSpec, getContext(), SymLoc);

      // Handle optional + or - offset, e.g. +XL8'0' or +XL8'16' or +16
      while (getLexer().is(AsmToken::Plus) || getLexer().is(AsmToken::Minus)) {
        bool IsAdd = getLexer().is(AsmToken::Plus);
        Lex(); // consume '+' or '-'

        const MCExpr *OffsetExpr = nullptr;
        // Check for XL<n>'<hex>' or X'<hex>' literal as offset
        if (getLexer().is(AsmToken::Identifier)) {
          std::string OffTypeStr = getTok().getString().upper();
          if (StringRef(OffTypeStr).starts_with("XL") || OffTypeStr == "X") {
            AsmLexer &Lex = getLexer();
            Lex.setLexHLASMStrings(false);
            Lex.setLexMasmStrings(true);
            this->Lex();
            AsmToken QuoteTok = getTok();
            Lex.setLexMasmStrings(false);
            Lex.setLexHLASMStrings(true);
            this->Lex();

            if (QuoteTok.isNot(AsmToken::String))
              return TokError("expected quoted hex string in offset");
            std::string RawStr = stripHLASMContinuations(
                QuoteTok.getString(), getColumn(QuoteTok.getLoc()));
            StringRef Raw = RawStr;
            if (Raw.size() < 2 || Raw.front() != '\'' || Raw.back() != '\'')
              return TokError("malformed hex string in offset");
            StringRef Hex = Raw.slice(1, Raw.size() - 1);
            uint64_t OffsetVal = 0;
            if (Hex.getAsInteger(16, OffsetVal))
              return TokError("invalid hex digits in offset");
            OffsetExpr = MCConstantExpr::create(OffsetVal, getContext());
          }
        }
        if (!OffsetExpr) {
          if (getParser().parseExpression(OffsetExpr))
            return true;
        }

        Value = IsAdd
                    ? MCBinaryExpr::createAdd(Value, OffsetExpr, getContext())
                    : MCBinaryExpr::createSub(Value, OffsetExpr, getContext());
      }
    } else {
      const MCExpr *SymExpr = nullptr;
      if (getParser().parseExpression(SymExpr))
        return true;

      Value = SymExpr;
      uint16_t S_None = *MAI.getSpecifierForName("AD");
      if (Spec != S_None) {
        if (const auto *SRE = dyn_cast<MCSymbolRefExpr>(SymExpr))
          Value = MCSpecifierExpr::create(&SRE->getSymbol(), Spec, getContext(),
                                          SymLoc);
      }
    }

    skipSpaces();
    if (getLexer().isNot(AsmToken::RParen))
      return TokError("expected ')' in DC " + TypeStr);
    Lex();

    for (unsigned D = 0; D < DupFactor; ++D)
      getStreamer().emitValue(Value, Size, Loc);
  } while (parseOptionalComma() && !isEOL());

  if (!isEOL())
    return TokError("unexpected token after DC");

  return false;
}

//===----------------------------------------------------------------------===//
// EXTRN / WXTRN  –  declare external symbol reference(s)
//
//   EXTRN sym[,sym...]
//   EXTRN PART(sym[,sym...])
//   WXTRN sym[,sym...]   (weak external)
//
// The PART form declares external references to parts (data symbols). The
// symbols are marked as external DATA references belonging to the class (ED)
// of the part (PR) section that is current at the point of the directive, as
// established by a prior CATTR ...,PART(...).
//===----------------------------------------------------------------------===//
bool GOFFAsmParser::parseDirectiveEXTRN(StringRef DirName, SMLoc Loc) {
  bool IsWeak = DirName.upper() == "WXTRN";

  skipSpaces();
  bool IsPart = false;
  if (getLexer().is(AsmToken::Identifier) &&
      getTok().getString().upper() == "PART" &&
      getLexer().peekTok().is(AsmToken::LParen)) {
    IsPart = true;
    Lex(); // consume PART
    Lex(); // consume '('
  }

  do {
    StringRef SymName;
    SMLoc SymLoc = getLexer().getLoc();
    if (parseHLASMSymName(SymName))
      return Error(SymLoc, "expected symbol name in " + DirName);

    // HLASM is case-insensitive — use the spelling of the first occurrence.
    MCSymbol *Sym = getContext().getOrCreateSymbol(
        getContext().getCanonicalHLASMName(SymName));
    auto *GOFFSym = static_cast<MCSymbolGOFF *>(Sym);
    GOFFSym->setExternal(true);
    if (IsWeak)
      GOFFSym->setWeak(true);

    if (IsPart) {
      // An external part reference is a DATA reference; the writer and the
      // HLASM streamer derive the class (ED) of the external PR from the
      // parent of the symbol's ADA section.
      auto *CurGOFF =
          static_cast<MCSectionGOFF *>(getStreamer().getCurrentSectionOnly());
      if (!CurGOFF || !CurGOFF->isPR())
        return Error(SymLoc, "EXTRN PART(" + SymName +
                                 ") must follow a CATTR with PART(...)");
      GOFFSym->setCodeData(GOFF::ESD_EXE_DATA);
      GOFFSym->setADA(CurGOFF);
    }

    getStreamer().emitSymbolAttribute(Sym, IsWeak ? MCSA_Weak : MCSA_Global);
  } while (parseOptionalComma() && !isEOL() &&
           !(IsPart && getLexer().is(AsmToken::RParen)));

  if (IsPart) {
    skipSpaces();
    if (getLexer().isNot(AsmToken::RParen))
      return TokError("expected ')' after PART symbol list");
    Lex();
  }

  if (!isEOL())
    return TokError("unexpected token in " + DirName + " directive");
  return false;
}

//===----------------------------------------------------------------------===//
// ENTRY  –  declare a public entry point
//
//   ENTRY sym
//
// Marks the named symbol as a global entry point.  This is equivalent to
// SCOPE(EXPORT) in XATTR.
//===----------------------------------------------------------------------===//
bool GOFFAsmParser::parseDirectiveENTRY(StringRef, SMLoc Loc) {
  StringRef EntryName;
  if (isEOL())
    return Error(Loc, "ENTRY requires a symbol name");
  if (parseHLASMSymName(EntryName))
    return TokError("expected symbol name in ENTRY directive");
  if (!isEOL())
    return TokError("unexpected token in ENTRY directive");

  MCSymbol *Sym = getContext().getOrCreateSymbol(EntryName.upper());
  getStreamer().emitSymbolAttribute(Sym, MCSA_Global);
  return false;
}

//===----------------------------------------------------------------------===//
// END  –  end of module
//
//   END [<entry>]
//===----------------------------------------------------------------------===//
bool GOFFAsmParser::parseDirectiveEND(StringRef, SMLoc Loc) {
  // Optional entry-point name.  Mark it as global so that its LD ESD record
  // carries ESD_BSC_ImportExport (B'0100') binding scope instead of the
  // default section scope (B'0001') assigned to unexported defined symbols.
  if (!isEOL()) {
    StringRef EntryName;
    if (parseHLASMSymName(EntryName))
      return TokError("expected identifier in END directive");

    // Mark the entry point as global so its LD ESD record carries
    // ESD_BSC_ImportExport binding scope.
    MCSymbol *EntrySym = getContext().getOrCreateSymbol(
        getContext().getCanonicalHLASMName(EntryName));
    getStreamer().emitSymbolAttribute(EntrySym, MCSA_Global);
  }
  if (!isEOL())
    return TokError("unexpected token in END directive");
  return false;
}

MCAsmParserExtension *llvm::createGOFFAsmParser() { return new GOFFAsmParser; }
