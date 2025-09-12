//===- COFFAsmParser.cpp - COFF Assembly Parser ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/COFF.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCParser/AsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParserExtension.h"
#include "llvm/MC/MCSectionCOFF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/TargetParser/Triple.h"
#include <cassert>
#include <cstdint>
#include <limits>
#include <utility>

using namespace llvm;

namespace {

class COFFAsmParser : public MCAsmParserExtension {
  template<bool (COFFAsmParser::*HandlerMethod)(StringRef, SMLoc)>
  void addDirectiveHandler(StringRef Directive) {
    MCAsmParser::ExtensionDirectiveHandler Handler = std::make_pair(
        this, HandleDirective<COFFAsmParser, HandlerMethod>);
    getParser().addDirectiveHandler(Directive, Handler);
  }

  bool parseSectionSwitch(StringRef Section, unsigned Characteristics);

  bool parseSectionSwitch(StringRef Section, unsigned Characteristics,
                          StringRef COMDATSymName, COFF::COMDATType Type,
                          unsigned UniqueID);

  bool parseSectionName(StringRef &SectionName);
  bool parseSectionFlags(StringRef SectionName, StringRef FlagsString,
                         unsigned *Flags);
  void Initialize(MCAsmParser &Parser) override {
    // Call the base implementation.
    MCAsmParserExtension::Initialize(Parser);

    addDirectiveHandler<&COFFAsmParser::parseSectionDirectiveText>(".text");
    addDirectiveHandler<&COFFAsmParser::parseSectionDirectiveData>(".data");
    addDirectiveHandler<&COFFAsmParser::parseSectionDirectiveBSS>(".bss");
    addDirectiveHandler<&COFFAsmParser::parseDirectiveSection>(".section");
    addDirectiveHandler<&COFFAsmParser::parseDirectivePushSection>(
        ".pushsection");
    addDirectiveHandler<&COFFAsmParser::parseDirectivePopSection>(
        ".popsection");
    addDirectiveHandler<&COFFAsmParser::parseDirectiveDef>(".def");
    addDirectiveHandler<&COFFAsmParser::parseDirectiveScl>(".scl");
    addDirectiveHandler<&COFFAsmParser::parseDirectiveType>(".type");
    addDirectiveHandler<&COFFAsmParser::parseDirectiveEndef>(".endef");
    addDirectiveHandler<&COFFAsmParser::parseDirectiveSecRel32>(".secrel32");
    addDirectiveHandler<&COFFAsmParser::parseDirectiveSymIdx>(".symidx");
    addDirectiveHandler<&COFFAsmParser::parseDirectiveSafeSEH>(".safeseh");
    addDirectiveHandler<&COFFAsmParser::parseDirectiveSecIdx>(".secidx");
    addDirectiveHandler<&COFFAsmParser::parseDirectiveLinkOnce>(".linkonce");
    addDirectiveHandler<&COFFAsmParser::parseDirectiveRVA>(".rva");
    addDirectiveHandler<&COFFAsmParser::parseDirectiveSymbolAttribute>(".weak");
    addDirectiveHandler<&COFFAsmParser::parseDirectiveSymbolAttribute>(
        ".weak_anti_dep");
    addDirectiveHandler<&COFFAsmParser::parseDirectiveCGProfile>(".cg_profile");
    addDirectiveHandler<&COFFAsmParser::parseDirectiveSecNum>(".secnum");
    addDirectiveHandler<&COFFAsmParser::parseDirectiveSecOffset>(".secoffset");

    // Win64 EH directives.
    addDirectiveHandler<&COFFAsmParser::parseSEHDirectiveStartProc>(
        ".seh_proc");
    addDirectiveHandler<&COFFAsmParser::parseSEHDirectiveEndProc>(
        ".seh_endproc");
    addDirectiveHandler<&COFFAsmParser::parseSEHDirectiveEndFuncletOrFunc>(
        ".seh_endfunclet");
    addDirectiveHandler<&COFFAsmParser::parseSEHDirectiveStartChained>(
        ".seh_startchained");
    addDirectiveHandler<&COFFAsmParser::parseSEHDirectiveEndChained>(
        ".seh_endchained");
    addDirectiveHandler<&COFFAsmParser::parseSEHDirectiveHandler>(
        ".seh_handler");
    addDirectiveHandler<&COFFAsmParser::parseSEHDirectiveHandlerData>(
        ".seh_handlerdata");
    addDirectiveHandler<&COFFAsmParser::parseSEHDirectiveAllocStack>(
        ".seh_stackalloc");
    addDirectiveHandler<&COFFAsmParser::parseSEHDirectiveEndProlog>(
        ".seh_endprologue");
    addDirectiveHandler<&COFFAsmParser::ParseSEHDirectiveBeginEpilog>(
        ".seh_startepilogue");
    addDirectiveHandler<&COFFAsmParser::ParseSEHDirectiveEndEpilog>(
        ".seh_endepilogue");
    addDirectiveHandler<&COFFAsmParser::ParseSEHDirectiveUnwindV2Start>(
        ".seh_unwindv2start");
    addDirectiveHandler<&COFFAsmParser::ParseSEHDirectiveUnwindVersion>(
        ".seh_unwindversion");
  }

  bool parseSectionDirectiveText(StringRef, SMLoc) {
    return parseSectionSwitch(".text", COFF::IMAGE_SCN_CNT_CODE |
                                           COFF::IMAGE_SCN_MEM_EXECUTE |
                                           COFF::IMAGE_SCN_MEM_READ);
  }

  bool parseSectionDirectiveData(StringRef, SMLoc) {
    return parseSectionSwitch(".data", COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                                           COFF::IMAGE_SCN_MEM_READ |
                                           COFF::IMAGE_SCN_MEM_WRITE);
  }

  bool parseSectionDirectiveBSS(StringRef, SMLoc) {
    return parseSectionSwitch(".bss", COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA |
                                          COFF::IMAGE_SCN_MEM_READ |
                                          COFF::IMAGE_SCN_MEM_WRITE);
  }

  bool parseDirectiveSection(StringRef, SMLoc);
  bool parseSectionArguments(StringRef, SMLoc);
  bool parseDirectivePushSection(StringRef, SMLoc);
  bool parseDirectivePopSection(StringRef, SMLoc);
  bool parseDirectiveDef(StringRef, SMLoc);
  bool parseDirectiveScl(StringRef, SMLoc);
  bool parseDirectiveType(StringRef, SMLoc);
  bool parseDirectiveEndef(StringRef, SMLoc);
  bool parseDirectiveSecRel32(StringRef, SMLoc);
  bool parseDirectiveSecIdx(StringRef, SMLoc);
  bool parseDirectiveSafeSEH(StringRef, SMLoc);
  bool parseDirectiveSymIdx(StringRef, SMLoc);
  bool parseCOMDATType(COFF::COMDATType &Type);
  bool parseDirectiveLinkOnce(StringRef, SMLoc);
  bool parseDirectiveRVA(StringRef, SMLoc);
  bool parseDirectiveCGProfile(StringRef, SMLoc);
  bool parseDirectiveSecNum(StringRef, SMLoc);
  bool parseDirectiveSecOffset(StringRef, SMLoc);

  // Win64 EH directives.
  bool parseSEHDirectiveStartProc(StringRef, SMLoc);
  bool parseSEHDirectiveEndProc(StringRef, SMLoc);
  bool parseSEHDirectiveEndFuncletOrFunc(StringRef, SMLoc);
  bool parseSEHDirectiveStartChained(StringRef, SMLoc);
  bool parseSEHDirectiveEndChained(StringRef, SMLoc);
  bool parseSEHDirectiveHandler(StringRef, SMLoc);
  bool parseSEHDirectiveHandlerData(StringRef, SMLoc);
  bool parseSEHDirectiveAllocStack(StringRef, SMLoc);
  bool parseSEHDirectiveEndProlog(StringRef, SMLoc);
  bool ParseSEHDirectiveBeginEpilog(StringRef, SMLoc);
  bool ParseSEHDirectiveEndEpilog(StringRef, SMLoc);
  bool ParseSEHDirectiveUnwindV2Start(StringRef, SMLoc);
  bool ParseSEHDirectiveUnwindVersion(StringRef, SMLoc);

  bool parseAtUnwindOrAtExcept(bool &unwind, bool &except);
  bool parseDirectiveSymbolAttribute(StringRef Directive, SMLoc);

public:
  COFFAsmParser() = default;
};

} // end anonymous namespace.

bool COFFAsmParser::parseSectionFlags(StringRef SectionName,
                                      StringRef FlagsString, unsigned *Flags) {
  enum {
    None = 0,
    Alloc = 1 << 0,
    Code = 1 << 1,
    Load = 1 << 2,
    InitData = 1 << 3,
    Shared = 1 << 4,
    NoLoad = 1 << 5,
    NoRead = 1 << 6,
    NoWrite = 1 << 7,
    Discardable = 1 << 8,
    Info = 1 << 9,
  };

  bool ReadOnlyRemoved = false;
  unsigned SecFlags = None;

  for (char FlagChar : FlagsString) {
    switch (FlagChar) {
    case 'a':
      // Ignored.
      break;

    case 'b': // bss section
      SecFlags |= Alloc;
      if (SecFlags & InitData)
        return TokError("conflicting section flags 'b' and 'd'.");
      SecFlags &= ~Load;
      break;

    case 'd': // data section
      SecFlags |= InitData;
      if (SecFlags & Alloc)
        return TokError("conflicting section flags 'b' and 'd'.");
      SecFlags &= ~NoWrite;
      if ((SecFlags & NoLoad) == 0)
        SecFlags |= Load;
      break;

    case 'n': // section is not loaded
      SecFlags |= NoLoad;
      SecFlags &= ~Load;
      break;

    case 'D': // discardable
      SecFlags |= Discardable;
      break;

    case 'r': // read-only
      ReadOnlyRemoved = false;
      SecFlags |= NoWrite;
      if ((SecFlags & Code) == 0)
        SecFlags |= InitData;
      if ((SecFlags & NoLoad) == 0)
        SecFlags |= Load;
      break;

    case 's': // shared section
      SecFlags |= Shared | InitData;
      SecFlags &= ~NoWrite;
      if ((SecFlags & NoLoad) == 0)
        SecFlags |= Load;
      break;

    case 'w': // writable
      SecFlags &= ~NoWrite;
      ReadOnlyRemoved = true;
      break;

    case 'x': // executable section
      SecFlags |= Code;
      if ((SecFlags & NoLoad) == 0)
        SecFlags |= Load;
      if (!ReadOnlyRemoved)
        SecFlags |= NoWrite;
      break;

    case 'y': // not readable
      SecFlags |= NoRead | NoWrite;
      break;

    case 'i': // info
      SecFlags |= Info;
      break;

    default:
      return TokError("unknown flag");
    }
  }

  *Flags = 0;

  if (SecFlags == None)
    SecFlags = InitData;

  if (SecFlags & Code)
    *Flags |= COFF::IMAGE_SCN_CNT_CODE | COFF::IMAGE_SCN_MEM_EXECUTE;
  if (SecFlags & InitData)
    *Flags |= COFF::IMAGE_SCN_CNT_INITIALIZED_DATA;
  if ((SecFlags & Alloc) && (SecFlags & Load) == 0)
    *Flags |= COFF::IMAGE_SCN_CNT_UNINITIALIZED_DATA;
  if (SecFlags & NoLoad)
    *Flags |= COFF::IMAGE_SCN_LNK_REMOVE;
  if ((SecFlags & Discardable) ||
      MCSectionCOFF::isImplicitlyDiscardable(SectionName))
    *Flags |= COFF::IMAGE_SCN_MEM_DISCARDABLE;
  if ((SecFlags & NoRead) == 0)
    *Flags |= COFF::IMAGE_SCN_MEM_READ;
  if ((SecFlags & NoWrite) == 0)
    *Flags |= COFF::IMAGE_SCN_MEM_WRITE;
  if (SecFlags & Shared)
    *Flags |= COFF::IMAGE_SCN_MEM_SHARED;
  if (SecFlags & Info)
    *Flags |= COFF::IMAGE_SCN_LNK_INFO;

  return false;
}

/// ParseDirectiveSymbolAttribute
///  ::= { ".weak", ... } [ identifier ( , identifier )* ]
bool COFFAsmParser::parseDirectiveSymbolAttribute(StringRef Directive, SMLoc) {
  MCSymbolAttr Attr = StringSwitch<MCSymbolAttr>(Directive)
    .Case(".weak", MCSA_Weak)
    .Case(".weak_anti_dep", MCSA_WeakAntiDep)
    .Default(MCSA_Invalid);
  assert(Attr != MCSA_Invalid && "unexpected symbol attribute directive!");
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    while (true) {
      MCSymbol *Sym;

      if (getParser().parseSymbol(Sym))
        return TokError("expected identifier in directive");

      getStreamer().emitSymbolAttribute(Sym, Attr);

      if (getLexer().is(AsmToken::EndOfStatement))
        break;

      if (getLexer().isNot(AsmToken::Comma))
        return TokError("unexpected token in directive");
      Lex();
    }
  }

  Lex();
  return false;
}

bool COFFAsmParser::parseDirectiveCGProfile(StringRef S, SMLoc Loc) {
  return MCAsmParserExtension::parseDirectiveCGProfile(S, Loc);
}

bool COFFAsmParser::parseSectionSwitch(StringRef Section,
                                       unsigned Characteristics) {
  return parseSectionSwitch(Section, Characteristics, "", (COFF::COMDATType)0,
                            MCSection::NonUniqueID);
}

bool COFFAsmParser::parseSectionSwitch(StringRef Section,
                                       unsigned Characteristics,
                                       StringRef COMDATSymName,
                                       COFF::COMDATType Type,
                                       unsigned UniqueID) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in section switching directive");
  Lex();

  getStreamer().switchSection(getContext().getCOFFSection(
      Section, Characteristics, COMDATSymName, Type, UniqueID));

  return false;
}

bool COFFAsmParser::parseSectionName(StringRef &SectionName) {
  if (!getLexer().is(AsmToken::Identifier) && !getLexer().is(AsmToken::String))
    return true;

  SectionName = getTok().getIdentifier();
  Lex();
  return false;
}

bool COFFAsmParser::parseDirectiveSection(StringRef directive, SMLoc loc) {
  return parseSectionArguments(directive, loc);
}

// .section name [, "flags"] [, identifier [ identifier ], identifier]
// .pushsection <same as above>
//
// Supported flags:
//   a: Ignored.
//   b: BSS section (uninitialized data)
//   d: data section (initialized data)
//   n: "noload" section (removed by linker)
//   D: Discardable section
//   r: Readable section
//   s: Shared section
//   w: Writable section
//   x: Executable section
//   y: Not-readable section (clears 'r')
//
// Subsections are not supported.
bool COFFAsmParser::parseSectionArguments(StringRef, SMLoc) {
  StringRef SectionName;

  if (parseSectionName(SectionName))
    return TokError("expected identifier in directive");

  unsigned Flags = COFF::IMAGE_SCN_CNT_INITIALIZED_DATA |
                   COFF::IMAGE_SCN_MEM_READ |
                   COFF::IMAGE_SCN_MEM_WRITE;

  if (getLexer().is(AsmToken::Comma)) {
    Lex();

    if (getLexer().isNot(AsmToken::String))
      return TokError("expected string in directive");

    StringRef FlagsStr = getTok().getStringContents();
    Lex();

    if (parseSectionFlags(SectionName, FlagsStr, &Flags))
      return true;
  }

  COFF::COMDATType Type = (COFF::COMDATType)0;
  StringRef COMDATSymName;
  if (getLexer().is(AsmToken::Comma) &&
      getLexer().peekTok().getString() != "unique") {
    Type = COFF::IMAGE_COMDAT_SELECT_ANY;
    Lex();

    Flags |= COFF::IMAGE_SCN_LNK_COMDAT;

    if (!getLexer().is(AsmToken::Identifier))
      return TokError("expected comdat type such as 'discard' or 'largest' "
                      "after protection bits");

    if (parseCOMDATType(Type))
      return true;

    if (getLexer().isNot(AsmToken::Comma))
      return TokError("expected comma in directive");
    Lex();

    if (getParser().parseIdentifier(COMDATSymName))
      return TokError("expected identifier in directive");
  }

  int64_t UniqueID = MCSection::NonUniqueID;
  if (maybeParseUniqueID(UniqueID))
    return true;

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in directive");

  if (Flags & COFF::IMAGE_SCN_CNT_CODE) {
    const Triple &T = getContext().getTargetTriple();
    if (T.getArch() == Triple::arm || T.getArch() == Triple::thumb)
      Flags |= COFF::IMAGE_SCN_MEM_16BIT;
  }
  parseSectionSwitch(SectionName, Flags, COMDATSymName, Type, UniqueID);
  return false;
}

bool COFFAsmParser::parseDirectivePushSection(StringRef directive, SMLoc loc) {
  getStreamer().pushSection();

  if (parseSectionArguments(directive, loc)) {
    getStreamer().popSection();
    return true;
  }

  return false;
}

bool COFFAsmParser::parseDirectivePopSection(StringRef, SMLoc) {
  if (!getStreamer().popSection())
    return TokError(".popsection without corresponding .pushsection");
  return false;
}

bool COFFAsmParser::parseDirectiveDef(StringRef, SMLoc) {
  MCSymbol *Sym;

  if (getParser().parseSymbol(Sym))
    return TokError("expected identifier in directive");

  getStreamer().beginCOFFSymbolDef(Sym);

  Lex();
  return false;
}

bool COFFAsmParser::parseDirectiveScl(StringRef, SMLoc) {
  int64_t SymbolStorageClass;
  if (getParser().parseAbsoluteExpression(SymbolStorageClass))
    return true;

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in directive");

  Lex();
  getStreamer().emitCOFFSymbolStorageClass(SymbolStorageClass);
  return false;
}

bool COFFAsmParser::parseDirectiveType(StringRef, SMLoc) {
  int64_t Type;
  if (getParser().parseAbsoluteExpression(Type))
    return true;

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in directive");

  Lex();
  getStreamer().emitCOFFSymbolType(Type);
  return false;
}

bool COFFAsmParser::parseDirectiveEndef(StringRef, SMLoc) {
  Lex();
  getStreamer().endCOFFSymbolDef();
  return false;
}

bool COFFAsmParser::parseDirectiveSecRel32(StringRef, SMLoc) {
  MCSymbol *Symbol;
  if (getParser().parseSymbol(Symbol))
    return TokError("expected identifier in directive");

  int64_t Offset = 0;
  SMLoc OffsetLoc;
  if (getLexer().is(AsmToken::Plus)) {
    OffsetLoc = getLexer().getLoc();
    if (getParser().parseAbsoluteExpression(Offset))
      return true;
  }

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in directive");

  if (Offset < 0 || Offset > std::numeric_limits<uint32_t>::max())
    return Error(
        OffsetLoc,
        "invalid '.secrel32' directive offset, can't be less "
        "than zero or greater than std::numeric_limits<uint32_t>::max()");

  Lex();
  getStreamer().emitCOFFSecRel32(Symbol, Offset);
  return false;
}

bool COFFAsmParser::parseDirectiveRVA(StringRef, SMLoc) {
  auto parseOp = [&]() -> bool {
    MCSymbol *Symbol;
    if (getParser().parseSymbol(Symbol))
      return TokError("expected identifier in directive");

    int64_t Offset = 0;
    SMLoc OffsetLoc;
    if (getLexer().is(AsmToken::Plus) || getLexer().is(AsmToken::Minus)) {
      OffsetLoc = getLexer().getLoc();
      if (getParser().parseAbsoluteExpression(Offset))
        return true;
    }

    if (Offset < std::numeric_limits<int32_t>::min() ||
        Offset > std::numeric_limits<int32_t>::max())
      return Error(OffsetLoc, "invalid '.rva' directive offset, can't be less "
                              "than -2147483648 or greater than "
                              "2147483647");

    getStreamer().emitCOFFImgRel32(Symbol, Offset);
    return false;
  };

  if (getParser().parseMany(parseOp))
    return addErrorSuffix(" in directive");
  return false;
}

bool COFFAsmParser::parseDirectiveSafeSEH(StringRef, SMLoc) {
  MCSymbol *Symbol;
  if (getParser().parseSymbol(Symbol))
    return TokError("expected identifier in directive");

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in directive");

  Lex();
  getStreamer().emitCOFFSafeSEH(Symbol);
  return false;
}

bool COFFAsmParser::parseDirectiveSecIdx(StringRef, SMLoc) {
  MCSymbol *Symbol;
  if (getParser().parseSymbol(Symbol))
    return TokError("expected identifier in directive");

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in directive");

  Lex();
  getStreamer().emitCOFFSectionIndex(Symbol);
  return false;
}

bool COFFAsmParser::parseDirectiveSymIdx(StringRef, SMLoc) {
  MCSymbol *Symbol;
  if (getParser().parseSymbol(Symbol))
    return TokError("expected identifier in directive");

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in directive");

  Lex();
  getStreamer().emitCOFFSymbolIndex(Symbol);
  return false;
}

bool COFFAsmParser::parseDirectiveSecNum(StringRef, SMLoc) {
  MCSymbol *Symbol;
  if (getParser().parseSymbol(Symbol))
    return TokError("expected identifier in directive");

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in directive");

  Lex();
  getStreamer().emitCOFFSecNumber(Symbol);
  return false;
}

bool COFFAsmParser::parseDirectiveSecOffset(StringRef, SMLoc) {
  MCSymbol *Symbol;
  if (getParser().parseSymbol(Symbol))
    return TokError("expected identifier in directive");

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in directive");

  Lex();
  getStreamer().emitCOFFSecOffset(Symbol);
  return false;
}

/// ::= [ identifier ]
bool COFFAsmParser::parseCOMDATType(COFF::COMDATType &Type) {
  StringRef TypeId = getTok().getIdentifier();

  Type = StringSwitch<COFF::COMDATType>(TypeId)
    .Case("one_only", COFF::IMAGE_COMDAT_SELECT_NODUPLICATES)
    .Case("discard", COFF::IMAGE_COMDAT_SELECT_ANY)
    .Case("same_size", COFF::IMAGE_COMDAT_SELECT_SAME_SIZE)
    .Case("same_contents", COFF::IMAGE_COMDAT_SELECT_EXACT_MATCH)
    .Case("associative", COFF::IMAGE_COMDAT_SELECT_ASSOCIATIVE)
    .Case("largest", COFF::IMAGE_COMDAT_SELECT_LARGEST)
    .Case("newest", COFF::IMAGE_COMDAT_SELECT_NEWEST)
    .Default((COFF::COMDATType)0);

  if (Type == 0)
    return TokError(Twine("unrecognized COMDAT type '" + TypeId + "'"));

  Lex();

  return false;
}

/// ParseDirectiveLinkOnce
///  ::= .linkonce [ identifier ]
bool COFFAsmParser::parseDirectiveLinkOnce(StringRef, SMLoc Loc) {
  COFF::COMDATType Type = COFF::IMAGE_COMDAT_SELECT_ANY;
  if (getLexer().is(AsmToken::Identifier))
    if (parseCOMDATType(Type))
      return true;

  const MCSectionCOFF *Current =
      static_cast<const MCSectionCOFF *>(getStreamer().getCurrentSectionOnly());

  if (Type == COFF::IMAGE_COMDAT_SELECT_ASSOCIATIVE)
    return Error(Loc, "cannot make section associative with .linkonce");

  if (Current->getCharacteristics() & COFF::IMAGE_SCN_LNK_COMDAT)
    return Error(Loc, Twine("section '") + Current->getName() +
                          "' is already linkonce");

  Current->setSelection(Type);

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in directive");

  return false;
}

bool COFFAsmParser::parseSEHDirectiveStartProc(StringRef, SMLoc Loc) {
  MCSymbol *Symbol;
  if (getParser().parseSymbol(Symbol))
    return true;

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in directive");

  Lex();
  getStreamer().emitWinCFIStartProc(Symbol, Loc);
  return false;
}

bool COFFAsmParser::parseSEHDirectiveEndProc(StringRef, SMLoc Loc) {
  Lex();
  getStreamer().emitWinCFIEndProc(Loc);
  return false;
}

bool COFFAsmParser::parseSEHDirectiveEndFuncletOrFunc(StringRef, SMLoc Loc) {
  Lex();
  getStreamer().emitWinCFIFuncletOrFuncEnd(Loc);
  return false;
}

bool COFFAsmParser::parseSEHDirectiveStartChained(StringRef, SMLoc Loc) {
  Lex();
  getStreamer().emitWinCFIStartChained(Loc);
  return false;
}

bool COFFAsmParser::parseSEHDirectiveEndChained(StringRef, SMLoc Loc) {
  Lex();
  getStreamer().emitWinCFIEndChained(Loc);
  return false;
}

bool COFFAsmParser::parseSEHDirectiveHandler(StringRef, SMLoc Loc) {
  MCSymbol *handler;
  if (getParser().parseSymbol(handler))
    return true;

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("you must specify one or both of @unwind or @except");
  Lex();
  bool unwind = false, except = false;
  if (parseAtUnwindOrAtExcept(unwind, except))
    return true;
  if (getLexer().is(AsmToken::Comma)) {
    Lex();
    if (parseAtUnwindOrAtExcept(unwind, except))
      return true;
  }
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in directive");

  Lex();
  getStreamer().emitWinEHHandler(handler, unwind, except, Loc);
  return false;
}

bool COFFAsmParser::parseSEHDirectiveHandlerData(StringRef, SMLoc Loc) {
  Lex();
  getStreamer().emitWinEHHandlerData();
  return false;
}

bool COFFAsmParser::parseSEHDirectiveAllocStack(StringRef, SMLoc Loc) {
  int64_t Size;
  if (getParser().parseAbsoluteExpression(Size))
    return true;

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in directive");

  Lex();
  getStreamer().emitWinCFIAllocStack(Size, Loc);
  return false;
}

bool COFFAsmParser::parseSEHDirectiveEndProlog(StringRef, SMLoc Loc) {
  Lex();
  getStreamer().emitWinCFIEndProlog(Loc);
  return false;
}

bool COFFAsmParser::ParseSEHDirectiveBeginEpilog(StringRef, SMLoc Loc) {
  Lex();
  getStreamer().emitWinCFIBeginEpilogue(Loc);
  return false;
}

bool COFFAsmParser::ParseSEHDirectiveEndEpilog(StringRef, SMLoc Loc) {
  Lex();
  getStreamer().emitWinCFIEndEpilogue(Loc);
  return false;
}

bool COFFAsmParser::ParseSEHDirectiveUnwindV2Start(StringRef, SMLoc Loc) {
  Lex();
  getStreamer().emitWinCFIUnwindV2Start(Loc);
  return false;
}

bool COFFAsmParser::ParseSEHDirectiveUnwindVersion(StringRef, SMLoc Loc) {
  int64_t Version;
  if (getParser().parseIntToken(Version, "expected unwind version number"))
    return true;

  if ((Version < 1) || (Version > UINT8_MAX))
    return Error(Loc, "invalid unwind version");

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in directive");

  Lex();
  getStreamer().emitWinCFIUnwindVersion(Version, Loc);
  return false;
}

bool COFFAsmParser::parseAtUnwindOrAtExcept(bool &unwind, bool &except) {
  StringRef identifier;
  if (getLexer().isNot(AsmToken::At) && getLexer().isNot(AsmToken::Percent))
    return TokError("a handler attribute must begin with '@' or '%'");
  SMLoc startLoc = getLexer().getLoc();
  Lex();
  if (getParser().parseIdentifier(identifier))
    return Error(startLoc, "expected @unwind or @except");
  if (identifier == "unwind")
    unwind = true;
  else if (identifier == "except")
    except = true;
  else
    return Error(startLoc, "expected @unwind or @except");
  return false;
}

namespace llvm {

MCAsmParserExtension *createCOFFAsmParser() {
  return new COFFAsmParser;
}

} // end namespace llvm
