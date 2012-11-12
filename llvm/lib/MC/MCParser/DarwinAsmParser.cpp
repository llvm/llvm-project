//===- DarwinAsmParser.cpp - Darwin (Mach-O) Assembly Parser --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCParser/MCAsmParserExtension.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
using namespace llvm;

namespace {

/// \brief Implementation of directive handling which is shared across all
/// Darwin targets.
class DarwinAsmParser : public MCAsmParserExtension {
  template<bool (DarwinAsmParser::*Handler)(StringRef, SMLoc)>
  void AddDirectiveHandler(StringRef Directive) {
    getParser().AddDirectiveHandler(this, Directive,
                                    HandleDirective<DarwinAsmParser, Handler>);
  }

  bool ParseSectionSwitch(const char *Segment, const char *Section,
                          unsigned TAA = 0, unsigned ImplicitAlign = 0,
                          unsigned StubSize = 0);

public:
  DarwinAsmParser() {}

  virtual void Initialize(MCAsmParser &Parser) {
    // Call the base implementation.
    this->MCAsmParserExtension::Initialize(Parser);

    AddDirectiveHandler<&DarwinAsmParser::ParseDirectiveDesc>(".desc");
    AddDirectiveHandler<&DarwinAsmParser::ParseDirectiveLsym>(".lsym");
    AddDirectiveHandler<&DarwinAsmParser::ParseDirectiveSubsectionsViaSymbols>(
      ".subsections_via_symbols");
    AddDirectiveHandler<&DarwinAsmParser::ParseDirectiveDumpOrLoad>(".dump");
    AddDirectiveHandler<&DarwinAsmParser::ParseDirectiveDumpOrLoad>(".load");
    AddDirectiveHandler<&DarwinAsmParser::ParseDirectiveSection>(".section");
    AddDirectiveHandler<&DarwinAsmParser::ParseDirectivePushSection>(".pushsection");
    AddDirectiveHandler<&DarwinAsmParser::ParseDirectivePopSection>(".popsection");
    AddDirectiveHandler<&DarwinAsmParser::ParseDirectivePrevious>(".previous");
    AddDirectiveHandler<&DarwinAsmParser::ParseDirectiveSecureLogUnique>(
      ".secure_log_unique");
    AddDirectiveHandler<&DarwinAsmParser::ParseDirectiveSecureLogReset>(
      ".secure_log_reset");
    AddDirectiveHandler<&DarwinAsmParser::ParseDirectiveTBSS>(".tbss");
    AddDirectiveHandler<&DarwinAsmParser::ParseDirectiveZerofill>(".zerofill");

    AddDirectiveHandler<&DarwinAsmParser::ParseDirectiveDataRegion>(".data_region");
    AddDirectiveHandler<&DarwinAsmParser::ParseDirectiveDataRegionEnd>(".end_data_region");

    // Special section directives.
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveConst>(".const");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveConstData>(".const_data");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveConstructor>(".constructor");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveCString>(".cstring");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveData>(".data");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveDestructor>(".destructor");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveDyld>(".dyld");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveFVMLibInit0>(".fvmlib_init0");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveFVMLibInit1>(".fvmlib_init1");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveLazySymbolPointers>(".lazy_symbol_pointer");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveLiteral16>(".literal16");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveLiteral4>(".literal4");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveLiteral8>(".literal8");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveModInitFunc>(".mod_init_func");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveModTermFunc>(".mod_term_func");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveNonLazySymbolPointers>(".non_lazy_symbol_pointer");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveObjCCatClsMeth>(".objc_cat_cls_meth");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveObjCCatInstMeth>(".objc_cat_inst_meth");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveObjCCategory>(".objc_category");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveObjCClass>(".objc_class");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveObjCClassNames>(".objc_class_names");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveObjCClassVars>(".objc_class_vars");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveObjCClsMeth>(".objc_cls_meth");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveObjCClsRefs>(".objc_cls_refs");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveObjCInstMeth>(".objc_inst_meth");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveObjCInstanceVars>(".objc_instance_vars");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveObjCMessageRefs>(".objc_message_refs");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveObjCMetaClass>(".objc_meta_class");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveObjCMethVarNames>(".objc_meth_var_names");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveObjCMethVarTypes>(".objc_meth_var_types");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveObjCModuleInfo>(".objc_module_info");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveObjCProtocol>(".objc_protocol");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveObjCSelectorStrs>(".objc_selector_strs");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveObjCStringObject>(".objc_string_object");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveObjCSymbols>(".objc_symbols");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectivePICSymbolStub>(".picsymbol_stub");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveStaticConst>(".static_const");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveStaticData>(".static_data");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveSymbolStub>(".symbol_stub");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveTData>(".tdata");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveText>(".text");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveThreadInitFunc>(".thread_init_func");
    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveTLV>(".tlv");

    AddDirectiveHandler<&DarwinAsmParser::ParseSectionDirectiveIdent>(".ident");
  }

  bool ParseDirectiveDesc(StringRef, SMLoc);
  bool ParseDirectiveDumpOrLoad(StringRef, SMLoc);
  bool ParseDirectiveLsym(StringRef, SMLoc);
  bool ParseDirectiveSection(StringRef, SMLoc);
  bool ParseDirectivePushSection(StringRef, SMLoc);
  bool ParseDirectivePopSection(StringRef, SMLoc);
  bool ParseDirectivePrevious(StringRef, SMLoc);
  bool ParseDirectiveSecureLogReset(StringRef, SMLoc);
  bool ParseDirectiveSecureLogUnique(StringRef, SMLoc);
  bool ParseDirectiveSubsectionsViaSymbols(StringRef, SMLoc);
  bool ParseDirectiveTBSS(StringRef, SMLoc);
  bool ParseDirectiveZerofill(StringRef, SMLoc);
  bool ParseDirectiveDataRegion(StringRef, SMLoc);
  bool ParseDirectiveDataRegionEnd(StringRef, SMLoc);

  // Named Section Directive
  bool ParseSectionDirectiveConst(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT", "__const");
  }
  bool ParseSectionDirectiveStaticConst(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT", "__static_const");
  }
  bool ParseSectionDirectiveCString(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT","__cstring",
                              MCSectionMachO::S_CSTRING_LITERALS);
  }
  bool ParseSectionDirectiveLiteral4(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT", "__literal4",
                              MCSectionMachO::S_4BYTE_LITERALS, 4);
  }
  bool ParseSectionDirectiveLiteral8(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT", "__literal8",
                              MCSectionMachO::S_8BYTE_LITERALS, 8);
  }
  bool ParseSectionDirectiveLiteral16(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT","__literal16",
                              MCSectionMachO::S_16BYTE_LITERALS, 16);
  }
  bool ParseSectionDirectiveConstructor(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT","__constructor");
  }
  bool ParseSectionDirectiveDestructor(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT","__destructor");
  }
  bool ParseSectionDirectiveFVMLibInit0(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT","__fvmlib_init0");
  }
  bool ParseSectionDirectiveFVMLibInit1(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT","__fvmlib_init1");
  }
  bool ParseSectionDirectiveSymbolStub(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT","__symbol_stub",
                              MCSectionMachO::S_SYMBOL_STUBS |
                              MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS,
                              // FIXME: Different on PPC and ARM.
                              0, 16);
  }
  bool ParseSectionDirectivePICSymbolStub(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT","__picsymbol_stub",
                              MCSectionMachO::S_SYMBOL_STUBS |
                              MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS, 0, 26);
  }
  bool ParseSectionDirectiveData(StringRef, SMLoc) {
    return ParseSectionSwitch("__DATA", "__data");
  }
  bool ParseSectionDirectiveStaticData(StringRef, SMLoc) {
    return ParseSectionSwitch("__DATA", "__static_data");
  }
  bool ParseSectionDirectiveNonLazySymbolPointers(StringRef, SMLoc) {
    return ParseSectionSwitch("__DATA", "__nl_symbol_ptr",
                              MCSectionMachO::S_NON_LAZY_SYMBOL_POINTERS, 4);
  }
  bool ParseSectionDirectiveLazySymbolPointers(StringRef, SMLoc) {
    return ParseSectionSwitch("__DATA", "__la_symbol_ptr",
                              MCSectionMachO::S_LAZY_SYMBOL_POINTERS, 4);
  }
  bool ParseSectionDirectiveDyld(StringRef, SMLoc) {
    return ParseSectionSwitch("__DATA", "__dyld");
  }
  bool ParseSectionDirectiveModInitFunc(StringRef, SMLoc) {
    return ParseSectionSwitch("__DATA", "__mod_init_func",
                              MCSectionMachO::S_MOD_INIT_FUNC_POINTERS, 4);
  }
  bool ParseSectionDirectiveModTermFunc(StringRef, SMLoc) {
    return ParseSectionSwitch("__DATA", "__mod_term_func",
                              MCSectionMachO::S_MOD_TERM_FUNC_POINTERS, 4);
  }
  bool ParseSectionDirectiveConstData(StringRef, SMLoc) {
    return ParseSectionSwitch("__DATA", "__const");
  }
  bool ParseSectionDirectiveObjCClass(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__class",
                              MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
  }
  bool ParseSectionDirectiveObjCMetaClass(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__meta_class",
                              MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
  }
  bool ParseSectionDirectiveObjCCatClsMeth(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__cat_cls_meth",
                              MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
  }
  bool ParseSectionDirectiveObjCCatInstMeth(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__cat_inst_meth",
                              MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
  }
  bool ParseSectionDirectiveObjCProtocol(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__protocol",
                              MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
  }
  bool ParseSectionDirectiveObjCStringObject(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__string_object",
                              MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
  }
  bool ParseSectionDirectiveObjCClsMeth(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__cls_meth",
                              MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
  }
  bool ParseSectionDirectiveObjCInstMeth(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__inst_meth",
                              MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
  }
  bool ParseSectionDirectiveObjCClsRefs(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__cls_refs",
                              MCSectionMachO::S_ATTR_NO_DEAD_STRIP |
                              MCSectionMachO::S_LITERAL_POINTERS, 4);
  }
  bool ParseSectionDirectiveObjCMessageRefs(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__message_refs",
                              MCSectionMachO::S_ATTR_NO_DEAD_STRIP |
                              MCSectionMachO::S_LITERAL_POINTERS, 4);
  }
  bool ParseSectionDirectiveObjCSymbols(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__symbols",
                              MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
  }
  bool ParseSectionDirectiveObjCCategory(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__category",
                              MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
  }
  bool ParseSectionDirectiveObjCClassVars(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__class_vars",
                              MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
  }
  bool ParseSectionDirectiveObjCInstanceVars(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__instance_vars",
                              MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
  }
  bool ParseSectionDirectiveObjCModuleInfo(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__module_info",
                              MCSectionMachO::S_ATTR_NO_DEAD_STRIP);
  }
  bool ParseSectionDirectiveObjCClassNames(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT", "__cstring",
                              MCSectionMachO::S_CSTRING_LITERALS);
  }
  bool ParseSectionDirectiveObjCMethVarTypes(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT", "__cstring",
                              MCSectionMachO::S_CSTRING_LITERALS);
  }
  bool ParseSectionDirectiveObjCMethVarNames(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT", "__cstring",
                              MCSectionMachO::S_CSTRING_LITERALS);
  }
  bool ParseSectionDirectiveObjCSelectorStrs(StringRef, SMLoc) {
    return ParseSectionSwitch("__OBJC", "__selector_strs",
                              MCSectionMachO::S_CSTRING_LITERALS);
  }
  bool ParseSectionDirectiveTData(StringRef, SMLoc) {
    return ParseSectionSwitch("__DATA", "__thread_data",
                              MCSectionMachO::S_THREAD_LOCAL_REGULAR);
  }
  bool ParseSectionDirectiveText(StringRef, SMLoc) {
    return ParseSectionSwitch("__TEXT", "__text",
                              MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS);
  }
  bool ParseSectionDirectiveTLV(StringRef, SMLoc) {
    return ParseSectionSwitch("__DATA", "__thread_vars",
                              MCSectionMachO::S_THREAD_LOCAL_VARIABLES);
  }
  bool ParseSectionDirectiveIdent(StringRef, SMLoc) {
    // Darwin silently ignores the .ident directive.
    getParser().EatToEndOfStatement();
    return false;
  }
  bool ParseSectionDirectiveThreadInitFunc(StringRef, SMLoc) {
    return ParseSectionSwitch("__DATA", "__thread_init",
                         MCSectionMachO::S_THREAD_LOCAL_INIT_FUNCTION_POINTERS);
  }

};

} // end anonymous namespace

bool DarwinAsmParser::ParseSectionSwitch(const char *Segment,
                                         const char *Section,
                                         unsigned TAA, unsigned Align,
                                         unsigned StubSize) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in section switching directive");
  Lex();

  // FIXME: Arch specific.
  bool isText = StringRef(Segment) == "__TEXT";  // FIXME: Hack.
  getStreamer().SwitchSection(getContext().getMachOSection(
                                Segment, Section, TAA, StubSize,
                                isText ? SectionKind::getText()
                                       : SectionKind::getDataRel()));

  // Set the implicit alignment, if any.
  //
  // FIXME: This isn't really what 'as' does; I think it just uses the implicit
  // alignment on the section (e.g., if one manually inserts bytes into the
  // section, then just issuing the section switch directive will not realign
  // the section. However, this is arguably more reasonable behavior, and there
  // is no good reason for someone to intentionally emit incorrectly sized
  // values into the implicitly aligned sections.
  if (Align)
    getStreamer().EmitValueToAlignment(Align, 0, 1, 0);

  return false;
}

/// ParseDirectiveDesc
///  ::= .desc identifier , expression
bool DarwinAsmParser::ParseDirectiveDesc(StringRef, SMLoc) {
  StringRef Name;
  if (getParser().ParseIdentifier(Name))
    return TokError("expected identifier in directive");

  // Handle the identifier as the key symbol.
  MCSymbol *Sym = getContext().GetOrCreateSymbol(Name);

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in '.desc' directive");
  Lex();

  int64_t DescValue;
  if (getParser().ParseAbsoluteExpression(DescValue))
    return true;

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.desc' directive");

  Lex();

  // Set the n_desc field of this Symbol to this DescValue
  getStreamer().EmitSymbolDesc(Sym, DescValue);

  return false;
}

/// ParseDirectiveDumpOrLoad
///  ::= ( .dump | .load ) "filename"
bool DarwinAsmParser::ParseDirectiveDumpOrLoad(StringRef Directive,
                                               SMLoc IDLoc) {
  bool IsDump = Directive == ".dump";
  if (getLexer().isNot(AsmToken::String))
    return TokError("expected string in '.dump' or '.load' directive");

  Lex();

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.dump' or '.load' directive");

  Lex();

  // FIXME: If/when .dump and .load are implemented they will be done in the
  // the assembly parser and not have any need for an MCStreamer API.
  if (IsDump)
    return Warning(IDLoc, "ignoring directive .dump for now");
  else
    return Warning(IDLoc, "ignoring directive .load for now");
}

/// ParseDirectiveLsym
///  ::= .lsym identifier , expression
bool DarwinAsmParser::ParseDirectiveLsym(StringRef, SMLoc) {
  StringRef Name;
  if (getParser().ParseIdentifier(Name))
    return TokError("expected identifier in directive");

  // Handle the identifier as the key symbol.
  MCSymbol *Sym = getContext().GetOrCreateSymbol(Name);

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in '.lsym' directive");
  Lex();

  const MCExpr *Value;
  if (getParser().ParseExpression(Value))
    return true;

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.lsym' directive");

  Lex();

  // We don't currently support this directive.
  //
  // FIXME: Diagnostic location!
  (void) Sym;
  return TokError("directive '.lsym' is unsupported");
}

/// ParseDirectiveSection:
///   ::= .section identifier (',' identifier)*
bool DarwinAsmParser::ParseDirectiveSection(StringRef, SMLoc) {
  SMLoc Loc = getLexer().getLoc();

  StringRef SectionName;
  if (getParser().ParseIdentifier(SectionName))
    return Error(Loc, "expected identifier after '.section' directive");

  // Verify there is a following comma.
  if (!getLexer().is(AsmToken::Comma))
    return TokError("unexpected token in '.section' directive");

  std::string SectionSpec = SectionName;
  SectionSpec += ",";

  // Add all the tokens until the end of the line, ParseSectionSpecifier will
  // handle this.
  StringRef EOL = getLexer().LexUntilEndOfStatement();
  SectionSpec.append(EOL.begin(), EOL.end());

  Lex();
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.section' directive");
  Lex();


  StringRef Segment, Section;
  unsigned StubSize;
  unsigned TAA;
  bool TAAParsed;
  std::string ErrorStr =
    MCSectionMachO::ParseSectionSpecifier(SectionSpec, Segment, Section,
                                          TAA, TAAParsed, StubSize);

  if (!ErrorStr.empty())
    return Error(Loc, ErrorStr.c_str());

  // FIXME: Arch specific.
  bool isText = Segment == "__TEXT";  // FIXME: Hack.
  getStreamer().SwitchSection(getContext().getMachOSection(
                                Segment, Section, TAA, StubSize,
                                isText ? SectionKind::getText()
                                : SectionKind::getDataRel()));
  return false;
}

/// ParseDirectivePushSection:
///   ::= .pushsection identifier (',' identifier)*
bool DarwinAsmParser::ParseDirectivePushSection(StringRef S, SMLoc Loc) {
  getStreamer().PushSection();

  if (ParseDirectiveSection(S, Loc)) {
    getStreamer().PopSection();
    return true;
  }

  return false;
}

/// ParseDirectivePopSection:
///   ::= .popsection
bool DarwinAsmParser::ParseDirectivePopSection(StringRef, SMLoc) {
  if (!getStreamer().PopSection())
    return TokError(".popsection without corresponding .pushsection");
  return false;
}

/// ParseDirectivePrevious:
///   ::= .previous
bool DarwinAsmParser::ParseDirectivePrevious(StringRef DirName, SMLoc) {
  const MCSection *PreviousSection = getStreamer().getPreviousSection();
  if (PreviousSection == NULL)
      return TokError(".previous without corresponding .section");
  getStreamer().SwitchSection(PreviousSection);
  return false;
}

/// ParseDirectiveSecureLogUnique
///  ::= .secure_log_unique ... message ...
bool DarwinAsmParser::ParseDirectiveSecureLogUnique(StringRef, SMLoc IDLoc) {
  StringRef LogMessage = getParser().ParseStringToEndOfStatement();
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.secure_log_unique' directive");

  if (getContext().getSecureLogUsed() != false)
    return Error(IDLoc, ".secure_log_unique specified multiple times");

  // Get the secure log path.
  const char *SecureLogFile = getContext().getSecureLogFile();
  if (SecureLogFile == NULL)
    return Error(IDLoc, ".secure_log_unique used but AS_SECURE_LOG_FILE "
                 "environment variable unset.");

  // Open the secure log file if we haven't already.
  raw_ostream *OS = getContext().getSecureLog();
  if (OS == NULL) {
    std::string Err;
    OS = new raw_fd_ostream(SecureLogFile, Err, raw_fd_ostream::F_Append);
    if (!Err.empty()) {
       delete OS;
       return Error(IDLoc, Twine("can't open secure log file: ") +
                    SecureLogFile + " (" + Err + ")");
    }
    getContext().setSecureLog(OS);
  }

  // Write the message.
  int CurBuf = getSourceManager().FindBufferContainingLoc(IDLoc);
  *OS << getSourceManager().getBufferInfo(CurBuf).Buffer->getBufferIdentifier()
      << ":" << getSourceManager().FindLineNumber(IDLoc, CurBuf) << ":"
      << LogMessage + "\n";

  getContext().setSecureLogUsed(true);

  return false;
}

/// ParseDirectiveSecureLogReset
///  ::= .secure_log_reset
bool DarwinAsmParser::ParseDirectiveSecureLogReset(StringRef, SMLoc IDLoc) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.secure_log_reset' directive");

  Lex();

  getContext().setSecureLogUsed(false);

  return false;
}

/// ParseDirectiveSubsectionsViaSymbols
///  ::= .subsections_via_symbols
bool DarwinAsmParser::ParseDirectiveSubsectionsViaSymbols(StringRef, SMLoc) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.subsections_via_symbols' directive");

  Lex();

  getStreamer().EmitAssemblerFlag(MCAF_SubsectionsViaSymbols);

  return false;
}

/// ParseDirectiveTBSS
///  ::= .tbss identifier, size, align
bool DarwinAsmParser::ParseDirectiveTBSS(StringRef, SMLoc) {
  SMLoc IDLoc = getLexer().getLoc();
  StringRef Name;
  if (getParser().ParseIdentifier(Name))
    return TokError("expected identifier in directive");

  // Handle the identifier as the key symbol.
  MCSymbol *Sym = getContext().GetOrCreateSymbol(Name);

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lex();

  int64_t Size;
  SMLoc SizeLoc = getLexer().getLoc();
  if (getParser().ParseAbsoluteExpression(Size))
    return true;

  int64_t Pow2Alignment = 0;
  SMLoc Pow2AlignmentLoc;
  if (getLexer().is(AsmToken::Comma)) {
    Lex();
    Pow2AlignmentLoc = getLexer().getLoc();
    if (getParser().ParseAbsoluteExpression(Pow2Alignment))
      return true;
  }

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.tbss' directive");

  Lex();

  if (Size < 0)
    return Error(SizeLoc, "invalid '.tbss' directive size, can't be less than"
                 "zero");

  // FIXME: Diagnose overflow.
  if (Pow2Alignment < 0)
    return Error(Pow2AlignmentLoc, "invalid '.tbss' alignment, can't be less"
                 "than zero");

  if (!Sym->isUndefined())
    return Error(IDLoc, "invalid symbol redefinition");

  getStreamer().EmitTBSSSymbol(getContext().getMachOSection(
                                 "__DATA", "__thread_bss",
                                 MCSectionMachO::S_THREAD_LOCAL_ZEROFILL,
                                 0, SectionKind::getThreadBSS()),
                               Sym, Size, 1 << Pow2Alignment);

  return false;
}

/// ParseDirectiveZerofill
///  ::= .zerofill segname , sectname [, identifier , size_expression [
///      , align_expression ]]
bool DarwinAsmParser::ParseDirectiveZerofill(StringRef, SMLoc) {
  StringRef Segment;
  if (getParser().ParseIdentifier(Segment))
    return TokError("expected segment name after '.zerofill' directive");

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lex();

  StringRef Section;
  if (getParser().ParseIdentifier(Section))
    return TokError("expected section name after comma in '.zerofill' "
                    "directive");

  // If this is the end of the line all that was wanted was to create the
  // the section but with no symbol.
  if (getLexer().is(AsmToken::EndOfStatement)) {
    // Create the zerofill section but no symbol
    getStreamer().EmitZerofill(getContext().getMachOSection(
                                 Segment, Section, MCSectionMachO::S_ZEROFILL,
                                 0, SectionKind::getBSS()));
    return false;
  }

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lex();

  SMLoc IDLoc = getLexer().getLoc();
  StringRef IDStr;
  if (getParser().ParseIdentifier(IDStr))
    return TokError("expected identifier in directive");

  // handle the identifier as the key symbol.
  MCSymbol *Sym = getContext().GetOrCreateSymbol(IDStr);

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lex();

  int64_t Size;
  SMLoc SizeLoc = getLexer().getLoc();
  if (getParser().ParseAbsoluteExpression(Size))
    return true;

  int64_t Pow2Alignment = 0;
  SMLoc Pow2AlignmentLoc;
  if (getLexer().is(AsmToken::Comma)) {
    Lex();
    Pow2AlignmentLoc = getLexer().getLoc();
    if (getParser().ParseAbsoluteExpression(Pow2Alignment))
      return true;
  }

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.zerofill' directive");

  Lex();

  if (Size < 0)
    return Error(SizeLoc, "invalid '.zerofill' directive size, can't be less "
                 "than zero");

  // NOTE: The alignment in the directive is a power of 2 value, the assembler
  // may internally end up wanting an alignment in bytes.
  // FIXME: Diagnose overflow.
  if (Pow2Alignment < 0)
    return Error(Pow2AlignmentLoc, "invalid '.zerofill' directive alignment, "
                 "can't be less than zero");

  if (!Sym->isUndefined())
    return Error(IDLoc, "invalid symbol redefinition");

  // Create the zerofill Symbol with Size and Pow2Alignment
  //
  // FIXME: Arch specific.
  getStreamer().EmitZerofill(getContext().getMachOSection(
                               Segment, Section, MCSectionMachO::S_ZEROFILL,
                               0, SectionKind::getBSS()),
                             Sym, Size, 1 << Pow2Alignment);

  return false;
}

/// ParseDirectiveDataRegion
///  ::= .data_region [ ( jt8 | jt16 | jt32 ) ]
bool DarwinAsmParser::ParseDirectiveDataRegion(StringRef, SMLoc) {
  if (getLexer().is(AsmToken::EndOfStatement)) {
    Lex();
    getStreamer().EmitDataRegion(MCDR_DataRegion);
    return false;
  }
  StringRef RegionType;
  SMLoc Loc = getParser().getTok().getLoc();
  if (getParser().ParseIdentifier(RegionType))
    return TokError("expected region type after '.data_region' directive");
  int Kind = StringSwitch<int>(RegionType)
    .Case("jt8", MCDR_DataRegionJT8)
    .Case("jt16", MCDR_DataRegionJT16)
    .Case("jt32", MCDR_DataRegionJT32)
    .Default(-1);
  if (Kind == -1)
    return Error(Loc, "unknown region type in '.data_region' directive");
  Lex();

  getStreamer().EmitDataRegion((MCDataRegionType)Kind);
  return false;
}

/// ParseDirectiveDataRegionEnd
///  ::= .end_data_region
bool DarwinAsmParser::ParseDirectiveDataRegionEnd(StringRef, SMLoc) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.end_data_region' directive");

  Lex();
  getStreamer().EmitDataRegion(MCDR_DataRegionEnd);
  return false;
}

namespace llvm {

MCAsmParserExtension *createDarwinAsmParser() {
  return new DarwinAsmParser;
}

} // end llvm namespace
