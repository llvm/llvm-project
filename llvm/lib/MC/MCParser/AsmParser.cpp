//===- AsmParser.cpp - Parser for Assembly Files --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class implements the parser for assembly files.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCParser/AsmCond.h"
#include "llvm/MC/MCParser/AsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>
#include <set>
#include <string>
#include <vector>
using namespace llvm;

static cl::opt<bool>
FatalAssemblerWarnings("fatal-assembler-warnings",
                       cl::desc("Consider warnings as error"));

MCAsmParserSemaCallback::~MCAsmParserSemaCallback() {} 

namespace {

/// \brief Helper class for tracking macro definitions.
typedef std::vector<AsmToken> MacroArgument;
typedef std::vector<MacroArgument> MacroArguments;
typedef std::pair<StringRef, MacroArgument> MacroParameter;
typedef std::vector<MacroParameter> MacroParameters;

struct Macro {
  StringRef Name;
  StringRef Body;
  MacroParameters Parameters;

public:
  Macro(StringRef N, StringRef B, const MacroParameters &P) :
    Name(N), Body(B), Parameters(P) {}
};

/// \brief Helper class for storing information about an active macro
/// instantiation.
struct MacroInstantiation {
  /// The macro being instantiated.
  const Macro *TheMacro;

  /// The macro instantiation with substitutions.
  MemoryBuffer *Instantiation;

  /// The location of the instantiation.
  SMLoc InstantiationLoc;

  /// The location where parsing should resume upon instantiation completion.
  SMLoc ExitLoc;

public:
  MacroInstantiation(const Macro *M, SMLoc IL, SMLoc EL,
                     MemoryBuffer *I);
};

//struct AsmRewrite;
struct ParseStatementInfo {
  /// ParsedOperands - The parsed operands from the last parsed statement.
  SmallVector<MCParsedAsmOperand*, 8> ParsedOperands;

  /// Opcode - The opcode from the last parsed instruction.
  unsigned Opcode;

  SmallVectorImpl<AsmRewrite> *AsmRewrites;

  ParseStatementInfo() : Opcode(~0U), AsmRewrites(0) {}
  ParseStatementInfo(SmallVectorImpl<AsmRewrite> *rewrites)
    : Opcode(~0), AsmRewrites(rewrites) {}

  ~ParseStatementInfo() {
    // Free any parsed operands.
    for (unsigned i = 0, e = ParsedOperands.size(); i != e; ++i)
      delete ParsedOperands[i];
    ParsedOperands.clear();
  }
};

/// \brief The concrete assembly parser instance.
class AsmParser : public MCAsmParser {
  friend class GenericAsmParser;

  AsmParser(const AsmParser &) LLVM_DELETED_FUNCTION;
  void operator=(const AsmParser &) LLVM_DELETED_FUNCTION;
private:
  AsmLexer Lexer;
  MCContext &Ctx;
  MCStreamer &Out;
  const MCAsmInfo &MAI;
  SourceMgr &SrcMgr;
  SourceMgr::DiagHandlerTy SavedDiagHandler;
  void *SavedDiagContext;
  MCAsmParserExtension *GenericParser;
  MCAsmParserExtension *PlatformParser;

  /// This is the current buffer index we're lexing from as managed by the
  /// SourceMgr object.
  int CurBuffer;

  AsmCond TheCondState;
  std::vector<AsmCond> TheCondStack;

  /// DirectiveMap - This is a table handlers for directives.  Each handler is
  /// invoked after the directive identifier is read and is responsible for
  /// parsing and validating the rest of the directive.  The handler is passed
  /// in the directive name and the location of the directive keyword.
  StringMap<std::pair<MCAsmParserExtension*, DirectiveHandler> > DirectiveMap;

  /// MacroMap - Map of currently defined macros.
  StringMap<Macro*> MacroMap;

  /// ActiveMacros - Stack of active macro instantiations.
  std::vector<MacroInstantiation*> ActiveMacros;

  /// Boolean tracking whether macro substitution is enabled.
  unsigned MacrosEnabled : 1;

  /// Flag tracking whether any errors have been encountered.
  unsigned HadError : 1;

  /// The values from the last parsed cpp hash file line comment if any.
  StringRef CppHashFilename;
  int64_t CppHashLineNumber;
  SMLoc CppHashLoc;
  int CppHashBuf;

  /// AssemblerDialect. ~OU means unset value and use value provided by MAI.
  unsigned AssemblerDialect;

  /// IsDarwin - is Darwin compatibility enabled?
  bool IsDarwin;

  /// ParsingInlineAsm - Are we parsing ms-style inline assembly?
  bool ParsingInlineAsm;

public:
  AsmParser(SourceMgr &SM, MCContext &Ctx, MCStreamer &Out,
            const MCAsmInfo &MAI);
  virtual ~AsmParser();

  virtual bool Run(bool NoInitialTextSection, bool NoFinalize = false);

  virtual void AddDirectiveHandler(MCAsmParserExtension *Object,
                                   StringRef Directive,
                                   DirectiveHandler Handler) {
    DirectiveMap[Directive] = std::make_pair(Object, Handler);
  }

public:
  /// @name MCAsmParser Interface
  /// {

  virtual SourceMgr &getSourceManager() { return SrcMgr; }
  virtual MCAsmLexer &getLexer() { return Lexer; }
  virtual MCContext &getContext() { return Ctx; }
  virtual MCStreamer &getStreamer() { return Out; }
  virtual unsigned getAssemblerDialect() { 
    if (AssemblerDialect == ~0U)
      return MAI.getAssemblerDialect(); 
    else
      return AssemblerDialect;
  }
  virtual void setAssemblerDialect(unsigned i) {
    AssemblerDialect = i;
  }

  virtual bool Warning(SMLoc L, const Twine &Msg,
                       ArrayRef<SMRange> Ranges = ArrayRef<SMRange>());
  virtual bool Error(SMLoc L, const Twine &Msg,
                     ArrayRef<SMRange> Ranges = ArrayRef<SMRange>());

  virtual const AsmToken &Lex();

  void setParsingInlineAsm(bool V) { ParsingInlineAsm = V; }
  bool isParsingInlineAsm() { return ParsingInlineAsm; }

  bool ParseMSInlineAsm(void *AsmLoc, std::string &AsmString,
                        unsigned &NumOutputs, unsigned &NumInputs,
                        SmallVectorImpl<std::pair<void *,bool> > &OpDecls,
                        SmallVectorImpl<std::string> &Constraints,
                        SmallVectorImpl<std::string> &Clobbers,
                        const MCInstrInfo *MII,
                        const MCInstPrinter *IP,
                        MCAsmParserSemaCallback &SI);

  bool ParseExpression(const MCExpr *&Res);
  virtual bool ParseExpression(const MCExpr *&Res, SMLoc &EndLoc);
  virtual bool ParseParenExpression(const MCExpr *&Res, SMLoc &EndLoc);
  virtual bool ParseAbsoluteExpression(int64_t &Res);

  /// }

private:
  void CheckForValidSection();

  bool ParseStatement(ParseStatementInfo &Info);
  void EatToEndOfLine();
  bool ParseCppHashLineFilenameComment(const SMLoc &L);

  bool HandleMacroEntry(StringRef Name, SMLoc NameLoc, const Macro *M);
  bool expandMacro(raw_svector_ostream &OS, StringRef Body,
                   const MacroParameters &Parameters,
                   const MacroArguments &A,
                   const SMLoc &L);
  void HandleMacroExit();

  void PrintMacroInstantiations();
  void PrintMessage(SMLoc Loc, SourceMgr::DiagKind Kind, const Twine &Msg,
                    ArrayRef<SMRange> Ranges = ArrayRef<SMRange>()) const {
    SrcMgr.PrintMessage(Loc, Kind, Msg, Ranges);
  }
  static void DiagHandler(const SMDiagnostic &Diag, void *Context);

  /// EnterIncludeFile - Enter the specified file. This returns true on failure.
  bool EnterIncludeFile(const std::string &Filename);
  /// ProcessIncbinFile - Process the specified file for the .incbin directive.
  /// This returns true on failure.
  bool ProcessIncbinFile(const std::string &Filename);

  /// \brief Reset the current lexer position to that given by \p Loc. The
  /// current token is not set; clients should ensure Lex() is called
  /// subsequently.
  void JumpToLoc(SMLoc Loc);

  virtual void EatToEndOfStatement();

  bool ParseMacroArgument(MacroArgument &MA,
                          AsmToken::TokenKind &ArgumentDelimiter);
  bool ParseMacroArguments(const Macro *M, MacroArguments &A);

  /// \brief Parse up to the end of statement and a return the contents from the
  /// current token until the end of the statement; the current token on exit
  /// will be either the EndOfStatement or EOF.
  virtual StringRef ParseStringToEndOfStatement();

  /// \brief Parse until the end of a statement or a comma is encountered,
  /// return the contents from the current token up to the end or comma.
  StringRef ParseStringToComma();

  bool ParseAssignment(StringRef Name, bool allow_redef,
                       bool NoDeadStrip = false);

  bool ParsePrimaryExpr(const MCExpr *&Res, SMLoc &EndLoc);
  bool ParseBinOpRHS(unsigned Precedence, const MCExpr *&Res, SMLoc &EndLoc);
  bool ParseParenExpr(const MCExpr *&Res, SMLoc &EndLoc);
  bool ParseBracketExpr(const MCExpr *&Res, SMLoc &EndLoc);

  /// ParseIdentifier - Parse an identifier or string (as a quoted identifier)
  /// and set \p Res to the identifier contents.
  virtual bool ParseIdentifier(StringRef &Res);

  // Directive Parsing.

 // ".ascii", ".asciiz", ".string"
  bool ParseDirectiveAscii(StringRef IDVal, bool ZeroTerminated);
  bool ParseDirectiveValue(unsigned Size); // ".byte", ".long", ...
  bool ParseDirectiveRealValue(const fltSemantics &); // ".single", ...
  bool ParseDirectiveFill(); // ".fill"
  bool ParseDirectiveSpace(); // ".space"
  bool ParseDirectiveZero(); // ".zero"
  bool ParseDirectiveSet(StringRef IDVal, bool allow_redef); // ".set", ".equ", ".equiv"
  bool ParseDirectiveOrg(); // ".org"
  // ".align{,32}", ".p2align{,w,l}"
  bool ParseDirectiveAlign(bool IsPow2, unsigned ValueSize);

  /// ParseDirectiveSymbolAttribute - Parse a directive like ".globl" which
  /// accepts a single symbol (which should be a label or an external).
  bool ParseDirectiveSymbolAttribute(MCSymbolAttr Attr);

  bool ParseDirectiveComm(bool IsLocal); // ".comm" and ".lcomm"

  bool ParseDirectiveAbort(); // ".abort"
  bool ParseDirectiveInclude(); // ".include"
  bool ParseDirectiveIncbin(); // ".incbin"

  bool ParseDirectiveIf(SMLoc DirectiveLoc); // ".if"
  // ".ifb" or ".ifnb", depending on ExpectBlank.
  bool ParseDirectiveIfb(SMLoc DirectiveLoc, bool ExpectBlank);
  // ".ifc" or ".ifnc", depending on ExpectEqual.
  bool ParseDirectiveIfc(SMLoc DirectiveLoc, bool ExpectEqual);
  // ".ifdef" or ".ifndef", depending on expect_defined
  bool ParseDirectiveIfdef(SMLoc DirectiveLoc, bool expect_defined);
  bool ParseDirectiveElseIf(SMLoc DirectiveLoc); // ".elseif"
  bool ParseDirectiveElse(SMLoc DirectiveLoc); // ".else"
  bool ParseDirectiveEndIf(SMLoc DirectiveLoc); // .endif

  /// ParseEscapedString - Parse the current token as a string which may include
  /// escaped characters and return the string contents.
  bool ParseEscapedString(std::string &Data);

  const MCExpr *ApplyModifierToExpr(const MCExpr *E,
                                    MCSymbolRefExpr::VariantKind Variant);

  // Macro-like directives
  Macro *ParseMacroLikeBody(SMLoc DirectiveLoc);
  void InstantiateMacroLikeBody(Macro *M, SMLoc DirectiveLoc,
                                raw_svector_ostream &OS);
  bool ParseDirectiveRept(SMLoc DirectiveLoc); // ".rept"
  bool ParseDirectiveIrp(SMLoc DirectiveLoc);  // ".irp"
  bool ParseDirectiveIrpc(SMLoc DirectiveLoc); // ".irpc"
  bool ParseDirectiveEndr(SMLoc DirectiveLoc); // ".endr"

  // "_emit"
  bool ParseDirectiveEmit(SMLoc DirectiveLoc, ParseStatementInfo &Info);
};

/// \brief Generic implementations of directive handling, etc. which is shared
/// (or the default, at least) for all assembler parser.
class GenericAsmParser : public MCAsmParserExtension {
  template<bool (GenericAsmParser::*Handler)(StringRef, SMLoc)>
  void AddDirectiveHandler(StringRef Directive) {
    getParser().AddDirectiveHandler(this, Directive,
                                    HandleDirective<GenericAsmParser, Handler>);
  }
public:
  GenericAsmParser() {}

  AsmParser &getParser() {
    return (AsmParser&) this->MCAsmParserExtension::getParser();
  }

  virtual void Initialize(MCAsmParser &Parser) {
    // Call the base implementation.
    this->MCAsmParserExtension::Initialize(Parser);

    // Debugging directives.
    AddDirectiveHandler<&GenericAsmParser::ParseDirectiveFile>(".file");
    AddDirectiveHandler<&GenericAsmParser::ParseDirectiveLine>(".line");
    AddDirectiveHandler<&GenericAsmParser::ParseDirectiveLoc>(".loc");
    AddDirectiveHandler<&GenericAsmParser::ParseDirectiveStabs>(".stabs");

    // CFI directives.
    AddDirectiveHandler<&GenericAsmParser::ParseDirectiveCFISections>(
                                                               ".cfi_sections");
    AddDirectiveHandler<&GenericAsmParser::ParseDirectiveCFIStartProc>(
                                                              ".cfi_startproc");
    AddDirectiveHandler<&GenericAsmParser::ParseDirectiveCFIEndProc>(
                                                                ".cfi_endproc");
    AddDirectiveHandler<&GenericAsmParser::ParseDirectiveCFIDefCfa>(
                                                         ".cfi_def_cfa");
    AddDirectiveHandler<&GenericAsmParser::ParseDirectiveCFIDefCfaOffset>(
                                                         ".cfi_def_cfa_offset");
    AddDirectiveHandler<&GenericAsmParser::ParseDirectiveCFIAdjustCfaOffset>(
                                                      ".cfi_adjust_cfa_offset");
    AddDirectiveHandler<&GenericAsmParser::ParseDirectiveCFIDefCfaRegister>(
                                                       ".cfi_def_cfa_register");
    AddDirectiveHandler<&GenericAsmParser::ParseDirectiveCFIOffset>(
                                                                 ".cfi_offset");
    AddDirectiveHandler<&GenericAsmParser::ParseDirectiveCFIRelOffset>(
                                                             ".cfi_rel_offset");
    AddDirectiveHandler<
     &GenericAsmParser::ParseDirectiveCFIPersonalityOrLsda>(".cfi_personality");
    AddDirectiveHandler<
            &GenericAsmParser::ParseDirectiveCFIPersonalityOrLsda>(".cfi_lsda");
    AddDirectiveHandler<
      &GenericAsmParser::ParseDirectiveCFIRememberState>(".cfi_remember_state");
    AddDirectiveHandler<
      &GenericAsmParser::ParseDirectiveCFIRestoreState>(".cfi_restore_state");
    AddDirectiveHandler<
      &GenericAsmParser::ParseDirectiveCFISameValue>(".cfi_same_value");
    AddDirectiveHandler<
      &GenericAsmParser::ParseDirectiveCFIRestore>(".cfi_restore");
    AddDirectiveHandler<
      &GenericAsmParser::ParseDirectiveCFIEscape>(".cfi_escape");
    AddDirectiveHandler<
      &GenericAsmParser::ParseDirectiveCFISignalFrame>(".cfi_signal_frame");

    // Macro directives.
    AddDirectiveHandler<&GenericAsmParser::ParseDirectiveMacrosOnOff>(
      ".macros_on");
    AddDirectiveHandler<&GenericAsmParser::ParseDirectiveMacrosOnOff>(
      ".macros_off");
    AddDirectiveHandler<&GenericAsmParser::ParseDirectiveMacro>(".macro");
    AddDirectiveHandler<&GenericAsmParser::ParseDirectiveEndMacro>(".endm");
    AddDirectiveHandler<&GenericAsmParser::ParseDirectiveEndMacro>(".endmacro");
    AddDirectiveHandler<&GenericAsmParser::ParseDirectivePurgeMacro>(".purgem");

    AddDirectiveHandler<&GenericAsmParser::ParseDirectiveLEB128>(".sleb128");
    AddDirectiveHandler<&GenericAsmParser::ParseDirectiveLEB128>(".uleb128");
  }

  bool ParseRegisterOrRegisterNumber(int64_t &Register, SMLoc DirectiveLoc);

  bool ParseDirectiveFile(StringRef, SMLoc DirectiveLoc);
  bool ParseDirectiveLine(StringRef, SMLoc DirectiveLoc);
  bool ParseDirectiveLoc(StringRef, SMLoc DirectiveLoc);
  bool ParseDirectiveStabs(StringRef, SMLoc DirectiveLoc);
  bool ParseDirectiveCFISections(StringRef, SMLoc DirectiveLoc);
  bool ParseDirectiveCFIStartProc(StringRef, SMLoc DirectiveLoc);
  bool ParseDirectiveCFIEndProc(StringRef, SMLoc DirectiveLoc);
  bool ParseDirectiveCFIDefCfa(StringRef, SMLoc DirectiveLoc);
  bool ParseDirectiveCFIDefCfaOffset(StringRef, SMLoc DirectiveLoc);
  bool ParseDirectiveCFIAdjustCfaOffset(StringRef, SMLoc DirectiveLoc);
  bool ParseDirectiveCFIDefCfaRegister(StringRef, SMLoc DirectiveLoc);
  bool ParseDirectiveCFIOffset(StringRef, SMLoc DirectiveLoc);
  bool ParseDirectiveCFIRelOffset(StringRef, SMLoc DirectiveLoc);
  bool ParseDirectiveCFIPersonalityOrLsda(StringRef, SMLoc DirectiveLoc);
  bool ParseDirectiveCFIRememberState(StringRef, SMLoc DirectiveLoc);
  bool ParseDirectiveCFIRestoreState(StringRef, SMLoc DirectiveLoc);
  bool ParseDirectiveCFISameValue(StringRef, SMLoc DirectiveLoc);
  bool ParseDirectiveCFIRestore(StringRef, SMLoc DirectiveLoc);
  bool ParseDirectiveCFIEscape(StringRef, SMLoc DirectiveLoc);
  bool ParseDirectiveCFISignalFrame(StringRef, SMLoc DirectiveLoc);

  bool ParseDirectiveMacrosOnOff(StringRef, SMLoc DirectiveLoc);
  bool ParseDirectiveMacro(StringRef, SMLoc DirectiveLoc);
  bool ParseDirectiveEndMacro(StringRef, SMLoc DirectiveLoc);
  bool ParseDirectivePurgeMacro(StringRef, SMLoc DirectiveLoc);

  bool ParseDirectiveLEB128(StringRef, SMLoc);
};

}

namespace llvm {

extern MCAsmParserExtension *createDarwinAsmParser();
extern MCAsmParserExtension *createELFAsmParser();
extern MCAsmParserExtension *createCOFFAsmParser();

}

enum { DEFAULT_ADDRSPACE = 0 };

AsmParser::AsmParser(SourceMgr &_SM, MCContext &_Ctx,
                     MCStreamer &_Out, const MCAsmInfo &_MAI)
  : Lexer(_MAI), Ctx(_Ctx), Out(_Out), MAI(_MAI), SrcMgr(_SM),
    GenericParser(new GenericAsmParser), PlatformParser(0),
    CurBuffer(0), MacrosEnabled(true), CppHashLineNumber(0),
    AssemblerDialect(~0U), IsDarwin(false), ParsingInlineAsm(false) {
  // Save the old handler.
  SavedDiagHandler = SrcMgr.getDiagHandler();
  SavedDiagContext = SrcMgr.getDiagContext();
  // Set our own handler which calls the saved handler.
  SrcMgr.setDiagHandler(DiagHandler, this);
  Lexer.setBuffer(SrcMgr.getMemoryBuffer(CurBuffer));

  // Initialize the generic parser.
  GenericParser->Initialize(*this);

  // Initialize the platform / file format parser.
  //
  // FIXME: This is a hack, we need to (majorly) cleanup how these objects are
  // created.
  if (_MAI.hasMicrosoftFastStdCallMangling()) {
    PlatformParser = createCOFFAsmParser();
    PlatformParser->Initialize(*this);
  } else if (_MAI.hasSubsectionsViaSymbols()) {
    PlatformParser = createDarwinAsmParser();
    PlatformParser->Initialize(*this);
    IsDarwin = true;
  } else {
    PlatformParser = createELFAsmParser();
    PlatformParser->Initialize(*this);
  }
}

AsmParser::~AsmParser() {
  assert(ActiveMacros.empty() && "Unexpected active macro instantiation!");

  // Destroy any macros.
  for (StringMap<Macro*>::iterator it = MacroMap.begin(),
         ie = MacroMap.end(); it != ie; ++it)
    delete it->getValue();

  delete PlatformParser;
  delete GenericParser;
}

void AsmParser::PrintMacroInstantiations() {
  // Print the active macro instantiation stack.
  for (std::vector<MacroInstantiation*>::const_reverse_iterator
         it = ActiveMacros.rbegin(), ie = ActiveMacros.rend(); it != ie; ++it)
    PrintMessage((*it)->InstantiationLoc, SourceMgr::DK_Note,
                 "while in macro instantiation");
}

bool AsmParser::Warning(SMLoc L, const Twine &Msg, ArrayRef<SMRange> Ranges) {
  if (FatalAssemblerWarnings)
    return Error(L, Msg, Ranges);
  PrintMessage(L, SourceMgr::DK_Warning, Msg, Ranges);
  PrintMacroInstantiations();
  return false;
}

bool AsmParser::Error(SMLoc L, const Twine &Msg, ArrayRef<SMRange> Ranges) {
  HadError = true;
  PrintMessage(L, SourceMgr::DK_Error, Msg, Ranges);
  PrintMacroInstantiations();
  return true;
}

bool AsmParser::EnterIncludeFile(const std::string &Filename) {
  std::string IncludedFile;
  int NewBuf = SrcMgr.AddIncludeFile(Filename, Lexer.getLoc(), IncludedFile);
  if (NewBuf == -1)
    return true;

  CurBuffer = NewBuf;

  Lexer.setBuffer(SrcMgr.getMemoryBuffer(CurBuffer));

  return false;
}

/// Process the specified .incbin file by seaching for it in the include paths
/// then just emitting the byte contents of the file to the streamer. This
/// returns true on failure.
bool AsmParser::ProcessIncbinFile(const std::string &Filename) {
  std::string IncludedFile;
  int NewBuf = SrcMgr.AddIncludeFile(Filename, Lexer.getLoc(), IncludedFile);
  if (NewBuf == -1)
    return true;

  // Pick up the bytes from the file and emit them.
  getStreamer().EmitBytes(SrcMgr.getMemoryBuffer(NewBuf)->getBuffer(),
                          DEFAULT_ADDRSPACE);
  return false;
}

void AsmParser::JumpToLoc(SMLoc Loc) {
  CurBuffer = SrcMgr.FindBufferContainingLoc(Loc);
  Lexer.setBuffer(SrcMgr.getMemoryBuffer(CurBuffer), Loc.getPointer());
}

const AsmToken &AsmParser::Lex() {
  const AsmToken *tok = &Lexer.Lex();

  if (tok->is(AsmToken::Eof)) {
    // If this is the end of an included file, pop the parent file off the
    // include stack.
    SMLoc ParentIncludeLoc = SrcMgr.getParentIncludeLoc(CurBuffer);
    if (ParentIncludeLoc != SMLoc()) {
      JumpToLoc(ParentIncludeLoc);
      tok = &Lexer.Lex();
    }
  }

  if (tok->is(AsmToken::Error))
    Error(Lexer.getErrLoc(), Lexer.getErr());

  return *tok;
}

bool AsmParser::Run(bool NoInitialTextSection, bool NoFinalize) {
  // Create the initial section, if requested.
  if (!NoInitialTextSection)
    Out.InitSections();

  // Prime the lexer.
  Lex();

  HadError = false;
  AsmCond StartingCondState = TheCondState;

  // If we are generating dwarf for assembly source files save the initial text
  // section and generate a .file directive.
  if (getContext().getGenDwarfForAssembly()) {
    getContext().setGenDwarfSection(getStreamer().getCurrentSection());
    MCSymbol *SectionStartSym = getContext().CreateTempSymbol();
    getStreamer().EmitLabel(SectionStartSym);
    getContext().setGenDwarfSectionStartSym(SectionStartSym);
    getStreamer().EmitDwarfFileDirective(getContext().nextGenDwarfFileNumber(),
      StringRef(), SrcMgr.getMemoryBuffer(CurBuffer)->getBufferIdentifier());
  }

  // While we have input, parse each statement.
  while (Lexer.isNot(AsmToken::Eof)) {
    ParseStatementInfo Info;
    if (!ParseStatement(Info)) continue;

    // We had an error, validate that one was emitted and recover by skipping to
    // the next line.
    assert(HadError && "Parse statement returned an error, but none emitted!");
    EatToEndOfStatement();
  }

  if (TheCondState.TheCond != StartingCondState.TheCond ||
      TheCondState.Ignore != StartingCondState.Ignore)
    return TokError("unmatched .ifs or .elses");

  // Check to see there are no empty DwarfFile slots.
  const std::vector<MCDwarfFile *> &MCDwarfFiles =
    getContext().getMCDwarfFiles();
  for (unsigned i = 1; i < MCDwarfFiles.size(); i++) {
    if (!MCDwarfFiles[i])
      TokError("unassigned file number: " + Twine(i) + " for .file directives");
  }

  // Check to see that all assembler local symbols were actually defined.
  // Targets that don't do subsections via symbols may not want this, though,
  // so conservatively exclude them. Only do this if we're finalizing, though,
  // as otherwise we won't necessarilly have seen everything yet.
  if (!NoFinalize && MAI.hasSubsectionsViaSymbols()) {
    const MCContext::SymbolTable &Symbols = getContext().getSymbols();
    for (MCContext::SymbolTable::const_iterator i = Symbols.begin(),
         e = Symbols.end();
         i != e; ++i) {
      MCSymbol *Sym = i->getValue();
      // Variable symbols may not be marked as defined, so check those
      // explicitly. If we know it's a variable, we have a definition for
      // the purposes of this check.
      if (Sym->isTemporary() && !Sym->isVariable() && !Sym->isDefined())
        // FIXME: We would really like to refer back to where the symbol was
        // first referenced for a source location. We need to add something
        // to track that. Currently, we just point to the end of the file.
        PrintMessage(getLexer().getLoc(), SourceMgr::DK_Error,
                     "assembler local symbol '" + Sym->getName() +
                     "' not defined");
    }
  }


  // Finalize the output stream if there are no errors and if the client wants
  // us to.
  if (!HadError && !NoFinalize)
    Out.Finish();

  return HadError;
}

void AsmParser::CheckForValidSection() {
  if (!ParsingInlineAsm && !getStreamer().getCurrentSection()) {
    TokError("expected section directive before assembly directive");
    Out.SwitchSection(Ctx.getMachOSection(
                        "__TEXT", "__text",
                        MCSectionMachO::S_ATTR_PURE_INSTRUCTIONS,
                        0, SectionKind::getText()));
  }
}

/// EatToEndOfStatement - Throw away the rest of the line for testing purposes.
void AsmParser::EatToEndOfStatement() {
  while (Lexer.isNot(AsmToken::EndOfStatement) &&
         Lexer.isNot(AsmToken::Eof))
    Lex();

  // Eat EOL.
  if (Lexer.is(AsmToken::EndOfStatement))
    Lex();
}

StringRef AsmParser::ParseStringToEndOfStatement() {
  const char *Start = getTok().getLoc().getPointer();

  while (Lexer.isNot(AsmToken::EndOfStatement) &&
         Lexer.isNot(AsmToken::Eof))
    Lex();

  const char *End = getTok().getLoc().getPointer();
  return StringRef(Start, End - Start);
}

StringRef AsmParser::ParseStringToComma() {
  const char *Start = getTok().getLoc().getPointer();

  while (Lexer.isNot(AsmToken::EndOfStatement) &&
         Lexer.isNot(AsmToken::Comma) &&
         Lexer.isNot(AsmToken::Eof))
    Lex();

  const char *End = getTok().getLoc().getPointer();
  return StringRef(Start, End - Start);
}

/// ParseParenExpr - Parse a paren expression and return it.
/// NOTE: This assumes the leading '(' has already been consumed.
///
/// parenexpr ::= expr)
///
bool AsmParser::ParseParenExpr(const MCExpr *&Res, SMLoc &EndLoc) {
  if (ParseExpression(Res)) return true;
  if (Lexer.isNot(AsmToken::RParen))
    return TokError("expected ')' in parentheses expression");
  EndLoc = Lexer.getLoc();
  Lex();
  return false;
}

/// ParseBracketExpr - Parse a bracket expression and return it.
/// NOTE: This assumes the leading '[' has already been consumed.
///
/// bracketexpr ::= expr]
///
bool AsmParser::ParseBracketExpr(const MCExpr *&Res, SMLoc &EndLoc) {
  if (ParseExpression(Res)) return true;
  if (Lexer.isNot(AsmToken::RBrac))
    return TokError("expected ']' in brackets expression");
  EndLoc = Lexer.getLoc();
  Lex();
  return false;
}

/// ParsePrimaryExpr - Parse a primary expression and return it.
///  primaryexpr ::= (parenexpr
///  primaryexpr ::= symbol
///  primaryexpr ::= number
///  primaryexpr ::= '.'
///  primaryexpr ::= ~,+,- primaryexpr
bool AsmParser::ParsePrimaryExpr(const MCExpr *&Res, SMLoc &EndLoc) {
  switch (Lexer.getKind()) {
  default:
    return TokError("unknown token in expression");
  // If we have an error assume that we've already handled it.
  case AsmToken::Error:
    return true;
  case AsmToken::Exclaim:
    Lex(); // Eat the operator.
    if (ParsePrimaryExpr(Res, EndLoc))
      return true;
    Res = MCUnaryExpr::CreateLNot(Res, getContext());
    return false;
  case AsmToken::Dollar:
  case AsmToken::String:
  case AsmToken::Identifier: {
    EndLoc = Lexer.getLoc();

    StringRef Identifier;
    if (ParseIdentifier(Identifier))
      return true;

    // This is a symbol reference.
    std::pair<StringRef, StringRef> Split = Identifier.split('@');
    MCSymbol *Sym = getContext().GetOrCreateSymbol(Split.first);

    // Lookup the symbol variant if used.
    MCSymbolRefExpr::VariantKind Variant = MCSymbolRefExpr::VK_None;
    if (Split.first.size() != Identifier.size()) {
      Variant = MCSymbolRefExpr::getVariantKindForName(Split.second);
      if (Variant == MCSymbolRefExpr::VK_Invalid) {
        Variant = MCSymbolRefExpr::VK_None;
        return TokError("invalid variant '" + Split.second + "'");
      }
    }

    // If this is an absolute variable reference, substitute it now to preserve
    // semantics in the face of reassignment.
    if (Sym->isVariable() && isa<MCConstantExpr>(Sym->getVariableValue())) {
      if (Variant)
        return Error(EndLoc, "unexpected modifier on variable reference");

      Res = Sym->getVariableValue();
      return false;
    }

    // Otherwise create a symbol ref.
    Res = MCSymbolRefExpr::Create(Sym, Variant, getContext());
    return false;
  }
  case AsmToken::Integer: {
    SMLoc Loc = getTok().getLoc();
    int64_t IntVal = getTok().getIntVal();
    Res = MCConstantExpr::Create(IntVal, getContext());
    EndLoc = Lexer.getLoc();
    Lex(); // Eat token.
    // Look for 'b' or 'f' following an Integer as a directional label
    if (Lexer.getKind() == AsmToken::Identifier) {
      StringRef IDVal = getTok().getString();
      if (IDVal == "f" || IDVal == "b"){
        MCSymbol *Sym = Ctx.GetDirectionalLocalSymbol(IntVal,
                                                      IDVal == "f" ? 1 : 0);
        Res = MCSymbolRefExpr::Create(Sym, MCSymbolRefExpr::VK_None,
                                      getContext());
        if (IDVal == "b" && Sym->isUndefined())
          return Error(Loc, "invalid reference to undefined symbol");
        EndLoc = Lexer.getLoc();
        Lex(); // Eat identifier.
      }
    }
    return false;
  }
  case AsmToken::Real: {
    APFloat RealVal(APFloat::IEEEdouble, getTok().getString());
    uint64_t IntVal = RealVal.bitcastToAPInt().getZExtValue();
    Res = MCConstantExpr::Create(IntVal, getContext());
    Lex(); // Eat token.
    return false;
  }
  case AsmToken::Dot: {
    // This is a '.' reference, which references the current PC.  Emit a
    // temporary label to the streamer and refer to it.
    MCSymbol *Sym = Ctx.CreateTempSymbol();
    Out.EmitLabel(Sym);
    Res = MCSymbolRefExpr::Create(Sym, MCSymbolRefExpr::VK_None, getContext());
    EndLoc = Lexer.getLoc();
    Lex(); // Eat identifier.
    return false;
  }
  case AsmToken::LParen:
    Lex(); // Eat the '('.
    return ParseParenExpr(Res, EndLoc);
  case AsmToken::LBrac:
    if (!PlatformParser->HasBracketExpressions())
      return TokError("brackets expression not supported on this target");
    Lex(); // Eat the '['.
    return ParseBracketExpr(Res, EndLoc);
  case AsmToken::Minus:
    Lex(); // Eat the operator.
    if (ParsePrimaryExpr(Res, EndLoc))
      return true;
    Res = MCUnaryExpr::CreateMinus(Res, getContext());
    return false;
  case AsmToken::Plus:
    Lex(); // Eat the operator.
    if (ParsePrimaryExpr(Res, EndLoc))
      return true;
    Res = MCUnaryExpr::CreatePlus(Res, getContext());
    return false;
  case AsmToken::Tilde:
    Lex(); // Eat the operator.
    if (ParsePrimaryExpr(Res, EndLoc))
      return true;
    Res = MCUnaryExpr::CreateNot(Res, getContext());
    return false;
  }
}

bool AsmParser::ParseExpression(const MCExpr *&Res) {
  SMLoc EndLoc;
  return ParseExpression(Res, EndLoc);
}

const MCExpr *
AsmParser::ApplyModifierToExpr(const MCExpr *E,
                               MCSymbolRefExpr::VariantKind Variant) {
  // Recurse over the given expression, rebuilding it to apply the given variant
  // if there is exactly one symbol.
  switch (E->getKind()) {
  case MCExpr::Target:
  case MCExpr::Constant:
    return 0;

  case MCExpr::SymbolRef: {
    const MCSymbolRefExpr *SRE = cast<MCSymbolRefExpr>(E);

    if (SRE->getKind() != MCSymbolRefExpr::VK_None) {
      TokError("invalid variant on expression '" +
               getTok().getIdentifier() + "' (already modified)");
      return E;
    }

    return MCSymbolRefExpr::Create(&SRE->getSymbol(), Variant, getContext());
  }

  case MCExpr::Unary: {
    const MCUnaryExpr *UE = cast<MCUnaryExpr>(E);
    const MCExpr *Sub = ApplyModifierToExpr(UE->getSubExpr(), Variant);
    if (!Sub)
      return 0;
    return MCUnaryExpr::Create(UE->getOpcode(), Sub, getContext());
  }

  case MCExpr::Binary: {
    const MCBinaryExpr *BE = cast<MCBinaryExpr>(E);
    const MCExpr *LHS = ApplyModifierToExpr(BE->getLHS(), Variant);
    const MCExpr *RHS = ApplyModifierToExpr(BE->getRHS(), Variant);

    if (!LHS && !RHS)
      return 0;

    if (!LHS) LHS = BE->getLHS();
    if (!RHS) RHS = BE->getRHS();

    return MCBinaryExpr::Create(BE->getOpcode(), LHS, RHS, getContext());
  }
  }

  llvm_unreachable("Invalid expression kind!");
}

/// ParseExpression - Parse an expression and return it.
///
///  expr ::= expr &&,|| expr               -> lowest.
///  expr ::= expr |,^,&,! expr
///  expr ::= expr ==,!=,<>,<,<=,>,>= expr
///  expr ::= expr <<,>> expr
///  expr ::= expr +,- expr
///  expr ::= expr *,/,% expr               -> highest.
///  expr ::= primaryexpr
///
bool AsmParser::ParseExpression(const MCExpr *&Res, SMLoc &EndLoc) {
  // Parse the expression.
  Res = 0;
  if (ParsePrimaryExpr(Res, EndLoc) || ParseBinOpRHS(1, Res, EndLoc))
    return true;

  // As a special case, we support 'a op b @ modifier' by rewriting the
  // expression to include the modifier. This is inefficient, but in general we
  // expect users to use 'a@modifier op b'.
  if (Lexer.getKind() == AsmToken::At) {
    Lex();

    if (Lexer.isNot(AsmToken::Identifier))
      return TokError("unexpected symbol modifier following '@'");

    MCSymbolRefExpr::VariantKind Variant =
      MCSymbolRefExpr::getVariantKindForName(getTok().getIdentifier());
    if (Variant == MCSymbolRefExpr::VK_Invalid)
      return TokError("invalid variant '" + getTok().getIdentifier() + "'");

    const MCExpr *ModifiedRes = ApplyModifierToExpr(Res, Variant);
    if (!ModifiedRes) {
      return TokError("invalid modifier '" + getTok().getIdentifier() +
                      "' (no symbols present)");
    }

    Res = ModifiedRes;
    Lex();
  }

  // Try to constant fold it up front, if possible.
  int64_t Value;
  if (Res->EvaluateAsAbsolute(Value))
    Res = MCConstantExpr::Create(Value, getContext());

  return false;
}

bool AsmParser::ParseParenExpression(const MCExpr *&Res, SMLoc &EndLoc) {
  Res = 0;
  return ParseParenExpr(Res, EndLoc) ||
         ParseBinOpRHS(1, Res, EndLoc);
}

bool AsmParser::ParseAbsoluteExpression(int64_t &Res) {
  const MCExpr *Expr;

  SMLoc StartLoc = Lexer.getLoc();
  if (ParseExpression(Expr))
    return true;

  if (!Expr->EvaluateAsAbsolute(Res))
    return Error(StartLoc, "expected absolute expression");

  return false;
}

static unsigned getBinOpPrecedence(AsmToken::TokenKind K,
                                   MCBinaryExpr::Opcode &Kind) {
  switch (K) {
  default:
    return 0;    // not a binop.

    // Lowest Precedence: &&, ||
  case AsmToken::AmpAmp:
    Kind = MCBinaryExpr::LAnd;
    return 1;
  case AsmToken::PipePipe:
    Kind = MCBinaryExpr::LOr;
    return 1;


    // Low Precedence: |, &, ^
    //
    // FIXME: gas seems to support '!' as an infix operator?
  case AsmToken::Pipe:
    Kind = MCBinaryExpr::Or;
    return 2;
  case AsmToken::Caret:
    Kind = MCBinaryExpr::Xor;
    return 2;
  case AsmToken::Amp:
    Kind = MCBinaryExpr::And;
    return 2;

    // Low Intermediate Precedence: ==, !=, <>, <, <=, >, >=
  case AsmToken::EqualEqual:
    Kind = MCBinaryExpr::EQ;
    return 3;
  case AsmToken::ExclaimEqual:
  case AsmToken::LessGreater:
    Kind = MCBinaryExpr::NE;
    return 3;
  case AsmToken::Less:
    Kind = MCBinaryExpr::LT;
    return 3;
  case AsmToken::LessEqual:
    Kind = MCBinaryExpr::LTE;
    return 3;
  case AsmToken::Greater:
    Kind = MCBinaryExpr::GT;
    return 3;
  case AsmToken::GreaterEqual:
    Kind = MCBinaryExpr::GTE;
    return 3;

    // Intermediate Precedence: <<, >>
  case AsmToken::LessLess:
    Kind = MCBinaryExpr::Shl;
    return 4;
  case AsmToken::GreaterGreater:
    Kind = MCBinaryExpr::Shr;
    return 4;

    // High Intermediate Precedence: +, -
  case AsmToken::Plus:
    Kind = MCBinaryExpr::Add;
    return 5;
  case AsmToken::Minus:
    Kind = MCBinaryExpr::Sub;
    return 5;

    // Highest Precedence: *, /, %
  case AsmToken::Star:
    Kind = MCBinaryExpr::Mul;
    return 6;
  case AsmToken::Slash:
    Kind = MCBinaryExpr::Div;
    return 6;
  case AsmToken::Percent:
    Kind = MCBinaryExpr::Mod;
    return 6;
  }
}


/// ParseBinOpRHS - Parse all binary operators with precedence >= 'Precedence'.
/// Res contains the LHS of the expression on input.
bool AsmParser::ParseBinOpRHS(unsigned Precedence, const MCExpr *&Res,
                              SMLoc &EndLoc) {
  while (1) {
    MCBinaryExpr::Opcode Kind = MCBinaryExpr::Add;
    unsigned TokPrec = getBinOpPrecedence(Lexer.getKind(), Kind);

    // If the next token is lower precedence than we are allowed to eat, return
    // successfully with what we ate already.
    if (TokPrec < Precedence)
      return false;

    Lex();

    // Eat the next primary expression.
    const MCExpr *RHS;
    if (ParsePrimaryExpr(RHS, EndLoc)) return true;

    // If BinOp binds less tightly with RHS than the operator after RHS, let
    // the pending operator take RHS as its LHS.
    MCBinaryExpr::Opcode Dummy;
    unsigned NextTokPrec = getBinOpPrecedence(Lexer.getKind(), Dummy);
    if (TokPrec < NextTokPrec) {
      if (ParseBinOpRHS(Precedence+1, RHS, EndLoc)) return true;
    }

    // Merge LHS and RHS according to operator.
    Res = MCBinaryExpr::Create(Kind, Res, RHS, getContext());
  }
}

/// ParseStatement:
///   ::= EndOfStatement
///   ::= Label* Directive ...Operands... EndOfStatement
///   ::= Label* Identifier OperandList* EndOfStatement
bool AsmParser::ParseStatement(ParseStatementInfo &Info) {
  if (Lexer.is(AsmToken::EndOfStatement)) {
    Out.AddBlankLine();
    Lex();
    return false;
  }

  // Statements always start with an identifier or are a full line comment.
  AsmToken ID = getTok();
  SMLoc IDLoc = ID.getLoc();
  StringRef IDVal;
  int64_t LocalLabelVal = -1;
  // A full line comment is a '#' as the first token.
  if (Lexer.is(AsmToken::Hash))
    return ParseCppHashLineFilenameComment(IDLoc);

  // Allow an integer followed by a ':' as a directional local label.
  if (Lexer.is(AsmToken::Integer)) {
    LocalLabelVal = getTok().getIntVal();
    if (LocalLabelVal < 0) {
      if (!TheCondState.Ignore)
        return TokError("unexpected token at start of statement");
      IDVal = "";
    }
    else {
      IDVal = getTok().getString();
      Lex(); // Consume the integer token to be used as an identifier token.
      if (Lexer.getKind() != AsmToken::Colon) {
        if (!TheCondState.Ignore)
          return TokError("unexpected token at start of statement");
      }
    }

  } else if (Lexer.is(AsmToken::Dot)) {
    // Treat '.' as a valid identifier in this context.
    Lex();
    IDVal = ".";

  } else if (ParseIdentifier(IDVal)) {
    if (!TheCondState.Ignore)
      return TokError("unexpected token at start of statement");
    IDVal = "";
  }


  // Handle conditional assembly here before checking for skipping.  We
  // have to do this so that .endif isn't skipped in a ".if 0" block for
  // example.
  if (IDVal == ".if")
    return ParseDirectiveIf(IDLoc);
  if (IDVal == ".ifb")
    return ParseDirectiveIfb(IDLoc, true);
  if (IDVal == ".ifnb")
    return ParseDirectiveIfb(IDLoc, false);
  if (IDVal == ".ifc")
    return ParseDirectiveIfc(IDLoc, true);
  if (IDVal == ".ifnc")
    return ParseDirectiveIfc(IDLoc, false);
  if (IDVal == ".ifdef")
    return ParseDirectiveIfdef(IDLoc, true);
  if (IDVal == ".ifndef" || IDVal == ".ifnotdef")
    return ParseDirectiveIfdef(IDLoc, false);
  if (IDVal == ".elseif")
    return ParseDirectiveElseIf(IDLoc);
  if (IDVal == ".else")
    return ParseDirectiveElse(IDLoc);
  if (IDVal == ".endif")
    return ParseDirectiveEndIf(IDLoc);

  // If we are in a ".if 0" block, ignore this statement.
  if (TheCondState.Ignore) {
    EatToEndOfStatement();
    return false;
  }

  // FIXME: Recurse on local labels?

  // See what kind of statement we have.
  switch (Lexer.getKind()) {
  case AsmToken::Colon: {
    CheckForValidSection();

    // identifier ':'   -> Label.
    Lex();

    // Diagnose attempt to use '.' as a label.
    if (IDVal == ".")
      return Error(IDLoc, "invalid use of pseudo-symbol '.' as a label");

    // Diagnose attempt to use a variable as a label.
    //
    // FIXME: Diagnostics. Note the location of the definition as a label.
    // FIXME: This doesn't diagnose assignment to a symbol which has been
    // implicitly marked as external.
    MCSymbol *Sym;
    if (LocalLabelVal == -1)
      Sym = getContext().GetOrCreateSymbol(IDVal);
    else
      Sym = Ctx.CreateDirectionalLocalSymbol(LocalLabelVal);
    if (!Sym->isUndefined() || Sym->isVariable())
      return Error(IDLoc, "invalid symbol redefinition");

    // Emit the label.
    Out.EmitLabel(Sym);

    // If we are generating dwarf for assembly source files then gather the
    // info to make a dwarf label entry for this label if needed.
    if (getContext().getGenDwarfForAssembly())
      MCGenDwarfLabelEntry::Make(Sym, &getStreamer(), getSourceManager(),
                                 IDLoc);

    // Consume any end of statement token, if present, to avoid spurious
    // AddBlankLine calls().
    if (Lexer.is(AsmToken::EndOfStatement)) {
      Lex();
      if (Lexer.is(AsmToken::Eof))
        return false;
    }

    return false;
  }

  case AsmToken::Equal:
    // identifier '=' ... -> assignment statement
    Lex();

    return ParseAssignment(IDVal, true);

  default: // Normal instruction or directive.
    break;
  }

  // If macros are enabled, check to see if this is a macro instantiation.
  if (MacrosEnabled)
    if (const Macro *M = MacroMap.lookup(IDVal))
      return HandleMacroEntry(IDVal, IDLoc, M);

  // Otherwise, we have a normal instruction or directive.
  if (IDVal[0] == '.' && IDVal != ".") {

    // Target hook for parsing target specific directives.
    if (!getTargetParser().ParseDirective(ID))
      return false;

    // Assembler features
    if (IDVal == ".set" || IDVal == ".equ")
      return ParseDirectiveSet(IDVal, true);
    if (IDVal == ".equiv")
      return ParseDirectiveSet(IDVal, false);

    // Data directives

    if (IDVal == ".ascii")
      return ParseDirectiveAscii(IDVal, false);
    if (IDVal == ".asciz" || IDVal == ".string")
      return ParseDirectiveAscii(IDVal, true);

    if (IDVal == ".byte")
      return ParseDirectiveValue(1);
    if (IDVal == ".short")
      return ParseDirectiveValue(2);
    if (IDVal == ".value")
      return ParseDirectiveValue(2);
    if (IDVal == ".2byte")
      return ParseDirectiveValue(2);
    if (IDVal == ".long")
      return ParseDirectiveValue(4);
    if (IDVal == ".int")
      return ParseDirectiveValue(4);
    if (IDVal == ".4byte")
      return ParseDirectiveValue(4);
    if (IDVal == ".quad")
      return ParseDirectiveValue(8);
    if (IDVal == ".8byte")
      return ParseDirectiveValue(8);
    if (IDVal == ".single" || IDVal == ".float")
      return ParseDirectiveRealValue(APFloat::IEEEsingle);
    if (IDVal == ".double")
      return ParseDirectiveRealValue(APFloat::IEEEdouble);

    if (IDVal == ".align") {
      bool IsPow2 = !getContext().getAsmInfo().getAlignmentIsInBytes();
      return ParseDirectiveAlign(IsPow2, /*ExprSize=*/1);
    }
    if (IDVal == ".align32") {
      bool IsPow2 = !getContext().getAsmInfo().getAlignmentIsInBytes();
      return ParseDirectiveAlign(IsPow2, /*ExprSize=*/4);
    }
    if (IDVal == ".balign")
      return ParseDirectiveAlign(/*IsPow2=*/false, /*ExprSize=*/1);
    if (IDVal == ".balignw")
      return ParseDirectiveAlign(/*IsPow2=*/false, /*ExprSize=*/2);
    if (IDVal == ".balignl")
      return ParseDirectiveAlign(/*IsPow2=*/false, /*ExprSize=*/4);
    if (IDVal == ".p2align")
      return ParseDirectiveAlign(/*IsPow2=*/true, /*ExprSize=*/1);
    if (IDVal == ".p2alignw")
      return ParseDirectiveAlign(/*IsPow2=*/true, /*ExprSize=*/2);
    if (IDVal == ".p2alignl")
      return ParseDirectiveAlign(/*IsPow2=*/true, /*ExprSize=*/4);

    if (IDVal == ".org")
      return ParseDirectiveOrg();

    if (IDVal == ".fill")
      return ParseDirectiveFill();
    if (IDVal == ".space" || IDVal == ".skip")
      return ParseDirectiveSpace();
    if (IDVal == ".zero")
      return ParseDirectiveZero();

    // Symbol attribute directives

    if (IDVal == ".extern") {
      EatToEndOfStatement(); // .extern is the default, ignore it.
      return false;
    }
    if (IDVal == ".globl" || IDVal == ".global")
      return ParseDirectiveSymbolAttribute(MCSA_Global);
    if (IDVal == ".indirect_symbol")
      return ParseDirectiveSymbolAttribute(MCSA_IndirectSymbol);
    if (IDVal == ".lazy_reference")
      return ParseDirectiveSymbolAttribute(MCSA_LazyReference);
    if (IDVal == ".no_dead_strip")
      return ParseDirectiveSymbolAttribute(MCSA_NoDeadStrip);
    if (IDVal == ".symbol_resolver")
      return ParseDirectiveSymbolAttribute(MCSA_SymbolResolver);
    if (IDVal == ".private_extern")
      return ParseDirectiveSymbolAttribute(MCSA_PrivateExtern);
    if (IDVal == ".reference")
      return ParseDirectiveSymbolAttribute(MCSA_Reference);
    if (IDVal == ".weak_definition")
      return ParseDirectiveSymbolAttribute(MCSA_WeakDefinition);
    if (IDVal == ".weak_reference")
      return ParseDirectiveSymbolAttribute(MCSA_WeakReference);
    if (IDVal == ".weak_def_can_be_hidden")
      return ParseDirectiveSymbolAttribute(MCSA_WeakDefAutoPrivate);

    if (IDVal == ".comm" || IDVal == ".common")
      return ParseDirectiveComm(/*IsLocal=*/false);
    if (IDVal == ".lcomm")
      return ParseDirectiveComm(/*IsLocal=*/true);

    if (IDVal == ".abort")
      return ParseDirectiveAbort();
    if (IDVal == ".include")
      return ParseDirectiveInclude();
    if (IDVal == ".incbin")
      return ParseDirectiveIncbin();

    if (IDVal == ".code16" || IDVal == ".code16gcc")
      return TokError(Twine(IDVal) + " not supported yet");

    // Macro-like directives
    if (IDVal == ".rept")
      return ParseDirectiveRept(IDLoc);
    if (IDVal == ".irp")
      return ParseDirectiveIrp(IDLoc);
    if (IDVal == ".irpc")
      return ParseDirectiveIrpc(IDLoc);
    if (IDVal == ".endr")
      return ParseDirectiveEndr(IDLoc);

    // Look up the handler in the handler table.
    std::pair<MCAsmParserExtension*, DirectiveHandler> Handler =
      DirectiveMap.lookup(IDVal);
    if (Handler.first)
      return (*Handler.second)(Handler.first, IDVal, IDLoc);


    return Error(IDLoc, "unknown directive");
  }

  // _emit
  if (ParsingInlineAsm && IDVal == "_emit")
    return ParseDirectiveEmit(IDLoc, Info);

  CheckForValidSection();

  // Canonicalize the opcode to lower case.
  SmallString<128> OpcodeStr;
  for (unsigned i = 0, e = IDVal.size(); i != e; ++i)
    OpcodeStr.push_back(tolower(IDVal[i]));

  ParseInstructionInfo IInfo(Info.AsmRewrites);
  bool HadError = getTargetParser().ParseInstruction(IInfo, OpcodeStr.str(),
                                                     IDLoc,Info.ParsedOperands);

  // Dump the parsed representation, if requested.
  if (getShowParsedOperands()) {
    SmallString<256> Str;
    raw_svector_ostream OS(Str);
    OS << "parsed instruction: [";
    for (unsigned i = 0; i != Info.ParsedOperands.size(); ++i) {
      if (i != 0)
        OS << ", ";
      Info.ParsedOperands[i]->print(OS);
    }
    OS << "]";

    PrintMessage(IDLoc, SourceMgr::DK_Note, OS.str());
  }

  // If we are generating dwarf for assembly source files and the current
  // section is the initial text section then generate a .loc directive for
  // the instruction.
  if (!HadError && getContext().getGenDwarfForAssembly() &&
      getContext().getGenDwarfSection() == getStreamer().getCurrentSection() ) {

     unsigned Line = SrcMgr.FindLineNumber(IDLoc, CurBuffer);

     // If we previously parsed a cpp hash file line comment then make sure the
     // current Dwarf File is for the CppHashFilename if not then emit the
     // Dwarf File table for it and adjust the line number for the .loc.
     const std::vector<MCDwarfFile *> &MCDwarfFiles =
       getContext().getMCDwarfFiles();
     if (CppHashFilename.size() != 0) {
       if(MCDwarfFiles[getContext().getGenDwarfFileNumber()]->getName() !=
          CppHashFilename)
	 getStreamer().EmitDwarfFileDirective(
	   getContext().nextGenDwarfFileNumber(), StringRef(), CppHashFilename);

       unsigned CppHashLocLineNo = SrcMgr.FindLineNumber(CppHashLoc,CppHashBuf);
       Line = CppHashLineNumber - 1 + (Line - CppHashLocLineNo);
     }

    getStreamer().EmitDwarfLocDirective(getContext().getGenDwarfFileNumber(),
                                        Line, 0, DWARF2_LINE_DEFAULT_IS_STMT ?
                                        DWARF2_FLAG_IS_STMT : 0, 0, 0,
                                        StringRef());
  }

  // If parsing succeeded, match the instruction.
  if (!HadError) {
    unsigned ErrorInfo;
    HadError = getTargetParser().MatchAndEmitInstruction(IDLoc, Info.Opcode,
                                                         Info.ParsedOperands,
                                                         Out, ErrorInfo,
                                                         ParsingInlineAsm);
  }

  // Don't skip the rest of the line, the instruction parser is responsible for
  // that.
  return false;
}

/// EatToEndOfLine uses the Lexer to eat the characters to the end of the line
/// since they may not be able to be tokenized to get to the end of line token.
void AsmParser::EatToEndOfLine() {
  if (!Lexer.is(AsmToken::EndOfStatement))
    Lexer.LexUntilEndOfLine();
 // Eat EOL.
 Lex();
}

/// ParseCppHashLineFilenameComment as this:
///   ::= # number "filename"
/// or just as a full line comment if it doesn't have a number and a string.
bool AsmParser::ParseCppHashLineFilenameComment(const SMLoc &L) {
  Lex(); // Eat the hash token.

  if (getLexer().isNot(AsmToken::Integer)) {
    // Consume the line since in cases it is not a well-formed line directive,
    // as if were simply a full line comment.
    EatToEndOfLine();
    return false;
  }

  int64_t LineNumber = getTok().getIntVal();
  Lex();

  if (getLexer().isNot(AsmToken::String)) {
    EatToEndOfLine();
    return false;
  }

  StringRef Filename = getTok().getString();
  // Get rid of the enclosing quotes.
  Filename = Filename.substr(1, Filename.size()-2);

  // Save the SMLoc, Filename and LineNumber for later use by diagnostics.
  CppHashLoc = L;
  CppHashFilename = Filename;
  CppHashLineNumber = LineNumber;
  CppHashBuf = CurBuffer;

  // Ignore any trailing characters, they're just comment.
  EatToEndOfLine();
  return false;
}

/// DiagHandler - will use the last parsed cpp hash line filename comment
/// for the Filename and LineNo if any in the diagnostic.
void AsmParser::DiagHandler(const SMDiagnostic &Diag, void *Context) {
  const AsmParser *Parser = static_cast<const AsmParser*>(Context);
  raw_ostream &OS = errs();

  const SourceMgr &DiagSrcMgr = *Diag.getSourceMgr();
  const SMLoc &DiagLoc = Diag.getLoc();
  int DiagBuf = DiagSrcMgr.FindBufferContainingLoc(DiagLoc);
  int CppHashBuf = Parser->SrcMgr.FindBufferContainingLoc(Parser->CppHashLoc);

  // Like SourceMgr::PrintMessage() we need to print the include stack if any
  // before printing the message.
  int DiagCurBuffer = DiagSrcMgr.FindBufferContainingLoc(DiagLoc);
  if (!Parser->SavedDiagHandler && DiagCurBuffer > 0) {
     SMLoc ParentIncludeLoc = DiagSrcMgr.getParentIncludeLoc(DiagCurBuffer);
     DiagSrcMgr.PrintIncludeStack(ParentIncludeLoc, OS);
  }

  // If we have not parsed a cpp hash line filename comment or the source 
  // manager changed or buffer changed (like in a nested include) then just
  // print the normal diagnostic using its Filename and LineNo.
  if (!Parser->CppHashLineNumber ||
      &DiagSrcMgr != &Parser->SrcMgr ||
      DiagBuf != CppHashBuf) {
    if (Parser->SavedDiagHandler)
      Parser->SavedDiagHandler(Diag, Parser->SavedDiagContext);
    else
      Diag.print(0, OS);
    return;
  }

  // Use the CppHashFilename and calculate a line number based on the 
  // CppHashLoc and CppHashLineNumber relative to this Diag's SMLoc for
  // the diagnostic.
  const std::string Filename = Parser->CppHashFilename;

  int DiagLocLineNo = DiagSrcMgr.FindLineNumber(DiagLoc, DiagBuf);
  int CppHashLocLineNo =
      Parser->SrcMgr.FindLineNumber(Parser->CppHashLoc, CppHashBuf);
  int LineNo = Parser->CppHashLineNumber - 1 +
               (DiagLocLineNo - CppHashLocLineNo);

  SMDiagnostic NewDiag(*Diag.getSourceMgr(), Diag.getLoc(),
                       Filename, LineNo, Diag.getColumnNo(),
                       Diag.getKind(), Diag.getMessage(),
                       Diag.getLineContents(), Diag.getRanges());

  if (Parser->SavedDiagHandler)
    Parser->SavedDiagHandler(NewDiag, Parser->SavedDiagContext);
  else
    NewDiag.print(0, OS);
}

// FIXME: This is mostly duplicated from the function in AsmLexer.cpp. The
// difference being that that function accepts '@' as part of identifiers and
// we can't do that. AsmLexer.cpp should probably be changed to handle
// '@' as a special case when needed.
static bool isIdentifierChar(char c) {
  return isalnum(c) || c == '_' || c == '$' || c == '.';
}

bool AsmParser::expandMacro(raw_svector_ostream &OS, StringRef Body,
                            const MacroParameters &Parameters,
                            const MacroArguments &A,
                            const SMLoc &L) {
  unsigned NParameters = Parameters.size();
  if (NParameters != 0 && NParameters != A.size())
    return Error(L, "Wrong number of arguments");

  // A macro without parameters is handled differently on Darwin:
  // gas accepts no arguments and does no substitutions
  while (!Body.empty()) {
    // Scan for the next substitution.
    std::size_t End = Body.size(), Pos = 0;
    for (; Pos != End; ++Pos) {
      // Check for a substitution or escape.
      if (!NParameters) {
        // This macro has no parameters, look for $0, $1, etc.
        if (Body[Pos] != '$' || Pos + 1 == End)
          continue;

        char Next = Body[Pos + 1];
        if (Next == '$' || Next == 'n' || isdigit(Next))
          break;
      } else {
        // This macro has parameters, look for \foo, \bar, etc.
        if (Body[Pos] == '\\' && Pos + 1 != End)
          break;
      }
    }

    // Add the prefix.
    OS << Body.slice(0, Pos);

    // Check if we reached the end.
    if (Pos == End)
      break;

    if (!NParameters) {
      switch (Body[Pos+1]) {
        // $$ => $
      case '$':
        OS << '$';
        break;

        // $n => number of arguments
      case 'n':
        OS << A.size();
        break;

        // $[0-9] => argument
      default: {
        // Missing arguments are ignored.
        unsigned Index = Body[Pos+1] - '0';
        if (Index >= A.size())
          break;

        // Otherwise substitute with the token values, with spaces eliminated.
        for (MacroArgument::const_iterator it = A[Index].begin(),
               ie = A[Index].end(); it != ie; ++it)
          OS << it->getString();
        break;
      }
      }
      Pos += 2;
    } else {
      unsigned I = Pos + 1;
      while (isIdentifierChar(Body[I]) && I + 1 != End)
        ++I;

      const char *Begin = Body.data() + Pos +1;
      StringRef Argument(Begin, I - (Pos +1));
      unsigned Index = 0;
      for (; Index < NParameters; ++Index)
        if (Parameters[Index].first == Argument)
          break;

      if (Index == NParameters) {
          if (Body[Pos+1] == '(' && Body[Pos+2] == ')')
            Pos += 3;
          else {
            OS << '\\' << Argument;
            Pos = I;
          }
      } else {
        for (MacroArgument::const_iterator it = A[Index].begin(),
               ie = A[Index].end(); it != ie; ++it)
          if (it->getKind() == AsmToken::String)
            OS << it->getStringContents();
          else
            OS << it->getString();

        Pos += 1 + Argument.size();
      }
    }
    // Update the scan point.
    Body = Body.substr(Pos);
  }

  return false;
}

MacroInstantiation::MacroInstantiation(const Macro *M, SMLoc IL, SMLoc EL,
                                       MemoryBuffer *I)
  : TheMacro(M), Instantiation(I), InstantiationLoc(IL), ExitLoc(EL)
{
}

static bool IsOperator(AsmToken::TokenKind kind)
{
  switch (kind)
  {
    default:
      return false;
    case AsmToken::Plus:
    case AsmToken::Minus:
    case AsmToken::Tilde:
    case AsmToken::Slash:
    case AsmToken::Star:
    case AsmToken::Dot:
    case AsmToken::Equal:
    case AsmToken::EqualEqual:
    case AsmToken::Pipe:
    case AsmToken::PipePipe:
    case AsmToken::Caret:
    case AsmToken::Amp:
    case AsmToken::AmpAmp:
    case AsmToken::Exclaim:
    case AsmToken::ExclaimEqual:
    case AsmToken::Percent:
    case AsmToken::Less:
    case AsmToken::LessEqual:
    case AsmToken::LessLess:
    case AsmToken::LessGreater:
    case AsmToken::Greater:
    case AsmToken::GreaterEqual:
    case AsmToken::GreaterGreater:
      return true;
  }
}

/// ParseMacroArgument - Extract AsmTokens for a macro argument.
/// This is used for both default macro parameter values and the
/// arguments in macro invocations
bool AsmParser::ParseMacroArgument(MacroArgument &MA,
                                   AsmToken::TokenKind &ArgumentDelimiter) {
  unsigned ParenLevel = 0;
  unsigned AddTokens = 0;

  // gas accepts arguments separated by whitespace, except on Darwin
  if (!IsDarwin)
    Lexer.setSkipSpace(false);

  for (;;) {
    if (Lexer.is(AsmToken::Eof) || Lexer.is(AsmToken::Equal)) {
      Lexer.setSkipSpace(true);
      return TokError("unexpected token in macro instantiation");
    }

    if (ParenLevel == 0 && Lexer.is(AsmToken::Comma)) {
      // Spaces and commas cannot be mixed to delimit parameters
      if (ArgumentDelimiter == AsmToken::Eof)
        ArgumentDelimiter = AsmToken::Comma;
      else if (ArgumentDelimiter != AsmToken::Comma) {
        Lexer.setSkipSpace(true);
        return TokError("expected ' ' for macro argument separator");
      }
      break;
    }

    if (Lexer.is(AsmToken::Space)) {
      Lex(); // Eat spaces

      // Spaces can delimit parameters, but could also be part an expression.
      // If the token after a space is an operator, add the token and the next
      // one into this argument
      if (ArgumentDelimiter == AsmToken::Space ||
          ArgumentDelimiter == AsmToken::Eof) {
        if (IsOperator(Lexer.getKind())) {
          // Check to see whether the token is used as an operator,
          // or part of an identifier
          const char *NextChar = getTok().getEndLoc().getPointer() + 1;
          if (*NextChar == ' ')
            AddTokens = 2;
        }

        if (!AddTokens && ParenLevel == 0) {
          if (ArgumentDelimiter == AsmToken::Eof &&
              !IsOperator(Lexer.getKind()))
            ArgumentDelimiter = AsmToken::Space;
          break;
        }
      }
    }

    // HandleMacroEntry relies on not advancing the lexer here
    // to be able to fill in the remaining default parameter values
    if (Lexer.is(AsmToken::EndOfStatement))
      break;

    // Adjust the current parentheses level.
    if (Lexer.is(AsmToken::LParen))
      ++ParenLevel;
    else if (Lexer.is(AsmToken::RParen) && ParenLevel)
      --ParenLevel;

    // Append the token to the current argument list.
    MA.push_back(getTok());
    if (AddTokens)
      AddTokens--;
    Lex();
  }

  Lexer.setSkipSpace(true);
  if (ParenLevel != 0)
    return TokError("unbalanced parentheses in macro argument");
  return false;
}

// Parse the macro instantiation arguments.
bool AsmParser::ParseMacroArguments(const Macro *M, MacroArguments &A) {
  const unsigned NParameters = M ? M->Parameters.size() : 0;
  // Argument delimiter is initially unknown. It will be set by
  // ParseMacroArgument()
  AsmToken::TokenKind ArgumentDelimiter = AsmToken::Eof;

  // Parse two kinds of macro invocations:
  // - macros defined without any parameters accept an arbitrary number of them
  // - macros defined with parameters accept at most that many of them
  for (unsigned Parameter = 0; !NParameters || Parameter < NParameters;
       ++Parameter) {
    MacroArgument MA;

    if (ParseMacroArgument(MA, ArgumentDelimiter))
      return true;

    if (!MA.empty() || !NParameters)
      A.push_back(MA);
    else if (NParameters) {
      if (!M->Parameters[Parameter].second.empty())
        A.push_back(M->Parameters[Parameter].second);
    }

    // At the end of the statement, fill in remaining arguments that have
    // default values. If there aren't any, then the next argument is
    // required but missing
    if (Lexer.is(AsmToken::EndOfStatement)) {
      if (NParameters && Parameter < NParameters - 1) {
        if (M->Parameters[Parameter + 1].second.empty())
          return TokError("macro argument '" +
                          Twine(M->Parameters[Parameter + 1].first) +
                          "' is missing");
        else
          continue;
      }
      return false;
    }

    if (Lexer.is(AsmToken::Comma))
      Lex();
  }
  return TokError("Too many arguments");
}

bool AsmParser::HandleMacroEntry(StringRef Name, SMLoc NameLoc,
                                 const Macro *M) {
  // Arbitrarily limit macro nesting depth, to match 'as'. We can eliminate
  // this, although we should protect against infinite loops.
  if (ActiveMacros.size() == 20)
    return TokError("macros cannot be nested more than 20 levels deep");

  MacroArguments A;
  if (ParseMacroArguments(M, A))
    return true;

  // Remove any trailing empty arguments. Do this after-the-fact as we have
  // to keep empty arguments in the middle of the list or positionality
  // gets off. e.g.,  "foo 1, , 2" vs. "foo 1, 2,"
  while (!A.empty() && A.back().empty())
    A.pop_back();

  // Macro instantiation is lexical, unfortunately. We construct a new buffer
  // to hold the macro body with substitutions.
  SmallString<256> Buf;
  StringRef Body = M->Body;
  raw_svector_ostream OS(Buf);

  if (expandMacro(OS, Body, M->Parameters, A, getTok().getLoc()))
    return true;

  // We include the .endmacro in the buffer as our queue to exit the macro
  // instantiation.
  OS << ".endmacro\n";

  MemoryBuffer *Instantiation =
    MemoryBuffer::getMemBufferCopy(OS.str(), "<instantiation>");

  // Create the macro instantiation object and add to the current macro
  // instantiation stack.
  MacroInstantiation *MI = new MacroInstantiation(M, NameLoc,
                                                  getTok().getLoc(),
                                                  Instantiation);
  ActiveMacros.push_back(MI);

  // Jump to the macro instantiation and prime the lexer.
  CurBuffer = SrcMgr.AddNewSourceBuffer(MI->Instantiation, SMLoc());
  Lexer.setBuffer(SrcMgr.getMemoryBuffer(CurBuffer));
  Lex();

  return false;
}

void AsmParser::HandleMacroExit() {
  // Jump to the EndOfStatement we should return to, and consume it.
  JumpToLoc(ActiveMacros.back()->ExitLoc);
  Lex();

  // Pop the instantiation entry.
  delete ActiveMacros.back();
  ActiveMacros.pop_back();
}

static bool IsUsedIn(const MCSymbol *Sym, const MCExpr *Value) {
  switch (Value->getKind()) {
  case MCExpr::Binary: {
    const MCBinaryExpr *BE = static_cast<const MCBinaryExpr*>(Value);
    return IsUsedIn(Sym, BE->getLHS()) || IsUsedIn(Sym, BE->getRHS());
    break;
  }
  case MCExpr::Target:
  case MCExpr::Constant:
    return false;
  case MCExpr::SymbolRef: {
    const MCSymbol &S = static_cast<const MCSymbolRefExpr*>(Value)->getSymbol();
    if (S.isVariable())
      return IsUsedIn(Sym, S.getVariableValue());
    return &S == Sym;
  }
  case MCExpr::Unary:
    return IsUsedIn(Sym, static_cast<const MCUnaryExpr*>(Value)->getSubExpr());
  }

  llvm_unreachable("Unknown expr kind!");
}

bool AsmParser::ParseAssignment(StringRef Name, bool allow_redef,
                                bool NoDeadStrip) {
  // FIXME: Use better location, we should use proper tokens.
  SMLoc EqualLoc = Lexer.getLoc();

  const MCExpr *Value;
  if (ParseExpression(Value))
    return true;

  // Note: we don't count b as used in "a = b". This is to allow
  // a = b
  // b = c

  if (Lexer.isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in assignment");

  // Error on assignment to '.'.
  if (Name == ".") {
    return Error(EqualLoc, ("assignment to pseudo-symbol '.' is unsupported "
                            "(use '.space' or '.org').)"));
  }

  // Eat the end of statement marker.
  Lex();

  // Validate that the LHS is allowed to be a variable (either it has not been
  // used as a symbol, or it is an absolute symbol).
  MCSymbol *Sym = getContext().LookupSymbol(Name);
  if (Sym) {
    // Diagnose assignment to a label.
    //
    // FIXME: Diagnostics. Note the location of the definition as a label.
    // FIXME: Diagnose assignment to protected identifier (e.g., register name).
    if (IsUsedIn(Sym, Value))
      return Error(EqualLoc, "Recursive use of '" + Name + "'");
    else if (Sym->isUndefined() && !Sym->isUsed() && !Sym->isVariable())
      ; // Allow redefinitions of undefined symbols only used in directives.
    else if (Sym->isVariable() && !Sym->isUsed() && allow_redef)
      ; // Allow redefinitions of variables that haven't yet been used.
    else if (!Sym->isUndefined() && (!Sym->isVariable() || !allow_redef))
      return Error(EqualLoc, "redefinition of '" + Name + "'");
    else if (!Sym->isVariable())
      return Error(EqualLoc, "invalid assignment to '" + Name + "'");
    else if (!isa<MCConstantExpr>(Sym->getVariableValue()))
      return Error(EqualLoc, "invalid reassignment of non-absolute variable '" +
                   Name + "'");

    // Don't count these checks as uses.
    Sym->setUsed(false);
  } else
    Sym = getContext().GetOrCreateSymbol(Name);

  // FIXME: Handle '.'.

  // Do the assignment.
  Out.EmitAssignment(Sym, Value);
  if (NoDeadStrip)
    Out.EmitSymbolAttribute(Sym, MCSA_NoDeadStrip);


  return false;
}

/// ParseIdentifier:
///   ::= identifier
///   ::= string
bool AsmParser::ParseIdentifier(StringRef &Res) {
  // The assembler has relaxed rules for accepting identifiers, in particular we
  // allow things like '.globl $foo', which would normally be separate
  // tokens. At this level, we have already lexed so we cannot (currently)
  // handle this as a context dependent token, instead we detect adjacent tokens
  // and return the combined identifier.
  if (Lexer.is(AsmToken::Dollar)) {
    SMLoc DollarLoc = getLexer().getLoc();

    // Consume the dollar sign, and check for a following identifier.
    Lex();
    if (Lexer.isNot(AsmToken::Identifier))
      return true;

    // We have a '$' followed by an identifier, make sure they are adjacent.
    if (DollarLoc.getPointer() + 1 != getTok().getLoc().getPointer())
      return true;

    // Construct the joined identifier and consume the token.
    Res = StringRef(DollarLoc.getPointer(),
                    getTok().getIdentifier().size() + 1);
    Lex();
    return false;
  }

  if (Lexer.isNot(AsmToken::Identifier) &&
      Lexer.isNot(AsmToken::String))
    return true;

  Res = getTok().getIdentifier();

  Lex(); // Consume the identifier token.

  return false;
}

/// ParseDirectiveSet:
///   ::= .equ identifier ',' expression
///   ::= .equiv identifier ',' expression
///   ::= .set identifier ',' expression
bool AsmParser::ParseDirectiveSet(StringRef IDVal, bool allow_redef) {
  StringRef Name;

  if (ParseIdentifier(Name))
    return TokError("expected identifier after '" + Twine(IDVal) + "'");

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in '" + Twine(IDVal) + "'");
  Lex();

  return ParseAssignment(Name, allow_redef, true);
}

bool AsmParser::ParseEscapedString(std::string &Data) {
  assert(getLexer().is(AsmToken::String) && "Unexpected current token!");

  Data = "";
  StringRef Str = getTok().getStringContents();
  for (unsigned i = 0, e = Str.size(); i != e; ++i) {
    if (Str[i] != '\\') {
      Data += Str[i];
      continue;
    }

    // Recognize escaped characters. Note that this escape semantics currently
    // loosely follows Darwin 'as'. Notably, it doesn't support hex escapes.
    ++i;
    if (i == e)
      return TokError("unexpected backslash at end of string");

    // Recognize octal sequences.
    if ((unsigned) (Str[i] - '0') <= 7) {
      // Consume up to three octal characters.
      unsigned Value = Str[i] - '0';

      if (i + 1 != e && ((unsigned) (Str[i + 1] - '0')) <= 7) {
        ++i;
        Value = Value * 8 + (Str[i] - '0');

        if (i + 1 != e && ((unsigned) (Str[i + 1] - '0')) <= 7) {
          ++i;
          Value = Value * 8 + (Str[i] - '0');
        }
      }

      if (Value > 255)
        return TokError("invalid octal escape sequence (out of range)");

      Data += (unsigned char) Value;
      continue;
    }

    // Otherwise recognize individual escapes.
    switch (Str[i]) {
    default:
      // Just reject invalid escape sequences for now.
      return TokError("invalid escape sequence (unrecognized character)");

    case 'b': Data += '\b'; break;
    case 'f': Data += '\f'; break;
    case 'n': Data += '\n'; break;
    case 'r': Data += '\r'; break;
    case 't': Data += '\t'; break;
    case '"': Data += '"'; break;
    case '\\': Data += '\\'; break;
    }
  }

  return false;
}

/// ParseDirectiveAscii:
///   ::= ( .ascii | .asciz | .string ) [ "string" ( , "string" )* ]
bool AsmParser::ParseDirectiveAscii(StringRef IDVal, bool ZeroTerminated) {
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    CheckForValidSection();

    for (;;) {
      if (getLexer().isNot(AsmToken::String))
        return TokError("expected string in '" + Twine(IDVal) + "' directive");

      std::string Data;
      if (ParseEscapedString(Data))
        return true;

      getStreamer().EmitBytes(Data, DEFAULT_ADDRSPACE);
      if (ZeroTerminated)
        getStreamer().EmitBytes(StringRef("\0", 1), DEFAULT_ADDRSPACE);

      Lex();

      if (getLexer().is(AsmToken::EndOfStatement))
        break;

      if (getLexer().isNot(AsmToken::Comma))
        return TokError("unexpected token in '" + Twine(IDVal) + "' directive");
      Lex();
    }
  }

  Lex();
  return false;
}

/// ParseDirectiveValue
///  ::= (.byte | .short | ... ) [ expression (, expression)* ]
bool AsmParser::ParseDirectiveValue(unsigned Size) {
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    CheckForValidSection();

    for (;;) {
      const MCExpr *Value;
      SMLoc ExprLoc = getLexer().getLoc();
      if (ParseExpression(Value))
        return true;

      // Special case constant expressions to match code generator.
      if (const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(Value)) {
        assert(Size <= 8 && "Invalid size");
        uint64_t IntValue = MCE->getValue();
        if (!isUIntN(8 * Size, IntValue) && !isIntN(8 * Size, IntValue))
          return Error(ExprLoc, "literal value out of range for directive");
        getStreamer().EmitIntValue(IntValue, Size, DEFAULT_ADDRSPACE);
      } else
        getStreamer().EmitValue(Value, Size, DEFAULT_ADDRSPACE);

      if (getLexer().is(AsmToken::EndOfStatement))
        break;

      // FIXME: Improve diagnostic.
      if (getLexer().isNot(AsmToken::Comma))
        return TokError("unexpected token in directive");
      Lex();
    }
  }

  Lex();
  return false;
}

/// ParseDirectiveRealValue
///  ::= (.single | .double) [ expression (, expression)* ]
bool AsmParser::ParseDirectiveRealValue(const fltSemantics &Semantics) {
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    CheckForValidSection();

    for (;;) {
      // We don't truly support arithmetic on floating point expressions, so we
      // have to manually parse unary prefixes.
      bool IsNeg = false;
      if (getLexer().is(AsmToken::Minus)) {
        Lex();
        IsNeg = true;
      } else if (getLexer().is(AsmToken::Plus))
        Lex();

      if (getLexer().isNot(AsmToken::Integer) &&
          getLexer().isNot(AsmToken::Real) &&
          getLexer().isNot(AsmToken::Identifier))
        return TokError("unexpected token in directive");

      // Convert to an APFloat.
      APFloat Value(Semantics);
      StringRef IDVal = getTok().getString();
      if (getLexer().is(AsmToken::Identifier)) {
        if (!IDVal.compare_lower("infinity") || !IDVal.compare_lower("inf"))
          Value = APFloat::getInf(Semantics);
        else if (!IDVal.compare_lower("nan"))
          Value = APFloat::getNaN(Semantics, false, ~0);
        else
          return TokError("invalid floating point literal");
      } else if (Value.convertFromString(IDVal, APFloat::rmNearestTiesToEven) ==
          APFloat::opInvalidOp)
        return TokError("invalid floating point literal");
      if (IsNeg)
        Value.changeSign();

      // Consume the numeric token.
      Lex();

      // Emit the value as an integer.
      APInt AsInt = Value.bitcastToAPInt();
      getStreamer().EmitIntValue(AsInt.getLimitedValue(),
                                 AsInt.getBitWidth() / 8, DEFAULT_ADDRSPACE);

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

/// ParseDirectiveSpace
///  ::= .space expression [ , expression ]
bool AsmParser::ParseDirectiveSpace() {
  CheckForValidSection();

  int64_t NumBytes;
  if (ParseAbsoluteExpression(NumBytes))
    return true;

  int64_t FillExpr = 0;
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    if (getLexer().isNot(AsmToken::Comma))
      return TokError("unexpected token in '.space' directive");
    Lex();

    if (ParseAbsoluteExpression(FillExpr))
      return true;

    if (getLexer().isNot(AsmToken::EndOfStatement))
      return TokError("unexpected token in '.space' directive");
  }

  Lex();

  if (NumBytes <= 0)
    return TokError("invalid number of bytes in '.space' directive");

  // FIXME: Sometimes the fill expr is 'nop' if it isn't supplied, instead of 0.
  getStreamer().EmitFill(NumBytes, FillExpr, DEFAULT_ADDRSPACE);

  return false;
}

/// ParseDirectiveZero
///  ::= .zero expression
bool AsmParser::ParseDirectiveZero() {
  CheckForValidSection();

  int64_t NumBytes;
  if (ParseAbsoluteExpression(NumBytes))
    return true;

  int64_t Val = 0;
  if (getLexer().is(AsmToken::Comma)) {
    Lex();
    if (ParseAbsoluteExpression(Val))
      return true;
  }

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.zero' directive");

  Lex();

  getStreamer().EmitFill(NumBytes, Val, DEFAULT_ADDRSPACE);

  return false;
}

/// ParseDirectiveFill
///  ::= .fill expression , expression , expression
bool AsmParser::ParseDirectiveFill() {
  CheckForValidSection();

  int64_t NumValues;
  if (ParseAbsoluteExpression(NumValues))
    return true;

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in '.fill' directive");
  Lex();

  int64_t FillSize;
  if (ParseAbsoluteExpression(FillSize))
    return true;

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in '.fill' directive");
  Lex();

  int64_t FillExpr;
  if (ParseAbsoluteExpression(FillExpr))
    return true;

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.fill' directive");

  Lex();

  if (FillSize != 1 && FillSize != 2 && FillSize != 4 && FillSize != 8)
    return TokError("invalid '.fill' size, expected 1, 2, 4, or 8");

  for (uint64_t i = 0, e = NumValues; i != e; ++i)
    getStreamer().EmitIntValue(FillExpr, FillSize, DEFAULT_ADDRSPACE);

  return false;
}

/// ParseDirectiveOrg
///  ::= .org expression [ , expression ]
bool AsmParser::ParseDirectiveOrg() {
  CheckForValidSection();

  const MCExpr *Offset;
  SMLoc Loc = getTok().getLoc();
  if (ParseExpression(Offset))
    return true;

  // Parse optional fill expression.
  int64_t FillExpr = 0;
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    if (getLexer().isNot(AsmToken::Comma))
      return TokError("unexpected token in '.org' directive");
    Lex();

    if (ParseAbsoluteExpression(FillExpr))
      return true;

    if (getLexer().isNot(AsmToken::EndOfStatement))
      return TokError("unexpected token in '.org' directive");
  }

  Lex();

  // Only limited forms of relocatable expressions are accepted here, it
  // has to be relative to the current section. The streamer will return
  // 'true' if the expression wasn't evaluatable.
  if (getStreamer().EmitValueToOffset(Offset, FillExpr))
    return Error(Loc, "expected assembly-time absolute expression");

  return false;
}

/// ParseDirectiveAlign
///  ::= {.align, ...} expression [ , expression [ , expression ]]
bool AsmParser::ParseDirectiveAlign(bool IsPow2, unsigned ValueSize) {
  CheckForValidSection();

  SMLoc AlignmentLoc = getLexer().getLoc();
  int64_t Alignment;
  if (ParseAbsoluteExpression(Alignment))
    return true;

  SMLoc MaxBytesLoc;
  bool HasFillExpr = false;
  int64_t FillExpr = 0;
  int64_t MaxBytesToFill = 0;
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    if (getLexer().isNot(AsmToken::Comma))
      return TokError("unexpected token in directive");
    Lex();

    // The fill expression can be omitted while specifying a maximum number of
    // alignment bytes, e.g:
    //  .align 3,,4
    if (getLexer().isNot(AsmToken::Comma)) {
      HasFillExpr = true;
      if (ParseAbsoluteExpression(FillExpr))
        return true;
    }

    if (getLexer().isNot(AsmToken::EndOfStatement)) {
      if (getLexer().isNot(AsmToken::Comma))
        return TokError("unexpected token in directive");
      Lex();

      MaxBytesLoc = getLexer().getLoc();
      if (ParseAbsoluteExpression(MaxBytesToFill))
        return true;

      if (getLexer().isNot(AsmToken::EndOfStatement))
        return TokError("unexpected token in directive");
    }
  }

  Lex();

  if (!HasFillExpr)
    FillExpr = 0;

  // Compute alignment in bytes.
  if (IsPow2) {
    // FIXME: Diagnose overflow.
    if (Alignment >= 32) {
      Error(AlignmentLoc, "invalid alignment value");
      Alignment = 31;
    }

    Alignment = 1ULL << Alignment;
  }

  // Diagnose non-sensical max bytes to align.
  if (MaxBytesLoc.isValid()) {
    if (MaxBytesToFill < 1) {
      Error(MaxBytesLoc, "alignment directive can never be satisfied in this "
            "many bytes, ignoring maximum bytes expression");
      MaxBytesToFill = 0;
    }

    if (MaxBytesToFill >= Alignment) {
      Warning(MaxBytesLoc, "maximum bytes expression exceeds alignment and "
              "has no effect");
      MaxBytesToFill = 0;
    }
  }

  // Check whether we should use optimal code alignment for this .align
  // directive.
  bool UseCodeAlign = getStreamer().getCurrentSection()->UseCodeAlign();
  if ((!HasFillExpr || Lexer.getMAI().getTextAlignFillValue() == FillExpr) &&
      ValueSize == 1 && UseCodeAlign) {
    getStreamer().EmitCodeAlignment(Alignment, MaxBytesToFill);
  } else {
    // FIXME: Target specific behavior about how the "extra" bytes are filled.
    getStreamer().EmitValueToAlignment(Alignment, FillExpr, ValueSize,
                                       MaxBytesToFill);
  }

  return false;
}

/// ParseDirectiveSymbolAttribute
///  ::= { ".globl", ".weak", ... } [ identifier ( , identifier )* ]
bool AsmParser::ParseDirectiveSymbolAttribute(MCSymbolAttr Attr) {
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    for (;;) {
      StringRef Name;
      SMLoc Loc = getTok().getLoc();

      if (ParseIdentifier(Name))
        return Error(Loc, "expected identifier in directive");

      MCSymbol *Sym = getContext().GetOrCreateSymbol(Name);

      // Assembler local symbols don't make any sense here. Complain loudly.
      if (Sym->isTemporary())
        return Error(Loc, "non-local symbol required in directive");

      getStreamer().EmitSymbolAttribute(Sym, Attr);

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

/// ParseDirectiveComm
///  ::= ( .comm | .lcomm ) identifier , size_expression [ , align_expression ]
bool AsmParser::ParseDirectiveComm(bool IsLocal) {
  CheckForValidSection();

  SMLoc IDLoc = getLexer().getLoc();
  StringRef Name;
  if (ParseIdentifier(Name))
    return TokError("expected identifier in directive");

  // Handle the identifier as the key symbol.
  MCSymbol *Sym = getContext().GetOrCreateSymbol(Name);

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lex();

  int64_t Size;
  SMLoc SizeLoc = getLexer().getLoc();
  if (ParseAbsoluteExpression(Size))
    return true;

  int64_t Pow2Alignment = 0;
  SMLoc Pow2AlignmentLoc;
  if (getLexer().is(AsmToken::Comma)) {
    Lex();
    Pow2AlignmentLoc = getLexer().getLoc();
    if (ParseAbsoluteExpression(Pow2Alignment))
      return true;

    LCOMM::LCOMMType LCOMM = Lexer.getMAI().getLCOMMDirectiveAlignmentType();
    if (IsLocal && LCOMM == LCOMM::NoAlignment)
      return Error(Pow2AlignmentLoc, "alignment not supported on this target");

    // If this target takes alignments in bytes (not log) validate and convert.
    if ((!IsLocal && Lexer.getMAI().getCOMMDirectiveAlignmentIsInBytes()) ||
        (IsLocal && LCOMM == LCOMM::ByteAlignment)) {
      if (!isPowerOf2_64(Pow2Alignment))
        return Error(Pow2AlignmentLoc, "alignment must be a power of 2");
      Pow2Alignment = Log2_64(Pow2Alignment);
    }
  }

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.comm' or '.lcomm' directive");

  Lex();

  // NOTE: a size of zero for a .comm should create a undefined symbol
  // but a size of .lcomm creates a bss symbol of size zero.
  if (Size < 0)
    return Error(SizeLoc, "invalid '.comm' or '.lcomm' directive size, can't "
                 "be less than zero");

  // NOTE: The alignment in the directive is a power of 2 value, the assembler
  // may internally end up wanting an alignment in bytes.
  // FIXME: Diagnose overflow.
  if (Pow2Alignment < 0)
    return Error(Pow2AlignmentLoc, "invalid '.comm' or '.lcomm' directive "
                 "alignment, can't be less than zero");

  if (!Sym->isUndefined())
    return Error(IDLoc, "invalid symbol redefinition");

  // Create the Symbol as a common or local common with Size and Pow2Alignment
  if (IsLocal) {
    getStreamer().EmitLocalCommonSymbol(Sym, Size, 1 << Pow2Alignment);
    return false;
  }

  getStreamer().EmitCommonSymbol(Sym, Size, 1 << Pow2Alignment);
  return false;
}

/// ParseDirectiveAbort
///  ::= .abort [... message ...]
bool AsmParser::ParseDirectiveAbort() {
  // FIXME: Use loc from directive.
  SMLoc Loc = getLexer().getLoc();

  StringRef Str = ParseStringToEndOfStatement();
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.abort' directive");

  Lex();

  if (Str.empty())
    Error(Loc, ".abort detected. Assembly stopping.");
  else
    Error(Loc, ".abort '" + Str + "' detected. Assembly stopping.");
  // FIXME: Actually abort assembly here.

  return false;
}

/// ParseDirectiveInclude
///  ::= .include "filename"
bool AsmParser::ParseDirectiveInclude() {
  if (getLexer().isNot(AsmToken::String))
    return TokError("expected string in '.include' directive");

  std::string Filename = getTok().getString();
  SMLoc IncludeLoc = getLexer().getLoc();
  Lex();

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.include' directive");

  // Strip the quotes.
  Filename = Filename.substr(1, Filename.size()-2);

  // Attempt to switch the lexer to the included file before consuming the end
  // of statement to avoid losing it when we switch.
  if (EnterIncludeFile(Filename)) {
    Error(IncludeLoc, "Could not find include file '" + Filename + "'");
    return true;
  }

  return false;
}

/// ParseDirectiveIncbin
///  ::= .incbin "filename"
bool AsmParser::ParseDirectiveIncbin() {
  if (getLexer().isNot(AsmToken::String))
    return TokError("expected string in '.incbin' directive");

  std::string Filename = getTok().getString();
  SMLoc IncbinLoc = getLexer().getLoc();
  Lex();

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.incbin' directive");

  // Strip the quotes.
  Filename = Filename.substr(1, Filename.size()-2);

  // Attempt to process the included file.
  if (ProcessIncbinFile(Filename)) {
    Error(IncbinLoc, "Could not find incbin file '" + Filename + "'");
    return true;
  }

  return false;
}

/// ParseDirectiveIf
/// ::= .if expression
bool AsmParser::ParseDirectiveIf(SMLoc DirectiveLoc) {
  TheCondStack.push_back(TheCondState);
  TheCondState.TheCond = AsmCond::IfCond;
  if (TheCondState.Ignore) {
    EatToEndOfStatement();
  } else {
    int64_t ExprValue;
    if (ParseAbsoluteExpression(ExprValue))
      return true;

    if (getLexer().isNot(AsmToken::EndOfStatement))
      return TokError("unexpected token in '.if' directive");

    Lex();

    TheCondState.CondMet = ExprValue;
    TheCondState.Ignore = !TheCondState.CondMet;
  }

  return false;
}

/// ParseDirectiveIfb
/// ::= .ifb string
bool AsmParser::ParseDirectiveIfb(SMLoc DirectiveLoc, bool ExpectBlank) {
  TheCondStack.push_back(TheCondState);
  TheCondState.TheCond = AsmCond::IfCond;

  if (TheCondState.Ignore) {
    EatToEndOfStatement();
  } else {
    StringRef Str = ParseStringToEndOfStatement();

    if (getLexer().isNot(AsmToken::EndOfStatement))
      return TokError("unexpected token in '.ifb' directive");

    Lex();

    TheCondState.CondMet = ExpectBlank == Str.empty();
    TheCondState.Ignore = !TheCondState.CondMet;
  }

  return false;
}

/// ParseDirectiveIfc
/// ::= .ifc string1, string2
bool AsmParser::ParseDirectiveIfc(SMLoc DirectiveLoc, bool ExpectEqual) {
  TheCondStack.push_back(TheCondState);
  TheCondState.TheCond = AsmCond::IfCond;

  if (TheCondState.Ignore) {
    EatToEndOfStatement();
  } else {
    StringRef Str1 = ParseStringToComma();

    if (getLexer().isNot(AsmToken::Comma))
      return TokError("unexpected token in '.ifc' directive");

    Lex();

    StringRef Str2 = ParseStringToEndOfStatement();

    if (getLexer().isNot(AsmToken::EndOfStatement))
      return TokError("unexpected token in '.ifc' directive");

    Lex();

    TheCondState.CondMet = ExpectEqual == (Str1 == Str2);
    TheCondState.Ignore = !TheCondState.CondMet;
  }

  return false;
}

/// ParseDirectiveIfdef
/// ::= .ifdef symbol
bool AsmParser::ParseDirectiveIfdef(SMLoc DirectiveLoc, bool expect_defined) {
  StringRef Name;
  TheCondStack.push_back(TheCondState);
  TheCondState.TheCond = AsmCond::IfCond;

  if (TheCondState.Ignore) {
    EatToEndOfStatement();
  } else {
    if (ParseIdentifier(Name))
      return TokError("expected identifier after '.ifdef'");

    Lex();

    MCSymbol *Sym = getContext().LookupSymbol(Name);

    if (expect_defined)
      TheCondState.CondMet = (Sym != NULL && !Sym->isUndefined());
    else
      TheCondState.CondMet = (Sym == NULL || Sym->isUndefined());
    TheCondState.Ignore = !TheCondState.CondMet;
  }

  return false;
}

/// ParseDirectiveElseIf
/// ::= .elseif expression
bool AsmParser::ParseDirectiveElseIf(SMLoc DirectiveLoc) {
  if (TheCondState.TheCond != AsmCond::IfCond &&
      TheCondState.TheCond != AsmCond::ElseIfCond)
      Error(DirectiveLoc, "Encountered a .elseif that doesn't follow a .if or "
                          " an .elseif");
  TheCondState.TheCond = AsmCond::ElseIfCond;

  bool LastIgnoreState = false;
  if (!TheCondStack.empty())
      LastIgnoreState = TheCondStack.back().Ignore;
  if (LastIgnoreState || TheCondState.CondMet) {
    TheCondState.Ignore = true;
    EatToEndOfStatement();
  }
  else {
    int64_t ExprValue;
    if (ParseAbsoluteExpression(ExprValue))
      return true;

    if (getLexer().isNot(AsmToken::EndOfStatement))
      return TokError("unexpected token in '.elseif' directive");

    Lex();
    TheCondState.CondMet = ExprValue;
    TheCondState.Ignore = !TheCondState.CondMet;
  }

  return false;
}

/// ParseDirectiveElse
/// ::= .else
bool AsmParser::ParseDirectiveElse(SMLoc DirectiveLoc) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.else' directive");

  Lex();

  if (TheCondState.TheCond != AsmCond::IfCond &&
      TheCondState.TheCond != AsmCond::ElseIfCond)
      Error(DirectiveLoc, "Encountered a .else that doesn't follow a .if or an "
                          ".elseif");
  TheCondState.TheCond = AsmCond::ElseCond;
  bool LastIgnoreState = false;
  if (!TheCondStack.empty())
    LastIgnoreState = TheCondStack.back().Ignore;
  if (LastIgnoreState || TheCondState.CondMet)
    TheCondState.Ignore = true;
  else
    TheCondState.Ignore = false;

  return false;
}

/// ParseDirectiveEndIf
/// ::= .endif
bool AsmParser::ParseDirectiveEndIf(SMLoc DirectiveLoc) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.endif' directive");

  Lex();

  if ((TheCondState.TheCond == AsmCond::NoCond) ||
      TheCondStack.empty())
    Error(DirectiveLoc, "Encountered a .endif that doesn't follow a .if or "
                        ".else");
  if (!TheCondStack.empty()) {
    TheCondState = TheCondStack.back();
    TheCondStack.pop_back();
  }

  return false;
}

/// ParseDirectiveFile
/// ::= .file [number] filename
/// ::= .file number directory filename
bool GenericAsmParser::ParseDirectiveFile(StringRef, SMLoc DirectiveLoc) {
  // FIXME: I'm not sure what this is.
  int64_t FileNumber = -1;
  SMLoc FileNumberLoc = getLexer().getLoc();
  if (getLexer().is(AsmToken::Integer)) {
    FileNumber = getTok().getIntVal();
    Lex();

    if (FileNumber < 1)
      return TokError("file number less than one");
  }

  if (getLexer().isNot(AsmToken::String))
    return TokError("unexpected token in '.file' directive");

  // Usually the directory and filename together, otherwise just the directory.
  StringRef Path = getTok().getString();
  Path = Path.substr(1, Path.size()-2);
  Lex();

  StringRef Directory;
  StringRef Filename;
  if (getLexer().is(AsmToken::String)) {
    if (FileNumber == -1)
      return TokError("explicit path specified, but no file number");
    Filename = getTok().getString();
    Filename = Filename.substr(1, Filename.size()-2);
    Directory = Path;
    Lex();
  } else {
    Filename = Path;
  }

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.file' directive");

  if (FileNumber == -1)
    getStreamer().EmitFileDirective(Filename);
  else {
    if (getContext().getGenDwarfForAssembly() == true)
      Error(DirectiveLoc, "input can't have .file dwarf directives when -g is "
                        "used to generate dwarf debug info for assembly code");

    if (getStreamer().EmitDwarfFileDirective(FileNumber, Directory, Filename))
      Error(FileNumberLoc, "file number already allocated");
  }

  return false;
}

/// ParseDirectiveLine
/// ::= .line [number]
bool GenericAsmParser::ParseDirectiveLine(StringRef, SMLoc DirectiveLoc) {
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    if (getLexer().isNot(AsmToken::Integer))
      return TokError("unexpected token in '.line' directive");

    int64_t LineNumber = getTok().getIntVal();
    (void) LineNumber;
    Lex();

    // FIXME: Do something with the .line.
  }

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.line' directive");

  return false;
}


/// ParseDirectiveLoc
/// ::= .loc FileNumber [LineNumber] [ColumnPos] [basic_block] [prologue_end]
///                                [epilogue_begin] [is_stmt VALUE] [isa VALUE]
/// The first number is a file number, must have been previously assigned with
/// a .file directive, the second number is the line number and optionally the
/// third number is a column position (zero if not specified).  The remaining
/// optional items are .loc sub-directives.
bool GenericAsmParser::ParseDirectiveLoc(StringRef, SMLoc DirectiveLoc) {

  if (getLexer().isNot(AsmToken::Integer))
    return TokError("unexpected token in '.loc' directive");
  int64_t FileNumber = getTok().getIntVal();
  if (FileNumber < 1)
    return TokError("file number less than one in '.loc' directive");
  if (!getContext().isValidDwarfFileNumber(FileNumber))
    return TokError("unassigned file number in '.loc' directive");
  Lex();

  int64_t LineNumber = 0;
  if (getLexer().is(AsmToken::Integer)) {
    LineNumber = getTok().getIntVal();
    if (LineNumber < 1)
      return TokError("line number less than one in '.loc' directive");
    Lex();
  }

  int64_t ColumnPos = 0;
  if (getLexer().is(AsmToken::Integer)) {
    ColumnPos = getTok().getIntVal();
    if (ColumnPos < 0)
      return TokError("column position less than zero in '.loc' directive");
    Lex();
  }

  unsigned Flags = DWARF2_LINE_DEFAULT_IS_STMT ? DWARF2_FLAG_IS_STMT : 0;
  unsigned Isa = 0;
  int64_t Discriminator = 0;
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    for (;;) {
      if (getLexer().is(AsmToken::EndOfStatement))
        break;

      StringRef Name;
      SMLoc Loc = getTok().getLoc();
      if (getParser().ParseIdentifier(Name))
        return TokError("unexpected token in '.loc' directive");

      if (Name == "basic_block")
        Flags |= DWARF2_FLAG_BASIC_BLOCK;
      else if (Name == "prologue_end")
        Flags |= DWARF2_FLAG_PROLOGUE_END;
      else if (Name == "epilogue_begin")
        Flags |= DWARF2_FLAG_EPILOGUE_BEGIN;
      else if (Name == "is_stmt") {
        SMLoc Loc = getTok().getLoc();
        const MCExpr *Value;
        if (getParser().ParseExpression(Value))
          return true;
        // The expression must be the constant 0 or 1.
        if (const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(Value)) {
          int Value = MCE->getValue();
          if (Value == 0)
            Flags &= ~DWARF2_FLAG_IS_STMT;
          else if (Value == 1)
            Flags |= DWARF2_FLAG_IS_STMT;
          else
            return Error(Loc, "is_stmt value not 0 or 1");
        }
        else {
          return Error(Loc, "is_stmt value not the constant value of 0 or 1");
        }
      }
      else if (Name == "isa") {
        SMLoc Loc = getTok().getLoc();
        const MCExpr *Value;
        if (getParser().ParseExpression(Value))
          return true;
        // The expression must be a constant greater or equal to 0.
        if (const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(Value)) {
          int Value = MCE->getValue();
          if (Value < 0)
            return Error(Loc, "isa number less than zero");
          Isa = Value;
        }
        else {
          return Error(Loc, "isa number not a constant value");
        }
      }
      else if (Name == "discriminator") {
        if (getParser().ParseAbsoluteExpression(Discriminator))
          return true;
      }
      else {
        return Error(Loc, "unknown sub-directive in '.loc' directive");
      }

      if (getLexer().is(AsmToken::EndOfStatement))
        break;
    }
  }

  getStreamer().EmitDwarfLocDirective(FileNumber, LineNumber, ColumnPos, Flags,
                                      Isa, Discriminator, StringRef());

  return false;
}

/// ParseDirectiveStabs
/// ::= .stabs string, number, number, number
bool GenericAsmParser::ParseDirectiveStabs(StringRef Directive,
                                           SMLoc DirectiveLoc) {
  return TokError("unsupported directive '" + Directive + "'");
}

/// ParseDirectiveCFISections
/// ::= .cfi_sections section [, section]
bool GenericAsmParser::ParseDirectiveCFISections(StringRef,
                                                 SMLoc DirectiveLoc) {
  StringRef Name;
  bool EH = false;
  bool Debug = false;

  if (getParser().ParseIdentifier(Name))
    return TokError("Expected an identifier");

  if (Name == ".eh_frame")
    EH = true;
  else if (Name == ".debug_frame")
    Debug = true;

  if (getLexer().is(AsmToken::Comma)) {
    Lex();

    if (getParser().ParseIdentifier(Name))
      return TokError("Expected an identifier");

    if (Name == ".eh_frame")
      EH = true;
    else if (Name == ".debug_frame")
      Debug = true;
  }

  getStreamer().EmitCFISections(EH, Debug);

  return false;
}

/// ParseDirectiveCFIStartProc
/// ::= .cfi_startproc
bool GenericAsmParser::ParseDirectiveCFIStartProc(StringRef,
                                                  SMLoc DirectiveLoc) {
  getStreamer().EmitCFIStartProc();
  return false;
}

/// ParseDirectiveCFIEndProc
/// ::= .cfi_endproc
bool GenericAsmParser::ParseDirectiveCFIEndProc(StringRef, SMLoc DirectiveLoc) {
  getStreamer().EmitCFIEndProc();
  return false;
}

/// ParseRegisterOrRegisterNumber - parse register name or number.
bool GenericAsmParser::ParseRegisterOrRegisterNumber(int64_t &Register,
                                                     SMLoc DirectiveLoc) {
  unsigned RegNo;

  if (getLexer().isNot(AsmToken::Integer)) {
    if (getParser().getTargetParser().ParseRegister(RegNo, DirectiveLoc,
      DirectiveLoc))
      return true;
    Register = getContext().getRegisterInfo().getDwarfRegNum(RegNo, true);
  } else
    return getParser().ParseAbsoluteExpression(Register);

  return false;
}

/// ParseDirectiveCFIDefCfa
/// ::= .cfi_def_cfa register,  offset
bool GenericAsmParser::ParseDirectiveCFIDefCfa(StringRef,
                                               SMLoc DirectiveLoc) {
  int64_t Register = 0;
  if (ParseRegisterOrRegisterNumber(Register, DirectiveLoc))
    return true;

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lex();

  int64_t Offset = 0;
  if (getParser().ParseAbsoluteExpression(Offset))
    return true;

  getStreamer().EmitCFIDefCfa(Register, Offset);
  return false;
}

/// ParseDirectiveCFIDefCfaOffset
/// ::= .cfi_def_cfa_offset offset
bool GenericAsmParser::ParseDirectiveCFIDefCfaOffset(StringRef,
                                                     SMLoc DirectiveLoc) {
  int64_t Offset = 0;
  if (getParser().ParseAbsoluteExpression(Offset))
    return true;

  getStreamer().EmitCFIDefCfaOffset(Offset);
  return false;
}

/// ParseDirectiveCFIAdjustCfaOffset
/// ::= .cfi_adjust_cfa_offset adjustment
bool GenericAsmParser::ParseDirectiveCFIAdjustCfaOffset(StringRef,
                                                        SMLoc DirectiveLoc) {
  int64_t Adjustment = 0;
  if (getParser().ParseAbsoluteExpression(Adjustment))
    return true;

  getStreamer().EmitCFIAdjustCfaOffset(Adjustment);
  return false;
}

/// ParseDirectiveCFIDefCfaRegister
/// ::= .cfi_def_cfa_register register
bool GenericAsmParser::ParseDirectiveCFIDefCfaRegister(StringRef,
                                                       SMLoc DirectiveLoc) {
  int64_t Register = 0;
  if (ParseRegisterOrRegisterNumber(Register, DirectiveLoc))
    return true;

  getStreamer().EmitCFIDefCfaRegister(Register);
  return false;
}

/// ParseDirectiveCFIOffset
/// ::= .cfi_offset register, offset
bool GenericAsmParser::ParseDirectiveCFIOffset(StringRef, SMLoc DirectiveLoc) {
  int64_t Register = 0;
  int64_t Offset = 0;

  if (ParseRegisterOrRegisterNumber(Register, DirectiveLoc))
    return true;

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lex();

  if (getParser().ParseAbsoluteExpression(Offset))
    return true;

  getStreamer().EmitCFIOffset(Register, Offset);
  return false;
}

/// ParseDirectiveCFIRelOffset
/// ::= .cfi_rel_offset register, offset
bool GenericAsmParser::ParseDirectiveCFIRelOffset(StringRef,
                                                  SMLoc DirectiveLoc) {
  int64_t Register = 0;

  if (ParseRegisterOrRegisterNumber(Register, DirectiveLoc))
    return true;

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lex();

  int64_t Offset = 0;
  if (getParser().ParseAbsoluteExpression(Offset))
    return true;

  getStreamer().EmitCFIRelOffset(Register, Offset);
  return false;
}

static bool isValidEncoding(int64_t Encoding) {
  if (Encoding & ~0xff)
    return false;

  if (Encoding == dwarf::DW_EH_PE_omit)
    return true;

  const unsigned Format = Encoding & 0xf;
  if (Format != dwarf::DW_EH_PE_absptr && Format != dwarf::DW_EH_PE_udata2 &&
      Format != dwarf::DW_EH_PE_udata4 && Format != dwarf::DW_EH_PE_udata8 &&
      Format != dwarf::DW_EH_PE_sdata2 && Format != dwarf::DW_EH_PE_sdata4 &&
      Format != dwarf::DW_EH_PE_sdata8 && Format != dwarf::DW_EH_PE_signed)
    return false;

  const unsigned Application = Encoding & 0x70;
  if (Application != dwarf::DW_EH_PE_absptr &&
      Application != dwarf::DW_EH_PE_pcrel)
    return false;

  return true;
}

/// ParseDirectiveCFIPersonalityOrLsda
/// ::= .cfi_personality encoding, [symbol_name]
/// ::= .cfi_lsda encoding, [symbol_name]
bool GenericAsmParser::ParseDirectiveCFIPersonalityOrLsda(StringRef IDVal,
                                                    SMLoc DirectiveLoc) {
  int64_t Encoding = 0;
  if (getParser().ParseAbsoluteExpression(Encoding))
    return true;
  if (Encoding == dwarf::DW_EH_PE_omit)
    return false;

  if (!isValidEncoding(Encoding))
    return TokError("unsupported encoding.");

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lex();

  StringRef Name;
  if (getParser().ParseIdentifier(Name))
    return TokError("expected identifier in directive");

  MCSymbol *Sym = getContext().GetOrCreateSymbol(Name);

  if (IDVal == ".cfi_personality")
    getStreamer().EmitCFIPersonality(Sym, Encoding);
  else {
    assert(IDVal == ".cfi_lsda");
    getStreamer().EmitCFILsda(Sym, Encoding);
  }
  return false;
}

/// ParseDirectiveCFIRememberState
/// ::= .cfi_remember_state
bool GenericAsmParser::ParseDirectiveCFIRememberState(StringRef IDVal,
                                                      SMLoc DirectiveLoc) {
  getStreamer().EmitCFIRememberState();
  return false;
}

/// ParseDirectiveCFIRestoreState
/// ::= .cfi_remember_state
bool GenericAsmParser::ParseDirectiveCFIRestoreState(StringRef IDVal,
                                                     SMLoc DirectiveLoc) {
  getStreamer().EmitCFIRestoreState();
  return false;
}

/// ParseDirectiveCFISameValue
/// ::= .cfi_same_value register
bool GenericAsmParser::ParseDirectiveCFISameValue(StringRef IDVal,
                                                  SMLoc DirectiveLoc) {
  int64_t Register = 0;

  if (ParseRegisterOrRegisterNumber(Register, DirectiveLoc))
    return true;

  getStreamer().EmitCFISameValue(Register);

  return false;
}

/// ParseDirectiveCFIRestore
/// ::= .cfi_restore register
bool GenericAsmParser::ParseDirectiveCFIRestore(StringRef IDVal,
                                                SMLoc DirectiveLoc) {
  int64_t Register = 0;
  if (ParseRegisterOrRegisterNumber(Register, DirectiveLoc))
    return true;

  getStreamer().EmitCFIRestore(Register);

  return false;
}

/// ParseDirectiveCFIEscape
/// ::= .cfi_escape expression[,...]
bool GenericAsmParser::ParseDirectiveCFIEscape(StringRef IDVal,
                                               SMLoc DirectiveLoc) {
  std::string Values;
  int64_t CurrValue;
  if (getParser().ParseAbsoluteExpression(CurrValue))
    return true;

  Values.push_back((uint8_t)CurrValue);

  while (getLexer().is(AsmToken::Comma)) {
    Lex();

    if (getParser().ParseAbsoluteExpression(CurrValue))
      return true;

    Values.push_back((uint8_t)CurrValue);
  }

  getStreamer().EmitCFIEscape(Values);
  return false;
}

/// ParseDirectiveCFISignalFrame
/// ::= .cfi_signal_frame
bool GenericAsmParser::ParseDirectiveCFISignalFrame(StringRef Directive,
                                                    SMLoc DirectiveLoc) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return Error(getLexer().getLoc(),
                 "unexpected token in '" + Directive + "' directive");

  getStreamer().EmitCFISignalFrame();

  return false;
}

/// ParseDirectiveMacrosOnOff
/// ::= .macros_on
/// ::= .macros_off
bool GenericAsmParser::ParseDirectiveMacrosOnOff(StringRef Directive,
                                                 SMLoc DirectiveLoc) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return Error(getLexer().getLoc(),
                 "unexpected token in '" + Directive + "' directive");

  getParser().MacrosEnabled = Directive == ".macros_on";

  return false;
}

/// ParseDirectiveMacro
/// ::= .macro name [parameters]
bool GenericAsmParser::ParseDirectiveMacro(StringRef Directive,
                                           SMLoc DirectiveLoc) {
  StringRef Name;
  if (getParser().ParseIdentifier(Name))
    return TokError("expected identifier in '.macro' directive");

  MacroParameters Parameters;
  // Argument delimiter is initially unknown. It will be set by
  // ParseMacroArgument()
  AsmToken::TokenKind ArgumentDelimiter = AsmToken::Eof;
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    for (;;) {
      MacroParameter Parameter;
      if (getParser().ParseIdentifier(Parameter.first))
        return TokError("expected identifier in '.macro' directive");

      if (getLexer().is(AsmToken::Equal)) {
        Lex();
        if (getParser().ParseMacroArgument(Parameter.second, ArgumentDelimiter))
          return true;
      }

      Parameters.push_back(Parameter);

      if (getLexer().is(AsmToken::Comma))
        Lex();
      else if (getLexer().is(AsmToken::EndOfStatement))
        break;
    }
  }

  // Eat the end of statement.
  Lex();

  AsmToken EndToken, StartToken = getTok();

  // Lex the macro definition.
  for (;;) {
    // Check whether we have reached the end of the file.
    if (getLexer().is(AsmToken::Eof))
      return Error(DirectiveLoc, "no matching '.endmacro' in definition");

    // Otherwise, check whether we have reach the .endmacro.
    if (getLexer().is(AsmToken::Identifier) &&
        (getTok().getIdentifier() == ".endm" ||
         getTok().getIdentifier() == ".endmacro")) {
      EndToken = getTok();
      Lex();
      if (getLexer().isNot(AsmToken::EndOfStatement))
        return TokError("unexpected token in '" + EndToken.getIdentifier() +
                        "' directive");
      break;
    }

    // Otherwise, scan til the end of the statement.
    getParser().EatToEndOfStatement();
  }

  if (getParser().MacroMap.lookup(Name)) {
    return Error(DirectiveLoc, "macro '" + Name + "' is already defined");
  }

  const char *BodyStart = StartToken.getLoc().getPointer();
  const char *BodyEnd = EndToken.getLoc().getPointer();
  StringRef Body = StringRef(BodyStart, BodyEnd - BodyStart);
  getParser().MacroMap[Name] = new Macro(Name, Body, Parameters);
  return false;
}

/// ParseDirectiveEndMacro
/// ::= .endm
/// ::= .endmacro
bool GenericAsmParser::ParseDirectiveEndMacro(StringRef Directive,
                                              SMLoc DirectiveLoc) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '" + Directive + "' directive");

  // If we are inside a macro instantiation, terminate the current
  // instantiation.
  if (!getParser().ActiveMacros.empty()) {
    getParser().HandleMacroExit();
    return false;
  }

  // Otherwise, this .endmacro is a stray entry in the file; well formed
  // .endmacro directives are handled during the macro definition parsing.
  return TokError("unexpected '" + Directive + "' in file, "
                  "no current macro definition");
}

/// ParseDirectivePurgeMacro
/// ::= .purgem
bool GenericAsmParser::ParseDirectivePurgeMacro(StringRef Directive,
                                                SMLoc DirectiveLoc) {
  StringRef Name;
  if (getParser().ParseIdentifier(Name))
    return TokError("expected identifier in '.purgem' directive");

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.purgem' directive");

  StringMap<Macro*>::iterator I = getParser().MacroMap.find(Name);
  if (I == getParser().MacroMap.end())
    return Error(DirectiveLoc, "macro '" + Name + "' is not defined");

  // Undefine the macro.
  delete I->getValue();
  getParser().MacroMap.erase(I);
  return false;
}

bool GenericAsmParser::ParseDirectiveLEB128(StringRef DirName, SMLoc) {
  getParser().CheckForValidSection();

  const MCExpr *Value;

  if (getParser().ParseExpression(Value))
    return true;

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in directive");

  if (DirName[1] == 's')
    getStreamer().EmitSLEB128Value(Value);
  else
    getStreamer().EmitULEB128Value(Value);

  return false;
}

Macro *AsmParser::ParseMacroLikeBody(SMLoc DirectiveLoc) {
  AsmToken EndToken, StartToken = getTok();

  unsigned NestLevel = 0;
  for (;;) {
    // Check whether we have reached the end of the file.
    if (getLexer().is(AsmToken::Eof)) {
      Error(DirectiveLoc, "no matching '.endr' in definition");
      return 0;
    }

    if (Lexer.is(AsmToken::Identifier) &&
        (getTok().getIdentifier() == ".rept")) {
      ++NestLevel;
    }

    // Otherwise, check whether we have reached the .endr.
    if (Lexer.is(AsmToken::Identifier) &&
        getTok().getIdentifier() == ".endr") {
      if (NestLevel == 0) {
        EndToken = getTok();
        Lex();
        if (Lexer.isNot(AsmToken::EndOfStatement)) {
          TokError("unexpected token in '.endr' directive");
          return 0;
        }
        break;
      }
      --NestLevel;
    }

    // Otherwise, scan till the end of the statement.
    EatToEndOfStatement();
  }

  const char *BodyStart = StartToken.getLoc().getPointer();
  const char *BodyEnd = EndToken.getLoc().getPointer();
  StringRef Body = StringRef(BodyStart, BodyEnd - BodyStart);

  // We Are Anonymous.
  StringRef Name;
  MacroParameters Parameters;
  return new Macro(Name, Body, Parameters);
}

void AsmParser::InstantiateMacroLikeBody(Macro *M, SMLoc DirectiveLoc,
                                         raw_svector_ostream &OS) {
  OS << ".endr\n";

  MemoryBuffer *Instantiation =
    MemoryBuffer::getMemBufferCopy(OS.str(), "<instantiation>");

  // Create the macro instantiation object and add to the current macro
  // instantiation stack.
  MacroInstantiation *MI = new MacroInstantiation(M, DirectiveLoc,
                                                  getTok().getLoc(),
                                                  Instantiation);
  ActiveMacros.push_back(MI);

  // Jump to the macro instantiation and prime the lexer.
  CurBuffer = SrcMgr.AddNewSourceBuffer(MI->Instantiation, SMLoc());
  Lexer.setBuffer(SrcMgr.getMemoryBuffer(CurBuffer));
  Lex();
}

bool AsmParser::ParseDirectiveRept(SMLoc DirectiveLoc) {
  int64_t Count;
  if (ParseAbsoluteExpression(Count))
    return TokError("unexpected token in '.rept' directive");

  if (Count < 0)
    return TokError("Count is negative");

  if (Lexer.isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.rept' directive");

  // Eat the end of statement.
  Lex();

  // Lex the rept definition.
  Macro *M = ParseMacroLikeBody(DirectiveLoc);
  if (!M)
    return true;

  // Macro instantiation is lexical, unfortunately. We construct a new buffer
  // to hold the macro body with substitutions.
  SmallString<256> Buf;
  MacroParameters Parameters;
  MacroArguments A;
  raw_svector_ostream OS(Buf);
  while (Count--) {
    if (expandMacro(OS, M->Body, Parameters, A, getTok().getLoc()))
      return true;
  }
  InstantiateMacroLikeBody(M, DirectiveLoc, OS);

  return false;
}

/// ParseDirectiveIrp
/// ::= .irp symbol,values
bool AsmParser::ParseDirectiveIrp(SMLoc DirectiveLoc) {
  MacroParameters Parameters;
  MacroParameter Parameter;

  if (ParseIdentifier(Parameter.first))
    return TokError("expected identifier in '.irp' directive");

  Parameters.push_back(Parameter);

  if (Lexer.isNot(AsmToken::Comma))
    return TokError("expected comma in '.irp' directive");

  Lex();

  MacroArguments A;
  if (ParseMacroArguments(0, A))
    return true;

  // Eat the end of statement.
  Lex();

  // Lex the irp definition.
  Macro *M = ParseMacroLikeBody(DirectiveLoc);
  if (!M)
    return true;

  // Macro instantiation is lexical, unfortunately. We construct a new buffer
  // to hold the macro body with substitutions.
  SmallString<256> Buf;
  raw_svector_ostream OS(Buf);

  for (MacroArguments::iterator i = A.begin(), e = A.end(); i != e; ++i) {
    MacroArguments Args;
    Args.push_back(*i);

    if (expandMacro(OS, M->Body, Parameters, Args, getTok().getLoc()))
      return true;
  }

  InstantiateMacroLikeBody(M, DirectiveLoc, OS);

  return false;
}

/// ParseDirectiveIrpc
/// ::= .irpc symbol,values
bool AsmParser::ParseDirectiveIrpc(SMLoc DirectiveLoc) {
  MacroParameters Parameters;
  MacroParameter Parameter;

  if (ParseIdentifier(Parameter.first))
    return TokError("expected identifier in '.irpc' directive");

  Parameters.push_back(Parameter);

  if (Lexer.isNot(AsmToken::Comma))
    return TokError("expected comma in '.irpc' directive");

  Lex();

  MacroArguments A;
  if (ParseMacroArguments(0, A))
    return true;

  if (A.size() != 1 || A.front().size() != 1)
    return TokError("unexpected token in '.irpc' directive");

  // Eat the end of statement.
  Lex();

  // Lex the irpc definition.
  Macro *M = ParseMacroLikeBody(DirectiveLoc);
  if (!M)
    return true;

  // Macro instantiation is lexical, unfortunately. We construct a new buffer
  // to hold the macro body with substitutions.
  SmallString<256> Buf;
  raw_svector_ostream OS(Buf);

  StringRef Values = A.front().front().getString();
  std::size_t I, End = Values.size();
  for (I = 0; I < End; ++I) {
    MacroArgument Arg;
    Arg.push_back(AsmToken(AsmToken::Identifier, Values.slice(I, I+1)));

    MacroArguments Args;
    Args.push_back(Arg);

    if (expandMacro(OS, M->Body, Parameters, Args, getTok().getLoc()))
      return true;
  }

  InstantiateMacroLikeBody(M, DirectiveLoc, OS);

  return false;
}

bool AsmParser::ParseDirectiveEndr(SMLoc DirectiveLoc) {
  if (ActiveMacros.empty())
    return TokError("unmatched '.endr' directive");

  // The only .repl that should get here are the ones created by
  // InstantiateMacroLikeBody.
  assert(getLexer().is(AsmToken::EndOfStatement));

  HandleMacroExit();
  return false;
}

bool AsmParser::ParseDirectiveEmit(SMLoc IDLoc, ParseStatementInfo &Info) {
  const MCExpr *Value;
  SMLoc ExprLoc = getLexer().getLoc();
  if (ParseExpression(Value))
    return true;
  const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(Value);
  if (!MCE)
    return Error(ExprLoc, "unexpected expression in _emit");
  uint64_t IntValue = MCE->getValue();
  if (!isUIntN(8, IntValue) && !isIntN(8, IntValue))
    return Error(ExprLoc, "literal value out of range for directive");

  Info.AsmRewrites->push_back(AsmRewrite(AOK_Emit, IDLoc, 5));
  return false;
}

bool AsmParser::ParseMSInlineAsm(void *AsmLoc, std::string &AsmString,
                                 unsigned &NumOutputs, unsigned &NumInputs,
                                 SmallVectorImpl<std::pair<void *, bool> > &OpDecls,
                                 SmallVectorImpl<std::string> &Constraints,
                                 SmallVectorImpl<std::string> &Clobbers,
                                 const MCInstrInfo *MII,
                                 const MCInstPrinter *IP,
                                 MCAsmParserSemaCallback &SI) {
  SmallVector<void *, 4> InputDecls;
  SmallVector<void *, 4> OutputDecls;
  SmallVector<bool, 4> InputDeclsOffsetOf;
  SmallVector<bool, 4> OutputDeclsOffsetOf;
  SmallVector<std::string, 4> InputConstraints;
  SmallVector<std::string, 4> OutputConstraints;
  std::set<std::string> ClobberRegs;

  SmallVector<struct AsmRewrite, 4> AsmStrRewrites;

  // Prime the lexer.
  Lex();

  // While we have input, parse each statement.
  unsigned InputIdx = 0;
  unsigned OutputIdx = 0;
  while (getLexer().isNot(AsmToken::Eof)) {
    ParseStatementInfo Info(&AsmStrRewrites);
    if (ParseStatement(Info))
      return true;

    if (Info.Opcode != ~0U) {
      const MCInstrDesc &Desc = MII->get(Info.Opcode);

      // Build the list of clobbers, outputs and inputs.
      for (unsigned i = 1, e = Info.ParsedOperands.size(); i != e; ++i) {
        MCParsedAsmOperand *Operand = Info.ParsedOperands[i];

        // Immediate.
        if (Operand->isImm()) {
          if (Operand->needAsmRewrite())
            AsmStrRewrites.push_back(AsmRewrite(AOK_ImmPrefix,
                                                Operand->getStartLoc()));
          continue;
        }

        // Register operand.
        if (Operand->isReg() && !Operand->isOffsetOf()) {
          unsigned NumDefs = Desc.getNumDefs();
          // Clobber.
          if (NumDefs && Operand->getMCOperandNum() < NumDefs) {
            std::string Reg;
            raw_string_ostream OS(Reg);
            IP->printRegName(OS, Operand->getReg());
            ClobberRegs.insert(StringRef(OS.str()));
          }
          continue;
        }

        // Expr/Input or Output.
        unsigned Size;
        void *OpDecl = SI.LookupInlineAsmIdentifier(Operand->getName(), AsmLoc,
                                                    Size);
        if (OpDecl) {
          bool isOutput = (i == 1) && Desc.mayStore();
          if (!Operand->isOffsetOf() && Operand->needSizeDirective())
            AsmStrRewrites.push_back(AsmRewrite(AOK_SizeDirective,
                                                Operand->getStartLoc(),
                                                /*Len*/0,
                                                Operand->getMemSize()));
          if (isOutput) {
            std::string Constraint = "=";
            ++InputIdx;
            OutputDecls.push_back(OpDecl);
            OutputDeclsOffsetOf.push_back(Operand->isOffsetOf());
            Constraint += Operand->getConstraint().str();
            OutputConstraints.push_back(Constraint);
            AsmStrRewrites.push_back(AsmRewrite(AOK_Output,
                                                Operand->getStartLoc(),
                                                Operand->getNameLen()));
          } else {
            InputDecls.push_back(OpDecl);
            InputDeclsOffsetOf.push_back(Operand->isOffsetOf());
            InputConstraints.push_back(Operand->getConstraint().str());
            AsmStrRewrites.push_back(AsmRewrite(AOK_Input,
                                                Operand->getStartLoc(),
                                                Operand->getNameLen()));
          }
        }
      }
    }
  }

  // Set the number of Outputs and Inputs.
  NumOutputs = OutputDecls.size();
  NumInputs = InputDecls.size();

  // Set the unique clobbers.
  for (std::set<std::string>::iterator I = ClobberRegs.begin(),
         E = ClobberRegs.end(); I != E; ++I)
    Clobbers.push_back(*I);

  // Merge the various outputs and inputs.  Output are expected first.
  if (NumOutputs || NumInputs) {
    unsigned NumExprs = NumOutputs + NumInputs;
    OpDecls.resize(NumExprs);
    Constraints.resize(NumExprs);
    // FIXME: Constraints are hard coded to 'm', but we need an 'r'
    // constraint for offsetof.  This needs to be cleaned up!
    for (unsigned i = 0; i < NumOutputs; ++i) {
      OpDecls[i] = std::make_pair(OutputDecls[i], OutputDeclsOffsetOf[i]);
      Constraints[i] = OutputDeclsOffsetOf[i] ? "=r" : OutputConstraints[i];
    }
    for (unsigned i = 0, j = NumOutputs; i < NumInputs; ++i, ++j) {
      OpDecls[j] = std::make_pair(InputDecls[i], InputDeclsOffsetOf[i]);
      Constraints[j] = InputDeclsOffsetOf[i] ? "r" : InputConstraints[i];
    }
  }

  // Build the IR assembly string.
  std::string AsmStringIR;
  AsmRewriteKind PrevKind = AOK_Imm;
  raw_string_ostream OS(AsmStringIR);
  const char *Start = SrcMgr.getMemoryBuffer(0)->getBufferStart();
  for (SmallVectorImpl<struct AsmRewrite>::iterator
         I = AsmStrRewrites.begin(), E = AsmStrRewrites.end(); I != E; ++I) {
    const char *Loc = (*I).Loc.getPointer();

    AsmRewriteKind Kind = (*I).Kind;

    // Emit everything up to the immediate/expression.  If the previous rewrite
    // was a size directive, then this has already been done.
    if (PrevKind != AOK_SizeDirective)
      OS << StringRef(Start, Loc - Start);
    PrevKind = Kind;

    // Skip the original expression.
    if (Kind == AOK_Skip) {
      Start = Loc + (*I).Len;
      continue;
    }

    // Rewrite expressions in $N notation.
    switch (Kind) {
    default: break;
    case AOK_Imm:
      OS << Twine("$$");
      OS << (*I).Val;
      break;
    case AOK_ImmPrefix:
      OS << Twine("$$");
      break;
    case AOK_Input:
      OS << '$';
      OS << InputIdx++;
      break;
    case AOK_Output:
      OS << '$';
      OS << OutputIdx++;
      break;
    case AOK_SizeDirective:
      switch((*I).Val) {
      default: break;
      case 8:  OS << "byte ptr "; break;
      case 16: OS << "word ptr "; break;
      case 32: OS << "dword ptr "; break;
      case 64: OS << "qword ptr "; break;
      case 80: OS << "xword ptr "; break;
      case 128: OS << "xmmword ptr "; break;
      case 256: OS << "ymmword ptr "; break;
      }
      break;
    case AOK_Emit:
      OS << ".byte";
      break;
    case AOK_DotOperator:
      OS << (*I).Val;
      break;
    }

    // Skip the original expression.
    if (Kind != AOK_SizeDirective)
      Start = Loc + (*I).Len;
  }

  // Emit the remainder of the asm string.
  const char *AsmEnd = SrcMgr.getMemoryBuffer(0)->getBufferEnd();
  if (Start != AsmEnd)
    OS << StringRef(Start, AsmEnd - Start);

  AsmString = OS.str();
  return false;
}

/// \brief Create an MCAsmParser instance.
MCAsmParser *llvm::createMCAsmParser(SourceMgr &SM,
                                     MCContext &C, MCStreamer &Out,
                                     const MCAsmInfo &MAI) {
  return new AsmParser(SM, C, Out, MAI);
}
