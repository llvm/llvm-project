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
#include "llvm/ADT/STLExtras.h"
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

/// \brief Helper types for tracking macro definitions.
typedef std::vector<AsmToken> MCAsmMacroArgument;
typedef std::vector<MCAsmMacroArgument> MCAsmMacroArguments;
typedef std::pair<StringRef, MCAsmMacroArgument> MCAsmMacroParameter;
typedef std::vector<MCAsmMacroParameter> MCAsmMacroParameters;

struct MCAsmMacro {
  StringRef Name;
  StringRef Body;
  MCAsmMacroParameters Parameters;

public:
  MCAsmMacro(StringRef N, StringRef B, const MCAsmMacroParameters &P) :
    Name(N), Body(B), Parameters(P) {}

  MCAsmMacro(const MCAsmMacro& Other)
    : Name(Other.Name), Body(Other.Body), Parameters(Other.Parameters) {}
};

/// \brief Helper class for storing information about an active macro
/// instantiation.
struct MacroInstantiation {
  /// The macro being instantiated.
  const MCAsmMacro *TheMacro;

  /// The macro instantiation with substitutions.
  MemoryBuffer *Instantiation;

  /// The location of the instantiation.
  SMLoc InstantiationLoc;

  /// The buffer where parsing should resume upon instantiation completion.
  int ExitBuffer;

  /// The location where parsing should resume upon instantiation completion.
  SMLoc ExitLoc;

public:
  MacroInstantiation(const MCAsmMacro *M, SMLoc IL, int EB, SMLoc EL,
                     MemoryBuffer *I);
};

struct ParseStatementInfo {
  /// ParsedOperands - The parsed operands from the last parsed statement.
  SmallVector<MCParsedAsmOperand*, 8> ParsedOperands;

  /// Opcode - The opcode from the last parsed instruction.
  unsigned Opcode;

  /// Error - Was there an error parsing the inline assembly?
  bool ParseError;

  SmallVectorImpl<AsmRewrite> *AsmRewrites;

  ParseStatementInfo() : Opcode(~0U), ParseError(false), AsmRewrites(0) {}
  ParseStatementInfo(SmallVectorImpl<AsmRewrite> *rewrites)
    : Opcode(~0), ParseError(false), AsmRewrites(rewrites) {}

  ~ParseStatementInfo() {
    // Free any parsed operands.
    for (unsigned i = 0, e = ParsedOperands.size(); i != e; ++i)
      delete ParsedOperands[i];
    ParsedOperands.clear();
  }
};

/// \brief The concrete assembly parser instance.
class AsmParser : public MCAsmParser {
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
  MCAsmParserExtension *PlatformParser;

  /// This is the current buffer index we're lexing from as managed by the
  /// SourceMgr object.
  int CurBuffer;

  AsmCond TheCondState;
  std::vector<AsmCond> TheCondStack;

  /// ExtensionDirectiveMap - maps directive names to handler methods in parser
  /// extensions. Extensions register themselves in this map by calling
  /// addDirectiveHandler.
  StringMap<ExtensionDirectiveHandler> ExtensionDirectiveMap;

  /// MacroMap - Map of currently defined macros.
  StringMap<MCAsmMacro*> MacroMap;

  /// ActiveMacros - Stack of active macro instantiations.
  std::vector<MacroInstantiation*> ActiveMacros;

  /// Boolean tracking whether macro substitution is enabled.
  unsigned MacrosEnabledFlag : 1;

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

  virtual void addDirectiveHandler(StringRef Directive,
                                   ExtensionDirectiveHandler Handler) {
    ExtensionDirectiveMap[Directive] = Handler;
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
                       ArrayRef<SMRange> Ranges = None);
  virtual bool Error(SMLoc L, const Twine &Msg,
                     ArrayRef<SMRange> Ranges = None);

  virtual const AsmToken &Lex();

  void setParsingInlineAsm(bool V) { ParsingInlineAsm = V; }
  bool isParsingInlineAsm() { return ParsingInlineAsm; }

  bool parseMSInlineAsm(void *AsmLoc, std::string &AsmString,
                        unsigned &NumOutputs, unsigned &NumInputs,
                        SmallVectorImpl<std::pair<void *,bool> > &OpDecls,
                        SmallVectorImpl<std::string> &Constraints,
                        SmallVectorImpl<std::string> &Clobbers,
                        const MCInstrInfo *MII,
                        const MCInstPrinter *IP,
                        MCAsmParserSemaCallback &SI);

  bool parseExpression(const MCExpr *&Res);
  virtual bool parseExpression(const MCExpr *&Res, SMLoc &EndLoc);
  virtual bool parsePrimaryExpr(const MCExpr *&Res, SMLoc &EndLoc);
  virtual bool parseParenExpression(const MCExpr *&Res, SMLoc &EndLoc);
  virtual bool parseAbsoluteExpression(int64_t &Res);

  /// parseIdentifier - Parse an identifier or string (as a quoted identifier)
  /// and set \p Res to the identifier contents.
  virtual bool parseIdentifier(StringRef &Res);
  virtual void eatToEndOfStatement();

  virtual void checkForValidSection();
  /// }

private:

  bool ParseStatement(ParseStatementInfo &Info);
  void EatToEndOfLine();
  bool ParseCppHashLineFilenameComment(const SMLoc &L);

  void CheckForBadMacro(SMLoc DirectiveLoc, StringRef Name, StringRef Body,
                        MCAsmMacroParameters Parameters);
  bool expandMacro(raw_svector_ostream &OS, StringRef Body,
                   const MCAsmMacroParameters &Parameters,
                   const MCAsmMacroArguments &A,
                   const SMLoc &L);

  /// \brief Are macros enabled in the parser?
  bool MacrosEnabled() {return MacrosEnabledFlag;}

  /// \brief Control a flag in the parser that enables or disables macros.
  void SetMacrosEnabled(bool Flag) {MacrosEnabledFlag = Flag;}

  /// \brief Lookup a previously defined macro.
  /// \param Name Macro name.
  /// \returns Pointer to macro. NULL if no such macro was defined.
  const MCAsmMacro* LookupMacro(StringRef Name);

  /// \brief Define a new macro with the given name and information.
  void DefineMacro(StringRef Name, const MCAsmMacro& Macro);

  /// \brief Undefine a macro. If no such macro was defined, it's a no-op.
  void UndefineMacro(StringRef Name);

  /// \brief Are we inside a macro instantiation?
  bool InsideMacroInstantiation() {return !ActiveMacros.empty();}

  /// \brief Handle entry to macro instantiation. 
  ///
  /// \param M The macro.
  /// \param NameLoc Instantiation location.
  bool HandleMacroEntry(const MCAsmMacro *M, SMLoc NameLoc);

  /// \brief Handle exit from macro instantiation.
  void HandleMacroExit();

  /// \brief Extract AsmTokens for a macro argument. If the argument delimiter
  /// is initially unknown, set it to AsmToken::Eof. It will be set to the
  /// correct delimiter by the method.
  bool ParseMacroArgument(MCAsmMacroArgument &MA,
                          AsmToken::TokenKind &ArgumentDelimiter);

  /// \brief Parse all macro arguments for a given macro.
  bool ParseMacroArguments(const MCAsmMacro *M, MCAsmMacroArguments &A);

  void PrintMacroInstantiations();
  void PrintMessage(SMLoc Loc, SourceMgr::DiagKind Kind, const Twine &Msg,
                    ArrayRef<SMRange> Ranges = None) const {
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
  ///
  /// \param InBuffer If not -1, should be the known buffer id that contains the
  /// location.
  void JumpToLoc(SMLoc Loc, int InBuffer=-1);

  /// \brief Parse up to the end of statement and a return the contents from the
  /// current token until the end of the statement; the current token on exit
  /// will be either the EndOfStatement or EOF.
  virtual StringRef parseStringToEndOfStatement();

  /// \brief Parse until the end of a statement or a comma is encountered,
  /// return the contents from the current token up to the end or comma.
  StringRef ParseStringToComma();

  bool ParseAssignment(StringRef Name, bool allow_redef,
                       bool NoDeadStrip = false);

  bool ParsePrimaryExpr(const MCExpr *&Res, SMLoc &EndLoc);
  bool ParseBinOpRHS(unsigned Precedence, const MCExpr *&Res, SMLoc &EndLoc);
  bool ParseParenExpr(const MCExpr *&Res, SMLoc &EndLoc);
  bool ParseBracketExpr(const MCExpr *&Res, SMLoc &EndLoc);

  bool ParseRegisterOrRegisterNumber(int64_t &Register, SMLoc DirectiveLoc);

  // Generic (target and platform independent) directive parsing.
  enum DirectiveKind {
    DK_NO_DIRECTIVE, // Placeholder
    DK_SET, DK_EQU, DK_EQUIV, DK_ASCII, DK_ASCIZ, DK_STRING, DK_BYTE, DK_SHORT,
    DK_VALUE, DK_2BYTE, DK_LONG, DK_INT, DK_4BYTE, DK_QUAD, DK_8BYTE, DK_SINGLE,
    DK_FLOAT, DK_DOUBLE, DK_ALIGN, DK_ALIGN32, DK_BALIGN, DK_BALIGNW,
    DK_BALIGNL, DK_P2ALIGN, DK_P2ALIGNW, DK_P2ALIGNL, DK_ORG, DK_FILL, DK_ENDR,
    DK_BUNDLE_ALIGN_MODE, DK_BUNDLE_LOCK, DK_BUNDLE_UNLOCK,
    DK_ZERO, DK_EXTERN, DK_GLOBL, DK_GLOBAL, DK_INDIRECT_SYMBOL,
    DK_LAZY_REFERENCE, DK_NO_DEAD_STRIP, DK_SYMBOL_RESOLVER, DK_PRIVATE_EXTERN,
    DK_REFERENCE, DK_WEAK_DEFINITION, DK_WEAK_REFERENCE,
    DK_WEAK_DEF_CAN_BE_HIDDEN, DK_COMM, DK_COMMON, DK_LCOMM, DK_ABORT,
    DK_INCLUDE, DK_INCBIN, DK_CODE16, DK_CODE16GCC, DK_REPT, DK_IRP, DK_IRPC,
    DK_IF, DK_IFB, DK_IFNB, DK_IFC, DK_IFNC, DK_IFDEF, DK_IFNDEF, DK_IFNOTDEF,
    DK_ELSEIF, DK_ELSE, DK_ENDIF,
    DK_SPACE, DK_SKIP, DK_FILE, DK_LINE, DK_LOC, DK_STABS,
    DK_CFI_SECTIONS, DK_CFI_STARTPROC, DK_CFI_ENDPROC, DK_CFI_DEF_CFA,
    DK_CFI_DEF_CFA_OFFSET, DK_CFI_ADJUST_CFA_OFFSET, DK_CFI_DEF_CFA_REGISTER,
    DK_CFI_OFFSET, DK_CFI_REL_OFFSET, DK_CFI_PERSONALITY, DK_CFI_LSDA,
    DK_CFI_REMEMBER_STATE, DK_CFI_RESTORE_STATE, DK_CFI_SAME_VALUE,
    DK_CFI_RESTORE, DK_CFI_ESCAPE, DK_CFI_SIGNAL_FRAME, DK_CFI_UNDEFINED,
    DK_CFI_REGISTER,
    DK_MACROS_ON, DK_MACROS_OFF, DK_MACRO, DK_ENDM, DK_ENDMACRO, DK_PURGEM,
    DK_SLEB128, DK_ULEB128
  };

  /// DirectiveKindMap - Maps directive name --> DirectiveKind enum, for
  /// directives parsed by this class.
  StringMap<DirectiveKind> DirectiveKindMap;

  // ".ascii", ".asciz", ".string"
  bool ParseDirectiveAscii(StringRef IDVal, bool ZeroTerminated);
  bool ParseDirectiveValue(unsigned Size); // ".byte", ".long", ...
  bool ParseDirectiveRealValue(const fltSemantics &); // ".single", ...
  bool ParseDirectiveFill(); // ".fill"
  bool ParseDirectiveZero(); // ".zero"
  // ".set", ".equ", ".equiv"
  bool ParseDirectiveSet(StringRef IDVal, bool allow_redef);
  bool ParseDirectiveOrg(); // ".org"
  // ".align{,32}", ".p2align{,w,l}"
  bool ParseDirectiveAlign(bool IsPow2, unsigned ValueSize);

  // ".file", ".line", ".loc", ".stabs"
  bool ParseDirectiveFile(SMLoc DirectiveLoc);
  bool ParseDirectiveLine();
  bool ParseDirectiveLoc();
  bool ParseDirectiveStabs();

  // .cfi directives
  bool ParseDirectiveCFIRegister(SMLoc DirectiveLoc);
  bool ParseDirectiveCFISections();
  bool ParseDirectiveCFIStartProc();
  bool ParseDirectiveCFIEndProc();
  bool ParseDirectiveCFIDefCfaOffset();
  bool ParseDirectiveCFIDefCfa(SMLoc DirectiveLoc);
  bool ParseDirectiveCFIAdjustCfaOffset();
  bool ParseDirectiveCFIDefCfaRegister(SMLoc DirectiveLoc);
  bool ParseDirectiveCFIOffset(SMLoc DirectiveLoc);
  bool ParseDirectiveCFIRelOffset(SMLoc DirectiveLoc);
  bool ParseDirectiveCFIPersonalityOrLsda(bool IsPersonality);
  bool ParseDirectiveCFIRememberState();
  bool ParseDirectiveCFIRestoreState();
  bool ParseDirectiveCFISameValue(SMLoc DirectiveLoc);
  bool ParseDirectiveCFIRestore(SMLoc DirectiveLoc);
  bool ParseDirectiveCFIEscape();
  bool ParseDirectiveCFISignalFrame();
  bool ParseDirectiveCFIUndefined(SMLoc DirectiveLoc);

  // macro directives
  bool ParseDirectivePurgeMacro(SMLoc DirectiveLoc);
  bool ParseDirectiveEndMacro(StringRef Directive);
  bool ParseDirectiveMacro(SMLoc DirectiveLoc);
  bool ParseDirectiveMacrosOnOff(StringRef Directive);

  // ".bundle_align_mode"
  bool ParseDirectiveBundleAlignMode();
  // ".bundle_lock"
  bool ParseDirectiveBundleLock();
  // ".bundle_unlock"
  bool ParseDirectiveBundleUnlock();

  // ".space", ".skip"
  bool ParseDirectiveSpace(StringRef IDVal);

  // .sleb128 (Signed=true) and .uleb128 (Signed=false)
  bool ParseDirectiveLEB128(bool Signed);

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
  virtual bool parseEscapedString(std::string &Data);

  const MCExpr *ApplyModifierToExpr(const MCExpr *E,
                                    MCSymbolRefExpr::VariantKind Variant);

  // Macro-like directives
  MCAsmMacro *ParseMacroLikeBody(SMLoc DirectiveLoc);
  void InstantiateMacroLikeBody(MCAsmMacro *M, SMLoc DirectiveLoc,
                                raw_svector_ostream &OS);
  bool ParseDirectiveRept(SMLoc DirectiveLoc); // ".rept"
  bool ParseDirectiveIrp(SMLoc DirectiveLoc);  // ".irp"
  bool ParseDirectiveIrpc(SMLoc DirectiveLoc); // ".irpc"
  bool ParseDirectiveEndr(SMLoc DirectiveLoc); // ".endr"

  // "_emit" or "__emit"
  bool ParseDirectiveMSEmit(SMLoc DirectiveLoc, ParseStatementInfo &Info,
                            size_t Len);

  // "align"
  bool ParseDirectiveMSAlign(SMLoc DirectiveLoc, ParseStatementInfo &Info);

  void initializeDirectiveKindMap();
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
    PlatformParser(0),
    CurBuffer(0), MacrosEnabledFlag(true), CppHashLineNumber(0),
    AssemblerDialect(~0U), IsDarwin(false), ParsingInlineAsm(false) {
  // Save the old handler.
  SavedDiagHandler = SrcMgr.getDiagHandler();
  SavedDiagContext = SrcMgr.getDiagContext();
  // Set our own handler which calls the saved handler.
  SrcMgr.setDiagHandler(DiagHandler, this);
  Lexer.setBuffer(SrcMgr.getMemoryBuffer(CurBuffer));

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

  initializeDirectiveKindMap();
}

AsmParser::~AsmParser() {
  assert(ActiveMacros.empty() && "Unexpected active macro instantiation!");

  // Destroy any macros.
  for (StringMap<MCAsmMacro*>::iterator it = MacroMap.begin(),
         ie = MacroMap.end(); it != ie; ++it)
    delete it->getValue();

  delete PlatformParser;
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

void AsmParser::JumpToLoc(SMLoc Loc, int InBuffer) {
  if (InBuffer != -1) {
    CurBuffer = InBuffer;
  } else {
    CurBuffer = SrcMgr.FindBufferContainingLoc(Loc);
  }
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
    getContext().setGenDwarfSection(getStreamer().getCurrentSection().first);
    MCSymbol *SectionStartSym = getContext().CreateTempSymbol();
    getStreamer().EmitLabel(SectionStartSym);
    getContext().setGenDwarfSectionStartSym(SectionStartSym);
    getStreamer().EmitDwarfFileDirective(getContext().nextGenDwarfFileNumber(),
                                         StringRef(),
                                         getContext().getMainFileName());
  }

  // While we have input, parse each statement.
  while (Lexer.isNot(AsmToken::Eof)) {
    ParseStatementInfo Info;
    if (!ParseStatement(Info)) continue;

    // We had an error, validate that one was emitted and recover by skipping to
    // the next line.
    assert(HadError && "Parse statement returned an error, but none emitted!");
    eatToEndOfStatement();
  }

  if (TheCondState.TheCond != StartingCondState.TheCond ||
      TheCondState.Ignore != StartingCondState.Ignore)
    return TokError("unmatched .ifs or .elses");

  // Check to see there are no empty DwarfFile slots.
  const SmallVectorImpl<MCDwarfFile *> &MCDwarfFiles =
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

void AsmParser::checkForValidSection() {
  if (!ParsingInlineAsm && !getStreamer().getCurrentSection().first) {
    TokError("expected section directive before assembly directive");
    Out.InitToTextSection();
  }
}

/// eatToEndOfStatement - Throw away the rest of the line for testing purposes.
void AsmParser::eatToEndOfStatement() {
  while (Lexer.isNot(AsmToken::EndOfStatement) &&
         Lexer.isNot(AsmToken::Eof))
    Lex();

  // Eat EOL.
  if (Lexer.is(AsmToken::EndOfStatement))
    Lex();
}

StringRef AsmParser::parseStringToEndOfStatement() {
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
  if (parseExpression(Res)) return true;
  if (Lexer.isNot(AsmToken::RParen))
    return TokError("expected ')' in parentheses expression");
  EndLoc = Lexer.getTok().getEndLoc();
  Lex();
  return false;
}

/// ParseBracketExpr - Parse a bracket expression and return it.
/// NOTE: This assumes the leading '[' has already been consumed.
///
/// bracketexpr ::= expr]
///
bool AsmParser::ParseBracketExpr(const MCExpr *&Res, SMLoc &EndLoc) {
  if (parseExpression(Res)) return true;
  if (Lexer.isNot(AsmToken::RBrac))
    return TokError("expected ']' in brackets expression");
  EndLoc = Lexer.getTok().getEndLoc();
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
  SMLoc FirstTokenLoc = getLexer().getLoc();
  AsmToken::TokenKind FirstTokenKind = Lexer.getKind();
  switch (FirstTokenKind) {
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
    StringRef Identifier;
    if (parseIdentifier(Identifier)) {
      if (FirstTokenKind == AsmToken::Dollar)
        return Error(FirstTokenLoc, "invalid token in expression");
      return true;
    }

    EndLoc = SMLoc::getFromPointer(Identifier.end());

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
    EndLoc = Lexer.getTok().getEndLoc();
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
        EndLoc = Lexer.getTok().getEndLoc();
        Lex(); // Eat identifier.
      }
    }
    return false;
  }
  case AsmToken::Real: {
    APFloat RealVal(APFloat::IEEEdouble, getTok().getString());
    uint64_t IntVal = RealVal.bitcastToAPInt().getZExtValue();
    Res = MCConstantExpr::Create(IntVal, getContext());
    EndLoc = Lexer.getTok().getEndLoc();
    Lex(); // Eat token.
    return false;
  }
  case AsmToken::Dot: {
    // This is a '.' reference, which references the current PC.  Emit a
    // temporary label to the streamer and refer to it.
    MCSymbol *Sym = Ctx.CreateTempSymbol();
    Out.EmitLabel(Sym);
    Res = MCSymbolRefExpr::Create(Sym, MCSymbolRefExpr::VK_None, getContext());
    EndLoc = Lexer.getTok().getEndLoc();
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

bool AsmParser::parseExpression(const MCExpr *&Res) {
  SMLoc EndLoc;
  return parseExpression(Res, EndLoc);
}

bool AsmParser::parsePrimaryExpr(const MCExpr *&Res, SMLoc &EndLoc) {
  return ParsePrimaryExpr(Res, EndLoc);
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

/// parseExpression - Parse an expression and return it.
///
///  expr ::= expr &&,|| expr               -> lowest.
///  expr ::= expr |,^,&,! expr
///  expr ::= expr ==,!=,<>,<,<=,>,>= expr
///  expr ::= expr <<,>> expr
///  expr ::= expr +,- expr
///  expr ::= expr *,/,% expr               -> highest.
///  expr ::= primaryexpr
///
bool AsmParser::parseExpression(const MCExpr *&Res, SMLoc &EndLoc) {
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

bool AsmParser::parseParenExpression(const MCExpr *&Res, SMLoc &EndLoc) {
  Res = 0;
  return ParseParenExpr(Res, EndLoc) ||
         ParseBinOpRHS(1, Res, EndLoc);
}

bool AsmParser::parseAbsoluteExpression(int64_t &Res) {
  const MCExpr *Expr;

  SMLoc StartLoc = Lexer.getLoc();
  if (parseExpression(Expr))
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
      if (ParseBinOpRHS(TokPrec+1, RHS, EndLoc)) return true;
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
    } else {
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
  } else if (parseIdentifier(IDVal)) {
    if (!TheCondState.Ignore)
      return TokError("unexpected token at start of statement");
    IDVal = "";
  }

  // Handle conditional assembly here before checking for skipping.  We
  // have to do this so that .endif isn't skipped in a ".if 0" block for
  // example.
  StringMap<DirectiveKind>::const_iterator DirKindIt =
    DirectiveKindMap.find(IDVal);
  DirectiveKind DirKind =
    (DirKindIt == DirectiveKindMap.end()) ? DK_NO_DIRECTIVE :
                                            DirKindIt->getValue();
  switch (DirKind) {
    default:
      break;
    case DK_IF:
      return ParseDirectiveIf(IDLoc);
    case DK_IFB:
      return ParseDirectiveIfb(IDLoc, true);
    case DK_IFNB:
      return ParseDirectiveIfb(IDLoc, false);
    case DK_IFC:
      return ParseDirectiveIfc(IDLoc, true);
    case DK_IFNC:
      return ParseDirectiveIfc(IDLoc, false);
    case DK_IFDEF:
      return ParseDirectiveIfdef(IDLoc, true);
    case DK_IFNDEF:
    case DK_IFNOTDEF:
      return ParseDirectiveIfdef(IDLoc, false);
    case DK_ELSEIF:
      return ParseDirectiveElseIf(IDLoc);
    case DK_ELSE:
      return ParseDirectiveElse(IDLoc);
    case DK_ENDIF:
      return ParseDirectiveEndIf(IDLoc);
  }

  // Ignore the statement if in the middle of inactive conditional
  // (e.g. ".if 0").
  if (TheCondState.Ignore) {
    eatToEndOfStatement();
    return false;
  }

  // FIXME: Recurse on local labels?

  // See what kind of statement we have.
  switch (Lexer.getKind()) {
  case AsmToken::Colon: {
    checkForValidSection();

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
    if (!ParsingInlineAsm)
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
  if (MacrosEnabled())
    if (const MCAsmMacro *M = LookupMacro(IDVal)) {
      return HandleMacroEntry(M, IDLoc);
    }

  // Otherwise, we have a normal instruction or directive.
  
  // Directives start with "."
  if (IDVal[0] == '.' && IDVal != ".") {
    // There are several entities interested in parsing directives:
    // 
    // 1. The target-specific assembly parser. Some directives are target
    //    specific or may potentially behave differently on certain targets.
    // 2. Asm parser extensions. For example, platform-specific parsers
    //    (like the ELF parser) register themselves as extensions.
    // 3. The generic directive parser implemented by this class. These are
    //    all the directives that behave in a target and platform independent
    //    manner, or at least have a default behavior that's shared between
    //    all targets and platforms.

    // First query the target-specific parser. It will return 'true' if it
    // isn't interested in this directive.
    if (!getTargetParser().ParseDirective(ID))
      return false;

    // Next, check the extention directive map to see if any extension has
    // registered itself to parse this directive.
    std::pair<MCAsmParserExtension*, DirectiveHandler> Handler =
      ExtensionDirectiveMap.lookup(IDVal);
    if (Handler.first)
      return (*Handler.second)(Handler.first, IDVal, IDLoc);

    // Finally, if no one else is interested in this directive, it must be
    // generic and familiar to this class.
    switch (DirKind) {
      default:
        break;
      case DK_SET:
      case DK_EQU:
        return ParseDirectiveSet(IDVal, true);
      case DK_EQUIV:
        return ParseDirectiveSet(IDVal, false);
      case DK_ASCII:
        return ParseDirectiveAscii(IDVal, false);
      case DK_ASCIZ:
      case DK_STRING:
        return ParseDirectiveAscii(IDVal, true);
      case DK_BYTE:
        return ParseDirectiveValue(1);
      case DK_SHORT:
      case DK_VALUE:
      case DK_2BYTE:
        return ParseDirectiveValue(2);
      case DK_LONG:
      case DK_INT:
      case DK_4BYTE:
        return ParseDirectiveValue(4);
      case DK_QUAD:
      case DK_8BYTE:
        return ParseDirectiveValue(8);
      case DK_SINGLE:
      case DK_FLOAT:
        return ParseDirectiveRealValue(APFloat::IEEEsingle);
      case DK_DOUBLE:
        return ParseDirectiveRealValue(APFloat::IEEEdouble);
      case DK_ALIGN: {
        bool IsPow2 = !getContext().getAsmInfo().getAlignmentIsInBytes();
        return ParseDirectiveAlign(IsPow2, /*ExprSize=*/1);
      }
      case DK_ALIGN32: {
        bool IsPow2 = !getContext().getAsmInfo().getAlignmentIsInBytes();
        return ParseDirectiveAlign(IsPow2, /*ExprSize=*/4);
      }
      case DK_BALIGN:
        return ParseDirectiveAlign(/*IsPow2=*/false, /*ExprSize=*/1);
      case DK_BALIGNW:
        return ParseDirectiveAlign(/*IsPow2=*/false, /*ExprSize=*/2);
      case DK_BALIGNL:
        return ParseDirectiveAlign(/*IsPow2=*/false, /*ExprSize=*/4);
      case DK_P2ALIGN:
        return ParseDirectiveAlign(/*IsPow2=*/true, /*ExprSize=*/1);
      case DK_P2ALIGNW:
        return ParseDirectiveAlign(/*IsPow2=*/true, /*ExprSize=*/2);
      case DK_P2ALIGNL:
        return ParseDirectiveAlign(/*IsPow2=*/true, /*ExprSize=*/4);
      case DK_ORG:
        return ParseDirectiveOrg();
      case DK_FILL:
        return ParseDirectiveFill();
      case DK_ZERO:
        return ParseDirectiveZero();
      case DK_EXTERN:
        eatToEndOfStatement(); // .extern is the default, ignore it.
        return false;
      case DK_GLOBL:
      case DK_GLOBAL:
        return ParseDirectiveSymbolAttribute(MCSA_Global);
      case DK_INDIRECT_SYMBOL:
        return ParseDirectiveSymbolAttribute(MCSA_IndirectSymbol);
      case DK_LAZY_REFERENCE:
        return ParseDirectiveSymbolAttribute(MCSA_LazyReference);
      case DK_NO_DEAD_STRIP:
        return ParseDirectiveSymbolAttribute(MCSA_NoDeadStrip);
      case DK_SYMBOL_RESOLVER:
        return ParseDirectiveSymbolAttribute(MCSA_SymbolResolver);
      case DK_PRIVATE_EXTERN:
        return ParseDirectiveSymbolAttribute(MCSA_PrivateExtern);
      case DK_REFERENCE:
        return ParseDirectiveSymbolAttribute(MCSA_Reference);
      case DK_WEAK_DEFINITION:
        return ParseDirectiveSymbolAttribute(MCSA_WeakDefinition);
      case DK_WEAK_REFERENCE:
        return ParseDirectiveSymbolAttribute(MCSA_WeakReference);
      case DK_WEAK_DEF_CAN_BE_HIDDEN:
        return ParseDirectiveSymbolAttribute(MCSA_WeakDefAutoPrivate);
      case DK_COMM:
      case DK_COMMON:
        return ParseDirectiveComm(/*IsLocal=*/false);
      case DK_LCOMM:
        return ParseDirectiveComm(/*IsLocal=*/true);
      case DK_ABORT:
        return ParseDirectiveAbort();
      case DK_INCLUDE:
        return ParseDirectiveInclude();
      case DK_INCBIN:
        return ParseDirectiveIncbin();
      case DK_CODE16:
      case DK_CODE16GCC:
        return TokError(Twine(IDVal) + " not supported yet");
      case DK_REPT:
        return ParseDirectiveRept(IDLoc);
      case DK_IRP:
        return ParseDirectiveIrp(IDLoc);
      case DK_IRPC:
        return ParseDirectiveIrpc(IDLoc);
      case DK_ENDR:
        return ParseDirectiveEndr(IDLoc);
      case DK_BUNDLE_ALIGN_MODE:
        return ParseDirectiveBundleAlignMode();
      case DK_BUNDLE_LOCK:
        return ParseDirectiveBundleLock();
      case DK_BUNDLE_UNLOCK:
        return ParseDirectiveBundleUnlock();
      case DK_SLEB128:
        return ParseDirectiveLEB128(true);
      case DK_ULEB128:
        return ParseDirectiveLEB128(false);
      case DK_SPACE:
      case DK_SKIP:
        return ParseDirectiveSpace(IDVal);
      case DK_FILE:
        return ParseDirectiveFile(IDLoc);
      case DK_LINE:
        return ParseDirectiveLine();
      case DK_LOC:
        return ParseDirectiveLoc();
      case DK_STABS:
        return ParseDirectiveStabs();
      case DK_CFI_SECTIONS:
        return ParseDirectiveCFISections();
      case DK_CFI_STARTPROC:
        return ParseDirectiveCFIStartProc();
      case DK_CFI_ENDPROC:
        return ParseDirectiveCFIEndProc();
      case DK_CFI_DEF_CFA:
        return ParseDirectiveCFIDefCfa(IDLoc);
      case DK_CFI_DEF_CFA_OFFSET:
        return ParseDirectiveCFIDefCfaOffset();
      case DK_CFI_ADJUST_CFA_OFFSET:
        return ParseDirectiveCFIAdjustCfaOffset();
      case DK_CFI_DEF_CFA_REGISTER:
        return ParseDirectiveCFIDefCfaRegister(IDLoc);
      case DK_CFI_OFFSET:
        return ParseDirectiveCFIOffset(IDLoc);
      case DK_CFI_REL_OFFSET:
        return ParseDirectiveCFIRelOffset(IDLoc);
      case DK_CFI_PERSONALITY:
        return ParseDirectiveCFIPersonalityOrLsda(true);
      case DK_CFI_LSDA:
        return ParseDirectiveCFIPersonalityOrLsda(false);
      case DK_CFI_REMEMBER_STATE:
        return ParseDirectiveCFIRememberState();
      case DK_CFI_RESTORE_STATE:
        return ParseDirectiveCFIRestoreState();
      case DK_CFI_SAME_VALUE:
        return ParseDirectiveCFISameValue(IDLoc);
      case DK_CFI_RESTORE:
        return ParseDirectiveCFIRestore(IDLoc);
      case DK_CFI_ESCAPE:
        return ParseDirectiveCFIEscape();
      case DK_CFI_SIGNAL_FRAME:
        return ParseDirectiveCFISignalFrame();
      case DK_CFI_UNDEFINED:
        return ParseDirectiveCFIUndefined(IDLoc);
      case DK_CFI_REGISTER:
        return ParseDirectiveCFIRegister(IDLoc);
      case DK_MACROS_ON:
      case DK_MACROS_OFF:
        return ParseDirectiveMacrosOnOff(IDVal);
      case DK_MACRO:
        return ParseDirectiveMacro(IDLoc);
      case DK_ENDM:
      case DK_ENDMACRO:
        return ParseDirectiveEndMacro(IDVal);
      case DK_PURGEM:
        return ParseDirectivePurgeMacro(IDLoc);
    }

    return Error(IDLoc, "unknown directive");
  }

  // __asm _emit or __asm __emit
  if (ParsingInlineAsm && (IDVal == "_emit" || IDVal == "__emit" ||
                           IDVal == "_EMIT" || IDVal == "__EMIT"))
    return ParseDirectiveMSEmit(IDLoc, Info, IDVal.size());

  // __asm align
  if (ParsingInlineAsm && (IDVal == "align" || IDVal == "ALIGN"))
    return ParseDirectiveMSAlign(IDLoc, Info);

  checkForValidSection();

  // Canonicalize the opcode to lower case.
  std::string OpcodeStr = IDVal.lower();
  ParseInstructionInfo IInfo(Info.AsmRewrites);
  bool HadError = getTargetParser().ParseInstruction(IInfo, OpcodeStr,
                                                     IDLoc, Info.ParsedOperands);
  Info.ParseError = HadError;

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
      getContext().getGenDwarfSection() ==
      getStreamer().getCurrentSection().first) {

    unsigned Line = SrcMgr.FindLineNumber(IDLoc, CurBuffer);

    // If we previously parsed a cpp hash file line comment then make sure the
    // current Dwarf File is for the CppHashFilename if not then emit the
    // Dwarf File table for it and adjust the line number for the .loc.
    const SmallVectorImpl<MCDwarfFile *> &MCDwarfFiles = 
      getContext().getMCDwarfFiles();
    if (CppHashFilename.size() != 0) {
      if (MCDwarfFiles[getContext().getGenDwarfFileNumber()]->getName() !=
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
  return isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '$' ||
         c == '.';
}

bool AsmParser::expandMacro(raw_svector_ostream &OS, StringRef Body,
                            const MCAsmMacroParameters &Parameters,
                            const MCAsmMacroArguments &A,
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
        if (Next == '$' || Next == 'n' ||
            isdigit(static_cast<unsigned char>(Next)))
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
        for (MCAsmMacroArgument::const_iterator it = A[Index].begin(),
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
        for (MCAsmMacroArgument::const_iterator it = A[Index].begin(),
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

MacroInstantiation::MacroInstantiation(const MCAsmMacro *M, SMLoc IL,
                                       int EB, SMLoc EL,
                                       MemoryBuffer *I)
  : TheMacro(M), Instantiation(I), InstantiationLoc(IL), ExitBuffer(EB),
    ExitLoc(EL)
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

bool AsmParser::ParseMacroArgument(MCAsmMacroArgument &MA,
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
          const char *NextChar = getTok().getEndLoc().getPointer();
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
bool AsmParser::ParseMacroArguments(const MCAsmMacro *M, MCAsmMacroArguments &A) {
  const unsigned NParameters = M ? M->Parameters.size() : 0;
  // Argument delimiter is initially unknown. It will be set by
  // ParseMacroArgument()
  AsmToken::TokenKind ArgumentDelimiter = AsmToken::Eof;

  // Parse two kinds of macro invocations:
  // - macros defined without any parameters accept an arbitrary number of them
  // - macros defined with parameters accept at most that many of them
  for (unsigned Parameter = 0; !NParameters || Parameter < NParameters;
       ++Parameter) {
    MCAsmMacroArgument MA;

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

const MCAsmMacro* AsmParser::LookupMacro(StringRef Name) {
  StringMap<MCAsmMacro*>::iterator I = MacroMap.find(Name);
  return (I == MacroMap.end()) ? NULL : I->getValue();
}

void AsmParser::DefineMacro(StringRef Name, const MCAsmMacro& Macro) {
  MacroMap[Name] = new MCAsmMacro(Macro);
}

void AsmParser::UndefineMacro(StringRef Name) {
  StringMap<MCAsmMacro*>::iterator I = MacroMap.find(Name);
  if (I != MacroMap.end()) {
    delete I->getValue();
    MacroMap.erase(I);
  }
}

bool AsmParser::HandleMacroEntry(const MCAsmMacro *M, SMLoc NameLoc) {
  // Arbitrarily limit macro nesting depth, to match 'as'. We can eliminate
  // this, although we should protect against infinite loops.
  if (ActiveMacros.size() == 20)
    return TokError("macros cannot be nested more than 20 levels deep");

  MCAsmMacroArguments A;
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

  // We include the .endmacro in the buffer as our cue to exit the macro
  // instantiation.
  OS << ".endmacro\n";

  MemoryBuffer *Instantiation =
    MemoryBuffer::getMemBufferCopy(OS.str(), "<instantiation>");

  // Create the macro instantiation object and add to the current macro
  // instantiation stack.
  MacroInstantiation *MI = new MacroInstantiation(M, NameLoc,
                                                  CurBuffer,
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
  JumpToLoc(ActiveMacros.back()->ExitLoc, ActiveMacros.back()->ExitBuffer);
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
  if (parseExpression(Value))
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

/// parseIdentifier:
///   ::= identifier
///   ::= string
bool AsmParser::parseIdentifier(StringRef &Res) {
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

  if (parseIdentifier(Name))
    return TokError("expected identifier after '" + Twine(IDVal) + "'");

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in '" + Twine(IDVal) + "'");
  Lex();

  return ParseAssignment(Name, allow_redef, true);
}

bool AsmParser::parseEscapedString(std::string &Data) {
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
    checkForValidSection();

    for (;;) {
      if (getLexer().isNot(AsmToken::String))
        return TokError("expected string in '" + Twine(IDVal) + "' directive");

      std::string Data;
      if (parseEscapedString(Data))
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
    checkForValidSection();

    for (;;) {
      const MCExpr *Value;
      SMLoc ExprLoc = getLexer().getLoc();
      if (parseExpression(Value))
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
    checkForValidSection();

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

/// ParseDirectiveZero
///  ::= .zero expression
bool AsmParser::ParseDirectiveZero() {
  checkForValidSection();

  int64_t NumBytes;
  if (parseAbsoluteExpression(NumBytes))
    return true;

  int64_t Val = 0;
  if (getLexer().is(AsmToken::Comma)) {
    Lex();
    if (parseAbsoluteExpression(Val))
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
  checkForValidSection();

  int64_t NumValues;
  if (parseAbsoluteExpression(NumValues))
    return true;

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in '.fill' directive");
  Lex();

  int64_t FillSize;
  if (parseAbsoluteExpression(FillSize))
    return true;

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in '.fill' directive");
  Lex();

  int64_t FillExpr;
  if (parseAbsoluteExpression(FillExpr))
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
  checkForValidSection();

  const MCExpr *Offset;
  SMLoc Loc = getTok().getLoc();
  if (parseExpression(Offset))
    return true;

  // Parse optional fill expression.
  int64_t FillExpr = 0;
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    if (getLexer().isNot(AsmToken::Comma))
      return TokError("unexpected token in '.org' directive");
    Lex();

    if (parseAbsoluteExpression(FillExpr))
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
  checkForValidSection();

  SMLoc AlignmentLoc = getLexer().getLoc();
  int64_t Alignment;
  if (parseAbsoluteExpression(Alignment))
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
      if (parseAbsoluteExpression(FillExpr))
        return true;
    }

    if (getLexer().isNot(AsmToken::EndOfStatement)) {
      if (getLexer().isNot(AsmToken::Comma))
        return TokError("unexpected token in directive");
      Lex();

      MaxBytesLoc = getLexer().getLoc();
      if (parseAbsoluteExpression(MaxBytesToFill))
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
  } else {
    // Reject alignments that aren't a power of two, for gas compatibility.
    if (!isPowerOf2_64(Alignment))
      Error(AlignmentLoc, "alignment must be a power of 2");
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
  bool UseCodeAlign = getStreamer().getCurrentSection().first->UseCodeAlign();
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

/// ParseDirectiveFile
/// ::= .file [number] filename
/// ::= .file number directory filename
bool AsmParser::ParseDirectiveFile(SMLoc DirectiveLoc) {
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
bool AsmParser::ParseDirectiveLine() {
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
bool AsmParser::ParseDirectiveLoc() {
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
      if (parseIdentifier(Name))
        return TokError("unexpected token in '.loc' directive");

      if (Name == "basic_block")
        Flags |= DWARF2_FLAG_BASIC_BLOCK;
      else if (Name == "prologue_end")
        Flags |= DWARF2_FLAG_PROLOGUE_END;
      else if (Name == "epilogue_begin")
        Flags |= DWARF2_FLAG_EPILOGUE_BEGIN;
      else if (Name == "is_stmt") {
        Loc = getTok().getLoc();
        const MCExpr *Value;
        if (parseExpression(Value))
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
        } else {
          return Error(Loc, "is_stmt value not the constant value of 0 or 1");
        }
      } else if (Name == "isa") {
        Loc = getTok().getLoc();
        const MCExpr *Value;
        if (parseExpression(Value))
          return true;
        // The expression must be a constant greater or equal to 0.
        if (const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(Value)) {
          int Value = MCE->getValue();
          if (Value < 0)
            return Error(Loc, "isa number less than zero");
          Isa = Value;
        } else {
          return Error(Loc, "isa number not a constant value");
        }
      } else if (Name == "discriminator") {
        if (parseAbsoluteExpression(Discriminator))
          return true;
      } else {
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
bool AsmParser::ParseDirectiveStabs() {
  return TokError("unsupported directive '.stabs'");
}

/// ParseDirectiveCFISections
/// ::= .cfi_sections section [, section]
bool AsmParser::ParseDirectiveCFISections() {
  StringRef Name;
  bool EH = false;
  bool Debug = false;

  if (parseIdentifier(Name))
    return TokError("Expected an identifier");

  if (Name == ".eh_frame")
    EH = true;
  else if (Name == ".debug_frame")
    Debug = true;

  if (getLexer().is(AsmToken::Comma)) {
    Lex();

    if (parseIdentifier(Name))
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
bool AsmParser::ParseDirectiveCFIStartProc() {
  getStreamer().EmitCFIStartProc();
  return false;
}

/// ParseDirectiveCFIEndProc
/// ::= .cfi_endproc
bool AsmParser::ParseDirectiveCFIEndProc() {
  getStreamer().EmitCFIEndProc();
  return false;
}

/// ParseRegisterOrRegisterNumber - parse register name or number.
bool AsmParser::ParseRegisterOrRegisterNumber(int64_t &Register,
                                              SMLoc DirectiveLoc) {
  unsigned RegNo;

  if (getLexer().isNot(AsmToken::Integer)) {
    if (getTargetParser().ParseRegister(RegNo, DirectiveLoc, DirectiveLoc))
      return true;
    Register = getContext().getRegisterInfo().getDwarfRegNum(RegNo, true);
  } else
    return parseAbsoluteExpression(Register);

  return false;
}

/// ParseDirectiveCFIDefCfa
/// ::= .cfi_def_cfa register,  offset
bool AsmParser::ParseDirectiveCFIDefCfa(SMLoc DirectiveLoc) {
  int64_t Register = 0;
  if (ParseRegisterOrRegisterNumber(Register, DirectiveLoc))
    return true;

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lex();

  int64_t Offset = 0;
  if (parseAbsoluteExpression(Offset))
    return true;

  getStreamer().EmitCFIDefCfa(Register, Offset);
  return false;
}

/// ParseDirectiveCFIDefCfaOffset
/// ::= .cfi_def_cfa_offset offset
bool AsmParser::ParseDirectiveCFIDefCfaOffset() {
  int64_t Offset = 0;
  if (parseAbsoluteExpression(Offset))
    return true;

  getStreamer().EmitCFIDefCfaOffset(Offset);
  return false;
}

/// ParseDirectiveCFIRegister
/// ::= .cfi_register register, register
bool AsmParser::ParseDirectiveCFIRegister(SMLoc DirectiveLoc) {
  int64_t Register1 = 0;
  if (ParseRegisterOrRegisterNumber(Register1, DirectiveLoc))
    return true;

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lex();

  int64_t Register2 = 0;
  if (ParseRegisterOrRegisterNumber(Register2, DirectiveLoc))
    return true;

  getStreamer().EmitCFIRegister(Register1, Register2);
  return false;
}

/// ParseDirectiveCFIAdjustCfaOffset
/// ::= .cfi_adjust_cfa_offset adjustment
bool AsmParser::ParseDirectiveCFIAdjustCfaOffset() {
  int64_t Adjustment = 0;
  if (parseAbsoluteExpression(Adjustment))
    return true;

  getStreamer().EmitCFIAdjustCfaOffset(Adjustment);
  return false;
}

/// ParseDirectiveCFIDefCfaRegister
/// ::= .cfi_def_cfa_register register
bool AsmParser::ParseDirectiveCFIDefCfaRegister(SMLoc DirectiveLoc) {
  int64_t Register = 0;
  if (ParseRegisterOrRegisterNumber(Register, DirectiveLoc))
    return true;

  getStreamer().EmitCFIDefCfaRegister(Register);
  return false;
}

/// ParseDirectiveCFIOffset
/// ::= .cfi_offset register, offset
bool AsmParser::ParseDirectiveCFIOffset(SMLoc DirectiveLoc) {
  int64_t Register = 0;
  int64_t Offset = 0;

  if (ParseRegisterOrRegisterNumber(Register, DirectiveLoc))
    return true;

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lex();

  if (parseAbsoluteExpression(Offset))
    return true;

  getStreamer().EmitCFIOffset(Register, Offset);
  return false;
}

/// ParseDirectiveCFIRelOffset
/// ::= .cfi_rel_offset register, offset
bool AsmParser::ParseDirectiveCFIRelOffset(SMLoc DirectiveLoc) {
  int64_t Register = 0;

  if (ParseRegisterOrRegisterNumber(Register, DirectiveLoc))
    return true;

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lex();

  int64_t Offset = 0;
  if (parseAbsoluteExpression(Offset))
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
/// IsPersonality true for cfi_personality, false for cfi_lsda
/// ::= .cfi_personality encoding, [symbol_name]
/// ::= .cfi_lsda encoding, [symbol_name]
bool AsmParser::ParseDirectiveCFIPersonalityOrLsda(bool IsPersonality) {
  int64_t Encoding = 0;
  if (parseAbsoluteExpression(Encoding))
    return true;
  if (Encoding == dwarf::DW_EH_PE_omit)
    return false;

  if (!isValidEncoding(Encoding))
    return TokError("unsupported encoding.");

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lex();

  StringRef Name;
  if (parseIdentifier(Name))
    return TokError("expected identifier in directive");

  MCSymbol *Sym = getContext().GetOrCreateSymbol(Name);

  if (IsPersonality)
    getStreamer().EmitCFIPersonality(Sym, Encoding);
  else
    getStreamer().EmitCFILsda(Sym, Encoding);
  return false;
}

/// ParseDirectiveCFIRememberState
/// ::= .cfi_remember_state
bool AsmParser::ParseDirectiveCFIRememberState() {
  getStreamer().EmitCFIRememberState();
  return false;
}

/// ParseDirectiveCFIRestoreState
/// ::= .cfi_remember_state
bool AsmParser::ParseDirectiveCFIRestoreState() {
  getStreamer().EmitCFIRestoreState();
  return false;
}

/// ParseDirectiveCFISameValue
/// ::= .cfi_same_value register
bool AsmParser::ParseDirectiveCFISameValue(SMLoc DirectiveLoc) {
  int64_t Register = 0;

  if (ParseRegisterOrRegisterNumber(Register, DirectiveLoc))
    return true;

  getStreamer().EmitCFISameValue(Register);
  return false;
}

/// ParseDirectiveCFIRestore
/// ::= .cfi_restore register
bool AsmParser::ParseDirectiveCFIRestore(SMLoc DirectiveLoc) {
  int64_t Register = 0;
  if (ParseRegisterOrRegisterNumber(Register, DirectiveLoc))
    return true;

  getStreamer().EmitCFIRestore(Register);
  return false;
}

/// ParseDirectiveCFIEscape
/// ::= .cfi_escape expression[,...]
bool AsmParser::ParseDirectiveCFIEscape() {
  std::string Values;
  int64_t CurrValue;
  if (parseAbsoluteExpression(CurrValue))
    return true;

  Values.push_back((uint8_t)CurrValue);

  while (getLexer().is(AsmToken::Comma)) {
    Lex();

    if (parseAbsoluteExpression(CurrValue))
      return true;

    Values.push_back((uint8_t)CurrValue);
  }

  getStreamer().EmitCFIEscape(Values);
  return false;
}

/// ParseDirectiveCFISignalFrame
/// ::= .cfi_signal_frame
bool AsmParser::ParseDirectiveCFISignalFrame() {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return Error(getLexer().getLoc(),
                 "unexpected token in '.cfi_signal_frame'");

  getStreamer().EmitCFISignalFrame();
  return false;
}

/// ParseDirectiveCFIUndefined
/// ::= .cfi_undefined register
bool AsmParser::ParseDirectiveCFIUndefined(SMLoc DirectiveLoc) {
  int64_t Register = 0;

  if (ParseRegisterOrRegisterNumber(Register, DirectiveLoc))
    return true;

  getStreamer().EmitCFIUndefined(Register);
  return false;
}

/// ParseDirectiveMacrosOnOff
/// ::= .macros_on
/// ::= .macros_off
bool AsmParser::ParseDirectiveMacrosOnOff(StringRef Directive) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return Error(getLexer().getLoc(),
                 "unexpected token in '" + Directive + "' directive");

  SetMacrosEnabled(Directive == ".macros_on");
  return false;
}

/// ParseDirectiveMacro
/// ::= .macro name [parameters]
bool AsmParser::ParseDirectiveMacro(SMLoc DirectiveLoc) {
  StringRef Name;
  if (parseIdentifier(Name))
    return TokError("expected identifier in '.macro' directive");

  MCAsmMacroParameters Parameters;
  // Argument delimiter is initially unknown. It will be set by
  // ParseMacroArgument()
  AsmToken::TokenKind ArgumentDelimiter = AsmToken::Eof;
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    for (;;) {
      MCAsmMacroParameter Parameter;
      if (parseIdentifier(Parameter.first))
        return TokError("expected identifier in '.macro' directive");

      if (getLexer().is(AsmToken::Equal)) {
        Lex();
        if (ParseMacroArgument(Parameter.second, ArgumentDelimiter))
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
    eatToEndOfStatement();
  }

  if (LookupMacro(Name)) {
    return Error(DirectiveLoc, "macro '" + Name + "' is already defined");
  }

  const char *BodyStart = StartToken.getLoc().getPointer();
  const char *BodyEnd = EndToken.getLoc().getPointer();
  StringRef Body = StringRef(BodyStart, BodyEnd - BodyStart);
  CheckForBadMacro(DirectiveLoc, Name, Body, Parameters);
  DefineMacro(Name, MCAsmMacro(Name, Body, Parameters));
  return false;
}

/// CheckForBadMacro
///
/// With the support added for named parameters there may be code out there that
/// is transitioning from positional parameters.  In versions of gas that did
/// not support named parameters they would be ignored on the macro defintion.
/// But to support both styles of parameters this is not possible so if a macro
/// defintion has named parameters but does not use them and has what appears
/// to be positional parameters, strings like $1, $2, ... and $n, then issue a
/// warning that the positional parameter found in body which have no effect.
/// Hoping the developer will either remove the named parameters from the macro
/// definiton so the positional parameters get used if that was what was
/// intended or change the macro to use the named parameters.  It is possible
/// this warning will trigger when the none of the named parameters are used
/// and the strings like $1 are infact to simply to be passed trough unchanged.
void AsmParser::CheckForBadMacro(SMLoc DirectiveLoc, StringRef Name,
                                 StringRef Body,
                                 MCAsmMacroParameters Parameters) {
  // If this macro is not defined with named parameters the warning we are
  // checking for here doesn't apply.
  unsigned NParameters = Parameters.size();
  if (NParameters == 0)
    return;

  bool NamedParametersFound = false;
  bool PositionalParametersFound = false;

  // Look at the body of the macro for use of both the named parameters and what
  // are likely to be positional parameters.  This is what expandMacro() is
  // doing when it finds the parameters in the body.
  while (!Body.empty()) {
    // Scan for the next possible parameter.
    std::size_t End = Body.size(), Pos = 0;
    for (; Pos != End; ++Pos) {
      // Check for a substitution or escape.
      // This macro is defined with parameters, look for \foo, \bar, etc.
      if (Body[Pos] == '\\' && Pos + 1 != End)
        break;

      // This macro should have parameters, but look for $0, $1, ..., $n too.
      if (Body[Pos] != '$' || Pos + 1 == End)
        continue;
      char Next = Body[Pos + 1];
      if (Next == '$' || Next == 'n' ||
          isdigit(static_cast<unsigned char>(Next)))
        break;
    }

    // Check if we reached the end.
    if (Pos == End)
      break;

    if (Body[Pos] == '$') {
      switch (Body[Pos+1]) {
        // $$ => $
      case '$':
        break;

        // $n => number of arguments
      case 'n':
        PositionalParametersFound = true;
        break;

        // $[0-9] => argument
      default: {
        PositionalParametersFound = true;
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
            Pos = I;
          }
      } else {
        NamedParametersFound = true;
        Pos += 1 + Argument.size();
      }
    }
    // Update the scan point.
    Body = Body.substr(Pos);
  }

  if (!NamedParametersFound && PositionalParametersFound)
    Warning(DirectiveLoc, "macro defined with named parameters which are not "
                          "used in macro body, possible positional parameter "
                          "found in body which will have no effect");
}

/// ParseDirectiveEndMacro
/// ::= .endm
/// ::= .endmacro
bool AsmParser::ParseDirectiveEndMacro(StringRef Directive) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '" + Directive + "' directive");

  // If we are inside a macro instantiation, terminate the current
  // instantiation.
  if (InsideMacroInstantiation()) {
    HandleMacroExit();
    return false;
  }

  // Otherwise, this .endmacro is a stray entry in the file; well formed
  // .endmacro directives are handled during the macro definition parsing.
  return TokError("unexpected '" + Directive + "' in file, "
                  "no current macro definition");
}

/// ParseDirectivePurgeMacro
/// ::= .purgem
bool AsmParser::ParseDirectivePurgeMacro(SMLoc DirectiveLoc) {
  StringRef Name;
  if (parseIdentifier(Name))
    return TokError("expected identifier in '.purgem' directive");

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.purgem' directive");

  if (!LookupMacro(Name))
    return Error(DirectiveLoc, "macro '" + Name + "' is not defined");

  UndefineMacro(Name);
  return false;
}

/// ParseDirectiveBundleAlignMode
/// ::= {.bundle_align_mode} expression
bool AsmParser::ParseDirectiveBundleAlignMode() {
  checkForValidSection();

  // Expect a single argument: an expression that evaluates to a constant
  // in the inclusive range 0-30.
  SMLoc ExprLoc = getLexer().getLoc();
  int64_t AlignSizePow2;
  if (parseAbsoluteExpression(AlignSizePow2))
    return true;
  else if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token after expression in"
                    " '.bundle_align_mode' directive");
  else if (AlignSizePow2 < 0 || AlignSizePow2 > 30)
    return Error(ExprLoc,
                 "invalid bundle alignment size (expected between 0 and 30)");

  Lex();

  // Because of AlignSizePow2's verified range we can safely truncate it to
  // unsigned.
  getStreamer().EmitBundleAlignMode(static_cast<unsigned>(AlignSizePow2));
  return false;
}

/// ParseDirectiveBundleLock
/// ::= {.bundle_lock} [align_to_end]
bool AsmParser::ParseDirectiveBundleLock() {
  checkForValidSection();
  bool AlignToEnd = false;

  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    StringRef Option;
    SMLoc Loc = getTok().getLoc();
    const char *kInvalidOptionError =
      "invalid option for '.bundle_lock' directive";

    if (parseIdentifier(Option))
      return Error(Loc, kInvalidOptionError);

    if (Option != "align_to_end")
      return Error(Loc, kInvalidOptionError);
    else if (getLexer().isNot(AsmToken::EndOfStatement))
      return Error(Loc,
                   "unexpected token after '.bundle_lock' directive option");
    AlignToEnd = true;
  }

  Lex();

  getStreamer().EmitBundleLock(AlignToEnd);
  return false;
}

/// ParseDirectiveBundleLock
/// ::= {.bundle_lock}
bool AsmParser::ParseDirectiveBundleUnlock() {
  checkForValidSection();

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.bundle_unlock' directive");
  Lex();

  getStreamer().EmitBundleUnlock();
  return false;
}

/// ParseDirectiveSpace
/// ::= (.skip | .space) expression [ , expression ]
bool AsmParser::ParseDirectiveSpace(StringRef IDVal) {
  checkForValidSection();

  int64_t NumBytes;
  if (parseAbsoluteExpression(NumBytes))
    return true;

  int64_t FillExpr = 0;
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    if (getLexer().isNot(AsmToken::Comma))
      return TokError("unexpected token in '" + Twine(IDVal) + "' directive");
    Lex();

    if (parseAbsoluteExpression(FillExpr))
      return true;

    if (getLexer().isNot(AsmToken::EndOfStatement))
      return TokError("unexpected token in '" + Twine(IDVal) + "' directive");
  }

  Lex();

  if (NumBytes <= 0)
    return TokError("invalid number of bytes in '" +
                    Twine(IDVal) + "' directive");

  // FIXME: Sometimes the fill expr is 'nop' if it isn't supplied, instead of 0.
  getStreamer().EmitFill(NumBytes, FillExpr, DEFAULT_ADDRSPACE);

  return false;
}

/// ParseDirectiveLEB128
/// ::= (.sleb128 | .uleb128) expression
bool AsmParser::ParseDirectiveLEB128(bool Signed) {
  checkForValidSection();
  const MCExpr *Value;

  if (parseExpression(Value))
    return true;

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in directive");

  if (Signed)
    getStreamer().EmitSLEB128Value(Value);
  else
    getStreamer().EmitULEB128Value(Value);

  return false;
}

/// ParseDirectiveSymbolAttribute
///  ::= { ".globl", ".weak", ... } [ identifier ( , identifier )* ]
bool AsmParser::ParseDirectiveSymbolAttribute(MCSymbolAttr Attr) {
  if (getLexer().isNot(AsmToken::EndOfStatement)) {
    for (;;) {
      StringRef Name;
      SMLoc Loc = getTok().getLoc();

      if (parseIdentifier(Name))
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
  checkForValidSection();

  SMLoc IDLoc = getLexer().getLoc();
  StringRef Name;
  if (parseIdentifier(Name))
    return TokError("expected identifier in directive");

  // Handle the identifier as the key symbol.
  MCSymbol *Sym = getContext().GetOrCreateSymbol(Name);

  if (getLexer().isNot(AsmToken::Comma))
    return TokError("unexpected token in directive");
  Lex();

  int64_t Size;
  SMLoc SizeLoc = getLexer().getLoc();
  if (parseAbsoluteExpression(Size))
    return true;

  int64_t Pow2Alignment = 0;
  SMLoc Pow2AlignmentLoc;
  if (getLexer().is(AsmToken::Comma)) {
    Lex();
    Pow2AlignmentLoc = getLexer().getLoc();
    if (parseAbsoluteExpression(Pow2Alignment))
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

  StringRef Str = parseStringToEndOfStatement();
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
    eatToEndOfStatement();
  } else {
    int64_t ExprValue;
    if (parseAbsoluteExpression(ExprValue))
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
    eatToEndOfStatement();
  } else {
    StringRef Str = parseStringToEndOfStatement();

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
    eatToEndOfStatement();
  } else {
    StringRef Str1 = ParseStringToComma();

    if (getLexer().isNot(AsmToken::Comma))
      return TokError("unexpected token in '.ifc' directive");

    Lex();

    StringRef Str2 = parseStringToEndOfStatement();

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
    eatToEndOfStatement();
  } else {
    if (parseIdentifier(Name))
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
    eatToEndOfStatement();
  } else {
    int64_t ExprValue;
    if (parseAbsoluteExpression(ExprValue))
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

void AsmParser::initializeDirectiveKindMap() {
  DirectiveKindMap[".set"] = DK_SET;
  DirectiveKindMap[".equ"] = DK_EQU;
  DirectiveKindMap[".equiv"] = DK_EQUIV;
  DirectiveKindMap[".ascii"] = DK_ASCII;
  DirectiveKindMap[".asciz"] = DK_ASCIZ;
  DirectiveKindMap[".string"] = DK_STRING;
  DirectiveKindMap[".byte"] = DK_BYTE;
  DirectiveKindMap[".short"] = DK_SHORT;
  DirectiveKindMap[".value"] = DK_VALUE;
  DirectiveKindMap[".2byte"] = DK_2BYTE;
  DirectiveKindMap[".long"] = DK_LONG;
  DirectiveKindMap[".int"] = DK_INT;
  DirectiveKindMap[".4byte"] = DK_4BYTE;
  DirectiveKindMap[".quad"] = DK_QUAD;
  DirectiveKindMap[".8byte"] = DK_8BYTE;
  DirectiveKindMap[".single"] = DK_SINGLE;
  DirectiveKindMap[".float"] = DK_FLOAT;
  DirectiveKindMap[".double"] = DK_DOUBLE;
  DirectiveKindMap[".align"] = DK_ALIGN;
  DirectiveKindMap[".align32"] = DK_ALIGN32;
  DirectiveKindMap[".balign"] = DK_BALIGN;
  DirectiveKindMap[".balignw"] = DK_BALIGNW;
  DirectiveKindMap[".balignl"] = DK_BALIGNL;
  DirectiveKindMap[".p2align"] = DK_P2ALIGN;
  DirectiveKindMap[".p2alignw"] = DK_P2ALIGNW;
  DirectiveKindMap[".p2alignl"] = DK_P2ALIGNL;
  DirectiveKindMap[".org"] = DK_ORG;
  DirectiveKindMap[".fill"] = DK_FILL;
  DirectiveKindMap[".zero"] = DK_ZERO;
  DirectiveKindMap[".extern"] = DK_EXTERN;
  DirectiveKindMap[".globl"] = DK_GLOBL;
  DirectiveKindMap[".global"] = DK_GLOBAL;
  DirectiveKindMap[".indirect_symbol"] = DK_INDIRECT_SYMBOL;
  DirectiveKindMap[".lazy_reference"] = DK_LAZY_REFERENCE;
  DirectiveKindMap[".no_dead_strip"] = DK_NO_DEAD_STRIP;
  DirectiveKindMap[".symbol_resolver"] = DK_SYMBOL_RESOLVER;
  DirectiveKindMap[".private_extern"] = DK_PRIVATE_EXTERN;
  DirectiveKindMap[".reference"] = DK_REFERENCE;
  DirectiveKindMap[".weak_definition"] = DK_WEAK_DEFINITION;
  DirectiveKindMap[".weak_reference"] = DK_WEAK_REFERENCE;
  DirectiveKindMap[".weak_def_can_be_hidden"] = DK_WEAK_DEF_CAN_BE_HIDDEN;
  DirectiveKindMap[".comm"] = DK_COMM;
  DirectiveKindMap[".common"] = DK_COMMON;
  DirectiveKindMap[".lcomm"] = DK_LCOMM;
  DirectiveKindMap[".abort"] = DK_ABORT;
  DirectiveKindMap[".include"] = DK_INCLUDE;
  DirectiveKindMap[".incbin"] = DK_INCBIN;
  DirectiveKindMap[".code16"] = DK_CODE16;
  DirectiveKindMap[".code16gcc"] = DK_CODE16GCC;
  DirectiveKindMap[".rept"] = DK_REPT;
  DirectiveKindMap[".irp"] = DK_IRP;
  DirectiveKindMap[".irpc"] = DK_IRPC;
  DirectiveKindMap[".endr"] = DK_ENDR;
  DirectiveKindMap[".bundle_align_mode"] = DK_BUNDLE_ALIGN_MODE;
  DirectiveKindMap[".bundle_lock"] = DK_BUNDLE_LOCK;
  DirectiveKindMap[".bundle_unlock"] = DK_BUNDLE_UNLOCK;
  DirectiveKindMap[".if"] = DK_IF;
  DirectiveKindMap[".ifb"] = DK_IFB;
  DirectiveKindMap[".ifnb"] = DK_IFNB;
  DirectiveKindMap[".ifc"] = DK_IFC;
  DirectiveKindMap[".ifnc"] = DK_IFNC;
  DirectiveKindMap[".ifdef"] = DK_IFDEF;
  DirectiveKindMap[".ifndef"] = DK_IFNDEF;
  DirectiveKindMap[".ifnotdef"] = DK_IFNOTDEF;
  DirectiveKindMap[".elseif"] = DK_ELSEIF;
  DirectiveKindMap[".else"] = DK_ELSE;
  DirectiveKindMap[".endif"] = DK_ENDIF;
  DirectiveKindMap[".skip"] = DK_SKIP;
  DirectiveKindMap[".space"] = DK_SPACE;
  DirectiveKindMap[".file"] = DK_FILE;
  DirectiveKindMap[".line"] = DK_LINE;
  DirectiveKindMap[".loc"] = DK_LOC;
  DirectiveKindMap[".stabs"] = DK_STABS;
  DirectiveKindMap[".sleb128"] = DK_SLEB128;
  DirectiveKindMap[".uleb128"] = DK_ULEB128;
  DirectiveKindMap[".cfi_sections"] = DK_CFI_SECTIONS;
  DirectiveKindMap[".cfi_startproc"] = DK_CFI_STARTPROC;
  DirectiveKindMap[".cfi_endproc"] = DK_CFI_ENDPROC;
  DirectiveKindMap[".cfi_def_cfa"] = DK_CFI_DEF_CFA;
  DirectiveKindMap[".cfi_def_cfa_offset"] = DK_CFI_DEF_CFA_OFFSET;
  DirectiveKindMap[".cfi_adjust_cfa_offset"] = DK_CFI_ADJUST_CFA_OFFSET;
  DirectiveKindMap[".cfi_def_cfa_register"] = DK_CFI_DEF_CFA_REGISTER;
  DirectiveKindMap[".cfi_offset"] = DK_CFI_OFFSET;
  DirectiveKindMap[".cfi_rel_offset"] = DK_CFI_REL_OFFSET;
  DirectiveKindMap[".cfi_personality"] = DK_CFI_PERSONALITY;
  DirectiveKindMap[".cfi_lsda"] = DK_CFI_LSDA;
  DirectiveKindMap[".cfi_remember_state"] = DK_CFI_REMEMBER_STATE;
  DirectiveKindMap[".cfi_restore_state"] = DK_CFI_RESTORE_STATE;
  DirectiveKindMap[".cfi_same_value"] = DK_CFI_SAME_VALUE;
  DirectiveKindMap[".cfi_restore"] = DK_CFI_RESTORE;
  DirectiveKindMap[".cfi_escape"] = DK_CFI_ESCAPE;
  DirectiveKindMap[".cfi_signal_frame"] = DK_CFI_SIGNAL_FRAME;
  DirectiveKindMap[".cfi_undefined"] = DK_CFI_UNDEFINED;
  DirectiveKindMap[".cfi_register"] = DK_CFI_REGISTER;
  DirectiveKindMap[".macros_on"] = DK_MACROS_ON;
  DirectiveKindMap[".macros_off"] = DK_MACROS_OFF;
  DirectiveKindMap[".macro"] = DK_MACRO;
  DirectiveKindMap[".endm"] = DK_ENDM;
  DirectiveKindMap[".endmacro"] = DK_ENDMACRO;
  DirectiveKindMap[".purgem"] = DK_PURGEM;
}


MCAsmMacro *AsmParser::ParseMacroLikeBody(SMLoc DirectiveLoc) {
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
    eatToEndOfStatement();
  }

  const char *BodyStart = StartToken.getLoc().getPointer();
  const char *BodyEnd = EndToken.getLoc().getPointer();
  StringRef Body = StringRef(BodyStart, BodyEnd - BodyStart);

  // We Are Anonymous.
  StringRef Name;
  MCAsmMacroParameters Parameters;
  return new MCAsmMacro(Name, Body, Parameters);
}

void AsmParser::InstantiateMacroLikeBody(MCAsmMacro *M, SMLoc DirectiveLoc,
                                         raw_svector_ostream &OS) {
  OS << ".endr\n";

  MemoryBuffer *Instantiation =
    MemoryBuffer::getMemBufferCopy(OS.str(), "<instantiation>");

  // Create the macro instantiation object and add to the current macro
  // instantiation stack.
  MacroInstantiation *MI = new MacroInstantiation(M, DirectiveLoc,
                                                  CurBuffer,
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
  if (parseAbsoluteExpression(Count))
    return TokError("unexpected token in '.rept' directive");

  if (Count < 0)
    return TokError("Count is negative");

  if (Lexer.isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.rept' directive");

  // Eat the end of statement.
  Lex();

  // Lex the rept definition.
  MCAsmMacro *M = ParseMacroLikeBody(DirectiveLoc);
  if (!M)
    return true;

  // Macro instantiation is lexical, unfortunately. We construct a new buffer
  // to hold the macro body with substitutions.
  SmallString<256> Buf;
  MCAsmMacroParameters Parameters;
  MCAsmMacroArguments A;
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
  MCAsmMacroParameters Parameters;
  MCAsmMacroParameter Parameter;

  if (parseIdentifier(Parameter.first))
    return TokError("expected identifier in '.irp' directive");

  Parameters.push_back(Parameter);

  if (Lexer.isNot(AsmToken::Comma))
    return TokError("expected comma in '.irp' directive");

  Lex();

  MCAsmMacroArguments A;
  if (ParseMacroArguments(0, A))
    return true;

  // Eat the end of statement.
  Lex();

  // Lex the irp definition.
  MCAsmMacro *M = ParseMacroLikeBody(DirectiveLoc);
  if (!M)
    return true;

  // Macro instantiation is lexical, unfortunately. We construct a new buffer
  // to hold the macro body with substitutions.
  SmallString<256> Buf;
  raw_svector_ostream OS(Buf);

  for (MCAsmMacroArguments::iterator i = A.begin(), e = A.end(); i != e; ++i) {
    MCAsmMacroArguments Args;
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
  MCAsmMacroParameters Parameters;
  MCAsmMacroParameter Parameter;

  if (parseIdentifier(Parameter.first))
    return TokError("expected identifier in '.irpc' directive");

  Parameters.push_back(Parameter);

  if (Lexer.isNot(AsmToken::Comma))
    return TokError("expected comma in '.irpc' directive");

  Lex();

  MCAsmMacroArguments A;
  if (ParseMacroArguments(0, A))
    return true;

  if (A.size() != 1 || A.front().size() != 1)
    return TokError("unexpected token in '.irpc' directive");

  // Eat the end of statement.
  Lex();

  // Lex the irpc definition.
  MCAsmMacro *M = ParseMacroLikeBody(DirectiveLoc);
  if (!M)
    return true;

  // Macro instantiation is lexical, unfortunately. We construct a new buffer
  // to hold the macro body with substitutions.
  SmallString<256> Buf;
  raw_svector_ostream OS(Buf);

  StringRef Values = A.front().front().getString();
  std::size_t I, End = Values.size();
  for (I = 0; I < End; ++I) {
    MCAsmMacroArgument Arg;
    Arg.push_back(AsmToken(AsmToken::Identifier, Values.slice(I, I+1)));

    MCAsmMacroArguments Args;
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

bool AsmParser::ParseDirectiveMSEmit(SMLoc IDLoc, ParseStatementInfo &Info,
                                     size_t Len) {
  const MCExpr *Value;
  SMLoc ExprLoc = getLexer().getLoc();
  if (parseExpression(Value))
    return true;
  const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(Value);
  if (!MCE)
    return Error(ExprLoc, "unexpected expression in _emit");
  uint64_t IntValue = MCE->getValue();
  if (!isUIntN(8, IntValue) && !isIntN(8, IntValue))
    return Error(ExprLoc, "literal value out of range for directive");

  Info.AsmRewrites->push_back(AsmRewrite(AOK_Emit, IDLoc, Len));
  return false;
}

bool AsmParser::ParseDirectiveMSAlign(SMLoc IDLoc, ParseStatementInfo &Info) {
  const MCExpr *Value;
  SMLoc ExprLoc = getLexer().getLoc();
  if (parseExpression(Value))
    return true;
  const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(Value);
  if (!MCE)
    return Error(ExprLoc, "unexpected expression in align");
  uint64_t IntValue = MCE->getValue();
  if (!isPowerOf2_64(IntValue))
    return Error(ExprLoc, "literal value not a power of two greater then zero");

  Info.AsmRewrites->push_back(AsmRewrite(AOK_Align, IDLoc, 5,
                                         Log2_64(IntValue)));
  return false;
}

// We are comparing pointers, but the pointers are relative to a single string.
// Thus, this should always be deterministic.
static int RewritesSort(const void *A, const void *B) {
  const AsmRewrite *AsmRewriteA = static_cast<const AsmRewrite *>(A);
  const AsmRewrite *AsmRewriteB = static_cast<const AsmRewrite *>(B);
  if (AsmRewriteA->Loc.getPointer() < AsmRewriteB->Loc.getPointer())
    return -1;
  if (AsmRewriteB->Loc.getPointer() < AsmRewriteA->Loc.getPointer())
    return 1;

  // It's possible to have a SizeDirective, Imm/ImmPrefix and an Input/Output
  // rewrite to the same location.  Make sure the SizeDirective rewrite is
  // performed first, then the Imm/ImmPrefix and finally the Input/Output.  This
  // ensures the sort algorithm is stable.
  if (AsmRewritePrecedence [AsmRewriteA->Kind] >
      AsmRewritePrecedence [AsmRewriteB->Kind])
    return -1;

  if (AsmRewritePrecedence [AsmRewriteA->Kind] <
      AsmRewritePrecedence [AsmRewriteB->Kind])
    return 1;
  llvm_unreachable ("Unstable rewrite sort.");
}

bool
AsmParser::parseMSInlineAsm(void *AsmLoc, std::string &AsmString,
                            unsigned &NumOutputs, unsigned &NumInputs,
                            SmallVectorImpl<std::pair<void *, bool> > &OpDecls,
                            SmallVectorImpl<std::string> &Constraints,
                            SmallVectorImpl<std::string> &Clobbers,
                            const MCInstrInfo *MII,
                            const MCInstPrinter *IP,
                            MCAsmParserSemaCallback &SI) {
  SmallVector<void *, 4> InputDecls;
  SmallVector<void *, 4> OutputDecls;
  SmallVector<bool, 4> InputDeclsAddressOf;
  SmallVector<bool, 4> OutputDeclsAddressOf;
  SmallVector<std::string, 4> InputConstraints;
  SmallVector<std::string, 4> OutputConstraints;
  SmallVector<unsigned, 4> ClobberRegs;

  SmallVector<AsmRewrite, 4> AsmStrRewrites;

  // Prime the lexer.
  Lex();

  // While we have input, parse each statement.
  unsigned InputIdx = 0;
  unsigned OutputIdx = 0;
  while (getLexer().isNot(AsmToken::Eof)) {
    ParseStatementInfo Info(&AsmStrRewrites);
    if (ParseStatement(Info))
      return true;

    if (Info.ParseError)
      return true;

    if (Info.Opcode == ~0U)
      continue;

    const MCInstrDesc &Desc = MII->get(Info.Opcode);

    // Build the list of clobbers, outputs and inputs.
    for (unsigned i = 1, e = Info.ParsedOperands.size(); i != e; ++i) {
      MCParsedAsmOperand *Operand = Info.ParsedOperands[i];

      // Immediate.
      if (Operand->isImm())
        continue;

      // Register operand.
      if (Operand->isReg() && !Operand->needAddressOf()) {
        unsigned NumDefs = Desc.getNumDefs();
        // Clobber.
        if (NumDefs && Operand->getMCOperandNum() < NumDefs)
          ClobberRegs.push_back(Operand->getReg());
        continue;
      }

      // Expr/Input or Output.
      StringRef SymName = Operand->getSymName();
      if (SymName.empty())
        continue;

      void *OpDecl = Operand->getOpDecl();
      if (!OpDecl)
        continue;

      bool isOutput = (i == 1) && Desc.mayStore();
      SMLoc Start = SMLoc::getFromPointer(SymName.data());
      if (isOutput) {
        ++InputIdx;
        OutputDecls.push_back(OpDecl);
        OutputDeclsAddressOf.push_back(Operand->needAddressOf());
        OutputConstraints.push_back('=' + Operand->getConstraint().str());
        AsmStrRewrites.push_back(AsmRewrite(AOK_Output, Start, SymName.size()));
      } else {
        InputDecls.push_back(OpDecl);
        InputDeclsAddressOf.push_back(Operand->needAddressOf());
        InputConstraints.push_back(Operand->getConstraint().str());
        AsmStrRewrites.push_back(AsmRewrite(AOK_Input, Start, SymName.size()));
      }
    }
  }

  // Set the number of Outputs and Inputs.
  NumOutputs = OutputDecls.size();
  NumInputs = InputDecls.size();

  // Set the unique clobbers.
  array_pod_sort(ClobberRegs.begin(), ClobberRegs.end());
  ClobberRegs.erase(std::unique(ClobberRegs.begin(), ClobberRegs.end()),
                    ClobberRegs.end());
  Clobbers.assign(ClobberRegs.size(), std::string());
  for (unsigned I = 0, E = ClobberRegs.size(); I != E; ++I) {
    raw_string_ostream OS(Clobbers[I]);
    IP->printRegName(OS, ClobberRegs[I]);
  }

  // Merge the various outputs and inputs.  Output are expected first.
  if (NumOutputs || NumInputs) {
    unsigned NumExprs = NumOutputs + NumInputs;
    OpDecls.resize(NumExprs);
    Constraints.resize(NumExprs);
    for (unsigned i = 0; i < NumOutputs; ++i) {
      OpDecls[i] = std::make_pair(OutputDecls[i], OutputDeclsAddressOf[i]);
      Constraints[i] = OutputConstraints[i];
    }
    for (unsigned i = 0, j = NumOutputs; i < NumInputs; ++i, ++j) {
      OpDecls[j] = std::make_pair(InputDecls[i], InputDeclsAddressOf[i]);
      Constraints[j] = InputConstraints[i];
    }
  }

  // Build the IR assembly string.
  std::string AsmStringIR;
  raw_string_ostream OS(AsmStringIR);
  const char *AsmStart = SrcMgr.getMemoryBuffer(0)->getBufferStart();
  const char *AsmEnd = SrcMgr.getMemoryBuffer(0)->getBufferEnd();
  array_pod_sort(AsmStrRewrites.begin(), AsmStrRewrites.end(), RewritesSort);
  for (SmallVectorImpl<AsmRewrite>::iterator I = AsmStrRewrites.begin(),
                                             E = AsmStrRewrites.end();
       I != E; ++I) {
    AsmRewriteKind Kind = (*I).Kind;
    if (Kind == AOK_Delete)
      continue;

    const char *Loc = (*I).Loc.getPointer();
    assert(Loc >= AsmStart && "Expected Loc to be at or after Start!");

    // Emit everything up to the immediate/expression.
    unsigned Len = Loc - AsmStart;
    if (Len)
      OS << StringRef(AsmStart, Len);

    // Skip the original expression.
    if (Kind == AOK_Skip) {
      AsmStart = Loc + (*I).Len;
      continue;
    }

    unsigned AdditionalSkip = 0;
    // Rewrite expressions in $N notation.
    switch (Kind) {
    default: break;
    case AOK_Imm:
      OS << "$$" << (*I).Val;
      break;
    case AOK_ImmPrefix:
      OS << "$$";
      break;
    case AOK_Input:
      OS << '$' << InputIdx++;
      break;
    case AOK_Output:
      OS << '$' << OutputIdx++;
      break;
    case AOK_SizeDirective:
      switch ((*I).Val) {
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
    case AOK_Align: {
      unsigned Val = (*I).Val;
      OS << ".align " << Val;

      // Skip the original immediate.
      assert(Val < 10 && "Expected alignment less then 2^10.");
      AdditionalSkip = (Val < 4) ? 2 : Val < 7 ? 3 : 4;
      break;
    }
    case AOK_DotOperator:
      OS << (*I).Val;
      break;
    }

    // Skip the original expression.
    AsmStart = Loc + (*I).Len + AdditionalSkip;
  }

  // Emit the remainder of the asm string.
  if (AsmStart != AsmEnd)
    OS << StringRef(AsmStart, AsmEnd - AsmStart);

  AsmString = OS.str();
  return false;
}

/// \brief Create an MCAsmParser instance.
MCAsmParser *llvm::createMCAsmParser(SourceMgr &SM,
                                     MCContext &C, MCStreamer &Out,
                                     const MCAsmInfo &MAI) {
  return new AsmParser(SM, C, Out, MAI);
}
