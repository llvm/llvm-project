//===- AsmParser.cpp - Parser for Assembly Files --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class implements the parser for assembly files.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeView.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCParser/AsmCond.h"
#include "llvm/MC/MCParser/AsmLexer.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCAsmParserExtension.h"
#include "llvm/MC/MCParser/MCAsmParserUtils.h"
#include "llvm/MC/MCParser/MCParsedAsmOperand.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace llvm;

extern cl::opt<unsigned> AsmMacroMaxNestingDepth;

namespace {

/// Helper types for tracking macro definitions.
typedef std::vector<AsmToken> MCAsmMacroArgument;
typedef std::vector<MCAsmMacroArgument> MCAsmMacroArguments;

/// Helper class for storing information about an active macro
/// instantiation.
struct MacroInstantiation {
  /// The location of the instantiation.
  SMLoc InstantiationLoc;

  /// The buffer where parsing should resume upon instantiation completion.
  unsigned ExitBuffer;

  /// The location where parsing should resume upon instantiation completion.
  SMLoc ExitLoc;

  /// The depth of TheCondStack at the start of the instantiation.
  size_t CondStackDepth;
};

struct ParseStatementInfo {
  /// The parsed operands from the last parsed statement.
  SmallVector<std::unique_ptr<MCParsedAsmOperand>, 8> ParsedOperands;

  /// The opcode from the last parsed instruction.
  unsigned Opcode = ~0U;

  /// Was there an error parsing the inline assembly?
  bool ParseError = false;

  SmallVectorImpl<AsmRewrite> *AsmRewrites = nullptr;

  ParseStatementInfo() = delete;
  ParseStatementInfo(SmallVectorImpl<AsmRewrite> *rewrites)
      : AsmRewrites(rewrites) {}
};

/// The concrete assembly parser instance.
// Note that this is a full MCAsmParser, not an MCAsmParserExtension!
// It's a peer of AsmParser, not of COFFAsmParser, WasmAsmParser, etc.
class MasmParser : public MCAsmParser {
private:
  AsmLexer Lexer;
  MCContext &Ctx;
  MCStreamer &Out;
  const MCAsmInfo &MAI;
  SourceMgr &SrcMgr;
  SourceMgr::DiagHandlerTy SavedDiagHandler;
  void *SavedDiagContext;
  std::unique_ptr<MCAsmParserExtension> PlatformParser;

  /// This is the current buffer index we're lexing from as managed by the
  /// SourceMgr object.
  unsigned CurBuffer;

  AsmCond TheCondState;
  std::vector<AsmCond> TheCondStack;

  /// maps directive names to handler methods in parser
  /// extensions. Extensions register themselves in this map by calling
  /// addDirectiveHandler.
  StringMap<ExtensionDirectiveHandler> ExtensionDirectiveMap;

  /// maps assembly-time variable names to variables
  struct Variable {
    StringRef Name;
    bool Redefinable = true;
    bool IsText = false;
    int64_t NumericValue = 0;
    std::string TextValue;
  };
  StringMap<Variable> Variables;

  /// Stack of active macro instantiations.
  std::vector<MacroInstantiation*> ActiveMacros;

  /// List of bodies of anonymous macros.
  std::deque<MCAsmMacro> MacroLikeBodies;

  /// Keeps track of how many .macro's have been instantiated.
  unsigned NumOfMacroInstantiations;

  /// The values from the last parsed cpp hash file line comment if any.
  struct CppHashInfoTy {
    StringRef Filename;
    int64_t LineNumber;
    SMLoc Loc;
    unsigned Buf;
    CppHashInfoTy() : Filename(), LineNumber(0), Loc(), Buf(0) {}
  };
  CppHashInfoTy CppHashInfo;

  /// The filename from the first cpp hash file line comment, if any.
  StringRef FirstCppHashFilename;

  /// List of forward directional labels for diagnosis at the end.
  SmallVector<std::tuple<SMLoc, CppHashInfoTy, MCSymbol *>, 4> DirLabels;

  /// AssemblerDialect. ~OU means unset value and use value provided by MAI.
  /// Defaults to 1U, meaning Intel.
  unsigned AssemblerDialect = 1U;

  /// is Darwin compatibility enabled?
  bool IsDarwin = false;

  /// Are we parsing ms-style inline assembly?
  bool ParsingMSInlineAsm = false;

  /// Did we already inform the user about inconsistent MD5 usage?
  bool ReportedInconsistentMD5 = false;

  // Is alt macro mode enabled.
  bool AltMacroMode = false;

public:
  MasmParser(SourceMgr &SM, MCContext &Ctx, MCStreamer &Out,
             const MCAsmInfo &MAI, unsigned CB);
  MasmParser(const MasmParser &) = delete;
  MasmParser &operator=(const MasmParser &) = delete;
  ~MasmParser() override;

  bool Run(bool NoInitialTextSection, bool NoFinalize = false) override;

  void addDirectiveHandler(StringRef Directive,
                           ExtensionDirectiveHandler Handler) override {
    ExtensionDirectiveMap[Directive] = Handler;
    if (DirectiveKindMap.find(Directive) == DirectiveKindMap.end()) {
      DirectiveKindMap[Directive] = DK_HANDLER_DIRECTIVE;
    }
  }

  void addAliasForDirective(StringRef Directive, StringRef Alias) override {
    DirectiveKindMap[Directive] = DirectiveKindMap[Alias];
  }

  /// @name MCAsmParser Interface
  /// {

  SourceMgr &getSourceManager() override { return SrcMgr; }
  MCAsmLexer &getLexer() override { return Lexer; }
  MCContext &getContext() override { return Ctx; }
  MCStreamer &getStreamer() override { return Out; }

  CodeViewContext &getCVContext() { return Ctx.getCVContext(); }

  unsigned getAssemblerDialect() override {
    if (AssemblerDialect == ~0U)
      return MAI.getAssemblerDialect();
    else
      return AssemblerDialect;
  }
  void setAssemblerDialect(unsigned i) override {
    AssemblerDialect = i;
  }

  void Note(SMLoc L, const Twine &Msg, SMRange Range = None) override;
  bool Warning(SMLoc L, const Twine &Msg, SMRange Range = None) override;
  bool printError(SMLoc L, const Twine &Msg, SMRange Range = None) override;

  const AsmToken &Lex() override;

  void setParsingMSInlineAsm(bool V) override {
    ParsingMSInlineAsm = V;
    // When parsing MS inline asm, we must lex 0b1101 and 0ABCH as binary and
    // hex integer literals.
    Lexer.setLexMasmIntegers(V);
  }
  bool isParsingMSInlineAsm() override { return ParsingMSInlineAsm; }

  bool parseMSInlineAsm(void *AsmLoc, std::string &AsmString,
                        unsigned &NumOutputs, unsigned &NumInputs,
                        SmallVectorImpl<std::pair<void *,bool>> &OpDecls,
                        SmallVectorImpl<std::string> &Constraints,
                        SmallVectorImpl<std::string> &Clobbers,
                        const MCInstrInfo *MII, const MCInstPrinter *IP,
                        MCAsmParserSemaCallback &SI) override;

  bool parseExpression(const MCExpr *&Res);
  bool parseExpression(const MCExpr *&Res, SMLoc &EndLoc) override;
  bool parsePrimaryExpr(const MCExpr *&Res, SMLoc &EndLoc) override;
  bool parseParenExpression(const MCExpr *&Res, SMLoc &EndLoc) override;
  bool parseParenExprOfDepth(unsigned ParenDepth, const MCExpr *&Res,
                             SMLoc &EndLoc) override;
  bool parseAbsoluteExpression(int64_t &Res) override;

  /// Parse a floating point expression using the float \p Semantics
  /// and set \p Res to the value.
  bool parseRealValue(const fltSemantics &Semantics, APInt &Res);

  /// Parse an identifier or string (as a quoted identifier)
  /// and set \p Res to the identifier contents.
  bool parseIdentifier(StringRef &Res) override;
  void eatToEndOfStatement() override;

  bool checkForValidSection() override;

  /// }

private:
  bool parseStatement(ParseStatementInfo &Info,
                      MCAsmParserSemaCallback *SI);
  bool parseCurlyBlockScope(SmallVectorImpl<AsmRewrite>& AsmStrRewrites);
  bool parseCppHashLineFilenameComment(SMLoc L);

  void checkForBadMacro(SMLoc DirectiveLoc, StringRef Name, StringRef Body,
                        ArrayRef<MCAsmMacroParameter> Parameters);
  bool expandMacro(raw_svector_ostream &OS, StringRef Body,
                   ArrayRef<MCAsmMacroParameter> Parameters,
                   ArrayRef<MCAsmMacroArgument> A, bool EnableAtPseudoVariable,
                   SMLoc L);

  /// Are we inside a macro instantiation?
  bool isInsideMacroInstantiation() {return !ActiveMacros.empty();}

  /// Handle entry to macro instantiation.
  ///
  /// \param M The macro.
  /// \param NameLoc Instantiation location.
  bool handleMacroEntry(const MCAsmMacro *M, SMLoc NameLoc);

  /// Handle exit from macro instantiation.
  void handleMacroExit();

  /// Extract AsmTokens for a macro argument.
  bool parseMacroArgument(MCAsmMacroArgument &MA, bool Vararg);

  /// Parse all macro arguments for a given macro.
  bool parseMacroArguments(const MCAsmMacro *M, MCAsmMacroArguments &A);

  void printMacroInstantiations();
  void printMessage(SMLoc Loc, SourceMgr::DiagKind Kind, const Twine &Msg,
                    SMRange Range = None) const {
    ArrayRef<SMRange> Ranges(Range);
    SrcMgr.PrintMessage(Loc, Kind, Msg, Ranges);
  }
  static void DiagHandler(const SMDiagnostic &Diag, void *Context);

  /// Should we emit DWARF describing this assembler source?  (Returns false if
  /// the source has .file directives, which means we don't want to generate
  /// info describing the assembler source itself.)
  bool enabledGenDwarfForAssembly();

  /// Enter the specified file. This returns true on failure.
  bool enterIncludeFile(const std::string &Filename);

  /// Reset the current lexer position to that given by \p Loc. The
  /// current token is not set; clients should ensure Lex() is called
  /// subsequently.
  ///
  /// \param InBuffer If not 0, should be the known buffer id that contains the
  /// location.
  void jumpToLoc(SMLoc Loc, unsigned InBuffer = 0);

  /// Parse up to the end of statement and a return the contents from the
  /// current token until the end of the statement; the current token on exit
  /// will be either the EndOfStatement or EOF.
  StringRef parseStringToEndOfStatement() override;

  bool parseTextItem(std::string &Data);

  unsigned getBinOpPrecedence(AsmToken::TokenKind K,
                              MCBinaryExpr::Opcode &Kind);

  bool parseBinOpRHS(unsigned Precedence, const MCExpr *&Res, SMLoc &EndLoc);
  bool parseParenExpr(const MCExpr *&Res, SMLoc &EndLoc);
  bool parseBracketExpr(const MCExpr *&Res, SMLoc &EndLoc);

  bool parseRegisterOrRegisterNumber(int64_t &Register, SMLoc DirectiveLoc);

  bool parseCVFunctionId(int64_t &FunctionId, StringRef DirectiveName);
  bool parseCVFileId(int64_t &FileId, StringRef DirectiveName);

  // Generic (target and platform independent) directive parsing.
  enum DirectiveKind {
    DK_NO_DIRECTIVE, // Placeholder
    DK_HANDLER_DIRECTIVE,
    DK_ASSIGN,
    DK_EQU,
    DK_TEXTEQU,
    DK_ASCII,
    DK_ASCIZ,
    DK_STRING,
    DK_BYTE,
    DK_SBYTE,
    DK_WORD,
    DK_SWORD,
    DK_DWORD,
    DK_SDWORD,
    DK_FWORD,
    DK_QWORD,
    DK_SQWORD,
    DK_DB,
    DK_DD,
    DK_DQ,
    DK_DW,
    DK_REAL4,
    DK_REAL8,
    DK_ALIGN,
    DK_ORG,
    DK_ENDR,
    DK_EXTERN,
    DK_PUBLIC,
    DK_COMM,
    DK_COMMENT,
    DK_INCLUDE,
    DK_REPT,
    DK_IRP,
    DK_IRPC,
    DK_IF,
    DK_IFE,
    DK_IFB,
    DK_IFNB,
    DK_IFDEF,
    DK_IFNDEF,
    DK_IFDIF,
    DK_IFDIFI,
    DK_IFIDN,
    DK_IFIDNI,
    DK_ELSEIF,
    DK_ELSEIFE,
    DK_ELSEIFB,
    DK_ELSEIFNB,
    DK_ELSEIFDEF,
    DK_ELSEIFNDEF,
    DK_ELSEIFDIF,
    DK_ELSEIFDIFI,
    DK_ELSEIFIDN,
    DK_ELSEIFIDNI,
    DK_ELSE,
    DK_ENDIF,
    DK_FILE,
    DK_LINE,
    DK_LOC,
    DK_STABS,
    DK_CV_FILE,
    DK_CV_FUNC_ID,
    DK_CV_INLINE_SITE_ID,
    DK_CV_LOC,
    DK_CV_LINETABLE,
    DK_CV_INLINE_LINETABLE,
    DK_CV_DEF_RANGE,
    DK_CV_STRINGTABLE,
    DK_CV_STRING,
    DK_CV_FILECHECKSUMS,
    DK_CV_FILECHECKSUM_OFFSET,
    DK_CV_FPO_DATA,
    DK_CFI_SECTIONS,
    DK_CFI_STARTPROC,
    DK_CFI_ENDPROC,
    DK_CFI_DEF_CFA,
    DK_CFI_DEF_CFA_OFFSET,
    DK_CFI_ADJUST_CFA_OFFSET,
    DK_CFI_DEF_CFA_REGISTER,
    DK_CFI_OFFSET,
    DK_CFI_REL_OFFSET,
    DK_CFI_PERSONALITY,
    DK_CFI_LSDA,
    DK_CFI_REMEMBER_STATE,
    DK_CFI_RESTORE_STATE,
    DK_CFI_SAME_VALUE,
    DK_CFI_RESTORE,
    DK_CFI_ESCAPE,
    DK_CFI_RETURN_COLUMN,
    DK_CFI_SIGNAL_FRAME,
    DK_CFI_UNDEFINED,
    DK_CFI_REGISTER,
    DK_CFI_WINDOW_SAVE,
    DK_CFI_B_KEY_FRAME,
    DK_ALTMACRO,
    DK_NOALTMACRO,
    DK_MACRO,
    DK_EXITM,
    DK_ENDM,
    DK_PURGEM,
    DK_ERR,
    DK_ERRB,
    DK_ERRNB,
    DK_ERRDEF,
    DK_ERRNDEF,
    DK_ERRDIF,
    DK_ERRDIFI,
    DK_ERRIDN,
    DK_ERRIDNI,
    DK_ERRE,
    DK_ERRNZ,
    DK_ECHO,
    DK_END
  };

  /// Maps directive name --> DirectiveKind enum, for
  /// directives parsed by this class.
  StringMap<DirectiveKind> DirectiveKindMap;

  // Codeview def_range type parsing.
  enum CVDefRangeType {
    CVDR_DEFRANGE = 0, // Placeholder
    CVDR_DEFRANGE_REGISTER,
    CVDR_DEFRANGE_FRAMEPOINTER_REL,
    CVDR_DEFRANGE_SUBFIELD_REGISTER,
    CVDR_DEFRANGE_REGISTER_REL
  };

  /// Maps Codeview def_range types --> CVDefRangeType enum, for
  /// Codeview def_range types parsed by this class.
  StringMap<CVDefRangeType> CVDefRangeTypeMap;

  bool parseInitValue(unsigned Size);

  // ".ascii", ".asciz", ".string"
  bool parseDirectiveAscii(StringRef IDVal, bool ZeroTerminated);

  // "byte", "word", ...
  bool parseScalarInstList(unsigned Size,
                           SmallVectorImpl<const MCExpr *> &Values);
  bool parseDirectiveValue(StringRef IDVal, unsigned Size);
  bool parseDirectiveNamedValue(StringRef IDVal, unsigned Size, StringRef Name,
                                SMLoc NameLoc);

  // "real4", "real8"
  bool parseDirectiveRealValue(StringRef IDVal, const fltSemantics &Semantics);
  bool parseRealInstList(const fltSemantics &Semantics,
                         SmallVectorImpl<APInt> &Values);
  bool parseDirectiveNamedRealValue(StringRef IDVal,
                                    const fltSemantics &Semantics,
                                    StringRef Name, SMLoc NameLoc);

  // "=", "equ", "textequ"
  bool parseDirectiveEquate(StringRef IDVal, StringRef Name,
                            DirectiveKind DirKind);

  bool parseDirectiveOrg(); // ".org"
  bool parseDirectiveAlign();  // "align"

  // ".file", ".line", ".loc", ".stabs"
  bool parseDirectiveFile(SMLoc DirectiveLoc);
  bool parseDirectiveLine();
  bool parseDirectiveLoc();
  bool parseDirectiveStabs();

  // ".cv_file", ".cv_func_id", ".cv_inline_site_id", ".cv_loc", ".cv_linetable",
  // ".cv_inline_linetable", ".cv_def_range", ".cv_string"
  bool parseDirectiveCVFile();
  bool parseDirectiveCVFuncId();
  bool parseDirectiveCVInlineSiteId();
  bool parseDirectiveCVLoc();
  bool parseDirectiveCVLinetable();
  bool parseDirectiveCVInlineLinetable();
  bool parseDirectiveCVDefRange();
  bool parseDirectiveCVString();
  bool parseDirectiveCVStringTable();
  bool parseDirectiveCVFileChecksums();
  bool parseDirectiveCVFileChecksumOffset();
  bool parseDirectiveCVFPOData();

  // .cfi directives
  bool parseDirectiveCFIRegister(SMLoc DirectiveLoc);
  bool parseDirectiveCFIWindowSave();
  bool parseDirectiveCFISections();
  bool parseDirectiveCFIStartProc();
  bool parseDirectiveCFIEndProc();
  bool parseDirectiveCFIDefCfaOffset();
  bool parseDirectiveCFIDefCfa(SMLoc DirectiveLoc);
  bool parseDirectiveCFIAdjustCfaOffset();
  bool parseDirectiveCFIDefCfaRegister(SMLoc DirectiveLoc);
  bool parseDirectiveCFIOffset(SMLoc DirectiveLoc);
  bool parseDirectiveCFIRelOffset(SMLoc DirectiveLoc);
  bool parseDirectiveCFIPersonalityOrLsda(bool IsPersonality);
  bool parseDirectiveCFIRememberState();
  bool parseDirectiveCFIRestoreState();
  bool parseDirectiveCFISameValue(SMLoc DirectiveLoc);
  bool parseDirectiveCFIRestore(SMLoc DirectiveLoc);
  bool parseDirectiveCFIEscape();
  bool parseDirectiveCFIReturnColumn(SMLoc DirectiveLoc);
  bool parseDirectiveCFISignalFrame();
  bool parseDirectiveCFIUndefined(SMLoc DirectiveLoc);

  // macro directives
  bool parseDirectivePurgeMacro(SMLoc DirectiveLoc);
  bool parseDirectiveExitMacro(StringRef Directive);
  bool parseDirectiveEndMacro(StringRef Directive);
  bool parseDirectiveMacro(SMLoc DirectiveLoc);
  // alternate macro mode directives
  bool parseDirectiveAltmacro(StringRef Directive);

  /// Parse a directive like ".globl" which
  /// accepts a single symbol (which should be a label or an external).
  bool parseDirectiveSymbolAttribute(MCSymbolAttr Attr);

  bool parseDirectiveComm(bool IsLocal); // ".comm" and ".lcomm"

  bool parseDirectiveComment(SMLoc DirectiveLoc); // "comment"

  bool parseDirectiveInclude(); // "include"

  // "if" or "ife"
  bool parseDirectiveIf(SMLoc DirectiveLoc, DirectiveKind DirKind);
  // "ifb" or "ifnb", depending on ExpectBlank.
  bool parseDirectiveIfb(SMLoc DirectiveLoc, bool ExpectBlank);
  // "ifidn", "ifdif", "ifidni", or "ifdifi", depending on ExpectEqual and
  // CaseInsensitive.
  bool parseDirectiveIfidn(SMLoc DirectiveLoc, bool ExpectEqual,
                           bool CaseInsensitive);
  // "ifdef" or "ifndef", depending on expect_defined
  bool parseDirectiveIfdef(SMLoc DirectiveLoc, bool expect_defined);
  // "elseif" or "elseife"
  bool parseDirectiveElseIf(SMLoc DirectiveLoc, DirectiveKind DirKind);
  // "elseifb" or "elseifnb", depending on ExpectBlank.
  bool parseDirectiveElseIfb(SMLoc DirectiveLoc, bool ExpectBlank);
  // ".elseifdef" or ".elseifndef", depending on expect_defined
  bool parseDirectiveElseIfdef(SMLoc DirectiveLoc, bool expect_defined);
  // "elseifidn", "elseifdif", "elseifidni", or "elseifdifi", depending on
  // ExpectEqual and CaseInsensitive.
  bool parseDirectiveElseIfidn(SMLoc DirectiveLoc, bool ExpectEqual,
                               bool CaseInsensitive);
  bool parseDirectiveElse(SMLoc DirectiveLoc);   // "else"
  bool parseDirectiveEndIf(SMLoc DirectiveLoc);  // "endif"
  bool parseEscapedString(std::string &Data) override;
  bool parseAngleBracketString(std::string &Data) override;

  // Macro-like directives
  MCAsmMacro *parseMacroLikeBody(SMLoc DirectiveLoc);
  void instantiateMacroLikeBody(MCAsmMacro *M, SMLoc DirectiveLoc,
                                raw_svector_ostream &OS);
  bool parseDirectiveRept(SMLoc DirectiveLoc, StringRef Directive);
  bool parseDirectiveIrp(SMLoc DirectiveLoc);  // ".irp"
  bool parseDirectiveIrpc(SMLoc DirectiveLoc); // ".irpc"
  bool parseDirectiveEndr(SMLoc DirectiveLoc); // ".endr"

  // "_emit" or "__emit"
  bool parseDirectiveMSEmit(SMLoc DirectiveLoc, ParseStatementInfo &Info,
                            size_t Len);

  // "align"
  bool parseDirectiveMSAlign(SMLoc DirectiveLoc, ParseStatementInfo &Info);

  // "end"
  bool parseDirectiveEnd(SMLoc DirectiveLoc);

  // ".err"
  bool parseDirectiveError(SMLoc DirectiveLoc);
  // ".errb" or ".errnb", depending on ExpectBlank.
  bool parseDirectiveErrorIfb(SMLoc DirectiveLoc, bool ExpectBlank);
  // ".errdef" or ".errndef", depending on ExpectBlank.
  bool parseDirectiveErrorIfdef(SMLoc DirectiveLoc, bool ExpectDefined);
  // ".erridn", ".errdif", ".erridni", or ".errdifi", depending on ExpectEqual
  // and CaseInsensitive.
  bool parseDirectiveErrorIfidn(SMLoc DirectiveLoc, bool ExpectEqual,
                                bool CaseInsensitive);
  // ".erre" or ".errnz", depending on ExpectZero.
  bool parseDirectiveErrorIfe(SMLoc DirectiveLoc, bool ExpectZero);

  // "echo"
  bool parseDirectiveEcho();

  void initializeDirectiveKindMap();
  void initializeCVDefRangeTypeMap();
};

} // end anonymous namespace

namespace llvm {

extern MCAsmParserExtension *createCOFFMasmParser();

} // end namespace llvm

enum { DEFAULT_ADDRSPACE = 0 };

MasmParser::MasmParser(SourceMgr &SM, MCContext &Ctx, MCStreamer &Out,
                       const MCAsmInfo &MAI, unsigned CB = 0)
    : Lexer(MAI), Ctx(Ctx), Out(Out), MAI(MAI), SrcMgr(SM),
      CurBuffer(CB ? CB : SM.getMainFileID()) {
  HadError = false;
  // Save the old handler.
  SavedDiagHandler = SrcMgr.getDiagHandler();
  SavedDiagContext = SrcMgr.getDiagContext();
  // Set our own handler which calls the saved handler.
  SrcMgr.setDiagHandler(DiagHandler, this);
  Lexer.setBuffer(SrcMgr.getMemoryBuffer(CurBuffer)->getBuffer());

  // Initialize the platform / file format parser.
  switch (Ctx.getObjectFileInfo()->getObjectFileType()) {
  case MCObjectFileInfo::IsCOFF:
    PlatformParser.reset(createCOFFMasmParser());
    break;
  default:
    report_fatal_error("llvm-ml currently supports only COFF output.");
    break;
  }

  initializeDirectiveKindMap();
  PlatformParser->Initialize(*this);
  initializeCVDefRangeTypeMap();

  NumOfMacroInstantiations = 0;
}

MasmParser::~MasmParser() {
  assert((HadError || ActiveMacros.empty()) &&
         "Unexpected active macro instantiation!");

  // Restore the saved diagnostics handler and context for use during
  // finalization.
  SrcMgr.setDiagHandler(SavedDiagHandler, SavedDiagContext);
}

void MasmParser::printMacroInstantiations() {
  // Print the active macro instantiation stack.
  for (std::vector<MacroInstantiation *>::const_reverse_iterator
           it = ActiveMacros.rbegin(),
           ie = ActiveMacros.rend();
       it != ie; ++it)
    printMessage((*it)->InstantiationLoc, SourceMgr::DK_Note,
                 "while in macro instantiation");
}

void MasmParser::Note(SMLoc L, const Twine &Msg, SMRange Range) {
  printPendingErrors();
  printMessage(L, SourceMgr::DK_Note, Msg, Range);
  printMacroInstantiations();
}

bool MasmParser::Warning(SMLoc L, const Twine &Msg, SMRange Range) {
  if (getTargetParser().getTargetOptions().MCNoWarn)
    return false;
  if (getTargetParser().getTargetOptions().MCFatalWarnings)
    return Error(L, Msg, Range);
  printMessage(L, SourceMgr::DK_Warning, Msg, Range);
  printMacroInstantiations();
  return false;
}

bool MasmParser::printError(SMLoc L, const Twine &Msg, SMRange Range) {
  HadError = true;
  printMessage(L, SourceMgr::DK_Error, Msg, Range);
  printMacroInstantiations();
  return true;
}

bool MasmParser::enterIncludeFile(const std::string &Filename) {
  std::string IncludedFile;
  unsigned NewBuf =
      SrcMgr.AddIncludeFile(Filename, Lexer.getLoc(), IncludedFile);
  if (!NewBuf)
    return true;

  CurBuffer = NewBuf;
  Lexer.setBuffer(SrcMgr.getMemoryBuffer(CurBuffer)->getBuffer());
  return false;
}

void MasmParser::jumpToLoc(SMLoc Loc, unsigned InBuffer) {
  CurBuffer = InBuffer ? InBuffer : SrcMgr.FindBufferContainingLoc(Loc);
  Lexer.setBuffer(SrcMgr.getMemoryBuffer(CurBuffer)->getBuffer(),
                  Loc.getPointer());
}

const AsmToken &MasmParser::Lex() {
  if (Lexer.getTok().is(AsmToken::Error))
    Error(Lexer.getErrLoc(), Lexer.getErr());

  // if it's a end of statement with a comment in it
  if (getTok().is(AsmToken::EndOfStatement)) {
    // if this is a line comment output it.
    if (!getTok().getString().empty() && getTok().getString().front() != '\n' &&
        getTok().getString().front() != '\r' && MAI.preserveAsmComments())
      Out.addExplicitComment(Twine(getTok().getString()));
  }

  const AsmToken *tok = &Lexer.Lex();

  while (tok->is(AsmToken::Identifier)) {
    auto it = Variables.find(tok->getIdentifier());
    if (it != Variables.end() && it->second.IsText) {
      std::unique_ptr<MemoryBuffer> Instantiation =
          MemoryBuffer::getMemBufferCopy(it->second.TextValue,
                                         "<instantiation>");

      // Jump to the macro instantiation and prime the lexer.
      CurBuffer = SrcMgr.AddNewSourceBuffer(std::move(Instantiation),
                                            getTok().getEndLoc());
      Lexer.setBuffer(SrcMgr.getMemoryBuffer(CurBuffer)->getBuffer(), nullptr,
                      /*EndStatementAtEOF=*/false);
      tok = &Lexer.Lex();
    } else {
      break;
    }
  }

  // Parse comments here to be deferred until end of next statement.
  while (tok->is(AsmToken::Comment)) {
    if (MAI.preserveAsmComments())
      Out.addExplicitComment(Twine(tok->getString()));
    tok = &Lexer.Lex();
  }

  if (tok->is(AsmToken::Eof)) {
    // If this is the end of an included file, pop the parent file off the
    // include stack.
    SMLoc ParentIncludeLoc = SrcMgr.getParentIncludeLoc(CurBuffer);
    if (ParentIncludeLoc != SMLoc()) {
      jumpToLoc(ParentIncludeLoc);
      return Lex();
    }
  }

  return *tok;
}

bool MasmParser::enabledGenDwarfForAssembly() {
  // Check whether the user specified -g.
  if (!getContext().getGenDwarfForAssembly())
    return false;
  // If we haven't encountered any .file directives (which would imply that
  // the assembler source was produced with debug info already) then emit one
  // describing the assembler source file itself.
  if (getContext().getGenDwarfFileNumber() == 0) {
    // Use the first #line directive for this, if any. It's preprocessed, so
    // there is no checksum, and of course no source directive.
    if (!FirstCppHashFilename.empty())
      getContext().setMCLineTableRootFile(/*CUID=*/0,
                                          getContext().getCompilationDir(),
                                          FirstCppHashFilename,
                                          /*Cksum=*/None, /*Source=*/None);
    const MCDwarfFile &RootFile =
        getContext().getMCDwarfLineTable(/*CUID=*/0).getRootFile();
    getContext().setGenDwarfFileNumber(getStreamer().emitDwarfFileDirective(
        /*CUID=*/0, getContext().getCompilationDir(), RootFile.Name,
        RootFile.Checksum, RootFile.Source));
  }
  return true;
}

bool MasmParser::Run(bool NoInitialTextSection, bool NoFinalize) {
  // Create the initial section, if requested.
  if (!NoInitialTextSection)
    Out.InitSections(false);

  // Prime the lexer.
  Lex();

  HadError = false;
  AsmCond StartingCondState = TheCondState;
  SmallVector<AsmRewrite, 4> AsmStrRewrites;

  // If we are generating dwarf for assembly source files save the initial text
  // section.  (Don't use enabledGenDwarfForAssembly() here, as we aren't
  // emitting any actual debug info yet and haven't had a chance to parse any
  // embedded .file directives.)
  if (getContext().getGenDwarfForAssembly()) {
    MCSection *Sec = getStreamer().getCurrentSectionOnly();
    if (!Sec->getBeginSymbol()) {
      MCSymbol *SectionStartSym = getContext().createTempSymbol();
      getStreamer().emitLabel(SectionStartSym);
      Sec->setBeginSymbol(SectionStartSym);
    }
    bool InsertResult = getContext().addGenDwarfSection(Sec);
    assert(InsertResult && ".text section should not have debug info yet");
    (void)InsertResult;
  }

  // While we have input, parse each statement.
  while (Lexer.isNot(AsmToken::Eof)) {
    ParseStatementInfo Info(&AsmStrRewrites);
    bool Parsed = parseStatement(Info, nullptr);

    // If we have a Lexer Error we are on an Error Token. Load in Lexer Error
    // for printing ErrMsg via Lex() only if no (presumably better) parser error
    // exists.
    if (Parsed && !hasPendingError() && Lexer.getTok().is(AsmToken::Error)) {
      Lex();
    }

    // parseStatement returned true so may need to emit an error.
    printPendingErrors();

    // Skipping to the next line if needed.
    if (Parsed && !getLexer().isAtStartOfStatement())
      eatToEndOfStatement();
  }

  getTargetParser().onEndOfFile();
  printPendingErrors();

  // All errors should have been emitted.
  assert(!hasPendingError() && "unexpected error from parseStatement");

  getTargetParser().flushPendingInstructions(getStreamer());

  if (TheCondState.TheCond != StartingCondState.TheCond ||
      TheCondState.Ignore != StartingCondState.Ignore)
    printError(getTok().getLoc(), "unmatched .ifs or .elses");
  // Check to see there are no empty DwarfFile slots.
  const auto &LineTables = getContext().getMCDwarfLineTables();
  if (!LineTables.empty()) {
    unsigned Index = 0;
    for (const auto &File : LineTables.begin()->second.getMCDwarfFiles()) {
      if (File.Name.empty() && Index != 0)
        printError(getTok().getLoc(), "unassigned file number: " +
                                          Twine(Index) +
                                          " for .file directives");
      ++Index;
    }
  }

  // Check to see that all assembler local symbols were actually defined.
  // Targets that don't do subsections via symbols may not want this, though,
  // so conservatively exclude them. Only do this if we're finalizing, though,
  // as otherwise we won't necessarilly have seen everything yet.
  if (!NoFinalize) {
    if (MAI.hasSubsectionsViaSymbols()) {
      for (const auto &TableEntry : getContext().getSymbols()) {
        MCSymbol *Sym = TableEntry.getValue();
        // Variable symbols may not be marked as defined, so check those
        // explicitly. If we know it's a variable, we have a definition for
        // the purposes of this check.
        if (Sym->isTemporary() && !Sym->isVariable() && !Sym->isDefined())
          // FIXME: We would really like to refer back to where the symbol was
          // first referenced for a source location. We need to add something
          // to track that. Currently, we just point to the end of the file.
          printError(getTok().getLoc(), "assembler local symbol '" +
                                            Sym->getName() + "' not defined");
      }
    }

    // Temporary symbols like the ones for directional jumps don't go in the
    // symbol table. They also need to be diagnosed in all (final) cases.
    for (std::tuple<SMLoc, CppHashInfoTy, MCSymbol *> &LocSym : DirLabels) {
      if (std::get<2>(LocSym)->isUndefined()) {
        // Reset the state of any "# line file" directives we've seen to the
        // context as it was at the diagnostic site.
        CppHashInfo = std::get<1>(LocSym);
        printError(std::get<0>(LocSym), "directional label undefined");
      }
    }
  }

  // Finalize the output stream if there are no errors and if the client wants
  // us to.
  if (!HadError && !NoFinalize)
    Out.Finish();

  return HadError || getContext().hadError();
}

bool MasmParser::checkForValidSection() {
  if (!ParsingMSInlineAsm && !getStreamer().getCurrentSectionOnly()) {
    Out.InitSections(false);
    return Error(getTok().getLoc(),
                 "expected section directive before assembly directive");
  }
  return false;
}

/// Throw away the rest of the line for testing purposes.
void MasmParser::eatToEndOfStatement() {
  while (Lexer.isNot(AsmToken::EndOfStatement) && Lexer.isNot(AsmToken::Eof))
    Lexer.Lex();

  // Eat EOL.
  if (Lexer.is(AsmToken::EndOfStatement))
    Lexer.Lex();
}

StringRef MasmParser::parseStringToEndOfStatement() {
  const char *Start = getTok().getLoc().getPointer();

  while (Lexer.isNot(AsmToken::EndOfStatement) && Lexer.isNot(AsmToken::Eof))
    Lexer.Lex();

  const char *End = getTok().getLoc().getPointer();
  return StringRef(Start, End - Start);
}

/// Parse a paren expression and return it.
/// NOTE: This assumes the leading '(' has already been consumed.
///
/// parenexpr ::= expr)
///
bool MasmParser::parseParenExpr(const MCExpr *&Res, SMLoc &EndLoc) {
  if (parseExpression(Res))
    return true;
  if (Lexer.isNot(AsmToken::RParen))
    return TokError("expected ')' in parentheses expression");
  EndLoc = Lexer.getTok().getEndLoc();
  Lex();
  return false;
}

/// Parse a bracket expression and return it.
/// NOTE: This assumes the leading '[' has already been consumed.
///
/// bracketexpr ::= expr]
///
bool MasmParser::parseBracketExpr(const MCExpr *&Res, SMLoc &EndLoc) {
  if (parseExpression(Res))
    return true;
  EndLoc = getTok().getEndLoc();
  if (parseToken(AsmToken::RBrac, "expected ']' in brackets expression"))
    return true;
  return false;
}

/// Parse a primary expression and return it.
///  primaryexpr ::= (parenexpr
///  primaryexpr ::= symbol
///  primaryexpr ::= number
///  primaryexpr ::= '.'
///  primaryexpr ::= ~,+,- primaryexpr
bool MasmParser::parsePrimaryExpr(const MCExpr *&Res, SMLoc &EndLoc) {
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
    if (parsePrimaryExpr(Res, EndLoc))
      return true;
    Res = MCUnaryExpr::createLNot(Res, getContext(), FirstTokenLoc);
    return false;
  case AsmToken::Dollar:
  case AsmToken::At:
  case AsmToken::String:
  case AsmToken::Identifier: {
    StringRef Identifier;
    if (parseIdentifier(Identifier)) {
      // We may have failed but $ may be a valid token.
      if (getTok().is(AsmToken::Dollar)) {
        if (Lexer.getMAI().getDollarIsPC()) {
          Lex();
          // This is a '$' reference, which references the current PC.  Emit a
          // temporary label to the streamer and refer to it.
          MCSymbol *Sym = Ctx.createTempSymbol();
          Out.emitLabel(Sym);
          Res = MCSymbolRefExpr::create(Sym, MCSymbolRefExpr::VK_None,
                                        getContext());
          EndLoc = FirstTokenLoc;
          return false;
        }
        return Error(FirstTokenLoc, "invalid token in expression");
      }
    }
    // Parse symbol variant
    std::pair<StringRef, StringRef> Split;
    if (!MAI.useParensForSymbolVariant()) {
      if (FirstTokenKind == AsmToken::String) {
        if (Lexer.is(AsmToken::At)) {
          Lex(); // eat @
          SMLoc AtLoc = getLexer().getLoc();
          StringRef VName;
          if (parseIdentifier(VName))
            return Error(AtLoc, "expected symbol variant after '@'");

          Split = std::make_pair(Identifier, VName);
        }
      } else {
        Split = Identifier.split('@');
      }
    } else if (Lexer.is(AsmToken::LParen)) {
      Lex(); // eat '('.
      StringRef VName;
      parseIdentifier(VName);
      // eat ')'.
      if (parseToken(AsmToken::RParen,
                     "unexpected token in variant, expected ')'"))
        return true;
      Split = std::make_pair(Identifier, VName);
    }

    EndLoc = SMLoc::getFromPointer(Identifier.end());

    // This is a symbol reference.
    StringRef SymbolName = Identifier;
    if (SymbolName.empty())
      return Error(getLexer().getLoc(), "expected a symbol reference");

    MCSymbolRefExpr::VariantKind Variant = MCSymbolRefExpr::VK_None;

    // Lookup the symbol variant if used.
    if (!Split.second.empty()) {
      Variant = MCSymbolRefExpr::getVariantKindForName(Split.second);
      if (Variant != MCSymbolRefExpr::VK_Invalid) {
        SymbolName = Split.first;
      } else if (MAI.doesAllowAtInName() && !MAI.useParensForSymbolVariant()) {
        Variant = MCSymbolRefExpr::VK_None;
      } else {
        return Error(SMLoc::getFromPointer(Split.second.begin()),
                     "invalid variant '" + Split.second + "'");
      }
    }

    MCSymbol *Sym = getContext().getInlineAsmLabel(SymbolName);
    if (!Sym)
      Sym = getContext().getOrCreateSymbol(SymbolName);

    // If this is an absolute variable reference, substitute it now to preserve
    // semantics in the face of reassignment.
    if (Sym->isVariable()) {
      auto V = Sym->getVariableValue(/*SetUsed*/ false);
      bool DoInline = isa<MCConstantExpr>(V) && !Variant;
      if (auto TV = dyn_cast<MCTargetExpr>(V))
        DoInline = TV->inlineAssignedExpr();
      if (DoInline) {
        if (Variant)
          return Error(EndLoc, "unexpected modifier on variable reference");
        Res = Sym->getVariableValue(/*SetUsed*/ false);
        return false;
      }
    }

    // Otherwise create a symbol ref.
    Res = MCSymbolRefExpr::create(Sym, Variant, getContext(), FirstTokenLoc);
    return false;
  }
  case AsmToken::BigNum:
    return TokError("literal value out of range for directive");
  case AsmToken::Integer: {
    SMLoc Loc = getTok().getLoc();
    int64_t IntVal = getTok().getIntVal();
    Res = MCConstantExpr::create(IntVal, getContext());
    EndLoc = Lexer.getTok().getEndLoc();
    Lex(); // Eat token.
    // Look for 'b' or 'f' following an Integer as a directional label
    if (Lexer.getKind() == AsmToken::Identifier) {
      StringRef IDVal = getTok().getString();
      // Lookup the symbol variant if used.
      std::pair<StringRef, StringRef> Split = IDVal.split('@');
      MCSymbolRefExpr::VariantKind Variant = MCSymbolRefExpr::VK_None;
      if (Split.first.size() != IDVal.size()) {
        Variant = MCSymbolRefExpr::getVariantKindForName(Split.second);
        if (Variant == MCSymbolRefExpr::VK_Invalid)
          return TokError("invalid variant '" + Split.second + "'");
        IDVal = Split.first;
      }
      if (IDVal == "f" || IDVal == "b") {
        MCSymbol *Sym =
            Ctx.getDirectionalLocalSymbol(IntVal, IDVal == "b");
        Res = MCSymbolRefExpr::create(Sym, Variant, getContext());
        if (IDVal == "b" && Sym->isUndefined())
          return Error(Loc, "directional label undefined");
        DirLabels.push_back(std::make_tuple(Loc, CppHashInfo, Sym));
        EndLoc = Lexer.getTok().getEndLoc();
        Lex(); // Eat identifier.
      }
    }
    return false;
  }
  case AsmToken::Real: {
    APFloat RealVal(APFloat::IEEEdouble(), getTok().getString());
    uint64_t IntVal = RealVal.bitcastToAPInt().getZExtValue();
    Res = MCConstantExpr::create(IntVal, getContext());
    EndLoc = Lexer.getTok().getEndLoc();
    Lex(); // Eat token.
    return false;
  }
  case AsmToken::Dot: {
    // This is a '.' reference, which references the current PC.  Emit a
    // temporary label to the streamer and refer to it.
    MCSymbol *Sym = Ctx.createTempSymbol();
    Out.emitLabel(Sym);
    Res = MCSymbolRefExpr::create(Sym, MCSymbolRefExpr::VK_None, getContext());
    EndLoc = Lexer.getTok().getEndLoc();
    Lex(); // Eat identifier.
    return false;
  }
  case AsmToken::LParen:
    Lex(); // Eat the '('.
    return parseParenExpr(Res, EndLoc);
  case AsmToken::LBrac:
    if (!PlatformParser->HasBracketExpressions())
      return TokError("brackets expression not supported on this target");
    Lex(); // Eat the '['.
    return parseBracketExpr(Res, EndLoc);
  case AsmToken::Minus:
    Lex(); // Eat the operator.
    if (parsePrimaryExpr(Res, EndLoc))
      return true;
    Res = MCUnaryExpr::createMinus(Res, getContext(), FirstTokenLoc);
    return false;
  case AsmToken::Plus:
    Lex(); // Eat the operator.
    if (parsePrimaryExpr(Res, EndLoc))
      return true;
    Res = MCUnaryExpr::createPlus(Res, getContext(), FirstTokenLoc);
    return false;
  case AsmToken::Tilde:
    Lex(); // Eat the operator.
    if (parsePrimaryExpr(Res, EndLoc))
      return true;
    Res = MCUnaryExpr::createNot(Res, getContext(), FirstTokenLoc);
    return false;
  // MIPS unary expression operators. The lexer won't generate these tokens if
  // MCAsmInfo::HasMipsExpressions is false for the target.
  case AsmToken::PercentCall16:
  case AsmToken::PercentCall_Hi:
  case AsmToken::PercentCall_Lo:
  case AsmToken::PercentDtprel_Hi:
  case AsmToken::PercentDtprel_Lo:
  case AsmToken::PercentGot:
  case AsmToken::PercentGot_Disp:
  case AsmToken::PercentGot_Hi:
  case AsmToken::PercentGot_Lo:
  case AsmToken::PercentGot_Ofst:
  case AsmToken::PercentGot_Page:
  case AsmToken::PercentGottprel:
  case AsmToken::PercentGp_Rel:
  case AsmToken::PercentHi:
  case AsmToken::PercentHigher:
  case AsmToken::PercentHighest:
  case AsmToken::PercentLo:
  case AsmToken::PercentNeg:
  case AsmToken::PercentPcrel_Hi:
  case AsmToken::PercentPcrel_Lo:
  case AsmToken::PercentTlsgd:
  case AsmToken::PercentTlsldm:
  case AsmToken::PercentTprel_Hi:
  case AsmToken::PercentTprel_Lo:
    Lex(); // Eat the operator.
    if (Lexer.isNot(AsmToken::LParen))
      return TokError("expected '(' after operator");
    Lex(); // Eat the operator.
    if (parseExpression(Res, EndLoc))
      return true;
    if (Lexer.isNot(AsmToken::RParen))
      return TokError("expected ')'");
    Lex(); // Eat the operator.
    Res = getTargetParser().createTargetUnaryExpr(Res, FirstTokenKind, Ctx);
    return !Res;
  }
}

bool MasmParser::parseExpression(const MCExpr *&Res) {
  SMLoc EndLoc;
  return parseExpression(Res, EndLoc);
}

/// This function checks if the next token is <string> type or arithmetic.
/// string that begin with character '<' must end with character '>'.
/// otherwise it is arithmetics.
/// If the function returns a 'true' value,
/// the End argument will be filled with the last location pointed to the '>'
/// character.

/// There is a gap between the AltMacro's documentation and the single quote
/// implementation. GCC does not fully support this feature and so we will not
/// support it.
/// TODO: Adding single quote as a string.
static bool isAngleBracketString(SMLoc &StrLoc, SMLoc &EndLoc) {
  assert((StrLoc.getPointer() != nullptr) &&
         "Argument to the function cannot be a NULL value");
  const char *CharPtr = StrLoc.getPointer();
  while ((*CharPtr != '>') && (*CharPtr != '\n') && (*CharPtr != '\r') &&
         (*CharPtr != '\0')) {
    if (*CharPtr == '!')
      CharPtr++;
    CharPtr++;
  }
  if (*CharPtr == '>') {
    EndLoc = StrLoc.getFromPointer(CharPtr + 1);
    return true;
  }
  return false;
}

/// creating a string without the escape characters '!'.
static std::string angleBracketString(StringRef AltMacroStr) {
  std::string Res;
  for (size_t Pos = 0; Pos < AltMacroStr.size(); Pos++) {
    if (AltMacroStr[Pos] == '!')
      Pos++;
    Res += AltMacroStr[Pos];
  }
  return Res;
}

/// Parse an expression and return it.
///
///  expr ::= expr &&,|| expr               -> lowest.
///  expr ::= expr |,^,&,! expr
///  expr ::= expr ==,!=,<>,<,<=,>,>= expr
///  expr ::= expr <<,>> expr
///  expr ::= expr +,- expr
///  expr ::= expr *,/,% expr               -> highest.
///  expr ::= primaryexpr
///
bool MasmParser::parseExpression(const MCExpr *&Res, SMLoc &EndLoc) {
  // Parse the expression.
  Res = nullptr;
  if (getTargetParser().parsePrimaryExpr(Res, EndLoc) ||
      parseBinOpRHS(1, Res, EndLoc))
    return true;

  // Try to constant fold it up front, if possible. Do not exploit
  // assembler here.
  int64_t Value;
  if (Res->evaluateAsAbsolute(Value))
    Res = MCConstantExpr::create(Value, getContext());

  return false;
}

bool MasmParser::parseParenExpression(const MCExpr *&Res, SMLoc &EndLoc) {
  Res = nullptr;
  return parseParenExpr(Res, EndLoc) || parseBinOpRHS(1, Res, EndLoc);
}

bool MasmParser::parseParenExprOfDepth(unsigned ParenDepth, const MCExpr *&Res,
                                       SMLoc &EndLoc) {
  if (parseParenExpr(Res, EndLoc))
    return true;

  for (; ParenDepth > 0; --ParenDepth) {
    if (parseBinOpRHS(1, Res, EndLoc))
      return true;

    // We don't Lex() the last RParen.
    // This is the same behavior as parseParenExpression().
    if (ParenDepth - 1 > 0) {
      EndLoc = getTok().getEndLoc();
      if (parseToken(AsmToken::RParen,
                     "expected ')' in parentheses expression"))
        return true;
    }
  }
  return false;
}

bool MasmParser::parseAbsoluteExpression(int64_t &Res) {
  const MCExpr *Expr;

  SMLoc StartLoc = Lexer.getLoc();
  if (parseExpression(Expr))
    return true;

  if (!Expr->evaluateAsAbsolute(Res, getStreamer().getAssemblerPtr()))
    return Error(StartLoc, "expected absolute expression");

  return false;
}

static unsigned getGNUBinOpPrecedence(AsmToken::TokenKind K,
                                      MCBinaryExpr::Opcode &Kind,
                                      bool ShouldUseLogicalShr) {
  switch (K) {
  default:
    return 0; // not a binop.

  // Lowest Precedence: &&, ||
  case AsmToken::AmpAmp:
    Kind = MCBinaryExpr::LAnd;
    return 2;
  case AsmToken::PipePipe:
    Kind = MCBinaryExpr::LOr;
    return 1;

  // Low Precedence: ==, !=, <>, <, <=, >, >=
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

  // Low Intermediate Precedence: +, -
  case AsmToken::Plus:
    Kind = MCBinaryExpr::Add;
    return 4;
  case AsmToken::Minus:
    Kind = MCBinaryExpr::Sub;
    return 4;

  // High Intermediate Precedence: |, &, ^
  //
  // FIXME: gas seems to support '!' as an infix operator?
  case AsmToken::Pipe:
    Kind = MCBinaryExpr::Or;
    return 5;
  case AsmToken::Caret:
    Kind = MCBinaryExpr::Xor;
    return 5;
  case AsmToken::Amp:
    Kind = MCBinaryExpr::And;
    return 5;

  // Highest Precedence: *, /, %, <<, >>
  case AsmToken::Star:
    Kind = MCBinaryExpr::Mul;
    return 6;
  case AsmToken::Slash:
    Kind = MCBinaryExpr::Div;
    return 6;
  case AsmToken::Percent:
    Kind = MCBinaryExpr::Mod;
    return 6;
  case AsmToken::LessLess:
    Kind = MCBinaryExpr::Shl;
    return 6;
  case AsmToken::GreaterGreater:
    Kind = ShouldUseLogicalShr ? MCBinaryExpr::LShr : MCBinaryExpr::AShr;
    return 6;
  }
}

unsigned MasmParser::getBinOpPrecedence(AsmToken::TokenKind K,
                                        MCBinaryExpr::Opcode &Kind) {
  bool ShouldUseLogicalShr = MAI.shouldUseLogicalShr();
  return getGNUBinOpPrecedence(K, Kind, ShouldUseLogicalShr);
}

/// Parse all binary operators with precedence >= 'Precedence'.
/// Res contains the LHS of the expression on input.
bool MasmParser::parseBinOpRHS(unsigned Precedence, const MCExpr *&Res,
                               SMLoc &EndLoc) {
  SMLoc StartLoc = Lexer.getLoc();
  while (true) {
    MCBinaryExpr::Opcode Kind = MCBinaryExpr::Add;
    unsigned TokPrec = getBinOpPrecedence(Lexer.getKind(), Kind);

    // If the next token is lower precedence than we are allowed to eat, return
    // successfully with what we ate already.
    if (TokPrec < Precedence)
      return false;

    Lex();

    // Eat the next primary expression.
    const MCExpr *RHS;
    if (getTargetParser().parsePrimaryExpr(RHS, EndLoc))
      return true;

    // If BinOp binds less tightly with RHS than the operator after RHS, let
    // the pending operator take RHS as its LHS.
    MCBinaryExpr::Opcode Dummy;
    unsigned NextTokPrec = getBinOpPrecedence(Lexer.getKind(), Dummy);
    if (TokPrec < NextTokPrec && parseBinOpRHS(TokPrec + 1, RHS, EndLoc))
      return true;

    // Merge LHS and RHS according to operator.
    Res = MCBinaryExpr::create(Kind, Res, RHS, getContext(), StartLoc);
  }
}

/// ParseStatement:
///   ::= EndOfStatement
///   ::= Label* Directive ...Operands... EndOfStatement
///   ::= Label* Identifier OperandList* EndOfStatement
bool MasmParser::parseStatement(ParseStatementInfo &Info,
                                MCAsmParserSemaCallback *SI) {
  assert(!hasPendingError() && "parseStatement started with pending error");
  // Eat initial spaces and comments
  while (Lexer.is(AsmToken::Space))
    Lex();
  if (Lexer.is(AsmToken::EndOfStatement)) {
    // if this is a line comment we can drop it safely
    if (getTok().getString().empty() || getTok().getString().front() == '\r' ||
        getTok().getString().front() == '\n')
      Out.AddBlankLine();
    Lex();
    return false;
  }
  // Statements always start with an identifier, unless we're dealing with a
  // processor directive (.386, .686, etc.) that lexes as a real.
  AsmToken ID = getTok();
  SMLoc IDLoc = ID.getLoc();
  StringRef IDVal;
  int64_t LocalLabelVal = -1;
  if (Lexer.is(AsmToken::HashDirective))
    return parseCppHashLineFilenameComment(IDLoc);
  // Allow an integer followed by a ':' as a directional local label.
  if (Lexer.is(AsmToken::Integer)) {
    LocalLabelVal = getTok().getIntVal();
    if (LocalLabelVal < 0) {
      if (!TheCondState.Ignore) {
        Lex(); // always eat a token
        return Error(IDLoc, "unexpected token at start of statement");
      }
      IDVal = "";
    } else {
      IDVal = getTok().getString();
      Lex(); // Consume the integer token to be used as an identifier token.
      if (Lexer.getKind() != AsmToken::Colon) {
        if (!TheCondState.Ignore) {
          Lex(); // always eat a token
          return Error(IDLoc, "unexpected token at start of statement");
        }
      }
    }
  } else if (Lexer.is(AsmToken::Dot)) {
    // Treat '.' as a valid identifier in this context.
    Lex();
    IDVal = ".";
  } else if (Lexer.is(AsmToken::LCurly)) {
    // Treat '{' as a valid identifier in this context.
    Lex();
    IDVal = "{";

  } else if (Lexer.is(AsmToken::RCurly)) {
    // Treat '}' as a valid identifier in this context.
    Lex();
    IDVal = "}";
  } else if (Lexer.is(AsmToken::Star) &&
             getTargetParser().starIsStartOfStatement()) {
    // Accept '*' as a valid start of statement.
    Lex();
    IDVal = "*";
  } else if (Lexer.is(AsmToken::Real)) {
    // Treat ".<number>" as a valid identifier in this context.
    IDVal = getTok().getString();
    Lex(); // always eat a token
    if (!IDVal.startswith("."))
      return Error(IDLoc, "unexpected token at start of statement");
  } else if (parseIdentifier(IDVal)) {
    if (!TheCondState.Ignore) {
      Lex(); // always eat a token
      return Error(IDLoc, "unexpected token at start of statement");
    }
    IDVal = "";
  }

  // Handle conditional assembly here before checking for skipping.  We
  // have to do this so that .endif isn't skipped in a ".if 0" block for
  // example.
  StringMap<DirectiveKind>::const_iterator DirKindIt =
      DirectiveKindMap.find(IDVal.lower());
  DirectiveKind DirKind = (DirKindIt == DirectiveKindMap.end())
                              ? DK_NO_DIRECTIVE
                              : DirKindIt->getValue();
  switch (DirKind) {
  default:
    break;
  case DK_IF:
  case DK_IFE:
    return parseDirectiveIf(IDLoc, DirKind);
  case DK_IFB:
    return parseDirectiveIfb(IDLoc, true);
  case DK_IFNB:
    return parseDirectiveIfb(IDLoc, false);
  case DK_IFDEF:
    return parseDirectiveIfdef(IDLoc, true);
  case DK_IFNDEF:
    return parseDirectiveIfdef(IDLoc, false);
  case DK_IFDIF:
    return parseDirectiveIfidn(IDLoc, /*ExpectEqual=*/false,
                               /*CaseInsensitive=*/false);
  case DK_IFDIFI:
    return parseDirectiveIfidn(IDLoc, /*ExpectEqual=*/false,
                               /*CaseInsensitive=*/true);
  case DK_IFIDN:
    return parseDirectiveIfidn(IDLoc, /*ExpectEqual=*/true,
                               /*CaseInsensitive=*/false);
  case DK_IFIDNI:
    return parseDirectiveIfidn(IDLoc, /*ExpectEqual=*/true,
                               /*CaseInsensitive=*/true);
  case DK_ELSEIF:
  case DK_ELSEIFE:
    return parseDirectiveElseIf(IDLoc, DirKind);
  case DK_ELSEIFB:
    return parseDirectiveElseIfb(IDLoc, true);
  case DK_ELSEIFNB:
    return parseDirectiveElseIfb(IDLoc, false);
  case DK_ELSEIFDEF:
    return parseDirectiveElseIfdef(IDLoc, true);
  case DK_ELSEIFNDEF:
    return parseDirectiveElseIfdef(IDLoc, false);
  case DK_ELSEIFDIF:
    return parseDirectiveElseIfidn(IDLoc, /*ExpectEqual=*/false,
                                   /*CaseInsensitive=*/false);
  case DK_ELSEIFDIFI:
    return parseDirectiveElseIfidn(IDLoc, /*ExpectEqual=*/false,
                                   /*CaseInsensitive=*/true);
  case DK_ELSEIFIDN:
    return parseDirectiveElseIfidn(IDLoc, /*ExpectEqual=*/true,
                                   /*CaseInsensitive=*/false);
  case DK_ELSEIFIDNI:
    return parseDirectiveElseIfidn(IDLoc, /*ExpectEqual=*/true,
                                   /*CaseInsensitive=*/true);
  case DK_ELSE:
    return parseDirectiveElse(IDLoc);
  case DK_ENDIF:
    return parseDirectiveEndIf(IDLoc);
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
    if (!getTargetParser().isLabel(ID))
      break;
    if (checkForValidSection())
      return true;

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
    if (LocalLabelVal == -1) {
      if (ParsingMSInlineAsm && SI) {
        StringRef RewrittenLabel =
            SI->LookupInlineAsmLabel(IDVal, getSourceManager(), IDLoc, true);
        assert(!RewrittenLabel.empty() &&
               "We should have an internal name here.");
        Info.AsmRewrites->emplace_back(AOK_Label, IDLoc, IDVal.size(),
                                       RewrittenLabel);
        IDVal = RewrittenLabel;
      }
      Sym = getContext().getOrCreateSymbol(IDVal);
    } else
      Sym = Ctx.createDirectionalLocalSymbol(LocalLabelVal);
    // End of Labels should be treated as end of line for lexing
    // purposes but that information is not available to the Lexer who
    // does not understand Labels. This may cause us to see a Hash
    // here instead of a preprocessor line comment.
    if (getTok().is(AsmToken::Hash)) {
      StringRef CommentStr = parseStringToEndOfStatement();
      Lexer.Lex();
      Lexer.UnLex(AsmToken(AsmToken::EndOfStatement, CommentStr));
    }

    // Consume any end of statement token, if present, to avoid spurious
    // AddBlankLine calls().
    if (getTok().is(AsmToken::EndOfStatement)) {
      Lex();
    }

    getTargetParser().doBeforeLabelEmit(Sym);

    // Emit the label.
    if (!getTargetParser().isParsingMSInlineAsm())
      Out.emitLabel(Sym, IDLoc);

    // If we are generating dwarf for assembly source files then gather the
    // info to make a dwarf label entry for this label if needed.
    if (enabledGenDwarfForAssembly())
      MCGenDwarfLabelEntry::Make(Sym, &getStreamer(), getSourceManager(),
                                 IDLoc);

    getTargetParser().onLabelParsed(Sym);

    return false;
  }

  default: // Normal instruction or directive.
    break;
  }

  // If macros are enabled, check to see if this is a macro instantiation.
  if (const MCAsmMacro *M = getContext().lookupMacro(IDVal)) {
    return handleMacroEntry(M, IDLoc);
  }

  // Otherwise, we have a normal instruction or directive.

  if (DirKind != DK_NO_DIRECTIVE) {
    // There are several entities interested in parsing directives:
    //
    // 1. Asm parser extensions. For example, platform-specific parsers
    //    (like the ELF parser) register themselves as extensions.
    // 2. The target-specific assembly parser. Some directives are target
    //    specific or may potentially behave differently on certain targets.
    // 3. The generic directive parser implemented by this class. These are
    //    all the directives that behave in a target and platform independent
    //    manner, or at least have a default behavior that's shared between
    //    all targets and platforms.

    getTargetParser().flushPendingInstructions(getStreamer());

    // First, check the extension directive map to see if any extension has
    // registered itself to parse this directive.
    std::pair<MCAsmParserExtension *, DirectiveHandler> Handler =
        ExtensionDirectiveMap.lookup(IDVal.lower());
    if (Handler.first)
      return (*Handler.second)(Handler.first, IDVal, IDLoc);

    // Next, let the target-specific assembly parser try.
    SMLoc StartTokLoc = getTok().getLoc();
    bool TPDirectiveReturn =
        ID.is(AsmToken::Identifier) && getTargetParser().ParseDirective(ID);

    if (hasPendingError())
      return true;
    // Currently the return value should be true if we are
    // uninterested but as this is at odds with the standard parsing
    // convention (return true = error) we have instances of a parsed
    // directive that fails returning true as an error. Catch these
    // cases as best as possible errors here.
    if (TPDirectiveReturn && StartTokLoc != getTok().getLoc())
      return true;
    // Return if we did some parsing or believe we succeeded.
    if (!TPDirectiveReturn || StartTokLoc != getTok().getLoc())
      return false;

    // Finally, if no one else is interested in this directive, it must be
    // generic and familiar to this class.
    switch (DirKind) {
    default:
      break;
    case DK_ASCII:
      return parseDirectiveAscii(IDVal, false);
    case DK_ASCIZ:
    case DK_STRING:
      return parseDirectiveAscii(IDVal, true);
    case DK_BYTE:
    case DK_SBYTE:
    case DK_DB:
      return parseDirectiveValue(IDVal, 1);
    case DK_WORD:
    case DK_SWORD:
    case DK_DW:
      return parseDirectiveValue(IDVal, 2);
    case DK_DWORD:
    case DK_SDWORD:
    case DK_DD:
      return parseDirectiveValue(IDVal, 4);
    case DK_FWORD:
      return parseDirectiveValue(IDVal, 6);
    case DK_QWORD:
    case DK_SQWORD:
    case DK_DQ:
      return parseDirectiveValue(IDVal, 8);
    case DK_REAL4:
      return parseDirectiveRealValue(IDVal, APFloat::IEEEsingle());
    case DK_REAL8:
      return parseDirectiveRealValue(IDVal, APFloat::IEEEdouble());
    case DK_ALIGN:
      return parseDirectiveAlign();
    case DK_ORG:
      return parseDirectiveOrg();
    case DK_EXTERN:
      eatToEndOfStatement(); // .extern is the default, ignore it.
      return false;
    case DK_PUBLIC:
      return parseDirectiveSymbolAttribute(MCSA_Global);
    case DK_COMM:
      return parseDirectiveComm(/*IsLocal=*/false);
    case DK_COMMENT:
      return parseDirectiveComment(IDLoc);
    case DK_INCLUDE:
      return parseDirectiveInclude();
    case DK_REPT:
      return parseDirectiveRept(IDLoc, IDVal);
    case DK_IRP:
      return parseDirectiveIrp(IDLoc);
    case DK_IRPC:
      return parseDirectiveIrpc(IDLoc);
    case DK_ENDR:
      return parseDirectiveEndr(IDLoc);
    case DK_FILE:
      return parseDirectiveFile(IDLoc);
    case DK_LINE:
      return parseDirectiveLine();
    case DK_LOC:
      return parseDirectiveLoc();
    case DK_STABS:
      return parseDirectiveStabs();
    case DK_CV_FILE:
      return parseDirectiveCVFile();
    case DK_CV_FUNC_ID:
      return parseDirectiveCVFuncId();
    case DK_CV_INLINE_SITE_ID:
      return parseDirectiveCVInlineSiteId();
    case DK_CV_LOC:
      return parseDirectiveCVLoc();
    case DK_CV_LINETABLE:
      return parseDirectiveCVLinetable();
    case DK_CV_INLINE_LINETABLE:
      return parseDirectiveCVInlineLinetable();
    case DK_CV_DEF_RANGE:
      return parseDirectiveCVDefRange();
    case DK_CV_STRING:
      return parseDirectiveCVString();
    case DK_CV_STRINGTABLE:
      return parseDirectiveCVStringTable();
    case DK_CV_FILECHECKSUMS:
      return parseDirectiveCVFileChecksums();
    case DK_CV_FILECHECKSUM_OFFSET:
      return parseDirectiveCVFileChecksumOffset();
    case DK_CV_FPO_DATA:
      return parseDirectiveCVFPOData();
    case DK_CFI_SECTIONS:
      return parseDirectiveCFISections();
    case DK_CFI_STARTPROC:
      return parseDirectiveCFIStartProc();
    case DK_CFI_ENDPROC:
      return parseDirectiveCFIEndProc();
    case DK_CFI_DEF_CFA:
      return parseDirectiveCFIDefCfa(IDLoc);
    case DK_CFI_DEF_CFA_OFFSET:
      return parseDirectiveCFIDefCfaOffset();
    case DK_CFI_ADJUST_CFA_OFFSET:
      return parseDirectiveCFIAdjustCfaOffset();
    case DK_CFI_DEF_CFA_REGISTER:
      return parseDirectiveCFIDefCfaRegister(IDLoc);
    case DK_CFI_OFFSET:
      return parseDirectiveCFIOffset(IDLoc);
    case DK_CFI_REL_OFFSET:
      return parseDirectiveCFIRelOffset(IDLoc);
    case DK_CFI_PERSONALITY:
      return parseDirectiveCFIPersonalityOrLsda(true);
    case DK_CFI_LSDA:
      return parseDirectiveCFIPersonalityOrLsda(false);
    case DK_CFI_REMEMBER_STATE:
      return parseDirectiveCFIRememberState();
    case DK_CFI_RESTORE_STATE:
      return parseDirectiveCFIRestoreState();
    case DK_CFI_SAME_VALUE:
      return parseDirectiveCFISameValue(IDLoc);
    case DK_CFI_RESTORE:
      return parseDirectiveCFIRestore(IDLoc);
    case DK_CFI_ESCAPE:
      return parseDirectiveCFIEscape();
    case DK_CFI_RETURN_COLUMN:
      return parseDirectiveCFIReturnColumn(IDLoc);
    case DK_CFI_SIGNAL_FRAME:
      return parseDirectiveCFISignalFrame();
    case DK_CFI_UNDEFINED:
      return parseDirectiveCFIUndefined(IDLoc);
    case DK_CFI_REGISTER:
      return parseDirectiveCFIRegister(IDLoc);
    case DK_CFI_WINDOW_SAVE:
      return parseDirectiveCFIWindowSave();
    case DK_MACRO:
      return parseDirectiveMacro(IDLoc);
    case DK_ALTMACRO:
    case DK_NOALTMACRO:
      return parseDirectiveAltmacro(IDVal);
    case DK_EXITM:
      return parseDirectiveExitMacro(IDVal);
    case DK_ENDM:
      return parseDirectiveEndMacro(IDVal);
    case DK_PURGEM:
      return parseDirectivePurgeMacro(IDLoc);
    case DK_END:
      return parseDirectiveEnd(IDLoc);
    case DK_ERR:
      return parseDirectiveError(IDLoc);
    case DK_ERRB:
      return parseDirectiveErrorIfb(IDLoc, true);
    case DK_ERRNB:
      return parseDirectiveErrorIfb(IDLoc, false);
    case DK_ERRDEF:
      return parseDirectiveErrorIfdef(IDLoc, true);
    case DK_ERRNDEF:
      return parseDirectiveErrorIfdef(IDLoc, false);
    case DK_ERRDIF:
      return parseDirectiveErrorIfidn(IDLoc, /*ExpectEqual=*/false,
                                      /*CaseInsensitive=*/false);
    case DK_ERRDIFI:
      return parseDirectiveErrorIfidn(IDLoc, /*ExpectEqual=*/false,
                                      /*CaseInsensitive=*/true);
    case DK_ERRIDN:
      return parseDirectiveErrorIfidn(IDLoc, /*ExpectEqual=*/true,
                                      /*CaseInsensitive=*/false);
    case DK_ERRIDNI:
      return parseDirectiveErrorIfidn(IDLoc, /*ExpectEqual=*/true,
                                      /*CaseInsensitive=*/true);
    case DK_ERRE:
      return parseDirectiveErrorIfe(IDLoc, true);
    case DK_ERRNZ:
      return parseDirectiveErrorIfe(IDLoc, false);
    case DK_ECHO:
      return parseDirectiveEcho();
    }

    return Error(IDLoc, "unknown directive");
  }

  // Non-conditional Microsoft directives sometimes follow their first argument.
  const AsmToken nextTok = getTok();
  const StringRef nextVal = nextTok.getString();
  const SMLoc nextLoc = nextTok.getLoc();

  // There are several entities interested in parsing infix directives:
  //
  // 1. Asm parser extensions. For example, platform-specific parsers
  //    (like the ELF parser) register themselves as extensions.
  // 2. The generic directive parser implemented by this class. These are
  //    all the directives that behave in a target and platform independent
  //    manner, or at least have a default behavior that's shared between
  //    all targets and platforms.

  getTargetParser().flushPendingInstructions(getStreamer());

  // First, check the extension directive map to see if any extension has
  // registered itself to parse this directive.
  std::pair<MCAsmParserExtension *, DirectiveHandler> Handler =
      ExtensionDirectiveMap.lookup(nextVal.lower());
  if (Handler.first) {
    Lex();
    Lexer.UnLex(ID);
    return (*Handler.second)(Handler.first, nextVal, nextLoc);
  }

  // Finally, if no one else is interested in this directive, it must be
  // generic and familiar to this class.
  DirKindIt = DirectiveKindMap.find(nextVal.lower());
  DirKind = (DirKindIt == DirectiveKindMap.end())
                ? DK_NO_DIRECTIVE
                : DirKindIt->getValue();
  switch (DirKind) {
  default:
    break;
  case DK_ASSIGN:
  case DK_EQU:
  case DK_TEXTEQU:
    Lex();
    return parseDirectiveEquate(nextVal, IDVal, DirKind);
  case DK_BYTE:
  case DK_DB:
    Lex();
    return parseDirectiveNamedValue(nextVal, 1, IDVal, IDLoc);
  case DK_WORD:
  case DK_DW:
    Lex();
    return parseDirectiveNamedValue(nextVal, 2, IDVal, IDLoc);
  case DK_DWORD:
  case DK_DD:
    Lex();
    return parseDirectiveNamedValue(nextVal, 4, IDVal, IDLoc);
  case DK_FWORD:
    Lex();
    return parseDirectiveNamedValue(nextVal, 6, IDVal, IDLoc);
  case DK_QWORD:
  case DK_DQ:
    Lex();
    return parseDirectiveNamedValue(nextVal, 8, IDVal, IDLoc);
  case DK_REAL4:
    Lex();
    return parseDirectiveNamedRealValue(nextVal, APFloat::IEEEsingle(), IDVal,
                                        IDLoc);
  case DK_REAL8:
    Lex();
    return parseDirectiveNamedRealValue(nextVal, APFloat::IEEEdouble(), IDVal,
                                        IDLoc);
  }

  // __asm _emit or __asm __emit
  if (ParsingMSInlineAsm && (IDVal == "_emit" || IDVal == "__emit" ||
                             IDVal == "_EMIT" || IDVal == "__EMIT"))
    return parseDirectiveMSEmit(IDLoc, Info, IDVal.size());

  // __asm align
  if (ParsingMSInlineAsm && (IDVal == "align" || IDVal == "ALIGN"))
    return parseDirectiveMSAlign(IDLoc, Info);

  if (ParsingMSInlineAsm && (IDVal == "even" || IDVal == "EVEN"))
    Info.AsmRewrites->emplace_back(AOK_EVEN, IDLoc, 4);
  if (checkForValidSection())
    return true;

  // Canonicalize the opcode to lower case.
  std::string OpcodeStr = IDVal.lower();
  ParseInstructionInfo IInfo(Info.AsmRewrites);
  bool ParseHadError = getTargetParser().ParseInstruction(IInfo, OpcodeStr, ID,
                                                          Info.ParsedOperands);
  Info.ParseError = ParseHadError;

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

    printMessage(IDLoc, SourceMgr::DK_Note, OS.str());
  }

  // Fail even if ParseInstruction erroneously returns false.
  if (hasPendingError() || ParseHadError)
    return true;

  // If we are generating dwarf for the current section then generate a .loc
  // directive for the instruction.
  if (!ParseHadError && enabledGenDwarfForAssembly() &&
      getContext().getGenDwarfSectionSyms().count(
          getStreamer().getCurrentSectionOnly())) {
    unsigned Line;
    if (ActiveMacros.empty())
      Line = SrcMgr.FindLineNumber(IDLoc, CurBuffer);
    else
      Line = SrcMgr.FindLineNumber(ActiveMacros.front()->InstantiationLoc,
                                   ActiveMacros.front()->ExitBuffer);

    // If we previously parsed a cpp hash file line comment then make sure the
    // current Dwarf File is for the CppHashFilename if not then emit the
    // Dwarf File table for it and adjust the line number for the .loc.
    if (!CppHashInfo.Filename.empty()) {
      unsigned FileNumber = getStreamer().emitDwarfFileDirective(
          0, StringRef(), CppHashInfo.Filename);
      getContext().setGenDwarfFileNumber(FileNumber);

      unsigned CppHashLocLineNo =
        SrcMgr.FindLineNumber(CppHashInfo.Loc, CppHashInfo.Buf);
      Line = CppHashInfo.LineNumber - 1 + (Line - CppHashLocLineNo);
    }

    getStreamer().emitDwarfLocDirective(
        getContext().getGenDwarfFileNumber(), Line, 0,
        DWARF2_LINE_DEFAULT_IS_STMT ? DWARF2_FLAG_IS_STMT : 0, 0, 0,
        StringRef());
  }

  // If parsing succeeded, match the instruction.
  if (!ParseHadError) {
    uint64_t ErrorInfo;
    if (getTargetParser().MatchAndEmitInstruction(
            IDLoc, Info.Opcode, Info.ParsedOperands, Out, ErrorInfo,
            getTargetParser().isParsingMSInlineAsm()))
      return true;
  }
  return false;
}

// Parse and erase curly braces marking block start/end
bool MasmParser::parseCurlyBlockScope(
    SmallVectorImpl<AsmRewrite> &AsmStrRewrites) {
  // Identify curly brace marking block start/end
  if (Lexer.isNot(AsmToken::LCurly) && Lexer.isNot(AsmToken::RCurly))
    return false;

  SMLoc StartLoc = Lexer.getLoc();
  Lex(); // Eat the brace
  if (Lexer.is(AsmToken::EndOfStatement))
    Lex(); // Eat EndOfStatement following the brace

  // Erase the block start/end brace from the output asm string
  AsmStrRewrites.emplace_back(AOK_Skip, StartLoc, Lexer.getLoc().getPointer() -
                                                  StartLoc.getPointer());
  return true;
}

/// parseCppHashLineFilenameComment as this:
///   ::= # number "filename"
bool MasmParser::parseCppHashLineFilenameComment(SMLoc L) {
  Lex(); // Eat the hash token.
  // Lexer only ever emits HashDirective if it fully formed if it's
  // done the checking already so this is an internal error.
  assert(getTok().is(AsmToken::Integer) &&
         "Lexing Cpp line comment: Expected Integer");
  int64_t LineNumber = getTok().getIntVal();
  Lex();
  assert(getTok().is(AsmToken::String) &&
         "Lexing Cpp line comment: Expected String");
  StringRef Filename = getTok().getString();
  Lex();

  // Get rid of the enclosing quotes.
  Filename = Filename.substr(1, Filename.size() - 2);

  // Save the SMLoc, Filename and LineNumber for later use by diagnostics
  // and possibly DWARF file info.
  CppHashInfo.Loc = L;
  CppHashInfo.Filename = Filename;
  CppHashInfo.LineNumber = LineNumber;
  CppHashInfo.Buf = CurBuffer;
  if (FirstCppHashFilename.empty())
    FirstCppHashFilename = Filename;
  return false;
}

/// will use the last parsed cpp hash line filename comment
/// for the Filename and LineNo if any in the diagnostic.
void MasmParser::DiagHandler(const SMDiagnostic &Diag, void *Context) {
  const MasmParser *Parser = static_cast<const MasmParser *>(Context);
  raw_ostream &OS = errs();

  const SourceMgr &DiagSrcMgr = *Diag.getSourceMgr();
  SMLoc DiagLoc = Diag.getLoc();
  unsigned DiagBuf = DiagSrcMgr.FindBufferContainingLoc(DiagLoc);
  unsigned CppHashBuf =
      Parser->SrcMgr.FindBufferContainingLoc(Parser->CppHashInfo.Loc);

  // Like SourceMgr::printMessage() we need to print the include stack if any
  // before printing the message.
  unsigned DiagCurBuffer = DiagSrcMgr.FindBufferContainingLoc(DiagLoc);
  if (!Parser->SavedDiagHandler && DiagCurBuffer &&
      DiagCurBuffer != DiagSrcMgr.getMainFileID()) {
    SMLoc ParentIncludeLoc = DiagSrcMgr.getParentIncludeLoc(DiagCurBuffer);
    DiagSrcMgr.PrintIncludeStack(ParentIncludeLoc, OS);
  }

  // If we have not parsed a cpp hash line filename comment or the source
  // manager changed or buffer changed (like in a nested include) then just
  // print the normal diagnostic using its Filename and LineNo.
  if (!Parser->CppHashInfo.LineNumber || &DiagSrcMgr != &Parser->SrcMgr ||
      DiagBuf != CppHashBuf) {
    if (Parser->SavedDiagHandler)
      Parser->SavedDiagHandler(Diag, Parser->SavedDiagContext);
    else
      Diag.print(nullptr, OS);
    return;
  }

  // Use the CppHashFilename and calculate a line number based on the
  // CppHashInfo.Loc and CppHashInfo.LineNumber relative to this Diag's SMLoc
  // for the diagnostic.
  const std::string &Filename = std::string(Parser->CppHashInfo.Filename);

  int DiagLocLineNo = DiagSrcMgr.FindLineNumber(DiagLoc, DiagBuf);
  int CppHashLocLineNo =
      Parser->SrcMgr.FindLineNumber(Parser->CppHashInfo.Loc, CppHashBuf);
  int LineNo =
      Parser->CppHashInfo.LineNumber - 1 + (DiagLocLineNo - CppHashLocLineNo);

  SMDiagnostic NewDiag(*Diag.getSourceMgr(), Diag.getLoc(), Filename, LineNo,
                       Diag.getColumnNo(), Diag.getKind(), Diag.getMessage(),
                       Diag.getLineContents(), Diag.getRanges());

  if (Parser->SavedDiagHandler)
    Parser->SavedDiagHandler(NewDiag, Parser->SavedDiagContext);
  else
    NewDiag.print(nullptr, OS);
}

// FIXME: This is mostly duplicated from the function in AsmLexer.cpp. The
// difference being that that function accepts '@' as part of identifiers and
// we can't do that. AsmLexer.cpp should probably be changed to handle
// '@' as a special case when needed.
static bool isIdentifierChar(char c) {
  return isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '$' ||
         c == '.';
}

bool MasmParser::expandMacro(raw_svector_ostream &OS, StringRef Body,
                             ArrayRef<MCAsmMacroParameter> Parameters,
                             ArrayRef<MCAsmMacroArgument> A,
                             bool EnableAtPseudoVariable, SMLoc L) {
  unsigned NParameters = Parameters.size();
  bool HasVararg = NParameters ? Parameters.back().Vararg : false;
  if ((!IsDarwin || NParameters != 0) && NParameters != A.size())
    return Error(L, "Wrong number of arguments");

  // A macro without parameters is handled differently on Darwin:
  // gas accepts no arguments and does no substitutions
  while (!Body.empty()) {
    // Scan for the next substitution.
    std::size_t End = Body.size(), Pos = 0;
    for (; Pos != End; ++Pos) {
      // Check for a substitution or escape.
      if (IsDarwin && !NParameters) {
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

    if (IsDarwin && !NParameters) {
      switch (Body[Pos + 1]) {
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
        unsigned Index = Body[Pos + 1] - '0';
        if (Index >= A.size())
          break;

        // Otherwise substitute with the token values, with spaces eliminated.
        for (const AsmToken &Token : A[Index])
          OS << Token.getString();
        break;
      }
      }
      Pos += 2;
    } else {
      unsigned I = Pos + 1;

      // Check for the \@ pseudo-variable.
      if (EnableAtPseudoVariable && Body[I] == '@' && I + 1 != End)
        ++I;
      else
        while (isIdentifierChar(Body[I]) && I + 1 != End)
          ++I;

      const char *Begin = Body.data() + Pos + 1;
      StringRef Argument(Begin, I - (Pos + 1));
      unsigned Index = 0;

      if (Argument == "@") {
        OS << NumOfMacroInstantiations;
        Pos += 2;
      } else {
        for (; Index < NParameters; ++Index)
          if (Parameters[Index].Name == Argument)
            break;

        if (Index == NParameters) {
          if (Body[Pos + 1] == '(' && Body[Pos + 2] == ')')
            Pos += 3;
          else {
            OS << '\\' << Argument;
            Pos = I;
          }
        } else {
          bool VarargParameter = HasVararg && Index == (NParameters - 1);
          for (const AsmToken &Token : A[Index])
            // For altmacro mode, you can write '%expr'.
            // The prefix '%' evaluates the expression 'expr'
            // and uses the result as a string (e.g. replace %(1+2) with the
            // string "3").
            // Here, we identify the integer token which is the result of the
            // absolute expression evaluation and replace it with its string
            // representation.
            if (AltMacroMode && Token.getString().front() == '%' &&
                Token.is(AsmToken::Integer))
              // Emit an integer value to the buffer.
              OS << Token.getIntVal();
            // Only Token that was validated as a string and begins with '<'
            // is considered altMacroString!!!
            else if (AltMacroMode && Token.getString().front() == '<' &&
                     Token.is(AsmToken::String)) {
              OS << angleBracketString(Token.getStringContents());
            }
            // We expect no quotes around the string's contents when
            // parsing for varargs.
            else if (Token.isNot(AsmToken::String) || VarargParameter)
              OS << Token.getString();
            else
              OS << Token.getStringContents();

          Pos += 1 + Argument.size();
        }
      }
    }
    // Update the scan point.
    Body = Body.substr(Pos);
  }

  return false;
}

static bool isOperator(AsmToken::TokenKind kind) {
  switch (kind) {
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

namespace {

class AsmLexerSkipSpaceRAII {
public:
  AsmLexerSkipSpaceRAII(AsmLexer &Lexer, bool SkipSpace) : Lexer(Lexer) {
    Lexer.setSkipSpace(SkipSpace);
  }

  ~AsmLexerSkipSpaceRAII() {
    Lexer.setSkipSpace(true);
  }

private:
  AsmLexer &Lexer;
};

} // end anonymous namespace

bool MasmParser::parseMacroArgument(MCAsmMacroArgument &MA, bool Vararg) {

  if (Vararg) {
    if (Lexer.isNot(AsmToken::EndOfStatement)) {
      StringRef Str = parseStringToEndOfStatement();
      MA.emplace_back(AsmToken::String, Str);
    }
    return false;
  }

  unsigned ParenLevel = 0;

  // Darwin doesn't use spaces to delmit arguments.
  AsmLexerSkipSpaceRAII ScopedSkipSpace(Lexer, IsDarwin);

  bool SpaceEaten;

  while (true) {
    SpaceEaten = false;
    if (Lexer.is(AsmToken::Eof) || Lexer.is(AsmToken::Equal))
      return TokError("unexpected token in macro instantiation");

    if (ParenLevel == 0) {

      if (Lexer.is(AsmToken::Comma))
        break;

      if (Lexer.is(AsmToken::Space)) {
        SpaceEaten = true;
        Lexer.Lex(); // Eat spaces
      }

      // Spaces can delimit parameters, but could also be part an expression.
      // If the token after a space is an operator, add the token and the next
      // one into this argument
      if (!IsDarwin) {
        if (isOperator(Lexer.getKind())) {
          MA.push_back(getTok());
          Lexer.Lex();

          // Whitespace after an operator can be ignored.
          if (Lexer.is(AsmToken::Space))
            Lexer.Lex();

          continue;
        }
      }
      if (SpaceEaten)
        break;
    }

    // handleMacroEntry relies on not advancing the lexer here
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
    Lexer.Lex();
  }

  if (ParenLevel != 0)
    return TokError("unbalanced parentheses in macro argument");
  return false;
}

// Parse the macro instantiation arguments.
bool MasmParser::parseMacroArguments(const MCAsmMacro *M,
                                     MCAsmMacroArguments &A) {
  const unsigned NParameters = M ? M->Parameters.size() : 0;
  bool NamedParametersFound = false;
  SmallVector<SMLoc, 4> FALocs;

  A.resize(NParameters);
  FALocs.resize(NParameters);

  // Parse two kinds of macro invocations:
  // - macros defined without any parameters accept an arbitrary number of them
  // - macros defined with parameters accept at most that many of them
  bool HasVararg = NParameters ? M->Parameters.back().Vararg : false;
  for (unsigned Parameter = 0; !NParameters || Parameter < NParameters;
       ++Parameter) {
    SMLoc IDLoc = Lexer.getLoc();
    MCAsmMacroParameter FA;

    if (Lexer.is(AsmToken::Identifier) && Lexer.peekTok().is(AsmToken::Equal)) {
      if (parseIdentifier(FA.Name))
        return Error(IDLoc, "invalid argument identifier for formal argument");

      if (Lexer.isNot(AsmToken::Equal))
        return TokError("expected '=' after formal parameter identifier");

      Lex();

      NamedParametersFound = true;
    }
    bool Vararg = HasVararg && Parameter == (NParameters - 1);

    if (NamedParametersFound && FA.Name.empty())
      return Error(IDLoc, "cannot mix positional and keyword arguments");

    SMLoc StrLoc = Lexer.getLoc();
    SMLoc EndLoc;
    if (AltMacroMode && Lexer.is(AsmToken::Percent)) {
      const MCExpr *AbsoluteExp;
      int64_t Value;
      /// Eat '%'
      Lex();
      if (parseExpression(AbsoluteExp, EndLoc))
        return false;
      if (!AbsoluteExp->evaluateAsAbsolute(Value,
                                           getStreamer().getAssemblerPtr()))
        return Error(StrLoc, "expected absolute expression");
      const char *StrChar = StrLoc.getPointer();
      const char *EndChar = EndLoc.getPointer();
      AsmToken newToken(AsmToken::Integer,
                        StringRef(StrChar, EndChar - StrChar), Value);
      FA.Value.push_back(newToken);
    } else if (AltMacroMode && Lexer.is(AsmToken::Less) &&
               isAngleBracketString(StrLoc, EndLoc)) {
      const char *StrChar = StrLoc.getPointer();
      const char *EndChar = EndLoc.getPointer();
      jumpToLoc(EndLoc, CurBuffer);
      /// Eat from '<' to '>'
      Lex();
      AsmToken newToken(AsmToken::String,
                        StringRef(StrChar, EndChar - StrChar));
      FA.Value.push_back(newToken);
    } else if(parseMacroArgument(FA.Value, Vararg))
      return true;

    unsigned PI = Parameter;
    if (!FA.Name.empty()) {
      unsigned FAI = 0;
      for (FAI = 0; FAI < NParameters; ++FAI)
        if (M->Parameters[FAI].Name == FA.Name)
          break;

      if (FAI >= NParameters) {
        assert(M && "expected macro to be defined");
        return Error(IDLoc, "parameter named '" + FA.Name +
                                "' does not exist for macro '" + M->Name + "'");
      }
      PI = FAI;
    }

    if (!FA.Value.empty()) {
      if (A.size() <= PI)
        A.resize(PI + 1);
      A[PI] = FA.Value;

      if (FALocs.size() <= PI)
        FALocs.resize(PI + 1);

      FALocs[PI] = Lexer.getLoc();
    }

    // At the end of the statement, fill in remaining arguments that have
    // default values. If there aren't any, then the next argument is
    // required but missing
    if (Lexer.is(AsmToken::EndOfStatement)) {
      bool Failure = false;
      for (unsigned FAI = 0; FAI < NParameters; ++FAI) {
        if (A[FAI].empty()) {
          if (M->Parameters[FAI].Required) {
            Error(FALocs[FAI].isValid() ? FALocs[FAI] : Lexer.getLoc(),
                  "missing value for required parameter "
                  "'" + M->Parameters[FAI].Name + "' in macro '" + M->Name + "'");
            Failure = true;
          }

          if (!M->Parameters[FAI].Value.empty())
            A[FAI] = M->Parameters[FAI].Value;
        }
      }
      return Failure;
    }

    if (Lexer.is(AsmToken::Comma))
      Lex();
  }

  return TokError("too many positional arguments");
}

bool MasmParser::handleMacroEntry(const MCAsmMacro *M, SMLoc NameLoc) {
  // Arbitrarily limit macro nesting depth (default matches 'as'). We can
  // eliminate this, although we should protect against infinite loops.
  unsigned MaxNestingDepth = AsmMacroMaxNestingDepth;
  if (ActiveMacros.size() == MaxNestingDepth) {
    std::ostringstream MaxNestingDepthError;
    MaxNestingDepthError << "macros cannot be nested more than "
                         << MaxNestingDepth << " levels deep."
                         << " Use -asm-macro-max-nesting-depth to increase "
                            "this limit.";
    return TokError(MaxNestingDepthError.str());
  }

  MCAsmMacroArguments A;
  if (parseMacroArguments(M, A))
    return true;

  // Macro instantiation is lexical, unfortunately. We construct a new buffer
  // to hold the macro body with substitutions.
  SmallString<256> Buf;
  StringRef Body = M->Body;
  raw_svector_ostream OS(Buf);

  if (expandMacro(OS, Body, M->Parameters, A, true, getTok().getLoc()))
    return true;

  // We include the .endmacro in the buffer as our cue to exit the macro
  // instantiation.
  OS << ".endmacro\n";

  std::unique_ptr<MemoryBuffer> Instantiation =
      MemoryBuffer::getMemBufferCopy(OS.str(), "<instantiation>");

  // Create the macro instantiation object and add to the current macro
  // instantiation stack.
  MacroInstantiation *MI = new MacroInstantiation{
      NameLoc, CurBuffer, getTok().getLoc(), TheCondStack.size()};
  ActiveMacros.push_back(MI);

  ++NumOfMacroInstantiations;

  // Jump to the macro instantiation and prime the lexer.
  CurBuffer = SrcMgr.AddNewSourceBuffer(std::move(Instantiation), SMLoc());
  Lexer.setBuffer(SrcMgr.getMemoryBuffer(CurBuffer)->getBuffer());
  Lex();

  return false;
}

void MasmParser::handleMacroExit() {
  // Jump to the EndOfStatement we should return to, and consume it.
  jumpToLoc(ActiveMacros.back()->ExitLoc, ActiveMacros.back()->ExitBuffer);
  Lex();

  // Pop the instantiation entry.
  delete ActiveMacros.back();
  ActiveMacros.pop_back();
}

/// parseIdentifier:
///   ::= identifier
///   ::= string
bool MasmParser::parseIdentifier(StringRef &Res) {
  // The assembler has relaxed rules for accepting identifiers, in particular we
  // allow things like '.globl $foo' and '.def @feat.00', which would normally
  // be separate tokens. At this level, we have already lexed so we cannot
  // (currently) handle this as a context dependent token, instead we detect
  // adjacent tokens and return the combined identifier.
  if (Lexer.is(AsmToken::Dollar) || Lexer.is(AsmToken::At)) {
    SMLoc PrefixLoc = getLexer().getLoc();

    // Consume the prefix character, and check for a following identifier.

    AsmToken Buf[1];
    Lexer.peekTokens(Buf, false);

    if (Buf[0].isNot(AsmToken::Identifier))
      return true;

    // We have a '$' or '@' followed by an identifier, make sure they are adjacent.
    if (PrefixLoc.getPointer() + 1 != Buf[0].getLoc().getPointer())
      return true;

    // eat $ or @
    Lexer.Lex(); // Lexer's Lex guarantees consecutive token.
    // Construct the joined identifier and consume the token.
    Res =
        StringRef(PrefixLoc.getPointer(), getTok().getIdentifier().size() + 1);
    Lex(); // Parser Lex to maintain invariants.
    return false;
  }

  if (Lexer.isNot(AsmToken::Identifier) && Lexer.isNot(AsmToken::String))
    return true;

  Res = getTok().getIdentifier();

  Lex(); // Consume the identifier token.

  return false;
}

/// parseDirectiveEquate:
///  ::= name "=" expression
///    | name "equ" expression    (not redefinable)
///    | name "equ" text-list
///    | name "textequ" text-list
bool MasmParser::parseDirectiveEquate(StringRef IDVal, StringRef Name,
                                      DirectiveKind DirKind) {
  Variable &Var = Variables[Name];
  if (Var.Name.empty()) {
    Var.Name = Name;
  } else if (!Var.Redefinable) {
    return TokError("invalid variable redefinition");
  }
  Var.Redefinable = (DirKind != DK_EQU);

  if (DirKind == DK_EQU || DirKind == DK_TEXTEQU) {
    // "equ" and "textequ" both allow text expressions.
    std::string Value;
    if (!parseTextItem(Value)) {
      Var.IsText = true;
      Var.TextValue = Value;

      // Accept a text-list, not just one text-item
      auto parseItem = [&]() -> bool {
        if (parseTextItem(Value))
          return true;
        Var.TextValue += Value;
        return false;
      };
      if (parseOptionalToken(AsmToken::Comma) && parseMany(parseItem))
        return addErrorSuffix(" in '" + Twine(IDVal) + "' directive");

      return false;
    }
  }
  if (DirKind == DK_TEXTEQU)
    return TokError("expected <text> in '" + Twine(IDVal) + "' directive");

  // Parse as expression assignment
  const MCExpr *Expr;
  SMLoc EndLoc, StartLoc = Lexer.getLoc();
  if (parseExpression(Expr, EndLoc))
    return addErrorSuffix(" in '" + Twine(IDVal) + "' directive");
  if (Expr->evaluateAsAbsolute(Var.NumericValue,
                               getStreamer().getAssemblerPtr()))
    return false;

  // Not an absolute expression; define as a text replacement.
  Var.IsText = true;
  Var.TextValue = StringRef(StartLoc.getPointer(),
                            EndLoc.getPointer() - StartLoc.getPointer()).str();
  return false;
}

bool MasmParser::parseEscapedString(std::string &Data) {
  if (check(getTok().isNot(AsmToken::String), "expected string"))
    return true;

  Data = "";
  StringRef Str = getTok().getStringContents();
  for (unsigned i = 0, e = Str.size(); i != e; ++i) {
    if (Str[i] != '\\') {
      Data += Str[i];
      continue;
    }

    // Recognize escaped characters. Note that this escape semantics currently
    // loosely follows Darwin 'as'.
    ++i;
    if (i == e)
      return TokError("unexpected backslash at end of string");

    // Recognize hex sequences similarly to GNU 'as'.
    if (Str[i] == 'x' || Str[i] == 'X') {
      size_t length = Str.size();
      if (i + 1 >= length || !isHexDigit(Str[i + 1]))
        return TokError("invalid hexadecimal escape sequence");

      // Consume hex characters. GNU 'as' reads all hexadecimal characters and
      // then truncates to the lower 16 bits. Seems reasonable.
      unsigned Value = 0;
      while (i + 1 < length && isHexDigit(Str[i + 1]))
        Value = Value * 16 + hexDigitValue(Str[++i]);

      Data += (unsigned char)(Value & 0xFF);
      continue;
    }

    // Recognize octal sequences.
    if ((unsigned)(Str[i] - '0') <= 7) {
      // Consume up to three octal characters.
      unsigned Value = Str[i] - '0';

      if (i + 1 != e && ((unsigned)(Str[i + 1] - '0')) <= 7) {
        ++i;
        Value = Value * 8 + (Str[i] - '0');

        if (i + 1 != e && ((unsigned)(Str[i + 1] - '0')) <= 7) {
          ++i;
          Value = Value * 8 + (Str[i] - '0');
        }
      }

      if (Value > 255)
        return TokError("invalid octal escape sequence (out of range)");

      Data += (unsigned char)Value;
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

  Lex();
  return false;
}

bool MasmParser::parseAngleBracketString(std::string &Data) {
  SMLoc EndLoc, StartLoc = getTok().getLoc();
  if (isAngleBracketString(StartLoc, EndLoc)) {
    const char *StartChar = StartLoc.getPointer() + 1;
    const char *EndChar = EndLoc.getPointer() - 1;
    jumpToLoc(EndLoc, CurBuffer);
    /// Eat from '<' to '>'
    Lex();

    Data = angleBracketString(StringRef(StartChar, EndChar - StartChar));
    return false;
  }
  return true;
}

/// textItem ::= textLiteral | textMacroID | % constExpr
bool MasmParser::parseTextItem(std::string &Data) {
  // TODO(epastor): Support textMacroID and % expansion of expressions.
  return parseAngleBracketString(Data);
}

/// parseDirectiveAscii:
///   ::= ( .ascii | .asciz | .string ) [ "string" ( , "string" )* ]
bool MasmParser::parseDirectiveAscii(StringRef IDVal, bool ZeroTerminated) {
  auto parseOp = [&]() -> bool {
    std::string Data;
    if (checkForValidSection() || parseEscapedString(Data))
      return true;
    getStreamer().emitBytes(Data);
    if (ZeroTerminated)
      getStreamer().emitBytes(StringRef("\0", 1));
    return false;
  };

  if (parseMany(parseOp))
    return addErrorSuffix(" in '" + Twine(IDVal) + "' directive");
  return false;
}

bool MasmParser::parseScalarInstList(unsigned Size,
                                     SmallVectorImpl<const MCExpr *> &Values) {
  do {
    if (getTok().is(AsmToken::String)) {
      StringRef Value = getTok().getStringContents();
      if (Size == 1) {
        // Treat each character as an initializer.
        for (const char CharVal : Value)
          Values.push_back(MCConstantExpr::create(CharVal, getContext()));
      } else {
        // Treat the string as an initial value in big-endian representation.
        if (Value.size() > Size)
          return Error(getTok().getLoc(), "out of range literal value");

        uint64_t IntValue = 0;
        for (const unsigned char CharVal : Value.bytes())
          IntValue = (IntValue << 8) | CharVal;
        Values.push_back(MCConstantExpr::create(IntValue, getContext()));
      }
      Lex();
    } else {
      const MCExpr *Value;
      if (checkForValidSection() || parseExpression(Value))
        return true;
      if (getTok().is(AsmToken::Identifier) &&
          getTok().getString().equals_lower("dup")) {
        Lex();  // eat 'dup'
        const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(Value);
        if (!MCE)
          return Error(Value->getLoc(),
                       "cannot repeat value a non-constant number of times");
        const int64_t Repetitions = MCE->getValue();
        if (Repetitions < 0)
          return Error(Value->getLoc(),
                       "cannot repeat value a negative number of times");

        SmallVector<const MCExpr *, 1> DuplicatedValues;
        if (parseToken(AsmToken::LParen,
                       "parentheses required for 'dup' contents") ||
            parseScalarInstList(Size, DuplicatedValues) ||
            parseToken(AsmToken::RParen, "unmatched parentheses"))
          return true;

        for (int i = 0; i < Repetitions; ++i)
          Values.append(DuplicatedValues.begin(), DuplicatedValues.end());
      } else {
        Values.push_back(Value);
      }
    }

    // Continue if we see a comma. (Also, allow line continuation.)
  } while (parseOptionalToken(AsmToken::Comma) &&
           (getTok().isNot(AsmToken::EndOfStatement) ||
            !parseToken(AsmToken::EndOfStatement)));

  return false;
}

/// parseDirectiveValue
///  ::= (byte | word | ... ) [ expression (, expression)* ]
bool MasmParser::parseDirectiveValue(StringRef IDVal, unsigned Size) {
  SmallVector<const MCExpr *, 1> Values;
  if (parseScalarInstList(Size, Values))
    return addErrorSuffix(" in '" + Twine(IDVal) + "' directive");

  for (const MCExpr *Value : Values) {
    // Special case constant expressions to match code generator.
    if (const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(Value)) {
      assert(Size <= 8 && "Invalid size");
      int64_t IntValue = MCE->getValue();
      if (!isUIntN(8 * Size, IntValue) && !isIntN(8 * Size, IntValue))
        return Error(MCE->getLoc(), "out of range literal value");
      getStreamer().emitIntValue(IntValue, Size);
    } else {
      const MCSymbolRefExpr *MSE = dyn_cast<MCSymbolRefExpr>(Value);
      if (MSE && MSE->getSymbol().getName() == "?") {
        // ? initializer; treat as 0.
        getStreamer().emitIntValue(0, Size);
      } else {
        getStreamer().emitValue(Value, Size, Value->getLoc());
      }
    }
  }
  return false;
}

/// parseDirectiveNamedValue
///  ::= name (byte | word | ... ) [ expression (, expression)* ]
bool MasmParser::parseDirectiveNamedValue(StringRef IDVal, unsigned Size,
                                          StringRef Name, SMLoc NameLoc) {
  MCSymbol *Sym = getContext().getOrCreateSymbol(Name);
  getStreamer().emitLabel(Sym);
  return parseDirectiveValue(IDVal, Size);
}

static bool parseHexOcta(MasmParser &Asm, uint64_t &hi, uint64_t &lo) {
  if (Asm.getTok().isNot(AsmToken::Integer) &&
      Asm.getTok().isNot(AsmToken::BigNum))
    return Asm.TokError("unknown token in expression");
  SMLoc ExprLoc = Asm.getTok().getLoc();
  APInt IntValue = Asm.getTok().getAPIntVal();
  Asm.Lex();
  if (!IntValue.isIntN(128))
    return Asm.Error(ExprLoc, "out of range literal value");
  if (!IntValue.isIntN(64)) {
    hi = IntValue.getHiBits(IntValue.getBitWidth() - 64).getZExtValue();
    lo = IntValue.getLoBits(64).getZExtValue();
  } else {
    hi = 0;
    lo = IntValue.getZExtValue();
  }
  return false;
}

bool MasmParser::parseRealValue(const fltSemantics &Semantics, APInt &Res) {
  // We don't truly support arithmetic on floating point expressions, so we
  // have to manually parse unary prefixes.
  bool IsNeg = false;
  if (getLexer().is(AsmToken::Minus)) {
    Lexer.Lex();
    IsNeg = true;
  } else if (getLexer().is(AsmToken::Plus))
    Lexer.Lex();

  if (Lexer.is(AsmToken::Error))
    return TokError(Lexer.getErr());
  if (Lexer.isNot(AsmToken::Integer) && Lexer.isNot(AsmToken::Real) &&
      Lexer.isNot(AsmToken::Identifier))
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
  } else if (errorToBool(
                 Value.convertFromString(IDVal, APFloat::rmNearestTiesToEven)
                     .takeError()))
    return TokError("invalid floating point literal");
  if (IsNeg)
    Value.changeSign();

  // Consume the numeric token.
  Lex();

  Res = Value.bitcastToAPInt();

  return false;
}

bool MasmParser::parseRealInstList(const fltSemantics &Semantics,
                                   SmallVectorImpl<APInt> &ValuesAsInt) {
  do {
    const AsmToken NextTok = Lexer.peekTok();
    if (NextTok.is(AsmToken::Identifier) &&
        NextTok.getString().equals_lower("dup")) {
      const MCExpr *Value;
      if (parseExpression(Value) || parseToken(AsmToken::Identifier))
        return true;
      const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(Value);
      if (!MCE)
        return Error(Value->getLoc(),
                     "cannot repeat value a non-constant number of times");
      const int64_t Repetitions = MCE->getValue();
      if (Repetitions < 0)
        return Error(Value->getLoc(),
                     "cannot repeat value a negative number of times");

      SmallVector<APInt, 1> DuplicatedValues;
      if (parseToken(AsmToken::LParen,
                     "parentheses required for 'dup' contents") ||
          parseRealInstList(Semantics, DuplicatedValues) ||
          parseToken(AsmToken::RParen, "unmatched parentheses"))
        return true;

      for (int i = 0; i < Repetitions; ++i)
        ValuesAsInt.append(DuplicatedValues.begin(), DuplicatedValues.end());
    } else {
      APInt AsInt;
      if (parseRealValue(Semantics, AsInt))
        return true;
      ValuesAsInt.push_back(AsInt);
    }
    // Continue if we see a comma. (Also, allow line continuation.)
  } while (parseOptionalToken(AsmToken::Comma) &&
           (getTok().isNot(AsmToken::EndOfStatement) ||
            !parseToken(AsmToken::EndOfStatement)));

  return false;
}

/// parseDirectiveRealValue
///  ::= (real4 | real8) [ expression (, expression)* ]
bool MasmParser::parseDirectiveRealValue(StringRef IDVal,
                                         const fltSemantics &Semantics) {
  if (checkForValidSection())
    return true;

  SmallVector<APInt, 1> ValuesAsInt;
  if (parseRealInstList(Semantics, ValuesAsInt))
    return addErrorSuffix(" in '" + Twine(IDVal) + "' directive");

  for (const APInt &AsInt : ValuesAsInt) {
    getStreamer().emitIntValue(AsInt.getLimitedValue(),
                               AsInt.getBitWidth() / 8);
  }
  return false;
}

/// parseDirectiveNamedRealValue
///  ::= name (real4 | real8) [ expression (, expression)* ]
bool MasmParser::parseDirectiveNamedRealValue(StringRef IDVal,
                                              const fltSemantics &Semantics,
                                              StringRef Name, SMLoc NameLoc) {
  MCSymbol *Sym = getContext().getOrCreateSymbol(Name);
  getStreamer().emitLabel(Sym);
  return parseDirectiveRealValue(IDVal, Semantics);
}

/// parseDirectiveOrg
///  ::= .org expression [ , expression ]
bool MasmParser::parseDirectiveOrg() {
  const MCExpr *Offset;
  SMLoc OffsetLoc = Lexer.getLoc();
  if (checkForValidSection() || parseExpression(Offset))
    return true;

  // Parse optional fill expression.
  int64_t FillExpr = 0;
  if (parseOptionalToken(AsmToken::Comma))
    if (parseAbsoluteExpression(FillExpr))
      return addErrorSuffix(" in '.org' directive");
  if (parseToken(AsmToken::EndOfStatement))
    return addErrorSuffix(" in '.org' directive");

  getStreamer().emitValueToOffset(Offset, FillExpr, OffsetLoc);
  return false;
}

/// parseDirectiveAlign
///  ::= align expression
bool MasmParser::parseDirectiveAlign() {
  SMLoc AlignmentLoc = getLexer().getLoc();
  int64_t Alignment;

  if (checkForValidSection())
    return addErrorSuffix(" in align directive");
  // Ignore empty 'align' directives
  if (getTok().is(AsmToken::EndOfStatement)) {
    Warning(AlignmentLoc, "align directive with no operand is ignored");
    return parseToken(AsmToken::EndOfStatement);
  }
  if (parseAbsoluteExpression(Alignment) ||
      parseToken(AsmToken::EndOfStatement))
    return addErrorSuffix(" in align directive");

  // Always emit an alignment here even if we thrown an error.
  bool ReturnVal = false;

  // Reject alignments that aren't either a power of two or zero,
  // for gas compatibility. Alignment of zero is silently rounded
  // up to one.
  if (Alignment == 0)
    Alignment = 1;
  if (!isPowerOf2_64(Alignment))
    ReturnVal |= Error(AlignmentLoc, "alignment must be a power of 2");

  // Check whether we should use optimal code alignment for this align
  // directive.
  const MCSection *Section = getStreamer().getCurrentSectionOnly();
  assert(Section && "must have section to emit alignment");
  if (Section->UseCodeAlign()) {
    getStreamer().emitCodeAlignment(Alignment, /*MaxBytesToEmit=*/0);
  } else {
    // FIXME: Target specific behavior about how the "extra" bytes are filled.
    getStreamer().emitValueToAlignment(Alignment, /*Value=*/0, /*ValueSize=*/1,
                                       /*MaxBytesToEmit=*/0);
  }

  return ReturnVal;
}

/// parseDirectiveFile
/// ::= .file filename
/// ::= .file number [directory] filename [md5 checksum] [source source-text]
bool MasmParser::parseDirectiveFile(SMLoc DirectiveLoc) {
  // FIXME: I'm not sure what this is.
  int64_t FileNumber = -1;
  if (getLexer().is(AsmToken::Integer)) {
    FileNumber = getTok().getIntVal();
    Lex();

    if (FileNumber < 0)
      return TokError("negative file number");
  }

  std::string Path;

  // Usually the directory and filename together, otherwise just the directory.
  // Allow the strings to have escaped octal character sequence.
  if (check(getTok().isNot(AsmToken::String),
            "unexpected token in '.file' directive") ||
      parseEscapedString(Path))
    return true;

  StringRef Directory;
  StringRef Filename;
  std::string FilenameData;
  if (getLexer().is(AsmToken::String)) {
    if (check(FileNumber == -1,
              "explicit path specified, but no file number") ||
        parseEscapedString(FilenameData))
      return true;
    Filename = FilenameData;
    Directory = Path;
  } else {
    Filename = Path;
  }

  uint64_t MD5Hi, MD5Lo;
  bool HasMD5 = false;

  Optional<StringRef> Source;
  bool HasSource = false;
  std::string SourceString;

  while (!parseOptionalToken(AsmToken::EndOfStatement)) {
    StringRef Keyword;
    if (check(getTok().isNot(AsmToken::Identifier),
              "unexpected token in '.file' directive") ||
        parseIdentifier(Keyword))
      return true;
    if (Keyword == "md5") {
      HasMD5 = true;
      if (check(FileNumber == -1,
                "MD5 checksum specified, but no file number") ||
          parseHexOcta(*this, MD5Hi, MD5Lo))
        return true;
    } else if (Keyword == "source") {
      HasSource = true;
      if (check(FileNumber == -1,
                "source specified, but no file number") ||
          check(getTok().isNot(AsmToken::String),
                "unexpected token in '.file' directive") ||
          parseEscapedString(SourceString))
        return true;
    } else {
      return TokError("unexpected token in '.file' directive");
    }
  }

  if (FileNumber == -1) {
    // Ignore the directive if there is no number and the target doesn't support
    // numberless .file directives. This allows some portability of assembler
    // between different object file formats.
    if (getContext().getAsmInfo()->hasSingleParameterDotFile())
      getStreamer().emitFileDirective(Filename);
  } else {
    // In case there is a -g option as well as debug info from directive .file,
    // we turn off the -g option, directly use the existing debug info instead.
    // Throw away any implicit file table for the assembler source.
    if (Ctx.getGenDwarfForAssembly()) {
      Ctx.getMCDwarfLineTable(0).resetFileTable();
      Ctx.setGenDwarfForAssembly(false);
    }

    Optional<MD5::MD5Result> CKMem;
    if (HasMD5) {
      MD5::MD5Result Sum;
      for (unsigned i = 0; i != 8; ++i) {
        Sum.Bytes[i] = uint8_t(MD5Hi >> ((7 - i) * 8));
        Sum.Bytes[i + 8] = uint8_t(MD5Lo >> ((7 - i) * 8));
      }
      CKMem = Sum;
    }
    if (HasSource) {
      char *SourceBuf = static_cast<char *>(Ctx.allocate(SourceString.size()));
      memcpy(SourceBuf, SourceString.data(), SourceString.size());
      Source = StringRef(SourceBuf, SourceString.size());
    }
    if (FileNumber == 0) {
      if (Ctx.getDwarfVersion() < 5)
        return Warning(DirectiveLoc, "file 0 not supported prior to DWARF-5");
      getStreamer().emitDwarfFile0Directive(Directory, Filename, CKMem, Source);
    } else {
      Expected<unsigned> FileNumOrErr = getStreamer().tryEmitDwarfFileDirective(
          FileNumber, Directory, Filename, CKMem, Source);
      if (!FileNumOrErr)
        return Error(DirectiveLoc, toString(FileNumOrErr.takeError()));
    }
    // Alert the user if there are some .file directives with MD5 and some not.
    // But only do that once.
    if (!ReportedInconsistentMD5 && !Ctx.isDwarfMD5UsageConsistent(0)) {
      ReportedInconsistentMD5 = true;
      return Warning(DirectiveLoc, "inconsistent use of MD5 checksums");
    }
  }

  return false;
}

/// parseDirectiveLine
/// ::= .line [number]
bool MasmParser::parseDirectiveLine() {
  int64_t LineNumber;
  if (getLexer().is(AsmToken::Integer)) {
    if (parseIntToken(LineNumber, "unexpected token in '.line' directive"))
      return true;
    (void)LineNumber;
    // FIXME: Do something with the .line.
  }
  if (parseToken(AsmToken::EndOfStatement,
                 "unexpected token in '.line' directive"))
    return true;

  return false;
}

/// parseDirectiveLoc
/// ::= .loc FileNumber [LineNumber] [ColumnPos] [basic_block] [prologue_end]
///                                [epilogue_begin] [is_stmt VALUE] [isa VALUE]
/// The first number is a file number, must have been previously assigned with
/// a .file directive, the second number is the line number and optionally the
/// third number is a column position (zero if not specified).  The remaining
/// optional items are .loc sub-directives.
bool MasmParser::parseDirectiveLoc() {
  int64_t FileNumber = 0, LineNumber = 0;
  SMLoc Loc = getTok().getLoc();
  if (parseIntToken(FileNumber, "unexpected token in '.loc' directive") ||
      check(FileNumber < 1 && Ctx.getDwarfVersion() < 5, Loc,
            "file number less than one in '.loc' directive") ||
      check(!getContext().isValidDwarfFileNumber(FileNumber), Loc,
            "unassigned file number in '.loc' directive"))
    return true;

  // optional
  if (getLexer().is(AsmToken::Integer)) {
    LineNumber = getTok().getIntVal();
    if (LineNumber < 0)
      return TokError("line number less than zero in '.loc' directive");
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

  auto parseLocOp = [&]() -> bool {
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
    return false;
  };

  if (parseMany(parseLocOp, false /*hasComma*/))
    return true;

  getStreamer().emitDwarfLocDirective(FileNumber, LineNumber, ColumnPos, Flags,
                                      Isa, Discriminator, StringRef());

  return false;
}

/// parseDirectiveStabs
/// ::= .stabs string, number, number, number
bool MasmParser::parseDirectiveStabs() {
  return TokError("unsupported directive '.stabs'");
}

/// parseDirectiveCVFile
/// ::= .cv_file number filename [checksum] [checksumkind]
bool MasmParser::parseDirectiveCVFile() {
  SMLoc FileNumberLoc = getTok().getLoc();
  int64_t FileNumber;
  std::string Filename;
  std::string Checksum;
  int64_t ChecksumKind = 0;

  if (parseIntToken(FileNumber,
                    "expected file number in '.cv_file' directive") ||
      check(FileNumber < 1, FileNumberLoc, "file number less than one") ||
      check(getTok().isNot(AsmToken::String),
            "unexpected token in '.cv_file' directive") ||
      parseEscapedString(Filename))
    return true;
  if (!parseOptionalToken(AsmToken::EndOfStatement)) {
    if (check(getTok().isNot(AsmToken::String),
              "unexpected token in '.cv_file' directive") ||
        parseEscapedString(Checksum) ||
        parseIntToken(ChecksumKind,
                      "expected checksum kind in '.cv_file' directive") ||
        parseToken(AsmToken::EndOfStatement,
                   "unexpected token in '.cv_file' directive"))
      return true;
  }

  Checksum = fromHex(Checksum);
  void *CKMem = Ctx.allocate(Checksum.size(), 1);
  memcpy(CKMem, Checksum.data(), Checksum.size());
  ArrayRef<uint8_t> ChecksumAsBytes(reinterpret_cast<const uint8_t *>(CKMem),
                                    Checksum.size());

  if (!getStreamer().EmitCVFileDirective(FileNumber, Filename, ChecksumAsBytes,
                                         static_cast<uint8_t>(ChecksumKind)))
    return Error(FileNumberLoc, "file number already allocated");

  return false;
}

bool MasmParser::parseCVFunctionId(int64_t &FunctionId,
                                   StringRef DirectiveName) {
  SMLoc Loc;
  return parseTokenLoc(Loc) ||
         parseIntToken(FunctionId, "expected function id in '" + DirectiveName +
                                       "' directive") ||
         check(FunctionId < 0 || FunctionId >= UINT_MAX, Loc,
               "expected function id within range [0, UINT_MAX)");
}

bool MasmParser::parseCVFileId(int64_t &FileNumber, StringRef DirectiveName) {
  SMLoc Loc;
  return parseTokenLoc(Loc) ||
         parseIntToken(FileNumber, "expected integer in '" + DirectiveName +
                                       "' directive") ||
         check(FileNumber < 1, Loc, "file number less than one in '" +
                                        DirectiveName + "' directive") ||
         check(!getCVContext().isValidFileNumber(FileNumber), Loc,
               "unassigned file number in '" + DirectiveName + "' directive");
}

/// parseDirectiveCVFuncId
/// ::= .cv_func_id FunctionId
///
/// Introduces a function ID that can be used with .cv_loc.
bool MasmParser::parseDirectiveCVFuncId() {
  SMLoc FunctionIdLoc = getTok().getLoc();
  int64_t FunctionId;

  if (parseCVFunctionId(FunctionId, ".cv_func_id") ||
      parseToken(AsmToken::EndOfStatement,
                 "unexpected token in '.cv_func_id' directive"))
    return true;

  if (!getStreamer().EmitCVFuncIdDirective(FunctionId))
    return Error(FunctionIdLoc, "function id already allocated");

  return false;
}

/// parseDirectiveCVInlineSiteId
/// ::= .cv_inline_site_id FunctionId
///         "within" IAFunc
///         "inlined_at" IAFile IALine [IACol]
///
/// Introduces a function ID that can be used with .cv_loc. Includes "inlined
/// at" source location information for use in the line table of the caller,
/// whether the caller is a real function or another inlined call site.
bool MasmParser::parseDirectiveCVInlineSiteId() {
  SMLoc FunctionIdLoc = getTok().getLoc();
  int64_t FunctionId;
  int64_t IAFunc;
  int64_t IAFile;
  int64_t IALine;
  int64_t IACol = 0;

  // FunctionId
  if (parseCVFunctionId(FunctionId, ".cv_inline_site_id"))
    return true;

  // "within"
  if (check((getLexer().isNot(AsmToken::Identifier) ||
             getTok().getIdentifier() != "within"),
            "expected 'within' identifier in '.cv_inline_site_id' directive"))
    return true;
  Lex();

  // IAFunc
  if (parseCVFunctionId(IAFunc, ".cv_inline_site_id"))
    return true;

  // "inlined_at"
  if (check((getLexer().isNot(AsmToken::Identifier) ||
             getTok().getIdentifier() != "inlined_at"),
            "expected 'inlined_at' identifier in '.cv_inline_site_id' "
            "directive") )
    return true;
  Lex();

  // IAFile IALine
  if (parseCVFileId(IAFile, ".cv_inline_site_id") ||
      parseIntToken(IALine, "expected line number after 'inlined_at'"))
    return true;

  // [IACol]
  if (getLexer().is(AsmToken::Integer)) {
    IACol = getTok().getIntVal();
    Lex();
  }

  if (parseToken(AsmToken::EndOfStatement,
                 "unexpected token in '.cv_inline_site_id' directive"))
    return true;

  if (!getStreamer().EmitCVInlineSiteIdDirective(FunctionId, IAFunc, IAFile,
                                                 IALine, IACol, FunctionIdLoc))
    return Error(FunctionIdLoc, "function id already allocated");

  return false;
}

/// parseDirectiveCVLoc
/// ::= .cv_loc FunctionId FileNumber [LineNumber] [ColumnPos] [prologue_end]
///                                [is_stmt VALUE]
/// The first number is a file number, must have been previously assigned with
/// a .file directive, the second number is the line number and optionally the
/// third number is a column position (zero if not specified).  The remaining
/// optional items are .loc sub-directives.
bool MasmParser::parseDirectiveCVLoc() {
  SMLoc DirectiveLoc = getTok().getLoc();
  int64_t FunctionId, FileNumber;
  if (parseCVFunctionId(FunctionId, ".cv_loc") ||
      parseCVFileId(FileNumber, ".cv_loc"))
    return true;

  int64_t LineNumber = 0;
  if (getLexer().is(AsmToken::Integer)) {
    LineNumber = getTok().getIntVal();
    if (LineNumber < 0)
      return TokError("line number less than zero in '.cv_loc' directive");
    Lex();
  }

  int64_t ColumnPos = 0;
  if (getLexer().is(AsmToken::Integer)) {
    ColumnPos = getTok().getIntVal();
    if (ColumnPos < 0)
      return TokError("column position less than zero in '.cv_loc' directive");
    Lex();
  }

  bool PrologueEnd = false;
  uint64_t IsStmt = 0;

  auto parseOp = [&]() -> bool {
    StringRef Name;
    SMLoc Loc = getTok().getLoc();
    if (parseIdentifier(Name))
      return TokError("unexpected token in '.cv_loc' directive");
    if (Name == "prologue_end")
      PrologueEnd = true;
    else if (Name == "is_stmt") {
      Loc = getTok().getLoc();
      const MCExpr *Value;
      if (parseExpression(Value))
        return true;
      // The expression must be the constant 0 or 1.
      IsStmt = ~0ULL;
      if (const auto *MCE = dyn_cast<MCConstantExpr>(Value))
        IsStmt = MCE->getValue();

      if (IsStmt > 1)
        return Error(Loc, "is_stmt value not 0 or 1");
    } else {
      return Error(Loc, "unknown sub-directive in '.cv_loc' directive");
    }
    return false;
  };

  if (parseMany(parseOp, false /*hasComma*/))
    return true;

  getStreamer().EmitCVLocDirective(FunctionId, FileNumber, LineNumber,
                                   ColumnPos, PrologueEnd, IsStmt, StringRef(),
                                   DirectiveLoc);
  return false;
}

/// parseDirectiveCVLinetable
/// ::= .cv_linetable FunctionId, FnStart, FnEnd
bool MasmParser::parseDirectiveCVLinetable() {
  int64_t FunctionId;
  StringRef FnStartName, FnEndName;
  SMLoc Loc = getTok().getLoc();
  if (parseCVFunctionId(FunctionId, ".cv_linetable") ||
      parseToken(AsmToken::Comma,
                 "unexpected token in '.cv_linetable' directive") ||
      parseTokenLoc(Loc) || check(parseIdentifier(FnStartName), Loc,
                                  "expected identifier in directive") ||
      parseToken(AsmToken::Comma,
                 "unexpected token in '.cv_linetable' directive") ||
      parseTokenLoc(Loc) || check(parseIdentifier(FnEndName), Loc,
                                  "expected identifier in directive"))
    return true;

  MCSymbol *FnStartSym = getContext().getOrCreateSymbol(FnStartName);
  MCSymbol *FnEndSym = getContext().getOrCreateSymbol(FnEndName);

  getStreamer().EmitCVLinetableDirective(FunctionId, FnStartSym, FnEndSym);
  return false;
}

/// parseDirectiveCVInlineLinetable
/// ::= .cv_inline_linetable PrimaryFunctionId FileId LineNum FnStart FnEnd
bool MasmParser::parseDirectiveCVInlineLinetable() {
  int64_t PrimaryFunctionId, SourceFileId, SourceLineNum;
  StringRef FnStartName, FnEndName;
  SMLoc Loc = getTok().getLoc();
  if (parseCVFunctionId(PrimaryFunctionId, ".cv_inline_linetable") ||
      parseTokenLoc(Loc) ||
      parseIntToken(
          SourceFileId,
          "expected SourceField in '.cv_inline_linetable' directive") ||
      check(SourceFileId <= 0, Loc,
            "File id less than zero in '.cv_inline_linetable' directive") ||
      parseTokenLoc(Loc) ||
      parseIntToken(
          SourceLineNum,
          "expected SourceLineNum in '.cv_inline_linetable' directive") ||
      check(SourceLineNum < 0, Loc,
            "Line number less than zero in '.cv_inline_linetable' directive") ||
      parseTokenLoc(Loc) || check(parseIdentifier(FnStartName), Loc,
                                  "expected identifier in directive") ||
      parseTokenLoc(Loc) || check(parseIdentifier(FnEndName), Loc,
                                  "expected identifier in directive"))
    return true;

  if (parseToken(AsmToken::EndOfStatement, "Expected End of Statement"))
    return true;

  MCSymbol *FnStartSym = getContext().getOrCreateSymbol(FnStartName);
  MCSymbol *FnEndSym = getContext().getOrCreateSymbol(FnEndName);
  getStreamer().EmitCVInlineLinetableDirective(PrimaryFunctionId, SourceFileId,
                                               SourceLineNum, FnStartSym,
                                               FnEndSym);
  return false;
}

void MasmParser::initializeCVDefRangeTypeMap() {
  CVDefRangeTypeMap["reg"] = CVDR_DEFRANGE_REGISTER;
  CVDefRangeTypeMap["frame_ptr_rel"] = CVDR_DEFRANGE_FRAMEPOINTER_REL;
  CVDefRangeTypeMap["subfield_reg"] = CVDR_DEFRANGE_SUBFIELD_REGISTER;
  CVDefRangeTypeMap["reg_rel"] = CVDR_DEFRANGE_REGISTER_REL;
}

/// parseDirectiveCVDefRange
/// ::= .cv_def_range RangeStart RangeEnd (GapStart GapEnd)*, bytes*
bool MasmParser::parseDirectiveCVDefRange() {
  SMLoc Loc;
  std::vector<std::pair<const MCSymbol *, const MCSymbol *>> Ranges;
  while (getLexer().is(AsmToken::Identifier)) {
    Loc = getLexer().getLoc();
    StringRef GapStartName;
    if (parseIdentifier(GapStartName))
      return Error(Loc, "expected identifier in directive");
    MCSymbol *GapStartSym = getContext().getOrCreateSymbol(GapStartName);

    Loc = getLexer().getLoc();
    StringRef GapEndName;
    if (parseIdentifier(GapEndName))
      return Error(Loc, "expected identifier in directive");
    MCSymbol *GapEndSym = getContext().getOrCreateSymbol(GapEndName);

    Ranges.push_back({GapStartSym, GapEndSym});
  }

  StringRef CVDefRangeTypeStr;
  if (parseToken(
          AsmToken::Comma,
          "expected comma before def_range type in .cv_def_range directive") ||
      parseIdentifier(CVDefRangeTypeStr))
    return Error(Loc, "expected def_range type in directive");

  StringMap<CVDefRangeType>::const_iterator CVTypeIt =
      CVDefRangeTypeMap.find(CVDefRangeTypeStr);
  CVDefRangeType CVDRType = (CVTypeIt == CVDefRangeTypeMap.end())
                                ? CVDR_DEFRANGE
                                : CVTypeIt->getValue();
  switch (CVDRType) {
  case CVDR_DEFRANGE_REGISTER: {
    int64_t DRRegister;
    if (parseToken(AsmToken::Comma, "expected comma before register number in "
                                    ".cv_def_range directive") ||
        parseAbsoluteExpression(DRRegister))
      return Error(Loc, "expected register number");

    codeview::DefRangeRegisterHeader DRHdr;
    DRHdr.Register = DRRegister;
    DRHdr.MayHaveNoName = 0;
    getStreamer().EmitCVDefRangeDirective(Ranges, DRHdr);
    break;
  }
  case CVDR_DEFRANGE_FRAMEPOINTER_REL: {
    int64_t DROffset;
    if (parseToken(AsmToken::Comma,
                   "expected comma before offset in .cv_def_range directive") ||
        parseAbsoluteExpression(DROffset))
      return Error(Loc, "expected offset value");

    codeview::DefRangeFramePointerRelHeader DRHdr;
    DRHdr.Offset = DROffset;
    getStreamer().EmitCVDefRangeDirective(Ranges, DRHdr);
    break;
  }
  case CVDR_DEFRANGE_SUBFIELD_REGISTER: {
    int64_t DRRegister;
    int64_t DROffsetInParent;
    if (parseToken(AsmToken::Comma, "expected comma before register number in "
                                    ".cv_def_range directive") ||
        parseAbsoluteExpression(DRRegister))
      return Error(Loc, "expected register number");
    if (parseToken(AsmToken::Comma,
                   "expected comma before offset in .cv_def_range directive") ||
        parseAbsoluteExpression(DROffsetInParent))
      return Error(Loc, "expected offset value");

    codeview::DefRangeSubfieldRegisterHeader DRHdr;
    DRHdr.Register = DRRegister;
    DRHdr.MayHaveNoName = 0;
    DRHdr.OffsetInParent = DROffsetInParent;
    getStreamer().EmitCVDefRangeDirective(Ranges, DRHdr);
    break;
  }
  case CVDR_DEFRANGE_REGISTER_REL: {
    int64_t DRRegister;
    int64_t DRFlags;
    int64_t DRBasePointerOffset;
    if (parseToken(AsmToken::Comma, "expected comma before register number in "
                                    ".cv_def_range directive") ||
        parseAbsoluteExpression(DRRegister))
      return Error(Loc, "expected register value");
    if (parseToken(
            AsmToken::Comma,
            "expected comma before flag value in .cv_def_range directive") ||
        parseAbsoluteExpression(DRFlags))
      return Error(Loc, "expected flag value");
    if (parseToken(AsmToken::Comma, "expected comma before base pointer offset "
                                    "in .cv_def_range directive") ||
        parseAbsoluteExpression(DRBasePointerOffset))
      return Error(Loc, "expected base pointer offset value");

    codeview::DefRangeRegisterRelHeader DRHdr;
    DRHdr.Register = DRRegister;
    DRHdr.Flags = DRFlags;
    DRHdr.BasePointerOffset = DRBasePointerOffset;
    getStreamer().EmitCVDefRangeDirective(Ranges, DRHdr);
    break;
  }
  default:
    return Error(Loc, "unexpected def_range type in .cv_def_range directive");
  }
  return true;
}

/// parseDirectiveCVString
/// ::= .cv_stringtable "string"
bool MasmParser::parseDirectiveCVString() {
  std::string Data;
  if (checkForValidSection() || parseEscapedString(Data))
    return addErrorSuffix(" in '.cv_string' directive");

  // Put the string in the table and emit the offset.
  std::pair<StringRef, unsigned> Insertion =
      getCVContext().addToStringTable(Data);
  getStreamer().emitIntValue(Insertion.second, 4);
  return false;
}

/// parseDirectiveCVStringTable
/// ::= .cv_stringtable
bool MasmParser::parseDirectiveCVStringTable() {
  getStreamer().EmitCVStringTableDirective();
  return false;
}

/// parseDirectiveCVFileChecksums
/// ::= .cv_filechecksums
bool MasmParser::parseDirectiveCVFileChecksums() {
  getStreamer().EmitCVFileChecksumsDirective();
  return false;
}

/// parseDirectiveCVFileChecksumOffset
/// ::= .cv_filechecksumoffset fileno
bool MasmParser::parseDirectiveCVFileChecksumOffset() {
  int64_t FileNo;
  if (parseIntToken(FileNo, "expected identifier in directive"))
    return true;
  if (parseToken(AsmToken::EndOfStatement, "Expected End of Statement"))
    return true;
  getStreamer().EmitCVFileChecksumOffsetDirective(FileNo);
  return false;
}

/// parseDirectiveCVFPOData
/// ::= .cv_fpo_data procsym
bool MasmParser::parseDirectiveCVFPOData() {
  SMLoc DirLoc = getLexer().getLoc();
  StringRef ProcName;
  if (parseIdentifier(ProcName))
    return TokError("expected symbol name");
  if (parseEOL("unexpected tokens"))
    return addErrorSuffix(" in '.cv_fpo_data' directive");
  MCSymbol *ProcSym = getContext().getOrCreateSymbol(ProcName);
  getStreamer().EmitCVFPOData(ProcSym, DirLoc);
  return false;
}

/// parseDirectiveCFISections
/// ::= .cfi_sections section [, section]
bool MasmParser::parseDirectiveCFISections() {
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

  getStreamer().emitCFISections(EH, Debug);
  return false;
}

/// parseDirectiveCFIStartProc
/// ::= .cfi_startproc [simple]
bool MasmParser::parseDirectiveCFIStartProc() {
  StringRef Simple;
  if (!parseOptionalToken(AsmToken::EndOfStatement)) {
    if (check(parseIdentifier(Simple) || Simple != "simple",
              "unexpected token") ||
        parseToken(AsmToken::EndOfStatement))
      return addErrorSuffix(" in '.cfi_startproc' directive");
  }

  // TODO(kristina): Deal with a corner case of incorrect diagnostic context
  // being produced if this directive is emitted as part of preprocessor macro
  // expansion which can *ONLY* happen if Clang's cc1as is the API consumer.
  // Tools like llvm-mc on the other hand are not affected by it, and report
  // correct context information.
  getStreamer().emitCFIStartProc(!Simple.empty(), Lexer.getLoc());
  return false;
}

/// parseDirectiveCFIEndProc
/// ::= .cfi_endproc
bool MasmParser::parseDirectiveCFIEndProc() {
  getStreamer().emitCFIEndProc();
  return false;
}

/// parse register name or number.
bool MasmParser::parseRegisterOrRegisterNumber(int64_t &Register,
                                               SMLoc DirectiveLoc) {
  unsigned RegNo;

  if (getLexer().isNot(AsmToken::Integer)) {
    if (getTargetParser().ParseRegister(RegNo, DirectiveLoc, DirectiveLoc))
      return true;
    Register = getContext().getRegisterInfo()->getDwarfRegNum(RegNo, true);
  } else
    return parseAbsoluteExpression(Register);

  return false;
}

/// parseDirectiveCFIDefCfa
/// ::= .cfi_def_cfa register,  offset
bool MasmParser::parseDirectiveCFIDefCfa(SMLoc DirectiveLoc) {
  int64_t Register = 0, Offset = 0;
  if (parseRegisterOrRegisterNumber(Register, DirectiveLoc) ||
      parseToken(AsmToken::Comma, "unexpected token in directive") ||
      parseAbsoluteExpression(Offset))
    return true;

  getStreamer().emitCFIDefCfa(Register, Offset);
  return false;
}

/// parseDirectiveCFIDefCfaOffset
/// ::= .cfi_def_cfa_offset offset
bool MasmParser::parseDirectiveCFIDefCfaOffset() {
  int64_t Offset = 0;
  if (parseAbsoluteExpression(Offset))
    return true;

  getStreamer().emitCFIDefCfaOffset(Offset);
  return false;
}

/// parseDirectiveCFIRegister
/// ::= .cfi_register register, register
bool MasmParser::parseDirectiveCFIRegister(SMLoc DirectiveLoc) {
  int64_t Register1 = 0, Register2 = 0;
  if (parseRegisterOrRegisterNumber(Register1, DirectiveLoc) ||
      parseToken(AsmToken::Comma, "unexpected token in directive") ||
      parseRegisterOrRegisterNumber(Register2, DirectiveLoc))
    return true;

  getStreamer().emitCFIRegister(Register1, Register2);
  return false;
}

/// parseDirectiveCFIWindowSave
/// ::= .cfi_window_save
bool MasmParser::parseDirectiveCFIWindowSave() {
  getStreamer().emitCFIWindowSave();
  return false;
}

/// parseDirectiveCFIAdjustCfaOffset
/// ::= .cfi_adjust_cfa_offset adjustment
bool MasmParser::parseDirectiveCFIAdjustCfaOffset() {
  int64_t Adjustment = 0;
  if (parseAbsoluteExpression(Adjustment))
    return true;

  getStreamer().emitCFIAdjustCfaOffset(Adjustment);
  return false;
}

/// parseDirectiveCFIDefCfaRegister
/// ::= .cfi_def_cfa_register register
bool MasmParser::parseDirectiveCFIDefCfaRegister(SMLoc DirectiveLoc) {
  int64_t Register = 0;
  if (parseRegisterOrRegisterNumber(Register, DirectiveLoc))
    return true;

  getStreamer().emitCFIDefCfaRegister(Register);
  return false;
}

/// parseDirectiveCFIOffset
/// ::= .cfi_offset register, offset
bool MasmParser::parseDirectiveCFIOffset(SMLoc DirectiveLoc) {
  int64_t Register = 0;
  int64_t Offset = 0;

  if (parseRegisterOrRegisterNumber(Register, DirectiveLoc) ||
      parseToken(AsmToken::Comma, "unexpected token in directive") ||
      parseAbsoluteExpression(Offset))
    return true;

  getStreamer().emitCFIOffset(Register, Offset);
  return false;
}

/// parseDirectiveCFIRelOffset
/// ::= .cfi_rel_offset register, offset
bool MasmParser::parseDirectiveCFIRelOffset(SMLoc DirectiveLoc) {
  int64_t Register = 0, Offset = 0;

  if (parseRegisterOrRegisterNumber(Register, DirectiveLoc) ||
      parseToken(AsmToken::Comma, "unexpected token in directive") ||
      parseAbsoluteExpression(Offset))
    return true;

  getStreamer().emitCFIRelOffset(Register, Offset);
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

/// parseDirectiveCFIPersonalityOrLsda
/// IsPersonality true for cfi_personality, false for cfi_lsda
/// ::= .cfi_personality encoding, [symbol_name]
/// ::= .cfi_lsda encoding, [symbol_name]
bool MasmParser::parseDirectiveCFIPersonalityOrLsda(bool IsPersonality) {
  int64_t Encoding = 0;
  if (parseAbsoluteExpression(Encoding))
    return true;
  if (Encoding == dwarf::DW_EH_PE_omit)
    return false;

  StringRef Name;
  if (check(!isValidEncoding(Encoding), "unsupported encoding.") ||
      parseToken(AsmToken::Comma, "unexpected token in directive") ||
      check(parseIdentifier(Name), "expected identifier in directive"))
    return true;

  MCSymbol *Sym = getContext().getOrCreateSymbol(Name);

  if (IsPersonality)
    getStreamer().emitCFIPersonality(Sym, Encoding);
  else
    getStreamer().emitCFILsda(Sym, Encoding);
  return false;
}

/// parseDirectiveCFIRememberState
/// ::= .cfi_remember_state
bool MasmParser::parseDirectiveCFIRememberState() {
  getStreamer().emitCFIRememberState();
  return false;
}

/// parseDirectiveCFIRestoreState
/// ::= .cfi_remember_state
bool MasmParser::parseDirectiveCFIRestoreState() {
  getStreamer().emitCFIRestoreState();
  return false;
}

/// parseDirectiveCFISameValue
/// ::= .cfi_same_value register
bool MasmParser::parseDirectiveCFISameValue(SMLoc DirectiveLoc) {
  int64_t Register = 0;

  if (parseRegisterOrRegisterNumber(Register, DirectiveLoc))
    return true;

  getStreamer().emitCFISameValue(Register);
  return false;
}

/// parseDirectiveCFIRestore
/// ::= .cfi_restore register
bool MasmParser::parseDirectiveCFIRestore(SMLoc DirectiveLoc) {
  int64_t Register = 0;
  if (parseRegisterOrRegisterNumber(Register, DirectiveLoc))
    return true;

  getStreamer().emitCFIRestore(Register);
  return false;
}

/// parseDirectiveCFIEscape
/// ::= .cfi_escape expression[,...]
bool MasmParser::parseDirectiveCFIEscape() {
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

  getStreamer().emitCFIEscape(Values);
  return false;
}

/// parseDirectiveCFIReturnColumn
/// ::= .cfi_return_column register
bool MasmParser::parseDirectiveCFIReturnColumn(SMLoc DirectiveLoc) {
  int64_t Register = 0;
  if (parseRegisterOrRegisterNumber(Register, DirectiveLoc))
    return true;
  getStreamer().emitCFIReturnColumn(Register);
  return false;
}

/// parseDirectiveCFISignalFrame
/// ::= .cfi_signal_frame
bool MasmParser::parseDirectiveCFISignalFrame() {
  if (parseToken(AsmToken::EndOfStatement,
                 "unexpected token in '.cfi_signal_frame'"))
    return true;

  getStreamer().emitCFISignalFrame();
  return false;
}

/// parseDirectiveCFIUndefined
/// ::= .cfi_undefined register
bool MasmParser::parseDirectiveCFIUndefined(SMLoc DirectiveLoc) {
  int64_t Register = 0;

  if (parseRegisterOrRegisterNumber(Register, DirectiveLoc))
    return true;

  getStreamer().emitCFIUndefined(Register);
  return false;
}

/// parseDirectiveAltmacro
/// ::= .altmacro
/// ::= .noaltmacro
bool MasmParser::parseDirectiveAltmacro(StringRef Directive) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '" + Directive + "' directive");
  AltMacroMode = (Directive == ".altmacro");
  return false;
}

/// parseDirectiveMacro
/// ::= .macro name[,] [parameters]
bool MasmParser::parseDirectiveMacro(SMLoc DirectiveLoc) {
  StringRef Name;
  if (parseIdentifier(Name))
    return TokError("expected identifier in '.macro' directive");

  if (getLexer().is(AsmToken::Comma))
    Lex();

  MCAsmMacroParameters Parameters;
  while (getLexer().isNot(AsmToken::EndOfStatement)) {

    if (!Parameters.empty() && Parameters.back().Vararg)
      return Error(Lexer.getLoc(),
                   "Vararg parameter '" + Parameters.back().Name +
                   "' should be last one in the list of parameters.");

    MCAsmMacroParameter Parameter;
    if (parseIdentifier(Parameter.Name))
      return TokError("expected identifier in '.macro' directive");

    // Emit an error if two (or more) named parameters share the same name
    for (const MCAsmMacroParameter& CurrParam : Parameters)
      if (CurrParam.Name.equals(Parameter.Name))
        return TokError("macro '" + Name + "' has multiple parameters"
                        " named '" + Parameter.Name + "'");

    if (Lexer.is(AsmToken::Colon)) {
      Lex();  // consume ':'

      SMLoc QualLoc;
      StringRef Qualifier;

      QualLoc = Lexer.getLoc();
      if (parseIdentifier(Qualifier))
        return Error(QualLoc, "missing parameter qualifier for "
                     "'" + Parameter.Name + "' in macro '" + Name + "'");

      if (Qualifier == "req")
        Parameter.Required = true;
      else if (Qualifier == "vararg")
        Parameter.Vararg = true;
      else
        return Error(QualLoc, Qualifier + " is not a valid parameter qualifier "
                     "for '" + Parameter.Name + "' in macro '" + Name + "'");
    }

    if (getLexer().is(AsmToken::Equal)) {
      Lex();

      SMLoc ParamLoc;

      ParamLoc = Lexer.getLoc();
      if (parseMacroArgument(Parameter.Value, /*Vararg=*/false ))
        return true;

      if (Parameter.Required)
        Warning(ParamLoc, "pointless default value for required parameter "
                "'" + Parameter.Name + "' in macro '" + Name + "'");
    }

    Parameters.push_back(std::move(Parameter));

    if (getLexer().is(AsmToken::Comma))
      Lex();
  }

  // Eat just the end of statement.
  Lexer.Lex();

  // Consuming deferred text, so use Lexer.Lex to ignore Lexing Errors
  AsmToken EndToken, StartToken = getTok();
  unsigned MacroDepth = 0;
  // Lex the macro definition.
  while (true) {
    // Ignore Lexing errors in macros.
    while (Lexer.is(AsmToken::Error)) {
      Lexer.Lex();
    }

    // Check whether we have reached the end of the file.
    if (getLexer().is(AsmToken::Eof))
      return Error(DirectiveLoc, "no matching '.endmacro' in definition");

    // Otherwise, check whether we have reach the .endmacro.
    if (getLexer().is(AsmToken::Identifier)) {
      if (getTok().getIdentifier() == ".endm" ||
          getTok().getIdentifier() == ".endmacro") {
        if (MacroDepth == 0) { // Outermost macro.
          EndToken = getTok();
          Lexer.Lex();
          if (getLexer().isNot(AsmToken::EndOfStatement))
            return TokError("unexpected token in '" + EndToken.getIdentifier() +
                            "' directive");
          break;
        } else {
          // Otherwise we just found the end of an inner macro.
          --MacroDepth;
        }
      } else if (getTok().getIdentifier() == ".macro") {
        // We allow nested macros. Those aren't instantiated until the outermost
        // macro is expanded so just ignore them for now.
        ++MacroDepth;
      }
    }

    // Otherwise, scan til the end of the statement.
    eatToEndOfStatement();
  }

  if (getContext().lookupMacro(Name)) {
    return Error(DirectiveLoc, "macro '" + Name + "' is already defined");
  }

  const char *BodyStart = StartToken.getLoc().getPointer();
  const char *BodyEnd = EndToken.getLoc().getPointer();
  StringRef Body = StringRef(BodyStart, BodyEnd - BodyStart);
  checkForBadMacro(DirectiveLoc, Name, Body, Parameters);
  MCAsmMacro Macro(Name, Body, std::move(Parameters));
  DEBUG_WITH_TYPE("asm-macros", dbgs() << "Defining new macro:\n";
                  Macro.dump());
  getContext().defineMacro(Name, std::move(Macro));
  return false;
}

/// checkForBadMacro
///
/// With the support added for named parameters there may be code out there that
/// is transitioning from positional parameters.  In versions of gas that did
/// not support named parameters they would be ignored on the macro definition.
/// But to support both styles of parameters this is not possible so if a macro
/// definition has named parameters but does not use them and has what appears
/// to be positional parameters, strings like $1, $2, ... and $n, then issue a
/// warning that the positional parameter found in body which have no effect.
/// Hoping the developer will either remove the named parameters from the macro
/// definition so the positional parameters get used if that was what was
/// intended or change the macro to use the named parameters.  It is possible
/// this warning will trigger when the none of the named parameters are used
/// and the strings like $1 are infact to simply to be passed trough unchanged.
void MasmParser::checkForBadMacro(SMLoc DirectiveLoc, StringRef Name,
                                  StringRef Body,
                                  ArrayRef<MCAsmMacroParameter> Parameters) {
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
      switch (Body[Pos + 1]) {
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

      const char *Begin = Body.data() + Pos + 1;
      StringRef Argument(Begin, I - (Pos + 1));
      unsigned Index = 0;
      for (; Index < NParameters; ++Index)
        if (Parameters[Index].Name == Argument)
          break;

      if (Index == NParameters) {
        if (Body[Pos + 1] == '(' && Body[Pos + 2] == ')')
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

/// parseDirectiveExitMacro
/// ::= .exitm
bool MasmParser::parseDirectiveExitMacro(StringRef Directive) {
  if (parseToken(AsmToken::EndOfStatement,
                 "unexpected token in '" + Directive + "' directive"))
    return true;

  if (!isInsideMacroInstantiation())
    return TokError("unexpected '" + Directive + "' in file, "
                                                 "no current macro definition");

  // Exit all conditionals that are active in the current macro.
  while (TheCondStack.size() != ActiveMacros.back()->CondStackDepth) {
    TheCondState = TheCondStack.back();
    TheCondStack.pop_back();
  }

  handleMacroExit();
  return false;
}

/// parseDirectiveEndMacro
/// ::= .endm
/// ::= .endmacro
bool MasmParser::parseDirectiveEndMacro(StringRef Directive) {
  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '" + Directive + "' directive");

  // If we are inside a macro instantiation, terminate the current
  // instantiation.
  if (isInsideMacroInstantiation()) {
    handleMacroExit();
    return false;
  }

  // Otherwise, this .endmacro is a stray entry in the file; well formed
  // .endmacro directives are handled during the macro definition parsing.
  return TokError("unexpected '" + Directive + "' in file, "
                                               "no current macro definition");
}

/// parseDirectivePurgeMacro
/// ::= .purgem
bool MasmParser::parseDirectivePurgeMacro(SMLoc DirectiveLoc) {
  StringRef Name;
  SMLoc Loc;
  if (parseTokenLoc(Loc) ||
      check(parseIdentifier(Name), Loc,
            "expected identifier in '.purgem' directive") ||
      parseToken(AsmToken::EndOfStatement,
                 "unexpected token in '.purgem' directive"))
    return true;

  if (!getContext().lookupMacro(Name))
    return Error(DirectiveLoc, "macro '" + Name + "' is not defined");

  getContext().undefineMacro(Name);
  DEBUG_WITH_TYPE("asm-macros", dbgs()
                                    << "Un-defining macro: " << Name << "\n");
  return false;
}

/// parseDirectiveSymbolAttribute
///  ::= { ".globl", ".weak", ... } [ identifier ( , identifier )* ]
bool MasmParser::parseDirectiveSymbolAttribute(MCSymbolAttr Attr) {
  auto parseOp = [&]() -> bool {
    StringRef Name;
    SMLoc Loc = getTok().getLoc();
    if (parseIdentifier(Name))
      return Error(Loc, "expected identifier");
    MCSymbol *Sym = getContext().getOrCreateSymbol(Name);

    // Assembler local symbols don't make any sense here. Complain loudly.
    if (Sym->isTemporary())
      return Error(Loc, "non-local symbol required");

    if (!getStreamer().emitSymbolAttribute(Sym, Attr))
      return Error(Loc, "unable to emit symbol attribute");
    return false;
  };

  if (parseMany(parseOp))
    return addErrorSuffix(" in directive");
  return false;
}

/// parseDirectiveComm
///  ::= ( .comm | .lcomm ) identifier , size_expression [ , align_expression ]
bool MasmParser::parseDirectiveComm(bool IsLocal) {
  if (checkForValidSection())
    return true;

  SMLoc IDLoc = getLexer().getLoc();
  StringRef Name;
  if (parseIdentifier(Name))
    return TokError("expected identifier in directive");

  // Handle the identifier as the key symbol.
  MCSymbol *Sym = getContext().getOrCreateSymbol(Name);

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

  if (parseToken(AsmToken::EndOfStatement,
                 "unexpected token in '.comm' or '.lcomm' directive"))
    return true;

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

  Sym->redefineIfPossible();
  if (!Sym->isUndefined())
    return Error(IDLoc, "invalid symbol redefinition");

  // Create the Symbol as a common or local common with Size and Pow2Alignment
  if (IsLocal) {
    getStreamer().emitLocalCommonSymbol(Sym, Size, 1 << Pow2Alignment);
    return false;
  }

  getStreamer().emitCommonSymbol(Sym, Size, 1 << Pow2Alignment);
  return false;
}

/// parseDirectiveComment
///  ::= comment delimiter [[text]]
///              [[text]]
///              [[text]] delimiter [[text]]
bool MasmParser::parseDirectiveComment(SMLoc DirectiveLoc) {
  StringRef FirstLine = parseStringToEndOfStatement();
  size_t DelimiterEnd = FirstLine.find_first_of("\b\t\v\f\r\x1A ");
  StringRef Delimiter = FirstLine.take_front(DelimiterEnd);
  if (Delimiter.empty())
    return Error(DirectiveLoc, "no delimiter in 'comment' directive");
  do {
    if (getTok().is(AsmToken::Eof))
      return Error(DirectiveLoc, "unmatched delimiter in 'comment' directive");
    Lex();  // eat end of statement
  } while (!parseStringToEndOfStatement().contains(Delimiter));
  return parseToken(AsmToken::EndOfStatement,
                    "unexpected token in 'comment' directive");
}

/// parseDirectiveInclude
///  ::= include <filename>
///    | include filename
bool MasmParser::parseDirectiveInclude() {
  // Allow the strings to have escaped octal character sequence.
  std::string Filename;
  SMLoc IncludeLoc = getTok().getLoc();

  if (!parseAngleBracketString(Filename))
    Filename = parseStringToEndOfStatement().str();
  if (check(!Filename.empty(), "missing filename in 'include' directive") ||
      check(getTok().isNot(AsmToken::EndOfStatement),
            "unexpected token in 'include' directive") ||
      // Attempt to switch the lexer to the included file before consuming the
      // end of statement to avoid losing it when we switch.
      check(enterIncludeFile(Filename), IncludeLoc,
            "Could not find include file '" + Filename + "'"))
    return true;

  return false;
}

/// parseDirectiveIf
/// ::= .if{,eq,ge,gt,le,lt,ne} expression
bool MasmParser::parseDirectiveIf(SMLoc DirectiveLoc, DirectiveKind DirKind) {
  TheCondStack.push_back(TheCondState);
  TheCondState.TheCond = AsmCond::IfCond;
  if (TheCondState.Ignore) {
    eatToEndOfStatement();
  } else {
    int64_t ExprValue;
    if (parseAbsoluteExpression(ExprValue) ||
        parseToken(AsmToken::EndOfStatement,
                   "unexpected token in '.if' directive"))
      return true;

    switch (DirKind) {
    default:
      llvm_unreachable("unsupported directive");
    case DK_IF:
      break;
    case DK_IFE:
      ExprValue = ExprValue == 0;
      break;
    }

    TheCondState.CondMet = ExprValue;
    TheCondState.Ignore = !TheCondState.CondMet;
  }

  return false;
}

/// parseDirectiveIfb
/// ::= .ifb string
bool MasmParser::parseDirectiveIfb(SMLoc DirectiveLoc, bool ExpectBlank) {
  TheCondStack.push_back(TheCondState);
  TheCondState.TheCond = AsmCond::IfCond;

  if (TheCondState.Ignore) {
    eatToEndOfStatement();
  } else {
    std::string Str;
    if (parseTextItem(Str))
      return TokError("expected string parameter for 'ifb' directive");

    if (parseToken(AsmToken::EndOfStatement,
                   "unexpected token in 'ifb' directive"))
      return true;

    TheCondState.CondMet = ExpectBlank == Str.empty();
    TheCondState.Ignore = !TheCondState.CondMet;
  }

  return false;
}

/// parseDirectiveIfidn
///   ::= ifidn string1, string2
bool MasmParser::parseDirectiveIfidn(SMLoc DirectiveLoc, bool ExpectEqual, bool CaseInsensitive) {
  std::string String1, String2;

  if (parseTextItem(String1)) {
    if (ExpectEqual)
      return TokError("expected string parameter for 'ifidn' directive");
    return TokError("expected string parameter for 'ifdif' directive");
  }

  if (Lexer.isNot(AsmToken::Comma)) {
    if (ExpectEqual)
      return TokError(
          "expected comma after first string for 'ifidn' directive");
    return TokError("expected comma after first string for 'ifdif' directive");
  }
  Lex();

  if (parseTextItem(String2)) {
    if (ExpectEqual)
      return TokError("expected string parameter for 'ifidn' directive");
    return TokError("expected string parameter for 'ifdif' directive");
  }

  TheCondStack.push_back(TheCondState);
  TheCondState.TheCond = AsmCond::IfCond;
  if (CaseInsensitive)
    TheCondState.CondMet =
        ExpectEqual == (StringRef(String1).equals_lower(String2));
  else
    TheCondState.CondMet = ExpectEqual == (String1 == String2);
  TheCondState.Ignore = !TheCondState.CondMet;

  return false;
}

/// parseDirectiveIfdef
/// ::= ifdef symbol
///   | ifdef variable
bool MasmParser::parseDirectiveIfdef(SMLoc DirectiveLoc, bool expect_defined) {
  TheCondStack.push_back(TheCondState);
  TheCondState.TheCond = AsmCond::IfCond;

  if (TheCondState.Ignore) {
    eatToEndOfStatement();
  } else {
    bool is_defined = false;
    unsigned RegNo;
    SMLoc StartLoc, EndLoc;
    is_defined = (getTargetParser().tryParseRegister(
                      RegNo, StartLoc, EndLoc) == MatchOperand_Success);
    if (!is_defined) {
      StringRef Name;
      if (check(parseIdentifier(Name), "expected identifier after 'ifdef'") ||
          parseToken(AsmToken::EndOfStatement, "unexpected token in 'ifdef'"))
        return true;

      if (Variables.find(Name) != Variables.end()) {
        is_defined = true;
      } else {
        MCSymbol *Sym = getContext().lookupSymbol(Name);
        is_defined = (Sym && !Sym->isUndefined(false));
      }
    }

    TheCondState.CondMet = (is_defined == expect_defined);
    TheCondState.Ignore = !TheCondState.CondMet;
  }

  return false;
}

/// parseDirectiveElseIf
/// ::= elseif expression
bool MasmParser::parseDirectiveElseIf(SMLoc DirectiveLoc,
                                      DirectiveKind DirKind) {
  if (TheCondState.TheCond != AsmCond::IfCond &&
      TheCondState.TheCond != AsmCond::ElseIfCond)
    return Error(DirectiveLoc, "Encountered a .elseif that doesn't follow an"
                               " .if or  an .elseif");
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

    if (parseToken(AsmToken::EndOfStatement,
                   "unexpected token in '.elseif' directive"))
      return true;

    switch (DirKind) {
    default:
      llvm_unreachable("unsupported directive");
    case DK_ELSEIF:
      break;
    case DK_ELSEIFE:
      ExprValue = ExprValue == 0;
      break;
    }

    TheCondState.CondMet = ExprValue;
    TheCondState.Ignore = !TheCondState.CondMet;
  }

  return false;
}

/// parseDirectiveElseIfb
/// ::= elseifb expression
bool MasmParser::parseDirectiveElseIfb(SMLoc DirectiveLoc, bool ExpectBlank) {
  if (TheCondState.TheCond != AsmCond::IfCond &&
      TheCondState.TheCond != AsmCond::ElseIfCond)
    return Error(DirectiveLoc, "Encountered an elseif that doesn't follow an"
                               " if or an elseif");
  TheCondState.TheCond = AsmCond::ElseIfCond;

  bool LastIgnoreState = false;
  if (!TheCondStack.empty())
    LastIgnoreState = TheCondStack.back().Ignore;
  if (LastIgnoreState || TheCondState.CondMet) {
    TheCondState.Ignore = true;
    eatToEndOfStatement();
  } else {
    std::string Str;
    if (parseTextItem(Str))
      return TokError("expected string parameter for 'elseifb' directive");

    if (parseToken(AsmToken::EndOfStatement,
                   "unexpected token in 'elseifb' directive"))
      return true;

    TheCondState.CondMet = ExpectBlank == Str.empty();
    TheCondState.Ignore = !TheCondState.CondMet;
  }

  return false;
}

/// parseDirectiveElseIfdef
/// ::= elseifdef symbol
///   | elseifdef variable
bool MasmParser::parseDirectiveElseIfdef(SMLoc DirectiveLoc,
                                         bool expect_defined) {
  if (TheCondState.TheCond != AsmCond::IfCond &&
      TheCondState.TheCond != AsmCond::ElseIfCond)
    return Error(DirectiveLoc, "Encountered an elseif that doesn't follow an"
                               " if or an elseif");
  TheCondState.TheCond = AsmCond::ElseIfCond;

  bool LastIgnoreState = false;
  if (!TheCondStack.empty())
    LastIgnoreState = TheCondStack.back().Ignore;
  if (LastIgnoreState || TheCondState.CondMet) {
    TheCondState.Ignore = true;
    eatToEndOfStatement();
  } else {
    bool is_defined = false;
    unsigned RegNo;
    SMLoc StartLoc, EndLoc;
    is_defined = (getTargetParser().tryParseRegister(RegNo, StartLoc, EndLoc) ==
                  MatchOperand_Success);
    if (!is_defined) {
      StringRef Name;
      if (check(parseIdentifier(Name),
                "expected identifier after 'elseifdef'") ||
          parseToken(AsmToken::EndOfStatement,
                     "unexpected token in 'elseifdef'"))
        return true;

      if (Variables.find(Name) != Variables.end()) {
        is_defined = true;
      } else {
        MCSymbol *Sym = getContext().lookupSymbol(Name);
        is_defined = (Sym && !Sym->isUndefined(false));
      }
    }

    TheCondState.CondMet = (is_defined == expect_defined);
    TheCondState.Ignore = !TheCondState.CondMet;
  }

  return false;
}

/// parseDirectiveElseIfidn
/// ::= elseifidn string1, string2
bool MasmParser::parseDirectiveElseIfidn(SMLoc DirectiveLoc, bool ExpectEqual,
                                         bool CaseInsensitive) {
  if (TheCondState.TheCond != AsmCond::IfCond &&
      TheCondState.TheCond != AsmCond::ElseIfCond)
    return Error(DirectiveLoc, "Encountered an elseif that doesn't follow an"
                               " if or an elseif");
  TheCondState.TheCond = AsmCond::ElseIfCond;

  bool LastIgnoreState = false;
  if (!TheCondStack.empty())
    LastIgnoreState = TheCondStack.back().Ignore;
  if (LastIgnoreState || TheCondState.CondMet) {
    TheCondState.Ignore = true;
    eatToEndOfStatement();
  } else {
    std::string String1, String2;

    if (parseTextItem(String1)) {
      if (ExpectEqual)
        return TokError("expected string parameter for 'elseifidn' directive");
      return TokError("expected string parameter for 'elseifdif' directive");
    }

    if (Lexer.isNot(AsmToken::Comma)) {
      if (ExpectEqual)
        return TokError(
            "expected comma after first string for 'elseifidn' directive");
      return TokError(
          "expected comma after first string for 'elseifdif' directive");
    }
    Lex();

    if (parseTextItem(String2)) {
      if (ExpectEqual)
        return TokError("expected string parameter for 'elseifidn' directive");
      return TokError("expected string parameter for 'elseifdif' directive");
    }

    if (CaseInsensitive)
      TheCondState.CondMet =
          ExpectEqual == (StringRef(String1).equals_lower(String2));
    else
      TheCondState.CondMet = ExpectEqual == (String1 == String2);
    TheCondState.Ignore = !TheCondState.CondMet;
  }

  return false;
}

/// parseDirectiveElse
/// ::= else
bool MasmParser::parseDirectiveElse(SMLoc DirectiveLoc) {
  if (parseToken(AsmToken::EndOfStatement,
                 "unexpected token in 'else' directive"))
    return true;

  if (TheCondState.TheCond != AsmCond::IfCond &&
      TheCondState.TheCond != AsmCond::ElseIfCond)
    return Error(DirectiveLoc, "Encountered an else that doesn't follow an if"
                               " or an elseif");
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

/// parseDirectiveEnd
/// ::= end
bool MasmParser::parseDirectiveEnd(SMLoc DirectiveLoc) {
  if (parseToken(AsmToken::EndOfStatement,
                 "unexpected token in 'end' directive"))
    return true;

  while (Lexer.isNot(AsmToken::Eof))
    Lexer.Lex();

  return false;
}

/// parseDirectiveError
///   ::= .err [message]
bool MasmParser::parseDirectiveError(SMLoc DirectiveLoc) {
  if (!TheCondStack.empty()) {
    if (TheCondStack.back().Ignore) {
      eatToEndOfStatement();
      return false;
    }
  }

  StringRef Message = ".err directive invoked in source file";
  if (Lexer.isNot(AsmToken::EndOfStatement))
    Message = parseStringToEndOfStatement();
  Lex();

  return Error(DirectiveLoc, Message);
}

/// parseDirectiveErrorIfb
///   ::= .errb textitem[, message]
bool MasmParser::parseDirectiveErrorIfb(SMLoc DirectiveLoc, bool ExpectBlank) {
  if (!TheCondStack.empty()) {
    if (TheCondStack.back().Ignore) {
      eatToEndOfStatement();
      return false;
    }
  }

  std::string Text;
  if (parseTextItem(Text))
    return Error(getTok().getLoc(), "missing text item in '.errb' directive");

  StringRef Message = ".errb directive invoked in source file";
  if (Lexer.isNot(AsmToken::EndOfStatement)) {
    if (parseToken(AsmToken::Comma))
      return addErrorSuffix(" in '.errb' directive");
    Message = parseStringToEndOfStatement();
  }
  Lex();

  if (Text.empty() == ExpectBlank)
    return Error(DirectiveLoc, Message);
  return false;
}

/// parseDirectiveErrorIfdef
///   ::= .errdef name[, message]
bool MasmParser::parseDirectiveErrorIfdef(SMLoc DirectiveLoc,
                                          bool ExpectDefined) {
  if (!TheCondStack.empty()) {
    if (TheCondStack.back().Ignore) {
      eatToEndOfStatement();
      return false;
    }
  }

  bool IsDefined = false;
  unsigned RegNo;
  SMLoc StartLoc, EndLoc;
  IsDefined = (getTargetParser().tryParseRegister(RegNo, StartLoc, EndLoc) ==
               MatchOperand_Success);
  if (!IsDefined) {
    StringRef Name;
    if (check(parseIdentifier(Name), "expected identifier after '.errdef'"))
      return true;

    if (Variables.find(Name) != Variables.end()) {
      IsDefined = true;
    } else {
      MCSymbol *Sym = getContext().lookupSymbol(Name);
      IsDefined = (Sym && !Sym->isUndefined(false));
    }
  }

  StringRef Message = ".errdef directive invoked in source file";
  if (Lexer.isNot(AsmToken::EndOfStatement)) {
    if (parseToken(AsmToken::Comma))
      return addErrorSuffix(" in '.errdef' directive");
    Message = parseStringToEndOfStatement();
  }
  Lex();

  if (IsDefined == ExpectDefined)
    return Error(DirectiveLoc, Message);
  return false;
}

/// parseDirectiveErrorIfidn
///   ::= .erridn textitem1, textitem2[, message]
bool MasmParser::parseDirectiveErrorIfidn(SMLoc DirectiveLoc, bool ExpectEqual,
                                          bool CaseInsensitive) {
  if (!TheCondStack.empty()) {
    if (TheCondStack.back().Ignore) {
      eatToEndOfStatement();
      return false;
    }
  }

  std::string String1, String2;

  if (parseTextItem(String1)) {
    if (ExpectEqual)
      return TokError("expected string parameter for '.erridn' directive");
    return TokError("expected string parameter for '.errdif' directive");
  }

  if (Lexer.isNot(AsmToken::Comma)) {
    if (ExpectEqual)
      return TokError(
          "expected comma after first string for '.erridn' directive");
    return TokError(
        "expected comma after first string for '.errdif' directive");
  }
  Lex();

  if (parseTextItem(String2)) {
    if (ExpectEqual)
      return TokError("expected string parameter for '.erridn' directive");
    return TokError("expected string parameter for '.errdif' directive");
  }

  StringRef Message;
  if (ExpectEqual)
    Message = ".erridn directive invoked in source file";
  else
    Message = ".errdif directive invoked in source file";
  if (Lexer.isNot(AsmToken::EndOfStatement)) {
    if (parseToken(AsmToken::Comma))
      return addErrorSuffix(" in '.erridn' directive");
    Message = parseStringToEndOfStatement();
  }
  Lex();

  if (CaseInsensitive)
    TheCondState.CondMet =
        ExpectEqual == (StringRef(String1).equals_lower(String2));
  else
    TheCondState.CondMet = ExpectEqual == (String1 == String2);
  TheCondState.Ignore = !TheCondState.CondMet;

  if ((CaseInsensitive &&
       ExpectEqual == StringRef(String1).equals_lower(String2)) ||
      (ExpectEqual == (String1 == String2)))
    return Error(DirectiveLoc, Message);
  return false;
}

/// parseDirectiveErrorIfe
///   ::= .erre expression[, message]
bool MasmParser::parseDirectiveErrorIfe(SMLoc DirectiveLoc, bool ExpectZero) {
  if (!TheCondStack.empty()) {
    if (TheCondStack.back().Ignore) {
      eatToEndOfStatement();
      return false;
    }
  }

  int64_t ExprValue;
  if (parseAbsoluteExpression(ExprValue))
    return addErrorSuffix(" in '.erre' directive");

  StringRef Message = ".erre directive invoked in source file";
  if (Lexer.isNot(AsmToken::EndOfStatement)) {
    if (parseToken(AsmToken::Comma))
      return addErrorSuffix(" in '.erre' directive");
    Message = parseStringToEndOfStatement();
  }
  Lex();

  if ((ExprValue == 0) == ExpectZero)
    return Error(DirectiveLoc, Message);
  return false;
}

/// parseDirectiveEndIf
/// ::= .endif
bool MasmParser::parseDirectiveEndIf(SMLoc DirectiveLoc) {
  if (parseToken(AsmToken::EndOfStatement,
                 "unexpected token in '.endif' directive"))
    return true;

  if ((TheCondState.TheCond == AsmCond::NoCond) || TheCondStack.empty())
    return Error(DirectiveLoc, "Encountered a .endif that doesn't follow "
                               "an .if or .else");
  if (!TheCondStack.empty()) {
    TheCondState = TheCondStack.back();
    TheCondStack.pop_back();
  }

  return false;
}

void MasmParser::initializeDirectiveKindMap() {
  DirectiveKindMap["="] = DK_ASSIGN;
  DirectiveKindMap["equ"] = DK_EQU;
  DirectiveKindMap["textequ"] = DK_TEXTEQU;
  // DirectiveKindMap[".ascii"] = DK_ASCII;
  // DirectiveKindMap[".asciz"] = DK_ASCIZ;
  // DirectiveKindMap[".string"] = DK_STRING;
  DirectiveKindMap["byte"] = DK_BYTE;
  DirectiveKindMap["sbyte"] = DK_SBYTE;
  DirectiveKindMap["word"] = DK_WORD;
  DirectiveKindMap["sword"] = DK_SWORD;
  DirectiveKindMap["dword"] = DK_DWORD;
  DirectiveKindMap["sdword"] = DK_SDWORD;
  DirectiveKindMap["fword"] = DK_FWORD;
  DirectiveKindMap["qword"] = DK_QWORD;
  DirectiveKindMap["sqword"] = DK_SQWORD;
  DirectiveKindMap["real4"] = DK_REAL4;
  DirectiveKindMap["real8"] = DK_REAL8;
  DirectiveKindMap["align"] = DK_ALIGN;
  // DirectiveKindMap[".org"] = DK_ORG;
  DirectiveKindMap["extern"] = DK_EXTERN;
  DirectiveKindMap["public"] = DK_PUBLIC;
  // DirectiveKindMap[".comm"] = DK_COMM;
  DirectiveKindMap["comment"] = DK_COMMENT;
  DirectiveKindMap["include"] = DK_INCLUDE;
  // DirectiveKindMap[".rept"] = DK_REPT;
  // DirectiveKindMap[".rep"] = DK_REPT;
  // DirectiveKindMap[".irp"] = DK_IRP;
  // DirectiveKindMap[".irpc"] = DK_IRPC;
  // DirectiveKindMap[".endr"] = DK_ENDR;
  DirectiveKindMap["if"] = DK_IF;
  DirectiveKindMap["ife"] = DK_IFE;
  DirectiveKindMap["ifb"] = DK_IFB;
  DirectiveKindMap["ifnb"] = DK_IFNB;
  DirectiveKindMap["ifdef"] = DK_IFDEF;
  DirectiveKindMap["ifndef"] = DK_IFNDEF;
  DirectiveKindMap["ifdif"] = DK_IFDIF;
  DirectiveKindMap["ifdifi"] = DK_IFDIFI;
  DirectiveKindMap["ifidn"] = DK_IFIDN;
  DirectiveKindMap["ifidni"] = DK_IFIDNI;
  DirectiveKindMap["elseif"] = DK_ELSEIF;
  DirectiveKindMap["elseifdef"] = DK_ELSEIFDEF;
  DirectiveKindMap["elseifndef"] = DK_ELSEIFNDEF;
  DirectiveKindMap["elseifdif"] = DK_ELSEIFDIF;
  DirectiveKindMap["elseifidn"] = DK_ELSEIFIDN;
  DirectiveKindMap["else"] = DK_ELSE;
  DirectiveKindMap["end"] = DK_END;
  DirectiveKindMap["endif"] = DK_ENDIF;
  // DirectiveKindMap[".file"] = DK_FILE;
  // DirectiveKindMap[".line"] = DK_LINE;
  // DirectiveKindMap[".loc"] = DK_LOC;
  // DirectiveKindMap[".stabs"] = DK_STABS;
  // DirectiveKindMap[".cv_file"] = DK_CV_FILE;
  // DirectiveKindMap[".cv_func_id"] = DK_CV_FUNC_ID;
  // DirectiveKindMap[".cv_loc"] = DK_CV_LOC;
  // DirectiveKindMap[".cv_linetable"] = DK_CV_LINETABLE;
  // DirectiveKindMap[".cv_inline_linetable"] = DK_CV_INLINE_LINETABLE;
  // DirectiveKindMap[".cv_inline_site_id"] = DK_CV_INLINE_SITE_ID;
  // DirectiveKindMap[".cv_def_range"] = DK_CV_DEF_RANGE;
  // DirectiveKindMap[".cv_string"] = DK_CV_STRING;
  // DirectiveKindMap[".cv_stringtable"] = DK_CV_STRINGTABLE;
  // DirectiveKindMap[".cv_filechecksums"] = DK_CV_FILECHECKSUMS;
  // DirectiveKindMap[".cv_filechecksumoffset"] = DK_CV_FILECHECKSUM_OFFSET;
  // DirectiveKindMap[".cv_fpo_data"] = DK_CV_FPO_DATA;
  // DirectiveKindMap[".cfi_sections"] = DK_CFI_SECTIONS;
  // DirectiveKindMap[".cfi_startproc"] = DK_CFI_STARTPROC;
  // DirectiveKindMap[".cfi_endproc"] = DK_CFI_ENDPROC;
  // DirectiveKindMap[".cfi_def_cfa"] = DK_CFI_DEF_CFA;
  // DirectiveKindMap[".cfi_def_cfa_offset"] = DK_CFI_DEF_CFA_OFFSET;
  // DirectiveKindMap[".cfi_adjust_cfa_offset"] = DK_CFI_ADJUST_CFA_OFFSET;
  // DirectiveKindMap[".cfi_def_cfa_register"] = DK_CFI_DEF_CFA_REGISTER;
  // DirectiveKindMap[".cfi_offset"] = DK_CFI_OFFSET;
  // DirectiveKindMap[".cfi_rel_offset"] = DK_CFI_REL_OFFSET;
  // DirectiveKindMap[".cfi_personality"] = DK_CFI_PERSONALITY;
  // DirectiveKindMap[".cfi_lsda"] = DK_CFI_LSDA;
  // DirectiveKindMap[".cfi_remember_state"] = DK_CFI_REMEMBER_STATE;
  // DirectiveKindMap[".cfi_restore_state"] = DK_CFI_RESTORE_STATE;
  // DirectiveKindMap[".cfi_same_value"] = DK_CFI_SAME_VALUE;
  // DirectiveKindMap[".cfi_restore"] = DK_CFI_RESTORE;
  // DirectiveKindMap[".cfi_escape"] = DK_CFI_ESCAPE;
  // DirectiveKindMap[".cfi_return_column"] = DK_CFI_RETURN_COLUMN;
  // DirectiveKindMap[".cfi_signal_frame"] = DK_CFI_SIGNAL_FRAME;
  // DirectiveKindMap[".cfi_undefined"] = DK_CFI_UNDEFINED;
  // DirectiveKindMap[".cfi_register"] = DK_CFI_REGISTER;
  // DirectiveKindMap[".cfi_window_save"] = DK_CFI_WINDOW_SAVE;
  // DirectiveKindMap[".cfi_b_key_frame"] = DK_CFI_B_KEY_FRAME;
  // DirectiveKindMap[".macro"] = DK_MACRO;
  // DirectiveKindMap[".exitm"] = DK_EXITM;
  // DirectiveKindMap[".endm"] = DK_ENDM;
  // DirectiveKindMap[".purgem"] = DK_PURGEM;
  DirectiveKindMap[".err"] = DK_ERR;
  DirectiveKindMap[".errb"] = DK_ERRB;
  DirectiveKindMap[".errnb"] = DK_ERRNB;
  DirectiveKindMap[".errdef"] = DK_ERRDEF;
  DirectiveKindMap[".errndef"] = DK_ERRNDEF;
  DirectiveKindMap[".errdif"] = DK_ERRDIF;
  DirectiveKindMap[".errdifi"] = DK_ERRDIFI;
  DirectiveKindMap[".erridn"] = DK_ERRIDN;
  DirectiveKindMap[".erridni"] = DK_ERRIDNI;
  DirectiveKindMap[".erre"] = DK_ERRE;
  DirectiveKindMap[".errnz"] = DK_ERRNZ;
  // DirectiveKindMap[".altmacro"] = DK_ALTMACRO;
  // DirectiveKindMap[".noaltmacro"] = DK_NOALTMACRO;
  DirectiveKindMap["db"] = DK_DB;
  DirectiveKindMap["dd"] = DK_DD;
  DirectiveKindMap["dq"] = DK_DQ;
  DirectiveKindMap["dw"] = DK_DW;
  DirectiveKindMap["echo"] = DK_ECHO;
}

MCAsmMacro *MasmParser::parseMacroLikeBody(SMLoc DirectiveLoc) {
  AsmToken EndToken, StartToken = getTok();

  unsigned NestLevel = 0;
  while (true) {
    // Check whether we have reached the end of the file.
    if (getLexer().is(AsmToken::Eof)) {
      printError(DirectiveLoc, "no matching '.endr' in definition");
      return nullptr;
    }

    if (Lexer.is(AsmToken::Identifier) &&
        (getTok().getIdentifier() == ".rep" ||
         getTok().getIdentifier() == ".rept" ||
         getTok().getIdentifier() == ".irp" ||
         getTok().getIdentifier() == ".irpc")) {
      ++NestLevel;
    }

    // Otherwise, check whether we have reached the .endr.
    if (Lexer.is(AsmToken::Identifier) && getTok().getIdentifier() == ".endr") {
      if (NestLevel == 0) {
        EndToken = getTok();
        Lex();
        if (Lexer.isNot(AsmToken::EndOfStatement)) {
          printError(getTok().getLoc(),
                     "unexpected token in '.endr' directive");
          return nullptr;
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
  MacroLikeBodies.emplace_back(StringRef(), Body, MCAsmMacroParameters());
  return &MacroLikeBodies.back();
}

void MasmParser::instantiateMacroLikeBody(MCAsmMacro *M, SMLoc DirectiveLoc,
                                          raw_svector_ostream &OS) {
  OS << ".endr\n";

  std::unique_ptr<MemoryBuffer> Instantiation =
      MemoryBuffer::getMemBufferCopy(OS.str(), "<instantiation>");

  // Create the macro instantiation object and add to the current macro
  // instantiation stack.
  MacroInstantiation *MI = new MacroInstantiation{
      DirectiveLoc, CurBuffer, getTok().getLoc(), TheCondStack.size()};
  ActiveMacros.push_back(MI);

  // Jump to the macro instantiation and prime the lexer.
  CurBuffer = SrcMgr.AddNewSourceBuffer(std::move(Instantiation), SMLoc());
  Lexer.setBuffer(SrcMgr.getMemoryBuffer(CurBuffer)->getBuffer());
  Lex();
}

/// parseDirectiveRept
///   ::= .rep | .rept count
bool MasmParser::parseDirectiveRept(SMLoc DirectiveLoc, StringRef Dir) {
  const MCExpr *CountExpr;
  SMLoc CountLoc = getTok().getLoc();
  if (parseExpression(CountExpr))
    return true;

  int64_t Count;
  if (!CountExpr->evaluateAsAbsolute(Count, getStreamer().getAssemblerPtr())) {
    return Error(CountLoc, "unexpected token in '" + Dir + "' directive");
  }

  if (check(Count < 0, CountLoc, "Count is negative") ||
      parseToken(AsmToken::EndOfStatement,
                 "unexpected token in '" + Dir + "' directive"))
    return true;

  // Lex the rept definition.
  MCAsmMacro *M = parseMacroLikeBody(DirectiveLoc);
  if (!M)
    return true;

  // Macro instantiation is lexical, unfortunately. We construct a new buffer
  // to hold the macro body with substitutions.
  SmallString<256> Buf;
  raw_svector_ostream OS(Buf);
  while (Count--) {
    // Note that the AtPseudoVariable is disabled for instantiations of .rep(t).
    if (expandMacro(OS, M->Body, None, None, false, getTok().getLoc()))
      return true;
  }
  instantiateMacroLikeBody(M, DirectiveLoc, OS);

  return false;
}

/// parseDirectiveIrp
/// ::= .irp symbol,values
bool MasmParser::parseDirectiveIrp(SMLoc DirectiveLoc) {
  MCAsmMacroParameter Parameter;
  MCAsmMacroArguments A;
  if (check(parseIdentifier(Parameter.Name),
            "expected identifier in '.irp' directive") ||
      parseToken(AsmToken::Comma, "expected comma in '.irp' directive") ||
      parseMacroArguments(nullptr, A) ||
      parseToken(AsmToken::EndOfStatement, "expected End of Statement"))
    return true;

  // Lex the irp definition.
  MCAsmMacro *M = parseMacroLikeBody(DirectiveLoc);
  if (!M)
    return true;

  // Macro instantiation is lexical, unfortunately. We construct a new buffer
  // to hold the macro body with substitutions.
  SmallString<256> Buf;
  raw_svector_ostream OS(Buf);

  for (const MCAsmMacroArgument &Arg : A) {
    // Note that the AtPseudoVariable is enabled for instantiations of .irp.
    // This is undocumented, but GAS seems to support it.
    if (expandMacro(OS, M->Body, Parameter, Arg, true, getTok().getLoc()))
      return true;
  }

  instantiateMacroLikeBody(M, DirectiveLoc, OS);

  return false;
}

/// parseDirectiveIrpc
/// ::= .irpc symbol,values
bool MasmParser::parseDirectiveIrpc(SMLoc DirectiveLoc) {
  MCAsmMacroParameter Parameter;
  MCAsmMacroArguments A;

  if (check(parseIdentifier(Parameter.Name),
            "expected identifier in '.irpc' directive") ||
      parseToken(AsmToken::Comma, "expected comma in '.irpc' directive") ||
      parseMacroArguments(nullptr, A))
    return true;

  if (A.size() != 1 || A.front().size() != 1)
    return TokError("unexpected token in '.irpc' directive");

  // Eat the end of statement.
  if (parseToken(AsmToken::EndOfStatement, "expected end of statement"))
    return true;

  // Lex the irpc definition.
  MCAsmMacro *M = parseMacroLikeBody(DirectiveLoc);
  if (!M)
    return true;

  // Macro instantiation is lexical, unfortunately. We construct a new buffer
  // to hold the macro body with substitutions.
  SmallString<256> Buf;
  raw_svector_ostream OS(Buf);

  StringRef Values = A.front().front().getString();
  for (std::size_t I = 0, End = Values.size(); I != End; ++I) {
    MCAsmMacroArgument Arg;
    Arg.emplace_back(AsmToken::Identifier, Values.slice(I, I + 1));

    // Note that the AtPseudoVariable is enabled for instantiations of .irpc.
    // This is undocumented, but GAS seems to support it.
    if (expandMacro(OS, M->Body, Parameter, Arg, true, getTok().getLoc()))
      return true;
  }

  instantiateMacroLikeBody(M, DirectiveLoc, OS);

  return false;
}

bool MasmParser::parseDirectiveEndr(SMLoc DirectiveLoc) {
  if (ActiveMacros.empty())
    return TokError("unmatched '.endr' directive");

  // The only .repl that should get here are the ones created by
  // instantiateMacroLikeBody.
  assert(getLexer().is(AsmToken::EndOfStatement));

  handleMacroExit();
  return false;
}

bool MasmParser::parseDirectiveMSEmit(SMLoc IDLoc, ParseStatementInfo &Info,
                                      size_t Len) {
  const MCExpr *Value;
  SMLoc ExprLoc = getLexer().getLoc();
  if (parseExpression(Value))
    return true;
  const MCConstantExpr *MCE = dyn_cast<MCConstantExpr>(Value);
  if (!MCE)
    return Error(ExprLoc, "unexpected expression in _emit");
  uint64_t IntValue = MCE->getValue();
  if (!isUInt<8>(IntValue) && !isInt<8>(IntValue))
    return Error(ExprLoc, "literal value out of range for directive");

  Info.AsmRewrites->emplace_back(AOK_Emit, IDLoc, Len);
  return false;
}

bool MasmParser::parseDirectiveMSAlign(SMLoc IDLoc, ParseStatementInfo &Info) {
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

  Info.AsmRewrites->emplace_back(AOK_Align, IDLoc, 5, Log2_64(IntValue));
  return false;
}

bool MasmParser::parseDirectiveEcho() {
  StringRef Message = parseStringToEndOfStatement();
  Lex();  // eat end of statement
  llvm::outs() << Message << '\n';
  return false;
}

// We are comparing pointers, but the pointers are relative to a single string.
// Thus, this should always be deterministic.
static int rewritesSort(const AsmRewrite *AsmRewriteA,
                        const AsmRewrite *AsmRewriteB) {
  if (AsmRewriteA->Loc.getPointer() < AsmRewriteB->Loc.getPointer())
    return -1;
  if (AsmRewriteB->Loc.getPointer() < AsmRewriteA->Loc.getPointer())
    return 1;

  // It's possible to have a SizeDirective, Imm/ImmPrefix and an Input/Output
  // rewrite to the same location.  Make sure the SizeDirective rewrite is
  // performed first, then the Imm/ImmPrefix and finally the Input/Output.  This
  // ensures the sort algorithm is stable.
  if (AsmRewritePrecedence[AsmRewriteA->Kind] >
      AsmRewritePrecedence[AsmRewriteB->Kind])
    return -1;

  if (AsmRewritePrecedence[AsmRewriteA->Kind] <
      AsmRewritePrecedence[AsmRewriteB->Kind])
    return 1;
  llvm_unreachable("Unstable rewrite sort.");
}

bool MasmParser::parseMSInlineAsm(
    void *AsmLoc, std::string &AsmString, unsigned &NumOutputs,
    unsigned &NumInputs, SmallVectorImpl<std::pair<void *, bool>> &OpDecls,
    SmallVectorImpl<std::string> &Constraints,
    SmallVectorImpl<std::string> &Clobbers, const MCInstrInfo *MII,
    const MCInstPrinter *IP, MCAsmParserSemaCallback &SI) {
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
    // Parse curly braces marking block start/end
    if (parseCurlyBlockScope(AsmStrRewrites))
      continue;

    ParseStatementInfo Info(&AsmStrRewrites);
    bool StatementErr = parseStatement(Info, &SI);

    if (StatementErr || Info.ParseError) {
      // Emit pending errors if any exist.
      printPendingErrors();
      return true;
    }

    // No pending error should exist here.
    assert(!hasPendingError() && "unexpected error from parseStatement");

    if (Info.Opcode == ~0U)
      continue;

    const MCInstrDesc &Desc = MII->get(Info.Opcode);

    // Build the list of clobbers, outputs and inputs.
    for (unsigned i = 1, e = Info.ParsedOperands.size(); i != e; ++i) {
      MCParsedAsmOperand &Operand = *Info.ParsedOperands[i];

      // Register operand.
      if (Operand.isReg() && !Operand.needAddressOf() &&
          !getTargetParser().OmitRegisterFromClobberLists(Operand.getReg())) {
        unsigned NumDefs = Desc.getNumDefs();
        // Clobber.
        if (NumDefs && Operand.getMCOperandNum() < NumDefs)
          ClobberRegs.push_back(Operand.getReg());
        continue;
      }

      // Expr/Input or Output.
      StringRef SymName = Operand.getSymName();
      if (SymName.empty())
        continue;

      void *OpDecl = Operand.getOpDecl();
      if (!OpDecl)
        continue;

      StringRef Constraint = Operand.getConstraint();
      if (Operand.isImm()) {
        // Offset as immediate
        if (Operand.isOffsetOfLocal())
          Constraint = "r";
        else
          Constraint = "i";
      }

      bool isOutput = (i == 1) && Desc.mayStore();
      SMLoc Start = SMLoc::getFromPointer(SymName.data());
      if (isOutput) {
        ++InputIdx;
        OutputDecls.push_back(OpDecl);
        OutputDeclsAddressOf.push_back(Operand.needAddressOf());
        OutputConstraints.push_back(("=" + Constraint).str());
        AsmStrRewrites.emplace_back(AOK_Output, Start, SymName.size());
      } else {
        InputDecls.push_back(OpDecl);
        InputDeclsAddressOf.push_back(Operand.needAddressOf());
        InputConstraints.push_back(Constraint.str());
        if (Desc.OpInfo[i - 1].isBranchTarget())
          AsmStrRewrites.emplace_back(AOK_CallInput, Start, SymName.size());
        else
          AsmStrRewrites.emplace_back(AOK_Input, Start, SymName.size());
      }
    }

    // Consider implicit defs to be clobbers.  Think of cpuid and push.
    ArrayRef<MCPhysReg> ImpDefs(Desc.getImplicitDefs(),
                                Desc.getNumImplicitDefs());
    ClobberRegs.insert(ClobberRegs.end(), ImpDefs.begin(), ImpDefs.end());
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
  StringRef ASMString =
      SrcMgr.getMemoryBuffer(SrcMgr.getMainFileID())->getBuffer();
  const char *AsmStart = ASMString.begin();
  const char *AsmEnd = ASMString.end();
  array_pod_sort(AsmStrRewrites.begin(), AsmStrRewrites.end(), rewritesSort);
  for (auto it = AsmStrRewrites.begin(); it != AsmStrRewrites.end(); ++it) {
    const AsmRewrite &AR = *it;
    // Check if this has already been covered by another rewrite...
    if (AR.Done)
      continue;
    AsmRewriteKind Kind = AR.Kind;

    const char *Loc = AR.Loc.getPointer();
    assert(Loc >= AsmStart && "Expected Loc to be at or after Start!");

    // Emit everything up to the immediate/expression.
    if (unsigned Len = Loc - AsmStart)
      OS << StringRef(AsmStart, Len);

    // Skip the original expression.
    if (Kind == AOK_Skip) {
      AsmStart = Loc + AR.Len;
      continue;
    }

    unsigned AdditionalSkip = 0;
    // Rewrite expressions in $N notation.
    switch (Kind) {
    default:
      break;
    case AOK_IntelExpr:
      assert(AR.IntelExp.isValid() && "cannot write invalid intel expression");
      if (AR.IntelExp.NeedBracs)
        OS << "[";
      if (AR.IntelExp.hasBaseReg())
        OS << AR.IntelExp.BaseReg;
      if (AR.IntelExp.hasIndexReg())
        OS << (AR.IntelExp.hasBaseReg() ? " + " : "")
           << AR.IntelExp.IndexReg;
      if (AR.IntelExp.Scale > 1)
        OS << " * $$" << AR.IntelExp.Scale;
      if (AR.IntelExp.hasOffset()) {
        if (AR.IntelExp.hasRegs())
          OS << " + ";
        // Fuse this rewrite with a rewrite of the offset name, if present.
        StringRef OffsetName = AR.IntelExp.OffsetName;
        SMLoc OffsetLoc = SMLoc::getFromPointer(AR.IntelExp.OffsetName.data());
        size_t OffsetLen = OffsetName.size();
        auto rewrite_it = std::find_if(
            it, AsmStrRewrites.end(), [&](const AsmRewrite &FusingAR) {
              return FusingAR.Loc == OffsetLoc && FusingAR.Len == OffsetLen &&
                     (FusingAR.Kind == AOK_Input ||
                      FusingAR.Kind == AOK_CallInput);
            });
        if (rewrite_it == AsmStrRewrites.end()) {
          OS << "offset " << OffsetName;
        } else if (rewrite_it->Kind == AOK_CallInput) {
          OS << "${" << InputIdx++ << ":P}";
          rewrite_it->Done = true;
        } else {
          OS << '$' << InputIdx++;
          rewrite_it->Done = true;
        }
      }
      if (AR.IntelExp.Imm || AR.IntelExp.emitImm())
        OS << (AR.IntelExp.emitImm() ? "$$" : " + $$") << AR.IntelExp.Imm;
      if (AR.IntelExp.NeedBracs)
        OS << "]";
      break;
    case AOK_Label:
      OS << Ctx.getAsmInfo()->getPrivateLabelPrefix() << AR.Label;
      break;
    case AOK_Input:
      OS << '$' << InputIdx++;
      break;
    case AOK_CallInput:
      OS << "${" << InputIdx++ << ":P}";
      break;
    case AOK_Output:
      OS << '$' << OutputIdx++;
      break;
    case AOK_SizeDirective:
      switch (AR.Val) {
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
      // MS alignment directives are measured in bytes. If the native assembler
      // measures alignment in bytes, we can pass it straight through.
      OS << ".align";
      if (getContext().getAsmInfo()->getAlignmentIsInBytes())
        break;

      // Alignment is in log2 form, so print that instead and skip the original
      // immediate.
      unsigned Val = AR.Val;
      OS << ' ' << Val;
      assert(Val < 10 && "Expected alignment less then 2^10.");
      AdditionalSkip = (Val < 4) ? 2 : Val < 7 ? 3 : 4;
      break;
    }
    case AOK_EVEN:
      OS << ".even";
      break;
    case AOK_EndOfStatement:
      OS << "\n\t";
      break;
    }

    // Skip the original expression.
    AsmStart = Loc + AR.Len + AdditionalSkip;
  }

  // Emit the remainder of the asm string.
  if (AsmStart != AsmEnd)
    OS << StringRef(AsmStart, AsmEnd - AsmStart);

  AsmString = OS.str();
  return false;
}

/// Create an MCAsmParser instance.
MCAsmParser *llvm::createMCMasmParser(SourceMgr &SM, MCContext &C,
                                      MCStreamer &Out, const MCAsmInfo &MAI,
                                      unsigned CB) {
  return new MasmParser(SM, C, Out, MAI, CB);
}
