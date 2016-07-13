//===-- llvm-mc.cpp - Machine Code Hacking Driver ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This utility is a simple driver that allows command line hacking on machine
// code.
//
//===----------------------------------------------------------------------===//

#include "Disassembler.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCParser/AsmLexer.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptionsCommandFlags.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Output filename"),
               cl::value_desc("filename"));

static cl::opt<bool>
ShowEncoding("show-encoding", cl::desc("Show instruction encodings"));

static cl::opt<bool> RelaxELFRel(
    "relax-relocations", cl::init(true),
    cl::desc("Emit R_X86_64_GOTPCRELX instead of R_X86_64_GOTPCREL"));

static cl::opt<DebugCompressionType>
CompressDebugSections("compress-debug-sections", cl::ValueOptional,
  cl::init(DebugCompressionType::DCT_None),
  cl::desc("Choose DWARF debug sections compression:"),
  cl::values(
    clEnumValN(DebugCompressionType::DCT_None, "none",
      "No compression"),
    clEnumValN(DebugCompressionType::DCT_Zlib, "zlib",
      "Use zlib compression"),
    clEnumValN(DebugCompressionType::DCT_ZlibGnu, "zlib-gnu",
      "Use zlib-gnu compression (depricated)"),
    clEnumValEnd));

static cl::opt<bool>
ShowInst("show-inst", cl::desc("Show internal instruction representation"));

static cl::opt<bool>
ShowInstOperands("show-inst-operands",
                 cl::desc("Show instructions operands as parsed"));

static cl::opt<unsigned>
OutputAsmVariant("output-asm-variant",
                 cl::desc("Syntax variant to use for output printing"));

static cl::opt<bool>
PrintImmHex("print-imm-hex", cl::init(false),
            cl::desc("Prefer hex format for immediate values"));

static cl::list<std::string>
DefineSymbol("defsym", cl::desc("Defines a symbol to be an integer constant"));

static cl::opt<bool>
    PreserveComments("preserve-comments",
                     cl::desc("Preserve Comments in outputted assembly"));

enum OutputFileType {
  OFT_Null,
  OFT_AssemblyFile,
  OFT_ObjectFile
};
static cl::opt<OutputFileType>
FileType("filetype", cl::init(OFT_AssemblyFile),
  cl::desc("Choose an output file type:"),
  cl::values(
       clEnumValN(OFT_AssemblyFile, "asm",
                  "Emit an assembly ('.s') file"),
       clEnumValN(OFT_Null, "null",
                  "Don't emit anything (for timing purposes)"),
       clEnumValN(OFT_ObjectFile, "obj",
                  "Emit a native object ('.o') file"),
       clEnumValEnd));

static cl::list<std::string>
IncludeDirs("I", cl::desc("Directory of include files"),
            cl::value_desc("directory"), cl::Prefix);

static cl::opt<std::string>
ArchName("arch", cl::desc("Target arch to assemble for, "
                          "see -version for available targets"));

static cl::opt<std::string>
TripleName("triple", cl::desc("Target triple to assemble for, "
                              "see -version for available targets"));

static cl::opt<std::string>
MCPU("mcpu",
     cl::desc("Target a specific cpu type (-mcpu=help for details)"),
     cl::value_desc("cpu-name"),
     cl::init(""));

static cl::list<std::string>
MAttrs("mattr",
  cl::CommaSeparated,
  cl::desc("Target specific attributes (-mattr=help for details)"),
  cl::value_desc("a1,+a2,-a3,..."));

static cl::opt<bool> PIC("position-independent",
                         cl::desc("Position independent"), cl::init(false));

static cl::opt<llvm::CodeModel::Model>
CMModel("code-model",
        cl::desc("Choose code model"),
        cl::init(CodeModel::Default),
        cl::values(clEnumValN(CodeModel::Default, "default",
                              "Target default code model"),
                   clEnumValN(CodeModel::Small, "small",
                              "Small code model"),
                   clEnumValN(CodeModel::Kernel, "kernel",
                              "Kernel code model"),
                   clEnumValN(CodeModel::Medium, "medium",
                              "Medium code model"),
                   clEnumValN(CodeModel::Large, "large",
                              "Large code model"),
                   clEnumValEnd));

static cl::opt<bool>
NoInitialTextSection("n", cl::desc("Don't assume assembly file starts "
                                   "in the text section"));

static cl::opt<bool>
GenDwarfForAssembly("g", cl::desc("Generate dwarf debugging info for assembly "
                                  "source files"));

static cl::opt<std::string>
DebugCompilationDir("fdebug-compilation-dir",
                    cl::desc("Specifies the debug info's compilation dir"));

static cl::opt<std::string>
MainFileName("main-file-name",
             cl::desc("Specifies the name we should consider the input file"));

static cl::opt<bool> SaveTempLabels("save-temp-labels",
                                    cl::desc("Don't discard temporary labels"));

static cl::opt<bool> NoExecStack("no-exec-stack",
                                 cl::desc("File doesn't need an exec stack"));

enum ActionType {
  AC_AsLex,
  AC_Assemble,
  AC_Disassemble,
  AC_MDisassemble,
};

static cl::opt<ActionType>
Action(cl::desc("Action to perform:"),
       cl::init(AC_Assemble),
       cl::values(clEnumValN(AC_AsLex, "as-lex",
                             "Lex tokens from a .s file"),
                  clEnumValN(AC_Assemble, "assemble",
                             "Assemble a .s file (default)"),
                  clEnumValN(AC_Disassemble, "disassemble",
                             "Disassemble strings of hex bytes"),
                  clEnumValN(AC_MDisassemble, "mdis",
                             "Marked up disassembly of strings of hex bytes"),
                  clEnumValEnd));

static const Target *GetTarget(const char *ProgName) {
  // Figure out the target triple.
  if (TripleName.empty())
    TripleName = sys::getDefaultTargetTriple();
  Triple TheTriple(Triple::normalize(TripleName));

  // Get the target specific parser.
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(ArchName, TheTriple,
                                                         Error);
  if (!TheTarget) {
    errs() << ProgName << ": " << Error;
    return nullptr;
  }

  // Update the triple name and return the found target.
  TripleName = TheTriple.getTriple();
  return TheTarget;
}

static std::unique_ptr<tool_output_file> GetOutputStream() {
  if (OutputFilename == "")
    OutputFilename = "-";

  std::error_code EC;
  auto Out = llvm::make_unique<tool_output_file>(OutputFilename, EC,
                                                 sys::fs::F_None);
  if (EC) {
    errs() << EC.message() << '\n';
    return nullptr;
  }

  return Out;
}

static std::string DwarfDebugFlags;
static void setDwarfDebugFlags(int argc, char **argv) {
  if (!getenv("RC_DEBUG_OPTIONS"))
    return;
  for (int i = 0; i < argc; i++) {
    DwarfDebugFlags += argv[i];
    if (i + 1 < argc)
      DwarfDebugFlags += " ";
  }
}

static std::string DwarfDebugProducer;
static void setDwarfDebugProducer() {
  if(!getenv("DEBUG_PRODUCER"))
    return;
  DwarfDebugProducer += getenv("DEBUG_PRODUCER");
}

static int AsLexInput(SourceMgr &SrcMgr, MCAsmInfo &MAI,
                      raw_ostream &OS) {

  AsmLexer Lexer(MAI);
  Lexer.setBuffer(SrcMgr.getMemoryBuffer(SrcMgr.getMainFileID())->getBuffer());

  bool Error = false;
  while (Lexer.Lex().isNot(AsmToken::Eof)) {
    const AsmToken &Tok = Lexer.getTok();

    switch (Tok.getKind()) {
    default:
      SrcMgr.PrintMessage(Lexer.getLoc(), SourceMgr::DK_Warning,
                          "unknown token");
      Error = true;
      break;
    case AsmToken::Error:
      Error = true; // error already printed.
      break;
    case AsmToken::Identifier:
      OS << "identifier: " << Lexer.getTok().getString();
      break;
    case AsmToken::Integer:
      OS << "int: " << Lexer.getTok().getString();
      break;
    case AsmToken::Real:
      OS << "real: " << Lexer.getTok().getString();
      break;
    case AsmToken::String:
      OS << "string: " << Lexer.getTok().getString();
      break;

    case AsmToken::Amp:            OS << "Amp"; break;
    case AsmToken::AmpAmp:         OS << "AmpAmp"; break;
    case AsmToken::At:             OS << "At"; break;
    case AsmToken::Caret:          OS << "Caret"; break;
    case AsmToken::Colon:          OS << "Colon"; break;
    case AsmToken::Comma:          OS << "Comma"; break;
    case AsmToken::Dollar:         OS << "Dollar"; break;
    case AsmToken::Dot:            OS << "Dot"; break;
    case AsmToken::EndOfStatement: OS << "EndOfStatement"; break;
    case AsmToken::Eof:            OS << "Eof"; break;
    case AsmToken::Equal:          OS << "Equal"; break;
    case AsmToken::EqualEqual:     OS << "EqualEqual"; break;
    case AsmToken::Exclaim:        OS << "Exclaim"; break;
    case AsmToken::ExclaimEqual:   OS << "ExclaimEqual"; break;
    case AsmToken::Greater:        OS << "Greater"; break;
    case AsmToken::GreaterEqual:   OS << "GreaterEqual"; break;
    case AsmToken::GreaterGreater: OS << "GreaterGreater"; break;
    case AsmToken::Hash:           OS << "Hash"; break;
    case AsmToken::LBrac:          OS << "LBrac"; break;
    case AsmToken::LCurly:         OS << "LCurly"; break;
    case AsmToken::LParen:         OS << "LParen"; break;
    case AsmToken::Less:           OS << "Less"; break;
    case AsmToken::LessEqual:      OS << "LessEqual"; break;
    case AsmToken::LessGreater:    OS << "LessGreater"; break;
    case AsmToken::LessLess:       OS << "LessLess"; break;
    case AsmToken::Minus:          OS << "Minus"; break;
    case AsmToken::Percent:        OS << "Percent"; break;
    case AsmToken::Pipe:           OS << "Pipe"; break;
    case AsmToken::PipePipe:       OS << "PipePipe"; break;
    case AsmToken::Plus:           OS << "Plus"; break;
    case AsmToken::RBrac:          OS << "RBrac"; break;
    case AsmToken::RCurly:         OS << "RCurly"; break;
    case AsmToken::RParen:         OS << "RParen"; break;
    case AsmToken::Slash:          OS << "Slash"; break;
    case AsmToken::Star:           OS << "Star"; break;
    case AsmToken::Tilde:          OS << "Tilde"; break;
    }

    // Print the token string.
    OS << " (\"";
    OS.write_escaped(Tok.getString());
    OS << "\")\n";
  }

  return Error;
}

static int fillCommandLineSymbols(MCAsmParser &Parser){
  for(auto &I: DefineSymbol){
    auto Pair = StringRef(I).split('=');
    if(Pair.second.empty()){
      errs() << "error: defsym must be of the form: sym=value: " << I;
      return 1;
    }
    int64_t Value;
    if(Pair.second.getAsInteger(0, Value)){
      errs() << "error: Value is not an integer: " << Pair.second;
      return 1;
    }
    auto &Context = Parser.getContext();
    auto Symbol = Context.getOrCreateSymbol(Pair.first);
    Parser.getStreamer().EmitAssignment(Symbol,
                                        MCConstantExpr::create(Value, Context));
  }
  return 0;
}

static int AssembleInput(const char *ProgName, const Target *TheTarget,
                         SourceMgr &SrcMgr, MCContext &Ctx, MCStreamer &Str,
                         MCAsmInfo &MAI, MCSubtargetInfo &STI,
                         MCInstrInfo &MCII, MCTargetOptions &MCOptions) {
  std::unique_ptr<MCAsmParser> Parser(
      createMCAsmParser(SrcMgr, Ctx, Str, MAI));
  std::unique_ptr<MCTargetAsmParser> TAP(
      TheTarget->createMCAsmParser(STI, *Parser, MCII, MCOptions));

  if (!TAP) {
    errs() << ProgName
           << ": error: this target does not support assembly parsing.\n";
    return 1;
  }

  int SymbolResult = fillCommandLineSymbols(*Parser);
  if(SymbolResult)
    return SymbolResult;
  Parser->setShowParsedOperands(ShowInstOperands);
  Parser->setTargetParser(*TAP);

  int Res = Parser->Run(NoInitialTextSection);

  return Res;
}

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  // Initialize targets and assembly printers/parsers.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllDisassemblers();

  // Register the target printer for --version.
  cl::AddExtraVersionPrinter(TargetRegistry::printRegisteredTargetsForVersion);

  cl::ParseCommandLineOptions(argc, argv, "llvm machine code playground\n");
  MCTargetOptions MCOptions = InitMCTargetOptionsFromFlags();
  TripleName = Triple::normalize(TripleName);
  setDwarfDebugFlags(argc, argv);

  setDwarfDebugProducer();

  const char *ProgName = argv[0];
  const Target *TheTarget = GetTarget(ProgName);
  if (!TheTarget)
    return 1;
  // Now that GetTarget() has (potentially) replaced TripleName, it's safe to
  // construct the Triple object.
  Triple TheTriple(TripleName);

  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferPtr =
      MemoryBuffer::getFileOrSTDIN(InputFilename);
  if (std::error_code EC = BufferPtr.getError()) {
    errs() << InputFilename << ": " << EC.message() << '\n';
    return 1;
  }
  MemoryBuffer *Buffer = BufferPtr->get();

  SourceMgr SrcMgr;

  // Tell SrcMgr about this buffer, which is what the parser will pick up.
  SrcMgr.AddNewSourceBuffer(std::move(*BufferPtr), SMLoc());

  // Record the location of the include directories so that the lexer can find
  // it later.
  SrcMgr.setIncludeDirs(IncludeDirs);

  std::unique_ptr<MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TripleName));
  assert(MRI && "Unable to create target register info!");

  std::unique_ptr<MCAsmInfo> MAI(TheTarget->createMCAsmInfo(*MRI, TripleName));
  assert(MAI && "Unable to create target asm info!");

  MAI->setRelaxELFRelocations(RelaxELFRel);

  if (CompressDebugSections != DebugCompressionType::DCT_None) {
    if (!zlib::isAvailable()) {
      errs() << ProgName
             << ": build tools with zlib to enable -compress-debug-sections";
      return 1;
    }
    MAI->setCompressDebugSections(CompressDebugSections);
  }
  MAI->setPreserveAsmComments(PreserveComments);

  // FIXME: This is not pretty. MCContext has a ptr to MCObjectFileInfo and
  // MCObjectFileInfo needs a MCContext reference in order to initialize itself.
  MCObjectFileInfo MOFI;
  MCContext Ctx(MAI.get(), MRI.get(), &MOFI, &SrcMgr);
  MOFI.InitMCObjectFileInfo(TheTriple, PIC, CMModel, Ctx);

  if (SaveTempLabels)
    Ctx.setAllowTemporaryLabels(false);

  Ctx.setGenDwarfForAssembly(GenDwarfForAssembly);
  // Default to 4 for dwarf version.
  unsigned DwarfVersion = MCOptions.DwarfVersion ? MCOptions.DwarfVersion : 4;
  if (DwarfVersion < 2 || DwarfVersion > 4) {
    errs() << ProgName << ": Dwarf version " << DwarfVersion
           << " is not supported." << '\n';
    return 1;
  }
  Ctx.setDwarfVersion(DwarfVersion);
  if (!DwarfDebugFlags.empty())
    Ctx.setDwarfDebugFlags(StringRef(DwarfDebugFlags));
  if (!DwarfDebugProducer.empty())
    Ctx.setDwarfDebugProducer(StringRef(DwarfDebugProducer));
  if (!DebugCompilationDir.empty())
    Ctx.setCompilationDir(DebugCompilationDir);
  else {
    // If no compilation dir is set, try to use the current directory.
    SmallString<128> CWD;
    if (!sys::fs::current_path(CWD))
      Ctx.setCompilationDir(CWD);
  }
  if (!MainFileName.empty())
    Ctx.setMainFileName(MainFileName);

  // Package up features to be passed to target/subtarget
  std::string FeaturesStr;
  if (MAttrs.size()) {
    SubtargetFeatures Features;
    for (unsigned i = 0; i != MAttrs.size(); ++i)
      Features.AddFeature(MAttrs[i]);
    FeaturesStr = Features.getString();
  }

  std::unique_ptr<tool_output_file> Out = GetOutputStream();
  if (!Out)
    return 1;

  std::unique_ptr<buffer_ostream> BOS;
  raw_pwrite_stream *OS = &Out->os();
  std::unique_ptr<MCStreamer> Str;

  std::unique_ptr<MCInstrInfo> MCII(TheTarget->createMCInstrInfo());
  std::unique_ptr<MCSubtargetInfo> STI(
      TheTarget->createMCSubtargetInfo(TripleName, MCPU, FeaturesStr));

  MCInstPrinter *IP = nullptr;
  if (FileType == OFT_AssemblyFile) {
    IP = TheTarget->createMCInstPrinter(Triple(TripleName), OutputAsmVariant,
                                        *MAI, *MCII, *MRI);

    // Set the display preference for hex vs. decimal immediates.
    IP->setPrintImmHex(PrintImmHex);

    // Set up the AsmStreamer.
    MCCodeEmitter *CE = nullptr;
    MCAsmBackend *MAB = nullptr;
    if (ShowEncoding) {
      CE = TheTarget->createMCCodeEmitter(*MCII, *MRI, Ctx);
      MAB = TheTarget->createMCAsmBackend(*MRI, TripleName, MCPU);
    }
    auto FOut = llvm::make_unique<formatted_raw_ostream>(*OS);
    Str.reset(TheTarget->createAsmStreamer(
        Ctx, std::move(FOut), /*asmverbose*/ true,
        /*useDwarfDirectory*/ true, IP, CE, MAB, ShowInst));

  } else if (FileType == OFT_Null) {
    Str.reset(TheTarget->createNullStreamer(Ctx));
  } else {
    assert(FileType == OFT_ObjectFile && "Invalid file type!");

    // Don't waste memory on names of temp labels.
    Ctx.setUseNamesOnTempLabels(false);

    if (!Out->os().supportsSeeking()) {
      BOS = make_unique<buffer_ostream>(Out->os());
      OS = BOS.get();
    }

    MCCodeEmitter *CE = TheTarget->createMCCodeEmitter(*MCII, *MRI, Ctx);
    MCAsmBackend *MAB = TheTarget->createMCAsmBackend(*MRI, TripleName, MCPU);
    Str.reset(TheTarget->createMCObjectStreamer(
        TheTriple, Ctx, *MAB, *OS, CE, *STI, MCOptions.MCRelaxAll,
        MCOptions.MCIncrementalLinkerCompatible,
        /*DWARFMustBeAtTheEnd*/ false));
    if (NoExecStack)
      Str->InitSections(true);
  }

  int Res = 1;
  bool disassemble = false;
  switch (Action) {
  case AC_AsLex:
    Res = AsLexInput(SrcMgr, *MAI, Out->os());
    break;
  case AC_Assemble:
    Res = AssembleInput(ProgName, TheTarget, SrcMgr, Ctx, *Str, *MAI, *STI,
                        *MCII, MCOptions);
    break;
  case AC_MDisassemble:
    assert(IP && "Expected assembly output");
    IP->setUseMarkup(1);
    disassemble = true;
    break;
  case AC_Disassemble:
    disassemble = true;
    break;
  }
  if (disassemble)
    Res = Disassembler::disassemble(*TheTarget, TripleName, *STI, *Str,
                                    *Buffer, SrcMgr, Out->os());

  // Keep output if no errors.
  if (Res == 0) Out->keep();
  return Res;
}
