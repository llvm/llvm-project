//===-- llvm-mc.cpp - Machine Code Hacking Driver -------------------------===//
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

#include "llvm/MC/MCParser/AsmLexer.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/system_error.h"
#include "Disassembler.h"
using namespace llvm;

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Output filename"),
               cl::value_desc("filename"));

static cl::opt<bool>
ShowEncoding("show-encoding", cl::desc("Show instruction encodings"));

static cl::opt<bool>
ShowInst("show-inst", cl::desc("Show internal instruction representation"));

static cl::opt<bool>
ShowInstOperands("show-inst-operands",
                 cl::desc("Show instructions operands as parsed"));

static cl::opt<unsigned>
OutputAsmVariant("output-asm-variant",
                 cl::desc("Syntax variant to use for output printing"));

static cl::opt<bool>
RelaxAll("mc-relax-all", cl::desc("Relax all fixups"));

static cl::opt<bool>
NoExecStack("mc-no-exec-stack", cl::desc("File doesn't need an exec stack"));

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

static cl::opt<Reloc::Model>
RelocModel("relocation-model",
             cl::desc("Choose relocation model"),
             cl::init(Reloc::Default),
             cl::values(
            clEnumValN(Reloc::Default, "default",
                       "Target default relocation model"),
            clEnumValN(Reloc::Static, "static",
                       "Non-relocatable code"),
            clEnumValN(Reloc::PIC_, "pic",
                       "Fully relocatable, position independent code"),
            clEnumValN(Reloc::DynamicNoPIC, "dynamic-no-pic",
                       "Relocatable external references, non-relocatable code"),
            clEnumValEnd));

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
SaveTempLabels("L", cl::desc("Don't discard temporary labels"));

static cl::opt<bool>
GenDwarfForAssembly("g", cl::desc("Generate dwarf debugging info for assembly "
                                  "source files"));

enum ActionType {
  AC_AsLex,
  AC_Assemble,
  AC_Disassemble,
  AC_EDisassemble,
  AC_MDisassemble
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
                  clEnumValN(AC_EDisassemble, "edis",
                             "Enhanced disassembly of strings of hex bytes"),
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
    return 0;
  }

  // Update the triple name and return the found target.
  TripleName = TheTriple.getTriple();
  return TheTarget;
}

static tool_output_file *GetOutputStream() {
  if (OutputFilename == "")
    OutputFilename = "-";

  std::string Err;
  tool_output_file *Out = new tool_output_file(OutputFilename.c_str(), Err,
                                               raw_fd_ostream::F_Binary);
  if (!Err.empty()) {
    errs() << Err << '\n';
    delete Out;
    return 0;
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

static int AsLexInput(SourceMgr &SrcMgr, MCAsmInfo &MAI, tool_output_file *Out) {

  AsmLexer Lexer(MAI);
  Lexer.setBuffer(SrcMgr.getMemoryBuffer(0));

  bool Error = false;
  while (Lexer.Lex().isNot(AsmToken::Eof)) {
    AsmToken Tok = Lexer.getTok();

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
      Out->os() << "identifier: " << Lexer.getTok().getString();
      break;
    case AsmToken::Integer:
      Out->os() << "int: " << Lexer.getTok().getString();
      break;
    case AsmToken::Real:
      Out->os() << "real: " << Lexer.getTok().getString();
      break;
    case AsmToken::Register:
      Out->os() << "register: " << Lexer.getTok().getRegVal();
      break;
    case AsmToken::String:
      Out->os() << "string: " << Lexer.getTok().getString();
      break;

    case AsmToken::Amp:            Out->os() << "Amp"; break;
    case AsmToken::AmpAmp:         Out->os() << "AmpAmp"; break;
    case AsmToken::At:             Out->os() << "At"; break;
    case AsmToken::Caret:          Out->os() << "Caret"; break;
    case AsmToken::Colon:          Out->os() << "Colon"; break;
    case AsmToken::Comma:          Out->os() << "Comma"; break;
    case AsmToken::Dollar:         Out->os() << "Dollar"; break;
    case AsmToken::Dot:            Out->os() << "Dot"; break;
    case AsmToken::EndOfStatement: Out->os() << "EndOfStatement"; break;
    case AsmToken::Eof:            Out->os() << "Eof"; break;
    case AsmToken::Equal:          Out->os() << "Equal"; break;
    case AsmToken::EqualEqual:     Out->os() << "EqualEqual"; break;
    case AsmToken::Exclaim:        Out->os() << "Exclaim"; break;
    case AsmToken::ExclaimEqual:   Out->os() << "ExclaimEqual"; break;
    case AsmToken::Greater:        Out->os() << "Greater"; break;
    case AsmToken::GreaterEqual:   Out->os() << "GreaterEqual"; break;
    case AsmToken::GreaterGreater: Out->os() << "GreaterGreater"; break;
    case AsmToken::Hash:           Out->os() << "Hash"; break;
    case AsmToken::LBrac:          Out->os() << "LBrac"; break;
    case AsmToken::LCurly:         Out->os() << "LCurly"; break;
    case AsmToken::LParen:         Out->os() << "LParen"; break;
    case AsmToken::Less:           Out->os() << "Less"; break;
    case AsmToken::LessEqual:      Out->os() << "LessEqual"; break;
    case AsmToken::LessGreater:    Out->os() << "LessGreater"; break;
    case AsmToken::LessLess:       Out->os() << "LessLess"; break;
    case AsmToken::Minus:          Out->os() << "Minus"; break;
    case AsmToken::Percent:        Out->os() << "Percent"; break;
    case AsmToken::Pipe:           Out->os() << "Pipe"; break;
    case AsmToken::PipePipe:       Out->os() << "PipePipe"; break;
    case AsmToken::Plus:           Out->os() << "Plus"; break;
    case AsmToken::RBrac:          Out->os() << "RBrac"; break;
    case AsmToken::RCurly:         Out->os() << "RCurly"; break;
    case AsmToken::RParen:         Out->os() << "RParen"; break;
    case AsmToken::Slash:          Out->os() << "Slash"; break;
    case AsmToken::Star:           Out->os() << "Star"; break;
    case AsmToken::Tilde:          Out->os() << "Tilde"; break;
    }

    // Print the token string.
    Out->os() << " (\"";
    Out->os().write_escaped(Tok.getString());
    Out->os() << "\")\n";
  }

  return Error;
}

static int AssembleInput(const char *ProgName, const Target *TheTarget, 
                         SourceMgr &SrcMgr, MCContext &Ctx, MCStreamer &Str,
                         MCAsmInfo &MAI, MCSubtargetInfo &STI) {
  OwningPtr<MCAsmParser> Parser(createMCAsmParser(SrcMgr, Ctx,
                                                  Str, MAI));
  OwningPtr<MCTargetAsmParser> TAP(TheTarget->createMCAsmParser(STI, *Parser));
  if (!TAP) {
    errs() << ProgName
           << ": error: this target does not support assembly parsing.\n";
    return 1;
  }

  Parser->setShowParsedOperands(ShowInstOperands);
  Parser->setTargetParser(*TAP.get());

  int Res = Parser->Run(NoInitialTextSection);

  return Res;
}

int main(int argc, char **argv) {
  // Print a stack trace if we signal out.
  sys::PrintStackTraceOnErrorSignal();
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
  TripleName = Triple::normalize(TripleName);
  setDwarfDebugFlags(argc, argv);

  const char *ProgName = argv[0];
  const Target *TheTarget = GetTarget(ProgName);
  if (!TheTarget)
    return 1;

  OwningPtr<MemoryBuffer> BufferPtr;
  if (error_code ec = MemoryBuffer::getFileOrSTDIN(InputFilename, BufferPtr)) {
    errs() << ProgName << ": " << ec.message() << '\n';
    return 1;
  }
  MemoryBuffer *Buffer = BufferPtr.take();

  SourceMgr SrcMgr;

  // Tell SrcMgr about this buffer, which is what the parser will pick up.
  SrcMgr.AddNewSourceBuffer(Buffer, SMLoc());

  // Record the location of the include directories so that the lexer can find
  // it later.
  SrcMgr.setIncludeDirs(IncludeDirs);


  llvm::OwningPtr<MCAsmInfo> MAI(TheTarget->createMCAsmInfo(TripleName));
  assert(MAI && "Unable to create target asm info!");

  llvm::OwningPtr<MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TripleName));
  assert(MRI && "Unable to create target register info!");

  // FIXME: This is not pretty. MCContext has a ptr to MCObjectFileInfo and
  // MCObjectFileInfo needs a MCContext reference in order to initialize itself.
  OwningPtr<MCObjectFileInfo> MOFI(new MCObjectFileInfo());
  MCContext Ctx(*MAI, *MRI, MOFI.get(), &SrcMgr);
  MOFI->InitMCObjectFileInfo(TripleName, RelocModel, CMModel, Ctx);

  if (SaveTempLabels)
    Ctx.setAllowTemporaryLabels(false);

  Ctx.setGenDwarfForAssembly(GenDwarfForAssembly);
  if (!DwarfDebugFlags.empty()) 
    Ctx.setDwarfDebugFlags(StringRef(DwarfDebugFlags));

  // Package up features to be passed to target/subtarget
  std::string FeaturesStr;
  if (MAttrs.size()) {
    SubtargetFeatures Features;
    for (unsigned i = 0; i != MAttrs.size(); ++i)
      Features.AddFeature(MAttrs[i]);
    FeaturesStr = Features.getString();
  }

  OwningPtr<tool_output_file> Out(GetOutputStream());
  if (!Out)
    return 1;

  formatted_raw_ostream FOS(Out->os());
  OwningPtr<MCStreamer> Str;

  OwningPtr<MCInstrInfo> MCII(TheTarget->createMCInstrInfo());
  OwningPtr<MCSubtargetInfo>
    STI(TheTarget->createMCSubtargetInfo(TripleName, MCPU, FeaturesStr));

  MCInstPrinter *IP;
  if (FileType == OFT_AssemblyFile) {
    IP =
      TheTarget->createMCInstPrinter(OutputAsmVariant, *MAI, *MCII, *MRI, *STI);
    MCCodeEmitter *CE = 0;
    MCAsmBackend *MAB = 0;
    if (ShowEncoding) {
      CE = TheTarget->createMCCodeEmitter(*MCII, *MRI, *STI, Ctx);
      MAB = TheTarget->createMCAsmBackend(TripleName, MCPU);
    }
    Str.reset(TheTarget->createAsmStreamer(Ctx, FOS, /*asmverbose*/true,
                                           /*useLoc*/ true,
                                           /*useCFI*/ true,
                                           /*useDwarfDirectory*/ true,
                                           IP, CE, MAB, ShowInst));

  } else if (FileType == OFT_Null) {
    Str.reset(createNullStreamer(Ctx));
  } else {
    assert(FileType == OFT_ObjectFile && "Invalid file type!");
    MCCodeEmitter *CE = TheTarget->createMCCodeEmitter(*MCII, *MRI, *STI, Ctx);
    MCAsmBackend *MAB = TheTarget->createMCAsmBackend(TripleName, MCPU);
    Str.reset(TheTarget->createMCObjectStreamer(TripleName, Ctx, *MAB,
                                                FOS, CE, RelaxAll,
                                                NoExecStack));
  }

  int Res = 1;
  switch (Action) {
  case AC_AsLex:
    Res = AsLexInput(SrcMgr, *MAI, Out.get());
    break;
  case AC_Assemble:
    Res = AssembleInput(ProgName, TheTarget, SrcMgr, Ctx, *Str, *MAI, *STI);
    break;
  case AC_MDisassemble:
    IP->setUseMarkup(1);
    // Fall through to do disassembly.
  case AC_Disassemble:
    Res = Disassembler::disassemble(*TheTarget, TripleName, *STI, *Str,
                                    *Buffer, SrcMgr, Out->os());
    break;
  case AC_EDisassemble:
    Res =  Disassembler::disassembleEnhanced(TripleName, *Buffer, SrcMgr, Out->os());
    break;
  }

  // Keep output if no errors.
  if (Res == 0) Out->keep();
  return Res;
}
