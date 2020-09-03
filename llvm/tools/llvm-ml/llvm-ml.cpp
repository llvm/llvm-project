//===-- llvm-ml.cpp - masm-compatible assembler -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A simple driver around MasmParser; based on llvm-mc.
//
//===----------------------------------------------------------------------===//

#include "Disassembler.h"

#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/AsmLexer.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptionsCommandFlags.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"

using namespace llvm;

static mc::RegisterMCTargetOptionsFlags MOF;

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Output filename"), cl::value_desc("filename"),
               cl::init("-"));

static cl::opt<bool>
ShowEncoding("show-encoding", cl::desc("Show instruction encodings"));

static cl::opt<bool>
ShowInst("show-inst", cl::desc("Show internal instruction representation"));

static cl::opt<bool>
ShowInstOperands("show-inst-operands",
                 cl::desc("Show instructions operands as parsed"));

static cl::opt<bool>
OutputATTAsm("output-att-asm", cl::desc("Use ATT syntax for output printing"));

static cl::opt<bool>
PrintImmHex("print-imm-hex", cl::init(false),
            cl::desc("Prefer hex format for immediate values"));

static cl::opt<bool>
PreserveComments("preserve-comments",
                 cl::desc("Preserve Comments in outputted assembly"));

enum OutputFileType {
  OFT_Null,
  OFT_AssemblyFile,
  OFT_ObjectFile
};
static cl::opt<OutputFileType>
FileType("filetype", cl::init(OFT_ObjectFile),
  cl::desc("Choose an output file type:"),
  cl::values(
       clEnumValN(OFT_AssemblyFile, "asm",
                  "Emit an assembly ('.s') file"),
       clEnumValN(OFT_Null, "null",
                  "Don't emit anything (for timing purposes)"),
       clEnumValN(OFT_ObjectFile, "obj",
                  "Emit a native object ('.o') file")));

static cl::list<std::string>
IncludeDirs("I", cl::desc("Directory of include files"),
            cl::value_desc("directory"), cl::Prefix);

enum BitnessType {
  m32,
  m64,
};
cl::opt<BitnessType> Bitness(cl::desc("Choose bitness:"), cl::init(m64),
                             cl::values(clEnumVal(m32, "32-bit"),
                                        clEnumVal(m64, "64-bit (default)")));

static cl::opt<std::string>
TripleName("triple", cl::desc("Target triple to assemble for, "
                              "see -version for available targets"));

static cl::opt<std::string>
DebugCompilationDir("fdebug-compilation-dir",
                    cl::desc("Specifies the debug info's compilation dir"));

static cl::list<std::string>
DebugPrefixMap("fdebug-prefix-map",
               cl::desc("Map file source paths in debug info"),
               cl::value_desc("= separated key-value pairs"));

static cl::opt<std::string>
MainFileName("main-file-name",
             cl::desc("Specifies the name we should consider the input file"));

static cl::opt<bool> SaveTempLabels("save-temp-labels",
                                    cl::desc("Don't discard temporary labels"));

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
                             "Lex tokens from a .asm file"),
                  clEnumValN(AC_Assemble, "assemble",
                             "Assemble a .asm file (default)"),
                  clEnumValN(AC_Disassemble, "disassemble",
                             "Disassemble strings of hex bytes"),
                  clEnumValN(AC_MDisassemble, "mdis",
                             "Marked up disassembly of strings of hex bytes")));

static const Target *GetTarget(const char *ProgName) {
  // Figure out the target triple.
  if (TripleName.empty()) {
    if (Bitness == m32)
      TripleName = "i386-pc-windows";
    else if (Bitness == m64)
      TripleName = "x86_64-pc-windows";
  }
  Triple TheTriple(Triple::normalize(TripleName));

  // Get the target specific parser.
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget("", TheTriple, Error);
  if (!TheTarget) {
    WithColor::error(errs(), ProgName) << Error;
    return nullptr;
  }

  // Update the triple name and return the found target.
  TripleName = TheTriple.getTriple();
  return TheTarget;
}

static std::unique_ptr<ToolOutputFile> GetOutputStream(StringRef Path) {
  std::error_code EC;
  auto Out = std::make_unique<ToolOutputFile>(Path, EC, sys::fs::F_None);
  if (EC) {
    WithColor::error() << EC.message() << '\n';
    return nullptr;
  }

  return Out;
}

static int AsLexInput(SourceMgr &SrcMgr, MCAsmInfo &MAI, raw_ostream &OS) {
  AsmLexer Lexer(MAI);
  Lexer.setBuffer(SrcMgr.getMemoryBuffer(SrcMgr.getMainFileID())->getBuffer());
  Lexer.setLexMasmIntegers(true);

  bool Error = false;
  while (Lexer.Lex().isNot(AsmToken::Eof)) {
    Lexer.getTok().dump(OS);
    OS << "\n";
    if (Lexer.getTok().getKind() == AsmToken::Error)
      Error = true;
  }

  return Error;
}

static int AssembleInput(const char *ProgName, const Target *TheTarget,
                         SourceMgr &SrcMgr, MCContext &Ctx, MCStreamer &Str,
                         MCAsmInfo &MAI, MCSubtargetInfo &STI,
                         MCInstrInfo &MCII, MCTargetOptions &MCOptions) {
  std::unique_ptr<MCAsmParser> Parser(
      createMCMasmParser(SrcMgr, Ctx, Str, MAI));
  std::unique_ptr<MCTargetAsmParser> TAP(
      TheTarget->createMCAsmParser(STI, *Parser, MCII, MCOptions));

  if (!TAP) {
    WithColor::error(errs(), ProgName)
        << "this target does not support assembly parsing.\n";
    return 1;
  }

  Parser->setShowParsedOperands(ShowInstOperands);
  Parser->setTargetParser(*TAP);
  Parser->getLexer().setLexMasmIntegers(true);

  int Res = Parser->Run(/*NoInitialTextSection=*/true);

  return Res;
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  // Initialize targets and assembly printers/parsers.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllDisassemblers();

  // Register the target printer for --version.
  cl::AddExtraVersionPrinter(TargetRegistry::printRegisteredTargetsForVersion);

  cl::ParseCommandLineOptions(argc, argv, "llvm machine code playground\n");
  MCTargetOptions MCOptions = mc::InitMCTargetOptionsFromFlags();
  MCOptions.AssemblyLanguage = "masm";

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
    WithColor::error(errs(), ProgName)
        << InputFilename << ": " << EC.message() << '\n';
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

  std::unique_ptr<MCAsmInfo> MAI(
      TheTarget->createMCAsmInfo(*MRI, TripleName, MCOptions));
  assert(MAI && "Unable to create target asm info!");

  MAI->setPreserveAsmComments(PreserveComments);

  // FIXME: This is not pretty. MCContext has a ptr to MCObjectFileInfo and
  // MCObjectFileInfo needs a MCContext reference in order to initialize itself.
  MCObjectFileInfo MOFI;
  MCContext Ctx(MAI.get(), MRI.get(), &MOFI, &SrcMgr);
  MOFI.InitMCObjectFileInfo(TheTriple, /*PIC=*/false, Ctx,
                            /*LargeCodeModel=*/true);

  if (SaveTempLabels)
    Ctx.setAllowTemporaryLabels(false);

  if (!DebugCompilationDir.empty()) {
    Ctx.setCompilationDir(DebugCompilationDir);
  } else {
    // If no compilation dir is set, try to use the current directory.
    SmallString<128> CWD;
    if (!sys::fs::current_path(CWD))
      Ctx.setCompilationDir(CWD);
  }
  for (const auto &Arg : DebugPrefixMap) {
    const auto &KV = StringRef(Arg).split('=');
    Ctx.addDebugPrefixMapEntry(std::string(KV.first), std::string(KV.second));
  }
  if (!MainFileName.empty())
    Ctx.setMainFileName(MainFileName);

  std::unique_ptr<ToolOutputFile> Out = GetOutputStream(OutputFilename);
  if (!Out)
    return 1;

  std::unique_ptr<buffer_ostream> BOS;
  raw_pwrite_stream *OS = &Out->os();
  std::unique_ptr<MCStreamer> Str;

  std::unique_ptr<MCInstrInfo> MCII(TheTarget->createMCInstrInfo());
  std::unique_ptr<MCSubtargetInfo> STI(TheTarget->createMCSubtargetInfo(
      TripleName, /*CPU=*/"", /*Features=*/""));

  MCInstPrinter *IP = nullptr;
  if (FileType == OFT_AssemblyFile) {
    const unsigned OutputAsmVariant = OutputATTAsm ? 0U   // ATT dialect
                                                   : 1U;  // Intel dialect
    IP = TheTarget->createMCInstPrinter(Triple(TripleName), OutputAsmVariant,
                                        *MAI, *MCII, *MRI);

    if (!IP) {
      WithColor::error()
          << "unable to create instruction printer for target triple '"
          << TheTriple.normalize() << "' with "
          << (OutputATTAsm ? "ATT" : "Intel") << " assembly variant.\n";
      return 1;
    }

    // Set the display preference for hex vs. decimal immediates.
    IP->setPrintImmHex(PrintImmHex);

    // Set up the AsmStreamer.
    std::unique_ptr<MCCodeEmitter> CE;
    if (ShowEncoding)
      CE.reset(TheTarget->createMCCodeEmitter(*MCII, *MRI, Ctx));

    std::unique_ptr<MCAsmBackend> MAB(
        TheTarget->createMCAsmBackend(*STI, *MRI, MCOptions));
    auto FOut = std::make_unique<formatted_raw_ostream>(*OS);
    Str.reset(
        TheTarget->createAsmStreamer(Ctx, std::move(FOut), /*asmverbose*/ true,
                                     /*useDwarfDirectory*/ true, IP,
                                     std::move(CE), std::move(MAB), ShowInst));

  } else if (FileType == OFT_Null) {
    Str.reset(TheTarget->createNullStreamer(Ctx));
  } else {
    assert(FileType == OFT_ObjectFile && "Invalid file type!");

    if (!Out->os().supportsSeeking()) {
      BOS = std::make_unique<buffer_ostream>(Out->os());
      OS = BOS.get();
    }

    MCCodeEmitter *CE = TheTarget->createMCCodeEmitter(*MCII, *MRI, Ctx);
    MCAsmBackend *MAB = TheTarget->createMCAsmBackend(*STI, *MRI, MCOptions);
    Str.reset(TheTarget->createMCObjectStreamer(
        TheTriple, Ctx, std::unique_ptr<MCAsmBackend>(MAB),
        MAB->createObjectWriter(*OS), std::unique_ptr<MCCodeEmitter>(CE), *STI,
        MCOptions.MCRelaxAll, MCOptions.MCIncrementalLinkerCompatible,
        /*DWARFMustBeAtTheEnd*/ false));
  }

  // Use Assembler information for parsing.
  Str->setUseAssemblerInfoForParsing(true);

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
    Res = Disassembler::disassemble(*TheTarget, TripleName, *STI, *Str, *Buffer,
                                    SrcMgr, Out->os());

  // Keep output if no errors.
  if (Res == 0)
    Out->keep();
  return Res;
}
