//===-- llvm-mca.cpp - Machine Code Analyzer -------------------*- C++ -* -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This utility is a simple driver that allows static performance analysis on
// machine code similarly to how IACA (Intel Architecture Code Analyzer) works.
//
//   llvm-mca [options] <file-name>
//      -march <type>
//      -mcpu <cpu>
//      -o <file>
//
// The target defaults to the host target.
// The cpu defaults to 'generic'.
// The output defaults to standard output.
//
//===----------------------------------------------------------------------===//

#include "BackendPrinter.h"
#include "BackendStatistics.h"
#include "InstructionInfoView.h"
#include "InstructionTables.h"
#include "ResourcePressureView.h"
#include "SummaryView.h"
#include "TimelineView.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
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

static cl::opt<std::string> OutputFilename("o", cl::desc("Output filename"),
                                           cl::init("-"),
                                           cl::value_desc("filename"));

static cl::opt<std::string>
    ArchName("march", cl::desc("Target arch to assemble for, "
                               "see -version for available targets"));

static cl::opt<std::string>
    TripleName("mtriple", cl::desc("Target triple to assemble for, "
                                   "see -version for available targets"));

static cl::opt<std::string>
    MCPU("mcpu",
         cl::desc("Target a specific cpu type (-mcpu=help for details)"),
         cl::value_desc("cpu-name"), cl::init("generic"));

static cl::opt<unsigned>
    OutputAsmVariant("output-asm-variant",
                     cl::desc("Syntax variant to use for output printing"));

static cl::opt<unsigned> Iterations("iterations",
                                    cl::desc("Number of iterations to run"),
                                    cl::init(0));

static cl::opt<unsigned> DispatchWidth(
    "dispatch",
    cl::desc("Dispatch Width. By default it is set equal to IssueWidth"),
    cl::init(0));

static cl::opt<unsigned> MaxRetirePerCycle(
    "max-retire-per-cycle",
    cl::desc("Maximum number of instructions that can be retired in one cycle"),
    cl::init(0));

static cl::opt<unsigned>
    RegisterFileSize("register-file-size",
                     cl::desc("Maximum number of temporary registers which can "
                              "be used for register mappings"),
                     cl::init(0));

static cl::opt<bool>
    PrintResourcePressureView("resource-pressure",
                              cl::desc("Print the resource pressure view"),
                              cl::init(true));

static cl::opt<bool> PrintTimelineView("timeline",
                                       cl::desc("Print the timeline view"),
                                       cl::init(false));

static cl::opt<unsigned> TimelineMaxIterations(
    "timeline-max-iterations",
    cl::desc("Maximum number of iterations to print in timeline view"),
    cl::init(0));

static cl::opt<unsigned> TimelineMaxCycles(
    "timeline-max-cycles",
    cl::desc(
        "Maximum number of cycles in the timeline view. Defaults to 80 cycles"),
    cl::init(80));

static cl::opt<bool> PrintModeVerbose("verbose",
                                      cl::desc("Enable verbose output"),
                                      cl::init(false));

static cl::opt<bool> AssumeNoAlias(
    "noalias",
    cl::desc("If set, it assumes that loads and stores do not alias"),
    cl::init(true));

static cl::opt<unsigned>
    LoadQueueSize("lqueue", cl::desc("Size of the load queue"), cl::init(0));

static cl::opt<unsigned>
    StoreQueueSize("squeue", cl::desc("Size of the store queue"), cl::init(0));

static cl::opt<bool>
    PrintInstructionTables("instruction-tables",
                           cl::desc("Print instruction tables"),
                           cl::init(false));

static cl::opt<bool>
    PrintInstructionInfoView("instruction-info",
                             cl::desc("Print the instruction info view"),
                             cl::init(true));

static const Target *getTarget(const char *ProgName) {
  TripleName = Triple::normalize(TripleName);
  if (TripleName.empty())
    TripleName = Triple::normalize(sys::getDefaultTargetTriple());
  Triple TheTriple(TripleName);

  // Get the target specific parser.
  std::string Error;
  const Target *TheTarget =
      TargetRegistry::lookupTarget(ArchName, TheTriple, Error);
  if (!TheTarget) {
    errs() << ProgName << ": " << Error;
    return nullptr;
  }

  // Return the found target.
  return TheTarget;
}

static int AssembleInput(const char *ProgName, const Target *TheTarget,
                         SourceMgr &SrcMgr, MCContext &Ctx, MCStreamer &Str,
                         MCAsmInfo &MAI, MCSubtargetInfo &STI,
                         MCInstrInfo &MCII, MCTargetOptions &MCOptions) {
  std::unique_ptr<MCAsmParser> Parser(createMCAsmParser(SrcMgr, Ctx, Str, MAI));
  std::unique_ptr<MCTargetAsmParser> TAP(
      TheTarget->createMCAsmParser(STI, *Parser, MCII, MCOptions));

  if (!TAP) {
    errs() << ProgName
           << ": error: this target does not support assembly parsing.\n";
    return 1;
  }

  Parser->setTargetParser(*TAP);
  return Parser->Run(false);
}

static ErrorOr<std::unique_ptr<ToolOutputFile>> getOutputStream() {
  if (OutputFilename == "")
    OutputFilename = "-";
  std::error_code EC;
  auto Out =
      llvm::make_unique<ToolOutputFile>(OutputFilename, EC, sys::fs::F_None);
  if (!EC)
    return std::move(Out);
  return EC;
}

namespace {

class MCStreamerWrapper final : public MCStreamer {
  using InstVec = std::vector<std::unique_ptr<const MCInst>>;
  InstVec &Insts;

public:
  MCStreamerWrapper(MCContext &Context, InstVec &Vec)
      : MCStreamer(Context), Insts(Vec) {}

  // We only want to intercept the emission of new instructions.
  virtual void EmitInstruction(const MCInst &Inst, const MCSubtargetInfo &STI,
                               bool /* unused */) override {
    Insts.emplace_back(new MCInst(Inst));
  }

  bool EmitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute) override {
    return true;
  }

  void EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                        unsigned ByteAlignment) override {}
  void EmitZerofill(MCSection *Section, MCSymbol *Symbol = nullptr,
                    uint64_t Size = 0, unsigned ByteAlignment = 0) override {}
  void EmitGPRel32Value(const MCExpr *Value) override {}
  void BeginCOFFSymbolDef(const MCSymbol *Symbol) override {}
  void EmitCOFFSymbolStorageClass(int StorageClass) override {}
  void EmitCOFFSymbolType(int Type) override {}
  void EndCOFFSymbolDef() override {}

  const InstVec &GetInstructionSequence() const { return Insts; }
};
} // end of anonymous namespace

int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);
  llvm_shutdown_obj Y; // Call llvm_shutdown() on exit.

  // Initialize targets and assembly parsers.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();

  // Enable printing of available targets when flag --version is specified.
  cl::AddExtraVersionPrinter(TargetRegistry::printRegisteredTargetsForVersion);

  // Parse flags and initialize target options.
  cl::ParseCommandLineOptions(argc, argv,
                              "llvm machine code performance analyzer.\n");
  MCTargetOptions MCOptions;
  MCOptions.PreserveAsmComments = false;

  // Get the target from the triple. If a triple is not specified, then select
  // the default triple for the host. If the triple doesn't correspond to any
  // registered target, then exit with an error message.
  const char *ProgName = argv[0];
  const Target *TheTarget = getTarget(ProgName);
  if (!TheTarget)
    return 1;

  // GetTarget() may replaced TripleName with a default triple.
  // For safety, reconstruct the Triple object.
  Triple TheTriple(TripleName);

  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferPtr =
      MemoryBuffer::getFileOrSTDIN(InputFilename);
  if (std::error_code EC = BufferPtr.getError()) {
    errs() << InputFilename << ": " << EC.message() << '\n';
    return 1;
  }

  SourceMgr SrcMgr;

  // Tell SrcMgr about this buffer, which is what the parser will pick up.
  SrcMgr.AddNewSourceBuffer(std::move(*BufferPtr), SMLoc());

  std::unique_ptr<MCRegisterInfo> MRI(TheTarget->createMCRegInfo(TripleName));
  assert(MRI && "Unable to create target register info!");

  std::unique_ptr<MCAsmInfo> MAI(TheTarget->createMCAsmInfo(*MRI, TripleName));
  assert(MAI && "Unable to create target asm info!");

  MCObjectFileInfo MOFI;
  MCContext Ctx(MAI.get(), MRI.get(), &MOFI, &SrcMgr);
  MOFI.InitMCObjectFileInfo(TheTriple, /* PIC= */ false, Ctx);

  std::unique_ptr<buffer_ostream> BOS;

  std::unique_ptr<mca::SourceMgr> S = llvm::make_unique<mca::SourceMgr>(
      PrintInstructionTables ? 1 : Iterations);
  MCStreamerWrapper Str(Ctx, S->getSequence());

  std::unique_ptr<MCInstrInfo> MCII(TheTarget->createMCInstrInfo());
  std::unique_ptr<MCSubtargetInfo> STI(
      TheTarget->createMCSubtargetInfo(TripleName, MCPU, /* FeaturesStr */ ""));
  if (!STI->isCPUStringValid(MCPU))
    return 1;

  if (!STI->getSchedModel().isOutOfOrder()) {
    errs() << "error: please specify an out-of-order cpu. '" << MCPU
           << "' is an in-order cpu.\n";
    return 1;
  }

  if (!STI->getSchedModel().hasInstrSchedModel()) {
    errs()
        << "error: unable to find instruction-level scheduling information for"
        << " target triple '" << TheTriple.normalize() << "' and cpu '" << MCPU
        << "'.\n";

    if (STI->getSchedModel().InstrItineraries)
      errs() << "note: cpu '" << MCPU << "' provides itineraries. However, "
             << "instruction itineraries are currently unsupported.\n";
    return 1;
  }

  std::unique_ptr<MCInstPrinter> IP(TheTarget->createMCInstPrinter(
      Triple(TripleName), OutputAsmVariant, *MAI, *MCII, *MRI));
  if (!IP) {
    errs() << "error: unable to create instruction printer for target triple '"
           << TheTriple.normalize() << "' with assembly variant "
           << OutputAsmVariant << ".\n";
    return 1;
  }

  int Res = AssembleInput(ProgName, TheTarget, SrcMgr, Ctx, Str, *MAI, *STI,
                          *MCII, MCOptions);
  if (Res)
    return Res;

  if (S->isEmpty()) {
    errs() << "error: no assembly instructions found.\n";
    return 1;
  }

  // Now initialize the output file.
  auto OF = getOutputStream();
  if (std::error_code EC = OF.getError()) {
    errs() << EC.message() << '\n';
    return 1;
  }

  std::unique_ptr<llvm::ToolOutputFile> TOF = std::move(*OF);

  const MCSchedModel &SM = STI->getSchedModel();

  unsigned Width = SM.IssueWidth;
  if (DispatchWidth)
    Width = DispatchWidth;

  // Create an instruction builder.
  std::unique_ptr<mca::InstrBuilder> IB =
      llvm::make_unique<mca::InstrBuilder>(*STI, *MCII);

  if (PrintInstructionTables) {
    mca::InstructionTables IT(STI->getSchedModel(), *IB, *S);

    if (PrintInstructionInfoView) {
      IT.addView(
          llvm::make_unique<mca::InstructionInfoView>(*STI, *MCII, *S, *IP));
    }

    IT.addView(llvm::make_unique<mca::ResourcePressureView>(*STI, *IP, *S));
    IT.run();
    IT.printReport(TOF->os());
    TOF->keep();
    return 0;
  }

  std::unique_ptr<mca::Backend> B = llvm::make_unique<mca::Backend>(
      *STI, *MRI, *IB, *S, Width, RegisterFileSize, MaxRetirePerCycle,
      LoadQueueSize, StoreQueueSize, AssumeNoAlias);

  std::unique_ptr<mca::BackendPrinter> Printer =
      llvm::make_unique<mca::BackendPrinter>(*B);

  Printer->addView(llvm::make_unique<mca::SummaryView>(*S, Width));

  if (PrintInstructionInfoView)
    Printer->addView(
        llvm::make_unique<mca::InstructionInfoView>(*STI, *MCII, *S, *IP));

  if (PrintModeVerbose)
    Printer->addView(llvm::make_unique<mca::BackendStatistics>(*STI));

  if (PrintResourcePressureView)
    Printer->addView(
        llvm::make_unique<mca::ResourcePressureView>(*STI, *IP, *S));

  if (PrintTimelineView) {
    Printer->addView(llvm::make_unique<mca::TimelineView>(
        *STI, *IP, *S, TimelineMaxIterations, TimelineMaxCycles));
  }

  B->run();
  Printer->printReport(TOF->os());
  TOF->keep();

  return 0;
}
