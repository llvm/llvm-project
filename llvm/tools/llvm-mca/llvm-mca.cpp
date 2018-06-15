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
// The cpu defaults to the 'native' host cpu.
// The output defaults to standard output.
//
//===----------------------------------------------------------------------===//

#include "BackendPrinter.h"
#include "CodeRegion.h"
#include "DispatchStatistics.h"
#include "FetchStage.h"
#include "InstructionInfoView.h"
#include "InstructionTables.h"
#include "RegisterFileStatistics.h"
#include "ResourcePressureView.h"
#include "RetireControlUnitStatistics.h"
#include "SchedulerStatistics.h"
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
#include "llvm/Support/Host.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/WithColor.h"

using namespace llvm;

static cl::OptionCategory ToolOptions("Tool Options");
static cl::OptionCategory ViewOptions("View Options");

static cl::opt<std::string> InputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::cat(ToolOptions), cl::init("-"));

static cl::opt<std::string> OutputFilename("o", cl::desc("Output filename"),
                                           cl::init("-"), cl::cat(ToolOptions),
                                           cl::value_desc("filename"));

static cl::opt<std::string>
    ArchName("march",
             cl::desc("Target arch to assemble for, "
                      "see -version for available targets"),
             cl::cat(ToolOptions));

static cl::opt<std::string>
    TripleName("mtriple",
               cl::desc("Target triple to assemble for, "
                        "see -version for available targets"),
               cl::cat(ToolOptions));

static cl::opt<std::string>
    MCPU("mcpu",
         cl::desc("Target a specific cpu type (-mcpu=help for details)"),
         cl::value_desc("cpu-name"), cl::cat(ToolOptions), cl::init("native"));

static cl::opt<int>
    OutputAsmVariant("output-asm-variant",
                     cl::desc("Syntax variant to use for output printing"),
                     cl::cat(ToolOptions), cl::init(-1));

static cl::opt<unsigned> Iterations("iterations",
                                    cl::desc("Number of iterations to run"),
                                    cl::cat(ToolOptions), cl::init(0));

static cl::opt<unsigned>
    DispatchWidth("dispatch", cl::desc("Override the processor dispatch width"),
                  cl::cat(ToolOptions), cl::init(0));

static cl::opt<unsigned>
    RegisterFileSize("register-file-size",
                     cl::desc("Maximum number of temporary registers which can "
                              "be used for register mappings"),
                     cl::cat(ToolOptions), cl::init(0));

static cl::opt<bool>
    PrintRegisterFileStats("register-file-stats",
                           cl::desc("Print register file statistics"),
                           cl::cat(ViewOptions), cl::init(false));

static cl::opt<bool> PrintDispatchStats("dispatch-stats",
                                        cl::desc("Print dispatch statistics"),
                                        cl::cat(ViewOptions), cl::init(false));

static cl::opt<bool>
    PrintSummaryView("summary-view", cl::Hidden,
                     cl::desc("Print summary view (enabled by default)"),
                     cl::cat(ViewOptions), cl::init(true));

static cl::opt<bool> PrintSchedulerStats("scheduler-stats",
                                         cl::desc("Print scheduler statistics"),
                                         cl::cat(ViewOptions), cl::init(false));

static cl::opt<bool>
    PrintRetireStats("retire-stats",
                     cl::desc("Print retire control unit statistics"),
                     cl::cat(ViewOptions), cl::init(false));

static cl::opt<bool> PrintResourcePressureView(
    "resource-pressure",
    cl::desc("Print the resource pressure view (enabled by default)"),
    cl::cat(ViewOptions), cl::init(true));

static cl::opt<bool> PrintTimelineView("timeline",
                                       cl::desc("Print the timeline view"),
                                       cl::cat(ViewOptions), cl::init(false));

static cl::opt<unsigned> TimelineMaxIterations(
    "timeline-max-iterations",
    cl::desc("Maximum number of iterations to print in timeline view"),
    cl::cat(ViewOptions), cl::init(0));

static cl::opt<unsigned> TimelineMaxCycles(
    "timeline-max-cycles",
    cl::desc(
        "Maximum number of cycles in the timeline view. Defaults to 80 cycles"),
    cl::cat(ViewOptions), cl::init(80));

static cl::opt<bool>
    AssumeNoAlias("noalias",
                  cl::desc("If set, assume that loads and stores do not alias"),
                  cl::cat(ToolOptions), cl::init(true));

static cl::opt<unsigned>
    LoadQueueSize("lqueue",
                  cl::desc("Size of the load queue (unbound by default)"),
                  cl::cat(ToolOptions), cl::init(0));

static cl::opt<unsigned>
    StoreQueueSize("squeue",
                   cl::desc("Size of the store queue (unbound by default)"),
                   cl::cat(ToolOptions), cl::init(0));

static cl::opt<bool>
    PrintInstructionTables("instruction-tables",
                           cl::desc("Print instruction tables"),
                           cl::cat(ToolOptions), cl::init(false));

static cl::opt<bool> PrintInstructionInfoView(
    "instruction-info",
    cl::desc("Print the instruction info view (enabled by default)"),
    cl::cat(ViewOptions), cl::init(true));

static cl::opt<bool> EnableAllStats("all-stats",
                                    cl::desc("Print all hardware statistics"),
                                    cl::cat(ViewOptions), cl::init(false));

static cl::opt<bool>
    EnableAllViews("all-views",
                   cl::desc("Print all views including hardware statistics"),
                   cl::cat(ViewOptions), cl::init(false));

namespace {

const Target *getTarget(const char *ProgName) {
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

// A comment consumer that parses strings.
// The only valid tokens are strings.
class MCACommentConsumer : public AsmCommentConsumer {
public:
  mca::CodeRegions &Regions;

  MCACommentConsumer(mca::CodeRegions &R) : Regions(R) {}
  void HandleComment(SMLoc Loc, StringRef CommentText) override {
    // Skip empty comments.
    StringRef Comment(CommentText);
    if (Comment.empty())
      return;

    // Skip spaces and tabs
    unsigned Position = Comment.find_first_not_of(" \t");
    if (Position >= Comment.size())
      // we reached the end of the comment. Bail out.
      return;

    Comment = Comment.drop_front(Position);
    if (Comment.consume_front("LLVM-MCA-END")) {
      Regions.endRegion(Loc);
      return;
    }

    // Now try to parse string LLVM-MCA-BEGIN
    if (!Comment.consume_front("LLVM-MCA-BEGIN"))
      return;

    // Skip spaces and tabs
    Position = Comment.find_first_not_of(" \t");
    if (Position < Comment.size())
      Comment = Comment.drop_front(Position);
    // Use the rest of the string as a descriptor for this code snippet.
    Regions.beginRegion(Comment, Loc);
  }
};

int AssembleInput(const char *ProgName, MCAsmParser &Parser,
                  const Target *TheTarget, MCSubtargetInfo &STI,
                  MCInstrInfo &MCII, MCTargetOptions &MCOptions) {
  std::unique_ptr<MCTargetAsmParser> TAP(
      TheTarget->createMCAsmParser(STI, Parser, MCII, MCOptions));

  if (!TAP) {
    WithColor::error() << "this target does not support assembly parsing.\n";
    return 1;
  }

  Parser.setTargetParser(*TAP);
  return Parser.Run(false);
}

ErrorOr<std::unique_ptr<ToolOutputFile>> getOutputStream() {
  if (OutputFilename == "")
    OutputFilename = "-";
  std::error_code EC;
  auto Out =
      llvm::make_unique<ToolOutputFile>(OutputFilename, EC, sys::fs::F_None);
  if (!EC)
    return std::move(Out);
  return EC;
}

class MCStreamerWrapper final : public MCStreamer {
  mca::CodeRegions &Regions;

public:
  MCStreamerWrapper(MCContext &Context, mca::CodeRegions &R)
      : MCStreamer(Context), Regions(R) {}

  // We only want to intercept the emission of new instructions.
  virtual void EmitInstruction(const MCInst &Inst, const MCSubtargetInfo &STI,
                               bool /* unused */) override {
    Regions.addInstruction(llvm::make_unique<const MCInst>(Inst));
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

  const std::vector<std::unique_ptr<const MCInst>> &
  GetInstructionSequence(unsigned Index) const {
    return Regions.getInstructionSequence(Index);
  }
};
} // end of anonymous namespace

static void processOptionImpl(cl::opt<bool> &O, const cl::opt<bool> &Default) {
  if (!O.getNumOccurrences() || O.getPosition() < Default.getPosition())
    O = Default.getValue();
}

static void processViewOptions() {
  if (!EnableAllViews.getNumOccurrences() &&
      !EnableAllStats.getNumOccurrences())
    return;

  if (EnableAllViews.getNumOccurrences()) {
    processOptionImpl(PrintSummaryView, EnableAllViews);
    processOptionImpl(PrintResourcePressureView, EnableAllViews);
    processOptionImpl(PrintTimelineView, EnableAllViews);
    processOptionImpl(PrintInstructionInfoView, EnableAllViews);
  }

  const cl::opt<bool> &Default =
      EnableAllViews.getPosition() < EnableAllStats.getPosition()
          ? EnableAllStats
          : EnableAllViews;
  processOptionImpl(PrintSummaryView, Default);
  processOptionImpl(PrintRegisterFileStats, Default);
  processOptionImpl(PrintDispatchStats, Default);
  processOptionImpl(PrintSchedulerStats, Default);
  processOptionImpl(PrintRetireStats, Default);
}

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);

  // Initialize targets and assembly parsers.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();

  // Enable printing of available targets when flag --version is specified.
  cl::AddExtraVersionPrinter(TargetRegistry::printRegisteredTargetsForVersion);

  cl::HideUnrelatedOptions({&ToolOptions, &ViewOptions});

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
    WithColor::error() << InputFilename << ": " << EC.message() << '\n';
    return 1;
  }

  // Apply overrides to llvm-mca specific options.
  processViewOptions();

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

  mca::CodeRegions Regions(SrcMgr);
  MCStreamerWrapper Str(Ctx, Regions);

  std::unique_ptr<MCInstrInfo> MCII(TheTarget->createMCInstrInfo());

  if (!MCPU.compare("native"))
    MCPU = llvm::sys::getHostCPUName();

  std::unique_ptr<MCSubtargetInfo> STI(
      TheTarget->createMCSubtargetInfo(TripleName, MCPU, /* FeaturesStr */ ""));
  if (!STI->isCPUStringValid(MCPU))
    return 1;

  if (!PrintInstructionTables && !STI->getSchedModel().isOutOfOrder()) {
    WithColor::error() << "please specify an out-of-order cpu. '" << MCPU
                       << "' is an in-order cpu.\n";
    return 1;
  }

  if (!STI->getSchedModel().hasInstrSchedModel()) {
    WithColor::error()
        << "unable to find instruction-level scheduling information for"
        << " target triple '" << TheTriple.normalize() << "' and cpu '" << MCPU
        << "'.\n";

    if (STI->getSchedModel().InstrItineraries)
      WithColor::note()
          << "cpu '" << MCPU << "' provides itineraries. However, "
          << "instruction itineraries are currently unsupported.\n";
    return 1;
  }

  std::unique_ptr<MCAsmParser> P(createMCAsmParser(SrcMgr, Ctx, Str, *MAI));
  MCAsmLexer &Lexer = P->getLexer();
  MCACommentConsumer CC(Regions);
  Lexer.setCommentConsumer(&CC);

  if (AssembleInput(ProgName, *P, TheTarget, *STI, *MCII, MCOptions))
    return 1;

  if (Regions.empty()) {
    WithColor::error() << "no assembly instructions found.\n";
    return 1;
  }

  // Now initialize the output file.
  auto OF = getOutputStream();
  if (std::error_code EC = OF.getError()) {
    WithColor::error() << EC.message() << '\n';
    return 1;
  }

  unsigned AssemblerDialect = P->getAssemblerDialect();
  if (OutputAsmVariant >= 0)
    AssemblerDialect = static_cast<unsigned>(OutputAsmVariant);
  std::unique_ptr<MCInstPrinter> IP(TheTarget->createMCInstPrinter(
      Triple(TripleName), AssemblerDialect, *MAI, *MCII, *MRI));
  if (!IP) {
    WithColor::error()
        << "unable to create instruction printer for target triple '"
        << TheTriple.normalize() << "' with assembly variant "
        << AssemblerDialect << ".\n";
    return 1;
  }

  std::unique_ptr<llvm::ToolOutputFile> TOF = std::move(*OF);

  const MCSchedModel &SM = STI->getSchedModel();

  unsigned Width = SM.IssueWidth;
  if (DispatchWidth)
    Width = DispatchWidth;

  // Create an instruction builder.
  mca::InstrBuilder IB(*STI, *MCII);

  // Number each region in the sequence.
  unsigned RegionIdx = 0;
  for (const std::unique_ptr<mca::CodeRegion> &Region : Regions) {
    // Skip empty code regions.
    if (Region->empty())
      continue;

    // Don't print the header of this region if it is the default region, and
    // it doesn't have an end location.
    if (Region->startLoc().isValid() || Region->endLoc().isValid()) {
      TOF->os() << "\n[" << RegionIdx++ << "] Code Region";
      StringRef Desc = Region->getDescription();
      if (!Desc.empty())
        TOF->os() << " - " << Desc;
      TOF->os() << "\n\n";
    }

    mca::SourceMgr S(Region->getInstructions(),
                     PrintInstructionTables ? 1 : Iterations);

    if (PrintInstructionTables) {
      mca::InstructionTables IT(STI->getSchedModel(), IB, S);

      if (PrintInstructionInfoView) {
        IT.addView(
            llvm::make_unique<mca::InstructionInfoView>(*STI, *MCII, S, *IP));
      }

      IT.addView(llvm::make_unique<mca::ResourcePressureView>(*STI, *IP, S));
      IT.run();
      IT.printReport(TOF->os());
      continue;
    }

    // Ideally, I'd like to expose the pipeline building here,
    // by registering all of the Stage instances.
    // But for now, it's just this single puppy.
    std::unique_ptr<mca::FetchStage> Fetch =
        llvm::make_unique<mca::FetchStage>(IB, S);
    mca::Backend B(*STI, *MRI, std::move(Fetch), Width, RegisterFileSize,
                   LoadQueueSize, StoreQueueSize, AssumeNoAlias);
    mca::BackendPrinter Printer(B);

    if (PrintSummaryView)
      Printer.addView(llvm::make_unique<mca::SummaryView>(SM, S, Width));

    if (PrintInstructionInfoView)
      Printer.addView(
          llvm::make_unique<mca::InstructionInfoView>(*STI, *MCII, S, *IP));

    if (PrintDispatchStats)
      Printer.addView(llvm::make_unique<mca::DispatchStatistics>());

    if (PrintSchedulerStats)
      Printer.addView(llvm::make_unique<mca::SchedulerStatistics>(*STI));

    if (PrintRetireStats)
      Printer.addView(llvm::make_unique<mca::RetireControlUnitStatistics>());

    if (PrintRegisterFileStats)
      Printer.addView(llvm::make_unique<mca::RegisterFileStatistics>(*STI));

    if (PrintResourcePressureView)
      Printer.addView(
          llvm::make_unique<mca::ResourcePressureView>(*STI, *IP, S));

    if (PrintTimelineView) {
      Printer.addView(llvm::make_unique<mca::TimelineView>(
          *STI, *IP, S, TimelineMaxIterations, TimelineMaxCycles));
    }

    B.run();
    Printer.printReport(TOF->os());
  }

  TOF->keep();
  return 0;
}
