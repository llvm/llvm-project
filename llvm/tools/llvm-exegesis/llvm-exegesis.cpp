//===-- llvm-exegesis.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Measures execution properties (latencies/uops) of an instruction.
///
//===----------------------------------------------------------------------===//

#include "lib/Analysis.h"
#include "lib/BenchmarkResult.h"
#include "lib/BenchmarkRunner.h"
#include "lib/Clustering.h"
#include "lib/LlvmState.h"
#include "lib/PerfHelper.h"
#include "lib/Target.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include <algorithm>
#include <random>
#include <string>
#include <unordered_map>

static llvm::cl::opt<unsigned>
    OpcodeIndex("opcode-index", llvm::cl::desc("opcode to measure, by index"),
                llvm::cl::init(0));

static llvm::cl::opt<std::string>
    OpcodeName("opcode-name", llvm::cl::desc("opcode to measure, by name"),
               llvm::cl::init(""));

static llvm::cl::opt<std::string>
    SnippetsFile("snippets-file", llvm::cl::desc("code snippets to measure"),
                 llvm::cl::init(""));

static llvm::cl::opt<std::string>
    BenchmarkFile("benchmarks-file", llvm::cl::desc(""), llvm::cl::init(""));

static llvm::cl::opt<exegesis::InstructionBenchmark::ModeE> BenchmarkMode(
    "mode", llvm::cl::desc("the mode to run"),
    llvm::cl::values(clEnumValN(exegesis::InstructionBenchmark::Latency,
                                "latency", "Instruction Latency"),
                     clEnumValN(exegesis::InstructionBenchmark::Uops, "uops",
                                "Uop Decomposition"),
                     // When not asking for a specific benchmark mode, we'll
                     // analyse the results.
                     clEnumValN(exegesis::InstructionBenchmark::Unknown,
                                "analysis", "Analysis")));

static llvm::cl::opt<unsigned>
    NumRepetitions("num-repetitions",
                   llvm::cl::desc("number of time to repeat the asm snippet"),
                   llvm::cl::init(10000));

static llvm::cl::opt<bool> IgnoreInvalidSchedClass(
    "ignore-invalid-sched-class",
    llvm::cl::desc("ignore instructions that do not define a sched class"),
    llvm::cl::init(false));

static llvm::cl::opt<unsigned> AnalysisNumPoints(
    "analysis-numpoints",
    llvm::cl::desc("minimum number of points in an analysis cluster"),
    llvm::cl::init(3));

static llvm::cl::opt<float>
    AnalysisEpsilon("analysis-epsilon",
                    llvm::cl::desc("dbscan epsilon for analysis clustering"),
                    llvm::cl::init(0.1));

static llvm::cl::opt<std::string>
    AnalysisClustersOutputFile("analysis-clusters-output-file",
                               llvm::cl::desc(""), llvm::cl::init("-"));
static llvm::cl::opt<std::string>
    AnalysisInconsistenciesOutputFile("analysis-inconsistencies-output-file",
                                      llvm::cl::desc(""), llvm::cl::init("-"));

namespace exegesis {

static llvm::ExitOnError ExitOnErr;

#ifdef LLVM_EXEGESIS_INITIALIZE_NATIVE_TARGET
void LLVM_EXEGESIS_INITIALIZE_NATIVE_TARGET();
#endif

// Checks that only one of OpcodeName, OpcodeIndex or SnippetsFile is provided,
// and returns the opcode index or 0 if snippets should be read from
// `SnippetsFile`.
static unsigned getOpcodeOrDie(const llvm::MCInstrInfo &MCInstrInfo) {
  const size_t NumSetFlags = (OpcodeName.empty() ? 0 : 1) +
                             (OpcodeIndex == 0 ? 0 : 1) +
                             (SnippetsFile.empty() ? 0 : 1);
  if (NumSetFlags != 1)
    llvm::report_fatal_error(
        "please provide one and only one of 'opcode-index', 'opcode-name' or "
        "'snippets-file'");
  if (!SnippetsFile.empty())
    return 0;
  if (OpcodeIndex > 0)
    return OpcodeIndex;
  // Resolve opcode name -> opcode.
  for (unsigned I = 0, E = MCInstrInfo.getNumOpcodes(); I < E; ++I)
    if (MCInstrInfo.getName(I) == OpcodeName)
      return I;
  llvm::report_fatal_error(llvm::Twine("unknown opcode ").concat(OpcodeName));
}

static BenchmarkResultContext
getBenchmarkResultContext(const LLVMState &State) {
  BenchmarkResultContext Ctx;

  const llvm::MCInstrInfo &InstrInfo = State.getInstrInfo();
  for (unsigned E = InstrInfo.getNumOpcodes(), I = 0; I < E; ++I)
    Ctx.addInstrEntry(I, InstrInfo.getName(I).data());

  const llvm::MCRegisterInfo &RegInfo = State.getRegInfo();
  for (unsigned E = RegInfo.getNumRegs(), I = 0; I < E; ++I)
    Ctx.addRegEntry(I, RegInfo.getName(I));

  return Ctx;
}

// Generates code snippets for opcode `Opcode`.
static llvm::Expected<std::vector<BenchmarkCode>>
generateSnippets(const LLVMState &State, unsigned Opcode) {
  const std::unique_ptr<SnippetGenerator> Generator =
      State.getExegesisTarget().createSnippetGenerator(BenchmarkMode, State);
  if (!Generator)
    llvm::report_fatal_error("cannot create snippet generator");

  const llvm::MCInstrDesc &InstrDesc = State.getInstrInfo().get(Opcode);
  // Ignore instructions that we cannot run.
  if (InstrDesc.isPseudo())
    return llvm::make_error<BenchmarkFailure>("Unsupported opcode: isPseudo");
  if (InstrDesc.isBranch() || InstrDesc.isIndirectBranch())
    return llvm::make_error<BenchmarkFailure>(
        "Unsupported opcode: isBranch/isIndirectBranch");
  if (InstrDesc.isCall() || InstrDesc.isReturn())
    return llvm::make_error<BenchmarkFailure>(
        "Unsupported opcode: isCall/isReturn");

  return Generator->generateConfigurations(Opcode);
}

namespace {

// An MCStreamer that reads a BenchmarkCode definition from a file.
// The BenchmarkCode definition is just an asm file, with additional comments to
// specify which registers should be defined or are live on entry.
class BenchmarkCodeStreamer : public llvm::MCStreamer,
                              public llvm::AsmCommentConsumer {
public:
  explicit BenchmarkCodeStreamer(llvm::MCContext *Context,
                                 const llvm::MCRegisterInfo *TheRegInfo,
                                 BenchmarkCode *Result)
      : llvm::MCStreamer(*Context), RegInfo(TheRegInfo), Result(Result) {}

  // Implementation of the llvm::MCStreamer interface. We only care about
  // instructions.
  void EmitInstruction(const llvm::MCInst &instruction,
                       const llvm::MCSubtargetInfo &mc_subtarget_info,
                       bool PrintSchedInfo) override {
    Result->Instructions.push_back(instruction);
  }

  // Implementation of the llvm::AsmCommentConsumer.
  void HandleComment(llvm::SMLoc Loc, llvm::StringRef CommentText) override {
    CommentText = CommentText.trim();
    if (!CommentText.consume_front("LLVM-EXEGESIS-"))
      return;
    if (CommentText.consume_front("DEFREG")) {
      // LLVM-EXEGESIS-DEFREF <reg> <hex_value>
      RegisterValue RegVal;
      llvm::SmallVector<llvm::StringRef, 2> Parts;
      CommentText.split(Parts, ' ', /*unlimited splits*/ -1,
                        /*do not keep empty strings*/ false);
      if (Parts.size() != 2) {
        llvm::errs() << "invalid comment 'LLVM-EXEGESIS-DEFREG " << CommentText
                     << "\n";
        ++InvalidComments;
      }
      if (!(RegVal.Register = findRegisterByName(Parts[0].trim()))) {
        llvm::errs() << "unknown register in 'LLVM-EXEGESIS-DEFREG "
                     << CommentText << "\n";
        ++InvalidComments;
        return;
      }
      const llvm::StringRef HexValue = Parts[1].trim();
      RegVal.Value = llvm::APInt(
          /* each hex digit is 4 bits */ HexValue.size() * 4, HexValue, 16);
      Result->RegisterInitialValues.push_back(std::move(RegVal));
      return;
    }
    if (CommentText.consume_front("LIVEIN")) {
      // LLVM-EXEGESIS-LIVEIN <reg>
      if (unsigned Reg = findRegisterByName(CommentText.ltrim()))
        Result->LiveIns.push_back(Reg);
      else {
        llvm::errs() << "unknown register in 'LLVM-EXEGESIS-LIVEIN "
                     << CommentText << "\n";
        ++InvalidComments;
      }
      return;
    }
  }

  unsigned numInvalidComments() const { return InvalidComments; }

private:
  // We only care about instructions, we don't implement this part of the API.
  void EmitCommonSymbol(llvm::MCSymbol *symbol, uint64_t size,
                        unsigned byte_alignment) override {}
  bool EmitSymbolAttribute(llvm::MCSymbol *symbol,
                           llvm::MCSymbolAttr attribute) override {
    return false;
  }
  void EmitValueToAlignment(unsigned byte_alignment, int64_t value,
                            unsigned value_size,
                            unsigned max_bytes_to_emit) override {}
  void EmitZerofill(llvm::MCSection *section, llvm::MCSymbol *symbol,
                    uint64_t size, unsigned byte_alignment,
                    llvm::SMLoc Loc) override {}

  unsigned findRegisterByName(const llvm::StringRef RegName) const {
    // FIXME: Can we do better than this ?
    for (unsigned I = 0, E = RegInfo->getNumRegs(); I < E; ++I) {
      if (RegName == RegInfo->getName(I))
        return I;
    }
    llvm::errs() << "'" << RegName
                 << "' is not a valid register name for the target\n";
    return 0;
  }

  const llvm::MCRegisterInfo *const RegInfo;
  BenchmarkCode *const Result;
  unsigned InvalidComments = 0;
};

} // namespace

// Reads code snippets from file `Filename`.
static llvm::Expected<std::vector<BenchmarkCode>>
readSnippets(const LLVMState &State, llvm::StringRef Filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> BufferPtr =
      llvm::MemoryBuffer::getFileOrSTDIN(Filename);
  if (std::error_code EC = BufferPtr.getError()) {
    return llvm::make_error<BenchmarkFailure>(
        "cannot read snippet: " + Filename + ": " + EC.message());
  }
  llvm::SourceMgr SM;
  SM.AddNewSourceBuffer(std::move(BufferPtr.get()), llvm::SMLoc());

  BenchmarkCode Result;

  llvm::MCObjectFileInfo ObjectFileInfo;
  const llvm::TargetMachine &TM = State.getTargetMachine();
  llvm::MCContext Context(TM.getMCAsmInfo(), TM.getMCRegisterInfo(),
                          &ObjectFileInfo);
  ObjectFileInfo.InitMCObjectFileInfo(TM.getTargetTriple(), /*PIC*/ false,
                                      Context);
  BenchmarkCodeStreamer Streamer(&Context, TM.getMCRegisterInfo(), &Result);
  const std::unique_ptr<llvm::MCAsmParser> AsmParser(
      llvm::createMCAsmParser(SM, Context, Streamer, *TM.getMCAsmInfo()));
  if (!AsmParser)
    return llvm::make_error<BenchmarkFailure>("cannot create asm parser");
  AsmParser->getLexer().setCommentConsumer(&Streamer);

  const std::unique_ptr<llvm::MCTargetAsmParser> TargetAsmParser(
      TM.getTarget().createMCAsmParser(*TM.getMCSubtargetInfo(), *AsmParser,
                                       *TM.getMCInstrInfo(),
                                       llvm::MCTargetOptions()));

  if (!TargetAsmParser)
    return llvm::make_error<BenchmarkFailure>(
        "cannot create target asm parser");
  AsmParser->setTargetParser(*TargetAsmParser);

  if (AsmParser->Run(false))
    return llvm::make_error<BenchmarkFailure>("cannot parse asm file");
  if (Streamer.numInvalidComments())
    return llvm::make_error<BenchmarkFailure>(
        llvm::Twine("found ")
            .concat(llvm::Twine(Streamer.numInvalidComments()))
            .concat(" invalid LLVM-EXEGESIS comments"));
  return std::vector<BenchmarkCode>{std::move(Result)};
}

void benchmarkMain() {
  if (exegesis::pfm::pfmInitialize())
    llvm::report_fatal_error("cannot initialize libpfm");

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();
#ifdef LLVM_EXEGESIS_INITIALIZE_NATIVE_TARGET
  LLVM_EXEGESIS_INITIALIZE_NATIVE_TARGET();
#endif

  const LLVMState State;
  const auto Opcode = getOpcodeOrDie(State.getInstrInfo());

  std::vector<BenchmarkCode> Configurations;
  if (Opcode > 0) {
    // Ignore instructions without a sched class if -ignore-invalid-sched-class
    // is passed.
    if (IgnoreInvalidSchedClass &&
        State.getInstrInfo().get(Opcode).getSchedClass() == 0) {
      llvm::errs() << "ignoring instruction without sched class\n";
      return;
    }
    Configurations = ExitOnErr(generateSnippets(State, Opcode));
  } else {
    Configurations = ExitOnErr(readSnippets(State, SnippetsFile));
  }

  const std::unique_ptr<BenchmarkRunner> Runner =
      State.getExegesisTarget().createBenchmarkRunner(BenchmarkMode, State);
  if (!Runner) {
    llvm::report_fatal_error("cannot create benchmark runner");
  }

  if (NumRepetitions == 0)
    llvm::report_fatal_error("--num-repetitions must be greater than zero");

  // Write to standard output if file is not set.
  if (BenchmarkFile.empty())
    BenchmarkFile = "-";

  const BenchmarkResultContext Context = getBenchmarkResultContext(State);

  for (const BenchmarkCode &Conf : Configurations) {
    InstructionBenchmark Result =
        Runner->runConfiguration(Conf, NumRepetitions);
    ExitOnErr(Result.writeYaml(Context, BenchmarkFile));
  }
  exegesis::pfm::pfmTerminate();
}

// Prints the results of running analysis pass `Pass` to file `OutputFilename`
// if OutputFilename is non-empty.
template <typename Pass>
static void maybeRunAnalysis(const Analysis &Analyzer, const std::string &Name,
                             const std::string &OutputFilename) {
  if (OutputFilename.empty())
    return;
  if (OutputFilename != "-") {
    llvm::errs() << "Printing " << Name << " results to file '"
                 << OutputFilename << "'\n";
  }
  std::error_code ErrorCode;
  llvm::raw_fd_ostream ClustersOS(OutputFilename, ErrorCode,
                                  llvm::sys::fs::FA_Read |
                                      llvm::sys::fs::FA_Write);
  if (ErrorCode)
    llvm::report_fatal_error("cannot open out file: " + OutputFilename);
  if (auto Err = Analyzer.run<Pass>(ClustersOS))
    llvm::report_fatal_error(std::move(Err));
}

static void analysisMain() {
  if (BenchmarkFile.empty())
    llvm::report_fatal_error("--benchmarks-file must be set.");

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetDisassembler();
  // Read benchmarks.
  const LLVMState State;
  const std::vector<InstructionBenchmark> Points =
      ExitOnErr(InstructionBenchmark::readYamls(
          getBenchmarkResultContext(State), BenchmarkFile));
  llvm::outs() << "Parsed " << Points.size() << " benchmark points\n";
  if (Points.empty()) {
    llvm::errs() << "no benchmarks to analyze\n";
    return;
  }
  // FIXME: Check that all points have the same triple/cpu.
  // FIXME: Merge points from several runs (latency and uops).

  std::string Error;
  const auto *TheTarget =
      llvm::TargetRegistry::lookupTarget(Points[0].LLVMTriple, Error);
  if (!TheTarget) {
    llvm::errs() << "unknown target '" << Points[0].LLVMTriple << "'\n";
    return;
  }
  const auto Clustering = ExitOnErr(InstructionBenchmarkClustering::create(
      Points, AnalysisNumPoints, AnalysisEpsilon));

  const Analysis Analyzer(*TheTarget, Clustering);

  maybeRunAnalysis<Analysis::PrintClusters>(Analyzer, "analysis clusters",
                                            AnalysisClustersOutputFile);
  maybeRunAnalysis<Analysis::PrintSchedClassInconsistencies>(
      Analyzer, "sched class consistency analysis",
      AnalysisInconsistenciesOutputFile);
}

} // namespace exegesis

int main(int Argc, char **Argv) {
  llvm::cl::ParseCommandLineOptions(Argc, Argv, "");

  exegesis::ExitOnErr.setExitCodeMapper([](const llvm::Error &Err) {
    if (Err.isA<llvm::StringError>())
      return EXIT_SUCCESS;
    return EXIT_FAILURE;
  });

  if (BenchmarkMode == exegesis::InstructionBenchmark::Unknown) {
    exegesis::analysisMain();
  } else {
    exegesis::benchmarkMain();
  }
  return EXIT_SUCCESS;
}
