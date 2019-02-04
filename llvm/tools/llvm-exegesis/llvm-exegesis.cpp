//===-- llvm-exegesis.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include <string>

namespace llvm {
namespace exegesis {

static cl::opt<int> OpcodeIndex("opcode-index",
                                cl::desc("opcode to measure, by index"),
                                cl::init(0));

static cl::opt<std::string>
    OpcodeNames("opcode-name",
                cl::desc("comma-separated list of opcodes to measure, by name"),
                cl::init(""));

static cl::opt<std::string> SnippetsFile("snippets-file",
                                         cl::desc("code snippets to measure"),
                                         cl::init(""));

static cl::opt<std::string> BenchmarkFile("benchmarks-file", cl::desc(""),
                                          cl::init(""));

static cl::opt<exegesis::InstructionBenchmark::ModeE> BenchmarkMode(
    "mode", cl::desc("the mode to run"),
    cl::values(clEnumValN(exegesis::InstructionBenchmark::Latency, "latency",
                          "Instruction Latency"),
               clEnumValN(exegesis::InstructionBenchmark::InverseThroughput,
                          "inverse_throughput",
                          "Instruction Inverse Throughput"),
               clEnumValN(exegesis::InstructionBenchmark::Uops, "uops",
                          "Uop Decomposition"),
               // When not asking for a specific benchmark mode,
               // we'll analyse the results.
               clEnumValN(exegesis::InstructionBenchmark::Unknown, "analysis",
                          "Analysis")));

static cl::opt<unsigned>
    NumRepetitions("num-repetitions",
                   cl::desc("number of time to repeat the asm snippet"),
                   cl::init(10000));

static cl::opt<bool> IgnoreInvalidSchedClass(
    "ignore-invalid-sched-class",
    cl::desc("ignore instructions that do not define a sched class"),
    cl::init(false));

static cl::opt<unsigned> AnalysisNumPoints(
    "analysis-numpoints",
    cl::desc("minimum number of points in an analysis cluster"), cl::init(3));

static cl::opt<float>
    AnalysisEpsilon("analysis-epsilon",
                    cl::desc("dbscan epsilon for analysis clustering"),
                    cl::init(0.1));

static cl::opt<std::string>
    AnalysisClustersOutputFile("analysis-clusters-output-file", cl::desc(""),
                               cl::init(""));
static cl::opt<std::string>
    AnalysisInconsistenciesOutputFile("analysis-inconsistencies-output-file",
                                      cl::desc(""), cl::init(""));

static cl::opt<std::string>
    CpuName("mcpu",
            cl::desc(
                "cpu name to use for pfm counters, leave empty to autodetect"),
            cl::init(""));


static ExitOnError ExitOnErr;

#ifdef LLVM_EXEGESIS_INITIALIZE_NATIVE_TARGET
void LLVM_EXEGESIS_INITIALIZE_NATIVE_TARGET();
#endif

// Checks that only one of OpcodeNames, OpcodeIndex or SnippetsFile is provided,
// and returns the opcode indices or {} if snippets should be read from
// `SnippetsFile`.
static std::vector<unsigned>
getOpcodesOrDie(const llvm::MCInstrInfo &MCInstrInfo) {
  const size_t NumSetFlags = (OpcodeNames.empty() ? 0 : 1) +
                             (OpcodeIndex == 0 ? 0 : 1) +
                             (SnippetsFile.empty() ? 0 : 1);
  if (NumSetFlags != 1)
    llvm::report_fatal_error(
        "please provide one and only one of 'opcode-index', 'opcode-name' or "
        "'snippets-file'");
  if (!SnippetsFile.empty())
    return {};
  if (OpcodeIndex > 0)
    return {static_cast<unsigned>(OpcodeIndex)};
  if (OpcodeIndex < 0) {
    std::vector<unsigned> Result;
    for (unsigned I = 1, E = MCInstrInfo.getNumOpcodes(); I < E; ++I)
      Result.push_back(I);
    return Result;
  }
  // Resolve opcode name -> opcode.
  const auto ResolveName =
      [&MCInstrInfo](llvm::StringRef OpcodeName) -> unsigned {
    for (unsigned I = 1, E = MCInstrInfo.getNumOpcodes(); I < E; ++I)
      if (MCInstrInfo.getName(I) == OpcodeName)
        return I;
    return 0u;
  };
  llvm::SmallVector<llvm::StringRef, 2> Pieces;
  llvm::StringRef(OpcodeNames.getValue())
      .split(Pieces, ",", /* MaxSplit */ -1, /* KeepEmpty */ false);
  std::vector<unsigned> Result;
  for (const llvm::StringRef OpcodeName : Pieces) {
    if (unsigned Opcode = ResolveName(OpcodeName))
      Result.push_back(Opcode);
    else
      llvm::report_fatal_error(
          llvm::Twine("unknown opcode ").concat(OpcodeName));
  }
  return Result;
}

// Generates code snippets for opcode `Opcode`.
static llvm::Expected<std::vector<BenchmarkCode>>
generateSnippets(const LLVMState &State, unsigned Opcode) {
  const Instruction &Instr = State.getIC().getInstr(Opcode);
  const llvm::MCInstrDesc &InstrDesc = *Instr.Description;
  // Ignore instructions that we cannot run.
  if (InstrDesc.isPseudo())
    return llvm::make_error<BenchmarkFailure>("Unsupported opcode: isPseudo");
  if (InstrDesc.isBranch() || InstrDesc.isIndirectBranch())
    return llvm::make_error<BenchmarkFailure>(
        "Unsupported opcode: isBranch/isIndirectBranch");
  if (InstrDesc.isCall() || InstrDesc.isReturn())
    return llvm::make_error<BenchmarkFailure>(
        "Unsupported opcode: isCall/isReturn");

  const std::unique_ptr<SnippetGenerator> Generator =
      State.getExegesisTarget().createSnippetGenerator(BenchmarkMode, State);
  if (!Generator)
    llvm::report_fatal_error("cannot create snippet generator");
  return Generator->generateConfigurations(Instr);
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
  void EmitInstruction(const llvm::MCInst &Instruction,
                       const llvm::MCSubtargetInfo &STI,
                       bool PrintSchedInfo) override {
    Result->Instructions.push_back(Instruction);
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
  void EmitCommonSymbol(llvm::MCSymbol *Symbol, uint64_t Size,
                        unsigned ByteAlignment) override {}
  bool EmitSymbolAttribute(llvm::MCSymbol *Symbol,
                           llvm::MCSymbolAttr Attribute) override {
    return false;
  }
  void EmitValueToAlignment(unsigned ByteAlignment, int64_t Value,
                            unsigned ValueSize,
                            unsigned MaxBytesToEmit) override {}
  void EmitZerofill(llvm::MCSection *Section, llvm::MCSymbol *Symbol,
                    uint64_t Size, unsigned ByteAlignment,
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

  const LLVMState State(CpuName);
  const auto Opcodes = getOpcodesOrDie(State.getInstrInfo());

  std::vector<BenchmarkCode> Configurations;
  if (!Opcodes.empty()) {
    for (const unsigned Opcode : Opcodes) {
      // Ignore instructions without a sched class if
      // -ignore-invalid-sched-class is passed.
      if (IgnoreInvalidSchedClass &&
          State.getInstrInfo().get(Opcode).getSchedClass() == 0) {
        llvm::errs() << State.getInstrInfo().getName(Opcode)
                     << ": ignoring instruction without sched class\n";
        continue;
      }
      auto ConfigsForInstr = generateSnippets(State, Opcode);
      if (!ConfigsForInstr) {
        llvm::logAllUnhandledErrors(
            ConfigsForInstr.takeError(), llvm::errs(),
            llvm::Twine(State.getInstrInfo().getName(Opcode)).concat(": "));
        continue;
      }
      std::move(ConfigsForInstr->begin(), ConfigsForInstr->end(),
                std::back_inserter(Configurations));
    }
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

  for (const BenchmarkCode &Conf : Configurations) {
    InstructionBenchmark Result =
        Runner->runConfiguration(Conf, NumRepetitions);
    ExitOnErr(Result.writeYaml(State, BenchmarkFile));
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

  if (AnalysisClustersOutputFile.empty() &&
      AnalysisInconsistenciesOutputFile.empty()) {
    llvm::report_fatal_error(
        "At least one of --analysis-clusters-output-file and "
        "--analysis-inconsistencies-output-file must be specified.");
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetDisassembler();
  // Read benchmarks.
  const LLVMState State("");
  const std::vector<InstructionBenchmark> Points =
      ExitOnErr(InstructionBenchmark::readYamls(State, BenchmarkFile));
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
} // namespace llvm

int main(int Argc, char **Argv) {
  using namespace llvm;
  cl::ParseCommandLineOptions(Argc, Argv, "");

  exegesis::ExitOnErr.setExitCodeMapper([](const llvm::Error &Err) {
    if (Err.isA<llvm::StringError>())
      return EXIT_SUCCESS;
    return EXIT_FAILURE;
  });

  if (exegesis::BenchmarkMode == exegesis::InstructionBenchmark::Unknown) {
    exegesis::analysisMain();
  } else {
    exegesis::benchmarkMain();
  }
  return EXIT_SUCCESS;
}
