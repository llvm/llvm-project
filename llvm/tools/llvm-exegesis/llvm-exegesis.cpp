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
#include "lib/Error.h"
#include "lib/LlvmState.h"
#include "lib/PerfHelper.h"
#include "lib/SnippetFile.h"
#include "lib/SnippetRepetitor.h"
#include "lib/Target.h"
#include "lib/TargetSelect.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
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

static cl::OptionCategory Options("llvm-exegesis options");
static cl::OptionCategory BenchmarkOptions("llvm-exegesis benchmark options");
static cl::OptionCategory AnalysisOptions("llvm-exegesis analysis options");

static cl::opt<int> OpcodeIndex("opcode-index",
                                cl::desc("opcode to measure, by index"),
                                cl::cat(BenchmarkOptions), cl::init(0));

static cl::opt<std::string>
    OpcodeNames("opcode-name",
                cl::desc("comma-separated list of opcodes to measure, by name"),
                cl::cat(BenchmarkOptions), cl::init(""));

static cl::opt<std::string> SnippetsFile("snippets-file",
                                         cl::desc("code snippets to measure"),
                                         cl::cat(BenchmarkOptions),
                                         cl::init(""));

static cl::opt<std::string>
    BenchmarkFile("benchmarks-file",
                  cl::desc("File to read (analysis mode) or write "
                           "(latency/uops/inverse_throughput modes) benchmark "
                           "results. “-” uses stdin/stdout."),
                  cl::cat(Options), cl::init(""));

static cl::opt<exegesis::InstructionBenchmark::ModeE> BenchmarkMode(
    "mode", cl::desc("the mode to run"), cl::cat(Options),
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

static cl::opt<exegesis::InstructionBenchmark::RepetitionModeE> RepetitionMode(
    "repetition-mode", cl::desc("how to repeat the instruction snippet"),
    cl::cat(BenchmarkOptions),
    cl::values(clEnumValN(exegesis::InstructionBenchmark::Duplicate,
                          "duplicate", "Duplicate the snippet"),
               clEnumValN(exegesis::InstructionBenchmark::Loop, "loop",
                          "Loop over the snippet")));

static cl::opt<unsigned>
    NumRepetitions("num-repetitions",
                   cl::desc("number of time to repeat the asm snippet"),
                   cl::cat(BenchmarkOptions), cl::init(10000));

static cl::opt<unsigned> MaxConfigsPerOpcode(
    "max-configs-per-opcode",
    cl::desc(
        "allow to snippet generator to generate at most that many configs"),
    cl::cat(BenchmarkOptions), cl::init(1));

static cl::opt<bool> IgnoreInvalidSchedClass(
    "ignore-invalid-sched-class",
    cl::desc("ignore instructions that do not define a sched class"),
    cl::cat(BenchmarkOptions), cl::init(false));

static cl::opt<exegesis::InstructionBenchmarkClustering::ModeE>
    AnalysisClusteringAlgorithm(
        "analysis-clustering", cl::desc("the clustering algorithm to use"),
        cl::cat(AnalysisOptions),
        cl::values(clEnumValN(exegesis::InstructionBenchmarkClustering::Dbscan,
                              "dbscan", "use DBSCAN/OPTICS algorithm"),
                   clEnumValN(exegesis::InstructionBenchmarkClustering::Naive,
                              "naive", "one cluster per opcode")),
        cl::init(exegesis::InstructionBenchmarkClustering::Dbscan));

static cl::opt<unsigned> AnalysisDbscanNumPoints(
    "analysis-numpoints",
    cl::desc("minimum number of points in an analysis cluster (dbscan only)"),
    cl::cat(AnalysisOptions), cl::init(3));

static cl::opt<float> AnalysisClusteringEpsilon(
    "analysis-clustering-epsilon",
    cl::desc("epsilon for benchmark point clustering"),
    cl::cat(AnalysisOptions), cl::init(0.1));

static cl::opt<float> AnalysisInconsistencyEpsilon(
    "analysis-inconsistency-epsilon",
    cl::desc("epsilon for detection of when the cluster is different from the "
             "LLVM schedule profile values"),
    cl::cat(AnalysisOptions), cl::init(0.1));

static cl::opt<std::string>
    AnalysisClustersOutputFile("analysis-clusters-output-file", cl::desc(""),
                               cl::cat(AnalysisOptions), cl::init(""));
static cl::opt<std::string>
    AnalysisInconsistenciesOutputFile("analysis-inconsistencies-output-file",
                                      cl::desc(""), cl::cat(AnalysisOptions),
                                      cl::init(""));

static cl::opt<bool> AnalysisDisplayUnstableOpcodes(
    "analysis-display-unstable-clusters",
    cl::desc("if there is more than one benchmark for an opcode, said "
             "benchmarks may end up not being clustered into the same cluster "
             "if the measured performance characteristics are different. by "
             "default all such opcodes are filtered out. this flag will "
             "instead show only such unstable opcodes"),
    cl::cat(AnalysisOptions), cl::init(false));

static cl::opt<std::string> CpuName(
    "mcpu",
    cl::desc("cpu name to use for pfm counters, leave empty to autodetect"),
    cl::cat(Options), cl::init(""));

static cl::opt<bool>
    DumpObjectToDisk("dump-object-to-disk",
                     cl::desc("dumps the generated benchmark object to disk "
                              "and prints a message to access it"),
                     cl::cat(BenchmarkOptions), cl::init(true));

static ExitOnError ExitOnErr;

// Checks that only one of OpcodeNames, OpcodeIndex or SnippetsFile is provided,
// and returns the opcode indices or {} if snippets should be read from
// `SnippetsFile`.
static std::vector<unsigned> getOpcodesOrDie(const MCInstrInfo &MCInstrInfo) {
  const size_t NumSetFlags = (OpcodeNames.empty() ? 0 : 1) +
                             (OpcodeIndex == 0 ? 0 : 1) +
                             (SnippetsFile.empty() ? 0 : 1);
  if (NumSetFlags != 1)
    report_fatal_error(
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
  const auto ResolveName = [&MCInstrInfo](StringRef OpcodeName) -> unsigned {
    for (unsigned I = 1, E = MCInstrInfo.getNumOpcodes(); I < E; ++I)
      if (MCInstrInfo.getName(I) == OpcodeName)
        return I;
    return 0u;
  };
  SmallVector<StringRef, 2> Pieces;
  StringRef(OpcodeNames.getValue())
      .split(Pieces, ",", /* MaxSplit */ -1, /* KeepEmpty */ false);
  std::vector<unsigned> Result;
  for (const StringRef &OpcodeName : Pieces) {
    if (unsigned Opcode = ResolveName(OpcodeName))
      Result.push_back(Opcode);
    else
      report_fatal_error(Twine("unknown opcode ").concat(OpcodeName));
  }
  return Result;
}

// Generates code snippets for opcode `Opcode`.
static Expected<std::vector<BenchmarkCode>>
generateSnippets(const LLVMState &State, unsigned Opcode,
                 const BitVector &ForbiddenRegs) {
  const Instruction &Instr = State.getIC().getInstr(Opcode);
  const MCInstrDesc &InstrDesc = Instr.Description;
  // Ignore instructions that we cannot run.
  if (InstrDesc.isPseudo())
    return make_error<Failure>("Unsupported opcode: isPseudo");
  if (InstrDesc.isBranch() || InstrDesc.isIndirectBranch())
    return make_error<Failure>("Unsupported opcode: isBranch/isIndirectBranch");
  if (InstrDesc.isCall() || InstrDesc.isReturn())
    return make_error<Failure>("Unsupported opcode: isCall/isReturn");

  SnippetGenerator::Options SnippetOptions;
  SnippetOptions.MaxConfigsPerOpcode = MaxConfigsPerOpcode;
  const std::unique_ptr<SnippetGenerator> Generator =
      State.getExegesisTarget().createSnippetGenerator(BenchmarkMode, State,
                                                       SnippetOptions);
  if (!Generator)
    report_fatal_error("cannot create snippet generator");
  return Generator->generateConfigurations(Instr, ForbiddenRegs);
}

void benchmarkMain() {
#ifndef HAVE_LIBPFM
  report_fatal_error(
      "benchmarking unavailable, LLVM was built without libpfm.");
#endif

  if (exegesis::pfm::pfmInitialize())
    report_fatal_error("cannot initialize libpfm");

  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetAsmParser();
  InitializeNativeExegesisTarget();

  const LLVMState State(CpuName);

  const std::unique_ptr<BenchmarkRunner> Runner =
      State.getExegesisTarget().createBenchmarkRunner(BenchmarkMode, State);
  if (!Runner) {
    report_fatal_error("cannot create benchmark runner");
  }

  const auto Opcodes = getOpcodesOrDie(State.getInstrInfo());

  const auto Repetitor = SnippetRepetitor::Create(RepetitionMode, State);

  std::vector<BenchmarkCode> Configurations;
  if (!Opcodes.empty()) {
    for (const unsigned Opcode : Opcodes) {
      // Ignore instructions without a sched class if
      // -ignore-invalid-sched-class is passed.
      if (IgnoreInvalidSchedClass &&
          State.getInstrInfo().get(Opcode).getSchedClass() == 0) {
        errs() << State.getInstrInfo().getName(Opcode)
               << ": ignoring instruction without sched class\n";
        continue;
      }
      auto ConfigsForInstr =
          generateSnippets(State, Opcode, Repetitor->getReservedRegs());
      if (!ConfigsForInstr) {
        logAllUnhandledErrors(
            ConfigsForInstr.takeError(), errs(),
            Twine(State.getInstrInfo().getName(Opcode)).concat(": "));
        continue;
      }
      std::move(ConfigsForInstr->begin(), ConfigsForInstr->end(),
                std::back_inserter(Configurations));
    }
  } else {
    Configurations = ExitOnErr(readSnippets(State, SnippetsFile));
  }

  if (NumRepetitions == 0)
    report_fatal_error("--num-repetitions must be greater than zero");

  // Write to standard output if file is not set.
  if (BenchmarkFile.empty())
    BenchmarkFile = "-";

  for (const BenchmarkCode &Conf : Configurations) {
    InstructionBenchmark Result = Runner->runConfiguration(
        Conf, NumRepetitions, *Repetitor, DumpObjectToDisk);
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
    errs() << "Printing " << Name << " results to file '" << OutputFilename
           << "'\n";
  }
  std::error_code ErrorCode;
  raw_fd_ostream ClustersOS(OutputFilename, ErrorCode,
                            sys::fs::FA_Read | sys::fs::FA_Write);
  if (ErrorCode)
    report_fatal_error("cannot open out file: " + OutputFilename);
  if (auto Err = Analyzer.run<Pass>(ClustersOS))
    report_fatal_error(std::move(Err));
}

static void analysisMain() {
  if (BenchmarkFile.empty())
    report_fatal_error("--benchmarks-file must be set.");

  if (AnalysisClustersOutputFile.empty() &&
      AnalysisInconsistenciesOutputFile.empty()) {
    report_fatal_error(
        "At least one of --analysis-clusters-output-file and "
        "--analysis-inconsistencies-output-file must be specified.");
  }

  InitializeNativeTarget();
  InitializeNativeTargetAsmPrinter();
  InitializeNativeTargetDisassembler();
  // Read benchmarks.
  const LLVMState State("");
  const std::vector<InstructionBenchmark> Points =
      ExitOnErr(InstructionBenchmark::readYamls(State, BenchmarkFile));
  outs() << "Parsed " << Points.size() << " benchmark points\n";
  if (Points.empty()) {
    errs() << "no benchmarks to analyze\n";
    return;
  }
  // FIXME: Check that all points have the same triple/cpu.
  // FIXME: Merge points from several runs (latency and uops).

  std::string Error;
  const auto *TheTarget =
      TargetRegistry::lookupTarget(Points[0].LLVMTriple, Error);
  if (!TheTarget) {
    errs() << "unknown target '" << Points[0].LLVMTriple << "'\n";
    return;
  }

  std::unique_ptr<MCInstrInfo> InstrInfo(TheTarget->createMCInstrInfo());

  const auto Clustering = ExitOnErr(InstructionBenchmarkClustering::create(
      Points, AnalysisClusteringAlgorithm, AnalysisDbscanNumPoints,
      AnalysisClusteringEpsilon, InstrInfo->getNumOpcodes()));

  const Analysis Analyzer(*TheTarget, std::move(InstrInfo), Clustering,
                          AnalysisInconsistencyEpsilon,
                          AnalysisDisplayUnstableOpcodes);

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

  exegesis::ExitOnErr.setExitCodeMapper([](const Error &Err) {
    if (Err.isA<StringError>())
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
