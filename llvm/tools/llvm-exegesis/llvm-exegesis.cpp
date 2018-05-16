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
#include "lib/Latency.h"
#include "lib/LlvmState.h"
#include "lib/PerfHelper.h"
#include "lib/Uops.h"
#include "lib/X86.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Path.h"
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
    BenchmarkFile("benchmarks-file", llvm::cl::desc(""), llvm::cl::init("-"));

enum class BenchmarkModeE { Latency, Uops, Analysis };
static llvm::cl::opt<BenchmarkModeE> BenchmarkMode(
    "benchmark-mode", llvm::cl::desc("the benchmark mode to run"),
    llvm::cl::values(
        clEnumValN(BenchmarkModeE::Latency, "latency", "Instruction Latency"),
        clEnumValN(BenchmarkModeE::Uops, "uops", "Uop Decomposition"),
        clEnumValN(BenchmarkModeE::Analysis, "analysis", "Analysis")));

static llvm::cl::opt<unsigned>
    NumRepetitions("num-repetitions",
                   llvm::cl::desc("number of time to repeat the asm snippet"),
                   llvm::cl::init(10000));

static llvm::cl::opt<unsigned> AnalysisNumPoints(
    "analysis-numpoints",
    llvm::cl::desc("minimum number of points in an analysis cluster"),
    llvm::cl::init(3));

static llvm::cl::opt<float>
    AnalysisEpsilon("analysis-epsilon",
                    llvm::cl::desc("dbscan epsilon for analysis clustering"),
                    llvm::cl::init(0.1));

static llvm::cl::opt<std::string> AnalysisClustersFile("analysis-clusters-file",
                                                       llvm::cl::desc(""),
                                                       llvm::cl::init("-"));

namespace exegesis {

void benchmarkMain() {
  if (exegesis::pfm::pfmInitialize())
    llvm::report_fatal_error("cannot initialize libpfm");

  if (OpcodeName.empty() == (OpcodeIndex == 0))
    llvm::report_fatal_error(
        "please provide one and only one of 'opcode-index' or 'opcode-name'");

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // FIXME: Target-specific filter.
  X86Filter Filter;

  const LLVMState State;

  if (!State.getSubtargetInfo().getSchedModel().hasExtraProcessorInfo())
    llvm::report_fatal_error("sched model is missing extra processor info!");

  unsigned Opcode = OpcodeIndex;
  if (Opcode == 0) {
    // Resolve opcode name -> opcode.
    for (unsigned I = 0, E = State.getInstrInfo().getNumOpcodes(); I < E; ++I) {
      if (State.getInstrInfo().getName(I) == OpcodeName) {
        Opcode = I;
        break;
      }
    }
    if (Opcode == 0) {
      llvm::report_fatal_error(
          llvm::Twine("unknown opcode ").concat(OpcodeName));
    }
  }

  std::unique_ptr<BenchmarkRunner> Runner;
  switch (BenchmarkMode) {
  case BenchmarkModeE::Latency:
    Runner = llvm::make_unique<LatencyBenchmarkRunner>();
    break;
  case BenchmarkModeE::Uops:
    Runner = llvm::make_unique<UopsBenchmarkRunner>();
    break;
  case BenchmarkModeE::Analysis:
    llvm_unreachable("not a benchmark");
  }

  Runner->run(State, Opcode, NumRepetitions > 0 ? NumRepetitions : 1, Filter)
      .writeYamlOrDie(BenchmarkFile);
  exegesis::pfm::pfmTerminate();
}

void analysisMain() {
  // Read benchmarks.
  const std::vector<InstructionBenchmark> Points =
      InstructionBenchmark::readYamlsOrDie(BenchmarkFile);
  llvm::outs() << "Parsed " << Points.size() << " benchmark points\n";
  if (Points.empty()) {
    llvm::errs() << "no benchmarks to analyze\n";
    return;
  }
  // FIXME: Check that all points have the same triple/cpu.
  // FIXME: Merge points from several runs (latency and uops).

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  std::string Error;
  const auto *TheTarget =
      llvm::TargetRegistry::lookupTarget(Points[0].LLVMTriple, Error);
  if (!TheTarget) {
    llvm::errs() << "unknown target '" << Points[0].LLVMTriple << "'\n";
    return;
  }
  const auto Clustering = llvm::cantFail(InstructionBenchmarkClustering::create(
      Points, AnalysisNumPoints, AnalysisEpsilon));

  const Analysis Analyzer(*TheTarget, Clustering);

  std::error_code ErrorCode;
  llvm::raw_fd_ostream ClustersOS(AnalysisClustersFile, ErrorCode,
                                  llvm::sys::fs::F_RW);
  if (ErrorCode)
    llvm::report_fatal_error("cannot open out file: " + AnalysisClustersFile);

  if (auto Err = Analyzer.printClusters(ClustersOS))
    llvm::report_fatal_error(std::move(Err));
}

} // namespace exegesis

int main(int Argc, char **Argv) {
  llvm::cl::ParseCommandLineOptions(Argc, Argv, "");

  if (BenchmarkMode == BenchmarkModeE::Analysis) {
    exegesis::analysisMain();
  } else {
    exegesis::benchmarkMain();
  }
  return EXIT_SUCCESS;
}
