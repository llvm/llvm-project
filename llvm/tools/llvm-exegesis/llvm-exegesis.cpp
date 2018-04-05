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

#include "lib/BenchmarkResult.h"
#include "lib/BenchmarkRunner.h"
#include "lib/Latency.h"
#include "lib/LlvmState.h"
#include "lib/PerfHelper.h"
#include "lib/Uops.h"
#include "lib/X86.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
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

enum class BenchmarkModeE { Latency, Uops };
static llvm::cl::opt<BenchmarkModeE>
    BenchmarkMode("benchmark-mode", llvm::cl::desc("the benchmark mode to run"),
                  llvm::cl::values(clEnumValN(BenchmarkModeE::Latency,
                                              "latency", "Instruction Latency"),
                                   clEnumValN(BenchmarkModeE::Uops, "uops",
                                              "Uop Decomposition")));

static llvm::cl::opt<unsigned>
    NumRepetitions("num-repetitions",
                   llvm::cl::desc("number of time to repeat the asm snippet"),
                   llvm::cl::init(10000));

namespace exegesis {

void main() {
  if (OpcodeName.empty() == (OpcodeIndex == 0)) {
    llvm::report_fatal_error(
        "please provide one and only one of 'opcode-index' or 'opcode-name' ");
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // FIXME: Target-specific filter.
  X86Filter Filter;

  const LLVMState State;

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
  }

  Runner->run(State, Opcode, NumRepetitions > 0 ? NumRepetitions : 1, Filter)
      .writeYamlOrDie("-");
}

} // namespace exegesis

int main(int Argc, char **Argv) {
  llvm::cl::ParseCommandLineOptions(Argc, Argv, "");

  if (exegesis::pfm::pfmInitialize()) {
    llvm::errs() << "cannot initialize libpfm\n";
    return EXIT_FAILURE;
  }

  exegesis::main();

  exegesis::pfm::pfmTerminate();
  return EXIT_SUCCESS;
}
