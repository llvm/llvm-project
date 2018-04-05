//===-- BenchmarkRunner.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines the abstract BenchmarkRunner class for measuring a certain execution
/// property of instructions (e.g. latency).
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_EXEGESIS_BENCHMARKRUNNER_H
#define LLVM_TOOLS_LLVM_EXEGESIS_BENCHMARKRUNNER_H

#include "BenchmarkResult.h"
#include "InMemoryAssembler.h"
#include "LlvmState.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/Error.h"
#include <vector>

namespace exegesis {

// Common code for all benchmark modes.
class BenchmarkRunner {
public:
  // Subtargets can disable running benchmarks for some instructions by
  // returning an error here.
  class InstructionFilter {
  public:
    virtual ~InstructionFilter();

    virtual llvm::Error shouldRun(const LLVMState &State,
                                  unsigned Opcode) const {
      return llvm::ErrorSuccess();
    }
  };

  virtual ~BenchmarkRunner();

  InstructionBenchmark run(const LLVMState &State, unsigned Opcode,
                           unsigned NumRepetitions,
                           const InstructionFilter &Filter) const;

private:
  virtual const char *getDisplayName() const = 0;

  virtual llvm::Expected<std::vector<llvm::MCInst>>
  createCode(const LLVMState &State, unsigned OpcodeIndex,
             unsigned NumRepetitions,
             const JitFunctionContext &Context) const = 0;

  virtual std::vector<BenchmarkMeasure>
  runMeasurements(const LLVMState &State, const JitFunction &Function,
                  unsigned NumRepetitions) const = 0;
};

} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_BENCHMARKRUNNER_H
