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

#include "Assembler.h"
#include "BenchmarkResult.h"
#include "LlvmState.h"
#include "RegisterAliasing.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/Error.h"
#include <vector>

namespace exegesis {

// A collection of instructions that are to be assembled, executed and measured.
struct BenchmarkConfiguration {
  // This code is run before the Snippet is iterated. Since it is part of the
  // measurement it should be as short as possible. It is usually used to setup
  // the content of the Registers.
  std::vector<llvm::MCInst> SnippetSetup;

  // The sequence of instructions that are to be repeated.
  std::vector<llvm::MCInst> Snippet;

  // Informations about how this configuration was built.
  std::string Info;
};

// Common code for all benchmark modes.
class BenchmarkRunner {
public:
  explicit BenchmarkRunner(const LLVMState &State);

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

  llvm::Expected<std::vector<InstructionBenchmark>>
  run(unsigned Opcode, const InstructionFilter &Filter,
      unsigned NumRepetitions);

protected:
  const LLVMState &State;
  const llvm::MCInstrInfo &MCInstrInfo;
  const llvm::MCRegisterInfo &MCRegisterInfo;

private:
  InstructionBenchmark runOne(const BenchmarkConfiguration &Configuration,
                              unsigned Opcode, unsigned NumRepetitions) const;

  virtual InstructionBenchmark::ModeE getMode() const = 0;

  virtual llvm::Expected<std::vector<BenchmarkConfiguration>>
  createConfigurations(RegisterAliasingTrackerCache &RATC,
                       unsigned Opcode) const = 0;

  virtual std::vector<BenchmarkMeasure>
  runMeasurements(const ExecutableFunction &EF,
                  const unsigned NumRepetitions) const = 0;

  llvm::Expected<std::string>
  writeObjectFile(llvm::ArrayRef<llvm::MCInst> Code) const;

  RegisterAliasingTrackerCache RATC;
};

} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_BENCHMARKRUNNER_H
