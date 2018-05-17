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

  InstructionBenchmark run(unsigned Opcode, const InstructionFilter &Filter,
                           unsigned NumRepetitions);

protected:
  const LLVMState &State;
  const llvm::MCInstrInfo &MCInstrInfo;
  const llvm::MCRegisterInfo &MCRegisterInfo;

private:
  virtual const char *getDisplayName() const = 0;

  virtual llvm::Expected<std::vector<llvm::MCInst>>
  createSnippet(RegisterAliasingTrackerCache &RATC, unsigned Opcode,
                llvm::raw_ostream &Debug) const = 0;

  virtual std::vector<BenchmarkMeasure>
  runMeasurements(const ExecutableFunction &EF,
                  const unsigned NumRepetitions) const = 0;

  llvm::Expected<std::string>
  writeObjectFile(llvm::ArrayRef<llvm::MCInst> Code) const;

  RegisterAliasingTrackerCache RATC;
};

} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_BENCHMARKRUNNER_H
