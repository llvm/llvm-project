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
#include "MCInstrDescView.h"
#include "RegisterAliasing.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/Error.h"
#include <cstdlib>
#include <memory>
#include <vector>

namespace exegesis {

// A class representing failures that happened during Benchmark, they are used
// to report informations to the user.
class BenchmarkFailure : public llvm::StringError {
public:
  BenchmarkFailure(const llvm::Twine &S);
};

// A collection of instructions that are to be assembled, executed and measured.
struct BenchmarkCode {
  // The sequence of instructions that are to be repeated.
  std::vector<llvm::MCInst> Instructions;

  // Before the code is executed some instructions are added to setup the
  // registers initial values.
  std::vector<unsigned> RegsToDef;

  // We also need to provide the registers that are live on entry for the
  // assembler to generate proper prologue/epilogue.
  std::vector<unsigned> LiveIns;

  // Informations about how this configuration was built.
  std::string Info;
};

// Common code for all benchmark modes.
class BenchmarkRunner {
public:
  explicit BenchmarkRunner(const LLVMState &State,
                           InstructionBenchmark::ModeE Mode);

  virtual ~BenchmarkRunner();

  llvm::Expected<std::vector<InstructionBenchmark>>
  run(unsigned Opcode, unsigned NumRepetitions);

  // Given a snippet, computes which registers the setup code needs to define.
  std::vector<unsigned>
  computeRegsToDef(const std::vector<InstructionBuilder> &Snippet) const;

  // Scratch space to run instructions that touch memory.
  struct ScratchSpace {
    static constexpr const size_t kAlignment = 1024;
    static constexpr const size_t kSize = 1 << 20; // 1MB.
    ScratchSpace()
        : UnalignedPtr(llvm::make_unique<char[]>(kSize + kAlignment)),
          AlignedPtr(
              UnalignedPtr.get() + kAlignment -
              (reinterpret_cast<intptr_t>(UnalignedPtr.get()) % kAlignment)) {}
    char *ptr() const { return AlignedPtr; }
    void clear() { std::memset(ptr(), 0, kSize); }

  private:
    const std::unique_ptr<char[]> UnalignedPtr;
    char *const AlignedPtr;
  };

protected:
  const LLVMState &State;
  const RegisterAliasingTrackerCache RATC;

  // Generates a single code template that has a self-dependency.
  llvm::Expected<CodeTemplate>
  generateSelfAliasingCodeTemplate(const Instruction &Instr) const;
  // Generates a single code template without assignment constraints.
  llvm::Expected<CodeTemplate>
  generateUnconstrainedCodeTemplate(const Instruction &Instr,
                                    llvm::StringRef Msg) const;

private:
  // API to be implemented by subclasses.
  virtual llvm::Expected<CodeTemplate>
  generateCodeTemplate(unsigned Opcode) const = 0;

  virtual std::vector<BenchmarkMeasure>
  runMeasurements(const ExecutableFunction &EF, ScratchSpace &Scratch,
                  const unsigned NumRepetitions) const = 0;

  // Internal helpers.
  InstructionBenchmark runConfiguration(const BenchmarkCode &Configuration,
                                        unsigned NumRepetitions) const;

  // Calls generateCodeTemplate and expands it into one or more BenchmarkCode.
  llvm::Expected<std::vector<BenchmarkCode>>
  generateConfigurations(unsigned Opcode) const;

  llvm::Expected<std::string>
  writeObjectFile(const BenchmarkCode &Configuration,
                  llvm::ArrayRef<llvm::MCInst> Code) const;

  const InstructionBenchmark::ModeE Mode;

  const std::unique_ptr<ScratchSpace> Scratch;
};

} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_BENCHMARKRUNNER_H
