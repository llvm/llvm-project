//===-- Target.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// Classes that handle the creation of target-specific objects. This is
/// similar to Target/TargetRegistry.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_EXEGESIS_TARGET_H
#define LLVM_TOOLS_LLVM_EXEGESIS_TARGET_H

#include "BenchmarkResult.h"
#include "BenchmarkRunner.h"
#include "Error.h"
#include "LlvmState.h"
#include "PerfHelper.h"
#include "SnippetGenerator.h"
#include "llvm/ADT/Triple.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace exegesis {

struct PfmCountersInfo {
  // An optional name of a performance counter that can be used to measure
  // cycles.
  const char *CycleCounter;

  // An optional name of a performance counter that can be used to measure
  // uops.
  const char *UopsCounter;

  // An IssueCounter specifies how to measure uops issued to specific proc
  // resources.
  struct IssueCounter {
    const char *Counter;
    // The name of the ProcResource that this counter measures.
    const char *ProcResName;
  };
  // An optional list of IssueCounters.
  const IssueCounter *IssueCounters;
  unsigned NumIssueCounters;

  static const PfmCountersInfo Default;
};

struct CpuAndPfmCounters {
  const char *CpuName;
  const PfmCountersInfo *PCI;
  bool operator<(StringRef S) const { return StringRef(CpuName) < S; }
};

class ExegesisTarget {
public:
  explicit ExegesisTarget(ArrayRef<CpuAndPfmCounters> CpuPfmCounters)
      : CpuPfmCounters(CpuPfmCounters) {}

  // Targets can use this to create target-specific perf counters.
  virtual Expected<std::unique_ptr<pfm::Counter>>
  createCounter(StringRef CounterName, const LLVMState &State) const;

  // Targets can use this to add target-specific passes in assembleToStream();
  virtual void addTargetSpecificPasses(PassManagerBase &PM) const {}

  // Generates code to move a constant into a the given register.
  // Precondition: Value must fit into Reg.
  virtual std::vector<MCInst> setRegTo(const MCSubtargetInfo &STI, unsigned Reg,
                                       const APInt &Value) const = 0;

  // Returns the register pointing to scratch memory, or 0 if this target
  // does not support memory operands. The benchmark function uses the
  // default calling convention.
  virtual unsigned getScratchMemoryRegister(const Triple &) const { return 0; }

  // Fills memory operands with references to the address at [Reg] + Offset.
  virtual void fillMemoryOperands(InstructionTemplate &IT, unsigned Reg,
                                  unsigned Offset) const {
    llvm_unreachable(
        "fillMemoryOperands() requires getScratchMemoryRegister() > 0");
  }

  // Returns a counter usable as a loop counter.
  virtual unsigned getLoopCounterRegister(const Triple &) const { return 0; }

  // Adds the code to decrement the loop counter and
  virtual void decrementLoopCounterAndJump(MachineBasicBlock &MBB,
                                           MachineBasicBlock &TargetMBB,
                                           const MCInstrInfo &MII) const {
    llvm_unreachable("decrementLoopCounterAndBranch() requires "
                     "getLoopCounterRegister() > 0");
  }

  // Returns a list of unavailable registers.
  // Targets can use this to prevent some registers to be automatically selected
  // for use in snippets.
  virtual ArrayRef<unsigned> getUnavailableRegisters() const { return {}; }

  // Returns the maximum number of bytes a load/store instruction can access at
  // once. This is typically the size of the largest register available on the
  // processor. Note that this only used as a hint to generate independant
  // load/stores to/from memory, so the exact returned value does not really
  // matter as long as it's large enough.
  virtual unsigned getMaxMemoryAccessSize() const { return 0; }

  // Assigns a random operand of the right type to variable Var.
  // The target is responsible for handling any operand starting from
  // OPERAND_FIRST_TARGET.
  virtual Error randomizeTargetMCOperand(const Instruction &Instr,
                                         const Variable &Var,
                                         MCOperand &AssignedValue,
                                         const BitVector &ForbiddenRegs) const {
    return make_error<Failure>(
        "targets with target-specific operands should implement this");
  }

  // Returns true if this instruction is supported as a back-to-back
  // instructions.
  // FIXME: Eventually we should discover this dynamically.
  virtual bool allowAsBackToBack(const Instruction &Instr) const {
    return true;
  }

  // For some instructions, it is interesting to measure how it's performance
  // characteristics differ depending on it's operands.
  // This allows us to produce all the interesting variants.
  virtual std::vector<InstructionTemplate>
  generateInstructionVariants(const Instruction &Instr,
                              unsigned MaxConfigsPerOpcode) const {
    // By default, we're happy with whatever randomizer will give us.
    return {&Instr};
  }

  // Creates a snippet generator for the given mode.
  std::unique_ptr<SnippetGenerator>
  createSnippetGenerator(InstructionBenchmark::ModeE Mode,
                         const LLVMState &State,
                         const SnippetGenerator::Options &Opts) const;
  // Creates a benchmark runner for the given mode.
  Expected<std::unique_ptr<BenchmarkRunner>>
  createBenchmarkRunner(InstructionBenchmark::ModeE Mode,
                        const LLVMState &State) const;

  // Returns the ExegesisTarget for the given triple or nullptr if the target
  // does not exist.
  static const ExegesisTarget *lookup(Triple TT);
  // Returns the default (unspecialized) ExegesisTarget.
  static const ExegesisTarget &getDefault();
  // Registers a target. Not thread safe.
  static void registerTarget(ExegesisTarget *T);

  virtual ~ExegesisTarget();

  // Returns the Pfm counters for the given CPU (or the default if no pfm
  // counters are defined for this CPU).
  const PfmCountersInfo &getPfmCounters(StringRef CpuName) const;

private:
  virtual bool matchesArch(Triple::ArchType Arch) const = 0;

  // Targets can implement their own snippet generators/benchmarks runners by
  // implementing these.
  std::unique_ptr<SnippetGenerator> virtual createSerialSnippetGenerator(
      const LLVMState &State, const SnippetGenerator::Options &Opts) const;
  std::unique_ptr<SnippetGenerator> virtual createParallelSnippetGenerator(
      const LLVMState &State, const SnippetGenerator::Options &Opts) const;
  std::unique_ptr<BenchmarkRunner> virtual createLatencyBenchmarkRunner(
      const LLVMState &State, InstructionBenchmark::ModeE Mode) const;
  std::unique_ptr<BenchmarkRunner> virtual createUopsBenchmarkRunner(
      const LLVMState &State) const;

  const ExegesisTarget *Next = nullptr;
  const ArrayRef<CpuAndPfmCounters> CpuPfmCounters;
};

} // namespace exegesis
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_EXEGESIS_TARGET_H
