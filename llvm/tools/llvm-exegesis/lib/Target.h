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
/// similar to llvm::Target/TargetRegistry.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_EXEGESIS_TARGET_H
#define LLVM_TOOLS_LLVM_EXEGESIS_TARGET_H

#include "BenchmarkResult.h"
#include "BenchmarkRunner.h"
#include "LlvmState.h"
#include "SnippetGenerator.h"
#include "llvm/ADT/Triple.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"

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
  bool operator<(llvm::StringRef S) const {
    return llvm::StringRef(CpuName) < S;
  }
};

class ExegesisTarget {
public:
  explicit ExegesisTarget(llvm::ArrayRef<CpuAndPfmCounters> CpuPfmCounters)
      : CpuPfmCounters(CpuPfmCounters) {}

  // Targets can use this to add target-specific passes in assembleToStream();
  virtual void addTargetSpecificPasses(llvm::PassManagerBase &PM) const {}

  // Generates code to move a constant into a the given register.
  // Precondition: Value must fit into Reg.
  virtual std::vector<llvm::MCInst>
  setRegTo(const llvm::MCSubtargetInfo &STI, unsigned Reg,
           const llvm::APInt &Value) const = 0;

  // Returns the register pointing to scratch memory, or 0 if this target
  // does not support memory operands. The benchmark function uses the
  // default calling convention.
  virtual unsigned getScratchMemoryRegister(const llvm::Triple &) const {
    return 0;
  }

  // Fills memory operands with references to the address at [Reg] + Offset.
  virtual void fillMemoryOperands(InstructionTemplate &IT, unsigned Reg,
                                  unsigned Offset) const {

    llvm_unreachable(
        "fillMemoryOperands() requires getScratchMemoryRegister() > 0");
  }

  // Returns the maximum number of bytes a load/store instruction can access at
  // once. This is typically the size of the largest register available on the
  // processor. Note that this only used as a hint to generate independant
  // load/stores to/from memory, so the exact returned value does not really
  // matter as long as it's large enough.
  virtual unsigned getMaxMemoryAccessSize() const { return 0; }

  // Creates a snippet generator for the given mode.
  std::unique_ptr<SnippetGenerator>
  createSnippetGenerator(InstructionBenchmark::ModeE Mode,
                         const LLVMState &State) const;
  // Creates a benchmark runner for the given mode.
  std::unique_ptr<BenchmarkRunner>
  createBenchmarkRunner(InstructionBenchmark::ModeE Mode,
                        const LLVMState &State) const;

  // Returns the ExegesisTarget for the given triple or nullptr if the target
  // does not exist.
  static const ExegesisTarget *lookup(llvm::Triple TT);
  // Returns the default (unspecialized) ExegesisTarget.
  static const ExegesisTarget &getDefault();
  // Registers a target. Not thread safe.
  static void registerTarget(ExegesisTarget *T);

  virtual ~ExegesisTarget();

  // Returns the Pfm counters for the given CPU (or the default if no pfm
  // counters are defined for this CPU).
  const PfmCountersInfo &getPfmCounters(llvm::StringRef CpuName) const;

private:
  virtual bool matchesArch(llvm::Triple::ArchType Arch) const = 0;

  // Targets can implement their own snippet generators/benchmarks runners by
  // implementing these.
  std::unique_ptr<SnippetGenerator> virtual createLatencySnippetGenerator(
      const LLVMState &State) const;
  std::unique_ptr<SnippetGenerator> virtual createUopsSnippetGenerator(
      const LLVMState &State) const;
  std::unique_ptr<BenchmarkRunner> virtual createLatencyBenchmarkRunner(
      const LLVMState &State, InstructionBenchmark::ModeE Mode) const;
  std::unique_ptr<BenchmarkRunner> virtual createUopsBenchmarkRunner(
      const LLVMState &State) const;

  const ExegesisTarget *Next = nullptr;
  const llvm::ArrayRef<CpuAndPfmCounters> CpuPfmCounters;
};

} // namespace exegesis
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_EXEGESIS_TARGET_H
