//===-- Target.h ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/ADT/Triple.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCRegisterInfo.h"

namespace exegesis {

class ExegesisTarget {
public:
  // Targets can use this to add target-specific passes in assembleToStream();
  virtual void addTargetSpecificPasses(llvm::PassManagerBase &PM) const {}

  // Generates code to move a constant into a the given register.
  virtual std::vector<llvm::MCInst>
  setRegToConstant(const llvm::MCSubtargetInfo &STI, unsigned Reg) const {
    return {};
  }

  // Returns the register pointing to scratch memory, or 0 if this target does
  // not support memory operands. The benchmark function uses the default
  // calling convention.
  virtual unsigned getScratchMemoryRegister(const llvm::Triple &) const {
    return 0;
  }

  // Fills memory operands with references to the address at [Reg] + Offset.
  virtual void fillMemoryOperands(InstructionInstance &II, unsigned Reg,
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

private:
  virtual bool matchesArch(llvm::Triple::ArchType Arch) const = 0;

  // Targets can implement their own Latency/Uops benchmarks runners by
  // implementing these.
  std::unique_ptr<BenchmarkRunner> virtual createLatencyBenchmarkRunner(
      const LLVMState &State) const;
  std::unique_ptr<BenchmarkRunner> virtual createUopsBenchmarkRunner(
      const LLVMState &State) const;

  const ExegesisTarget *Next = nullptr;
};

} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_TARGET_H
