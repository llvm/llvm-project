//===-- Target.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "../Target.h"
#include "../Latency.h"
#include "AArch64.h"

namespace exegesis {

namespace {

class AArch64LatencyBenchmarkRunner : public LatencyBenchmarkRunner {
public:
  AArch64LatencyBenchmarkRunner(const LLVMState &State)
      : LatencyBenchmarkRunner(State) {}

private:
  const char *getCounterName() const override {
    // All AArch64 subtargets have CPU_CYCLES as the cycle counter name
    return "CPU_CYCLES";
  }
};

class ExegesisAArch64Target : public ExegesisTarget {
  bool matchesArch(llvm::Triple::ArchType Arch) const override {
    return Arch == llvm::Triple::aarch64 || Arch == llvm::Triple::aarch64_be;
  }
  void addTargetSpecificPasses(llvm::PassManagerBase &PM) const override {
    // Function return is a pseudo-instruction that needs to be expanded
    PM.add(llvm::createAArch64ExpandPseudoPass());
  }
  std::unique_ptr<BenchmarkRunner>
  createLatencyBenchmarkRunner(const LLVMState &State) const override {
    return llvm::make_unique<AArch64LatencyBenchmarkRunner>(State);
  }
};

} // namespace

static ExegesisTarget *getTheExegesisAArch64Target() {
  static ExegesisAArch64Target Target;
  return &Target;
}

void InitializeAArch64ExegesisTarget() {
  ExegesisTarget::registerTarget(getTheExegesisAArch64Target());
}

} // namespace exegesis
