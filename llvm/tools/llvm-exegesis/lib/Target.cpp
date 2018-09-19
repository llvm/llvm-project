//===-- Target.cpp ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "Target.h"

#include "Latency.h"
#include "Uops.h"

namespace exegesis {

ExegesisTarget::~ExegesisTarget() {} // anchor.

static ExegesisTarget *FirstTarget = nullptr;

const ExegesisTarget *ExegesisTarget::lookup(llvm::Triple TT) {
  for (const ExegesisTarget *T = FirstTarget; T != nullptr; T = T->Next) {
    if (T->matchesArch(TT.getArch()))
      return T;
  }
  return nullptr;
}

void ExegesisTarget::registerTarget(ExegesisTarget *Target) {
  if (FirstTarget == nullptr) {
    FirstTarget = Target;
    return;
  }
  if (Target->Next != nullptr)
    return; // Already registered.
  Target->Next = FirstTarget;
  FirstTarget = Target;
}

std::unique_ptr<SnippetGenerator>
ExegesisTarget::createSnippetGenerator(InstructionBenchmark::ModeE Mode,
                                       const LLVMState &State) const {
  switch (Mode) {
  case InstructionBenchmark::Unknown:
    return nullptr;
  case InstructionBenchmark::Latency:
    return createLatencySnippetGenerator(State);
  case InstructionBenchmark::Uops:
    return createUopsSnippetGenerator(State);
  }
  return nullptr;
}

std::unique_ptr<BenchmarkRunner>
ExegesisTarget::createBenchmarkRunner(InstructionBenchmark::ModeE Mode,
                                      const LLVMState &State) const {
  switch (Mode) {
  case InstructionBenchmark::Unknown:
    return nullptr;
  case InstructionBenchmark::Latency:
    return createLatencyBenchmarkRunner(State);
  case InstructionBenchmark::Uops:
    return createUopsBenchmarkRunner(State);
  }
  return nullptr;
}

std::unique_ptr<SnippetGenerator>
ExegesisTarget::createLatencySnippetGenerator(const LLVMState &State) const {
  return llvm::make_unique<LatencySnippetGenerator>(State);
}

std::unique_ptr<SnippetGenerator>
ExegesisTarget::createUopsSnippetGenerator(const LLVMState &State) const {
  return llvm::make_unique<UopsSnippetGenerator>(State);
}

std::unique_ptr<BenchmarkRunner>
ExegesisTarget::createLatencyBenchmarkRunner(const LLVMState &State) const {
  return llvm::make_unique<LatencyBenchmarkRunner>(State);
}

std::unique_ptr<BenchmarkRunner>
ExegesisTarget::createUopsBenchmarkRunner(const LLVMState &State) const {
  return llvm::make_unique<UopsBenchmarkRunner>(State);
}

namespace {

// Default implementation.
class ExegesisDefaultTarget : public ExegesisTarget {
private:
  std::vector<llvm::MCInst> setRegToConstant(const llvm::MCSubtargetInfo &STI,
                                             unsigned Reg) const override {
    llvm_unreachable("Not yet implemented");
  }

  std::vector<llvm::MCInst> setRegTo(const llvm::MCSubtargetInfo &STI,
                                     const llvm::APInt &Value,
                                     unsigned Reg) const override {
    llvm_unreachable("Not yet implemented");
  }

  unsigned getScratchMemoryRegister(const llvm::Triple &) const override {
    llvm_unreachable("Not yet implemented");
  }

  void fillMemoryOperands(InstructionBuilder &IB, unsigned Reg,
                          unsigned Offset) const override {
    llvm_unreachable("Not yet implemented");
  }

  unsigned getMaxMemoryAccessSize() const override {
    llvm_unreachable("Not yet implemented");
  }

  bool matchesArch(llvm::Triple::ArchType Arch) const override {
    llvm_unreachable("never called");
    return false;
  }
};

} // namespace

const ExegesisTarget &ExegesisTarget::getDefault() {
  static ExegesisDefaultTarget Target;
  return Target;
}

} // namespace exegesis
