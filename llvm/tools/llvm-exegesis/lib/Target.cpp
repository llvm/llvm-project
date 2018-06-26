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
  assert(Target->Next == nullptr && "target has already been registered");
  if (Target->Next != nullptr)
    return;
  Target->Next = FirstTarget;
  FirstTarget = Target;
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
