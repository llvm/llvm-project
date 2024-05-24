//===-- RISCVCounters.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines RISC-V perf counters.
//
//===----------------------------------------------------------------------===//

#include "RISCVCounters.h"

namespace llvm {
namespace exegesis {

// This implementation of RISCV target for Exegesis doesn't use libpfm
// and provides manual implementation of performance counters.

inline uint64_t getRISCVCpuCyclesCount() {
#ifdef __riscv
  uint64_t Counter;
  asm("csrr %0, cycle" : "=r"(Counter)::"memory");
  return Counter;
#else
  return 0;
#endif
}

class RISCVCpuCyclesCounter : public pfm::CounterGroup {
  uint64_t StartValue;
  uint64_t EndValue;
  uint64_t MeasurementCycles;

public:
  explicit RISCVCpuCyclesCounter(pfm::PerfEvent &&Event);

  void start() override { StartValue = getRISCVCpuCyclesCount(); }

  void stop() override { EndValue = getRISCVCpuCyclesCount(); }

  Expected<llvm::SmallVector<int64_t, 4>>
  readOrError(StringRef FunctionBytes) const override;
};

RISCVCpuCyclesCounter::RISCVCpuCyclesCounter(pfm::PerfEvent &&Event)
    : CounterGroup(std::move(Event), {}) {
  StartValue = getRISCVCpuCyclesCount();
  EndValue = getRISCVCpuCyclesCount();
  MeasurementCycles = EndValue - StartValue;
  // If values of two calls CpuCyclesCounters don`t differ
  // it means that counters don`t configured properly, report error.
  // MeasurementCycles the smallest interval between two counter calls.
  if (MeasurementCycles == 0) {
    report_fatal_error("MeasurementCycles == 0, "
                       "performance counters are not configured.");
  }
  StartValue = EndValue = 0;
}

Expected<SmallVector<int64_t, 4>>
RISCVCpuCyclesCounter::readOrError(StringRef FunctionBytes) const {
  uint64_t Counter = EndValue - StartValue - MeasurementCycles;
  return SmallVector<int64_t, 4>({static_cast<int64_t>(Counter)});
}
std::unique_ptr<pfm::CounterGroup>
createRISCVCpuCyclesCounter(pfm::PerfEvent &&Event) {
  return std::make_unique<RISCVCpuCyclesCounter>(std::move(Event));
}

} // namespace exegesis
} // namespace llvm
