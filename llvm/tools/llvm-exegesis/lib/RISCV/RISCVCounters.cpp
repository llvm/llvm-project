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
#include <linux/sysctl.h>
#include <sys/types.h>

namespace llvm {
namespace exegesis {

// This implementation of RISCV target for Exegesis doesn't use libpfm
// and provides manual implementation of performance counters.

inline uint64_t getRISCVCpuCyclesCount() {
#if defined(__riscv) && defined(__linux__)
  uint32_t Cycle;
  size_t Length = sizeof(Cycle);
  if (!sysctlbyname("kernel.perf_user_access", &Cycle, &Length, NULL, 0) ||
      Cycle != 1)
    report_fatal_error(
        "Please write 'sudo echo 1 > /proc/sys/kernel/perf_user_access'");
#if __riscv_xlen == 32
  uint32_t cycles_lo, cycles_hi0, cycles_hi1;
  asm volatile("rdcycleh %0\n"
               "rdcycle %1\n"
               "rdcycleh %2\n"
               "sub %0, %0, %2\n"
               "seqz %0, %0\n"
               "sub %0, zero, %0\n"
               "and %1, %1, %0\n"
               : "=r"(cycles_hi0), "=r"(cycles_lo), "=r"(cycles_hi1));
  return static_cast<uint64_t>((static_cast<uint64_t>(cycles_hi1) << 32) |
                               cycles_lo);
#else
  uint64_t cycles;
  asm volatile("rdcycle %0" : "=r"(cycles));
  return static_cast<uint64_t>(cycles);
#endif
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
