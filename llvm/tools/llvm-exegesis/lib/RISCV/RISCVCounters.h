//===-- RISCVCounters.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// RISC-V perf counters.
///
/// More info at: https://lwn.net/Articles/680985
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_EXEGESIS_LIB_RISCV_RISCVCOUNTERS_H
#define LLVM_TOOLS_LLVM_EXEGESIS_LIB_RISCV_RISCVCOUNTERS_H

#include "../PerfHelper.h"
#include "../Target.h"
#include <memory>

namespace llvm {
namespace exegesis {

std::unique_ptr<pfm::CounterGroup>
createRISCVCpuCyclesCounter(pfm::PerfEvent &&Event);

} // namespace exegesis
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_EXEGESIS_LIB_RISCV_RISCVCOUNTERS_H
