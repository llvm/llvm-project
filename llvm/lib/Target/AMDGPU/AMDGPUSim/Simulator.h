//===- AMDGPUSim/Simulator.h - Single-wave simulator ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Defines the representation independent single wave simulator interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_SIMULATOR_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUSIM_SIMULATOR_H

#include "HWModel.h"
#include "InstrInfo.h"
#include "SimInstInfo.h"
#include "SimState.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

namespace llvm {
namespace AMDGPUSim {

/// Optional diagnostic logging configuration.
/// Log is used only when Verbose is true.
struct SimulatorConfig {
  bool Verbose = false;
  raw_ostream *Log = nullptr;
};

/// Simulates one wave in program order and retains state between instructions.
class Simulator {
  class Impl;
  std::unique_ptr<Impl> PImpl;

public:
  Simulator(const SimInstInfo &II, const HWModel &Model,
            SimulatorConfig Config = {});
  ~Simulator();

  /// Simulate \p Inst at the current cycle and update simulator state.
  /// The instruction must originate from the configured adapter.
  /// \p Lookahead supplies the next instruction when determining whether an
  /// exposed S_SET_VGPR_MSB stall is masked.
  InstrSimInfo simulateInst(const SimInst &Inst,
                            ArrayRef<SimInst> Lookahead = {});

  /// Advance without issuing an instruction.
  /// Resources that complete during the interval are retired.
  void advanceCycles(unsigned Count);

  /// Return the current single wave simulation state.
  const GPUSimState &getState() const;

  /// Return the simulator configuration.
  const SimulatorConfig &getConfig() const;

  /// Return the hardware model used by the simulator.
  const HWModel &getModel() const;
};

} // namespace AMDGPUSim
} // namespace llvm

#endif
