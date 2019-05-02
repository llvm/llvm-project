//===--------------------- BottleneckAnalysis.h -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements the bottleneck analysis view.
/// 
/// This view internally observes backend pressure increase events in order to
/// identify potential sources of bottlenecks.
/// 
/// Example of bottleneck analysis report:
///
/// Cycles with backend pressure increase [ 33.40% ]
///  Throughput Bottlenecks:
///  Resource Pressure       [ 0.52% ]
///  - JLAGU  [ 0.52% ]
///  Data Dependencies:      [ 32.88% ]
///  - Register Dependencies [ 32.88% ]
///  - Memory Dependencies   [ 0.00% ]
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_BOTTLENECK_ANALYSIS_H
#define LLVM_TOOLS_LLVM_MCA_BOTTLENECK_ANALYSIS_H

#include "Views/View.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace mca {

/// A view that collects and prints a few performance numbers.
class BottleneckAnalysis : public View {
  const llvm::MCSchedModel &SM;
  unsigned TotalCycles;

  struct BackPressureInfo {
    // Cycles where backpressure increased.
    unsigned PressureIncreaseCycles;
    // Cycles where backpressure increased because of pipeline pressure.
    unsigned ResourcePressureCycles;
    // Cycles where backpressure increased because of data dependencies.
    unsigned DataDependencyCycles;
    // Cycles where backpressure increased because of register dependencies.
    unsigned RegisterDependencyCycles;
    // Cycles where backpressure increased because of memory dependencies.
    unsigned MemoryDependencyCycles;
  };
  BackPressureInfo BPI;

  // Resource pressure distribution. There is an element for every processor
  // resource declared by the scheduling model. Quantities are number of cycles.
  llvm::SmallVector<unsigned, 8> ResourcePressureDistribution;

  // Each processor resource is associated with a so-called processor resource
  // mask. This vector allows to correlate processor resource IDs with processor
  // resource masks. There is exactly one element per each processor resource
  // declared by the scheduling model.
  llvm::SmallVector<uint64_t, 8> ProcResourceMasks;

  // Used to map resource indices to actual processor resource IDs.
  llvm::SmallVector<unsigned, 8> ResIdx2ProcResID;

  // True if resource pressure events were notified during this cycle.
  bool PressureIncreasedBecauseOfResources;
  bool PressureIncreasedBecauseOfDataDependencies;

  // True if throughput was affected by dispatch stalls.
  bool SeenStallCycles;

  // Prints a bottleneck message to OS.
  void printBottleneckHints(llvm::raw_ostream &OS) const;

public:
  BottleneckAnalysis(const llvm::MCSchedModel &Model);

  void onCycleEnd() override {
    ++TotalCycles;
    if (PressureIncreasedBecauseOfResources ||
        PressureIncreasedBecauseOfDataDependencies) {
      ++BPI.PressureIncreaseCycles;
      if (PressureIncreasedBecauseOfDataDependencies)
        ++BPI.DataDependencyCycles;
      PressureIncreasedBecauseOfResources = false;
      PressureIncreasedBecauseOfDataDependencies = false;
    }
  }

  void onEvent(const HWStallEvent &Event) override { SeenStallCycles = true; }

  void onEvent(const HWPressureEvent &Event) override;

  void printView(llvm::raw_ostream &OS) const override;
};

} // namespace mca
} // namespace llvm

#endif
