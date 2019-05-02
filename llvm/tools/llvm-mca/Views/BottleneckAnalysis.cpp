//===--------------------- BottleneckAnalysis.cpp ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements the functionalities used by the BottleneckAnalysis
/// to report bottleneck info.
///
//===----------------------------------------------------------------------===//

#include "Views/BottleneckAnalysis.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MCA/Support.h"
#include "llvm/Support/Format.h"

namespace llvm {
namespace mca {

#define DEBUG_TYPE "llvm-mca"

BottleneckAnalysis::BottleneckAnalysis(const MCSchedModel &Model)
    : SM(Model), TotalCycles(0), BPI({0, 0, 0, 0, 0}),
      ResourcePressureDistribution(Model.getNumProcResourceKinds(), 0),
      ProcResourceMasks(Model.getNumProcResourceKinds()),
      ResIdx2ProcResID(Model.getNumProcResourceKinds(), 0),
      PressureIncreasedBecauseOfResources(false),
      PressureIncreasedBecauseOfDataDependencies(false),
      SeenStallCycles(false) {
  computeProcResourceMasks(SM, ProcResourceMasks);
  for (unsigned I = 1, E = SM.getNumProcResourceKinds(); I < E; ++I) {
    unsigned Index = getResourceStateIndex(ProcResourceMasks[I]);
    ResIdx2ProcResID[Index] = I;
  }
}

void BottleneckAnalysis::onEvent(const HWPressureEvent &Event) {
  assert(Event.Reason != HWPressureEvent::INVALID &&
         "Unexpected invalid event!");

  switch (Event.Reason) {
  default:
    break;

  case HWPressureEvent::RESOURCES: {
    PressureIncreasedBecauseOfResources = true;
    ++BPI.ResourcePressureCycles;
    uint64_t ResourceMask = Event.ResourceMask;
    while (ResourceMask) {
      uint64_t Current = ResourceMask & (-ResourceMask);
      unsigned Index = getResourceStateIndex(Current);
      unsigned ProcResID = ResIdx2ProcResID[Index];
      const MCProcResourceDesc &PRDesc = *SM.getProcResource(ProcResID);
      if (!PRDesc.SubUnitsIdxBegin) {
        ResourcePressureDistribution[Index]++;
        ResourceMask ^= Current;
        continue;
      }

      for (unsigned I = 0, E = PRDesc.NumUnits; I < E; ++I) {
        unsigned OtherProcResID = PRDesc.SubUnitsIdxBegin[I];
        unsigned OtherMask = ProcResourceMasks[OtherProcResID];
        ResourcePressureDistribution[getResourceStateIndex(OtherMask)]++;
      }

      ResourceMask ^= Current;
    }
    break;
  }

  case HWPressureEvent::REGISTER_DEPS:
    PressureIncreasedBecauseOfDataDependencies = true;
    ++BPI.RegisterDependencyCycles;
    break;
  case HWPressureEvent::MEMORY_DEPS:
    PressureIncreasedBecauseOfDataDependencies = true;
    ++BPI.MemoryDependencyCycles;
    break;
  }
}

void BottleneckAnalysis::printBottleneckHints(raw_ostream &OS) const {
  if (!SeenStallCycles || !BPI.PressureIncreaseCycles) {
    OS << "\nNo resource or data dependency bottlenecks discovered.\n";
    return;
  }

  double PressurePerCycle =
      (double)BPI.PressureIncreaseCycles * 100 / TotalCycles;
  double ResourcePressurePerCycle =
      (double)BPI.ResourcePressureCycles * 100 / TotalCycles;
  double DDPerCycle = (double)BPI.DataDependencyCycles * 100 / TotalCycles;
  double RegDepPressurePerCycle =
      (double)BPI.RegisterDependencyCycles * 100 / TotalCycles;
  double MemDepPressurePerCycle =
      (double)BPI.MemoryDependencyCycles * 100 / TotalCycles;

  OS << "\nCycles with backend pressure increase [ "
     << format("%.2f", floor((PressurePerCycle * 100) + 0.5) / 100) << "% ]";

  OS << "\nThroughput Bottlenecks: "
     << "\n  Resource Pressure       [ "
     << format("%.2f", floor((ResourcePressurePerCycle * 100) + 0.5) / 100)
     << "% ]";

  if (BPI.PressureIncreaseCycles) {
    for (unsigned I = 0, E = ResourcePressureDistribution.size(); I < E; ++I) {
      if (ResourcePressureDistribution[I]) {
        double Frequency =
            (double)ResourcePressureDistribution[I] * 100 / TotalCycles;
        unsigned Index = ResIdx2ProcResID[getResourceStateIndex(1ULL << I)];
        const MCProcResourceDesc &PRDesc = *SM.getProcResource(Index);
        OS << "\n  - " << PRDesc.Name << "  [ "
           << format("%.2f", floor((Frequency * 100) + 0.5) / 100) << "% ]";
      }
    }
  }

  OS << "\n  Data Dependencies:      [ "
     << format("%.2f", floor((DDPerCycle * 100) + 0.5) / 100) << "% ]";

  OS << "\n  - Register Dependencies [ "
     << format("%.2f", floor((RegDepPressurePerCycle * 100) + 0.5) / 100)
     << "% ]";

  OS << "\n  - Memory Dependencies   [ "
     << format("%.2f", floor((MemDepPressurePerCycle * 100) + 0.5) / 100)
     << "% ]\n\n";
}

void BottleneckAnalysis::printView(raw_ostream &OS) const {
  std::string Buffer;
  raw_string_ostream TempStream(Buffer);
  printBottleneckHints(TempStream);
  TempStream.flush();
  OS << Buffer;
}
} // namespace mca.
} // namespace llvm
