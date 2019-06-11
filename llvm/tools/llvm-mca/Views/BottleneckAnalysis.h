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
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace mca {

class PressureTracker {
  const MCSchedModel &SM;

  // Resource pressure distribution. There is an element for every processor
  // resource declared by the scheduling model. Quantities are number of cycles.
  SmallVector<unsigned, 4> ResourcePressureDistribution;

  // Each processor resource is associated with a so-called processor resource
  // mask. This vector allows to correlate processor resource IDs with processor
  // resource masks. There is exactly one element per each processor resource
  // declared by the scheduling model.
  SmallVector<uint64_t, 4> ProcResID2Mask;

  // Maps processor resource state indices (returned by calls to
  // `getResourceStateIndex(Mask)` to processor resource identifiers.
  SmallVector<unsigned, 4> ResIdx2ProcResID;

  // Maps Processor Resource identifiers to ResourceUsers indices.
  SmallVector<unsigned, 4> ProcResID2ResourceUsersIndex;

  // Identifies the last user of a processor resource unit.
  // This vector is updated on every instruction issued event.
  // There is one entry for every processor resource unit declared by the
  // processor model. An all_ones value is treated like an invalid instruction
  // identifier.
  using User = std::pair<unsigned, unsigned>;
  SmallVector<User, 4> ResourceUsers;

  struct InstructionPressureInfo {
    unsigned RegisterPressureCycles;
    unsigned MemoryPressureCycles;
    unsigned ResourcePressureCycles;
  };
  DenseMap<unsigned, InstructionPressureInfo> IPI;

  void updateResourcePressureDistribution(uint64_t CumulativeMask);

  User getResourceUser(unsigned ProcResID, unsigned UnitID) const {
    unsigned Index = ProcResID2ResourceUsersIndex[ProcResID];
    return ResourceUsers[Index + UnitID];
  }

public:
  PressureTracker(const MCSchedModel &Model);

  ArrayRef<unsigned> getResourcePressureDistribution() const {
    return ResourcePressureDistribution;
  }

  void getResourceUsers(uint64_t ResourceMask,
                        SmallVectorImpl<User> &Users) const;

  unsigned getRegisterPressureCycles(unsigned IID) const {
    assert(IPI.find(IID) != IPI.end() && "Instruction is not tracked!");
    const InstructionPressureInfo &Info = IPI.find(IID)->second;
    return Info.RegisterPressureCycles;
  }

  unsigned getMemoryPressureCycles(unsigned IID) const {
    assert(IPI.find(IID) != IPI.end() && "Instruction is not tracked!");
    const InstructionPressureInfo &Info = IPI.find(IID)->second;
    return Info.MemoryPressureCycles;
  }

  unsigned getResourcePressureCycles(unsigned IID) const {
    assert(IPI.find(IID) != IPI.end() && "Instruction is not tracked!");
    const InstructionPressureInfo &Info = IPI.find(IID)->second;
    return Info.ResourcePressureCycles;
  }

  void onInstructionDispatched(unsigned IID);
  void onInstructionExecuted(unsigned IID);

  void handlePressureEvent(const HWPressureEvent &Event);
  void handleInstructionIssuedEvent(const HWInstructionIssuedEvent &Event);
};

class DependencyGraph {
  enum DependencyType { DT_REGISTER, DT_MEMORY, DT_RESOURCE };

  struct DependencyEdge {
    DependencyType Type;
    unsigned IID;
    uint64_t ResourceOrRegID;
    uint64_t Cycles;
  };

  struct DGNode {
    unsigned NumPredecessors;
    SmallVector<DependencyEdge, 8> OutgoingEdges;
  };
  SmallVector<DGNode, 16> Nodes;

  DependencyGraph(const DependencyGraph &) = delete;
  DependencyGraph &operator=(const DependencyGraph &) = delete;

  void addDependency(unsigned From, DependencyEdge &&DE);

#ifndef NDEBUG
  void dumpDependencyEdge(raw_ostream &OS, unsigned FromIID,
                          const DependencyEdge &DE, MCInstPrinter &MCIP) const;
#endif

public:
  DependencyGraph(unsigned Size) : Nodes(Size) {}

  void addRegisterDep(unsigned From, unsigned To, unsigned RegID, unsigned Cy) {
    addDependency(From, {DT_REGISTER, To, RegID, Cy});
  }

  void addMemoryDep(unsigned From, unsigned To, unsigned Cy) {
    addDependency(From, {DT_MEMORY, To, /* unused */ 0, Cy});
  }

  void addResourceDep(unsigned From, unsigned To, uint64_t Mask, unsigned Cy) {
    addDependency(From, {DT_RESOURCE, To, Mask, Cy});
  }

#ifndef NDEBUG
  void dump(raw_ostream &OS, MCInstPrinter &MCIP) const;
#endif
};

/// A view that collects and prints a few performance numbers.
class BottleneckAnalysis : public View {
  const MCSubtargetInfo &STI;
  PressureTracker Tracker;
  DependencyGraph DG;

  ArrayRef<MCInst> Source;
  unsigned TotalCycles;

  bool PressureIncreasedBecauseOfResources;
  bool PressureIncreasedBecauseOfRegisterDependencies;
  bool PressureIncreasedBecauseOfMemoryDependencies;
  // True if throughput was affected by dispatch stalls.
  bool SeenStallCycles;

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

  // Prints a bottleneck message to OS.
  void printBottleneckHints(raw_ostream &OS) const;

  // Used to populate the dependency graph DG.
  void addRegisterDep(unsigned From, unsigned To, unsigned RegID, unsigned Cy);
  void addMemoryDep(unsigned From, unsigned To, unsigned Cy);
  void addResourceDep(unsigned From, unsigned To, uint64_t Mask, unsigned Cy);

public:
  BottleneckAnalysis(const MCSubtargetInfo &STI, MCInstPrinter &MCIP,
                     ArrayRef<MCInst> Sequence);

  void onCycleEnd() override;
  void onEvent(const HWStallEvent &Event) override { SeenStallCycles = true; }
  void onEvent(const HWPressureEvent &Event) override;
  void onEvent(const HWInstructionEvent &Event) override;

  void printView(raw_ostream &OS) const override;

#ifndef NDEBUG
  void dump(raw_ostream &OS, MCInstPrinter &MCIP) const { DG.dump(OS, MCIP); }
#endif
};

} // namespace mca
} // namespace llvm

#endif
