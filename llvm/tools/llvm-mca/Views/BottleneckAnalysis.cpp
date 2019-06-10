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
#include "llvm/MC/MCInst.h"
#include "llvm/MCA/Support.h"
#include "llvm/Support/Format.h"

namespace llvm {
namespace mca {

#define DEBUG_TYPE "llvm-mca"

PressureTracker::PressureTracker(const MCSchedModel &Model)
    : SM(Model),
      ResourcePressureDistribution(Model.getNumProcResourceKinds(), 0),
      ProcResID2Mask(Model.getNumProcResourceKinds(), 0),
      ResIdx2ProcResID(Model.getNumProcResourceKinds(), 0),
      ProcResID2ResourceUsersIndex(Model.getNumProcResourceKinds(), 0) {
  computeProcResourceMasks(SM, ProcResID2Mask);

  // Ignore the invalid resource at index zero.
  unsigned NextResourceUsersIdx = 0;
  for (unsigned I = 1, E = Model.getNumProcResourceKinds(); I < E; ++I) {
    const MCProcResourceDesc &ProcResource = *SM.getProcResource(I);
    ProcResID2ResourceUsersIndex[I] = NextResourceUsersIdx;
    NextResourceUsersIdx += ProcResource.NumUnits;
    uint64_t ResourceMask = ProcResID2Mask[I];
    ResIdx2ProcResID[getResourceStateIndex(ResourceMask)] = I;
  }

  ResourceUsers.resize(NextResourceUsersIdx);
  std::fill(ResourceUsers.begin(), ResourceUsers.end(),
            std::make_pair<unsigned, unsigned>(~0U, 0U));
}

void PressureTracker::getResourceUsers(uint64_t ResourceMask,
                                       SmallVectorImpl<User> &Users) const {
  unsigned Index = getResourceStateIndex(ResourceMask);
  unsigned ProcResID = ResIdx2ProcResID[Index];
  const MCProcResourceDesc &PRDesc = *SM.getProcResource(ProcResID);
  for (unsigned I = 0, E = PRDesc.NumUnits; I < E; ++I) {
    const User U = getResourceUser(ProcResID, I);
    if (U.second && IPI.find(U.first) != IPI.end())
      Users.emplace_back(U);
  }
}

void PressureTracker::onInstructionDispatched(unsigned IID) {
  IPI.insert(std::make_pair(IID, InstructionPressureInfo()));
}

void PressureTracker::onInstructionExecuted(unsigned IID) { IPI.erase(IID); }

void PressureTracker::handleInstructionIssuedEvent(
    const HWInstructionIssuedEvent &Event) {
  unsigned IID = Event.IR.getSourceIndex();
  using ResourceRef = HWInstructionIssuedEvent::ResourceRef;
  using ResourceUse = std::pair<ResourceRef, ResourceCycles>;
  for (const ResourceUse &Use : Event.UsedResources) {
    const ResourceRef &RR = Use.first;
    unsigned Index = ProcResID2ResourceUsersIndex[RR.first];
    Index += countTrailingZeros(RR.second);
    ResourceUsers[Index] = std::make_pair(IID, Use.second.getNumerator());
  }
}

void PressureTracker::updateResourcePressureDistribution(
    uint64_t CumulativeMask) {
  while (CumulativeMask) {
    uint64_t Current = CumulativeMask & (-CumulativeMask);
    unsigned ResIdx = getResourceStateIndex(Current);
    unsigned ProcResID = ResIdx2ProcResID[ResIdx];
    uint64_t Mask = ProcResID2Mask[ProcResID];

    if (Mask == Current) {
      ResourcePressureDistribution[ProcResID]++;
      CumulativeMask ^= Current;
      continue;
    }

    Mask ^= Current;
    while (Mask) {
      uint64_t SubUnit = Mask & (-Mask);
      ResIdx = getResourceStateIndex(SubUnit);
      ProcResID = ResIdx2ProcResID[ResIdx];
      ResourcePressureDistribution[ProcResID]++;
      Mask ^= SubUnit;
    }

    CumulativeMask ^= Current;
  }
}

void PressureTracker::handlePressureEvent(const HWPressureEvent &Event) {
  assert(Event.Reason != HWPressureEvent::INVALID &&
         "Unexpected invalid event!");

  switch (Event.Reason) {
  default:
    break;

  case HWPressureEvent::RESOURCES: {
    const uint64_t ResourceMask = Event.ResourceMask;
    updateResourcePressureDistribution(Event.ResourceMask);

    for (const InstRef &IR : Event.AffectedInstructions) {
      const Instruction &IS = *IR.getInstruction();
      unsigned BusyResources = IS.getCriticalResourceMask() & ResourceMask;
      if (!BusyResources)
        continue;

      unsigned IID = IR.getSourceIndex();
      IPI[IID].ResourcePressureCycles++;
    }
    break;
  }

  case HWPressureEvent::REGISTER_DEPS:
    for (const InstRef &IR : Event.AffectedInstructions) {
      unsigned IID = IR.getSourceIndex();
      IPI[IID].RegisterPressureCycles++;
    }
    break;

  case HWPressureEvent::MEMORY_DEPS:
    for (const InstRef &IR : Event.AffectedInstructions) {
      unsigned IID = IR.getSourceIndex();
      IPI[IID].MemoryPressureCycles++;
    }
  }
}

#ifndef NDEBUG
void DependencyGraph::dumpDependencyEdge(raw_ostream &OS, unsigned FromIID,
                                         const DependencyEdge &DE,
                                         MCInstPrinter &MCIP) const {
  bool LoopCarried = FromIID >= DE.IID;
  OS << " FROM: " << FromIID << " TO: " << DE.IID
     << (LoopCarried ? " (loop carried)" : "             ");
  if (DE.Type == DT_REGISTER) {
    OS << " - REGISTER: ";
    MCIP.printRegName(OS, DE.ResourceOrRegID);
  } else if (DE.Type == DT_MEMORY) {
    OS << " - MEMORY";
  } else {
    assert(DE.Type == DT_RESOURCE && "Unexpected unsupported dependency type!");
    OS << " - RESOURCE MASK: " << DE.ResourceOrRegID;
  }
  OS << " - CYCLES: " << DE.Cycles << '\n';
}

void DependencyGraph::dump(raw_ostream &OS, MCInstPrinter &MCIP) const {
  OS << "\nREG DEPS\n";
  for (unsigned I = 0, E = Nodes.size(); I < E; ++I) {
    const DGNode &Node = Nodes[I];
    for (const DependencyEdge &DE : Node.OutgoingEdges) {
      if (DE.Type == DT_REGISTER)
        dumpDependencyEdge(OS, I, DE, MCIP);
    }
  }

  OS << "\nMEM DEPS\n";
  for (unsigned I = 0, E = Nodes.size(); I < E; ++I) {
    const DGNode &Node = Nodes[I];
    for (const DependencyEdge &DE : Node.OutgoingEdges) {
      if (DE.Type == DT_MEMORY)
        dumpDependencyEdge(OS, I, DE, MCIP);
    }
  }

  OS << "\nRESOURCE DEPS\n";
  for (unsigned I = 0, E = Nodes.size(); I < E; ++I) {
    const DGNode &Node = Nodes[I];
    for (const DependencyEdge &DE : Node.OutgoingEdges) {
      if (DE.Type == DT_RESOURCE)
        dumpDependencyEdge(OS, I, DE, MCIP);
    }
  }
}
#endif // NDEBUG

void DependencyGraph::addDependency(unsigned From, DependencyEdge &&Dep) {
  DGNode &NodeFrom = Nodes[From];
  DGNode &NodeTo = Nodes[Dep.IID];
  SmallVectorImpl<DependencyEdge> &Vec = NodeFrom.OutgoingEdges;

  auto It = find_if(Vec, [Dep](DependencyEdge &DE) {
    return DE.IID == Dep.IID && DE.ResourceOrRegID == Dep.ResourceOrRegID;
  });

  if (It != Vec.end()) {
    It->Cycles += Dep.Cycles;
    return;
  }

  Vec.emplace_back(Dep);
  NodeTo.NumPredecessors++;
}

BottleneckAnalysis::BottleneckAnalysis(const MCSubtargetInfo &sti,
                                       MCInstPrinter &Printer,
                                       ArrayRef<MCInst> S)
    : STI(sti), Tracker(STI.getSchedModel()), DG(S.size() * 3),
      Source(S), TotalCycles(0), PressureIncreasedBecauseOfResources(false),
      PressureIncreasedBecauseOfRegisterDependencies(false),
      PressureIncreasedBecauseOfMemoryDependencies(false),
      SeenStallCycles(false), BPI() {}

void BottleneckAnalysis::addRegisterDep(unsigned From, unsigned To,
                                        unsigned RegID, unsigned Cy) {
  bool IsLoopCarried = From >= To;
  unsigned SourceSize = Source.size();
  if (IsLoopCarried) {
    DG.addRegisterDep(From, To + SourceSize, RegID, Cy);
    DG.addRegisterDep(From + SourceSize, To + (SourceSize * 2), RegID, Cy);
    return;
  }
  DG.addRegisterDep(From + SourceSize, To + SourceSize, RegID, Cy);
}

void BottleneckAnalysis::addMemoryDep(unsigned From, unsigned To, unsigned Cy) {
  bool IsLoopCarried = From >= To;
  unsigned SourceSize = Source.size();
  if (IsLoopCarried) {
    DG.addMemoryDep(From, To + SourceSize, Cy);
    DG.addMemoryDep(From + SourceSize, To + (SourceSize * 2), Cy);
    return;
  }
  DG.addMemoryDep(From + SourceSize, To + SourceSize, Cy);
}

void BottleneckAnalysis::addResourceDep(unsigned From, unsigned To,
                                        uint64_t Mask, unsigned Cy) {
  bool IsLoopCarried = From >= To;
  unsigned SourceSize = Source.size();
  if (IsLoopCarried) {
    DG.addResourceDep(From, To + SourceSize, Mask, Cy);
    DG.addResourceDep(From + SourceSize, To + (SourceSize * 2), Mask, Cy);
    return;
  }
  DG.addResourceDep(From + SourceSize, To + SourceSize, Mask, Cy);
}

void BottleneckAnalysis::onEvent(const HWInstructionEvent &Event) {
  const unsigned IID = Event.IR.getSourceIndex();
  if (Event.Type == HWInstructionEvent::Dispatched) {
    Tracker.onInstructionDispatched(IID);
    return;
  }
  if (Event.Type == HWInstructionEvent::Executed) {
    Tracker.onInstructionExecuted(IID);
    return;
  }

  if (Event.Type != HWInstructionEvent::Issued)
    return;

  const Instruction &IS = *Event.IR.getInstruction();
  unsigned To = IID % Source.size();

  unsigned Cycles = Tracker.getResourcePressureCycles(IID);
  if (Cycles) {
    uint64_t ResourceMask = IS.getCriticalResourceMask();
    SmallVector<std::pair<unsigned, unsigned>, 4> Users;
    while (ResourceMask) {
      uint64_t Current = ResourceMask & (-ResourceMask);
      Tracker.getResourceUsers(Current, Users);
      for (const std::pair<unsigned, unsigned> &U : Users) {
        unsigned Cost = std::min(U.second, Cycles);
        addResourceDep(U.first % Source.size(), To, Current, Cost);
      }
      Users.clear();
      ResourceMask ^= Current;
    }
  }

  Cycles = Tracker.getRegisterPressureCycles(IID);
  if (Cycles) {
    const CriticalDependency &RegDep = IS.getCriticalRegDep();
    unsigned From = RegDep.IID % Source.size();
    addRegisterDep(From, To, RegDep.RegID, Cycles);
  }

  Cycles = Tracker.getMemoryPressureCycles(IID);
  if (Cycles) {
    const CriticalDependency &MemDep = IS.getCriticalMemDep();
    unsigned From = MemDep.IID % Source.size();
    addMemoryDep(From, To, Cycles);
  }

  Tracker.handleInstructionIssuedEvent(
      static_cast<const HWInstructionIssuedEvent &>(Event));
}

void BottleneckAnalysis::onEvent(const HWPressureEvent &Event) {
  assert(Event.Reason != HWPressureEvent::INVALID &&
         "Unexpected invalid event!");

  Tracker.handlePressureEvent(Event);

  switch (Event.Reason) {
  default:
    break;

  case HWPressureEvent::RESOURCES:
    PressureIncreasedBecauseOfResources = true;
    break;
  case HWPressureEvent::REGISTER_DEPS:
    PressureIncreasedBecauseOfRegisterDependencies = true;
    break;
  case HWPressureEvent::MEMORY_DEPS:
    PressureIncreasedBecauseOfMemoryDependencies = true;
    break;
  }
}

void BottleneckAnalysis::onCycleEnd() {
  ++TotalCycles;

  bool PressureIncreasedBecauseOfDataDependencies =
      PressureIncreasedBecauseOfRegisterDependencies ||
      PressureIncreasedBecauseOfMemoryDependencies;
  if (!PressureIncreasedBecauseOfResources &&
      !PressureIncreasedBecauseOfDataDependencies)
    return;

  ++BPI.PressureIncreaseCycles;
  if (PressureIncreasedBecauseOfRegisterDependencies)
    ++BPI.RegisterDependencyCycles;
  if (PressureIncreasedBecauseOfMemoryDependencies)
    ++BPI.MemoryDependencyCycles;
  if (PressureIncreasedBecauseOfDataDependencies)
    ++BPI.DataDependencyCycles;
  if (PressureIncreasedBecauseOfResources)
    ++BPI.ResourcePressureCycles;
  PressureIncreasedBecauseOfResources = false;
  PressureIncreasedBecauseOfRegisterDependencies = false;
  PressureIncreasedBecauseOfMemoryDependencies = false;
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
    ArrayRef<unsigned> Distribution = Tracker.getResourcePressureDistribution();
    const MCSchedModel &SM = STI.getSchedModel();
    for (unsigned I = 0, E = Distribution.size(); I < E; ++I) {
      unsigned ResourceCycles = Distribution[I];
      if (ResourceCycles) {
        double Frequency = (double)ResourceCycles * 100 / TotalCycles;
        const MCProcResourceDesc &PRDesc = *SM.getProcResource(I);
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
