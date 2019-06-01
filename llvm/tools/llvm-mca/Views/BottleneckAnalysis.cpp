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
  std::fill(ResourceUsers.begin(), ResourceUsers.end(), ~0U);
}

void PressureTracker::getUniqueUsers(
    uint64_t ResourceMask, SmallVectorImpl<unsigned> &UniqueUsers) const {
  unsigned Index = getResourceStateIndex(ResourceMask);
  unsigned ProcResID = ResIdx2ProcResID[Index];
  const MCProcResourceDesc &PRDesc = *SM.getProcResource(ProcResID);
  for (unsigned I = 0, E = PRDesc.NumUnits; I < E; ++I) {
    unsigned From = getResourceUser(ProcResID, I);
    if (find(UniqueUsers, From) == UniqueUsers.end())
      UniqueUsers.emplace_back(From);
  }
}

void PressureTracker::handleInstructionEvent(const HWInstructionEvent &Event) {
  unsigned IID = Event.IR.getSourceIndex();
  switch (Event.Type) {
  default:
    break;
  case HWInstructionEvent::Dispatched:
    IPI.insert(std::make_pair(IID, InstructionPressureInfo()));
    break;
  case HWInstructionEvent::Executed:
    IPI.erase(IID);
    break;
  case HWInstructionEvent::Issued: {
    const auto &IIE = static_cast<const HWInstructionIssuedEvent &>(Event);
    using ResourceRef = HWInstructionIssuedEvent::ResourceRef;
    using ResourceUse = std::pair<ResourceRef, ResourceCycles>;
    for (const ResourceUse &Use : IIE.UsedResources) {
      const ResourceRef &RR = Use.first;
      unsigned Index = ProcResID2ResourceUsersIndex[RR.first];
      Index += countTrailingZeros(RR.second);
      ResourceUsers[Index] = IID;
    }
  }
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

      IPI[IR.getSourceIndex()].ResourcePressureCycles++;
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
void DependencyGraph::dumpRegDeps(raw_ostream &OS, MCInstPrinter &MCIP) const {
  OS << "\nREG DEPS\n";
  for (unsigned I = 0, E = Nodes.size(); I < E; ++I) {
    const DGNode &Node = Nodes[I];
    for (const DependencyEdge &DE : Node.RegDeps) {
      bool LoopCarried = I >= DE.IID;
      OS << " FROM: " << I << " TO: " << DE.IID
         << (LoopCarried ? " (loop carried)" : "             ")
         << " - REGISTER: ";
      MCIP.printRegName(OS, DE.ResourceOrRegID);
      OS << " - CYCLES: " << DE.Cycles << '\n';
    }
  }
}

void DependencyGraph::dumpMemDeps(raw_ostream &OS) const {
  OS << "\nMEM DEPS\n";
  for (unsigned I = 0, E = Nodes.size(); I < E; ++I) {
    const DGNode &Node = Nodes[I];
    for (const DependencyEdge &DE : Node.MemDeps) {
      bool LoopCarried = I >= DE.IID;
      OS << " FROM: " << I << " TO: " << DE.IID
         << (LoopCarried ? " (loop carried)" : "             ")
         << " - MEMORY - CYCLES: " << DE.Cycles << '\n';
    }
  }
}

void DependencyGraph::dumpResDeps(raw_ostream &OS) const {
  OS << "\nRESOURCE DEPS\n";
  for (unsigned I = 0, E = Nodes.size(); I < E; ++I) {
    const DGNode &Node = Nodes[I];
    for (const DependencyEdge &DE : Node.ResDeps) {
      bool LoopCarried = I >= DE.IID;
      OS << " FROM: " << I << " TO: " << DE.IID
         << (LoopCarried ? "(loop carried)" : "             ")
         << " - RESOURCE MASK: " << DE.ResourceOrRegID;
      OS << " - CYCLES: " << DE.Cycles << '\n';
    }
  }
}
#endif // NDEBUG

void DependencyGraph::addDepImpl(SmallVectorImpl<DependencyEdge> &Vec,
                                 DependencyEdge &&Dep) {
  auto It = find_if(Vec, [Dep](DependencyEdge &DE) {
    return DE.IID == Dep.IID && DE.ResourceOrRegID == Dep.ResourceOrRegID;
  });

  if (It != Vec.end()) {
    It->Cycles += Dep.Cycles;
    return;
  }

  Vec.emplace_back(Dep);
  Nodes[Dep.IID].NumPredecessors++;
}

BottleneckAnalysis::BottleneckAnalysis(const MCSubtargetInfo &sti,
                                       ArrayRef<MCInst> Sequence)
    : STI(sti), Tracker(STI.getSchedModel()), DG(Sequence.size()),
      Source(Sequence), TotalCycles(0),
      PressureIncreasedBecauseOfResources(false),
      PressureIncreasedBecauseOfRegisterDependencies(false),
      PressureIncreasedBecauseOfMemoryDependencies(false),
      SeenStallCycles(false), BPI() {}

void BottleneckAnalysis::onEvent(const HWInstructionEvent &Event) {
  Tracker.handleInstructionEvent(Event);
  if (Event.Type != HWInstructionEvent::Issued)
    return;

  const unsigned IID = Event.IR.getSourceIndex();
  const Instruction &IS = *Event.IR.getInstruction();
  unsigned Cycles = Tracker.getRegisterPressureCycles(IID);
  unsigned To = IID % Source.size();
  if (Cycles) {
    const CriticalDependency &RegDep = IS.getCriticalRegDep();
    unsigned From = RegDep.IID % Source.size();
    DG.addRegDep(From, To, RegDep.RegID, Cycles);
  }
  Cycles = Tracker.getMemoryPressureCycles(IID);
  if (Cycles) {
    const CriticalDependency &MemDep = IS.getCriticalMemDep();
    unsigned From = MemDep.IID % Source.size();
    DG.addMemDep(From, To, Cycles);
  }
}

void BottleneckAnalysis::onEvent(const HWPressureEvent &Event) {
  assert(Event.Reason != HWPressureEvent::INVALID &&
         "Unexpected invalid event!");

  Tracker.handlePressureEvent(Event);

  switch (Event.Reason) {
  default:
    break;

  case HWPressureEvent::RESOURCES: {
    PressureIncreasedBecauseOfResources = true;

    SmallVector<unsigned, 4> UniqueUsers;
    for (const InstRef &IR : Event.AffectedInstructions) {
      const Instruction &IS = *IR.getInstruction();
      unsigned To = IR.getSourceIndex() % Source.size();
      unsigned BusyResources =
          IS.getCriticalResourceMask() & Event.ResourceMask;
      while (BusyResources) {
        uint64_t Current = BusyResources & (-BusyResources);
        Tracker.getUniqueUsers(Current, UniqueUsers);
        for (unsigned User : UniqueUsers)
          DG.addResourceDep(User % Source.size(), To, Current, 1);
        BusyResources ^= Current;
      }
      UniqueUsers.clear();
    }

    break;
  }

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
