//===--------------------- InstructionTables.cpp ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements method InstructionTables::run().
/// Method run() prints a theoretical resource pressure distribution based on
/// the information available in the scheduling model, and without running
/// the backend pipeline.
///
//===----------------------------------------------------------------------===//

#include "InstructionTables.h"

namespace mca {

using namespace llvm;

using ResourceRef = std::pair<uint64_t, uint64_t>;

void InstructionTables::run() {
  ArrayRef<uint64_t> Masks = IB.getProcResourceMasks();
  SmallVector<std::pair<ResourceRef, double>, 4> UsedResources;

  // Create an instruction descriptor for every instruction in the sequence.
  while (S.hasNext()) {
    UsedResources.clear();
    SourceRef SR = S.peekNext();
    std::unique_ptr<Instruction> Inst = IB.createInstruction(*SR.second);
    const InstrDesc &Desc = Inst->getDesc();
    // Now identify the resources consumed by this instruction.
    for (const std::pair<uint64_t, ResourceUsage> Resource : Desc.Resources) {
      // Skip zero-cycle resources (i.e. unused resources).
      if (!Resource.second.size())
        continue;
      double Cycles = static_cast<double>(Resource.second.size());
      unsigned Index =
          std::distance(Masks.begin(), std::find(Masks.begin(), Masks.end(),
                                                 Resource.first));
      const MCProcResourceDesc &ProcResource = *SM.getProcResource(Index);
      unsigned NumUnits = ProcResource.NumUnits;
      if (!ProcResource.SubUnitsIdxBegin) {
        // The number of cycles consumed by each unit.
        Cycles /= NumUnits;
        for (unsigned I = 0, E = NumUnits; I < E; ++I) {
          ResourceRef ResourceUnit = std::make_pair(Index, 1U << I);
          UsedResources.emplace_back(std::make_pair(ResourceUnit, Cycles));
        }
        continue;
      }

      // This is a group. Obtain the set of resources contained in this
      // group. Some of these resources may implement multiple units.
      // Uniformly distribute Cycles across all of the units.
      for (unsigned I1 = 0; I1 < NumUnits; ++I1) {
        unsigned SubUnitIdx = ProcResource.SubUnitsIdxBegin[I1];
        const MCProcResourceDesc &SubUnit = *SM.getProcResource(SubUnitIdx);
        // Compute the number of cycles consumed by each resource unit.
        double RUCycles = Cycles / (NumUnits * SubUnit.NumUnits);
        for (unsigned I2 = 0, E2 = SubUnit.NumUnits; I2 < E2; ++I2) {
          ResourceRef ResourceUnit = std::make_pair(SubUnitIdx, 1U << I2);
          UsedResources.emplace_back(std::make_pair(ResourceUnit, RUCycles));
        }
      }
    }

    // Now send a fake instruction issued event to all the views.
    InstRef IR(SR.first, Inst.get());
    HWInstructionIssuedEvent Event(IR, UsedResources);
    for (std::unique_ptr<View> &Listener : Views)
      Listener->onInstructionEvent(Event);
    S.updateNext();
  }
}

void InstructionTables::printReport(llvm::raw_ostream &OS) const {
  for (const std::unique_ptr<View> &V : Views)
    V->printView(OS);
}

} // namespace mca
