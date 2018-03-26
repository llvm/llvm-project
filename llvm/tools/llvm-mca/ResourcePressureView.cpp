//===--------------------- ResourcePressureView.cpp -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements methods in the ResourcePressureView interface.
///
//===----------------------------------------------------------------------===//

#include "ResourcePressureView.h"
#include "llvm/Support/raw_ostream.h"

namespace mca {

using namespace llvm;

void ResourcePressureView::initialize() {
  // Populate the map of resource descriptors.
  unsigned R2VIndex = 0;
  const MCSchedModel &SM = STI.getSchedModel();
  for (unsigned I = 0, E = SM.getNumProcResourceKinds(); I < E; ++I) {
    const MCProcResourceDesc &ProcResource = *SM.getProcResource(I);
    unsigned NumUnits = ProcResource.NumUnits;
    // Skip groups and invalid resources with zero units.
    if (ProcResource.SubUnitsIdxBegin || !NumUnits)
      continue;

    Resource2VecIndex.insert(std::pair<unsigned, unsigned>(I, R2VIndex));
    R2VIndex += ProcResource.NumUnits;
  }

  NumResourceUnits = R2VIndex;
  ResourceUsage.resize(NumResourceUnits * (Source.size() + 1));
  std::fill(ResourceUsage.begin(), ResourceUsage.end(), 0.0);
}

void ResourcePressureView::onInstructionEvent(const HWInstructionEvent &Event) {
  // We're only interested in Issue events.
  if (Event.Type != HWInstructionEvent::Issued)
    return;
  const auto &IssueEvent = static_cast<const HWInstructionIssuedEvent &>(Event);
  unsigned SourceIdx = Event.Index % Source.size();
  for (const std::pair<ResourceRef, double> &Use : IssueEvent.UsedResources) {
    const ResourceRef &RR = Use.first;
    assert(Resource2VecIndex.find(RR.first) != Resource2VecIndex.end());
    unsigned R2VIndex = Resource2VecIndex[RR.first];
    R2VIndex += countTrailingZeros(RR.second);
    ResourceUsage[R2VIndex + NumResourceUnits * SourceIdx] += Use.second;
    ResourceUsage[R2VIndex + NumResourceUnits * Source.size()] += Use.second;
  }
}

static void printColumnNames(raw_string_ostream &OS, const MCSchedModel &SM) {
  for (unsigned I = 1, ResourceIndex = 0, E = SM.getNumProcResourceKinds();
       I < E; ++I) {
    const MCProcResourceDesc &ProcResource = *SM.getProcResource(I);
    unsigned NumUnits = ProcResource.NumUnits;
    // Skip groups and invalid resources with zero units.
    if (ProcResource.SubUnitsIdxBegin || !NumUnits)
      continue;

    if (NumUnits == 1) {
      OS << '[' << ResourceIndex << ']';
      if (ResourceIndex < 10)
        OS << "    ";
      else
        OS << "   ";
    } else {
      for (unsigned J = 0; J < NumUnits; ++J) {
        OS << "[" << ResourceIndex << '.' << J << ']';
        if (ResourceIndex < 10)
          OS << "  ";
        else
          OS << ' ';
      }
    }
    ResourceIndex++;
  }
}

void ResourcePressureView::printResourcePressurePerIteration(
    raw_ostream &OS, unsigned Executions) const {
  std::string Buffer;
  raw_string_ostream TempStream(Buffer);

  TempStream << "\n\nResources:\n";
  const MCSchedModel &SM = STI.getSchedModel();
  for (unsigned I = 1, ResourceIndex = 0, E = SM.getNumProcResourceKinds();
       I < E; ++I) {
    const MCProcResourceDesc &ProcResource = *SM.getProcResource(I);
    unsigned NumUnits = ProcResource.NumUnits;
    // Skip groups and invalid resources with zero units.
    if (ProcResource.SubUnitsIdxBegin || !NumUnits)
      continue;

    for (unsigned J = 0; J < NumUnits; ++J) {
      TempStream << '[' << ResourceIndex;
      if (NumUnits > 1)
        TempStream << '.' << J;
      TempStream << "] - " << ProcResource.Name << '\n';
    }

    ResourceIndex++;
  }

  TempStream << "\n\nResource pressure per iteration:\n";
  printColumnNames(TempStream, SM);
  TempStream << '\n';

  for (unsigned I = 0, E = NumResourceUnits; I < E; ++I) {
    double Usage = ResourceUsage[I + Source.size() * E];
    if (!Usage) {
      TempStream << " -     ";
      continue;
    }

    double Pressure = Usage / Executions;
    TempStream << format("%.2f", Pressure);
    if (Pressure < 10.0)
      TempStream << "   ";
    else if (Pressure < 100.0)
      TempStream << "  ";
    else
      TempStream << ' ';
  }

  TempStream.flush();
  OS << Buffer;
}

void ResourcePressureView::printResourcePressurePerInstruction(
    raw_ostream &OS, unsigned Executions) const {
  std::string Buffer;
  raw_string_ostream TempStream(Buffer);

  TempStream << "\n\nResource pressure by instruction:\n";
  printColumnNames(TempStream, STI.getSchedModel());
  TempStream << "\tInstructions:\n";

  for (unsigned I = 0, E = Source.size(); I < E; ++I) {
    for (unsigned J = 0; J < NumResourceUnits; ++J) {
      double Usage = ResourceUsage[J + I * NumResourceUnits];
      if (!Usage) {
        TempStream << " -     ";
      } else {
        double Pressure = Usage / Executions;
        if (Pressure < 0.005) {
          TempStream << " -     ";
        } else {
          TempStream << format("%.2f", Pressure);
          if (Pressure < 10.0)
            TempStream << "   ";
          else if (Pressure < 100.0)
            TempStream << "  ";
          else
            TempStream << ' ';
        }
      }
    }

    MCIP.printInst(&Source.getMCInstFromIndex(I), TempStream, "", STI);
    TempStream << '\n';
    TempStream.flush();
    OS << Buffer;
    Buffer = "";
  }
}
} // namespace mca
