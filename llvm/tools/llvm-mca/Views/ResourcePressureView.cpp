//===--------------------- ResourcePressureView.cpp -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements methods in the ResourcePressureView interface.
///
//===----------------------------------------------------------------------===//

#include "Views/ResourcePressureView.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace mca {

ResourcePressureView::ResourcePressureView(const llvm::MCSubtargetInfo &sti,
                                           MCInstPrinter &Printer,
                                           ArrayRef<MCInst> S)
    : InstructionView(sti, Printer, S), LastInstructionIdx(0) {
  // Populate the map of resource descriptors.
  unsigned R2VIndex = 0;
  const MCSchedModel &SM = getSubTargetInfo().getSchedModel();
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
  ResourceUsage.resize(getSource().size());

  ResourceReleaseAtCycles InitValue{0, 0};
  auto Generator = [&InitValue]() {
    ResourceReleaseAtCycles Old = InitValue;
    ++InitValue.ResourceIdx;
    return Old;
  };
  std::generate_n(std::back_inserter(CommonResourceUsage), NumResourceUnits,
                  Generator);
}

void ResourcePressureView::onEvent(const HWInstructionEvent &Event) {
  if (Event.Type == HWInstructionEvent::Dispatched) {
    LastInstructionIdx = Event.IR.getSourceIndex();
    return;
  }

  // We're only interested in Issue events.
  if (Event.Type != HWInstructionEvent::Issued)
    return;

  const auto &IssueEvent = static_cast<const HWInstructionIssuedEvent &>(Event);
  ArrayRef<llvm::MCInst> Source = getSource();
  const unsigned SourceIdx = Event.IR.getSourceIndex() % Source.size();
  for (const std::pair<ResourceRef, ReleaseAtCycles> &Use :
       IssueEvent.UsedResources) {
    const ResourceRef &RR = Use.first;
    assert(Resource2VecIndex.contains(RR.first));
    unsigned R2VIndex = Resource2VecIndex[RR.first];
    R2VIndex += llvm::countr_zero(RR.second);

    InstResourceUsage &RU = ResourceUsage[SourceIdx];
    ResourceReleaseAtCycles NewUsage{R2VIndex, Use.second};
    auto ResCyclesIt =
        lower_bound(RU, NewUsage, [](const auto &L, const auto &R) {
          return L.ResourceIdx < R.ResourceIdx;
        });
    if (ResCyclesIt != RU.end() && ResCyclesIt->ResourceIdx == R2VIndex)
      ResCyclesIt->Cycles += NewUsage.Cycles;
    else
      RU.insert(ResCyclesIt, std::move(NewUsage));

    CommonResourceUsage[R2VIndex].Cycles += NewUsage.Cycles;
  }
}

static void printColumnNames(formatted_raw_ostream &OS,
                             const MCSchedModel &SM) {
  unsigned Column = OS.getColumn();
  for (unsigned I = 1, ResourceIndex = 0, E = SM.getNumProcResourceKinds();
       I < E; ++I) {
    const MCProcResourceDesc &ProcResource = *SM.getProcResource(I);
    unsigned NumUnits = ProcResource.NumUnits;
    // Skip groups and invalid resources with zero units.
    if (ProcResource.SubUnitsIdxBegin || !NumUnits)
      continue;

    for (unsigned J = 0; J < NumUnits; ++J) {
      Column += 7;
      OS << "[" << ResourceIndex;
      if (NumUnits > 1)
        OS << '.' << J;
      OS << ']';
      OS.PadToColumn(Column);
    }

    ResourceIndex++;
  }
}

static void printResourcePressure(formatted_raw_ostream &OS, double Pressure,
                                  unsigned Col) {
  if (!Pressure || Pressure < 0.005) {
    OS << " - ";
  } else {
    // Round to the value to the nearest hundredth and then print it.
    OS << format("%.2f", floor((Pressure * 100) + 0.5) / 100);
  }
  OS.PadToColumn(Col);
}

void ResourcePressureView::printResourcePressurePerIter(raw_ostream &OS) const {
  std::string Buffer;
  raw_string_ostream TempStream(Buffer);
  formatted_raw_ostream FOS(TempStream);

  FOS << "\n\nResources:\n";
  const MCSchedModel &SM = getSubTargetInfo().getSchedModel();
  for (unsigned I = 1, ResourceIndex = 0, E = SM.getNumProcResourceKinds();
       I < E; ++I) {
    const MCProcResourceDesc &ProcResource = *SM.getProcResource(I);
    unsigned NumUnits = ProcResource.NumUnits;
    // Skip groups and invalid resources with zero units.
    if (ProcResource.SubUnitsIdxBegin || !NumUnits)
      continue;

    for (unsigned J = 0; J < NumUnits; ++J) {
      FOS << '[' << ResourceIndex;
      if (NumUnits > 1)
        FOS << '.' << J;
      FOS << ']';
      FOS.PadToColumn(6);
      FOS << "- " << ProcResource.Name << '\n';
    }

    ResourceIndex++;
  }

  FOS << "\n\nResource pressure per iteration:\n";
  FOS.flush();
  printColumnNames(FOS, SM);
  FOS << '\n';
  FOS.flush();

  ArrayRef<llvm::MCInst> Source = getSource();
  const unsigned Executions = LastInstructionIdx / Source.size() + 1;
  auto UsageEntryEnd = CommonResourceUsage.end();
  auto UsageEntryIt = CommonResourceUsage.begin();
  for (unsigned I = 0, E = NumResourceUnits; I < E; ++I) {
    double Pressure = 0.0;
    if (UsageEntryIt != UsageEntryEnd && UsageEntryIt->ResourceIdx == I) {
      Pressure = UsageEntryIt->Cycles / Executions;
      ++UsageEntryIt;
    }
    printResourcePressure(FOS, Pressure, (I + 1) * 7);
  }
  assert(UsageEntryIt == UsageEntryEnd);

  FOS.flush();
  OS << Buffer;
}

void ResourcePressureView::printResourcePressurePerInst(raw_ostream &OS) const {
  std::string Buffer;
  raw_string_ostream TempStream(Buffer);
  formatted_raw_ostream FOS(TempStream);

  FOS << "\n\nResource pressure by instruction:\n";
  printColumnNames(FOS, getSubTargetInfo().getSchedModel());
  FOS << "Instructions:\n";

  unsigned InstrIndex = 0;
  ArrayRef<llvm::MCInst> Source = getSource();
  const unsigned Executions = LastInstructionIdx / Source.size() + 1;
  for (const MCInst &MCI : Source) {
    auto UsageEntryEnd = ResourceUsage[InstrIndex].end();
    auto UsageEntryIt = ResourceUsage[InstrIndex].begin();
    for (unsigned J = 0; J < NumResourceUnits; ++J) {
      double Pressure = 0.0;
      if (UsageEntryIt != UsageEntryEnd && UsageEntryIt->ResourceIdx == J) {
        Pressure = UsageEntryIt->Cycles / Executions;
        ++UsageEntryIt;
      }
      printResourcePressure(FOS, Pressure, (J + 1) * 7);
    }
    assert(UsageEntryIt == UsageEntryEnd);

    FOS << printInstructionString(MCI) << '\n';
    FOS.flush();
    OS << Buffer;
    Buffer = "";

    ++InstrIndex;
  }
}

json::Value ResourcePressureView::toJSON() const {
  // We're dumping the instructions and the ResourceUsage array.
  json::Array ResourcePressureInfo;

  // The ResourceUsage matrix is sparse, so we only consider
  // non-zero values.
  ArrayRef<llvm::MCInst> Source = getSource();
  const unsigned Executions = LastInstructionIdx / Source.size() + 1;

  auto AddToJSON = [&ResourcePressureInfo, Executions](
                       const ResourceReleaseAtCycles &RU, unsigned InstIndex) {
    assert(RU.Cycles.getNumerator() != 0);
    double Usage = RU.Cycles / Executions;
    ResourcePressureInfo.push_back(
        json::Object({{"InstructionIndex", InstIndex},
                      {"ResourceIndex", RU.ResourceIdx},
                      {"ResourceUsage", Usage}}));
  };
  for (const auto &[InstIndex, Usages] : enumerate(ResourceUsage))
    for (const auto &RU : Usages)
      AddToJSON(RU, InstIndex);
  for (const auto &RU : CommonResourceUsage) {
    if (RU.Cycles.getNumerator() != 0)
      AddToJSON(RU, Source.size());
  }

  json::Object JO({{"ResourcePressureInfo", std::move(ResourcePressureInfo)}});
  return JO;
}
} // namespace mca
} // namespace llvm
