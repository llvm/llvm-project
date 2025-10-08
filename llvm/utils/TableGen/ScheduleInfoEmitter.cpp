//===- ScheduleInfoEmitter.cpp - Generate scheduling info JSON -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits scheduling information in JSON format for
// external analysis tools like llvm-exegesis verification scripts.
//
//===----------------------------------------------------------------------===//

#include "Common/CodeGenSchedule.h"
#include "Common/CodeGenTarget.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <map>
#include <vector>

using namespace llvm;

namespace {

class ScheduleInfoEmitter {
  CodeGenTarget Target;
  CodeGenSchedModels SchedModels;

public:
  ScheduleInfoEmitter(const RecordKeeper &R)
    : Target(R), SchedModels(R, Target) {}

  void run(raw_ostream &OS);

private:
  json::Object emitWriteResInfo(const Record *WriteDef, const Record *WriteRes,
                                 const CodeGenProcModel &ProcModel);
  json::Object emitProcModelInfo(const CodeGenProcModel &ProcModel);
  unsigned getNumUnitsInGroup(const Record *ProcResGroup);
};

void ScheduleInfoEmitter::run(raw_ostream &OS) {
  json::Object Root;

  // Emit info for each processor model
  for (const CodeGenProcModel &PM : SchedModels.procModels()) {
    if (!PM.hasInstrSchedModel())
      continue;

    std::string ProcName = PM.ModelName;
    if (ProcName.empty())
      continue;

    Root[ProcName] = emitProcModelInfo(PM);
  }

  OS << formatv("{0:2}", json::Value(std::move(Root))) << "\n";
}

json::Object ScheduleInfoEmitter::emitProcModelInfo(
    const CodeGenProcModel &ProcModel) {
  json::Object ProcInfo;
  json::Object SchedWrites;

  // Process each WriteRes mapping for this processor
  for (const auto &[WriteDef, WriteRes] : ProcModel.WriteResMap) {
    // Get the write name from the WriteDef
    std::string WriteName;
    if (WriteDef->isSubClassOf("SchedWrite")) {
      WriteName = WriteDef->getName();
    } else if (WriteDef->isSubClassOf("SchedWriteVariant")) {
      WriteName = WriteDef->getName();
    } else if (WriteDef->isSubClassOf("SchedWriteRes")) {
      WriteName = WriteDef->getName();
    } else {
      WriteName = WriteDef->getName();
    }

    SchedWrites[WriteName] = emitWriteResInfo(WriteDef, WriteRes, ProcModel);
  }

  ProcInfo["SchedWrites"] = std::move(SchedWrites);

  // Add processor resource information
  json::Object Resources;
  for (const Record *ProcRes : ProcModel.ProcResourceDefs) {
    if (ProcRes->isSubClassOf("ProcResGroup")) {
      json::Object ResInfo;
      std::vector<const Record *> ResUnits =
        ProcRes->getValueAsListOfDefs("Resources");
      ResInfo["NumUnits"] = static_cast<int64_t>(ResUnits.size());

      json::Array Units;
      for (const Record *Unit : ResUnits)
        Units.push_back(Unit->getName());
      ResInfo["Units"] = std::move(Units);

      Resources[ProcRes->getName()] = std::move(ResInfo);
    } else {
      json::Object ResInfo;
      ResInfo["NumUnits"] = ProcRes->getValueAsInt("NumUnits");
      Resources[ProcRes->getName()] = std::move(ResInfo);
    }
  }

  ProcInfo["Resources"] = std::move(Resources);

  return ProcInfo;
}

json::Object ScheduleInfoEmitter::emitWriteResInfo(
    const Record *WriteDef, const Record *WriteRes,
    const CodeGenProcModel &ProcModel) {
  json::Object WriteInfo;

  if (!WriteRes) {
    WriteInfo["Error"] = "No WriteRes found";
    return WriteInfo;
  }

  // Extract scheduling information
  int Latency = WriteRes->getValueAsInt("Latency");
  WriteInfo["Latency"] = Latency;

  std::vector<const Record *> ProcResources =
    WriteRes->getValueAsListOfDefs("ProcResources");
  std::vector<int> ReleaseAtCycles;

  if (WriteRes->isSubClassOf("WriteRes") ||
      WriteRes->isSubClassOf("SchedWriteRes")) {
    for (int Val : WriteRes->getValueAsListOfInts("ReleaseAtCycles")) {
      ReleaseAtCycles.push_back(Val);
    }
  }

  // Make sure we have matching resources and release cycles
  if (ReleaseAtCycles.empty() && !ProcResources.empty()) {
    ReleaseAtCycles.resize(ProcResources.size(), 1);
  }

  json::Array ResourceArray;
  double TotalInverseThroughput = 0.0;

  for (size_t i = 0; i < ProcResources.size(); ++i) {
    const Record *ProcRes = ProcResources[i];
    int ReleaseCycles = i < ReleaseAtCycles.size() ? ReleaseAtCycles[i] : 1;

    json::Object ResInfo;
    ResInfo["Name"] = ProcRes->getName();
    ResInfo["ReleaseAtCycles"] = ReleaseCycles;

    // Calculate number of units
    unsigned NumUnits = getNumUnitsInGroup(ProcRes);
    ResInfo["NumUnits"] = static_cast<int64_t>(NumUnits);

    // Calculate inverse throughput for this resource
    double InvThroughput = static_cast<double>(ReleaseCycles) / NumUnits;
    ResInfo["InverseThroughput"] = InvThroughput;

    // Track maximum inverse throughput (bottleneck)
    TotalInverseThroughput = std::max(TotalInverseThroughput, InvThroughput);

    ResourceArray.push_back(std::move(ResInfo));
  }

  WriteInfo["Resources"] = std::move(ResourceArray);
  WriteInfo["InverseThroughput"] = TotalInverseThroughput;
  WriteInfo["NumMicroOps"] = WriteRes->getValueAsInt("NumMicroOps");

  return WriteInfo;
}

unsigned ScheduleInfoEmitter::getNumUnitsInGroup(const Record *ProcRes) {
  if (ProcRes->isSubClassOf("ProcResGroup")) {
    // It's a group - count the units in the group
    std::vector<const Record *> Resources =
      ProcRes->getValueAsListOfDefs("Resources");
    unsigned TotalUnits = 0;
    for (const Record *Res : Resources) {
      // Recursively handle nested groups
      TotalUnits += getNumUnitsInGroup(Res);
    }
    return TotalUnits;
  } else {
    // It's a single resource - return its NumUnits
    return ProcRes->getValueAsInt("NumUnits");
  }
}

} // end anonymous namespace

static TableGen::Emitter::OptClass<ScheduleInfoEmitter>
    X("gen-sched-info", "Generate scheduling information JSON");