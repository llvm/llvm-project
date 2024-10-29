//===-- Analysis.cpp --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis.h"
#include "BenchmarkResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Regex.h"
#include <string>
#include <vector>

namespace llvm {
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
static cl::opt<std::string>
    SchedClassAnalysisBlackList("sched-class-analysis-blacklist",
                                cl::desc("Regex of sched class to exclude from"
                                         " analysis"),
                                cl::Hidden, cl::init(""));
#endif

namespace exegesis {

void Analysis::printSnippet(raw_ostream &OS, ArrayRef<uint8_t> Bytes,
                            const char *Separator) const {
  ListSeparator LS(Separator);
  std::string Line;
  raw_string_ostream LineSS(Line);
  // Parse the asm snippet and print it.
  while (!Bytes.empty()) {
    MCInst MI;
    uint64_t MISize = 0;
    if (!DisasmHelper_->decodeInst(MI, MISize, Bytes)) {
      OS << LS << "[error decoding asm snippet]";
      return;
    }
    Line.clear();
    DisasmHelper_->printInst(&MI, LineSS);
    OS << LS << StringRef(Line).trim();
    Bytes = Bytes.drop_front(MISize);
  }
}

Analysis::Analysis(const LLVMState &State,
                   const BenchmarkClustering &Clustering,
                   double AnalysisInconsistencyEpsilon,
                   bool AnalysisDisplayUnstableOpcodes)
    : Clustering_(Clustering), State_(State),
      AnalysisInconsistencyEpsilonSquared_(AnalysisInconsistencyEpsilon *
                                           AnalysisInconsistencyEpsilon),
      AnalysisDisplayUnstableOpcodes_(AnalysisDisplayUnstableOpcodes) {
  if (Clustering.getPoints().empty())
    return;

  DisasmHelper_ = std::make_unique<DisassemblerHelper>(State);
}

template <>
Expected<typename Analysis::PrintClusters::Result>
Analysis::exportResult<Analysis::PrintClusters>() const {
  typename Analysis::PrintClusters::Result Clusters;

  for (const auto &Measurement : Clustering_.getPoints().front().Measurements)
    Clusters.MeasurementNames.push_back(Measurement.Key);

  auto &Entries = Clusters.Data;
  for (const auto &ClusterIt : Clustering_.getValidClusters())
    for (const size_t PointId : ClusterIt.PointIndices) {
      Entries.emplace_back();
      auto &Data = Entries.back();
      const Benchmark &Point = Clustering_.getPoints()[PointId];
      Data.Id = Clustering_.getClusterIdForPoint(PointId);
      raw_string_ostream SS(Data.Snippet);
      printSnippet(SS, Point.AssembledSnippet, /*Separator=*/"; ");
      Data.Config = Point.Key.Config;

      assert(!Point.Key.Instructions.empty());
      const MCInst &MCI = Point.keyInstruction();
      unsigned SchedClassId;
      std::tie(SchedClassId, std::ignore) =
          ResolvedSchedClass::resolveSchedClassId(State_.getSubtargetInfo(),
                                                  State_.getInstrInfo(), MCI);
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
      const MCSchedClassDesc *const SCDesc =
          State_.getSubtargetInfo().getSchedModel().getSchedClassDesc(
              SchedClassId);
      Data.SchedClass = SCDesc->Name;
#else
      Data.SchedClass = SchedClassId;
#endif

      for (const auto &Measurement : Point.Measurements)
        Data.Measurements.push_back(Measurement.PerInstructionValue);
    }

  return Clusters;
}

template <>
Error Analysis::run<Analysis::PrintClusters>(
    raw_ostream &OS, Analysis::OutputFormat Format) const {
  if (Clustering_.getPoints().empty())
    return Error::success();

  auto Result = exportResult<Analysis::PrintClusters>();
  if (!Result)
    return Result.takeError();

  switch (Format) {
  case OF_Default:
    AnalysisResult::printCSV(OS, *Result);
    break;
  case OF_YAML:
    AnalysisResult::printYAML(OS, *Result);
    break;
  default:
    llvm_unreachable("Unsupported output format");
  }

  return Error::success();
}

Analysis::ResolvedSchedClassAndPoints::ResolvedSchedClassAndPoints(
    ResolvedSchedClass &&RSC)
    : RSC(std::move(RSC)) {}

std::vector<Analysis::ResolvedSchedClassAndPoints>
Analysis::makePointsPerSchedClass() const {
  std::vector<ResolvedSchedClassAndPoints> Entries;
  // Maps SchedClassIds to index in result.
  std::unordered_map<unsigned, size_t> SchedClassIdToIndex;
  const auto &Points = Clustering_.getPoints();
  for (size_t PointId = 0, E = Points.size(); PointId < E; ++PointId) {
    const Benchmark &Point = Points[PointId];
    if (!Point.Error.empty())
      continue;
    assert(!Point.Key.Instructions.empty());
    // FIXME: we should be using the tuple of classes for instructions in the
    // snippet as key.
    const MCInst &MCI = Point.keyInstruction();
    unsigned SchedClassId;
    bool WasVariant;
    std::tie(SchedClassId, WasVariant) =
        ResolvedSchedClass::resolveSchedClassId(State_.getSubtargetInfo(),
                                                State_.getInstrInfo(), MCI);
    const auto IndexIt = SchedClassIdToIndex.find(SchedClassId);
    if (IndexIt == SchedClassIdToIndex.end()) {
      // Create a new entry.
      SchedClassIdToIndex.emplace(SchedClassId, Entries.size());
      ResolvedSchedClassAndPoints Entry(ResolvedSchedClass(
          State_.getSubtargetInfo(), SchedClassId, WasVariant));
      Entry.PointIds.push_back(PointId);
      Entries.push_back(std::move(Entry));
    } else {
      // Append to the existing entry.
      Entries[IndexIt->second].PointIds.push_back(PointId);
    }
  }
  return Entries;
}

void Analysis::SchedClassCluster::addPoint(
    size_t PointId, const BenchmarkClustering &Clustering) {
  PointIds.push_back(PointId);
  const auto &Point = Clustering.getPoints()[PointId];
  if (ClusterId.isUndef())
    ClusterId = Clustering.getClusterIdForPoint(PointId);
  assert(ClusterId == Clustering.getClusterIdForPoint(PointId));

  Centroid.addPoint(Point.Measurements);
}

bool Analysis::SchedClassCluster::measurementsMatch(
    const MCSubtargetInfo &STI, const ResolvedSchedClass &RSC,
    const BenchmarkClustering &Clustering,
    const double AnalysisInconsistencyEpsilonSquared_) const {
  assert(!Clustering.getPoints().empty());
  const Benchmark::ModeE Mode = Clustering.getPoints()[0].Mode;

  if (!Centroid.validate(Mode))
    return false;

  const std::vector<BenchmarkMeasure> ClusterCenterPoint =
      Centroid.getAsPoint();

  const std::vector<BenchmarkMeasure> SchedClassPoint =
      RSC.getAsPoint(Mode, STI, Centroid.getStats());
  if (SchedClassPoint.empty())
    return false; // In Uops mode validate() may not be enough.

  assert(ClusterCenterPoint.size() == SchedClassPoint.size() &&
         "Expected measured/sched data dimensions to match.");

  return Clustering.isNeighbour(ClusterCenterPoint, SchedClassPoint,
                                AnalysisInconsistencyEpsilonSquared_);
}

// Returns false to exclude the given MCSchedClassDesc from analysis.
static bool filterMCSchedClass(const MCSchedClassDesc &SCDesc) {
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  static Regex Filter(SchedClassAnalysisBlackList);
  if (Filter.isValid() && Filter.match(SCDesc.Name))
    return false;
#endif
  return true;
}

template <>
Expected<typename Analysis::PrintSchedClassInconsistencies::Result>
Analysis::exportResult<Analysis::PrintSchedClassInconsistencies>() const {
  AnalysisResult::SchedClassInconsistencies Result;

  const MCInstrInfo &II = State_.getInstrInfo();
  const auto &SI = State_.getSubtargetInfo();
  const auto &SM = SI.getSchedModel();

  const auto &Points = Clustering_.getPoints();
  const auto &FirstPoint = Points[0];
  Result.Triple = FirstPoint.LLVMTriple;
  Result.CPUName = FirstPoint.CpuName;
  Result.Epsilon = std::sqrt(AnalysisInconsistencyEpsilonSquared_);

  std::vector<SchedClassCluster> SchedClassClusters;
  for (const auto &RSCAndPoints : makePointsPerSchedClass()) {
    const auto &RSC = RSCAndPoints.RSC;
    if (!RSC.SCDesc)
      continue;

    if (!filterMCSchedClass(*RSC.SCDesc))
      continue;

    // Bucket sched class points into sched class clusters.
    SchedClassClusters.clear();
    for (const size_t PointId : RSCAndPoints.PointIds) {
      const auto &ClusterId = Clustering_.getClusterIdForPoint(PointId);
      if (!ClusterId.isValid())
        continue; // Ignore noise and errors. FIXME: take noise into account ?
      if (ClusterId.isUnstable() ^ AnalysisDisplayUnstableOpcodes_)
        continue; // Either display stable or unstable clusters only.
      auto SchedClassClusterIt = llvm::find_if(
          SchedClassClusters, [ClusterId](const SchedClassCluster &C) {
            return C.id() == ClusterId;
          });
      if (SchedClassClusterIt == SchedClassClusters.end()) {
        SchedClassClusters.emplace_back();
        SchedClassClusterIt = std::prev(SchedClassClusters.end());
      }
      SchedClassClusterIt->addPoint(PointId, Clustering_);
    }

    // Print any scheduling class that has at least one cluster that does not
    // match the checked-in data.
    if (all_of(
            SchedClassClusters, [this, &RSC, &SI](const SchedClassCluster &C) {
              return C.measurementsMatch(SI, RSC, Clustering_,
                                         AnalysisInconsistencyEpsilonSquared_);
            }))
      continue; // Nothing weird.

    Result.Inconsistencies.emplace_back();
    auto &ResultEntry = Result.Inconsistencies.back();
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    ResultEntry.Name = RSC.SCDesc->Name;
#else
    ResultEntry.Name = RSC.SchedClassId;
#endif

    assert(!SchedClassClusters.empty());
    for (const auto &Measurement :
         Points[SchedClassClusters[0].getPointIds()[0]].Measurements)
      ResultEntry.MeasurementNames.push_back(Measurement.Key);

    // Measurements
    for (const SchedClassCluster &Cluster : SchedClassClusters) {
      ResultEntry.Measurements.emplace_back();
      auto &Measurement = ResultEntry.Measurements.back();
      Measurement.ClusterId = Cluster.id();
      Measurement.IsInconsistent = !Cluster.measurementsMatch(
          SI, RSC, Clustering_, AnalysisInconsistencyEpsilonSquared_);

      // Description of points in this cluster.
      for (const size_t PointId : Cluster.getPointIds()) {
        Measurement.Points.emplace_back();
        auto &ResPoint = Measurement.Points.back();
        const auto &Point = Points[PointId];
        if (!Point.Key.Instructions.empty())
          ResPoint.Opcode = II.getName(Point.Key.Instructions[0].getOpcode());
        ResPoint.Config = Point.Key.Config;
        raw_string_ostream SS(ResPoint.Snippet);
        printSnippet(SS, Point.AssembledSnippet);
      }

      // Measured data.
      for (const auto &Stats : Cluster.getCentroid().getStats()) {
        Measurement.Data.emplace_back();
        Measurement.Data.back() = {Stats.min(), Stats.avg(), Stats.max()};
      }
    }

    // SchedModel data
    ResultEntry.IsVariant = RSC.WasVariant;
    ResultEntry.NumMicroOps = RSC.SCDesc->NumMicroOps;
    // Latencies.
    for (int I = 0, E = RSC.SCDesc->NumWriteLatencyEntries; I < E; ++I) {
      const auto *const Entry = SI.getWriteLatencyEntry(RSC.SCDesc, I);
      ResultEntry.Latency.emplace_back(
          std::make_pair(Entry->WriteResourceID,
                         RSC.computeNormalizedWriteLatency(Entry, SI)));
    }

    // Inverse throughput.
    ResultEntry.RThroughput =
        MCSchedModel::getReciprocalThroughput(SI, *RSC.SCDesc);

    // Used processor resources and pressures.
    auto PressureIt = RSC.IdealizedProcResPressure.begin();
    auto EndPressureIt = RSC.IdealizedProcResPressure.end();
    for (const auto &WPR : RSC.NonRedundantWriteProcRes) {
      ResultEntry.WriteProcResEntries.emplace_back();
      auto &ResWPR = ResultEntry.WriteProcResEntries.back();
      ResWPR.ProcResName = SM.getProcResource(WPR.ProcResourceIdx)->Name;
      ResWPR.AcquireAtCycle = WPR.AcquireAtCycle;
      ResWPR.ReleaseAtCycle = WPR.ReleaseAtCycle;
      if (PressureIt != EndPressureIt &&
          WPR.ProcResourceIdx == PressureIt->first) {
        ResWPR.ResourcePressure = PressureIt->second;
        ++PressureIt;
      } else {
        ResWPR.ResourcePressure = std::nullopt;
      }
    }
  }

  return Result;
}

template <>
Error Analysis::run<Analysis::PrintSchedClassInconsistencies>(
    raw_ostream &OS, Analysis::OutputFormat Format) const {
  if (Clustering_.getPoints().empty())
    return Error::success();

  auto Result = exportResult<Analysis::PrintSchedClassInconsistencies>();
  if (!Result)
    return Result.takeError();

  switch (Format) {
  case OF_Default:
    AnalysisResult::printHTML(OS, *Result);
    break;
  case OF_YAML:
    AnalysisResult::printYAML(OS, *Result);
    break;
  default:
    llvm_unreachable("Unsupported output format");
  }

  return Error::success();
}

} // namespace exegesis
} // namespace llvm
