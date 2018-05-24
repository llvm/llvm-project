//===-- Analysis.cpp --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Analysis.h"
#include "BenchmarkResult.h"
#include "llvm/Support/FormatVariadic.h"
#include <unordered_set>
#include <vector>

namespace exegesis {

static const char kCsvSep = ',';

namespace {

enum EscapeTag { kEscapeCsv, kEscapeHtml };

template <EscapeTag Tag>
void writeEscaped(llvm::raw_ostream &OS, const llvm::StringRef S);

template <>
void writeEscaped<kEscapeCsv>(llvm::raw_ostream &OS, const llvm::StringRef S) {
  if (std::find(S.begin(), S.end(), kCsvSep) == S.end()) {
    OS << S;
  } else {
    // Needs escaping.
    OS << '"';
    for (const char C : S) {
      if (C == '"')
        OS << "\"\"";
      else
        OS << C;
    }
    OS << '"';
  }
}

template <>
void writeEscaped<kEscapeHtml>(llvm::raw_ostream &OS, const llvm::StringRef S) {
  for (const char C : S) {
    if (C == '<')
      OS << "&lt;";
    else if (C == '>')
      OS << "&gt;";
    else if (C == '&')
      OS << "&amp;";
    else
      OS << C;
  }
}

} // namespace

template <EscapeTag Tag>
static void
writeClusterId(llvm::raw_ostream &OS,
               const InstructionBenchmarkClustering::ClusterId &CID) {
  if (CID.isNoise())
    writeEscaped<Tag>(OS, "[noise]");
  else if (CID.isError())
    writeEscaped<Tag>(OS, "[error]");
  else
    OS << CID.getId();
}

template <EscapeTag Tag>
static void writeMeasurementValue(llvm::raw_ostream &OS, const double Value) {
  writeEscaped<Tag>(OS, llvm::formatv("{0:F}", Value).str());
}

// Prints a row representing an instruction, along with scheduling info and
// point coordinates (measurements).
void Analysis::printInstructionRowCsv(const size_t PointId,
                                      llvm::raw_ostream &OS) const {
  const InstructionBenchmark &Point = Clustering_.getPoints()[PointId];
  writeClusterId<kEscapeCsv>(OS, Clustering_.getClusterIdForPoint(PointId));
  OS << kCsvSep;
  writeEscaped<kEscapeCsv>(OS, Point.Key.OpcodeName);
  OS << kCsvSep;
  writeEscaped<kEscapeCsv>(OS, Point.Key.Config);
  OS << kCsvSep;
  const auto OpcodeIt = MnemonicToOpcode_.find(Point.Key.OpcodeName);
  if (OpcodeIt != MnemonicToOpcode_.end()) {
    const unsigned SchedClassId =
        InstrInfo_->get(OpcodeIt->second).getSchedClass();
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    const auto &SchedModel = SubtargetInfo_->getSchedModel();
    const llvm::MCSchedClassDesc *const SCDesc =
        SchedModel.getSchedClassDesc(SchedClassId);
    writeEscaped<kEscapeCsv>(OS, SCDesc->Name);
#else
    OS << SchedClassId;
#endif
  }
  // FIXME: Print the sched class once InstructionBenchmark separates key into
  // (mnemonic, mode, opaque).
  for (const auto &Measurement : Point.Measurements) {
    OS << kCsvSep;
    writeMeasurementValue<kEscapeCsv>(OS, Measurement.Value);
  }
  OS << "\n";
}

Analysis::Analysis(const llvm::Target &Target,
                   const InstructionBenchmarkClustering &Clustering)
    : Clustering_(Clustering) {
  if (Clustering.getPoints().empty())
    return;

  InstrInfo_.reset(Target.createMCInstrInfo());
  const InstructionBenchmark &FirstPoint = Clustering.getPoints().front();
  SubtargetInfo_.reset(Target.createMCSubtargetInfo(FirstPoint.LLVMTriple,
                                                    FirstPoint.CpuName, ""));

  // Build an index of mnemonic->opcode.
  for (int I = 0, E = InstrInfo_->getNumOpcodes(); I < E; ++I)
    MnemonicToOpcode_.emplace(InstrInfo_->getName(I), I);
}

template <>
llvm::Error
Analysis::run<Analysis::PrintClusters>(llvm::raw_ostream &OS) const {
  if (Clustering_.getPoints().empty())
    return llvm::Error::success();

  // Write the header.
  OS << "cluster_id" << kCsvSep << "opcode_name" << kCsvSep << "config"
     << kCsvSep << "sched_class";
  for (const auto &Measurement : Clustering_.getPoints().front().Measurements) {
    OS << kCsvSep;
    writeEscaped<kEscapeCsv>(OS, Measurement.Key);
  }
  OS << "\n";

  // Write the points.
  const auto &Clusters = Clustering_.getValidClusters();
  for (size_t I = 0, E = Clusters.size(); I < E; ++I) {
    for (const size_t PointId : Clusters[I].PointIndices) {
      printInstructionRowCsv(PointId, OS);
    }
    OS << "\n\n";
  }
  return llvm::Error::success();
}

std::unordered_map<unsigned, std::vector<size_t>>
Analysis::makePointsPerSchedClass() const {
  std::unordered_map<unsigned, std::vector<size_t>> PointsPerSchedClass;
  const auto &Points = Clustering_.getPoints();
  for (size_t PointId = 0, E = Points.size(); PointId < E; ++PointId) {
    const InstructionBenchmark &Point = Points[PointId];
    if (!Point.Error.empty())
      continue;
    const auto OpcodeIt = MnemonicToOpcode_.find(Point.Key.OpcodeName);
    if (OpcodeIt == MnemonicToOpcode_.end())
      continue;
    const unsigned SchedClassId =
        InstrInfo_->get(OpcodeIt->second).getSchedClass();
    PointsPerSchedClass[SchedClassId].push_back(PointId);
  }
  return PointsPerSchedClass;
}

void Analysis::printSchedClassClustersHtml(std::vector<size_t> PointIds,
                                           llvm::raw_ostream &OS) const {
  assert(!PointIds.empty());
  // Sort the points by cluster id so that we can display them grouped by
  // cluster.
  std::sort(PointIds.begin(), PointIds.end(),
            [this](const size_t A, const size_t B) {
              return Clustering_.getClusterIdForPoint(A) <
                     Clustering_.getClusterIdForPoint(B);
            });
  const auto &Points = Clustering_.getPoints();
  OS << "<table class=\"sched-class-clusters\">";
  OS << "<tr><th>ClusterId</th><th>Opcode/Config</th>";
  for (const auto &Measurement : Points[PointIds[0]].Measurements) {
    OS << "<th>";
    if (Measurement.DebugString.empty())
      writeEscaped<kEscapeHtml>(OS, Measurement.Key);
    else
      writeEscaped<kEscapeHtml>(OS, Measurement.DebugString);
    OS << "</th>";
  }
  OS << "</tr>";
  for (size_t I = 0, E = PointIds.size(); I < E;) {
    const auto &CurrentClusterId =
        Clustering_.getClusterIdForPoint(PointIds[I]);
    OS << "<tr><td>";
    writeClusterId<kEscapeHtml>(OS, CurrentClusterId);
    OS << "</td><td><ul>";
    std::vector<BenchmarkMeasureStats> MeasurementStats(
        Points[PointIds[I]].Measurements.size());
    for (; I < E &&
           Clustering_.getClusterIdForPoint(PointIds[I]) == CurrentClusterId;
         ++I) {
      const auto &Point = Points[PointIds[I]];
      OS << "<li><span class=\"mono\">";
      writeEscaped<kEscapeHtml>(OS, Point.Key.OpcodeName);
      OS << "</span> <span class=\"mono\">";
      writeEscaped<kEscapeHtml>(OS, Point.Key.Config);
      OS << "</span></li>";
      for (size_t J = 0, F = Point.Measurements.size(); J < F; ++J) {
        MeasurementStats[J].push(Point.Measurements[J]);
      }
    }
    OS << "</ul></td>";
    for (const auto &Stats : MeasurementStats) {
      OS << "<td class=\"measurement\">";
      writeMeasurementValue<kEscapeHtml>(OS, Stats.avg());
      OS << "<br><span class=\"minmax\">[";
      writeMeasurementValue<kEscapeHtml>(OS, Stats.min());
      OS << ";";
      writeMeasurementValue<kEscapeHtml>(OS, Stats.max());
      OS << "]</span></td>";
    }
    OS << "</tr>";
  }
  OS << "</table>";
}

// Return the non-redundant list of WriteProcRes used by the given sched class.
// The scheduling model for LLVM is such that each instruction has a certain
// number of uops which consume resources which are described by WriteProcRes
// entries. Each entry describe how many cycles are spent on a specific ProcRes
// kind.
// For example, an instruction might have 3 uOps, one dispatching on P0
// (ProcResIdx=1) and two on P06 (ProcResIdx = 7).
// Note that LLVM additionally denormalizes resource consumption to include
// usage of super resources by subresources. So in practice if there exists a
// P016 (ProcResIdx=10), then the cycles consumed by P0 are also consumed by
// P06 (ProcResIdx = 7) and P016 (ProcResIdx = 10), and the resources consumed
// by P06 are also consumed by P016. In the figure below, parenthesized cycles
// denote implied usage of superresources by subresources:
//            P0      P06    P016
//     uOp1    1      (1)     (1)
//     uOp2            1      (1)
//     uOp3            1      (1)
//     =============================
//             1       3       3
// Eventually we end up with three entries for the WriteProcRes of the
// instruction:
//    {ProcResIdx=1,  Cycles=1}  // P0
//    {ProcResIdx=7,  Cycles=3}  // P06
//    {ProcResIdx=10, Cycles=3}  // P016
//
// Note that in this case, P016 does not contribute any cycles, so it would
// be removed by this function.
// FIXME: Move this to MCSubtargetInfo and use it in llvm-mca.
static llvm::SmallVector<llvm::MCWriteProcResEntry, 8>
getNonRedundantWriteProcRes(const llvm::MCSchedClassDesc &SCDesc,
                            const llvm::MCSubtargetInfo &STI) {
  llvm::SmallVector<llvm::MCWriteProcResEntry, 8> Result;
  const auto &SM = STI.getSchedModel();
  const unsigned NumProcRes = SM.getNumProcResourceKinds();

  // This assumes that the ProcResDescs are sorted in topological order, which
  // is guaranteed by the tablegen backend.
  llvm::SmallVector<float, 32> ProcResUnitUsage(NumProcRes);
  for (const auto *WPR = STI.getWriteProcResBegin(&SCDesc),
                  *const WPREnd = STI.getWriteProcResEnd(&SCDesc);
       WPR != WPREnd; ++WPR) {
    const llvm::MCProcResourceDesc *const ProcResDesc =
        SM.getProcResource(WPR->ProcResourceIdx);
    if (ProcResDesc->SubUnitsIdxBegin == nullptr) {
      // This is a ProcResUnit.
      Result.push_back({WPR->ProcResourceIdx, WPR->Cycles});
      ProcResUnitUsage[WPR->ProcResourceIdx] += WPR->Cycles;
    } else {
      // This is a ProcResGroup. First see if it contributes any cycles or if
      // it has cycles just from subunits.
      float RemainingCycles = WPR->Cycles;
      for (const auto *SubResIdx = ProcResDesc->SubUnitsIdxBegin;
           SubResIdx != ProcResDesc->SubUnitsIdxBegin + ProcResDesc->NumUnits;
           ++SubResIdx) {
        RemainingCycles -= ProcResUnitUsage[*SubResIdx];
      }
      if (RemainingCycles < 0.01f) {
        // The ProcResGroup contributes no cycles of its own.
        continue;
      }
      // The ProcResGroup contributes `RemainingCycles` cycles of its own.
      Result.push_back({WPR->ProcResourceIdx,
                        static_cast<uint16_t>(std::round(RemainingCycles))});
      // Spread the remaining cycles over all subunits.
      for (const auto *SubResIdx = ProcResDesc->SubUnitsIdxBegin;
           SubResIdx != ProcResDesc->SubUnitsIdxBegin + ProcResDesc->NumUnits;
           ++SubResIdx) {
        ProcResUnitUsage[*SubResIdx] += RemainingCycles / ProcResDesc->NumUnits;
      }
    }
  }
  return Result;
}

void Analysis::printSchedClassDescHtml(const llvm::MCSchedClassDesc &SCDesc,
                                       llvm::raw_ostream &OS) const {
  OS << "<table class=\"sched-class-desc\">";
  OS << "<tr><th>Valid</th><th>Variant</th><th>uOps</th><th>Latency</"
        "th><th>WriteProcRes</th></tr>";
  if (SCDesc.isValid()) {
    OS << "<tr><td>&#10004;</td>";
    OS << "<td>" << (SCDesc.isVariant() ? "&#10004;" : "&#10005;") << "</td>";
    OS << "<td>" << SCDesc.NumMicroOps << "</td>";
    // Latencies.
    OS << "<td><ul>";
    for (int I = 0, E = SCDesc.NumWriteLatencyEntries; I < E; ++I) {
      const auto *const Entry =
          SubtargetInfo_->getWriteLatencyEntry(&SCDesc, I);
      OS << "<li>" << Entry->Cycles;
      if (SCDesc.NumWriteLatencyEntries > 1) {
        // Dismabiguate if more than 1 latency.
        OS << " (WriteResourceID " << Entry->WriteResourceID << ")";
      }
      OS << "</li>";
    }
    OS << "</ul></td>";
    // WriteProcRes.
    OS << "<td><ul>";
    for (const auto &WPR :
         getNonRedundantWriteProcRes(SCDesc, *SubtargetInfo_)) {
      OS << "<li><span class=\"mono\">";
      writeEscaped<kEscapeHtml>(OS, SubtargetInfo_->getSchedModel()
                                        .getProcResource(WPR.ProcResourceIdx)
                                        ->Name);
      OS << "</span>: " << WPR.Cycles << "</li>";
    }
    OS << "</ul></td>";
    OS << "</tr>";
  } else {
    OS << "<tr><td>&#10005;</td><td></td><td></td></tr>";
  }
  OS << "</table>";
}

static constexpr const char kHtmlHead[] = R"(
<head>
<title>llvm-exegesis Analysis Results</title>
<style>
body {
  font-family: sans-serif
}
span.sched-class-name {
  font-weight: bold;
  font-family: monospace;
}
span.opcode {
  font-family: monospace;
}
span.config {
  font-family: monospace;
}
div.inconsistency {
  margin-top: 50px;
}
table {
  margin-left: 50px;
  border-collapse: collapse;
}
table, table tr,td,th {
  border: 1px solid #444;
}
table ul {
  padding-left: 0px;
  margin: 0px;
  list-style-type: none;
}
table.sched-class-clusters td {
  padding-left: 10px;
  padding-right: 10px;
  padding-top: 10px;
  padding-bottom: 10px;
}
table.sched-class-desc td {
  padding-left: 10px;
  padding-right: 10px;
  padding-top: 2px;
  padding-bottom: 2px;
}
span.mono {
  font-family: monospace;
}
span.minmax {
  color: #888;
}
td.measurement {
  text-align: center;
}
</style>
</head>
)";

template <>
llvm::Error Analysis::run<Analysis::PrintSchedClassInconsistencies>(
    llvm::raw_ostream &OS) const {
  // Print the header.
  OS << "<!DOCTYPE html><html>" << kHtmlHead << "<body>";
  OS << "<h1><span class=\"mono\">llvm-exegesis</span> Analysis Results</h1>";
  OS << "<h3>Triple: <span class=\"mono\">";
  writeEscaped<kEscapeHtml>(OS, Clustering_.getPoints()[0].LLVMTriple);
  OS << "</span></h3><h3>Cpu: <span class=\"mono\">";
  writeEscaped<kEscapeHtml>(OS, Clustering_.getPoints()[0].CpuName);
  OS << "</span></h3>";

  // All the points in a scheduling class should be in the same cluster.
  // Print any scheduling class for which this is not the case.
  for (const auto &SchedClassAndPoints : makePointsPerSchedClass()) {
    std::unordered_set<size_t> ClustersForSchedClass;
    for (const size_t PointId : SchedClassAndPoints.second) {
      const auto &ClusterId = Clustering_.getClusterIdForPoint(PointId);
      if (!ClusterId.isValid())
        continue; // Ignore noise and errors.
      ClustersForSchedClass.insert(ClusterId.getId());
    }
    if (ClustersForSchedClass.size() <= 1)
      continue; // Nothing weird.

    const auto &SchedModel = SubtargetInfo_->getSchedModel();
    const llvm::MCSchedClassDesc *const SCDesc =
        SchedModel.getSchedClassDesc(SchedClassAndPoints.first);
    if (!SCDesc)
      continue;
    OS << "<div class=\"inconsistency\"><p>Sched Class <span "
          "class=\"sched-class-name\">";
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    writeEscaped<kEscapeHtml>(OS, SCDesc->Name);
#else
    OS << SchedClassAndPoints.first;
#endif
    OS << "</span> contains instructions with distinct performance "
          "characteristics, falling into "
       << ClustersForSchedClass.size() << " clusters:</p>";
    printSchedClassClustersHtml(SchedClassAndPoints.second, OS);
    OS << "<p>llvm data:</p>";
    printSchedClassDescHtml(*SCDesc, OS);
    OS << "</div>";
  }

  OS << "</body></html>";
  return llvm::Error::success();
}

} // namespace exegesis
