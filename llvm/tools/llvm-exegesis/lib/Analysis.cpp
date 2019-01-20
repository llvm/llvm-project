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
#include "llvm/Support/FormatVariadic.h"
#include <limits>
#include <unordered_set>
#include <vector>

namespace llvm {
namespace exegesis {

static const char kCsvSep = ',';

static unsigned resolveSchedClassId(const llvm::MCSubtargetInfo &STI,
                                    unsigned SchedClassId,
                                    const llvm::MCInst &MCI) {
  const auto &SM = STI.getSchedModel();
  while (SchedClassId && SM.getSchedClassDesc(SchedClassId)->isVariant())
    SchedClassId =
        STI.resolveVariantSchedClass(SchedClassId, &MCI, SM.getProcessorID());
  return SchedClassId;
}

namespace {

enum EscapeTag { kEscapeCsv, kEscapeHtml, kEscapeHtmlString };

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

template <>
void writeEscaped<kEscapeHtmlString>(llvm::raw_ostream &OS,
                                     const llvm::StringRef S) {
  for (const char C : S) {
    if (C == '"')
      OS << "\\\"";
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
  // Given Value, if we wanted to serialize it to a string,
  // how many base-10 digits will we need to store, max?
  static constexpr auto MaxDigitCount =
      std::numeric_limits<decltype(Value)>::max_digits10;
  // Also, we will need a decimal separator.
  static constexpr auto DecimalSeparatorLen = 1; // '.' e.g.
  // So how long of a string will the serialization produce, max?
  static constexpr auto SerializationLen = MaxDigitCount + DecimalSeparatorLen;

  // WARNING: when changing the format, also adjust the small-size estimate ^.
  static constexpr StringLiteral SimpleFloatFormat = StringLiteral("{0:F}");

  writeEscaped<Tag>(
      OS,
      llvm::formatv(SimpleFloatFormat.data(), Value).sstr<SerializationLen>());
}

template <typename EscapeTag, EscapeTag Tag>
void Analysis::writeSnippet(llvm::raw_ostream &OS,
                            llvm::ArrayRef<uint8_t> Bytes,
                            const char *Separator) const {
  llvm::SmallVector<std::string, 3> Lines;
  // Parse the asm snippet and print it.
  while (!Bytes.empty()) {
    llvm::MCInst MI;
    uint64_t MISize = 0;
    if (!Disasm_->getInstruction(MI, MISize, Bytes, 0, llvm::nulls(),
                                 llvm::nulls())) {
      writeEscaped<Tag>(OS, llvm::join(Lines, Separator));
      writeEscaped<Tag>(OS, Separator);
      writeEscaped<Tag>(OS, "[error decoding asm snippet]");
      return;
    }
    llvm::SmallString<128> InstPrinterStr; // FIXME: magic number.
    llvm::raw_svector_ostream OSS(InstPrinterStr);
    InstPrinter_->printInst(&MI, OSS, "", *SubtargetInfo_);
    Bytes = Bytes.drop_front(MISize);
    Lines.emplace_back(llvm::StringRef(InstPrinterStr).trim());
  }
  writeEscaped<Tag>(OS, llvm::join(Lines, Separator));
}

// Prints a row representing an instruction, along with scheduling info and
// point coordinates (measurements).
void Analysis::printInstructionRowCsv(const size_t PointId,
                                      llvm::raw_ostream &OS) const {
  const InstructionBenchmark &Point = Clustering_.getPoints()[PointId];
  writeClusterId<kEscapeCsv>(OS, Clustering_.getClusterIdForPoint(PointId));
  OS << kCsvSep;
  writeSnippet<EscapeTag, kEscapeCsv>(OS, Point.AssembledSnippet, "; ");
  OS << kCsvSep;
  writeEscaped<kEscapeCsv>(OS, Point.Key.Config);
  OS << kCsvSep;
  assert(!Point.Key.Instructions.empty());
  const llvm::MCInst &MCI = Point.Key.Instructions[0];
  const unsigned SchedClassId = resolveSchedClassId(
      *SubtargetInfo_, InstrInfo_->get(MCI.getOpcode()).getSchedClass(), MCI);

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  const llvm::MCSchedClassDesc *const SCDesc =
      SubtargetInfo_->getSchedModel().getSchedClassDesc(SchedClassId);
  writeEscaped<kEscapeCsv>(OS, SCDesc->Name);
#else
  OS << SchedClassId;
#endif
  for (const auto &Measurement : Point.Measurements) {
    OS << kCsvSep;
    writeMeasurementValue<kEscapeCsv>(OS, Measurement.PerInstructionValue);
  }
  OS << "\n";
}

Analysis::Analysis(const llvm::Target &Target,
                   const InstructionBenchmarkClustering &Clustering)
    : Clustering_(Clustering) {
  if (Clustering.getPoints().empty())
    return;

  const InstructionBenchmark &FirstPoint = Clustering.getPoints().front();
  InstrInfo_.reset(Target.createMCInstrInfo());
  RegInfo_.reset(Target.createMCRegInfo(FirstPoint.LLVMTriple));
  AsmInfo_.reset(Target.createMCAsmInfo(*RegInfo_, FirstPoint.LLVMTriple));
  SubtargetInfo_.reset(Target.createMCSubtargetInfo(FirstPoint.LLVMTriple,
                                                    FirstPoint.CpuName, ""));
  InstPrinter_.reset(Target.createMCInstPrinter(
      llvm::Triple(FirstPoint.LLVMTriple), 0 /*default variant*/, *AsmInfo_,
      *InstrInfo_, *RegInfo_));

  Context_ = llvm::make_unique<llvm::MCContext>(AsmInfo_.get(), RegInfo_.get(),
                                                &ObjectFileInfo_);
  Disasm_.reset(Target.createMCDisassembler(*SubtargetInfo_, *Context_));
  assert(Disasm_ && "cannot create MCDisassembler. missing call to "
                    "InitializeXXXTargetDisassembler ?");
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
    const InstructionBenchmark &Point = Points[PointId];
    if (!Point.Error.empty())
      continue;
    assert(!Point.Key.Instructions.empty());
    // FIXME: we should be using the tuple of classes for instructions in the
    // snippet as key.
    const llvm::MCInst &MCI = Point.Key.Instructions[0];
    unsigned SchedClassId = InstrInfo_->get(MCI.getOpcode()).getSchedClass();
    const bool WasVariant = SchedClassId && SubtargetInfo_->getSchedModel()
                                                .getSchedClassDesc(SchedClassId)
                                                ->isVariant();
    SchedClassId = resolveSchedClassId(*SubtargetInfo_, SchedClassId, MCI);
    const auto IndexIt = SchedClassIdToIndex.find(SchedClassId);
    if (IndexIt == SchedClassIdToIndex.end()) {
      // Create a new entry.
      SchedClassIdToIndex.emplace(SchedClassId, Entries.size());
      ResolvedSchedClassAndPoints Entry(
          ResolvedSchedClass(*SubtargetInfo_, SchedClassId, WasVariant));
      Entry.PointIds.push_back(PointId);
      Entries.push_back(std::move(Entry));
    } else {
      // Append to the existing entry.
      Entries[IndexIt->second].PointIds.push_back(PointId);
    }
  }
  return Entries;
}

// Uops repeat the same opcode over again. Just show this opcode and show the
// whole snippet only on hover.
static void writeUopsSnippetHtml(llvm::raw_ostream &OS,
                                 const std::vector<llvm::MCInst> &Instructions,
                                 const llvm::MCInstrInfo &InstrInfo) {
  if (Instructions.empty())
    return;
  writeEscaped<kEscapeHtml>(OS, InstrInfo.getName(Instructions[0].getOpcode()));
  if (Instructions.size() > 1)
    OS << " (x" << Instructions.size() << ")";
}

// Latency tries to find a serial path. Just show the opcode path and show the
// whole snippet only on hover.
static void
writeLatencySnippetHtml(llvm::raw_ostream &OS,
                        const std::vector<llvm::MCInst> &Instructions,
                        const llvm::MCInstrInfo &InstrInfo) {
  bool First = true;
  for (const llvm::MCInst &Instr : Instructions) {
    if (First)
      First = false;
    else
      OS << " &rarr; ";
    writeEscaped<kEscapeHtml>(OS, InstrInfo.getName(Instr.getOpcode()));
  }
}

void Analysis::printSchedClassClustersHtml(
    const std::vector<SchedClassCluster> &Clusters,
    const ResolvedSchedClass &RSC, llvm::raw_ostream &OS) const {
  const auto &Points = Clustering_.getPoints();
  OS << "<table class=\"sched-class-clusters\">";
  OS << "<tr><th>ClusterId</th><th>Opcode/Config</th>";
  assert(!Clusters.empty());
  for (const auto &Measurement :
       Points[Clusters[0].getPointIds()[0]].Measurements) {
    OS << "<th>";
    writeEscaped<kEscapeHtml>(OS, Measurement.Key);
    OS << "</th>";
  }
  OS << "</tr>";
  for (const SchedClassCluster &Cluster : Clusters) {
    OS << "<tr class=\""
       << (Cluster.measurementsMatch(*SubtargetInfo_, RSC, Clustering_)
               ? "good-cluster"
               : "bad-cluster")
       << "\"><td>";
    writeClusterId<kEscapeHtml>(OS, Cluster.id());
    OS << "</td><td><ul>";
    for (const size_t PointId : Cluster.getPointIds()) {
      const auto &Point = Points[PointId];
      OS << "<li><span class=\"mono\" title=\"";
      writeSnippet<EscapeTag, kEscapeHtmlString>(OS, Point.AssembledSnippet,
                                                 "\n");
      OS << "\">";
      switch (Point.Mode) {
      case InstructionBenchmark::Latency:
        writeLatencySnippetHtml(OS, Point.Key.Instructions, *InstrInfo_);
        break;
      case InstructionBenchmark::Uops:
        writeUopsSnippetHtml(OS, Point.Key.Instructions, *InstrInfo_);
        break;
      default:
        llvm_unreachable("invalid mode");
      }
      OS << "</span> <span class=\"mono\">";
      writeEscaped<kEscapeHtml>(OS, Point.Key.Config);
      OS << "</span></li>";
    }
    OS << "</ul></td>";
    for (const auto &Stats : Cluster.getRepresentative()) {
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

Analysis::ResolvedSchedClass::ResolvedSchedClass(
    const llvm::MCSubtargetInfo &STI, unsigned ResolvedSchedClassId,
    bool WasVariant)
    : SchedClassId(ResolvedSchedClassId), SCDesc(STI.getSchedModel().getSchedClassDesc(ResolvedSchedClassId)),
      WasVariant(WasVariant),
      NonRedundantWriteProcRes(getNonRedundantWriteProcRes(*SCDesc, STI)),
      IdealizedProcResPressure(computeIdealizedProcResPressure(
          STI.getSchedModel(), NonRedundantWriteProcRes)) {
  assert((SCDesc == nullptr || !SCDesc->isVariant()) &&
         "ResolvedSchedClass should never be variant");
}

void Analysis::SchedClassCluster::addPoint(
    size_t PointId, const InstructionBenchmarkClustering &Clustering) {
  PointIds.push_back(PointId);
  const auto &Point = Clustering.getPoints()[PointId];
  if (ClusterId.isUndef()) {
    ClusterId = Clustering.getClusterIdForPoint(PointId);
    Representative.resize(Point.Measurements.size());
  }
  for (size_t I = 0, E = Point.Measurements.size(); I < E; ++I) {
    Representative[I].push(Point.Measurements[I]);
  }
  assert(ClusterId == Clustering.getClusterIdForPoint(PointId));
}

// Returns a ProxResIdx by id or name.
static unsigned findProcResIdx(const llvm::MCSubtargetInfo &STI,
                               const llvm::StringRef NameOrId) {
  // Interpret the key as an ProcResIdx.
  unsigned ProcResIdx = 0;
  if (llvm::to_integer(NameOrId, ProcResIdx, 10))
    return ProcResIdx;
  // Interpret the key as a ProcRes name.
  const auto &SchedModel = STI.getSchedModel();
  for (int I = 0, E = SchedModel.getNumProcResourceKinds(); I < E; ++I) {
    if (NameOrId == SchedModel.getProcResource(I)->Name)
      return I;
  }
  return 0;
}

bool Analysis::SchedClassCluster::measurementsMatch(
    const llvm::MCSubtargetInfo &STI, const ResolvedSchedClass &RSC,
    const InstructionBenchmarkClustering &Clustering) const {
  const size_t NumMeasurements = Representative.size();
  std::vector<BenchmarkMeasure> ClusterCenterPoint(NumMeasurements);
  std::vector<BenchmarkMeasure> SchedClassPoint(NumMeasurements);
  // Latency case.
  assert(!Clustering.getPoints().empty());
  const InstructionBenchmark::ModeE Mode = Clustering.getPoints()[0].Mode;
  if (Mode == InstructionBenchmark::Latency) {
    if (NumMeasurements != 1) {
      llvm::errs()
          << "invalid number of measurements in latency mode: expected 1, got "
          << NumMeasurements << "\n";
      return false;
    }
    // Find the latency.
    SchedClassPoint[0].PerInstructionValue = 0.0;
    for (unsigned I = 0; I < RSC.SCDesc->NumWriteLatencyEntries; ++I) {
      const llvm::MCWriteLatencyEntry *const WLE =
          STI.getWriteLatencyEntry(RSC.SCDesc, I);
      SchedClassPoint[0].PerInstructionValue =
          std::max<double>(SchedClassPoint[0].PerInstructionValue, WLE->Cycles);
    }
    ClusterCenterPoint[0].PerInstructionValue = Representative[0].avg();
  } else if (Mode == InstructionBenchmark::Uops) {
    for (int I = 0, E = Representative.size(); I < E; ++I) {
      const auto Key = Representative[I].key();
      uint16_t ProcResIdx = findProcResIdx(STI, Key);
      if (ProcResIdx > 0) {
        // Find the pressure on ProcResIdx `Key`.
        const auto ProcResPressureIt =
            std::find_if(RSC.IdealizedProcResPressure.begin(),
                         RSC.IdealizedProcResPressure.end(),
                         [ProcResIdx](const std::pair<uint16_t, float> &WPR) {
                           return WPR.first == ProcResIdx;
                         });
        SchedClassPoint[I].PerInstructionValue =
            ProcResPressureIt == RSC.IdealizedProcResPressure.end()
                ? 0.0
                : ProcResPressureIt->second;
      } else if (Key == "NumMicroOps") {
        SchedClassPoint[I].PerInstructionValue = RSC.SCDesc->NumMicroOps;
      } else {
        llvm::errs() << "expected `key` to be either a ProcResIdx or a ProcRes "
                        "name, got "
                     << Key << "\n";
        return false;
      }
      ClusterCenterPoint[I].PerInstructionValue = Representative[I].avg();
    }
  } else {
    llvm::errs() << "unimplemented measurement matching for mode " << Mode
                 << "\n";
    return false;
  }
  return Clustering.isNeighbour(ClusterCenterPoint, SchedClassPoint);
}

void Analysis::printSchedClassDescHtml(const ResolvedSchedClass &RSC,
                                       llvm::raw_ostream &OS) const {
  OS << "<table class=\"sched-class-desc\">";
  OS << "<tr><th>Valid</th><th>Variant</th><th>NumMicroOps</th><th>Latency</"
        "th><th>WriteProcRes</th><th title=\"This is the idealized unit "
        "resource (port) pressure assuming ideal distribution\">Idealized "
        "Resource Pressure</th></tr>";
  if (RSC.SCDesc->isValid()) {
    const auto &SM = SubtargetInfo_->getSchedModel();
    OS << "<tr><td>&#10004;</td>";
    OS << "<td>" << (RSC.WasVariant ? "&#10004;" : "&#10005;") << "</td>";
    OS << "<td>" << RSC.SCDesc->NumMicroOps << "</td>";
    // Latencies.
    OS << "<td><ul>";
    for (int I = 0, E = RSC.SCDesc->NumWriteLatencyEntries; I < E; ++I) {
      const auto *const Entry =
          SubtargetInfo_->getWriteLatencyEntry(RSC.SCDesc, I);
      OS << "<li>" << Entry->Cycles;
      if (RSC.SCDesc->NumWriteLatencyEntries > 1) {
        // Dismabiguate if more than 1 latency.
        OS << " (WriteResourceID " << Entry->WriteResourceID << ")";
      }
      OS << "</li>";
    }
    OS << "</ul></td>";
    // WriteProcRes.
    OS << "<td><ul>";
    for (const auto &WPR : RSC.NonRedundantWriteProcRes) {
      OS << "<li><span class=\"mono\">";
      writeEscaped<kEscapeHtml>(OS,
                                SM.getProcResource(WPR.ProcResourceIdx)->Name);
      OS << "</span>: " << WPR.Cycles << "</li>";
    }
    OS << "</ul></td>";
    // Idealized port pressure.
    OS << "<td><ul>";
    for (const auto &Pressure : RSC.IdealizedProcResPressure) {
      OS << "<li><span class=\"mono\">";
      writeEscaped<kEscapeHtml>(OS, SubtargetInfo_->getSchedModel()
                                        .getProcResource(Pressure.first)
                                        ->Name);
      OS << "</span>: ";
      writeMeasurementValue<kEscapeHtml>(OS, Pressure.second);
      OS << "</li>";
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
td.measurement {
  text-align: center;
}
tr.good-cluster td.measurement {
  color: #292
}
tr.bad-cluster td.measurement {
  color: #922
}
tr.good-cluster td.measurement span.minmax {
  color: #888;
}
tr.bad-cluster td.measurement span.minmax {
  color: #888;
}
</style>
</head>
)";

template <>
llvm::Error Analysis::run<Analysis::PrintSchedClassInconsistencies>(
    llvm::raw_ostream &OS) const {
  const auto &FirstPoint = Clustering_.getPoints()[0];
  // Print the header.
  OS << "<!DOCTYPE html><html>" << kHtmlHead << "<body>";
  OS << "<h1><span class=\"mono\">llvm-exegesis</span> Analysis Results</h1>";
  OS << "<h3>Triple: <span class=\"mono\">";
  writeEscaped<kEscapeHtml>(OS, FirstPoint.LLVMTriple);
  OS << "</span></h3><h3>Cpu: <span class=\"mono\">";
  writeEscaped<kEscapeHtml>(OS, FirstPoint.CpuName);
  OS << "</span></h3>";

  for (const auto &RSCAndPoints : makePointsPerSchedClass()) {
    if (!RSCAndPoints.RSC.SCDesc)
      continue;
    // Bucket sched class points into sched class clusters.
    std::vector<SchedClassCluster> SchedClassClusters;
    for (const size_t PointId : RSCAndPoints.PointIds) {
      const auto &ClusterId = Clustering_.getClusterIdForPoint(PointId);
      if (!ClusterId.isValid())
        continue; // Ignore noise and errors. FIXME: take noise into account ?
      auto SchedClassClusterIt =
          std::find_if(SchedClassClusters.begin(), SchedClassClusters.end(),
                       [ClusterId](const SchedClassCluster &C) {
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
    if (llvm::all_of(SchedClassClusters,
                     [this, &RSCAndPoints](const SchedClassCluster &C) {
                       return C.measurementsMatch(
                           *SubtargetInfo_, RSCAndPoints.RSC, Clustering_);
                     }))
      continue; // Nothing weird.

    OS << "<div class=\"inconsistency\"><p>Sched Class <span "
          "class=\"sched-class-name\">";
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    writeEscaped<kEscapeHtml>(OS, RSCAndPoints.RSC.SCDesc->Name);
#else
    OS << RSCAndPoints.RSC.SchedClassId;
#endif
    OS << "</span> contains instructions whose performance characteristics do"
          " not match that of LLVM:</p>";
    printSchedClassClustersHtml(SchedClassClusters, RSCAndPoints.RSC, OS);
    OS << "<p>llvm SchedModel data:</p>";
    printSchedClassDescHtml(RSCAndPoints.RSC, OS);
    OS << "</div>";
  }

  OS << "</body></html>";
  return llvm::Error::success();
}

// Distributes a pressure budget as evenly as possible on the provided subunits
// given the already existing port pressure distribution.
//
// The algorithm is as follows: while there is remaining pressure to
// distribute, find the subunits with minimal pressure, and distribute
// remaining pressure equally up to the pressure of the unit with
// second-to-minimal pressure.
// For example, let's assume we want to distribute 2*P1256
// (Subunits = [P1,P2,P5,P6]), and the starting DensePressure is:
//     DensePressure =        P0   P1   P2   P3   P4   P5   P6   P7
//                           0.1  0.3  0.2  0.0  0.0  0.5  0.5  0.5
//     RemainingPressure = 2.0
// We sort the subunits by pressure:
//     Subunits = [(P2,p=0.2), (P1,p=0.3), (P5,p=0.5), (P6, p=0.5)]
// We'll first start by the subunits with minimal pressure, which are at
// the beginning of the sorted array. In this example there is one (P2).
// The subunit with second-to-minimal pressure is the next one in the
// array (P1). So we distribute 0.1 pressure to P2, and remove 0.1 cycles
// from the budget.
//     Subunits = [(P2,p=0.3), (P1,p=0.3), (P5,p=0.5), (P5,p=0.5)]
//     RemainingPressure = 1.9
// We repeat this process: distribute 0.2 pressure on each of the minimal
// P2 and P1, decrease budget by 2*0.2:
//     Subunits = [(P2,p=0.5), (P1,p=0.5), (P5,p=0.5), (P5,p=0.5)]
//     RemainingPressure = 1.5
// There are no second-to-minimal subunits so we just share the remaining
// budget (1.5 cycles) equally:
//     Subunits = [(P2,p=0.875), (P1,p=0.875), (P5,p=0.875), (P5,p=0.875)]
//     RemainingPressure = 0.0
// We stop as there is no remaining budget to distribute.
void distributePressure(float RemainingPressure,
                        llvm::SmallVector<uint16_t, 32> Subunits,
                        llvm::SmallVector<float, 32> &DensePressure) {
  // Find the number of subunits with minimal pressure (they are at the
  // front).
  llvm::sort(Subunits, [&DensePressure](const uint16_t A, const uint16_t B) {
    return DensePressure[A] < DensePressure[B];
  });
  const auto getPressureForSubunit = [&DensePressure,
                                      &Subunits](size_t I) -> float & {
    return DensePressure[Subunits[I]];
  };
  size_t NumMinimalSU = 1;
  while (NumMinimalSU < Subunits.size() &&
         getPressureForSubunit(NumMinimalSU) == getPressureForSubunit(0)) {
    ++NumMinimalSU;
  }
  while (RemainingPressure > 0.0f) {
    if (NumMinimalSU == Subunits.size()) {
      // All units are minimal, just distribute evenly and be done.
      for (size_t I = 0; I < NumMinimalSU; ++I) {
        getPressureForSubunit(I) += RemainingPressure / NumMinimalSU;
      }
      return;
    }
    // Distribute the remaining pressure equally.
    const float MinimalPressure = getPressureForSubunit(NumMinimalSU - 1);
    const float SecondToMinimalPressure = getPressureForSubunit(NumMinimalSU);
    assert(MinimalPressure < SecondToMinimalPressure);
    const float Increment = SecondToMinimalPressure - MinimalPressure;
    if (RemainingPressure <= NumMinimalSU * Increment) {
      // There is not enough remaining pressure.
      for (size_t I = 0; I < NumMinimalSU; ++I) {
        getPressureForSubunit(I) += RemainingPressure / NumMinimalSU;
      }
      return;
    }
    // Bump all minimal pressure subunits to `SecondToMinimalPressure`.
    for (size_t I = 0; I < NumMinimalSU; ++I) {
      getPressureForSubunit(I) = SecondToMinimalPressure;
      RemainingPressure -= SecondToMinimalPressure;
    }
    while (NumMinimalSU < Subunits.size() &&
           getPressureForSubunit(NumMinimalSU) == SecondToMinimalPressure) {
      ++NumMinimalSU;
    }
  }
}

std::vector<std::pair<uint16_t, float>> computeIdealizedProcResPressure(
    const llvm::MCSchedModel &SM,
    llvm::SmallVector<llvm::MCWriteProcResEntry, 8> WPRS) {
  // DensePressure[I] is the port pressure for Proc Resource I.
  llvm::SmallVector<float, 32> DensePressure(SM.getNumProcResourceKinds());
  llvm::sort(WPRS, [](const llvm::MCWriteProcResEntry &A,
                      const llvm::MCWriteProcResEntry &B) {
    return A.ProcResourceIdx < B.ProcResourceIdx;
  });
  for (const llvm::MCWriteProcResEntry &WPR : WPRS) {
    // Get units for the entry.
    const llvm::MCProcResourceDesc *const ProcResDesc =
        SM.getProcResource(WPR.ProcResourceIdx);
    if (ProcResDesc->SubUnitsIdxBegin == nullptr) {
      // This is a ProcResUnit.
      DensePressure[WPR.ProcResourceIdx] += WPR.Cycles;
    } else {
      // This is a ProcResGroup.
      llvm::SmallVector<uint16_t, 32> Subunits(ProcResDesc->SubUnitsIdxBegin,
                                               ProcResDesc->SubUnitsIdxBegin +
                                                   ProcResDesc->NumUnits);
      distributePressure(WPR.Cycles, Subunits, DensePressure);
    }
  }
  // Turn dense pressure into sparse pressure by removing zero entries.
  std::vector<std::pair<uint16_t, float>> Pressure;
  for (unsigned I = 0, E = SM.getNumProcResourceKinds(); I < E; ++I) {
    if (DensePressure[I] > 0.0f)
      Pressure.emplace_back(I, DensePressure[I]);
  }
  return Pressure;
}

} // namespace exegesis
} // namespace llvm
