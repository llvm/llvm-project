//===-- AnalysisPrinters.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis.h"
#include "BenchmarkResult.h"
#include "Clustering.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/YAMLTraits.h"
#include <limits>

using namespace llvm;
using namespace llvm::exegesis;

static const char kCsvSep = ',';

namespace {
enum EscapeTag { kNone, kEscapeCsv, kEscapeHtml };

template <EscapeTag Tag> void writeEscaped(raw_ostream &OS, const StringRef S) {
  OS << S;
}

template <> void writeEscaped<kEscapeCsv>(raw_ostream &OS, const StringRef S) {
  if (!S.contains(kCsvSep)) {
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

template <> void writeEscaped<kEscapeHtml>(raw_ostream &OS, const StringRef S) {
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

template <EscapeTag Tag>
void writeClusterId(raw_ostream &OS,
                    const BenchmarkClustering::ClusterId &CID) {
  if (CID.isNoise())
    writeEscaped<Tag>(OS, "[noise]");
  else if (CID.isError())
    writeEscaped<Tag>(OS, "[error]");
  else
    OS << CID.getId();
}

template <EscapeTag Tag>
void writeMeasurementValue(raw_ostream &OS, const double Value) {
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
      OS, formatv(SimpleFloatFormat.data(), Value).sstr<SerializationLen>());
}
} // anonymous namespace

void llvm::exegesis::AnalysisResult::printCSV(
    raw_ostream &OS, const AnalysisResult::Clusters &Result) {
  // Write the header.
  OS << "cluster_id" << kCsvSep << "opcode_name" << kCsvSep << "config"
     << kCsvSep << "sched_class";
  for (StringRef Name : Result.MeasurementNames) {
    OS << kCsvSep;
    writeEscaped<kEscapeCsv>(OS, Name);
  }
  OS << "\n";

  // Prints a row representing an instruction, along with scheduling info and
  // point coordinates (measurements).
  for (const auto &Row : Result.Data) {
    writeClusterId<kEscapeCsv>(OS, Row.Id);
    OS << kCsvSep;
    writeEscaped<kEscapeCsv>(OS, Row.Snippet);
    OS << kCsvSep;
    writeEscaped<kEscapeCsv>(OS, Row.Config);
    OS << kCsvSep;
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    writeEscaped<kEscapeCsv>(OS, Row.SchedClass);
#else
    OS << Row.SchedClass;
#endif
    for (double Measurement : Row.Measurements) {
      OS << kCsvSep;
      writeMeasurementValue<kEscapeCsv>(OS, Measurement);
    }
    OS << "\n";
  }
}

namespace llvm {
namespace yaml {
template <> struct ScalarTraits<BenchmarkClustering::ClusterId> {
  static void output(const BenchmarkClustering::ClusterId &Value, void *,
                     raw_ostream &OS) {
    if (Value.isUnstable()) {
      OS << "unstable<";
      writeClusterId<kNone>(OS, Value);
      OS << ">";
    } else {
      writeClusterId<kNone>(OS, Value);
    }
  }

  static StringRef input(StringRef Text, void *,
                         BenchmarkClustering::ClusterId &Value) {
    size_t Id;

    if (Text == "[noise]") {
      Value = BenchmarkClustering::ClusterId::noise();
    } else if (Text == "[error]") {
      Value = BenchmarkClustering::ClusterId::error();
    } else if (Text.consume_front("unstable<")) {
      if (!Text.consumeInteger(10, Id) && Text == ">")
        Value = BenchmarkClustering::ClusterId::makeValidUnstable(Id);
      else
        return "Expect 'unstable<cluster id>'";
    } else if (!Text.getAsInteger(10, Id)) {
      Value = BenchmarkClustering::ClusterId::makeValid(Id);
    } else {
      return "Unrecognized ClusterId value";
    }

    return StringRef();
  }

  static QuotingType mustQuote(StringRef) { return QuotingType::Single; }

  static const bool flow = true;
};

template <> struct SequenceElementTraits<AnalysisResult::Cluster> {
  static const bool flow = false;
};

template <> struct MappingTraits<AnalysisResult::Cluster> {
  static void mapping(IO &Io, AnalysisResult::Cluster &Obj) {
    Io.mapRequired("id", Obj.Id);
    Io.mapRequired("snippet", Obj.Snippet);
    Io.mapRequired("config", Obj.Config);
    Io.mapRequired("sched_class", Obj.SchedClass);
    Io.mapRequired("measurements", Obj.Measurements);
  }
};

template <> struct MappingTraits<AnalysisResult::Clusters> {
  static void mapping(IO &Io, AnalysisResult::Clusters &Obj) {
    Io.mapRequired("measurement_names", Obj.MeasurementNames);
    Io.mapRequired("data", Obj.Data);
  }
};
} // namespace yaml
} // namespace llvm

void llvm::exegesis::AnalysisResult::printYAML(
    raw_ostream &OS, const AnalysisResult::Clusters &Result) {
  yaml::Output YOS(OS, /*Ctx=*/nullptr, /*WrapColumn=*/200);
  YOS << const_cast<AnalysisResult::Clusters &>(Result);
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

namespace {
using namespace AnalysisResult;
void printSchedClassClustersHTML(
    raw_ostream &OS,
    ArrayRef<SchedClassInconsistency::Measurement> Measurements,
    ArrayRef<StringRef> MeasurementNames) {
  OS << "<table class=\"sched-class-clusters\">";
  OS << "<tr><th>ClusterId</th><th>Opcode/Config</th>";
  for (StringRef Name : MeasurementNames) {
    OS << "<th>";
    writeEscaped<kEscapeHtml>(OS, Name);
    OS << "</th>";
  }
  OS << "</tr>";
  for (const auto &M : Measurements) {
    OS << "<tr class=\"" << (M.IsInconsistent ? "bad-cluster" : "good-cluster")
       << "\"><td>";
    writeClusterId<kEscapeHtml>(OS, M.ClusterId);
    OS << "</td><td><ul>";
    for (const auto &P : M.Points) {
      // Show up when the cursor is hovered over.
      OS << "<li><span class=\"mono\" title=\"";
      writeEscaped<kEscapeHtml>(OS, P.Snippet);
      OS << "\">";

      writeEscaped<kEscapeHtml>(OS, P.Opcode);
      OS << "</span> <span class=\"mono\">";
      writeEscaped<kEscapeHtml>(OS, P.Config);
      OS << "</span></li>";
    }
    OS << "</ul></td>";

    for (const auto &Stats : M.Data) {
      OS << "<td class=\"measurement\">";
      writeMeasurementValue<kEscapeHtml>(OS, Stats[1]);
      OS << "<br><span class=\"minmax\">[";
      writeMeasurementValue<kEscapeHtml>(OS, Stats[0]);
      OS << ";";
      writeMeasurementValue<kEscapeHtml>(OS, Stats[2]);
      OS << "]</span></td>";
    }
    OS << "</tr>";
  }
  OS << "</table>";
}

void printSchedClassDescHTML(raw_ostream &OS,
                             const SchedClassInconsistency &SCI) {
  OS << "<table class=\"sched-class-desc\">";
  OS << "<tr><th>Valid</th><th>Variant</th><th>NumMicroOps</th><th>Normalized "
        "Latency</"
        "th><th>RThroughput</th><th>WriteProcRes</th><th title=\"This is the "
        "idealized unit resource (port) pressure assuming ideal "
        "distribution\">Idealized Resource Pressure</th></tr>";

  OS << "<tr><td>&#10004;</td>";
  OS << "<td>" << (SCI.IsVariant ? "&#10004;" : "&#10005;") << "</td>";
  OS << "<td>" << SCI.NumMicroOps << "</td>";
  // Latencies.
  OS << "<td><ul>";
  for (const auto &L : SCI.Latency) {
    OS << "<li>" << L.second;
    if (SCI.Latency.size() > 1) {
      // Dismabiguate if more than 1 latency.
      OS << " (WriteResourceID " << L.first << ")";
    }
    OS << "</li>";
  }
  OS << "</ul></td>";
  // Inverse throughput.
  OS << "<td>";
  writeMeasurementValue<kEscapeHtml>(OS, SCI.RThroughput);
  OS << "</td>";
  // WriteProcRes.
  OS << "<td><ul>";
  for (const auto &WPR : SCI.WriteProcResEntries) {
    OS << "<li><span class=\"mono\">";
    writeEscaped<kEscapeHtml>(OS, WPR.ProcResName);
    OS << "</span>: "
       << formatv("[{0}, {1}]", WPR.AcquireAtCycle, WPR.ReleaseAtCycle)
       << "</li>";
  }
  OS << "</ul></td>";
  // Idealized port pressure.
  OS << "<td><ul>";
  for (const auto &WPR : SCI.WriteProcResEntries) {
    if (!WPR.ResourcePressure.has_value())
      continue;
    OS << "<li><span class=\"mono\">";
    writeEscaped<kEscapeHtml>(OS, WPR.ProcResName);
    OS << "</span>: ";
    writeMeasurementValue<kEscapeHtml>(OS, *WPR.ResourcePressure);
    OS << "</li>";
  }
  OS << "</ul></td>";
  OS << "</tr>";
  OS << "</table>";
}
} // anonymous namespace

void llvm::exegesis::AnalysisResult::printHTML(
    raw_ostream &OS, const AnalysisResult::SchedClassInconsistencies &Result) {
  // Print the header.
  OS << "<!DOCTYPE html><html>" << kHtmlHead << "<body>";
  OS << "<h1><span class=\"mono\">llvm-exegesis</span> Analysis Results</h1>";
  OS << "<h3>Triple: <span class=\"mono\">";
  writeEscaped<kEscapeHtml>(OS, Result.Triple);
  OS << "</span></h3><h3>Cpu: <span class=\"mono\">";
  writeEscaped<kEscapeHtml>(OS, Result.CPUName);
  OS << "</span></h3>";
  OS << "<h3>Epsilon: <span class=\"mono\">" << format("%0.2f", Result.Epsilon)
     << "</span></h3>";

  for (const auto &SCI : Result.Inconsistencies) {
    OS << "<div class=\"inconsistency\"><p>Sched Class <span "
          "class=\"sched-class-name\">";
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    writeEscaped<kEscapeHtml>(OS, SCI.Name);
#else
    OS << SCI.Name;
#endif
    OS << "</span> contains instructions whose performance characteristics do"
          " not match that of LLVM:</p>";
    printSchedClassClustersHTML(OS, SCI.Measurements, SCI.MeasurementNames);
    OS << "<p>llvm SchedModel data:</p>";
    printSchedClassDescHTML(OS, SCI);
    OS << "</div>";
  }

  // TODO: Print noise data points.
  OS << "</body></html>";
}

namespace llvm {
namespace yaml {

template <>
struct SequenceElementTraits<AnalysisResult::SchedClassInconsistency> {
  static const bool flow = false;
};

template <>
struct SequenceElementTraits<
    AnalysisResult::SchedClassInconsistency::WriteProcResEntry> {
  static const bool flow = false;
};

template <>
struct MappingTraits<
    AnalysisResult::SchedClassInconsistency::WriteProcResEntry> {
  static void
  mapping(IO &Io,
          AnalysisResult::SchedClassInconsistency::WriteProcResEntry &Obj) {
    Io.mapRequired("name", Obj.ProcResName);
    Io.mapRequired("acquire_cycle", Obj.AcquireAtCycle);
    Io.mapRequired("release_cycle", Obj.ReleaseAtCycle);
    Io.mapOptional("pressure", Obj.ResourcePressure);
  }

  static const bool flow = true;
};

template <>
struct SequenceElementTraits<AnalysisResult::SchedClassInconsistency::Point> {
  static const bool flow = false;
};

template <>
struct MappingTraits<AnalysisResult::SchedClassInconsistency::Point> {
  static void mapping(IO &Io,
                      AnalysisResult::SchedClassInconsistency::Point &Obj) {
    Io.mapRequired("opcode", Obj.Opcode);
    Io.mapRequired("config", Obj.Config);
    Io.mapRequired("snippet", Obj.Snippet);
  }
};

template <>
struct SequenceElementTraits<
    AnalysisResult::SchedClassInconsistency::DataPoint> {
  static const bool flow = true;
};

template <>
struct SequenceTraits<AnalysisResult::SchedClassInconsistency::DataPoint> {
  using DataPoint = AnalysisResult::SchedClassInconsistency::DataPoint;
  static size_t size(IO &, DataPoint &Obj) { return Obj.size(); }

  static DataPoint::value_type &element(IO &, DataPoint &Obj, size_t Index) {
    return Obj[Index];
  }

  static const bool flow = true;
};

template <>
struct SequenceElementTraits<
    AnalysisResult::SchedClassInconsistency::Measurement> {
  static const bool flow = false;
};

template <>
struct MappingTraits<AnalysisResult::SchedClassInconsistency::Measurement> {
  static void
  mapping(IO &Io, AnalysisResult::SchedClassInconsistency::Measurement &Obj) {
    Io.mapRequired("cluster_id", Obj.ClusterId);
    Io.mapRequired("points", Obj.Points);
    Io.mapRequired("data", Obj.Data);
    Io.mapRequired("inconsistent", Obj.IsInconsistent);
  }
};

template <> struct SequenceTraits<std::pair<unsigned, unsigned>> {
  using Pair = std::pair<unsigned, unsigned>;
  static size_t size(IO &, Pair &) { return 2; }

  static unsigned &element(IO &, Pair &Obj, size_t Index) {
    return Index == 0 ? Obj.first : Obj.second;
  }

  static const bool flow = true;
};

template <> struct SequenceElementTraits<std::pair<unsigned, unsigned>> {
  static const bool flow = true;
};

template <> struct MappingTraits<AnalysisResult::SchedClassInconsistency> {
  static void mapping(IO &Io, AnalysisResult::SchedClassInconsistency &Obj) {
    Io.mapRequired("name", Obj.Name);
    Io.mapRequired("variant", Obj.IsVariant);
    Io.mapRequired("num_microops", Obj.NumMicroOps);
    Io.mapRequired("latency", Obj.Latency);
    Io.mapRequired("rthroughput", Obj.RThroughput);

    Io.mapRequired("write_proc_res", Obj.WriteProcResEntries);

    Io.mapRequired("measurement_names", Obj.MeasurementNames);
    Io.mapRequired("measurements", Obj.Measurements);
  }
};

template <> struct MappingTraits<AnalysisResult::SchedClassInconsistencies> {
  static void mapping(IO &Io, AnalysisResult::SchedClassInconsistencies &Obj) {
    Io.mapRequired("triple", Obj.Triple);
    Io.mapRequired("cpu", Obj.CPUName);
    Io.mapOptional("epsilon", Obj.Epsilon);
    Io.mapRequired("inconsistencies", Obj.Inconsistencies);
  }
};
} // namespace yaml
} // namespace llvm

void llvm::exegesis::AnalysisResult::printYAML(
    raw_ostream &OS, const AnalysisResult::SchedClassInconsistencies &Result) {
  yaml::Output YOS(OS, /*Ctx=*/nullptr, /*WrapColumn=*/200);
  YOS << const_cast<AnalysisResult::SchedClassInconsistencies &>(Result);
}
