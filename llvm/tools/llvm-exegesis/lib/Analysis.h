//===-- Analysis.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Analysis output for benchmark results.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_EXEGESIS_ANALYSIS_H
#define LLVM_TOOLS_LLVM_EXEGESIS_ANALYSIS_H

#include "Clustering.h"
#include "DisassemblerHelper.h"
#include "SchedClassResolution.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <memory>

namespace llvm {
namespace exegesis {

// Abstractions over analysis results which make it easier
// to print them in different formats.
namespace AnalysisResult {
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
using SchedClassName = StringRef;
#else
using SchedClassName = unsigned;
#endif

struct Cluster {
  BenchmarkClustering::ClusterId Id;
  std::string Snippet;
  StringRef Config;
  SchedClassName SchedClass;
  SmallVector<double, 2> Measurements;
};
struct Clusters {
  SmallVector<StringRef, 2> MeasurementNames;
  std::vector<Cluster> Data;
};

struct SchedClassInconsistency {
  // === SchedClass properties ===
  SchedClassName Name;
  bool IsVariant;
  unsigned NumMicroOps;

  // {WriteResourceID, Latency}
  SmallVector<std::pair<unsigned, unsigned>, 2> Latency;

  double RThroughput;

  struct WriteProcResEntry {
    StringRef ProcResName;
    uint16_t AcquireAtCycle;
    uint16_t ReleaseAtCycle;
    std::optional<double> ResourcePressure;
  };
  SmallVector<WriteProcResEntry, 2> WriteProcResEntries;

  // === Collected data ===
  struct Point {
    StringRef Opcode;
    StringRef Config;
    std::string Snippet;
  };
  // [min, mean, max]
  using DataPoint = std::array<double, 3>;

  struct Measurement {
    BenchmarkClustering::ClusterId ClusterId;
    SmallVector<Point, 32> Points;
    SmallVector<DataPoint, 2> Data;
    bool IsInconsistent;
  };
  SmallVector<StringRef, 2> MeasurementNames;
  SmallVector<Measurement, 4> Measurements;
};
struct SchedClassInconsistencies {
  StringRef Triple;
  StringRef CPUName;
  double Epsilon;

  std::vector<SchedClassInconsistency> Inconsistencies;
};

/// Printers
void printCSV(raw_ostream &OS, const Clusters &Data);
void printYAML(raw_ostream &OS, const Clusters &Data);

void printHTML(raw_ostream &OS, const SchedClassInconsistencies &Data);
void printYAML(raw_ostream &OS, const SchedClassInconsistencies &Data);
} // namespace AnalysisResult

// A helper class to analyze benchmark results for a target.
class Analysis {
public:
  Analysis(const LLVMState &State,
           const BenchmarkClustering &Clustering,
           double AnalysisInconsistencyEpsilon,
           bool AnalysisDisplayUnstableOpcodes);

  // Prints a csv of instructions for each cluster.
  struct PrintClusters {
    using Result = AnalysisResult::Clusters;
  };
  // Find potential errors in the scheduling information given measurements.
  struct PrintSchedClassInconsistencies {
    using Result = AnalysisResult::SchedClassInconsistencies;
  };

  enum OutputFormat { OF_Default, OF_YAML, OF_JSON };
  template <typename Pass>
  Error run(raw_ostream &OS, OutputFormat Format) const;

private:
  using ClusterId = BenchmarkClustering::ClusterId;

  template <typename Pass, typename ResultT = typename Pass::Result>
  Expected<ResultT> exportResult() const;

  // Represents the intersection of a sched class and a cluster.
  class SchedClassCluster {
  public:
    const BenchmarkClustering::ClusterId &id() const {
      return ClusterId;
    }

    const std::vector<size_t> &getPointIds() const { return PointIds; }

    void addPoint(size_t PointId,
                  const BenchmarkClustering &Clustering);

    // Return the cluster centroid.
    const SchedClassClusterCentroid &getCentroid() const { return Centroid; }

    // Returns true if the cluster representative measurements match that of SC.
    bool
    measurementsMatch(const MCSubtargetInfo &STI, const ResolvedSchedClass &SC,
                      const BenchmarkClustering &Clustering,
                      const double AnalysisInconsistencyEpsilonSquared_) const;

  private:
    BenchmarkClustering::ClusterId ClusterId;
    std::vector<size_t> PointIds;
    // Measurement stats for the points in the SchedClassCluster.
    SchedClassClusterCentroid Centroid;
  };

  // A pair of (Sched Class, indices of points that belong to the sched
  // class).
  struct ResolvedSchedClassAndPoints {
    explicit ResolvedSchedClassAndPoints(ResolvedSchedClass &&RSC);

    ResolvedSchedClass RSC;
    std::vector<size_t> PointIds;
  };

  // Builds a list of ResolvedSchedClassAndPoints.
  std::vector<ResolvedSchedClassAndPoints> makePointsPerSchedClass() const;

  // Print non-escaped snippet.
  void printSnippet(raw_ostream &OS, ArrayRef<uint8_t> Bytes,
                    const char *Separator = "\n") const;

  const BenchmarkClustering &Clustering_;
  const LLVMState &State_;
  std::unique_ptr<DisassemblerHelper> DisasmHelper_;
  const double AnalysisInconsistencyEpsilonSquared_;
  const bool AnalysisDisplayUnstableOpcodes_;
};

} // namespace exegesis
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_EXEGESIS_CLUSTERING_H
