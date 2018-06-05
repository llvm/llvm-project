//===-- Analysis.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
#include <string>
#include <unordered_map>

namespace exegesis {

// A helper class to analyze benchmark results for a target.
class Analysis {
public:
  Analysis(const llvm::Target &Target,
           const InstructionBenchmarkClustering &Clustering);

  // Prints a csv of instructions for each cluster.
  struct PrintClusters {};
  // Find potential errors in the scheduling information given measurements.
  struct PrintSchedClassInconsistencies {};

  template <typename Pass> llvm::Error run(llvm::raw_ostream &OS) const;

private:
  using ClusterId = InstructionBenchmarkClustering::ClusterId;

  // An llvm::MCSchedClassDesc augmented with some additional data.
  struct SchedClass {
    SchedClass(const llvm::MCSchedClassDesc &SD,
               const llvm::MCSubtargetInfo &STI);

    const llvm::MCSchedClassDesc &SCDesc;
    const llvm::SmallVector<llvm::MCWriteProcResEntry, 8>
        NonRedundantWriteProcRes;
    const std::vector<std::pair<uint16_t, float>> IdealizedProcResPressure;
  };

  // Represents the intersection of a sched class and a cluster.
  class SchedClassCluster {
  public:
    const InstructionBenchmarkClustering::ClusterId &id() const {
      return ClusterId;
    }

    const std::vector<size_t> &getPointIds() const { return PointIds; }

    // Return the cluster centroid.
    const std::vector<BenchmarkMeasureStats> &getRepresentative() const {
      return Representative;
    }

    // Returns true if the cluster representative measurements match that of SC.
    bool
    measurementsMatch(const llvm::MCSubtargetInfo &STI, const SchedClass &SC,
                      const InstructionBenchmarkClustering &Clustering) const;

    void addPoint(size_t PointId,
                  const InstructionBenchmarkClustering &Clustering);

  private:
    InstructionBenchmarkClustering::ClusterId ClusterId;
    std::vector<size_t> PointIds;
    // Measurement stats for the points in the SchedClassCluster.
    std::vector<BenchmarkMeasureStats> Representative;
  };

  void printInstructionRowCsv(size_t PointId, llvm::raw_ostream &OS) const;

  void
  printSchedClassClustersHtml(const std::vector<SchedClassCluster> &Clusters,
                              const SchedClass &SC,
                              llvm::raw_ostream &OS) const;
  void printSchedClassDescHtml(const SchedClass &SC,
                               llvm::raw_ostream &OS) const;

  // Builds a map of Sched Class -> indices of points that belong to the sched
  // class.
  std::unordered_map<unsigned, std::vector<size_t>>
  makePointsPerSchedClass() const;

  const InstructionBenchmarkClustering &Clustering_;
  std::unique_ptr<llvm::MCSubtargetInfo> SubtargetInfo_;
  std::unique_ptr<llvm::MCInstrInfo> InstrInfo_;
  std::unordered_map<std::string, unsigned> MnemonicToOpcode_;
};

// Computes the idealized ProcRes Unit pressure. This is the expected
// distribution if the CPU scheduler can distribute the load as evenly as
// possible.
std::vector<std::pair<uint16_t, float>> computeIdealizedProcResPressure(
    const llvm::MCSchedModel &SM,
    llvm::SmallVector<llvm::MCWriteProcResEntry, 8> WPRS);

} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_CLUSTERING_H
