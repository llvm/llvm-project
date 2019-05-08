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
#include "SchedClassResolution.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <set>
#include <string>
#include <unordered_map>

namespace llvm {
namespace exegesis {

// A helper class to analyze benchmark results for a target.
class Analysis {
public:
  Analysis(const llvm::Target &Target,
           std::unique_ptr<llvm::MCInstrInfo> InstrInfo,
           const InstructionBenchmarkClustering &Clustering,
           double AnalysisInconsistencyEpsilon,
           bool AnalysisDisplayUnstableOpcodes);

  // Prints a csv of instructions for each cluster.
  struct PrintClusters {};
  // Find potential errors in the scheduling information given measurements.
  struct PrintSchedClassInconsistencies {};

  template <typename Pass> llvm::Error run(llvm::raw_ostream &OS) const;

private:
  using ClusterId = InstructionBenchmarkClustering::ClusterId;

  // Represents the intersection of a sched class and a cluster.
  class SchedClassCluster {
  public:
    const InstructionBenchmarkClustering::ClusterId &id() const {
      return ClusterId;
    }

    const std::vector<size_t> &getPointIds() const { return PointIds; }

    void addPoint(size_t PointId,
                  const InstructionBenchmarkClustering &Clustering);

    // Return the cluster centroid.
    const SchedClassClusterCentroid &getCentroid() const { return Centroid; }

    // Returns true if the cluster representative measurements match that of SC.
    bool
    measurementsMatch(const llvm::MCSubtargetInfo &STI,
                      const ResolvedSchedClass &SC,
                      const InstructionBenchmarkClustering &Clustering,
                      const double AnalysisInconsistencyEpsilonSquared_) const;

  private:
    InstructionBenchmarkClustering::ClusterId ClusterId;
    std::vector<size_t> PointIds;
    // Measurement stats for the points in the SchedClassCluster.
    SchedClassClusterCentroid Centroid;
  };

  void printInstructionRowCsv(size_t PointId, llvm::raw_ostream &OS) const;

  void
  printSchedClassClustersHtml(const std::vector<SchedClassCluster> &Clusters,
                              const ResolvedSchedClass &SC,
                              llvm::raw_ostream &OS) const;
  void printSchedClassDescHtml(const ResolvedSchedClass &SC,
                               llvm::raw_ostream &OS) const;

  // A pair of (Sched Class, indices of points that belong to the sched
  // class).
  struct ResolvedSchedClassAndPoints {
    explicit ResolvedSchedClassAndPoints(ResolvedSchedClass &&RSC);

    ResolvedSchedClass RSC;
    std::vector<size_t> PointIds;
  };

  // Builds a list of ResolvedSchedClassAndPoints.
  std::vector<ResolvedSchedClassAndPoints> makePointsPerSchedClass() const;

  template <typename EscapeTag, EscapeTag Tag>
  void writeSnippet(llvm::raw_ostream &OS, llvm::ArrayRef<uint8_t> Bytes,
                    const char *Separator) const;

  const InstructionBenchmarkClustering &Clustering_;
  llvm::MCObjectFileInfo ObjectFileInfo_;
  std::unique_ptr<llvm::MCContext> Context_;
  std::unique_ptr<llvm::MCSubtargetInfo> SubtargetInfo_;
  std::unique_ptr<llvm::MCInstrInfo> InstrInfo_;
  std::unique_ptr<llvm::MCRegisterInfo> RegInfo_;
  std::unique_ptr<llvm::MCAsmInfo> AsmInfo_;
  std::unique_ptr<llvm::MCInstPrinter> InstPrinter_;
  std::unique_ptr<llvm::MCDisassembler> Disasm_;
  const double AnalysisInconsistencyEpsilonSquared_;
  const bool AnalysisDisplayUnstableOpcodes_;
};

} // namespace exegesis
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_EXEGESIS_CLUSTERING_H
