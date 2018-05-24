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
  void printInstructionRowCsv(size_t PointId, llvm::raw_ostream &OS) const;

  void printSchedClassClustersHtml(std::vector<size_t> PointIds,
                                   llvm::raw_ostream &OS) const;
  void printSchedClassDescHtml(const llvm::MCSchedClassDesc &SCDesc,
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

} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_CLUSTERING_H
