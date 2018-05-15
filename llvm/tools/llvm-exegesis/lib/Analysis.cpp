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
#include <vector>

namespace exegesis {

static const char kCsvSep = ',';

static void writeCsvEscaped(llvm::raw_ostream &OS, const std::string &S) {
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

// Prints a row representing an instruction, along with scheduling info and
// point coordinates (measurements).
static void printInstructionRow(const InstructionBenchmark &Point,
                                const llvm::MCSubtargetInfo &STI,
                                const size_t ClusterId, llvm::raw_ostream &OS) {
  OS << ClusterId << kCsvSep;
  writeCsvEscaped(OS, Point.Key.OpcodeName);
  OS << kCsvSep;
  writeCsvEscaped(OS, Point.Key.Config);
  // FIXME: Print the sched class once InstructionBenchmark separates key into
  // (mnemonic, mode, opaque).
  for (const auto &Measurement : Point.Measurements) {
    OS << kCsvSep;
    writeCsvEscaped(OS, llvm::formatv("{0:F}", Measurement.Value));
  }
  OS << "\n";
}

static void printCluster(const std::vector<InstructionBenchmark> &Points,
                         const llvm::MCSubtargetInfo &STI,
                         const size_t ClusterId,
                         const InstructionBenchmarkClustering::Cluster &Cluster,
                         llvm::raw_ostream &OS) {
  // Print all points.
  for (const auto &PointId : Cluster.PointIndices) {
    printInstructionRow(Points[PointId], STI, ClusterId, OS);
  }
}

llvm::Error
printAnalysisClusters(const InstructionBenchmarkClustering &Clustering,
                      const llvm::MCSubtargetInfo &STI, llvm::raw_ostream &OS) {
  if (Clustering.getPoints().empty())
    return llvm::Error::success();

  // Write the header.
  OS << "cluster_id" << kCsvSep << "opcode_name" << kCsvSep << "config"
     << kCsvSep << "sched_class";
  for (const auto &Measurement : Clustering.getPoints().front().Measurements) {
    OS << kCsvSep;
    writeCsvEscaped(OS, Measurement.Key);
  }
  OS << "\n";

  // Write the points.
  for (size_t I = 0, E = Clustering.getValidClusters().size(); I < E; ++I) {
    printCluster(Clustering.getPoints(), STI, I,
                 Clustering.getValidClusters()[I], OS);
    OS << "\n\n";
  }
  return llvm::Error::success();
}

} // namespace exegesis
