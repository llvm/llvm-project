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
void Analysis::printInstructionRow(const size_t ClusterId, const size_t PointId,
                                   llvm::raw_ostream &OS) const {
  const InstructionBenchmark &Point = Clustering_.getPoints()[PointId];

  OS << ClusterId << kCsvSep;
  writeCsvEscaped(OS, Point.Key.OpcodeName);
  OS << kCsvSep;
  writeCsvEscaped(OS, Point.Key.Config);
  OS << kCsvSep;
  const auto OpcodeIt = MnemonicToOpcode_.find(Point.Key.OpcodeName);
  if (OpcodeIt != MnemonicToOpcode_.end()) {
    const unsigned SchedClassId = InstrInfo_->get(OpcodeIt->second).getSchedClass();
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
    const auto &SchedModel = SubtargetInfo_->getSchedModel();
    const llvm::MCSchedClassDesc *const SCDesc =
       SchedModel.getSchedClassDesc(SchedClassId);
    writeCsvEscaped(OS, SCDesc->Name);
#else
    OS << SchedClassId;
#endif
  }
  // FIXME: Print the sched class once InstructionBenchmark separates key into
  // (mnemonic, mode, opaque).
  for (const auto &Measurement : Point.Measurements) {
    OS << kCsvSep;
    writeCsvEscaped(OS, llvm::formatv("{0:F}", Measurement.Value));
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
  SubtargetInfo_.reset(Target.createMCSubtargetInfo(
      FirstPoint.LLVMTriple, FirstPoint.CpuName, ""));

  // Build an index of mnemonic->opcode.
  for (int I = 0, E = InstrInfo_->getNumOpcodes(); I < E; ++I)
    MnemonicToOpcode_.emplace(InstrInfo_->getName(I), I);
}

llvm::Error Analysis::printClusters(llvm::raw_ostream &OS) const {
  if (Clustering_.getPoints().empty())
    return llvm::Error::success();

  // Write the header.
  OS << "cluster_id" << kCsvSep << "opcode_name" << kCsvSep << "config"
     << kCsvSep << "sched_class";
  for (const auto &Measurement : Clustering_.getPoints().front().Measurements) {
    OS << kCsvSep;
    writeCsvEscaped(OS, Measurement.Key);
  }
  OS << "\n";

  // Write the points.
  const auto& Clusters = Clustering_.getValidClusters();
  for (size_t I = 0, E = Clusters.size(); I < E; ++I) {
    for (const size_t PointId : Clusters[I].PointIndices) {
      printInstructionRow(I, PointId, OS);
    }
    OS << "\n\n";
  }
  return llvm::Error::success();
}

} // namespace exegesis
