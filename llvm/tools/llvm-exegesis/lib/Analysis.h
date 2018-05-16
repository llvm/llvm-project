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
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <unordered_map>

namespace exegesis {

// A helper class to analyze benchmark results for a target.
class Analysis {
public:
  Analysis(const llvm::Target& Target, const InstructionBenchmarkClustering &Clustering);

  // Prints a csv of instructions for each cluster.
  llvm::Error printClusters(llvm::raw_ostream &OS) const;

 private:
   void printInstructionRow(size_t ClusterId,  size_t PointId,
                                   llvm::raw_ostream &OS) const;

   const InstructionBenchmarkClustering & Clustering_;
   std::unique_ptr<llvm::MCSubtargetInfo> SubtargetInfo_;
   std::unique_ptr<llvm::MCInstrInfo> InstrInfo_;
   std::unordered_map<std::string, unsigned> MnemonicToOpcode_;
};

} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_CLUSTERING_H
