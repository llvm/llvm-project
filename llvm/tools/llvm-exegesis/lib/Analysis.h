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
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

namespace exegesis {

// Prints a csv of instructions for each cluster.
llvm::Error
printAnalysisClusters(const InstructionBenchmarkClustering &Clustering,
                      const llvm::MCSubtargetInfo &STI, llvm::raw_ostream &OS);

} // namespace exegesis

#endif // LLVM_TOOLS_LLVM_EXEGESIS_CLUSTERING_H
