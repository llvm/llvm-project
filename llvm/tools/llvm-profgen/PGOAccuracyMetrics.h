//===-- PGOAccuracyMetrics.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_LLVM_PROFGEN_PGOACCURACYMETRICS_H
#define LLVM_TOOLS_LLVM_PROFGEN_PGOACCURACYMETRICS_H

#include "PerfReader.h"
#include "llvm/Object/ELFTypes.h"

namespace llvm {
namespace sampleprof {
class PGOAccuracyMetrics {
public:
  PGOAccuracyMetrics(const ContextSampleCounterMap &CSCM,
                     const std::vector<BBAddrMap> &BBAM,
                     const std::vector<PGOAnalysisMap> &PGOAM)
      : SampleCounters(CSCM), BBAddrMaps(BBAM), PGOAnalysisMaps(PGOAM) {
    init();
  }
  void init();
  void emitPGOAccuracyMetrics();

private:
  const ContextSampleCounterMap &SampleCounters;
  const std::vector<BBAddrMap> &BBAddrMaps;
  const std::vector<PGOAnalysisMap> &PGOAnalysisMaps;

  // key is bb's address, value is function's address and bb's unique id;
  std::map<uint64_t, std::pair<uint64_t, uint32_t>> BBAddrToID;
  std::unordered_map<uint64_t,
                     std::map<std::pair<uint32_t, uint32_t>, uint64_t>>
      EdgeCounter;

  uint64_t UnknownBranches = 0;
  uint64_t MatchedBranches = 0;
  uint64_t DismatchedBranches = 0;
};
} // namespace sampleprof
} // namespace llvm

#endif