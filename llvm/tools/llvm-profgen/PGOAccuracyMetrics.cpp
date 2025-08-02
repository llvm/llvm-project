#include "PGOAccuracyMetrics.h"
static cl::opt<double> BranchProbabilityThreshold(
    "pgo-accuracy-metrics-branch-probability-threshold",
    cl::desc("PGO accuracy metrics branch probability threshold"),
    cl::init(20.0));

namespace llvm {
namespace sampleprof {

void PGOAccuracyMetrics::init() {
  for (auto &Function : BBAddrMaps) {
    for (auto &Range : Function.getBBRanges()) {
      uint64_t BaseAddress = Range.BaseAddress;
      for (auto &BB : Range.BBEntries) {
        uint64_t Address = BaseAddress + BB.Offset;
        BBAddrToID[Address] =
            std::make_pair(Function.getFunctionAddress(), BB.ID);
      }
    }
  }

  for (auto It : SampleCounters) {
    const RangeSample &RangeCounter = It.second.RangeCounter;
    const BranchSample &BranchCounter = It.second.BranchCounter;

    for (auto Item : RangeCounter) {
      auto From = BBAddrToID.lower_bound(Item.first.first);
      auto End = BBAddrToID.upper_bound(Item.first.second);
      if (From == End)
        continue;
      for (auto To = std::next(From); To != End; ++To) {
        assert(From->second.first == To->second.first);
        EdgeCounter[From->second.first]
                   [std::make_pair(From->second.second, To->second.second)] +=
            Item.second;
        From = To;
      }
    }

    for (auto Item : BranchCounter) {
      auto Ptr = BBAddrToID.upper_bound(Item.first.first);
      if (Ptr == BBAddrToID.begin())
        continue;
      auto From = std::prev(Ptr)->second;
      auto To = BBAddrToID[Item.first.second];
      assert(From.first == To.first);
      EdgeCounter[From.first][{From.second, To.second}] += Item.second;
    }
  }
}

void PGOAccuracyMetrics::emitPGOAccuracyMetrics() {
  assert(BBAddrMaps.size() == PGOAnalysisMaps.size());

  for (uint64_t I = 0; I < BBAddrMaps.size(); ++I) {
    auto &BBAddrMap = BBAddrMaps[I];
    auto &PGOAnalysisMap = PGOAnalysisMaps[I];
    assert(BBAddrMap.getNumBBEntries() == PGOAnalysisMap.BBEntries.size());
    size_t Pos = 0;
    for (auto &Range : BBAddrMap.getBBRanges()) {
      for (auto &BB : Range.BBEntries) {
        uint64_t Sum = 0;
        for (auto Successor : PGOAnalysisMap.BBEntries[Pos].Successors) {
          Sum += EdgeCounter[BBAddrMap.getFunctionAddress()]
                            [std::make_pair(BB.ID, Successor.ID)];
        }

        for (auto Successor : PGOAnalysisMap.BBEntries[Pos].Successors) {
          if (Successor.Prob.isUnknown()) {
            ++UnknownBranches;
            continue;
          }
          double SampleBranchProb =
              rint((Sum == 0
                        ? (double)1 /
                              PGOAnalysisMap.BBEntries[Pos].Successors.size()
                        : (double)EdgeCounter[BBAddrMap.getFunctionAddress()]
                                             [std::make_pair(BB.ID,
                                                             Successor.ID)] /
                              Sum) *
                   100.0 * 100.0) /
              100.0;
          double RecordBranchProb =
              rint(((double)Successor.Prob.getNumerator() /
                    Successor.Prob.getDenominator()) *
                   100.0 * 100.0) /
              100.0;
          if (abs(SampleBranchProb - RecordBranchProb) <
              BranchProbabilityThreshold) {
            ++MatchedBranches;
          } else {
            ++DismatchedBranches;
          }
        }
        ++Pos;
      }
    }
  }

  llvm::outs() << "UnknownBranches: " << UnknownBranches << "\n";
  llvm::outs() << "MatchedBranches: " << MatchedBranches << "\n";
  llvm::outs() << "DismatchedBranches: " << DismatchedBranches << "\n";
}

} // namespace sampleprof
} // namespace llvm
