//===- FuzzerCorpus.h - Internal header for the Fuzzer ----------*- C++ -* ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// fuzzer::InputCorpus
//===----------------------------------------------------------------------===//

#ifndef LLVM_FUZZER_CORPUS
#define LLVM_FUZZER_CORPUS

#include <random>
#include <unordered_set>

#include "FuzzerDefs.h"
#include "FuzzerRandom.h"
#include "FuzzerTracePC.h"

namespace fuzzer {

struct InputInfo {
  Unit U;  // The actual input data.
  uint8_t Sha1[kSHA1NumBytes];  // Checksum.
  // Number of features that this input has and no smaller input has.
  size_t NumFeatures = 0;
  size_t Tmp = 0; // Used by ValidateFeatureSet.
  // Stats.
  size_t NumExecutedMutations = 0;
  size_t NumSuccessfullMutations = 0;
};

class InputCorpus {
 public:
  InputCorpus() {
    Inputs.reserve(1 << 14);  // Avoid too many resizes.
    memset(FeatureSet, 0, sizeof(FeatureSet));
  }
  size_t size() const { return Inputs.size(); }
  bool empty() const { return Inputs.empty(); }
  const Unit &operator[] (size_t Idx) const { return Inputs[Idx].U; }
  void AddToCorpus(const Unit &U) {
    uint8_t Hash[kSHA1NumBytes];
    ComputeSHA1(U.data(), U.size(), Hash);
    if (!Hashes.insert(Sha1ToString(Hash)).second) return;
    Inputs.push_back(InputInfo());
    InputInfo &II = Inputs.back();
    II.U = U;
    memcpy(II.Sha1, Hash, kSHA1NumBytes);
    UpdateFeatureSet(Inputs.size() - 1);
    UpdateCorpusDistribution();
  }

  typedef const std::vector<InputInfo>::const_iterator ConstIter;
  ConstIter begin() const { return Inputs.begin(); }
  ConstIter end() const { return Inputs.end(); }

  bool HasUnit(const Unit &U) { return Hashes.count(Hash(U)); }
  InputInfo &ChooseUnitToMutate(Random &Rand) {
    return Inputs[ChooseUnitIdxToMutate(Rand)];
  };

  // Returns an index of random unit from the corpus to mutate.
  // Hypothesis: units added to the corpus last are more likely to be
  // interesting. This function gives more weight to the more recent units.
  size_t ChooseUnitIdxToMutate(Random &Rand) {
    size_t Idx = static_cast<size_t>(CorpusDistribution(Rand.Get_mt19937()));
    assert(Idx < Inputs.size());
    return Idx;
  }

  void PrintStats() {
    for (size_t i = 0; i < Inputs.size(); i++) {
      const auto &II = Inputs[i];
      Printf("  [%zd %s]\tsz: %zd\truns: %zd\tsucc: %zd\n", i,
             Sha1ToString(II.Sha1).c_str(), II.U.size(),
             II.NumExecutedMutations, II.NumSuccessfullMutations);
    }
  }

  void PrintFeatureSet() {
    Printf("Features [id: cnt idx sz] ");
    for (size_t i = 0; i < kFeatureSetSize; i++) {
      auto &Fe = FeatureSet[i];
      if (!Fe.Count) continue;
      Printf("[%zd: %zd %zd] ", i, Fe.SmallestElementIdx,
             Fe.SmallestElementSize);
    }
    Printf("\n\t");
    for (size_t i = 0; i < Inputs.size(); i++)
      if (size_t N = Inputs[i].NumFeatures)
        Printf(" %zd=>%zd ", i, N);
    Printf("\n");
  }

private:

  static const bool FeatureDebug = false;
  static const size_t kFeatureSetSize = TracePC::kFeatureSetSize;

  void ValidateFeatureSet() {
    for (size_t Idx = 0; Idx < kFeatureSetSize; Idx++) {
      Feature &Fe = FeatureSet[Idx];
      if(Fe.Count && Fe.SmallestElementSize)
        Inputs[Fe.SmallestElementIdx].Tmp++;
    }
    for (auto &II: Inputs) {
      assert(II.Tmp == II.NumFeatures);
      II.Tmp = 0;
    }
  }

  void UpdateFeatureSet(size_t CurrentElementIdx) {
    auto &II = Inputs[CurrentElementIdx];
    size_t Size = II.U.size();
    if (!Size)
      return;
    bool Updated = false;
    for (size_t Idx = 0; Idx < kFeatureSetSize; Idx++) {
      if (!TPC.HasFeature(Idx))
        continue;
      Feature &Fe = FeatureSet[Idx];
      Fe.Count++;
      if (!Fe.SmallestElementSize ||
          Fe.SmallestElementSize > Size) {
        II.NumFeatures++;
        if (Fe.SmallestElementSize > Size) {
          auto &OlderII = Inputs[Fe.SmallestElementIdx];
          assert(OlderII.NumFeatures > 0);
          OlderII.NumFeatures--;
          if (!OlderII.NumFeatures && FeatureDebug)
            Printf("EVICTED %zd\n", Fe.SmallestElementIdx);
        }
        Fe.SmallestElementIdx = CurrentElementIdx;
        Fe.SmallestElementSize = Size;
        Updated = true;
      }
    }
    if (Updated && FeatureDebug) PrintFeatureSet();
    ValidateFeatureSet();
  }

  // Updates the probability distribution for the units in the corpus.
  // Must be called whenever the corpus or unit weights are changed.
  void UpdateCorpusDistribution() {
    size_t N = Inputs.size();
    std::vector<double> Intervals(N + 1);
    std::vector<double> Weights(N);
    std::iota(Intervals.begin(), Intervals.end(), 0);
    std::iota(Weights.begin(), Weights.end(), 1);
    CorpusDistribution = std::piecewise_constant_distribution<double>(
        Intervals.begin(), Intervals.end(), Weights.begin());
  }
  std::piecewise_constant_distribution<double> CorpusDistribution;

  std::unordered_set<std::string> Hashes;
  std::vector<InputInfo> Inputs;

  struct Feature {
    size_t Count;
    size_t SmallestElementIdx;
    size_t SmallestElementSize;
  };
  Feature FeatureSet[kFeatureSetSize];
};

}  // namespace fuzzer

#endif  // LLVM_FUZZER_CORPUS
