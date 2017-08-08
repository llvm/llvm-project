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

#include "FuzzerDefs.h"
#include "FuzzerIO.h"
#include "FuzzerRandom.h"
#include "FuzzerSHA1.h"
#include "FuzzerTracePC.h"
#include <algorithm>
#include <numeric>
#include <random>
#include <unordered_set>

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
  bool MayDeleteFile = false;
  bool Reduced = false;
  std::vector<uint32_t> UniqFeatureSet;
};

class InputCorpus {
  static const size_t kFeatureSetSize = 1 << 21;
 public:
  InputCorpus(const std::string &OutputCorpus) : OutputCorpus(OutputCorpus) {
    memset(InputSizesPerFeature, 0, sizeof(InputSizesPerFeature));
    memset(SmallestElementPerFeature, 0, sizeof(SmallestElementPerFeature));
  }
  ~InputCorpus() {
    for (auto II : Inputs)
      delete II;
  }
  size_t size() const { return Inputs.size(); }
  size_t SizeInBytes() const {
    size_t Res = 0;
    for (auto II : Inputs)
      Res += II->U.size();
    return Res;
  }
  size_t NumActiveUnits() const {
    size_t Res = 0;
    for (auto II : Inputs)
      Res += !II->U.empty();
    return Res;
  }
  size_t MaxInputSize() const {
    size_t Res = 0;
    for (auto II : Inputs)
        Res = std::max(Res, II->U.size());
    return Res;
  }
  bool empty() const { return Inputs.empty(); }
  const Unit &operator[] (size_t Idx) const { return Inputs[Idx]->U; }
  void AddToCorpus(const Unit &U, size_t NumFeatures, bool MayDeleteFile,
                   const std::vector<uint32_t> &FeatureSet) {
    assert(!U.empty());
    if (FeatureDebug)
      Printf("ADD_TO_CORPUS %zd NF %zd\n", Inputs.size(), NumFeatures);
    Inputs.push_back(new InputInfo());
    InputInfo &II = *Inputs.back();
    II.U = U;
    II.NumFeatures = NumFeatures;
    II.MayDeleteFile = MayDeleteFile;
    II.UniqFeatureSet = FeatureSet;
    std::sort(II.UniqFeatureSet.begin(), II.UniqFeatureSet.end());
    ComputeSHA1(U.data(), U.size(), II.Sha1);
    Hashes.insert(Sha1ToString(II.Sha1));
    UpdateCorpusDistribution();
    PrintCorpus();
    // ValidateFeatureSet();
  }

  // Debug-only
  void PrintUnit(const Unit &U) {
    if (!FeatureDebug) return;
    for (uint8_t C : U) {
      if (C != 'F' && C != 'U' && C != 'Z')
        C = '.';
      Printf("%c", C);
    }
  }

  // Debug-only
  void PrintFeatureSet(const std::vector<uint32_t> &FeatureSet) {
    if (!FeatureDebug) return;
    Printf("{");
    for (uint32_t Feature: FeatureSet)
      Printf("%u,", Feature);
    Printf("}");
  }

  // Debug-only
  void PrintCorpus() {
    if (!FeatureDebug) return;
    Printf("======= CORPUS:\n");
    int i = 0;
    for (auto II : Inputs) {
      if (std::find(II->U.begin(), II->U.end(), 'F') != II->U.end()) {
        Printf("[%2d] ", i);
        Printf("%s sz=%zd ", Sha1ToString(II->Sha1).c_str(), II->U.size());
        PrintUnit(II->U);
        Printf(" ");
        PrintFeatureSet(II->UniqFeatureSet);
        Printf("\n");
      }
      i++;
    }
  }

  void Replace(InputInfo *II, const Unit &U) {
    assert(II->U.size() > U.size());
    Hashes.erase(Sha1ToString(II->Sha1));
    DeleteFile(*II);
    ComputeSHA1(U.data(), U.size(), II->Sha1);
    Hashes.insert(Sha1ToString(II->Sha1));
    II->U = U;
    II->Reduced = true;
  }

  bool HasUnit(const Unit &U) { return Hashes.count(Hash(U)); }
  bool HasUnit(const std::string &H) { return Hashes.count(H); }
  InputInfo &ChooseUnitToMutate(Random &Rand) {
    InputInfo &II = *Inputs[ChooseUnitIdxToMutate(Rand)];
    assert(!II.U.empty());
    return II;
  };

  // Returns an index of random unit from the corpus to mutate.
  // Hypothesis: units added to the corpus last are more likely to be
  // interesting. This function gives more weight to the more recent units.
  size_t ChooseUnitIdxToMutate(Random &Rand) {
    size_t Idx = static_cast<size_t>(CorpusDistribution(Rand));
    assert(Idx < Inputs.size());
    return Idx;
  }

  void PrintStats() {
    for (size_t i = 0; i < Inputs.size(); i++) {
      const auto &II = *Inputs[i];
      Printf("  [%zd %s]\tsz: %zd\truns: %zd\tsucc: %zd\n", i,
             Sha1ToString(II.Sha1).c_str(), II.U.size(),
             II.NumExecutedMutations, II.NumSuccessfullMutations);
    }
  }

  void PrintFeatureSet() {
    for (size_t i = 0; i < kFeatureSetSize; i++) {
      if(size_t Sz = GetFeature(i))
        Printf("[%zd: id %zd sz%zd] ", i, SmallestElementPerFeature[i], Sz);
    }
    Printf("\n\t");
    for (size_t i = 0; i < Inputs.size(); i++)
      if (size_t N = Inputs[i]->NumFeatures)
        Printf(" %zd=>%zd ", i, N);
    Printf("\n");
  }

  void DeleteFile(const InputInfo &II) {
    if (!OutputCorpus.empty() && II.MayDeleteFile)
      RemoveFile(DirPlusFile(OutputCorpus, Sha1ToString(II.Sha1)));
  }

  void DeleteInput(size_t Idx) {
    InputInfo &II = *Inputs[Idx];
    DeleteFile(II);
    Unit().swap(II.U);
    if (FeatureDebug)
      Printf("EVICTED %zd\n", Idx);
  }

  bool AddFeature(size_t Idx, uint32_t NewSize, bool Shrink) {
    assert(NewSize);
    Idx = Idx % kFeatureSetSize;
    uint32_t OldSize = GetFeature(Idx);
    if (OldSize == 0 || (Shrink && OldSize > NewSize)) {
      if (OldSize > 0) {
        size_t OldIdx = SmallestElementPerFeature[Idx];
        InputInfo &II = *Inputs[OldIdx];
        assert(II.NumFeatures > 0);
        II.NumFeatures--;
        if (II.NumFeatures == 0)
          DeleteInput(OldIdx);
      } else {
        NumAddedFeatures++;
      }
      NumUpdatedFeatures++;
      if (FeatureDebug)
        Printf("ADD FEATURE %zd sz %d\n", Idx, NewSize);
      SmallestElementPerFeature[Idx] = Inputs.size();
      InputSizesPerFeature[Idx] = NewSize;
      return true;
    }
    return false;
  }

  size_t NumFeatures() const { return NumAddedFeatures; }
  size_t NumFeatureUpdates() const { return NumUpdatedFeatures; }

  void ResetFeatureSet() {
    assert(Inputs.empty());
    memset(InputSizesPerFeature, 0, sizeof(InputSizesPerFeature));
    memset(SmallestElementPerFeature, 0, sizeof(SmallestElementPerFeature));
  }

private:

  static const bool FeatureDebug = false;

  size_t GetFeature(size_t Idx) const { return InputSizesPerFeature[Idx]; }

  void ValidateFeatureSet() {
    if (FeatureDebug)
      PrintFeatureSet();
    for (size_t Idx = 0; Idx < kFeatureSetSize; Idx++)
      if (GetFeature(Idx))
        Inputs[SmallestElementPerFeature[Idx]]->Tmp++;
    for (auto II: Inputs) {
      if (II->Tmp != II->NumFeatures)
        Printf("ZZZ %zd %zd\n", II->Tmp, II->NumFeatures);
      assert(II->Tmp == II->NumFeatures);
      II->Tmp = 0;
    }
  }

  // Updates the probability distribution for the units in the corpus.
  // Must be called whenever the corpus or unit weights are changed.
  void UpdateCorpusDistribution() {
    size_t N = Inputs.size();
    assert(N);
    Intervals.resize(N + 1);
    Weights.resize(N);
    std::iota(Intervals.begin(), Intervals.end(), 0);
    for (size_t i = 0; i < N; i++)
      Weights[i] = Inputs[i]->NumFeatures * (i + 1);
    CorpusDistribution = std::piecewise_constant_distribution<double>(
        Intervals.begin(), Intervals.end(), Weights.begin());
  }
  std::piecewise_constant_distribution<double> CorpusDistribution;

  std::vector<double> Intervals;
  std::vector<double> Weights;

  std::unordered_set<std::string> Hashes;
  std::vector<InputInfo*> Inputs;

  size_t NumAddedFeatures = 0;
  size_t NumUpdatedFeatures = 0;
  uint32_t InputSizesPerFeature[kFeatureSetSize];
  uint32_t SmallestElementPerFeature[kFeatureSetSize];

  std::string OutputCorpus;
};

}  // namespace fuzzer

#endif  // LLVM_FUZZER_CORPUS
