//=-- Profilesummary.cpp - Profile summary computation ----------------------=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for computing profile summary data.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/ProfileCommon.h"
#include "llvm/ProfileData/InstrProf.h"

using namespace llvm;

void ProfileSummary::addRecord(const InstrProfRecord &R) {
  NumFunctions++;
  if (R.Counts[0] > MaxFunctionCount)
    MaxFunctionCount = R.Counts[0];

  for (size_t I = 0, E = R.Counts.size(); I < E; ++I)
    addCount(R.Counts[I], (I == 0));
}

// The argument to this method is a vector of cutoff percentages and the return
// value is a vector of (Cutoff, MinBlockCount, NumBlocks) triplets.
void ProfileSummary::computeDetailedSummary() {
  if (DetailedSummaryCutoffs.empty())
    return;
  auto Iter = CountFrequencies.begin();
  auto End = CountFrequencies.end();
  std::sort(DetailedSummaryCutoffs.begin(), DetailedSummaryCutoffs.end());

  uint32_t BlocksSeen = 0;
  uint64_t CurrSum = 0, Count = 0;

  for (uint32_t Cutoff : DetailedSummaryCutoffs) {
    assert(Cutoff <= 999999);
    APInt Temp(128, TotalCount);
    APInt N(128, Cutoff);
    APInt D(128, ProfileSummary::Scale);
    Temp *= N;
    Temp = Temp.sdiv(D);
    uint64_t DesiredCount = Temp.getZExtValue();
    assert(DesiredCount <= TotalCount);
    while (CurrSum < DesiredCount && Iter != End) {
      Count = Iter->first;
      uint32_t Freq = Iter->second;
      CurrSum += (Count * Freq);
      BlocksSeen += Freq;
      Iter++;
    }
    assert(CurrSum >= DesiredCount);
    ProfileSummaryEntry PSE = {Cutoff, Count, BlocksSeen};
    DetailedSummary.push_back(PSE);
  }
}

ProfileSummary::ProfileSummary(const IndexedInstrProf::Summary &S)
    : TotalCount(S.get(IndexedInstrProf::Summary::TotalBlockCount)),
      MaxBlockCount(S.get(IndexedInstrProf::Summary::MaxBlockCount)),
      MaxInternalBlockCount(
          S.get(IndexedInstrProf::Summary::MaxInternalBlockCount)),
      MaxFunctionCount(S.get(IndexedInstrProf::Summary::MaxFunctionCount)),
      NumBlocks(S.get(IndexedInstrProf::Summary::TotalNumBlocks)),
      NumFunctions(S.get(IndexedInstrProf::Summary::TotalNumFunctions)) {
  for (unsigned I = 0; I < S.NumCutoffEntries; I++) {
    const IndexedInstrProf::Summary::Entry &Ent = S.getEntry(I);
    DetailedSummary.emplace_back((uint32_t)Ent.Cutoff, Ent.MinBlockCount,
                                 Ent.NumBlocks);
  }
}
