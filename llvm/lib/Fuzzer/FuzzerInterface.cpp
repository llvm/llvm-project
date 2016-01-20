//===- FuzzerInterface.cpp - Mutate a test input --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Parts of public interface for libFuzzer.
//===----------------------------------------------------------------------===//


#include "FuzzerInterface.h"
#include "FuzzerInternal.h"
#include <random>

namespace fuzzer {

void FuzzerRandomLibc::ResetSeed(unsigned int seed) { srand(seed); }

size_t FuzzerRandomLibc::Rand() { return rand(); }

struct FuzzerRandom_mt19937::Impl : public std::mt19937 {
  Impl(unsigned seed) : std::mt19937(seed) {}
};

void FuzzerRandom_mt19937::ResetSeed(unsigned int seed) {
  delete R;
  R = new Impl(seed);
}

FuzzerRandom_mt19937::~FuzzerRandom_mt19937() { delete R; }

size_t FuzzerRandom_mt19937::Rand() { return (*R)(); }

UserSuppliedFuzzer::UserSuppliedFuzzer(FuzzerRandomBase *Rand)
    : Rand(Rand), MD(new MutationDispatcher(*Rand)) {}

UserSuppliedFuzzer::~UserSuppliedFuzzer() {
  if (OwnRand)
    delete Rand;
  delete MD;
}

size_t UserSuppliedFuzzer::Mutate(uint8_t *Data, size_t Size, size_t MaxSize) {
  return GetMD().Mutate(Data, Size, MaxSize);
}

size_t UserSuppliedFuzzer::CrossOver(const uint8_t *Data1, size_t Size1,
                                     const uint8_t *Data2, size_t Size2,
                                     uint8_t *Out, size_t MaxOutSize) {
  return GetMD().CrossOver(Data1, Size1, Data2, Size2, Out, MaxOutSize);
}

size_t Mutate(uint8_t *Data, size_t Size, size_t MaxSize,
              FuzzerRandomBase &Rand) {
  MutationDispatcher MD(Rand);
  return MD.Mutate(Data, Size, MaxSize);
}



}  // namespace fuzzer.
