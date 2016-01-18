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

namespace fuzzer {

void FuzzerRandomLibc::ResetSeed(int seed) { srand(seed); }

size_t FuzzerRandomLibc::Rand() { return rand(); }

UserSuppliedFuzzer::UserSuppliedFuzzer(FuzzerRandomBase *Rand)
    : Rand(Rand), MD(*Rand) {}

UserSuppliedFuzzer::~UserSuppliedFuzzer() {
  if (OwnRand)
    delete Rand;
}

}  // namespace fuzzer.
