//===- FuzzerMutate.cpp - Mutate a test input -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Mutate a test input.
//===----------------------------------------------------------------------===//

#include <cstring>

#include "FuzzerInternal.h"

#include <algorithm>

namespace fuzzer {

struct Mutator {
  size_t (MutationDispatcher::*Fn)(uint8_t *Data, size_t Size, size_t Max);
  const char *Name;
};

struct MutationDispatcher::Impl {
  std::vector<Unit> Dictionary;
  std::vector<Mutator> Mutators;
  std::vector<Mutator> CurrentMutatorSequence;
  const std::vector<Unit> *Corpus = nullptr;

  void Add(Mutator M) { Mutators.push_back(M); }
  Impl() {
    Add({&MutationDispatcher::Mutate_EraseByte, "EraseByte"});
    Add({&MutationDispatcher::Mutate_InsertByte, "InsertByte"});
    Add({&MutationDispatcher::Mutate_ChangeByte, "ChangeByte"});
    Add({&MutationDispatcher::Mutate_ChangeBit, "ChangeBit"});
    Add({&MutationDispatcher::Mutate_ShuffleBytes, "ShuffleBytes"});
    Add({&MutationDispatcher::Mutate_ChangeASCIIInteger, "ChangeASCIIInt"});
    Add({&MutationDispatcher::Mutate_CrossOver, "CrossOver"});
  }
  void AddWordToDictionary(const uint8_t *Word, size_t Size) {
    if (Dictionary.empty()) {
      Add({&MutationDispatcher::Mutate_AddWordFromDictionary, "AddFromDict"});
    }
    Dictionary.push_back(Unit(Word, Word + Size));
  }
  void SetCorpus(const std::vector<Unit> *Corpus) { this->Corpus = Corpus; }
};

static char FlipRandomBit(char X, FuzzerRandomBase &Rand) {
  int Bit = Rand(8);
  char Mask = 1 << Bit;
  char R;
  if (X & (1 << Bit))
    R = X & ~Mask;
  else
    R = X | Mask;
  assert(R != X);
  return R;
}

static char RandCh(FuzzerRandomBase &Rand) {
  if (Rand.RandBool()) return Rand(256);
  const char *Special = "!*'();:@&=+$,/?%#[]123ABCxyz-`~.";
  return Special[Rand(sizeof(Special) - 1)];
}

size_t MutationDispatcher::Mutate_ShuffleBytes(uint8_t *Data, size_t Size,
                                               size_t MaxSize) {
  assert(Size);
  size_t ShuffleAmount = Rand(std::min(Size, (size_t)8)) + 1;  // [1,8] and <= Size.
  size_t ShuffleStart = Rand(Size - ShuffleAmount);
  assert(ShuffleStart + ShuffleAmount <= Size);
  std::random_shuffle(Data + ShuffleStart, Data + ShuffleStart + ShuffleAmount,
                      Rand);
  return Size;
}

size_t MutationDispatcher::Mutate_EraseByte(uint8_t *Data, size_t Size,
                                            size_t MaxSize) {
  assert(Size);
  if (Size == 1) return 0;
  size_t Idx = Rand(Size);
  // Erase Data[Idx].
  memmove(Data + Idx, Data + Idx + 1, Size - Idx - 1);
  return Size - 1;
}

size_t MutationDispatcher::Mutate_InsertByte(uint8_t *Data, size_t Size,
                                             size_t MaxSize) {
  if (Size == MaxSize) return 0;
  size_t Idx = Rand(Size + 1);
  // Insert new value at Data[Idx].
  memmove(Data + Idx + 1, Data + Idx, Size - Idx);
  Data[Idx] = RandCh(Rand);
  return Size + 1;
}

size_t MutationDispatcher::Mutate_ChangeByte(uint8_t *Data, size_t Size,
                                             size_t MaxSize) {
  size_t Idx = Rand(Size);
  Data[Idx] = RandCh(Rand);
  return Size;
}

size_t MutationDispatcher::Mutate_ChangeBit(uint8_t *Data, size_t Size,
                                            size_t MaxSize) {
  size_t Idx = Rand(Size);
  Data[Idx] = FlipRandomBit(Data[Idx], Rand);
  return Size;
}

size_t MutationDispatcher::Mutate_AddWordFromDictionary(uint8_t *Data,
                                                        size_t Size,
                                                        size_t MaxSize) {
  auto &D = MDImpl->Dictionary;
  assert(!D.empty());
  if (D.empty()) return 0;
  const Unit &Word = D[Rand(D.size())];
  if (Rand.RandBool()) {  // Insert Word.
    if (Size + Word.size() > MaxSize) return 0;
    size_t Idx = Rand(Size + 1);
    memmove(Data + Idx + Word.size(), Data + Idx, Size - Idx);
    memcpy(Data + Idx, Word.data(), Word.size());
    return Size + Word.size();
  } else {  // Overwrite some bytes with Word.
    if (Word.size() > Size) return 0;
    size_t Idx = Rand(Size - Word.size());
    memcpy(Data + Idx, Word.data(), Word.size());
    return Size;
  }
}

size_t MutationDispatcher::Mutate_ChangeASCIIInteger(uint8_t *Data, size_t Size,
                                                     size_t MaxSize) {
  size_t B = Rand(Size);
  while (B < Size && !isdigit(Data[B])) B++;
  if (B == Size) return 0;
  size_t E = B;
  while (E < Size && isdigit(Data[E])) E++;
  assert(B < E);
  // now we have digits in [B, E).
  // strtol and friends don't accept non-zero-teminated data, parse it manually.
  uint64_t Val = Data[B] - '0';
  for (size_t i = B + 1; i < E; i++)
    Val = Val * 10 + Data[i] - '0';

  // Mutate the integer value.
  switch(Rand(5)) {
    case 0: Val++; break;
    case 1: Val--; break;
    case 2: Val /= 2; break;
    case 3: Val *= 2; break;
    case 4: Val = Rand(Val * Val); break;
    default: assert(0);
  }
  // Just replace the bytes with the new ones, don't bother moving bytes.
  for (size_t i = B; i < E; i++) {
    size_t Idx = E + B - i - 1;
    assert(Idx >= B && Idx < E);
    Data[Idx] = (Val % 10) + '0';
    Val /= 10;
  }
  return Size;
}

size_t MutationDispatcher::Mutate_CrossOver(uint8_t *Data, size_t Size,
                                            size_t MaxSize) {
  auto Corpus = MDImpl->Corpus;
  if (!Corpus || Corpus->size() < 2 || Size == 0) return 0;
  size_t Idx = Rand(Corpus->size());
  const Unit &Other = (*Corpus)[Idx];
  if (Other.empty()) return 0;
  Unit U(MaxSize);
  size_t NewSize =
      CrossOver(Data, Size, Other.data(), Other.size(), U.data(), U.size());
  assert(NewSize > 0 && "CrossOver returned empty unit");
  assert(NewSize <= MaxSize && "CrossOver returned overisized unit");
  memcpy(Data, U.data(), NewSize);
  return NewSize;
}

void MutationDispatcher::StartMutationSequence() {
  MDImpl->CurrentMutatorSequence.clear();
}

void MutationDispatcher::PrintMutationSequence() {
  Printf("MS: %zd ", MDImpl->CurrentMutatorSequence.size());
  for (auto M : MDImpl->CurrentMutatorSequence)
    Printf("%s-", M.Name);
}

// Mutates Data in place, returns new size.
size_t MutationDispatcher::Mutate(uint8_t *Data, size_t Size, size_t MaxSize) {
  assert(MaxSize > 0);
  assert(Size <= MaxSize);
  if (Size == 0) {
    for (size_t i = 0; i < MaxSize; i++)
      Data[i] = RandCh(Rand);
    return MaxSize;
  }
  assert(Size > 0);
  // Some mutations may fail (e.g. can't insert more bytes if Size == MaxSize),
  // in which case they will return 0.
  // Try several times before returning un-mutated data.
  for (int Iter = 0; Iter < 10; Iter++) {
    size_t MutatorIdx = Rand(MDImpl->Mutators.size());
    auto M = MDImpl->Mutators[MutatorIdx];
    size_t NewSize = (this->*(M.Fn))(Data, Size, MaxSize);
    if (NewSize) {
      MDImpl->CurrentMutatorSequence.push_back(M);
      return NewSize;
    }
  }
  return Size;
}

void MutationDispatcher::SetCorpus(const std::vector<Unit> *Corpus) {
  MDImpl->SetCorpus(Corpus);
}

void MutationDispatcher::AddWordToDictionary(const uint8_t *Word, size_t Size) {
  MDImpl->AddWordToDictionary(Word, Size);
}

MutationDispatcher::MutationDispatcher(FuzzerRandomBase &Rand) : Rand(Rand) {
  MDImpl = new Impl;
}

MutationDispatcher::~MutationDispatcher() { delete MDImpl; }

}  // namespace fuzzer
