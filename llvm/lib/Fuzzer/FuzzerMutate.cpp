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

class DictionaryEntry {
 public:
  DictionaryEntry() {}
  DictionaryEntry(Word W) : W(W) {}
  DictionaryEntry(Word W, size_t PositionHint) : W(W), PositionHint(PositionHint) {}
  const Word &GetW() const { return W; }

  bool HasPositionHint() const { return PositionHint != std::numeric_limits<size_t>::max(); }
  size_t GetPositionHint() const {
    assert(HasPositionHint());
    return PositionHint;
  }
  void IncUseCount() { UseCount++; }
  void IncSuccessCount() { SuccessCount++; }
  size_t GetUseCount() const { return UseCount; }
  size_t GetSuccessCount() const {return SuccessCount; }

private:
  Word W;
  size_t PositionHint = std::numeric_limits<size_t>::max();
  size_t UseCount = 0;
  size_t SuccessCount = 0;
};

class Dictionary {
 public:
  static const size_t kMaxDictSize = 1 << 14;

  bool ContainsWord(const Word &W) const {
    return std::any_of(begin(), end(), [&](const DictionaryEntry &DE) {
      return DE.GetW() == W;
    });
  }
  const DictionaryEntry *begin() const { return &DE[0]; }
  const DictionaryEntry *end() const { return begin() + Size; }
  DictionaryEntry & operator[] (size_t Idx) {
    assert(Idx < Size);
    return DE[Idx];
  }
  void push_back(DictionaryEntry DE) {
    if (Size < kMaxDictSize)
      this->DE[Size++] = DE;
  }
  void clear() { Size = 0; }
  bool empty() const { return Size == 0; }
  size_t size() const { return Size; }

private:
  DictionaryEntry DE[kMaxDictSize];
  size_t Size = 0;
};

const size_t Dictionary::kMaxDictSize;

struct MutationDispatcher::Impl {
  // Dictionary provided by the user via -dict=DICT_FILE.
  Dictionary ManualDictionary;
  // Temporary dictionary modified by the fuzzer itself,
  // recreated periodically.
  Dictionary TempAutoDictionary;
  // Persistent dictionary modified by the fuzzer, consists of
  // entries that led to successfull discoveries in the past mutations.
  Dictionary PersistentAutoDictionary;

  std::vector<Mutator> Mutators;
  std::vector<Mutator> CurrentMutatorSequence;
  std::vector<DictionaryEntry *> CurrentDictionaryEntrySequence;
  const std::vector<Unit> *Corpus = nullptr;
  FuzzerRandomBase &Rand;

  void Add(Mutator M) { Mutators.push_back(M); }
  Impl(FuzzerRandomBase &Rand) : Rand(Rand) {
    Add({&MutationDispatcher::Mutate_EraseByte, "EraseByte"});
    Add({&MutationDispatcher::Mutate_InsertByte, "InsertByte"});
    Add({&MutationDispatcher::Mutate_ChangeByte, "ChangeByte"});
    Add({&MutationDispatcher::Mutate_ChangeBit, "ChangeBit"});
    Add({&MutationDispatcher::Mutate_ShuffleBytes, "ShuffleBytes"});
    Add({&MutationDispatcher::Mutate_ChangeASCIIInteger, "ChangeASCIIInt"});
    Add({&MutationDispatcher::Mutate_CrossOver, "CrossOver"});
    Add({&MutationDispatcher::Mutate_AddWordFromManualDictionary,
         "AddFromManualDict"});
    Add({&MutationDispatcher::Mutate_AddWordFromTemporaryAutoDictionary,
         "AddFromTempAutoDict"});
    Add({&MutationDispatcher::Mutate_AddWordFromPersistentAutoDictionary,
         "AddFromPersAutoDict"});
  }
  void SetCorpus(const std::vector<Unit> *Corpus) { this->Corpus = Corpus; }
  size_t AddWordFromDictionary(Dictionary &D, uint8_t *Data, size_t Size,
                               size_t MaxSize);
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
  size_t ShuffleAmount =
      Rand(std::min(Size, (size_t)8)) + 1; // [1,8] and <= Size.
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

size_t MutationDispatcher::Mutate_AddWordFromManualDictionary(uint8_t *Data,
                                                              size_t Size,
                                                              size_t MaxSize) {
  return MDImpl->AddWordFromDictionary(MDImpl->ManualDictionary, Data, Size,
                                       MaxSize);
}

size_t MutationDispatcher::Mutate_AddWordFromTemporaryAutoDictionary(
    uint8_t *Data, size_t Size, size_t MaxSize) {
  return MDImpl->AddWordFromDictionary(MDImpl->TempAutoDictionary, Data, Size,
                                       MaxSize);
}

size_t MutationDispatcher::Mutate_AddWordFromPersistentAutoDictionary(
    uint8_t *Data, size_t Size, size_t MaxSize) {
  return MDImpl->AddWordFromDictionary(MDImpl->PersistentAutoDictionary, Data, Size,
                                       MaxSize);
}

size_t MutationDispatcher::Impl::AddWordFromDictionary(Dictionary &D,
                                                       uint8_t *Data,
                                                       size_t Size,
                                                       size_t MaxSize) {
  if (D.empty()) return 0;
  DictionaryEntry &DE = D[Rand(D.size())];
  const Word &W = DE.GetW();
  bool UsePositionHint = DE.HasPositionHint() &&
                         DE.GetPositionHint() + W.size() < Size && Rand.RandBool();
  if (Rand.RandBool()) {  // Insert W.
    if (Size + W.size() > MaxSize) return 0;
    size_t Idx = UsePositionHint ? DE.GetPositionHint() : Rand(Size + 1);
    memmove(Data + Idx + W.size(), Data + Idx, Size - Idx);
    memcpy(Data + Idx, W.data(), W.size());
    Size += W.size();
  } else {  // Overwrite some bytes with W.
    if (W.size() > Size) return 0;
    size_t Idx = UsePositionHint ? DE.GetPositionHint() : Rand(Size - W.size());
    memcpy(Data + Idx, W.data(), W.size());
  }
  DE.IncUseCount();
  CurrentDictionaryEntrySequence.push_back(&DE);
  return Size;
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
  MDImpl->CurrentDictionaryEntrySequence.clear();
}

// Copy successful dictionary entries to PersistentAutoDictionary.
void MutationDispatcher::RecordSuccessfulMutationSequence() {
  for (auto DE : MDImpl->CurrentDictionaryEntrySequence) {
    // MDImpl->PersistentAutoDictionary.AddWithSuccessCountOne(DE);
    DE->IncSuccessCount();
    // Linear search is fine here as this happens seldom.
    if (!MDImpl->PersistentAutoDictionary.ContainsWord(DE->GetW()))
      MDImpl->PersistentAutoDictionary.push_back({DE->GetW(), 1});
  }
}

void MutationDispatcher::PrintRecommendedDictionary() {
  std::vector<DictionaryEntry> V;
  for (auto &DE : MDImpl->PersistentAutoDictionary)
    if (!MDImpl->ManualDictionary.ContainsWord(DE.GetW()))
      V.push_back(DE);
  if (V.empty()) return;
  Printf("###### Recommended dictionary. ######\n");
  for (auto &DE: V) {
    Printf("\"");
    PrintASCII(DE.GetW(), "\"");
    Printf(" # Uses: %zd\n", DE.GetUseCount());
  }
  Printf("###### End of recommended dictionary. ######\n");
}

void MutationDispatcher::PrintMutationSequence() {
  Printf("MS: %zd ", MDImpl->CurrentMutatorSequence.size());
  for (auto M : MDImpl->CurrentMutatorSequence)
    Printf("%s-", M.Name);
  if (!MDImpl->CurrentDictionaryEntrySequence.empty()) {
    Printf(" DE: ");
    for (auto DE : MDImpl->CurrentDictionaryEntrySequence) {
      Printf("\"");
      PrintASCII(DE->GetW(), "\"-");
    }
  }
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

void MutationDispatcher::AddWordToManualDictionary(const Word &W) {
  MDImpl->ManualDictionary.push_back(
      {W, std::numeric_limits<size_t>::max()});
}

void MutationDispatcher::AddWordToAutoDictionary(const Word &W,
                                                 size_t PositionHint) {
  static const size_t kMaxAutoDictSize = 1 << 14;
  if (MDImpl->TempAutoDictionary.size() >= kMaxAutoDictSize) return;
  MDImpl->TempAutoDictionary.push_back({W, PositionHint});
}

void MutationDispatcher::ClearAutoDictionary() {
  MDImpl->TempAutoDictionary.clear();
}

MutationDispatcher::MutationDispatcher(FuzzerRandomBase &Rand) : Rand(Rand) {
  MDImpl = new Impl(Rand);
}

MutationDispatcher::~MutationDispatcher() { delete MDImpl; }

}  // namespace fuzzer
