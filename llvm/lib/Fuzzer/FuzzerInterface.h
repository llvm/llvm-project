//===- FuzzerInterface.h - Interface header for the Fuzzer ------*- C++ -* ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Define the interface between the Fuzzer and the library being tested.
//===----------------------------------------------------------------------===//

// WARNING: keep the interface free of STL or any other header-based C++ lib,
// to avoid bad interactions between the code used in the fuzzer and
// the code used in the target function.

#ifndef LLVM_FUZZER_INTERFACE_H
#define LLVM_FUZZER_INTERFACE_H

#include <limits>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>

namespace fuzzer {
typedef std::vector<uint8_t> Unit;

/// Returns an int 0. Values other than zero are reserved for future.
typedef int (*UserCallback)(const uint8_t *Data, size_t Size);
/** Simple C-like interface with a single user-supplied callback.

Usage:

#\code
#include "FuzzerInterface.h"

int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  DoStuffWithData(Data, Size);
  return 0;
}

// Implement your own main() or use the one from FuzzerMain.cpp.
int main(int argc, char **argv) {
  InitializeMeIfNeeded();
  return fuzzer::FuzzerDriver(argc, argv, LLVMFuzzerTestOneInput);
}
#\endcode
*/
int FuzzerDriver(int argc, char **argv, UserCallback Callback);

class FuzzerRandomBase {
 public:
  FuzzerRandomBase(){}
  virtual ~FuzzerRandomBase(){};
  virtual void ResetSeed(unsigned int seed) = 0;
  // Return a random number.
  virtual size_t Rand() = 0;
  // Return a random number in range [0,n).
  size_t operator()(size_t n) { return n ? Rand() % n : 0; }
  bool RandBool() { return Rand() % 2; }
};

class FuzzerRandomLibc : public FuzzerRandomBase {
 public:
  FuzzerRandomLibc(unsigned int seed) { ResetSeed(seed); }
  void ResetSeed(unsigned int seed) override;
  ~FuzzerRandomLibc() override {}
  size_t Rand() override;
};

class MutationDispatcher {
 public:
  MutationDispatcher(FuzzerRandomBase &Rand);
  ~MutationDispatcher();
  /// Indicate that we are about to start a new sequence of mutations.
  void StartMutationSequence();
  /// Print the current sequence of mutations.
  void PrintMutationSequence();
  /// Mutates data by shuffling bytes.
  size_t Mutate_ShuffleBytes(uint8_t *Data, size_t Size, size_t MaxSize);
  /// Mutates data by erasing a byte.
  size_t Mutate_EraseByte(uint8_t *Data, size_t Size, size_t MaxSize);
  /// Mutates data by inserting a byte.
  size_t Mutate_InsertByte(uint8_t *Data, size_t Size, size_t MaxSize);
  /// Mutates data by chanding one byte.
  size_t Mutate_ChangeByte(uint8_t *Data, size_t Size, size_t MaxSize);
  /// Mutates data by chanding one bit.
  size_t Mutate_ChangeBit(uint8_t *Data, size_t Size, size_t MaxSize);

  /// Mutates data by adding a word from the manual dictionary.
  size_t Mutate_AddWordFromManualDictionary(uint8_t *Data, size_t Size,
                                            size_t MaxSize);

  /// Mutates data by adding a word from the automatic dictionary.
  size_t Mutate_AddWordFromAutoDictionary(uint8_t *Data, size_t Size,
                                          size_t MaxSize);

  /// Tries to find an ASCII integer in Data, changes it to another ASCII int.
  size_t Mutate_ChangeASCIIInteger(uint8_t *Data, size_t Size, size_t MaxSize);

  /// CrossOver Data with some other element of the corpus.
  size_t Mutate_CrossOver(uint8_t *Data, size_t Size, size_t MaxSize);

  /// Applies one of the above mutations.
  /// Returns the new size of data which could be up to MaxSize.
  size_t Mutate(uint8_t *Data, size_t Size, size_t MaxSize);

  /// Creates a cross-over of two pieces of Data, returns its size.
  size_t CrossOver(const uint8_t *Data1, size_t Size1, const uint8_t *Data2,
                   size_t Size2, uint8_t *Out, size_t MaxOutSize);

  void AddWordToManualDictionary(const Unit &Word);

  void AddWordToAutoDictionary(const Unit &Word, size_t PositionHint);
  void ClearAutoDictionary();

  void SetCorpus(const std::vector<Unit> *Corpus);

 private:
  FuzzerRandomBase &Rand;
  struct Impl;
  Impl *MDImpl;
};

// For backward compatibility only, deprecated.
static inline size_t Mutate(uint8_t *Data, size_t Size, size_t MaxSize,
                            FuzzerRandomBase &Rand) {
  MutationDispatcher MD(Rand);
  return MD.Mutate(Data, Size, MaxSize);
}

/** An abstract class that allows to use user-supplied mutators with libFuzzer.

Usage:

#\code
#include "FuzzerInterface.h"
class MyFuzzer : public fuzzer::UserSuppliedFuzzer {
 public:
  MyFuzzer(fuzzer::FuzzerRandomBase *Rand);
  // Must define the target function.
  int TargetFunction(...) { ...; return 0; }
  // Optionally define the mutator.
  size_t Mutate(...) { ... }
  // Optionally define the CrossOver method.
  size_t CrossOver(...) { ... }
};

int main(int argc, char **argv) {
  MyFuzzer F;
  fuzzer::FuzzerDriver(argc, argv, F);
}
#\endcode
*/
class UserSuppliedFuzzer {
 public:
  UserSuppliedFuzzer(FuzzerRandomBase *Rand);
  /// Executes the target function on 'Size' bytes of 'Data'.
  virtual int TargetFunction(const uint8_t *Data, size_t Size) = 0;
  virtual void StartMutationSequence() { MD.StartMutationSequence(); }
  virtual void PrintMutationSequence() { MD.PrintMutationSequence(); }
  virtual void SetCorpus(const std::vector<Unit> *Corpus) {
    MD.SetCorpus(Corpus);
  }
  /// Mutates 'Size' bytes of data in 'Data' inplace into up to 'MaxSize' bytes,
  /// returns the new size of the data, which should be positive.
  virtual size_t Mutate(uint8_t *Data, size_t Size, size_t MaxSize) {
    return MD.Mutate(Data, Size, MaxSize);
  }
  /// Crosses 'Data1' and 'Data2', writes up to 'MaxOutSize' bytes into Out,
  /// returns the number of bytes written, which should be positive.
  virtual size_t CrossOver(const uint8_t *Data1, size_t Size1,
                           const uint8_t *Data2, size_t Size2,
                           uint8_t *Out, size_t MaxOutSize) {
    return MD.CrossOver(Data1, Size1, Data2, Size2, Out, MaxOutSize);
  }
  virtual ~UserSuppliedFuzzer();

  FuzzerRandomBase &GetRand() { return *Rand; }

  MutationDispatcher &GetMD() { return MD; }

 private:
  bool OwnRand = false;
  FuzzerRandomBase *Rand;
  MutationDispatcher MD;
};

/// Runs the fuzzing with the UserSuppliedFuzzer.
int FuzzerDriver(int argc, char **argv, UserSuppliedFuzzer &USF);

/// More C++-ish interface.
int FuzzerDriver(const std::vector<std::string> &Args, UserSuppliedFuzzer &USF);
int FuzzerDriver(const std::vector<std::string> &Args, UserCallback Callback);

}  // namespace fuzzer

#endif  // LLVM_FUZZER_INTERFACE_H
