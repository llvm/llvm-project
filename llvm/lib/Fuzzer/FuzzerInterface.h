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

// Optional.
// Define this only if you need to read/modify argc/argv at startup
// and you are using libFuzzer's main().
// Must return 0.
int LLVMFuzzerInitialize(int *argc, char ***argv) {
  ReadAndMaybeModify(argc, argv);
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

// Using libc's stand/rand.
class FuzzerRandomLibc : public FuzzerRandomBase {
 public:
  FuzzerRandomLibc(unsigned int seed) { ResetSeed(seed); }
  void ResetSeed(unsigned int seed) override;
  ~FuzzerRandomLibc() override {};
  size_t Rand() override;
};

// Using std::mt19937
class FuzzerRandom_mt19937 : public FuzzerRandomBase {
 public:
  FuzzerRandom_mt19937(unsigned int seed) { ResetSeed(seed); }
  void ResetSeed(unsigned int seed) override;
  ~FuzzerRandom_mt19937() override;
  size_t Rand() override;
 private:
  struct Impl;
  Impl *R = nullptr;
};

// For backward compatibility only, deprecated.
size_t Mutate(uint8_t *Data, size_t Size, size_t MaxSize,
              FuzzerRandomBase &Rand);

class MutationDispatcher;

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
  /// Mutates 'Size' bytes of data in 'Data' inplace into up to 'MaxSize' bytes,
  /// returns the new size of the data, which should be positive.
  virtual size_t Mutate(uint8_t *Data, size_t Size, size_t MaxSize);
  /// Crosses 'Data1' and 'Data2', writes up to 'MaxOutSize' bytes into Out,
  /// returns the number of bytes written, which should be positive.
  virtual size_t CrossOver(const uint8_t *Data1, size_t Size1,
                           const uint8_t *Data2, size_t Size2,
                           uint8_t *Out, size_t MaxOutSize);
  virtual ~UserSuppliedFuzzer();

  FuzzerRandomBase &GetRand() { return *Rand; }

  MutationDispatcher &GetMD() { return *MD; }

 private:
  bool OwnRand = false;
  FuzzerRandomBase *Rand;
  MutationDispatcher *MD;
};

/// Runs the fuzzing with the UserSuppliedFuzzer.
int FuzzerDriver(int argc, char **argv, UserSuppliedFuzzer &USF);

/// More C++-ish interface.
int FuzzerDriver(const std::vector<std::string> &Args, UserSuppliedFuzzer &USF);
int FuzzerDriver(const std::vector<std::string> &Args, UserCallback Callback);

}  // namespace fuzzer

#endif  // LLVM_FUZZER_INTERFACE_H
