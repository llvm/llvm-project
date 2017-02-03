//===- FuzzerTraceState.cpp - Trace-based fuzzer mutator ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Data tracing.
//===----------------------------------------------------------------------===//

#include "FuzzerDictionary.h"
#include "FuzzerInternal.h"
#include "FuzzerIO.h"
#include "FuzzerMutate.h"
#include "FuzzerTracePC.h"
#include <algorithm>
#include <cstring>
#include <map>
#include <set>
#include <thread>

namespace fuzzer {

// Declared as static globals for faster checks inside the hooks.
static bool RecordingMemmem = false;
static bool DoingMyOwnMemmem = false;

ScopedDoingMyOwnMemmem::ScopedDoingMyOwnMemmem() { DoingMyOwnMemmem = true; }
ScopedDoingMyOwnMemmem::~ScopedDoingMyOwnMemmem() { DoingMyOwnMemmem = false; }

class TraceState {
public:
  TraceState(MutationDispatcher &MD, const FuzzingOptions &Options,
             const Fuzzer *F)
      : MD(MD), Options(Options), F(F) {}

  void StartTraceRecording() {
    if (!Options.UseMemmem)
      return;
    RecordingMemmem = true;
    InterestingWords.clear();
    MD.ClearAutoDictionary();
  }

  void StopTraceRecording() {
    if (!RecordingMemmem)
      return;
    for (auto &W : InterestingWords)
      MD.AddWordToAutoDictionary({W});
  }

  void AddInterestingWord(const uint8_t *Data, size_t Size) {
    if (!RecordingMemmem || !F->InFuzzingThread()) return;
    if (Size <= 1) return;
    Size = std::min(Size, Word::GetMaxSize());
    Word W(Data, Size);
    InterestingWords.insert(W);
  }

 private:

  // TODO: std::set is too inefficient, need to have a custom DS here.
  std::set<Word> InterestingWords;
  MutationDispatcher &MD;
  const FuzzingOptions Options;
  const Fuzzer *F;
};

static TraceState *TS;

void Fuzzer::StartTraceRecording() {
  if (!TS) return;
  TS->StartTraceRecording();
}

void Fuzzer::StopTraceRecording() {
  if (!TS) return;
  TS->StopTraceRecording();
}

void Fuzzer::InitializeTraceState() {
  if (!Options.UseMemmem) return;
  TS = new TraceState(MD, Options, this);
}

static size_t InternalStrnlen(const char *S, size_t MaxLen) {
  size_t Len = 0;
  for (; Len < MaxLen && S[Len]; Len++) {}
  return Len;
}

}  // namespace fuzzer

using fuzzer::TS;

extern "C" {

// We may need to avoid defining weak hooks to stay compatible with older clang.
#ifndef LLVM_FUZZER_DEFINES_SANITIZER_WEAK_HOOOKS
# define LLVM_FUZZER_DEFINES_SANITIZER_WEAK_HOOOKS 1
#endif

#if LLVM_FUZZER_DEFINES_SANITIZER_WEAK_HOOOKS

ATTRIBUTE_INTERFACE ATTRIBUTE_NO_SANITIZE_MEMORY
void __sanitizer_weak_hook_memcmp(void *caller_pc, const void *s1,
                                  const void *s2, size_t n, int result) {
  if (result == 0) return;  // No reason to mutate.
  if (n <= 1) return;  // Not interesting.
  fuzzer::TPC.AddValueForMemcmp(caller_pc, s1, s2, n, /*StopAtZero*/false);
}

ATTRIBUTE_INTERFACE ATTRIBUTE_NO_SANITIZE_MEMORY
void __sanitizer_weak_hook_strncmp(void *caller_pc, const char *s1,
                                   const char *s2, size_t n, int result) {
  if (result == 0) return;  // No reason to mutate.
  size_t Len1 = fuzzer::InternalStrnlen(s1, n);
  size_t Len2 = fuzzer::InternalStrnlen(s2, n);
  n = std::min(n, Len1);
  n = std::min(n, Len2);
  if (n <= 1) return;  // Not interesting.
  fuzzer::TPC.AddValueForMemcmp(caller_pc, s1, s2, n, /*StopAtZero*/true);
}


ATTRIBUTE_INTERFACE ATTRIBUTE_NO_SANITIZE_MEMORY
void __sanitizer_weak_hook_strcmp(void *caller_pc, const char *s1,
                                   const char *s2, int result) {
  if (result == 0) return;  // No reason to mutate.
  size_t Len1 = strlen(s1);
  size_t Len2 = strlen(s2);
  size_t N = std::min(Len1, Len2);
  if (N <= 1) return;  // Not interesting.
  fuzzer::TPC.AddValueForMemcmp(caller_pc, s1, s2, N, /*StopAtZero*/true);
}

ATTRIBUTE_INTERFACE ATTRIBUTE_NO_SANITIZE_MEMORY
void __sanitizer_weak_hook_strncasecmp(void *called_pc, const char *s1,
                                       const char *s2, size_t n, int result) {
  return __sanitizer_weak_hook_strncmp(called_pc, s1, s2, n, result);
}

ATTRIBUTE_INTERFACE ATTRIBUTE_NO_SANITIZE_MEMORY
void __sanitizer_weak_hook_strcasecmp(void *called_pc, const char *s1,
                                      const char *s2, int result) {
  return __sanitizer_weak_hook_strcmp(called_pc, s1, s2, result);
}

ATTRIBUTE_INTERFACE ATTRIBUTE_NO_SANITIZE_MEMORY
void __sanitizer_weak_hook_strstr(void *called_pc, const char *s1,
                                  const char *s2, char *result) {
  TS->AddInterestingWord(reinterpret_cast<const uint8_t *>(s2), strlen(s2));
}

ATTRIBUTE_INTERFACE ATTRIBUTE_NO_SANITIZE_MEMORY
void __sanitizer_weak_hook_strcasestr(void *called_pc, const char *s1,
                                      const char *s2, char *result) {
  TS->AddInterestingWord(reinterpret_cast<const uint8_t *>(s2), strlen(s2));
}

ATTRIBUTE_INTERFACE ATTRIBUTE_NO_SANITIZE_MEMORY
void __sanitizer_weak_hook_memmem(void *called_pc, const void *s1, size_t len1,
                                  const void *s2, size_t len2, void *result) {
  if (fuzzer::DoingMyOwnMemmem) return;
  TS->AddInterestingWord(reinterpret_cast<const uint8_t *>(s2), len2);
}

#endif  // LLVM_FUZZER_DEFINES_SANITIZER_WEAK_HOOOKS
}  // extern "C"
