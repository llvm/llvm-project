//===- FuzzerDefs.h - Internal header for the Fuzzer ------------*- C++ -* ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Basic definitions.
//===----------------------------------------------------------------------===//

#ifndef LLVM_FUZZER_DEFS_H
#define LLVM_FUZZER_DEFS_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// Platform detection.
#ifdef __linux__
#define LIBFUZZER_APPLE 0
#define LIBFUZZER_LINUX 1
#define LIBFUZZER_WINDOWS 0
#elif __APPLE__
#define LIBFUZZER_APPLE 1
#define LIBFUZZER_LINUX 0
#define LIBFUZZER_WINDOWS 0
#elif _WIN32
#define LIBFUZZER_APPLE 0
#define LIBFUZZER_LINUX 0
#define LIBFUZZER_WINDOWS 1
#else
#error "Support for your platform has not been implemented"
#endif

#define LIBFUZZER_POSIX LIBFUZZER_APPLE || LIBFUZZER_LINUX

#ifdef __x86_64
#define ATTRIBUTE_TARGET_POPCNT __attribute__((target("popcnt")))
#else
#define ATTRIBUTE_TARGET_POPCNT
#endif


#ifdef __clang__  // avoid gcc warning.
#  define ATTRIBUTE_NO_SANITIZE_MEMORY __attribute__((no_sanitize("memory")))
#  define ALWAYS_INLINE __attribute__((always_inline))
#else
#  define ATTRIBUTE_NO_SANITIZE_MEMORY
#  define ALWAYS_INLINE
#endif // __clang__

#define ATTRIBUTE_NO_SANITIZE_ADDRESS __attribute__((no_sanitize_address))

#if defined(__has_feature)
#  if __has_feature(address_sanitizer)
#    define ATTRIBUTE_NO_SANITIZE_ALL ATTRIBUTE_NO_SANITIZE_ADDRESS
#  elif __has_feature(memory_sanitizer)
#    define ATTRIBUTE_NO_SANITIZE_ALL ATTRIBUTE_NO_SANITIZE_MEMORY
#  else
#    define ATTRIBUTE_NO_SANITIZE_ALL
#  endif
#else
#  define ATTRIBUTE_NO_SANITIZE_ALL
#endif

#if LIBFUZZER_WINDOWS
#define ATTRIBUTE_INTERFACE __declspec(dllexport)
#else
#define ATTRIBUTE_INTERFACE __attribute__((visibility("default")))
#endif

namespace fuzzer {

template <class T> T Min(T a, T b) { return a < b ? a : b; }
template <class T> T Max(T a, T b) { return a > b ? a : b; }

class Random;
class Dictionary;
class DictionaryEntry;
class MutationDispatcher;
struct FuzzingOptions;
class InputCorpus;
struct InputInfo;
struct ExternalFunctions;

// Global interface to functions that may or may not be available.
extern ExternalFunctions *EF;

typedef std::vector<uint8_t> Unit;
typedef std::vector<Unit> UnitVector;
typedef int (*UserCallback)(const uint8_t *Data, size_t Size);

int FuzzerDriver(int *argc, char ***argv, UserCallback Callback);

struct ScopedDoingMyOwnMemOrStr {
  ScopedDoingMyOwnMemOrStr() { DoingMyOwnMemOrStr++; }
  ~ScopedDoingMyOwnMemOrStr() { DoingMyOwnMemOrStr--; }
  static int DoingMyOwnMemOrStr;
};

inline uint8_t  Bswap(uint8_t x)  { return x; }
inline uint16_t Bswap(uint16_t x) { return __builtin_bswap16(x); }
inline uint32_t Bswap(uint32_t x) { return __builtin_bswap32(x); }
inline uint64_t Bswap(uint64_t x) { return __builtin_bswap64(x); }

uint8_t *ExtraCountersBegin();
uint8_t *ExtraCountersEnd();
void ClearExtraCounters();

}  // namespace fuzzer

#endif  // LLVM_FUZZER_DEFS_H
