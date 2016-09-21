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
#include <string>
#include <vector>

// Platform detection.
#ifdef __linux__
#define LIBFUZZER_LINUX 1
#define LIBFUZZER_APPLE 0
#elif __APPLE__
#define LIBFUZZER_LINUX 0
#define LIBFUZZER_APPLE 1
#else
#error "Support for your platform has not been implemented"
#endif

#ifdef __x86_64
#define ATTRIBUTE_TARGET_POPCNT __attribute__((target("popcnt")))
#else
#define ATTRIBUTE_TARGET_POPCNT
#endif

namespace fuzzer {

class Random;
class Dictionary;
class DictionaryEntry;
class MutationDispatcher;
struct FuzzingOptions;
class InputCorpus;
struct ExternalFunctions;

// Global interface to functions that may or may not be available.
extern ExternalFunctions *EF;

typedef std::vector<uint8_t> Unit;
typedef std::vector<Unit> UnitVector;
typedef int (*UserCallback)(const uint8_t *Data, size_t Size);
int FuzzerDriver(int *argc, char ***argv, UserCallback Callback);

bool IsFile(const std::string &Path);
long GetEpoch(const std::string &Path);
std::string FileToString(const std::string &Path);
Unit FileToVector(const std::string &Path, size_t MaxSize = 0);
void ReadDirToVectorOfUnits(const char *Path, std::vector<Unit> *V,
                            long *Epoch, size_t MaxSize);
void WriteToFile(const Unit &U, const std::string &Path);
void CopyFileToErr(const std::string &Path);
// Returns "Dir/FileName" or equivalent for the current OS.
std::string DirPlusFile(const std::string &DirPath,
                        const std::string &FileName);

void DupAndCloseStderr();
void CloseStdout();
void Printf(const char *Fmt, ...);
void PrintHexArray(const Unit &U, const char *PrintAfter = "");
void PrintHexArray(const uint8_t *Data, size_t Size,
                   const char *PrintAfter = "");
void PrintASCII(const uint8_t *Data, size_t Size, const char *PrintAfter = "");
void PrintASCII(const Unit &U, const char *PrintAfter = "");

void PrintPC(const char *SymbolizedFMT, const char *FallbackFMT, uintptr_t PC);
std::string Hash(const Unit &U);
void SetTimer(int Seconds);
void SetSigSegvHandler();
void SetSigBusHandler();
void SetSigAbrtHandler();
void SetSigIllHandler();
void SetSigFpeHandler();
void SetSigIntHandler();
void SetSigTermHandler();
std::string Base64(const Unit &U);
int ExecuteCommand(const std::string &Command);
size_t GetPeakRSSMb();

// Private copy of SHA1 implementation.
static const int kSHA1NumBytes = 20;
// Computes SHA1 hash of 'Len' bytes in 'Data', writes kSHA1NumBytes to 'Out'.
void ComputeSHA1(const uint8_t *Data, size_t Len, uint8_t *Out);
std::string Sha1ToString(uint8_t Sha1[kSHA1NumBytes]);

// Changes U to contain only ASCII (isprint+isspace) characters.
// Returns true iff U has been changed.
bool ToASCII(uint8_t *Data, size_t Size);
bool IsASCII(const Unit &U);
bool IsASCII(const uint8_t *Data, size_t Size);

int NumberOfCpuCores();
int GetPid();
void SleepSeconds(int Seconds);

}  // namespace fuzzer
#endif  // LLVM_FUZZER_DEFS_H
