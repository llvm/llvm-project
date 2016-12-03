//===- FuzzerUtil.h - Internal header for the Fuzzer Utils ------*- C++ -* ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Util functions.
//===----------------------------------------------------------------------===//
#ifndef LLVM_FUZZER_UTIL_H
#define LLVM_FUZZER_UTIL_H

#include "FuzzerDefs.h"

namespace fuzzer {

void PrintHexArray(const Unit &U, const char *PrintAfter = "");

void PrintHexArray(const uint8_t *Data, size_t Size,
                   const char *PrintAfter = "");

void PrintASCII(const uint8_t *Data, size_t Size, const char *PrintAfter = "");

void PrintASCII(const Unit &U, const char *PrintAfter = "");

// Changes U to contain only ASCII (isprint+isspace) characters.
// Returns true iff U has been changed.
bool ToASCII(uint8_t *Data, size_t Size);

bool IsASCII(const Unit &U);

bool IsASCII(const uint8_t *Data, size_t Size);

std::string Base64(const Unit &U);

void PrintPC(const char *SymbolizedFMT, const char *FallbackFMT, uintptr_t PC);

std::string DescribePC(const char *SymbolizedFMT, uintptr_t PC);

int NumberOfCpuCores();

bool ExecuteCommandAndReadOutput(const std::string &Command, std::string *Out);

// Platform specific functions.
void SetTimer(int Seconds);

void SetSigSegvHandler();
void SetSigBusHandler();
void SetSigAbrtHandler();
void SetSigIllHandler();
void SetSigFpeHandler();
void SetSigIntHandler();
void SetSigTermHandler();

void SleepSeconds(int Seconds);

int GetPid();

size_t GetPeakRSSMb();

int ExecuteCommand(const std::string &Command);

FILE *OpenProcessPipe(const char *Command, const char *Mode);

const void *SearchMemory(const void *haystack, size_t haystacklen,
                         const void *needle, size_t needlelen);

}  // namespace fuzzer
#endif  // LLVM_FUZZER_UTIL_H
