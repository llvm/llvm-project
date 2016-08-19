//===- FuzzerUtil.cpp - Misc utils ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Misc utils.
//===----------------------------------------------------------------------===//

#include "FuzzerInternal.h"
#include <sstream>
#include <iomanip>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <cassert>
#include <chrono>
#include <cstring>
#include <signal.h>
#include <sstream>
#include <unistd.h>
#include <errno.h>
#include <thread>

namespace fuzzer {

void PrintHexArray(const uint8_t *Data, size_t Size,
                   const char *PrintAfter) {
  for (size_t i = 0; i < Size; i++)
    Printf("0x%x,", (unsigned)Data[i]);
  Printf("%s", PrintAfter);
}

void Print(const Unit &v, const char *PrintAfter) {
  PrintHexArray(v.data(), v.size(), PrintAfter);
}

void PrintASCIIByte(uint8_t Byte) {
  if (Byte == '\\')
    Printf("\\\\");
  else if (Byte == '"')
    Printf("\\\"");
  else if (Byte >= 32 && Byte < 127)
    Printf("%c", Byte);
  else
    Printf("\\x%02x", Byte);
}

void PrintASCII(const uint8_t *Data, size_t Size, const char *PrintAfter) {
  for (size_t i = 0; i < Size; i++)
    PrintASCIIByte(Data[i]);
  Printf("%s", PrintAfter);
}

void PrintASCII(const Word &W, const char *PrintAfter) {
  PrintASCII(W.data(), W.size(), PrintAfter);
}

void PrintASCII(const Unit &U, const char *PrintAfter) {
  PrintASCII(U.data(), U.size(), PrintAfter);
}

std::string Sha1ToString(uint8_t Sha1[kSHA1NumBytes]) {
  std::stringstream SS;
  for (int i = 0; i < kSHA1NumBytes; i++)
    SS << std::hex << std::setfill('0') << std::setw(2) << (unsigned)Sha1[i];
  return SS.str();
}

std::string Hash(const Unit &U) {
  uint8_t Hash[kSHA1NumBytes];
  ComputeSHA1(U.data(), U.size(), Hash);
  return Sha1ToString(Hash);
}

static void AlarmHandler(int, siginfo_t *, void *) {
  Fuzzer::StaticAlarmCallback();
}

static void CrashHandler(int, siginfo_t *, void *) {
  Fuzzer::StaticCrashSignalCallback();
}

static void InterruptHandler(int, siginfo_t *, void *) {
  Fuzzer::StaticInterruptCallback();
}

static void SetSigaction(int signum,
                         void (*callback)(int, siginfo_t *, void *)) {
  struct sigaction sigact;
  memset(&sigact, 0, sizeof(sigact));
  sigact.sa_sigaction = callback;
  if (sigaction(signum, &sigact, 0)) {
    Printf("libFuzzer: sigaction failed with %d\n", errno);
    exit(1);
  }
}

void SetTimer(int Seconds) {
  struct itimerval T {{Seconds, 0}, {Seconds, 0}};
  if (setitimer(ITIMER_REAL, &T, nullptr)) {
    Printf("libFuzzer: setitimer failed with %d\n", errno);
    exit(1);
  }
  SetSigaction(SIGALRM, AlarmHandler);
}

void SetSigSegvHandler() { SetSigaction(SIGSEGV, CrashHandler); }
void SetSigBusHandler() { SetSigaction(SIGBUS, CrashHandler); }
void SetSigAbrtHandler() { SetSigaction(SIGABRT, CrashHandler); }
void SetSigIllHandler() { SetSigaction(SIGILL, CrashHandler); }
void SetSigFpeHandler() { SetSigaction(SIGFPE, CrashHandler); }
void SetSigIntHandler() { SetSigaction(SIGINT, InterruptHandler); }
void SetSigTermHandler() { SetSigaction(SIGTERM, InterruptHandler); }

int NumberOfCpuCores() {
  const char *CmdLine = nullptr;
  if (LIBFUZZER_LINUX) {
    CmdLine = "nproc";
  } else if (LIBFUZZER_APPLE) {
    CmdLine = "sysctl -n hw.ncpu";
  } else {
    assert(0 && "NumberOfCpuCores() is not implemented for your platform");
  }

  FILE *F = popen(CmdLine, "r");
  int N = 1;
  if (!F || fscanf(F, "%d", &N) != 1) {
    Printf("WARNING: Failed to parse output of command \"%s\" in %s(). "
           "Assuming CPU count of 1.\n",
           CmdLine, __func__);
    N = 1;
  }

  if (pclose(F)) {
    Printf("WARNING: Executing command \"%s\" failed in %s(). "
           "Assuming CPU count of 1.\n",
           CmdLine, __func__);
    N = 1;
  }
  if (N < 1) {
    Printf("WARNING: Reported CPU count (%d) from command \"%s\" was invalid "
           "in %s(). Assuming CPU count of 1.\n",
           N, CmdLine, __func__);
    N = 1;
  }
  return N;
}

bool ToASCII(uint8_t *Data, size_t Size) {
  bool Changed = false;
  for (size_t i = 0; i < Size; i++) {
    uint8_t &X = Data[i];
    auto NewX = X;
    NewX &= 127;
    if (!isspace(NewX) && !isprint(NewX))
      NewX = ' ';
    Changed |= NewX != X;
    X = NewX;
  }
  return Changed;
}

bool IsASCII(const Unit &U) { return IsASCII(U.data(), U.size()); }

bool IsASCII(const uint8_t *Data, size_t Size) {
  for (size_t i = 0; i < Size; i++)
    if (!(isprint(Data[i]) || isspace(Data[i]))) return false;
  return true;
}

bool ParseOneDictionaryEntry(const std::string &Str, Unit *U) {
  U->clear();
  if (Str.empty()) return false;
  size_t L = 0, R = Str.size() - 1;  // We are parsing the range [L,R].
  // Skip spaces from both sides.
  while (L < R && isspace(Str[L])) L++;
  while (R > L && isspace(Str[R])) R--;
  if (R - L < 2) return false;
  // Check the closing "
  if (Str[R] != '"') return false;
  R--;
  // Find the opening "
  while (L < R && Str[L] != '"') L++;
  if (L >= R) return false;
  assert(Str[L] == '\"');
  L++;
  assert(L <= R);
  for (size_t Pos = L; Pos <= R; Pos++) {
    uint8_t V = (uint8_t)Str[Pos];
    if (!isprint(V) && !isspace(V)) return false;
    if (V =='\\') {
      // Handle '\\'
      if (Pos + 1 <= R && (Str[Pos + 1] == '\\' || Str[Pos + 1] == '"')) {
        U->push_back(Str[Pos + 1]);
        Pos++;
        continue;
      }
      // Handle '\xAB'
      if (Pos + 3 <= R && Str[Pos + 1] == 'x'
           && isxdigit(Str[Pos + 2]) && isxdigit(Str[Pos + 3])) {
        char Hex[] = "0xAA";
        Hex[2] = Str[Pos + 2];
        Hex[3] = Str[Pos + 3];
        U->push_back(strtol(Hex, nullptr, 16));
        Pos += 3;
        continue;
      }
      return false;  // Invalid escape.
    } else {
      // Any other character.
      U->push_back(V);
    }
  }
  return true;
}

bool ParseDictionaryFile(const std::string &Text, std::vector<Unit> *Units) {
  if (Text.empty()) {
    Printf("ParseDictionaryFile: file does not exist or is empty\n");
    return false;
  }
  std::istringstream ISS(Text);
  Units->clear();
  Unit U;
  int LineNo = 0;
  std::string S;
  while (std::getline(ISS, S, '\n')) {
    LineNo++;
    size_t Pos = 0;
    while (Pos < S.size() && isspace(S[Pos])) Pos++;  // Skip spaces.
    if (Pos == S.size()) continue;  // Empty line.
    if (S[Pos] == '#') continue;  // Comment line.
    if (ParseOneDictionaryEntry(S, &U)) {
      Units->push_back(U);
    } else {
      Printf("ParseDictionaryFile: error in line %d\n\t\t%s\n", LineNo,
             S.c_str());
      return false;
    }
  }
  return true;
}

void SleepSeconds(int Seconds) {
  std::this_thread::sleep_for(std::chrono::seconds(Seconds));
}

int GetPid() { return getpid(); }

std::string Base64(const Unit &U) {
  static const char Table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                              "abcdefghijklmnopqrstuvwxyz"
                              "0123456789+/";
  std::string Res;
  size_t i;
  for (i = 0; i + 2 < U.size(); i += 3) {
    uint32_t x = (U[i] << 16) + (U[i + 1] << 8) + U[i + 2];
    Res += Table[(x >> 18) & 63];
    Res += Table[(x >> 12) & 63];
    Res += Table[(x >> 6) & 63];
    Res += Table[x & 63];
  }
  if (i + 1 == U.size()) {
    uint32_t x = (U[i] << 16);
    Res += Table[(x >> 18) & 63];
    Res += Table[(x >> 12) & 63];
    Res += "==";
  } else if (i + 2 == U.size()) {
    uint32_t x = (U[i] << 16) + (U[i + 1] << 8);
    Res += Table[(x >> 18) & 63];
    Res += Table[(x >> 12) & 63];
    Res += Table[(x >> 6) & 63];
    Res += "=";
  }
  return Res;
}

size_t GetPeakRSSMb() {
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage))
    return 0;
  if (LIBFUZZER_LINUX) {
    // ru_maxrss is in KiB
    return usage.ru_maxrss >> 10;
  } else if (LIBFUZZER_APPLE) {
    // ru_maxrss is in bytes
    return usage.ru_maxrss >> 20;
  }
  assert(0 && "GetPeakRSSMb() is not implemented for your platform");
  return 0;
}

}  // namespace fuzzer
