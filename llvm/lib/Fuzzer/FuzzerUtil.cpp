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
#include <sys/time.h>
#include <cassert>
#include <cstring>
#include <signal.h>
#include <sstream>
#include <unistd.h>

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

void PrintASCII(const Unit &U, const char *PrintAfter) {
  for (auto X : U)
    PrintASCIIByte(X);
  Printf("%s", PrintAfter);
}

std::string Hash(const Unit &U) {
  uint8_t Hash[kSHA1NumBytes];
  ComputeSHA1(U.data(), U.size(), Hash);
  std::stringstream SS;
  for (int i = 0; i < kSHA1NumBytes; i++)
    SS << std::hex << std::setfill('0') << std::setw(2) << (unsigned)Hash[i];
  return SS.str();
}

static void AlarmHandler(int, siginfo_t *, void *) {
  Fuzzer::StaticAlarmCallback();
}

void SetTimer(int Seconds) {
  struct itimerval T {{Seconds, 0}, {Seconds, 0}};
  int Res = setitimer(ITIMER_REAL, &T, nullptr);
  assert(Res == 0);
  struct sigaction sigact;
  memset(&sigact, 0, sizeof(sigact));
  sigact.sa_sigaction = AlarmHandler;
  Res = sigaction(SIGALRM, &sigact, 0);
  assert(Res == 0);
}

int NumberOfCpuCores() {
  FILE *F = popen("nproc", "r");
  int N = 0;
  fscanf(F, "%d", &N);
  fclose(F);
  return N;
}

int ExecuteCommand(const std::string &Command) {
  return system(Command.c_str());
}

bool ToASCII(Unit &U) {
  bool Changed = false;
  for (auto &X : U) {
    auto NewX = X;
    NewX &= 127;
    if (!isspace(NewX) && !isprint(NewX))
      NewX = ' ';
    Changed |= NewX != X;
    X = NewX;
  }
  return Changed;
}

bool IsASCII(const Unit &U) {
  for (auto X : U)
    if (!(isprint(X) || isspace(X))) return false;
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

}  // namespace fuzzer
