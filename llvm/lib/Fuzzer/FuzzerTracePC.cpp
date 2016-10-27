//===- FuzzerTracePC.cpp - PC tracing--------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Trace PCs.
// This module implements __sanitizer_cov_trace_pc_guard[_init],
// the callback required for -fsanitize-coverage=trace-pc-guard instrumentation.
//
//===----------------------------------------------------------------------===//

#include <map>
#include <set>
#include <sstream>

#include "FuzzerCorpus.h"
#include "FuzzerDefs.h"
#include "FuzzerDictionary.h"
#include "FuzzerExtFunctions.h"
#include "FuzzerTracePC.h"
#include "FuzzerValueBitMap.h"

namespace fuzzer {

TracePC TPC;

void TracePC::HandleTrace(uint32_t *Guard, uintptr_t PC) {
  uint32_t Idx = *Guard;
  if (!Idx) return;
  PCs[Idx % kNumPCs] = PC;
  Counters[Idx % kNumCounters]++;
}

size_t TracePC::GetTotalPCCoverage() {
  size_t Res = 0;
  for (size_t i = 1; i < GetNumPCs(); i++)
    if (PCs[i])
      Res++;
  return Res;
}

void TracePC::HandleInit(uint32_t *Start, uint32_t *Stop) {
  if (Start == Stop || *Start) return;
  assert(NumModules < sizeof(Modules) / sizeof(Modules[0]));
  for (uint32_t *P = Start; P < Stop; P++)
    *P = ++NumGuards;
  Modules[NumModules].Start = Start;
  Modules[NumModules].Stop = Stop;
  NumModules++;
}

void TracePC::PrintModuleInfo() {
  Printf("INFO: Loaded %zd modules (%zd guards): ", NumModules, NumGuards);
  for (size_t i = 0; i < NumModules; i++)
    Printf("[%p, %p), ", Modules[i].Start, Modules[i].Stop);
  Printf("\n");
}

size_t TracePC::FinalizeTrace(InputCorpus *C, size_t InputSize, bool Shrink) {
  if (!UsingTracePcGuard()) return 0;
  size_t Res = 0;
  const size_t Step = 8;
  assert(reinterpret_cast<uintptr_t>(Counters) % Step == 0);
  size_t N = Min(kNumCounters, NumGuards + 1);
  N = (N + Step - 1) & ~(Step - 1);  // Round up.
  for (size_t Idx = 0; Idx < N; Idx += Step) {
    uint64_t Bundle = *reinterpret_cast<uint64_t*>(&Counters[Idx]);
    if (!Bundle) continue;
    for (size_t i = Idx; i < Idx + Step; i++) {
      uint8_t Counter = (Bundle >> (i * 8)) & 0xff;
      if (!Counter) continue;
      Counters[i] = 0;
      unsigned Bit = 0;
      /**/ if (Counter >= 128) Bit = 7;
      else if (Counter >= 32) Bit = 6;
      else if (Counter >= 16) Bit = 5;
      else if (Counter >= 8) Bit = 4;
      else if (Counter >= 4) Bit = 3;
      else if (Counter >= 3) Bit = 2;
      else if (Counter >= 2) Bit = 1;
      size_t Feature = (i * 8 + Bit);
      if (C->AddFeature(Feature, InputSize, Shrink))
        Res++;
    }
  }
  if (UseValueProfile)
    ValueProfileMap.ForEach([&](size_t Idx) {
      if (C->AddFeature(NumGuards + Idx, InputSize, Shrink))
        Res++;
    });
  return Res;
}

void TracePC::HandleCallerCallee(uintptr_t Caller, uintptr_t Callee) {
  const uintptr_t kBits = 12;
  const uintptr_t kMask = (1 << kBits) - 1;
  uintptr_t Idx = (Caller & kMask) | ((Callee & kMask) << kBits);
  HandleValueProfile(Idx);
}

static bool IsInterestingCoverageFile(std::string &File) {
  if (File.find("compiler-rt/lib/") != std::string::npos)
    return false; // sanitizer internal.
  if (File.find("/usr/lib/") != std::string::npos)
    return false;
  if (File.find("/usr/include/") != std::string::npos)
    return false;
  if (File == "<null>")
    return false;
  return true;
}

void TracePC::PrintNewPCs() {
  if (DoPrintNewPCs) {
    if (!PrintedPCs)
      PrintedPCs = new std::set<uintptr_t>;
    for (size_t i = 1; i < GetNumPCs(); i++)
      if (PCs[i] && PrintedPCs->insert(PCs[i]).second)
        PrintPC("\tNEW_PC: %p %F %L\n", "\tNEW_PC: %p\n", PCs[i]);
  }
}

void TracePC::PrintCoverage() {
  if (!EF->__sanitizer_symbolize_pc) {
    Printf("INFO: __sanitizer_symbolize_pc is not available,"
           " not printing coverage\n");
    return;
  }
  std::map<std::string, std::vector<uintptr_t>> CoveredPCsPerModule;
  std::map<std::string, uintptr_t> ModuleOffsets;
  std::set<std::string> CoveredFiles, CoveredFunctions, CoveredLines;
  Printf("COVERAGE:\n");
  for (size_t i = 1; i < GetNumPCs(); i++) {
    if (!PCs[i]) continue;
    std::string FileStr = DescribePC("%s", PCs[i]);
    if (!IsInterestingCoverageFile(FileStr)) continue;
    std::string FixedPCStr = DescribePC("%p", PCs[i]);
    std::string FunctionStr = DescribePC("%F", PCs[i]);
    std::string LineStr = DescribePC("%l", PCs[i]);
    // TODO(kcc): get the module using some other way since this
    // does not work with ASAN_OPTIONS=strip_path_prefix=something.
    std::string Module = DescribePC("%m", PCs[i]);
    std::string OffsetStr = DescribePC("%o", PCs[i]);
    uintptr_t FixedPC = std::stol(FixedPCStr, 0, 16);
    uintptr_t PcOffset = std::stol(OffsetStr, 0, 16);
    ModuleOffsets[Module] = FixedPC - PcOffset;
    CoveredPCsPerModule[Module].push_back(PcOffset);
    CoveredFunctions.insert(FunctionStr);
    CoveredFiles.insert(FileStr);
    if (!CoveredLines.insert(FileStr + ":" + LineStr).second)
      continue;
    Printf("COVERED: %s %s:%s\n", FunctionStr.c_str(),
           FileStr.c_str(), LineStr.c_str());
  }

  for (auto &M : CoveredPCsPerModule) {
    std::set<std::string> UncoveredFiles, UncoveredFunctions;
    std::map<std::string, std::set<int> > UncoveredLines;  // Func+File => lines
    auto &ModuleName = M.first;
    auto &CoveredOffsets = M.second;
    uintptr_t ModuleOffset = ModuleOffsets[ModuleName];
    std::sort(CoveredOffsets.begin(), CoveredOffsets.end());
    Printf("MODULE_WITH_COVERAGE: %s\n", ModuleName.c_str());
    // sancov does not yet fully support DSOs.
    // std::string Cmd = "sancov -print-coverage-pcs " + ModuleName;
    std::string Cmd = "objdump -d " + ModuleName +
        " | grep 'call.*__sanitizer_cov_trace_pc_guard' | awk -F: '{print $1}'";
    std::string SanCovOutput;
    if (!ExecuteCommandAndReadOutput(Cmd, &SanCovOutput)) {
      Printf("INFO: Command failed: %s\n", Cmd.c_str());
      continue;
    }
    std::istringstream ISS(SanCovOutput);
    std::string S;
    while (std::getline(ISS, S, '\n')) {
      uintptr_t PcOffset = std::stol(S, 0, 16);
      if (!std::binary_search(CoveredOffsets.begin(), CoveredOffsets.end(),
                              PcOffset)) {
        uintptr_t PC = ModuleOffset + PcOffset;
        auto FileStr = DescribePC("%s", PC);
        if (!IsInterestingCoverageFile(FileStr)) continue;
        if (CoveredFiles.count(FileStr) == 0) {
          UncoveredFiles.insert(FileStr);
          continue;
        }
        auto FunctionStr = DescribePC("%F", PC);
        if (CoveredFunctions.count(FunctionStr) == 0) {
          UncoveredFunctions.insert(FunctionStr);
          continue;
        }
        std::string LineStr = DescribePC("%l", PC);
        uintptr_t Line = std::stoi(LineStr);
        std::string FileLineStr = FileStr + ":" + LineStr;
        if (CoveredLines.count(FileLineStr) == 0)
          UncoveredLines[FunctionStr + " " + FileStr].insert(Line);
      }
    }
    for (auto &FileLine: UncoveredLines)
      for (int Line : FileLine.second)
        Printf("UNCOVERED_LINE: %s:%d\n", FileLine.first.c_str(), Line);
    for (auto &Func : UncoveredFunctions)
      Printf("UNCOVERED_FUNC: %s\n", Func.c_str());
    for (auto &File : UncoveredFiles)
      Printf("UNCOVERED_FILE: %s\n", File.c_str());
  }
}

// Value profile.
// We keep track of various values that affect control flow.
// These values are inserted into a bit-set-based hash map.
// Every new bit in the map is treated as a new coverage.
//
// For memcmp/strcmp/etc the interesting value is the length of the common
// prefix of the parameters.
// For cmp instructions the interesting value is a XOR of the parameters.
// The interesting value is mixed up with the PC and is then added to the map.

void TracePC::AddValueForMemcmp(void *caller_pc, const void *s1, const void *s2,
                              size_t n) {
  if (!n) return;
  size_t Len = std::min(n, (size_t)32);
  const uint8_t *A1 = reinterpret_cast<const uint8_t *>(s1);
  const uint8_t *A2 = reinterpret_cast<const uint8_t *>(s2);
  size_t I = 0;
  for (; I < Len; I++)
    if (A1[I] != A2[I])
      break;
  size_t PC = reinterpret_cast<size_t>(caller_pc);
  size_t Idx = I;
  // if (I < Len)
  //  Idx += __builtin_popcountl((A1[I] ^ A2[I])) - 1;
  TPC.HandleValueProfile((PC & 4095) | (Idx << 12));
}

void TracePC::AddValueForStrcmp(void *caller_pc, const char *s1, const char *s2,
                              size_t n) {
  if (!n) return;
  size_t Len = std::min(n, (size_t)32);
  const uint8_t *A1 = reinterpret_cast<const uint8_t *>(s1);
  const uint8_t *A2 = reinterpret_cast<const uint8_t *>(s2);
  size_t I = 0;
  for (; I < Len; I++)
    if (A1[I] != A2[I] || A1[I] == 0)
      break;
  size_t PC = reinterpret_cast<size_t>(caller_pc);
  size_t Idx = I;
  // if (I < Len && A1[I])
  //  Idx += __builtin_popcountl((A1[I] ^ A2[I])) - 1;
  TPC.HandleValueProfile((PC & 4095) | (Idx << 12));
}

template <class T>
ATTRIBUTE_TARGET_POPCNT
#ifdef __clang__  // g++ can't handle this __attribute__ here :(
__attribute__((always_inline))
#endif  // __clang__
void TracePC::HandleCmp(void *PC, T Arg1, T Arg2) {
  uintptr_t PCuint = reinterpret_cast<uintptr_t>(PC);
  uint64_t ArgXor = Arg1 ^ Arg2;
  uint64_t ArgDistance = __builtin_popcountl(ArgXor) + 1; // [1,65]
  uintptr_t Idx = ((PCuint & 4095) + 1) * ArgDistance;
  if (sizeof(T) == 4)
      TORC4.Insert(ArgXor, Arg1, Arg2);
  else if (sizeof(T) == 8)
      TORC8.Insert(ArgXor, Arg1, Arg2);
  HandleValueProfile(Idx);
}

} // namespace fuzzer

extern "C" {
__attribute__((visibility("default")))
void __sanitizer_cov_trace_pc_guard(uint32_t *Guard) {
  uintptr_t PC = (uintptr_t)__builtin_return_address(0);
  fuzzer::TPC.HandleTrace(Guard, PC);
}

__attribute__((visibility("default")))
void __sanitizer_cov_trace_pc_guard_init(uint32_t *Start, uint32_t *Stop) {
  fuzzer::TPC.HandleInit(Start, Stop);
}

__attribute__((visibility("default")))
void __sanitizer_cov_trace_pc_indir(uintptr_t Callee) {
  uintptr_t PC = (uintptr_t)__builtin_return_address(0);
  fuzzer::TPC.HandleCallerCallee(PC, Callee);
}

__attribute__((visibility("default")))
void __sanitizer_cov_trace_cmp8(uint64_t Arg1, uint64_t Arg2) {
  fuzzer::TPC.HandleCmp(__builtin_return_address(0), Arg1, Arg2);
}
__attribute__((visibility("default")))
void __sanitizer_cov_trace_cmp4(uint32_t Arg1, uint32_t Arg2) {
  fuzzer::TPC.HandleCmp(__builtin_return_address(0), Arg1, Arg2);
}
__attribute__((visibility("default")))
void __sanitizer_cov_trace_cmp2(uint16_t Arg1, uint16_t Arg2) {
  fuzzer::TPC.HandleCmp(__builtin_return_address(0), Arg1, Arg2);
}
__attribute__((visibility("default")))
void __sanitizer_cov_trace_cmp1(uint8_t Arg1, uint8_t Arg2) {
  fuzzer::TPC.HandleCmp(__builtin_return_address(0), Arg1, Arg2);
}

__attribute__((visibility("default")))
void __sanitizer_cov_trace_switch(uint64_t Val, uint64_t *Cases) {
  uint64_t N = Cases[0];
  uint64_t *Vals = Cases + 2;
  char *PC = (char*)__builtin_return_address(0);
  for (size_t i = 0; i < N; i++)
    if (Val != Vals[i])
      fuzzer::TPC.HandleCmp(PC + i, Val, Vals[i]);
}

__attribute__((visibility("default")))
void __sanitizer_cov_trace_div4(uint32_t Val) {
  fuzzer::TPC.HandleCmp(__builtin_return_address(0), Val, (uint32_t)0);
}
__attribute__((visibility("default")))
void __sanitizer_cov_trace_div8(uint64_t Val) {
  fuzzer::TPC.HandleCmp(__builtin_return_address(0), Val, (uint64_t)0);
}
__attribute__((visibility("default")))
void __sanitizer_cov_trace_gep(uintptr_t Idx) {
  fuzzer::TPC.HandleCmp(__builtin_return_address(0), Idx, (uintptr_t)0);
}

}  // extern "C"
