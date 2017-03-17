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

#include "FuzzerCorpus.h"
#include "FuzzerDefs.h"
#include "FuzzerDictionary.h"
#include "FuzzerExtFunctions.h"
#include "FuzzerIO.h"
#include "FuzzerTracePC.h"
#include "FuzzerUtil.h"
#include "FuzzerValueBitMap.h"
#include <map>
#include <set>
#include <sstream>

// The coverage counters and PCs.
// These are declared as global variables named "__sancov_*" to simplify
// experiments with inlined instrumentation.
alignas(8) ATTRIBUTE_INTERFACE
uint8_t __sancov_trace_pc_guard_8bit_counters[fuzzer::TracePC::kNumPCs];

ATTRIBUTE_INTERFACE
uintptr_t __sancov_trace_pc_pcs[fuzzer::TracePC::kNumPCs];

namespace fuzzer {

TracePC TPC;

uint8_t *TracePC::Counters() const {
  return __sancov_trace_pc_guard_8bit_counters;
}

uintptr_t *TracePC::PCs() const {
  return __sancov_trace_pc_pcs;
}

size_t TracePC::GetTotalPCCoverage() {
  size_t Res = 0;
  for (size_t i = 1, N = GetNumPCs(); i < N; i++)
    if (PCs()[i])
      Res++;
  return Res;
}

void TracePC::HandleInit(uint32_t *Start, uint32_t *Stop) {
  if (Start == Stop || *Start) return;
  assert(NumModules < sizeof(Modules) / sizeof(Modules[0]));
  for (uint32_t *P = Start; P < Stop; P++) {
    NumGuards++;
    if (NumGuards == kNumPCs) {
      RawPrint(
          "WARNING: The binary has too many instrumented PCs.\n"
          "         You may want to reduce the size of the binary\n"
          "         for more efficient fuzzing and precise coverage data\n");
    }
    *P = NumGuards % kNumPCs;
  }
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

ATTRIBUTE_NO_SANITIZE_ALL
void TracePC::HandleCallerCallee(uintptr_t Caller, uintptr_t Callee) {
  const uintptr_t kBits = 12;
  const uintptr_t kMask = (1 << kBits) - 1;
  uintptr_t Idx = (Caller & kMask) | ((Callee & kMask) << kBits);
  ValueProfileMap.AddValueModPrime(Idx);
}

void TracePC::InitializePrintNewPCs() {
  if (!DoPrintNewPCs) return;
  assert(!PrintedPCs);
  PrintedPCs = new std::set<uintptr_t>;
  for (size_t i = 1; i < GetNumPCs(); i++)
    if (PCs()[i])
      PrintedPCs->insert(PCs()[i]);
}

void TracePC::PrintNewPCs() {
  if (!DoPrintNewPCs) return;
  assert(PrintedPCs);
  for (size_t i = 1; i < GetNumPCs(); i++)
    if (PCs()[i] && PrintedPCs->insert(PCs()[i]).second)
      PrintPC("\tNEW_PC: %p %F %L\n", "\tNEW_PC: %p\n", PCs()[i]);
}

void TracePC::PrintCoverage() {
  if (!EF->__sanitizer_symbolize_pc ||
      !EF->__sanitizer_get_module_and_offset_for_pc) {
    Printf("INFO: __sanitizer_symbolize_pc or "
           "__sanitizer_get_module_and_offset_for_pc is not available,"
           " not printing coverage\n");
    return;
  }
  std::map<std::string, std::vector<uintptr_t>> CoveredPCsPerModule;
  std::map<std::string, uintptr_t> ModuleOffsets;
  std::set<std::string> CoveredDirs, CoveredFiles, CoveredFunctions,
      CoveredLines;
  Printf("COVERAGE:\n");
  for (size_t i = 1; i < GetNumPCs(); i++) {
    uintptr_t PC = PCs()[i];
    if (!PC) continue;
    std::string FileStr = DescribePC("%s", PC);
    if (!IsInterestingCoverageFile(FileStr)) continue;
    std::string FixedPCStr = DescribePC("%p", PC);
    std::string FunctionStr = DescribePC("%F", PC);
    std::string LineStr = DescribePC("%l", PC);
    char ModulePathRaw[4096] = "";  // What's PATH_MAX in portable C++?
    void *OffsetRaw = nullptr;
    if (!EF->__sanitizer_get_module_and_offset_for_pc(
            reinterpret_cast<void *>(PC), ModulePathRaw,
            sizeof(ModulePathRaw), &OffsetRaw))
      continue;
    std::string Module = ModulePathRaw;
    uintptr_t FixedPC = std::stoull(FixedPCStr, 0, 16);
    uintptr_t PcOffset = reinterpret_cast<uintptr_t>(OffsetRaw);
    ModuleOffsets[Module] = FixedPC - PcOffset;
    CoveredPCsPerModule[Module].push_back(PcOffset);
    CoveredFunctions.insert(FunctionStr);
    CoveredFiles.insert(FileStr);
    CoveredDirs.insert(DirName(FileStr));
    if (!CoveredLines.insert(FileStr + ":" + LineStr).second)
      continue;
    Printf("COVERED: %s %s:%s\n", FunctionStr.c_str(),
           FileStr.c_str(), LineStr.c_str());
  }

  std::string CoveredDirsStr;
  for (auto &Dir : CoveredDirs) {
    if (!CoveredDirsStr.empty())
      CoveredDirsStr += ",";
    CoveredDirsStr += Dir;
  }
  Printf("COVERED_DIRS: %s\n", CoveredDirsStr.c_str());

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
    std::string Cmd = DisassembleCmd(ModuleName) + " | " +
        SearchRegexCmd("call.*__sanitizer_cov_trace_pc_guard");
    std::string SanCovOutput;
    if (!ExecuteCommandAndReadOutput(Cmd, &SanCovOutput)) {
      Printf("INFO: Command failed: %s\n", Cmd.c_str());
      continue;
    }
    std::istringstream ISS(SanCovOutput);
    std::string S;
    while (std::getline(ISS, S, '\n')) {
      size_t PcOffsetEnd = S.find(':');
      if (PcOffsetEnd == std::string::npos)
        continue;
      S.resize(PcOffsetEnd);
      uintptr_t PcOffset = std::stoull(S, 0, 16);
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

inline ALWAYS_INLINE uintptr_t GetPreviousInstructionPc(uintptr_t PC) {
  // TODO: this implementation is x86 only.
  // see sanitizer_common GetPreviousInstructionPc for full implementation.
  return PC - 1;
}

void TracePC::DumpCoverage() {
  if (EF->__sanitizer_dump_coverage) {
    std::vector<uintptr_t> PCsCopy(GetNumPCs());
    for (size_t i = 0; i < GetNumPCs(); i++)
      PCsCopy[i] = PCs()[i] ? GetPreviousInstructionPc(PCs()[i]) : 0;
    EF->__sanitizer_dump_coverage(PCsCopy.data(), PCsCopy.size());
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

ATTRIBUTE_NO_SANITIZE_ALL
void TracePC::AddValueForMemcmp(void *caller_pc, const void *s1, const void *s2,
                                size_t n, bool StopAtZero) {
  if (!n) return;
  size_t Len = std::min(n, Word::GetMaxSize());
  const uint8_t *A1 = reinterpret_cast<const uint8_t *>(s1);
  const uint8_t *A2 = reinterpret_cast<const uint8_t *>(s2);
  uint8_t B1[Word::kMaxSize];
  uint8_t B2[Word::kMaxSize];
  // Copy the data into locals in this non-msan-instrumented function
  // to avoid msan complaining further.
  size_t Hash = 0;  // Compute some simple hash of both strings.
  for (size_t i = 0; i < Len; i++) {
    B1[i] = A1[i];
    B2[i] = A2[i];
    size_t T = B1[i];
    Hash ^= (T << 8) | B2[i];
  }
  size_t I = 0;
  for (; I < Len; I++)
    if (B1[I] != B2[I] || (StopAtZero && B1[I] == 0))
      break;
  size_t PC = reinterpret_cast<size_t>(caller_pc);
  size_t Idx = (PC & 4095) | (I << 12);
  ValueProfileMap.AddValue(Idx);
  TORCW.Insert(Idx ^ Hash, Word(B1, Len), Word(B2, Len));
}

template <class T>
ATTRIBUTE_TARGET_POPCNT ALWAYS_INLINE
ATTRIBUTE_NO_SANITIZE_ALL
void TracePC::HandleCmp(uintptr_t PC, T Arg1, T Arg2) {
  uint64_t ArgXor = Arg1 ^ Arg2;
  uint64_t ArgDistance = __builtin_popcountll(ArgXor) + 1; // [1,65]
  uintptr_t Idx = ((PC & 4095) + 1) * ArgDistance;
  if (sizeof(T) == 4)
      TORC4.Insert(ArgXor, Arg1, Arg2);
  else if (sizeof(T) == 8)
      TORC8.Insert(ArgXor, Arg1, Arg2);
  ValueProfileMap.AddValue(Idx);
}

} // namespace fuzzer

extern "C" {
ATTRIBUTE_INTERFACE
ATTRIBUTE_NO_SANITIZE_ALL
void __sanitizer_cov_trace_pc_guard(uint32_t *Guard) {
  uintptr_t PC = reinterpret_cast<uintptr_t>(__builtin_return_address(0));
  uint32_t Idx = *Guard;
  __sancov_trace_pc_pcs[Idx] = PC;
  __sancov_trace_pc_guard_8bit_counters[Idx]++;
}

ATTRIBUTE_INTERFACE
void __sanitizer_cov_trace_pc_guard_init(uint32_t *Start, uint32_t *Stop) {
  fuzzer::TPC.HandleInit(Start, Stop);
}

ATTRIBUTE_INTERFACE
ATTRIBUTE_NO_SANITIZE_ALL
void __sanitizer_cov_trace_pc_indir(uintptr_t Callee) {
  uintptr_t PC = reinterpret_cast<uintptr_t>(__builtin_return_address(0));
  fuzzer::TPC.HandleCallerCallee(PC, Callee);
}

ATTRIBUTE_INTERFACE
ATTRIBUTE_NO_SANITIZE_ALL
ATTRIBUTE_TARGET_POPCNT
void __sanitizer_cov_trace_cmp8(uint64_t Arg1, uint64_t Arg2) {
  uintptr_t PC = reinterpret_cast<uintptr_t>(__builtin_return_address(0));
  fuzzer::TPC.HandleCmp(PC, Arg1, Arg2);
}

ATTRIBUTE_INTERFACE
ATTRIBUTE_NO_SANITIZE_ALL
ATTRIBUTE_TARGET_POPCNT
void __sanitizer_cov_trace_cmp4(uint32_t Arg1, uint32_t Arg2) {
  uintptr_t PC = reinterpret_cast<uintptr_t>(__builtin_return_address(0));
  fuzzer::TPC.HandleCmp(PC, Arg1, Arg2);
}

ATTRIBUTE_INTERFACE
ATTRIBUTE_NO_SANITIZE_ALL
ATTRIBUTE_TARGET_POPCNT
void __sanitizer_cov_trace_cmp2(uint16_t Arg1, uint16_t Arg2) {
  uintptr_t PC = reinterpret_cast<uintptr_t>(__builtin_return_address(0));
  fuzzer::TPC.HandleCmp(PC, Arg1, Arg2);
}

ATTRIBUTE_INTERFACE
ATTRIBUTE_NO_SANITIZE_ALL
ATTRIBUTE_TARGET_POPCNT
void __sanitizer_cov_trace_cmp1(uint8_t Arg1, uint8_t Arg2) {
  uintptr_t PC = reinterpret_cast<uintptr_t>(__builtin_return_address(0));
  fuzzer::TPC.HandleCmp(PC, Arg1, Arg2);
}

ATTRIBUTE_INTERFACE
ATTRIBUTE_NO_SANITIZE_ALL
ATTRIBUTE_TARGET_POPCNT
void __sanitizer_cov_trace_switch(uint64_t Val, uint64_t *Cases) {
  uint64_t N = Cases[0];
  uint64_t ValSizeInBits = Cases[1];
  uint64_t *Vals = Cases + 2;
  // Skip the most common and the most boring case.
  if (Vals[N - 1]  < 256 && Val < 256)
    return;
  uintptr_t PC = reinterpret_cast<uintptr_t>(__builtin_return_address(0));
  size_t i;
  uint64_t Token = 0;
  for (i = 0; i < N; i++) {
    Token = Val ^ Vals[i];
    if (Val < Vals[i])
      break;
  }

  if (ValSizeInBits == 16)
    fuzzer::TPC.HandleCmp(PC + i, static_cast<uint16_t>(Token), (uint16_t)(0));
  else if (ValSizeInBits == 32)
    fuzzer::TPC.HandleCmp(PC + i, static_cast<uint32_t>(Token), (uint32_t)(0));
  else
    fuzzer::TPC.HandleCmp(PC + i, Token, (uint64_t)(0));
}

ATTRIBUTE_INTERFACE
ATTRIBUTE_NO_SANITIZE_ALL
ATTRIBUTE_TARGET_POPCNT
void __sanitizer_cov_trace_div4(uint32_t Val) {
  uintptr_t PC = reinterpret_cast<uintptr_t>(__builtin_return_address(0));
  fuzzer::TPC.HandleCmp(PC, Val, (uint32_t)0);
}

ATTRIBUTE_INTERFACE
ATTRIBUTE_NO_SANITIZE_ALL
ATTRIBUTE_TARGET_POPCNT
void __sanitizer_cov_trace_div8(uint64_t Val) {
  uintptr_t PC = reinterpret_cast<uintptr_t>(__builtin_return_address(0));
  fuzzer::TPC.HandleCmp(PC, Val, (uint64_t)0);
}

ATTRIBUTE_INTERFACE
ATTRIBUTE_NO_SANITIZE_ALL
ATTRIBUTE_TARGET_POPCNT
void __sanitizer_cov_trace_gep(uintptr_t Idx) {
  uintptr_t PC = reinterpret_cast<uintptr_t>(__builtin_return_address(0));
  fuzzer::TPC.HandleCmp(PC, Idx, (uintptr_t)0);
}
}  // extern "C"
