//===- FuzzerTraceState.cpp - Trace-based fuzzer mutator ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This file implements a mutation algorithm based on instruction traces and
// on taint analysis feedback from DFSan.
//
// Instruction traces are special hooks inserted by the compiler around
// interesting instructions. Currently supported traces:
//   * __sanitizer_cov_trace_cmp -- inserted before every ICMP instruction,
//    receives the type, size and arguments of ICMP.
//
// Every time a traced event is intercepted we analyse the data involved
// in the event and suggest a mutation for future executions.
// For example if 4 bytes of data that derive from input bytes {4,5,6,7}
// are compared with a constant 12345,
// we try to insert 12345, 12344, 12346 into bytes
// {4,5,6,7} of the next fuzzed inputs.
//
// The fuzzer can work only with the traces, or with both traces and DFSan.
//
// DataFlowSanitizer (DFSan) is a tool for
// generalised dynamic data flow (taint) analysis:
// http://clang.llvm.org/docs/DataFlowSanitizer.html .
//
// The approach with DFSan-based fuzzing has some similarity to
// "Taint-based Directed Whitebox Fuzzing"
// by Vijay Ganesh & Tim Leek & Martin Rinard:
// http://dspace.mit.edu/openaccess-disseminate/1721.1/59320,
// but it uses a full blown LLVM IR taint analysis and separate instrumentation
// to analyze all of the "attack points" at once.
//
// Workflow with DFSan:
//   * lib/Fuzzer/Fuzzer*.cpp is compiled w/o any instrumentation.
//   * The code under test is compiled with DFSan *and* with instruction traces.
//   * Every call to HOOK(a,b) is replaced by DFSan with
//     __dfsw_HOOK(a, b, label(a), label(b)) so that __dfsw_HOOK
//     gets all the taint labels for the arguments.
//   * At the Fuzzer startup we assign a unique DFSan label
//     to every byte of the input string (Fuzzer::CurrentUnitData) so that
//     for any chunk of data we know which input bytes it has derived from.
//   * The __dfsw_* functions (implemented in this file) record the
//     parameters (i.e. the application data and the corresponding taint labels)
//     in a global state.
//
// Parts of this code will not function when DFSan is not linked in.
// Instead of using ifdefs and thus requiring a separate build of lib/Fuzzer
// we redeclare the dfsan_* interface functions as weak and check if they
// are nullptr before calling.
// If this approach proves to be useful we may add attribute(weak) to the
// dfsan declarations in dfsan_interface.h
//
// This module is in the "proof of concept" stage.
// It is capable of solving only the simplest puzzles
// like test/dfsan/DFSanSimpleCmpTest.cpp.
//===----------------------------------------------------------------------===//

/* Example of manual usage (-fsanitize=dataflow is optional):
(
  cd $LLVM/lib/Fuzzer/
  clang  -fPIC -c -g -O2 -std=c++11 Fuzzer*.cpp
  clang++ -O0 -std=c++11 -fsanitize-coverage=edge,trace-cmp \
    -fsanitize=dataflow \
    test/SimpleCmpTest.cpp Fuzzer*.o
  ./a.out -use_traces=1
)
*/

#include "FuzzerDFSan.h"
#include "FuzzerInternal.h"

#include <algorithm>
#include <cstring>
#include <thread>
#include <map>

#if !LLVM_FUZZER_SUPPORTS_DFSAN
// Stubs for dfsan for platforms where dfsan does not exist and weak
// functions don't work.
extern "C" {
dfsan_label dfsan_create_label(const char *desc, void *userdata) { return 0; }
void dfsan_set_label(dfsan_label label, void *addr, size_t size) {}
void dfsan_add_label(dfsan_label label, void *addr, size_t size) {}
const struct dfsan_label_info *dfsan_get_label_info(dfsan_label label) {
  return nullptr;
}
dfsan_label dfsan_read_label(const void *addr, size_t size) { return 0; }
}  // extern "C"
#endif  // !LLVM_FUZZER_SUPPORTS_DFSAN

namespace fuzzer {

// These values are copied from include/llvm/IR/InstrTypes.h.
// We do not include the LLVM headers here to remain independent.
// If these values ever change, an assertion in ComputeCmp will fail.
enum Predicate {
  ICMP_EQ = 32,  ///< equal
  ICMP_NE = 33,  ///< not equal
  ICMP_UGT = 34, ///< unsigned greater than
  ICMP_UGE = 35, ///< unsigned greater or equal
  ICMP_ULT = 36, ///< unsigned less than
  ICMP_ULE = 37, ///< unsigned less or equal
  ICMP_SGT = 38, ///< signed greater than
  ICMP_SGE = 39, ///< signed greater or equal
  ICMP_SLT = 40, ///< signed less than
  ICMP_SLE = 41, ///< signed less or equal
};

template <class U, class S>
bool ComputeCmp(size_t CmpType, U Arg1, U Arg2) {
  switch(CmpType) {
    case ICMP_EQ : return Arg1 == Arg2;
    case ICMP_NE : return Arg1 != Arg2;
    case ICMP_UGT: return Arg1 > Arg2;
    case ICMP_UGE: return Arg1 >= Arg2;
    case ICMP_ULT: return Arg1 < Arg2;
    case ICMP_ULE: return Arg1 <= Arg2;
    case ICMP_SGT: return (S)Arg1 > (S)Arg2;
    case ICMP_SGE: return (S)Arg1 >= (S)Arg2;
    case ICMP_SLT: return (S)Arg1 < (S)Arg2;
    case ICMP_SLE: return (S)Arg1 <= (S)Arg2;
    default: assert(0 && "unsupported CmpType");
  }
  return false;
}

static bool ComputeCmp(size_t CmpSize, size_t CmpType, uint64_t Arg1,
                       uint64_t Arg2) {
  if (CmpSize == 8) return ComputeCmp<uint64_t, int64_t>(CmpType, Arg1, Arg2);
  if (CmpSize == 4) return ComputeCmp<uint32_t, int32_t>(CmpType, Arg1, Arg2);
  if (CmpSize == 2) return ComputeCmp<uint16_t, int16_t>(CmpType, Arg1, Arg2);
  if (CmpSize == 1) return ComputeCmp<uint8_t, int8_t>(CmpType, Arg1, Arg2);
  // Other size, ==
  if (CmpType == ICMP_EQ) return Arg1 == Arg2;
  // assert(0 && "unsupported cmp and type size combination");
  return true;
}

// As a simplification we use the range of input bytes instead of a set of input
// bytes.
struct LabelRange {
  uint16_t Beg, End;  // Range is [Beg, End), thus Beg==End is an empty range.

  LabelRange(uint16_t Beg = 0, uint16_t End = 0) : Beg(Beg), End(End) {}

  static LabelRange Join(LabelRange LR1, LabelRange LR2) {
    if (LR1.Beg == LR1.End) return LR2;
    if (LR2.Beg == LR2.End) return LR1;
    return {std::min(LR1.Beg, LR2.Beg), std::max(LR1.End, LR2.End)};
  }
  LabelRange &Join(LabelRange LR) {
    return *this = Join(*this, LR);
  }
  static LabelRange Singleton(const dfsan_label_info *LI) {
    uint16_t Idx = (uint16_t)(uintptr_t)LI->userdata;
    assert(Idx > 0);
    return {(uint16_t)(Idx - 1), Idx};
  }
};

// For now, very simple: put Size bytes of Data at position Pos.
struct TraceBasedMutation {
  static const size_t kMaxSize = 28;
  uint32_t Pos : 24;
  uint32_t Size : 8;
  uint8_t  Data[kMaxSize];
};

const size_t TraceBasedMutation::kMaxSize;

// Declared as static globals for faster checks inside the hooks.
static bool RecordingTraces = false;
static bool RecordingMemcmp = false;

class TraceState {
 public:
   TraceState(UserSuppliedFuzzer &USF, const Fuzzer::FuzzingOptions &Options,
              uint8_t **CurrentUnitData, size_t *CurrentUnitSize)
       : USF(USF), Options(Options), CurrentUnitData(CurrentUnitData),
         CurrentUnitSize(CurrentUnitSize) {
     // Current trace collection is not thread-friendly and it probably
     // does not have to be such, but at least we should not crash in presence
     // of threads. So, just ignore all traces coming from all threads but one.
     IsMyThread = true;
  }

  LabelRange GetLabelRange(dfsan_label L);
  void DFSanCmpCallback(uintptr_t PC, size_t CmpSize, size_t CmpType,
                        uint64_t Arg1, uint64_t Arg2, dfsan_label L1,
                        dfsan_label L2);
  void DFSanMemcmpCallback(size_t CmpSize, const uint8_t *Data1,
                           const uint8_t *Data2, dfsan_label L1,
                           dfsan_label L2);
  void DFSanSwitchCallback(uint64_t PC, size_t ValSizeInBits, uint64_t Val,
                           size_t NumCases, uint64_t *Cases, dfsan_label L);
  void TraceCmpCallback(uintptr_t PC, size_t CmpSize, size_t CmpType,
                        uint64_t Arg1, uint64_t Arg2);
  void TraceMemcmpCallback(size_t CmpSize, const uint8_t *Data1,
                           const uint8_t *Data2);

  void TraceSwitchCallback(uintptr_t PC, size_t ValSizeInBits, uint64_t Val,
                           size_t NumCases, uint64_t *Cases);
  int TryToAddDesiredData(uint64_t PresentData, uint64_t DesiredData,
                           size_t DataSize);
  int TryToAddDesiredData(const uint8_t *PresentData,
                          const uint8_t *DesiredData, size_t DataSize);

  void StartTraceRecording() {
    if (!Options.UseTraces && !Options.UseMemcmp) return;
    RecordingTraces = Options.UseTraces;
    RecordingMemcmp = Options.UseMemcmp;
    NumMutations = 0;
    USF.GetMD().ClearAutoDictionary();
  }

  void StopTraceRecording() {
    if (!RecordingTraces && !RecordingMemcmp) return;
    RecordingTraces = false;
    RecordingMemcmp = false;
    for (size_t i = 0; i < NumMutations; i++) {
      auto &M = Mutations[i];
      Unit U(M.Data, M.Data + M.Size);
      if (Options.Verbosity >= 2) {
        AutoDictUnitCounts[U]++;
        AutoDictAdds++;
        if ((AutoDictAdds & (AutoDictAdds - 1)) == 0) {
          typedef std::pair<size_t, Unit> CU;
          std::vector<CU> CountedUnits;
          for (auto &I : AutoDictUnitCounts)
            CountedUnits.push_back(std::make_pair(I.second, I.first));
          std::sort(CountedUnits.begin(), CountedUnits.end(),
                    [](const CU &a, const CU &b) { return a.first > b.first; });
          Printf("AutoDict:\n");
          for (auto &I : CountedUnits) {
            Printf("   %zd ", I.first);
            PrintASCII(I.second);
            Printf("\n");
          }
        }
      }
      USF.GetMD().AddWordToAutoDictionary(U, M.Pos);
    }
  }

  void AddMutation(uint32_t Pos, uint32_t Size, const uint8_t *Data) {
    if (NumMutations >= kMaxMutations) return;
    assert(Size <= TraceBasedMutation::kMaxSize);
    auto &M = Mutations[NumMutations++];
    M.Pos = Pos;
    M.Size = Size;
    memcpy(M.Data, Data, Size);
  }

  void AddMutation(uint32_t Pos, uint32_t Size, uint64_t Data) {
    assert(Size <= sizeof(Data));
    AddMutation(Pos, Size, reinterpret_cast<uint8_t*>(&Data));
  }

 private:
  bool IsTwoByteData(uint64_t Data) {
    int64_t Signed = static_cast<int64_t>(Data);
    Signed >>= 16;
    return Signed == 0 || Signed == -1L;
  }
  static const size_t kMaxMutations = 1 << 16;
  size_t NumMutations;
  TraceBasedMutation Mutations[kMaxMutations];
  LabelRange LabelRanges[1 << (sizeof(dfsan_label) * 8)];
  UserSuppliedFuzzer &USF;
  const Fuzzer::FuzzingOptions &Options;
  uint8_t **CurrentUnitData;
  size_t *CurrentUnitSize;
  std::map<Unit, size_t> AutoDictUnitCounts;
  size_t AutoDictAdds = 0;
  static thread_local bool IsMyThread;
};

thread_local bool TraceState::IsMyThread;

LabelRange TraceState::GetLabelRange(dfsan_label L) {
  LabelRange &LR = LabelRanges[L];
  if (LR.Beg < LR.End || L == 0)
    return LR;
  const dfsan_label_info *LI = dfsan_get_label_info(L);
  if (LI->l1 || LI->l2)
    return LR = LabelRange::Join(GetLabelRange(LI->l1), GetLabelRange(LI->l2));
  return LR = LabelRange::Singleton(LI);
}

void TraceState::DFSanCmpCallback(uintptr_t PC, size_t CmpSize, size_t CmpType,
                                  uint64_t Arg1, uint64_t Arg2, dfsan_label L1,
                                  dfsan_label L2) {
  assert(ReallyHaveDFSan());
  if (!RecordingTraces || !IsMyThread) return;
  if (L1 == 0 && L2 == 0)
    return;  // Not actionable.
  if (L1 != 0 && L2 != 0)
    return;  // Probably still actionable.
  bool Res = ComputeCmp(CmpSize, CmpType, Arg1, Arg2);
  uint64_t Data = L1 ? Arg2 : Arg1;
  LabelRange LR = L1 ? GetLabelRange(L1) : GetLabelRange(L2);

  for (size_t Pos = LR.Beg; Pos + CmpSize <= LR.End; Pos++) {
    AddMutation(Pos, CmpSize, Data);
    AddMutation(Pos, CmpSize, Data + 1);
    AddMutation(Pos, CmpSize, Data - 1);
  }

  if (CmpSize > LR.End - LR.Beg)
    AddMutation(LR.Beg, (unsigned)(LR.End - LR.Beg), Data);


  if (Options.Verbosity >= 3)
    Printf("DFSanCmpCallback: PC %lx S %zd T %zd A1 %llx A2 %llx R %d L1 %d L2 "
           "%d MU %zd\n",
           PC, CmpSize, CmpType, Arg1, Arg2, Res, L1, L2, NumMutations);
}

void TraceState::DFSanMemcmpCallback(size_t CmpSize, const uint8_t *Data1,
                                     const uint8_t *Data2, dfsan_label L1,
                                     dfsan_label L2) {

  assert(ReallyHaveDFSan());
  if (!RecordingMemcmp || !IsMyThread) return;
  if (L1 == 0 && L2 == 0)
    return;  // Not actionable.
  if (L1 != 0 && L2 != 0)
    return;  // Probably still actionable.

  const uint8_t *Data = L1 ? Data2 : Data1;
  LabelRange LR = L1 ? GetLabelRange(L1) : GetLabelRange(L2);
  for (size_t Pos = LR.Beg; Pos + CmpSize <= LR.End; Pos++) {
    AddMutation(Pos, CmpSize, Data);
    if (Options.Verbosity >= 3)
      Printf("DFSanMemcmpCallback: Pos %d Size %d\n", Pos, CmpSize);
  }
}

void TraceState::DFSanSwitchCallback(uint64_t PC, size_t ValSizeInBits,
                                     uint64_t Val, size_t NumCases,
                                     uint64_t *Cases, dfsan_label L) {
  assert(ReallyHaveDFSan());
  if (!RecordingTraces || !IsMyThread) return;
  if (!L) return;  // Not actionable.
  LabelRange LR = GetLabelRange(L);
  size_t ValSize = ValSizeInBits / 8;
  bool TryShort = IsTwoByteData(Val);
  for (size_t i = 0; i < NumCases; i++)
    TryShort &= IsTwoByteData(Cases[i]);

  for (size_t Pos = LR.Beg; Pos + ValSize <= LR.End; Pos++)
    for (size_t i = 0; i < NumCases; i++)
      AddMutation(Pos, ValSize, Cases[i]);

  if (TryShort)
    for (size_t Pos = LR.Beg; Pos + 2 <= LR.End; Pos++)
      for (size_t i = 0; i < NumCases; i++)
        AddMutation(Pos, 2, Cases[i]);

  if (Options.Verbosity >= 3)
    Printf("DFSanSwitchCallback: PC %lx Val %zd SZ %zd # %zd L %d: {%d, %d} "
           "TryShort %d\n",
           PC, Val, ValSize, NumCases, L, LR.Beg, LR.End, TryShort);
}

int TraceState::TryToAddDesiredData(uint64_t PresentData, uint64_t DesiredData,
                                    size_t DataSize) {
  int Res = 0;
  const uint8_t *Beg = *CurrentUnitData;
  const uint8_t *End = Beg + *CurrentUnitSize;
  for (const uint8_t *Cur = Beg; Cur < End; Cur++) {
    Cur = (uint8_t *)memmem(Cur, End - Cur, &PresentData, DataSize);
    if (!Cur)
      break;
    size_t Pos = Cur - Beg;
    assert(Pos < *CurrentUnitSize);
    AddMutation(Pos, DataSize, DesiredData);
    AddMutation(Pos, DataSize, DesiredData + 1);
    AddMutation(Pos, DataSize, DesiredData - 1);
    Res++;
  }
  return Res;
}

int TraceState::TryToAddDesiredData(const uint8_t *PresentData,
                                    const uint8_t *DesiredData,
                                    size_t DataSize) {
  int Res = 0;
  const uint8_t *Beg = *CurrentUnitData;
  const uint8_t *End = Beg + *CurrentUnitSize;
  for (const uint8_t *Cur = Beg; Cur < End; Cur++) {
    Cur = (uint8_t *)memmem(Cur, End - Cur, PresentData, DataSize);
    if (!Cur)
      break;
    size_t Pos = Cur - Beg;
    assert(Pos < *CurrentUnitSize);
    AddMutation(Pos, DataSize, DesiredData);
    Res++;
  }
  return Res;
}

void TraceState::TraceCmpCallback(uintptr_t PC, size_t CmpSize, size_t CmpType,
                                  uint64_t Arg1, uint64_t Arg2) {
  if (!RecordingTraces || !IsMyThread) return;
  if ((CmpType == ICMP_EQ || CmpType == ICMP_NE) && Arg1 == Arg2)
    return;  // No reason to mutate.
  int Added = 0;
  Added += TryToAddDesiredData(Arg1, Arg2, CmpSize);
  Added += TryToAddDesiredData(Arg2, Arg1, CmpSize);
  if (!Added && CmpSize == 4 && IsTwoByteData(Arg1) && IsTwoByteData(Arg2)) {
    Added += TryToAddDesiredData(Arg1, Arg2, 2);
    Added += TryToAddDesiredData(Arg2, Arg1, 2);
  }
  if (Options.Verbosity >= 3 && Added)
    Printf("TraceCmp %zd/%zd: %p %zd %zd\n", CmpSize, CmpType, PC, Arg1, Arg2);
}

void TraceState::TraceMemcmpCallback(size_t CmpSize, const uint8_t *Data1,
                                     const uint8_t *Data2) {
  if (!RecordingMemcmp || !IsMyThread) return;
  CmpSize = std::min(CmpSize, TraceBasedMutation::kMaxSize);
  int Added2 = TryToAddDesiredData(Data1, Data2, CmpSize);
  int Added1 = TryToAddDesiredData(Data2, Data1, CmpSize);
  if ((Added1 || Added2) && Options.Verbosity >= 3) {
    Printf("MemCmp Added %d%d: ", Added1, Added2);
    if (Added1) PrintASCII(Data1, CmpSize);
    if (Added2) PrintASCII(Data2, CmpSize);
    Printf("\n");
  }
}

void TraceState::TraceSwitchCallback(uintptr_t PC, size_t ValSizeInBits,
                                     uint64_t Val, size_t NumCases,
                                     uint64_t *Cases) {
  if (!RecordingTraces || !IsMyThread) return;
  size_t ValSize = ValSizeInBits / 8;
  bool TryShort = IsTwoByteData(Val);
  for (size_t i = 0; i < NumCases; i++)
    TryShort &= IsTwoByteData(Cases[i]);

  if (Options.Verbosity >= 3)
    Printf("TraceSwitch: %p %zd # %zd; TryShort %d\n", PC, Val, NumCases,
           TryShort);

  for (size_t i = 0; i < NumCases; i++) {
    TryToAddDesiredData(Val, Cases[i], ValSize);
    if (TryShort)
      TryToAddDesiredData(Val, Cases[i], 2);
  }
}

static TraceState *TS;

void Fuzzer::StartTraceRecording() {
  if (!TS) return;
  TS->StartTraceRecording();
}

void Fuzzer::StopTraceRecording() {
  if (!TS) return;
  TS->StopTraceRecording();
}

void Fuzzer::AssignTaintLabels(uint8_t *Data, size_t Size) {
  if (!Options.UseTraces && !Options.UseMemcmp) return;
  if (!ReallyHaveDFSan()) return;
  for (size_t i = 0; i < Size; i++)
    dfsan_set_label(i + 1, &Data[i], 1);
}

void Fuzzer::InitializeTraceState() {
  if (!Options.UseTraces && !Options.UseMemcmp) return;
  TS = new TraceState(USF, Options, &CurrentUnitData, &CurrentUnitSize);
  if (ReallyHaveDFSan()) {
    for (size_t i = 0; i < static_cast<size_t>(Options.MaxLen); i++) {
      dfsan_label L = dfsan_create_label("input", (void *)(i + 1));
      // We assume that no one else has called dfsan_create_label before.
      if (L != i + 1) {
        Printf("DFSan labels are not starting from 1, exiting\n");
        exit(1);
      }
    }
  }
}

static size_t InternalStrnlen(const char *S, size_t MaxLen) {
  size_t Len = 0;
  for (; Len < MaxLen && S[Len]; Len++) {}
  return Len;
}

}  // namespace fuzzer

using fuzzer::TS;
using fuzzer::RecordingTraces;
using fuzzer::RecordingMemcmp;

extern "C" {
void __dfsw___sanitizer_cov_trace_cmp(uint64_t SizeAndType, uint64_t Arg1,
                                      uint64_t Arg2, dfsan_label L0,
                                      dfsan_label L1, dfsan_label L2) {
  if (!RecordingTraces) return;
  assert(L0 == 0);
  uintptr_t PC = reinterpret_cast<uintptr_t>(__builtin_return_address(0));
  uint64_t CmpSize = (SizeAndType >> 32) / 8;
  uint64_t Type = (SizeAndType << 32) >> 32;
  TS->DFSanCmpCallback(PC, CmpSize, Type, Arg1, Arg2, L1, L2);
}

void __dfsw___sanitizer_cov_trace_switch(uint64_t Val, uint64_t *Cases,
                                         dfsan_label L1, dfsan_label L2) {
  if (!RecordingTraces) return;
  uintptr_t PC = reinterpret_cast<uintptr_t>(__builtin_return_address(0));
  TS->DFSanSwitchCallback(PC, Cases[1], Val, Cases[0], Cases+2, L1);
}

void dfsan_weak_hook_memcmp(void *caller_pc, const void *s1, const void *s2,
                            size_t n, dfsan_label s1_label,
                            dfsan_label s2_label, dfsan_label n_label) {
  if (!RecordingMemcmp) return;
  dfsan_label L1 = dfsan_read_label(s1, n);
  dfsan_label L2 = dfsan_read_label(s2, n);
  TS->DFSanMemcmpCallback(n, reinterpret_cast<const uint8_t *>(s1),
                          reinterpret_cast<const uint8_t *>(s2), L1, L2);
}

void dfsan_weak_hook_strncmp(void *caller_pc, const char *s1, const char *s2,
                             size_t n, dfsan_label s1_label,
                             dfsan_label s2_label, dfsan_label n_label) {
  if (!RecordingMemcmp) return;
  n = std::min(n, fuzzer::InternalStrnlen(s1, n));
  n = std::min(n, fuzzer::InternalStrnlen(s2, n));
  dfsan_label L1 = dfsan_read_label(s1, n);
  dfsan_label L2 = dfsan_read_label(s2, n);
  TS->DFSanMemcmpCallback(n, reinterpret_cast<const uint8_t *>(s1),
                          reinterpret_cast<const uint8_t *>(s2), L1, L2);
}

void dfsan_weak_hook_strcmp(void *caller_pc, const char *s1, const char *s2,
                            dfsan_label s1_label, dfsan_label s2_label) {
  if (!RecordingMemcmp) return;
  size_t Len1 = strlen(s1);
  size_t Len2 = strlen(s2);
  size_t N = std::min(Len1, Len2);
  if (N <= 1) return;  // Not interesting.
  dfsan_label L1 = dfsan_read_label(s1, Len1);
  dfsan_label L2 = dfsan_read_label(s2, Len2);
  TS->DFSanMemcmpCallback(N, reinterpret_cast<const uint8_t *>(s1),
                          reinterpret_cast<const uint8_t *>(s2), L1, L2);
}

// We may need to avoid defining weak hooks to stay compatible with older clang.
#ifndef LLVM_FUZZER_DEFINES_SANITIZER_WEAK_HOOOKS
# define LLVM_FUZZER_DEFINES_SANITIZER_WEAK_HOOOKS 1
#endif

#if LLVM_FUZZER_DEFINES_SANITIZER_WEAK_HOOOKS
void __sanitizer_weak_hook_memcmp(void *caller_pc, const void *s1,
                                  const void *s2, size_t n, int result) {
  if (!RecordingMemcmp) return;
  if (result == 0) return;  // No reason to mutate.
  if (n <= 1) return;  // Not interesting.
  TS->TraceMemcmpCallback(n, reinterpret_cast<const uint8_t *>(s1),
                          reinterpret_cast<const uint8_t *>(s2));
}

void __sanitizer_weak_hook_strncmp(void *caller_pc, const char *s1,
                                   const char *s2, size_t n, int result) {
  if (!RecordingMemcmp) return;
  if (result == 0) return;  // No reason to mutate.
  size_t Len1 = fuzzer::InternalStrnlen(s1, n);
  size_t Len2 = fuzzer::InternalStrnlen(s2, n);
  n = std::min(n, Len1);
  n = std::min(n, Len2);
  if (n <= 1) return;  // Not interesting.
  TS->TraceMemcmpCallback(n, reinterpret_cast<const uint8_t *>(s1),
                          reinterpret_cast<const uint8_t *>(s2));
}

void __sanitizer_weak_hook_strcmp(void *caller_pc, const char *s1,
                                   const char *s2, int result) {
  if (!RecordingMemcmp) return;
  if (result == 0) return;  // No reason to mutate.
  size_t Len1 = strlen(s1);
  size_t Len2 = strlen(s2);
  size_t N = std::min(Len1, Len2);
  if (N <= 1) return;  // Not interesting.
  TS->TraceMemcmpCallback(N, reinterpret_cast<const uint8_t *>(s1),
                          reinterpret_cast<const uint8_t *>(s2));
}

#endif  // LLVM_FUZZER_DEFINES_SANITIZER_WEAK_HOOOKS

__attribute__((visibility("default")))
void __sanitizer_cov_trace_cmp(uint64_t SizeAndType, uint64_t Arg1,
                               uint64_t Arg2) {
  if (!RecordingTraces) return;
  uintptr_t PC = reinterpret_cast<uintptr_t>(__builtin_return_address(0));
  uint64_t CmpSize = (SizeAndType >> 32) / 8;
  uint64_t Type = (SizeAndType << 32) >> 32;
  TS->TraceCmpCallback(PC, CmpSize, Type, Arg1, Arg2);
}

__attribute__((visibility("default")))
void __sanitizer_cov_trace_switch(uint64_t Val, uint64_t *Cases) {
  if (!RecordingTraces) return;
  uintptr_t PC = reinterpret_cast<uintptr_t>(__builtin_return_address(0));
  TS->TraceSwitchCallback(PC, Cases[1], Val, Cases[0], Cases + 2);
}

}  // extern "C"
