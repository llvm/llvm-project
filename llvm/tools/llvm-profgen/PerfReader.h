//===-- PerfReader.h - perfscript reader -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_PROFGEN_PERFREADER_H
#define LLVM_TOOLS_LLVM_PROFGEN_PERFREADER_H
#include "ErrorHandling.h"
#include "ProfiledBinary.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Regex.h"
#include <fstream>
#include <list>
#include <map>
#include <vector>

using namespace llvm;
using namespace sampleprof;

namespace llvm {
namespace sampleprof {

// Stream based trace line iterator
class TraceStream {
  std::string CurrentLine;
  std::ifstream Fin;
  bool IsAtEoF = false;
  uint64_t LineNumber = 0;

public:
  TraceStream(StringRef Filename) : Fin(Filename.str()) {
    if (!Fin.good())
      exitWithError("Error read input perf script file", Filename);
    advance();
  }

  StringRef getCurrentLine() {
    assert(!IsAtEoF && "Line iterator reaches the End-of-File!");
    return CurrentLine;
  }

  uint64_t getLineNumber() { return LineNumber; }

  bool isAtEoF() { return IsAtEoF; }

  // Read the next line
  void advance() {
    if (!std::getline(Fin, CurrentLine)) {
      IsAtEoF = true;
      return;
    }
    LineNumber++;
  }
};

// The type of perfscript
enum PerfScriptType {
  PERF_INVILID = 0,
  PERF_LBR = 1,       // Only LBR sample
  PERF_LBR_STACK = 2, // Hybrid sample including call stack and LBR stack.
};

// The parsed LBR sample entry.
struct LBREntry {
  uint64_t Source = 0;
  uint64_t Target = 0;
  // An artificial branch stands for a series of consecutive branches starting
  // from the current binary with a transition through external code and
  // eventually landing back in the current binary.
  bool IsArtificial = false;
  LBREntry(uint64_t S, uint64_t T, bool I)
      : Source(S), Target(T), IsArtificial(I) {}
};

// Hash interface for generic data of type T
// Data should implement a \fn getHashCode and a \fn isEqual
// Currently getHashCode is non-virtual to avoid the overhead of calling vtable,
// i.e we explicitly calculate hash of derived class, assign to base class's
// HashCode. This also provides the flexibility for calculating the hash code
// incrementally(like rolling hash) during frame stack unwinding since unwinding
// only changes the leaf of frame stack. \fn isEqual is a virtual function,
// which will have perf overhead. In the future, if we redesign a better hash
// function, then we can just skip this or switch to non-virtual function(like
// just ignore comparision if hash conflicts probabilities is low)
template <class T> class Hashable {
public:
  std::shared_ptr<T> Data;
  Hashable(const std::shared_ptr<T> &D) : Data(D) {}

  // Hash code generation
  struct Hash {
    uint64_t operator()(const Hashable<T> &Key) const {
      // Don't make it virtual for getHashCode
      assert(Key.Data->getHashCode() && "Should generate HashCode for it!");
      return Key.Data->getHashCode();
    }
  };

  // Hash equal
  struct Equal {
    bool operator()(const Hashable<T> &LHS, const Hashable<T> &RHS) const {
      // Precisely compare the data, vtable will have overhead.
      return LHS.Data->isEqual(RHS.Data.get());
    }
  };

  T *getPtr() const { return Data.get(); }
};

// Base class to extend for all types of perf sample
struct PerfSample {
  uint64_t HashCode = 0;

  virtual ~PerfSample() = default;
  uint64_t getHashCode() const { return HashCode; }
  virtual bool isEqual(const PerfSample *K) const {
    return HashCode == K->HashCode;
  };

  // Utilities for LLVM-style RTTI
  enum PerfKind { PK_HybridSample };
  const PerfKind Kind;
  PerfKind getKind() const { return Kind; }
  PerfSample(PerfKind K) : Kind(K){};
};

// The parsed hybrid sample including call stack and LBR stack.
struct HybridSample : public PerfSample {
  // Profiled binary that current frame address belongs to
  ProfiledBinary *Binary;
  // Call stack recorded in FILO(leaf to root) order
  std::list<uint64_t> CallStack;
  // LBR stack recorded in FIFO order
  SmallVector<LBREntry, 16> LBRStack;

  HybridSample() : PerfSample(PK_HybridSample){};
  static bool classof(const PerfSample *K) {
    return K->getKind() == PK_HybridSample;
  }

  // Used for sample aggregation
  bool isEqual(const PerfSample *K) const override {
    const HybridSample *Other = dyn_cast<HybridSample>(K);
    if (Other->Binary != Binary)
      return false;
    const std::list<uint64_t> &OtherCallStack = Other->CallStack;
    const SmallVector<LBREntry, 16> &OtherLBRStack = Other->LBRStack;

    if (CallStack.size() != OtherCallStack.size() ||
        LBRStack.size() != OtherLBRStack.size())
      return false;

    auto Iter = CallStack.begin();
    for (auto Address : OtherCallStack) {
      if (Address != *Iter++)
        return false;
    }

    for (size_t I = 0; I < OtherLBRStack.size(); I++) {
      if (LBRStack[I].Source != OtherLBRStack[I].Source ||
          LBRStack[I].Target != OtherLBRStack[I].Target)
        return false;
    }
    return true;
  }

  void genHashCode() {
    // Use simple DJB2 hash
    auto HashCombine = [](uint64_t H, uint64_t V) {
      return ((H << 5) + H) + V;
    };
    uint64_t Hash = 5381;
    Hash = HashCombine(Hash, reinterpret_cast<uint64_t>(Binary));
    for (const auto &Value : CallStack) {
      Hash = HashCombine(Hash, Value);
    }
    for (const auto &Entry : LBRStack) {
      Hash = HashCombine(Hash, Entry.Source);
      Hash = HashCombine(Hash, Entry.Target);
    }
    HashCode = Hash;
  }
};

// After parsing the sample, we record the samples by aggregating them
// into this counter. The key stores the sample data and the value is
// the sample repeat times.
using AggregatedCounter =
    std::unordered_map<Hashable<PerfSample>, uint64_t,
                       Hashable<PerfSample>::Hash, Hashable<PerfSample>::Equal>;

// The state for the unwinder, it doesn't hold the data but only keep the
// pointer/index of the data, While unwinding, the CallStack is changed
// dynamicially and will be recorded as the context of the sample
struct UnwindState {
  // Profiled binary that current frame address belongs to
  const ProfiledBinary *Binary;
  // TODO: switch to use trie for call stack
  std::list<uint64_t> CallStack;
  // Used to fall through the LBR stack
  uint32_t LBRIndex = 0;
  // Reference to HybridSample.LBRStack
  const SmallVector<LBREntry, 16> &LBRStack;
  // Used to iterate the address range
  InstructionPointer InstPtr;
  UnwindState(const HybridSample *Sample)
      : Binary(Sample->Binary), CallStack(Sample->CallStack),
        LBRStack(Sample->LBRStack),
        InstPtr(Sample->Binary, Sample->CallStack.front()) {}

  bool validateInitialState() {
    uint64_t LBRLeaf = LBRStack[LBRIndex].Target;
    uint64_t StackLeaf = CallStack.front();
    // When we take a stack sample, ideally the sampling distance between the
    // leaf IP of stack and the last LBR target shouldn't be very large.
    // Use a heuristic size (0x100) to filter out broken records.
    if (StackLeaf < LBRLeaf || StackLeaf >= LBRLeaf + 0x100) {
      WithColor::warning() << "Bogus trace: stack tip = "
                           << format("%#010x", StackLeaf)
                           << ", LBR tip = " << format("%#010x\n", LBRLeaf);
      return false;
    }
    return true;
  }

  void checkStateConsistency() {
    assert(InstPtr.Address == CallStack.front() &&
           "IP should align with context leaf");
  }

  std::string getExpandedContextStr() const {
    return Binary->getExpandedContextStr(CallStack);
  }
  const ProfiledBinary *getBinary() const { return Binary; }
  bool hasNextLBR() const { return LBRIndex < LBRStack.size(); }
  uint64_t getCurrentLBRSource() const { return LBRStack[LBRIndex].Source; }
  uint64_t getCurrentLBRTarget() const { return LBRStack[LBRIndex].Target; }
  const LBREntry &getCurrentLBR() const { return LBRStack[LBRIndex]; }
  void advanceLBR() { LBRIndex++; }
};

// Base class for sample counter key with context
struct ContextKey {
  uint64_t HashCode = 0;
  virtual ~ContextKey() = default;
  uint64_t getHashCode() const { return HashCode; }
  virtual bool isEqual(const ContextKey *K) const {
    return HashCode == K->HashCode;
  };

  // Utilities for LLVM-style RTTI
  enum ContextKind { CK_StringBased, CK_ProbeBased };
  const ContextKind Kind;
  ContextKind getKind() const { return Kind; }
  ContextKey(ContextKind K) : Kind(K){};
};

// String based context id
struct StringBasedCtxKey : public ContextKey {
  std::string Context;
  StringBasedCtxKey() : ContextKey(CK_StringBased){};
  static bool classof(const ContextKey *K) {
    return K->getKind() == CK_StringBased;
  }

  bool isEqual(const ContextKey *K) const override {
    const StringBasedCtxKey *Other = dyn_cast<StringBasedCtxKey>(K);
    return Context == Other->Context;
  }

  void genHashCode() { HashCode = hash_value(Context); }
};

// Probe based context key as the intermediate key of context
// String based context key will introduce redundant string handling
// since the callee context is inferred from the context string which
// need to be splitted by '@' to get the last location frame, so we
// can just use probe instead and generate the string in the end.
struct ProbeBasedCtxKey : public ContextKey {
  SmallVector<const PseudoProbe *, 16> Probes;

  ProbeBasedCtxKey() : ContextKey(CK_ProbeBased) {}
  static bool classof(const ContextKey *K) {
    return K->getKind() == CK_ProbeBased;
  }

  bool isEqual(const ContextKey *K) const override {
    const ProbeBasedCtxKey *O = dyn_cast<ProbeBasedCtxKey>(K);
    assert(O != nullptr && "Probe based key shouldn't be null in isEqual");
    return std::equal(Probes.begin(), Probes.end(), O->Probes.begin(),
                      O->Probes.end());
  }

  void genHashCode() {
    for (const auto *P : Probes) {
      HashCode = hash_combine(HashCode, P);
    }
    if (HashCode == 0) {
      // Avoid zero value of HashCode when it's an empty list
      HashCode = 1;
    }
  }
};

// The counter of branch samples for one function indexed by the branch,
// which is represented as the source and target offset pair.
using BranchSample = std::map<std::pair<uint64_t, uint64_t>, uint64_t>;
// The counter of range samples for one function indexed by the range,
// which is represented as the start and end offset pair.
using RangeSample = std::map<std::pair<uint64_t, uint64_t>, uint64_t>;
// Wrapper for sample counters including range counter and branch counter
struct SampleCounter {
  RangeSample RangeCounter;
  BranchSample BranchCounter;

  void recordRangeCount(uint64_t Start, uint64_t End, uint64_t Repeat) {
    RangeCounter[{Start, End}] += Repeat;
  }
  void recordBranchCount(uint64_t Source, uint64_t Target, uint64_t Repeat) {
    BranchCounter[{Source, Target}] += Repeat;
  }
};

// Sample counter with context to support context-sensitive profile
using ContextSampleCounterMap =
    std::unordered_map<Hashable<ContextKey>, SampleCounter,
                       Hashable<ContextKey>::Hash, Hashable<ContextKey>::Equal>;

/*
As in hybrid sample we have a group of LBRs and the most recent sampling call
stack, we can walk through those LBRs to infer more call stacks which would be
used as context for profile. VirtualUnwinder is the class to do the call stack
unwinding based on LBR state. Two types of unwinding are processd here:
1) LBR unwinding and 2) linear range unwinding.
Specifically, for each LBR entry(can be classified into call, return, regular
branch), LBR unwinding will replay the operation by pushing, popping or
switching leaf frame towards the call stack and since the initial call stack
is most recently sampled, the replay should be in anti-execution order, i.e. for
the regular case, pop the call stack when LBR is call, push frame on call stack
when LBR is return. After each LBR processed, it also needs to align with the
next LBR by going through instructions from previous LBR's target to current
LBR's source, which is the linear unwinding. As instruction from linear range
can come from different function by inlining, linear unwinding will do the range
splitting and record counters by the range with same inline context. Over those
unwinding process we will record each call stack as context id and LBR/linear
range as sample counter for further CS profile generation.
*/
class VirtualUnwinder {
public:
  VirtualUnwinder(ContextSampleCounterMap *Counter) : CtxCounterMap(Counter) {}

  bool isCallState(UnwindState &State) const {
    // The tail call frame is always missing here in stack sample, we will
    // use a specific tail call tracker to infer it.
    return State.getBinary()->addressIsCall(State.getCurrentLBRSource());
  }

  bool isReturnState(UnwindState &State) const {
    // Simply check addressIsReturn, as ret is always reliable, both for
    // regular call and tail call.
    return State.getBinary()->addressIsReturn(State.getCurrentLBRSource());
  }

  void unwindCall(UnwindState &State);
  void unwindLinear(UnwindState &State, uint64_t Repeat);
  void unwindReturn(UnwindState &State);
  void unwindBranchWithinFrame(UnwindState &State);
  bool unwind(const HybridSample *Sample, uint64_t Repeat);
  void recordRangeCount(uint64_t Start, uint64_t End, UnwindState &State,
                        uint64_t Repeat);
  void recordBranchCount(const LBREntry &Branch, UnwindState &State,
                         uint64_t Repeat);
  SampleCounter &getOrCreateCounter(const ProfiledBinary *Binary,
                                    std::list<uint64_t> &CallStack);
  // Use pseudo probe based context key to get the sample counter
  // A context stands for a call path from 'main' to an uninlined
  // callee with all inline frames recovered on that path. The probes
  // belonging to that call path is the probes either originated from
  // the callee or from any functions inlined into the callee. Since
  // pseudo probes are organized in a tri-tree style after decoded,
  // the tree path from the tri-tree root (which is the uninlined
  // callee) to the probe node forms an inline context.
  // Here we use a list of probe(pointer) as the context key to speed up
  // aggregation and the final context string will be generate in
  // ProfileGenerator
  SampleCounter &getOrCreateCounterForProbe(const ProfiledBinary *Binary,
                                            std::list<uint64_t> &CallStack);

private:
  ContextSampleCounterMap *CtxCounterMap;
};

// Filename to binary map
using BinaryMap = StringMap<ProfiledBinary>;
// Address to binary map for fast look-up
using AddressBinaryMap = std::map<uint64_t, ProfiledBinary *>;
// Binary to ContextSampleCounters Map to support multiple binary, we may have
// same binary loaded at different addresses, they should share the same sample
// counter
using BinarySampleCounterMap =
    std::unordered_map<ProfiledBinary *, ContextSampleCounterMap>;

// Load binaries and read perf trace to parse the events and samples
class PerfReader {

public:
  PerfReader(cl::list<std::string> &BinaryFilenames);

  // Hybrid sample(call stack + LBRs) profile traces are seprated by double line
  // break, search for that within the first 4k charactors to avoid going
  // through the whole file.
  static bool isHybridPerfScript(StringRef FileName) {
    auto BufOrError = MemoryBuffer::getFileOrSTDIN(FileName, 4000);
    if (!BufOrError)
      exitWithError(BufOrError.getError(), FileName);
    auto Buffer = std::move(BufOrError.get());
    if (Buffer->getBuffer().find("\n\n") == StringRef::npos)
      return false;
    return true;
  }

  // The parsed MMap event
  struct MMapEvent {
    uint64_t PID = 0;
    uint64_t BaseAddress = 0;
    uint64_t Size = 0;
    uint64_t Offset = 0;
    StringRef BinaryPath;
  };

  /// Load symbols and disassemble the code of a give binary.
  /// Also register the binary in the binary table.
  ///
  ProfiledBinary &loadBinary(const StringRef BinaryPath,
                             bool AllowNameConflict = true);
  void updateBinaryAddress(const MMapEvent &Event);
  PerfScriptType getPerfScriptType() const { return PerfType; }
  // Entry of the reader to parse multiple perf traces
  void parsePerfTraces(cl::list<std::string> &PerfTraceFilenames);
  const BinarySampleCounterMap &getBinarySampleCounters() const {
    return BinarySampleCounters;
  }

private:
  /// Parse a single line of a PERF_RECORD_MMAP2 event looking for a
  /// mapping between the binary name and its memory layout.
  ///
  void parseMMap2Event(TraceStream &TraceIt);
  // Parse perf events/samples and do aggregation
  void parseAndAggregateTrace(StringRef Filename);
  // Parse either an MMAP event or a perf sample
  void parseEventOrSample(TraceStream &TraceIt);
  // Parse the hybrid sample including the call and LBR line
  void parseHybridSample(TraceStream &TraceIt);
  // Extract call stack from the perf trace lines
  bool extractCallstack(TraceStream &TraceIt, std::list<uint64_t> &CallStack);
  // Extract LBR stack from one perf trace line
  bool extractLBRStack(TraceStream &TraceIt,
                       SmallVector<LBREntry, 16> &LBRStack,
                       ProfiledBinary *Binary);
  void checkAndSetPerfType(cl::list<std::string> &PerfTraceFilenames);
  // Post process the profile after trace aggregation, we will do simple range
  // overlap computation for AutoFDO, or unwind for CSSPGO(hybrid sample).
  void generateRawProfile();
  // Unwind the hybrid samples after aggregration
  void unwindSamples();
  void printUnwinderOutput();
  // Helper function for looking up binary in AddressBinaryMap
  ProfiledBinary *getBinary(uint64_t Address);

  BinaryMap BinaryTable;
  AddressBinaryMap AddrToBinaryMap; // Used by address-based lookup.

private:
  BinarySampleCounterMap BinarySampleCounters;
  // Samples with the repeating time generated by the perf reader
  AggregatedCounter AggregatedSamples;
  PerfScriptType PerfType;
};

} // end namespace sampleprof
} // end namespace llvm

#endif
