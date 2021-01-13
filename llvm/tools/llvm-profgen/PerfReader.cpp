//===-- PerfReader.cpp - perfscript reader  ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "PerfReader.h"

static cl::opt<bool> ShowMmapEvents("show-mmap-events", cl::ReallyHidden,
                                    cl::init(false), cl::ZeroOrMore,
                                    cl::desc("Print binary load events."));

static cl::opt<bool> ShowUnwinderOutput("show-unwinder-output",
                                        cl::ReallyHidden, cl::init(false),
                                        cl::ZeroOrMore,
                                        cl::desc("Print unwinder output"));

namespace llvm {
namespace sampleprof {

void VirtualUnwinder::unwindCall(UnwindState &State) {
  // The 2nd frame after leaf could be missing if stack sample is
  // taken when IP is within prolog/epilog, as frame chain isn't
  // setup yet. Fill in the missing frame in that case.
  // TODO: Currently we just assume all the addr that can't match the
  // 2nd frame is in prolog/epilog. In the future, we will switch to
  // pro/epi tracker(Dwarf CFI) for the precise check.
  uint64_t Source = State.getCurrentLBRSource();
  auto Iter = State.CallStack.begin();
  if (State.CallStack.size() == 1 || *(++Iter) != Source) {
    State.CallStack.front() = Source;
  } else {
    State.CallStack.pop_front();
  }
  State.InstPtr.update(Source);
}

void VirtualUnwinder::unwindLinear(UnwindState &State, uint64_t Repeat) {
  InstructionPointer &IP = State.InstPtr;
  uint64_t Target = State.getCurrentLBRTarget();
  uint64_t End = IP.Address;
  if (State.getBinary()->usePseudoProbes()) {
    // The outcome of the virtual unwinding with pseudo probes is a
    // map from a context key to the address range being unwound.
    // This means basically linear unwinding is not needed for pseudo
    // probes. The range will be simply recorded here and will be
    // converted to a list of pseudo probes to report in ProfileGenerator.
    recordRangeCount(Target, End, State, Repeat);
  } else {
    // Unwind linear execution part
    while (IP.Address >= Target) {
      uint64_t PrevIP = IP.Address;
      IP.backward();
      // Break into segments for implicit call/return due to inlining
      bool SameInlinee =
          State.getBinary()->inlineContextEqual(PrevIP, IP.Address);
      if (!SameInlinee || PrevIP == Target) {
        recordRangeCount(PrevIP, End, State, Repeat);
        End = IP.Address;
      }
      State.CallStack.front() = IP.Address;
    }
  }
}

void VirtualUnwinder::unwindReturn(UnwindState &State) {
  // Add extra frame as we unwind through the return
  const LBREntry &LBR = State.getCurrentLBR();
  uint64_t CallAddr = State.getBinary()->getCallAddrFromFrameAddr(LBR.Target);
  State.CallStack.front() = CallAddr;
  State.CallStack.push_front(LBR.Source);
  State.InstPtr.update(LBR.Source);
}

void VirtualUnwinder::unwindBranchWithinFrame(UnwindState &State) {
  // TODO: Tolerate tail call for now, as we may see tail call from libraries.
  // This is only for intra function branches, excluding tail calls.
  uint64_t Source = State.getCurrentLBRSource();
  State.CallStack.front() = Source;
  State.InstPtr.update(Source);
}

SampleCounter &
VirtualUnwinder::getOrCreateCounter(const ProfiledBinary *Binary,
                                    std::list<uint64_t> &CallStack) {
  if (Binary->usePseudoProbes()) {
    return getOrCreateCounterForProbe(Binary, CallStack);
  }
  std::shared_ptr<StringBasedCtxKey> KeyStr =
      std::make_shared<StringBasedCtxKey>();
  KeyStr->Context = Binary->getExpandedContextStr(CallStack);
  KeyStr->genHashCode();
  auto Ret =
      CtxCounterMap->emplace(Hashable<ContextKey>(KeyStr), SampleCounter());
  return Ret.first->second;
}

SampleCounter &
VirtualUnwinder::getOrCreateCounterForProbe(const ProfiledBinary *Binary,
                                            std::list<uint64_t> &CallStack) {
  std::shared_ptr<ProbeBasedCtxKey> ProbeBasedKey =
      std::make_shared<ProbeBasedCtxKey>();
  if (CallStack.size() > 1) {
    // We don't need to top frame probe since it should be extracted
    // from the range.
    // The top of stack is an instruction from the function where
    // the LBR address range physcially resides. Strip it since
    // the function is not a part of the call context. We also
    // don't need its inline context since the probes being unwound
    // come with an inline context all the way back to the uninlined
    // function in their prefix tree.
    auto Iter = CallStack.rbegin();
    auto EndT = std::prev(CallStack.rend());
    for (; Iter != EndT; Iter++) {
      uint64_t Address = *Iter;
      const PseudoProbe *CallProbe = Binary->getCallProbeForAddr(Address);
      // We may not find a probe for a merged or external callsite.
      // Callsite merging may cause the loss of original probe IDs.
      // Cutting off the context from here since the inline will
      // not know how to consume a context with unknown callsites.
      if (!CallProbe)
        break;
      ProbeBasedKey->Probes.emplace_back(CallProbe);
    }
  }
  ProbeBasedKey->genHashCode();
  Hashable<ContextKey> ContextId(ProbeBasedKey);
  auto Ret = CtxCounterMap->emplace(ContextId, SampleCounter());
  return Ret.first->second;
}

void VirtualUnwinder::recordRangeCount(uint64_t Start, uint64_t End,
                                       UnwindState &State, uint64_t Repeat) {
  uint64_t StartOffset = State.getBinary()->virtualAddrToOffset(Start);
  uint64_t EndOffset = State.getBinary()->virtualAddrToOffset(End);
  SampleCounter &SCounter =
      getOrCreateCounter(State.getBinary(), State.CallStack);
  SCounter.recordRangeCount(StartOffset, EndOffset, Repeat);
}

void VirtualUnwinder::recordBranchCount(const LBREntry &Branch,
                                        UnwindState &State, uint64_t Repeat) {
  if (Branch.IsArtificial)
    return;
  uint64_t SourceOffset = State.getBinary()->virtualAddrToOffset(Branch.Source);
  uint64_t TargetOffset = State.getBinary()->virtualAddrToOffset(Branch.Target);
  SampleCounter &SCounter =
      getOrCreateCounter(State.getBinary(), State.CallStack);
  SCounter.recordBranchCount(SourceOffset, TargetOffset, Repeat);
}

bool VirtualUnwinder::unwind(const HybridSample *Sample, uint64_t Repeat) {
  // Capture initial state as starting point for unwinding.
  UnwindState State(Sample);

  // Sanity check - making sure leaf of LBR aligns with leaf of stack sample
  // Stack sample sometimes can be unreliable, so filter out bogus ones.
  if (!State.validateInitialState())
    return false;

  // Also do not attempt linear unwind for the leaf range as it's incomplete.
  bool IsLeaf = true;

  // Now process the LBR samples in parrallel with stack sample
  // Note that we do not reverse the LBR entry order so we can
  // unwind the sample stack as we walk through LBR entries.
  while (State.hasNextLBR()) {
    State.checkStateConsistency();

    // Unwind implicit calls/returns from inlining, along the linear path,
    // break into smaller sub section each with its own calling context.
    if (!IsLeaf) {
      unwindLinear(State, Repeat);
    }
    IsLeaf = false;

    // Save the LBR branch before it gets unwound.
    const LBREntry &Branch = State.getCurrentLBR();

    if (isCallState(State)) {
      // Unwind calls - we know we encountered call if LBR overlaps with
      // transition between leaf the 2nd frame. Note that for calls that
      // were not in the original stack sample, we should have added the
      // extra frame when processing the return paired with this call.
      unwindCall(State);
    } else if (isReturnState(State)) {
      // Unwind returns - check whether the IP is indeed at a return instruction
      unwindReturn(State);
    } else {
      // Unwind branches - for regular intra function branches, we only
      // need to record branch with context.
      unwindBranchWithinFrame(State);
    }
    State.advanceLBR();
    // Record `branch` with calling context after unwinding.
    recordBranchCount(Branch, State, Repeat);
  }

  return true;
}

PerfReader::PerfReader(cl::list<std::string> &BinaryFilenames) {
  // Load the binaries.
  for (auto Filename : BinaryFilenames)
    loadBinary(Filename, /*AllowNameConflict*/ false);
}

ProfiledBinary &PerfReader::loadBinary(const StringRef BinaryPath,
                                       bool AllowNameConflict) {
  // The binary table is currently indexed by the binary name not the full
  // binary path. This is because the user-given path may not match the one
  // that was actually executed.
  StringRef BinaryName = llvm::sys::path::filename(BinaryPath);

  // Call to load the binary in the ctor of ProfiledBinary.
  auto Ret = BinaryTable.insert({BinaryName, ProfiledBinary(BinaryPath)});

  if (!Ret.second && !AllowNameConflict) {
    std::string ErrorMsg = "Binary name conflict: " + BinaryPath.str() +
                           " and " + Ret.first->second.getPath().str() + " \n";
    exitWithError(ErrorMsg);
  }

  return Ret.first->second;
}

void PerfReader::updateBinaryAddress(const MMapEvent &Event) {
  // Load the binary.
  StringRef BinaryPath = Event.BinaryPath;
  StringRef BinaryName = llvm::sys::path::filename(BinaryPath);

  auto I = BinaryTable.find(BinaryName);
  // Drop the event which doesn't belong to user-provided binaries
  // or if its image is loaded at the same address
  if (I == BinaryTable.end() || Event.BaseAddress == I->second.getBaseAddress())
    return;

  ProfiledBinary &Binary = I->second;

  // A binary image could be uploaded and then reloaded at different
  // place, so update the address map here
  AddrToBinaryMap.erase(Binary.getBaseAddress());
  AddrToBinaryMap[Event.BaseAddress] = &Binary;

  // Update binary load address.
  Binary.setBaseAddress(Event.BaseAddress);
}

ProfiledBinary *PerfReader::getBinary(uint64_t Address) {
  auto Iter = AddrToBinaryMap.lower_bound(Address);
  if (Iter == AddrToBinaryMap.end() || Iter->first != Address) {
    if (Iter == AddrToBinaryMap.begin())
      return nullptr;
    Iter--;
  }
  return Iter->second;
}

// Use ordered map to make the output deterministic
using OrderedCounterForPrint = std::map<std::string, RangeSample>;

static void printSampleCounter(OrderedCounterForPrint &OrderedCounter) {
  for (auto Range : OrderedCounter) {
    outs() << Range.first << "\n";
    for (auto I : Range.second) {
      outs() << "  (" << format("%" PRIx64, I.first.first) << ", "
             << format("%" PRIx64, I.first.second) << "): " << I.second << "\n";
    }
  }
}

static std::string getContextKeyStr(ContextKey *K,
                                    const ProfiledBinary *Binary) {
  std::string ContextStr;
  if (const auto *CtxKey = dyn_cast<StringBasedCtxKey>(K)) {
    return CtxKey->Context;
  } else if (const auto *CtxKey = dyn_cast<ProbeBasedCtxKey>(K)) {
    SmallVector<std::string, 16> ContextStack;
    for (const auto *Probe : CtxKey->Probes) {
      Binary->getInlineContextForProbe(Probe, ContextStack, true);
    }
    for (const auto &Context : ContextStack) {
      if (ContextStr.size())
        ContextStr += " @ ";
      ContextStr += Context;
    }
  }
  return ContextStr;
}

static void printRangeCounter(ContextSampleCounterMap &Counter,
                              const ProfiledBinary *Binary) {
  OrderedCounterForPrint OrderedCounter;
  for (auto &CI : Counter) {
    OrderedCounter[getContextKeyStr(CI.first.getPtr(), Binary)] =
        CI.second.RangeCounter;
  }
  printSampleCounter(OrderedCounter);
}

static void printBranchCounter(ContextSampleCounterMap &Counter,
                               const ProfiledBinary *Binary) {
  OrderedCounterForPrint OrderedCounter;
  for (auto &CI : Counter) {
    OrderedCounter[getContextKeyStr(CI.first.getPtr(), Binary)] =
        CI.second.BranchCounter;
  }
  printSampleCounter(OrderedCounter);
}

void PerfReader::printUnwinderOutput() {
  for (auto I : BinarySampleCounters) {
    const ProfiledBinary *Binary = I.first;
    outs() << "Binary(" << Binary->getName().str() << ")'s Range Counter:\n";
    printRangeCounter(I.second, Binary);
    outs() << "\nBinary(" << Binary->getName().str() << ")'s Branch Counter:\n";
    printBranchCounter(I.second, Binary);
  }
}

void PerfReader::unwindSamples() {
  for (const auto &Item : AggregatedSamples) {
    const HybridSample *Sample = dyn_cast<HybridSample>(Item.first.getPtr());
    VirtualUnwinder Unwinder(&BinarySampleCounters[Sample->Binary]);
    Unwinder.unwind(Sample, Item.second);
  }

  if (ShowUnwinderOutput)
    printUnwinderOutput();
}

bool PerfReader::extractLBRStack(TraceStream &TraceIt,
                                 SmallVector<LBREntry, 16> &LBRStack,
                                 ProfiledBinary *Binary) {
  // The raw format of LBR stack is like:
  // 0x4005c8/0x4005dc/P/-/-/0 0x40062f/0x4005b0/P/-/-/0 ...
  //                           ... 0x4005c8/0x4005dc/P/-/-/0
  // It's in FIFO order and seperated by whitespace.
  SmallVector<StringRef, 32> Records;
  TraceIt.getCurrentLine().split(Records, " ");

  // Extract leading instruction pointer if present, use single
  // list to pass out as reference.
  size_t Index = 0;
  if (!Records.empty() && Records[0].find('/') == StringRef::npos) {
    Index = 1;
  }
  // Now extract LBR samples - note that we do not reverse the
  // LBR entry order so we can unwind the sample stack as we walk
  // through LBR entries.
  uint64_t PrevTrDst = 0;

  while (Index < Records.size()) {
    auto &Token = Records[Index++];
    if (Token.size() == 0)
      continue;

    SmallVector<StringRef, 8> Addresses;
    Token.split(Addresses, "/");
    uint64_t Src;
    uint64_t Dst;
    Addresses[0].substr(2).getAsInteger(16, Src);
    Addresses[1].substr(2).getAsInteger(16, Dst);

    bool SrcIsInternal = Binary->addressIsCode(Src);
    bool DstIsInternal = Binary->addressIsCode(Dst);
    bool IsArtificial = false;
    // Ignore branches outside the current binary.
    if (!SrcIsInternal && !DstIsInternal)
      continue;
    if (!SrcIsInternal && DstIsInternal) {
      // For transition from external code (such as dynamic libraries) to
      // the current binary, keep track of the branch target which will be
      // grouped with the Source of the last transition from the current
      // binary.
      PrevTrDst = Dst;
      continue;
    }
    if (SrcIsInternal && !DstIsInternal) {
      // For transition to external code, group the Source with the next
      // availabe transition target.
      if (!PrevTrDst)
        continue;
      Dst = PrevTrDst;
      PrevTrDst = 0;
      IsArtificial = true;
    }
    // TODO: filter out buggy duplicate branches on Skylake

    LBRStack.emplace_back(LBREntry(Src, Dst, IsArtificial));
  }
  TraceIt.advance();
  return !LBRStack.empty();
}

bool PerfReader::extractCallstack(TraceStream &TraceIt,
                                  std::list<uint64_t> &CallStack) {
  // The raw format of call stack is like:
  //            4005dc      # leaf frame
  //	          400634
  //	          400684      # root frame
  // It's in bottom-up order with each frame in one line.

  // Extract stack frames from sample
  ProfiledBinary *Binary = nullptr;
  while (!TraceIt.isAtEoF() && !TraceIt.getCurrentLine().startswith(" 0x")) {
    StringRef FrameStr = TraceIt.getCurrentLine().ltrim();
    // We might get an empty line at the beginning or comments, skip it
    uint64_t FrameAddr = 0;
    if (FrameStr.getAsInteger(16, FrameAddr)) {
      TraceIt.advance();
      break;
    }
    TraceIt.advance();
    if (!Binary) {
      Binary = getBinary(FrameAddr);
      // we might have addr not match the MMAP, skip it
      if (!Binary) {
        if (AddrToBinaryMap.size() == 0)
          WithColor::warning() << "No MMAP event in the perfscript, create it "
                                  "with '--show-mmap-events'\n";
        break;
      }
    }
    // Currently intermixed frame from different binaries is not supported.
    // Ignore bottom frames not from binary of interest.
    if (!Binary->addressIsCode(FrameAddr))
      break;

    // We need to translate return address to call address
    // for non-leaf frames
    if (!CallStack.empty()) {
      FrameAddr = Binary->getCallAddrFromFrameAddr(FrameAddr);
    }

    CallStack.emplace_back(FrameAddr);
  }

  if (CallStack.empty())
    return false;
  // Skip other unrelated line, find the next valid LBR line
  while (!TraceIt.isAtEoF() && !TraceIt.getCurrentLine().startswith(" 0x")) {
    TraceIt.advance();
  }
  // Filter out broken stack sample. We may not have complete frame info
  // if sample end up in prolog/epilog, the result is dangling context not
  // connected to entry point. This should be relatively rare thus not much
  // impact on overall profile quality. However we do want to filter them
  // out to reduce the number of different calling contexts. One instance
  // of such case - when sample landed in prolog/epilog, somehow stack
  // walking will be broken in an unexpected way that higher frames will be
  // missing.
  return !Binary->addressInPrologEpilog(CallStack.front());
}

void PerfReader::parseHybridSample(TraceStream &TraceIt) {
  // The raw hybird sample started with call stack in FILO order and followed
  // intermediately by LBR sample
  // e.g.
  // 	          4005dc    # call stack leaf
  //	          400634
  //	          400684    # call stack root
  // 0x4005c8/0x4005dc/P/-/-/0   0x40062f/0x4005b0/P/-/-/0 ...
  //          ... 0x4005c8/0x4005dc/P/-/-/0    # LBR Entries
  //
  std::shared_ptr<HybridSample> Sample = std::make_shared<HybridSample>();

  // Parsing call stack and populate into HybridSample.CallStack
  if (!extractCallstack(TraceIt, Sample->CallStack)) {
    // Skip the next LBR line matched current call stack
    if (!TraceIt.isAtEoF() && TraceIt.getCurrentLine().startswith(" 0x"))
      TraceIt.advance();
    return;
  }
  // Set the binary current sample belongs to
  Sample->Binary = getBinary(Sample->CallStack.front());

  if (!TraceIt.isAtEoF() && TraceIt.getCurrentLine().startswith(" 0x")) {
    // Parsing LBR stack and populate into HybridSample.LBRStack
    if (extractLBRStack(TraceIt, Sample->LBRStack, Sample->Binary)) {
      // Canonicalize stack leaf to avoid 'random' IP from leaf frame skew LBR
      // ranges
      Sample->CallStack.front() = Sample->LBRStack[0].Target;
      // Record samples by aggregation
      Sample->genHashCode();
      AggregatedSamples[Hashable<PerfSample>(Sample)]++;
    }
  } else {
    // LBR sample is encoded in single line after stack sample
    exitWithError("'Hybrid perf sample is corrupted, No LBR sample line");
  }
}

void PerfReader::parseMMap2Event(TraceStream &TraceIt) {
  // Parse a line like:
  //  PERF_RECORD_MMAP2 2113428/2113428: [0x7fd4efb57000(0x204000) @ 0
  //  08:04 19532229 3585508847]: r-xp /usr/lib64/libdl-2.17.so
  constexpr static const char *const Pattern =
      "PERF_RECORD_MMAP2 ([0-9]+)/[0-9]+: "
      "\\[(0x[a-f0-9]+)\\((0x[a-f0-9]+)\\) @ "
      "(0x[a-f0-9]+|0) .*\\]: [-a-z]+ (.*)";
  // Field 0 - whole line
  // Field 1 - PID
  // Field 2 - base address
  // Field 3 - mmapped size
  // Field 4 - page offset
  // Field 5 - binary path
  enum EventIndex {
    WHOLE_LINE = 0,
    PID = 1,
    BASE_ADDRESS = 2,
    MMAPPED_SIZE = 3,
    PAGE_OFFSET = 4,
    BINARY_PATH = 5
  };

  Regex RegMmap2(Pattern);
  SmallVector<StringRef, 6> Fields;
  bool R = RegMmap2.match(TraceIt.getCurrentLine(), &Fields);
  if (!R) {
    std::string ErrorMsg = "Cannot parse mmap event: Line" +
                           Twine(TraceIt.getLineNumber()).str() + ": " +
                           TraceIt.getCurrentLine().str() + " \n";
    exitWithError(ErrorMsg);
  }
  MMapEvent Event;
  Fields[PID].getAsInteger(10, Event.PID);
  Fields[BASE_ADDRESS].getAsInteger(0, Event.BaseAddress);
  Fields[MMAPPED_SIZE].getAsInteger(0, Event.Size);
  Fields[PAGE_OFFSET].getAsInteger(0, Event.Offset);
  Event.BinaryPath = Fields[BINARY_PATH];
  updateBinaryAddress(Event);
  if (ShowMmapEvents) {
    outs() << "Mmap: Binary " << Event.BinaryPath << " loaded at "
           << format("0x%" PRIx64 ":", Event.BaseAddress) << " \n";
  }
  TraceIt.advance();
}

void PerfReader::parseEventOrSample(TraceStream &TraceIt) {
  if (TraceIt.getCurrentLine().startswith("PERF_RECORD_MMAP2"))
    parseMMap2Event(TraceIt);
  else if (getPerfScriptType() == PERF_LBR_STACK)
    parseHybridSample(TraceIt);
  else {
    // TODO: parse other type sample
    TraceIt.advance();
  }
}

void PerfReader::parseAndAggregateTrace(StringRef Filename) {
  // Trace line iterator
  TraceStream TraceIt(Filename);
  while (!TraceIt.isAtEoF())
    parseEventOrSample(TraceIt);
}

void PerfReader::checkAndSetPerfType(
    cl::list<std::string> &PerfTraceFilenames) {
  bool HasHybridPerf = true;
  for (auto FileName : PerfTraceFilenames) {
    if (!isHybridPerfScript(FileName)) {
      HasHybridPerf = false;
      break;
    }
  }

  if (HasHybridPerf) {
    // Set up ProfileIsCS to enable context-sensitive functionalities
    // in SampleProf
    FunctionSamples::ProfileIsCS = true;
    PerfType = PERF_LBR_STACK;

  } else {
    // TODO: Support other type of perf script
    PerfType = PERF_INVILID;
  }

  if (BinaryTable.size() > 1) {
    // TODO: remove this if everything is ready to support multiple binaries.
    exitWithError("Currently only support one input binary, multiple binaries' "
                  "profile will be merged in one profile and make profile "
                  "summary info inaccurate. Please use `perfdata` to merge "
                  "profiles from multiple binaries.");
  }
}

void PerfReader::generateRawProfile() {
  if (getPerfScriptType() == PERF_LBR_STACK) {
    // Unwind samples if it's hybird sample
    unwindSamples();
  } else if (getPerfScriptType() == PERF_LBR) {
    // TODO: range overlap computation for regular AutoFDO
  }
}

void PerfReader::parsePerfTraces(cl::list<std::string> &PerfTraceFilenames) {
  // Check and set current perfscript type
  checkAndSetPerfType(PerfTraceFilenames);
  // Parse perf traces and do aggregation.
  for (auto Filename : PerfTraceFilenames)
    parseAndAggregateTrace(Filename);

  generateRawProfile();
}

} // end namespace sampleprof
} // end namespace llvm
