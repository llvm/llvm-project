//===-- PerfReader.cpp - perfscript reader  ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "PerfReader.h"
#include "ProfileGenerator.h"
#include "llvm/Support/FileSystem.h"

static cl::opt<bool> ShowMmapEvents("show-mmap-events", cl::ReallyHidden,
                                    cl::init(false), cl::ZeroOrMore,
                                    cl::desc("Print binary load events."));

static cl::opt<bool> ShowUnwinderOutput("show-unwinder-output",
                                        cl::ReallyHidden, cl::init(false),
                                        cl::ZeroOrMore,
                                        cl::desc("Print unwinder output"));

extern cl::opt<bool> ShowDisassemblyOnly;
extern cl::opt<bool> ShowSourceLocations;

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
  auto *ParentFrame = State.getParentFrame();
  if (ParentFrame == State.getDummyRootPtr() ||
      ParentFrame->Address != Source) {
    State.switchToFrame(Source);
  } else {
    State.popFrame();
  }
  State.InstPtr.update(Source);
}

void VirtualUnwinder::unwindLinear(UnwindState &State, uint64_t Repeat) {
  InstructionPointer &IP = State.InstPtr;
  uint64_t Target = State.getCurrentLBRTarget();
  uint64_t End = IP.Address;
  if (Binary->usePseudoProbes()) {
    // We don't need to top frame probe since it should be extracted
    // from the range.
    // The outcome of the virtual unwinding with pseudo probes is a
    // map from a context key to the address range being unwound.
    // This means basically linear unwinding is not needed for pseudo
    // probes. The range will be simply recorded here and will be
    // converted to a list of pseudo probes to report in ProfileGenerator.
    State.getParentFrame()->recordRangeCount(Target, End, Repeat);
  } else {
    // Unwind linear execution part
    uint64_t LeafAddr = State.CurrentLeafFrame->Address;
    while (IP.Address >= Target) {
      uint64_t PrevIP = IP.Address;
      IP.backward();
      // Break into segments for implicit call/return due to inlining
      bool SameInlinee = Binary->inlineContextEqual(PrevIP, IP.Address);
      if (!SameInlinee || PrevIP == Target) {
        State.switchToFrame(LeafAddr);
        State.CurrentLeafFrame->recordRangeCount(PrevIP, End, Repeat);
        End = IP.Address;
      }
      LeafAddr = IP.Address;
    }
  }
}

void VirtualUnwinder::unwindReturn(UnwindState &State) {
  // Add extra frame as we unwind through the return
  const LBREntry &LBR = State.getCurrentLBR();
  uint64_t CallAddr = Binary->getCallAddrFromFrameAddr(LBR.Target);
  State.switchToFrame(CallAddr);
  State.pushFrame(LBR.Source);
  State.InstPtr.update(LBR.Source);
}

void VirtualUnwinder::unwindBranchWithinFrame(UnwindState &State) {
  // TODO: Tolerate tail call for now, as we may see tail call from libraries.
  // This is only for intra function branches, excluding tail calls.
  uint64_t Source = State.getCurrentLBRSource();
  State.switchToFrame(Source);
  State.InstPtr.update(Source);
}

std::shared_ptr<StringBasedCtxKey> FrameStack::getContextKey() {
  std::shared_ptr<StringBasedCtxKey> KeyStr =
      std::make_shared<StringBasedCtxKey>();
  KeyStr->Context =
      Binary->getExpandedContextStr(Stack, KeyStr->WasLeafInlined);
  if (KeyStr->Context.empty())
    return nullptr;
  KeyStr->genHashCode();
  return KeyStr;
}

std::shared_ptr<ProbeBasedCtxKey> ProbeStack::getContextKey() {
  std::shared_ptr<ProbeBasedCtxKey> ProbeBasedKey =
      std::make_shared<ProbeBasedCtxKey>();
  for (auto CallProbe : Stack) {
    ProbeBasedKey->Probes.emplace_back(CallProbe);
  }
  CSProfileGenerator::compressRecursionContext<const MCDecodedPseudoProbe *>(
      ProbeBasedKey->Probes);
  CSProfileGenerator::trimContext<const MCDecodedPseudoProbe *>(
      ProbeBasedKey->Probes);

  ProbeBasedKey->genHashCode();
  return ProbeBasedKey;
}

template <typename T>
void VirtualUnwinder::collectSamplesFromFrame(UnwindState::ProfiledFrame *Cur,
                                              T &Stack) {
  if (Cur->RangeSamples.empty() && Cur->BranchSamples.empty())
    return;

  std::shared_ptr<ContextKey> Key = Stack.getContextKey();
  if (Key == nullptr)
    return;
  auto Ret = CtxCounterMap->emplace(Hashable<ContextKey>(Key), SampleCounter());
  SampleCounter &SCounter = Ret.first->second;
  for (auto &Item : Cur->RangeSamples) {
    uint64_t StartOffset = Binary->virtualAddrToOffset(std::get<0>(Item));
    uint64_t EndOffset = Binary->virtualAddrToOffset(std::get<1>(Item));
    SCounter.recordRangeCount(StartOffset, EndOffset, std::get<2>(Item));
  }

  for (auto &Item : Cur->BranchSamples) {
    uint64_t SourceOffset = Binary->virtualAddrToOffset(std::get<0>(Item));
    uint64_t TargetOffset = Binary->virtualAddrToOffset(std::get<1>(Item));
    SCounter.recordBranchCount(SourceOffset, TargetOffset, std::get<2>(Item));
  }
}

template <typename T>
void VirtualUnwinder::collectSamplesFromFrameTrie(
    UnwindState::ProfiledFrame *Cur, T &Stack) {
  if (!Cur->isDummyRoot()) {
    if (!Stack.pushFrame(Cur)) {
      // Process truncated context
      // Start a new traversal ignoring its bottom context
      T EmptyStack(Binary);
      collectSamplesFromFrame(Cur, EmptyStack);
      for (const auto &Item : Cur->Children) {
        collectSamplesFromFrameTrie(Item.second.get(), EmptyStack);
      }
      return;
    }
  }

  collectSamplesFromFrame(Cur, Stack);
  // Process children frame
  for (const auto &Item : Cur->Children) {
    collectSamplesFromFrameTrie(Item.second.get(), Stack);
  }
  // Recover the call stack
  Stack.popFrame();
}

void VirtualUnwinder::collectSamplesFromFrameTrie(
    UnwindState::ProfiledFrame *Cur) {
  if (Binary->usePseudoProbes()) {
    ProbeStack Stack(Binary);
    collectSamplesFromFrameTrie<ProbeStack>(Cur, Stack);
  } else {
    FrameStack Stack(Binary);
    collectSamplesFromFrameTrie<FrameStack>(Cur, Stack);
  }
}

void VirtualUnwinder::recordBranchCount(const LBREntry &Branch,
                                        UnwindState &State, uint64_t Repeat) {
  if (Branch.IsArtificial)
    return;

  if (Binary->usePseudoProbes()) {
    // Same as recordRangeCount, We don't need to top frame probe since we will
    // extract it from branch's source address
    State.getParentFrame()->recordBranchCount(Branch.Source, Branch.Target,
                                              Repeat);
  } else {
    State.CurrentLeafFrame->recordBranchCount(Branch.Source, Branch.Target,
                                              Repeat);
  }
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
  // As samples are aggregated on trie, record them into counter map
  collectSamplesFromFrameTrie(State.getDummyRootPtr());

  return true;
}

void PerfReaderBase::validateCommandLine(
    cl::list<std::string> &BinaryFilenames,
    cl::list<std::string> &PerfTraceFilenames) {
  // Allow the invalid perfscript if we only use to show binary disassembly
  if (!ShowDisassemblyOnly) {
    for (auto &File : PerfTraceFilenames) {
      if (!llvm::sys::fs::exists(File)) {
        std::string Msg = "Input perf script(" + File + ") doesn't exist!";
        exitWithError(Msg);
      }
    }
  }
  if (BinaryFilenames.size() > 1) {
    // TODO: remove this if everything is ready to support multiple binaries.
    exitWithError(
        "Currently only support one input binary, multiple binaries' "
        "profile will be merged in one profile and make profile "
        "summary info inaccurate. Please use `llvm-perfdata` to merge "
        "profiles from multiple binaries.");
  }
  for (auto &Binary : BinaryFilenames) {
    if (!llvm::sys::fs::exists(Binary)) {
      std::string Msg = "Input binary(" + Binary + ") doesn't exist!";
      exitWithError(Msg);
    }
  }
  if (CSProfileGenerator::MaxCompressionSize < -1) {
    exitWithError("Value of --compress-recursion should >= -1");
  }
  if (ShowSourceLocations && !ShowDisassemblyOnly) {
    exitWithError("--show-source-locations should work together with "
                  "--show-disassembly-only!");
  }
}

std::unique_ptr<PerfReaderBase>
PerfReaderBase::create(cl::list<std::string> &BinaryFilenames,
                       cl::list<std::string> &PerfTraceFilenames) {
  validateCommandLine(BinaryFilenames, PerfTraceFilenames);

  PerfScriptType PerfType = extractPerfType(PerfTraceFilenames);
  std::unique_ptr<PerfReaderBase> PerfReader;
  if (PerfType == PERF_LBR_STACK) {
    PerfReader.reset(new HybridPerfReader(BinaryFilenames));
  } else if (PerfType == PERF_LBR) {
    // TODO:
    exitWithError("Unsupported perfscript!");
  } else {
    exitWithError("Unsupported perfscript!");
  }

  return PerfReader;
}

PerfReaderBase::PerfReaderBase(cl::list<std::string> &BinaryFilenames) {
  // Load the binaries.
  for (auto Filename : BinaryFilenames)
    loadBinary(Filename, /*AllowNameConflict*/ false);
}

ProfiledBinary &PerfReaderBase::loadBinary(const StringRef BinaryPath,
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

  // Initialize the base address to preferred address.
  ProfiledBinary &B = Ret.first->second;
  uint64_t PreferredAddr = B.getPreferredBaseAddress();
  AddrToBinaryMap[PreferredAddr] = &B;
  B.setBaseAddress(PreferredAddr);

  return B;
}

void PerfReaderBase::updateBinaryAddress(const MMapEvent &Event) {
  // Load the binary.
  StringRef BinaryPath = Event.BinaryPath;
  StringRef BinaryName = llvm::sys::path::filename(BinaryPath);

  auto I = BinaryTable.find(BinaryName);
  // Drop the event which doesn't belong to user-provided binaries
  if (I == BinaryTable.end())
    return;

  ProfiledBinary &Binary = I->second;
  // Drop the event if its image is loaded at the same address
  if (Event.Address == Binary.getBaseAddress()) {
    Binary.setIsLoadedByMMap(true);
    return;
  }

  if (Event.Offset == Binary.getTextSegmentOffset()) {
    // A binary image could be unloaded and then reloaded at different
    // place, so update the address map here.
    // Only update for the first executable segment and assume all other
    // segments are loaded at consecutive memory addresses, which is the case on
    // X64.
    AddrToBinaryMap.erase(Binary.getBaseAddress());
    AddrToBinaryMap[Event.Address] = &Binary;

    // Update binary load address.
    Binary.setBaseAddress(Event.Address);

    Binary.setIsLoadedByMMap(true);
  } else {
    // Verify segments are loaded consecutively.
    const auto &Offsets = Binary.getTextSegmentOffsets();
    auto It = std::lower_bound(Offsets.begin(), Offsets.end(), Event.Offset);
    if (It != Offsets.end() && *It == Event.Offset) {
      // The event is for loading a separate executable segment.
      auto I = std::distance(Offsets.begin(), It);
      const auto &PreferredAddrs = Binary.getPreferredTextSegmentAddresses();
      if (PreferredAddrs[I] - Binary.getPreferredBaseAddress() !=
          Event.Address - Binary.getBaseAddress())
        exitWithError("Executable segments not loaded consecutively");
    } else {
      if (It == Offsets.begin())
        exitWithError("File offset not found");
      else {
        // Find the segment the event falls in. A large segment could be loaded
        // via multiple mmap calls with consecutive memory addresses.
        --It;
        assert(*It < Event.Offset);
        if (Event.Offset - *It != Event.Address - Binary.getBaseAddress())
          exitWithError("Segment not loaded by consecutive mmaps");
      }
    }
  }
}

ProfiledBinary *PerfReaderBase::getBinary(uint64_t Address) {
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

void HybridPerfReader::printUnwinderOutput() {
  for (auto I : BinarySampleCounters) {
    const ProfiledBinary *Binary = I.first;
    outs() << "Binary(" << Binary->getName().str() << ")'s Range Counter:\n";
    printRangeCounter(I.second, Binary);
    outs() << "\nBinary(" << Binary->getName().str() << ")'s Branch Counter:\n";
    printBranchCounter(I.second, Binary);
  }
}

void HybridPerfReader::unwindSamples() {
  for (const auto &Item : AggregatedSamples) {
    const HybridSample *Sample = dyn_cast<HybridSample>(Item.first.getPtr());
    VirtualUnwinder Unwinder(&BinarySampleCounters[Sample->Binary],
                             Sample->Binary);
    Unwinder.unwind(Sample, Item.second);
  }

  if (ShowUnwinderOutput)
    printUnwinderOutput();
}

bool PerfReaderBase::extractLBRStack(TraceStream &TraceIt,
                                     SmallVectorImpl<LBREntry> &LBRStack,
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
    bool IsExternal = !SrcIsInternal && !DstIsInternal;
    bool IsIncoming = !SrcIsInternal && DstIsInternal;
    bool IsOutgoing = SrcIsInternal && !DstIsInternal;
    bool IsArtificial = false;

    // Ignore branches outside the current binary.
    if (IsExternal)
      continue;

    if (IsOutgoing) {
      if (!PrevTrDst) {
        // This is unpaired outgoing jump which is likely due to interrupt or
        // incomplete LBR trace. Ignore current and subsequent entries since
        // they are likely in different contexts.
        break;
      }

      if (Binary->addressIsReturn(Src)) {
        // In a callback case, a return from internal code, say A, to external
        // runtime can happen. The external runtime can then call back to
        // another internal routine, say B. Making an artificial branch that
        // looks like a return from A to B can confuse the unwinder to treat
        // the instruction before B as the call instruction.
        break;
      }

      // For transition to external code, group the Source with the next
      // availabe transition target.
      Dst = PrevTrDst;
      PrevTrDst = 0;
      IsArtificial = true;
    } else {
      if (PrevTrDst) {
        // If we have seen an incoming transition from external code to internal
        // code, but not a following outgoing transition, the incoming
        // transition is likely due to interrupt which is usually unpaired.
        // Ignore current and subsequent entries since they are likely in
        // different contexts.
        break;
      }

      if (IsIncoming) {
        // For transition from external code (such as dynamic libraries) to
        // the current binary, keep track of the branch target which will be
        // grouped with the Source of the last transition from the current
        // binary.
        PrevTrDst = Dst;
        continue;
      }
    }

    // TODO: filter out buggy duplicate branches on Skylake

    LBRStack.emplace_back(LBREntry(Src, Dst, IsArtificial));
  }
  TraceIt.advance();
  return !LBRStack.empty();
}

bool PerfReaderBase::extractCallstack(TraceStream &TraceIt,
                                      SmallVectorImpl<uint64_t> &CallStack) {
  // The raw format of call stack is like:
  //            4005dc      # leaf frame
  //	          400634
  //	          400684      # root frame
  // It's in bottom-up order with each frame in one line.

  // Extract stack frames from sample
  ProfiledBinary *Binary = nullptr;
  while (!TraceIt.isAtEoF() && !TraceIt.getCurrentLine().startswith(" 0x")) {
    StringRef FrameStr = TraceIt.getCurrentLine().ltrim();
    uint64_t FrameAddr = 0;
    if (FrameStr.getAsInteger(16, FrameAddr)) {
      // We might parse a non-perf sample line like empty line and comments,
      // skip it
      TraceIt.advance();
      return false;
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

  // Skip other unrelated line, find the next valid LBR line
  // Note that even for empty call stack, we should skip the address at the
  // bottom, otherwise the following pass may generate a truncated callstack
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
  return !CallStack.empty() &&
         !Binary->addressInPrologEpilog(CallStack.front());
}

void HybridPerfReader::parseSample(TraceStream &TraceIt, uint64_t Count) {
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
  ProfiledBinary *PB = getBinary(Sample->CallStack.front());
  Sample->Binary = PB;
  if (!PB->getMissingMMapWarned() && !PB->getIsLoadedByMMap()) {
    WithColor::warning() << "No relevant mmap event is matched, will use "
                            "preferred address as the base loading address!\n";
    // Avoid redundant warning, only warn at the first unmatched sample.
    PB->setMissingMMapWarned(true);
  }

  if (!TraceIt.isAtEoF() && TraceIt.getCurrentLine().startswith(" 0x")) {
    // Parsing LBR stack and populate into HybridSample.LBRStack
    if (extractLBRStack(TraceIt, Sample->LBRStack, Sample->Binary)) {
      // Canonicalize stack leaf to avoid 'random' IP from leaf frame skew LBR
      // ranges
      Sample->CallStack.front() = Sample->LBRStack[0].Target;
      // Record samples by aggregation
      Sample->genHashCode();
      AggregatedSamples[Hashable<PerfSample>(Sample)] += Count;
    }
  } else {
    // LBR sample is encoded in single line after stack sample
    exitWithError("'Hybrid perf sample is corrupted, No LBR sample line");
  }
}

uint64_t PerfReaderBase::parseAggregatedCount(TraceStream &TraceIt) {
  // The aggregated count is optional, so do not skip the line and return 1 if
  // it's unmatched
  uint64_t Count = 1;
  if (!TraceIt.getCurrentLine().getAsInteger(10, Count))
    TraceIt.advance();
  return Count;
}

void PerfReaderBase::parseSample(TraceStream &TraceIt) {
  uint64_t Count = parseAggregatedCount(TraceIt);
  assert(Count >= 1 && "Aggregated count should be >= 1!");
  parseSample(TraceIt, Count);
}

void PerfReaderBase::parseMMap2Event(TraceStream &TraceIt) {
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
    MMAPPED_ADDRESS = 2,
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
  Fields[MMAPPED_ADDRESS].getAsInteger(0, Event.Address);
  Fields[MMAPPED_SIZE].getAsInteger(0, Event.Size);
  Fields[PAGE_OFFSET].getAsInteger(0, Event.Offset);
  Event.BinaryPath = Fields[BINARY_PATH];
  updateBinaryAddress(Event);
  if (ShowMmapEvents) {
    outs() << "Mmap: Binary " << Event.BinaryPath << " loaded at "
           << format("0x%" PRIx64 ":", Event.Address) << " \n";
  }
  TraceIt.advance();
}

void PerfReaderBase::parseEventOrSample(TraceStream &TraceIt) {
  if (TraceIt.getCurrentLine().startswith("PERF_RECORD_MMAP2"))
    parseMMap2Event(TraceIt);
  else
    parseSample(TraceIt);
}

void PerfReaderBase::parseAndAggregateTrace(StringRef Filename) {
  // Trace line iterator
  TraceStream TraceIt(Filename);
  while (!TraceIt.isAtEoF())
    parseEventOrSample(TraceIt);
}

PerfScriptType
PerfReaderBase::extractPerfType(cl::list<std::string> &PerfTraceFilenames) {
  PerfScriptType PerfType = PERF_UNKNOWN;
  for (auto FileName : PerfTraceFilenames) {
    PerfScriptType Type = checkPerfScriptType(FileName);
    if (Type == PERF_INVALID)
      exitWithError("Invalid perf script input!");
    if (PerfType != PERF_UNKNOWN && PerfType != Type)
      exitWithError("Inconsistent sample among different perf scripts");
    PerfType = Type;
  }
  return PerfType;
}

void HybridPerfReader::generateRawProfile() { unwindSamples(); }

void PerfReaderBase::parsePerfTraces(
    cl::list<std::string> &PerfTraceFilenames) {
  // Parse perf traces and do aggregation.
  for (auto Filename : PerfTraceFilenames)
    parseAndAggregateTrace(Filename);

  generateRawProfile();
}

} // end namespace sampleprof
} // end namespace llvm
