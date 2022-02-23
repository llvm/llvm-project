//===-- ProfiledBinary.h - Binary decoder -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_PROFGEN_PROFILEDBINARY_H
#define LLVM_TOOLS_LLVM_PROFGEN_PROFILEDBINARY_H

#include "CallContext.h"
#include "ErrorHandling.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/Symbolize/Symbolize.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCPseudoProbe.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/ProfileData/SampleProf.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/Transforms/IPO/SampleContextTracker.h"
#include <list>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

extern cl::opt<bool> EnableCSPreInliner;
extern cl::opt<bool> UseContextCostForPreInliner;

using namespace llvm;
using namespace sampleprof;
using namespace llvm::object;

namespace llvm {
namespace sampleprof {

class ProfiledBinary;

struct InstructionPointer {
  const ProfiledBinary *Binary;
  union {
    // Offset of the executable segment of the binary.
    uint64_t Offset = 0;
    // Also used as address in unwinder
    uint64_t Address;
  };
  // Index to the sorted code address array of the binary.
  uint64_t Index = 0;
  InstructionPointer(const ProfiledBinary *Binary, uint64_t Address,
                     bool RoundToNext = false);
  bool advance();
  bool backward();
  void update(uint64_t Addr);
};

// The special frame addresses.
enum SpecialFrameAddr {
  // Dummy root of frame trie.
  DummyRoot = 0,
  // Represent all the addresses outside of current binary.
  // This's also used to indicate the call stack should be truncated since this
  // isn't a real call context the compiler will see.
  ExternalAddr = 1,
};

using RangesTy = std::vector<std::pair<uint64_t, uint64_t>>;

struct BinaryFunction {
  StringRef FuncName;
  // End of range is an exclusive bound.
  RangesTy Ranges;

  uint64_t getFuncSize() {
    uint64_t Sum = 0;
    for (auto &R : Ranges) {
      Sum += R.second - R.first;
    }
    return Sum;
  }
};

// Info about function range. A function can be split into multiple
// non-continuous ranges, each range corresponds to one FuncRange.
struct FuncRange {
  uint64_t StartOffset;
  // EndOffset is an exclusive bound.
  uint64_t EndOffset;
  // Function the range belongs to
  BinaryFunction *Func;
  // Whether the start offset is the real entry of the function.
  bool IsFuncEntry = false;

  StringRef getFuncName() { return Func->FuncName; }
};

// PrologEpilog offset tracker, used to filter out broken stack samples
// Currently we use a heuristic size (two) to infer prolog and epilog
// based on the start address and return address. In the future,
// we will switch to Dwarf CFI based tracker
struct PrologEpilogTracker {
  // A set of prolog and epilog offsets. Used by virtual unwinding.
  std::unordered_set<uint64_t> PrologEpilogSet;
  ProfiledBinary *Binary;
  PrologEpilogTracker(ProfiledBinary *Bin) : Binary(Bin){};

  // Take the two addresses from the start of function as prolog
  void inferPrologOffsets(std::map<uint64_t, FuncRange> &FuncStartOffsetMap) {
    for (auto I : FuncStartOffsetMap) {
      PrologEpilogSet.insert(I.first);
      InstructionPointer IP(Binary, I.first);
      if (!IP.advance())
        break;
      PrologEpilogSet.insert(IP.Offset);
    }
  }

  // Take the last two addresses before the return address as epilog
  void inferEpilogOffsets(std::unordered_set<uint64_t> &RetAddrs) {
    for (auto Addr : RetAddrs) {
      PrologEpilogSet.insert(Addr);
      InstructionPointer IP(Binary, Addr);
      if (!IP.backward())
        break;
      PrologEpilogSet.insert(IP.Offset);
    }
  }
};

// Track function byte size under different context (outlined version as well as
// various inlined versions). It also provides query support to get function
// size with the best matching context, which is used to help pre-inliner use
// accurate post-optimization size to make decisions.
// TODO: If an inlinee is completely optimized away, ideally we should have zero
// for its context size, currently we would misss such context since it doesn't
// have instructions. To fix this, we need to mark all inlinee with entry probe
// but without instructions as having zero size.
class BinarySizeContextTracker {
public:
  // Add instruction with given size to a context
  void addInstructionForContext(const SampleContextFrameVector &Context,
                                uint32_t InstrSize);

  // Get function size with a specific context. When there's no exact match
  // for the given context, try to retrieve the size of that function from
  // closest matching context.
  uint32_t getFuncSizeForContext(const SampleContext &Context);

  // For inlinees that are full optimized away, we can establish zero size using
  // their remaining probes.
  void trackInlineesOptimizedAway(MCPseudoProbeDecoder &ProbeDecoder);

  using ProbeFrameStack = SmallVector<std::pair<StringRef, uint32_t>>;
  void trackInlineesOptimizedAway(MCPseudoProbeDecoder &ProbeDecoder,
                              MCDecodedPseudoProbeInlineTree &ProbeNode,
                              ProbeFrameStack &Context);

  void dump() { RootContext.dumpTree(); }

private:
  // Root node for context trie tree, node that this is a reverse context trie
  // with callee as parent and caller as child. This way we can traverse from
  // root to find the best/longest matching context if an exact match does not
  // exist. It gives us the best possible estimate for function's post-inline,
  // post-optimization byte size.
  ContextTrieNode RootContext;
};

using OffsetRange = std::pair<uint64_t, uint64_t>;

class ProfiledBinary {
  // Absolute path of the executable binary.
  std::string Path;
  // Path of the debug info binary.
  std::string DebugBinaryPath;
  // Path of symbolizer path which should be pointed to binary with debug info.
  StringRef SymbolizerPath;
  // The target triple.
  Triple TheTriple;
  // The runtime base address that the first executable segment is loaded at.
  uint64_t BaseAddress = 0;
  // The runtime base address that the first loadabe segment is loaded at.
  uint64_t FirstLoadableAddress = 0;
  // The preferred load address of each executable segment.
  std::vector<uint64_t> PreferredTextSegmentAddresses;
  // The file offset of each executable segment.
  std::vector<uint64_t> TextSegmentOffsets;

  // Mutiple MC component info
  std::unique_ptr<const MCRegisterInfo> MRI;
  std::unique_ptr<const MCAsmInfo> AsmInfo;
  std::unique_ptr<const MCSubtargetInfo> STI;
  std::unique_ptr<const MCInstrInfo> MII;
  std::unique_ptr<MCDisassembler> DisAsm;
  std::unique_ptr<const MCInstrAnalysis> MIA;
  std::unique_ptr<MCInstPrinter> IPrinter;
  // A list of text sections sorted by start RVA and size. Used to check
  // if a given RVA is a valid code address.
  std::set<std::pair<uint64_t, uint64_t>> TextSections;

  // A map of mapping function name to BinaryFunction info.
  std::unordered_map<std::string, BinaryFunction> BinaryFunctions;

  // An ordered map of mapping function's start offset to function range
  // relevant info. Currently to determine if the offset of ELF is the start of
  // a real function, we leverage the function range info from DWARF.
  std::map<uint64_t, FuncRange> StartOffset2FuncRangeMap;

  // Offset to context location map. Used to expand the context.
  std::unordered_map<uint64_t, SampleContextFrameVector> Offset2LocStackMap;

  // Offset to instruction size map. Also used for quick offset lookup.
  std::unordered_map<uint64_t, uint64_t> Offset2InstSizeMap;

  // An array of offsets of all instructions sorted in increasing order. The
  // sorting is needed to fast advance to the next forward/backward instruction.
  std::vector<uint64_t> CodeAddrOffsets;
  // A set of call instruction offsets. Used by virtual unwinding.
  std::unordered_set<uint64_t> CallOffsets;
  // A set of return instruction offsets. Used by virtual unwinding.
  std::unordered_set<uint64_t> RetOffsets;
  // A set of branch instruction offsets.
  std::unordered_set<uint64_t> BranchOffsets;

  // Estimate and track function prolog and epilog ranges.
  PrologEpilogTracker ProEpilogTracker;

  // Track function sizes under different context
  BinarySizeContextTracker FuncSizeTracker;

  // The symbolizer used to get inline context for an instruction.
  std::unique_ptr<symbolize::LLVMSymbolizer> Symbolizer;

  // String table owning function name strings created from the symbolizer.
  std::unordered_set<std::string> NameStrings;

  // A collection of functions to print disassembly for.
  StringSet<> DisassembleFunctionSet;

  // Pseudo probe decoder
  MCPseudoProbeDecoder ProbeDecoder;

  // Function name to probe frame map for top-level outlined functions.
  StringMap<MCDecodedPseudoProbeInlineTree *> TopLevelProbeFrameMap;

  bool UsePseudoProbes = false;

  bool UseFSDiscriminator = false;

  // Whether we need to symbolize all instructions to get function context size.
  bool TrackFuncContextSize = false;

  // Indicate if the base loading address is parsed from the mmap event or uses
  // the preferred address
  bool IsLoadedByMMap = false;
  // Use to avoid redundant warning.
  bool MissingMMapWarned = false;

  void setPreferredTextSegmentAddresses(const ELFObjectFileBase *O);

  template <class ELFT>
  void setPreferredTextSegmentAddresses(const ELFFile<ELFT> &Obj, StringRef FileName);

  void decodePseudoProbe(const ELFObjectFileBase *Obj);

  void
  checkUseFSDiscriminator(const ELFObjectFileBase *Obj,
                          std::map<SectionRef, SectionSymbolsTy> &AllSymbols);

  // Set up disassembler and related components.
  void setUpDisassembler(const ELFObjectFileBase *Obj);
  void setupSymbolizer();

  // Load debug info of subprograms from DWARF section.
  void loadSymbolsFromDWARF(ObjectFile &Obj);

  // Load debug info from DWARF unit.
  void loadSymbolsFromDWARFUnit(DWARFUnit &CompilationUnit);

  // A function may be spilt into multiple non-continuous address ranges. We use
  // this to set whether start offset of a function is the real entry of the
  // function and also set false to the non-function label.
  void setIsFuncEntry(uint64_t Offset, StringRef RangeSymName);

  // Warn if no entry range exists in the function.
  void warnNoFuncEntry();

  /// Dissassemble the text section and build various address maps.
  void disassemble(const ELFObjectFileBase *O);

  /// Helper function to dissassemble the symbol and extract info for unwinding
  bool dissassembleSymbol(std::size_t SI, ArrayRef<uint8_t> Bytes,
                          SectionSymbolsTy &Symbols, const SectionRef &Section);
  /// Symbolize a given instruction pointer and return a full call context.
  SampleContextFrameVector symbolize(const InstructionPointer &IP,
                                     bool UseCanonicalFnName = false,
                                     bool UseProbeDiscriminator = false);
  /// Decode the interesting parts of the binary and build internal data
  /// structures. On high level, the parts of interest are:
  ///   1. Text sections, including the main code section and the PLT
  ///   entries that will be used to handle cross-module call transitions.
  ///   2. The .debug_line section, used by Dwarf-based profile generation.
  ///   3. Pseudo probe related sections, used by probe-based profile
  ///   generation.
  void load();

public:
  ProfiledBinary(const StringRef ExeBinPath, const StringRef DebugBinPath)
      : Path(ExeBinPath), DebugBinaryPath(DebugBinPath), ProEpilogTracker(this),
        TrackFuncContextSize(EnableCSPreInliner &&
                             UseContextCostForPreInliner) {
    // Point to executable binary if debug info binary is not specified.
    SymbolizerPath = DebugBinPath.empty() ? ExeBinPath : DebugBinPath;
    setupSymbolizer();
    load();
  }
  uint64_t virtualAddrToOffset(uint64_t VirtualAddress) const {
    return VirtualAddress - BaseAddress;
  }
  uint64_t offsetToVirtualAddr(uint64_t Offset) const {
    return Offset + BaseAddress;
  }
  StringRef getPath() const { return Path; }
  StringRef getName() const { return llvm::sys::path::filename(Path); }
  uint64_t getBaseAddress() const { return BaseAddress; }
  void setBaseAddress(uint64_t Address) { BaseAddress = Address; }

  // Return the preferred load address for the first executable segment.
  uint64_t getPreferredBaseAddress() const { return PreferredTextSegmentAddresses[0]; }
  // Return the preferred load address for the first loadable segment.
  uint64_t getFirstLoadableAddress() const { return FirstLoadableAddress; }
  // Return the file offset for the first executable segment.
  uint64_t getTextSegmentOffset() const { return TextSegmentOffsets[0]; }
  const std::vector<uint64_t> &getPreferredTextSegmentAddresses() const {
    return PreferredTextSegmentAddresses;
  }
  const std::vector<uint64_t> &getTextSegmentOffsets() const {
    return TextSegmentOffsets;
  }

  uint64_t getInstSize(uint64_t Offset) const {
    auto I = Offset2InstSizeMap.find(Offset);
    if (I == Offset2InstSizeMap.end())
      return 0;
    return I->second;
  }

  bool offsetIsCode(uint64_t Offset) const {
    return Offset2InstSizeMap.find(Offset) != Offset2InstSizeMap.end();
  }
  bool addressIsCode(uint64_t Address) const {
    uint64_t Offset = virtualAddrToOffset(Address);
    return offsetIsCode(Offset);
  }
  bool addressIsCall(uint64_t Address) const {
    uint64_t Offset = virtualAddrToOffset(Address);
    return CallOffsets.count(Offset);
  }
  bool addressIsReturn(uint64_t Address) const {
    uint64_t Offset = virtualAddrToOffset(Address);
    return RetOffsets.count(Offset);
  }
  bool addressInPrologEpilog(uint64_t Address) const {
    uint64_t Offset = virtualAddrToOffset(Address);
    return ProEpilogTracker.PrologEpilogSet.count(Offset);
  }

  bool offsetIsTransfer(uint64_t Offset) {
    return BranchOffsets.count(Offset) || RetOffsets.count(Offset) ||
           CallOffsets.count(Offset);
  }

  uint64_t getAddressforIndex(uint64_t Index) const {
    return offsetToVirtualAddr(CodeAddrOffsets[Index]);
  }

  size_t getCodeOffsetsSize() const { return CodeAddrOffsets.size(); }

  bool usePseudoProbes() const { return UsePseudoProbes; }
  bool useFSDiscriminator() const { return UseFSDiscriminator; }
  // Get the index in CodeAddrOffsets for the address
  // As we might get an address which is not the code
  // here it would round to the next valid code address by
  // using lower bound operation
  uint32_t getIndexForOffset(uint64_t Offset) const {
    auto Low = llvm::lower_bound(CodeAddrOffsets, Offset);
    return Low - CodeAddrOffsets.begin();
  }
  uint32_t getIndexForAddr(uint64_t Address) const {
    uint64_t Offset = virtualAddrToOffset(Address);
    return getIndexForOffset(Offset);
  }

  uint64_t getCallAddrFromFrameAddr(uint64_t FrameAddr) const {
    if (FrameAddr == ExternalAddr)
      return ExternalAddr;
    auto I = getIndexForAddr(FrameAddr);
    FrameAddr = I ? getAddressforIndex(I - 1) : 0;
    if (FrameAddr && addressIsCall(FrameAddr))
      return FrameAddr;
    return 0;
  }

  FuncRange *findFuncRangeForStartOffset(uint64_t Offset) {
    auto I = StartOffset2FuncRangeMap.find(Offset);
    if (I == StartOffset2FuncRangeMap.end())
      return nullptr;
    return &I->second;
  }

  // Binary search the function range which includes the input offset.
  FuncRange *findFuncRangeForOffset(uint64_t Offset) {
    auto I = StartOffset2FuncRangeMap.upper_bound(Offset);
    if (I == StartOffset2FuncRangeMap.begin())
      return nullptr;
    I--;

    if (Offset >= I->second.EndOffset)
      return nullptr;

    return &I->second;
  }

  // Get all ranges of one function.
  RangesTy getRangesForOffset(uint64_t Offset) {
    auto *FRange = findFuncRangeForOffset(Offset);
    // Ignore the range which falls into plt section or system lib.
    if (!FRange)
      return RangesTy();

    return FRange->Func->Ranges;
  }

  const std::unordered_map<std::string, BinaryFunction> &
  getAllBinaryFunctions() {
    return BinaryFunctions;
  }

  BinaryFunction *getBinaryFunction(StringRef FName) {
    auto I = BinaryFunctions.find(FName.str());
    if (I == BinaryFunctions.end())
      return nullptr;
    return &I->second;
  }

  uint32_t getFuncSizeForContext(SampleContext &Context) {
    return FuncSizeTracker.getFuncSizeForContext(Context);
  }

  // Load the symbols from debug table and populate into symbol list.
  void populateSymbolListFromDWARF(ProfileSymbolList &SymbolList);

  const SampleContextFrameVector &
  getFrameLocationStack(uint64_t Offset, bool UseProbeDiscriminator = false) {
    auto I = Offset2LocStackMap.emplace(Offset, SampleContextFrameVector());
    if (I.second) {
      InstructionPointer IP(this, Offset);
      I.first->second = symbolize(IP, true, UseProbeDiscriminator);
    }
    return I.first->second;
  }

  Optional<SampleContextFrame> getInlineLeafFrameLoc(uint64_t Offset) {
    const auto &Stack = getFrameLocationStack(Offset);
    if (Stack.empty())
      return {};
    return Stack.back();
  }

  void flushSymbolizer() { Symbolizer.reset(); }

  // Compare two addresses' inline context
  bool inlineContextEqual(uint64_t Add1, uint64_t Add2);

  // Get the full context of the current stack with inline context filled in.
  // It will search the disassembling info stored in Offset2LocStackMap. This is
  // used as the key of function sample map
  SampleContextFrameVector
  getExpandedContext(const SmallVectorImpl<uint64_t> &Stack,
                     bool &WasLeafInlined);
  // Go through instructions among the given range and record its size for the
  // inline context.
  void computeInlinedContextSizeForRange(uint64_t StartOffset,
                                         uint64_t EndOffset);

  void computeInlinedContextSizeForFunc(const BinaryFunction *Func);

  const MCDecodedPseudoProbe *getCallProbeForAddr(uint64_t Address) const {
    return ProbeDecoder.getCallProbeForAddr(Address);
  }

  void getInlineContextForProbe(const MCDecodedPseudoProbe *Probe,
                                SampleContextFrameVector &InlineContextStack,
                                bool IncludeLeaf = false) const {
    SmallVector<MCPseduoProbeFrameLocation, 16> ProbeInlineContext;
    ProbeDecoder.getInlineContextForProbe(Probe, ProbeInlineContext,
                                          IncludeLeaf);
    for (uint32_t I = 0; I < ProbeInlineContext.size(); I++) {
      auto &Callsite = ProbeInlineContext[I];
      // Clear the current context for an unknown probe.
      if (Callsite.second == 0 && I != ProbeInlineContext.size() - 1) {
        InlineContextStack.clear();
        continue;
      }
      InlineContextStack.emplace_back(Callsite.first,
                                      LineLocation(Callsite.second, 0));
    }
  }
  const AddressProbesMap &getAddress2ProbesMap() const {
    return ProbeDecoder.getAddress2ProbesMap();
  }
  const MCPseudoProbeFuncDesc *getFuncDescForGUID(uint64_t GUID) {
    return ProbeDecoder.getFuncDescForGUID(GUID);
  }

  const MCPseudoProbeFuncDesc *
  getInlinerDescForProbe(const MCDecodedPseudoProbe *Probe) {
    return ProbeDecoder.getInlinerDescForProbe(Probe);
  }

  bool getTrackFuncContextSize() { return TrackFuncContextSize; }

  bool getIsLoadedByMMap() { return IsLoadedByMMap; }

  void setIsLoadedByMMap(bool Value) { IsLoadedByMMap = Value; }

  bool getMissingMMapWarned() { return MissingMMapWarned; }

  void setMissingMMapWarned(bool Value) { MissingMMapWarned = Value; }
};

} // end namespace sampleprof
} // end namespace llvm

#endif
