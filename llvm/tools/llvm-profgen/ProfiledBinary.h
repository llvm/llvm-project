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
#include "PseudoProbe.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/Symbolize/Symbolize.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/ProfileData/SampleProf.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include <list>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace llvm;
using namespace sampleprof;
using namespace llvm::object;

namespace llvm {
namespace sampleprof {

class ProfiledBinary;

struct InstructionPointer {
  ProfiledBinary *Binary;
  union {
    // Offset of the executable segment of the binary.
    uint64_t Offset = 0;
    // Also used as address in unwinder
    uint64_t Address;
  };
  // Index to the sorted code address array of the binary.
  uint64_t Index = 0;
  InstructionPointer(ProfiledBinary *Binary, uint64_t Address,
                     bool RoundToNext = false);
  void advance();
  void backward();
  void update(uint64_t Addr);
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
  void inferPrologOffsets(
      std::unordered_map<uint64_t, std::string> &FuncStartAddrMap) {
    for (auto I : FuncStartAddrMap) {
      PrologEpilogSet.insert(I.first);
      InstructionPointer IP(Binary, I.first);
      IP.advance();
      PrologEpilogSet.insert(IP.Offset);
    }
  }

  // Take the last two addresses before the return address as epilog
  void inferEpilogOffsets(std::unordered_set<uint64_t> &RetAddrs) {
    for (auto Addr : RetAddrs) {
      PrologEpilogSet.insert(Addr);
      InstructionPointer IP(Binary, Addr);
      IP.backward();
      PrologEpilogSet.insert(IP.Offset);
    }
  }
};

class ProfiledBinary {
  // Absolute path of the binary.
  std::string Path;
  // The target triple.
  Triple TheTriple;
  // The runtime base address that the first executable segment is loaded at.
  uint64_t BaseAddress = 0;
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
  // Function offset to name mapping.
  std::unordered_map<uint64_t, std::string> FuncStartAddrMap;
  // Offset to context location map. Used to expand the context.
  std::unordered_map<uint64_t, FrameLocationStack> Offset2LocStackMap;
  // An array of offsets of all instructions sorted in increasing order. The
  // sorting is needed to fast advance to the next forward/backward instruction.
  std::vector<uint64_t> CodeAddrs;
  // A set of call instruction offsets. Used by virtual unwinding.
  std::unordered_set<uint64_t> CallAddrs;
  // A set of return instruction offsets. Used by virtual unwinding.
  std::unordered_set<uint64_t> RetAddrs;

  PrologEpilogTracker ProEpilogTracker;

  // The symbolizer used to get inline context for an instruction.
  std::unique_ptr<symbolize::LLVMSymbolizer> Symbolizer;

  // Pseudo probe decoder
  PseudoProbeDecoder ProbeDecoder;

  bool UsePseudoProbes = false;

  // Indicate if the base loading address is parsed from the mmap event or uses
  // the preferred address
  bool IsLoadedByMMap = false;
  // Use to avoid redundant warning.
  bool MissingMMapWarned = false;

  void setPreferredTextSegmentAddresses(const ELFObjectFileBase *O);

  template <class ELFT>
  void setPreferredTextSegmentAddresses(const ELFFile<ELFT> &Obj, StringRef FileName);

  void decodePseudoProbe(const ELFObjectFileBase *Obj);

  // Set up disassembler and related components.
  void setUpDisassembler(const ELFObjectFileBase *Obj);
  void setupSymbolizer();

  /// Dissassemble the text section and build various address maps.
  void disassemble(const ELFObjectFileBase *O);

  /// Helper function to dissassemble the symbol and extract info for unwinding
  bool dissassembleSymbol(std::size_t SI, ArrayRef<uint8_t> Bytes,
                          SectionSymbolsTy &Symbols, const SectionRef &Section);
  /// Symbolize a given instruction pointer and return a full call context.
  FrameLocationStack symbolize(const InstructionPointer &IP,
                               bool UseCanonicalFnName = false);

  /// Decode the interesting parts of the binary and build internal data
  /// structures. On high level, the parts of interest are:
  ///   1. Text sections, including the main code section and the PLT
  ///   entries that will be used to handle cross-module call transitions.
  ///   2. The .debug_line section, used by Dwarf-based profile generation.
  ///   3. Pseudo probe related sections, used by probe-based profile
  ///   generation.
  void load();
  const FrameLocationStack &getFrameLocationStack(uint64_t Offset) const {
    auto I = Offset2LocStackMap.find(Offset);
    assert(I != Offset2LocStackMap.end() &&
           "Can't find location for offset in the binary");
    return I->second;
  }

public:
  ProfiledBinary(StringRef Path) : Path(Path), ProEpilogTracker(this) {
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
  // Return the file offset for the first executable segment.
  uint64_t getTextSegmentOffset() const { return TextSegmentOffsets[0]; }
  const std::vector<uint64_t> &getPreferredTextSegmentAddresses() const {
    return PreferredTextSegmentAddresses;
  }
  const std::vector<uint64_t> &getTextSegmentOffsets() const {
    return TextSegmentOffsets;
  }

  bool addressIsCode(uint64_t Address) const {
    uint64_t Offset = virtualAddrToOffset(Address);
    return Offset2LocStackMap.find(Offset) != Offset2LocStackMap.end();
  }
  bool addressIsCall(uint64_t Address) const {
    uint64_t Offset = virtualAddrToOffset(Address);
    return CallAddrs.count(Offset);
  }
  bool addressIsReturn(uint64_t Address) const {
    uint64_t Offset = virtualAddrToOffset(Address);
    return RetAddrs.count(Offset);
  }
  bool addressInPrologEpilog(uint64_t Address) const {
    uint64_t Offset = virtualAddrToOffset(Address);
    return ProEpilogTracker.PrologEpilogSet.count(Offset);
  }

  uint64_t getAddressforIndex(uint64_t Index) const {
    return offsetToVirtualAddr(CodeAddrs[Index]);
  }

  bool usePseudoProbes() const { return UsePseudoProbes; }
  // Get the index in CodeAddrs for the address
  // As we might get an address which is not the code
  // here it would round to the next valid code address by
  // using lower bound operation
  uint32_t getIndexForAddr(uint64_t Address) const {
    uint64_t Offset = virtualAddrToOffset(Address);
    auto Low = llvm::lower_bound(CodeAddrs, Offset);
    return Low - CodeAddrs.begin();
  }

  uint64_t getCallAddrFromFrameAddr(uint64_t FrameAddr) const {
    return getAddressforIndex(getIndexForAddr(FrameAddr) - 1);
  }

  StringRef getFuncFromStartOffset(uint64_t Offset) {
    return FuncStartAddrMap[Offset];
  }

  Optional<FrameLocation> getInlineLeafFrameLoc(uint64_t Offset) {
    const auto &Stack = getFrameLocationStack(Offset);
    if (Stack.empty())
      return {};
    return Stack.back();
  }

  // Compare two addresses' inline context
  bool inlineContextEqual(uint64_t Add1, uint64_t Add2) const;

  // Get the context string of the current stack with inline context filled in.
  // It will search the disassembling info stored in Offset2LocStackMap. This is
  // used as the key of function sample map
  std::string getExpandedContextStr(const SmallVectorImpl<uint64_t> &Stack,
                                    bool &WasLeafInlined) const;

  const PseudoProbe *getCallProbeForAddr(uint64_t Address) const {
    return ProbeDecoder.getCallProbeForAddr(Address);
  }
  void
  getInlineContextForProbe(const PseudoProbe *Probe,
                           SmallVectorImpl<std::string> &InlineContextStack,
                           bool IncludeLeaf = false) const {
    return ProbeDecoder.getInlineContextForProbe(Probe, InlineContextStack,
                                                 IncludeLeaf);
  }
  const AddressProbesMap &getAddress2ProbesMap() const {
    return ProbeDecoder.getAddress2ProbesMap();
  }
  const PseudoProbeFuncDesc *getFuncDescForGUID(uint64_t GUID) {
    return ProbeDecoder.getFuncDescForGUID(GUID);
  }
  const PseudoProbeFuncDesc *getInlinerDescForProbe(const PseudoProbe *Probe) {
    return ProbeDecoder.getInlinerDescForProbe(Probe);
  }

  bool getIsLoadedByMMap() { return IsLoadedByMMap; }

  void setIsLoadedByMMap(bool Value) { IsLoadedByMMap = Value; }

  bool getMissingMMapWarned() { return MissingMMapWarned; }

  void setMissingMMapWarned(bool Value) { MissingMMapWarned = Value; }
};

} // end namespace sampleprof
} // end namespace llvm

#endif
