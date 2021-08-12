//===- MCPseudoProbe.h - Pseudo probe encoding support ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MCPseudoProbe to support the pseudo
// probe encoding for AutoFDO. Pseudo probes together with their inline context
// are encoded in a DFS recursive way in the .pseudoprobe sections. For each
// .pseudoprobe section, the encoded binary data consist of a single or mutiple
// function records each for one outlined function. A function record has the
// following format :
//
// FUNCTION BODY (one for each outlined function present in the text section)
//    GUID (uint64)
//        GUID of the function
//    NPROBES (ULEB128)
//        Number of probes originating from this function.
//    NUM_INLINED_FUNCTIONS (ULEB128)
//        Number of callees inlined into this function, aka number of
//        first-level inlinees
//    PROBE RECORDS
//        A list of NPROBES entries. Each entry contains:
//          INDEX (ULEB128)
//          TYPE (uint4)
//            0 - block probe, 1 - indirect call, 2 - direct call
//          ATTRIBUTE (uint3)
//            1 - reserved
//          ADDRESS_TYPE (uint1)
//            0 - code address, 1 - address delta
//          CODE_ADDRESS (uint64 or ULEB128)
//            code address or address delta, depending on ADDRESS_TYPE
//    INLINED FUNCTION RECORDS
//        A list of NUM_INLINED_FUNCTIONS entries describing each of the inlined
//        callees.  Each record contains:
//          INLINE SITE
//            ID of the callsite probe (ULEB128)
//          FUNCTION BODY
//            A FUNCTION BODY entry describing the inlined function.
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCPSEUDOPROBE_H
#define LLVM_MC_MCPSEUDOPROBE_H

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/PseudoProbe.h"
#include "llvm/MC/MCSection.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <functional>
#include <list>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llvm {

class MCStreamer;
class MCSymbol;
class MCObjectStreamer;

enum class MCPseudoProbeFlag {
  // If set, indicates that the probe is encoded as an address delta
  // instead of a real code address.
  AddressDelta = 0x1,
};

// Function descriptor decoded from .pseudo_probe_desc section
struct MCPseudoProbeFuncDesc {
  uint64_t FuncGUID = 0;
  uint64_t FuncHash = 0;
  std::string FuncName;

  MCPseudoProbeFuncDesc(uint64_t GUID, uint64_t Hash, StringRef Name)
      : FuncGUID(GUID), FuncHash(Hash), FuncName(Name){};

  void print(raw_ostream &OS);
};

class MCPseudoProbe;
class MCDecodedPseudoProbe;

// An inline frame has the form <Guid, ProbeID>
using InlineSite = std::tuple<uint64_t, uint32_t>;
using MCPseudoProbeInlineStack = SmallVector<InlineSite, 8>;
// GUID to PseudoProbeFuncDesc map
using GUIDProbeFunctionMap =
    std::unordered_map<uint64_t, MCPseudoProbeFuncDesc>;
// Address to pseudo probes map.
using AddressProbesMap =
    std::unordered_map<uint64_t, std::list<MCDecodedPseudoProbe>>;

class MCPseudoProbeInlineTree;
class MCDecodedPseudoProbeInlineTree;

class MCPseudoProbeBase {
protected:
  uint64_t Guid;
  uint64_t Index;
  uint8_t Attributes;
  uint8_t Type;
  // The value should be equal to PseudoProbeReservedId::Last + 1 which is
  // defined in SampleProfileProbe.h. The header file is not included here to
  // reduce the dependency from MC to IPO.
  const static uint32_t PseudoProbeFirstId = 1;

public:
  MCPseudoProbeBase(uint64_t G, uint64_t I, uint64_t At, uint8_t T)
      : Guid(G), Index(I), Attributes(At), Type(T) {}

  bool isEntry() const { return Index == PseudoProbeFirstId; }

  uint64_t getGuid() const { return Guid; }

  uint64_t getIndex() const { return Index; }

  uint8_t getAttributes() const { return Attributes; }

  uint8_t getType() const { return Type; }

  bool isBlock() const {
    return Type == static_cast<uint8_t>(PseudoProbeType::Block);
  }

  bool isIndirectCall() const {
    return Type == static_cast<uint8_t>(PseudoProbeType::IndirectCall);
  }

  bool isDirectCall() const {
    return Type == static_cast<uint8_t>(PseudoProbeType::DirectCall);
  }

  bool isCall() const { return isIndirectCall() || isDirectCall(); }

  void setAttributes(uint8_t Attr) { Attributes = Attr; }
};

/// Instances of this class represent a pseudo probe instance for a pseudo probe
/// table entry, which is created during a machine instruction is assembled and
/// uses an address from a temporary label created at the current address in the
/// current section.
class MCPseudoProbe : public MCPseudoProbeBase {
  MCSymbol *Label;

public:
  MCPseudoProbe(MCSymbol *Label, uint64_t Guid, uint64_t Index, uint64_t Type,
                uint64_t Attributes)
      : MCPseudoProbeBase(Guid, Index, Attributes, Type), Label(Label) {
    assert(Type <= 0xFF && "Probe type too big to encode, exceeding 2^8");
    assert(Attributes <= 0xFF &&
           "Probe attributes too big to encode, exceeding 2^16");
  }

  MCSymbol *getLabel() const { return Label; }
  void emit(MCObjectStreamer *MCOS, const MCPseudoProbe *LastProbe) const;
};

class MCDecodedPseudoProbe : public MCPseudoProbeBase {
  uint64_t Address;
  MCDecodedPseudoProbeInlineTree *InlineTree;

public:
  MCDecodedPseudoProbe(uint64_t Ad, uint64_t G, uint32_t I, PseudoProbeType K,
                       uint8_t At, MCDecodedPseudoProbeInlineTree *Tree)
      : MCPseudoProbeBase(G, I, At, static_cast<uint8_t>(K)), Address(Ad),
        InlineTree(Tree){};

  uint64_t getAddress() const { return Address; }

  void setAddress(uint64_t Addr) { Address = Addr; }

  MCDecodedPseudoProbeInlineTree *getInlineTreeNode() const {
    return InlineTree;
  }

  // Get the inlined context by traversing current inline tree backwards,
  // each tree node has its InlineSite which is taken as the context.
  // \p ContextStack is populated in root to leaf order
  void getInlineContext(SmallVectorImpl<std::string> &ContextStack,
                        const GUIDProbeFunctionMap &GUID2FuncMAP,
                        bool ShowName) const;

  // Helper function to get the string from context stack
  std::string getInlineContextStr(const GUIDProbeFunctionMap &GUID2FuncMAP,
                                  bool ShowName) const;

  // Print pseudo probe while disassembling
  void print(raw_ostream &OS, const GUIDProbeFunctionMap &GUID2FuncMAP,
             bool ShowName) const;
};

template <typename ProbeType, typename DerivedProbeInlineTreeType>
class MCPseudoProbeInlineTreeBase {
  struct InlineSiteHash {
    uint64_t operator()(const InlineSite &Site) const {
      return std::get<0>(Site) ^ std::get<1>(Site);
    }
  };

protected:
  // Track children (e.g. inlinees) of current context
  using InlinedProbeTreeMap = std::unordered_map<
      InlineSite, std::unique_ptr<DerivedProbeInlineTreeType>, InlineSiteHash>;
  InlinedProbeTreeMap Children;
  // Set of probes that come with the function.
  std::vector<ProbeType> Probes;
  MCPseudoProbeInlineTreeBase() {
    static_assert(std::is_base_of<MCPseudoProbeInlineTreeBase,
                                  DerivedProbeInlineTreeType>::value,
                  "DerivedProbeInlineTreeType must be subclass of "
                  "MCPseudoProbeInlineTreeBase");
  }

public:
  uint64_t Guid = 0;

  // Root node has a GUID 0.
  bool isRoot() const { return Guid == 0; }
  InlinedProbeTreeMap &getChildren() { return Children; }
  const InlinedProbeTreeMap &getChildren() const { return Children; }
  std::vector<ProbeType> &getProbes() { return Probes; }
  void addProbes(ProbeType Probe) { Probes.push_back(Probe); }
  // Caller node of the inline site
  MCPseudoProbeInlineTreeBase<ProbeType, DerivedProbeInlineTreeType> *Parent;
  DerivedProbeInlineTreeType *getOrAddNode(const InlineSite &Site) {
    auto Ret = Children.emplace(
        Site, std::make_unique<DerivedProbeInlineTreeType>(Site));
    Ret.first->second->Parent = this;
    return Ret.first->second.get();
  };
};

// A Tri-tree based data structure to group probes by inline stack.
// A tree is allocated for a standalone .text section. A fake
// instance is created as the root of a tree.
// A real instance of this class is created for each function, either an
// unlined function that has code in .text section or an inlined function.

class MCPseudoProbeInlineTree
    : public MCPseudoProbeInlineTreeBase<MCPseudoProbe,
                                         MCPseudoProbeInlineTree> {
public:
  MCPseudoProbeInlineTree() = default;
  MCPseudoProbeInlineTree(uint64_t Guid) { this->Guid = Guid; }
  MCPseudoProbeInlineTree(const InlineSite &Site) {
    this->Guid = std::get<0>(Site);
  }

  // MCPseudoProbeInlineTree method based on Inlinees
  void addPseudoProbe(const MCPseudoProbe &Probe,
                      const MCPseudoProbeInlineStack &InlineStack);
  void emit(MCObjectStreamer *MCOS, const MCPseudoProbe *&LastProbe);
};

// inline tree node for the decoded pseudo probe
class MCDecodedPseudoProbeInlineTree
    : public MCPseudoProbeInlineTreeBase<MCDecodedPseudoProbe *,
                                         MCDecodedPseudoProbeInlineTree> {
public:
  InlineSite ISite;
  // Used for decoding
  uint32_t ChildrenToProcess = 0;

  MCDecodedPseudoProbeInlineTree(){};
  MCDecodedPseudoProbeInlineTree(const InlineSite &Site) : ISite(Site){};

  // Return false if it's a dummy inline site
  bool hasInlineSite() const { return std::get<0>(ISite) != 0; }
};

/// Instances of this class represent the pseudo probes inserted into a compile
/// unit.
class MCPseudoProbeSection {
public:
  void addPseudoProbe(MCSection *Sec, const MCPseudoProbe &Probe,
                      const MCPseudoProbeInlineStack &InlineStack) {
    MCProbeDivisions[Sec].addPseudoProbe(Probe, InlineStack);
  }

  // TODO: Sort by getOrdinal to ensure a determinstic section order
  using MCProbeDivisionMap = std::map<MCSection *, MCPseudoProbeInlineTree>;

private:
  // A collection of MCPseudoProbe for each text section. The MCPseudoProbes
  // are grouped by GUID of the functions where they are from and will be
  // encoded by groups. In the comdat scenario where a text section really only
  // contains the code of a function solely, the probes associated with a comdat
  // function are still grouped by GUIDs due to inlining that can bring probes
  // from different functions into one function.
  MCProbeDivisionMap MCProbeDivisions;

public:
  const MCProbeDivisionMap &getMCProbes() const { return MCProbeDivisions; }

  bool empty() const { return MCProbeDivisions.empty(); }

  void emit(MCObjectStreamer *MCOS);
};

class MCPseudoProbeTable {
  // A collection of MCPseudoProbe in the current module grouped by text
  // sections. MCPseudoProbes will be encoded into a corresponding
  // .pseudoprobe section. With functions emitted as separate comdats,
  // a text section really only contains the code of a function solely, and the
  // probes associated with the text section will be emitted into a standalone
  // .pseudoprobe section that shares the same comdat group with the function.
  MCPseudoProbeSection MCProbeSections;

public:
  static void emit(MCObjectStreamer *MCOS);

  MCPseudoProbeSection &getProbeSections() { return MCProbeSections; }

#ifndef NDEBUG
  static int DdgPrintIndent;
#endif
};

class MCPseudoProbeDecoder {
  // GUID to PseudoProbeFuncDesc map.
  GUIDProbeFunctionMap GUID2FuncDescMap;

  // Address to probes map.
  AddressProbesMap Address2ProbesMap;

  // The dummy root of the inline trie, all the outlined function will directly
  // be the children of the dummy root, all the inlined function will be the
  // children of its inlineer. So the relation would be like:
  // DummyRoot --> OutlinedFunc --> InlinedFunc1 --> InlinedFunc2
  MCDecodedPseudoProbeInlineTree DummyInlineRoot;

  /// Points to the current location in the buffer.
  const uint8_t *Data = nullptr;

  /// Points to the end of the buffer.
  const uint8_t *End = nullptr;

  // Decoding helper function
  template <typename T> ErrorOr<T> readUnencodedNumber();
  template <typename T> ErrorOr<T> readUnsignedNumber();
  template <typename T> ErrorOr<T> readSignedNumber();
  ErrorOr<StringRef> readString(uint32_t Size);

public:
  // Decode pseudo_probe_desc section to build GUID to PseudoProbeFuncDesc map.
  bool buildGUID2FuncDescMap(const uint8_t *Start, std::size_t Size);

  // Decode pseudo_probe section to build address to probes map.
  bool buildAddress2ProbeMap(const uint8_t *Start, std::size_t Size);

  // Print pseudo_probe_desc section info
  void printGUID2FuncDescMap(raw_ostream &OS);

  // Print pseudo_probe section info, used along with show-disassembly
  void printProbeForAddress(raw_ostream &OS, uint64_t Address);

  // do printProbeForAddress for all addresses
  void printProbesForAllAddresses(raw_ostream &OS);

  // Look up the probe of a call for the input address
  const MCDecodedPseudoProbe *getCallProbeForAddr(uint64_t Address) const;

  const MCPseudoProbeFuncDesc *getFuncDescForGUID(uint64_t GUID) const;

  // Helper function to populate one probe's inline stack into
  // \p InlineContextStack.
  // Current leaf location info will be added if IncludeLeaf is true
  // Example:
  //  Current probe(bar:3) inlined at foo:2 then inlined at main:1
  //  IncludeLeaf = true,  Output: [main:1, foo:2, bar:3]
  //  IncludeLeaf = false, Output: [main:1, foo:2]
  void
  getInlineContextForProbe(const MCDecodedPseudoProbe *Probe,
                           SmallVectorImpl<std::string> &InlineContextStack,
                           bool IncludeLeaf) const;

  const AddressProbesMap &getAddress2ProbesMap() const {
    return Address2ProbesMap;
  }

  AddressProbesMap &getAddress2ProbesMap() { return Address2ProbesMap; }

  const GUIDProbeFunctionMap &getGUID2FuncDescMap() const {
    return GUID2FuncDescMap;
  }

  const MCPseudoProbeFuncDesc *
  getInlinerDescForProbe(const MCDecodedPseudoProbe *Probe) const;

  const MCDecodedPseudoProbeInlineTree &getDummyInlineRoot() const {
    return DummyInlineRoot;
  }
};

} // end namespace llvm

#endif // LLVM_MC_MCPSEUDOPROBE_H
