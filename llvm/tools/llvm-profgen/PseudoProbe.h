//===--- PseudoProbe.h - Pseudo probe decoding utilities ---------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_TOOLS_LLVM_PROFGEN_PSEUDOPROBE_H
#define LLVM_TOOLS_LLVM_PROFGEN_PSEUDOPROBE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/PseudoProbe.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/SampleProfileProbe.h"
#include <algorithm>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace llvm {
namespace sampleprof {

enum PseudoProbeAttributes { TAILCALL = 1, DANGLING = 2 };

// Use func GUID and index as the location info of the inline site
using InlineSite = std::tuple<uint64_t, uint32_t>;

struct PseudoProbe;

// Tree node to represent the inline relation and its inline site, we use a
// dummy root in the PseudoProbeDecoder to lead the tree, the outlined
// function will directly be the children of the dummy root. For the inlined
// function, all the inlinee will be connected to its inlineer, then further to
// its outlined function. Pseudo probes originating from the function stores the
// tree's leaf node which we can process backwards to get its inline context
class PseudoProbeInlineTree {
  std::vector<PseudoProbe *> ProbeVector;

  struct InlineSiteHash {
    uint64_t operator()(const InlineSite &Site) const {
      return std::get<0>(Site) ^ std::get<1>(Site);
    }
  };
  std::unordered_map<InlineSite, std::unique_ptr<PseudoProbeInlineTree>,
                     InlineSiteHash>
      Children;

public:
  // Inlinee function GUID
  uint64_t GUID = 0;
  // Inline site to indicate the location in its inliner. As the node could also
  // be an outlined function, it will use a dummy InlineSite whose GUID and
  // Index is 0 connected to the dummy root
  InlineSite ISite;
  // Used for decoding
  uint32_t ChildrenToProcess = 0;
  // Caller node of the inline site
  PseudoProbeInlineTree *Parent;

  PseudoProbeInlineTree(){};
  PseudoProbeInlineTree(const InlineSite &Site) : ISite(Site){};

  PseudoProbeInlineTree *getOrAddNode(const InlineSite &Site) {
    auto Ret =
        Children.emplace(Site, std::make_unique<PseudoProbeInlineTree>(Site));
    Ret.first->second->Parent = this;
    return Ret.first->second.get();
  }

  void addProbes(PseudoProbe *Probe) { ProbeVector.push_back(Probe); }
  // Return false if it's a dummy inline site
  bool hasInlineSite() const { return std::get<0>(ISite) != 0; }
};

// Function descriptor decoded from .pseudo_probe_desc section
struct PseudoProbeFuncDesc {
  uint64_t FuncGUID = 0;
  uint64_t FuncHash = 0;
  std::string FuncName;

  PseudoProbeFuncDesc(uint64_t GUID, uint64_t Hash, StringRef Name)
      : FuncGUID(GUID), FuncHash(Hash), FuncName(Name){};

  void print(raw_ostream &OS);
};

// GUID to PseudoProbeFuncDesc map
using GUIDProbeFunctionMap = std::unordered_map<uint64_t, PseudoProbeFuncDesc>;
// Address to pseudo probes map.
using AddressProbesMap = std::unordered_map<uint64_t, std::vector<PseudoProbe>>;

/*
A pseudo probe has the format like below:
  INDEX (ULEB128)
  TYPE (uint4)
    0 - block probe, 1 - indirect call, 2 - direct call
  ATTRIBUTE (uint3)
    1 - tail call, 2 - dangling
  ADDRESS_TYPE (uint1)
    0 - code address, 1 - address delta
  CODE_ADDRESS (uint64 or ULEB128)
  code address or address delta, depending on Flag
*/
struct PseudoProbe {
  uint64_t Address;
  uint64_t GUID;
  uint32_t Index;
  PseudoProbeType Type;
  uint8_t Attribute;
  PseudoProbeInlineTree *InlineTree;
  const static uint32_t PseudoProbeFirstId =
      static_cast<uint32_t>(PseudoProbeReservedId::Last) + 1;

  PseudoProbe(uint64_t Ad, uint64_t G, uint32_t I, PseudoProbeType K,
              uint8_t At, PseudoProbeInlineTree *Tree)
      : Address(Ad), GUID(G), Index(I), Type(K), Attribute(At),
        InlineTree(Tree){};

  bool isEntry() const { return Index == PseudoProbeFirstId; }

  bool isDangling() const {
    return Attribute & static_cast<uint8_t>(PseudoProbeAttributes::DANGLING);
  }

  bool isTailCall() const {
    return Attribute & static_cast<uint8_t>(PseudoProbeAttributes::TAILCALL);
  }

  bool isBlock() const { return Type == PseudoProbeType::Block; }
  bool isIndirectCall() const { return Type == PseudoProbeType::IndirectCall; }
  bool isDirectCall() const { return Type == PseudoProbeType::DirectCall; }
  bool isCall() const { return isIndirectCall() || isDirectCall(); }

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
             bool ShowName);
};

/*
Decode pseudo probe info from ELF section, used along with ELF reader
Two sections are decoded here:
  1) \fn buildGUID2FunctionMap is responsible for .pseudo_probe_desc
  section which encodes all function descriptors.
  2) \fn buildAddress2ProbeMap is responsible for .pseudoprobe section
    which encodes an inline function forest and each tree includes its
    inlined function and all pseudo probes inside the function.
see \file MCPseudoProbe.h for the details of the section encoding format.
*/
class PseudoProbeDecoder {
  // GUID to PseudoProbeFuncDesc map.
  GUIDProbeFunctionMap GUID2FuncDescMap;

  // Address to probes map.
  AddressProbesMap Address2ProbesMap;

  // The dummy root of the inline trie, all the outlined function will directly
  // be the children of the dummy root, all the inlined function will be the
  // children of its inlineer. So the relation would be like:
  // DummyRoot --> OutlinedFunc --> InlinedFunc1 --> InlinedFunc2
  PseudoProbeInlineTree DummyInlineRoot;

  /// Points to the current location in the buffer.
  const uint8_t *Data = nullptr;

  /// Points to the end of the buffer.
  const uint8_t *End = nullptr;

  /// SectionName used for debug
  std::string SectionName;

  // Decoding helper function
  template <typename T> T readUnencodedNumber();
  template <typename T> T readUnsignedNumber();
  template <typename T> T readSignedNumber();
  StringRef readString(uint32_t Size);

public:
  // Decode pseudo_probe_desc section to build GUID to PseudoProbeFuncDesc map.
  void buildGUID2FuncDescMap(const uint8_t *Start, std::size_t Size);

  // Decode pseudo_probe section to build address to probes map.
  void buildAddress2ProbeMap(const uint8_t *Start, std::size_t Size);

  // Print pseudo_probe_desc section info
  void printGUID2FuncDescMap(raw_ostream &OS);

  // Print pseudo_probe section info, used along with show-disassembly
  void printProbeForAddress(raw_ostream &OS, uint64_t Address);

  // Look up the probe of a call for the input address
  const PseudoProbe *getCallProbeForAddr(uint64_t Address) const;

  const PseudoProbeFuncDesc *getFuncDescForGUID(uint64_t GUID) const;

  // Helper function to populate one probe's inline stack into
  // \p InlineContextStack.
  // Current leaf location info will be added if IncludeLeaf is true
  // Example:
  //  Current probe(bar:3) inlined at foo:2 then inlined at main:1
  //  IncludeLeaf = true,  Output: [main:1, foo:2, bar:3]
  //  IncludeLeaf = false, Output: [main:1, foo:2]
  void
  getInlineContextForProbe(const PseudoProbe *Probe,
                           SmallVectorImpl<std::string> &InlineContextStack,
                           bool IncludeLeaf) const;

  const AddressProbesMap &getAddress2ProbesMap() const {
    return Address2ProbesMap;
  }

  const PseudoProbeFuncDesc *
  getInlinerDescForProbe(const PseudoProbe *Probe) const;
};

} // end namespace sampleprof
} // end namespace llvm

#endif
