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
#include "llvm/MC/MCSection.h"
#include <functional>
#include <map>
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

/// Instances of this class represent a pseudo probe instance for a pseudo probe
/// table entry, which is created during a machine instruction is assembled and
/// uses an address from a temporary label created at the current address in the
/// current section.
class MCPseudoProbe {
  MCSymbol *Label;
  uint64_t Guid;
  uint64_t Index;
  uint8_t Type;
  uint8_t Attributes;

public:
  MCPseudoProbe(MCSymbol *Label, uint64_t Guid, uint64_t Index, uint64_t Type,
                uint64_t Attributes)
      : Label(Label), Guid(Guid), Index(Index), Type(Type),
        Attributes(Attributes) {
    assert(Type <= 0xFF && "Probe type too big to encode, exceeding 2^8");
    assert(Attributes <= 0xFF &&
           "Probe attributes too big to encode, exceeding 2^16");
  }

  MCSymbol *getLabel() const { return Label; }

  uint64_t getGuid() const { return Guid; }

  uint64_t getIndex() const { return Index; }

  uint8_t getType() const { return Type; }

  uint8_t getAttributes() const { return Attributes; }

  void emit(MCObjectStreamer *MCOS, const MCPseudoProbe *LastProbe) const;
};

// An inline frame has the form <Guid, ProbeID>
using InlineSite = std::tuple<uint64_t, uint32_t>;
using MCPseudoProbeInlineStack = SmallVector<InlineSite, 8>;

// A Tri-tree based data structure to group probes by inline stack.
// A tree is allocated for a standalone .text section. A fake
// instance is created as the root of a tree.
// A real instance of this class is created for each function, either an
// unlined function that has code in .text section or an inlined function.
class MCPseudoProbeInlineTree {
  uint64_t Guid;
  // Set of probes that come with the function.
  std::vector<MCPseudoProbe> Probes;
  // Use std::map for a deterministic output.
  std::map<InlineSite, MCPseudoProbeInlineTree *> Inlinees;

  // Root node has a GUID 0.
  bool isRoot() { return Guid == 0; }
  MCPseudoProbeInlineTree *getOrAddNode(InlineSite Site);

public:
  MCPseudoProbeInlineTree() = default;
  MCPseudoProbeInlineTree(uint64_t Guid) : Guid(Guid) {}
  ~MCPseudoProbeInlineTree();
  void addPseudoProbe(const MCPseudoProbe &Probe,
                      const MCPseudoProbeInlineStack &InlineStack);
  void emit(MCObjectStreamer *MCOS, const MCPseudoProbe *&LastProbe);
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
} // end namespace llvm

#endif // LLVM_MC_MCPSEUDOPROBE_H
