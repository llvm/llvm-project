//===--- PseudoProbe.cpp - Pseudo probe decoding utilities  ------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PseudoProbe.h"
#include "ErrorHandling.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
#include <memory>

using namespace llvm;
using namespace sampleprof;
using namespace support;

namespace llvm {
namespace sampleprof {

static StringRef getProbeFNameForGUID(const GUIDProbeFunctionMap &GUID2FuncMAP,
                                      uint64_t GUID) {
  auto It = GUID2FuncMAP.find(GUID);
  assert(It != GUID2FuncMAP.end() &&
         "Probe function must exist for a valid GUID");
  return It->second.FuncName;
}

void PseudoProbeFuncDesc::print(raw_ostream &OS) {
  OS << "GUID: " << FuncGUID << " Name: " << FuncName << "\n";
  OS << "Hash: " << FuncHash << "\n";
}

void PseudoProbe::getInlineContext(SmallVectorImpl<std::string> &ContextStack,
                                   const GUIDProbeFunctionMap &GUID2FuncMAP,
                                   bool ShowName) const {
  uint32_t Begin = ContextStack.size();
  PseudoProbeInlineTree *Cur = InlineTree;
  // It will add the string of each node's inline site during iteration.
  // Note that it won't include the probe's belonging function(leaf location)
  while (Cur->hasInlineSite()) {
    std::string ContextStr;
    if (ShowName) {
      StringRef FuncName =
          getProbeFNameForGUID(GUID2FuncMAP, std::get<0>(Cur->ISite));
      ContextStr += FuncName.str();
    } else {
      ContextStr += Twine(std::get<0>(Cur->ISite)).str();
    }
    ContextStr += ":";
    ContextStr += Twine(std::get<1>(Cur->ISite)).str();
    ContextStack.emplace_back(ContextStr);
    Cur = Cur->Parent;
  }
  // Make the ContextStack in caller-callee order
  std::reverse(ContextStack.begin() + Begin, ContextStack.end());
}

std::string
PseudoProbe::getInlineContextStr(const GUIDProbeFunctionMap &GUID2FuncMAP,
                                 bool ShowName) const {
  std::ostringstream OContextStr;
  SmallVector<std::string, 16> ContextStack;
  getInlineContext(ContextStack, GUID2FuncMAP, ShowName);
  for (auto &CxtStr : ContextStack) {
    if (OContextStr.str().size())
      OContextStr << " @ ";
    OContextStr << CxtStr;
  }
  return OContextStr.str();
}

static const char *PseudoProbeTypeStr[3] = {"Block", "IndirectCall",
                                            "DirectCall"};

void PseudoProbe::print(raw_ostream &OS,
                        const GUIDProbeFunctionMap &GUID2FuncMAP,
                        bool ShowName) {
  OS << "FUNC: ";
  if (ShowName) {
    StringRef FuncName = getProbeFNameForGUID(GUID2FuncMAP, GUID);
    OS << FuncName.str() << " ";
  } else {
    OS << GUID << " ";
  }
  OS << "Index: " << Index << "  ";
  OS << "Type: " << PseudoProbeTypeStr[static_cast<uint8_t>(Type)] << "  ";

  std::string InlineContextStr = getInlineContextStr(GUID2FuncMAP, ShowName);
  if (InlineContextStr.size()) {
    OS << "Inlined: @ ";
    OS << InlineContextStr;
  }
  OS << "\n";
}

template <typename T> T PseudoProbeDecoder::readUnencodedNumber() {
  if (Data + sizeof(T) > End) {
    exitWithError("Decode unencoded number error in " + SectionName +
                  " section");
  }
  T Val = endian::readNext<T, little, unaligned>(Data);
  return Val;
}

template <typename T> T PseudoProbeDecoder::readUnsignedNumber() {
  unsigned NumBytesRead = 0;
  uint64_t Val = decodeULEB128(Data, &NumBytesRead);
  if (Val > std::numeric_limits<T>::max() || (Data + NumBytesRead > End)) {
    exitWithError("Decode number error in " + SectionName + " section");
  }
  Data += NumBytesRead;
  return static_cast<T>(Val);
}

template <typename T> T PseudoProbeDecoder::readSignedNumber() {
  unsigned NumBytesRead = 0;
  int64_t Val = decodeSLEB128(Data, &NumBytesRead);
  if (Val > std::numeric_limits<T>::max() || (Data + NumBytesRead > End)) {
    exitWithError("Decode number error in " + SectionName + " section");
  }
  Data += NumBytesRead;
  return static_cast<T>(Val);
}

StringRef PseudoProbeDecoder::readString(uint32_t Size) {
  StringRef Str(reinterpret_cast<const char *>(Data), Size);
  if (Data + Size > End) {
    exitWithError("Decode string error in " + SectionName + " section");
  }
  Data += Size;
  return Str;
}

void PseudoProbeDecoder::buildGUID2FuncDescMap(const uint8_t *Start,
                                               std::size_t Size) {
  // The pseudo_probe_desc section has a format like:
  // .section .pseudo_probe_desc,"",@progbits
  // .quad -5182264717993193164   // GUID
  // .quad 4294967295             // Hash
  // .uleb 3                      // Name size
  // .ascii "foo"                 // Name
  // .quad -2624081020897602054
  // .quad 174696971957
  // .uleb 34
  // .ascii "main"
#ifndef NDEBUG
  SectionName = "pseudo_probe_desc";
#endif
  Data = Start;
  End = Data + Size;

  while (Data < End) {
    uint64_t GUID = readUnencodedNumber<uint64_t>();
    uint64_t Hash = readUnencodedNumber<uint64_t>();
    uint32_t NameSize = readUnsignedNumber<uint32_t>();
    StringRef Name = FunctionSamples::getCanonicalFnName(readString(NameSize));

    // Initialize PseudoProbeFuncDesc and populate it into GUID2FuncDescMap
    GUID2FuncDescMap.emplace(GUID, PseudoProbeFuncDesc(GUID, Hash, Name));
  }
  assert(Data == End && "Have unprocessed data in pseudo_probe_desc section");
}

void PseudoProbeDecoder::buildAddress2ProbeMap(const uint8_t *Start,
                                               std::size_t Size) {
  // The pseudo_probe section encodes an inline forest and each tree has a
  // format like:
  //  FUNCTION BODY (one for each uninlined function present in the text
  //  section)
  //     GUID (uint64)
  //         GUID of the function
  //     NPROBES (ULEB128)
  //         Number of probes originating from this function.
  //     NUM_INLINED_FUNCTIONS (ULEB128)
  //         Number of callees inlined into this function, aka number of
  //         first-level inlinees
  //     PROBE RECORDS
  //         A list of NPROBES entries. Each entry contains:
  //           INDEX (ULEB128)
  //           TYPE (uint4)
  //             0 - block probe, 1 - indirect call, 2 - direct call
  //           ATTRIBUTE (uint3)
  //             1 - reserved
  //           ADDRESS_TYPE (uint1)
  //             0 - code address, 1 - address delta
  //           CODE_ADDRESS (uint64 or ULEB128)
  //             code address or address delta, depending on Flag
  //     INLINED FUNCTION RECORDS
  //         A list of NUM_INLINED_FUNCTIONS entries describing each of the
  //         inlined callees.  Each record contains:
  //           INLINE SITE
  //             Index of the callsite probe (ULEB128)
  //           FUNCTION BODY
  //             A FUNCTION BODY entry describing the inlined function.
#ifndef NDEBUG
  SectionName = "pseudo_probe";
#endif
  Data = Start;
  End = Data + Size;

  PseudoProbeInlineTree *Root = &DummyInlineRoot;
  PseudoProbeInlineTree *Cur = &DummyInlineRoot;
  uint64_t LastAddr = 0;
  uint32_t Index = 0;
  // A DFS-based decoding
  while (Data < End) {
    if (Root == Cur) {
      // Use a sequential id for top level inliner.
      Index = Root->getChildren().size();
    } else {
      // Read inline site for inlinees
      Index = readUnsignedNumber<uint32_t>();
    }
    // Switch/add to a new tree node(inlinee)
    Cur = Cur->getOrAddNode(std::make_tuple(Cur->GUID, Index));
    // Read guid
    Cur->GUID = readUnencodedNumber<uint64_t>();
    // Read number of probes in the current node.
    uint32_t NodeCount = readUnsignedNumber<uint32_t>();
    // Read number of direct inlinees
    Cur->ChildrenToProcess = readUnsignedNumber<uint32_t>();
    // Read all probes in this node
    for (std::size_t I = 0; I < NodeCount; I++) {
      // Read index
      uint32_t Index = readUnsignedNumber<uint32_t>();
      // Read type | flag.
      uint8_t Value = readUnencodedNumber<uint8_t>();
      uint8_t Kind = Value & 0xf;
      uint8_t Attr = (Value & 0x70) >> 4;
      // Read address
      uint64_t Addr = 0;
      if (Value & 0x80) {
        int64_t Offset = readSignedNumber<int64_t>();
        Addr = LastAddr + Offset;
      } else {
        Addr = readUnencodedNumber<int64_t>();
      }
      // Populate Address2ProbesMap
      auto &Probes = Address2ProbesMap[Addr];
      Probes.emplace_back(Addr, Cur->GUID, Index, PseudoProbeType(Kind), Attr,
                          Cur);
      Cur->addProbes(&Probes.back());
      LastAddr = Addr;
    }

    // Look for the parent for the next node by subtracting the current
    // node count from tree counts along the parent chain. The first node
    // in the chain that has a non-zero tree count is the target.
    while (Cur != Root) {
      if (Cur->ChildrenToProcess == 0) {
        Cur = Cur->Parent;
        if (Cur != Root) {
          assert(Cur->ChildrenToProcess > 0 &&
                 "Should have some unprocessed nodes");
          Cur->ChildrenToProcess -= 1;
        }
      } else {
        break;
      }
    }
  }

  assert(Data == End && "Have unprocessed data in pseudo_probe section");
  assert(Cur == Root &&
         " Cur should point to root when the forest is fully built up");
}

void PseudoProbeDecoder::printGUID2FuncDescMap(raw_ostream &OS) {
  OS << "Pseudo Probe Desc:\n";
  // Make the output deterministic
  std::map<uint64_t, PseudoProbeFuncDesc> OrderedMap(GUID2FuncDescMap.begin(),
                                                     GUID2FuncDescMap.end());
  for (auto &I : OrderedMap) {
    I.second.print(OS);
  }
}

void PseudoProbeDecoder::printProbeForAddress(raw_ostream &OS,
                                              uint64_t Address) {
  auto It = Address2ProbesMap.find(Address);
  if (It != Address2ProbesMap.end()) {
    for (auto &Probe : It->second) {
      OS << " [Probe]:\t";
      Probe.print(OS, GUID2FuncDescMap, true);
    }
  }
}

const PseudoProbe *
PseudoProbeDecoder::getCallProbeForAddr(uint64_t Address) const {
  auto It = Address2ProbesMap.find(Address);
  if (It == Address2ProbesMap.end())
    return nullptr;
  const auto &Probes = It->second;

  const PseudoProbe *CallProbe = nullptr;
  for (const auto &Probe : Probes) {
    if (Probe.isCall()) {
      assert(!CallProbe &&
             "There should be only one call probe corresponding to address "
             "which is a callsite.");
      CallProbe = &Probe;
    }
  }
  return CallProbe;
}

const PseudoProbeFuncDesc *
PseudoProbeDecoder::getFuncDescForGUID(uint64_t GUID) const {
  auto It = GUID2FuncDescMap.find(GUID);
  assert(It != GUID2FuncDescMap.end() && "Function descriptor doesn't exist");
  return &It->second;
}

void PseudoProbeDecoder::getInlineContextForProbe(
    const PseudoProbe *Probe, SmallVectorImpl<std::string> &InlineContextStack,
    bool IncludeLeaf) const {
  Probe->getInlineContext(InlineContextStack, GUID2FuncDescMap, true);
  if (!IncludeLeaf)
    return;
  // Note that the context from probe doesn't include leaf frame,
  // hence we need to retrieve and prepend leaf if requested.
  const auto *FuncDesc = getFuncDescForGUID(Probe->GUID);
  InlineContextStack.emplace_back(FuncDesc->FuncName + ":" +
                                  Twine(Probe->Index).str());
}

const PseudoProbeFuncDesc *
PseudoProbeDecoder::getInlinerDescForProbe(const PseudoProbe *Probe) const {
  PseudoProbeInlineTree *InlinerNode = Probe->InlineTree;
  if (!InlinerNode->hasInlineSite())
    return nullptr;
  return getFuncDescForGUID(std::get<0>(InlinerNode->ISite));
}

} // end namespace sampleprof
} // end namespace llvm
