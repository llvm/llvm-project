//===- lib/MC/MCPseudoProbe.cpp - Pseudo probe encoding support ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCPseudoProbe.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
#include <memory>

#define DEBUG_TYPE "mcpseudoprobe"

using namespace llvm;
using namespace support;

#ifndef NDEBUG
int MCPseudoProbeTable::DdgPrintIndent = 0;
#endif

static const MCExpr *buildSymbolDiff(MCObjectStreamer *MCOS, const MCSymbol *A,
                                     const MCSymbol *B) {
  MCContext &Context = MCOS->getContext();
  MCSymbolRefExpr::VariantKind Variant = MCSymbolRefExpr::VK_None;
  const MCExpr *ARef = MCSymbolRefExpr::create(A, Variant, Context);
  const MCExpr *BRef = MCSymbolRefExpr::create(B, Variant, Context);
  const MCExpr *AddrDelta =
      MCBinaryExpr::create(MCBinaryExpr::Sub, ARef, BRef, Context);
  return AddrDelta;
}

void MCPseudoProbe::emit(MCObjectStreamer *MCOS,
                         const MCPseudoProbe *LastProbe) const {
  // Emit Index
  MCOS->emitULEB128IntValue(Index);
  // Emit Type and the flag:
  // Type (bit 0 to 3), with bit 4 to 6 for attributes.
  // Flag (bit 7, 0 - code address, 1 - address delta). This indicates whether
  // the following field is a symbolic code address or an address delta.
  assert(Type <= 0xF && "Probe type too big to encode, exceeding 15");
  assert(Attributes <= 0x7 &&
         "Probe attributes too big to encode, exceeding 7");
  uint8_t PackedType = Type | (Attributes << 4);
  uint8_t Flag = LastProbe ? ((int8_t)MCPseudoProbeFlag::AddressDelta << 7) : 0;
  MCOS->emitInt8(Flag | PackedType);

  if (LastProbe) {
    // Emit the delta between the address label and LastProbe.
    const MCExpr *AddrDelta =
        buildSymbolDiff(MCOS, Label, LastProbe->getLabel());
    int64_t Delta;
    if (AddrDelta->evaluateAsAbsolute(Delta, MCOS->getAssemblerPtr())) {
      MCOS->emitSLEB128IntValue(Delta);
    } else {
      MCOS->insert(new MCPseudoProbeAddrFragment(AddrDelta));
    }
  } else {
    // Emit label as a symbolic code address.
    MCOS->emitSymbolValue(
        Label, MCOS->getContext().getAsmInfo()->getCodePointerSize());
  }

  LLVM_DEBUG({
    dbgs().indent(MCPseudoProbeTable::DdgPrintIndent);
    dbgs() << "Probe: " << Index << "\n";
  });
}

void MCPseudoProbeInlineTree::addPseudoProbe(
    const MCPseudoProbe &Probe, const MCPseudoProbeInlineStack &InlineStack) {
  // The function should not be called on the root.
  assert(isRoot() && "Should not be called on root");

  // When it comes here, the input look like:
  //    Probe: GUID of C, ...
  //    InlineStack: [88, A], [66, B]
  // which means, Function A inlines function B at call site with a probe id of
  // 88, and B inlines C at probe 66. The tri-tree expects a tree path like {[0,
  // A], [88, B], [66, C]} to locate the tree node where the probe should be
  // added. Note that the edge [0, A] means A is the top-level function we are
  // emitting probes for.

  // Make a [0, A] edge.
  // An empty inline stack means the function that the probe originates from
  // is a top-level function.
  InlineSite Top;
  if (InlineStack.empty()) {
    Top = InlineSite(Probe.getGuid(), 0);
  } else {
    Top = InlineSite(std::get<0>(InlineStack.front()), 0);
  }

  auto *Cur = getOrAddNode(Top);

  // Make interior edges by walking the inline stack. Once it's done, Cur should
  // point to the node that the probe originates from.
  if (!InlineStack.empty()) {
    auto Iter = InlineStack.begin();
    auto Index = std::get<1>(*Iter);
    Iter++;
    for (; Iter != InlineStack.end(); Iter++) {
      // Make an edge by using the previous probe id and current GUID.
      Cur = Cur->getOrAddNode(InlineSite(std::get<0>(*Iter), Index));
      Index = std::get<1>(*Iter);
    }
    Cur = Cur->getOrAddNode(InlineSite(Probe.getGuid(), Index));
  }

  Cur->Probes.push_back(Probe);
}

void MCPseudoProbeInlineTree::emit(MCObjectStreamer *MCOS,
                                   const MCPseudoProbe *&LastProbe) {
  LLVM_DEBUG({
    dbgs().indent(MCPseudoProbeTable::DdgPrintIndent);
    dbgs() << "Group [\n";
    MCPseudoProbeTable::DdgPrintIndent += 2;
  });
  // Emit probes grouped by GUID.
  if (Guid != 0) {
    LLVM_DEBUG({
      dbgs().indent(MCPseudoProbeTable::DdgPrintIndent);
      dbgs() << "GUID: " << Guid << "\n";
    });
    // Emit Guid
    MCOS->emitInt64(Guid);
    // Emit number of probes in this node
    MCOS->emitULEB128IntValue(Probes.size());
    // Emit number of direct inlinees
    MCOS->emitULEB128IntValue(Children.size());
    // Emit probes in this group
    for (const auto &Probe : Probes) {
      Probe.emit(MCOS, LastProbe);
      LastProbe = &Probe;
    }
  } else {
    assert(Probes.empty() && "Root should not have probes");
  }

  // Emit sorted descendant
  // InlineSite is unique for each pair,
  // so there will be no ordering of Inlinee based on MCPseudoProbeInlineTree*
  std::map<InlineSite, MCPseudoProbeInlineTree *> Inlinees;
  for (auto Child = Children.begin(); Child != Children.end(); ++Child)
    Inlinees[Child->first] = Child->second.get();

  for (const auto &Inlinee : Inlinees) {
    if (Guid) {
      // Emit probe index
      MCOS->emitULEB128IntValue(std::get<1>(Inlinee.first));
      LLVM_DEBUG({
        dbgs().indent(MCPseudoProbeTable::DdgPrintIndent);
        dbgs() << "InlineSite: " << std::get<1>(Inlinee.first) << "\n";
      });
    }
    // Emit the group
    Inlinee.second->emit(MCOS, LastProbe);
  }

  LLVM_DEBUG({
    MCPseudoProbeTable::DdgPrintIndent -= 2;
    dbgs().indent(MCPseudoProbeTable::DdgPrintIndent);
    dbgs() << "]\n";
  });
}

void MCPseudoProbeSection::emit(MCObjectStreamer *MCOS) {
  MCContext &Ctx = MCOS->getContext();

  for (auto &ProbeSec : MCProbeDivisions) {
    const MCPseudoProbe *LastProbe = nullptr;
    if (auto *S =
            Ctx.getObjectFileInfo()->getPseudoProbeSection(ProbeSec.first)) {
      // Switch to the .pseudoprobe section or a comdat group.
      MCOS->SwitchSection(S);
      // Emit probes grouped by GUID.
      ProbeSec.second.emit(MCOS, LastProbe);
    }
  }
}

//
// This emits the pseudo probe tables.
//
void MCPseudoProbeTable::emit(MCObjectStreamer *MCOS) {
  MCContext &Ctx = MCOS->getContext();
  auto &ProbeTable = Ctx.getMCPseudoProbeTable();

  // Bail out early so we don't switch to the pseudo_probe section needlessly
  // and in doing so create an unnecessary (if empty) section.
  auto &ProbeSections = ProbeTable.getProbeSections();
  if (ProbeSections.empty())
    return;

  LLVM_DEBUG(MCPseudoProbeTable::DdgPrintIndent = 0);

  // Put out the probe.
  ProbeSections.emit(MCOS);
}

static StringRef getProbeFNameForGUID(const GUIDProbeFunctionMap &GUID2FuncMAP,
                                      uint64_t GUID) {
  auto It = GUID2FuncMAP.find(GUID);
  assert(It != GUID2FuncMAP.end() &&
         "Probe function must exist for a valid GUID");
  return It->second.FuncName;
}

void MCPseudoProbeFuncDesc::print(raw_ostream &OS) {
  OS << "GUID: " << FuncGUID << " Name: " << FuncName << "\n";
  OS << "Hash: " << FuncHash << "\n";
}

void MCDecodedPseudoProbe::getInlineContext(
    SmallVectorImpl<std::string> &ContextStack,
    const GUIDProbeFunctionMap &GUID2FuncMAP, bool ShowName) const {
  uint32_t Begin = ContextStack.size();
  MCDecodedPseudoProbeInlineTree *Cur = InlineTree;
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
    Cur = static_cast<MCDecodedPseudoProbeInlineTree *>(Cur->Parent);
  }
  // Make the ContextStack in caller-callee order
  std::reverse(ContextStack.begin() + Begin, ContextStack.end());
}

std::string MCDecodedPseudoProbe::getInlineContextStr(
    const GUIDProbeFunctionMap &GUID2FuncMAP, bool ShowName) const {
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

void MCDecodedPseudoProbe::print(raw_ostream &OS,
                                 const GUIDProbeFunctionMap &GUID2FuncMAP,
                                 bool ShowName) const {
  OS << "FUNC: ";
  if (ShowName) {
    StringRef FuncName = getProbeFNameForGUID(GUID2FuncMAP, Guid);
    OS << FuncName.str() << " ";
  } else {
    OS << Guid << " ";
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

template <typename T> ErrorOr<T> MCPseudoProbeDecoder::readUnencodedNumber() {
  if (Data + sizeof(T) > End) {
    return std::error_code();
  }
  T Val = endian::readNext<T, little, unaligned>(Data);
  return ErrorOr<T>(Val);
}

template <typename T> ErrorOr<T> MCPseudoProbeDecoder::readUnsignedNumber() {
  unsigned NumBytesRead = 0;
  uint64_t Val = decodeULEB128(Data, &NumBytesRead);
  if (Val > std::numeric_limits<T>::max() || (Data + NumBytesRead > End)) {
    return std::error_code();
  }
  Data += NumBytesRead;
  return ErrorOr<T>(static_cast<T>(Val));
}

template <typename T> ErrorOr<T> MCPseudoProbeDecoder::readSignedNumber() {
  unsigned NumBytesRead = 0;
  int64_t Val = decodeSLEB128(Data, &NumBytesRead);
  if (Val > std::numeric_limits<T>::max() || (Data + NumBytesRead > End)) {
    return std::error_code();
  }
  Data += NumBytesRead;
  return ErrorOr<T>(static_cast<T>(Val));
}

ErrorOr<StringRef> MCPseudoProbeDecoder::readString(uint32_t Size) {
  StringRef Str(reinterpret_cast<const char *>(Data), Size);
  if (Data + Size > End) {
    return std::error_code();
  }
  Data += Size;
  return ErrorOr<StringRef>(Str);
}

bool MCPseudoProbeDecoder::buildGUID2FuncDescMap(const uint8_t *Start,
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

  Data = Start;
  End = Data + Size;

  while (Data < End) {
    auto ErrorOrGUID = readUnencodedNumber<uint64_t>();
    if (!ErrorOrGUID)
      return false;

    auto ErrorOrHash = readUnencodedNumber<uint64_t>();
    if (!ErrorOrHash)
      return false;

    auto ErrorOrNameSize = readUnsignedNumber<uint32_t>();
    if (!ErrorOrNameSize)
      return false;
    uint32_t NameSize = std::move(*ErrorOrNameSize);

    auto ErrorOrName = readString(NameSize);
    if (!ErrorOrName)
      return false;

    uint64_t GUID = std::move(*ErrorOrGUID);
    uint64_t Hash = std::move(*ErrorOrHash);
    StringRef Name = std::move(*ErrorOrName);

    // Initialize PseudoProbeFuncDesc and populate it into GUID2FuncDescMap
    GUID2FuncDescMap.emplace(GUID, MCPseudoProbeFuncDesc(GUID, Hash, Name));
  }
  assert(Data == End && "Have unprocessed data in pseudo_probe_desc section");
  return true;
}

bool MCPseudoProbeDecoder::buildAddress2ProbeMap(const uint8_t *Start,
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
  //             1 - tail call, 2 - dangling
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

  Data = Start;
  End = Data + Size;

  MCDecodedPseudoProbeInlineTree *Root = &DummyInlineRoot;
  MCDecodedPseudoProbeInlineTree *Cur = &DummyInlineRoot;
  uint64_t LastAddr = 0;
  uint32_t Index = 0;
  // A DFS-based decoding
  while (Data < End) {
    if (Root == Cur) {
      // Use a sequential id for top level inliner.
      Index = Root->getChildren().size();
    } else {
      // Read inline site for inlinees
      auto ErrorOrIndex = readUnsignedNumber<uint32_t>();
      if (!ErrorOrIndex)
        return false;
      Index = std::move(*ErrorOrIndex);
    }
    // Switch/add to a new tree node(inlinee)
    Cur = Cur->getOrAddNode(std::make_tuple(Cur->Guid, Index));
    // Read guid
    auto ErrorOrCurGuid = readUnencodedNumber<uint64_t>();
    if (!ErrorOrCurGuid)
      return false;
    Cur->Guid = std::move(*ErrorOrCurGuid);
    // Read number of probes in the current node.
    auto ErrorOrNodeCount = readUnsignedNumber<uint32_t>();
    if (!ErrorOrNodeCount)
      return false;
    uint32_t NodeCount = std::move(*ErrorOrNodeCount);
    // Read number of direct inlinees
    auto ErrorOrCurChildrenToProcess = readUnsignedNumber<uint32_t>();
    if (!ErrorOrCurChildrenToProcess)
      return false;
    Cur->ChildrenToProcess = std::move(*ErrorOrCurChildrenToProcess);
    // Read all probes in this node
    for (std::size_t I = 0; I < NodeCount; I++) {
      // Read index
      auto ErrorOrIndex = readUnsignedNumber<uint32_t>();
      if (!ErrorOrIndex)
        return false;
      uint32_t Index = std::move(*ErrorOrIndex);
      // Read type | flag.
      auto ErrorOrValue = readUnencodedNumber<uint8_t>();
      if (!ErrorOrValue)
        return false;
      uint8_t Value = std::move(*ErrorOrValue);
      uint8_t Kind = Value & 0xf;
      uint8_t Attr = (Value & 0x70) >> 4;
      // Read address
      uint64_t Addr = 0;
      if (Value & 0x80) {
        auto ErrorOrOffset = readSignedNumber<int64_t>();
        if (!ErrorOrOffset)
          return false;
        int64_t Offset = std::move(*ErrorOrOffset);
        Addr = LastAddr + Offset;
      } else {
        auto ErrorOrAddr = readUnencodedNumber<int64_t>();
        if (!ErrorOrAddr)
          return false;
        Addr = std::move(*ErrorOrAddr);
      }
      // Populate Address2ProbesMap
      auto &Probes = Address2ProbesMap[Addr];
      Probes.emplace_back(Addr, Cur->Guid, Index, PseudoProbeType(Kind), Attr,
                          Cur);
      Cur->addProbes(&Probes.back());
      LastAddr = Addr;
    }

    // Look for the parent for the next node by subtracting the current
    // node count from tree counts along the parent chain. The first node
    // in the chain that has a non-zero tree count is the target.
    while (Cur != Root) {
      if (Cur->ChildrenToProcess == 0) {
        Cur = static_cast<MCDecodedPseudoProbeInlineTree *>(Cur->Parent);
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
  return true;
}

void MCPseudoProbeDecoder::printGUID2FuncDescMap(raw_ostream &OS) {
  OS << "Pseudo Probe Desc:\n";
  // Make the output deterministic
  std::map<uint64_t, MCPseudoProbeFuncDesc> OrderedMap(GUID2FuncDescMap.begin(),
                                                       GUID2FuncDescMap.end());
  for (auto &I : OrderedMap) {
    I.second.print(OS);
  }
}

void MCPseudoProbeDecoder::printProbeForAddress(raw_ostream &OS,
                                                uint64_t Address) {
  auto It = Address2ProbesMap.find(Address);
  if (It != Address2ProbesMap.end()) {
    for (auto &Probe : It->second) {
      OS << " [Probe]:\t";
      Probe.print(OS, GUID2FuncDescMap, true);
    }
  }
}

void MCPseudoProbeDecoder::printProbesForAllAddresses(raw_ostream &OS) {
  std::vector<uint64_t> Addresses;
  for (auto Entry : Address2ProbesMap)
    Addresses.push_back(Entry.first);
  std::sort(Addresses.begin(), Addresses.end());
  for (auto K : Addresses) {
    OS << "Address:\t";
    OS << K;
    OS << "\n";
    printProbeForAddress(OS, K);
  }
}

const MCDecodedPseudoProbe *
MCPseudoProbeDecoder::getCallProbeForAddr(uint64_t Address) const {
  auto It = Address2ProbesMap.find(Address);
  if (It == Address2ProbesMap.end())
    return nullptr;
  const auto &Probes = It->second;

  const MCDecodedPseudoProbe *CallProbe = nullptr;
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

const MCPseudoProbeFuncDesc *
MCPseudoProbeDecoder::getFuncDescForGUID(uint64_t GUID) const {
  auto It = GUID2FuncDescMap.find(GUID);
  assert(It != GUID2FuncDescMap.end() && "Function descriptor doesn't exist");
  return &It->second;
}

void MCPseudoProbeDecoder::getInlineContextForProbe(
    const MCDecodedPseudoProbe *Probe,
    SmallVectorImpl<std::string> &InlineContextStack, bool IncludeLeaf) const {
  Probe->getInlineContext(InlineContextStack, GUID2FuncDescMap, true);
  if (!IncludeLeaf)
    return;
  // Note that the context from probe doesn't include leaf frame,
  // hence we need to retrieve and prepend leaf if requested.
  const auto *FuncDesc = getFuncDescForGUID(Probe->getGuid());
  InlineContextStack.emplace_back(FuncDesc->FuncName + ":" +
                                  Twine(Probe->getIndex()).str());
}

const MCPseudoProbeFuncDesc *MCPseudoProbeDecoder::getInlinerDescForProbe(
    const MCDecodedPseudoProbe *Probe) const {
  MCDecodedPseudoProbeInlineTree *InlinerNode = Probe->getInlineTreeNode();
  if (!InlinerNode->hasInlineSite())
    return nullptr;
  return getFuncDescForGUID(std::get<0>(InlinerNode->ISite));
}
