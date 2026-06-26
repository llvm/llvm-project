//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Shared yaml2obj BBAddrMap writer, in a standalone header so type-only
/// includers don't pull in its dependencies.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECTYAML_BBADDRMAPYAMLEMITTER_H
#define LLVM_OBJECTYAML_BBADDRMAPYAMLEMITTER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/bit.h"
#include "llvm/Object/BBAddrMap.h"
#include "llvm/ObjectYAML/BBAddrMapYAML.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/WithColor.h"
#include <cassert>
#include <cstdint>
#include <vector>

namespace llvm {
namespace BBAddrMapYAML {

/// CRTP interface for emitting the BBAddrMap payload to a target-specific
/// writer, so the encode logic can be shared across formats.
template <typename Derived> class Writer {
  llvm::endianness Endian;
  unsigned AddressSize;

  Derived &derived() { return static_cast<Derived &>(*this); }

public:
  Writer(llvm::endianness Endian, unsigned AddressSize)
      : Endian(Endian), AddressSize(AddressSize) {
    assert((AddressSize == 4 || AddressSize == 8) && "invalid address size");
  }

  template <typename T> void writeInt(T Val) {
    derived().template emitInt<T>(Val, Endian);
  }

  // Pointer-sized: 4 or 8 bytes per AddressSize.
  void writeAddress(uint64_t Val) {
    if (AddressSize == 8)
      writeInt<uint64_t>(Val);
    else
      writeInt<uint32_t>(static_cast<uint32_t>(Val));
  }

  void writeULEB128(uint64_t Val) { derived().emitULEB128(Val); }
};

/// Encodes the BBAddrMap section body through \p W, walking \p PGOAnalyses
/// alongside \p Entries when non-null. Warns and continues on malformed YAML,
/// so yaml2obj can still emit intentionally-broken test objects.
template <typename Derived>
void encodePayload(ArrayRef<BBAddrMapEntry> Entries,
                   const std::vector<PGOAnalysisMapEntry> *PGOAnalyses,
                   Writer<Derived> &W) {
  for (const auto &[Idx, E] : llvm::enumerate(Entries)) {
    // Write version and feature values.
    if (E.Version > 5)
      WithColor::warning() << "unsupported BB address map version: "
                           << static_cast<int>(E.Version)
                           << "; encoding using the most recent version";
    W.template writeInt<uint8_t>(E.Version);
    if (E.Version < 5)
      W.template writeInt<uint8_t>(static_cast<uint8_t>(E.Feature));
    else
      W.template writeInt<uint16_t>(E.Feature);
    auto FeatureOrErr = llvm::object::BBAddrMap::Features::decode(E.Feature);
    if (!FeatureOrErr) {
      // Invalid feature: warn and skip the entry.
      WithColor::warning() << toString(FeatureOrErr.takeError());
      continue;
    }
    bool MultiBBRangeFeatureEnabled = FeatureOrErr->MultiBBRange;
    bool MultiBBRange =
        MultiBBRangeFeatureEnabled ||
        (E.NumBBRanges.has_value() && E.NumBBRanges.value() != 1) ||
        (E.BBRanges && E.BBRanges->size() != 1);
    if (MultiBBRange && !MultiBBRangeFeatureEnabled)
      WithColor::warning() << "feature value(" << E.Feature
                           << ") does not support multiple BB ranges.";
    if (MultiBBRange) {
      // Write the number of basic block ranges, which is overridden by the
      // 'NumBBRanges' field when specified.
      uint64_t NumBBRanges =
          E.NumBBRanges.value_or(E.BBRanges ? E.BBRanges->size() : 0);
      W.writeULEB128(NumBBRanges);
    }
    if (!E.BBRanges)
      continue;
    uint64_t TotalNumBlocks = 0;
    bool EmitCallsiteEndOffsets =
        FeatureOrErr->CallsiteEndOffsets || E.hasAnyCallsiteEndOffsets();
    for (const BBAddrMapEntry::BBRangeEntry &BBR : *E.BBRanges) {
      // Write the base address of the range.
      W.writeAddress(BBR.BaseAddress);
      // Write number of BBEntries (number of basic blocks in this basic block
      // range). This is overridden by the 'NumBlocks' YAML field when
      // specified.
      uint64_t NumBlocks =
          BBR.NumBlocks.value_or(BBR.BBEntries ? BBR.BBEntries->size() : 0);
      W.writeULEB128(NumBlocks);
      // Write all BBEntries in this BBRange.
      if (!BBR.BBEntries || FeatureOrErr->OmitBBEntries)
        continue;
      for (const BBAddrMapEntry::BBEntry &BBE : *BBR.BBEntries) {
        ++TotalNumBlocks;
        if (E.Version > 1)
          W.writeULEB128(BBE.ID);
        W.writeULEB128(BBE.AddressOffset);
        if (EmitCallsiteEndOffsets) {
          size_t NumCallsiteEndOffsets =
              BBE.CallsiteEndOffsets ? BBE.CallsiteEndOffsets->size() : 0;
          W.writeULEB128(NumCallsiteEndOffsets);
          if (BBE.CallsiteEndOffsets) {
            for (uint32_t Offset : *BBE.CallsiteEndOffsets)
              W.writeULEB128(Offset);
          }
        }
        W.writeULEB128(BBE.Size);
        W.writeULEB128(BBE.Metadata);
        if (FeatureOrErr->BBHash || BBE.Hash.has_value()) {
          uint64_t Hash =
              BBE.Hash.has_value() ? BBE.Hash.value() : llvm::yaml::Hex64(0);
          W.template writeInt<uint64_t>(Hash);
        }
      }
    }
    if (!PGOAnalyses)
      continue;
    const PGOAnalysisMapEntry &PGOEntry = PGOAnalyses->at(Idx);

    if (PGOEntry.FuncEntryCount)
      W.writeULEB128(*PGOEntry.FuncEntryCount);

    if (!PGOEntry.PGOBBEntries)
      continue;

    const auto &PGOBBEntries = PGOEntry.PGOBBEntries.value();
    if (TotalNumBlocks != PGOBBEntries.size()) {
      WithColor::warning() << "PGOBBEntries must be the same length as "
                              "BBEntries in the BB address map.\n"
                           << "Mismatch on function with address: "
                           << E.getFunctionAddress();
      continue;
    }

    for (const auto &PGOBBE : PGOBBEntries) {
      if (PGOBBE.BBFreq)
        W.writeULEB128(*PGOBBE.BBFreq);
      if (FeatureOrErr->PostLinkCfg || PGOBBE.PostLinkBBFreq.has_value())
        W.writeULEB128(PGOBBE.PostLinkBBFreq.value_or(0));
      if (PGOBBE.Successors) {
        W.writeULEB128(PGOBBE.Successors->size());
        for (const auto &[ID, BrProb, PostLinkBrFreq] : *PGOBBE.Successors) {
          W.writeULEB128(ID);
          W.writeULEB128(BrProb);
          if (FeatureOrErr->PostLinkCfg || PostLinkBrFreq.has_value())
            W.writeULEB128(PostLinkBrFreq.value_or(0));
        }
      }
    }
  }
}

} // end namespace BBAddrMapYAML
} // end namespace llvm

#endif // LLVM_OBJECTYAML_BBADDRMAPYAMLEMITTER_H
