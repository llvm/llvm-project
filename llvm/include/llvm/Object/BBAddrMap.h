//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares common types and utilities for basic-block address maps.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_BBADDRMAP_H
#define LLVM_OBJECT_BBADDRMAP_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/BlockFrequency.h"
#include "llvm/Support/BranchProbability.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/UniqueBBID.h"

namespace llvm {
namespace object {

// Struct representing the BBAddrMap for one function.
struct BBAddrMap {

  // Bitfield of optional features to control the extra information
  // emitted/encoded in the section.
  struct Features {
    bool FuncEntryCount : 1;
    bool BBFreq : 1;
    bool BrProb : 1;
    bool MultiBBRange : 1;
    bool OmitBBEntries : 1;
    bool CallsiteEndOffsets : 1;
    bool BBHash : 1;
    bool PostLinkCfg : 1;

    bool hasPGOAnalysis() const { return FuncEntryCount || BBFreq || BrProb; }

    bool hasPGOAnalysisBBData() const { return BBFreq || BrProb; }

    // Encodes to minimum bit width representation.
    uint16_t encode() const {
      return (static_cast<uint16_t>(FuncEntryCount) << 0) |
             (static_cast<uint16_t>(BBFreq) << 1) |
             (static_cast<uint16_t>(BrProb) << 2) |
             (static_cast<uint16_t>(MultiBBRange) << 3) |
             (static_cast<uint16_t>(OmitBBEntries) << 4) |
             (static_cast<uint16_t>(CallsiteEndOffsets) << 5) |
             (static_cast<uint16_t>(BBHash) << 6) |
             (static_cast<uint16_t>(PostLinkCfg) << 7);
    }

    // Decodes from minimum bit width representation and validates no
    // unnecessary bits are used.
    static Expected<Features> decode(uint16_t Val) {
      Features Feat{
          static_cast<bool>(Val & (1 << 0)), static_cast<bool>(Val & (1 << 1)),
          static_cast<bool>(Val & (1 << 2)), static_cast<bool>(Val & (1 << 3)),
          static_cast<bool>(Val & (1 << 4)), static_cast<bool>(Val & (1 << 5)),
          static_cast<bool>(Val & (1 << 6)), static_cast<bool>(Val & (1 << 7))};
      if (Feat.encode() != Val)
        return createStringError(
            "invalid encoding for BBAddrMap::Features: 0x%x", Val);
      return Feat;
    }

    bool operator==(const Features &Other) const {
      return std::tie(FuncEntryCount, BBFreq, BrProb, MultiBBRange,
                      OmitBBEntries, CallsiteEndOffsets, BBHash, PostLinkCfg) ==
             std::tie(Other.FuncEntryCount, Other.BBFreq, Other.BrProb,
                      Other.MultiBBRange, Other.OmitBBEntries,
                      Other.CallsiteEndOffsets, Other.BBHash,
                      Other.PostLinkCfg);
    }
  };

  // Struct representing the BBAddrMap information for one basic block.
  struct BBEntry {
    struct Metadata {
      bool HasReturn : 1;         // If this block ends with a return (or tail
                                  // call).
      bool HasTailCall : 1;       // If this block ends with a tail call.
      bool IsEHPad : 1;           // If this is an exception handling block.
      bool CanFallThrough : 1;    // If this block can fall through to its next.
      bool HasIndirectBranch : 1; // If this block ends with an indirect branch
                                  // (branch via a register).

      bool operator==(const Metadata &Other) const {
        return HasReturn == Other.HasReturn &&
               HasTailCall == Other.HasTailCall && IsEHPad == Other.IsEHPad &&
               CanFallThrough == Other.CanFallThrough &&
               HasIndirectBranch == Other.HasIndirectBranch;
      }

      // Encodes this struct as a uint32_t value.
      uint32_t encode() const {
        return static_cast<uint32_t>(HasReturn) |
               (static_cast<uint32_t>(HasTailCall) << 1) |
               (static_cast<uint32_t>(IsEHPad) << 2) |
               (static_cast<uint32_t>(CanFallThrough) << 3) |
               (static_cast<uint32_t>(HasIndirectBranch) << 4);
      }

      // Decodes and returns a Metadata struct from a uint32_t value.
      static Expected<Metadata> decode(uint32_t V) {
        Metadata MD{/*HasReturn=*/static_cast<bool>(V & 1),
                    /*HasTailCall=*/static_cast<bool>(V & (1 << 1)),
                    /*IsEHPad=*/static_cast<bool>(V & (1 << 2)),
                    /*CanFallThrough=*/static_cast<bool>(V & (1 << 3)),
                    /*HasIndirectBranch=*/static_cast<bool>(V & (1 << 4))};
        if (MD.encode() != V)
          return createStringError(
              "invalid encoding for BBEntry::Metadata: 0x%x", V);
        return MD;
      }
    };

    uint32_t ID = 0;     // Unique ID of this basic block.
    uint32_t Offset = 0; // Offset of basic block relative to the base address.
    uint32_t Size = 0;   // Size of the basic block.
    Metadata MD = {false, false, false, false,
                   false}; // Metadata for this basic block.
    // Offsets of end of call instructions, relative to the basic block start.
    SmallVector<uint32_t, 1> CallsiteEndOffsets;
    uint64_t Hash = 0; // Hash for this basic block.

    BBEntry(uint32_t ID, uint32_t Offset, uint32_t Size, Metadata MD,
            SmallVector<uint32_t, 1> CallsiteEndOffsets, uint64_t Hash)
        : ID(ID), Offset(Offset), Size(Size), MD(MD),
          CallsiteEndOffsets(std::move(CallsiteEndOffsets)), Hash(Hash) {}

    UniqueBBID getID() const { return {ID, 0}; }

    bool operator==(const BBEntry &Other) const {
      return ID == Other.ID && Offset == Other.Offset && Size == Other.Size &&
             MD == Other.MD && CallsiteEndOffsets == Other.CallsiteEndOffsets &&
             Hash == Other.Hash;
    }

    bool hasReturn() const { return MD.HasReturn; }
    bool hasTailCall() const { return MD.HasTailCall; }
    bool isEHPad() const { return MD.IsEHPad; }
    bool canFallThrough() const { return MD.CanFallThrough; }
    bool hasIndirectBranch() const { return MD.HasIndirectBranch; }
  };

  // Struct representing the BBAddrMap information for a contiguous range of
  // basic blocks (a function or a basic block section).
  struct BBRangeEntry {
    uint64_t BaseAddress = 0;       // Base address of the range.
    std::vector<BBEntry> BBEntries; // Basic block entries for this range.

    // Equality operator for unit testing.
    bool operator==(const BBRangeEntry &Other) const {
      return BaseAddress == Other.BaseAddress && BBEntries == Other.BBEntries;
    }
  };

  // All ranges for this function. Cannot be empty. The first range always
  // corresponds to the function entry.
  std::vector<BBRangeEntry> BBRanges;

  // Returns the function address associated with this BBAddrMap, which is
  // stored as the `BaseAddress` of its first BBRangeEntry.
  uint64_t getFunctionAddress() const {
    assert(!BBRanges.empty());
    return BBRanges.front().BaseAddress;
  }

  // Returns the total number of bb entries in all bb ranges.
  size_t getNumBBEntries() const {
    size_t NumBBEntries = 0;
    for (const auto &BBR : BBRanges)
      NumBBEntries += BBR.BBEntries.size();
    return NumBBEntries;
  }

  // Returns the index of the bb range with the given base address, or
  // `std::nullopt` if no such range exists.
  std::optional<size_t>
  getBBRangeIndexForBaseAddress(uint64_t BaseAddress) const {
    for (size_t I = 0; I < BBRanges.size(); ++I)
      if (BBRanges[I].BaseAddress == BaseAddress)
        return I;
    return {};
  }

  // Returns bb entries in the first range.
  const std::vector<BBEntry> &getBBEntries() const {
    return BBRanges.front().BBEntries;
  }

  const std::vector<BBRangeEntry> &getBBRanges() const { return BBRanges; }

  // Equality operator for unit testing.
  bool operator==(const BBAddrMap &Other) const {
    return BBRanges == Other.BBRanges;
  }
};

/// A feature extension of BBAddrMap that holds information relevant to PGO.
struct PGOAnalysisMap {
  /// Extra basic block data with fields for block frequency and branch
  /// probability.
  struct PGOBBEntry {
    /// Single successor of a given basic block that contains the tag and branch
    /// probability associated with it.
    struct SuccessorEntry {
      /// Unique ID of this successor basic block.
      uint32_t ID = 0;
      /// Branch Probability of the edge to this successor taken from MBPI.
      BranchProbability Prob;
      /// Raw edge count from the post link profile (e.g., from bolt or
      /// propeller).
      uint64_t PostLinkFreq = 0;

      bool operator==(const SuccessorEntry &Other) const {
        return std::tie(ID, Prob, PostLinkFreq) ==
               std::tie(Other.ID, Other.Prob, Other.PostLinkFreq);
      }
    };

    /// Block frequency taken from MBFI
    BlockFrequency BlockFreq;
    /// Raw block count taken from the post link profile (e.g., from bolt or
    /// propeller).
    uint64_t PostLinkBlockFreq = 0;
    /// List of successors of the current block
    llvm::SmallVector<SuccessorEntry, 2> Successors;

    bool operator==(const PGOBBEntry &Other) const {
      return std::tie(BlockFreq, PostLinkBlockFreq, Successors) ==
             std::tie(Other.BlockFreq, Other.PostLinkBlockFreq,
                      Other.Successors);
    }
  };

  uint64_t FuncEntryCount;           // Prof count from IR function
  std::vector<PGOBBEntry> BBEntries; // Extended basic block entries

  // Flags to indicate if each PGO related info was enabled in this function
  BBAddrMap::Features FeatEnable;

  bool operator==(const PGOAnalysisMap &Other) const {
    return std::tie(FuncEntryCount, BBEntries, FeatEnable) ==
           std::tie(Other.FuncEntryCount, Other.BBEntries, Other.FeatEnable);
  }
};

/// Extracts addresses from a data stream.
/// The base implementation reads the address directly.
/// Subclasses can override to handle format-specific details such as relocation
/// resolution.
class AddressExtractor {
  const DataExtractor &Data;
  unsigned AddressSize;

public:
  AddressExtractor(const DataExtractor &Data, unsigned AddressSize)
      : Data(Data), AddressSize(AddressSize) {}

  virtual ~AddressExtractor() = default;

  const DataExtractor &getDataExtractor() const { return Data; }

  /// Extract and resolve an address at the current \p Cur position.
  virtual Expected<uint64_t> extractAddress(DataExtractor::Cursor &Cur) {
    uint64_t Address = Data.getUnsigned(Cur, AddressSize);
    if (!Cur)
      return Cur.takeError();
    return Address;
  }
};

/// Decodes one BB address map section payload.
///
/// \p Extractor provides address extraction and the underlying DataExtractor.
/// \p PGOAnalyses if non-null, receives the decoded PGO analysis data. On
///   error, \p PGOAnalyses may be partially populated.
Expected<std::vector<BBAddrMap>>
decodeBBAddrMapPayload(AddressExtractor &Extractor,
                       std::vector<PGOAnalysisMap> *PGOAnalyses = nullptr);

} // end namespace object.
} // end namespace llvm.

#endif // LLVM_OBJECT_BBADDRMAP_H
