//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements shared utilities for basic-block address maps.
///
//===----------------------------------------------------------------------===//

#include "llvm/Object/BBAddrMap.h"
#include "llvm/Object/Error.h"

using namespace llvm;
using namespace object;

namespace {

// Helper to extract and decode the next ULEB128 value as an unsigned integer
// type. Returns zero and sets ULEBSizeErr if the ULEB128 value exceeds the
// destination type's limit.
// Also returns zero if ULEBSizeErr is already in an error state.
// ULEBSizeErr is an out variable if an error occurs.
template <typename IntTy, std::enable_if_t<std::is_unsigned_v<IntTy>, int> = 0>
static IntTy readULEB128As(const DataExtractor &Data,
                           DataExtractor::Cursor &Cur, Error &ULEBSizeErr) {
  // Bail out and do not extract data if ULEBSizeErr is already set.
  if (ULEBSizeErr)
    return 0;
  uint64_t Offset = Cur.tell();
  uint64_t Value = Data.getULEB128(Cur);
  if (Value > std::numeric_limits<IntTy>::max()) {
    ULEBSizeErr = createError("ULEB128 value at offset 0x" +
                              Twine::utohexstr(Offset) + " exceeds UINT" +
                              Twine(std::numeric_limits<IntTy>::digits) +
                              "_MAX (0x" + Twine::utohexstr(Value) + ")");
    return 0;
  }
  return static_cast<IntTy>(Value);
}
} // end anonymous namespace

Expected<std::vector<BBAddrMap>>
llvm::object::decodeBBAddrMapPayload(AddressExtractor &Extractor,
                                     std::vector<PGOAnalysisMap> *PGOAnalyses) {
  const DataExtractor &Data = Extractor.getDataExtractor();
  std::vector<BBAddrMap> FunctionEntries;

  DataExtractor::Cursor Cur(0);
  Error ULEBSizeErr = Error::success();
  Error MetadataDecodeErr = Error::success();

  // Use int for Version to avoid Twine treating uint8_t as char.
  int Version = 0;
  uint16_t Feature = 0;
  BBAddrMap::Features FeatEnable{};
  while (!ULEBSizeErr && !MetadataDecodeErr && Cur &&
         Cur.tell() < Data.getData().size()) {
    Version = Data.getU8(Cur);
    if (!Cur)
      break;
    if (Version < 2 || Version > 5)
      return createError("unsupported BB address map version: " +
                         Twine(Version));
    Feature = Version < 5 ? Data.getU8(Cur) : Data.getU16(Cur);
    if (!Cur)
      break;
    auto FeatEnableOrErr = BBAddrMap::Features::decode(Feature);
    if (!FeatEnableOrErr)
      return FeatEnableOrErr.takeError();
    FeatEnable = *FeatEnableOrErr;
    if (FeatEnable.CallsiteEndOffsets && Version < 3)
      return createError("version should be >= 3 for BB address map when "
                         "callsite offsets feature is enabled: version = " +
                         Twine(Version) + " feature = " + Twine(Feature));
    if (FeatEnable.BBHash && Version < 4)
      return createError("version should be >= 4 for BB address map when "
                         "basic block hash feature is enabled: version = " +
                         Twine(Version) + " feature = " + Twine(Feature));
    if (FeatEnable.PostLinkCfg && Version < 5)
      return createError("version should be >= 5 for BB address map when "
                         "post link cfg feature is enabled: version = " +
                         Twine(Version) + " feature = " + Twine(Feature));
    uint32_t NumBlocksInBBRange = 0;
    uint32_t NumBBRanges = 1;
    uint64_t RangeBaseAddress = 0;
    if (FeatEnable.MultiBBRange) {
      NumBBRanges = readULEB128As<uint32_t>(Data, Cur, ULEBSizeErr);
      if (!Cur || ULEBSizeErr)
        break;
      if (!NumBBRanges)
        return createError("invalid zero number of BB ranges at offset " +
                           Twine::utohexstr(Cur.tell()));
    } else {
      auto AddressOrErr = Extractor.extractAddress(Cur);
      if (!AddressOrErr)
        return AddressOrErr.takeError();
      RangeBaseAddress = *AddressOrErr;
      NumBlocksInBBRange = readULEB128As<uint32_t>(Data, Cur, ULEBSizeErr);
    }
    std::vector<BBAddrMap::BBRangeEntry> BBRangeEntries;
    uint32_t TotalNumBlocks = 0;
    for (uint32_t BBRangeIndex = 0; BBRangeIndex < NumBBRanges;
         ++BBRangeIndex) {
      uint32_t PrevBBEndOffset = 0;
      if (FeatEnable.MultiBBRange) {
        auto AddressOrErr = Extractor.extractAddress(Cur);
        if (!AddressOrErr)
          return AddressOrErr.takeError();
        RangeBaseAddress = *AddressOrErr;
        NumBlocksInBBRange = readULEB128As<uint32_t>(Data, Cur, ULEBSizeErr);
      }
      std::vector<BBAddrMap::BBEntry> BBEntries;
      if (!FeatEnable.OmitBBEntries) {
        for (uint32_t BlockIndex = 0; !MetadataDecodeErr && !ULEBSizeErr &&
                                      Cur && (BlockIndex < NumBlocksInBBRange);
             ++BlockIndex) {
          uint32_t ID = readULEB128As<uint32_t>(Data, Cur, ULEBSizeErr);
          uint32_t Offset = readULEB128As<uint32_t>(Data, Cur, ULEBSizeErr);
          // Read the callsite offsets.
          uint32_t LastCallsiteEndOffset = 0;
          SmallVector<uint32_t, 1> CallsiteEndOffsets;
          if (FeatEnable.CallsiteEndOffsets) {
            uint32_t NumCallsites =
                readULEB128As<uint32_t>(Data, Cur, ULEBSizeErr);
            CallsiteEndOffsets.reserve(NumCallsites);
            for (uint32_t CallsiteIndex = 0;
                 !ULEBSizeErr && Cur && (CallsiteIndex < NumCallsites);
                 ++CallsiteIndex) {
              LastCallsiteEndOffset +=
                  readULEB128As<uint32_t>(Data, Cur, ULEBSizeErr);
              CallsiteEndOffsets.push_back(LastCallsiteEndOffset);
            }
          }
          uint32_t Size = readULEB128As<uint32_t>(Data, Cur, ULEBSizeErr) +
                          LastCallsiteEndOffset;
          uint32_t MD = readULEB128As<uint32_t>(Data, Cur, ULEBSizeErr);
          uint64_t Hash = FeatEnable.BBHash ? Data.getU64(Cur) : 0;
          Expected<BBAddrMap::BBEntry::Metadata> MetadataOrErr =
              BBAddrMap::BBEntry::Metadata::decode(MD);
          if (!MetadataOrErr) {
            MetadataDecodeErr = MetadataOrErr.takeError();
            break;
          }
          BBEntries.push_back({ID, Offset + PrevBBEndOffset, Size,
                               *MetadataOrErr, CallsiteEndOffsets, Hash});
          PrevBBEndOffset += Offset + Size;
        }
        TotalNumBlocks += BBEntries.size();
      }
      BBRangeEntries.push_back({RangeBaseAddress, std::move(BBEntries)});
    }
    FunctionEntries.push_back({std::move(BBRangeEntries)});

    if (PGOAnalyses || FeatEnable.hasPGOAnalysis()) {
      // Function entry count
      uint64_t FuncEntryCount =
          FeatEnable.FuncEntryCount
              ? readULEB128As<uint64_t>(Data, Cur, ULEBSizeErr)
              : 0;

      std::vector<PGOAnalysisMap::PGOBBEntry> PGOBBEntries;
      for (uint32_t BlockIndex = 0;
           FeatEnable.hasPGOAnalysisBBData() && !MetadataDecodeErr &&
           !ULEBSizeErr && Cur && (BlockIndex < TotalNumBlocks);
           ++BlockIndex) {
        // Block frequency
        uint64_t BBF = FeatEnable.BBFreq
                           ? readULEB128As<uint64_t>(Data, Cur, ULEBSizeErr)
                           : 0;
        uint32_t PostLinkBBFreq =
            FeatEnable.PostLinkCfg
                ? readULEB128As<uint32_t>(Data, Cur, ULEBSizeErr)
                : 0;

        // Branch probability
        llvm::SmallVector<PGOAnalysisMap::PGOBBEntry::SuccessorEntry, 2>
            Successors;
        if (FeatEnable.BrProb) {
          auto SuccCount = readULEB128As<uint64_t>(Data, Cur, ULEBSizeErr);
          for (uint64_t I = 0; I < SuccCount; ++I) {
            uint32_t BBID = readULEB128As<uint32_t>(Data, Cur, ULEBSizeErr);
            uint32_t BrProb = readULEB128As<uint32_t>(Data, Cur, ULEBSizeErr);
            uint32_t PostLinkFreq =
                FeatEnable.PostLinkCfg
                    ? readULEB128As<uint32_t>(Data, Cur, ULEBSizeErr)
                    : 0;

            if (PGOAnalyses)
              Successors.push_back(
                  {BBID, BranchProbability::getRaw(BrProb), PostLinkFreq});
          }
        }

        if (PGOAnalyses)
          PGOBBEntries.push_back(
              {BlockFrequency(BBF), PostLinkBBFreq, std::move(Successors)});
      }

      if (PGOAnalyses)
        PGOAnalyses->push_back(
            {FuncEntryCount, std::move(PGOBBEntries), FeatEnable});
    }
  }

  // Either Cur is in the error state, or we have an error in ULEBSizeErr or
  // MetadataDecodeErr (but not both), but we join all errors here to be safe.
  if (!Cur || ULEBSizeErr || MetadataDecodeErr)
    return joinErrors(joinErrors(Cur.takeError(), std::move(ULEBSizeErr)),
                      std::move(MetadataDecodeErr));
  return FunctionEntries;
}
