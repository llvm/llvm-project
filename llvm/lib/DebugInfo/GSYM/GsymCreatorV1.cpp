//===- GsymCreatorV1.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/GSYM/GsymCreatorV1.h"
#include "llvm/DebugInfo/GSYM/FileWriter.h"
#include "llvm/DebugInfo/GSYM/Header.h"
#include "llvm/DebugInfo/GSYM/LineTable.h"
#include "llvm/DebugInfo/GSYM/OutputAggregator.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <vector>

using namespace llvm;
using namespace gsym;

GsymCreatorV1::GsymCreatorV1(bool Quiet)
    : StrTab(StringTableBuilder::ELF), Quiet(Quiet) {
  insertFile(StringRef());
}

uint32_t GsymCreatorV1::insertFile(StringRef Path, llvm::sys::path::Style Style) {
  llvm::StringRef directory = llvm::sys::path::parent_path(Path, Style);
  llvm::StringRef filename = llvm::sys::path::filename(Path, Style);
  const uint32_t Dir = insertString(directory);
  const uint32_t Base = insertString(filename);
  return insertFileEntry(FileEntry(Dir, Base));
}

uint32_t GsymCreatorV1::insertFileEntry(FileEntry FE) {
  std::lock_guard<std::mutex> Guard(Mutex);
  const auto NextIndex = Files.size();
  auto R = FileEntryToIndex.insert(std::make_pair(FE, NextIndex));
  if (R.second)
    Files.emplace_back(FE);
  return R.first->second;
}

uint32_t GsymCreatorV1::copyFile(const GsymCreatorV1 &SrcGC, uint32_t FileIdx) {
  if (FileIdx == 0)
    return 0;
  const FileEntry SrcFE = SrcGC.Files[FileIdx];
  uint32_t Dir =
      SrcFE.Dir == 0
          ? 0
          : StrTab.add(SrcGC.StringOffsetMap.find(SrcFE.Dir)->second);
  uint32_t Base = StrTab.add(SrcGC.StringOffsetMap.find(SrcFE.Base)->second);
  FileEntry DstFE(Dir, Base);
  return insertFileEntry(DstFE);
}

llvm::Error GsymCreatorV1::save(StringRef Path, llvm::endianness ByteOrder,
                              std::optional<uint64_t> SegmentSize) const {
  if (SegmentSize)
    return saveSegments(Path, ByteOrder, *SegmentSize);
  std::error_code EC;
  raw_fd_ostream OutStrm(Path, EC);
  if (EC)
    return llvm::errorCodeToError(EC);
  FileWriter O(OutStrm, ByteOrder);
  return encode(O);
}

llvm::Error GsymCreatorV1::encode(FileWriter &O) const {
  std::lock_guard<std::mutex> Guard(Mutex);
  if (Funcs.empty())
    return createStringError(std::errc::invalid_argument,
                             "no functions to encode");
  if (!Finalized)
    return createStringError(std::errc::invalid_argument,
                             "GsymCreator wasn't finalized prior to encoding");

  if (Funcs.size() > UINT32_MAX)
    return createStringError(std::errc::invalid_argument,
                             "too many FunctionInfos");

  std::optional<uint64_t> BaseAddress = getBaseAddress();
  if (!BaseAddress)
    return createStringError(std::errc::invalid_argument,
                             "invalid base address");
  Header Hdr;
  Hdr.Magic = GSYM_MAGIC;
  Hdr.Version = GSYM_VERSION;
  Hdr.AddrOffSize = getAddressOffsetSize();
  Hdr.UUIDSize = static_cast<uint8_t>(UUID.size());
  Hdr.BaseAddress = *BaseAddress;
  Hdr.NumAddresses = static_cast<uint32_t>(Funcs.size());
  Hdr.StrtabOffset = 0;
  Hdr.StrtabSize = 0;
  memset(Hdr.UUID, 0, sizeof(Hdr.UUID));
  if (UUID.size() > sizeof(Hdr.UUID))
    return createStringError(std::errc::invalid_argument,
                             "invalid UUID size %u", (uint32_t)UUID.size());
  if (UUID.size() > 0)
    memcpy(Hdr.UUID, UUID.data(), UUID.size());
  llvm::Error Err = Hdr.encode(O);
  if (Err)
    return Err;

  const uint64_t MaxAddressOffset = getMaxAddressOffset();
  O.alignTo(Hdr.AddrOffSize);
  for (const auto &FuncInfo : Funcs) {
    uint64_t AddrOffset = FuncInfo.startAddress() - Hdr.BaseAddress;
    assert(AddrOffset <= MaxAddressOffset);
    (void)MaxAddressOffset;
    switch (Hdr.AddrOffSize) {
    case 1:
      O.writeU8(static_cast<uint8_t>(AddrOffset));
      break;
    case 2:
      O.writeU16(static_cast<uint16_t>(AddrOffset));
      break;
    case 4:
      O.writeU32(static_cast<uint32_t>(AddrOffset));
      break;
    case 8:
      O.writeU64(AddrOffset);
      break;
    }
  }

  O.alignTo(4);
  const off_t AddrInfoOffsetsOffset = O.tell();
  for (size_t i = 0, n = Funcs.size(); i < n; ++i)
    O.writeU32(0);

  O.alignTo(4);
  assert(!Files.empty());
  assert(Files[0].Dir == 0);
  assert(Files[0].Base == 0);
  size_t NumFiles = Files.size();
  if (NumFiles > UINT32_MAX)
    return createStringError(std::errc::invalid_argument, "too many files");
  O.writeU32(static_cast<uint32_t>(NumFiles));
  for (auto File : Files) {
    O.writeU32(File.Dir);
    O.writeU32(File.Base);
  }

  const off_t StrtabOffset = O.tell();
  StrTab.write(O.get_stream());
  const off_t StrtabSize = O.tell() - StrtabOffset;
  std::vector<uint32_t> AddrInfoOffsets;

  if (StrtabSize > UINT32_MAX) {
    return createStringError(std::errc::invalid_argument,
                             "string table size exceeded 32-bit max");
  }

  for (const auto &FuncInfo : Funcs) {
    if (Expected<uint64_t> OffsetOrErr = FuncInfo.encode(O)) {
      uint64_t Offset = OffsetOrErr.get();
      if (Offset > UINT32_MAX) {
        return createStringError(std::errc::invalid_argument,
                                 "address info offset exceeded 32-bit max");
      }

      AddrInfoOffsets.push_back(Offset);
    } else
      return OffsetOrErr.takeError();
  }
  O.fixup32((uint32_t)StrtabOffset, offsetof(Header, StrtabOffset));
  O.fixup32((uint32_t)StrtabSize, offsetof(Header, StrtabSize));

  uint64_t Offset = 0;
  for (auto AddrInfoOffset : AddrInfoOffsets) {
    O.fixup32(AddrInfoOffset, AddrInfoOffsetsOffset + Offset);
    Offset += 4;
  }
  return ErrorSuccess();
}

llvm::Error GsymCreatorV1::loadCallSitesFromYAML(StringRef YAMLFile) {
  CallSiteInfoLoader Loader(*this, Funcs);
  return Loader.loadYAML(YAMLFile);
}

void GsymCreatorV1::prepareMergedFunctions(OutputAggregator &Out) {
  if (Funcs.size() < 2)
    return;

  llvm::stable_sort(Funcs);
  std::vector<FunctionInfo> TopLevelFuncs;

  TopLevelFuncs.emplace_back(std::move(Funcs.front()));

  for (size_t Idx = 1; Idx < Funcs.size(); ++Idx) {
    FunctionInfo &TopFunc = TopLevelFuncs.back();
    FunctionInfo &MatchFunc = Funcs[Idx];
    if (TopFunc.Range == MatchFunc.Range) {
      if (!TopFunc.MergedFunctions)
        TopFunc.MergedFunctions = MergedFunctionsInfo();
      else if (TopFunc.MergedFunctions->MergedFunctions.back() == MatchFunc)
        continue;
      TopFunc.MergedFunctions->MergedFunctions.emplace_back(
          std::move(MatchFunc));
    } else
      TopLevelFuncs.emplace_back(std::move(MatchFunc));
  }

  uint32_t mergedCount = Funcs.size() - TopLevelFuncs.size();
  if (mergedCount != 0)
    Out << "Have " << mergedCount
        << " merged functions as children of other functions\n";

  std::swap(Funcs, TopLevelFuncs);
}

llvm::Error GsymCreatorV1::finalize(OutputAggregator &Out) {
  std::lock_guard<std::mutex> Guard(Mutex);
  if (Finalized)
    return createStringError(std::errc::invalid_argument, "already finalized");
  Finalized = true;

  StrTab.finalizeInOrder();

  const auto NumBefore = Funcs.size();
  if (!IsSegment) {
    if (NumBefore > 1) {
      llvm::stable_sort(Funcs);
      std::vector<FunctionInfo> FinalizedFuncs;
      FinalizedFuncs.reserve(Funcs.size());
      FinalizedFuncs.emplace_back(std::move(Funcs.front()));
      for (size_t Idx=1; Idx < NumBefore; ++Idx) {
        FunctionInfo &Prev = FinalizedFuncs.back();
        FunctionInfo &Curr = Funcs[Idx];
        const bool ranges_equal = Prev.Range == Curr.Range;
        if (ranges_equal || Prev.Range.intersects(Curr.Range)) {
          if (ranges_equal) {
            if (!(Prev == Curr)) {
              if (Prev.hasRichInfo() && Curr.hasRichInfo())
                Out.Report(
                    "Duplicate address ranges with different debug info.",
                    [&](raw_ostream &OS) {
                      OS << "warning: same address range contains "
                            "different debug "
                         << "info. Removing:\n"
                         << Prev << "\nIn favor of this one:\n"
                         << Curr << "\n";
                    });

              std::swap(Prev, Curr);
            }
          } else {
            Out.Report("Overlapping function ranges", [&](raw_ostream &OS) {
              OS << "warning: function ranges overlap:\n"
                << Prev << "\n"
                << Curr << "\n";
            });
            FinalizedFuncs.emplace_back(std::move(Curr));
          }
        } else {
          if (Prev.Range.size() == 0 && Curr.Range.contains(Prev.Range.start())) {
            std::swap(Prev, Curr);
          } else {
            FinalizedFuncs.emplace_back(std::move(Curr));
          }
        }
      }
      std::swap(Funcs, FinalizedFuncs);
    }
    if (!Funcs.empty() && Funcs.back().Range.size() == 0 && ValidTextRanges) {
      if (auto Range =
              ValidTextRanges->getRangeThatContains(Funcs.back().Range.start())) {
        Funcs.back().Range = {Funcs.back().Range.start(), Range->end()};
      }
    }
    Out << "Pruned " << NumBefore - Funcs.size() << " functions, ended with "
        << Funcs.size() << " total\n";
  }
  return Error::success();
}

uint32_t GsymCreatorV1::copyString(const GsymCreatorV1 &SrcGC, uint32_t StrOff) {
  if (StrOff == 0)
    return 0;
  return StrTab.add(SrcGC.StringOffsetMap.find(StrOff)->second);
}

uint32_t GsymCreatorV1::insertString(StringRef S, bool Copy) {
  if (S.empty())
    return 0;

  CachedHashStringRef CHStr(S);
  std::lock_guard<std::mutex> Guard(Mutex);
  if (Copy) {
    if (!StrTab.contains(CHStr))
      CHStr = CachedHashStringRef{StringStorage.insert(S).first->getKey(),
                                  CHStr.hash()};
  }
  const uint32_t StrOff = StrTab.add(CHStr);
  StringOffsetMap.try_emplace(StrOff, CHStr);
  return StrOff;
}

StringRef GsymCreatorV1::getString(uint32_t Offset) {
  auto I = StringOffsetMap.find(Offset);
  assert(I != StringOffsetMap.end() &&
         "GsymCreatorV1::getString expects a valid offset as parameter.");
  return I->second.val();
}

void GsymCreatorV1::addFunctionInfo(FunctionInfo &&FI) {
  std::lock_guard<std::mutex> Guard(Mutex);
  Funcs.emplace_back(std::move(FI));
}

void GsymCreatorV1::forEachFunctionInfo(
    std::function<bool(FunctionInfo &)> const &Callback) {
  std::lock_guard<std::mutex> Guard(Mutex);
  for (auto &FI : Funcs) {
    if (!Callback(FI))
      break;
  }
}

void GsymCreatorV1::forEachFunctionInfo(
    std::function<bool(const FunctionInfo &)> const &Callback) const {
  std::lock_guard<std::mutex> Guard(Mutex);
  for (const auto &FI : Funcs) {
    if (!Callback(FI))
      break;
  }
}

size_t GsymCreatorV1::getNumFunctionInfos() const {
  std::lock_guard<std::mutex> Guard(Mutex);
  return Funcs.size();
}

bool GsymCreatorV1::IsValidTextAddress(uint64_t Addr) const {
  if (ValidTextRanges)
    return ValidTextRanges->contains(Addr);
  return true;
}

std::optional<uint64_t> GsymCreatorV1::getFirstFunctionAddress() const {
  if ((Finalized || IsSegment) && !Funcs.empty())
    return std::optional<uint64_t>(Funcs.front().startAddress());
  return std::nullopt;
}

std::optional<uint64_t> GsymCreatorV1::getLastFunctionAddress() const {
  if ((Finalized || IsSegment) && !Funcs.empty())
    return std::optional<uint64_t>(Funcs.back().startAddress());
  return std::nullopt;
}

std::optional<uint64_t> GsymCreatorV1::getBaseAddress() const {
  if (BaseAddress)
    return BaseAddress;
  return getFirstFunctionAddress();
}

uint64_t GsymCreatorV1::getMaxAddressOffset() const {
  switch (getAddressOffsetSize()) {
    case 1: return UINT8_MAX;
    case 2: return UINT16_MAX;
    case 4: return UINT32_MAX;
    case 8: return UINT64_MAX;
  }
  llvm_unreachable("invalid address offset");
}

uint8_t GsymCreatorV1::getAddressOffsetSize() const {
  const std::optional<uint64_t> BaseAddress = getBaseAddress();
  const std::optional<uint64_t> LastFuncAddr = getLastFunctionAddress();
  if (BaseAddress && LastFuncAddr) {
    const uint64_t AddrDelta = *LastFuncAddr - *BaseAddress;
    if (AddrDelta <= UINT8_MAX)
      return 1;
    else if (AddrDelta <= UINT16_MAX)
      return 2;
    else if (AddrDelta <= UINT32_MAX)
      return 4;
    return 8;
  }
  return 1;
}

uint64_t GsymCreatorV1::calculateHeaderAndTableSize() const {
  uint64_t Size = sizeof(Header);
  const size_t NumFuncs = Funcs.size();
  Size += NumFuncs * getAddressOffsetSize();
  Size += NumFuncs * sizeof(uint32_t);
  Size += Files.size() * sizeof(FileEntry);
  Size += StrTab.getSize();

  return Size;
}

void GsymCreatorV1::fixupInlineInfo(const GsymCreatorV1 &SrcGC, InlineInfo &II) {
  II.Name = copyString(SrcGC, II.Name);
  II.CallFile = copyFile(SrcGC, II.CallFile);
  for (auto &ChildII: II.Children)
    fixupInlineInfo(SrcGC, ChildII);
}

uint64_t GsymCreatorV1::copyFunctionInfo(const GsymCreatorV1 &SrcGC, size_t FuncIdx) {
  const FunctionInfo &SrcFI = SrcGC.Funcs[FuncIdx];

  FunctionInfo DstFI;
  DstFI.Range = SrcFI.Range;
  DstFI.Name = copyString(SrcGC, SrcFI.Name);
  if (SrcFI.OptLineTable) {
    DstFI.OptLineTable = LineTable(SrcFI.OptLineTable.value());
    LineTable &DstLT = DstFI.OptLineTable.value();
    const size_t NumLines = DstLT.size();
    for (size_t I=0; I<NumLines; ++I) {
      LineEntry &LE = DstLT.get(I);
      LE.File = copyFile(SrcGC, LE.File);
    }
  }
  if (SrcFI.Inline) {
    DstFI.Inline = SrcFI.Inline.value();
    fixupInlineInfo(SrcGC, *DstFI.Inline);
  }
  std::lock_guard<std::mutex> Guard(Mutex);
  Funcs.emplace_back(DstFI);
  return Funcs.back().cacheEncoding();
}

llvm::Error GsymCreatorV1::saveSegments(StringRef Path,
                                      llvm::endianness ByteOrder,
                                      uint64_t SegmentSize) const {
  if (SegmentSize == 0)
    return createStringError(std::errc::invalid_argument,
                             "invalid segment size zero");

  size_t FuncIdx = 0;
  const size_t NumFuncs = Funcs.size();
  while (FuncIdx < NumFuncs) {
    llvm::Expected<std::unique_ptr<GsymCreatorV1>> ExpectedGC =
        createSegment(SegmentSize, FuncIdx);
    if (ExpectedGC) {
      GsymCreatorV1 *GC = ExpectedGC->get();
      if (!GC)
        break;
      OutputAggregator Out(nullptr);
      llvm::Error Err = GC->finalize(Out);
      if (Err)
        return Err;
      std::string SegmentedGsymPath;
      raw_string_ostream SGP(SegmentedGsymPath);
      std::optional<uint64_t> FirstFuncAddr = GC->getFirstFunctionAddress();
      if (FirstFuncAddr) {
        SGP << Path << "-" << llvm::format_hex(*FirstFuncAddr, 1);
        Err = GC->save(SegmentedGsymPath, ByteOrder, std::nullopt);
        if (Err)
          return Err;
      }
    } else {
      return ExpectedGC.takeError();
    }
  }
  return Error::success();
}

llvm::Expected<std::unique_ptr<GsymCreatorV1>>
GsymCreatorV1::createSegment(uint64_t SegmentSize, size_t &FuncIdx) const {
  if (FuncIdx >= Funcs.size())
    return std::unique_ptr<GsymCreatorV1>();

  std::unique_ptr<GsymCreatorV1> GC(new GsymCreatorV1(/*Quiet=*/true));

  GC->setIsSegment();

  if (BaseAddress)
    GC->setBaseAddress(*BaseAddress);
  GC->setUUID(UUID);
  const size_t NumFuncs = Funcs.size();
  uint64_t SegmentFuncInfosSize = 0;
  for (; FuncIdx < NumFuncs; ++FuncIdx) {
    const uint64_t HeaderAndTableSize = GC->calculateHeaderAndTableSize();
    if (HeaderAndTableSize + SegmentFuncInfosSize >= SegmentSize) {
      if (SegmentFuncInfosSize == 0)
        return createStringError(std::errc::invalid_argument,
                                 "a segment size of %" PRIu64 " is to small to "
                                 "fit any function infos, specify a larger value",
                                 SegmentSize);

      break;
    }
    SegmentFuncInfosSize += alignTo(GC->copyFunctionInfo(*this, FuncIdx), 4);
  }
  return std::move(GC);
}
