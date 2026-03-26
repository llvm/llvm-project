//===- GsymCreatorV2.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/GSYM/GsymCreatorV2.h"
#include "llvm/DebugInfo/GSYM/FileWriter.h"
#include "llvm/DebugInfo/GSYM/GlobalData.h"
#include "llvm/DebugInfo/GSYM/HeaderV2.h"
#include "llvm/DebugInfo/GSYM/LineTable.h"
#include "llvm/DebugInfo/GSYM/OutputAggregator.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <vector>

using namespace llvm;
using namespace gsym;

GsymCreatorV2::GsymCreatorV2(bool Quiet)
    : StrTab(StringTableBuilder::ELF), Quiet(Quiet) {
  insertFile(StringRef());
}

uint32_t GsymCreatorV2::insertFile(StringRef Path, llvm::sys::path::Style Style) {
  llvm::StringRef directory = llvm::sys::path::parent_path(Path, Style);
  llvm::StringRef filename = llvm::sys::path::filename(Path, Style);
  // We must insert the strings first, then call the FileEntry constructor.
  // If we inline the insertString() function call into the constructor, the
  // call order is undefined due to parameter lists not having any ordering
  // requirements.
  const uint32_t Dir = insertString(directory);
  const uint32_t Base = insertString(filename);
  return insertFileEntry(FileEntry(Dir, Base));
}

uint32_t GsymCreatorV2::insertFileEntry(FileEntry FE) {
  std::lock_guard<std::mutex> Guard(Mutex);
  const auto NextIndex = Files.size();
  // Find FE in hash map and insert if not present.
  auto R = FileEntryToIndex.insert(std::make_pair(FE, NextIndex));
  if (R.second)
    Files.emplace_back(FE);
  return R.first->second;
}

uint32_t GsymCreatorV2::copyFile(const GsymCreatorV2 &SrcGC, uint32_t FileIdx) {
  // File index zero is reserved for a FileEntry with no directory and no
  // filename. Any other file and we need to copy the strings for the directory
  // and filename.
  if (FileIdx == 0)
    return 0;
  const FileEntry SrcFE = SrcGC.Files[FileIdx];
  // Copy the strings for the file and then add the newly converted file entry.
  uint32_t Dir =
      SrcFE.Dir == 0
          ? 0
          : StrTab.add(SrcGC.StringOffsetMap.find(SrcFE.Dir)->second);
  uint32_t Base = StrTab.add(SrcGC.StringOffsetMap.find(SrcFE.Base)->second);
  FileEntry DstFE(Dir, Base);
  return insertFileEntry(DstFE);
}

llvm::Error GsymCreatorV2::save(StringRef Path, llvm::endianness ByteOrder,
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

/// Write a single GlobalData entry to the output stream.
static void writeGlobalDataEntry(FileWriter &O, GlobalInfoType Type,
                                 uint64_t FileOffset, uint64_t FileSize) {
  O.writeU32(static_cast<uint32_t>(Type));
  O.writeU32(0); // Padding
  O.writeU64(FileOffset);
  O.writeU64(FileSize);
}

llvm::Error GsymCreatorV2::encode(FileWriter &O) const {
  std::lock_guard<std::mutex> Guard(Mutex);
  if (Funcs.empty())
    return createStringError(std::errc::invalid_argument,
                             "no functions to encode");
  if (!Finalized)
    return createStringError(std::errc::invalid_argument,
                             "GsymCreatorV2 wasn't finalized prior to encoding");
  if (Funcs.size() > UINT32_MAX)
    return createStringError(std::errc::invalid_argument,
                             "too many FunctionInfos");

  std::optional<uint64_t> BaseAddr = getBaseAddress();
  if (!BaseAddr)
    return createStringError(std::errc::invalid_argument,
                             "invalid base address");

  const uint8_t AddrOffSize = getAddressOffsetSize();

  // Pre-encode all FunctionInfo objects into a temporary buffer so we know the
  // total FunctionInfo section size and each function's offset within it.
  SmallVector<char, 0> FIBuf;
  raw_svector_ostream FIOS(FIBuf);
  FileWriter FIFW(FIOS, O.getByteOrder());
  std::vector<uint64_t> FIRelativeOffsets;
  for (const auto &FI : Funcs) {
    if (auto OffOrErr = FI.encode(FIFW))
      FIRelativeOffsets.push_back(*OffOrErr);
    else
      return OffOrErr.takeError();
  }
  const uint64_t FISectionSize = FIBuf.size();
  const uint64_t StringTableSize = StrTab.getSize();

  // Determine StrpSize based on string table size.
  const uint8_t StrpSize = (StringTableSize > UINT32_MAX) ? 8 : 4;

  // Compute number of GlobalData entries.
  const bool HasUUID = !UUID.empty();
  // Sections: AddrOffsets, AddrInfoOffsets, StringTable, FileTable, FunctionInfo
  // Plus UUID if present, plus EndOfList terminator.
  const uint32_t NumGlobalDataEntries = 5 + (HasUUID ? 1 : 0) + 1;
  const uint64_t GlobalDataArraySize =
      static_cast<uint64_t>(NumGlobalDataEntries) * 24;

  // Plan the file layout. All offsets are relative to the start of the GSYM
  // data (i.e., the start of the header). We place sections sequentially after
  // the header and GlobalData entries in a convenient order.
  constexpr uint64_t HeaderSize = 24;
  uint64_t CurOffset = HeaderSize + GlobalDataArraySize;

  // AddrOffsets section.
  CurOffset = llvm::alignTo(CurOffset, AddrOffSize);
  const uint64_t AddrOffsetsOffset = CurOffset;
  const uint64_t AddrOffsetsSize = Funcs.size() * AddrOffSize;
  CurOffset += AddrOffsetsSize;

  // Determine AddrInfoOffSize: if the estimated end of the FunctionInfo section
  // would exceed UINT32_MAX, use 8-byte offsets.
  uint8_t AddrInfoOffSize = 4;
  {
    // Estimate conservatively with 4-byte AddrInfoOffsets.
    uint64_t Est = CurOffset;
    Est = llvm::alignTo(Est, 4);
    Est += Funcs.size() * 4; // AddrInfoOffsets
    Est = llvm::alignTo(Est, 4);
    Est += 4 + Files.size() * sizeof(FileEntry); // FileTable
    Est += StringTableSize;
    Est = llvm::alignTo(Est, 4);
    Est += FISectionSize;
    if (Est > UINT32_MAX)
      AddrInfoOffSize = 8;
  }

  // AddrInfoOffsets section.
  CurOffset = llvm::alignTo(CurOffset, AddrInfoOffSize);
  const uint64_t AddrInfoOffsetsOffset = CurOffset;
  const uint64_t AddrInfoOffsetsSize = Funcs.size() * AddrInfoOffSize;
  CurOffset += AddrInfoOffsetsSize;

  // FileTable section.
  CurOffset = llvm::alignTo(CurOffset, 4);
  const uint64_t FileTableOffset = CurOffset;
  const uint64_t FileTableSize = 4 + Files.size() * sizeof(FileEntry);
  CurOffset += FileTableSize;

  // StringTable section (no alignment requirement).
  const uint64_t StringTableOffset = CurOffset;
  CurOffset += StringTableSize;

  // FunctionInfo section.
  CurOffset = llvm::alignTo(CurOffset, 4);
  const uint64_t FISectionOffset = CurOffset;
  CurOffset += FISectionSize;

  // UUID section (no alignment requirement).
  const uint64_t UUIDOffset = CurOffset;
  const uint64_t UUIDSectionSize = UUID.size();

  // Build and write the header.
  HeaderV2 Hdr;
  Hdr.Magic = GSYM_MAGIC;
  Hdr.Version = GSYM_VERSION_2;
  Hdr.Padding = 0;
  Hdr.BaseAddress = *BaseAddr;
  Hdr.NumAddresses = static_cast<uint32_t>(Funcs.size());
  Hdr.AddrOffSize = AddrOffSize;
  Hdr.AddrInfoOffSize = AddrInfoOffSize;
  Hdr.StrpSize = StrpSize;
  Hdr.Padding2 = 0;
  if (auto Err = Hdr.encode(O))
    return Err;

  // Write GlobalData entries.
  writeGlobalDataEntry(O, GlobalInfoType::AddrOffsets,
                       AddrOffsetsOffset, AddrOffsetsSize);
  writeGlobalDataEntry(O, GlobalInfoType::AddrInfoOffsets,
                       AddrInfoOffsetsOffset, AddrInfoOffsetsSize);
  writeGlobalDataEntry(O, GlobalInfoType::FileTable,
                       FileTableOffset, FileTableSize);
  writeGlobalDataEntry(O, GlobalInfoType::StringTable,
                       StringTableOffset, StringTableSize);
  writeGlobalDataEntry(O, GlobalInfoType::FunctionInfo,
                       FISectionOffset, FISectionSize);
  if (HasUUID)
    writeGlobalDataEntry(O, GlobalInfoType::UUID,
                         UUIDOffset, UUIDSectionSize);
  // EndOfList terminator.
  writeGlobalDataEntry(O, GlobalInfoType::EndOfList, 0, 0);

  // Write AddrOffsets section.
  O.alignTo(AddrOffSize);
  assert(O.tell() == AddrOffsetsOffset);
  const uint64_t MaxAddressOffset = getMaxAddressOffset();
  for (const auto &FI : Funcs) {
    uint64_t AddrOffset = FI.startAddress() - *BaseAddr;
    assert(AddrOffset <= MaxAddressOffset);
    (void)MaxAddressOffset;
    switch (AddrOffSize) {
    case 1: O.writeU8(static_cast<uint8_t>(AddrOffset)); break;
    case 2: O.writeU16(static_cast<uint16_t>(AddrOffset)); break;
    case 4: O.writeU32(static_cast<uint32_t>(AddrOffset)); break;
    case 8: O.writeU64(AddrOffset); break;
    }
  }

  // Write AddrInfoOffsets section. Each entry is the absolute file offset
  // (from the start of the GSYM data) to the corresponding FunctionInfo.
  O.alignTo(AddrInfoOffSize);
  assert(O.tell() == AddrInfoOffsetsOffset);
  for (uint64_t RelOff : FIRelativeOffsets) {
    uint64_t AbsOff = FISectionOffset + RelOff;
    if (AddrInfoOffSize == 4) {
      if (AbsOff > UINT32_MAX)
        return createStringError(std::errc::invalid_argument,
                                 "addr info offset exceeded 32-bit max");
      O.writeU32(static_cast<uint32_t>(AbsOff));
    } else {
      O.writeU64(AbsOff);
    }
  }

  // Write FileTable section.
  O.alignTo(4);
  assert(O.tell() == FileTableOffset);
  assert(!Files.empty());
  assert(Files[0].Dir == 0);
  assert(Files[0].Base == 0);
  if (Files.size() > UINT32_MAX)
    return createStringError(std::errc::invalid_argument, "too many files");
  O.writeU32(static_cast<uint32_t>(Files.size()));
  for (const auto &File : Files) {
    O.writeU32(File.Dir);
    O.writeU32(File.Base);
  }

  // Write StringTable section.
  assert(O.tell() == StringTableOffset);
  StrTab.write(O.get_stream());

  // Write FunctionInfo section (pre-encoded data).
  O.alignTo(4);
  assert(O.tell() == FISectionOffset);
  O.writeData(ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(FIBuf.data()),
                                FIBuf.size()));

  // Write UUID section.
  if (HasUUID) {
    assert(O.tell() == UUIDOffset);
    O.writeData(ArrayRef<uint8_t>(UUID.data(), UUID.size()));
  }

  return Error::success();
}

llvm::Error GsymCreatorV2::loadCallSitesFromYAML(StringRef YAMLFile) {
  // TODO: Implement V2-specific call site loading.
  return createStringError(std::errc::not_supported,
                           "call site loading not yet supported in V2");
}

void GsymCreatorV2::prepareMergedFunctions(OutputAggregator &Out) {
  // Nothing to do if we have less than 2 functions.
  if (Funcs.size() < 2)
    return;

  // Sort the function infos by address range first, preserving input order
  llvm::stable_sort(Funcs);
  std::vector<FunctionInfo> TopLevelFuncs;

  // Add the first function info to the top level functions
  TopLevelFuncs.emplace_back(std::move(Funcs.front()));

  // Now if the next function info has the same address range as the top level,
  // then merge it into the top level function, otherwise add it to the top
  // level.
  for (size_t Idx = 1; Idx < Funcs.size(); ++Idx) {
    FunctionInfo &TopFunc = TopLevelFuncs.back();
    FunctionInfo &MatchFunc = Funcs[Idx];
    if (TopFunc.Range == MatchFunc.Range) {
      // Both have the same range - add the 2nd func as a child of the 1st func
      if (!TopFunc.MergedFunctions)
        TopFunc.MergedFunctions = MergedFunctionsInfo();
      // Avoid adding duplicate functions to MergedFunctions. Since functions
      // are already ordered within the Funcs array, we can just check equality
      // against the last function in the merged array.
      else if (TopFunc.MergedFunctions->MergedFunctions.back() == MatchFunc)
        continue;
      TopFunc.MergedFunctions->MergedFunctions.emplace_back(
          std::move(MatchFunc));
    } else
      // No match, add the function as a top-level function
      TopLevelFuncs.emplace_back(std::move(MatchFunc));
  }

  uint32_t mergedCount = Funcs.size() - TopLevelFuncs.size();
  // If any functions were merged, print a message about it.
  if (mergedCount != 0)
    Out << "Have " << mergedCount
        << " merged functions as children of other functions\n";

  std::swap(Funcs, TopLevelFuncs);
}

llvm::Error GsymCreatorV2::finalize(OutputAggregator &Out) {
  std::lock_guard<std::mutex> Guard(Mutex);
  if (Finalized)
    return createStringError(std::errc::invalid_argument, "already finalized");
  Finalized = true;

  // Don't let the string table indexes change by finalizing in order.
  StrTab.finalizeInOrder();

  // Remove duplicates function infos that have both entries from debug info
  // (DWARF or Breakpad) and entries from the SymbolTable.
  //
  // Also handle overlapping function. Usually there shouldn't be any, but they
  // can and do happen in some rare cases.
  //
  // (a)          (b)         (c)
  //     ^  ^       ^            ^
  //     |X |Y      |X ^         |X
  //     |  |       |  |Y        |  ^
  //     |  |       |  v         v  |Y
  //     v  v       v               v
  //
  // In (a) and (b), Y is ignored and X will be reported for the full range.
  // In (c), both functions will be included in the result and lookups for an
  // address in the intersection will return Y because of binary search.
  //
  // Note that in case of (b), we cannot include Y in the result because then
  // we wouldn't find any function for range (end of Y, end of X)
  // with binary search

  const auto NumBefore = Funcs.size();
  // Only sort and unique if this isn't a segment. If this is a segment we
  // already finalized the main GsymCreatorV2 with all of the function infos
  // and then the already sorted and uniqued function infos were added to this
  // object.
  if (!IsSegment) {
    if (NumBefore > 1) {
      // Sort function infos so we can emit sorted functions. Use stable sort to
      // ensure determinism.
      llvm::stable_sort(Funcs);
      std::vector<FunctionInfo> FinalizedFuncs;
      FinalizedFuncs.reserve(Funcs.size());
      FinalizedFuncs.emplace_back(std::move(Funcs.front()));
      for (size_t Idx=1; Idx < NumBefore; ++Idx) {
        FunctionInfo &Prev = FinalizedFuncs.back();
        FunctionInfo &Curr = Funcs[Idx];
        // Empty ranges won't intersect, but we still need to
        // catch the case where we have multiple symbols at the
        // same address and coalesce them.
        const bool ranges_equal = Prev.Range == Curr.Range;
        if (ranges_equal || Prev.Range.intersects(Curr.Range)) {
          // Overlapping ranges or empty identical ranges.
          if (ranges_equal) {
            // Same address range. Check if one is from debug
            // info and the other is from a symbol table. If
            // so, then keep the one with debug info. Our
            // sorting guarantees that entries with matching
            // address ranges that have debug info are last in
            // the sort.
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

              // We want to swap the current entry with the previous since
              // later entries with the same range always have more debug info
              // or different debug info.
              std::swap(Prev, Curr);
            }
          } else {
            Out.Report("Overlapping function ranges", [&](raw_ostream &OS) {
              // print warnings about overlaps
              OS << "warning: function ranges overlap:\n"
                << Prev << "\n"
                << Curr << "\n";
            });
            FinalizedFuncs.emplace_back(std::move(Curr));
          }
        } else {
          if (Prev.Range.size() == 0 && Curr.Range.contains(Prev.Range.start())) {
            // Symbols on macOS don't have address ranges, so if the range
            // doesn't match and the size is zero, then we replace the empty
            // symbol function info with the current one.
            std::swap(Prev, Curr);
          } else {
            FinalizedFuncs.emplace_back(std::move(Curr));
          }
        }
      }
      std::swap(Funcs, FinalizedFuncs);
    }
    // If our last function info entry doesn't have a size and if we have valid
    // text ranges, we should set the size of the last entry since any search for
    // a high address might match our last entry. By fixing up this size, we can
    // help ensure we don't cause lookups to always return the last symbol that
    // has no size when doing lookups.
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

uint32_t GsymCreatorV2::copyString(const GsymCreatorV2 &SrcGC, uint32_t StrOff) {
  // String offset at zero is always the empty string, no copying needed.
  if (StrOff == 0)
    return 0;
  return StrTab.add(SrcGC.StringOffsetMap.find(StrOff)->second);
}

uint32_t GsymCreatorV2::insertString(StringRef S, bool Copy) {
  if (S.empty())
    return 0;

  // The hash can be calculated outside the lock.
  CachedHashStringRef CHStr(S);
  std::lock_guard<std::mutex> Guard(Mutex);
  if (Copy) {
    // We need to provide backing storage for the string if requested
    // since StringTableBuilder stores references to strings. Any string
    // that comes from a section in an object file doesn't need to be
    // copied, but any string created by code will need to be copied.
    // This allows GsymCreatorV2 to be really fast when parsing DWARF and
    // other object files as most strings don't need to be copied.
    if (!StrTab.contains(CHStr))
      CHStr = CachedHashStringRef{StringStorage.insert(S).first->getKey(),
                                  CHStr.hash()};
  }
  const uint32_t StrOff = StrTab.add(CHStr);
  // Save a mapping of string offsets to the cached string reference in case
  // we need to segment the GSYM file and copy string from one string table to
  // another.
  StringOffsetMap.try_emplace(StrOff, CHStr);
  return StrOff;
}

StringRef GsymCreatorV2::getString(uint32_t Offset) {
  auto I = StringOffsetMap.find(Offset);
  assert(I != StringOffsetMap.end() &&
         "GsymCreatorV2::getString expects a valid offset as parameter.");
  return I->second.val();
}

void GsymCreatorV2::addFunctionInfo(FunctionInfo &&FI) {
  std::lock_guard<std::mutex> Guard(Mutex);
  Funcs.emplace_back(std::move(FI));
}

void GsymCreatorV2::forEachFunctionInfo(
    std::function<bool(FunctionInfo &)> const &Callback) {
  std::lock_guard<std::mutex> Guard(Mutex);
  for (auto &FI : Funcs) {
    if (!Callback(FI))
      break;
  }
}

void GsymCreatorV2::forEachFunctionInfo(
    std::function<bool(const FunctionInfo &)> const &Callback) const {
  std::lock_guard<std::mutex> Guard(Mutex);
  for (const auto &FI : Funcs) {
    if (!Callback(FI))
      break;
  }
}

size_t GsymCreatorV2::getNumFunctionInfos() const {
  std::lock_guard<std::mutex> Guard(Mutex);
  return Funcs.size();
}

bool GsymCreatorV2::IsValidTextAddress(uint64_t Addr) const {
  if (ValidTextRanges)
    return ValidTextRanges->contains(Addr);
  return true; // No valid text ranges has been set, so accept all ranges.
}

std::optional<uint64_t> GsymCreatorV2::getFirstFunctionAddress() const {
  // If we have finalized then Funcs are sorted. If we are a segment then
  // Funcs will be sorted as well since function infos get added from an
  // already finalized GsymCreatorV2 object where its functions were sorted and
  // uniqued.
  if ((Finalized || IsSegment) && !Funcs.empty())
    return std::optional<uint64_t>(Funcs.front().startAddress());
  return std::nullopt;
}

std::optional<uint64_t> GsymCreatorV2::getLastFunctionAddress() const {
  // If we have finalized then Funcs are sorted. If we are a segment then
  // Funcs will be sorted as well since function infos get added from an
  // already finalized GsymCreatorV2 object where its functions were sorted and
  // uniqued.
  if ((Finalized || IsSegment) && !Funcs.empty())
    return std::optional<uint64_t>(Funcs.back().startAddress());
  return std::nullopt;
}

std::optional<uint64_t> GsymCreatorV2::getBaseAddress() const {
  if (BaseAddress)
    return BaseAddress;
  return getFirstFunctionAddress();
}

uint64_t GsymCreatorV2::getMaxAddressOffset() const {
  switch (getAddressOffsetSize()) {
    case 1: return UINT8_MAX;
    case 2: return UINT16_MAX;
    case 4: return UINT32_MAX;
    case 8: return UINT64_MAX;
  }
  llvm_unreachable("invalid address offset");
}

uint8_t GsymCreatorV2::getAddressOffsetSize() const {
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

uint64_t GsymCreatorV2::calculateHeaderAndTableSize() const {
  constexpr uint64_t HeaderSize = 24;
  const size_t NumFuncs = Funcs.size();
  // GlobalData entries: 5 sections + UUID (if any) + EndOfList terminator.
  const uint32_t NumEntries = 5 + (UUID.empty() ? 0 : 1) + 1;
  uint64_t Size = HeaderSize + NumEntries * 24;
  // AddrOffsets
  Size = llvm::alignTo(Size, getAddressOffsetSize());
  Size += NumFuncs * getAddressOffsetSize();
  // AddrInfoOffsets (assume 4-byte entries for estimation)
  Size = llvm::alignTo(Size, 4);
  Size += NumFuncs * 4;
  // FileTable
  Size = llvm::alignTo(Size, 4);
  Size += 4 + Files.size() * sizeof(FileEntry);
  // StringTable
  Size += StrTab.getSize();
  // UUID
  Size += UUID.size();
  return Size;
}

// This function takes a InlineInfo class that was copy constructed from an
// InlineInfo from the \a SrcGC and updates all members that point to strings
// and files to point to strings and files from this GsymCreatorV2.
void GsymCreatorV2::fixupInlineInfo(const GsymCreatorV2 &SrcGC, InlineInfo &II) {
  II.Name = copyString(SrcGC, II.Name);
  II.CallFile = copyFile(SrcGC, II.CallFile);
  for (auto &ChildII: II.Children)
    fixupInlineInfo(SrcGC, ChildII);
}

uint64_t GsymCreatorV2::copyFunctionInfo(const GsymCreatorV2 &SrcGC, size_t FuncIdx) {
  // To copy a function info we need to copy any files and strings over into
  // this GsymCreatorV2 and then copy the function info and update the string
  // table offsets to match the new offsets.
  const FunctionInfo &SrcFI = SrcGC.Funcs[FuncIdx];

  FunctionInfo DstFI;
  DstFI.Range = SrcFI.Range;
  DstFI.Name = copyString(SrcGC, SrcFI.Name);
  // Copy the line table if there is one.
  if (SrcFI.OptLineTable) {
    // Copy the entire line table.
    DstFI.OptLineTable = LineTable(SrcFI.OptLineTable.value());
    // Fixup all LineEntry::File entries which are indexes in the the file table
    // from SrcGC and must be converted to file indexes from this GsymCreatorV2.
    LineTable &DstLT = DstFI.OptLineTable.value();
    const size_t NumLines = DstLT.size();
    for (size_t I=0; I<NumLines; ++I) {
      LineEntry &LE = DstLT.get(I);
      LE.File = copyFile(SrcGC, LE.File);
    }
  }
  // Copy the inline information if needed.
  if (SrcFI.Inline) {
    // Make a copy of the source inline information.
    DstFI.Inline = SrcFI.Inline.value();
    // Fixup all strings and files in the copied inline information.
    fixupInlineInfo(SrcGC, *DstFI.Inline);
  }
  std::lock_guard<std::mutex> Guard(Mutex);
  Funcs.emplace_back(DstFI);
  return Funcs.back().cacheEncoding();
}

llvm::Error GsymCreatorV2::saveSegments(StringRef Path,
                                      llvm::endianness ByteOrder,
                                      uint64_t SegmentSize) const {
  if (SegmentSize == 0)
    return createStringError(std::errc::invalid_argument,
                             "invalid segment size zero");

  size_t FuncIdx = 0;
  const size_t NumFuncs = Funcs.size();
  while (FuncIdx < NumFuncs) {
    llvm::Expected<std::unique_ptr<GsymCreatorV2>> ExpectedGC =
        createSegment(SegmentSize, FuncIdx);
    if (ExpectedGC) {
      GsymCreatorV2 *GC = ExpectedGC->get();
      if (!GC)
        break; // We had not more functions to encode.
      // Don't collect any messages at all
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

llvm::Expected<std::unique_ptr<GsymCreatorV2>>
GsymCreatorV2::createSegment(uint64_t SegmentSize, size_t &FuncIdx) const {
  // No function entries, return empty unique pointer
  if (FuncIdx >= Funcs.size())
    return std::unique_ptr<GsymCreatorV2>();

  std::unique_ptr<GsymCreatorV2> GC(new GsymCreatorV2(/*Quiet=*/true));

  // Tell the creator that this is a segment.
  GC->setIsSegment();

  // Set the base address if there is one.
  if (BaseAddress)
    GC->setBaseAddress(*BaseAddress);
  // Copy the UUID value from this object into the new creator.
  GC->setUUID(UUID);
  const size_t NumFuncs = Funcs.size();
  // Track how big the function infos are for the current segment so we can
  // emit segments that are close to the requested size. It is quick math to
  // determine the current header and tables sizes, so we can do that each loop.
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
