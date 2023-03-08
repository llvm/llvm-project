//===- GsymCreator.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/GSYM/GsymCreator.h"
#include "llvm/DebugInfo/GSYM/FileWriter.h"
#include "llvm/DebugInfo/GSYM/Header.h"
#include "llvm/DebugInfo/GSYM/LineTable.h"
#include "llvm/MC/StringTableBuilder.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <vector>

using namespace llvm;
using namespace gsym;

GsymCreator::GsymCreator(bool Quiet)
    : StrTab(StringTableBuilder::ELF), Quiet(Quiet) {
  insertFile(StringRef());
}

uint32_t GsymCreator::insertFile(StringRef Path, llvm::sys::path::Style Style) {
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

uint32_t GsymCreator::insertFileEntry(FileEntry FE) {
  std::lock_guard<std::mutex> Guard(Mutex);
  const auto NextIndex = Files.size();
  // Find FE in hash map and insert if not present.
  auto R = FileEntryToIndex.insert(std::make_pair(FE, NextIndex));
  if (R.second)
    Files.emplace_back(FE);
  return R.first->second;
}

uint32_t GsymCreator::copyFile(const GsymCreator &SrcGC, uint32_t FileIdx) {
  // File index zero is reserved for a FileEntry with no directory and no
  // filename. Any other file and we need to copy the strings for the directory
  // and filename.
  if (FileIdx == 0)
    return 0;
  const FileEntry SrcFE = SrcGC.Files[FileIdx];
  // Copy the strings for the file and then add the newly converted file entry.
  uint32_t Dir = StrTab.add(SrcGC.StringOffsetMap.find(SrcFE.Dir)->second);
  uint32_t Base = StrTab.add(SrcGC.StringOffsetMap.find(SrcFE.Base)->second);
  FileEntry DstFE(Dir, Base);
  return insertFileEntry(DstFE);
}


llvm::Error GsymCreator::save(StringRef Path,
                              llvm::support::endianness ByteOrder,
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

llvm::Error GsymCreator::encode(FileWriter &O) const {
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
  // Base address should be valid if we have any functions.
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
  Hdr.StrtabOffset = 0; // We will fix this up later.
  Hdr.StrtabSize = 0;   // We will fix this up later.
  memset(Hdr.UUID, 0, sizeof(Hdr.UUID));
  if (UUID.size() > sizeof(Hdr.UUID))
    return createStringError(std::errc::invalid_argument,
                             "invalid UUID size %u", (uint32_t)UUID.size());
  // Copy the UUID value if we have one.
  if (UUID.size() > 0)
    memcpy(Hdr.UUID, UUID.data(), UUID.size());
  // Write out the header.
  llvm::Error Err = Hdr.encode(O);
  if (Err)
    return Err;

  const uint64_t MaxAddressOffset = getMaxAddressOffset();
  // Write out the address offsets.
  O.alignTo(Hdr.AddrOffSize);
  for (const auto &FuncInfo : Funcs) {
    uint64_t AddrOffset = FuncInfo.startAddress() - Hdr.BaseAddress;
    // Make sure we calculated the address offsets byte size correctly by
    // verifying the current address offset is within ranges. We have seen bugs
    // introduced when the code changes that can cause problems here so it is
    // good to catch this during testing.
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

  // Write out all zeros for the AddrInfoOffsets.
  O.alignTo(4);
  const off_t AddrInfoOffsetsOffset = O.tell();
  for (size_t i = 0, n = Funcs.size(); i < n; ++i)
    O.writeU32(0);

  // Write out the file table
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

  // Write out the string table.
  const off_t StrtabOffset = O.tell();
  StrTab.write(O.get_stream());
  const off_t StrtabSize = O.tell() - StrtabOffset;
  std::vector<uint32_t> AddrInfoOffsets;

  // Write out the address infos for each function info.
  for (const auto &FuncInfo : Funcs) {
    if (Expected<uint64_t> OffsetOrErr = FuncInfo.encode(O))
      AddrInfoOffsets.push_back(OffsetOrErr.get());
    else
      return OffsetOrErr.takeError();
  }
  // Fixup the string table offset and size in the header
  O.fixup32((uint32_t)StrtabOffset, offsetof(Header, StrtabOffset));
  O.fixup32((uint32_t)StrtabSize, offsetof(Header, StrtabSize));

  // Fixup all address info offsets
  uint64_t Offset = 0;
  for (auto AddrInfoOffset : AddrInfoOffsets) {
    O.fixup32(AddrInfoOffset, AddrInfoOffsetsOffset + Offset);
    Offset += 4;
  }
  return ErrorSuccess();
}

// Similar to std::remove_if, but the predicate is binary and it is passed both
// the previous and the current element.
template <class ForwardIt, class BinaryPredicate>
static ForwardIt removeIfBinary(ForwardIt FirstIt, ForwardIt LastIt,
                                BinaryPredicate Pred) {
  if (FirstIt != LastIt) {
    auto PrevIt = FirstIt++;
    FirstIt = std::find_if(FirstIt, LastIt, [&](const auto &Curr) {
      return Pred(*PrevIt++, Curr);
    });
    if (FirstIt != LastIt)
      for (ForwardIt CurrIt = FirstIt; ++CurrIt != LastIt;)
        if (!Pred(*PrevIt, *CurrIt)) {
          PrevIt = FirstIt;
          *FirstIt++ = std::move(*CurrIt);
        }
  }
  return FirstIt;
}

llvm::Error GsymCreator::finalize(llvm::raw_ostream &OS) {
  std::lock_guard<std::mutex> Guard(Mutex);
  if (Finalized)
    return createStringError(std::errc::invalid_argument, "already finalized");
  Finalized = true;

  // Sort function infos so we can emit sorted functions.
  llvm::sort(Funcs);

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
  auto NumBefore = Funcs.size();
  Funcs.erase(
      removeIfBinary(Funcs.begin(), Funcs.end(),
                     [&](const auto &Prev, const auto &Curr) {
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
                           if (Prev == Curr) {
                             // FunctionInfo entries match exactly (range,
                             // lines, inlines)

                             // We used to output a warning here, but this was
                             // so frequent on some binaries, in particular
                             // when those were built with GCC, that it slowed
                             // down processing extremely.
                             return true;
                           } else {
                             if (!Prev.hasRichInfo() && Curr.hasRichInfo()) {
                               // Same address range, one with no debug info
                               // (symbol) and the next with debug info. Keep
                               // the latter.
                               return true;
                             } else {
                               if (!Quiet) {
                                 OS << "warning: same address range contains "
                                       "different debug "
                                    << "info. Removing:\n"
                                    << Prev << "\nIn favor of this one:\n"
                                    << Curr << "\n";
                               }
                               return true;
                             }
                           }
                         } else {
                           if (!Quiet) { // print warnings about overlaps
                             OS << "warning: function ranges overlap:\n"
                                << Prev << "\n"
                                << Curr << "\n";
                           }
                         }
                       } else if (Prev.Range.size() == 0 &&
                                  Curr.Range.contains(Prev.Range.start())) {
                         if (!Quiet) {
                           OS << "warning: removing symbol:\n"
                              << Prev << "\nKeeping:\n"
                              << Curr << "\n";
                         }
                         return true;
                       }

                       return false;
                     }),
      Funcs.end());

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
  OS << "Pruned " << NumBefore - Funcs.size() << " functions, ended with "
     << Funcs.size() << " total\n";
  return Error::success();
}

uint32_t GsymCreator::copyString(const GsymCreator &SrcGC, uint32_t StrOff) {
  // String offset at zero is always the empty string, no copying needed.
  if (StrOff == 0)
    return 0;
  return StrTab.add(SrcGC.StringOffsetMap.find(StrOff)->second);
}

uint32_t GsymCreator::insertString(StringRef S, bool Copy) {
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
    // This allows GsymCreator to be really fast when parsing DWARF and
    // other object files as most strings don't need to be copied.
    if (!StrTab.contains(CHStr))
      CHStr = CachedHashStringRef{StringStorage.insert(S).first->getKey(),
                                  CHStr.hash()};
  }
  const uint32_t StrOff = StrTab.add(CHStr);
  // Save a mapping of string offsets to the cached string reference in case
  // we need to segment the GSYM file and copy string from one string table to
  // another.
  if (StringOffsetMap.count(StrOff) == 0)
    StringOffsetMap.insert(std::make_pair(StrOff, CHStr));
  return StrOff;
}

void GsymCreator::addFunctionInfo(FunctionInfo &&FI) {
  std::lock_guard<std::mutex> Guard(Mutex);
  Ranges.insert(FI.Range);
  Funcs.emplace_back(std::move(FI));
}

void GsymCreator::forEachFunctionInfo(
    std::function<bool(FunctionInfo &)> const &Callback) {
  std::lock_guard<std::mutex> Guard(Mutex);
  for (auto &FI : Funcs) {
    if (!Callback(FI))
      break;
  }
}

void GsymCreator::forEachFunctionInfo(
    std::function<bool(const FunctionInfo &)> const &Callback) const {
  std::lock_guard<std::mutex> Guard(Mutex);
  for (const auto &FI : Funcs) {
    if (!Callback(FI))
      break;
  }
}

size_t GsymCreator::getNumFunctionInfos() const {
  std::lock_guard<std::mutex> Guard(Mutex);
  return Funcs.size();
}

bool GsymCreator::IsValidTextAddress(uint64_t Addr) const {
  if (ValidTextRanges)
    return ValidTextRanges->contains(Addr);
  return true; // No valid text ranges has been set, so accept all ranges.
}

bool GsymCreator::hasFunctionInfoForAddress(uint64_t Addr) const {
  std::lock_guard<std::mutex> Guard(Mutex);
  return Ranges.contains(Addr);
}

std::optional<uint64_t> GsymCreator::getFirstFunctionAddress() const {
  if (Finalized && !Funcs.empty())
    return std::optional<uint64_t>(Funcs.front().startAddress());
  // This code gets used by the segmentation of GSYM files to help determine the
  // size of the GSYM header while continually adding new FunctionInfo objects
  // to this object, so we haven't finalized this object yet.
  if (Ranges.empty())
    return std::nullopt;
  return std::optional<uint64_t>(Ranges.begin()->start());
}

std::optional<uint64_t> GsymCreator::getLastFunctionAddress() const {
  if (Finalized && !Funcs.empty())
    return std::optional<uint64_t>(Funcs.back().startAddress());
  // This code gets used by the segmentation of GSYM files to help determine the
  // size of the GSYM header while continually adding new FunctionInfo objects
  // to this object, so we haven't finalized this object yet.
  if (Ranges.empty())
    return std::nullopt;
  return std::optional<uint64_t>((Ranges.end() - 1)->end());
}

std::optional<uint64_t> GsymCreator::getBaseAddress() const {
  if (BaseAddress)
    return BaseAddress;
  return getFirstFunctionAddress();
}

uint64_t GsymCreator::getMaxAddressOffset() const {
  switch (getAddressOffsetSize()) {
    case 1: return UINT8_MAX;
    case 2: return UINT16_MAX;
    case 4: return UINT32_MAX;
    case 8: return UINT64_MAX;
  }
  llvm_unreachable("invalid address offset");
}

uint8_t GsymCreator::getAddressOffsetSize() const {
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

uint64_t GsymCreator::calculateHeaderAndTableSize() const {
  uint64_t Size = sizeof(Header);
  const size_t NumFuncs = Funcs.size();
  // Add size of address offset table
  Size += NumFuncs * getAddressOffsetSize();
  // Add size of address info offsets which are 32 bit integers in version 1.
  Size += NumFuncs * sizeof(uint32_t);
  // Add file table size
  Size += Files.size() * sizeof(FileEntry);
  // Add string table size
  Size += StrTab.getSize();

  return Size;
}

// This function takes a InlineInfo class that was copy constructed from an
// InlineInfo from the \a SrcGC and updates all members that point to strings
// and files to point to strings and files from this GsymCreator.
void GsymCreator::fixupInlineInfo(const GsymCreator &SrcGC, InlineInfo &II) {
  II.Name = copyString(SrcGC, II.Name);
  II.CallFile = copyFile(SrcGC, II.CallFile);
  for (auto &ChildII: II.Children)
    fixupInlineInfo(SrcGC, ChildII);
}

uint64_t GsymCreator::copyFunctionInfo(const GsymCreator &SrcGC, size_t FuncIdx) {
  // To copy a function info we need to copy any files and strings over into
  // this GsymCreator and then copy the function info and update the string
  // table offsets to match the new offsets.
  const FunctionInfo &SrcFI = SrcGC.Funcs[FuncIdx];
  Ranges.insert(SrcFI.Range);

  FunctionInfo DstFI;
  DstFI.Range = SrcFI.Range;
  DstFI.Name = copyString(SrcGC, SrcFI.Name);
  // Copy the line table if there is one.
  if (SrcFI.OptLineTable) {
    // Copy the entire line table.
    DstFI.OptLineTable = LineTable(SrcFI.OptLineTable.value());
    // Fixup all LineEntry::File entries which are indexes in the the file table
    // from SrcGC and must be converted to file indexes from this GsymCreator.
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
  Funcs.push_back(DstFI);
  return Funcs.back().cacheEncoding();
}

llvm::Error GsymCreator::saveSegments(StringRef Path,
                                      llvm::support::endianness ByteOrder,
                                      uint64_t SegmentSize) const {
  if (SegmentSize == 0)
    return createStringError(std::errc::invalid_argument,
                             "invalid segment size zero");

  size_t FuncIdx = 0;
  const size_t NumFuncs = Funcs.size();
  while (FuncIdx < NumFuncs) {
    llvm::Expected<std::unique_ptr<GsymCreator>> ExpectedGC =
        createSegment(SegmentSize, FuncIdx);
    if (ExpectedGC) {
      GsymCreator *GC = ExpectedGC->get();
      if (GC == NULL)
        break; // We had not more functions to encode.
      raw_null_ostream ErrorStrm;
      llvm::Error Err = GC->finalize(ErrorStrm);
      if (Err)
        return Err;
      std::string SegmentedGsymPath;
      raw_string_ostream SGP(SegmentedGsymPath);
      std::optional<uint64_t> FirstFuncAddr = GC->getFirstFunctionAddress();
      if (FirstFuncAddr) {
        SGP << Path << "-" << llvm::format_hex(*FirstFuncAddr, 1);
        SGP.flush();
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

llvm::Expected<std::unique_ptr<GsymCreator>>
GsymCreator::createSegment(uint64_t SegmentSize, size_t &FuncIdx) const {
  // No function entries, return empty unique pointer
  if (FuncIdx >= Funcs.size())
    return std::unique_ptr<GsymCreator>();

  std::unique_ptr<GsymCreator> GC(new GsymCreator(/*Quiet=*/true));
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
