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


GsymCreator::GsymCreator() : StrTab(StringTableBuilder::ELF) {
  insertFile(StringRef());
}

uint32_t GsymCreator::insertFile(StringRef Path,
                                 llvm::sys::path::Style Style) {
  llvm::StringRef directory = llvm::sys::path::parent_path(Path, Style);
  llvm::StringRef filename = llvm::sys::path::filename(Path, Style);
  // We must insert the strings first, then call the FileEntry constructor.
  // If we inline the insertString() function call into the constructor, the
  // call order is undefined due to parameter lists not having any ordering
  // requirements.
  const uint32_t Dir = insertString(directory);
  const uint32_t Base = insertString(filename);
  FileEntry FE(Dir, Base);

  std::lock_guard<std::recursive_mutex> Guard(Mutex);
  const auto NextIndex = Files.size();
  // Find FE in hash map and insert if not present.
  auto R = FileEntryToIndex.insert(std::make_pair(FE, NextIndex));
  if (R.second)
    Files.emplace_back(FE);
  return R.first->second;
}

llvm::Error GsymCreator::save(StringRef Path,
                              llvm::support::endianness ByteOrder) const {
  std::error_code EC;
  raw_fd_ostream OutStrm(Path, EC);
  if (EC)
    return llvm::errorCodeToError(EC);
  FileWriter O(OutStrm, ByteOrder);
  return encode(O);
}

llvm::Error GsymCreator::encode(FileWriter &O) const {
  std::lock_guard<std::recursive_mutex> Guard(Mutex);
  if (Funcs.empty())
    return createStringError(std::errc::invalid_argument,
                             "no functions to encode");
  if (!Finalized)
    return createStringError(std::errc::invalid_argument,
                             "GsymCreator wasn't finalized prior to encoding");

  if (Funcs.size() > UINT32_MAX)
    return createStringError(std::errc::invalid_argument,
                             "too many FunctionInfos");

  const uint64_t MinAddr = BaseAddress ? *BaseAddress : Funcs.front().startAddress();
  const uint64_t MaxAddr = Funcs.back().startAddress();
  const uint64_t AddrDelta = MaxAddr - MinAddr;
  Header Hdr;
  Hdr.Magic = GSYM_MAGIC;
  Hdr.Version = GSYM_VERSION;
  Hdr.AddrOffSize = 0;
  Hdr.UUIDSize = static_cast<uint8_t>(UUID.size());
  Hdr.BaseAddress = MinAddr;
  Hdr.NumAddresses = static_cast<uint32_t>(Funcs.size());
  Hdr.StrtabOffset = 0; // We will fix this up later.
  Hdr.StrtabSize = 0; // We will fix this up later.
  memset(Hdr.UUID, 0, sizeof(Hdr.UUID));
  if (UUID.size() > sizeof(Hdr.UUID))
    return createStringError(std::errc::invalid_argument,
                             "invalid UUID size %u", (uint32_t)UUID.size());
  // Set the address offset size correctly in the GSYM header.
  if (AddrDelta <= UINT8_MAX)
    Hdr.AddrOffSize = 1;
  else if (AddrDelta <= UINT16_MAX)
    Hdr.AddrOffSize = 2;
  else if (AddrDelta <= UINT32_MAX)
    Hdr.AddrOffSize = 4;
  else
    Hdr.AddrOffSize = 8;
  // Copy the UUID value if we have one.
  if (UUID.size() > 0)
    memcpy(Hdr.UUID, UUID.data(), UUID.size());
  // Write out the header.
  llvm::Error Err = Hdr.encode(O);
  if (Err)
    return Err;

  // Write out the address offsets.
  O.alignTo(Hdr.AddrOffSize);
  for (const auto &FuncInfo : Funcs) {
    uint64_t AddrOffset = FuncInfo.startAddress() - Hdr.BaseAddress;
    switch(Hdr.AddrOffSize) {
      case 1: O.writeU8(static_cast<uint8_t>(AddrOffset)); break;
      case 2: O.writeU16(static_cast<uint16_t>(AddrOffset)); break;
      case 4: O.writeU32(static_cast<uint32_t>(AddrOffset)); break;
      case 8: O.writeU64(AddrOffset); break;
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
    return createStringError(std::errc::invalid_argument,
                             "too many files");
  O.writeU32(static_cast<uint32_t>(NumFiles));
  for (auto File: Files) {
      O.writeU32(File.Dir);
      O.writeU32(File.Base);
  }

  // Write out the sting table.
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
  for (auto AddrInfoOffset: AddrInfoOffsets) {
    O.fixup32(AddrInfoOffset, AddrInfoOffsetsOffset + Offset);
    Offset += 4;
  }
  return ErrorSuccess();
}

llvm::Error GsymCreator::finalize(llvm::raw_ostream &OS) {
  std::lock_guard<std::recursive_mutex> Guard(Mutex);
  if (Finalized)
    return createStringError(std::errc::invalid_argument,
                             "already finalized");
  Finalized = true;

  // Sort function infos so we can emit sorted functions.
  llvm::sort(Funcs.begin(), Funcs.end());

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
  auto Curr = Funcs.begin();
  auto Prev = Funcs.end();
  while (Curr != Funcs.end()) {
    // Can't check for overlaps or same address ranges if we don't have a
    // previous entry
    if (Prev != Funcs.end()) {
      if (Prev->Range.intersects(Curr->Range)) {
        // Overlapping address ranges.
        if (Prev->Range == Curr->Range) {
          // Same address range. Check if one is from debug info and the other
          // is from a symbol table. If so, then keep the one with debug info.
          // Our sorting guarantees that entries with matching address ranges
          // that have debug info are last in the sort.
          if (*Prev == *Curr) {
            // FunctionInfo entries match exactly (range, lines, inlines)
            OS << "warning: duplicate function info entries for range: "
               << Curr->Range << '\n';
            Curr = Funcs.erase(Prev);
          } else {
            if (!Prev->hasRichInfo() && Curr->hasRichInfo()) {
              // Same address range, one with no debug info (symbol) and the
              // next with debug info. Keep the latter.
              Curr = Funcs.erase(Prev);
            } else {
              OS << "warning: same address range contains different debug "
                 << "info. Removing:\n"
                 << *Prev << "\nIn favor of this one:\n"
                 << *Curr << "\n";
              Curr = Funcs.erase(Prev);
            }
          }
        } else {
          // print warnings about overlaps
          OS << "warning: function ranges overlap:\n"
             << *Prev << "\n"
             << *Curr << "\n";
        }
      } else if (Prev->Range.size() == 0 &&
                 Curr->Range.contains(Prev->Range.Start)) {
        OS << "warning: removing symbol:\n"
           << *Prev << "\nKeeping:\n"
           << *Curr << "\n";
        Curr = Funcs.erase(Prev);
      }
    }
    if (Curr == Funcs.end())
      break;
    Prev = Curr++;
  }

  // If our last function info entry doesn't have a size and if we have valid
  // text ranges, we should set the size of the last entry since any search for
  // a high address might match our last entry. By fixing up this size, we can
  // help ensure we don't cause lookups to always return the last symbol that
  // has no size when doing lookups.
  if (!Funcs.empty() && Funcs.back().Range.size() == 0 && ValidTextRanges) {
    if (auto Range = ValidTextRanges->getRangeThatContains(
          Funcs.back().Range.Start)) {
      Funcs.back().Range.End = Range->End;
    }
  }
  OS << "Pruned " << NumBefore - Funcs.size() << " functions, ended with "
     << Funcs.size() << " total\n";
  return Error::success();
}

uint32_t GsymCreator::insertString(StringRef S, bool Copy) {
  if (S.empty())
    return 0;
  std::lock_guard<std::recursive_mutex> Guard(Mutex);
  if (Copy) {
    // We need to provide backing storage for the string if requested
    // since StringTableBuilder stores references to strings. Any string
    // that comes from a section in an object file doesn't need to be
    // copied, but any string created by code will need to be copied.
    // This allows GsymCreator to be really fast when parsing DWARF and
    // other object files as most strings don't need to be copied.
    CachedHashStringRef CHStr(S);
    if (!StrTab.contains(CHStr))
      S = StringStorage.insert(S).first->getKey();
  }
  return StrTab.add(S);
}

void GsymCreator::addFunctionInfo(FunctionInfo &&FI) {
  std::lock_guard<std::recursive_mutex> Guard(Mutex);
  Ranges.insert(FI.Range);
  Funcs.emplace_back(FI);
}

void GsymCreator::forEachFunctionInfo(
    std::function<bool(FunctionInfo &)> const &Callback) {
  std::lock_guard<std::recursive_mutex> Guard(Mutex);
  for (auto &FI : Funcs) {
    if (!Callback(FI))
      break;
  }
}

void GsymCreator::forEachFunctionInfo(
    std::function<bool(const FunctionInfo &)> const &Callback) const {
  std::lock_guard<std::recursive_mutex> Guard(Mutex);
  for (const auto &FI : Funcs) {
    if (!Callback(FI))
      break;
  }
}

size_t GsymCreator::getNumFunctionInfos() const{
  std::lock_guard<std::recursive_mutex> Guard(Mutex);
  return Funcs.size();
}

bool GsymCreator::IsValidTextAddress(uint64_t Addr) const {
  if (ValidTextRanges)
    return ValidTextRanges->contains(Addr);
  return true; // No valid text ranges has been set, so accept all ranges.
}

bool GsymCreator::hasFunctionInfoForAddress(uint64_t Addr) const {
  std::lock_guard<std::recursive_mutex> Guard(Mutex);
  return Ranges.contains(Addr);
}
