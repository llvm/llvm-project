//===- GsymReader.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/GSYM/GsymReader.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#include "llvm/DebugInfo/GSYM/InlineInfo.h"
#include "llvm/DebugInfo/GSYM/LineTable.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace gsym;

GsymReader::GsymReader(std::unique_ptr<MemoryBuffer> Buffer)
    : MemBuffer(std::move(Buffer)), Endian(llvm::endianness::native) {}

GsymReader::GsymReader(GsymReader &&RHS) = default;

GsymReader::~GsymReader() = default;

llvm::Expected<GsymReader> GsymReader::openFile(StringRef Filename) {
  // Open the input file and return an appropriate error if needed.
  ErrorOr<std::unique_ptr<MemoryBuffer>> BuffOrErr =
      MemoryBuffer::getFileOrSTDIN(Filename);
  auto Err = BuffOrErr.getError();
  if (Err)
    return llvm::errorCodeToError(Err);
  return create(BuffOrErr.get());
}

llvm::Expected<GsymReader> GsymReader::copyBuffer(StringRef Bytes) {
  auto MemBuffer = MemoryBuffer::getMemBufferCopy(Bytes, "GSYM bytes");
  return create(MemBuffer);
}

llvm::Expected<llvm::gsym::GsymReader>
GsymReader::create(std::unique_ptr<MemoryBuffer> &MemBuffer) {
  if (!MemBuffer)
    return createStringError(std::errc::invalid_argument,
                             "invalid memory buffer");
  GsymReader GR(std::move(MemBuffer));
  llvm::Error Err = GR.parse();
  if (Err)
    return std::move(Err);
  return std::move(GR);
}

llvm::Error
GsymReader::parse() {
  BinaryStreamReader FileData(MemBuffer->getBuffer(), llvm::endianness::native);
  // Check for the magic bytes. This file format is designed to be mmap'ed
  // into a process and accessed as read only. This is done for performance
  // and efficiency for symbolicating and parsing GSYM data.
  if (FileData.readObject(Hdr))
    return createStringError(std::errc::invalid_argument,
                             "not enough data for a GSYM header");

  const auto HostByteOrder = llvm::endianness::native;
  switch (Hdr->Magic) {
    case GSYM_MAGIC:
      Endian = HostByteOrder;
      break;
    case GSYM_CIGAM:
      // This is a GSYM file, but not native endianness.
      Endian = sys::IsBigEndianHost ? llvm::endianness::little
                                    : llvm::endianness::big;
      Swap.reset(new SwappedData);
      break;
    default:
      return createStringError(std::errc::invalid_argument,
                               "not a GSYM file");
  }

  bool DataIsLittleEndian = HostByteOrder != llvm::endianness::little;
  // Read a correctly byte swapped header if we need to.
  if (Swap) {
    DataExtractor Data(MemBuffer->getBuffer(), DataIsLittleEndian, 4);
    if (auto ExpectedHdr = Header::decode(Data))
      Swap->Hdr = ExpectedHdr.get();
    else
      return ExpectedHdr.takeError();
    Hdr = &Swap->Hdr;
  }

  // Detect errors in the header and report any that are found. If we make it
  // past this without errors, we know we have a good magic value, a supported
  // version number, verified address offset size and a valid UUID size.
  if (Error Err = Hdr->checkForError())
    return Err;

  if (!Swap) {
    // This is the native endianness case that is most common and optimized for
    // efficient lookups. Here we just grab pointers to the native data and
    // use ArrayRef objects to allow efficient read only access.

    // Read the address offsets.
    if (FileData.padToAlignment(Hdr->AddrOffSize) ||
        FileData.readArray(AddrOffsets,
                           Hdr->NumAddresses * Hdr->AddrOffSize))
      return createStringError(std::errc::invalid_argument,
                              "failed to read address table");

    // Read the address info offsets.
    if (FileData.padToAlignment(4) ||
        FileData.readArray(AddrInfoOffsets, Hdr->NumAddresses))
      return createStringError(std::errc::invalid_argument,
                              "failed to read address info offsets table");

    // Read the file table.
    uint32_t NumFiles = 0;
    if (FileData.readInteger(NumFiles) || FileData.readArray(Files, NumFiles))
      return createStringError(std::errc::invalid_argument,
                              "failed to read file table");

    // Get the string table.
    FileData.setOffset(Hdr->StrtabOffset);
    if (FileData.readFixedString(StrTab.Data, Hdr->StrtabSize))
      return createStringError(std::errc::invalid_argument,
                              "failed to read string table");
} else {
  // This is the non native endianness case that is not common and not
  // optimized for lookups. Here we decode the important tables into local
  // storage and then set the ArrayRef objects to point to these swapped
  // copies of the read only data so lookups can be as efficient as possible.
  DataExtractor Data(MemBuffer->getBuffer(), DataIsLittleEndian, 4);

  // Read the address offsets.
  uint64_t Offset = alignTo(sizeof(Header), Hdr->AddrOffSize);
  Swap->AddrOffsets.resize(Hdr->NumAddresses * Hdr->AddrOffSize);
  switch (Hdr->AddrOffSize) {
    case 1:
      if (!Data.getU8(&Offset, Swap->AddrOffsets.data(), Hdr->NumAddresses))
        return createStringError(std::errc::invalid_argument,
                                  "failed to read address table");
      break;
    case 2:
      if (!Data.getU16(&Offset,
                        reinterpret_cast<uint16_t *>(Swap->AddrOffsets.data()),
                        Hdr->NumAddresses))
        return createStringError(std::errc::invalid_argument,
                                  "failed to read address table");
      break;
    case 4:
      if (!Data.getU32(&Offset,
                        reinterpret_cast<uint32_t *>(Swap->AddrOffsets.data()),
                        Hdr->NumAddresses))
        return createStringError(std::errc::invalid_argument,
                                  "failed to read address table");
      break;
    case 8:
      if (!Data.getU64(&Offset,
                        reinterpret_cast<uint64_t *>(Swap->AddrOffsets.data()),
                        Hdr->NumAddresses))
        return createStringError(std::errc::invalid_argument,
                                  "failed to read address table");
    }
    AddrOffsets = ArrayRef<uint8_t>(Swap->AddrOffsets);

    // Read the address info offsets.
    Offset = alignTo(Offset, 4);
    Swap->AddrInfoOffsets.resize(Hdr->NumAddresses);
    if (Data.getU32(&Offset, Swap->AddrInfoOffsets.data(), Hdr->NumAddresses))
      AddrInfoOffsets = ArrayRef<uint32_t>(Swap->AddrInfoOffsets);
    else
      return createStringError(std::errc::invalid_argument,
                               "failed to read address table");
    // Read the file table.
    const uint32_t NumFiles = Data.getU32(&Offset);
    if (NumFiles > 0) {
      Swap->Files.resize(NumFiles);
      if (Data.getU32(&Offset, &Swap->Files[0].Dir, NumFiles*2))
        Files = ArrayRef<FileEntry>(Swap->Files);
      else
        return createStringError(std::errc::invalid_argument,
                                 "failed to read file table");
    }
    // Get the string table.
    StrTab.Data = MemBuffer->getBuffer().substr(Hdr->StrtabOffset,
                                                Hdr->StrtabSize);
    if (StrTab.Data.empty())
      return createStringError(std::errc::invalid_argument,
                               "failed to read string table");
  }
  return Error::success();

}

const Header &GsymReader::getHeader() const {
  // The only way to get a GsymReader is from GsymReader::openFile(...) or
  // GsymReader::copyBuffer() and the header must be valid and initialized to
  // a valid pointer value, so the assert below should not trigger.
  assert(Hdr);
  return *Hdr;
}

std::optional<uint64_t> GsymReader::getAddress(size_t Index) const {
  switch (Hdr->AddrOffSize) {
  case 1: return addressForIndex<uint8_t>(Index);
  case 2: return addressForIndex<uint16_t>(Index);
  case 4: return addressForIndex<uint32_t>(Index);
  case 8: return addressForIndex<uint64_t>(Index);
  }
  return std::nullopt;
}

std::optional<uint64_t> GsymReader::getAddressInfoOffset(size_t Index) const {
  const auto NumAddrInfoOffsets = AddrInfoOffsets.size();
  if (Index < NumAddrInfoOffsets)
    return AddrInfoOffsets[Index];
  return std::nullopt;
}

Expected<uint64_t>
GsymReader::getAddressIndex(const uint64_t Addr) const {
  if (Addr >= Hdr->BaseAddress) {
    const uint64_t AddrOffset = Addr - Hdr->BaseAddress;
    std::optional<uint64_t> AddrOffsetIndex;
    switch (Hdr->AddrOffSize) {
    case 1:
      AddrOffsetIndex = getAddressOffsetIndex<uint8_t>(AddrOffset);
      break;
    case 2:
      AddrOffsetIndex = getAddressOffsetIndex<uint16_t>(AddrOffset);
      break;
    case 4:
      AddrOffsetIndex = getAddressOffsetIndex<uint32_t>(AddrOffset);
      break;
    case 8:
      AddrOffsetIndex = getAddressOffsetIndex<uint64_t>(AddrOffset);
      break;
    default:
      return createStringError(std::errc::invalid_argument,
                               "unsupported address offset size %u",
                               Hdr->AddrOffSize);
    }
    if (AddrOffsetIndex)
      return *AddrOffsetIndex;
  }
  return createStringError(std::errc::invalid_argument,
                           "address 0x%" PRIx64 " is not in GSYM", Addr);

}

llvm::Expected<DataExtractor>
GsymReader::getFunctionInfoDataForAddress(uint64_t Addr,
                                          uint64_t &FuncStartAddr) const {
  Expected<uint64_t> ExpectedAddrIdx = getAddressIndex(Addr);
  if (!ExpectedAddrIdx)
    return ExpectedAddrIdx.takeError();
  const uint64_t FirstAddrIdx = *ExpectedAddrIdx;
  // The AddrIdx is the first index of the function info entries that match
  // \a Addr. We need to iterate over all function info objects that start with
  // the same address until we find a range that contains \a Addr.
  std::optional<uint64_t> FirstFuncStartAddr;
  const size_t NumAddresses = getNumAddresses();
  for (uint64_t AddrIdx = FirstAddrIdx; AddrIdx < NumAddresses; ++AddrIdx) {
    auto ExpextedData = getFunctionInfoDataAtIndex(AddrIdx, FuncStartAddr);
    // If there was an error, return the error.
    if (!ExpextedData)
      return ExpextedData;

    // Remember the first function start address if it hasn't already been set.
    // If it is already valid, check to see if it matches the first function
    // start address and only continue if it matches.
    if (FirstFuncStartAddr.has_value()) {
      if (*FirstFuncStartAddr != FuncStartAddr)
        break; // Done with consecutive function entries with same address.
    } else {
      FirstFuncStartAddr = FuncStartAddr;
    }
    // Make sure the current function address ranges contains \a Addr.
    // Some symbols on Darwin don't have valid sizes, so if we run into a
    // symbol with zero size, then we have found a match for our address.

    // The first thing the encoding of a FunctionInfo object is the function
    // size.
    uint64_t Offset = 0;
    uint32_t FuncSize = ExpextedData->getU32(&Offset);
    if (FuncSize == 0 ||
        AddressRange(FuncStartAddr, FuncStartAddr + FuncSize).contains(Addr))
      return ExpextedData;
  }
  return createStringError(std::errc::invalid_argument,
                           "address 0x%" PRIx64 " is not in GSYM", Addr);
}

llvm::Expected<DataExtractor>
GsymReader::getFunctionInfoDataAtIndex(uint64_t AddrIdx,
                                       uint64_t &FuncStartAddr) const {
  if (AddrIdx >= getNumAddresses())
    return createStringError(std::errc::invalid_argument,
                             "invalid address index %" PRIu64, AddrIdx);
  const uint32_t AddrInfoOffset = AddrInfoOffsets[AddrIdx];
  assert((Endian == endianness::big || Endian == endianness::little) &&
         "Endian must be either big or little");
  StringRef Bytes = MemBuffer->getBuffer().substr(AddrInfoOffset);
  if (Bytes.empty())
    return createStringError(std::errc::invalid_argument,
                             "invalid address info offset 0x%" PRIx32,
                             AddrInfoOffset);
  std::optional<uint64_t> OptFuncStartAddr = getAddress(AddrIdx);
  if (!OptFuncStartAddr)
    return createStringError(std::errc::invalid_argument,
                             "failed to extract address[%" PRIu64 "]", AddrIdx);
  FuncStartAddr = *OptFuncStartAddr;
  return DataExtractor(Bytes, Endian == llvm::endianness::little, 4);
}

llvm::Expected<FunctionInfo> GsymReader::getFunctionInfo(uint64_t Addr) const {
  uint64_t FuncStartAddr = 0;
  if (auto ExpectedData = getFunctionInfoDataForAddress(Addr, FuncStartAddr))
    return FunctionInfo::decode(*ExpectedData, FuncStartAddr);
  else
    return ExpectedData.takeError();
}

llvm::Expected<FunctionInfo>
GsymReader::getFunctionInfoAtIndex(uint64_t Idx) const {
  uint64_t FuncStartAddr = 0;
  if (auto ExpectedData = getFunctionInfoDataAtIndex(Idx, FuncStartAddr))
    return FunctionInfo::decode(*ExpectedData, FuncStartAddr);
  else
    return ExpectedData.takeError();
}

llvm::Expected<LookupResult>
GsymReader::lookup(uint64_t Addr,
                   std::optional<DataExtractor> *MergedFunctionsData) const {
  uint64_t FuncStartAddr = 0;
  if (auto ExpectedData = getFunctionInfoDataForAddress(Addr, FuncStartAddr))
    return FunctionInfo::lookup(*ExpectedData, *this, FuncStartAddr, Addr,
                                MergedFunctionsData);
  else
    return ExpectedData.takeError();
}

llvm::Expected<std::vector<LookupResult>>
GsymReader::lookupAll(uint64_t Addr) const {
  std::vector<LookupResult> Results;
  std::optional<DataExtractor> MergedFunctionsData;

  // First perform a lookup to get the primary function info result.
  auto MainResult = lookup(Addr, &MergedFunctionsData);
  if (!MainResult)
    return MainResult.takeError();

  // Add the main result as the first entry.
  Results.push_back(std::move(*MainResult));

  // Now process any merged functions data that was found during the lookup.
  if (MergedFunctionsData) {
    // Get data extractors for each merged function.
    auto ExpectedMergedFuncExtractors =
        MergedFunctionsInfo::getFuncsDataExtractors(*MergedFunctionsData);
    if (!ExpectedMergedFuncExtractors)
      return ExpectedMergedFuncExtractors.takeError();

    // Process each merged function data.
    for (DataExtractor &MergedData : *ExpectedMergedFuncExtractors) {
      if (auto FI = FunctionInfo::lookup(MergedData, *this,
                                         MainResult->FuncRange.start(), Addr)) {
        Results.push_back(std::move(*FI));
      } else {
        return FI.takeError();
      }
    }
  }

  return Results;
}

void GsymReader::dump(raw_ostream &OS) {
  const auto &Header = getHeader();
  // Dump the GSYM header.
  OS << Header << "\n";
  // Dump the address table.
  OS << "Address Table:\n";
  OS << "INDEX  OFFSET";

  switch (Hdr->AddrOffSize) {
  case 1: OS << "8 "; break;
  case 2: OS << "16"; break;
  case 4: OS << "32"; break;
  case 8: OS << "64"; break;
  default: OS << "??"; break;
  }
  OS << " (ADDRESS)\n";
  OS << "====== =============================== \n";
  for (uint32_t I = 0; I < Header.NumAddresses; ++I) {
    OS << format("[%4u] ", I);
    switch (Hdr->AddrOffSize) {
    case 1: OS << HEX8(getAddrOffsets<uint8_t>()[I]); break;
    case 2: OS << HEX16(getAddrOffsets<uint16_t>()[I]); break;
    case 4: OS << HEX32(getAddrOffsets<uint32_t>()[I]); break;
    case 8: OS << HEX32(getAddrOffsets<uint64_t>()[I]); break;
    default: break;
    }
    OS << " (" << HEX64(*getAddress(I)) << ")\n";
  }
  // Dump the address info offsets table.
  OS << "\nAddress Info Offsets:\n";
  OS << "INDEX  Offset\n";
  OS << "====== ==========\n";
  for (uint32_t I = 0; I < Header.NumAddresses; ++I)
    OS << format("[%4u] ", I) << HEX32(AddrInfoOffsets[I]) << "\n";
  // Dump the file table.
  OS << "\nFiles:\n";
  OS << "INDEX  DIRECTORY  BASENAME   PATH\n";
  OS << "====== ========== ========== ==============================\n";
  for (uint32_t I = 0; I < Files.size(); ++I) {
    OS << format("[%4u] ", I) << HEX32(Files[I].Dir) << ' '
       << HEX32(Files[I].Base) << ' ';
    dump(OS, getFile(I));
    OS << "\n";
  }
  OS << "\n" << StrTab << "\n";

  for (uint32_t I = 0; I < Header.NumAddresses; ++I) {
    OS << "FunctionInfo @ " << HEX32(AddrInfoOffsets[I]) << ": ";
    if (auto FI = getFunctionInfoAtIndex(I))
      dump(OS, *FI);
    else
      logAllUnhandledErrors(FI.takeError(), OS, "FunctionInfo:");
  }
}

void GsymReader::dumpStatistics(StringRef GSYMPath, raw_ostream &OS,
                                StatisticsFormat Format) {
  const auto &Hdr = getHeader();

  // Get the total file size.
  uint64_t FileSize = 0;
  if (auto EC = sys::fs::file_size(GSYMPath, FileSize)) {
    OS << "error: cannot stat file '" << GSYMPath << "': " << EC.message()
       << "\n";
    return;
  }

  // Header section.
  const uint64_t HeaderSize = sizeof(Header);

  // Address table: aligned to AddrOffSize, contains NumAddresses entries.
  const uint64_t AddrTableStart = alignTo(HeaderSize, Hdr.AddrOffSize);
  const uint64_t AddrTableSize =
      static_cast<uint64_t>(Hdr.NumAddresses) * Hdr.AddrOffSize;

  // Address info offsets table: aligned to 4 bytes, NumAddresses uint32_t's.
  const uint64_t AddrInfoOffsetsStart =
      alignTo(AddrTableStart + AddrTableSize, sizeof(uint32_t));
  const uint64_t AddrInfoOffsetsSize =
      static_cast<uint64_t>(Hdr.NumAddresses) * sizeof(uint32_t);

  // File table: immediately follows address info offsets.
  const uint64_t FileTableStart = AddrInfoOffsetsStart + AddrInfoOffsetsSize;
  const uint64_t FileTableSize = Hdr.StrtabOffset - FileTableStart;

  // String table: offset and size from header.
  const uint64_t StrtabSize = Hdr.StrtabSize;

  // Function info data: everything after the string table to end of file.
  const uint64_t FuncInfoStart =
      static_cast<uint64_t>(Hdr.StrtabOffset) + Hdr.StrtabSize;
  const uint64_t FuncInfoSize = FileSize - FuncInfoStart;

  // Collect per-InfoType statistics across all functions.
  std::map<uint32_t, uint64_t> InfoTypeStats;
  std::map<uint32_t, uint64_t> MergedInfoTypeStats;
  for (uint32_t I = 0; I < Hdr.NumAddresses; ++I) {
    uint64_t FuncStartAddr = 0;
    if (auto ExpData = getFunctionInfoDataAtIndex(I, FuncStartAddr)) {
      DataExtractor Data = *ExpData;
      FunctionInfo::parseStatistics(Data, InfoTypeStats, &MergedInfoTypeStats);
    } else {
      consumeError(ExpData.takeError());
    }
  }

  auto GetStat = [](const std::map<uint32_t, uint64_t> &Stats, uint32_t Type) {
    auto It = Stats.find(Type);
    return It != Stats.end() ? It->second : 0;
  };

  if (Format == StatisticsFormat::JSON ||
      Format == StatisticsFormat::PrettyJSON) {
    json::Object MergedFuncInfo{
        {"line_table_info",
         static_cast<int64_t>(GetStat(MergedInfoTypeStats, 1))},
        {"inline_info", static_cast<int64_t>(GetStat(MergedInfoTypeStats, 2))},
        {"merged_func_info",
         static_cast<int64_t>(GetStat(MergedInfoTypeStats, 3))},
        {"call_site_info",
         static_cast<int64_t>(GetStat(MergedInfoTypeStats, 4))},
        {"overhead", static_cast<int64_t>(GetStat(MergedInfoTypeStats, 9))}};

    json::Object FuncInfoData{
        {"line_table_info", static_cast<int64_t>(GetStat(InfoTypeStats, 1))},
        {"inline_info", static_cast<int64_t>(GetStat(InfoTypeStats, 2))},
        {"call_site_info", static_cast<int64_t>(GetStat(InfoTypeStats, 4))},
        {"overhead", static_cast<int64_t>(GetStat(InfoTypeStats, 9))},
        {"merged_func_info",
         json::Object{
             {"total", static_cast<int64_t>(GetStat(InfoTypeStats, 3))},
             {"breakdown", std::move(MergedFuncInfo)}}}};

    json::Object RootObj{
        {"file", GSYMPath.str()},
        {"file_size", static_cast<int64_t>(FileSize)},
        {"header", static_cast<int64_t>(HeaderSize)},
        {"address_table", static_cast<int64_t>(AddrTableSize)},
        {"addr_info_offsets", static_cast<int64_t>(AddrInfoOffsetsSize)},
        {"file_table", static_cast<int64_t>(FileTableSize)},
        {"string_table", static_cast<int64_t>(StrtabSize)},
        {"function_info_data",
         json::Object{{"total", static_cast<int64_t>(FuncInfoSize)},
                      {"breakdown", std::move(FuncInfoData)}}},
        {"num_addresses", static_cast<int64_t>(Hdr.NumAddresses)}};

    json::Value Root(std::move(RootObj));
    if (Format == StatisticsFormat::PrettyJSON)
      OS << formatv("{0:2}", Root) << "\n";
    else
      OS << Root << "\n";
    return;
  }

  // Text format output.
  auto Fmt = [](uint64_t Value) {
    std::string Num = std::to_string(Value);
    int InsertPosition = Num.length() - 3;
    while (InsertPosition > 0) {
      Num.insert(InsertPosition, ",");
      InsertPosition -= 3;
    }
    return std::string(std::max((size_t)0, 14 - Num.length()), ' ') + Num;
  };

  auto Pct = [&](uint64_t Value) -> std::string {
    char Buf[16];
    snprintf(Buf, sizeof(Buf), "(%5.2f%%)", 100.0 * Value / FileSize);
    return Buf;
  };

  OS << "GSYM statistics for \"" << GSYMPath << "\":\n";
  OS << "  File size:           " << Fmt(FileSize) << " bytes\n";
  OS << "  Header:              " << Fmt(HeaderSize) << " bytes "
     << Pct(HeaderSize) << "\n";
  OS << "  Address table:       " << Fmt(AddrTableSize) << " bytes "
     << Pct(AddrTableSize) << "\n";
  OS << "  Addr info offsets:   " << Fmt(AddrInfoOffsetsSize) << " bytes "
     << Pct(AddrInfoOffsetsSize) << "\n";
  OS << "  File table:          " << Fmt(FileTableSize) << " bytes "
     << Pct(FileTableSize) << "\n";
  OS << "  String table:        " << Fmt(StrtabSize) << " bytes "
     << Pct(StrtabSize) << "\n";
  OS << "  Function info data:  " << Fmt(FuncInfoSize) << " bytes "
     << Pct(FuncInfoSize) << "\n";

  auto PrintInfoType = [&](const char *Indent, const char *Name, uint32_t Type,
                           const std::map<uint32_t, uint64_t> &Stats) {
    auto It = Stats.find(Type);
    if (It != Stats.end())
      OS << Indent << Name << Fmt(It->second) << " bytes " << Pct(It->second)
         << "\n";
  };
  PrintInfoType("    ", "Line table info:   ", 1, InfoTypeStats);
  PrintInfoType("    ", "Inline info:       ", 2, InfoTypeStats);
  PrintInfoType("    ", "Call site info:    ", 4, InfoTypeStats);
  PrintInfoType("    ", "Overhead:          ", 9, InfoTypeStats);
  PrintInfoType("    ", "Merged func info:  ", 3, InfoTypeStats);
  PrintInfoType("      ", "Line table info: ", 1, MergedInfoTypeStats);
  PrintInfoType("      ", "Inline info:     ", 2, MergedInfoTypeStats);
  PrintInfoType("      ", "Merged func info:", 3, MergedInfoTypeStats);
  PrintInfoType("      ", "Call site info:  ", 4, MergedInfoTypeStats);
  PrintInfoType("      ", "Overhead:        ", 9, MergedInfoTypeStats);

  OS << "  Number of addresses: " << Fmt(Hdr.NumAddresses) << "\n";
}

void GsymReader::dump(raw_ostream &OS, const FunctionInfo &FI,
                      uint32_t Indent) {
  OS.indent(Indent);
  OS << FI.Range << " \"" << getString(FI.Name) << "\"\n";
  if (FI.OptLineTable)
    dump(OS, *FI.OptLineTable, Indent);
  if (FI.Inline)
    dump(OS, *FI.Inline, Indent);

  if (FI.CallSites)
    dump(OS, *FI.CallSites, Indent);

  if (FI.MergedFunctions) {
    assert(Indent == 0 && "MergedFunctionsInfo should only exist at top level");
    dump(OS, *FI.MergedFunctions);
  }
}

void GsymReader::dump(raw_ostream &OS, const MergedFunctionsInfo &MFI) {
  for (uint32_t inx = 0; inx < MFI.MergedFunctions.size(); inx++) {
    OS << "++ Merged FunctionInfos[" << inx << "]:\n";
    dump(OS, MFI.MergedFunctions[inx], 4);
  }
}

void GsymReader::dump(raw_ostream &OS, const CallSiteInfo &CSI) {
  OS << HEX16(CSI.ReturnOffset);

  std::string Flags;
  auto addFlag = [&](const char *Flag) {
    if (!Flags.empty())
      Flags += " | ";
    Flags += Flag;
  };

  if (CSI.Flags == CallSiteInfo::Flags::None)
    Flags = "None";
  else {
    if (CSI.Flags & CallSiteInfo::Flags::InternalCall)
      addFlag("InternalCall");

    if (CSI.Flags & CallSiteInfo::Flags::ExternalCall)
      addFlag("ExternalCall");
  }
  OS << " Flags[" << Flags << "]";

  if (!CSI.MatchRegex.empty()) {
    OS << " MatchRegex[";
    for (uint32_t i = 0; i < CSI.MatchRegex.size(); ++i) {
      if (i > 0)
        OS << ";";
      OS << getString(CSI.MatchRegex[i]);
    }
    OS << "]";
  }
}

void GsymReader::dump(raw_ostream &OS, const CallSiteInfoCollection &CSIC,
                      uint32_t Indent) {
  OS.indent(Indent);
  OS << "CallSites (by relative return offset):\n";
  for (const auto &CS : CSIC.CallSites) {
    OS.indent(Indent);
    OS << "  ";
    dump(OS, CS);
    OS << "\n";
  }
}

void GsymReader::dump(raw_ostream &OS, const LineTable &LT, uint32_t Indent) {
  OS.indent(Indent);
  OS << "LineTable:\n";
  for (auto &LE: LT) {
    OS.indent(Indent);
    OS << "  " << HEX64(LE.Addr) << ' ';
    if (LE.File)
      dump(OS, getFile(LE.File));
    OS << ':' << LE.Line << '\n';
  }
}

void GsymReader::dump(raw_ostream &OS, const InlineInfo &II, uint32_t Indent) {
  if (Indent == 0)
    OS << "InlineInfo:\n";
  else
    OS.indent(Indent);
  OS << II.Ranges << ' ' << getString(II.Name);
  if (II.CallFile != 0) {
    if (auto File = getFile(II.CallFile)) {
      OS << " called from ";
      dump(OS, File);
      OS << ':' << II.CallLine;
    }
  }
  OS << '\n';
  for (const auto &ChildII: II.Children)
    dump(OS, ChildII, Indent + 2);
}

void GsymReader::dump(raw_ostream &OS, std::optional<FileEntry> FE) {
  if (FE) {
    // IF we have the file from index 0, then don't print anything
    if (FE->Dir == 0 && FE->Base == 0)
      return;
    StringRef Dir = getString(FE->Dir);
    StringRef Base = getString(FE->Base);
    if (!Dir.empty()) {
      OS << Dir;
      if (Dir.contains('\\') && !Dir.contains('/'))
        OS << '\\';
      else
        OS << '/';
    }
    if (!Base.empty()) {
      OS << Base;
    }
    if (!Dir.empty() || !Base.empty())
      return;
  }
  OS << "<invalid-file>";
}
