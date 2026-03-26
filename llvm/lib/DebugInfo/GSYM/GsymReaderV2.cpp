//===- GsymReaderV2.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/GSYM/GsymReaderV2.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#include "llvm/DebugInfo/GSYM/GlobalData.h"
#include "llvm/DebugInfo/GSYM/InlineInfo.h"
#include "llvm/DebugInfo/GSYM/LineTable.h"
#include "llvm/Support/BinaryStreamReader.h"
#include "llvm/Support/DataExtractor.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace gsym;

GsymReaderV2::GsymReaderV2(std::unique_ptr<MemoryBuffer> Buffer)
    : MemBuffer(std::move(Buffer)), Endian(llvm::endianness::native) {}

GsymReaderV2::GsymReaderV2(GsymReaderV2 &&RHS) = default;

GsymReaderV2::~GsymReaderV2() = default;

llvm::Expected<GsymReaderV2> GsymReaderV2::openFile(StringRef Filename) {
  // Open the input file and return an appropriate error if needed.
  ErrorOr<std::unique_ptr<MemoryBuffer>> BuffOrErr =
      MemoryBuffer::getFileOrSTDIN(Filename);
  auto Err = BuffOrErr.getError();
  if (Err)
    return llvm::errorCodeToError(Err);
  return create(BuffOrErr.get());
}

llvm::Expected<GsymReaderV2> GsymReaderV2::copyBuffer(StringRef Bytes) {
  auto MemBuffer = MemoryBuffer::getMemBufferCopy(Bytes, "GSYM bytes");
  return create(MemBuffer);
}

llvm::Expected<llvm::gsym::GsymReaderV2>
GsymReaderV2::create(std::unique_ptr<MemoryBuffer> &MemBuffer) {
  if (!MemBuffer)
    return createStringError(std::errc::invalid_argument,
                             "invalid memory buffer");
  GsymReaderV2 GR(std::move(MemBuffer));
  llvm::Error Err = GR.parse();
  if (Err)
    return std::move(Err);
  return std::move(GR);
}

/// Helper to parse GlobalData entries and populate section offsets/sizes.
/// Works for both native and swapped endianness paths.
static llvm::Error
parseGlobalDataEntries(DataExtractor &DE, uint64_t &Offset,
                       uint64_t BufSize,
                       uint64_t &AddrOffsetsOff, uint64_t &AddrOffsetsSize,
                       uint64_t &AddrInfoOffsetsOff, uint64_t &AddrInfoOffsetsSize,
                       uint64_t &StringTableOff, uint64_t &StringTableSize,
                       uint64_t &FileTableOff, uint64_t &FileTableSize,
                       uint64_t &FuncInfoOff, uint64_t &FuncInfoSize) {
  while (Offset + 24 <= BufSize) {
    auto Type = static_cast<GlobalInfoType>(DE.getU32(&Offset));
    uint32_t Pad = DE.getU32(&Offset);
    uint64_t FileOffset = DE.getU64(&Offset);
    uint64_t FileSize = DE.getU64(&Offset);
    (void)Pad;

    if (Type == GlobalInfoType::EndOfList)
      return Error::success();

    // Validate that the section fits within the buffer.
    if (FileOffset + FileSize > BufSize)
      return createStringError(std::errc::invalid_argument,
                               "GlobalData section type %u extends beyond "
                               "buffer (offset=%" PRIu64 ", size=%" PRIu64
                               ", bufsize=%" PRIu64 ")",
                               static_cast<uint32_t>(Type), FileOffset,
                               FileSize, BufSize);

    switch (Type) {
    case GlobalInfoType::AddrOffsets:
      AddrOffsetsOff = FileOffset;
      AddrOffsetsSize = FileSize;
      break;
    case GlobalInfoType::AddrInfoOffsets:
      AddrInfoOffsetsOff = FileOffset;
      AddrInfoOffsetsSize = FileSize;
      break;
    case GlobalInfoType::StringTable:
      StringTableOff = FileOffset;
      StringTableSize = FileSize;
      break;
    case GlobalInfoType::FileTable:
      FileTableOff = FileOffset;
      FileTableSize = FileSize;
      break;
    case GlobalInfoType::FunctionInfo:
      FuncInfoOff = FileOffset;
      FuncInfoSize = FileSize;
      break;
    case GlobalInfoType::UUID:
      // UUID is noted but not needed for lookups.
      break;
    case GlobalInfoType::EndOfList:
      llvm_unreachable("handled above");
    }
  }
  return createStringError(std::errc::invalid_argument,
                           "GlobalData array not terminated by EndOfList");
}

llvm::Error
GsymReaderV2::parse() {
  const StringRef Buf = MemBuffer->getBuffer();
  const uint64_t BufSize = Buf.size();

  if (BufSize < 24)
    return createStringError(std::errc::invalid_argument,
                             "not enough data for a GSYM V2 header");

  // Check magic to determine endianness.
  const auto HostByteOrder = llvm::endianness::native;
  uint32_t Magic;
  memcpy(&Magic, Buf.data(), 4);

  switch (Magic) {
  case GSYM_MAGIC:
    Endian = HostByteOrder;
    break;
  case GSYM_CIGAM:
    Endian = sys::IsBigEndianHost ? llvm::endianness::little
                                  : llvm::endianness::big;
    Swap.reset(new SwappedData);
    break;
  default:
    return createStringError(std::errc::invalid_argument, "not a GSYM file");
  }

  const bool IsLittleEndian = (Endian == llvm::endianness::little);

  // Decode the header.
  DataExtractor DE(Buf, IsLittleEndian, 8);
  if (Swap) {
    auto ExpectedHdr = HeaderV2::decode(DE);
    if (!ExpectedHdr)
      return ExpectedHdr.takeError();
    Swap->Hdr = *ExpectedHdr;
    Hdr = &Swap->Hdr;
  } else {
    // Native endianness — cast directly from the mmap'd buffer.
    Hdr = reinterpret_cast<const HeaderV2 *>(Buf.data());
  }

  if (Error Err = Hdr->checkForError())
    return Err;

  // Parse GlobalData entries to find section locations.
  uint64_t Offset = 24; // Fixed header size.
  uint64_t AddrOffsetsOff = 0, AddrOffsetsSize = 0;
  uint64_t AddrInfoOffsetsOff = 0, AddrInfoOffsetsSize = 0;
  uint64_t StringTableOff = 0, StringTableSize = 0;
  uint64_t FileTableOff = 0, FileTableSize = 0;
  uint64_t FuncInfoOff = 0, FuncInfoSize = 0;

  if (auto Err = parseGlobalDataEntries(
          DE, Offset, BufSize, AddrOffsetsOff, AddrOffsetsSize,
          AddrInfoOffsetsOff, AddrInfoOffsetsSize, StringTableOff,
          StringTableSize, FileTableOff, FileTableSize, FuncInfoOff,
          FuncInfoSize))
    return Err;

  // Validate required sections are present.
  if (!AddrOffsetsSize)
    return createStringError(std::errc::invalid_argument,
                             "missing AddrOffsets section");
  if (!AddrInfoOffsetsSize)
    return createStringError(std::errc::invalid_argument,
                             "missing AddrInfoOffsets section");
  if (!StringTableSize)
    return createStringError(std::errc::invalid_argument,
                             "missing StringTable section");
  if (!FileTableSize)
    return createStringError(std::errc::invalid_argument,
                             "missing FileTable section");

  // Validate AddrOffsets size matches header.
  if (AddrOffsetsSize !=
      static_cast<uint64_t>(Hdr->NumAddresses) * Hdr->AddrOffSize)
    return createStringError(std::errc::invalid_argument,
                             "AddrOffsets section size mismatch");

  // Validate AddrInfoOffsets size matches header.
  if (AddrInfoOffsetsSize !=
      static_cast<uint64_t>(Hdr->NumAddresses) * Hdr->AddrInfoOffSize)
    return createStringError(std::errc::invalid_argument,
                             "AddrInfoOffsets section size mismatch");

  if (!Swap) {
    // Native endianness — point ArrayRefs directly into the buffer.
    AddrOffsets = ArrayRef<uint8_t>(
        reinterpret_cast<const uint8_t *>(Buf.data() + AddrOffsetsOff),
        AddrOffsetsSize);

    if (Hdr->AddrInfoOffSize == 4) {
      AddrInfoOffsets = ArrayRef<uint32_t>(
          reinterpret_cast<const uint32_t *>(Buf.data() + AddrInfoOffsetsOff),
          Hdr->NumAddresses);
    } else {
      return createStringError(std::errc::not_supported,
                               "8-byte AddrInfoOffsets not yet supported");
    }

    // FileTable: first 4 bytes is NumFiles, then FileEntry array.
    if (FileTableSize < 4)
      return createStringError(std::errc::invalid_argument,
                               "FileTable section too small");
    uint32_t NumFiles;
    memcpy(&NumFiles, Buf.data() + FileTableOff, 4);
    if (FileTableSize < 4 + NumFiles * sizeof(FileEntry))
      return createStringError(std::errc::invalid_argument,
                               "FileTable section too small for %u files",
                               NumFiles);
    Files = ArrayRef<FileEntry>(
        reinterpret_cast<const FileEntry *>(Buf.data() + FileTableOff + 4),
        NumFiles);

    // String table.
    StrTab.Data = Buf.substr(StringTableOff, StringTableSize);
  } else {
    // Swapped endianness — decode into local storage.

    // AddrOffsets.
    uint64_t AOff = AddrOffsetsOff;
    Swap->AddrOffsets.resize(AddrOffsetsSize);
    switch (Hdr->AddrOffSize) {
    case 1:
      if (!DE.getU8(&AOff, Swap->AddrOffsets.data(), Hdr->NumAddresses))
        return createStringError(std::errc::invalid_argument,
                                 "failed to read address table");
      break;
    case 2:
      if (!DE.getU16(&AOff,
                     reinterpret_cast<uint16_t *>(Swap->AddrOffsets.data()),
                     Hdr->NumAddresses))
        return createStringError(std::errc::invalid_argument,
                                 "failed to read address table");
      break;
    case 4:
      if (!DE.getU32(&AOff,
                     reinterpret_cast<uint32_t *>(Swap->AddrOffsets.data()),
                     Hdr->NumAddresses))
        return createStringError(std::errc::invalid_argument,
                                 "failed to read address table");
      break;
    case 8:
      if (!DE.getU64(&AOff,
                     reinterpret_cast<uint64_t *>(Swap->AddrOffsets.data()),
                     Hdr->NumAddresses))
        return createStringError(std::errc::invalid_argument,
                                 "failed to read address table");
      break;
    }
    AddrOffsets = ArrayRef<uint8_t>(Swap->AddrOffsets);

    // AddrInfoOffsets.
    if (Hdr->AddrInfoOffSize == 4) {
      uint64_t AIOff = AddrInfoOffsetsOff;
      Swap->AddrInfoOffsets.resize(Hdr->NumAddresses);
      if (!DE.getU32(&AIOff, Swap->AddrInfoOffsets.data(), Hdr->NumAddresses))
        return createStringError(std::errc::invalid_argument,
                                 "failed to read address info offsets");
      AddrInfoOffsets = ArrayRef<uint32_t>(Swap->AddrInfoOffsets);
    } else {
      return createStringError(std::errc::not_supported,
                               "8-byte AddrInfoOffsets not yet supported");
    }

    // FileTable.
    uint64_t FTOff = FileTableOff;
    uint32_t NumFiles = DE.getU32(&FTOff);
    if (NumFiles > 0) {
      Swap->Files.resize(NumFiles);
      if (!DE.getU32(&FTOff, &Swap->Files[0].Dir, NumFiles * 2))
        return createStringError(std::errc::invalid_argument,
                                 "failed to read file table");
      Files = ArrayRef<FileEntry>(Swap->Files);
    }

    // String table — raw bytes, no swapping needed.
    StrTab.Data = Buf.substr(StringTableOff, StringTableSize);
  }
  return Error::success();
}

const HeaderV2 &GsymReaderV2::getHeader() const {
  assert(Hdr);
  return *Hdr;
}

std::optional<uint64_t> GsymReaderV2::getAddress(size_t Index) const {
  switch (Hdr->AddrOffSize) {
  case 1: return addressForIndex<uint8_t>(Index);
  case 2: return addressForIndex<uint16_t>(Index);
  case 4: return addressForIndex<uint32_t>(Index);
  case 8: return addressForIndex<uint64_t>(Index);
  }
  return std::nullopt;
}

std::optional<uint64_t> GsymReaderV2::getAddressInfoOffset(size_t Index) const {
  const auto NumAddrInfoOffsets = AddrInfoOffsets.size();
  if (Index < NumAddrInfoOffsets)
    return AddrInfoOffsets[Index];
  return std::nullopt;
}

Expected<uint64_t>
GsymReaderV2::getAddressIndex(const uint64_t Addr) const {
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
GsymReaderV2::getFunctionInfoDataForAddress(uint64_t Addr,
                                          uint64_t &FuncStartAddr) const {
  Expected<uint64_t> ExpectedAddrIdx = getAddressIndex(Addr);
  if (!ExpectedAddrIdx)
    return ExpectedAddrIdx.takeError();
  const uint64_t FirstAddrIdx = *ExpectedAddrIdx;
  std::optional<uint64_t> FirstFuncStartAddr;
  const size_t NumAddresses = getNumAddresses();
  for (uint64_t AddrIdx = FirstAddrIdx; AddrIdx < NumAddresses; ++AddrIdx) {
    auto ExpextedData = getFunctionInfoDataAtIndex(AddrIdx, FuncStartAddr);
    if (!ExpextedData)
      return ExpextedData;

    if (FirstFuncStartAddr.has_value()) {
      if (*FirstFuncStartAddr != FuncStartAddr)
        break;
    } else {
      FirstFuncStartAddr = FuncStartAddr;
    }

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
GsymReaderV2::getFunctionInfoDataAtIndex(uint64_t AddrIdx,
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

llvm::Expected<FunctionInfo> GsymReaderV2::getFunctionInfo(uint64_t Addr) const {
  uint64_t FuncStartAddr = 0;
  if (auto ExpectedData = getFunctionInfoDataForAddress(Addr, FuncStartAddr))
    return FunctionInfo::decode(*ExpectedData, FuncStartAddr);
  else
    return ExpectedData.takeError();
}

llvm::Expected<FunctionInfo>
GsymReaderV2::getFunctionInfoAtIndex(uint64_t Idx) const {
  uint64_t FuncStartAddr = 0;
  if (auto ExpectedData = getFunctionInfoDataAtIndex(Idx, FuncStartAddr))
    return FunctionInfo::decode(*ExpectedData, FuncStartAddr);
  else
    return ExpectedData.takeError();
}

llvm::Expected<LookupResult>
GsymReaderV2::lookup(uint64_t Addr,
                   std::optional<DataExtractor> *MergedFunctionsData) const {
  uint64_t FuncStartAddr = 0;
  if (auto ExpectedData = getFunctionInfoDataForAddress(Addr, FuncStartAddr))
    return FunctionInfo::lookup(*ExpectedData, *this, FuncStartAddr, Addr,
                                MergedFunctionsData);
  else
    return ExpectedData.takeError();
}

llvm::Expected<std::vector<LookupResult>>
GsymReaderV2::lookupAll(uint64_t Addr) const {
  std::vector<LookupResult> Results;
  std::optional<DataExtractor> MergedFunctionsData;

  auto MainResult = lookup(Addr, &MergedFunctionsData);
  if (!MainResult)
    return MainResult.takeError();

  Results.push_back(std::move(*MainResult));

  if (MergedFunctionsData) {
    auto ExpectedMergedFuncExtractors =
        MergedFunctionsInfo::getFuncsDataExtractors(*MergedFunctionsData);
    if (!ExpectedMergedFuncExtractors)
      return ExpectedMergedFuncExtractors.takeError();

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

void GsymReaderV2::dump(raw_ostream &OS) {
  const auto &Header = getHeader();
  OS << Header << "\n";
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
  OS << "\nAddress Info Offsets:\n";
  OS << "INDEX  Offset\n";
  OS << "====== ==========\n";
  for (uint32_t I = 0; I < Header.NumAddresses; ++I)
    OS << format("[%4u] ", I) << HEX32(AddrInfoOffsets[I]) << "\n";
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

void GsymReaderV2::dump(raw_ostream &OS, const FunctionInfo &FI,
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

void GsymReaderV2::dump(raw_ostream &OS, const MergedFunctionsInfo &MFI) {
  for (uint32_t inx = 0; inx < MFI.MergedFunctions.size(); inx++) {
    OS << "++ Merged FunctionInfos[" << inx << "]:\n";
    dump(OS, MFI.MergedFunctions[inx], 4);
  }
}

void GsymReaderV2::dump(raw_ostream &OS, const CallSiteInfo &CSI) {
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

void GsymReaderV2::dump(raw_ostream &OS, const CallSiteInfoCollection &CSIC,
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

void GsymReaderV2::dump(raw_ostream &OS, const LineTable &LT, uint32_t Indent) {
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

void GsymReaderV2::dump(raw_ostream &OS, const InlineInfo &II, uint32_t Indent) {
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

void GsymReaderV2::dump(raw_ostream &OS, std::optional<FileEntry> FE) {
  if (FE) {
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
