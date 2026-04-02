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

#include "llvm/DebugInfo/GSYM/GsymReaderV1.h"
#include "llvm/DebugInfo/GSYM/GsymReaderV2.h"
#include "llvm/DebugInfo/GSYM/Header.h"
#include "llvm/DebugInfo/GSYM/HeaderV2.h"
#include "llvm/DebugInfo/GSYM/InlineInfo.h"
#include "llvm/DebugInfo/GSYM/LineTable.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace gsym;

GsymReader::GsymReader(std::unique_ptr<MemoryBuffer> Buffer)
    : MemBuffer(std::move(Buffer)), Endian(llvm::endianness::native) {}

GsymReader::GsymReader(GsymReader &&RHS) = default;

/// Check magic bytes and return the GSYM version from raw bytes.
/// If magic bytes are invalid, return error.
static Expected<uint16_t> checkMagicAndDetectVersion(StringRef Data) {
  if (Data.size() < 6)
    return createStringError(std::errc::invalid_argument,
                             "data too small to be a GSYM file");
  const bool IsHostLittleEndian =
      llvm::endianness::native == llvm::endianness::little;
  DataExtractor DE(Data, IsHostLittleEndian, 4);
  uint64_t Offset = 0;
  uint32_t Magic = DE.getU32(&Offset);
  bool NeedSwap;
  if (Magic == GSYM_MAGIC)
    NeedSwap = false;
  else if (Magic == GSYM_CIGAM)
    NeedSwap = true;
  else
    return createStringError(std::errc::invalid_argument,
                             "not a GSYM file (bad magic)");
  if (NeedSwap) {
    // Re-create DataExtractor with swapped endianness to read version.
    DE = DataExtractor(Data, !IsHostLittleEndian, 4);
    Offset = 4;
  }
  return DE.getU16(&Offset);
}

llvm::Expected<std::unique_ptr<GsymReader>>
GsymReader::openFile(StringRef Path) {
  auto BufOrErr = MemoryBuffer::getFileOrSTDIN(Path);
  if (!BufOrErr)
    return createStringError(BufOrErr.getError(), "failed to open '%s'",
                             Path.str().c_str());
  auto VersionOrErr = checkMagicAndDetectVersion((*BufOrErr)->getBuffer());
  if (!VersionOrErr)
    return VersionOrErr.takeError();
  switch (*VersionOrErr) {
  case Header::getVersion(): {
    auto R = GsymReaderV1::openFile(Path);
    if (!R)
      return R.takeError();
    return std::make_unique<GsymReaderV1>(std::move(*R));
  }
  case HeaderV2::getVersion(): {
    auto R = GsymReaderV2::openFile(Path);
    if (!R)
      return R.takeError();
    return std::make_unique<GsymReaderV2>(std::move(*R));
  }
  default:
    return createStringError(std::errc::invalid_argument,
                             "unsupported GSYM version %u", *VersionOrErr);
  }
}

llvm::Expected<std::unique_ptr<GsymReader>>
GsymReader::copyBuffer(StringRef Bytes) {
  auto VersionOrErr = checkMagicAndDetectVersion(Bytes);
  if (!VersionOrErr)
    return VersionOrErr.takeError();
  switch (*VersionOrErr) {
  case Header::getVersion(): {
    auto R = GsymReaderV1::copyBuffer(Bytes);
    if (!R)
      return R.takeError();
    return std::make_unique<GsymReaderV1>(std::move(*R));
  }
  case HeaderV2::getVersion(): {
    auto R = GsymReaderV2::copyBuffer(Bytes);
    if (!R)
      return R.takeError();
    return std::make_unique<GsymReaderV2>(std::move(*R));
  }
  default:
    return createStringError(std::errc::invalid_argument,
                             "unsupported GSYM version %u", *VersionOrErr);
  }
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
  const uint64_t AddrInfoOffset = getAddressInfoOffset(AddrIdx);
  assert((Endian == endianness::big || Endian == endianness::little) &&
         "Endian must be either big or little");
  StringRef Bytes = MemBuffer->getBuffer().substr(AddrInfoOffset);
  if (Bytes.empty())
    return createStringError(std::errc::invalid_argument,
                             "invalid address info offset 0x%" PRIx64,
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
  if (auto ExpectedData = getFunctionInfoDataForAddress(Addr, FuncStartAddr)) {
    ExpectedData->setStringOffsetSize(getStringOffsetByteSize());
    return FunctionInfo::decode(*ExpectedData, FuncStartAddr);
  } else
    return ExpectedData.takeError();
}

llvm::Expected<FunctionInfo>
GsymReader::getFunctionInfoAtIndex(uint64_t Idx) const {
  uint64_t FuncStartAddr = 0;
  if (auto ExpectedData = getFunctionInfoDataAtIndex(Idx, FuncStartAddr)) {
    ExpectedData->setStringOffsetSize(getStringOffsetByteSize());
    return FunctionInfo::decode(*ExpectedData, FuncStartAddr);
  } else
    return ExpectedData.takeError();
}

llvm::Expected<LookupResult>
GsymReader::lookup(uint64_t Addr,
                   std::optional<DataExtractor> *MergedFunctionsData) const {
  uint64_t FuncStartAddr = 0;
  if (auto ExpectedData = getFunctionInfoDataForAddress(Addr, FuncStartAddr)) {
    ExpectedData->setStringOffsetSize(getStringOffsetByteSize());
    return FunctionInfo::lookup(*ExpectedData, *this, FuncStartAddr, Addr,
                                MergedFunctionsData);
  } else
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
      MergedData.setStringOffsetSize(getStringOffsetByteSize());
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
