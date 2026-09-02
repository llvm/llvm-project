//===- llvm/MC/DXContainerInfo.cpp - DXContainer Info -----*- C++ -------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/DXContainerInfo.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/Config/config.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SwapByteOrder.h"
#include "llvm/Support/VCSRevision.h"
#include <type_traits>

using namespace llvm;
using namespace llvm::mcdxbc;

static uint64_t align(uint64_t Size) {
  return alignTo(Size, dxbc::DXCONTAINER_STRUCT_ALIGNMENT);
}

template <typename StructT>
static void writeStruct(raw_ostream &OS, StructT S) {
  static_assert(std::is_class<StructT>() &&
                "This method must be used for writing structure types");
  if (sys::IsBigEndianHost)
    S.swapBytes();
  OS.write(reinterpret_cast<const char *>(&S), sizeof(StructT));
}

static void writeString(raw_ostream &OS, StringRef S) {
  OS.write(S.data(), S.size());
  // Write null terminator.
  OS.write_zeros(1);
}

static void writePadding(raw_ostream &OS, uint64_t Prev) {
  uint64_t UnpaddedSize = OS.tell() - Prev;
  uint64_t Padding = align(UnpaddedSize) - UnpaddedSize;
  if (Padding)
    OS.write_zeros(Padding);
}

void DebugName::setFilename(StringRef DebugFilename) {
  Parameters.NameLength = DebugFilename.size();
  Filename = DebugFilename;
}

void DebugName::write(raw_ostream &OS) const {
  writeStruct(OS, Parameters);
  writeString(OS, Filename.substr(0, Parameters.NameLength));
}

CompilerVersion::CompilerVersion() {
  Parameters.Major = LLVM_VERSION_MAJOR;
  Parameters.Minor = LLVM_VERSION_MINOR;
  Parameters.Flags = dxbc::CompilerVersionFlags::Default;
#ifndef NDEBUG
  Parameters.Flags |= dxbc::CompilerVersionFlags::Debug;
#endif
  Parameters.CommitCount = 0;
  Parameters.ContentSizeInBytes = 0;
#ifdef LLVM_REVISION
  CommitSha = LLVM_REVISION;
#else
  CommitSha = "";
#endif
  CustomVersionString = PACKAGE_VERSION;
  updateContentSize();
}

void CompilerVersion::setCommitSha(StringRef CommitSha) {
  this->CommitSha = CommitSha;
  updateContentSize();
}

void CompilerVersion::setVersionString(StringRef VersionString) {
  this->CustomVersionString = VersionString;
  updateContentSize();
}

void CompilerVersion::updateContentSize() {
  this->Parameters.ContentSizeInBytes =
      CommitSha.size() + 1 + CustomVersionString.size() + 1;
}

void CompilerVersion::write(raw_ostream &OS) const {
  writeStruct(OS, Parameters);
  SmallString<64> Content;
  raw_svector_ostream ContentStream(Content);
  writeString(ContentStream, CommitSha);
  writeString(ContentStream, CustomVersionString);
  Content.resize(Parameters.ContentSizeInBytes);
  OS.write(Content.data(), Parameters.ContentSizeInBytes);
}

void SourceInfo::Section::computeGenericHeader(
    uint32_t ContentSize, dxbc::SourceInfo::SectionType SecType) {
  GenericHeader.AlignedSizeInBytes = align(sizeof(GenericHeader) + ContentSize);
  GenericHeader.Flags = 0;
  GenericHeader.Type = SecType;
}

void SourceInfo::SourceContents::Entry::compute() {
  Parameters.Flags = 0;
  Parameters.ContentSizeInBytes = FileContent.size() + 1;
  Parameters.AlignedSizeInBytes =
      align(Parameters.ContentSizeInBytes +
            sizeof(dxbc::SourceInfo::Contents::Entry));
}

void SourceInfo::SourceContents::computeUncompressed(
    dxbc::SourceInfo::Contents::CompressionType CompType) {
  size_t ContentEntriesSize = 0;
  for (const SourceContents::Entry &ContentEntry : Entries)
    ContentEntriesSize += ContentEntry.Parameters.AlignedSizeInBytes;

  Parameters.Flags = 0;
  Parameters.Type = CompType;
  Parameters.Count = Entries.size();

  Parameters.UncompressedEntriesSizeInBytes = align(ContentEntriesSize);
  computeFinalSize(Parameters.UncompressedEntriesSizeInBytes);
}

void SourceInfo::SourceContents::computeFinalSize(uint32_t CompressedSize) {
  Parameters.EntriesSizeInBytes = CompressedSize;
  Parameters.AlignedSizeInBytes =
      align(Parameters.EntriesSizeInBytes +
            sizeof(dxbc::SourceInfo::Contents::Header));
  computeGenericHeader(Parameters.AlignedSizeInBytes,
                       dxbc::SourceInfo::SectionType::SourceContents);
}

void SourceInfo::SourceNames::Entry::compute(uint32_t ContentSize) {
  Parameters.Flags = 0;
  Parameters.NameSizeInBytes = FileName.size() + 1;
  Parameters.ContentSizeInBytes = ContentSize;
  Parameters.AlignedSizeInBytes = align(Parameters.NameSizeInBytes +
                                        sizeof(dxbc::SourceInfo::Names::Entry));
}

SourceInfo::SourceNames::Header::Header(
    const dxbc::SourceInfo::Names::HeaderOnDisk &H) {
  const auto *HPtr = reinterpret_cast<const uint8_t *>(&H);
  Flags = support::endian::read32le(HPtr);
  Count = support::endian::read32le(HPtr + 4);
  EntriesSizeInBytes = support::endian::read16le(HPtr + 8);
}

void SourceInfo::SourceNames::compute() {
  size_t NameEntriesSize = 0;
  for (const SourceNames::Entry &NameEntry : Entries)
    NameEntriesSize += NameEntry.Parameters.AlignedSizeInBytes;

  Parameters.Flags = 0;
  Parameters.Count = Entries.size();
  Parameters.EntriesSizeInBytes = align(NameEntriesSize);

  computeGenericHeader(Parameters.EntriesSizeInBytes +
                           sizeof(dxbc::SourceInfo::Names::HeaderOnDisk),
                       dxbc::SourceInfo::SectionType::SourceNames);
}

void SourceInfo::ProgramArgs::compute() {
  size_t ArgEntriesSize = 0;
  for (auto [ArgName, ArgVal] : Args) {
    // Null-terminated argument name and empty null-terminated argument value.
    ArgEntriesSize += ArgName.size() + 1 + ArgVal.size() + 1;
  }

  Parameters.Flags = 0;
  Parameters.SizeInBytes = ArgEntriesSize;
  Parameters.Count = Args.size();

  computeGenericHeader(Parameters.SizeInBytes +
                           sizeof(dxbc::SourceInfo::Args::Header),
                       dxbc::SourceInfo::SectionType::Args);
}

void SourceInfo::compute() {
  Parameters.Flags = 0;
  Parameters.SectionCount = 3;
  Parameters.AlignedSizeInBytes =
      align(sizeof(dxbc::SourceInfo::Header) +
            Names.GenericHeader.AlignedSizeInBytes +
            Contents.GenericHeader.AlignedSizeInBytes +
            Args.GenericHeader.AlignedSizeInBytes);
}

template <typename VecT> static void clearAndReserve(VecT &Vec, size_t N) {
  Vec.clear();
  Vec.reserve(N);
}

void SourceInfoBuilder::computeEntries() {
  IsFilled = true;
  clearAndReserve(BaseData.Names.Entries, FileNamesAndContents.size());
  clearAndReserve(BaseData.Contents.Entries, FileNamesAndContents.size());
  for (const auto &NameContent : FileNamesAndContents) {
    SourceInfo::SourceContents::Entry ContentEntry;
    ContentEntry.FileContent = NameContent.second.str();
    ContentEntry.compute();
    BaseData.Contents.Entries.emplace_back(std::move(ContentEntry));

    SourceInfo::SourceNames::Entry NameEntry;
    NameEntry.FileName = NameContent.first;
    NameEntry.compute(ContentEntry.Parameters.ContentSizeInBytes);
    BaseData.Names.Entries.emplace_back(std::move(NameEntry));
  }

  clearAndReserve(BaseData.Args.Args, Args.size());
  for (auto [ArgName, ArgValue] : Args)
    BaseData.Args.Args.emplace_back(ArgName, ArgValue);
}

void SourceInfoBuilder::recomputeAfterCompression(uint32_t CompressedSize) {
  BaseData.Contents.computeFinalSize(CompressedSize);
  BaseData.compute();
}

void SourceInfoBuilder::finalize() {
  assert(IsFilled && "SourceInfo::computeEntries() must be called before "
                     "SourceInfo::computeUncompressed()");
  assert(CompressionType && "Compression type must be set.");

  IsFinalized = true;

  BaseData.Contents.computeUncompressed(*CompressionType);
  BaseData.Names.compute();
  BaseData.Args.compute();
  BaseData.compute();

  // Compress Contents right here, to calculate compressed size.
  CompressedContents.clear();
  SmallString<256> Data;
  {
    raw_svector_ostream OS(Data);
    for (auto &E : BaseData.Contents.Entries) {
      uint64_t EntryOffset = OS.tell();
      writeStruct(OS, E.Parameters);
      writeString(OS, E.FileContent);
      writePadding(OS, EntryOffset);
    }
    writePadding(OS, 0);
  }
  uint32_t CompressedSize = 0;
  switch (BaseData.Contents.Parameters.Type) {
  case dxbc::SourceInfo::Contents::CompressionType::Zlib: {
    if (!compression::zlib::isAvailable())
      reportFatalUsageError(Twine("DXContainer SRCI Contents should be "
                                  "compressed with Zlib, but ") +
                            Twine(compression::getReasonIfUnsupported(
                                compression::Format::Zlib)));

    SmallVector<uint8_t, 128> CompressedData;
    compression::zlib::compress(
        ArrayRef(reinterpret_cast<uint8_t *>(Data.data()), Data.size()),
        CompressedData, compression::zlib::BestSizeCompression);
    raw_svector_ostream OS(CompressedContents);
    OS.write(reinterpret_cast<char *>(CompressedData.data()),
             CompressedData.size());
    CompressedSize = CompressedContents.size();
    break;
  }
  case dxbc::SourceInfo::Contents::CompressionType::None: {
    CompressedContents = std::move(Data);
    CompressedSize = align(CompressedContents.size());
    break;
  }
  }
  recomputeAfterCompression(CompressedSize);
}

void SourceInfoBuilder::write(raw_ostream &OS) const {
  assert(IsFinalized &&
         "SourceInfo::finalize() must be called before SourceInfo::write()");

  writeStruct(OS, BaseData.Parameters);

  // Write Names section.
  auto &Names = BaseData.Names;
  uint64_t NamesOffset = OS.tell();
  writeStruct(OS, Names.GenericHeader);
  support::endian::write(OS, Names.Parameters.Flags, endianness::little);
  support::endian::write(OS, Names.Parameters.Count, endianness::little);
  support::endian::write(OS, Names.Parameters.EntriesSizeInBytes,
                         endianness::little);
  for (auto &E : Names.Entries) {
    uint64_t EntryOffset = OS.tell();
    writeStruct(OS, E.Parameters);
    writeString(OS, E.FileName);
    writePadding(OS, EntryOffset);
  }
  writePadding(OS, NamesOffset);

  // Write Contents section.
  auto &Contents = BaseData.Contents;
  uint64_t ContentsOffset = OS.tell();
  writeStruct(OS, Contents.GenericHeader);
  writeStruct(OS, Contents.Parameters);

  if (BaseData.Contents.Parameters.EntriesSizeInBytes !=
      CompressedContents.size())
    reportFatalUsageError(formatv(
        "DXContainer SRCI Contents compressed size in header ({0} bytes) "
        "doesn't match the actual compressed size ({1} bytes)",
        BaseData.Contents.Parameters.EntriesSizeInBytes,
        CompressedContents.size()));

  OS.write(CompressedContents.data(), CompressedContents.size());
  writePadding(OS, ContentsOffset);

  // Write Args section.
  auto &Args = BaseData.Args;
  uint64_t ArgsOffset = OS.tell();
  writeStruct(OS, Args.GenericHeader);
  writeStruct(OS, Args.Parameters);
  for (auto &E : Args.Args) {
    writeString(OS, E.first);
    writeString(OS, E.second);
  }
  writePadding(OS, ArgsOffset);
}
