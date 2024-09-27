//===- Minidump.cpp - Minidump object file implementation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/Minidump.h"
#include "llvm/Object/Error.h"
#include "llvm/Support/ConvertUTF.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::minidump;

std::optional<ArrayRef<uint8_t>>
MinidumpFile::getRawStream(minidump::StreamType Type) const {
  auto It = StreamMap.find(Type);
  if (It != StreamMap.end())
    return getRawStream(Streams[It->second]);
  return std::nullopt;
}

Expected<std::string> MinidumpFile::getString(size_t Offset) const {
  // Minidump strings consist of a 32-bit length field, which gives the size of
  // the string in *bytes*. This is followed by the actual string encoded in
  // UTF16.
  auto ExpectedSize =
      getDataSliceAs<support::ulittle32_t>(getData(), Offset, 1);
  if (!ExpectedSize)
    return ExpectedSize.takeError();
  size_t Size = (*ExpectedSize)[0];
  if (Size % 2 != 0)
    return createError("String size not even");
  Size /= 2;
  if (Size == 0)
    return "";

  Offset += sizeof(support::ulittle32_t);
  auto ExpectedData =
      getDataSliceAs<support::ulittle16_t>(getData(), Offset, Size);
  if (!ExpectedData)
    return ExpectedData.takeError();

  SmallVector<UTF16, 32> WStr(Size);
  copy(*ExpectedData, WStr.begin());

  std::string Result;
  if (!convertUTF16ToUTF8String(WStr, Result))
    return createError("String decoding failed");

  return Result;
}

iterator_range<llvm::object::MinidumpFile::ExceptionStreamsIterator>
MinidumpFile::getExceptionStreams() const {
  return make_range(ExceptionStreamsIterator(ExceptionStreams, this),
                    ExceptionStreamsIterator({}, this));
}

Expected<iterator_range<MinidumpFile::MemoryInfoIterator>>
MinidumpFile::getMemoryInfoList() const {
  std::optional<ArrayRef<uint8_t>> Stream =
      getRawStream(StreamType::MemoryInfoList);
  if (!Stream)
    return createError("No such stream");
  auto ExpectedHeader =
      getDataSliceAs<minidump::MemoryInfoListHeader>(*Stream, 0, 1);
  if (!ExpectedHeader)
    return ExpectedHeader.takeError();
  const minidump::MemoryInfoListHeader &H = ExpectedHeader.get()[0];
  Expected<ArrayRef<uint8_t>> Data =
      getDataSlice(*Stream, H.SizeOfHeader, H.SizeOfEntry * H.NumberOfEntries);
  if (!Data)
    return Data.takeError();
  return make_range(MemoryInfoIterator(*Data, H.SizeOfEntry),
                    MemoryInfoIterator({}, H.SizeOfEntry));
}

template <typename T>
Expected<ArrayRef<T>> MinidumpFile::getListStream(StreamType Type) const {
  std::optional<ArrayRef<uint8_t>> Stream = getRawStream(Type);
  if (!Stream)
    return createError("No such stream");
  auto ExpectedSize = getDataSliceAs<support::ulittle32_t>(*Stream, 0, 1);
  if (!ExpectedSize)
    return ExpectedSize.takeError();

  size_t ListSize = ExpectedSize.get()[0];

  size_t ListOffset = 4;
  // Some producers insert additional padding bytes to align the list to an
  // 8-byte boundary. Check for that by comparing the list size with the overall
  // stream size.
  if (ListOffset + sizeof(T) * ListSize < Stream->size())
    ListOffset = 8;

  return getDataSliceAs<T>(*Stream, ListOffset, ListSize);
}
template Expected<ArrayRef<Module>>
    MinidumpFile::getListStream(StreamType) const;
template Expected<ArrayRef<Thread>>
    MinidumpFile::getListStream(StreamType) const;
template Expected<ArrayRef<MemoryDescriptor>>
    MinidumpFile::getListStream(StreamType) const;

Expected<ArrayRef<uint8_t>> MinidumpFile::getDataSlice(ArrayRef<uint8_t> Data,
                                                       uint64_t Offset,
                                                       uint64_t Size) {
  // Check for overflow.
  if (Offset + Size < Offset || Offset + Size < Size ||
      Offset + Size > Data.size())
    return createEOFError();
  return Data.slice(Offset, Size);
}

Expected<std::unique_ptr<MinidumpFile>>
MinidumpFile::create(MemoryBufferRef Source) {
  ArrayRef<uint8_t> Data = arrayRefFromStringRef(Source.getBuffer());
  auto ExpectedHeader = getDataSliceAs<minidump::Header>(Data, 0, 1);
  if (!ExpectedHeader)
    return ExpectedHeader.takeError();

  const minidump::Header &Hdr = (*ExpectedHeader)[0];
  if (Hdr.Signature != Header::MagicSignature)
    return createError("Invalid signature");
  if ((Hdr.Version & 0xffff) != Header::MagicVersion)
    return createError("Invalid version");

  auto ExpectedStreams = getDataSliceAs<Directory>(Data, Hdr.StreamDirectoryRVA,
                                                   Hdr.NumberOfStreams);
  if (!ExpectedStreams)
    return ExpectedStreams.takeError();

  DenseMap<StreamType, std::size_t> StreamMap;
  std::vector<Directory> ExceptionStreams;
  for (const auto &StreamDescriptor : llvm::enumerate(*ExpectedStreams)) {
    StreamType Type = StreamDescriptor.value().Type;
    const LocationDescriptor &Loc = StreamDescriptor.value().Location;

    Expected<ArrayRef<uint8_t>> Stream =
        getDataSlice(Data, Loc.RVA, Loc.DataSize);
    if (!Stream)
      return Stream.takeError();

    if (Type == StreamType::Unused && Loc.DataSize == 0) {
      // Ignore dummy streams. This is technically ill-formed, but a number of
      // existing minidumps seem to contain such streams.
      continue;
    }

    // Exceptions can be treated as a special case of streams. Other streams
    // represent a list of entities, but exceptions are unique per stream.
    if (Type == StreamType::Exception) {
      ExceptionStreams.push_back(StreamDescriptor.value());
      continue;
    }

    if (Type == DenseMapInfo<StreamType>::getEmptyKey() ||
        Type == DenseMapInfo<StreamType>::getTombstoneKey())
      return createError("Cannot handle one of the minidump streams");

    // Update the directory map, checking for duplicate stream types.
    if (!StreamMap.try_emplace(Type, StreamDescriptor.index()).second)
      return createError("Duplicate stream type");
  }

  return std::unique_ptr<MinidumpFile>(
      new MinidumpFile(Source, Hdr, *ExpectedStreams, std::move(StreamMap),
                       std::move(ExceptionStreams)));
}

iterator_range<MinidumpFile::FallibleMemory64Iterator>
MinidumpFile::getMemory64List(Error &Err) const {
  ErrorAsOutParameter ErrAsOutParam(&Err);
  auto end = FallibleMemory64Iterator::end(Memory64Iterator::end());
  Expected<minidump::Memory64ListHeader> ListHeader = getMemoryList64Header();
  if (!ListHeader) {
    Err = ListHeader.takeError();
    return make_range(end, end);
  }

  std::optional<ArrayRef<uint8_t>> Stream =
      getRawStream(StreamType::Memory64List);
  if (!Stream) {
    Err = createError("No such stream");
    return make_range(end, end);
  }

  Expected<ArrayRef<minidump::MemoryDescriptor_64>> Descriptors =
      getDataSliceAs<minidump::MemoryDescriptor_64>(
          *Stream, sizeof(Memory64ListHeader),
          ListHeader->NumberOfMemoryRanges);

  if (!Descriptors) {
    Err = Descriptors.takeError();
    return make_range(end, end);
  }

  if (!Descriptors->empty() &&
      ListHeader->BaseRVA + Descriptors->front().DataSize > getData().size()) {
    Err = createError("Memory64List header RVA out of range");
    return make_range(end, end);
  }

  return make_range(FallibleMemory64Iterator::itr(
                        Memory64Iterator::begin(
                            getData().slice(ListHeader->BaseRVA), *Descriptors),
                        Err),
                    FallibleMemory64Iterator::end(Memory64Iterator::end()));
}
