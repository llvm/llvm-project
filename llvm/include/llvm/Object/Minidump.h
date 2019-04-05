//===- Minidump.h - Minidump object file implementation ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_MINIDUMP_H
#define LLVM_OBJECT_MINIDUMP_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/Minidump.h"
#include "llvm/Object/Binary.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace object {

/// A class providing access to the contents of a minidump file.
class MinidumpFile : public Binary {
public:
  /// Construct a new MinidumpFile object from the given memory buffer. Returns
  /// an error if this file cannot be identified as a minidump file, or if its
  /// contents are badly corrupted (i.e. we cannot read the stream directory).
  static Expected<std::unique_ptr<MinidumpFile>> create(MemoryBufferRef Source);

  static bool classof(const Binary *B) { return B->isMinidump(); }

  /// Returns the contents of the minidump header.
  const minidump::Header &header() const { return Header; }

  /// Returns the list of streams (stream directory entries) in this file.
  ArrayRef<minidump::Directory> streams() const { return Streams; }

  /// Returns the raw contents of the stream given by the directory entry.
  ArrayRef<uint8_t> getRawStream(const minidump::Directory &Stream) const {
    return getData().slice(Stream.Location.RVA, Stream.Location.DataSize);
  }

  /// Returns the raw contents of the stream of the given type, or None if the
  /// file does not contain a stream of this type.
  Optional<ArrayRef<uint8_t>> getRawStream(minidump::StreamType Type) const;

  /// Returns the minidump string at the given offset. An error is returned if
  /// we fail to parse the string, or the string is invalid UTF16.
  Expected<std::string> getString(size_t Offset) const;

  /// Returns the contents of the SystemInfo stream, cast to the appropriate
  /// type. An error is returned if the file does not contain this stream, or
  /// the stream is smaller than the size of the SystemInfo structure. The
  /// internal consistency of the stream is not checked in any way.
  Expected<const minidump::SystemInfo &> getSystemInfo() const {
    return getStream<minidump::SystemInfo>(minidump::StreamType::SystemInfo);
  }

private:
  static Error createError(StringRef Str,
                           object_error Err = object_error::parse_failed) {
    return make_error<GenericBinaryError>(Str, Err);
  }

  static Error createEOFError() {
    return createError("Unexpected EOF", object_error::unexpected_eof);
  }

  /// Return a slice of the given data array, with bounds checking.
  static Expected<ArrayRef<uint8_t>> getDataSlice(ArrayRef<uint8_t> Data,
                                                  size_t Offset, size_t Size);

  /// Return the slice of the given data array as an array of objects of the
  /// given type. The function checks that the input array is large enough to
  /// contain the correct number of objects of the given type.
  template <typename T>
  static Expected<ArrayRef<T>> getDataSliceAs(ArrayRef<uint8_t> Data,
                                              size_t Offset, size_t Count);

  MinidumpFile(MemoryBufferRef Source, const minidump::Header &Header,
               ArrayRef<minidump::Directory> Streams,
               DenseMap<minidump::StreamType, std::size_t> StreamMap)
      : Binary(ID_Minidump, Source), Header(Header), Streams(Streams),
        StreamMap(std::move(StreamMap)) {}

  ArrayRef<uint8_t> getData() const {
    return arrayRefFromStringRef(Data.getBuffer());
  }

  /// Return the stream of the given type, cast to the appropriate type. Checks
  /// that the stream is large enough to hold an object of this type.
  template <typename T>
  Expected<const T &> getStream(minidump::StreamType Stream) const;

  const minidump::Header &Header;
  ArrayRef<minidump::Directory> Streams;
  DenseMap<minidump::StreamType, std::size_t> StreamMap;
};

template <typename T>
Expected<const T &> MinidumpFile::getStream(minidump::StreamType Stream) const {
  if (auto OptionalStream = getRawStream(Stream)) {
    if (OptionalStream->size() >= sizeof(T))
      return *reinterpret_cast<const T *>(OptionalStream->data());
    return createError("Malformed stream", object_error::unexpected_eof);
  }
  return createError("No such stream", object_error::invalid_section_index);
}

template <typename T>
Expected<ArrayRef<T>> MinidumpFile::getDataSliceAs(ArrayRef<uint8_t> Data,
                                                   size_t Offset,
                                                   size_t Count) {
  // Check for overflow.
  if (Count > std::numeric_limits<size_t>::max() / sizeof(T))
    return createEOFError();
  auto ExpectedArray = getDataSlice(Data, Offset, sizeof(T) * Count);
  if (!ExpectedArray)
    return ExpectedArray.takeError();
  return ArrayRef<T>(reinterpret_cast<const T *>(ExpectedArray->data()), Count);
}

} // end namespace object
} // end namespace llvm

#endif // LLVM_OBJECT_MINIDUMP_H
