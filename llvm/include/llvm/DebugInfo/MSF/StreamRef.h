//===- StreamRef.h - A copyable reference to a stream -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_MSF_STREAMREF_H
#define LLVM_DEBUGINFO_MSF_STREAMREF_H

#include "llvm/DebugInfo/MSF/MSFError.h"
#include "llvm/DebugInfo/MSF/StreamInterface.h"

namespace llvm {
namespace msf {
template <class StreamType, class RefType> class StreamRefBase {
public:
  StreamRefBase() : Stream(nullptr), ViewOffset(0), Length(0) {}
  StreamRefBase(const StreamType &Stream, uint32_t Offset, uint32_t Length)
      : Stream(&Stream), ViewOffset(Offset), Length(Length) {}

  uint32_t getLength() const { return Length; }
  const StreamType *getStream() const { return Stream; }

  RefType drop_front(uint32_t N) const {
    if (!Stream)
      return RefType();

    N = std::min(N, Length);
    return RefType(*Stream, ViewOffset + N, Length - N);
  }

  RefType keep_front(uint32_t N) const {
    if (!Stream)
      return RefType();
    N = std::min(N, Length);
    return RefType(*Stream, ViewOffset, N);
  }

  RefType slice(uint32_t Offset, uint32_t Len) const {
    return drop_front(Offset).keep_front(Len);
  }

  bool operator==(const RefType &Other) const {
    if (Stream != Other.Stream)
      return false;
    if (ViewOffset != Other.ViewOffset)
      return false;
    if (Length != Other.Length)
      return false;
    return true;
  }

protected:
  const StreamType *Stream;
  uint32_t ViewOffset;
  uint32_t Length;
};

class ReadableStreamRef
    : public StreamRefBase<ReadableStream, ReadableStreamRef> {
public:
  ReadableStreamRef() : StreamRefBase() {}
  ReadableStreamRef(const ReadableStream &Stream)
      : StreamRefBase(Stream, 0, Stream.getLength()) {}
  ReadableStreamRef(const ReadableStream &Stream, uint32_t Offset,
                    uint32_t Length)
      : StreamRefBase(Stream, Offset, Length) {}

  // Use StreamRef.slice() instead.
  ReadableStreamRef(const ReadableStreamRef &S, uint32_t Offset,
                    uint32_t Length) = delete;

  Error readBytes(uint32_t Offset, uint32_t Size,
                  ArrayRef<uint8_t> &Buffer) const {
    if (ViewOffset + Offset < Offset)
      return make_error<MSFError>(msf_error_code::insufficient_buffer);
    if (Size + Offset > Length)
      return make_error<MSFError>(msf_error_code::insufficient_buffer);
    return Stream->readBytes(ViewOffset + Offset, Size, Buffer);
  }

  // Given an offset into the stream, read as much as possible without copying
  // any data.
  Error readLongestContiguousChunk(uint32_t Offset,
                                   ArrayRef<uint8_t> &Buffer) const {
    if (Offset >= Length)
      return make_error<MSFError>(msf_error_code::insufficient_buffer);

    if (auto EC = Stream->readLongestContiguousChunk(Offset, Buffer))
      return EC;
    // This StreamRef might refer to a smaller window over a larger stream.  In
    // that case we will have read out more bytes than we should return, because
    // we should not read past the end of the current view.
    uint32_t MaxLength = Length - Offset;
    if (Buffer.size() > MaxLength)
      Buffer = Buffer.slice(0, MaxLength);
    return Error::success();
  }
};

class WritableStreamRef
    : public StreamRefBase<WritableStream, WritableStreamRef> {
public:
  WritableStreamRef() : StreamRefBase() {}
  WritableStreamRef(const WritableStream &Stream)
      : StreamRefBase(Stream, 0, Stream.getLength()) {}
  WritableStreamRef(const WritableStream &Stream, uint32_t Offset,
                    uint32_t Length)
      : StreamRefBase(Stream, Offset, Length) {}

  // Use StreamRef.slice() instead.
  WritableStreamRef(const WritableStreamRef &S, uint32_t Offset,
                    uint32_t Length) = delete;

  Error writeBytes(uint32_t Offset, ArrayRef<uint8_t> Data) const {
    if (Data.size() + Offset > Length)
      return make_error<MSFError>(msf_error_code::insufficient_buffer);
    return Stream->writeBytes(ViewOffset + Offset, Data);
  }

  Error commit() const { return Stream->commit(); }
};

} // namespace msf
} // namespace llvm

#endif // LLVM_DEBUGINFO_MSF_STREAMREF_H
