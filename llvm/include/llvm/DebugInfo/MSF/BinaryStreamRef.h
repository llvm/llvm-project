//===- BinaryStreamRef.h - A copyable reference to a stream -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_MSF_BINARYSTREAMREF_H
#define LLVM_DEBUGINFO_MSF_BINARYSTREAMREF_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/MSF/BinaryStream.h"
#include "llvm/DebugInfo/MSF/BinaryStreamError.h"
#include "llvm/Support/Error.h"
#include <algorithm>
#include <cstdint>

namespace llvm {

/// Common stuff for mutable and immutable StreamRefs.
template <class StreamType, class RefType> class BinaryStreamRefBase {
public:
  BinaryStreamRefBase() : Stream(nullptr), ViewOffset(0), Length(0) {}
  BinaryStreamRefBase(StreamType &Stream, uint32_t Offset, uint32_t Length)
      : Stream(&Stream), ViewOffset(Offset), Length(Length) {}

  llvm::support::endianness getEndian() const { return Stream->getEndian(); }

  uint32_t getLength() const { return Length; }
  const StreamType *getStream() const { return Stream; }

  /// Return a new BinaryStreamRef with the first \p N elements removed.
  RefType drop_front(uint32_t N) const {
    if (!Stream)
      return RefType();

    N = std::min(N, Length);
    return RefType(*Stream, ViewOffset + N, Length - N);
  }

  /// Return a new BinaryStreamRef with only the first \p N elements remaining.
  RefType keep_front(uint32_t N) const {
    if (!Stream)
      return RefType();
    N = std::min(N, Length);
    return RefType(*Stream, ViewOffset, N);
  }

  /// Return a new BinaryStreamRef with the first \p Offset elements removed,
  /// and retaining exactly \p Len elements.
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
  Error checkOffset(uint32_t Offset, uint32_t DataSize) const {
    if (Offset > getLength())
      return make_error<BinaryStreamError>(stream_error_code::invalid_offset);
    if (getLength() < DataSize + Offset)
      return make_error<BinaryStreamError>(stream_error_code::stream_too_short);
    return Error::success();
  }

  StreamType *Stream;
  uint32_t ViewOffset;
  uint32_t Length;
};

/// \brief BinaryStreamRef is to BinaryStream what ArrayRef is to an Array.  It
/// provides copy-semantics and read only access to a "window" of the underlying
/// BinaryStream. Note that BinaryStreamRef is *not* a BinaryStream.  That is to
/// say, it does not inherit and override the methods of BinaryStream.  In
/// general, you should not pass around pointers or references to BinaryStreams
/// and use inheritance to achieve polymorphism.  Instead, you should pass
/// around BinaryStreamRefs by value and achieve polymorphism that way.
class BinaryStreamRef
    : public BinaryStreamRefBase<BinaryStream, BinaryStreamRef> {
public:
  BinaryStreamRef() = default;
  BinaryStreamRef(BinaryStream &Stream)
      : BinaryStreamRefBase(Stream, 0, Stream.getLength()) {}
  BinaryStreamRef(BinaryStream &Stream, uint32_t Offset, uint32_t Length)
      : BinaryStreamRefBase(Stream, Offset, Length) {}

  // Use BinaryStreamRef.slice() instead.
  BinaryStreamRef(BinaryStreamRef &S, uint32_t Offset,
                  uint32_t Length) = delete;

  /// Given an Offset into this StreamRef and a Size, return a reference to a
  /// buffer owned by the stream.
  ///
  /// \returns a success error code if the entire range of data is within the
  /// bounds of this BinaryStreamRef's view and the implementation could read
  /// the data, and an appropriate error code otherwise.
  Error readBytes(uint32_t Offset, uint32_t Size,
                  ArrayRef<uint8_t> &Buffer) const {
    if (auto EC = checkOffset(Offset, Size))
      return EC;

    return Stream->readBytes(ViewOffset + Offset, Size, Buffer);
  }

  /// Given an Offset into this BinaryStreamRef, return a reference to the
  /// largest buffer the stream could support without necessitating a copy.
  ///
  /// \returns a success error code if implementation could read the data,
  /// and an appropriate error code otherwise.
  Error readLongestContiguousChunk(uint32_t Offset,
                                   ArrayRef<uint8_t> &Buffer) const {
    if (auto EC = checkOffset(Offset, 1))
      return EC;

    if (auto EC =
            Stream->readLongestContiguousChunk(ViewOffset + Offset, Buffer))
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

class WritableBinaryStreamRef
    : public BinaryStreamRefBase<WritableBinaryStream,
                                 WritableBinaryStreamRef> {
public:
  WritableBinaryStreamRef() = default;
  WritableBinaryStreamRef(WritableBinaryStream &Stream)
      : BinaryStreamRefBase(Stream, 0, Stream.getLength()) {}
  WritableBinaryStreamRef(WritableBinaryStream &Stream, uint32_t Offset,
                          uint32_t Length)
      : BinaryStreamRefBase(Stream, Offset, Length) {}

  // Use WritableBinaryStreamRef.slice() instead.
  WritableBinaryStreamRef(WritableBinaryStreamRef &S, uint32_t Offset,
                          uint32_t Length) = delete;

  /// Given an Offset into this WritableBinaryStreamRef and some input data,
  /// writes the data to the underlying stream.
  ///
  /// \returns a success error code if the data could fit within the underlying
  /// stream at the specified location and the implementation could write the
  /// data, and an appropriate error code otherwise.
  Error writeBytes(uint32_t Offset, ArrayRef<uint8_t> Data) const {
    if (auto EC = checkOffset(Offset, Data.size()))
      return EC;

    return Stream->writeBytes(ViewOffset + Offset, Data);
  }

  operator BinaryStreamRef() { return BinaryStreamRef(*Stream); }

  /// \brief For buffered streams, commits changes to the backing store.
  Error commit() { return Stream->commit(); }
};

} // end namespace llvm

#endif // LLVM_DEBUGINFO_MSF_BINARYSTREAMREF_H
