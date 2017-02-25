//===- BinaryItemStream.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_BINARYITEMSTREAM_H
#define LLVM_SUPPORT_BINARYITEMSTREAM_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/MSF/BinaryStream.h"
#include "llvm/Support/Error.h"
#include <cstddef>
#include <cstdint>

namespace llvm {

template <typename T> struct BinaryItemTraits {
  size_t length(const T &Item) = delete;
  ArrayRef<uint8_t> bytes(const T &Item) = delete;
};

/// BinaryItemStream represents a sequence of objects stored in some kind of
/// external container but for which it is useful to view as a stream of
/// contiguous bytes.  An example of this might be if you have a collection of
/// records and you serialize each one into a buffer, and store these serialized
/// records in a container.  The pointers themselves are not laid out
/// contiguously in memory, but we may wish to read from or write to these
/// records as if they were.
template <typename T, typename ItemTraits = BinaryItemTraits<T>>
class BinaryItemStream : public BinaryStream {
public:
  explicit BinaryItemStream(llvm::support::endianness Endian)
      : Endian(Endian) {}

  llvm::support::endianness getEndian() const override { return Endian; }

  Error readBytes(uint32_t Offset, uint32_t Size,
                  ArrayRef<uint8_t> &Buffer) override {
    if (auto EC = readLongestContiguousChunk(Offset, Buffer))
      return EC;

    if (Size > Buffer.size())
      return errorCodeToError(make_error_code(std::errc::no_buffer_space));

    Buffer = Buffer.take_front(Size);
    return Error::success();
  }

  Error readLongestContiguousChunk(uint32_t Offset,
                                   ArrayRef<uint8_t> &Buffer) override {
    uint32_t Index;
    uint32_t ByteOffset;
    if (auto EC = translateOffsetIndex(Offset, Index, ByteOffset))
      return EC;
    const auto &Item = Items[Index];
    Buffer = Traits.bytes(Item).drop_front(ByteOffset);
    return Error::success();
  }

  void setItems(ArrayRef<T> ItemArray) { Items = ItemArray; }

  uint32_t getLength() override {
    uint32_t Size = 0;
    for (const auto &Item : Items)
      Size += Traits.length(Item);
    return Size;
  }

private:
  Error translateOffsetIndex(uint32_t Offset, uint32_t &ItemIndex,
                             uint32_t &ByteOffset) {
    ItemIndex = 0;
    ByteOffset = 0;
    uint32_t PrevOffset = 0;
    uint32_t CurrentOffset = 0;
    if (Offset > 0) {
      for (const auto &Item : Items) {
        PrevOffset = CurrentOffset;
        CurrentOffset += Traits.length(Item);
        if (CurrentOffset > Offset)
          break;
        ++ItemIndex;
      }
    }
    if (CurrentOffset < Offset)
      return errorCodeToError(make_error_code(std::errc::no_buffer_space));
    ByteOffset = Offset - PrevOffset;
    return Error::success();
  }

  llvm::support::endianness Endian;
  ItemTraits Traits;
  ArrayRef<T> Items;
};

} // end namespace llvm

#endif // LLVM_SUPPORT_BINARYITEMSTREAM_H
