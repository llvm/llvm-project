//===- BinaryItemStream.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_MSF_BINARYITEMSTREAM_H
#define LLVM_DEBUGINFO_MSF_BINARYITEMSTREAM_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/MSF/BinaryStream.h"
#include "llvm/DebugInfo/MSF/MSFError.h"
#include "llvm/Support/Error.h"
#include <cstddef>
#include <cstdint>

namespace llvm {

template <typename T> struct BinaryItemTraits {
  static size_t length(const T &Item) = delete;
  static ArrayRef<uint8_t> bytes(const T &Item) = delete;
};

/// BinaryItemStream represents a sequence of objects stored in some kind of
/// external container but for which it is useful to view as a stream of
/// contiguous bytes.  An example of this might be if you have a collection of
/// records and you serialize each one into a buffer, and store these serialized
/// records in a container.  The pointers themselves are not laid out
/// contiguously in memory, but we may wish to read from or write to these
/// records as if they were.
template <typename T, typename Traits = BinaryItemTraits<T>>
class BinaryItemStream : public BinaryStream {
public:
  explicit BinaryItemStream(llvm::support::endianness Endian)
      : Endian(Endian) {}

  llvm::support::endianness getEndian() const override { return Endian; }

  Error readBytes(uint32_t Offset, uint32_t Size,
                  ArrayRef<uint8_t> &Buffer) override {
    auto ExpectedIndex = translateOffsetIndex(Offset);
    if (!ExpectedIndex)
      return ExpectedIndex.takeError();
    const auto &Item = Items[*ExpectedIndex];
    if (Size > Traits::length(Item))
      return make_error<msf::MSFError>(
          msf::msf_error_code::insufficient_buffer);
    Buffer = Traits::bytes(Item).take_front(Size);
    return Error::success();
  }

  Error readLongestContiguousChunk(uint32_t Offset,
                                   ArrayRef<uint8_t> &Buffer) override {
    auto ExpectedIndex = translateOffsetIndex(Offset);
    if (!ExpectedIndex)
      return ExpectedIndex.takeError();
    Buffer = Traits::bytes(Items[*ExpectedIndex]);
    return Error::success();
  }

  void setItems(ArrayRef<T> ItemArray) { Items = ItemArray; }

  uint32_t getLength() override {
    uint32_t Size = 0;
    for (const auto &Item : Items)
      Size += Traits::length(Item);
    return Size;
  }

private:
  Expected<uint32_t> translateOffsetIndex(uint32_t Offset) const {
    uint32_t CurrentOffset = 0;
    uint32_t CurrentIndex = 0;
    for (const auto &Item : Items) {
      if (CurrentOffset >= Offset)
        break;
      CurrentOffset += Traits::length(Item);
      ++CurrentIndex;
    }
    if (CurrentOffset != Offset)
      return make_error<msf::MSFError>(
          msf::msf_error_code::insufficient_buffer);
    return CurrentIndex;
  }

  llvm::support::endianness Endian;
  ArrayRef<T> Items;
};

} // end namespace llvm

#endif // LLVM_DEBUGINFO_MSF_BINARYITEMSTREAM_H
