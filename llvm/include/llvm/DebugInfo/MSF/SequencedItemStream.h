//===- SequencedItemStream.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_MSF_SEQUENCEDITEMSTREAM_H
#define LLVM_DEBUGINFO_MSF_SEQUENCEDITEMSTREAM_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/MSF/MSFError.h"
#include "llvm/DebugInfo/MSF/StreamInterface.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <memory>
#include <type_traits>

namespace llvm {
namespace msf {
template <typename T> struct SequencedItemTraits {
  static size_t length(const T &Item) = delete;
  static ArrayRef<uint8_t> bytes(const T &Item) = delete;
};

/// SequencedItemStream represents a sequence of objects stored in a
/// standard container but for which it is useful to view as a stream of
/// contiguous bytes.  An example of this might be if you have a std::vector
/// of TPI records, where each record contains a byte sequence that
/// represents that one record serialized, but where each consecutive item
/// might not be allocated immediately after the previous item.  Using a
/// SequencedItemStream, we can adapt the VarStreamArray class to trivially
/// extract one item at a time, allowing the data to be used anywhere a
/// VarStreamArray could be used.
template <typename T, typename Traits = SequencedItemTraits<T>>
class SequencedItemStream : public ReadableStream {
public:
  SequencedItemStream() {}

  Error readBytes(uint32_t Offset, uint32_t Size,
                  ArrayRef<uint8_t> &Buffer) const override {
    auto ExpectedIndex = translateOffsetIndex(Offset);
    if (!ExpectedIndex)
      return ExpectedIndex.takeError();
    const auto &Item = Items[*ExpectedIndex];
    if (Size > Traits::length(Item))
      return make_error<MSFError>(msf_error_code::insufficient_buffer);
    Buffer = Traits::bytes(Item).take_front(Size);
    return Error::success();
  }

  Error readLongestContiguousChunk(uint32_t Offset,
                                   ArrayRef<uint8_t> &Buffer) const override {
    auto ExpectedIndex = translateOffsetIndex(Offset);
    if (!ExpectedIndex)
      return ExpectedIndex.takeError();
    Buffer = Traits::bytes(Items[*ExpectedIndex]);
    return Error::success();
  }

  void setItems(ArrayRef<T> ItemArray) { Items = ItemArray; }

  uint32_t getLength() const override {
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
      return make_error<MSFError>(msf_error_code::insufficient_buffer);
    return CurrentIndex;
  }
  ArrayRef<T> Items;
};
} // end namespace msf
} // end namespace llvm

#endif // LLVM_DEBUGINFO_MSF_SEQUENCEDITEMSTREAM_H
