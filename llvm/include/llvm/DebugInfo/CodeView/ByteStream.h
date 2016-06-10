//===- ByteStream.h - Reads stream data from a byte sequence ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_BYTESTREAM_H
#define LLVM_DEBUGINFO_CODEVIEW_BYTESTREAM_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/CodeView/StreamInterface.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <memory>
#include <type_traits>

namespace llvm {
namespace codeview {
class StreamReader;

template <bool Writable = false> class ByteStream : public StreamInterface {
  typedef typename std::conditional<Writable, MutableArrayRef<uint8_t>,
                                    ArrayRef<uint8_t>>::type ArrayType;

public:
  ByteStream() {}
  explicit ByteStream(ArrayType Data) : Data(Data) {}
  ~ByteStream() override {}

  Error readBytes(uint32_t Offset, uint32_t Size,
                  ArrayRef<uint8_t> &Buffer) const override;
  Error readLongestContiguousChunk(uint32_t Offset,
                                   ArrayRef<uint8_t> &Buffer) const override;

  Error writeBytes(uint32_t Offset, ArrayRef<uint8_t> Buffer) const override;

  uint32_t getLength() const override;

  ArrayRef<uint8_t> data() const { return Data; }
  StringRef str() const;

private:
  ArrayType Data;
};

extern template class ByteStream<true>;
extern template class ByteStream<false>;

} // end namespace pdb
} // end namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_BYTESTREAM_H
