//===- ByteStream.cpp - Reads stream data from a byte sequence ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/CodeView/ByteStream.h"
#include "llvm/DebugInfo/CodeView/CodeViewError.h"
#include "llvm/DebugInfo/CodeView/StreamReader.h"
#include <cstring>

using namespace llvm;
using namespace llvm::codeview;

static Error writeBytes(uint32_t Offset, ArrayRef<uint8_t> Src,
                        ArrayRef<uint8_t> Dest) {
  return make_error<CodeViewError>(cv_error_code::operation_unsupported,
                                   "ByteStream is immutable.");
}

static Error writeBytes(uint32_t Offset, ArrayRef<uint8_t> Src,
                        MutableArrayRef<uint8_t> Dest) {
  if (Dest.size() < Src.size())
    return make_error<CodeViewError>(cv_error_code::insufficient_buffer);
  if (Offset > Src.size() - Dest.size())
    return make_error<CodeViewError>(cv_error_code::insufficient_buffer);

  ::memcpy(Dest.data() + Offset, Src.data(), Src.size());
  return Error::success();
}

template <bool Writable>
Error ByteStream<Writable>::readBytes(uint32_t Offset, uint32_t Size,
                                      ArrayRef<uint8_t> &Buffer) const {
  if (Offset > Data.size())
    return make_error<CodeViewError>(cv_error_code::insufficient_buffer);
  if (Data.size() < Size + Offset)
    return make_error<CodeViewError>(cv_error_code::insufficient_buffer);
  Buffer = Data.slice(Offset, Size);
  return Error::success();
}

template <bool Writable>
Error ByteStream<Writable>::readLongestContiguousChunk(
    uint32_t Offset, ArrayRef<uint8_t> &Buffer) const {
  if (Offset >= Data.size())
    return make_error<CodeViewError>(cv_error_code::insufficient_buffer);
  Buffer = Data.slice(Offset);
  return Error::success();
}

template <bool Writable>
Error ByteStream<Writable>::writeBytes(uint32_t Offset,
                                       ArrayRef<uint8_t> Buffer) const {
  return ::writeBytes(Offset, Buffer, Data);
}

template <bool Writable> uint32_t ByteStream<Writable>::getLength() const {
  return Data.size();
}

template <bool Writable> StringRef ByteStream<Writable>::str() const {
  const char *CharData = reinterpret_cast<const char *>(Data.data());
  return StringRef(CharData, Data.size());
}

namespace llvm {
namespace codeview {
template class ByteStream<true>;
template class ByteStream<false>;
}
}
