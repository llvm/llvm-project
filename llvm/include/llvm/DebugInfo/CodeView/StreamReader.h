//===- StreamReader.h - Reads bytes and objects from a stream ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CODEVIEW_STREAMREADER_H
#define LLVM_DEBUGINFO_CODEVIEW_STREAMREADER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/CodeView/CodeViewError.h"
#include "llvm/DebugInfo/CodeView/StreamArray.h"
#include "llvm/DebugInfo/CodeView/StreamInterface.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"

#include <string>

namespace llvm {
namespace codeview {

class StreamRef;

class StreamReader {
public:
  StreamReader(StreamRef Stream);

  Error readLongestContiguousChunk(ArrayRef<uint8_t> &Buffer);
  Error readBytes(ArrayRef<uint8_t> &Buffer, uint32_t Size);
  Error readInteger(uint16_t &Dest);
  Error readInteger(uint32_t &Dest);
  Error readZeroString(StringRef &Dest);
  Error readFixedString(StringRef &Dest, uint32_t Length);
  Error readStreamRef(StreamRef &Ref);
  Error readStreamRef(StreamRef &Ref, uint32_t Length);

  template <typename T> Error readEnum(T &Dest) {
    typename std::underlying_type<T>::type N;
    if (auto EC = readInteger(N))
      return EC;
    Dest = static_cast<T>(N);
    return Error::success();
  }

  template <typename T> Error readObject(const T *&Dest) {
    ArrayRef<uint8_t> Buffer;
    if (auto EC = readBytes(Buffer, sizeof(T)))
      return EC;
    Dest = reinterpret_cast<const T *>(Buffer.data());
    return Error::success();
  }

  template <typename T>
  Error readArray(ArrayRef<T> &Array, uint32_t NumElements) {
    ArrayRef<uint8_t> Bytes;
    if (NumElements == 0) {
      Array = ArrayRef<T>();
      return Error::success();
    }

    if (NumElements > UINT32_MAX/sizeof(T))
      return make_error<CodeViewError>(cv_error_code::insufficient_buffer);

    if (auto EC = readBytes(Bytes, NumElements * sizeof(T)))
      return EC;
    Array = ArrayRef<T>(reinterpret_cast<const T *>(Bytes.data()), NumElements);
    return Error::success();
  }

  template <typename T, typename U>
  Error readArray(VarStreamArray<T, U> &Array, uint32_t Size) {
    StreamRef S;
    if (auto EC = readStreamRef(S, Size))
      return EC;
    Array = VarStreamArray<T, U>(S, Array.getExtractor());
    return Error::success();
  }

  template <typename T>
  Error readArray(FixedStreamArray<T> &Array, uint32_t NumItems) {
    if (NumItems == 0) {
      Array = FixedStreamArray<T>();
      return Error::success();
    }
    uint32_t Length = NumItems * sizeof(T);
    if (Length / sizeof(T) != NumItems)
      return make_error<CodeViewError>(cv_error_code::corrupt_record);
    if (Offset + Length > Stream.getLength())
      return make_error<CodeViewError>(cv_error_code::insufficient_buffer);
    StreamRef View = Stream.slice(Offset, Length);
    Array = FixedStreamArray<T>(View);
    Offset += Length;
    return Error::success();
  }

  void setOffset(uint32_t Off) { Offset = Off; }
  uint32_t getOffset() const { return Offset; }
  uint32_t getLength() const { return Stream.getLength(); }
  uint32_t bytesRemaining() const { return getLength() - getOffset(); }

private:
  StreamRef Stream;
  uint32_t Offset;
};
} // namespace codeview
} // namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_STREAMREADER_H
