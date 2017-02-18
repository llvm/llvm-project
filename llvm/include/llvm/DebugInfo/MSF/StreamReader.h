//===- StreamReader.h - Reads bytes and objects from a stream ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_MSF_STREAMREADER_H
#define LLVM_DEBUGINFO_MSF_STREAMREADER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/MSF/MSFError.h"
#include "llvm/DebugInfo/MSF/StreamArray.h"
#include "llvm/DebugInfo/MSF/StreamInterface.h"
#include "llvm/DebugInfo/MSF/StreamRef.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/type_traits.h"

#include <string>
#include <type_traits>

namespace llvm {
namespace msf {

class StreamReader {
public:
  StreamReader(ReadableStreamRef Stream);

  Error readLongestContiguousChunk(ArrayRef<uint8_t> &Buffer);
  Error readBytes(ArrayRef<uint8_t> &Buffer, uint32_t Size);

  template <typename T>
  Error readInteger(T &Dest,
                    llvm::support::endianness Endian = llvm::support::native) {
    static_assert(std::is_integral<T>::value,
                  "Cannot call readInteger with non-integral value!");

    ArrayRef<uint8_t> Bytes;
    if (auto EC = readBytes(Bytes, sizeof(T)))
      return EC;

    Dest = llvm::support::endian::read<T, llvm::support::unaligned>(
        Bytes.data(), Endian);
    return Error::success();
  }

  Error readZeroString(StringRef &Dest);
  Error readFixedString(StringRef &Dest, uint32_t Length);
  Error readStreamRef(ReadableStreamRef &Ref);
  Error readStreamRef(ReadableStreamRef &Ref, uint32_t Length);

  template <typename T>
  Error readEnum(T &Dest,
                 llvm::support::endianness Endian = llvm::support::native) {
    static_assert(std::is_enum<T>::value,
                  "Cannot call readEnum with non-enum value!");
    typename std::underlying_type<T>::type N;
    if (auto EC = readInteger(N, Endian))
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

    if (NumElements > UINT32_MAX / sizeof(T))
      return make_error<MSFError>(msf_error_code::insufficient_buffer);

    if (auto EC = readBytes(Bytes, NumElements * sizeof(T)))
      return EC;
    Array = ArrayRef<T>(reinterpret_cast<const T *>(Bytes.data()), NumElements);
    return Error::success();
  }

  template <typename T, typename U>
  Error readArray(VarStreamArray<T, U> &Array, uint32_t Size) {
    ReadableStreamRef S;
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
      return make_error<MSFError>(msf_error_code::invalid_format);
    if (Offset + Length > Stream.getLength())
      return make_error<MSFError>(msf_error_code::insufficient_buffer);
    ReadableStreamRef View = Stream.slice(Offset, Length);
    Array = FixedStreamArray<T>(View);
    Offset += Length;
    return Error::success();
  }

  bool empty() const { return bytesRemaining() == 0; }
  void setOffset(uint32_t Off) { Offset = Off; }
  uint32_t getOffset() const { return Offset; }
  uint32_t getLength() const { return Stream.getLength(); }
  uint32_t bytesRemaining() const { return getLength() - getOffset(); }

  Error skip(uint32_t Amount);

  uint8_t peek() const;

private:
  ReadableStreamRef Stream;
  uint32_t Offset;
};
} // namespace msf
} // namespace llvm

#endif // LLVM_DEBUGINFO_MSF_STREAMREADER_H
