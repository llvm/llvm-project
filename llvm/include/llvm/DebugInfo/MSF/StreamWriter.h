//===- StreamWriter.h - Writes bytes and objects to a stream ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_MSF_STREAMWRITER_H
#define LLVM_DEBUGINFO_MSF_STREAMWRITER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/DebugInfo/MSF/MSFError.h"
#include "llvm/DebugInfo/MSF/StreamArray.h"
#include "llvm/DebugInfo/MSF/StreamInterface.h"
#include "llvm/DebugInfo/MSF/StreamRef.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"

#include <string>

namespace llvm {
namespace msf {

class StreamWriter {
public:
  StreamWriter(WritableStreamRef Stream);

  Error writeBytes(ArrayRef<uint8_t> Buffer);
  Error writeInteger(uint16_t Dest);
  Error writeInteger(uint32_t Dest);
  Error writeZeroString(StringRef Str);
  Error writeFixedString(StringRef Str);
  Error writeStreamRef(ReadableStreamRef Ref);
  Error writeStreamRef(ReadableStreamRef Ref, uint32_t Size);

  template <typename T> Error writeEnum(T Num) {
    return writeInteger(
        static_cast<typename std::underlying_type<T>::type>(Num));
  }

  template <typename T> Error writeObject(const T &Obj) {
    static_assert(!std::is_pointer<T>::value,
                  "writeObject should not be used with pointers, to write "
                  "the pointed-to value dereference the pointer before calling "
                  "writeObject");
    return writeBytes(
        ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(&Obj), sizeof(T)));
  }

  template <typename T> Error writeArray(ArrayRef<T> Array) {
    if (Array.size() == 0)
      return Error::success();

    if (Array.size() > UINT32_MAX / sizeof(T))
      return make_error<MSFError>(msf_error_code::insufficient_buffer);

    return writeBytes(
        ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(Array.data()),
                          Array.size() * sizeof(T)));
  }

  template <typename T, typename U>
  Error writeArray(VarStreamArray<T, U> Array) {
    return writeStreamRef(Array.getUnderlyingStream());
  }

  template <typename T> Error writeArray(FixedStreamArray<T> Array) {
    return writeStreamRef(Array.getUnderlyingStream());
  }

  void setOffset(uint32_t Off) { Offset = Off; }
  uint32_t getOffset() const { return Offset; }
  uint32_t getLength() const { return Stream.getLength(); }
  uint32_t bytesRemaining() const { return getLength() - getOffset(); }

private:
  WritableStreamRef Stream;
  uint32_t Offset;
};
} // namespace msf
} // namespace llvm

#endif // LLVM_DEBUGINFO_MSF_STREAMWRITER_H
