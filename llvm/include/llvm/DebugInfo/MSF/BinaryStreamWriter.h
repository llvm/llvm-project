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
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/MSF/BinaryStreamArray.h"
#include "llvm/DebugInfo/MSF/BinaryStreamRef.h"
#include "llvm/DebugInfo/MSF/MSFError.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <type_traits>

namespace llvm {
namespace msf {

class StreamWriter {
public:
  StreamWriter() = default;
  explicit StreamWriter(WritableStreamRef Stream);

  Error writeBytes(ArrayRef<uint8_t> Buffer);

  template <typename T>
  Error writeInteger(T Value,
                     llvm::support::endianness Endian = llvm::support::native) {
    static_assert(std::is_integral<T>::value,
                  "Cannot call writeInteger with non-integral value!");
    uint8_t Buffer[sizeof(T)];
    llvm::support::endian::write<T, llvm::support::unaligned>(Buffer, Value,
                                                              Endian);
    return writeBytes(Buffer);
  }

  Error writeZeroString(StringRef Str);
  Error writeFixedString(StringRef Str);
  Error writeStreamRef(ReadableStreamRef Ref);
  Error writeStreamRef(ReadableStreamRef Ref, uint32_t Size);

  template <typename T>
  Error writeEnum(T Num,
                  llvm::support::endianness Endian = llvm::support::native) {
    static_assert(std::is_enum<T>::value,
                  "Cannot call writeEnum with non-Enum type");

    using U = typename std::underlying_type<T>::type;
    return writeInteger<U>(static_cast<U>(Num), Endian);
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
    if (Array.empty())
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
  uint32_t Offset = 0;
};

} // end namespace msf
} // end namespace llvm

#endif // LLVM_DEBUGINFO_MSF_STREAMWRITER_H
