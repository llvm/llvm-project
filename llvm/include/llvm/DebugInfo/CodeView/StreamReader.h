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
  StreamReader(const StreamInterface &S);

  Error readBytes(uint32_t Size, ArrayRef<uint8_t> &Buffer);
  Error readInteger(uint16_t &Dest);
  Error readInteger(uint32_t &Dest);
  Error readZeroString(StringRef &Dest);
  Error readFixedString(StringRef &Dest, uint32_t Length);
  Error readStreamRef(StreamRef &Ref);
  Error readStreamRef(StreamRef &Ref, uint32_t Length);
  Error readBytes(MutableArrayRef<uint8_t> Buffer);

  template <typename T> Error readObject(const T *&Dest) {
    ArrayRef<uint8_t> Buffer;
    if (auto EC = readBytes(sizeof(T), Buffer))
      return EC;
    Dest = reinterpret_cast<const T *>(Buffer.data());
    return Error::success();
  }

  template <typename T>
  Error readArray(FixedStreamArray<T> &Array, uint32_t NumItems) {
    if (NumItems == 0) {
      Array = FixedStreamArray<T>();
      return Error::success();
    }
    uint32_t Length = NumItems * sizeof(T);
    if (Offset + Length > Stream.getLength())
      return make_error<CodeViewError>(cv_error_code::insufficient_buffer);
    StreamRef View(Stream, Offset, Length);
    Array = FixedStreamArray<T>(View);
    Offset += Length;
    return Error::success();
  }

  void setOffset(uint32_t Off) { Offset = Off; }
  uint32_t getOffset() const { return Offset; }
  uint32_t getLength() const { return Stream.getLength(); }
  uint32_t bytesRemaining() const { return getLength() - getOffset(); }

private:
  const StreamInterface &Stream;
  uint32_t Offset;
};
} // namespace codeview
} // namespace llvm

#endif // LLVM_DEBUGINFO_CODEVIEW_STREAMREADER_H
