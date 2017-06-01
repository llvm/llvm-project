//===- StreamInterface.h - Base interface for a stream of data --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_MSF_STREAMINTERFACE_H
#define LLVM_DEBUGINFO_MSF_STREAMINTERFACE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include <cstdint>

namespace llvm {
namespace msf {

class ReadableStream {
public:
  virtual ~ReadableStream() = default;

  // Given an offset into the stream and a number of bytes, attempt to read
  // the bytes and set the output ArrayRef to point to a reference into the
  // stream, without copying any data.
  virtual Error readBytes(uint32_t Offset, uint32_t Size,
                          ArrayRef<uint8_t> &Buffer) const = 0;

  // Given an offset into the stream, read as much as possible without copying
  // any data.
  virtual Error readLongestContiguousChunk(uint32_t Offset,
                                           ArrayRef<uint8_t> &Buffer) const = 0;

  virtual uint32_t getLength() const = 0;
};

class WritableStream : public ReadableStream {
public:
  ~WritableStream() override = default;

  // Attempt to write the given bytes into the stream at the desired offset.
  // This will always necessitate a copy.  Cannot shrink or grow the stream,
  // only writes into existing allocated space.
  virtual Error writeBytes(uint32_t Offset, ArrayRef<uint8_t> Data) const = 0;

  virtual Error commit() const = 0;
};

} // end namespace msf
} // end namespace llvm

#endif // LLVM_DEBUGINFO_MSF_STREAMINTERFACE_H
