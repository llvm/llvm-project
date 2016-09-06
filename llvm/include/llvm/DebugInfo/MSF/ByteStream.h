//===- ByteStream.h - Reads stream data from a byte sequence ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_MSF_BYTESTREAM_H
#define LLVM_DEBUGINFO_MSF_BYTESTREAM_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/MSF/MSFError.h"
#include "llvm/DebugInfo/MSF/StreamInterface.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cstdint>
#include <memory>
#include <type_traits>

namespace llvm {
namespace msf {

class ByteStream : public ReadableStream {
public:
  ByteStream() {}
  explicit ByteStream(ArrayRef<uint8_t> Data) : Data(Data) {}

  Error readBytes(uint32_t Offset, uint32_t Size,
                  ArrayRef<uint8_t> &Buffer) const override {
    if (Offset > Data.size())
      return make_error<MSFError>(msf_error_code::insufficient_buffer);
    if (Data.size() < Size + Offset)
      return make_error<MSFError>(msf_error_code::insufficient_buffer);
    Buffer = Data.slice(Offset, Size);
    return Error::success();
  }
  Error readLongestContiguousChunk(uint32_t Offset,
                                   ArrayRef<uint8_t> &Buffer) const override {
    if (Offset >= Data.size())
      return make_error<MSFError>(msf_error_code::insufficient_buffer);
    Buffer = Data.slice(Offset);
    return Error::success();
  }

  uint32_t getLength() const override { return Data.size(); }

  ArrayRef<uint8_t> data() const { return Data; }

  StringRef str() const {
    const char *CharData = reinterpret_cast<const char *>(Data.data());
    return StringRef(CharData, Data.size());
  }

protected:
  ArrayRef<uint8_t> Data;
};

// MemoryBufferByteStream behaves like a read-only ByteStream, but has its data
// backed by an llvm::MemoryBuffer.  It also owns the underlying MemoryBuffer.
class MemoryBufferByteStream : public ByteStream {
public:
  explicit MemoryBufferByteStream(std::unique_ptr<MemoryBuffer> Buffer)
      : ByteStream(ArrayRef<uint8_t>(Buffer->getBuffer().bytes_begin(),
                                     Buffer->getBuffer().bytes_end())),
        MemBuffer(std::move(Buffer)) {}

  std::unique_ptr<MemoryBuffer> MemBuffer;
};

class MutableByteStream : public WritableStream {
public:
  MutableByteStream() {}
  explicit MutableByteStream(MutableArrayRef<uint8_t> Data)
      : Data(Data), ImmutableStream(Data) {}

  Error readBytes(uint32_t Offset, uint32_t Size,
                  ArrayRef<uint8_t> &Buffer) const override {
    return ImmutableStream.readBytes(Offset, Size, Buffer);
  }
  Error readLongestContiguousChunk(uint32_t Offset,
                                   ArrayRef<uint8_t> &Buffer) const override {
    return ImmutableStream.readLongestContiguousChunk(Offset, Buffer);
  }

  uint32_t getLength() const override { return ImmutableStream.getLength(); }

  Error writeBytes(uint32_t Offset, ArrayRef<uint8_t> Buffer) const override {
    if (Data.size() < Buffer.size())
      return make_error<MSFError>(msf_error_code::insufficient_buffer);
    if (Offset > Buffer.size() - Data.size())
      return make_error<MSFError>(msf_error_code::insufficient_buffer);

    uint8_t *DataPtr = const_cast<uint8_t *>(Data.data());
    ::memcpy(DataPtr + Offset, Buffer.data(), Buffer.size());
    return Error::success();
  }

  Error commit() const override { return Error::success(); }

  MutableArrayRef<uint8_t> data() const { return Data; }

private:
  MutableArrayRef<uint8_t> Data;
  ByteStream ImmutableStream;
};

// A simple adapter that acts like a ByteStream but holds ownership over
// and underlying FileOutputBuffer.
class FileBufferByteStream : public WritableStream {
private:
  class StreamImpl : public MutableByteStream {
  public:
    StreamImpl(std::unique_ptr<FileOutputBuffer> Buffer)
        : MutableByteStream(MutableArrayRef<uint8_t>(Buffer->getBufferStart(),
                                                     Buffer->getBufferEnd())),
          FileBuffer(std::move(Buffer)) {}

    Error commit() const override {
      if (FileBuffer->commit())
        return llvm::make_error<MSFError>(msf_error_code::not_writable);
      return Error::success();
    }

  private:
    std::unique_ptr<FileOutputBuffer> FileBuffer;
  };

public:
  explicit FileBufferByteStream(std::unique_ptr<FileOutputBuffer> Buffer)
      : Impl(std::move(Buffer)) {}

  Error readBytes(uint32_t Offset, uint32_t Size,
                  ArrayRef<uint8_t> &Buffer) const override {
    return Impl.readBytes(Offset, Size, Buffer);
  }
  Error readLongestContiguousChunk(uint32_t Offset,
                                   ArrayRef<uint8_t> &Buffer) const override {
    return Impl.readLongestContiguousChunk(Offset, Buffer);
  }

  uint32_t getLength() const override { return Impl.getLength(); }

  Error writeBytes(uint32_t Offset, ArrayRef<uint8_t> Data) const override {
    return Impl.writeBytes(Offset, Data);
  }
  Error commit() const override { return Impl.commit(); }

private:
  StreamImpl Impl;
};


} // end namespace msf
} // end namespace llvm

#endif // LLVM_DEBUGINFO_MSF_BYTESTREAM_H
