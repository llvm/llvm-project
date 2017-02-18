//===- StreamWrite.cpp - Writes bytes and objects to a stream -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/MSF/StreamWriter.h"

#include "llvm/DebugInfo/MSF/MSFError.h"
#include "llvm/DebugInfo/MSF/StreamReader.h"
#include "llvm/DebugInfo/MSF/StreamRef.h"

using namespace llvm;
using namespace llvm::msf;

StreamWriter::StreamWriter(WritableStreamRef S) : Stream(S), Offset(0) {}

Error StreamWriter::writeBytes(ArrayRef<uint8_t> Buffer) {
  if (auto EC = Stream.writeBytes(Offset, Buffer))
    return EC;
  Offset += Buffer.size();
  return Error::success();
}

Error StreamWriter::writeZeroString(StringRef Str) {
  if (auto EC = writeFixedString(Str))
    return EC;
  if (auto EC = writeObject('\0'))
    return EC;

  return Error::success();
}

Error StreamWriter::writeFixedString(StringRef Str) {
  ArrayRef<uint8_t> Bytes(Str.bytes_begin(), Str.bytes_end());
  if (auto EC = Stream.writeBytes(Offset, Bytes))
    return EC;

  Offset += Str.size();
  return Error::success();
}

Error StreamWriter::writeStreamRef(ReadableStreamRef Ref) {
  if (auto EC = writeStreamRef(Ref, Ref.getLength()))
    return EC;
  // Don't increment Offset here, it is done by the overloaded call to
  // writeStreamRef.
  return Error::success();
}

Error StreamWriter::writeStreamRef(ReadableStreamRef Ref, uint32_t Length) {
  Ref = Ref.slice(0, Length);

  StreamReader SrcReader(Ref);
  // This is a bit tricky.  If we just call readBytes, we are requiring that it
  // return us the entire stream as a contiguous buffer.  For large streams this
  // will allocate a huge amount of space from the pool.  Instead, iterate over
  // each contiguous chunk until we've consumed the entire stream.
  while (SrcReader.bytesRemaining() > 0) {
    ArrayRef<uint8_t> Chunk;
    if (auto EC = SrcReader.readLongestContiguousChunk(Chunk))
      return EC;
    if (auto EC = writeBytes(Chunk))
      return EC;
  }
  return Error::success();
}
