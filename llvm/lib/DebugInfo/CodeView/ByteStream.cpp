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

ByteStream::ByteStream() {}

ByteStream::ByteStream(MutableArrayRef<uint8_t> Data) : Data(Data) {}

ByteStream::~ByteStream() {}

void ByteStream::reset() {
  Ownership.reset();
  Data = MutableArrayRef<uint8_t>();
}

void ByteStream::load(uint32_t Length) {
  reset();
  if (Length > 0)
    Data = MutableArrayRef<uint8_t>(new uint8_t[Length], Length);
  Ownership.reset(Data.data());
}

Error ByteStream::load(StreamReader &Reader, uint32_t Length) {
  load(Length);
  auto EC = Reader.readBytes(Data);
  if (EC)
    reset();
  return EC;
}

Error ByteStream::readBytes(uint32_t Offset,
                            MutableArrayRef<uint8_t> Buffer) const {
  if (Data.size() < Buffer.size() + Offset)
    return make_error<CodeViewError>(cv_error_code::insufficient_buffer);
  ::memcpy(Buffer.data() + Offset, Data.data(), Buffer.size());
  return Error::success();
}

Error ByteStream::readBytes(uint32_t Offset, uint32_t Size,
                            ArrayRef<uint8_t> &Buffer) const {
  if (Data.size() < Buffer.size() + Offset)
    return make_error<CodeViewError>(cv_error_code::insufficient_buffer);
  Buffer = Data.slice(Offset, Size);
  return Error::success();
}

uint32_t ByteStream::getLength() const { return Data.size(); }

StringRef ByteStream::str() const {
  const char *CharData = reinterpret_cast<const char *>(Data.data());
  return StringRef(CharData, Data.size());
}
