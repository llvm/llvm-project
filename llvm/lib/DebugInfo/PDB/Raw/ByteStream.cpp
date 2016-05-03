//===- ByteStream.cpp - Reads stream data from a byte sequence ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/PDB/Raw/ByteStream.h"
#include "llvm/DebugInfo/PDB/Raw/StreamReader.h"

using namespace llvm;
using namespace llvm::pdb;

ByteStream::ByteStream() : Owned(false) {}

ByteStream::ByteStream(MutableArrayRef<uint8_t> Bytes) : Owned(false) {
  initialize(Bytes);
}

ByteStream::ByteStream(uint32_t Length) : Owned(false) { initialize(Length); }

ByteStream::~ByteStream() { reset(); }

void ByteStream::reset() {
  if (Owned)
    delete[] Data.data();
  Owned = false;
  Data = MutableArrayRef<uint8_t>();
}

void ByteStream::initialize(MutableArrayRef<uint8_t> Bytes) {
  reset();
  Data = Bytes;
  Owned = false;
}

void ByteStream::initialize(uint32_t Length) {
  reset();
  Data = MutableArrayRef<uint8_t>(new uint8_t[Length], Length);
  Owned = true;
}

std::error_code ByteStream::initialize(StreamReader &Reader, uint32_t Length) {
  initialize(Length);
  std::error_code EC = Reader.readBytes(Data);
  if (EC)
    reset();
  return EC;
}

std::error_code ByteStream::readBytes(uint32_t Offset,
                                      MutableArrayRef<uint8_t> Buffer) const {
  if (Data.size() < Buffer.size() + Offset)
    return std::make_error_code(std::errc::bad_address);
  ::memcpy(Buffer.data(), Data.data() + Offset, Buffer.size());
  return std::error_code();
}

std::error_code ByteStream::getArrayRef(uint32_t Offset,
                                        ArrayRef<uint8_t> &Buffer,
                                        uint32_t Length) const {
  if (Data.size() < Length + Offset)
    return std::make_error_code(std::errc::bad_address);
  Buffer = Data.slice(Offset, Length);
  return std::error_code();
}

uint32_t ByteStream::getLength() const { return Data.size(); }

StringRef ByteStream::str() const {
  const char *CharData = reinterpret_cast<const char *>(Data.data());
  return StringRef(CharData, Data.size());
}
