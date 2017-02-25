//===- BinaryStreamReader.cpp - Reads objects from a binary stream --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/MSF/BinaryStreamReader.h"

#include "llvm/DebugInfo/MSF/BinaryStreamRef.h"

using namespace llvm;

BinaryStreamReader::BinaryStreamReader(BinaryStreamRef S)
    : Stream(S), Offset(0) {}

Error BinaryStreamReader::readLongestContiguousChunk(
    ArrayRef<uint8_t> &Buffer) {
  if (auto EC = Stream.readLongestContiguousChunk(Offset, Buffer))
    return EC;
  Offset += Buffer.size();
  return Error::success();
}

Error BinaryStreamReader::readBytes(ArrayRef<uint8_t> &Buffer, uint32_t Size) {
  if (auto EC = Stream.readBytes(Offset, Size, Buffer))
    return EC;
  Offset += Size;
  return Error::success();
}

Error BinaryStreamReader::readInteger(uint64_t &Dest, uint32_t ByteSize) {
  assert(ByteSize == 1 || ByteSize == 2 || ByteSize == 4 || ByteSize == 8);
  ArrayRef<uint8_t> Bytes;

  if (auto EC = readBytes(Bytes, ByteSize))
    return EC;
  switch (ByteSize) {
  case 1:
    Dest = Bytes[0];
    return Error::success();
  case 2:
    Dest = llvm::support::endian::read16(Bytes.data(), Stream.getEndian());
    return Error::success();
  case 4:
    Dest = llvm::support::endian::read32(Bytes.data(), Stream.getEndian());
    return Error::success();
  case 8:
    Dest = llvm::support::endian::read64(Bytes.data(), Stream.getEndian());
    return Error::success();
  }
  llvm_unreachable("Unreachable!");
  return Error::success();
}

Error BinaryStreamReader::readCString(StringRef &Dest) {
  // TODO: This could be made more efficient by using readLongestContiguousChunk
  // and searching for null terminators in the resulting buffer.

  uint32_t Length = 0;
  // First compute the length of the string by reading 1 byte at a time.
  uint32_t OriginalOffset = getOffset();
  const char *C;
  while (true) {
    if (auto EC = readObject(C))
      return EC;
    if (*C == '\0')
      break;
    ++Length;
  }
  // Now go back and request a reference for that many bytes.
  uint32_t NewOffset = getOffset();
  setOffset(OriginalOffset);

  if (auto EC = readFixedString(Dest, Length))
    return EC;

  // Now set the offset back to where it was after we calculated the length.
  setOffset(NewOffset);
  return Error::success();
}

Error BinaryStreamReader::readFixedString(StringRef &Dest, uint32_t Length) {
  ArrayRef<uint8_t> Bytes;
  if (auto EC = readBytes(Bytes, Length))
    return EC;
  Dest = StringRef(reinterpret_cast<const char *>(Bytes.begin()), Bytes.size());
  return Error::success();
}

Error BinaryStreamReader::readStreamRef(BinaryStreamRef &Ref) {
  return readStreamRef(Ref, bytesRemaining());
}

Error BinaryStreamReader::readStreamRef(BinaryStreamRef &Ref, uint32_t Length) {
  if (bytesRemaining() < Length)
    return errorCodeToError(make_error_code(std::errc::no_buffer_space));
  Ref = Stream.slice(Offset, Length);
  Offset += Length;
  return Error::success();
}

Error BinaryStreamReader::skip(uint32_t Amount) {
  if (Amount > bytesRemaining())
    return errorCodeToError(make_error_code(std::errc::no_buffer_space));
  Offset += Amount;
  return Error::success();
}

uint8_t BinaryStreamReader::peek() const {
  ArrayRef<uint8_t> Buffer;
  auto EC = Stream.readBytes(Offset, 1, Buffer);
  assert(!EC && "Cannot peek an empty buffer!");
  llvm::consumeError(std::move(EC));
  return Buffer[0];
}
