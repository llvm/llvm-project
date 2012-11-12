//===- StreamableMemoryObject.cpp - Streamable data interface -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/StreamableMemoryObject.h"
#include "llvm/Support/Compiler.h"
#include <cassert>
#include <cstring>


using namespace llvm;

namespace {

class RawMemoryObject : public StreamableMemoryObject {
public:
  RawMemoryObject(const unsigned char *Start, const unsigned char *End) :
    FirstChar(Start), LastChar(End) {
    assert(LastChar >= FirstChar && "Invalid start/end range");
  }

  virtual uint64_t getBase() const LLVM_OVERRIDE { return 0; }
  virtual uint64_t getExtent() const LLVM_OVERRIDE {
    return LastChar - FirstChar;
  }
  virtual int readByte(uint64_t address, uint8_t* ptr) const LLVM_OVERRIDE;
  virtual int readBytes(uint64_t address,
                        uint64_t size,
                        uint8_t* buf,
                        uint64_t* copied) const LLVM_OVERRIDE;
  virtual const uint8_t *getPointer(uint64_t address,
                                    uint64_t size) const LLVM_OVERRIDE;
  virtual bool isValidAddress(uint64_t address) const LLVM_OVERRIDE {
    return validAddress(address);
  }
  virtual bool isObjectEnd(uint64_t address) const LLVM_OVERRIDE {
    return objectEnd(address);
  }

private:
  const uint8_t* const FirstChar;
  const uint8_t* const LastChar;

  // These are implemented as inline functions here to avoid multiple virtual
  // calls per public function
  bool validAddress(uint64_t address) const {
    return static_cast<ptrdiff_t>(address) < LastChar - FirstChar;
  }
  bool objectEnd(uint64_t address) const {
    return static_cast<ptrdiff_t>(address) == LastChar - FirstChar;
  }

  RawMemoryObject(const RawMemoryObject&) LLVM_DELETED_FUNCTION;
  void operator=(const RawMemoryObject&) LLVM_DELETED_FUNCTION;
};

int RawMemoryObject::readByte(uint64_t address, uint8_t* ptr) const {
  if (!validAddress(address)) return -1;
  *ptr = *((uint8_t *)(uintptr_t)(address + FirstChar));
  return 0;
}

int RawMemoryObject::readBytes(uint64_t address,
                               uint64_t size,
                               uint8_t* buf,
                               uint64_t* copied) const {
  if (!validAddress(address) || !validAddress(address + size - 1)) return -1;
  memcpy(buf, (uint8_t *)(uintptr_t)(address + FirstChar), size);
  if (copied) *copied = size;
  return size;
}

const uint8_t *RawMemoryObject::getPointer(uint64_t address,
                                           uint64_t size) const {
  return FirstChar + address;
}
} // anonymous namespace

namespace llvm {
// If the bitcode has a header, then its size is known, and we don't have to
// block until we actually want to read it.
bool StreamingMemoryObject::isValidAddress(uint64_t address) const {
  if (ObjectSize && address < ObjectSize) return true;
    return fetchToPos(address);
}

bool StreamingMemoryObject::isObjectEnd(uint64_t address) const {
  if (ObjectSize) return address == ObjectSize;
  fetchToPos(address);
  return address == ObjectSize && address != 0;
}

uint64_t StreamingMemoryObject::getExtent() const {
  if (ObjectSize) return ObjectSize;
  size_t pos = BytesRead + kChunkSize;
  // keep fetching until we run out of bytes
  while (fetchToPos(pos)) pos += kChunkSize;
  return ObjectSize;
}

int StreamingMemoryObject::readByte(uint64_t address, uint8_t* ptr) const {
  if (!fetchToPos(address)) return -1;
  *ptr = Bytes[address + BytesSkipped];
  return 0;
}

int StreamingMemoryObject::readBytes(uint64_t address,
                                     uint64_t size,
                                     uint8_t* buf,
                                     uint64_t* copied) const {
  if (!fetchToPos(address + size - 1)) return -1;
  memcpy(buf, &Bytes[address + BytesSkipped], size);
  if (copied) *copied = size;
  return 0;
}

bool StreamingMemoryObject::dropLeadingBytes(size_t s) {
  if (BytesRead < s) return true;
  BytesSkipped = s;
  BytesRead -= s;
  return false;
}

void StreamingMemoryObject::setKnownObjectSize(size_t size) {
  ObjectSize = size;
  Bytes.reserve(size);
}

StreamableMemoryObject *getNonStreamedMemoryObject(
    const unsigned char *Start, const unsigned char *End) {
  return new RawMemoryObject(Start, End);
}

StreamableMemoryObject::~StreamableMemoryObject() { }

StreamingMemoryObject::StreamingMemoryObject(DataStreamer *streamer) :
  Bytes(kChunkSize), Streamer(streamer), BytesRead(0), BytesSkipped(0),
  ObjectSize(0), EOFReached(false) {
  BytesRead = streamer->GetBytes(&Bytes[0], kChunkSize);
}
}
