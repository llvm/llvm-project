//===- StreamableMemoryObject.h - Streamable data interface -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#ifndef STREAMABLEMEMORYOBJECT_H_
#define STREAMABLEMEMORYOBJECT_H_

#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MemoryObject.h"
#include "llvm/Support/DataStream.h"
#include <vector>

namespace llvm {

/// StreamableMemoryObject - Interface to data which might be streamed.
/// Streamability has 2 important implications/restrictions. First, the data
/// might not yet exist in memory when the request is made. This just means
/// that readByte/readBytes might have to block or do some work to get it.
/// More significantly, the exact size of the object might not be known until
/// it has all been fetched. This means that to return the right result,
/// getExtent must also wait for all the data to arrive; therefore it should
/// not be called on objects which are actually streamed (this would defeat
/// the purpose of streaming). Instead, isValidAddress and isObjectEnd can be
/// used to test addresses without knowing the exact size of the stream.
/// Finally, getPointer can be used instead of readBytes to avoid extra copying.
class StreamableMemoryObject : public MemoryObject {
 public:
  /// Destructor      - Override as necessary.
  virtual ~StreamableMemoryObject();

  /// getBase         - Returns the lowest valid address in the region.
  ///
  /// @result         - The lowest valid address.
  virtual uint64_t getBase() const = 0;

  /// getExtent       - Returns the size of the region in bytes.  (The region is
  ///                   contiguous, so the highest valid address of the region
  ///                   is getBase() + getExtent() - 1).
  ///                   May block until all bytes in the stream have been read
  ///
  /// @result         - The size of the region.
  virtual uint64_t getExtent() const = 0;

  /// readByte        - Tries to read a single byte from the region.
  ///                   May block until (address - base) bytes have been read
  /// @param address  - The address of the byte, in the same space as getBase().
  /// @param ptr      - A pointer to a byte to be filled in.  Must be non-NULL.
  /// @result         - 0 if successful; -1 if not.  Failure may be due to a
  ///                   bounds violation or an implementation-specific error.
  virtual int readByte(uint64_t address, uint8_t* ptr) const = 0;

  /// readBytes       - Tries to read a contiguous range of bytes from the
  ///                   region, up to the end of the region.
  ///                   May block until (address - base + size) bytes have
  ///                   been read. Additionally, StreamableMemoryObjects will
  ///                   not do partial reads - if size bytes cannot be read,
  ///                   readBytes will fail.
  ///
  /// @param address  - The address of the first byte, in the same space as
  ///                   getBase().
  /// @param size     - The maximum number of bytes to copy.
  /// @param buf      - A pointer to a buffer to be filled in.  Must be non-NULL
  ///                   and large enough to hold size bytes.
  /// @param copied   - A pointer to a nunber that is filled in with the number
  ///                   of bytes actually read.  May be NULL.
  /// @result         - 0 if successful; -1 if not.  Failure may be due to a
  ///                   bounds violation or an implementation-specific error.
  virtual int readBytes(uint64_t address,
                        uint64_t size,
                        uint8_t* buf,
                        uint64_t* copied) const = 0;

  /// getPointer  - Ensures that the requested data is in memory, and returns
  ///               A pointer to it. More efficient than using readBytes if the
  ///               data is already in memory.
  ///               May block until (address - base + size) bytes have been read
  /// @param address - address of the byte, in the same space as getBase()
  /// @param size    - amount of data that must be available on return
  /// @result        - valid pointer to the requested data
  virtual const uint8_t *getPointer(uint64_t address, uint64_t size) const = 0;

  /// isValidAddress - Returns true if the address is within the object
  ///                  (i.e. between base and base + extent - 1 inclusive)
  ///                  May block until (address - base) bytes have been read
  /// @param address - address of the byte, in the same space as getBase()
  /// @result        - true if the address may be read with readByte()
  virtual bool isValidAddress(uint64_t address) const = 0;

  /// isObjectEnd    - Returns true if the address is one past the end of the
  ///                  object (i.e. if it is equal to base + extent)
  ///                  May block until (address - base) bytes have been read
  /// @param address - address of the byte, in the same space as getBase()
  /// @result        - true if the address is equal to base + extent
  virtual bool isObjectEnd(uint64_t address) const = 0;
};

/// StreamingMemoryObject - interface to data which is actually streamed from
/// a DataStreamer. In addition to inherited members, it has the
/// dropLeadingBytes and setKnownObjectSize methods which are not applicable
/// to non-streamed objects.
class StreamingMemoryObject : public StreamableMemoryObject {
public:
  StreamingMemoryObject(DataStreamer *streamer);
  virtual uint64_t getBase() const LLVM_OVERRIDE { return 0; }
  virtual uint64_t getExtent() const LLVM_OVERRIDE;
  virtual int readByte(uint64_t address, uint8_t* ptr) const LLVM_OVERRIDE;
  virtual int readBytes(uint64_t address,
                        uint64_t size,
                        uint8_t* buf,
                        uint64_t* copied) const LLVM_OVERRIDE;
  virtual const uint8_t *getPointer(uint64_t address,
                                    uint64_t size) const LLVM_OVERRIDE {
    // This could be fixed by ensuring the bytes are fetched and making a copy,
    // requiring that the bitcode size be known, or otherwise ensuring that
    // the memory doesn't go away/get reallocated, but it's
    // not currently necessary. Users that need the pointer don't stream.
    assert(0 && "getPointer in streaming memory objects not allowed");
    return NULL;
  }
  virtual bool isValidAddress(uint64_t address) const LLVM_OVERRIDE;
  virtual bool isObjectEnd(uint64_t address) const LLVM_OVERRIDE;

  /// Drop s bytes from the front of the stream, pushing the positions of the
  /// remaining bytes down by s. This is used to skip past the bitcode header,
  /// since we don't know a priori if it's present, and we can't put bytes
  /// back into the stream once we've read them.
  bool dropLeadingBytes(size_t s);

  /// If the data object size is known in advance, many of the operations can
  /// be made more efficient, so this method should be called before reading
  /// starts (although it can be called anytime).
  void setKnownObjectSize(size_t size);

private:
  const static uint32_t kChunkSize = 4096 * 4;
  mutable std::vector<unsigned char> Bytes;
  OwningPtr<DataStreamer> Streamer;
  mutable size_t BytesRead;   // Bytes read from stream
  size_t BytesSkipped;// Bytes skipped at start of stream (e.g. wrapper/header)
  mutable size_t ObjectSize; // 0 if unknown, set if wrapper seen or EOF reached
  mutable bool EOFReached;

  // Fetch enough bytes such that Pos can be read or EOF is reached
  // (i.e. BytesRead > Pos). Return true if Pos can be read.
  // Unlike most of the functions in BitcodeReader, returns true on success.
  // Most of the requests will be small, but we fetch at kChunkSize bytes
  // at a time to avoid making too many potentially expensive GetBytes calls
  bool fetchToPos(size_t Pos) const {
    if (EOFReached) return Pos < ObjectSize;
    while (Pos >= BytesRead) {
      Bytes.resize(BytesRead + BytesSkipped + kChunkSize);
      size_t bytes = Streamer->GetBytes(&Bytes[BytesRead + BytesSkipped],
                                        kChunkSize);
      BytesRead += bytes;
      if (bytes < kChunkSize) {
        if (ObjectSize && BytesRead < Pos)
          assert(0 && "Unexpected short read fetching bitcode");
        if (BytesRead <= Pos) { // reached EOF/ran out of bytes
          ObjectSize = BytesRead;
          EOFReached = true;
          return false;
        }
      }
    }
    return true;
  }

  StreamingMemoryObject(const StreamingMemoryObject&) LLVM_DELETED_FUNCTION;
  void operator=(const StreamingMemoryObject&) LLVM_DELETED_FUNCTION;
};

StreamableMemoryObject *getNonStreamedMemoryObject(
    const unsigned char *Start, const unsigned char *End);

}
#endif  // STREAMABLEMEMORYOBJECT_H_
