//===- MemoryObject.h - Abstract memory interface ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef MEMORYOBJECT_H
#define MEMORYOBJECT_H

#include "llvm/Support/DataTypes.h"

namespace llvm {

/// MemoryObject - Abstract base class for contiguous addressable memory.
///   Necessary for cases in which the memory is in another process, in a
///   file, or on a remote machine.
///   All size and offset parameters are uint64_ts, to allow 32-bit processes
///   access to 64-bit address spaces.
class MemoryObject {
public:
  /// Destructor      - Override as necessary.
  virtual ~MemoryObject();

  /// getBase         - Returns the lowest valid address in the region.
  ///
  /// @result         - The lowest valid address.
  virtual uint64_t getBase() const = 0;

  /// getExtent       - Returns the size of the region in bytes.  (The region is
  ///                   contiguous, so the highest valid address of the region
  ///                   is getBase() + getExtent() - 1).
  ///
  /// @result         - The size of the region.
  virtual uint64_t getExtent() const = 0;

  /// readByte        - Tries to read a single byte from the region.
  ///
  /// @param address  - The address of the byte, in the same space as getBase().
  /// @param ptr      - A pointer to a byte to be filled in.  Must be non-NULL.
  /// @result         - 0 if successful; -1 if not.  Failure may be due to a
  ///                   bounds violation or an implementation-specific error.
  virtual int readByte(uint64_t address, uint8_t* ptr) const = 0;

  /// readBytes       - Tries to read a contiguous range of bytes from the
  ///                   region, up to the end of the region.
  ///                   You should override this function if there is a quicker
  ///                   way than going back and forth with individual bytes.
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
                        uint64_t* copied) const;
};

}

#endif
