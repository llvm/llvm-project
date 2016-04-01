//==- SHA1.h - SHA1 implementation for LLVM                     --*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This code is taken from public domain
// (http://oauth.googlecode.com/svn/code/c/liboauth/src/sha1.c)
// and modified by wrapping it in a C++ interface for LLVM,
// and removing unnecessary code.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_SHA1_H
#define LLVM_SUPPORT_SHA1_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace llvm {

/// A class that wrap the SHA1 algorithm.
class SHA1 {
public:
  SHA1() { init(); }

  /// Reinitialize the internal state
  void init();

  /// Digest more data.
  void update(ArrayRef<uint8_t> Data);

  /// Return a reference to the current raw 160-bits SHA1 for the digested data
  /// since the last call to init(). This call will add data to the internal
  /// state and as such is not suited for getting an intermediate result
  /// (see result()).
  StringRef final();

  /// Return a reference to the current raw 160-bits SHA1 for the digested data
  /// since the last call to init(). This is suitable for getting the SHA1 at
  /// any time without invalidating the internal state so that more calls can be
  /// made into update.
  StringRef result();

private:
  /// Define some constants.
  /// "static constexpr" would be cleaner but MSVC does not support it yet.
  enum { BLOCK_LENGTH = 64 };
  enum { HASH_LENGTH = 20 };

  // Internal State
  struct {
    uint32_t Buffer[BLOCK_LENGTH / 4];
    uint32_t State[HASH_LENGTH / 4];
    uint32_t ByteCount;
    uint8_t BufferOffset;
  } InternalState;

  // Internal copy of the hash, populated and accessed on calls to result()
  uint32_t HashResult[HASH_LENGTH / 4];

  // Helper
  void writebyte(uint8_t data);
  void hashBlock();
  void addUncounted(uint8_t data);
  void pad();
};

} // end llvm namespace

#endif
