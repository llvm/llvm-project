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
  void write(const char *data, size_t len);

  /// Return a reference to the current SHA1 for the digested data since the
  /// last call to init()
  StringRef result();

private:
  static constexpr int BLOCK_LENGTH = 64;
  static constexpr int HASH_LENGTH = 20;

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
