//======- SHA1.h - Private copy of the SHA1 implementation ---*- C++ -* ======//
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

/* This code is public-domain - it is based on libcrypt
 * placed in the public domain by Wei Dai and other contributors.
 */

#include "llvm/Support/SHA1.h"
using namespace llvm;

#include <stdint.h>
#include <string.h>

#ifdef __BIG_ENDIAN__
#define SHA_BIG_ENDIAN
#elif defined __LITTLE_ENDIAN__
/* override */
#elif defined __BYTE_ORDER
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define SHA_BIG_ENDIAN
#endif
#else               // ! defined __LITTLE_ENDIAN__
#include <endian.h> // machine/endian.h
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define SHA_BIG_ENDIAN
#endif
#endif

/* code */
#define SHA1_K0 0x5a827999
#define SHA1_K20 0x6ed9eba1
#define SHA1_K40 0x8f1bbcdc
#define SHA1_K60 0xca62c1d6

void SHA1::init() {
  InternalState.State[0] = 0x67452301;
  InternalState.State[1] = 0xefcdab89;
  InternalState.State[2] = 0x98badcfe;
  InternalState.State[3] = 0x10325476;
  InternalState.State[4] = 0xc3d2e1f0;
  InternalState.ByteCount = 0;
  InternalState.BufferOffset = 0;
}

static uint32_t rol32(uint32_t number, uint8_t bits) {
  return ((number << bits) | (number >> (32 - bits)));
}

void SHA1::hashBlock() {
  uint8_t i;
  uint32_t a, b, c, d, e, t;

  a = InternalState.State[0];
  b = InternalState.State[1];
  c = InternalState.State[2];
  d = InternalState.State[3];
  e = InternalState.State[4];
  for (i = 0; i < 80; i++) {
    if (i >= 16) {
      t = InternalState.Buffer[(i + 13) & 15] ^
          InternalState.Buffer[(i + 8) & 15] ^
          InternalState.Buffer[(i + 2) & 15] ^ InternalState.Buffer[i & 15];
      InternalState.Buffer[i & 15] = rol32(t, 1);
    }
    if (i < 20) {
      t = (d ^ (b & (c ^ d))) + SHA1_K0;
    } else if (i < 40) {
      t = (b ^ c ^ d) + SHA1_K20;
    } else if (i < 60) {
      t = ((b & c) | (d & (b | c))) + SHA1_K40;
    } else {
      t = (b ^ c ^ d) + SHA1_K60;
    }
    t += rol32(a, 5) + e + InternalState.Buffer[i & 15];
    e = d;
    d = c;
    c = rol32(b, 30);
    b = a;
    a = t;
  }
  InternalState.State[0] += a;
  InternalState.State[1] += b;
  InternalState.State[2] += c;
  InternalState.State[3] += d;
  InternalState.State[4] += e;
}

void SHA1::addUncounted(uint8_t data) {
  uint8_t *const b = (uint8_t *)InternalState.Buffer;
#ifdef SHA_BIG_ENDIAN
  b[InternalState.BufferOffset] = data;
#else
  b[InternalState.BufferOffset ^ 3] = data;
#endif
  InternalState.BufferOffset++;
  if (InternalState.BufferOffset == BLOCK_LENGTH) {
    hashBlock();
    InternalState.BufferOffset = 0;
  }
}

void SHA1::writebyte(uint8_t data) {
  ++InternalState.ByteCount;
  addUncounted(data);
}

void SHA1::write(const char *data, size_t len) {
  for (; len--;)
    writebyte((uint8_t)*data++);
}

void SHA1::pad() {
  // Implement SHA-1 padding (fips180-2 §5.1.1)

  // Pad with 0x80 followed by 0x00 until the end of the block
  addUncounted(0x80);
  while (InternalState.BufferOffset != 56)
    addUncounted(0x00);

  // Append length in the last 8 bytes
  addUncounted(0); // We're only using 32 bit lengths
  addUncounted(0); // But SHA-1 supports 64 bit lengths
  addUncounted(0); // So zero pad the top bits
  addUncounted(InternalState.ByteCount >> 29); // Shifting to multiply by 8
  addUncounted(InternalState.ByteCount >>
               21); // as SHA-1 supports bitstreams as well as
  addUncounted(InternalState.ByteCount >> 13); // byte.
  addUncounted(InternalState.ByteCount >> 5);
  addUncounted(InternalState.ByteCount << 3);
}
#include <cstdio>
StringRef SHA1::result() {
  auto StateToRestore = InternalState;

  // Pad to complete the last block
  pad();

#ifndef SHA_BIG_ENDIAN
  // Swap byte order back
  for (int i = 0; i < 5; i++) {
    HashResult[i] = (((InternalState.State[i]) << 24) & 0xff000000) |
                    (((InternalState.State[i]) << 8) & 0x00ff0000) |
                    (((InternalState.State[i]) >> 8) & 0x0000ff00) |
                    (((InternalState.State[i]) >> 24) & 0x000000ff);
  }
#else
  // Just copy the current state
  for (int i = 0; i < 5; i++) {
    HashResult[i] = InternalState.State[i];
  }
#endif

  // Restore the state
  InternalState = StateToRestore;

  // Return pointer to hash (20 characters)
  return StringRef((char *)HashResult, HASH_LENGTH);
}
