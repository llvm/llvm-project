//===- CryptoUtils.h - Cryptographically Secure Pseudo-Random
// Generator------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains includes and defines for the AES CTR PRNG
// The AES implementation has been derived and adapted
// from libtomcrypt (see http://libtom.org)
// Created on: 22 juin 2012
// Author(s): jrinaldini, pjunod
//===----------------------------------------------------------------------===//

#ifndef LLVM_CryptoUtils_H
#define LLVM_CryptoUtils_H

#include "llvm/Support/ManagedStatic.h"

#include <stdint.h>
#include <cstdio>
#include <string>

namespace llvm {

class CryptoUtils;
extern ManagedStatic<CryptoUtils> cryptoutils;

#define BYTE(x, n) (((x) >> (8 * (n))) & 0xFF)

#if defined(__i386) || defined(__i386__) || defined(_M_IX86) ||                \
    defined(INTEL_CC)

#ifndef ENDIAN_LITTLE
#define ENDIAN_LITTLE
#endif
#define ENDIAN_32BITWORD
#define UNALIGNED

#elif defined(__alpha)

#ifndef ENDIAN_LITTLE
#define ENDIAN_LITTLE
#endif
#define ENDIAN_64BITWORD

#elif defined(__x86_64__)

#ifndef ENDIAN_LITTLE
#define ENDIAN_LITTLE
#endif
#define ENDIAN_64BITWORD
#define UNALIGNED

#elif(defined(__R5900) || defined(R5900) || defined(__R5900__)) &&             \
    (defined(_mips) || defined(__mips__) || defined(mips))

#ifndef ENDIAN_LITTLE
#define ENDIAN_LITTLE
#endif
#define ENDIAN_64BITWORD

#elif defined(__sparc)

#ifndef ENDIAN_BIG
#define ENDIAN_BIG
#endif
#if defined(__arch64__)
#define ENDIAN_64BITWORD
#else
#define ENDIAN_32BITWORD
#endif

#endif

#if defined(__BIG_ENDIAN__) || defined(_BIG_ENDIAN)
#define ENDIAN_BIG
#endif

#if !defined(ENDIAN_BIG) && !defined(ENDIAN_LITTLE)
#error                                                                         \
    "Unknown endianness of the compilation platform, check this header aes_encrypt.h"
#endif

#ifdef ENDIAN_LITTLE

#define STORE32H(y, x)                                                         \
  {                                                                            \
    (y)[0] = (uint8_t)(((x) >> 24) & 0xFF);                                    \
    (y)[1] = (uint8_t)(((x) >> 16) & 0xFF);                                    \
    (y)[2] = (uint8_t)(((x) >> 8) & 0xFF);                                     \
    (y)[3] = (uint8_t)(((x) >> 0) & 0xFF);                                     \
  }
#define LOAD32H(x, y)                                                          \
  {                                                                            \
    (x) = ((uint32_t)((y)[0] & 0xFF) << 24) |                                  \
          ((uint32_t)((y)[1] & 0xFF) << 16) |                                  \
          ((uint32_t)((y)[2] & 0xFF) << 8) | ((uint32_t)((y)[3] & 0xFF) << 0); \
  }

#define LOAD64H(x, y)                                                          \
  {                                                                            \
    (x) = ((uint64_t)((y)[0] & 0xFF) << 56) |                                  \
          ((uint64_t)((y)[1] & 0xFF) << 48) |                                  \
          ((uint64_t)((y)[2] & 0xFF) << 40) |                                  \
          ((uint64_t)((y)[3] & 0xFF) << 32) |                                  \
          ((uint64_t)((y)[4] & 0xFF) << 24) |                                  \
          ((uint64_t)((y)[5] & 0xFF) << 16) |                                  \
          ((uint64_t)((y)[6] & 0xFF) << 8) | ((uint64_t)((y)[7] & 0xFF) << 0); \
  }

#define STORE64H(y, x)                                                         \
  {                                                                            \
    (y)[0] = (uint8_t)(((x) >> 56) & 0xFF);                                    \
    (y)[1] = (uint8_t)(((x) >> 48) & 0xFF);                                    \
    (y)[2] = (uint8_t)(((x) >> 40) & 0xFF);                                    \
    (y)[3] = (uint8_t)(((x) >> 32) & 0xFF);                                    \
    (y)[4] = (uint8_t)(((x) >> 24) & 0xFF);                                    \
    (y)[5] = (uint8_t)(((x) >> 16) & 0xFF);                                    \
    (y)[6] = (uint8_t)(((x) >> 8) & 0xFF);                                     \
    (y)[7] = (uint8_t)(((x) >> 0) & 0xFF);                                     \
  }

#endif /* ENDIAN_LITTLE */

#ifdef ENDIAN_BIG

#define STORE32H(y, x)                                                         \
  {                                                                            \
    (y)[3] = (uint8_t)(((x) >> 24) & 0xFF);                                    \
    (y)[2] = (uint8_t)(((x) >> 16) & 0xFF);                                    \
    (y)[1] = (uint8_t)(((x) >> 8) & 0xFF);                                     \
    (y)[0] = (uint8_t)(((x) >> 0) & 0xFF);                                     \
  }
#define STORE64H(y, x)                                                         \
  {                                                                            \
    (y)[7] = (uint8_t)(((x) >> 56) & 0xFF);                                    \
    (y)[6] = (uint8_t)(((x) >> 48) & 0xFF);                                    \
    (y)[5] = (uint8_t)(((x) >> 40) & 0xFF);                                    \
    (y)[4] = (uint8_t)(((x) >> 32) & 0xFF);                                    \
    (y)[3] = (uint8_t)(((x) >> 24) & 0xFF);                                    \
    (y)[2] = (uint8_t)(((x) >> 16) & 0xFF);                                    \
    (y)[1] = (uint8_t)(((x) >> 8) & 0xFF);                                     \
    (y)[0] = (uint8_t)(((x) >> 0) & 0xFF);                                     \
  }
#define LOAD32H(x, y)                                                          \
  {                                                                            \
    (x) = ((uint32_t)((y)[3] & 0xFF) << 24) |                                  \
          ((uint32_t)((y)[2] & 0xFF) << 16) |                                  \
          ((uint32_t)((y)[1] & 0xFF) << 8) | ((uint32_t)((y)[0] & 0xFF) << 0); \
}

#define LOAD64H(x, y)                                                          \
  {                                                                            \
    (x) = ((uint64_t)((y)[7] & 0xFF) << 56) |                                  \
          ((uint64_t)((y)[6] & 0xFF) << 48) |                                  \
          ((uint64_t)((y)[5] & 0xFF) << 40) |                                  \
          ((uint64_t)((y)[4] & 0xFF) << 32) |                                  \
          ((uint64_t)((y)[3] & 0xFF) << 24) |                                  \
          ((uint64_t)((y)[2] & 0xFF) << 16) |                                  \
          ((uint64_t)((y)[1] & 0xFF) << 8) | ((uint64_t)((y)[0] & 0xFF) << 0); \
  }

#endif /* ENDIAN_BIG */

#define AES_TE0(x) AES_PRECOMP_TE0[(x)]
#define AES_TE1(x) AES_PRECOMP_TE1[(x)]
#define AES_TE2(x) AES_PRECOMP_TE2[(x)]
#define AES_TE3(x) AES_PRECOMP_TE3[(x)]

#define AES_TE4_0(x) AES_PRECOMP_TE4_0[(x)]
#define AES_TE4_1(x) AES_PRECOMP_TE4_1[(x)]
#define AES_TE4_2(x) AES_PRECOMP_TE4_2[(x)]
#define AES_TE4_3(x) AES_PRECOMP_TE4_3[(x)]

#define CryptoUtils_POOL_SIZE (0x1 << 17) // 2^17

#define DUMP(x, l, s)                                                          \
  fprintf(stderr, "%s :", (s));                                                \
  for (int ii = 0; ii < (l); ii++) {                                           \
    fprintf(stderr, "%02hhX", *((x) + ii));                                    \
  }                                                                            \
  fprintf(stderr, "\n");

// SHA256
/* Various logical functions */
#define Ch(x, y, z) (z ^ (x &(y ^ z)))
#define Maj(x, y, z) (((x | y) & z) | (x &y))
#define S(x, n) RORc((x), (n))
#define R1(x, n) (((x) & 0xFFFFFFFFUL) >> (n))
#define Sigma0(x) (S(x, 2) ^ S(x, 13) ^ S(x, 22))
#define Sigma1(x) (S(x, 6) ^ S(x, 11) ^ S(x, 25))
#define Gamma0(x) (S(x, 7) ^ S(x, 18) ^ R1(x, 3))
#define Gamma1(x) (S(x, 17) ^ S(x, 19) ^ R1(x, 10))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define RND(a, b, c, d, e, f, g, h, i, ki)                                     \
  t0 = h + Sigma1(e) + Ch(e, f, g) + ki + W[i];                                \
  t1 = Sigma0(a) + Maj(a, b, c);                                               \
  d += t0;                                                                     \
  h = t0 + t1;

#define RORc(x, y)                                                             \
  (((((unsigned long)(x) & 0xFFFFFFFFUL) >> (unsigned long)((y) & 31)) |       \
    ((unsigned long)(x) << (unsigned long)(32 - ((y) & 31)))) &                \
   0xFFFFFFFFUL)

class CryptoUtils {
public:
  CryptoUtils();
  ~CryptoUtils();

  char *get_seed();
  void get_bytes(char *buffer, const int len);
  char get_char();
  bool prng_seed(const std::string seed);

  // Returns a uniformly distributed 8-bit value
  uint8_t get_uint8_t();
  // Returns a uniformly distributed 32-bit value
  uint32_t get_uint32_t();
  // Returns an integer uniformly distributed on [0, max[
  uint32_t get_range(const uint32_t max);
  // Returns a uniformly distributed 64-bit value
  uint64_t get_uint64_t();

  // Scramble a 32-bit value depending on a 128-bit value
  unsigned scramble32(const unsigned in, const char key[16]);

  int sha256(const char *msg, unsigned char *hash);

private:
  uint32_t ks[44];
  char key[16];
  char ctr[16];
  char pool[CryptoUtils_POOL_SIZE];
  uint32_t idx;
  std::string seed;
  bool seeded;

  typedef struct {
    uint64_t length;
    uint32_t state[8], curlen;
    unsigned char buf[64];
  } sha256_state;

  void aes_compute_ks(uint32_t *ks, const char *k);
  void aes_encrypt(char *out, const char *in, const uint32_t *ks);
  bool prng_seed();
  void inc_ctr();
  void populate_pool();
  int sha256_done(sha256_state *md, unsigned char *out);
  int sha256_init(sha256_state *md);
  static int sha256_compress(sha256_state *md, unsigned char *buf);
  int sha256_process(sha256_state *md, const unsigned char *in,
                     unsigned long inlen);
};
}

#endif // LLVM_CryptoUtils_H

