//===--- SipHash.h - An implementation of SipHash -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This is a header-only implementation of SipHash. It lacks library
//  dependencies so it can be used from LLVM and compiler-rt.
//
//===----------------------------------------------------------------------===//

#include <stddef.h>
#include <stdint.h>

// Lightly adapted from the SipHash reference C implementation:
//   https://github.com/veorq/SipHash
// by Jean-Philippe Aumasson and Daniel J. Bernstein

#define ROTL(x, b) (uint64_t)(((x) << (b)) | ((x) >> (64 - (b))))

#define U32TO8_LE(p, v)                                                        \
    (p)[0] = (uint8_t)((v));                                                   \
    (p)[1] = (uint8_t)((v) >> 8);                                              \
    (p)[2] = (uint8_t)((v) >> 16);                                             \
    (p)[3] = (uint8_t)((v) >> 24);

#define U64TO8_LE(p, v)                                                        \
    U32TO8_LE((p), (uint32_t)((v)));                                           \
    U32TO8_LE((p) + 4, (uint32_t)((v) >> 32));

#define U8TO64_LE(p)                                                           \
    (((uint64_t)((p)[0])) | ((uint64_t)((p)[1]) << 8) |                        \
     ((uint64_t)((p)[2]) << 16) | ((uint64_t)((p)[3]) << 24) |                 \
     ((uint64_t)((p)[4]) << 32) | ((uint64_t)((p)[5]) << 40) |                 \
     ((uint64_t)((p)[6]) << 48) | ((uint64_t)((p)[7]) << 56))

#define SIPROUND                                                               \
  do {                                                                         \
    v0 += v1;                                                                  \
    v1 = ROTL(v1, 13);                                                         \
    v1 ^= v0;                                                                  \
    v0 = ROTL(v0, 32);                                                         \
    v2 += v3;                                                                  \
    v3 = ROTL(v3, 16);                                                         \
    v3 ^= v2;                                                                  \
    v0 += v3;                                                                  \
    v3 = ROTL(v3, 21);                                                         \
    v3 ^= v0;                                                                  \
    v2 += v1;                                                                  \
    v1 = ROTL(v1, 17);                                                         \
    v1 ^= v2;                                                                  \
    v2 = ROTL(v2, 32);                                                         \
  } while (0)

namespace {

/// Computes a SipHash value
///
/// \param in: pointer to input data (read-only)
/// \param inlen: input data length in bytes (any size_t value)
/// \param k: reference to the key data 16-byte array (read-only)
/// \returns output data, must be 8 or 16 bytes
///
template <int cROUNDS, int dROUNDS, size_t outlen>
void siphash(const unsigned char *in, uint64_t inlen,
             const unsigned char (&k)[16], unsigned char (&out)[outlen]) {

  const unsigned char *ni = (const unsigned char *)in;
  const unsigned char *kk = (const unsigned char *)k;

  static_assert(outlen == 8 || outlen == 16, "result should be 8 or 16 bytes");

  uint64_t v0 = UINT64_C(0x736f6d6570736575);
  uint64_t v1 = UINT64_C(0x646f72616e646f6d);
  uint64_t v2 = UINT64_C(0x6c7967656e657261);
  uint64_t v3 = UINT64_C(0x7465646279746573);
  uint64_t k0 = U8TO64_LE(kk);
  uint64_t k1 = U8TO64_LE(kk + 8);
  uint64_t m;
  int i;
  const unsigned char *end = ni + inlen - (inlen % sizeof(uint64_t));
  const int left = inlen & 7;
  uint64_t b = ((uint64_t)inlen) << 56;
  v3 ^= k1;
  v2 ^= k0;
  v1 ^= k1;
  v0 ^= k0;

  if (outlen == 16)
    v1 ^= 0xee;

  for (; ni != end; ni += 8) {
    m = U8TO64_LE(ni);
    v3 ^= m;

    for (i = 0; i < cROUNDS; ++i)
      SIPROUND;

    v0 ^= m;
  }

  switch (left) {
  case 7:
    b |= ((uint64_t)ni[6]) << 48;
    [[fallthrough]];
  case 6:
    b |= ((uint64_t)ni[5]) << 40;
    [[fallthrough]];
  case 5:
    b |= ((uint64_t)ni[4]) << 32;
    [[fallthrough]];
  case 4:
    b |= ((uint64_t)ni[3]) << 24;
    [[fallthrough]];
  case 3:
    b |= ((uint64_t)ni[2]) << 16;
    [[fallthrough]];
  case 2:
    b |= ((uint64_t)ni[1]) << 8;
    [[fallthrough]];
  case 1:
    b |= ((uint64_t)ni[0]);
    break;
  case 0:
    break;
  }

  v3 ^= b;

  for (i = 0; i < cROUNDS; ++i)
    SIPROUND;

  v0 ^= b;

  if (outlen == 16)
    v2 ^= 0xee;
  else
    v2 ^= 0xff;

  for (i = 0; i < dROUNDS; ++i)
    SIPROUND;

  b = v0 ^ v1 ^ v2 ^ v3;
  U64TO8_LE(out, b);

  if (outlen == 8)
    return;

  v1 ^= 0xdd;

  for (i = 0; i < dROUNDS; ++i)
    SIPROUND;

  b = v0 ^ v1 ^ v2 ^ v3;
  U64TO8_LE(out + 8, b);
}

} // end anonymous namespace
