// Reduced from llvm/lib/Support/xxhash.cpp. LLVM describes the secret of the
// last accumulate_512 with DW_OP_convert operations ahead of the address of
// kSecret.
#include <arm_neon.h>

static const unsigned char kSecret[192] = {
    0xb8, 0xfe, 0x6c, 0x39, 0x23, 0xa4, 0x4b, 0xbe, 0x7c, 0x01, 0x81, 0x2c,
    0xf7, 0x21, 0xad, 0x1c, 0xde, 0xd4, 0x6d, 0xe9, 0x83, 0x90, 0x97, 0xdb};

static inline __attribute__((always_inline)) void
accumulate_512(unsigned long long *acc, const unsigned char *input,
               const unsigned char *secret) {
  uint64x2_t *xacc = (uint64x2_t *)acc;
  for (unsigned long i = 0; i < 4; ++i) {
    uint64x2_t data = vreinterpretq_u64_u8(vld1q_u8(input + i * 16));
    uint64x2_t key = vreinterpretq_u64_u8(vld1q_u8(secret + i * 16));
    xacc[i] = vaddq_u64(xacc[i], veorq_u64(data, key));
  }
}

static inline __attribute__((always_inline)) void
accumulate(unsigned long long *acc, const unsigned char *input,
           const unsigned char *secret, unsigned long nb_stripes) {
  for (unsigned long n = 0; n < nb_stripes; ++n)
    accumulate_512(acc, input + n * 64, secret + n * 8);
}

__attribute__((noinline)) static unsigned long long
hash_long(const unsigned char *input, unsigned long len,
          const unsigned char *secret, unsigned long secret_size) {
  unsigned long stripes_per_block = (secret_size - 64) / 8;
  unsigned long block_len = 64 * stripes_per_block;
  unsigned long nb_blocks = (len - 1) / block_len;
  __attribute__((aligned(16))) unsigned long long acc[8] = {0};
  unsigned long nb_stripes = (len - 1 - block_len * nb_blocks) / 64;
  accumulate(acc, input + nb_blocks * block_len, secret, nb_stripes);
  accumulate_512(acc, input + len - 64, secret + secret_size - 64 - 7);
  return acc[0] ^ acc[1] ^ acc[2] ^ acc[3] ^ acc[4] ^ acc[5] ^ acc[6] ^ acc[7];
}

unsigned long long xxh3(const unsigned char *in, unsigned long len) {
  return hash_long(in, len, kSecret, sizeof(kSecret));
}
