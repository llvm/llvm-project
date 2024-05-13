#include <stdint.h>

struct Bits {
  uint32_t    : 1;
  uint32_t    b1 : 1;
  uint32_t    b2 : 2;
  uint32_t    : 2;
  uint32_t    b3 : 3;
  // Unnamed bitfield (this will get removed).
  uint32_t    : 2;
  uint32_t    b4 __attribute__ ((aligned(16)));
  uint32_t    b5 : 5;
  uint32_t    b6 : 6;
  uint32_t    b7 : 7;
  uint32_t    four : 4;
};

struct MoreBits {
  uint32_t a : 3;
  uint8_t : 1;
  uint8_t b : 1;
  uint8_t c : 1;
  uint8_t d : 1;
};
struct MoreBits more_bits;

struct ManySingleBits {
  uint16_t b1 : 1, b2 : 1, b3 : 1, b4 : 1, b5 : 1, b6 : 1, b7 : 1, b8 : 1,
      b9 : 1, b10 : 1, b11 : 1, b12 : 1, b13 : 1, b14 : 1, b15 : 1, b16 : 1,
      b17 : 1;
};
struct ManySingleBits many_single_bits;

struct LargePackedBits {
  uint64_t a : 36;
  uint64_t b : 36;
} __attribute__((packed));

#pragma pack(1)
struct PackedBits {
  char a;
  uint32_t b : 5, c : 27;
};
#pragma pack()

int main(int argc, char const *argv[]) {
  struct Bits bits;
  bits.b1 = 1;
  bits.b2 = 3;
  bits.b3 = 7;
  bits.b4 = 15;
  bits.b5 = 31;
  bits.b6 = 63;
  bits.b7 = 127;
  bits.four = 15;

  more_bits.a = 3;
  more_bits.b = 0;
  more_bits.c = 1;
  more_bits.d = 0;

  many_single_bits.b1 = 1;
  many_single_bits.b5 = 1;
  many_single_bits.b7 = 1;
  many_single_bits.b13 = 1;

  struct PackedBits packed;
  packed.a = 'a';
  packed.b = 10;
  packed.c = 0x7112233;

  struct LargePackedBits large_packed =
      (struct LargePackedBits){0xcbbbbaaaa, 0xdffffeeee};
  struct LargePackedBits *large_packed_ptr = &large_packed;

  return 0; // break here
}
