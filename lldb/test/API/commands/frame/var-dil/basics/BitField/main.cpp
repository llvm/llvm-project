#include <cstdint>

int main(int argc, char **argv) {
  enum BitFieldEnum : uint32_t { kZero, kOne };

  struct BitFieldStruct {
    uint16_t a : 10;
    uint32_t b : 4;
    bool c : 1;
    bool d : 1;
    int32_t e : 32;
    uint32_t f : 32;
    uint32_t g : 31;
    uint64_t h : 31;
    uint64_t i : 33;
    BitFieldEnum j : 10;
  };

  BitFieldStruct bf;
  bf.a = 0b1111111111;
  bf.b = 0b1001;
  bf.c = 0b0;
  bf.d = 0b1;
  bf.e = 0b1;
  bf.f = 0b1;
  bf.g = 0b1;
  bf.h = 0b1;
  bf.i = 0b1;
  bf.j = BitFieldEnum::kOne;

  struct AlignedBitFieldStruct {
    uint16_t a : 10;
    uint8_t b : 4;
    unsigned char : 0;
    uint16_t c : 2;
  };

  uint32_t data = ~0;
  AlignedBitFieldStruct abf = (AlignedBitFieldStruct &)data;

  return 0; // Set a breakpoint here
}
