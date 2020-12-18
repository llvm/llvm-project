#include <stdint.h>

int main(int argc, char const *argv[]) {
  struct LargeBitsA {
    unsigned int : 30, a : 20;
  } lba;

  struct LargeBitsB {
    unsigned int a : 1, : 11, : 12, b : 20;
  } lbb;

  struct LargeBitsC {
    unsigned int : 13, : 9, a : 1, b : 1, c : 5, d : 1, e : 20;
  } lbc;

  struct LargeBitsD {
    char arr[3];
    unsigned int : 30, a : 20;
  } lbd;

  // This case came up when debugging clang and models RecordDeclBits
  struct BitExampleFromClangDeclContext {
    class fields {
      uint64_t : 13;
      uint64_t : 9;

      uint64_t a: 1;
      uint64_t b: 1;
      uint64_t c: 1;
      uint64_t d: 1;
      uint64_t e: 1;
      uint64_t f: 1;
      uint64_t g: 1;
      uint64_t h: 1;
      uint64_t i: 1;
      uint64_t j: 1;
      uint64_t k: 1;

      // In order to reproduce the crash for this case we need the
      // members of fields to stay private :-(
      friend struct BitExampleFromClangDeclContext;
    };

    union {
      struct fields f;
    };

    BitExampleFromClangDeclContext() {
  f.a = 1;
  f.b = 0;
  f.c = 1;
  f.d = 0;
  f.e = 1;
  f.f = 0;
  f.g = 1;
  f.h = 0;
  f.i = 1;
  f.j = 0;
  f.k = 1;
    }
  } clang_example;

  class B {
  public:
    uint32_t b_a;
  };

  class D : public B {
  public:
    uint32_t d_a : 1;
  } derived;

  union union_with_bitfields {
      unsigned int a : 8;
      unsigned int b : 16;
      unsigned int c : 32;
      unsigned int x;
  } uwbf;

  union union_with_unnamed_bitfield {
   unsigned int : 16, a : 24;
   unsigned int x;
  } uwubf;

  lba.a = 2;

  lbb.a = 1;
  lbb.b = 3;

  lbc.a = 1;
  lbc.b = 0;
  lbc.c = 4;
  lbc.d = 1;
  lbc.e = 20;

  lbd.arr[0] = 'a';
  lbd.arr[1] = 'b';
  lbd.arr[2] = '\0';
  lbd.a = 5;

  derived.b_a = 2;
  derived.d_a = 1;

  uwbf.x = 0xFFFFFFFF;
  uwubf.x = 0xFFFFFFFF;

  return 0; // Set break point at this line.
}
