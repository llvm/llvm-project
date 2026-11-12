#include <stdint.h>

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

struct BitfieldsInStructInUnion {
  class fields {
    uint64_t : 13;
    uint64_t : 9;

    uint64_t a : 1;
    uint64_t b : 1;
    uint64_t c : 1;
    uint64_t d : 1;
    uint64_t e : 1;
    uint64_t f : 1;
    uint64_t g : 1;
    uint64_t h : 1;
    uint64_t i : 1;
    uint64_t j : 1;
    uint64_t k : 1;

    // In order to reproduce the crash for this case we need the
    // members of fields to stay private :-(
    friend struct BitfieldsInStructInUnion;
  };

  union {
    struct fields f;
  };

  BitfieldsInStructInUnion() {
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
} bitfields_in_struct_in_union;

class Base {
public:
  uint32_t b_a;
};

class Derived : public Base {
public:
  uint32_t d_a : 1;
} derived;

union UnionWithBitfields {
  unsigned int a : 8;
  unsigned int b : 16;
  unsigned int c : 32;
  unsigned int x;
} uwbf;

union UnionWithUnnamedBitfield {
  unsigned int : 16, a : 24;
  unsigned int x;
} uwubf;

struct BoolBits {
  bool a : 1;
  bool b : 1;
  bool c : 2;
  bool d : 2;
};

struct WithVTable {
  virtual ~WithVTable() {}
  unsigned a : 4;
  unsigned b : 4;
  unsigned c : 4;
};
WithVTable with_vtable;

struct WithVTableAndUnnamed {
  virtual ~WithVTableAndUnnamed() {}
  unsigned : 4;
  unsigned b : 4;
  unsigned c : 4;
};
WithVTableAndUnnamed with_vtable_and_unnamed;

struct BaseWithVTable {
  virtual ~BaseWithVTable() {}
};
struct HasBaseWithVTable : BaseWithVTable {
  unsigned a : 4;
  unsigned b : 4;
  unsigned c : 4;
};
HasBaseWithVTable base_with_vtable;

struct DerivedWithVTable : public Base {
  virtual ~DerivedWithVTable() {}
  unsigned a : 1;
};
DerivedWithVTable derived_with_vtable;

int main(int argc, char const *argv[]) {
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

  BoolBits bb;
  bb.a = 0b1;
  bb.b = 0b0;
  bb.c = 0b11;
  bb.d = 0b01;

  with_vtable.a = 5;
  with_vtable.b = 0;
  with_vtable.c = 5;

  with_vtable_and_unnamed.b = 0;
  with_vtable_and_unnamed.c = 5;

  base_with_vtable.a = 5;
  base_with_vtable.b = 0;
  base_with_vtable.c = 5;

  derived_with_vtable.b_a = 2;
  derived_with_vtable.a = 1;

  return 0; // break here
}
