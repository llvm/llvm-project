// Structured binding in C++ can bind identifiers to subobjects of an object.
//
// There are three cases we need to test:
// 1) arrays
// 2) tuples like objects
// 3) non-static data members
//
// They can also bind by copy, reference or rvalue reference.

#include <tuple>

struct A {
  int x;
  int y;
};

// We want to cover a mix of types and also different sizes to make sure we
// hande the offsets correctly.
struct MixedTypesAndSizesStruct {
  A a;
  char b1;
  char b2;
  short b3;
  int b4;
  char b5;
};

int main() {
  MixedTypesAndSizesStruct b{{20, 30}, 'a', 'b', 50, 60, 'c'};

  auto [a1, b1, c1, d1, e1, f1] = b;
  auto &[a2, b2, c2, d2, e2, f2] = b;
  auto &&[a3, b3, c3, d3, e3, f3] =
      MixedTypesAndSizesStruct{{20, 30}, 'a', 'b', 50, 60, 'c'};

  // Array with different sized types
  char carr[]{'a', 'b', 'c'};
  short sarr[]{11, 12, 13};
  int iarr[]{22, 33, 44};

  auto [carr_copy1, carr_copy2, carr_copy3] = carr;
  auto [sarr_copy1, sarr_copy2, sarr_copy3] = sarr;
  auto [iarr_copy1, iarr_copy2, iarr_copy3] = iarr;

  auto &[carr_ref1, carr_ref2, carr_ref3] = carr;
  auto &[sarr_ref1, sarr_ref2, sarr_ref3] = sarr;
  auto &[iarr_ref1, iarr_ref2, iarr_ref3] = iarr;

  auto &&[carr_rref1, carr_rref2, carr_rref3] = carr;
  auto &&[sarr_rref1, sarr_rref2, sarr_rref3] = sarr;
  auto &&[iarr_rref1, iarr_rref2, iarr_rref3] = iarr;

  float x{4.0};
  char y{'z'};
  int z{10};

  std::tuple<float, char, int> tpl(x, y, z);
  auto [tx1, ty1, tz1] = tpl;
  auto &[tx2, ty2, tz2] = tpl;

  return a1.x + b1 + c1 + d1 + e1 + f1 + a2.y + b2 + c2 + d2 + e2 + f2 + a3.x +
         b3 + c3 + d3 + e3 + f3 + carr_copy1 + carr_copy2 + carr_copy3 +
         sarr_copy1 + sarr_copy2 + sarr_copy3 + iarr_copy1 + iarr_copy2 +
         iarr_copy3 + carr_ref1 + carr_ref2 + carr_ref3 + sarr_ref1 +
         sarr_ref2 + sarr_ref3 + iarr_ref1 + iarr_ref2 + iarr_ref3 +
         carr_rref1 + carr_rref2 + carr_rref3 + sarr_rref1 + sarr_rref2 +
         sarr_rref3 + iarr_rref1 + iarr_rref2 + iarr_rref3 + tx1 + ty1 + tz1 +
         tx2 + ty2 + tz2; // break here
}
