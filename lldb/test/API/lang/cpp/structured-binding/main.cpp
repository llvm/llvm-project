// Structured binding in C++ can bind identifiers to subobjects of an object.
//
// There are four cases we need to test:
// 1) arrays
// 2) tuple-like objects with `get` member functions
// 3) tuple-like objects with `get` free functions
// 4) non-static data members
//
// They can also bind by copy, reference or rvalue reference.

struct MyPair {
  int m1;
  int m2;

  // Helpers to enable tuple-like decomposition.
  template <unsigned> int get();
  template <> int get<0>() { return m1; }
  template <> int get<1>() { return m2; }
};

namespace std {
template <typename T1, typename T2, typename T3> struct mock_tuple {
  T1 m1;
  T2 m2;
  T3 m3;
};

template <typename T> struct tuple_size;

template <unsigned, typename T> struct tuple_element;

// Helpers to enable tuple-like decomposition for MyPair
template <unsigned I> struct tuple_element<I, MyPair> {
  using type = int;
};

template <> struct tuple_size<MyPair> {
  static constexpr unsigned value = 2;
};

// Helpers to enable tuple-like decomposition for mock_tuple
template <typename T1, typename T2, typename T3>
struct tuple_element<0, mock_tuple<T1, T2, T3>> {
  using type = T1;
};

template <typename T1, typename T2, typename T3>
struct tuple_element<1, mock_tuple<T1, T2, T3>> {
  using type = T2;
};

template <typename T1, typename T2, typename T3>
struct tuple_element<2, mock_tuple<T1, T2, T3>> {
  using type = T3;
};

template <typename T1, typename T2, typename T3>
struct tuple_size<mock_tuple<T1, T2, T3>> {
  static constexpr unsigned value = 3;
};

template <unsigned I, typename T1, typename T2, typename T3>
typename tuple_element<I, mock_tuple<T1, T2, T3>>::type
get(mock_tuple<T1, T2, T3> p) {
  switch (I) {
  case 0:
    return p.m1;
  case 1:
    return p.m2;
  case 2:
    return p.m3;
  default:
    __builtin_trap();
  }
}

} // namespace std

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

  std::mock_tuple<float, char, int> tpl{.m1 = x, .m2 = y, .m3 = z};
  auto [tx1, ty1, tz1] = tpl;
  auto &[tx2, ty2, tz2] = tpl;

  auto [mp1, mp2] = MyPair{.m1 = 1, .m2 = 2};

  return a1.x + b1 + c1 + d1 + e1 + f1 + a2.y + b2 + c2 + d2 + e2 + f2 + a3.x +
         b3 + c3 + d3 + e3 + f3 + carr_copy1 + carr_copy2 + carr_copy3 +
         sarr_copy1 + sarr_copy2 + sarr_copy3 + iarr_copy1 + iarr_copy2 +
         iarr_copy3 + carr_ref1 + carr_ref2 + carr_ref3 + sarr_ref1 +
         sarr_ref2 + sarr_ref3 + iarr_ref1 + iarr_ref2 + iarr_ref3 +
         carr_rref1 + carr_rref2 + carr_rref3 + sarr_rref1 + sarr_rref2 +
         sarr_rref3 + iarr_rref1 + iarr_rref2 + iarr_rref3 + tx1 + ty1 + tz1 +
         tx2 + ty2 + tz2 + mp1 + mp2; // break here
}
