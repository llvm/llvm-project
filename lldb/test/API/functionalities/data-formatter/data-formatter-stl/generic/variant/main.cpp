#include <cstdio>
#include <string>
#include <variant>
#include <vector>

struct S {
  S() = default;
  S(S &&) { throw 42; }
  S &operator=(S &&) = default;
};

int main() {
  std::variant<int, double, char> v1;
  std::variant<int, double, char> &v1_ref = v1;

  using V1_typedef = std::variant<int, double, char>;
  V1_typedef v1_typedef;
  V1_typedef &v1_typedef_ref = v1_typedef;

  std::variant<int, double, char> v2;
  std::variant<int, double, char> v3;
  std::variant<std::variant<int, double, char>> v_v1;
  std::variant<int, char, S> v_valueless = 5;
  // The next variant has 300 types, meaning the type index does not fit in
  // a byte and must be `unsigned short` instead of `unsigned char` when
  // using the unstable libc++ ABI. With stable libc++ ABI, the type index
  // is always just `unsigned int`.
  std::variant<
      int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
      int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
      int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
      int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
      int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
      int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
      int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
      int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
      int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
      int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
      int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
      int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
      int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
      int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
      int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
      int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
      int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
      int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
      int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
      int, int, int, int, int, int, int, int, int, int, int, int, int, int, int,
      S>
      v_300_types_valueless;

  std::variant<int, bool, std::string> v4 = 4;

  v_valueless = 5;
  v_300_types_valueless.emplace<0>(10);

  v1 = 12; // v contains int
  v1_typedef = v1;
  v_v1 = v1;
  int i = std::get<int>(v1);
  printf("%d\n", i); // break here

  v2 = 2.0;
  double d = std::get<double>(v2);
  printf("%f\n", d);

  v3 = 'A';
  char c = std::get<char>(v3);
  printf("%d\n", c);

  // Checking v1 above and here to make sure we done maintain the incorrect
  // state when we change its value.
  v1 = 2.0;
  d = std::get<double>(v1);

  v4 = "a string";

  printf("%f\n", d); // break here

  try {
    // Exception in type-changing move-assignment is guaranteed to put
    // std::variant into a valueless state.
    v_valueless = S();
  } catch (...) {
  }

  printf("%d\n", v_valueless.valueless_by_exception());

  try {
    // Exception in move-assignment is guaranteed to put std::variant into a
    // valueless state.
    v_300_types_valueless = S();
  } catch (...) {
  }

  printf("%d\n", v_300_types_valueless.valueless_by_exception());

  return 0; // break here
}
