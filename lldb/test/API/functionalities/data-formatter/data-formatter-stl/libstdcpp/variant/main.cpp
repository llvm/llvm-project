#include <cstdio>
#include <string>
#include <variant>
#include <vector>

struct S {
  operator int() { throw 42; }
};

int main() {
  bool has_variant = true;

  printf("%d\n", has_variant); // break here

  std::variant<int, double, char> v1;
  std::variant<int, double, char> &v1_ref = v1;
  std::variant<int, double, char> v2;
  std::variant<int, double, char> v3;
  std::variant<std::variant<int, double, char>> v_v1;
  std::variant<int, double, char> v_no_value;
  // The next variant has many types, meaning the type index does not fit in
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
      int, int, int, int, int, int, int, int, int, int, int, int>
      v_many_types_no_value;

  v1 = 12; // v contains int
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
  printf("%f\n", d); // break here

  try {
    v_no_value.emplace<0>(S());
  } catch (...) {
  }

  printf("%zu\n", v_no_value.index());

  try {
    v_many_types_no_value.emplace<0>(S());
  } catch (...) {
  }

  printf("%zu\n", v_many_types_no_value.index());

  return 0; // break here
}
