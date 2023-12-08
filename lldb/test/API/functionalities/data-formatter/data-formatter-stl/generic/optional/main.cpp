#include <cstdio>
#include <string>
#include <vector>

// If we have libc++ 4.0 or greater we should have <optional>
// According to libc++ C++1z status page
// https://libcxx.llvm.org/cxx1z_status.html
#if !defined(_LIBCPP_VERSION) || _LIBCPP_VERSION >= 4000
#include <optional>
#define HAVE_OPTIONAL 1
#else
#define HAVE_OPTIONAL 0
#endif

int main() {
  bool has_optional = HAVE_OPTIONAL;

  printf("%d\n", has_optional); // break here

#if HAVE_OPTIONAL == 1
  using int_vect = std::vector<int>;
  using optional_int = std::optional<int>;
  using optional_int_vect = std::optional<int_vect>;
  using optional_string = std::optional<std::string>;

  optional_int number_not_engaged;
  optional_int number_engaged = 42;

  printf("%d\n", *number_engaged);

  optional_int_vect numbers{{1, 2, 3, 4}};

  printf("%d %d\n", numbers.value()[0], numbers.value()[1]);

  optional_string ostring = "hello";

  printf("%s\n", ostring->c_str());
#endif

  return 0; // break here
}
