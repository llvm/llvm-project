#include <array>
#include <span>
#include <stdio.h>
#include <string>
#include <vector>

template <class T, size_t N>
void by_ref_and_ptr(std::span<T, N> &ref, std::span<T, N> *ptr) {
  printf("Stop here to check by ref");
  return;
}

int main() {
  std::array numbers = {1, 12, 123, 1234, 12345};

  using dynamic_string_span = std::span<std::string>;

  // Test span of ints

  //   Full view of numbers with static extent
  std::span numbers_span = numbers;

  printf("break here");

  by_ref_and_ptr(numbers_span, &numbers_span);

  // Test spans of strings
  std::vector<std::string> strings{"goofy", "is", "smart", "!!!"};
  strings.reserve(strings.size() + 1);

  //   Partial view of strings with dynamic extent
  dynamic_string_span strings_span{std::span{strings}.subspan(2)};

  auto strings_span_it = strings_span.begin();

  printf("break here");

  //   Vector size doesn't increase, span should
  //   print unchanged and the strings_span_it
  //   remains valid
  strings.emplace_back("???");

  printf("break here");

  // Now some empty spans
  std::span<int, 0> static_zero_span;
  std::span<int> dynamic_zero_span;

  // Multiple spans
  std::array span_arr{strings_span, strings_span};
  std::span<std::span<std::string>, 2> nested = span_arr;

  printf("break here");

  return 0; // break here
}
