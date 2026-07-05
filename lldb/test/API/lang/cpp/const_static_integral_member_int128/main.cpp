#include <limits>

struct A {
  const static auto uint128_max = std::numeric_limits<__uint128_t>::max();
  const static auto uint128_min = std::numeric_limits<__uint128_t>::min();
  const static auto int128_max = std::numeric_limits<__int128_t>::max();
  const static auto int128_min = std::numeric_limits<__int128_t>::min();
};

int main() {
  A a;

  auto int128_max = A::int128_max;
  auto uint128_max = A::uint128_max;
  auto int128_min = A::int128_min;
  auto uint128_min = A::uint128_min;
  return 0; // break here
}
