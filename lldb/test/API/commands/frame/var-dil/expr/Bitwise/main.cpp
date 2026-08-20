#include <cstdint>
enum UnscopedEnum { kZero, kOne };

int main(int argc, char **argv) {
  struct S {
  } s;

  auto enum_one = UnscopedEnum::kOne;
  uint64_t i64 = 1;

  return 0; // Set a breakpoint here
}
