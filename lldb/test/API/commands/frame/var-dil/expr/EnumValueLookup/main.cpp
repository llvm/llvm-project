#include <cstdint>

enum UnscopedEnum { kZero, kOne };
enum UnscopedEnumInt8 : int8_t { kZero8, kOne8 };
enum class ScopedEnum { kZeroS, kOneS };
namespace ns {
enum NSUnscopedEnum { kZeroNS, kOneNS };
}

void stop() {}
int main(int argc, char **argv) {
  auto enum_one = UnscopedEnum::kOne;
  auto enum_one_8 = UnscopedEnumInt8::kOne8;
  auto sc_enum_one = ScopedEnum::kOneS;
  auto ns_enum_one = ns::NSUnscopedEnum::kOneNS;

  stop(); // Set a breakpoint here
  return 0;
}
