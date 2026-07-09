#include <cstddef>
#include <cstdint>

enum class ScopedEnum { kZeroS, kOneS };
enum class ScopedEnumInt8 : int8_t { kZeroS8, kOneS8 };

void stop() {}

int main(int argc, char **argv) {
  auto enum_one = ScopedEnum::kOneS;
  auto enum_one_8 = ScopedEnumInt8::kOneS8;

  bool trueVar = true;

  int i = 1;
  int j = 2;
  int &iref = i;
  int array[2] = {1, 2};

  struct S {
  } s;

  int *p_int0 = &array[0];
  int *p_int1 = &array[1];
  const char *p_char1 = "hello";
  void *p_void = (void *)p_char1;
  void **pp_void0 = &p_void;
  void **pp_void1 = pp_void0 + 1;

  std::nullptr_t std_nullptr_t = nullptr;

  stop(); // Set a breakpoint here
  return 0;
}
