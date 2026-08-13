#include <cstdint>
#include <limits>

void stop() {}

int main(int argc, char **argv) {
  bool trueVar = true;
  bool falseVar = false;
  bool &trueRef = trueVar;

  const char *p_ptr = "str";
  const char *p_nullptr = nullptr;

  int array[2] = {1, 2};

  struct S {
  } s;

  stop(); // Set a breakpoint here
  return 0;
}
