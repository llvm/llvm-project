// RUN: %libomptarget-compilexx-run-and-check-generic

#include <algorithm>

extern "C" int printf(const char *, ...);

// std::equal is lowered to libc function memcmp.
void test_memcpy() {
  int r = 0;
#pragma omp target map(from: r)
  {
    int x[2] = {0, 0};
    int y[2] = {0, 0};
    int z[2] = {0, 1};
    bool eq1 = std::equal(x, x + 2, y);
    bool eq2 = std::equal(x, x + 2, z);
    r = eq1 && !eq2;
  }
  printf("memcmp: %s\n", r ? "PASS" : "FAIL");
}

int main(int argc, char *argv[]) {
  test_memcpy();

  return 0;
}

// CHECK: memcmp: PASS
