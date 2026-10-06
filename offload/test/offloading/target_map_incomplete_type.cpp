// Check that mapping a pointer to an incomplete type compiles and offloads
// correctly. The definition of the pointee must stay below the enter data
// directive so the type is still incomplete in directive codegen. That is
// what causes the assertion in vtable registration if the bug is present.

// RUN: %libomptarget-compilexx-run-and-check-generic

// REQUIRES: gpu

#include <cassert>
#include <cstdio>
#include <omp.h>

struct Incomplete;
Incomplete *p;

static void map_incomplete_pointer() {
#pragma omp target enter data map(alloc : p)
}

struct Incomplete {
  int a;
  int b;
};

int main() {
  constexpr int N = 5;
  auto *host = new Incomplete[N];
  for (int i = 0; i < N; ++i) {
    host[i].a = i + 100;
    host[i].b = i * 2 + 200;
  }
  p = host;

  map_incomplete_pointer();

  int isHost = 1;
#pragma omp target map(tofrom : p[0:N]) map(from : isHost)
  {
    isHost = omp_is_initial_device();
    for (int i = 0; i < N; ++i) {
      p[i].a += 1;
      p[i].b += 2;
    }
  }

#pragma omp target exit data map(delete : p)

  for (int i = 0; i < N; ++i) {
    assert(host[i].a == i + 101);
    assert(host[i].b == i * 2 + 202);
  }

  delete[] host;

  // CHECK: Target region executed on the device
  std::printf("Target region executed on the %s\n",
              isHost ? "host" : "device");

  if (isHost)
    return 1;

  // CHECK: PASS
  std::printf("PASS\n");
  return 0;
}
