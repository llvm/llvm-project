// REQUIRES: amdgpu
// RUN: %libomptarget-compile-run-and-check-generic

#include <cstdio>
#include <cstdlib>

struct vec {
  int len;
  int *data;
};

// Map the dynamic payload with tofrom semantics via a user-defined mapper.
#pragma omp declare mapper(default : vec v) map(tofrom : v.data [0:v.len])

int main() {
  vec s{};
  s.len = 16;
  s.data = (int *)malloc(sizeof(int) * s.len);
  for (int i = 0; i < s.len; ++i)
    s.data[i] = 1;

  // Offload with the mapper and update payload on device. Avoid reading s.len
  // on device; use a firstprivate copy of the length.
  int n = s.len;
  // Intentionally map the struct itself with 'alloc'. The mapper specifies
  // tofrom semantics for the payload. Without the fix that propagates mapper
  // to/from into ALLOC branches for components, the device writes would not
  // be copied back and this test would fail.
#pragma omp target map(mapper(default), alloc : s) firstprivate(n)
  {
    for (int i = 0; i < n; ++i)
      s.data[i] = 7;
  }

  long sum = 0;
  for (int i = 0; i < s.len; ++i)
    sum += s.data[i];

  if (sum == 7L * s.len) {
    std::printf("Test passed!\n");
  } else {
    std::printf("Test failed! sum=%ld\n", sum);
  }

  free(s.data);
  return 0;
}

// CHECK: Test passed!
