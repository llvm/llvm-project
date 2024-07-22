// clang-format off
// RUN: %libomptarget-compilexx-generic
// RUN: env OMPX_APU_MAPS=1 HSA_XNACK=1 LIBOMPTARGET_INFO=60 %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefix=CHECK

// REQUIRES: amdgpu
// REQUIRES: unified_shared_memory

// clang-format on

#include <cstdint>
#include <cstdio>

/// Test for globals under automatic zero-copy.
/// Because we are building without unified_shared_memory
/// requirement pragma, all globals are allocated in the device
/// memory of all used GPUs. To ensure those globals contain the intended
/// values, we need to execute H2D and D2H memory copies even if we are running
/// in automatic zero-copy. This only applies to globals. Local variables (their
/// host pointers) are passed to the kernels by-value, according to the
/// automatic zero-copy behavior.

#pragma omp begin declare target
int32_t x;     // 4 bytes
int32_t z[10]; // 40 bytes
int32_t *k;    // 20 bytes
#pragma omp end declare target

int main() {
  int32_t *dev_k = nullptr;
  x = 3;
  int32_t y = -1;
  for (size_t t = 0; t < 10; t++)
    z[t] = t;
  k = new int32_t[5];

  printf("Host pointer for k = %p\n", k);
  for (size_t t = 0; t < 5; t++)
    k[t] = -t;

/// target update to forces a copy between host and device global, which we must
/// execute to keep the two global copies consistent. CHECK: Copying data from
/// host to device, HstPtr={{.*}}, TgtPtr={{.*}}, Size=40, Name=z
#pragma omp target update to(z[ : 10])

/// target map with always modifier (for x) forces a copy between host and
/// device global, which we must execute to keep the two global copies
/// consistent. k's content (host address) is passed by-value to the kernel
/// (Size=20 case). y, being a local variable, is also passed by-value to the
/// kernel (Size=4 case) CHECK: Return HstPtrBegin {{.*}} Size=4 for unified
/// shared memory CHECK: Return HstPtrBegin {{.*}} Size=20 for unified shared
/// memory CHECK: Copying data from host to device, HstPtr={{.*}},
/// TgtPtr={{.*}}, Size=4, Name=x
#pragma omp target map(to : k[ : 5]) map(always, tofrom : x) map(tofrom : y)   \
    map(from : dev_k)
  {
    x++;
    y++;
    for (size_t t = 0; t < 10; t++)
      z[t]++;
    dev_k = k;
  }
/// CHECK-NOT: Copying data from device to host, TgtPtr={{.*}}, HstPtr={{.*}},
/// Size=20, Name=k

/// CHECK: Copying data from device to host, TgtPtr={{.*}}, HstPtr={{.*}},
/// Size=4, Name=x

/// CHECK: Copying data from device to host, TgtPtr={{.*}}, HstPtr={{.*}},
/// Size=40, Name=z
#pragma omp target update from(z[ : 10])

  /// CHECK-NOT: k pointer not correctly passed to kernel
  if (dev_k != k)
    printf("k pointer not correctly passed to kernel\n");

  delete[] k;
  return 0;
}
