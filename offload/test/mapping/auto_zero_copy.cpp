// clang-format off
// RUN: %libomptarget-compilexx-generic
// RUN: env OMPX_APU_MAPS=1 HSA_XNACK=1 LIBOMPTARGET_INFO=30 %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefix=INFO_ZERO -check-prefix=CHECK

// RUN: %libomptarget-compilexx-generic
// RUN: env HSA_XNACK=0 LIBOMPTARGET_INFO=30 %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic -check-prefix=INFO_COPY -check-prefix=CHECK

// REQUIRES: amdgpu
// REQUIRES: unified_shared_memory

// clang-format on

#include <cstdio>

int main() {
  int n = 1024;

  // test various mapping types
  int *a = new int[n];
  int k = 3;
  int b[n];

  for (int i = 0; i < n; i++)
    b[i] = i;

    // clang-format off
  // INFO_ZERO: Return HstPtrBegin 0x{{.*}} Size=4096 for unified shared memory
  // INFO_ZERO: Return HstPtrBegin 0x{{.*}} Size=4096 for unified shared memory

  // INFO_COPY: Creating new map entry with HstPtrBase=0x{{.*}}, HstPtrBegin=0x{{.*}}, TgtAllocBegin=0x{{.*}}, TgtPtrBegin=0x{{.*}}, Size=4096,
  // INFO_COPY: Creating new map entry with HstPtrBase=0x{{.*}}, HstPtrBegin=0x{{.*}}, TgtAllocBegin=0x{{.*}}, TgtPtrBegin=0x{{.*}}, Size=4096,
  // INFO_COPY: Mapping exists with HstPtrBegin=0x{{.*}}, TgtPtrBegin=0x{{.*}}, Size=4096, DynRefCount=1 (update suppressed)
  // INFO_COPY: Mapping exists with HstPtrBegin=0x{{.*}}, TgtPtrBegin=0x{{.*}}, Size=4096, DynRefCount=1 (update suppressed)
// clang-format on
#pragma omp target teams distribute parallel for map(tofrom : a[ : n])         \
    map(to : b[ : n])
  for (int i = 0; i < n; i++)
    a[i] = i + b[i] + k;

  int err = 0;
  for (int i = 0; i < n; i++)
    if (a[i] != i + b[i] + k)
      err++;

  // CHECK: PASS
  if (err == 0)
    printf("PASS\n");
  return err;
}
