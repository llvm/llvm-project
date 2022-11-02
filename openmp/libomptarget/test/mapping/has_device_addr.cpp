// RUN: %libomptarget-compilexx-generic -fopenmp-version=51
// RUN: %libomptarget-run-generic 2>&1 \
// RUN: | %fcheck-generic

#include <assert.h>
#include <iostream>
#include <omp.h>

struct view {
  const int size = 10;
  int *data_host;
  int *data_device;
  void foo() {
    std::size_t bytes = size * sizeof(int);
    const int host_id = omp_get_initial_device();
    const int device_id = omp_get_default_device();
    data_host = (int *)malloc(bytes);
    data_device = (int *)omp_target_alloc(bytes, device_id);
#pragma omp target teams distribute parallel for has_device_addr(data_device[0])
    for (int i = 0; i < size; ++i)
      data_device[i] = i;
    omp_target_memcpy(data_host, data_device, bytes, 0, 0, host_id, device_id);
    for (int i = 0; i < size; ++i)
      assert(data_host[i] == i);
  }
};

void poo() {
  short a = 1;
  short &ar = a;

#pragma omp target data map(tofrom : ar) use_device_addr(ar)
  {
#pragma omp target has_device_addr(ar)
    {
      ar = 222;
      // CHECK: 222
      printf("%hd %p\n", ar, &ar); // 222 p2
    }
  }
  // CHECK: 222
  printf("%hd %p\n", ar, &ar); // 222 p1
}

void noo() {
  short *b = (short *)malloc(sizeof(short));
  short *&br = b;
  br = br - 1;

  br[1] = 111;
#pragma omp target data map(tofrom : br[1]) use_device_addr(br[1])
#pragma omp target has_device_addr(br[1])
  {
    br[1] = 222;
    // CHECK: 222
    printf("%hd %p %p %p\n", br[1], br, &br[1], &br);
  }
  // CHECK: 222
  printf("%hd %p %p %p\n", br[1], br, &br[1], &br);
}

void ooo() {
  short a = 1;

#pragma omp target data map(tofrom : a) use_device_addr(a)
#pragma omp target has_device_addr(a)
  {
    a = 222;
    // CHECK: 222
    printf("%hd %p\n", a, &a);
  }
  // CHECK: 222
  printf("%hd %p\n", a, &a);
}

int main() {
  view a;
  a.foo();
  poo();
  noo();
  ooo();
  // CHECK: PASSED
  printf("PASSED\n");
}
