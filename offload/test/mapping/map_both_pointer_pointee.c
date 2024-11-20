// RUN: %libomptarget-compilexx-run-and-check-aarch64-unknown-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-powerpc64-ibm-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-powerpc64le-ibm-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-x86_64-unknown-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-nvptx64-nvidia-cuda

// REQUIRES: unified_shared_memory
// UNSUPPORTED: amdgcn-amd-amdhsa

#pragma omp declare target
int *ptr1;
#pragma omp end declare target
int a[10];

#include <stdio.h>
#include <stdlib.h>
int main() {
  ptr1 = (int *)malloc(sizeof(int) * 100);
  int *ptr2;
  ptr2 = (int *)malloc(sizeof(int) * 100);
#pragma omp target map(ptr1, ptr1[ : 100])
  { ptr1[1] = 6; }
  // CHECK: 6
  printf(" %d \n", ptr1[1]);
#pragma omp target data map(ptr1[ : 5])
  {
#pragma omp target map(ptr1[2], ptr1, ptr1[3]) map(ptr2, ptr2[2])
    {
      ptr1[2] = 7;
      ptr1[3] = 9;
      ptr2[2] = 7;
    }
  }
  // CHECK: 7 7 9
  printf(" %d %d %d \n", ptr2[2], ptr1[2], ptr1[3]);
  free(ptr1);
#pragma omp target map(ptr2, ptr2[ : 100])
  { ptr2[1] = 6; }
  // CHECK: 6
  printf(" %d \n", ptr2[1]);
  free(ptr2);

  a[1] = 111;
  int *p = &a[0];
  // CHECK: 111
  printf("%d %p %p\n", p[1], p, &p); // 111 hst_p1 hst_p2
#pragma omp target data map(to : p[1 : 3]) map(p)
#pragma omp target data use_device_addr(p)
  {
#pragma omp target has_device_addr(p)
    {
      // CHECK: 111
      printf("%d %p %p\n", p[1], p, &p); // 111 dev_p1 dev_p2
      p[1] = 222;
      // CHECK: 222
      printf("%d %p %p\n", p[1], p, &p); // 222 dev_p1 dev_p2
    }
  }
  // CHECK: 111
  printf("%d %p %p\n", p[1], p, &p); // 111 hst_p1 hst_p2
  return 0;
}
