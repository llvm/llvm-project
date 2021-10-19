// RUN: %libomptarget-compilexx-run-and-check-aarch64-unknown-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-powerpc64-ibm-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-powerpc64le-ibm-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-x86_64-pc-linux-gnu
// RUN: %libomptarget-compilexx-run-and-check-nvptx64-nvidia-cuda

#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 2

class MyObjectA {
public:
  MyObjectA() {
    data1 = 1;
    data2 = 2;
  }
  void show() {
    printf("\t\tObject A Contents:\n");
    printf("\t\t\tdata1 = %d  data2 = %d\n", data1, data2);
  }
  void foo() {
    data1 += 10;
    data2 += 20;
  }
  int data1;
  int data2;
};

class MyObjectB {
public:
  MyObjectB() {
    arr = new MyObjectA[N];
    len = N;
  }
  void show() {
    printf("\tObject B Contents:\n");
    for (int i = 0; i < len; i++)
      arr[i].show();
  }
  void foo() {
    for (int i = 0; i < len; i++)
      arr[i].foo();
  }
  MyObjectA *arr;
  int len;
};
#pragma omp declare mapper(MyObjectB obj) map(obj, obj.arr[:obj.len])

class MyObjectC {
public:
  MyObjectC() {
    arr = new MyObjectB[N];
    len = N;
  }
  void show() {
    printf("Object C Contents:\n");
    for (int i = 0; i < len; i++)
      arr[i].show();
  }
  void foo() {
    for (int i = 0; i < len; i++)
      arr[i].foo();
  }
  MyObjectB *arr;
  int len;
};
#pragma omp declare mapper(MyObjectC obj) map(obj, obj.arr[:obj.len])

int main(void) {
  MyObjectC *outer = new MyObjectC[N];

  printf("Original data hierarchy:\n");
  for (int i = 0; i < N; i++)
    outer[i].show();

  printf("Sending data to device...\n");
#pragma omp target enter data map(to : outer[:N])

  printf("Calling foo()...\n");
#pragma omp target teams distribute parallel for
  for (int i = 0; i < N; i++)
    outer[i].foo();

  printf("foo() complete!\n");

  printf("Sending data back to host...\n");
#pragma omp target exit data map(from : outer[:N])

  printf("Modified Data Hierarchy:\n");
  for (int i = 0; i < N; i++)
    outer[i].show();

  printf("Testing for correctness...\n");
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      for (int k = 0; k < N; ++k) {
        printf("outer[%d].arr[%d].arr[%d].data1 = %d.\n", i, j, k,
               outer[i].arr[j].arr[k].data1);
        printf("outer[%d].arr[%d].arr[%d].data2 = %d.\n", i, j, k,
               outer[i].arr[j].arr[k].data2);
        assert(outer[i].arr[j].arr[k].data1 == 11 &&
               outer[i].arr[j].arr[k].data2 == 22);
      }
  // CHECK: outer[0].arr[0].arr[0].data1 = 11.
  // CHECK: outer[0].arr[0].arr[0].data2 = 22.
  // CHECK: outer[0].arr[0].arr[1].data1 = 11.
  // CHECK: outer[0].arr[0].arr[1].data2 = 22.
  // CHECK: outer[0].arr[1].arr[0].data1 = 11.
  // CHECK: outer[0].arr[1].arr[0].data2 = 22.
  // CHECK: outer[0].arr[1].arr[1].data1 = 11.
  // CHECK: outer[0].arr[1].arr[1].data2 = 22.
  // CHECK: outer[1].arr[0].arr[0].data1 = 11.
  // CHECK: outer[1].arr[0].arr[0].data2 = 22.
  // CHECK: outer[1].arr[0].arr[1].data1 = 11.
  // CHECK: outer[1].arr[0].arr[1].data2 = 22.
  // CHECK: outer[1].arr[1].arr[0].data1 = 11.
  // CHECK: outer[1].arr[1].arr[0].data2 = 22.
  // CHECK: outer[1].arr[1].arr[1].data1 = 11.
  // CHECK: outer[1].arr[1].arr[1].data2 = 22.
  assert(outer[1].arr[1].arr[0].data1 == 11 &&
         outer[1].arr[1].arr[0].data2 == 22 &&
         outer[1].arr[1].arr[1].data1 == 11 &&
         outer[1].arr[1].arr[1].data2 == 22);

  return 0;
}
