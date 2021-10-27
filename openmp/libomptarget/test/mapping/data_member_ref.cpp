// RUN: %libomptarget-compilexx-run-and-check-generic

// amdgcn does not have printf definition
// XFAIL: amdgcn-amd-amdhsa
// XFAIL: amdgcn-amd-amdhsa-newRTL

#include <stdio.h>

struct View {
  int Data;
};

struct ViewPtr {
  int *Data;
};

template <typename T> struct Foo {
  Foo(T &V) : VRef(V) {}
  T &VRef;
};

int main() {
  View V;
  V.Data = 123456;
  Foo<View> Bar(V);
  ViewPtr V1;
  int Data = 123456;
  V1.Data = &Data;
  Foo<ViewPtr> Baz(V1);

  // CHECK: Host 123456.
  printf("Host %d.\n", Bar.VRef.Data);
#pragma omp target map(Bar.VRef)
  {
    // CHECK: Device 123456.
    printf("Device %d.\n", Bar.VRef.Data);
    V.Data = 654321;
    // CHECK: Device 654321.
    printf("Device %d.\n", Bar.VRef.Data);
  }
  // CHECK: Host 654321 654321.
  printf("Host %d %d.\n", Bar.VRef.Data, V.Data);
  V.Data = 123456;
  // CHECK: Host 123456.
  printf("Host %d.\n", Bar.VRef.Data);
#pragma omp target map(Bar) map(Bar.VRef)
  {
    // CHECK: Device 123456.
    printf("Device %d.\n", Bar.VRef.Data);
    V.Data = 654321;
    // CHECK: Device 654321.
    printf("Device %d.\n", Bar.VRef.Data);
  }
  // CHECK: Host 654321 654321.
  printf("Host %d %d.\n", Bar.VRef.Data, V.Data);
  // CHECK: Host 123456.
  printf("Host %d.\n", *Baz.VRef.Data);
#pragma omp target map(*Baz.VRef.Data)
  {
    // CHECK: Device 123456.
    printf("Device %d.\n", *Baz.VRef.Data);
    *V1.Data = 654321;
    // CHECK: Device 654321.
    printf("Device %d.\n", *Baz.VRef.Data);
  }
  // CHECK: Host 654321 654321 654321.
  printf("Host %d %d %d.\n", *Baz.VRef.Data, *V1.Data, Data);
  return 0;
}
