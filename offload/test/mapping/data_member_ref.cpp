// RUN: %libomptarget-compilexx-run-and-check-generic

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
  int D1, D2;

  // CHECK: Host 123456.
  printf("Host %d.\n", Bar.VRef.Data);
#pragma omp target map(Bar.VRef) map(from : D1, D2)
  {
    // CHECK: Device 123456.
    D1 = Bar.VRef.Data;
    printf("Device %d.\n", D1);
    V.Data = 654321;
    // CHECK: Device 654321.
    D2 = Bar.VRef.Data;
    printf("Device %d.\n", D2);
  }
  printf("Device %d.\n", D1);
  printf("Device %d.\n", D2);
  // CHECK: Host 654321 654321.
  printf("Host %d %d.\n", Bar.VRef.Data, V.Data);
  V.Data = 123456;
  // CHECK: Host 123456.
  printf("Host %d.\n", Bar.VRef.Data);
#pragma omp target map(Bar) map(Bar.VRef) map(from : D1, D2)
  {
    // CHECK: Device 123456.
    D1 = Bar.VRef.Data;
    printf("Device %d.\n", D1);
    V.Data = 654321;
    // CHECK: Device 654321.
    D2 = Bar.VRef.Data;
    printf("Device %d.\n", D2);
  }
  printf("Device %d.\n", D1);
  printf("Device %d.\n", D2);
  // CHECK: Host 654321 654321.
  printf("Host %d %d.\n", Bar.VRef.Data, V.Data);
  // CHECK: Host 123456.
  printf("Host %d.\n", *Baz.VRef.Data);
#pragma omp target map(Baz.VRef.Data) map(*Baz.VRef.Data) map(V1.Data[0 : 0])  \
    map(from : D1, D2)
  {
    // CHECK: Device 123456.
    D1 = *Baz.VRef.Data;
    printf("Device %d.\n", D1);
    *V1.Data = 654321;
    // CHECK: Device 654321.
    D2 = *Baz.VRef.Data;
    printf("Device %d.\n", D2);
  }
  printf("Device %d.\n", D1);
  printf("Device %d.\n", D2);
  // CHECK: Host 654321 654321 654321.
  printf("Host %d %d %d.\n", *Baz.VRef.Data, *V1.Data, Data);
  return 0;
}
