// Dumping a type in `target modules dump symfile` can complete it, and
// completing a type can parse more types and add them to the type list that is
// being dumped. Check that every type in the list is still dumped.

// RUN: %clangxx --target=x86_64-pc-linux -g -fstandalone-debug -c %s -o %t
// RUN: %lldb %t -b -o "image lookup -t Holder" -o "target modules dump symfile" | FileCheck %s

// `image lookup -t Holder` completes Holder, which puts Holder, Big *, Big, A,
// int, B and C in the type list. Big is only used through a pointer, so it
// stays a forward declaration. Dumping Big completes it, which adds its 300
// member types to the list while the list is being dumped.

// CHECK-LABEL: Types:
// CHECK-DAG: name = "A"
// CHECK-DAG: name = "B"
// CHECK-DAG: name = "C"
// CHECK-LABEL: Compile units:

template <int N> struct S {
  int x;
};

#define M1(n) S<n> s##n;
#define M10(n) M1(n##0) M1(n##1) M1(n##2) M1(n##3) M1(n##4) M1(n##5) M1(n##6) M1(n##7) M1(n##8) M1(n##9)
#define M100(n) M10(n##0) M10(n##1) M10(n##2) M10(n##3) M10(n##4) M10(n##5) M10(n##6) M10(n##7) M10(n##8) M10(n##9)

struct Big {
  M100(1) M100(2) M100(3)
};

struct A {
  int a;
};
struct B {
  int b;
};
struct C {
  int c;
};
struct Holder {
  Big *big;
  A a;
  B b;
  C c;
};

Big g_big;
Holder g_holder = {&g_big, {1}, {2}, {3}};
