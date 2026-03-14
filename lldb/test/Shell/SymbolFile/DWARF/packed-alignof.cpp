// XFAIL: *
//
// RUN: %clangxx_host -gdwarf -o %t %s
// RUN: %lldb %t \
// RUN:   -o "expr alignof(base)" \
// RUN:   -o "expr alignof(packed_base)" \
// RUN:   -o "expr alignof(derived)" \
// RUN:   -o "expr sizeof(derived)" \
// RUN:   -o exit | FileCheck %s

struct __attribute__((packed)) packed {
  int x;
  char y;
  int z;
} g_packed_struct;

// LLDB incorrectly calculates alignof(base)
struct foo {};
struct base : foo { int x; };
static_assert(alignof(base) == 4);

// CHECK:      (lldb) expr alignof(base)
// CHECK-NEXT: ${{.*}} = 4

// LLDB incorrectly calculates alignof(packed_base)
struct __attribute__((packed)) packed_base { int x; };
static_assert(alignof(packed_base) == 1);

// CHECK:      (lldb) expr alignof(packed_base)
// CHECK-NEXT: ${{.*}} = 1

struct derived : packed, packed_base {
  short s;
} g_derived;
static_assert(alignof(derived) == 2);
static_assert(sizeof(derived) == 16);

// CHECK:      (lldb) expr alignof(derived)
// CHECK-NEXT: ${{.*}} = 2
// CHECK:      (lldb) expr sizeof(derived)
// CHECK-NEXT: ${{.*}} = 16

int main() {}
