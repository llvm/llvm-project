// XFAIL: target-windows
// RUN: %clangxx_host -gdwarf -o %t %s
// RUN: %lldb %t \
// RUN:   -o "expr alignof(packed)" \
// RUN:   -o "expr sizeof(packed)" \
// RUN:   -o "expr alignof(packed_and_aligned)" \
// RUN:   -o "expr sizeof(packed_and_aligned)" \
// RUN:   -o "expr alignof(derived)" \
// RUN:   -o "expr sizeof(derived)" \
// RUN:   -o exit | FileCheck %s

struct __attribute__((packed)) packed {
  int x;
  char y;
  int z;
} g_packed_struct;
static_assert(alignof(packed) == 1);
static_assert(sizeof(packed) == 9);

// CHECK:      (lldb) expr alignof(packed)
// CHECK-NEXT: ${{.*}} = 1
// CHECK:      (lldb) expr sizeof(packed)
// CHECK-NEXT: ${{.*}} = 9

struct __attribute__((packed, aligned(16))) packed_and_aligned {
  int x;
  char y;
  int z;
} g_packed_and_aligned_struct;
static_assert(alignof(packed_and_aligned) == 16);
static_assert(sizeof(packed_and_aligned) == 16);

// CHECK:      (lldb) expr alignof(packed_and_aligned)
// CHECK-NEXT: ${{.*}} = 16
// CHECK:      (lldb) expr sizeof(packed_and_aligned)
// CHECK-NEXT: ${{.*}} = 16

struct __attribute__((packed)) packed_base { int x; };
static_assert(alignof(packed_base) == 1);

struct derived : packed, packed_base {} g_derived;
static_assert(alignof(derived) == 1);
static_assert(sizeof(derived) == 13);

// CHECK:      (lldb) expr alignof(derived)
// CHECK-NEXT: ${{.*}} = 1
// CHECK:      (lldb) expr sizeof(derived)
// CHECK-NEXT: ${{.*}} = 13

int main() {}
