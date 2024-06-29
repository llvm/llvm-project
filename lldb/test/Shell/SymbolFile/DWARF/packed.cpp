// RUN: %clangxx_host -gdwarf -o %t %s
// RUN: %lldb %t \
// RUN:   -o "expr alignof(packed)" \
// RUN:   -o "expr sizeof(packed)" \
// RUN:   -o "expr alignof(packed_and_aligned)" \
// RUN:   -o "expr sizeof(packed_and_aligned)" \
// RUN:   -o exit | FileCheck %s

// CHECK:      (lldb) expr alignof(packed)
// CHECK-NEXT: ${{.*}} = 1
// CHECK:      (lldb) expr sizeof(packed)
// CHECK-NEXT: ${{.*}} = 9

// CHECK:      (lldb) expr alignof(packed_and_aligned)
// CHECK-NEXT: ${{.*}} = 16
// CHECK:      (lldb) expr sizeof(packed_and_aligned)
// CHECK-NEXT: ${{.*}} = 16

struct __attribute__((packed)) packed {
  int x;
  char y;
  int z;
} g_packed_struct;
static_assert(alignof(packed) == 1);
static_assert(sizeof(packed) == 9);

struct __attribute__((packed, aligned(16))) packed_and_aligned {
  int x;
  char y;
  int z;
} g_packed_and_aligned_struct;
static_assert(alignof(packed_and_aligned) == 16);
static_assert(sizeof(packed_and_aligned) == 16);

int main() {}
