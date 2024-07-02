// XFAIL: *

// RUN: %clangxx_host -gdwarf -o %t %s
// RUN: %lldb %t \
// RUN:   -o "expr alignof(OverlappingDerived)" \
// RUN:   -o "expr sizeof(OverlappingDerived)" \
// RUN:   -o exit | FileCheck %s

struct Empty {};

struct OverlappingBase {
  [[no_unique_address]] Empty e;
};
static_assert(sizeof(OverlappingBase) == 1);
static_assert(alignof(OverlappingBase) == 1);

struct Base {
  int mem;
};

struct OverlappingDerived : Base, OverlappingBase {};
static_assert(alignof(OverlappingDerived) == 4);
static_assert(sizeof(OverlappingDerived) == 4);

// CHECK:      (lldb) expr alignof(OverlappingDerived)
// CHECK-NEXT: ${{.*}} = 4
// CHECK:      (lldb) expr sizeof(OverlappingDerived)
// CHECK-NEXT: ${{.*}} = 4

int main() { OverlappingDerived d; }
