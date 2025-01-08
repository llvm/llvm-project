// XFAIL: *

// RUN: %clang --target=x86_64-apple-macosx -c -gdwarf -o %t %s
// RUN: %lldb %t \
// RUN:   -o "expr alignof(OverlappingFields)" \
// RUN:   -o "expr sizeof(OverlappingFields)" \
// RUN:   -o exit | FileCheck %s

// CHECK:      (lldb) expr alignof(OverlappingFields)
// CHECK-NEXT: ${{.*}} = 4
// CHECK:      (lldb) expr sizeof(OverlappingFields)
// CHECK-NEXT: ${{.*}} = 8

struct Empty {};

struct OverlappingFields {
  char y;
  [[no_unique_address]] Empty e;
  int z;
} g_overlapping_struct;
static_assert(alignof(OverlappingFields) == 4);
static_assert(sizeof(OverlappingFields) == 8);

int main() {}
