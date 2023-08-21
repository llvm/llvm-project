/* Compiled with: clang -arch=arm64 -O2 -g -c inlined_low_pc.c */

static int foo(int i) { return 42 + i; }
int bar(int a) { return foo(a); }

// RUN: dsymutil -f -y %p/dummy-debug-map-amr64.map -oso-prepend-path %p/../Inputs/inlined-low_pc -o - | llvm-dwarfdump - | FileCheck %s

// CHECK: DW_TAG_subprogram
// CHECK: DW_AT_low_pc{{.*}}0x0000000000010000
// CHECK: DW_AT_name{{.*}}"bar"
// CHECK-NOT: NULL
// CHECK: DW_TAG_inlined_subroutine
// CHECK-NEXT: DW_AT_abstract_origin{{.*}}"foo"
// CHECK-NEXT: DW_AT_low_pc{{.*}}0x0000000000010000

