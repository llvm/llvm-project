// RUN: dsymutil -f -y %p/dummy-debug-map.map -oso-prepend-path %p/../Inputs/inlined-static-variable -o - | llvm-dwarfdump - | FileCheck %s --implicit-check-not "{{DW_AT_low_pc|DW_AT_high_pc|DW_AT_location|DW_TAG|NULL}}"

// clang -g -c inlined-static-variable.cpp -o 4.o

// The functions removed and not_removed are not in the debug map and are
// considered dead, but they are also inlined into the function foo which is
// in the debug map. Those function-local globals are alive and thus should
// have locations in the debug info even if their functions do not.

inline __attribute__((always_inline)) int removed() {
  static int a = 0;
  return ++a;
}

__attribute__((always_inline)) int not_removed() {
  static int b = 0;
  return ++b;
}

int unused() {
  static int c = 0;
  return ++c;
}

int foo() {
  return removed() + not_removed();
}

// CHECK: DW_TAG_compile_unit
// CHECK:   DW_AT_low_pc
// CHECK:   DW_AT_high_pc

// CHECK:   DW_TAG_subprogram
// CHECK:     DW_AT_name	("removed")
// CHECK:     DW_TAG_variable
// CHECK:       DW_AT_name	("a")
// CHECK:       DW_AT_location
// CHECK:     NULL

// CHECK:   DW_TAG_base_type
// CHECK:   DW_TAG_subprogram
// CHECK:     DW_AT_name	("not_removed")
// CHECK:     DW_TAG_variable
// CHECK:       DW_AT_name	("b")
// CHECK:       DW_AT_location
// CHECK:     NULL

// CHECK:   DW_TAG_subprogram
// CHECK:     DW_AT_low_pc
// CHECK:     DW_AT_high_pc
// CHECK:     DW_AT_name	("foo")
// CHECK:     DW_TAG_inlined_subroutine
// CHECK:       DW_AT_low_pc
// CHECK:       DW_AT_high_pc
// CHECK:     DW_TAG_inlined_subroutine
// CHECK:       DW_AT_low_pc
// CHECK:       DW_AT_high_pc
// CHECK:     NULL
// CHECK:   NULL
