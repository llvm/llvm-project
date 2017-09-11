/* Compile with:
   cat >modules.modulemap <<EOF
     module Foo {
       header "Foo.h"
       export *
     }
     module Bar {
       header "Bar.h"
       export *
     }
EOF
   clang -D BAR_H -E -o Bar.h modules.m
   clang -D FOO_H -E -o Foo.h modules.m
   clang -D ODR_VIOLATION_C -E -o odr_violation.c modules.m
   clang -c -fmodules -fmodule-map-file=modules.modulemap \
     -g -gmodules -fmodules-cache-path=. \
     -Xclang -fdisable-module-hash modules.m -o 1.o
   clang -c -g odr_violation.c -o 2.o
*/

// RUN: llvm-dsymutil -f -oso-prepend-path=%p/../Inputs/modules \
// RUN:   -y %p/dummy-debug-map.map -o - \
// RUN:     | llvm-dwarfdump --debug-info - | FileCheck %s

// RUN: llvm-dsymutil -f -oso-prepend-path=%p/../Inputs/modules -y \
// RUN:   %p/dummy-debug-map.map -o %t 2>&1 | FileCheck --check-prefix=WARN %s

// WARN-NOT: warning: hash mismatch

// ---------------------------------------------------------------------
#ifdef BAR_H
// ---------------------------------------------------------------------
// CHECK:            DW_TAG_compile_unit
// CHECK-NOT:        DW_TAG
// CHECK:              DW_TAG_module
// CHECK-NEXT:           DW_AT_name{{.*}}"Bar"
// CHECK: 0x0[[BAR:.*]]: DW_TAG_structure_type
// CHECK-NOT:              DW_TAG
// CHECK:                  DW_AT_name {{.*}}"Bar"
// CHECK-NOT:              DW_TAG
// CHECK:                  DW_TAG_member
// CHECK:                    DW_AT_name {{.*}}"value"
// CHECK:                DW_TAG_structure_type
// CHECK-NOT:              DW_TAG
// CHECK:                  DW_AT_name {{.*}}"PruneMeNot"

struct Bar {
  int value;
};

struct PruneMeNot;

#else
// ---------------------------------------------------------------------
#ifdef FOO_H
// ---------------------------------------------------------------------
// CHECK:               DW_TAG_compile_unit
// CHECK-NOT:             DW_TAG
// CHECK: 0x0[[FOO:.*]]:  DW_TAG_module
// CHECK-NEXT:              DW_AT_name{{.*}}"Foo"
// CHECK-NOT:               DW_TAG
// CHECK: 0x0[[BARTD:.*]]: DW_TAG_typedef
// CHECK-NOT:                 DW_TAG
// CHECK:                     DW_AT_type [DW_FORM_ref_addr] (0x{{0*}}[[BAR]])
// CHECK:                   DW_TAG_structure_type
// CHECK-NEXT:                DW_AT_name{{.*}}"S"
// CHECK-NOT:                 DW_TAG
// CHECK: 0x0[[INTERFACE:.*]]: DW_TAG_structure_type
// CHECK-NEXT:                DW_AT_name{{.*}}"Foo"

@import Bar;
typedef struct Bar Bar;
struct S {};

@interface Foo {
  int ivar;
}
@end

#else
// ---------------------------------------------------------------------
#ifdef ODR_VIOLATION_C
// ---------------------------------------------------------------------

struct Bar {
  int i;
};
typedef struct Bar Bar;
Bar odr_violation = { 42 };

// ---------------------------------------------------------------------
#else
// ---------------------------------------------------------------------

// CHECK:  DW_TAG_compile_unit
// CHECK:    DW_AT_low_pc
// CHECK-NOT:  DW_TAG_module
// CHECK-NOT:  DW_TAG_typedef
//
// CHECK:   DW_TAG_imported_declaration
// CHECK-NOT: DW_TAG
// CHECK:     DW_AT_import [DW_FORM_ref_addr] (0x{{0*}}[[FOO]]
//
// CHECK:   DW_TAG_subprogram
// CHECK:     DW_AT_name {{.*}}"main"
//
// CHECK:     DW_TAG_variable
// CHECK-NOT:   DW_TAG
// CHECK:       DW_AT_name{{.*}}"bar"
// CHECK-NOT:   DW_TAG
// CHECK:       DW_AT_type [DW_FORM_ref_addr] (0x{{0*}}[[BARTD]]
// CHECK:     DW_TAG_variable
// CHECK-NOT:   DW_TAG
// CHECK:       DW_AT_name{{.*}}"foo"
// CHECK-NOT:   DW_TAG
// CHECK:       DW_AT_type {{.*}}{0x{{0*}}[[PTR:.*]]}
//
// CHECK: 0x{{0*}}[[PTR]]: DW_TAG_pointer_type
// CHECK-NEXT   DW_AT_type [DW_FORM_ref_addr] {0x{{0*}}[[INTERFACE]])
extern int odr_violation;

@import Foo;
int main(int argc, char **argv) {
  Bar bar;
  Foo *foo = 0;
  bar.value = odr_violation;
  return bar.value;
}
#endif
#endif
#endif

// CHECK: DW_TAG_compile_unit
// CHECK:   DW_AT_name {{.*}}"odr_violation.c"
// CHECK: DW_TAG_variable
// CHECK:   DW_AT_name {{.*}}"odr_violation"
// CHECK:   DW_AT_type [DW_FORM_ref4] ({{.*}}{0x{{0*}}[[BAR2:.*]]})
// CHECK: 0x{{0*}}[[BAR2]]: DW_TAG_typedef
// CHECK:   DW_AT_type [DW_FORM_ref4] ({{.*}}{0x{{0*}}[[BAR3:.*]]})
// CHECK:   DW_AT_name {{.*}}"Bar"
// CHECK: 0x{{0*}}[[BAR3]]: DW_TAG_structure_type
// CHECK-NEXT:   DW_AT_name {{.*}}"Bar"
