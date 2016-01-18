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
   clang -cc1 -emit-obj -fmodules -fmodule-map-file=modules.modulemap \
     -fmodule-format=obj -g -dwarf-ext-refs -fmodules-cache-path=. \
     -fdisable-module-hash modules.m -o 1.o
*/

// RUN: llvm-dsymutil -f -oso-prepend-path=%p/../Inputs/modules \
// RUN:   -y %p/dummy-debug-map.map -o - \
// RUN:     | llvm-dwarfdump --debug-dump=info - | FileCheck %s

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
// CHECK:                   DW_TAG_typedef

@import Bar;
typedef struct Bar Bar;
struct S {};

@interface Foo {
  int ivar;
}
@end

// ---------------------------------------------------------------------
#else
// ---------------------------------------------------------------------

// CHECK:   DW_TAG_compile_unit
// CHECK:     DW_TAG_module
// CHECK-NEXT:  DW_AT_name{{.*}}"Bar"
// CHECK:     DW_TAG_module
// CHECK-NEXT:  DW_AT_name{{.*}}"Foo"
// CHECK-NOT:   DW_TAG
// CHECK:       DW_TAG_typedef
// CHECK-NOT:     DW_TAG
// CHECK:         DW_AT_type [DW_FORM_ref_addr] (0x{{0*}}[[BAR]])
// CHECK: 0x0[[INTERFACE:.*]]: DW_TAG_structure_type
// CHECK-NOT:     DW_TAG
// CHECK:         DW_AT_name{{.*}}"Foo"

//
// CHECK:   DW_TAG_imported_declaration
// CHECK-NOT: DW_TAG
// CHECK:     DW_AT_import [DW_FORM_ref_addr] (0x{{0*}}[[FOO]]
//
// CHECK:   DW_TAG_subprogram
// CHECK:     DW_AT_name {{.*}}"main"
//
// CHECK:     DW_TAG_variable
// CHECK:     DW_TAG_variable
// CHECK-NOT:   DW_TAG
// CHECK:       DW_AT_name{{.*}}"foo"
// CHECK-NOT:   DW_TAG
// CHECK:       DW_AT_type {{.*}}{0x{{0*}}[[PTR:.*]]}
//
// CHECK: 0x{{0*}}[[PTR]]: DW_TAG_pointer_type
// CHECK-NEXT   DW_AT_type [DW_FORM_ref_addr] {0x{{0*}}[[INTERFACE]])

@import Foo;
int main(int argc, char **argv) {
  Bar bar;
  Foo *foo = 0;
  bar.value = 42;
  return bar.value;
}
#endif
#endif
