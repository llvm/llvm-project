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

// RUN: dsymutil --linker classic -f -oso-prepend-path=%p/../Inputs/modules \
// RUN:   -y %p/dummy-debug-map.map -o - \
// RUN:     | llvm-dwarfdump -v --debug-info - \
// RUN:     | FileCheck %s --check-prefixes=CHECK,CLASSIC

// RUN: dsymutil --linker classic -f -oso-prepend-path=%p/../Inputs/modules -y \
// RUN:   %p/dummy-debug-map.map -o %t 2>&1 | FileCheck --check-prefix=WARN %s
// RUN: dsymutil --linker parallel -f -oso-prepend-path=%p/../Inputs/modules -y \
// RUN:   %p/dummy-debug-map.map -o %t 2>&1 | FileCheck --check-prefix=WARN %s

// The classic linker drops DW_AT_GNU_dwo_id; the parallel linker keeps it.
// RUN: rm -rf %t.classic.dSYM
// RUN: dsymutil --linker classic --accelerator Dwarf -verify -f \
// RUN:   -oso-prepend-path=%p/../Inputs/modules \
// RUN:   -y %p/dummy-debug-map.map -o %t.classic.dSYM
// RUN: llvm-dwarfdump -v --debug-info %t.classic.dSYM \
// RUN:     | FileCheck --check-prefix=ACCEL %s
// RUN: rm -rf %t.parallel.dSYM
// RUN: dsymutil --linker parallel --accelerator Dwarf -verify -f \
// RUN:   -oso-prepend-path=%p/../Inputs/modules \
// RUN:   -y %p/dummy-debug-map.map -o %t.parallel.dSYM
// RUN: llvm-dwarfdump -v --debug-info %t.parallel.dSYM \
// RUN:     | FileCheck --check-prefix=ACCEL %s

// Foo imports Bar, so Foo.pcm carries a skeleton of Bar next to the module it
// describes. That import has to resolve to the unit built from Bar.pcm, which
// is the only one describing Bar in full. Both units offer the parallel linker
// an anchor for Bar, so the winner must not depend on which is cloned first.

// RUN: dsymutil --linker parallel -f -oso-prepend-path=%p/../Inputs/modules \
// RUN:   -y %p/dummy-debug-map.map -o %t.threaded
// RUN: dsymutil --linker parallel -f --num-threads 1 \
// RUN:   -oso-prepend-path=%p/../Inputs/modules \
// RUN:   -y %p/dummy-debug-map.map -o %t.serial
// RUN: cmp %t.threaded %t.serial
// RUN: llvm-dwarfdump --debug-info %t.threaded \
// RUN:     | FileCheck --check-prefix=MODIMPORT %s

// MODIMPORT:      0x0[[BAR:[0-9a-f]+]]: DW_TAG_module
// MODIMPORT-NEXT:   DW_AT_name {{.*}}"Bar"
// MODIMPORT:          DW_TAG_structure_type
// MODIMPORT-NEXT:       DW_AT_name {{.*}}"Bar"
// MODIMPORT:              DW_AT_name {{.*}}"value"
// MODIMPORT:          DW_TAG_structure_type
// MODIMPORT-NEXT:       DW_AT_name {{.*}}"PruneMeNot"

// MODIMPORT:      DW_TAG_module
// MODIMPORT-NEXT:   DW_AT_name {{.*}}"Foo"
// MODIMPORT:      DW_TAG_imported_declaration
// MODIMPORT-NOT:    DW_TAG
// MODIMPORT:        DW_AT_import {{.*}}(0x{{0*}}[[BAR]] "Bar")

// ACCEL: DW_TAG_compile_unit

// WARN-NOT: warning: hash mismatch

// ---------------------------------------------------------------------
#ifdef BAR_H
// ---------------------------------------------------------------------
// CHECK:            DW_TAG_compile_unit
// CLASSIC-NOT:        DW_AT_GNU_dwo_id
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
// CLASSIC-NOT:           DW_AT_GNU_dwo_id
// CHECK-NOT:             DW_TAG
// CHECK: 0x0[[FOO:.*]]:  DW_TAG_module
// CHECK-NEXT:              DW_AT_name{{.*}}"Foo"
// CHECK-NOT:               DW_TAG
// CHECK: 0x0[[BARTD:.*]]: DW_TAG_typedef
// CHECK-NOT:                 DW_TAG
// CHECK:                     DW_AT_type [DW_FORM_ref_addr] (0x{{0*}}[[BAR]]
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

// CHECK:    DW_TAG_compile_unit
// CLASSIC-NOT: DW_AT_GNU_dwo_id
// CHECK:      DW_AT_low_pc
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
// CHECK-NEXT:  DW_AT_type [DW_FORM_ref_addr] (0x{{0*}}[[INTERFACE]]
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

// CHECK:     DW_TAG_compile_unit
// CLASSIC-NOT: DW_AT_GNU_dwo_id
// CHECK:       DW_AT_name {{.*}}"odr_violation.c"
// CHECK: DW_TAG_variable
// CHECK:   DW_AT_name {{.*}}"odr_violation"
// CHECK:   DW_AT_type [DW_FORM_ref4] ({{.*}}{0x{{0*}}[[BAR2:.*]]}
// CHECK: 0x{{0*}}[[BAR2]]: DW_TAG_typedef
// CHECK:   DW_AT_type [DW_FORM_ref4] ({{.*}}{0x{{0*}}[[BAR3:.*]]}
// CHECK:   DW_AT_name {{.*}}"Bar"
// CHECK: 0x{{0*}}[[BAR3]]: DW_TAG_structure_type
// CHECK-NEXT:   DW_AT_name {{.*}}"Bar"
