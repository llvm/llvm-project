// RUN: llvm-mc -triple x86_64-linux-gnu -filetype=obj %s | elf-dump | FileCheck %s

// Tests that relocation value fits in the provided size
// Original bug http://llvm.org/bugs/show_bug.cgi?id=10568

L: movq $(L + 2147483648),%rax


// CHECK:          Relocation 0
// CHECK-NEXT:     ('r_offset', 0x0000000000000003)
// CHECK-NEXT:     ('r_sym'
// CHECK-NEXT:     ('r_type', 0x0000000b)
// CHECK-NEXT:     ('r_addend', 0x0000000080000000
