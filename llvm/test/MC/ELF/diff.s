// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

        .global zed
foo:
        nop
bar:
        nop
zed:
        mov zed+(bar-foo), %eax

// CHECK:       # Relocation 0
// CHECK-NEXT:  (('r_offset', 0x0000000000000005)
// CHECK-NEXT:   ('r_sym', 0x00000006)
// CHECK-NEXT:   ('r_type', 0x0000000b)
// CHECK-NEXT:   ('r_addend', 0x0000000000000001)
