// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump   | FileCheck %s

        .zero 4
foo:
        .zero 4
        .org foo+16

// CHECK:     (('sh_name', 0x00000001) # '.text'
// CHECK-NEXT: ('sh_type',
// CHECK-NEXT: ('sh_flags',
// CHECK-NEXT: ('sh_addr',
// CHECK-NEXT: ('sh_offset'
// CHECK-NEXT: ('sh_size', 0x0000000000000014)
