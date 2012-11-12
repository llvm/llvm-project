// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump | FileCheck %s

// Test that the dwarf debug_line section contains no line directives.

        .file   1 "test.c"
        .globl  c
c:
        .asciz   "hi\n"

// CHECK:      # Section 4
// CHECK-NEXT: (('sh_name', 0x0000000c) # '.debug_line'
// CHECK-NEXT:  ('sh_type', 0x00000001)
// CHECK-NEXT:  ('sh_flags', 0x0000000000000000)
// CHECK-NEXT:  ('sh_addr', 0x0000000000000000)
// CHECK-NEXT:  ('sh_offset', 0x0000000000000044)
// CHECK-NEXT:  ('sh_size', 0x0000000000000027)
// CHECK-NEXT:  ('sh_link', 0x00000000)
// CHECK-NEXT:  ('sh_info', 0x00000000)
// CHECK-NEXT:  ('sh_addralign', 0x0000000000000001)
// CHECK-NEXT:  ('sh_entsize', 0x0000000000000000)
// CHECK-NEXT: ),
