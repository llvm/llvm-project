// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  --dump-section-data | FileCheck  %s

        .2byte 42, 1, 2, 3
        .4byte 42, 1, 2, 3
        .8byte 42, 1, 2, 3
        .int 42, 1, 2, 3

// CHECK:      # Section 1
// CHECK-NEXT: (('sh_name', 0x00000001) # '.text'
// CHECK-NEXT:  ('sh_type', 0x00000001)
// CHECK-NEXT:  ('sh_flags', 0x0000000000000006)
// CHECK-NEXT:  ('sh_addr', 0x0000000000000000)
// CHECK-NEXT:  ('sh_offset', 0x0000000000000040)
// CHECK-NEXT:  ('sh_size', 0x0000000000000048)
// CHECK-NEXT:  ('sh_link', 0x00000000)
// CHECK-NEXT:  ('sh_info', 0x00000000)
// CHECK-NEXT:  ('sh_addralign', 0x0000000000000004)
// CHECK-NEXT:  ('sh_entsize', 0x0000000000000000)
// CHECK-NEXT:  ('_section_data', '2a000100 02000300 2a000000 01000000 02000000 03000000 2a000000 00000000 01000000 00000000 02000000 00000000 03000000 00000000 2a000000 01000000 02000000 03000000')
// CHECK-NEXT: ),
