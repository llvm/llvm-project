// RUN: llvm-mc -filetype=obj -triple i386-pc-linux-gnu %s -o - | elf-dump --dump-section-data | FileCheck  %s

.version "1234"
.version "123"

// CHECK:       (('sh_name', 0x0000000c) # '.note'
// CHECK-NEXT:   ('sh_type', 0x00000007)
// CHECK-NEXT:   ('sh_flags', 0x00000000)
// CHECK-NEXT:   ('sh_addr', 0x00000000)
// CHECK-NEXT:   ('sh_offset', 0x00000034)
// CHECK-NEXT:   ('sh_size', 0x00000024)
// CHECK-NEXT:   ('sh_link', 0x00000000)
// CHECK-NEXT:   ('sh_info', 0x00000000)
// CHECK-NEXT:   ('sh_addralign', 0x00000004)
// CHECK-NEXT:   ('sh_entsize', 0x00000000)
// CHECK-NEXT:   ('_section_data', '05000000 00000000 01000000 31323334 00000000 04000000 00000000 01000000 31323300')
// CHECK-NEXT:  ),
