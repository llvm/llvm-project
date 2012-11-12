// RUN: llvm-mc -mc-no-exec-stack -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck  %s

// CHECK:       # Section 4
// CHECK-NEXT:  (('sh_name', 0x0000000c) # '.note.GNU-stack'
// CHECK-NEXT:   ('sh_type', 0x00000001)
// CHECK-NEXT:   ('sh_flags', 0x0000000000000000)
// CHECK-NEXT:   ('sh_addr', 0x0000000000000000)
// CHECK-NEXT:   ('sh_offset', 0x0000000000000040)
// CHECK-NEXT:   ('sh_size', 0x0000000000000000)
// CHECK-NEXT:   ('sh_link', 0x00000000)
// CHECK-NEXT:   ('sh_info', 0x00000000)
// CHECK-NEXT:   ('sh_addralign', 0x0000000000000001)
// CHECK-NEXT:   ('sh_entsize', 0x0000000000000000)
// CHECK-NEXT:  ),

// CHECK:       # Symbol 4
// CHECK-NEXT:  (('st_name', 0x00000000) # ''
// CHECK-NEXT:   ('st_bind', 0x0)
// CHECK-NEXT:   ('st_type', 0x3)
// CHECK-NEXT:   ('st_other', 0x00)
// CHECK-NEXT:   ('st_shndx', 0x0004)
// CHECK-NEXT:   ('st_value', 0x0000000000000000)
// CHECK-NEXT:   ('st_size', 0x0000000000000000)
// CHECK-NEXT:  ),
