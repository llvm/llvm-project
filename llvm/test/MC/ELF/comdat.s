// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump   | FileCheck %s

// Test that we produce the group sections and that they are a the beginning
// of the file.

// CHECK:       # Section 1
// CHECK-NEXT:  (('sh_name', 0x0000001b) # '.group'
// CHECK-NEXT:   ('sh_type', 0x00000011)
// CHECK-NEXT:   ('sh_flags', 0x0000000000000000)
// CHECK-NEXT:   ('sh_addr', 0x0000000000000000)
// CHECK-NEXT:   ('sh_offset', 0x0000000000000040)
// CHECK-NEXT:   ('sh_size', 0x000000000000000c)
// CHECK-NEXT:   ('sh_link', 0x0000000d)
// CHECK-NEXT:   ('sh_info', 0x00000001)
// CHECK-NEXT:   ('sh_addralign', 0x0000000000000004)
// CHECK-NEXT:   ('sh_entsize', 0x0000000000000004)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Section 2
// CHECK-NEXT:  (('sh_name', 0x0000001b) # '.group'
// CHECK-NEXT:   ('sh_type', 0x00000011)
// CHECK-NEXT:   ('sh_flags', 0x0000000000000000)
// CHECK-NEXT:   ('sh_addr', 0x0000000000000000)
// CHECK-NEXT:   ('sh_offset', 0x000000000000004c)
// CHECK-NEXT:   ('sh_size', 0x0000000000000008)
// CHECK-NEXT:   ('sh_link', 0x0000000d)
// CHECK-NEXT:   ('sh_info', 0x00000002)
// CHECK-NEXT:   ('sh_addralign', 0x0000000000000004)
// CHECK-NEXT:   ('sh_entsize', 0x0000000000000004)
// CHECK-NEXT:  ),
// CHECK-NEXT:  # Section 3
// CHECK-NEXT:  (('sh_name', 0x0000001b) # '.group'
// CHECK-NEXT:   ('sh_type', 0x00000011)
// CHECK-NEXT:   ('sh_flags', 0x0000000000000000)
// CHECK-NEXT:   ('sh_addr', 0x0000000000000000)
// CHECK-NEXT:   ('sh_offset', 0x0000000000000054)
// CHECK-NEXT:   ('sh_size', 0x0000000000000008)
// CHECK-NEXT:   ('sh_link', 0x0000000d)
// CHECK-NEXT:   ('sh_info', 0x0000000d)
// CHECK-NEXT:   ('sh_addralign', 0x0000000000000004)
// CHECK-NEXT:   ('sh_entsize', 0x0000000000000004)
// CHECK-NEXT:  ),

// Test that g1 and g2 are local, but g3 is an undefined global.

// CHECK:      # Symbol 1
// CHECK-NEXT: (('st_name', 0x00000001) # 'g1'
// CHECK-NEXT:  ('st_bind', 0x0)
// CHECK-NEXT:  ('st_type', 0x0)
// CHECK-NEXT:  ('st_other', 0x00)
// CHECK-NEXT:  ('st_shndx', 0x0007)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 2
// CHECK-NEXT: (('st_name', 0x00000004) # 'g2'
// CHECK-NEXT:  ('st_bind', 0x0)
// CHECK-NEXT:  ('st_type', 0x0)
// CHECK-NEXT:  ('st_other', 0x00)
// CHECK-NEXT:  ('st_shndx', 0x0002)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),

// CHECK:      # Symbol 13
// CHECK-NEXT: (('st_name', 0x00000007) # 'g3'
// CHECK-NEXT:  ('st_bind', 0x1)
// CHECK-NEXT:  ('st_type', 0x0)
// CHECK-NEXT:  ('st_other', 0x00)
// CHECK-NEXT:  ('st_shndx', 0x0000)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),


	.section	.foo,"axG",@progbits,g1,comdat
g1:
        nop

        .section	.bar,"axG",@progbits,g1,comdat
        nop

        .section	.zed,"axG",@progbits,g2,comdat
        nop

        .section	.baz,"axG",@progbits,g3,comdat
        .long g3
