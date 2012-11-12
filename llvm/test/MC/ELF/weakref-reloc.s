// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test that the relocations point to the correct symbols. We used to get the
// symbol index wrong for weakrefs when creating _GLOBAL_OFFSET_TABLE_.

	.weakref	bar,foo
        call    zed@PLT
	call	bar

// CHECK:      # Relocation 0
// CHECK-NEXT: (('r_offset', 0x0000000000000001)
// CHECK-NEXT:  ('r_sym', 0x00000006)
// CHECK-NEXT:  ('r_type', 0x00000004)
// CHECK-NEXT:  ('r_addend', 0xfffffffffffffffc)
// CHECK-NEXT: ),
// CHECK-NEXT: # Relocation 1
// CHECK-NEXT: (('r_offset', 0x0000000000000006)
// CHECK-NEXT:  ('r_sym', 0x00000005)
// CHECK-NEXT:  ('r_type', 0x00000002)
// CHECK-NEXT:  ('r_addend', 0xfffffffffffffffc)
// CHECK-NEXT: ),

// CHECK:      # Symbol 4
// CHECK-NEXT: (('st_name', 0x00000009) # '_GLOBAL_OFFSET_TABLE_'
// CHECK-NEXT:  ('st_bind', 0x1)
// CHECK-NEXT:  ('st_type', 0x0)
// CHECK-NEXT:  ('st_other', 0x00)
// CHECK-NEXT:  ('st_shndx', 0x0000)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 5
// CHECK-NEXT: (('st_name', 0x00000001) # 'foo'
// CHECK-NEXT:  ('st_bind', 0x2)
// CHECK-NEXT:  ('st_type', 0x0)
// CHECK-NEXT:  ('st_other', 0x00)
// CHECK-NEXT:  ('st_shndx', 0x0000)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
// CHECK-NEXT: # Symbol 6
// CHECK-NEXT: (('st_name', 0x00000005) # 'zed'
// CHECK-NEXT:  ('st_bind', 0x1)
// CHECK-NEXT:  ('st_type', 0x0)
// CHECK-NEXT:  ('st_other', 0x00)
// CHECK-NEXT:  ('st_shndx', 0x0000)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
// CHECK-NEXT: ),
