// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test that zed will be an ABS symbol

.Lfoo:
.Lbar:
        zed = .Lfoo - .Lbar

// CHECK:      # Symbol 1
// CHECK-NEXT: (('st_name', 0x00000001) # 'zed'
// CHECK-NEXT:  ('st_bind', 0x0)
// CHECK-NEXT:  ('st_type', 0x0)
// CHECK-NEXT:  ('st_other', 0x00)
// CHECK-NEXT:  ('st_shndx', 0xfff1)
// CHECK-NEXT:  ('st_value', 0x0000000000000000)
// CHECK-NEXT:  ('st_size', 0x0000000000000000)
