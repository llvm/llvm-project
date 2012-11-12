// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

// Test that this produces an undefined reference to .Lfoo

        je	.Lfoo

// CHECK: ('_symbols', [
// CHECK:      (('st_name', 0x00000001) # '.Lfoo'
// CHECK-NEXT:  ('st_bind', 0x1)
// CHECK: (('sh_name', 0x0000001b) # '.strtab'
