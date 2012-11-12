// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | elf-dump  | FileCheck %s

	.weakref	bar,foo
	call	bar@PLT

// CHECK:      # Symbol 5
// CHECK-NEXT: (('st_name', 0x00000001) # 'foo'
// CHECK-NEXT:  ('st_bind', 0x2)
