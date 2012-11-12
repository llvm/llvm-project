// RUN: llvm-mc -g -triple i386-apple-darwin10 %s -filetype=obj -o %t
// RUN: llvm-dwarfdump %t | FileCheck %s

# 100 "t.s" 1
.globl _bar
_bar:
	movl	$0, %eax
L1:	leave
	ret

// rdar://9275556

// We check that the source name "t.s" is picked up
// CHECK:                 Dir  Mod Time   File Len   File Name
// CHECK:                 ---- ---------- ---------- ---------------------------
// CHECK: file_names[  1]    1 0x00000000 0x00000000 gen-dwarf-cpp.s
// CHECK: file_names[  2]    0 0x00000000 0x00000000 t.s

// We check that the source line number 100 is picked up before the "movl"
// CHECK: Address            Line   Column File   ISA Flags
// CHECK: ------------------ ------ ------ ------ --- -------------
// CHECK: 0x0000000000000000    102      0      2   0  is_stmt
