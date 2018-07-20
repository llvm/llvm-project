// RUN: llvm-mc -g -triple i386-apple-darwin10 %s -filetype=obj -o %t
// RUN: llvm-dwarfdump -debug-line %t | FileCheck %s

# 100 "t.s" 1
.globl _bar
_bar:
	movl	$0, %eax
# 3 "inc/g.s"
	movl	$0, %eax
L1:	leave
# 42 "t.s"
	ret

// rdar://9275556

// We check that the source name "t.s" is picked up
// CHECK: include_directories[  1] = "{{.*(/|\\\\)}}test{{(/|\\\\)}}MC{{(/|\\\\)}}MachO"
// CHECK: include_directories[  2] = "inc"
// CHECK: file_names[  1]:
// CHECK-NEXT: name: "gen-dwarf-cpp.s"
// CHECK-NEXT: dir_index: 1
// CHECK: file_names[  2]:
// CHECK-NEXT: name: "t.s"
// CHECK-NEXT: dir_index: 0
// CHECK: file_names[  3]:
// CHECK-NEXT: name: "g.s"
// CHECK-NEXT: dir_index: 2
// CHECK-NOT: file_names

// We check that the source line number 100 is picked up before the "movl"
// CHECK: Address            Line   Column File   ISA Discriminator Flags
// CHECK: ------------------ ------ ------ ------ --- ------------- -------------
// CHECK: 0x0000000000000000    102      0      2   0             0  is_stmt
