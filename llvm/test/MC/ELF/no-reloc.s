// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -r - | FileCheck %s

// CHECK: Relocations [
// CHECK-NEXT: ]

	.section	.test1_foo
.Ltest1_1:
.Ltest1_2 = .Ltest1_1
	.section	.test1_bar
	.long .Ltest1_1-.Ltest1_2


        .section test2

.Ltest2_a:
.Ltest2_b = .Ltest2_a
.Ltest2_c:
.Ltest2_d = .Ltest2_c-.Ltest2_b
	.long	.Ltest2_d
