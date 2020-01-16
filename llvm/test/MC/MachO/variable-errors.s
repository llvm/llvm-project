// RUN: not --crash llvm-mc -triple x86_64-apple-darwin10 %s -filetype=obj -o %t.o 2> %t.err
// RUN: FileCheck < %t.err %s

        .data
t0_a:
t0_x = t0_a - t0_b
// CHECK: unable to evaluate offset to undefined symbol 't0_b'
	.long	t0_x
