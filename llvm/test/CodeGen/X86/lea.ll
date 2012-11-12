; RUN: llc < %s -mtriple=x86_64-linux | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-win32 | FileCheck %s

define i32 @test1(i32 %x) nounwind {
        %tmp1 = shl i32 %x, 3
        %tmp2 = add i32 %tmp1, 7
        ret i32 %tmp2
; CHECK: test1:
; CHECK:    leal 7(,[[A0:%rdi|%rcx]],8), %eax
}


; ISel the add of -4 with a neg and use an lea for the rest of the
; arithemtic.
define i32 @test2(i32 %x_offs) nounwind readnone {
entry:
	%t0 = icmp sgt i32 %x_offs, 4
	br i1 %t0, label %bb.nph, label %bb2

bb.nph:
	%tmp = add i32 %x_offs, -5
	%tmp6 = lshr i32 %tmp, 2
	%tmp7 = mul i32 %tmp6, -4
	%tmp8 = add i32 %tmp7, %x_offs
	%tmp9 = add i32 %tmp8, -4
	ret i32 %tmp9

bb2:
	ret i32 %x_offs
; CHECK: test2:
; CHECK:	leal	-5([[A0]]), %eax
; CHECK:	andl	$-4, %eax
; CHECK:	negl	%eax
; CHECK:	leal	-4([[A0]],%rax), %eax
}
