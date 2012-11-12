; RUN: llc -march=msp430 < %s | FileCheck %s
target datalayout = "e-p:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:16:32"
target triple = "msp430-generic-generic"

define i16 @sccweqand(i16 %a, i16 %b) nounwind {
	%t1 = and i16 %a, %b
	%t2 = icmp eq i16 %t1, 0
	%t3 = zext i1 %t2 to i16
	ret i16 %t3
}
; CHECK: sccweqand:
; CHECK:	bit.w	r14, r15
; CHECK:	mov.w	r2, r15
; CHECK:	rra.w   r15
; CHECK:	and.w	#1, r15

define i16 @sccwneand(i16 %a, i16 %b) nounwind {
	%t1 = and i16 %a, %b
	%t2 = icmp ne i16 %t1, 0
	%t3 = zext i1 %t2 to i16
	ret i16 %t3
}
; CHECK: sccwneand:
; CHECK: 	bit.w	r14, r15
; CHECK:	mov.w	r2, r15
; CHECK:	and.w	#1, r15

define i16 @sccwne(i16 %a, i16 %b) nounwind {
	%t1 = icmp ne i16 %a, %b
	%t2 = zext i1 %t1 to i16
	ret i16 %t2
}
; CHECK:sccwne:
; CHECK:	cmp.w	r14, r15
; CHECK:	mov.w	r2, r15
; CHECK:	rra.w	r15
; CHECK:	and.w	#1, r15
; CHECK:	xor.w   #1, r15

define i16 @sccweq(i16 %a, i16 %b) nounwind {
	%t1 = icmp eq i16 %a, %b
	%t2 = zext i1 %t1 to i16
	ret i16 %t2
}
; CHECK:sccweq:
; CHECK:	cmp.w	r14, r15
; CHECK:	mov.w	r2, r15
; CHECK:	rra.w	r15
; CHECK:	and.w	#1, r15

define i16 @sccwugt(i16 %a, i16 %b) nounwind {
	%t1 = icmp ugt i16 %a, %b
	%t2 = zext i1 %t1 to i16
	ret i16 %t2
}
; CHECK:sccwugt:
; CHECK:	cmp.w	r15, r14
; CHECK:	mov.w	r2, r15
; CHECK:	and.w	#1, r15
; CHECK:	xor.w	#1, r15

define i16 @sccwuge(i16 %a, i16 %b) nounwind {
	%t1 = icmp uge i16 %a, %b
	%t2 = zext i1 %t1 to i16
	ret i16 %t2
}
; CHECK:sccwuge:
; CHECK:	cmp.w	r14, r15
; CHECK:	mov.w	r2, r15
; CHECK:	and.w	#1, r15

define i16 @sccwult(i16 %a, i16 %b) nounwind {
	%t1 = icmp ult i16 %a, %b
	%t2 = zext i1 %t1 to i16
	ret i16 %t2
}
; CHECK:sccwult:
; CHECK:	cmp.w	r14, r15
; CHECK:	mov.w	r2, r15
; CHECK:	and.w	#1, r15
; CHECK:	xor.w	#1, r15

define i16 @sccwule(i16 %a, i16 %b) nounwind {
	%t1 = icmp ule i16 %a, %b
	%t2 = zext i1 %t1 to i16
	ret i16 %t2
}
; CHECK:sccwule:
; CHECK:	cmp.w	r15, r14
; CHECK:	mov.w	r2, r15
; CHECK:	and.w	#1, r15

define i16 @sccwsgt(i16 %a, i16 %b) nounwind {
	%t1 = icmp sgt i16 %a, %b
	%t2 = zext i1 %t1 to i16
	ret i16 %t2
}

define i16 @sccwsge(i16 %a, i16 %b) nounwind {
	%t1 = icmp sge i16 %a, %b
	%t2 = zext i1 %t1 to i16
	ret i16 %t2
}

define i16 @sccwslt(i16 %a, i16 %b) nounwind {
	%t1 = icmp slt i16 %a, %b
	%t2 = zext i1 %t1 to i16
	ret i16 %t2
}

define i16 @sccwsle(i16 %a, i16 %b) nounwind {
	%t1 = icmp sle i16 %a, %b
	%t2 = zext i1 %t1 to i16
	ret i16 %t2
}

