; RUN: llc < %s -mtriple=i686-- -mcpu=yonah | FileCheck %s
; rdar://5752025

; We want:
;      CHECK: movl	4(%esp), %ecx
; CHECK-NEXT: andl	$15, %ecx
; CHECK-NEXT: movl	$42, %eax
; CHECK-NEXT: cmovel	%ecx, %eax
; CHECK-NEXT: ret
;
; We don't want:
;	movl	4(%esp), %eax
;	movl	%eax, %ecx     # bad: extra copy
;	andl	$15, %ecx
;	testl	$15, %eax      # bad: peep obstructed
;	movl	$42, %eax
;	cmovel	%ecx, %eax
;	ret
;
; We also don't want:
;	movl	$15, %ecx      # bad: larger encoding
;	andl	4(%esp), %ecx
;	movl	$42, %eax
;	cmovel	%ecx, %eax
;	ret
;
; We also don't want:
;	movl	4(%esp), %ecx
;	andl	$15, %ecx
;	testl	%ecx, %ecx     # bad: unnecessary test
;	movl	$42, %eax
;	cmovel	%ecx, %eax
;	ret

define i32 @t1(i32 %X) nounwind  {
entry:
	%tmp2 = and i32 %X, 15		; <i32> [#uses=2]
	%tmp4 = icmp eq i32 %tmp2, 0		; <i1> [#uses=1]
	%retval = select i1 %tmp4, i32 %tmp2, i32 42		; <i32> [#uses=1]
	ret i32 %retval
}
