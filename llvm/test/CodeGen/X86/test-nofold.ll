; RUN: llvm-as < %s | llc -march=x86 -mcpu=yonah | grep {testl.*%e.x.*%e.x}
; rdar://5752025

; We don't want to fold the and into the test, because the and clobbers its
; input forcing a copy.  We want:
;	movl	$15, %ecx
;	andl	4(%esp), %ecx
;	testl	%ecx, %ecx
;	movl	$42, %eax
;	cmove	%ecx, %eax
;	ret
;
; Not:
;	movl	4(%esp), %eax
;	movl	%eax, %ecx
;	andl	$15, %ecx
;	testl	$15, %eax
;	movl	$42, %eax
;	cmove	%ecx, %eax
;	ret

define i32 @t1(i32 %X) nounwind  {
entry:
	%tmp2 = and i32 %X, 15		; <i32> [#uses=2]
	%tmp4 = icmp eq i32 %tmp2, 0		; <i1> [#uses=1]
	%retval = select i1 %tmp4, i32 %tmp2, i32 42		; <i32> [#uses=1]
	ret i32 %retval
}

