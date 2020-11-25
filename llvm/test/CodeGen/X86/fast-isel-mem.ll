; RUN: llc < %s -fast-isel -mtriple=i386-apple-darwin -mcpu=generic | FileCheck %s
; RUN: llc < %s -fast-isel -mtriple=i386-apple-darwin -mcpu=atom | FileCheck -check-prefix=ATOM %s

@src = external global i32

; rdar://6653118
define i32 @loadgv() nounwind {
entry:
	%0 = load i32, i32* @src, align 4
	%1 = load i32, i32* @src, align 4
        %2 = add i32 %0, %1
        store i32 %2, i32* @src
	ret i32 %2
; This should fold one of the loads into the add.
; CHECK-LABEL: loadgv:
; CHECK: 	movl	L_src$non_lazy_ptr, %eax
; CHECK: 	movl	(%eax), %eax
; CHECK: 	movl	L_src$non_lazy_ptr, %ecx
; CHECK: 	addl	(%ecx), %eax
; CHECK: 	movl	L_src$non_lazy_ptr, %ecx
; CHECK: 	movl	%eax, (%ecx)
; CHECK: 	ret

; ATOM:	loadgv:
; ATOM:		movl    L_src$non_lazy_ptr, %eax
; ATOM:         movl    (%eax), %eax
; ATOM:		    movl    L_src$non_lazy_ptr, %ecx
; ATOM:         addl    (%ecx), %eax
; ATOM:		    movl    L_src$non_lazy_ptr, %ecx
; ATOM:         movl    %eax, (%ecx)
; ATOM:         ret

}

%stuff = type { i32 (...)** }
@LotsStuff = external constant [4 x i32 (...)*]

define void @t(%stuff* %this) nounwind {
entry:
	store i32 (...)** getelementptr ([4 x i32 (...)*], [4 x i32 (...)*]* @LotsStuff, i32 0, i32 2), i32 (...)*** null, align 4
	ret void
; CHECK: _t:
; CHECK:	xorl    %eax, %eax
; CHECK:	movl	L_LotsStuff$non_lazy_ptr, %ecx

; ATOM: _t:
; ATOM:         movl    L_LotsStuff$non_lazy_ptr, %e{{..}}
; ATOM:         xorl    %e{{..}}, %e{{..}}

}
