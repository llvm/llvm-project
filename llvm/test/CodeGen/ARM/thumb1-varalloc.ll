; RUN: llc < %s -mtriple=thumbv6-apple-darwin | FileCheck %s
; RUN: llc < %s -mtriple=thumbv6-apple-darwin -regalloc=basic | FileCheck %s
; rdar://8819685

@__bar = external hidden global i8*
@__baz = external hidden global i8*

define i8* @_foo() {
entry:
; CHECK: foo:

	%size = alloca i32, align 4
	%0 = load i8** @__bar, align 4
	%1 = icmp eq i8* %0, null
	br i1 %1, label %bb1, label %bb3
; CHECK: bne
		
bb1:
	store i32 1026, i32* %size, align 4
	%2 = alloca [1026 x i8], align 1
; CHECK: mov     [[R0:r[0-9]+]], sp
; CHECK: adds    {{r[0-9]+}}, [[R0]], {{r[0-9]+}}
	%3 = getelementptr inbounds [1026 x i8]* %2, i32 0, i32 0
	%4 = call i32 @_called_func(i8* %3, i32* %size) nounwind
	%5 = icmp eq i32 %4, 0
	br i1 %5, label %bb2, label %bb3
	
bb2:
	%6 = call i8* @strdup(i8* %3) nounwind
	store i8* %6, i8** @__baz, align 4
	br label %bb3
	
bb3:
	%.0 = phi i8* [ %0, %entry ], [ %6, %bb2 ], [ %3, %bb1 ]
; CHECK: subs    r4, #5
; CHECK-NEXT: mov     sp, r4
; CHECK-NEXT: pop     {r4, r5, r6, r7, pc}
	ret i8* %.0
}

declare noalias i8* @strdup(i8* nocapture) nounwind
declare i32 @_called_func(i8*, i32*) nounwind