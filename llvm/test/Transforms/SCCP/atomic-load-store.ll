; RUN: opt < %s -passes=ipsccp -S | FileCheck %s

; This transformation is safe for atomic loads and stores; check that it works.

@G = internal global i32 17
@C = internal constant i32 222

define i32 @test1() {
	%V = load atomic i32, i32* @G seq_cst, align 4
	%C = icmp eq i32 %V, 17
	br i1 %C, label %T, label %F
T:
	store atomic i32 17, i32* @G seq_cst, align 4
	ret i32 %V
F:	
	store atomic i32 123, i32* @G seq_cst, align 4
	ret i32 0
}
; CHECK-LABEL: define i32 @test1(
; CHECK-NOT: store
; CHECK: ret i32 17

define i32 @test2() {
	%V = load atomic i32, i32* @C seq_cst, align 4
	ret i32 %V
}

; CHECK-LABEL: define i32 @test2(
; CHECK-NOT: load
; CHECK: ret i32 222
