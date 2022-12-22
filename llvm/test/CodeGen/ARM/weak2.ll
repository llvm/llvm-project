; RUN: llc -mtriple=arm-eabi %s -o - | FileCheck %s

define i32 @f(i32 %a) {
entry:
	%tmp2 = icmp eq i32 %a, 0		; <i1> [#uses=1]
	%t.0 = select i1 %tmp2, ptr null, ptr @test_weak		; <ptr> [#uses=2]
	%tmp5 = icmp eq ptr %t.0, null		; <i1> [#uses=1]
	br i1 %tmp5, label %UnifiedReturnBlock, label %cond_true8

cond_true8:		; preds = %entry
	%tmp10 = tail call i32 (...) %t.0( )		; <i32> [#uses=1]
	ret i32 %tmp10

UnifiedReturnBlock:		; preds = %entry
	ret i32 250
}

declare extern_weak i32 @test_weak(...)

; CHECK: {{.}}weak

