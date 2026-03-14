; RUN: llc < %s | FileCheck %s
; rdar://5671654
; The loads should fold into the testl instructions, no matter how
; the inputs are commuted.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin7"

define i32 @test(ptr %P, ptr %G) nounwind {
; CHECK-LABEL: test:
; CHECK-NOT: ret
; CHECK: testl %{{.*}}, (%{{.*}})
; CHECK: ret

entry:
	%0 = load i32, ptr %P, align 4		; <i32> [#uses=3]
	%1 = load i32, ptr %G, align 4		; <i32> [#uses=1]
	%2 = and i32 %1, %0		; <i32> [#uses=1]
	%3 = icmp eq i32 %2, 0		; <i1> [#uses=1]
	br i1 %3, label %bb1, label %bb

bb:		; preds = %entry
	%4 = tail call i32 @bar() nounwind		; <i32> [#uses=0]
	ret i32 %0

bb1:		; preds = %entry
	ret i32 %0
}

define i32 @test2(ptr %P, ptr %G) nounwind {
; CHECK-LABEL: test2:
; CHECK-NOT: ret
; CHECK: testl %{{.*}}, (%{{.*}})
; CHECK: ret

entry:
	%0 = load i32, ptr %P, align 4		; <i32> [#uses=3]
	%1 = load i32, ptr %G, align 4		; <i32> [#uses=1]
	%2 = and i32 %0, %1		; <i32> [#uses=1]
	%3 = icmp eq i32 %2, 0		; <i1> [#uses=1]
	br i1 %3, label %bb1, label %bb

bb:		; preds = %entry
	%4 = tail call i32 @bar() nounwind		; <i32> [#uses=0]
	ret i32 %0

bb1:		; preds = %entry
	ret i32 %0
}

define i32 @test3(ptr %P, ptr %G) nounwind {
; CHECK-LABEL: test3:
; CHECK-NOT: ret
; CHECK: testl %{{.*}}, (%{{.*}})
; CHECK: ret

entry:
	%0 = load i32, ptr %P, align 4		; <i32> [#uses=3]
	%1 = load i32, ptr %G, align 4		; <i32> [#uses=1]
	%2 = and i32 %0, %1		; <i32> [#uses=1]
	%3 = icmp eq i32 %2, 0		; <i1> [#uses=1]
	br i1 %3, label %bb1, label %bb

bb:		; preds = %entry
	%4 = tail call i32 @bar() nounwind		; <i32> [#uses=0]
	ret i32 %1

bb1:		; preds = %entry
	ret i32 %1
}

declare i32 @bar()
