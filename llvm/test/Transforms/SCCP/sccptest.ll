; RUN: opt < %s -sccp -S | FileCheck %s

; This is a basic sanity check for constant propagation.  The add instruction 
; should be eliminated.

define i32 @test1(i1 %B) {
	br i1 %B, label %BB1, label %BB2
BB1:		; preds = %0
	%Val = add i32 0, 0		; <i32> [#uses=1]
	br label %BB3
BB2:		; preds = %0
	br label %BB3
BB3:		; preds = %BB2, %BB1
	%Ret = phi i32 [ %Val, %BB1 ], [ 1, %BB2 ]		; <i32> [#uses=1]
	ret i32 %Ret
        
; CHECK-LABEL: @test1(
; CHECK: %Ret = phi i32 [ 0, %BB1 ], [ 1, %BB2 ]
}

; This is the test case taken from appel's book that illustrates a hard case
; that SCCP gets right.
;
define i32 @test2(i32 %i0, i32 %j0) {
; CHECK-LABEL: @test2(
BB1:
	br label %BB2
BB2:
	%j2 = phi i32 [ %j4, %BB7 ], [ 1, %BB1 ]
	%k2 = phi i32 [ %k4, %BB7 ], [ 0, %BB1 ]
	%kcond = icmp slt i32 %k2, 100
	br i1 %kcond, label %BB3, label %BB4
BB3:
	%jcond = icmp slt i32 %j2, 20
	br i1 %jcond, label %BB5, label %BB6
; CHECK: BB3:
; CHECK-NEXT: br i1 true, label %BB5, label %BB6
BB4:
	ret i32 %j2
; CHECK: BB4:
; CHECK-NEXT: ret i32 1
BB5:
	%k3 = add i32 %k2, 1
	br label %BB7
BB6:
	%k5 = add i32 %k2, 1
	br label %BB7
; CHECK: BB6:
; CHECK-NEXT: br label %BB7
BB7:
	%j4 = phi i32 [ 1, %BB5 ], [ %k2, %BB6 ]
	%k4 = phi i32 [ %k3, %BB5 ], [ %k5, %BB6 ]
	br label %BB2
; CHECK: BB7:
; CHECK-NEXT: %k4 = phi i32 [ %k3, %BB5 ], [ undef, %BB6 ]
; CHECK-NEXT: br label %BB2
}

