; RUN: opt < %s -tailcallelim -S | FileCheck %s

define i32 @test1_factorial(i32 %x) {
entry:
	%tmp.1 = icmp sgt i32 %x, 0		; <i1> [#uses=1]
	br i1 %tmp.1, label %then, label %else
then:		; preds = %entry
	%tmp.6 = add i32 %x, -1		; <i32> [#uses=1]
	%tmp.4 = call i32 @test1_factorial( i32 %tmp.6 )		; <i32> [#uses=1]
	%tmp.7 = mul i32 %tmp.4, %x		; <i32> [#uses=1]
	ret i32 %tmp.7
else:		; preds = %entry
	ret i32 1
}

; CHECK: define i32 @test1_factorial
; CHECK: phi i32
; CHECK-NOT: call i32
; CHECK: else:

; This is a more aggressive form of accumulator recursion insertion, which 
; requires noticing that X doesn't change as we perform the tailcall.

define i32 @test2_mul(i32 %x, i32 %y) {
entry:
	%tmp.1 = icmp eq i32 %y, 0		; <i1> [#uses=1]
	br i1 %tmp.1, label %return, label %endif
endif:		; preds = %entry
	%tmp.8 = add i32 %y, -1		; <i32> [#uses=1]
	%tmp.5 = call i32 @test2_mul( i32 %x, i32 %tmp.8 )		; <i32> [#uses=1]
	%tmp.9 = add i32 %tmp.5, %x		; <i32> [#uses=1]
	ret i32 %tmp.9
return:		; preds = %entry
	ret i32 %x
}

; CHECK: define i32 @test2_mul
; CHECK: phi i32
; CHECK-NOT: call i32
; CHECK: return:


define i64 @test3_fib(i64 %n) nounwind readnone {
; CHECK: @test3_fib
entry:
; CHECK: tailrecurse:
; CHECK: %accumulator.tr = phi i64 [ %n, %entry ], [ %3, %bb1 ]
; CHECK: %n.tr = phi i64 [ %n, %entry ], [ %2, %bb1 ]
  switch i64 %n, label %bb1 [
; CHECK: switch i64 %n.tr, label %bb1 [
    i64 0, label %bb2
    i64 1, label %bb2
  ]

bb1:
; CHECK: bb1:
  %0 = add i64 %n, -1
; CHECK: %0 = add i64 %n.tr, -1
  %1 = tail call i64 @test3_fib(i64 %0) nounwind
; CHECK: %1 = tail call i64 @test3_fib(i64 %0)
  %2 = add i64 %n, -2
; CHECK: %2 = add i64 %n.tr, -2
  %3 = tail call i64 @test3_fib(i64 %2) nounwind
; CHECK-NOT: tail call i64 @test3_fib
  %4 = add nsw i64 %3, %1
; CHECK: add nsw i64 %accumulator.tr, %1
  ret i64 %4
; CHECK: br label %tailrecurse

bb2:
; CHECK: bb2:
  ret i64 %n
; CHECK: ret i64 %accumulator.tr
}
