; RUN: opt < %s -passes=indvars -S | FileCheck %s

; These tests ensure that we can compute the trip count of various forms of
; loops.  If the trip count of the loop is computable, then we will know what
; the exit value of the loop will be for some value, allowing us to substitute
; it directly into users outside of the loop, making the loop dead.

; CHECK-LABEL: @linear_setne(
; CHECK: ret i32 100

define i32 @linear_setne() {
entry:
	br label %loop

loop:		; preds = %loop, %entry
	%i = phi i32 [ 0, %entry ], [ %i.next, %loop ]		; <i32> [#uses=3]
	%i.next = add i32 %i, 1		; <i32> [#uses=1]
	%c = icmp ne i32 %i, 100		; <i1> [#uses=1]
	br i1 %c, label %loop, label %loopexit

loopexit:		; preds = %loop
	ret i32 %i
}

; CHECK-LABEL: @linear_setne_2(
; CHECK: ret i32 100

define i32 @linear_setne_2() {
entry:
	br label %loop

loop:		; preds = %loop, %entry
	%i = phi i32 [ 0, %entry ], [ %i.next, %loop ]		; <i32> [#uses=3]
	%i.next = add i32 %i, 2		; <i32> [#uses=1]
	%c = icmp ne i32 %i, 100		; <i1> [#uses=1]
	br i1 %c, label %loop, label %loopexit

loopexit:		; preds = %loop
	ret i32 %i
}

; CHECK-LABEL: @linear_setne_overflow(
; CHECK: ret i32 0

define i32 @linear_setne_overflow() {
entry:
	br label %loop

loop:		; preds = %loop, %entry
	%i = phi i32 [ 1024, %entry ], [ %i.next, %loop ]		; <i32> [#uses=3]
	%i.next = add i32 %i, 1024		; <i32> [#uses=1]
	%c = icmp ne i32 %i, 0		; <i1> [#uses=1]
	br i1 %c, label %loop, label %loopexit

loopexit:		; preds = %loop
	ret i32 %i
}

; CHECK-LABEL: @linear_setlt(
; CHECK: ret i32 100

define i32 @linear_setlt() {
entry:
	br label %loop

loop:		; preds = %loop, %entry
	%i = phi i32 [ 0, %entry ], [ %i.next, %loop ]		; <i32> [#uses=3]
	%i.next = add i32 %i, 1		; <i32> [#uses=1]
	%c = icmp slt i32 %i, 100		; <i1> [#uses=1]
	br i1 %c, label %loop, label %loopexit

loopexit:		; preds = %loop
	ret i32 %i
}

; CHECK-LABEL: @quadratic_setlt(
; CHECK: ret i32 34

define i32 @quadratic_setlt() {
entry:
	br label %loop

loop:		; preds = %loop, %entry
	%i = phi i32 [ 7, %entry ], [ %i.next, %loop ]		; <i32> [#uses=4]
	%i.next = add i32 %i, 3		; <i32> [#uses=1]
	%i2 = mul i32 %i, %i		; <i32> [#uses=1]
	%c = icmp slt i32 %i2, 1000		; <i1> [#uses=1]
	br i1 %c, label %loop, label %loopexit

loopexit:		; preds = %loop
	ret i32 %i
}

; CHECK-LABEL: @chained(
; CHECK: ret i32 200

define i32 @chained() {
entry:
	br label %loop

loop:		; preds = %loop, %entry
	%i = phi i32 [ 0, %entry ], [ %i.next, %loop ]		; <i32> [#uses=3]
	%i.next = add i32 %i, 1		; <i32> [#uses=1]
	%c = icmp ne i32 %i, 100		; <i1> [#uses=1]
	br i1 %c, label %loop, label %loopexit

loopexit:		; preds = %loop
	br label %loop2

loop2:		; preds = %loop2, %loopexit
	%j = phi i32 [ %i, %loopexit ], [ %j.next, %loop2 ]		; <i32> [#uses=3]
	%j.next = add i32 %j, 1		; <i32> [#uses=1]
	%c2 = icmp ne i32 %j, 200		; <i1> [#uses=1]
	br i1 %c2, label %loop2, label %loopexit2

loopexit2:		; preds = %loop2
	ret i32 %j
}

; CHECK-LABEL: @chained4(
; CHECK: ret i32 400

define i32 @chained4() {
entry:
  br label %loop

loop:                                             ; preds = %loop, %entry
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]  ; <i32> [#uses=3]
  %i.next = add i32 %i, 1                         ; <i32> [#uses=1]
  %c = icmp ne i32 %i.next, 100                   ; <i1> [#uses=1]
  br i1 %c, label %loop, label %loopexit

loopexit:                                         ; preds = %loop
  br label %loop2

loop2:                                            ; preds = %loop2, %loopexit
  %j = phi i32 [ %i.next, %loopexit ], [ %j.next, %loop2 ] ; <i32> [#uses=3]
  %j.next = add i32 %j, 1                         ; <i32> [#uses=1]
  %c2 = icmp ne i32 %j.next, 200                  ; <i1> [#uses=1]
  br i1 %c2, label %loop2, label %loopexit2

loopexit2:                                        ; preds = %loop
  br label %loop8

loop8:                                            ; preds = %loop2, %loopexit
  %k = phi i32 [ %j.next, %loopexit2 ], [ %k.next, %loop8 ] ; <i32> [#uses=3]
  %k.next = add i32 %k, 1                         ; <i32> [#uses=1]
  %c8 = icmp ne i32 %k.next, 300                  ; <i1> [#uses=1]
  br i1 %c8, label %loop8, label %loopexit8

loopexit8:                                        ; preds = %loop2
  br label %loop9

loop9:                                            ; preds = %loop2, %loopexit
  %l = phi i32 [ %k.next, %loopexit8 ], [ %l.next, %loop9 ] ; <i32> [#uses=3]
  %l.next = add i32 %l, 1                         ; <i32> [#uses=1]
  %c9 = icmp ne i32 %l.next, 400                  ; <i1> [#uses=1]
  br i1 %c9, label %loop9, label %loopexit9

loopexit9:                                        ; preds = %loop2
  ret i32 %l.next
}

; PR18449. Check that the early exit is reduced to never taken.
;
; CHECK-LABEL: @twoexit
; CHECK-LABEL: loop:
; CHECK: phi
; CHECK: br i1 false
; CHECK: br
; CHECK: ret
define void @twoexit() {
"function top level":
  br label %loop

loop:                                             ; preds = %body, %"function top level"
  %0 = phi i64 [ 0, %"function top level" ], [ %2, %body ]
  %1 = icmp ugt i64 %0, 2
  br i1 %1, label %fail, label %body

fail:                                             ; preds = %loop
  tail call void @bounds_fail()
  unreachable

body:                                             ; preds = %loop
  %2 = add i64 %0, 1
  %3 = icmp slt i64 %2, 3
  br i1 %3, label %loop, label %out

out:                                              ; preds = %body
  ret void
}
declare void @bounds_fail()
