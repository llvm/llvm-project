; RUN: opt < %s -correlated-propagation -S | FileCheck %s
; PR2581

; CHECK-LABEL: @test1(
define i32 @test1(i1 %C) nounwind  {
        br i1 %C, label %exit, label %body

body:           ; preds = %0
; CHECK-NOT: select
        %A = select i1 %C, i32 10, i32 11               ; <i32> [#uses=1]
; CHECK: ret i32 11
        ret i32 %A

exit:           ; preds = %0
; CHECK: ret i32 10
        ret i32 10
}

; PR4420
declare i1 @ext()
; CHECK-LABEL: @test2(
define i1 @test2() {
entry:
        %cond = tail call i1 @ext()             ; <i1> [#uses=2]
        br i1 %cond, label %bb1, label %bb2

bb1:            ; preds = %entry
        %cond2 = tail call i1 @ext()            ; <i1> [#uses=1]
        br i1 %cond2, label %bb3, label %bb2

bb2:            ; preds = %bb1, %entry
; CHECK-NOT: phi i1
        %cond_merge = phi i1 [ %cond, %entry ], [ false, %bb1 ]         ; <i1> [#uses=1]
; CHECK: ret i1 false
        ret i1 %cond_merge

bb3:            ; preds = %bb1
        %res = tail call i1 @ext()              ; <i1> [#uses=1]
; CHECK: ret i1 %res
        ret i1 %res
}

; PR4855
@gv = internal constant i8 7
; CHECK-LABEL: @test3(
define i8 @test3(i8* %a) nounwind {
entry:
        %cond = icmp eq i8* %a, @gv
        br i1 %cond, label %bb2, label %bb

bb:             ; preds = %entry
        ret i8 0

bb2:            ; preds = %entry
; CHECK: %should_be_const = load i8, i8* @gv
        %should_be_const = load i8, i8* %a
        ret i8 %should_be_const
}

; PR1757
; CHECK-LABEL: @test4(
define i32 @test4(i32) {
EntryBlock:
; CHECK: icmp sgt i32 %0, 2  
  %.demorgan = icmp sgt i32 %0, 2    
  br i1 %.demorgan, label %GreaterThanTwo, label %LessThanOrEqualToTwo

GreaterThanTwo:
; CHECK-NOT: icmp eq i32 %0, 2
  icmp eq i32 %0, 2
; CHECK: br i1 false
  br i1 %1, label %Impossible, label %NotTwoAndGreaterThanTwo

NotTwoAndGreaterThanTwo:
  ret i32 2

Impossible:
  ret i32 1

LessThanOrEqualToTwo:
  ret i32 0
}

declare i32* @f(i32*)
define void @test5(i32* %x, i32* %y) {
; CHECK-LABEL: @test5(
entry:
  %pre = icmp eq i32* %x, null
  br i1 %pre, label %return, label %loop

loop:
  %phi = phi i32* [ %sel, %loop ], [ %x, %entry ]
; CHECK: %phi = phi i32* [ %f, %loop ], [ %x, %entry ]
  %f = tail call i32* @f(i32* %phi)
  %cmp1 = icmp ne i32* %f, %y
  %sel = select i1 %cmp1, i32* %f, i32* null
  %cmp2 = icmp eq i32* %sel, null
  br i1 %cmp2, label %return, label %loop

return:
  ret void
}

define i32 @switch1(i32 %s) {
; CHECK-LABEL: @switch1(
entry:
  %cmp = icmp slt i32 %s, 0
  br i1 %cmp, label %negative, label %out

negative:
  switch i32 %s, label %out [
; CHECK: switch i32 %s, label %out
    i32 0, label %out
; CHECK-NOT: i32 0
    i32 1, label %out
; CHECK-NOT: i32 1
    i32 -1, label %next
; CHECK: i32 -1, label %next
    i32 -2, label %next
; CHECK: i32 -2, label %next
    i32 2, label %out
; CHECK-NOT: i32 2
    i32 3, label %out
; CHECK-NOT: i32 3
  ]

out:
  %p = phi i32 [ 1, %entry ], [ -1, %negative ], [ -1, %negative ], [ -1, %negative ], [ -1, %negative ], [ -1, %negative ]
  ret i32 %p

next:
  %q = phi i32 [ 0, %negative ], [ 0, %negative ]
  ret i32 %q
}

define i32 @switch2(i32 %s) {
; CHECK-LABEL: @switch2(
entry:
  %cmp = icmp sgt i32 %s, 0
  br i1 %cmp, label %positive, label %out

positive:
  switch i32 %s, label %out [
    i32 0, label %out
    i32 -1, label %next
    i32 -2, label %next
  ]
; CHECK: br label %out

out:
  %p = phi i32 [ -1, %entry ], [ 1, %positive ], [ 1, %positive ]
  ret i32 %p

next:
  %q = phi i32 [ 0, %positive ], [ 0, %positive ]
  ret i32 %q
}

define i32 @switch3(i32 %s) {
; CHECK-LABEL: @switch3(
entry:
  %cmp = icmp sgt i32 %s, 0
  br i1 %cmp, label %positive, label %out

positive:
  switch i32 %s, label %out [
    i32 -1, label %out
    i32 -2, label %next
    i32 -3, label %next
  ]
; CHECK: br label %out

out:
  %p = phi i32 [ -1, %entry ], [ 1, %positive ], [ 1, %positive ]
  ret i32 %p

next:
  %q = phi i32 [ 0, %positive ], [ 0, %positive ]
  ret i32 %q
}

define void @switch4(i32 %s) {
; CHECK-LABEL: @switch4(
entry:
  %cmp = icmp eq i32 %s, 0
  br i1 %cmp, label %zero, label %out

zero:
  switch i32 %s, label %out [
    i32 0, label %next
    i32 1, label %out
    i32 -1, label %out
  ]
; CHECK: br label %next

out:
  ret void

next:
  ret void
}

define i1 @arg_attribute(i8* nonnull %a) {
; CHECK-LABEL: @arg_attribute(
; CHECK: ret i1 false
  %cmp = icmp eq i8* %a, null
  br label %exit

exit:
  ret i1 %cmp
}

declare nonnull i8* @return_nonnull()
define i1 @call_attribute() {
; CHECK-LABEL: @call_attribute(
; CHECK: ret i1 false
  %a = call i8* @return_nonnull()
  %cmp = icmp eq i8* %a, null
  br label %exit

exit:
  ret i1 %cmp
}

define i1 @umin(i32 %a, i32 %b) {
; CHECK-LABEL: @umin(
entry:
  %cmp = icmp ult i32 %a, 5
  br i1 %cmp, label %a_guard, label %out

a_guard:
  %cmp2 = icmp ult i32 %b, 20
  br i1 %cmp2, label %b_guard, label %out

b_guard:
  %sel_cmp = icmp ult i32 %a, %b
  %min = select i1 %sel_cmp, i32 %a, i32 %b
  %res = icmp eq i32 %min, 7
  br label %next
next:
; CHECK: next:
; CHECK: ret i1 false
  ret i1 %res
out:
  ret i1 false
}

define i1 @smin(i32 %a, i32 %b) {
; CHECK-LABEL: @smin(
entry:
  %cmp = icmp ult i32 %a, 5
  br i1 %cmp, label %a_guard, label %out

a_guard:
  %cmp2 = icmp ult i32 %b, 20
  br i1 %cmp2, label %b_guard, label %out

b_guard:
  %sel_cmp = icmp sle i32 %a, %b
  %min = select i1 %sel_cmp, i32 %a, i32 %b
  %res = icmp eq i32 %min, 7
  br label %next
next:
; CHECK: next:
; CHECK: ret i1 false
  ret i1 %res
out:
  ret i1 false
}

define i1 @smax(i32 %a, i32 %b) {
; CHECK-LABEL: @smax(
entry:
  %cmp = icmp sgt i32 %a, 5
  br i1 %cmp, label %a_guard, label %out

a_guard:
  %cmp2 = icmp sgt i32 %b, 20
  br i1 %cmp2, label %b_guard, label %out

b_guard:
  %sel_cmp = icmp sge i32 %a, %b
  %max = select i1 %sel_cmp, i32 %a, i32 %b
  %res = icmp eq i32 %max, 7
  br label %next
next:
; CHECK: next:
; CHECK: ret i1 false
  ret i1 %res
out:
  ret i1 false
}

define i1 @umax(i32 %a, i32 %b) {
; CHECK-LABEL: @umax(
entry:
  %cmp = icmp sgt i32 %a, 5
  br i1 %cmp, label %a_guard, label %out

a_guard:
  %cmp2 = icmp sgt i32 %b, 20
  br i1 %cmp2, label %b_guard, label %out

b_guard:
  %sel_cmp = icmp uge i32 %a, %b
  %max = select i1 %sel_cmp, i32 %a, i32 %b
  %res = icmp eq i32 %max, 7
  br label %next
next:
; CHECK: next:
; CHECK: ret i1 false
  ret i1 %res
out:
  ret i1 false
}

define i1 @clamp_low1(i32 %a) {
; CHECK-LABEL: @clamp_low1(
entry:
  %cmp = icmp sge i32 %a, 5
  br i1 %cmp, label %a_guard, label %out

a_guard:
  %sel_cmp = icmp eq i32 %a, 5
  %add = add i32 %a, -1
  %sel = select i1 %sel_cmp, i32 5, i32 %a
  %res = icmp eq i32 %sel, 4
  br label %next
next:
; CHECK: next:
; CHECK: ret i1 false
  ret i1 %res
out:
  ret i1 false
}

define i1 @clamp_low2(i32 %a) {
; CHECK-LABEL: @clamp_low2(
entry:
  %cmp = icmp sge i32 %a, 5
  br i1 %cmp, label %a_guard, label %out

a_guard:
  %sel_cmp = icmp ne i32 %a, 5
  %add = add i32 %a, -1
  %sel = select i1 %sel_cmp, i32 %a, i32 5
  %res = icmp eq i32 %sel, 4
  br label %next
next:
; CHECK: next:
; CHECK: ret i1 false
  ret i1 %res
out:
  ret i1 false
}

define i1 @clamp_high1(i32 %a) {
; CHECK-LABEL: @clamp_high1(
entry:
  %cmp = icmp sle i32 %a, 5
  br i1 %cmp, label %a_guard, label %out

a_guard:
  %sel_cmp = icmp eq i32 %a, 5
  %add = add i32 %a, 1
  %sel = select i1 %sel_cmp, i32 5, i32 %a
  %res = icmp eq i32 %sel, 6
  br label %next
next:
; CHECK: next:
; CHECK: ret i1 false
  ret i1 %res
out:
  ret i1 false
}

define i1 @clamp_high2(i32 %a) {
; CHECK-LABEL: @clamp_high2(
entry:
  %cmp = icmp sle i32 %a, 5
  br i1 %cmp, label %a_guard, label %out

a_guard:
  %sel_cmp = icmp ne i32 %a, 5
  %add = add i32 %a, 1
  %sel = select i1 %sel_cmp, i32 %a, i32 5
  %res = icmp eq i32 %sel, 6
  br label %next
next:
; CHECK: next:
; CHECK: ret i1 false
  ret i1 %res
out:
  ret i1 false
}

; Just showing arbitrary constants work, not really a clamp
define i1 @clamp_high3(i32 %a) {
; CHECK-LABEL: @clamp_high3(
entry:
  %cmp = icmp sle i32 %a, 5
  br i1 %cmp, label %a_guard, label %out

a_guard:
  %sel_cmp = icmp ne i32 %a, 5
  %add = add i32 %a, 100
  %sel = select i1 %sel_cmp, i32 %a, i32 5
  %res = icmp eq i32 %sel, 105
  br label %next
next:
; CHECK: next:
; CHECK: ret i1 false
  ret i1 %res
out:
  ret i1 false
}

define i1 @zext_unknown(i8 %a) {
; CHECK-LABEL: @zext_unknown
; CHECK: ret i1 true
entry:
  %a32 = zext i8 %a to i32
  %cmp = icmp sle i32 %a32, 256
  br label %exit
exit:
  ret i1 %cmp
}

define i1 @trunc_unknown(i32 %a) {
; CHECK-LABEL: @trunc_unknown
; CHECK: ret i1 true
entry:
  %a8 = trunc i32 %a to i8
  %a32 = sext i8 %a8 to i32
  %cmp = icmp sle i32 %a32, 128
  br label %exit
exit:
  ret i1 %cmp
}

; TODO: missed optimization
; Make sure we exercise non-integer inputs to unary operators (i.e. crash 
; check).
define i1 @bitcast_unknown(float %a) {
; CHECK-LABEL: @bitcast_unknown
; CHECK: ret i1 %cmp
entry:
  %a32 = bitcast float %a to i32
  %cmp = icmp sle i32 %a32, 128
  br label %exit
exit:
  ret i1 %cmp
}

define i1 @bitcast_unknown2(i8* %p) {
; CHECK-LABEL: @bitcast_unknown2
; CHECK: ret i1 %cmp
entry:
  %p64 = ptrtoint i8* %p to i64
  %cmp = icmp sle i64 %p64, 128
  br label %exit
exit:
  ret i1 %cmp
}


define i1 @and_unknown(i32 %a) {
; CHECK-LABEL: @and_unknown
; CHECK: ret i1 true
entry:
  %and = and i32 %a, 128
  %cmp = icmp sle i32 %and, 128
  br label %exit
exit:
  ret i1 %cmp
}

define i1 @lshr_unknown(i32 %a) {
; CHECK-LABEL: @lshr_unknown
; CHECK: ret i1 true
entry:
  %and = lshr i32 %a, 30
  %cmp = icmp sle i32 %and, 128
  br label %exit
exit:
  ret i1 %cmp
}
