; RUN: opt < %s -correlated-propagation -S | FileCheck %s
; PR2581

; CHECK: @test1
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
; CHECK: @test2
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
; CHECK: @test3
define i8 @test3(i8* %a) nounwind {
entry:
        %cond = icmp eq i8* %a, @gv
        br i1 %cond, label %bb2, label %bb

bb:             ; preds = %entry
        ret i8 0

bb2:            ; preds = %entry
; CHECK: %should_be_const = load i8* @gv
        %should_be_const = load i8* %a
        ret i8 %should_be_const
}

; PR1757
; CHECK: @test4
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

define i32 @switch1(i32 %s) {
; CHECK: @switch1
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
  %p = phi i32 [ 1, %entry ], [ -1, %negative ], [ -1, %negative ], [ -1, %negative ], [ -1, %negative ]
  ret i32 %p

next:
  %q = phi i32 [ 0, %negative ], [ 0, %negative ]
  ret i32 %q
}

define i32 @switch2(i32 %s) {
; CHECK: @switch2
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
; CHECK: @switch3
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
; CHECK: @switch4
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
