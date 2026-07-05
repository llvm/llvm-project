; RUN: opt -passes=jump-threading -S < %s | FileCheck %s

declare void @side_effect(i32)

define void @test0(i32 %i, i32 %len) {
; CHECK-LABEL: @test0(
 entry:
  call void @side_effect(i32 0)
  %i.inc = add nuw i32 %i, 1
  %c0 = icmp ult i32 %i.inc, %len
  br i1 %c0, label %left, label %right

 left:
; CHECK: entry:
; CHECK: br i1 %c0, label %left0, label %right

; CHECK: left0:
; CHECK: call void @side_effect
; CHECK-NOT: br i1 %c1
; CHECK: call void @side_effect
  call void @side_effect(i32 0)
  %c1 = icmp ult i32 %i, %len
  br i1 %c1, label %left0, label %right

 left0:
  call void @side_effect(i32 0)
  ret void

 right:
  %t = phi i32 [ 1, %left ], [ 2, %entry ]
  call void @side_effect(i32 %t)
  ret void
}

define void @test1(i32 %i, i32 %len) {
; CHECK-LABEL: @test1(
 entry:
  call void @side_effect(i32 0)
  %i.inc = add nsw i32 %i, 1
  %c0 = icmp slt i32 %i.inc, %len
  br i1 %c0, label %left, label %right

 left:
; CHECK: entry:
; CHECK: br i1 %c0, label %left0, label %right

; CHECK: left0:
; CHECK: call void @side_effect
; CHECK-NOT: br i1 %c1
; CHECK: call void @side_effect
  call void @side_effect(i32 0)
  %c1 = icmp slt i32 %i, %len
  br i1 %c1, label %left0, label %right

 left0:
  call void @side_effect(i32 0)
  ret void

 right:
  %t = phi i32 [ 1, %left ], [ 2, %entry ]
  call void @side_effect(i32 %t)
  ret void
}

define void @test2(i32 %i, i32 %len, ptr %c.ptr) {
; CHECK-LABEL: @test2(

; CHECK: entry:
; CHECK: br i1 %c0, label %cont, label %right
; CHECK: cont:
; CHECK: br i1 %c, label %left0, label %right
; CHECK: left0:
; CHECK: call void @side_effect(i32 0)
; CHECK: call void @side_effect(i32 0)
 entry:
  call void @side_effect(i32 0)
  %i.inc = add nsw i32 %i, 1
  %c0 = icmp slt i32 %i.inc, %len
  br i1 %c0, label %cont, label %right

 cont:
  %c = load i1, ptr %c.ptr
  br i1 %c, label %left, label %right

 left:
  call void @side_effect(i32 0)
  %c1 = icmp slt i32 %i, %len
  br i1 %c1, label %left0, label %right

 left0:
  call void @side_effect(i32 0)
  ret void

 right:
  %t = phi i32 [ 1, %left ], [ 2, %entry ], [ 3, %cont ]
  call void @side_effect(i32 %t)
  ret void
}

; A s<= B implies A s> B is false.
; CHECK-LABEL: @test3(
; CHECK: entry:
; CHECK: br i1 %cmp, label %if.end, label %if.end3
; CHECK-NOT: br i1 %cmp1, label %if.then2, label %if.end
; CHECK-NOT: call void @side_effect(i32 0)
; CHECK: br label %if.end3
; CHECK: ret void

define void @test3(i32 %a, i32 %b) {
entry:
  %cmp = icmp sle i32 %a, %b
  br i1 %cmp, label %if.then, label %if.end3

if.then:
  %cmp1 = icmp sgt i32 %a, %b
  br i1 %cmp1, label %if.then2, label %if.end

if.then2:
  call void @side_effect(i32 0)
  br label %if.end

if.end:
  br label %if.end3

if.end3:
  ret void
}

declare void @is(i1)

; If A >=s B is false then A <=s B is implied true.
; CHECK-LABEL: @test_sge_sle
; CHECK: call void @is(i1 true)
; CHECK-NOT: call void @is(i1 false)
define void @test_sge_sle(i32 %a, i32 %b) {
  %cmp1 = icmp sge i32 %a, %b
  br i1 %cmp1, label %untaken, label %taken

taken:
  %cmp2 = icmp sle i32 %a, %b
  br i1 %cmp2, label %istrue, label %isfalse

istrue:
  call void @is(i1 true)
  ret void

isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}

; If A <=s B is false then A <=s B is implied false.
; CHECK-LABEL: @test_sle_sle
; CHECK-NOT: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test_sle_sle(i32 %a, i32 %b) {
  %cmp1 = icmp sle i32 %a, %b
  br i1 %cmp1, label %untaken, label %taken

taken:
  %cmp2 = icmp sle i32 %a, %b
  br i1 %cmp2, label %istrue, label %isfalse

istrue:
  call void @is(i1 true)
  ret void

isfalse:
  call void @is(i1 false)
  ret void

untaken:
  ret void
}
