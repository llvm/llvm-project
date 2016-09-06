; RUN: opt %s -S -simplifycfg | FileCheck %s

; cmp1 implies cmp2 is false
; CHECK-LABEL: @test1
; CHECK-NOT: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test1(i32 %a) {
  %cmp1 = icmp eq i32 %a, 0
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp eq i32 %a, 1
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

; cmp1 implies cmp2 is false
; CHECK-LABEL: @test2
; CHECK-NOT: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test2(i32 %a) {
  %cmp1 = icmp ugt i32 %a, 5
  br i1 %cmp1, label %untaken, label %taken

taken:
  %cmp2 = icmp ugt i32 %a, 6
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

; cmp1 implies cmp2 is false
; CHECK-LABEL: @test3
; CHECK-NOT: call void @is(i1 true)
; CHECK: call void @is(i1 false)
define void @test3(i32 %a) {
  %cmp1 = icmp ugt i32 %a, 1
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp eq i32 %a, 0
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

; cmp1 implies cmp2 is true
; CHECK-LABEL: @test4
; CHECK: call void @is(i1 true)
; CHECK-NOT: call void @is(i1 false)
define void @test4(i32 %a) {
  %cmp1 = icmp sgt i32 %a, 1
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp ugt i32 %a, 0
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

; cmp1 implies cmp2 is true
; CHECK-LABEL: @test5
; CHECK: call void @is(i1 true)
; CHECK-NOT: call void @is(i1 false)
define void @test5(i32 %a) {
  %cmp1 = icmp sgt i32 %a, 5
  br i1 %cmp1, label %taken, label %untaken

taken:
  %cmp2 = icmp sgt i32 %a, -1
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

declare void @is(i1)
