; RUN: opt %s -debugify -jump-threading -S | FileCheck %s
; Tests Bug 37966

define void @test0(i32 %i) {
; CHECK-LABEL: @test0(
; CHECK: left:
; CHECK: br label %left, !dbg ![[DBG0:[0-9]+]]
 entry:
  %c0 = icmp ult i32 %i, 5
  br i1 %c0, label %left, label %right

 left:
  br i1 %c0, label %left, label %right

 right:
  ret void
}

define void @test1(i32 %i, i32 %len) {
; CHECK-LABEL: @test1(
; CHECK: left:
; CHECK: br label %right, !dbg ![[DBG1:[0-9]+]]
 entry:
  %i.inc = add nuw i32 %i, 1
  %c0 = icmp ult i32 %i.inc, %len
  br i1 %c0, label %left, label %right

 left:
  %c1 = icmp ult i32 %i, %len
  br i1 %c1, label %right, label %left0

 left0:
  ret void

 right:
  ret void
}

; CHECK-DAG: ![[DBG0]] = !DILocation(
; CHECK-DAG: ![[DBG1]] = !DILocation(

