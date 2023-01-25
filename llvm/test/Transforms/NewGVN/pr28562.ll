; RUN: opt -S -passes=newgvn < %s | FileCheck %s
define ptr @test1(ptr %a) {
  %x1 = getelementptr inbounds i32, ptr %a, i32 10
  %x2 = getelementptr i32, ptr %a, i32 10
  ret ptr %x2
; CHECK-LABEL: @test1(
; CHECK: %[[x:.*]] = getelementptr i32, ptr %a, i32 10
; CHECK: ret ptr %[[x]]
}
