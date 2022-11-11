; RUN: opt < %s -passes=instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin10.0"

declare noalias ptr @malloc(i64) nounwind allockind("alloc,uninitialized") "alloc-family"="malloc"
declare void @free(ptr) allockind("free") "alloc-family"="malloc"

; PR5130
define i1 @test1() {
  %A = call noalias ptr @malloc(i64 4) nounwind
  %B = icmp eq ptr %A, null
  store i8 0, ptr %A

  call void @free(ptr %A)
  ret i1 %B

; CHECK-LABEL: @test1(
; CHECK: ret i1 false
}

; CHECK-LABEL: @test2(
define noalias ptr @test2() nounwind {
entry:
; CHECK: @malloc
  %A = call noalias ptr @malloc(i64 4) nounwind
; CHECK: icmp eq
  %tobool = icmp eq ptr %A, null
; CHECK: br i1
  br i1 %tobool, label %return, label %if.end

if.end:
; CHECK: store
  store i8 7, ptr %A
  br label %return

return:
; CHECK: phi
  %retval.0 = phi ptr [ %A, %if.end ], [ null, %entry ]
  ret ptr %retval.0
}
